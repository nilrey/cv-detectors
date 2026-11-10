from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from src.config.schema import AppConfig, RuntimeParams
from src.domain.models import BatchResult, FrameResult, VideoResult
from src.factories import build_detector, build_tracker
from src.io.json_writer import JsonWriter
from src.io.video_reader import VideoReader
from src.io.video_writer import VideoWriter
from src.logging.clearml import ClearMLLogger
from src.logging.local import LocalLogger
from src.pipeline.drawing import draw_frame

LOG_FRAME = 50


def _log(msg: str) -> None:
    print(msg)


def _apply_runtime(detector, tracker, params: RuntimeParams) -> None:
    if params.conf is not None and hasattr(detector, "conf"):
        detector.conf = params.conf
    if params.iou is not None and hasattr(detector, "iou"):
        detector.iou = params.iou
    if params.threshold is not None and hasattr(detector, "threshold"):
        detector.threshold = params.threshold
    if params.device is not None and hasattr(detector, "device"):
        detector.device = params.device


def _make_run_dirs(config: AppConfig) -> tuple[Path, Path, Path]:
    if config.output.base_dir:
        run_dir = Path(config.output.base_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        video_dir = run_dir / "video"
        json_dir = run_dir / "json"
        log_dir = run_dir / "logs"
        for d in (video_dir, json_dir, log_dir):
            d.mkdir(parents=True, exist_ok=True)
        return video_dir, json_dir, log_dir
    return Path(config.output.video_dir), Path(config.output.json_dir), Path("outputs/logs")


async def _process_video(
    source: str,
    params: RuntimeParams,
    config: AppConfig,
    detector,
    tracker,
    loggers: list,
    video_dir: Path,
    json_dir: Path,
) -> VideoResult:
    reader = VideoReader(source)
    meta = reader.open()
    ts = datetime.now(timezone.utc).isoformat()
    video_result = VideoResult(
        source=source,
        timestamp=ts,
        width=meta.width,
        height=meta.height,
        fps=meta.fps,
        total_frames=meta.total_frames,
    )
    tracker.reset()

    save_video = (
        params.save_video if params.save_video is not None else config.pipeline.save_video
    )
    writer = None
    if save_video:
        out_path = video_dir / f"{Path(source).stem}.mp4"
        writer = VideoWriter(out_path, meta.width, meta.height, meta.fps)

    json_writer = JsonWriter(
        json_dir,
        save_json=config.pipeline.save_json,
        save_jsonl=config.pipeline.save_jsonl,
        json_format=config.pipeline.json_format,
    )

    writers = [json_writer]
    if writer is not None:
        writers.append(writer)

    for lg in loggers:
        lg.on_video_start(video_result)
    for w in writers:
        w.on_video_start(video_result)

    _log(f"start video: {source} ({meta.width}x{meta.height}, {meta.total_frames} frames)")

    try:
        for idx, frame in reader.frames():
            det_out = await detector.detect(frame, idx)
            tracks = await tracker.update(det_out.detections, frame, idx)
            rois = None
            if det_out.aux and "rois" in det_out.aux:
                rois = det_out.aux["rois"]
            fr = FrameResult(frame_idx=idx, objects=tracks, rois=rois)
            video_result.frames.append(fr)

            for lg in loggers:
                lg.on_frame(fr)
            json_writer.on_frame(fr)

            if writer is not None:
                drawn = draw_frame(frame, fr, config.drawing)
                writer.write(drawn)

            if idx < 6 or idx % LOG_FRAME == 0:
                _log(f"frame {idx}/{meta.total_frames}")
    finally:
        for lg in loggers:
            lg.on_video_end()
        for w in writers:
            w.on_video_end()
        if writer is not None:
            writer.finalize()
        json_writer.finalize()
        reader.close()

    video_result.summary = {
        "processed_frames": len(video_result.frames),
        "total_objects": sum(len(f.objects) for f in video_result.frames),
    }
    _log(f"done video: {source}, frames={len(video_result.frames)}")
    return video_result


async def run(params: RuntimeParams, config: AppConfig) -> BatchResult:
    _log("-" * 40)
    _log(f"run: {params.run_id} experiment={params.experiment_name}")
    _log(f"detector={config.detector.name} tracker={config.tracker.name}")
    _log(f"videos={len(params.source)}")
    weights = config.detector.params.get("weights", "n/a")
    _log(f"detector weights: {Path(weights).name}")

    video_dir, json_dir, log_dir = _make_run_dirs(config)
    _log(f"output dir: {video_dir.parent}")

    detector = build_detector(config, params)
    tracker = build_tracker(config, params)
    _apply_runtime(detector, tracker, params)
    _log(f"loading detector: {detector.name}")
    await detector.load()
    _log(f"detector loaded: {detector.name} device={getattr(detector, 'device', 'n/a')}")
    try:
        await detector.warmup()
    except Exception as exc:
        _log(f"warmup skipped: {exc}")

    loggers: list = []
    if config.logging.local:
        loggers.append(LocalLogger(log_dir=log_dir))

    clearml_enabled = (
        params.clearml.enabled
        if params.clearml.enabled is not None
        else config.logging.clearml.enabled
    )
    if clearml_enabled:
        try:
            from clearml import Task

            project_name = params.clearml.project_name or config.logging.clearml.project_name
            task_name = params.clearml.task_name or f"{params.experiment_name}_{params.run_id}"
            task = Task.init(
                project_name=project_name,
                task_name=task_name,
                tags=params.clearml.tags,
            )
            loggers.append(ClearMLLogger(task))
            _log(f"clearml: project={project_name} task={task_name}")
        except Exception as exc:
            _log(f"clearml disabled: {exc}")

    batch = BatchResult()
    for source in params.source:
        local = next((lg for lg in loggers if isinstance(lg, LocalLogger)), None)
        if local is not None:
            local.start_video(source)
            local.event("video_start", source=source)
        try:
            vr = await _process_video(
                source, params, config, detector, tracker, loggers, video_dir, json_dir
            )
            batch.results.append(vr)
            batch.success_count += 1
        except Exception as exc:
            msg = f"{source}: {exc}"
            _log(f"error: {msg}")
            batch.errors.append(msg)
            batch.failed_count += 1
            if local is not None:
                local.error(msg)

    batch.summary = {
        "success": batch.success_count,
        "failed": batch.failed_count,
        "videos": len(batch.results),
    }
    for lg in loggers:
        lg.on_batch_end(batch)
        lg.finalize()
    _log(f"batch done: success={batch.success_count} failed={batch.failed_count}")
    _log("-" * 40)
    return batch