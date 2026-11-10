from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.domain.models import BatchResult, FrameResult, VideoResult


def _frame_to_dict(fr: FrameResult) -> dict[str, Any]:
    return {
        "frame_idx": fr.frame_idx,
        "objects": [
            {
                "track_id": t.track_id,
                "bbox": list(t.bbox),
                "score": t.score,
                "class_id": t.class_id,
                "class_name": t.class_name,
                "is_detected": t.is_detected,
                "roi": list(t.roi) if t.roi is not None else None,
            }
            for t in fr.objects
        ],
        "rois": [list(r) for r in fr.rois] if fr.rois is not None else None,
    }


def _video_to_dict(vr: VideoResult) -> dict[str, Any]:
    return {
        "source": vr.source,
        "timestamp": vr.timestamp,
        "width": vr.width,
        "height": vr.height,
        "fps": vr.fps,
        "total_frames": vr.total_frames,
        "frames": [_frame_to_dict(f) for f in vr.frames],
        "summary": vr.summary,
    }


class JsonWriter:
    def __init__(
        self,
        json_dir: str | Path,
        save_json: bool = True,
        save_jsonl: bool = True,
        json_format: str = "full",
    ):
        self.json_dir = Path(json_dir)
        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.save_json = save_json
        self.save_jsonl = save_jsonl
        self.json_format = json_format
        self._jsonl_fh = None
        self._video: VideoResult | None = None

    def on_video_start(self, video_result: VideoResult) -> None:
        self._video = video_result
        if self.save_jsonl:
            path = self.json_dir / f"{Path(video_result.source).stem}.jsonl"
            self._jsonl_fh = path.open("w", encoding="utf-8")

    def on_frame(self, frame_result: FrameResult) -> None:
        if self._jsonl_fh is not None and self._video is not None:
            payload = {
                "source": self._video.source,
                "width": self._video.width,
                "height": self._video.height,
                "fps": self._video.fps,
                **_frame_to_dict(frame_result),
            }
            self._jsonl_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def on_video_end(self) -> None:
        if self._video is None:
            return
        if self.save_json:
            path = self.json_dir / f"{Path(self._video.source).stem}.json"
            with path.open("w", encoding="utf-8") as f:
                json.dump(_video_to_dict(self._video), f, ensure_ascii=False)
        if self._jsonl_fh is not None:
            self._jsonl_fh.close()
            self._jsonl_fh = None
        self._video = None

    def finalize(self) -> None:
        if self._jsonl_fh is not None:
            self._jsonl_fh.close()
            self._jsonl_fh = None

    def reset(self) -> None:
        self.finalize()
        self._video = None