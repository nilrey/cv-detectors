from __future__ import annotations

from collections import defaultdict


class ClearMLLogger:
    name = "clearml"

    def __init__(self, task):
        self.task = task
        self._class_counts: dict[str, int] = defaultdict(int)
        self._track_ids: dict[str, set[int]] = defaultdict(set)
        self._confidences: list[float] = []
        self._frame_count = 0

    def on_video_start(self, video_result) -> None:
        self._class_counts = defaultdict(int)
        self._track_ids = defaultdict(set)
        self._confidences = []
        self._frame_count = 0
        self.task.get_logger().report_text(f"start video: {video_result.source}")

    def on_frame(self, frame_result) -> None:
        self._frame_count += 1
        for obj in frame_result.objects:
            self._class_counts[obj.class_name] += 1
            self._track_ids[obj.class_name].add(obj.track_id)
            self._confidences.append(obj.score)

    def on_video_end(self) -> None:
        logger = self.task.get_logger()
        logger.report_scalar("frames_processed", "value", self._frame_count, 0)
        for cls, cnt in self._class_counts.items():
            logger.report_scalar("class_counts", cls, cnt, 0)
        for cls, ids in self._track_ids.items():
            logger.report_scalar("unique_tracks", cls, len(ids), 0)
        if self._confidences:
            logger.report_histogram(
                "confidence", "dist", self._confidences, xaxis="conf", yaxis="count"
            )

    def on_batch_end(self, batch) -> None:
        self.task.get_logger().report_text(
            f"batch done: success={batch.success_count}, failed={batch.failed_count}"
        )

    def finalize(self) -> None:
        return

    def reset(self) -> None:
        self._class_counts = defaultdict(int)
        self._track_ids = defaultdict(set)
        self._confidences = []
        self._frame_count = 0