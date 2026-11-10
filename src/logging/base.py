from __future__ import annotations

from typing import Protocol

from src.domain.models import BatchResult, FrameResult, VideoResult


class Logger(Protocol):
    name: str

    def on_video_start(self, video_result: VideoResult) -> None: ...

    def on_frame(self, frame_result: FrameResult) -> None: ...

    def on_video_end(self) -> None: ...

    def on_batch_end(self, batch: BatchResult) -> None: ...

    def finalize(self) -> None: ...

    def reset(self) -> None: ...