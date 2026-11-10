from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class VideoWriter:
    def __init__(self, path: str | Path, width: int, height: int, fps: float):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(
            str(self.path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps if fps > 0 else 25.0,
            (width, height),
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"cannot open writer: {self.path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()

    def on_video_start(self, meta) -> None:
        return

    def on_frame(self, frame_result, frame: np.ndarray) -> None:
        return

    def on_video_end(self) -> None:
        return

    def finalize(self) -> None:
        self.close()

    def reset(self) -> None:
        return