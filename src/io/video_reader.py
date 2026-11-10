from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoMeta:
    width: int
    height: int
    fps: float
    total_frames: int


class VideoReader:
    def __init__(self, source: str):
        self.source = source
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> VideoMeta:
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open video: {self.source}")
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 25.0
        total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return VideoMeta(width=width, height=height, fps=fps, total_frames=total)

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        if self._cap is None:
            raise RuntimeError("reader is not opened")
        idx = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()