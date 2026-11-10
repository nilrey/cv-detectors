from __future__ import annotations

from typing import Protocol

import numpy as np

from src.domain.models import DetectorOutput


class Detector(Protocol):
    name: str
    class_map: dict[int, str]

    async def load(self) -> None: ...

    async def warmup(self) -> None: ...

    async def detect(self, frame: np.ndarray, frame_idx: int) -> DetectorOutput: ...