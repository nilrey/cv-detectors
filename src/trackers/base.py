from __future__ import annotations

from typing import Protocol

import numpy as np

from src.domain.models import Detection, Track


class Tracker(Protocol):
    name: str

    async def update(
        self, detections: list[Detection], frame: np.ndarray, frame_idx: int
    ) -> list[Track]: ...

    def reset(self) -> None: ...