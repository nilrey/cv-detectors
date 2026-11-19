from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RefObj:
    img_id: int
    frame_idx: int
    track_id: int
    cls_name: str
    bbox: tuple[float, float, float, float]  # xyxy


@dataclass
class DetObj:
    frame_idx: int
    track_id: int
    cls_name: str
    score: float
    bbox: tuple[float, float, float, float]  # xyxy
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchedObj:
    frame_idx: int
    match: str  # tp / fp / fn
    track_id: int | None
    ref_track_id: int | None
    cls_name: str
    score: float | None
    bbox: tuple[float, float, float, float]
    iou: float | None = None