from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]
    score: float
    class_id: int
    class_name: str


@dataclass
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    score: float
    class_id: int
    class_name: str
    is_detected: bool
    roi: tuple[float, float, float, float] | None = None


@dataclass
class DetectorOutput:
    detections: list[Detection]
    aux: dict[str, Any] | None = None


@dataclass
class FrameResult:
    frame_idx: int
    objects: list[Track]
    rois: list[tuple[float, float, float, float]] | None = None


@dataclass
class VideoResult:
    source: str
    timestamp: str
    width: int
    height: int
    fps: float
    total_frames: int
    frames: list[FrameResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    results: list[VideoResult] = field(default_factory=list)
    success_count: int = 0
    failed_count: int = 0
    errors: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)