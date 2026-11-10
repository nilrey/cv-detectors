from __future__ import annotations

from src.config.schema import AppConfig, RuntimeParams
from src.detectors.cascade import CascadeDetector
from src.detectors.detr import DetrDetector
from src.detectors.yolo import YoloDetector
from src.trackers.botsort import BotSortTracker
from src.trackers.iou import IouTracker


def _build_single_detector(name: str, params: dict):
    if name == "yolo":
        return YoloDetector(**params)
    if name == "detr":
        return DetrDetector(**params)
    raise ValueError(f"unknown detector: {name}")


def build_detector(config: AppConfig, params: RuntimeParams):
    name = config.detector.name
    cfg = dict(config.detector.params)
    if name == "cascade":
        seg = cfg.pop("segmenter")
        inner = cfg.pop("inner")
        segmenter = _build_single_detector(seg["name"], seg.get("params", {}))
        inner_det = _build_single_detector(inner["name"], inner.get("params", {}))
        return CascadeDetector(segmenter=segmenter, inner=inner_det, **cfg)
    return _build_single_detector(name, cfg)


def build_tracker(config: AppConfig, params: RuntimeParams):
    name = config.tracker.name
    cfg = dict(config.tracker.params)
    if name == "iou":
        return IouTracker(**cfg)
    if name == "botsort":
        return BotSortTracker(**cfg)
    raise ValueError(f"unknown tracker: {name}")