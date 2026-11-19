from __future__ import annotations

import cv2
import numpy as np

from src.config.schema import DrawingConfig
from src.domain.models import FrameResult, Track


def _color(cfg: DrawingConfig, key: str) -> tuple[int, int, int]:
    rgb = cfg.colors.get(key, [255, 255, 255])
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def _pick_color(cfg: DrawingConfig, cls_name: str) -> tuple[int, int, int]:
    if cls_name in cfg.class_colors:
        rgb = cfg.class_colors[cls_name]
        return int(rgb[0]), int(rgb[1]), int(rgb[2])
    return _color(cfg, "bbox")


def draw_frame(frame: np.ndarray, fr: FrameResult, cfg: DrawingConfig) -> np.ndarray:
    out = frame.copy()
    if cfg.show_roi and fr.rois:
        roi_color = _color(cfg, "roi")
        for r in fr.rois:
            x1, y1, x2, y2 = (int(v) for v in r)
            cv2.rectangle(out, (x1, y1), (x2, y2), roi_color, cfg.box_thickness)
    if cfg.show_bbox:
        for obj in fr.objects:
            bbox_color = _pick_color(cfg, obj.class_name)
            _draw_track(out, obj, bbox_color, cfg)
    return out


def _draw_track(
    frame: np.ndarray,
    track: Track,
    bbox_color: tuple[int, int, int],
    cfg: DrawingConfig,
) -> None:
    x1, y1, x2, y2 = (int(v) for v in track.bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, cfg.box_thickness)
    if not cfg.show_label:
        return
    text = f"{track.class_name} #{track.track_id}:{track.score:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale, 1)
    
    # фон подписи - цвет класса
    cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 4, y1), bbox_color, -1)
    
    # сам текст - белый (или тоже цвет класса, если хочешь)
    cv2.putText(
        frame,
        text,
        (x1 + 2, y1 - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        cfg.font_scale,
        (255, 255, 255), # белый текст на цветном фоне
        1,
        cv2.LINE_AA,
    )