from __future__ import annotations

import cv2
import numpy as np

from src.config.schema import DrawingConfig
from src.domain.models import FrameResult, Track


def _color(cfg: DrawingConfig, key: str) -> tuple[int, int, int]:
    rgb = cfg.colors.get(key, [255, 255, 255])
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def draw_frame(frame: np.ndarray, fr: FrameResult, cfg: DrawingConfig) -> np.ndarray:
    out = frame.copy()
    if cfg.show_roi and fr.rois:
        roi_color = _color(cfg, "roi")
        for r in fr.rois:
            x1, y1, x2, y2 = (int(v) for v in r)
            cv2.rectangle(out, (x1, y1), (x2, y2), roi_color, 2)
    if cfg.show_bbox:
        bbox_color = _color(cfg, "bbox")
        label_color = _color(cfg, "label")
        for obj in fr.objects:
            _draw_track(out, obj, bbox_color, label_color, cfg.show_label)
    return out


def _draw_track(
    frame: np.ndarray,
    track: Track,
    bbox_color: tuple[int, int, int],
    label_color: tuple[int, int, int],
    show_label: bool,
) -> None:
    x1, y1, x2, y2 = (int(v) for v in track.bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
    if not show_label:
        return
    text = f"{track.class_name} #{track.track_id} {track.score:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 4, y1), bbox_color, -1)
    cv2.putText(
        frame,
        text,
        (x1 + 2, y1 - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        label_color,
        1,
        cv2.LINE_AA,
    )