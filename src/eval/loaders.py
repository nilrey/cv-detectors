from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from src.eval.types import DetObj, RefObj


def _xywh_to_xyxy(b: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = b
    return (float(x), float(y), float(x + w), float(y + h))


def _xyxy(b: list[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = b
    return (float(x1), float(y1), float(x2), float(y2))


def load_ref(path: str | Path) -> tuple[dict[int, int], list[RefObj]]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in data.get("categories", [])}
    img_id_to_idx = {img["id"]: i for i, img in enumerate(data.get("images", []))}

    refs: list[RefObj] = []
    for ann in data.get("annotations", []):
        img_id = int(ann["image_id"])
        if img_id not in img_id_to_idx:
            continue
        frame_idx = img_id_to_idx[img_id]
        attrs = ann.get("attributes") or {}
        refs.append(
            RefObj(
                img_id=img_id,
                frame_idx=frame_idx,
                track_id=int(attrs.get("track_id", -1)),
                cls_name=cat_id_to_name.get(ann["category_id"], str(ann["category_id"])),
                bbox=_xywh_to_xyxy(ann["bbox"]),
            )
        )
    return img_id_to_idx, refs


def iter_dets(path: str | Path) -> Iterator[DetObj]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            frame_idx = int(row["frame_idx"])
            for obj in row.get("objects", []):
                yield DetObj(
                    frame_idx=frame_idx,
                    track_id=int(obj["track_id"]),
                    cls_name=str(obj["class_name"]),
                    score=float(obj.get("score", 0.0)),
                    bbox=_xyxy(obj["bbox"]),
                    raw=obj,
                )