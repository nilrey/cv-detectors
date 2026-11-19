from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from src.eval.types import MatchedObj


IOU_BINS = [i / 10 for i in range(6, 11)]  # 0.6..1.0


def _hist(vals: list[float]) -> dict[str, int]:
    out = {f"{b:.1f}": 0 for b in IOU_BINS}
    for v in vals:
        for b in reversed(IOU_BINS):
            if v >= b:
                out[f"{b:.1f}"] += 1
                break
    return out


def summarize(objs: Iterable[MatchedObj]) -> dict:
    per_frame: dict[int, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    tp = fp = fn = 0
    ious: list[float] = []

    for o in objs:
        per_frame[o.frame_idx][o.match] += 1
        if o.match == "tp":
            tp += 1
            if o.iou is not None:
                ious.append(o.iou)
        elif o.match == "fp":
            fp += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou_hist": _hist(ious),
        "per_frame": {str(k): v for k, v in sorted(per_frame.items())},
    }