from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from src.eval.metrics import IOU_BINS

CONF_BINS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
CONF_LABELS = ["0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]


def _read_sync(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _conf_idx(score: float) -> int | None:
    for i, (lo, hi) in enumerate(CONF_BINS):
        if lo <= score < hi:
            return i
    return None


def build_stats(sync_path: str | Path) -> dict:
    rows = _read_sync(sync_path)

    tp = fp = fn = 0
    ious: list[float] = []
    per_frame_counts: list[int] = []
    per_frame_change: list[int] = []

    det_tracks_by_cls: dict[str, set[int]] = defaultdict(set)
    ref_tracks_by_cls: dict[str, set[int]] = defaultdict(set)
    det_objs_by_cls: dict[str, int] = defaultdict(int)
    ref_objs_by_cls: dict[str, int] = defaultdict(int)
    det_conf = [0] * len(CONF_BINS)
    tp_conf = [0] * len(CONF_BINS)

    prev_cnt = 0
    for row in rows:
        objs = row.get("objects", [])
        per_frame_counts.append(len(objs))
        per_frame_change.append(abs(len(objs) - prev_cnt))
        prev_cnt = len(objs)

        for o in objs:
            m = o["match"]
            cls = o["class_name"]
            if m == "tp":
                tp += 1
                if o.get("iou") is not None:
                    ious.append(float(o["iou"]))
                if o.get("track_id") is not None:
                    det_tracks_by_cls[cls].add(int(o["track_id"]))
                if o.get("ref_track_id") is not None:
                    ref_tracks_by_cls[cls].add(int(o["ref_track_id"]))
                det_objs_by_cls[cls] += 1
                ref_objs_by_cls[cls] += 1
                if o.get("score") is not None:
                    ci = _conf_idx(float(o["score"]))
                    if ci is not None:
                        tp_conf[ci] += 1
            elif m == "fp":
                fp += 1
                if o.get("track_id") is not None:
                    det_tracks_by_cls[cls].add(int(o["track_id"]))
                det_objs_by_cls[cls] += 1
            else:
                fn += 1
                if o.get("ref_track_id") is not None:
                    ref_tracks_by_cls[cls].add(int(o["ref_track_id"]))
                ref_objs_by_cls[cls] += 1

            if o.get("score") is not None:
                ci = _conf_idx(float(o["score"]))
                if ci is not None:
                    det_conf[ci] += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    iou_hist = {f"{b:.1f}": 0 for b in IOU_BINS}
    for v in ious:
        for b in reversed(IOU_BINS):
            if v >= b:
                iou_hist[f"{b:.1f}"] += 1
                break

    classes = sorted(set(det_objs_by_cls) | set(ref_objs_by_cls))

    return {
        "totals": {
            "frames": len(rows),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "per_frame_counts": per_frame_counts,
        "per_frame_change": per_frame_change,
        "conf_det": dict(zip(CONF_LABELS, det_conf)),
        "conf_tp": dict(zip(CONF_LABELS, tp_conf)),
        "iou_hist": iou_hist,
        "classes": classes,
        "det_objs_by_cls": {c: det_objs_by_cls.get(c, 0) for c in classes},
        "ref_objs_by_cls": {c: ref_objs_by_cls.get(c, 0) for c in classes},
        "det_tracks_by_cls": {c: len(det_tracks_by_cls.get(c, set())) for c in classes},
        "ref_tracks_by_cls": {c: len(ref_tracks_by_cls.get(c, set())) for c in classes},
    }


def render_bars(stats: dict, out_dir: str | Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    def _bar(labels, values, title, xlabel, ylabel, name):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, values)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        p = out_dir / f"{name}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(str(p))

    t = stats["totals"]
    _bar(
        ["TP", "FP", "FN"],
        [t["tp"], t["fp"], t["fn"]],
        f"TP/FP/FN (precision={t['precision']:.3f}, recall={t['recall']:.3f}, F1={t['f1']:.3f})",
        "metric",
        "count",
        "tp_fp_fn",
    )

    _bar(
        list(stats["det_objs_by_cls"].keys()),
        list(stats["det_objs_by_cls"].values()),
        "Обнаруженные объекты по классам",
        "class",
        "count",
        "det_objs_by_cls",
    )
    _bar(
        list(stats["ref_objs_by_cls"].keys()),
        list(stats["ref_objs_by_cls"].values()),
        "Ссылка на объекты по классу",
        "class",
        "count",
        "ref_objs_by_cls",
    )
    _bar(
        list(stats["det_tracks_by_cls"].keys()),
        list(stats["det_tracks_by_cls"].values()),
        "Детекция уникальных треков по классам",
        "class",
        "count",
        "det_tracks_by_cls",
    )
    _bar(
        list(stats["ref_tracks_by_cls"].keys()),
        list(stats["ref_tracks_by_cls"].values()),
        "Ссылочные уникальные треки по классам",
        "class",
        "count",
        "ref_tracks_by_cls",
    )
    _bar(
        list(stats["conf_det"].keys()),
        list(stats["conf_det"].values()),
        "Обнаруженное распределение достоверности",
        "confidence",
        "count",
        "conf_det",
    )
    _bar(
        list(stats["conf_tp"].keys()),
        list(stats["conf_tp"].values()),
        "TP распределение достоверности",
        "confidence",
        "count",
        "conf_tp",
    )
    _bar(
        list(stats["iou_hist"].keys()),
        list(stats["iou_hist"].values()),
        "IoU достоверности (TP)",
        "iou bin",
        "count",
        "iou_hist",
    )

    return paths