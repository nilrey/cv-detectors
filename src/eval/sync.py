from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from src.eval.loaders import iter_dets, load_ref
from src.eval.match import match_frame
from src.eval.metrics import summarize
from src.eval.types import DetObj, RefObj

IOU_THR = 0.5


def _group_dets(dets: list[DetObj]) -> dict[int, list[DetObj]]:
    g: dict[int, list[DetObj]] = defaultdict(list)
    for d in dets:
        g[d.frame_idx].append(d)
    return g


def _group_refs(refs: list[RefObj]) -> dict[int, list[RefObj]]:
    g: dict[int, list[RefObj]] = defaultdict(list)
    for r in refs:
        g[r.frame_idx].append(r)
    return g


def _obj_to_dict(o) -> dict:
    return {
        "frame_idx": o.frame_idx,
        "match": o.match,
        "track_id": o.track_id,
        "ref_track_id": o.ref_track_id,
        "class_name": o.cls_name,
        "score": o.score,
        "bbox": list(o.bbox),
        "iou": o.iou,
    }


def sync(
    ref_path: str | Path,
    det_path: str | Path,
    out_dir: str | Path,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, refs = load_ref(ref_path)
    dets = list(iter_dets(det_path))

    refs_by_frame = _group_refs(refs)
    dets_by_frame = _group_dets(dets)

    frames = sorted(set(refs_by_frame) | set(dets_by_frame))
    all_matched = []
    for idx in frames:
        all_matched.extend(
            match_frame(idx, dets_by_frame.get(idx, []), refs_by_frame.get(idx, []), IOU_THR)
        )

    sync_path = out_dir / "sync.jsonl"
    with sync_path.open("w", encoding="utf-8") as f:
        for idx in frames:
            frame_objs = [o for o in all_matched if o.frame_idx == idx]
            payload = {
                "frame_idx": idx,
                "objects": [_obj_to_dict(o) for o in frame_objs],
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    metrics = summarize(all_matched)
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return {"sync": str(sync_path), "metrics": str(metrics_path), "summary": metrics}