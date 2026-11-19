from __future__ import annotations

from src.eval.types import DetObj, MatchedObj, RefObj


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_frame(
    frame_idx: int,
    dets: list[DetObj],
    refs: list[RefObj],
    iou_thr: float,
) -> list[MatchedObj]:
    pairs: list[tuple[float, int, int]] = []
    for i, det in enumerate(dets):
        for j, ref in enumerate(refs):
            if det.cls_name != ref.cls_name:
                continue
            v = iou(det.bbox, ref.bbox)
            if v >= iou_thr:
                pairs.append((v, i, j))
    pairs.sort(key=lambda t: t[0], reverse=True)

    used_det: set[int] = set()
    used_ref: set[int] = set()
    out: list[MatchedObj] = []

    for v, i, j in pairs:
        if i in used_det or j in used_ref:
            continue
        used_det.add(i)
        used_ref.add(j)
        det = dets[i]
        ref = refs[j]
        out.append(
            MatchedObj(
                frame_idx=frame_idx,
                match="tp",
                track_id=det.track_id,
                ref_track_id=ref.track_id,
                cls_name=det.cls_name,
                score=det.score,
                bbox=det.bbox,
                iou=v,
            )
        )

    for i, det in enumerate(dets):
        if i in used_det:
            continue
        out.append(
            MatchedObj(
                frame_idx=frame_idx,
                match="fp",
                track_id=det.track_id,
                ref_track_id=None,
                cls_name=det.cls_name,
                score=det.score,
                bbox=det.bbox,
                iou=None,
            )
        )

    for j, ref in enumerate(refs):
        if j in used_ref:
            continue
        out.append(
            MatchedObj(
                frame_idx=frame_idx,
                match="fn",
                track_id=None,
                ref_track_id=ref.track_id,
                cls_name=ref.cls_name,
                score=None,
                bbox=ref.bbox,
                iou=None,
            )
        )

    return out