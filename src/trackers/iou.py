from __future__ import annotations

import numpy as np

from src.domain.models import Detection, Track


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
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


class IouTracker:
    name = "iou"

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30, min_hits: int = 3):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.reset()

    def reset(self) -> None:
        self._tracks: list[Track] = []
        self._ages: dict[int, int] = {}
        self._hits: dict[int, int] = {}
        self._next_id = 1

    async def update(
        self, detections: list[Detection], frame: np.ndarray, frame_idx: int
    ) -> list[Track]:
        unmatched = list(range(len(detections)))
        result: list[Track] = []

        for tr in self._tracks:
            best_iou = 0.0
            best_j = -1
            for j in unmatched:
                iou = _iou(tr.bbox, detections[j].bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0 and best_iou >= self.iou_threshold:
                det = detections[best_j]
                tr.bbox = det.bbox
                tr.score = det.score
                tr.class_id = det.class_id
                tr.class_name = det.class_name
                tr.is_detected = True
                self._ages[tr.track_id] = 0
                self._hits[tr.track_id] = self._hits.get(tr.track_id, 0) + 1
                unmatched.remove(best_j)
                result.append(tr)
            else:
                tr.is_detected = False
                self._ages[tr.track_id] = self._ages.get(tr.track_id, 0) + 1
                if self._ages[tr.track_id] <= self.max_age:
                    result.append(tr)

        for j in unmatched:
            det = detections[j]
            tr = Track(
                track_id=self._next_id,
                bbox=det.bbox,
                score=det.score,
                class_id=det.class_id,
                class_name=det.class_name,
                is_detected=True,
            )
            self._next_id += 1
            self._ages[tr.track_id] = 0
            self._hits[tr.track_id] = 1
            result.append(tr)

        self._tracks = [t for t in result if self._hits.get(t.track_id, 0) >= 0]
        return result