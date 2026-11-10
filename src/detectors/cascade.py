from __future__ import annotations

import numpy as np

from src.domain.models import Detection, DetectorOutput


class CascadeDetector:
    name = "cascade"

    def __init__(
        self,
        segmenter,
        inner,
        roi_padding: int = 8,
        min_roi_area: int = 64,
    ):
        self.segmenter = segmenter
        self.inner = inner
        self.roi_padding = roi_padding
        self.min_roi_area = min_roi_area
        self.class_map = getattr(inner, "class_map", {})

    async def load(self) -> None:
        await self.segmenter.load()
        await self.inner.load()

    async def warmup(self) -> None:
        await self.segmenter.warmup()
        await self.inner.warmup()

    async def detect(self, frame: np.ndarray, frame_idx: int) -> DetectorOutput:
        seg_out = await self.segmenter.detect(frame, frame_idx)
        rois: list[tuple[float, float, float, float]] = []
        h, w = frame.shape[:2]
        for det in seg_out.detections:
            x1, y1, x2, y2 = det.bbox
            area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
            if area < self.min_roi_area:
                continue
            rx1 = max(0, int(x1) - self.roi_padding)
            ry1 = max(0, int(y1) - self.roi_padding)
            rx2 = min(w, int(x2) + self.roi_padding)
            ry2 = min(h, int(y2) + self.roi_padding)
            if rx2 <= rx1 or ry2 <= ry1:
                continue
            rois.append((float(rx1), float(ry1), float(rx2), float(ry2)))

        detections: list[Detection] = []
        for roi in rois:
            rx1, ry1, rx2, ry2 = (int(v) for v in roi)
            crop = frame[ry1:ry2, rx1:rx2]
            if crop.size == 0:
                continue
            inner_out = await self.inner.detect(crop, frame_idx)
            for det in inner_out.detections:
                x1, y1, x2, y2 = det.bbox
                detections.append(
                    Detection(
                        bbox=(x1 + rx1, y1 + ry1, x2 + rx1, y2 + ry1),
                        score=det.score,
                        class_id=det.class_id,
                        class_name=det.class_name,
                    )
                )
        return DetectorOutput(detections=detections, aux={"rois": rois})