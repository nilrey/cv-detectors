from __future__ import annotations

import numpy as np

from src.domain.models import Detection, Track


class YoloTrackTracker:
    """Адаптер-обертка над model.track() из ultralytics.

    Не основной путь; используется, если нужно сравнить с встроенным трекингом.
    """

    name = "yolo_track"

    def __init__(self, model, conf: float = 0.25, iou: float = 0.5, tracker: str = "botsort.yaml"):
        self.model = model
        self.conf = conf
        self.iou = iou
        self.tracker = tracker

    def reset(self) -> None:
        return

    async def update(
        self, detections: list[Detection], frame: np.ndarray, frame_idx: int
    ) -> list[Track]:
        import asyncio

        results = await asyncio.to_thread(
            self.model.track,
            frame,
            conf=self.conf,
            iou=self.iou,
            tracker=self.tracker,
            persist=True,
            verbose=False,
        )
        out: list[Track] = []
        if not results:
            return out
        r = results[0]
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            return out
        ids = getattr(boxes, "id", None)
        for i, b in enumerate(boxes):
            if ids is None:
                continue
            tid = ids[i]
            if tid is None:
                continue
            xyxy = b.xyxy[0].tolist()
            cls_id = int(b.cls[0].item())
            score = float(b.conf[0].item())
            out.append(
                Track(
                    track_id=int(tid.item()),
                    bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                    score=score,
                    class_id=cls_id,
                    class_name=str(cls_id),
                    is_detected=True,
                )
            )
        return out