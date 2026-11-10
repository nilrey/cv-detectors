from __future__ import annotations

import asyncio

import numpy as np

from src.detectors._device import resolve_device
from src.domain.coco import COCO_ID_TO_NAME
from src.domain.models import Detection, DetectorOutput


class YoloDetector:
    name = "yolo"
    class_map = COCO_ID_TO_NAME

    def __init__(self, weights: str, device: str = "auto", conf: float = 0.25, iou: float = 0.5):
        self.weights = weights
        self.device = device
        self.conf = conf
        self.iou = iou
        self._model = None

    async def load(self) -> None:
        from ultralytics import YOLO

        self._model = await asyncio.to_thread(YOLO, self.weights)
        self.device = resolve_device(self.device)
        self._model.to(self.device)

    async def warmup(self) -> None:
        if self._model is None:
            return
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        await asyncio.to_thread(self._model.predict, dummy, conf=self.conf, iou=self.iou, verbose=False)

    async def detect(self, frame: np.ndarray, frame_idx: int) -> DetectorOutput:
        results = await asyncio.to_thread(
            self._model.predict, frame, conf=self.conf, iou=self.iou, verbose=False
        )
        detections: list[Detection] = []
        if results:
            r = results[0]
            boxes = getattr(r, "boxes", None)
            if boxes is not None:
                for b in boxes:
                    xyxy = b.xyxy[0].tolist()
                    cls_id = int(b.cls[0].item())
                    score = float(b.conf[0].item())
                    detections.append(
                        Detection(
                            bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                            score=score,
                            class_id=cls_id,
                            class_name=COCO_ID_TO_NAME.get(cls_id, str(cls_id)),
                        )
                    )
        return DetectorOutput(detections=detections)