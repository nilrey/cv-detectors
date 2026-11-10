from __future__ import annotations

import asyncio

import numpy as np

from src.detectors._device import resolve_device
from src.domain.coco import COCO_ID_TO_NAME
from src.domain.models import Detection, DetectorOutput


class DetrDetector:
    name = "detr"
    class_map = COCO_ID_TO_NAME

    def __init__(self, weights: str = "facebook/detr-resnet-50", device: str = "auto", threshold: float = 0.5):
        self.weights = weights
        self.device = device
        self.threshold = threshold
        self._processor = None
        self._model = None

    async def load(self) -> None:
        from transformers import DetrForObjectDetection, DetrImageProcessor

        self.device = resolve_device(self.device)
        self._processor = DetrImageProcessor.from_pretrained(self.weights)
        self._model = DetrForObjectDetection.from_pretrained(self.weights)
        self._model.to(self.device)
        self._model.eval()

    async def warmup(self) -> None:
        return

    def _infer(self, frame: np.ndarray) -> DetectorOutput:
        import torch
        from PIL import Image

        image = Image.fromarray(frame[:, :, ::-1])
        inputs = self._processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        results = self._processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.threshold
        )[0]
        detections: list[Detection] = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            cls_id = int(label.item())
            detections.append(
                Detection(
                    bbox=tuple(float(x) for x in box.tolist()),
                    score=float(score.item()),
                    class_id=cls_id,
                    class_name=COCO_ID_TO_NAME.get(cls_id, str(cls_id)),
                )
            )
        return DetectorOutput(detections=detections)

    async def detect(self, frame: np.ndarray, frame_idx: int) -> DetectorOutput:
        return await asyncio.to_thread(self._infer, frame)