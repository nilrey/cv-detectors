from __future__ import annotations

import asyncio

import numpy as np

from src.detectors._device import resolve_device
from src.domain.coco import COCO_ID_TO_NAME
from src.domain.models import Detection, DetectorOutput


class RtDetrDetector:
    name = "rtdetr"
    class_map = COCO_ID_TO_NAME

    def __init__(
        self,
        weights: str = "PekingU/rtdetr_r50vd",
        device: str = "auto",
        threshold: float = 0.5,
        class_map: dict[int, str] | None = None,
    ):
        self.weights = weights
        self.device = device
        self.threshold = threshold
        self._map = class_map
        self._excluded: set[str] = set()
        self._processor = None
        self._model = None

    async def load(self) -> None:
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

        self.device = resolve_device(self.device)
        cache_dir = "weights/hub"

        self._processor = RTDetrImageProcessor.from_pretrained(self.weights, cache_dir=cache_dir)
        self._model = RTDetrForObjectDetection.from_pretrained(self.weights, cache_dir=cache_dir)
        self._model.to(self.device)
        self._model.eval()

    async def warmup(self) -> None:
        return

    def _infer(self, frame: np.ndarray) -> DetectorOutput:
        import torch
        from PIL import Image
        from torchvision.ops import nms

        image = Image.fromarray(frame[:, :, ::-1])
        inputs = self._processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        results = self._processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.threshold
        )[0]

        boxes = results["boxes"]
        scores = results["scores"]
        labels = results["labels"]

        keep = nms(boxes, scores, iou_threshold=0.5)

        detections: list[Detection] = []
        for idx in keep:
            score = scores[idx]
            label = labels[idx]
            box = boxes[idx]
            
            cls_id = int(label.item())
            native_name = COCO_ID_TO_NAME.get(cls_id, str(cls_id))
            
            if self._map:
                target = self._map.get(cls_id)
                if target is None:
                    self._excluded.add(native_name)
                    continue
                cls_name = target
            else:
                cls_name = native_name
                
            detections.append(
                Detection(
                    bbox=tuple(float(x) for x in box.tolist()),
                    score=float(score.item()),
                    class_id=cls_id,
                    class_name=cls_name,
                )
            )
        return DetectorOutput(detections=detections)

    async def detect(self, frame: np.ndarray, frame_idx: int) -> DetectorOutput:
        return await asyncio.to_thread(self._infer, frame)