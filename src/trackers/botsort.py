from __future__ import annotations

import asyncio

import numpy as np

from src.domain.models import Detection, Track


class BotSortTracker:
    name = "botsort"

    def __init__(self, **params):
        self.params = params
        self._impl = None
        self._known: dict[int, Track] = {}


    def _build(self):
        from boxmot import create_tracker, BotSortConfig
        from boxmot.reid import create_reid_encoder
        from pathlib import Path

        algo_keys = {
            "track_high_thresh", "track_low_thresh", "new_track_thresh",
            "max_lost_frames", "match_thresh", "proximity_thresh",
            "appearance_thresh", "frame_rate",
        }
        algo_params = {k: v for k, v in self.params.items() if k in algo_keys}
        config = BotSortConfig(**algo_params)

        with_reid = self.params.get("with_reid", False)
        if not with_reid:
            return create_tracker(spec="botsort", config=config, per_class=False)

        reid_path = Path(self.params.get("reid_weights", "weights/trackers/osnet_x0_25_msmt17.pt"))
        if not reid_path.exists():
            raise FileNotFoundError(f"ReID weights not found: {reid_path}")

        encoder = create_reid_encoder(
            "osnet-x0-25-msmt17",
            device="cpu",
            allow_download=False,
            options={"model_path": str(reid_path)},  
        )

        return create_tracker(
            spec="botsort",
            config=config,
            reid=encoder,
            per_class=False,
        )

    async def _ensure(self) -> None:
        if self._impl is None:
            self._impl = await asyncio.to_thread(self._build)

    def reset(self) -> None:
        self._impl = None
        self._known = {}

    async def update(
        self, detections: list[Detection], frame: np.ndarray, frame_idx: int
    ) -> list[Track]:
        await self._ensure()
        if detections:
            arr = np.array(
                [[*d.bbox, d.score, float(d.class_id)] for d in detections], dtype=np.float32
            )
        else:
            arr = np.empty((0, 6), dtype=np.float32)
        tracks = await asyncio.to_thread(self._impl.update, arr, frame)
        result: list[Track] = []
        for row in tracks:
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            tid = int(row[4])
            score = float(row[5]) if len(row) > 5 else 1.0
            cls_id = int(row[6]) if len(row) > 6 else 0
            prev = self._known.get(tid)
            class_name = prev.class_name if prev else str(cls_id)
            tr = Track(
                track_id=tid,
                bbox=(x1, y1, x2, y2),
                score=score,
                class_id=cls_id,
                class_name=class_name,
                is_detected=True,
            )
            self._known[tid] = tr
            result.append(tr)
        return result