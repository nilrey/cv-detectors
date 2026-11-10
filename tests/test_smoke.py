from __future__ import annotations

import asyncio

import numpy as np

from src.domain.models import Detection
from src.trackers.iou import IouTracker


def test_iou_tracker_basic():
    tracker = IouTracker(iou_threshold=0.3, max_age=2, min_hits=1)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    dets = [Detection(bbox=(10, 10, 50, 50), score=0.9, class_id=0, class_name="person")]
    tracks = asyncio.run(tracker.update(dets, frame, 0))
    assert len(tracks) == 1
    assert tracks[0].is_detected
    tid = tracks[0].track_id

    tracks2 = asyncio.run(tracker.update(dets, frame, 1))
    assert any(t.track_id == tid for t in tracks2)

    tracker.reset()
    tracks3 = asyncio.run(tracker.update(dets, frame, 2))
    assert tracks3[0].track_id != tid