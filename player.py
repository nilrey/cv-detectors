from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

COLOR_TP = (0, 255, 0)
COLOR_FP = (0, 0, 255)
COLOR_FN = (0, 255, 255)


def load_jsonl(path: str | Path) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[int(row["frame_idx"])] = row.get("objects", [])
    return out


def _color(match: str) -> tuple[int, int, int]:
    if match == "tp":
        return COLOR_TP
    if match == "fp":
        return COLOR_FP
    return COLOR_FN


def _label(o: dict) -> str:
    parts = [str(o.get("match", "-"))]
    # parts.append(f"ref={o.get('ref_track_id')}")
    parts.append(f"trk={o.get('track_id')}")
    # score = o.get("score")
    # parts.append(f"sc={score:.2f}" if score is not None else "sc=-")
    # iou = o.get("iou")
    # parts.append(f"iou={iou:.2f}" if iou is not None else "iou=-")
    return " ".join(parts)


def draw_objects(frame, objects, ref_ids, trk_ids, matches):
    for o in objects:
        if matches and o.get("match") not in matches:
            continue
        if ref_ids and o.get("ref_track_id") not in ref_ids:
            continue
        if trk_ids and o.get("track_id") not in trk_ids:
            continue
        x1, y1, x2, y2 = o["bbox"]
        color = _color(o.get("match"))
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
        cv2.putText(
            frame,
            _label(o),
            (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )
    return frame
 

def main(json_path, video_path, output_path, ref_ids, trk_ids, matches):
    frames_map = load_jsonl(json_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("cannot open video")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        objs = frames_map.get(idx, [])
        frame = draw_objects(frame, objs, ref_ids, trk_ids, matches)
        out.write(frame)
        if idx % 30 == 0:
            print(f"frame {idx}/{total}")
        idx += 1

    cap.release()
    out.release()
    print(f"saved {output_path}")


def _out_name(matches, trk_ids) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [ts]
    if matches:
        parts.append("-".join(sorted(matches)))
    if trk_ids:
        parts.append("-".join(str(i) for i in sorted(trk_ids)))
    return "_".join(parts) + ".mp4"

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    PATH_ORIG_VIDEO = "data/input/spb_gostiny_dvor_001.mp4"
    PATH_SYNC = "data/sync"
    p.add_argument("--json", default=f"{PATH_SYNC}/results/sync.jsonl")
    p.add_argument("--video", default=PATH_ORIG_VIDEO)
    p.add_argument("--out", default=None)
    p.add_argument("--ref-ids", nargs="*", type=int, default=[])
    p.add_argument("--trk-ids", nargs="*", type=int, default=[])
    p.add_argument("--matches", nargs="*", default=["tp", "fp", "fn"])
    args = p.parse_args()

    matches = set(args.matches)
    trk_ids = set(args.trk_ids)
    out_path = args.out or f"{PATH_SYNC}/video/{_out_name(matches, trk_ids)}"
    main(args.json, args.video, out_path, set(args.ref_ids), trk_ids, matches)