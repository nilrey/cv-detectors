from __future__ import annotations

import asyncio
import time
from pathlib import Path

from src.config.loader import load_config
from src.config.schema import ClearMLRuntime, RuntimeParams
from src.pipeline.runner import run

CONFIGS = [
    "configs/rtdetr_r18vd.yaml",
    "configs/rtdetr_r34vd.yaml",
    "configs/rtdetr_r50vd.yaml",
    "configs/rtdetr_r50vd_coco_o365.yaml",
    "configs/detr-resnet-50-dc5.yaml",
    "configs/detr-resnet-50.yaml",
    "configs/rtdetr_r101.yaml",
    "configs/rtdetr_r101vd_coco_o365.yaml",
    "configs/detr-resnet-101-dc5.yaml",
    "configs/detr-resnet-101.yaml",
]

def _build_params(name: str) -> RuntimeParams:
    return RuntimeParams(
        source=["data/input/spb-10-sec.mp4"],
        output_name=f"run_{name}",
        run_id=f"run_{name}",
        experiment_name="baseline",
        threshold=None,
        conf=None,
        iou=None,
        classes=None,
        device=None,
        extra={},
        clearml=ClearMLRuntime(task_name=name, tags=["baseline"], enabled=None),
        save_video=None,
    )

def main() -> None:
    for cfg_path in CONFIGS:
        name = Path(cfg_path).stem
        print(f"\n{'='*50}")
        print(f"STARTING: {name}")
        
        config = load_config(cfg_path)
        params = _build_params(name)
        
        start_time = time.time()
        try:
            batch = asyncio.run(run(params, config))
            elapsed = time.time() - start_time
            print("result:")
            print(f"  success: {batch.success_count}")
            print(f"  failed: {batch.failed_count}")
            print(f"  errors: {len(batch.errors)}")
            print(f"  time: {elapsed:.2f}s")
        except Exception as e:
            print(f"ERROR processing {name}: {e}")

if __name__ == "__main__":
    main()