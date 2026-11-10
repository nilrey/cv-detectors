from __future__ import annotations

import asyncio
from pathlib import Path

from src.config.loader import load_config
from src.config.schema import ClearMLRuntime, RuntimeParams
from src.pipeline.runner import run


def _build_params(config_path: str) -> tuple[RuntimeParams, object]:
    config = load_config(config_path)
    params = RuntimeParams(
        source=[
            "data/input/spb-10-sec.mp4",
        ],
        output_name="default_run",
        run_id="run_001",
        experiment_name="baseline",
        threshold=None,
        conf=None,
        iou=None,
        classes=None,
        device=None,
        extra={},
        clearml=ClearMLRuntime(task_name="baseline", tags=["baseline"], enabled=None),
        save_video=None,
    )
    return params, config


def main() -> None:
    config_path = str(Path("configs/default.yaml"))
    params, config = _build_params(config_path)
    batch = asyncio.run(run(params, config))
    print("result:")
    print(f"  success: {batch.success_count}")
    print(f"  failed: {batch.failed_count}")
    print(f"  errors: {len(batch.errors)}")


if __name__ == "__main__":
    main()