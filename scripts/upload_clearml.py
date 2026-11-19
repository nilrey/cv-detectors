from __future__ import annotations

import argparse
import json
from pathlib import Path

from clearml import Task


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sync", default="data/sync/results/sync.jsonl")
    p.add_argument("--metrics", default="data/sync/results/metrics.json")
    p.add_argument("--project", default="Синхронизированная разметка")
    p.add_argument("--task", default="RT-DETR r18vd")
    args = p.parse_args()

    task = Task.init(project_name=args.project, task_name=args.task)

    sync_path = Path(args.sync)
    metrics_path = Path(args.metrics)

    task.upload_artifact("sync", str(sync_path))
    task.upload_artifact("metrics", str(metrics_path))

    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            m = json.load(f)
        logger = task.get_logger()
        logger.report_scalar("metrics", "tp", m["tp"], 0)
        logger.report_scalar("metrics", "fp", m["fp"], 0)
        logger.report_scalar("metrics", "fn", m["fn"], 0)
        logger.report_scalar("metrics", "precision", m["precision"], 0)
        logger.report_scalar("metrics", "recall", m["recall"], 0)
        logger.report_scalar("metrics", "f1", m["f1"], 0)
        for k, v in m.get("iou_hist", {}).items():
            logger.report_scalar("iou_hist", k, v, 0)

    task.close()
    print("uploaded")


if __name__ == "__main__":
    main()