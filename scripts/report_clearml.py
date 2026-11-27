from __future__ import annotations

import argparse
import json
from pathlib import Path

from clearml import Task

from src.eval.report import build_stats, render_bars


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sync", default="data/sync/results/sync.jsonl")
    p.add_argument("--out", default="data/sync/results/report")
    p.add_argument("--project", default="Синхронизированная разметка")
    p.add_argument("--task", default="Yolo 12n")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = build_stats(args.sync)
    stats_path = out_dir / "stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    plots = render_bars(stats, out_dir)

    task = Task.init(project_name=args.project, task_name=args.task)
    task.upload_artifact("sync", args.sync)
    task.upload_artifact("stats", str(stats_path))
    for pth in plots:
        task.upload_artifact(Path(pth).stem, pth)

    log = task.get_logger()
    t = stats["totals"]
    log.report_scalar("metrics", "tp", t["tp"], 0)
    log.report_scalar("metrics", "fp", t["fp"], 0)
    log.report_scalar("metrics", "fn", t["fn"], 0)
    log.report_scalar("metrics", "precision", t["precision"], 0)
    log.report_scalar("metrics", "recall", t["recall"], 0)
    log.report_scalar("metrics", "f1", t["f1"], 0)

    log.report_histogram(
        title="TP/FP/FN",
        series="count",
        values=[t["tp"], t["fp"], t["fn"]],
        xlabels=["TP", "FP", "FN"],
        xaxis="metric",
        yaxis="count",
    )
    log.report_histogram(
        title="Обнаруженные объекты по классам",
        series="count",
        values=list(stats["det_objs_by_cls"].values()),
        xlabels=list(stats["det_objs_by_cls"].keys()),
        xaxis="class",
        yaxis="count",
    )
    log.report_histogram(
        title="Ссылка на объекты по классу",
        series="count",
        values=list(stats["ref_objs_by_cls"].values()),
        xlabels=list(stats["ref_objs_by_cls"].keys()),
        xaxis="class",
        yaxis="count",
    )
    log.report_histogram(
        title="Детекция уникальных треков по классам",
        series="count",
        values=list(stats["det_tracks_by_cls"].values()),
        xlabels=list(stats["det_tracks_by_cls"].keys()),
        xaxis="class",
        yaxis="count",
    )
    log.report_histogram(
        title="Ссылочные уникальные треки по классам",
        series="count",
        values=list(stats["ref_tracks_by_cls"].values()),
        xlabels=list(stats["ref_tracks_by_cls"].keys()),
        xaxis="class",
        yaxis="count",
    )
    log.report_histogram(
        title="Обнаруженное распределение достоверности",
        series="count",
        values=list(stats["conf_det"].values()),
        xlabels=list(stats["conf_det"].keys()),
        xaxis="confidence",
        yaxis="count",
    )
    log.report_histogram(
        title="TP распределение достоверности",
        series="count",
        values=list(stats["conf_tp"].values()),
        xlabels=list(stats["conf_tp"].keys()),
        xaxis="confidence",
        yaxis="count",
    )
    log.report_histogram(
        title="IoU достоверности (TP)",
        series="count",
        values=list(stats["iou_hist"].values()),
        xlabels=list(stats["iou_hist"].keys()),
        xaxis="iou bin",
        yaxis="count",
    )

    log.report_single_value("frames", t["frames"])
    log.report_single_value("tp", t["tp"])
    log.report_single_value("fp", t["fp"])
    log.report_single_value("fn", t["fn"])
    log.report_single_value("precision", t["precision"])
    log.report_single_value("recall", t["recall"])
    log.report_single_value("f1", t["f1"])

    task.close()
    print("report done:", out_dir)


if __name__ == "__main__":
    main()