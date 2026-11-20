from __future__ import annotations

import argparse

from src.eval.sync import sync


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ref", default="data/sync/reference/annotation_gostinny_dvor_001_completed.json")
    p.add_argument("--det", default="data/sync/input/spb_gostiny_dvor_001.jsonl")
    p.add_argument("--out", default="data/sync/results")
    args = p.parse_args()

    res = sync(args.ref, args.det, args.out)
    print("sync:", res["sync"])
    print("metrics:", res["metrics"])
    s = res["summary"]
    print(
        "tp/fp/fn:",
        s["tp"],
        s["fp"],
        s["fn"],
        "precision:",
        round(s["precision"], 4),
        "recall:",
        round(s["recall"], 4),
        "f1:",
        round(s["f1"], 4),
    )


if __name__ == "__main__":
    main()