from __future__ import annotations

import asyncio
import json
from pathlib import Path

from src.detectors.resdetr import ResDetrDetector


async def main() -> None:
    det = ResDetrDetector(weights="PekingU/resdetr_r50vd")
    await det.load()
    out = Path("data/output/classes_resdetr.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(det.class_map, f, ensure_ascii=False, indent=2)
    print(out)


if __name__ == "__main__":
    asyncio.run(main())