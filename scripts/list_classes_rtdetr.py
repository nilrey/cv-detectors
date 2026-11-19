# Получение всех классов модели RT-DETR 
# Запуск из корня: python -m scripts.list_classes_rtdetr
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from src.detectors.rtdetr import RtDetrDetector


async def main() -> None:
    det = RtDetrDetector(weights="PekingU/rtdetr_r50vd")
    await det.load()
    out = Path("data/output/classes_rtdetr.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(det.class_map, f, ensure_ascii=False, indent=2)
    print(out)


if __name__ == "__main__":
    asyncio.run(main())