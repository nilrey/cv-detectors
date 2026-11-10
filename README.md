Замечания:
default.yaml использует yolo+botsort как базовую схему.
cascade.yaml демонстрирует aux={"rois": ...}: segmenter -> rois -> inner detector, roi_padding и min_roi_area в params детектора.
json_format: full — заглушка под возможные варианты (например, compact), логика в json_writer.


# CV pipeline template

Единый шаблон для проектов детекции и трекинга на видео.

## Назначение

Унифицирует домен, детекторы, трекеры, вывод и логирование. Покрывает сценарии:
DETR + BoT-SORT, YOLO + ClearML, DeepLab + YOLO + IoU, а также каскады вида U-Net + YOLO.

## Структура

- `configs/` - yaml-конфиги (`default.yaml`, `cascade.yaml`).
- `src/config/` - pydantic-схемы и загрузчик.
- `src/domain/` - модели данных и константа COCO.
- `src/detectors/` - протокол и адаптеры (yolo, detr, cascade).
- `src/trackers/` - протокол и реализации (iou, botsort, yolo_track).
- `src/io/` - чтение/запись видео, json/jsonl.
- `src/logging/` - локальный логгер и ClearML.
- `src/pipeline/` - оркестратор `run` и отрисовка.
- `src/api/` - задел HTTP API.
- `src/factories.py` - сборка компонентов по конфигу.
- `main.py` - точка входа.
- `tests/` - smoke-тест.

## Добавить детектор

1. Создать класс в `src/detectors/<name>.py` с атрибутами `name`, `class_map` и методами `load`, `warmup`, `detect`.
2. Возвращать `DetectorOutput(detections=..., aux=...)`.
3. Зарегистрировать в `factories._build_single_detector`.

## Добавить трекер

1. Создать класс в `src/trackers/<name>.py` с методами `update(detections, frame, frame_idx)` и `reset()`.
2. Зарегистрировать в `factories.build_tracker`.

## Запуск

```bash
pip install -r requirements.txt
python main.py