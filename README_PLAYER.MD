## Файл player.py

# Наложение результатов детекции и синхронизации на видео.

Читает исходное видео и jsonl с объектами, рисует bbox с подписями, сохраняет новое видео.

## Вход

- Видео: исходный mp4.
- jsonl: результат синхронизации (`sync.jsonl`) или обычный результат модели.
  В синхронизированном файле у объектов есть поля `match`, `ref_track_id`, `track_id`, `score`, `iou`.
  В обычном результате модели есть `track_id`, `class_name`, `score`, `bbox`, но нет `match` и `ref_track_id`.
  player работает с обоими: отсутствующие поля подставляет как `-`.

Формат строки jsonl:
```bash
{"frame_idx": 0, "objects": [{"match": "tp", "ref_track_id": 121, "track_id": 1, "class_name": "car", "score": 0.94, "bbox": [336.1, 323.7, 501.1, 417.5], "iou": 0.98}]}
```

## Параметры
```bash
    --json — путь к jsonl. По умолчанию data/sync/results/sync.jsonl.

    --video — путь к видео. По умолчанию data/input/spb_gostiny_dvor_001.mp4.

    --out — путь к выходному mp4. Если не задан, имя собирается автоматически.

    --matches — какие объекты рисовать: tp, fp, fn. Можно несколько. По умолчанию все три.

    --ref-ids — фильтр по эталонным id (ref_track_id). Можно несколько.

    --trk-ids — фильтр по id модели (track_id). Можно несколько.
```
Фильтры комбинируются через логическое И.

## Имя выходного файла

Если --out не задан: <timestamp>_<matches>_<trk>.mp4 в папке PATH_SYNC/video/.

    <timestamp> — YYYYMMDD_HHMMSS.

    <matches> — значения через -, отсортированы.

    <trk> — значения --trk-ids через -, отсортированы.

    Пустые сегменты пропускаются.

## Примеры:

    20261120_103008_tp-fp-fn.mp4

    20261120_103008_fp_2-7-21.mp4

## Цвета и подписи

Цвет bbox:

    tp — зеленый

    fp — красный

    fn — желтый

Подпись объекта:
```bash
<match> ref=<ref_track_id> trk=<track_id> sc=<score> iou=<iou>
```
Если поля нет, вместо значения ставится -.

Собирается в _label(o).

## Примеры запуска

Все объекты:
```bash
python player.py
```
Только FP:
```bash
python player.py --matches fp
```
TP и FN:
```bash
python player.py --matches tp fn
```
Конкретные треки модели:
```bash
python player.py --trk-ids 2 7 21
```
Комбинация:
```bash
python player.py --matches tp fp --trk-ids 52 35 66 93 40
```
Явный выходной файл (необяз. по умолчанию пишется в timestamp файл):
```bash
python player.py --out data/sync/video/my_overlay.mp4
```
## Поведение и ограничения

    Кадры сопоставляются по индексу: frame_idx в jsonl к номеру кадра видео, начиная с 0.

    Если для кадра нет объектов, кадр пишется без разметки.

    Работает и на несинхронизированном jsonl: отсутствующие поля отображаются как -.

    Выходной файл — mp4 (кодек mp4v).