from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DetectorConfig(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class TrackerConfig(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class InputConfig(BaseModel):
    source: str | None = None


class OutputConfig(BaseModel):
    base_dir: str | None = "data/output"
    video_dir: str = "outputs/video"
    json_dir: str = "outputs/json"


class ClearMLLoggingConfig(BaseModel):
    enabled: bool = False
    project_name: str = "cv-pipeline"


class LoggingConfig(BaseModel):
    local: bool = True
    clearml: ClearMLLoggingConfig = Field(default_factory=ClearMLLoggingConfig)


class DrawingConfig(BaseModel):
    show_bbox: bool = True
    show_roi: bool = False
    show_label: bool = True
    colors: dict[str, list[int]] = Field(
        default_factory=lambda: {
            "bbox": [0, 255, 0],
            "roi": [255, 128, 0],
            "label": [255, 255, 255],
        }
    )


class PipelineConfig(BaseModel):
    save_video: bool = True
    save_json: bool = True
    save_jsonl: bool = True
    json_format: str = "full"


class AppConfig(BaseModel):
    detector: DetectorConfig
    tracker: TrackerConfig
    classes: list[int] = Field(default_factory=list)
    input: InputConfig = Field(default_factory=InputConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    drawing: DrawingConfig = Field(default_factory=DrawingConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)


class ClearMLRuntime(BaseModel):
    task_name: str | None = None
    project_name: str | None = None
    tags: list[str] = Field(default_factory=list)
    enabled: bool | None = None


class RuntimeParams(BaseModel):
    source: list[str]
    output_name: str
    run_id: str
    experiment_name: str
    threshold: float | None = None
    conf: float | None = None
    iou: float | None = None
    classes: list[int] | None = None
    device: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    clearml: ClearMLRuntime = Field(default_factory=ClearMLRuntime)
    save_video: bool | None = None