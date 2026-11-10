from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ClearMLRequest(BaseModel):
    enabled: bool | None = None
    project_name: str | None = None
    task_name: str | None = None
    tags: list[str] = Field(default_factory=list)


class RuntimeRequest(BaseModel):
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
    save_video: bool | None = None
    clearml: ClearMLRequest = Field(default_factory=ClearMLRequest)


class VideoSummary(BaseModel):
    source: str
    total_frames: int
    processed_frames: int


class BatchResponse(BaseModel):
    success_count: int
    failed_count: int
    errors: list[str]
    videos: list[VideoSummary] = Field(default_factory=list)