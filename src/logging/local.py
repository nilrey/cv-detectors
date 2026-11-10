from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


class LocalLogger:
    name = "local"

    def __init__(self, log_dir: str | Path = "outputs/logs", echo: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self._fh = None

    def _emit(self, event: str, **data: Any) -> None:
        record = {"event": event, **data}
        line = json.dumps(record, ensure_ascii=False)
        if self._fh is not None:
            self._fh.write(line + "\n")
            self._fh.flush()
        if self.echo:
            sys.stdout.write(line + "\n")

    def start_video(self, source: str) -> None:
        if self._fh is not None:
            self._fh.close()
        path = self.log_dir / f"{Path(source).stem}.jsonl"
        self._fh = path.open("w", encoding="utf-8")

    def event(self, name: str, **data: Any) -> None:
        self._emit(name, **data)

    def error(self, message: str) -> None:
        self._emit("error", message=message)

    def on_video_start(self, video_result) -> None:
        return

    def on_frame(self, frame_result) -> None:
        return

    def on_video_end(self) -> None:
        self._emit("video_end")

    def on_batch_end(self, batch) -> None:
        self._emit(
            "batch_end",
            success=batch.success_count,
            failed=batch.failed_count,
            errors=len(batch.errors),
        )

    def finalize(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def reset(self) -> None:
        self.finalize()