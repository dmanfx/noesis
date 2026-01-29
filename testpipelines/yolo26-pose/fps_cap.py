"""Frame-rate cap for file-based test pipelines."""

from __future__ import annotations

import time
from typing import Any

from pyservicemaker import BatchMetadataOperator


class FpsCap(BatchMetadataOperator):
    def __init__(self, target_fps: float) -> None:
        super().__init__()
        if target_fps <= 0:
            raise ValueError("target_fps must be > 0")
        self._period = 1.0 / float(target_fps)
        self._next_time: float | None = None

    def handle_metadata(self, batch_meta: Any) -> None:  # noqa: ARG002
        now = time.perf_counter()
        if self._next_time is None:
            self._next_time = now + self._period
            return
        if now < self._next_time:
            time.sleep(self._next_time - now)
        self._next_time += self._period
