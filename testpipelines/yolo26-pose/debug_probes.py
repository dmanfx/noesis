"""Debug probes for YOLO26 pose test pipeline."""

from __future__ import annotations

import logging
import time
from typing import Any

from pyservicemaker import BatchMetadataOperator

logger = logging.getLogger(__name__)


class FrameCounter(BatchMetadataOperator):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self._last_log = 0.0
        self._frames = 0
        self._batches = 0

    def handle_metadata(self, batch_meta: Any) -> None:
        frame_items = getattr(batch_meta, "frame_items", None)
        if frame_items is None:
            return
        self._batches += 1
        for _ in frame_items:
            self._frames += 1

        now = time.time()
        if (now - self._last_log) >= 1.0:
            logger.info(
                "%s: batches=%d frames=%d",
                self.name,
                self._batches,
                self._frames,
            )
            self._batches = 0
            self._frames = 0
            self._last_log = now
