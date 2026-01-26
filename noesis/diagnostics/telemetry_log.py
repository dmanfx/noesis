from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import logging

logger = logging.getLogger(__name__)

try:
    from models import convert_numpy_types
except Exception:  # pragma: no cover - fallback when frontend utilities absent
    def convert_numpy_types(payload: Any) -> Any:  # type: ignore[override]
        return payload


_ENV_TRUE = {"1", "true", "yes", "y", "on"}


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


@dataclass
class TrackingDiagnosticsLogger:
    output_path: Path
    flush_every: int = 8
    max_queue: int = 2048
    _queue: "queue.Queue[Optional[str]]" = field(default_factory=queue.Queue, init=False)
    _thread: Optional[threading.Thread] = field(default=None, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _dropped: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.output_path = Path(self.output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._queue = queue.Queue(maxsize=int(self.max_queue))
        self._thread = threading.Thread(target=self._writer_loop, name="V3DT-DiagLogger", daemon=True)
        self._thread.start()

    @classmethod
    def from_env(cls) -> Optional["TrackingDiagnosticsLogger"]:
        flag = str(os.environ.get("NOESIS_V3DT_DIAG_LOG", "") or "").strip()
        if not flag:
            return None

        if flag.lower() in _ENV_TRUE:
            out_dir = str(os.environ.get("NOESIS_V3DT_DIAG_DIR", "diagnostics") or "diagnostics")
            session = str(os.environ.get("NOESIS_V3DT_DIAG_SESSION", "") or "").strip() or _now_stamp()
            output_path = Path(out_dir) / f"v3dt_frames_{session}.ndjson"
            return cls(output_path=output_path)

        output_path = Path(flag)
        if output_path.suffix.lower() != ".ndjson":
            out_dir = Path(flag)
            session = str(os.environ.get("NOESIS_V3DT_DIAG_SESSION", "") or "").strip() or _now_stamp()
            output_path = out_dir / f"v3dt_frames_{session}.ndjson"
        return cls(output_path=output_path)

    def log_event(self, payload: Dict[str, Any]) -> None:
        self._enqueue(payload)

    def log_frame(self, payload: Dict[str, Any]) -> None:
        self._enqueue(payload)

    def close(self, timeout: float = 2.0) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        if self._dropped > 0:
            logger.warning("V3DT diagnostics dropped %d records (queue full)", int(self._dropped))

    def _enqueue(self, payload: Dict[str, Any]) -> None:
        payload = convert_numpy_types(payload)
        try:
            record = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
        except Exception:
            return
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self._dropped += 1
            if self._dropped % 200 == 1:
                logger.warning("V3DT diagnostics queue full; dropping records (dropped=%d)", int(self._dropped))

    def _writer_loop(self) -> None:
        buffer = []
        try:
            with self.output_path.open("a", encoding="utf-8") as handle:
                while True:
                    try:
                        item = self._queue.get(timeout=0.5)
                    except queue.Empty:
                        item = None
                    if item is None:
                        if buffer:
                            handle.write("\n".join(buffer) + "\n")
                            handle.flush()
                            buffer.clear()
                        if self._stop.is_set():
                            break
                        continue
                    buffer.append(item)
                    if len(buffer) >= max(1, int(self.flush_every)):
                        handle.write("\n".join(buffer) + "\n")
                        handle.flush()
                        buffer.clear()
        except Exception:
            logger.exception("V3DT diagnostics writer failed")

    def __del__(self) -> None:
        try:
            self.close(timeout=0.5)
        except Exception:
            pass
