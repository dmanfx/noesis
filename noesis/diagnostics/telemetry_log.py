from __future__ import annotations

import json
import os
import queue
import re
import stat
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import logging

from noesis_core.private_paths import (
    ensure_private_directory,
    prepare_private_writable_file,
    validate_private_file,
)
from noesis_core.runtime_secrets import public_pipeline_config

logger = logging.getLogger(__name__)

try:
    from noesis.models import convert_numpy_types
except Exception:  # pragma: no cover - fallback when frontend utilities absent
    def convert_numpy_types(payload: Any) -> Any:  # type: ignore[override]
        return payload


_ENV_TRUE = {"1", "true", "yes", "y", "on"}
_SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_DEFAULT_MAX_BYTES = 64 * 1024 * 1024
_DEFAULT_MAX_FILES = 8
_PUBLIC_ENV_NAMES = (
    "NOESIS_BEV_FRAME",
    "NOESIS_CALIBRATION_POSE_ONLY",
    "NOESIS_PGIE_PROFILE",
    "NOESIS_REID_ENABLED",
    "NOESIS_TRACKING_MODE",
    "NOESIS_V3DT_AUTOGEN_CAMINFO",
    "NOESIS_V3DT_CAMINFO_INVERT_E",
    "NOESIS_V3DT_CAMINFO_MATRIX_TYPE",
    "NOESIS_V3DT_CAMINFO_WORLD_AXES",
    "NOESIS_V3DT_CAMINFO_WORLD_SCALE",
    "NOESIS_V3DT_CAMINFO_Y_FLIP",
    "NOESIS_V3DT_META_EXTRACT",
)


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _bounded_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = str(os.environ.get(name, default) or default).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def public_v3dt_environment(
    environment: Mapping[str, str] | None = None,
) -> Dict[str, str]:
    """Return only non-secret runtime switches useful to V3DT forensics."""

    values = os.environ if environment is None else environment
    return {name: str(values[name]) for name in _PUBLIC_ENV_NAMES if name in values}


def build_v3dt_session_start_payload(
    *,
    pipeline_config: Mapping[str, Any],
    camera_labels: Mapping[Any, Any],
    sensor_id_map: Mapping[Any, Any],
    environment: Mapping[str, str] | None = None,
    observed_at: float | None = None,
) -> Dict[str, Any]:
    """Build a serialization-safe V3DT session header without runtime secrets."""

    values = os.environ if environment is None else environment
    return {
        "type": "v3dt_session_start",
        "ts": float(time.time() if observed_at is None else observed_at),
        "camera_labels": dict(camera_labels),
        "sensor_id_map": dict(sensor_id_map),
        "pipeline_config": public_pipeline_config(pipeline_config),
        "env": public_v3dt_environment(values),
    }


def _prune_old_sessions(directory: Path, current: Path, max_files: int) -> None:
    candidates: list[Path] = []
    for path in directory.glob("v3dt_frames_*.ndjson"):
        if path == current:
            continue
        candidates.append(validate_private_file(path, label="V3DT diagnostic log"))
    candidates.sort(key=lambda path: path.lstat().st_mtime_ns)
    excess = max(0, len(candidates) - max(0, int(max_files) - 1))
    for path in candidates[:excess]:
        before = path.lstat()
        current_info = path.lstat()
        if (before.st_dev, before.st_ino) != (current_info.st_dev, current_info.st_ino):
            raise RuntimeError("V3DT diagnostic log changed during retention")
        path.unlink()


@dataclass
class TrackingDiagnosticsLogger:
    output_path: Path
    flush_every: int = 8
    max_queue: int = 2048
    max_bytes: int = _DEFAULT_MAX_BYTES
    max_files: int = _DEFAULT_MAX_FILES
    _queue: "queue.Queue[Optional[str]]" = field(default_factory=queue.Queue, init=False)
    _thread: Optional[threading.Thread] = field(default=None, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _capacity_exhausted: threading.Event = field(default_factory=threading.Event, init=False)
    _dropped: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.flush_every = int(self.flush_every)
        self.max_queue = int(self.max_queue)
        self.max_bytes = int(self.max_bytes)
        self.max_files = int(self.max_files)
        if self.flush_every < 1:
            raise ValueError("flush_every must be positive")
        if self.max_queue < 1:
            raise ValueError("max_queue must be positive")
        if self.max_bytes < 1024:
            raise ValueError("max_bytes must be at least 1024")
        if self.max_files < 1:
            raise ValueError("max_files must be positive")

        candidate = Path(self.output_path).expanduser()
        directory = ensure_private_directory(
            candidate.parent,
            label="V3DT diagnostic log parent",
        )
        candidate = directory / candidate.name
        _prune_old_sessions(directory, candidate, self.max_files)
        self.output_path = prepare_private_writable_file(
            candidate,
            label="V3DT diagnostic log",
        )
        self._queue = queue.Queue(maxsize=int(self.max_queue))
        self._thread = threading.Thread(target=self._writer_loop, name="V3DT-DiagLogger", daemon=True)
        self._thread.start()

    @classmethod
    def from_env(cls) -> Optional["TrackingDiagnosticsLogger"]:
        flag = str(os.environ.get("NOESIS_V3DT_DIAG_LOG", "") or "").strip()
        if not flag:
            return None

        if flag.lower() in _ENV_TRUE:
            out_dir = str(
                os.environ.get(
                    "NOESIS_V3DT_DIAG_DIR",
                    Path.home() / ".local" / "state" / "noesis" / "diagnostics",
                )
            )
            session = str(os.environ.get("NOESIS_V3DT_DIAG_SESSION", "") or "").strip() or _now_stamp()
            if not _SESSION_RE.fullmatch(session):
                raise ValueError("NOESIS_V3DT_DIAG_SESSION contains unsafe characters")
            output_path = Path(out_dir) / f"v3dt_frames_{session}.ndjson"
            return cls(
                output_path=output_path,
                max_bytes=_bounded_int_env(
                    "NOESIS_V3DT_DIAG_MAX_BYTES",
                    _DEFAULT_MAX_BYTES,
                    1024 * 1024,
                    512 * 1024 * 1024,
                ),
                max_files=_bounded_int_env(
                    "NOESIS_V3DT_DIAG_MAX_FILES",
                    _DEFAULT_MAX_FILES,
                    1,
                    64,
                ),
            )

        output_path = Path(flag)
        if output_path.suffix.lower() != ".ndjson":
            out_dir = Path(flag)
            session = str(os.environ.get("NOESIS_V3DT_DIAG_SESSION", "") or "").strip() or _now_stamp()
            if not _SESSION_RE.fullmatch(session):
                raise ValueError("NOESIS_V3DT_DIAG_SESSION contains unsafe characters")
            output_path = out_dir / f"v3dt_frames_{session}.ndjson"
        return cls(
            output_path=output_path,
            max_bytes=_bounded_int_env(
                "NOESIS_V3DT_DIAG_MAX_BYTES",
                _DEFAULT_MAX_BYTES,
                1024 * 1024,
                512 * 1024 * 1024,
            ),
            max_files=_bounded_int_env(
                "NOESIS_V3DT_DIAG_MAX_FILES",
                _DEFAULT_MAX_FILES,
                1,
                64,
            ),
        )

    def log_event(self, payload: Dict[str, Any]) -> None:
        self._enqueue(payload)

    def log_frame(self, payload: Dict[str, Any]) -> None:
        self._enqueue(payload)

    def close(self, timeout: float = 2.0) -> None:
        if not self._stop.is_set():
            self._stop.set()
            try:
                self._queue.put_nowait(None)
            except Exception:
                pass
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                raise RuntimeError(
                    "V3DT diagnostics writer did not stop within the shutdown bound"
                )
        if self._dropped > 0:
            logger.warning(
                "V3DT diagnostics dropped %d records (queue or session capacity)",
                int(self._dropped),
            )

    def _enqueue(self, payload: Dict[str, Any]) -> None:
        if self._capacity_exhausted.is_set():
            self._dropped += 1
            return
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
        buffer: list[str] = []
        descriptor = -1
        try:
            expected_path = validate_private_file(
                self.output_path,
                label="V3DT diagnostic log",
            )
            expected = expected_path.lstat()
            descriptor = os.open(
                expected_path,
                os.O_WRONLY
                | os.O_APPEND
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            opened = os.fstat(descriptor)
            if (
                (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino)
                or not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
            ):
                raise RuntimeError("V3DT diagnostic log changed while opening")
            bytes_written = int(opened.st_size)

            with os.fdopen(descriptor, "a", encoding="utf-8", closefd=True) as handle:
                descriptor = -1

                def _flush() -> None:
                    nonlocal bytes_written
                    if not buffer:
                        return
                    chunk = "\n".join(buffer) + "\n"
                    chunk_bytes = len(chunk.encode("utf-8"))
                    if bytes_written + chunk_bytes > self.max_bytes:
                        self._dropped += len(buffer)
                        self._capacity_exhausted.set()
                        buffer.clear()
                        logger.warning(
                            "V3DT diagnostics reached the %d-byte session limit; dropping later records",
                            self.max_bytes,
                        )
                        return
                    handle.write(chunk)
                    handle.flush()
                    bytes_written += chunk_bytes
                    buffer.clear()

                while True:
                    try:
                        item = self._queue.get(timeout=0.5)
                    except queue.Empty:
                        item = None
                    if item is None:
                        _flush()
                        if self._stop.is_set():
                            break
                        continue
                    buffer.append(item)
                    if len(buffer) >= self.flush_every:
                        _flush()
                os.fsync(handle.fileno())
        except Exception:
            logger.exception("V3DT diagnostics writer failed")
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def __del__(self) -> None:
        try:
            self.close(timeout=0.5)
        except Exception:
            pass


__all__ = [
    "TrackingDiagnosticsLogger",
    "build_v3dt_session_start_payload",
    "public_v3dt_environment",
]
