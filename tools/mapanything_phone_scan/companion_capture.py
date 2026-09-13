"""Bounded paired capture of one canonical static camera and tracking stream.

This module is intentionally independent of the phone reconstruction workers.
It owns the short lived companion session used while a browser phone walk is
recorded.  The static stream is remuxed in its encoded form; no decoded frame
branch or re-encode is created here.  Runtime source and tracking adapters are
injected so this service cannot accidentally become a second perception
authority.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import threading
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
from urllib.parse import urlsplit, urlunsplit

from .static_capture_sources import (
    CanonicalTrackingRecorder,
    StaticCaptureObserver,
    StaticCaptureSourceError,
    StaticSourceAuthority,
    list_active_camera_sources,
    resolve_active_source_authority,
)
from noesis_core.runtime_secrets import redact_runtime_secrets


COMPANION_SCHEMA = "noesis.companion_capture.session.v1"
COMPANION_SESSION_PATTERN = re.compile(r"^companion-[0-9]{8}-[0-9]{6}-[a-f0-9]{8}$")
COMPANION_CAPTURE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
MAX_MARKER_BYTES = 16 * 1024
MAX_CLOCK_EXCHANGES = 256
MAX_MARKERS = 10_000
MAX_TRACKING_RECORDS = 1_000_000
MAX_PACKET_RECORDS = 2_000_000


class CompanionCaptureError(RuntimeError):
    """Base class for fail-closed companion capture errors."""


class CompanionCaptureBusy(CompanionCaptureError):
    """A different session currently owns the single companion lane."""


class CompanionCaptureConflict(CompanionCaptureError):
    """An idempotency key or phone archive conflicts with prior evidence."""


@dataclass(frozen=True)
class CompanionCaptureLimits:
    max_duration_s: float = 15 * 60.0
    lease_s: float = 45.0
    readiness_timeout_s: float = 15.0
    max_session_bytes: int = 8 * 1024 * 1024 * 1024
    max_tracking_records: int = MAX_TRACKING_RECORDS
    max_packet_records: int = MAX_PACKET_RECORDS
    max_markers: int = MAX_MARKERS

    def __post_init__(self) -> None:
        if self.max_duration_s <= 0 or self.max_duration_s > 15 * 60.0:
            raise ValueError("companion max_duration_s must be in (0, 900]")
        if self.lease_s <= 0 or self.lease_s > 45.0:
            raise ValueError("companion lease_s must be in (0, 45]")
        if self.readiness_timeout_s <= 0 or self.readiness_timeout_s > 15.0:
            raise ValueError("companion readiness_timeout_s must be in (0, 15]")
        if self.max_session_bytes <= 0:
            raise ValueError("companion max_session_bytes must be positive")
        if not 1 <= self.max_tracking_records <= MAX_TRACKING_RECORDS:
            raise ValueError("companion max_tracking_records is out of bounds")
        if not 1 <= self.max_packet_records <= MAX_PACKET_RECORDS:
            raise ValueError("companion max_packet_records is out of bounds")
        if not 1 <= self.max_markers <= MAX_MARKERS:
            raise ValueError("companion max_markers is out of bounds")


@dataclass(frozen=True)
class CompanionCamera:
    camera_id: str
    label: str
    source_reference: str
    source_authority: Mapping[str, Any]
    source_id: int | None = None


class RecorderProtocol(Protocol):
    def start(self) -> None: ...

    def stop(self, reason: str = "user") -> Mapping[str, Any] | None: ...

    def status(self) -> Mapping[str, Any]: ...


class TrackingObserverProtocol(Protocol):
    def start(self) -> None: ...

    def stop(self, reason: str = "user") -> Mapping[str, Any] | None: ...

    def status(self) -> Mapping[str, Any]: ...


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_json(value: Any, *, max_bytes: int = MAX_MARKER_BYTES) -> Any:
    try:
        encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise CompanionCaptureError("marker data must be JSON serializable") from exc
    if len(encoded.encode("utf-8")) > max_bytes:
        raise CompanionCaptureError("marker data exceeds its size limit")
    return json.loads(encoded)


def _redact_uri(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return "<redacted>"
    if parsed.scheme.lower() not in {"rtsp", "rtsps"}:
        return value
    hostname = parsed.hostname or "unknown"
    if parsed.port:
        hostname = f"{hostname}:{parsed.port}"
    return urlunsplit((parsed.scheme, hostname, parsed.path, parsed.query, parsed.fragment))


def _public_authority(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Retain runtime/source provenance without persisting a camera secret."""

    def _public_value(value: Any, depth: int = 0) -> Any:
        # Source snapshots can legitimately contain calibration arrays.  Keep
        # those arrays while applying a finite shape bound so a malformed
        # runtime payload cannot turn a status request into an unbounded copy.
        if depth > 8:
            return None
        if isinstance(value, Mapping):
            return _public_authority(value)
        if isinstance(value, (list, tuple)):
            return [_public_value(item, depth + 1) for item in list(value)[:4096]]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return None

    result: dict[str, Any] = {}
    for key, value in raw.items():
        lowered = str(key).lower()
        if lowered in {"uri", "url", "location", "username", "password", "credential", "secret"}:
            continue
        result[str(key)] = _public_value(value)
    return result


def _status_value(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return dict(obj)
    status = getattr(obj, "status", None)
    if callable(status):
        try:
            return _status_value(status())
        except Exception:
            return {}
    return {}


class _UnavailableTrackingObserver:
    def __init__(self, reason: str):
        self.reason = reason
        self._started = False

    def start(self) -> None:
        self._started = True

    def stop(self, reason: str = "user") -> Mapping[str, Any]:
        return {"stopped": True, "reason": reason, "error": self.reason}

    def status(self) -> Mapping[str, Any]:
        return {
            "ready": False,
            "tracking_ready": False,
            "record_count": 0,
            "error": self.reason,
        }


class _UnavailableRecorder:
    def __init__(self, output_dir: Path, reason: str):
        self.output_dir = output_dir
        self.reason = reason
        self._started = False

    def start(self) -> None:
        self._started = True

    def stop(self, reason: str = "user") -> Mapping[str, Any]:
        return {"stopped": True, "reason": reason, "error": self.reason}

    def status(self) -> Mapping[str, Any]:
        return {"ready": False, "encoded_ready": False, "bytes": 0, "error": self.reason}


class GstCompressedRecorder:
    """Remux one RTSP H.264 stream while retaining encoded timing evidence.

    GI is loaded only when a session starts.  The pipeline uses rtspsrc,
    depay, h264parse, matroskamux, and filesink; it never exposes raw video to
    Python.  The parser pad admits only video/H264 RTP and records compact
    encoded timing metadata beside the remuxed MKV.
    """

    EOS_TIMEOUT_S = 3.0
    THREAD_JOIN_TIMEOUT_S = 3.0
    WRITER_READY_TIMEOUT_S = 2.0
    PACKET_QUEUE_BYTES_LIMIT = 32 * 1024 * 1024

    def __init__(
        self,
        source_uri: str,
        output_dir: Path,
        *,
        max_duration_s: float = 900.0,
        max_bytes: int = 8 * 1024 * 1024 * 1024,
        max_packet_records: int = MAX_PACKET_RECORDS,
        stale_timeout_s: float = 5.0,
    ) -> None:
        if not str(source_uri).lower().startswith(("rtsp://", "rtsps://")):
            raise CompanionCaptureError("companion source must resolve to RTSP")
        if float(max_duration_s) <= 0 or int(max_bytes) <= 0 or int(max_packet_records) < 1:
            raise ValueError("recorder limits must be positive")
        if float(stale_timeout_s) <= 0:
            raise ValueError("stale_timeout_s must be positive")
        self._source_uri = str(source_uri)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.max_duration_s = float(max_duration_s)
        self.max_bytes = int(max_bytes)
        self.max_packet_records = int(max_packet_records)
        self.stale_timeout_s = float(stale_timeout_s)
        self.video_path = self.output_dir / "static_camera.mkv"
        self.packet_path = self.output_dir / "packet_timing.jsonl"
        self._pipeline: Any = None
        self._parser_src_pad: Any = None
        self._gst: Any = None
        self._bus_thread: threading.Thread | None = None
        self._watchdog_thread: threading.Thread | None = None
        self._packet_writer: threading.Thread | None = None
        self._packet_queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=4096)
        self._packet_queue_bytes = 0
        self._packet_writer_ready = threading.Event()
        self._stop_event = threading.Event()
        self._eos_event = threading.Event()
        self._lock = threading.RLock()
        self._finalize_lock = threading.Lock()
        self._finalized_event = threading.Event()
        self._finalized = False
        self._finalizing = False
        self._status: dict[str, Any] = {
            "ready": False,
            "encoded_ready": False,
            "status": "created",
            "bytes": 0,
            "packet_count": 0,
            "dropped_packet_records": 0,
            "partial": False,
            "error": None,
        }
        self._segment_object: Any = None
        self._segment_snapshot: dict[str, Any] | None = None
        self._parser_caps_snapshot: dict[str, Any] | None = None

    @staticmethod
    def _redacted_error(value: Any) -> str:
        return redact_runtime_secrets(str(value))[:500]

    def _failure(self, reason: str, detail: Any | None = None) -> None:
        error = str(reason)
        if detail is not None:
            error = f"{error}:{self._redacted_error(detail)}"
        with self._lock:
            self._status["partial"] = True
            self._status["error"] = self._status.get("error") or self._redacted_error(error)
            self._status["stop_requested"] = True
            self._status["stop_reason"] = str(reason)
            self._status["status"] = "failed"
        self._stop_event.set()

    def _gst_time(self, value: Any) -> int | None:
        if value is None:
            return None
        try:
            number = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        gst = self._gst
        clock_none = getattr(gst, "CLOCK_TIME_NONE", None) if gst is not None else None
        if number < 0 or clock_none is not None and number == int(clock_none):
            return None
        return number

    @staticmethod
    def _metadata_value(value: Any, depth: int = 0) -> Any:
        if depth > 4:
            return None
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        numerator = getattr(value, "numerator", None)
        denominator = getattr(value, "denominator", None)
        if numerator is not None and denominator is not None:
            try:
                return {"numerator": int(numerator), "denominator": int(denominator)}
            except (TypeError, ValueError, OverflowError):
                return str(value)
        if isinstance(value, Mapping):
            return {str(key): GstCompressedRecorder._metadata_value(item, depth + 1) for key, item in list(value.items())[:64]}
        if isinstance(value, (list, tuple)):
            return [GstCompressedRecorder._metadata_value(item, depth + 1) for item in list(value)[:64]]
        to_string = getattr(value, "to_string", None)
        if callable(to_string):
            try:
                return str(to_string())
            except Exception:
                pass
        return str(value)

    def _caps_snapshot(self, pad: Any) -> dict[str, Any] | None:
        if pad is None:
            return self._parser_caps_snapshot
        try:
            caps = pad.get_current_caps() or pad.query_caps(None)
            if caps is None or caps.get_size() < 1:
                return self._parser_caps_snapshot
            structure = caps.get_structure(0)
        except Exception:
            return self._parser_caps_snapshot
        result: dict[str, Any] = {}
        try:
            result["caps_string"] = str(caps.to_string())
        except Exception:
            pass
        for key in (
            "media",
            "encoding-name",
            "clock-rate",
            "width",
            "height",
            "framerate",
            "stream-format",
            "alignment",
            "profile",
            "level",
        ):
            try:
                value = structure.get_value(key)
            except Exception:
                value = None
            if value is None:
                try:
                    value = structure.get_string(key)
                except Exception:
                    value = None
            if value is not None:
                result[key] = self._metadata_value(value)
        return result or self._parser_caps_snapshot

    def _segment_details(self, segment: Any) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key in ("format", "start", "stop", "time", "base", "offset", "position"):
            value = getattr(segment, key, None)
            if key == "format":
                result[key] = self._metadata_value(value)
            else:
                result[key] = self._gst_time(value)
        for key in ("rate", "applied_rate"):
            value = getattr(segment, key, None)
            result[key] = self._metadata_value(value)
        return result

    def _remember_segment(self, segment: Any) -> None:
        with self._lock:
            self._segment_object = segment
            self._segment_snapshot = self._segment_details(segment)
            self._status["last_segment"] = deepcopy(self._segment_snapshot)

    def _pipeline_clock_details(self) -> dict[str, Any]:
        pipeline = self._pipeline
        if pipeline is None:
            return {"clock_time_ns": None, "base_time_ns": None, "running_time_ns": None}
        try:
            clock = pipeline.get_clock()
            clock_time = self._gst_time(clock.get_time()) if clock is not None else None
        except Exception:
            clock_time = None
        try:
            base_time = self._gst_time(pipeline.get_base_time())
        except Exception:
            base_time = None
        running = None if clock_time is None or base_time is None else max(0, clock_time - base_time)
        return {
            "clock_time_ns": clock_time,
            "base_time_ns": base_time,
            "running_time_ns": running,
        }

    def _running_time(self, pts_ns: int | None) -> int | None:
        with self._lock:
            segment = self._segment_object
        if segment is None or pts_ns is None or self._gst is None:
            return None
        try:
            value = segment.to_running_time(self._gst.Format.TIME, pts_ns)
        except Exception:
            return None
        return self._gst_time(value)

    def _record_packet(self, buffer: Any, pad: Any = None) -> None:
        now_mono = time.monotonic_ns()
        now_utc = time.time_ns()
        pts = self._gst_time(getattr(buffer, "pts", None))
        dts = self._gst_time(getattr(buffer, "dts", None))
        duration = self._gst_time(getattr(buffer, "duration", None))
        parser_caps = self._caps_snapshot(pad)
        if parser_caps is not None:
            with self._lock:
                self._parser_caps_snapshot = deepcopy(parser_caps)
        keyframe: bool | None = None
        try:
            flags = buffer.get_flags()
            delta_flag = getattr(getattr(self._gst, "BufferFlags", None), "DELTA_UNIT", None)
            if delta_flag is not None:
                keyframe = not bool(int(flags) & int(delta_flag))
        except Exception:
            keyframe = None
        reference_metadata: dict[str, Any] | None = None
        try:
            # Passing None requests the original reference caps attached by
            # rtspsrc; synthesizing timestamp/x-rtcp would lose source detail.
            meta = buffer.get_reference_timestamp_meta(None)
            if meta is not None:
                reference_caps = getattr(meta, "reference", None)
                if reference_caps is None:
                    reference_caps = getattr(meta, "caps", None)
                try:
                    caps_text = reference_caps.to_string() if reference_caps is not None else None
                except Exception:
                    caps_text = str(reference_caps) if reference_caps is not None else None
                reference_metadata = {
                    "timestamp_ns": self._gst_time(getattr(meta, "timestamp", None)),
                    "duration_ns": self._gst_time(getattr(meta, "duration", None)),
                    "caps": caps_text,
                }
        except Exception:
            reference_metadata = None
        row: dict[str, Any] = {
            "receive_monotonic_ns": now_mono,
            "receive_utc_ns": now_utc,
            "pts_ns": pts,
            "dts_ns": dts,
            "duration_ns": duration,
            "running_time_ns": self._running_time(pts),
            "pipeline_clock": self._pipeline_clock_details(),
            "segment": deepcopy(self._segment_snapshot),
            "parser_caps": deepcopy(parser_caps),
            "keyframe": keyframe,
            "reference_timestamp_meta": reference_metadata,
            "timestamp_source": "gstreamer_buffer_pts_dts_segment_pipeline_clock_and_host_receive",
        }
        try:
            encoded = json.dumps(row, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"
        except (TypeError, ValueError, OverflowError) as exc:
            self._failure("packet_metadata_not_json", exc)
            return
        with self._lock:
            if self._status["packet_count"] >= self.max_packet_records:
                self._status["dropped_packet_records"] += 1
                self._failure("max_packet_records")
                return
            if self._packet_queue_bytes + len(encoded) > self.PACKET_QUEUE_BYTES_LIMIT:
                self._status["dropped_packet_records"] += 1
                self._failure("packet_timing_queue_overflow")
                return
            try:
                self._packet_queue.put_nowait(row)
            except queue.Full:
                self._status["dropped_packet_records"] += 1
                self._failure("packet_timing_queue_overflow")
                return
            self._packet_queue_bytes += len(encoded)
            self._status["packet_count"] += 1
            self._status["encoded_ready"] = True
            self._status["ready"] = True
            self._status.setdefault("first_packet_monotonic_ns", now_mono)
            self._status.setdefault("first_packet_utc_ns", now_utc)
            self._status["last_packet_monotonic_ns"] = now_mono
            self._status["last_packet_utc_ns"] = now_utc

    def _packet_probe(self, pad: Any, info: Any) -> Any:
        try:
            event_getter = getattr(info, "get_event", None)
            event = event_getter() if callable(event_getter) else None
            if event is not None:
                event_type = getattr(event, "type", None)
                segment_type = getattr(getattr(self._gst, "EventType", None), "SEGMENT", None)
                if segment_type is not None and event_type == segment_type:
                    parse_segment = getattr(event, "parse_segment", None)
                    if callable(parse_segment):
                        segment = parse_segment()
                        if segment is not None:
                            self._remember_segment(segment)
                return getattr(getattr(self._gst, "PadProbeReturn", None), "OK", 0)
            get_buffer = getattr(info, "get_buffer", None)
            buffer = get_buffer() if callable(get_buffer) else None
            if buffer is not None:
                self._record_packet(buffer, pad)
        except Exception as exc:  # noqa: BLE001 - metadata failure is capture-local
            self._failure("packet_probe_failure", exc)
        return getattr(getattr(self._gst, "PadProbeReturn", None), "OK", 0)

    def _queue_bytes_for_row(self, row: Mapping[str, Any]) -> int:
        try:
            return len(json.dumps(row, separators=(",", ":"), ensure_ascii=False).encode("utf-8")) + 1
        except Exception:
            return 0

    def _drain_packet_queue(self, *, count_dropped: bool) -> None:
        with self._lock:
            while True:
                try:
                    item = self._packet_queue.get_nowait()
                except queue.Empty:
                    self._packet_queue_bytes = 0
                    return
                else:
                    if item is not None:
                        self._packet_queue_bytes = max(0, self._packet_queue_bytes - self._queue_bytes_for_row(item))
                        if count_dropped:
                            self._status["dropped_packet_records"] += 1
                    self._packet_queue.task_done()

    def _packet_writer_loop(self) -> None:
        handle = None
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            handle = self.packet_path.open("w", encoding="utf-8")
            with self._lock:
                self._status["packet_writer_ready"] = True
            self._packet_writer_ready.set()
            while True:
                try:
                    row = self._packet_queue.get(timeout=0.25)
                except queue.Empty:
                    if self._stop_event.is_set():
                        break
                    continue
                if row is None:
                    self._packet_queue.task_done()
                    break
                row_bytes = self._queue_bytes_for_row(row)
                try:
                    handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
                    handle.flush()
                except Exception as exc:  # noqa: BLE001 - disk is observer-local
                    with self._lock:
                        self._packet_queue_bytes = max(0, self._packet_queue_bytes - row_bytes)
                        self._packet_queue.task_done()
                    self._failure("packet_writer_failure", exc)
                    break
                with self._lock:
                    self._packet_queue_bytes = max(0, self._packet_queue_bytes - row_bytes)
                self._packet_queue.task_done()
        except Exception as exc:  # noqa: BLE001 - setup failure is observer-local
            self._failure("packet_writer_setup_failure", exc)
            self._packet_writer_ready.set()
        finally:
            if handle is not None:
                try:
                    handle.close()
                except Exception as exc:
                    self._failure("packet_writer_close_failure", exc)
            self._drain_packet_queue(count_dropped=True)
            with self._lock:
                self._status["packet_writer_stopped"] = True
            self._packet_writer_ready.set()

    def _bus_loop(self) -> None:
        try:
            bus = self._pipeline.get_bus()
            # Keep draining the bus after a failure has requested shutdown so
            # the coordinator can still observe the EOS it sends below.
            while not self._eos_event.is_set():
                message = bus.timed_pop_filtered(
                    200 * self._gst.MSECOND,
                    self._gst.MessageType.ERROR | self._gst.MessageType.EOS | self._gst.MessageType.STATE_CHANGED,
                )
                if message is None:
                    continue
                message_type = getattr(message, "type", None)
                if message_type == self._gst.MessageType.ERROR:
                    error, debug = message.parse_error()
                    self._failure("gstreamer_error", error.message if error is not None else "unknown")
                    if debug:
                        with self._lock:
                            self._status["debug"] = self._redacted_error(debug)[-500:]
                    self._eos_event.set()
                    return
                if message_type == self._gst.MessageType.EOS:
                    self._eos_event.set()
                    with self._lock:
                        requested = bool(self._status.get("stop_requested"))
                    if not requested:
                        self._failure("gstreamer_eos")
                    return
        except Exception as exc:  # noqa: BLE001 - bus failure is capture-local
            self._failure("gstreamer_bus_failure", exc)
            self._eos_event.set()

    def _watchdog_loop(self) -> None:
        current = threading.current_thread()
        while not self._stop_event.wait(0.25):
            with self._lock:
                started = self._status.get("started_monotonic_ns")
                last_packet = self._status.get("last_packet_monotonic_ns")
                packet_count = int(self._status.get("packet_count") or 0)
            now = time.monotonic_ns()
            if started is not None and (now - int(started)) / 1e9 >= self.max_duration_s:
                self._failure("max_duration_s")
                self._finalize_stop("max_duration_s", current=current)
                return
            try:
                size = self.video_path.stat().st_size if self.video_path.is_file() else 0
            except OSError as exc:
                self._failure("video_stat_failure", exc)
                self._finalize_stop("video_stat_failure", current=current)
                return
            with self._lock:
                self._status["bytes"] = int(size)
            if size > self.max_bytes:
                self._failure("max_session_bytes")
                self._finalize_stop("max_session_bytes", current=current)
                return
            if started is not None:
                reference = int(last_packet) if last_packet is not None else int(started)
                if (now - reference) / 1e9 >= self.stale_timeout_s:
                    self._failure("video_stale_timeout")
                    self._finalize_stop("video_stale_timeout", current=current)
                    return
            if packet_count and self._overflow_requested():
                reason = self._status.get("stop_reason") or "packet_timing_overflow"
                self._finalize_stop(str(reason), current=current)
                return
        with self._lock:
            reason = self._status.get("stop_reason")
        if reason:
            self._finalize_stop(str(reason), current=current)

    def _overflow_requested(self) -> bool:
        with self._lock:
            return bool(self._status.get("stop_requested"))

    def _set_pipeline_null(self, pipeline: Any) -> None:
        if pipeline is None:
            return
        try:
            pipeline.set_state(self._gst.State.NULL)
        except Exception as exc:
            self._failure("pipeline_cleanup_failure", exc)

    def _cleanup_start_failure(self, pipeline: Any) -> None:
        self._stop_event.set()
        self._set_pipeline_null(pipeline)
        self._drain_packet_queue(count_dropped=True)
        if self._packet_writer is not None and self._packet_writer.is_alive() and self._packet_writer is not threading.current_thread():
            self._packet_writer.join(timeout=self.THREAD_JOIN_TIMEOUT_S)
        if self._packet_writer is not None and self._packet_writer.is_alive():
            self._failure("packet_writer_join_timeout")

    def start(self) -> None:
        with self._lock:
            if self._status["status"] != "created":
                return
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._stop_event.clear()
            self._eos_event.clear()
            self._packet_writer_ready.clear()
            self._finalized_event.clear()
            self._finalized = False
            self._finalizing = False
            self._packet_writer = threading.Thread(target=self._packet_writer_loop, name="CompanionPacketWriter", daemon=True)
            self._packet_writer.start()
        if not self._packet_writer_ready.wait(self.WRITER_READY_TIMEOUT_S):
            self._failure("packet_writer_start_timeout")
            self._cleanup_start_failure(None)
            raise CompanionCaptureError("packet timing writer did not become ready")
        with self._lock:
            if self._status.get("error"):
                self._cleanup_start_failure(None)
                raise CompanionCaptureError("packet timing writer could not start")
        pipeline = None
        try:
            import gi  # type: ignore

            gi.require_version("Gst", "1.0")
            from gi.repository import Gst  # type: ignore

            self._gst = Gst
            Gst.init(None)
            pipeline = Gst.Pipeline.new("noesis-companion-remux")
            self._pipeline = pipeline
            source = Gst.ElementFactory.make("rtspsrc", "source")
            depay = Gst.ElementFactory.make("rtph264depay", "depay")
            parser = Gst.ElementFactory.make("h264parse", "parser")
            mux = Gst.ElementFactory.make("matroskamux", "mux")
            sink = Gst.ElementFactory.make("filesink", "sink")
            elements = (pipeline, source, depay, parser, mux, sink)
            if any(element is None for element in elements):
                raise CompanionCaptureError("GStreamer H.264 remux elements are unavailable")
            source.set_property("location", self._source_uri)
            source.set_property("protocols", 4)  # TCP, matching canonical RTSP
            if source.find_property("add-reference-timestamp-meta") is not None:
                source.set_property("add-reference-timestamp-meta", True)
            sink.set_property("location", str(self.video_path))
            if mux.find_property("offset-to-zero") is not None:
                mux.set_property("offset-to-zero", False)
            if parser.find_property("config-interval") is not None:
                parser.set_property("config-interval", -1)
            for element in (source, depay, parser, mux, sink):
                pipeline.add(element)
            if not depay.link(parser) or not parser.link(mux) or not mux.link(sink):
                raise CompanionCaptureError("GStreamer remux links could not be created")
            source.connect("pad-added", lambda _src, pad: self._link_dynamic(pad, depay))
            parser_src_pad = parser.get_static_pad("src")
            if parser_src_pad is None:
                raise CompanionCaptureError("GStreamer parser source pad is unavailable")
            self._parser_src_pad = parser_src_pad
            probe_type = Gst.PadProbeType.BUFFER
            event_downstream = getattr(Gst.PadProbeType, "EVENT_DOWNSTREAM", None)
            if event_downstream is not None:
                probe_type |= event_downstream
            parser_src_pad.add_probe(probe_type, self._packet_probe)
            with self._lock:
                self._status.update(
                    {
                        "status": "recording",
                        "started_monotonic_ns": time.monotonic_ns(),
                        "started_utc_ns": time.time_ns(),
                        "pipeline": "rtspsrc-rtph264depay-h264parse-matroskamux-filesink",
                        "encoded_format": "h264-in-matroska",
                        "reference_timestamp_meta_requested": True,
                        "reference_timestamp_meta_caps": "original_caps_via_get_reference_timestamp_meta_none",
                        "mux_offset_to_zero": False,
                        "source_clock": "rtsp_source_clock_unverified",
                        "video_pad_filter": "media=video,encoding-name=H264",
                    }
                )
            if self._stop_event.is_set():
                raise CompanionCaptureError("recorder was stopped during startup")
            state_result = pipeline.set_state(Gst.State.PLAYING)
            failure_state = getattr(getattr(Gst, "StateChangeReturn", None), "FAILURE", None)
            if failure_state is not None and state_result == failure_state:
                raise CompanionCaptureError("GStreamer pipeline could not enter PLAYING")
            self._bus_thread = threading.Thread(target=self._bus_loop, name="CompanionGstBus", daemon=True)
            self._bus_thread.start()
            self._watchdog_thread = threading.Thread(target=self._watchdog_loop, name="CompanionGstWatchdog", daemon=True)
            self._watchdog_thread.start()
        except Exception as exc:
            with self._lock:
                self._status["status"] = "failed"
                self._status["partial"] = True
                self._status["error"] = self._status.get("error") or self._redacted_error(f"start:{type(exc).__name__}: {exc}")
            self._cleanup_start_failure(pipeline)
            raise CompanionCaptureError("GStreamer static-camera remux could not start") from exc

    @staticmethod
    def _link_dynamic(pad: Any, depay: Any) -> None:
        sink_pad = depay.get_static_pad("sink")
        if sink_pad is None or sink_pad.is_linked():
            return
        try:
            caps = pad.get_current_caps() or pad.query_caps(None)
            if caps is None or caps.get_size() < 1:
                return
            structure = caps.get_structure(0)
            name = str(structure.get_name() or "")
            if not name.startswith("application/x-rtp"):
                return
            media = str(structure.get_string("media") or "").lower()
            encoding = str(structure.get_string("encoding-name") or "").upper()
        except Exception:
            return
        if media != "video" or encoding != "H264":
            return
        pad.link(sink_pad)

    def _finalize_stop(self, reason: str, *, current: threading.Thread | None = None) -> Mapping[str, Any]:
        current_thread = current or threading.current_thread()
        with self._finalize_lock:
            if self._finalized:
                return self.status()
            if self._finalizing:
                if current_thread is not self._watchdog_thread:
                    self._finalized_event.wait(self.EOS_TIMEOUT_S + self.THREAD_JOIN_TIMEOUT_S)
                return self.status()
            self._finalizing = True
        try:
            with self._lock:
                self._status["stop_requested"] = True
                self._status["stop_reason"] = str(reason)
                pipeline = self._pipeline
            if pipeline is not None:
                try:
                    pipeline.send_event(self._gst.Event.new_eos())
                except Exception as exc:
                    self._failure("eos_request_failure", exc)
                if self._bus_thread is None:
                    self._eos_event.set()
                elif not self._eos_event.wait(self.EOS_TIMEOUT_S):
                    self._failure("eos_timeout")
                self._set_pipeline_null(pipeline)
            self._stop_event.set()
            if self._packet_writer is not None and self._packet_writer is not current_thread:
                self._packet_writer.join(timeout=self.THREAD_JOIN_TIMEOUT_S)
            if self._bus_thread is not None and self._bus_thread is not current_thread:
                self._bus_thread.join(timeout=self.THREAD_JOIN_TIMEOUT_S)
            if self._watchdog_thread is not None and self._watchdog_thread is not current_thread:
                self._watchdog_thread.join(timeout=self.THREAD_JOIN_TIMEOUT_S)
            with self._lock:
                if self._packet_writer is not None and self._packet_writer.is_alive():
                    self._status["partial"] = True
                    self._status["error"] = self._status.get("error") or "packet_writer_join_timeout"
                if self._bus_thread is not None and self._bus_thread.is_alive():
                    self._status["partial"] = True
                    self._status["error"] = self._status.get("error") or "bus_thread_join_timeout"
                if self._watchdog_thread is not None and self._watchdog_thread.is_alive() and self._watchdog_thread is not current_thread:
                    self._status["partial"] = True
                    self._status["error"] = self._status.get("error") or "watchdog_join_timeout"
                try:
                    self._status["bytes"] = int(self.video_path.stat().st_size) if self.video_path.is_file() else 0
                except OSError as exc:
                    self._status["bytes"] = 0
                    self._status["partial"] = True
                    self._status["error"] = self._status.get("error") or self._redacted_error(
                        f"video_stat_failure:{exc}"
                    )
                self._status["packet_timing_path"] = self.packet_path.name if self.packet_path.is_file() else None
                self._status["video_path"] = self.video_path.name if self.video_path.is_file() else None
                self._status["artifacts"] = {
                    "video": self._status["video_path"],
                    "packet_timing": self._status["packet_timing_path"],
                }
                if self._status.get("status") != "failed":
                    self._status["status"] = "stopped"
                self._finalized = True
        finally:
            self._finalized_event.set()
            with self._finalize_lock:
                self._finalizing = False
        return self.status()

    def stop(self, reason: str = "user") -> Mapping[str, Any]:
        with self._lock:
            if self._finalized and self._status["status"] in {"stopped", "complete", "failed"}:
                return dict(self._status)
            self._status["stop_requested"] = True
            self._status["stop_reason"] = str(reason)
        return self._finalize_stop(str(reason))

    def status(self) -> Mapping[str, Any]:
        with self._lock:
            result = dict(self._status)
            try:
                result["bytes"] = int(self.video_path.stat().st_size) if self.video_path.is_file() else 0
            except OSError:
                result["bytes"] = 0
            result.setdefault("packet_timing_path", self.packet_path.name if self.packet_path.is_file() else None)
            result.setdefault("video_path", self.video_path.name if self.video_path.is_file() else None)
            result.setdefault(
                "artifacts",
                {
                    "video": result.get("video_path"),
                    "packet_timing": result.get("packet_timing_path"),
                },
            )
            return result

def _default_camera_catalog() -> list[CompanionCamera]:
    """Resolve the active runtime camera inventory without exposing URIs."""

    try:
        loaded = list_active_camera_sources(require_process=True)
    except Exception as exc:
        raise CompanionCaptureError(
            redact_runtime_secrets(
                f"canonical camera source resolution failed: {type(exc).__name__}: {exc}"
            )[:500]
        ) from exc
    if not isinstance(loaded, Mapping) or loaded.get("available") is not True:
        reason = str(loaded.get("reason") or "no active camera source") if isinstance(loaded, Mapping) else "invalid source inventory"
        raise CompanionCaptureError(reason)
    cameras: list[CompanionCamera] = []
    for row in loaded.get("cameras") or []:
        if not isinstance(row, Mapping) or row.get("available") is not True:
            continue
        camera_id = str(row.get("camera_id") or "").strip()
        if not camera_id:
            continue
        cameras.append(
            CompanionCamera(
                camera_id=camera_id,
                label=str(row.get("label") or camera_id),
                source_reference=f"camera-secret:{camera_id}",
                source_authority={
                    "camera_id": camera_id,
                    "source_id": row.get("source_id"),
                    "config_authority": loaded.get("config_authority"),
                    "source_config_sha256": loaded.get("source_config_sha256"),
                },
                source_id=int(row["source_id"]) if row.get("source_id") is not None else None,
            )
        )
    if not cameras:
        raise CompanionCaptureError("no active camera source is available")
    return cameras


class CompanionCaptureManager:
    """Own one globally active, durable paired capture session."""

    def __init__(
        self,
        storage_root: Path,
        *,
        limits: CompanionCaptureLimits = CompanionCaptureLimits(),
        camera_catalog: Callable[[], list[CompanionCamera]] = _default_camera_catalog,
        source_resolver: Callable[..., Any] | None = None,
        recorder_factory: Callable[..., RecorderProtocol] | None = None,
        tracking_observer_factory: Callable[..., TrackingObserverProtocol] | None = None,
    ) -> None:
        self.storage_root = Path(storage_root).expanduser().resolve()
        self.session_root = self.storage_root / ".companion-captures"
        self.session_root.mkdir(parents=True, exist_ok=True)
        self.limits = limits
        self.camera_catalog = camera_catalog
        self.source_resolver = source_resolver or resolve_active_source_authority
        self.recorder_factory = recorder_factory or GstCompressedRecorder
        self.tracking_observer_factory = tracking_observer_factory
        self._lock = threading.RLock()
        self._active_session_id: str | None = None
        self._runtime: dict[str, tuple[RecorderProtocol, TrackingObserverProtocol, threading.Event]] = {}
        self._stop_event = threading.Event()
        self._watchdog = threading.Thread(target=self._watchdog_loop, name="CompanionCaptureLease", daemon=True)
        self._watchdog.start()
        self.recover_orphans()

    def shutdown(self) -> None:
        self._stop_event.set()
        with self._lock:
            session_id = self._active_session_id
        if session_id:
            try:
                self.stop_session(session_id, reason="service_shutdown")
            except Exception:
                pass
        if self._watchdog.is_alive():
            self._watchdog.join(timeout=2.0)

    def _watchdog_loop(self) -> None:
        while not self._stop_event.wait(0.5):
            with self._lock:
                session_id = self._active_session_id
            if not session_id:
                continue
            try:
                state = self._read(session_id)
                if state.get("status") not in {"starting", "recording"}:
                    continue
                now = time.monotonic()
                lease_deadline = float(state.get("lease", {}).get("expires_monotonic_s") or 0.0)
                started = float(state.get("started_monotonic_s") or now)
                if now >= lease_deadline:
                    self.stop_session(session_id, reason="lease_expired")
                elif now - started >= self.limits.max_duration_s:
                    self.stop_session(session_id, reason="max_duration_s")
                else:
                    failure = self._refresh_runtime_status(session_id)
                    if failure:
                        self.stop_session(session_id, reason=failure)
            except Exception:
                continue

    def _session_path(self, session_id: str) -> Path:
        if not COMPANION_SESSION_PATTERN.fullmatch(str(session_id)):
            raise CompanionCaptureError("invalid companion session ID")
        return self.session_root / session_id / "session.json"

    def _read(self, session_id: str) -> dict[str, Any]:
        path = self._session_path(session_id)
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CompanionCaptureError("companion session state is unreadable") from exc
        if not isinstance(state, dict) or state.get("session_id") != session_id:
            raise CompanionCaptureError("companion session state is invalid")
        return state

    def _write(self, session_id: str, state: dict[str, Any]) -> None:
        path = self._session_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        state["updated_at"] = _utc_now()
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)

    def _refresh_runtime_status(self, session_id: str) -> str | None:
        with self._lock:
            runtime = self._runtime.get(session_id)
            if runtime is None:
                return None
            recorder, observer, _ = runtime
            state = self._read(session_id)
            readiness = state.setdefault("readiness", {})
            readiness["recorder"] = _status_value(recorder)
            readiness["tracking"] = _status_value(observer)
            self._write(session_id, state)
            return self._active_component_failure(state)

    def list_cameras(self) -> list[dict[str, Any]]:
        cameras = self.camera_catalog()
        return [
            {
                "id": camera.camera_id,
                "camera_id": camera.camera_id,
                "label": camera.label,
                "available": True,
                "source_reference": camera.source_reference,
                "authority": _public_authority(camera.source_authority),
            }
            for camera in cameras
        ]

    def _camera(self, camera_id: str) -> CompanionCamera:
        wanted = str(camera_id or "").strip()
        if not wanted:
            raise CompanionCaptureError("camera_id is required")
        for camera in self.camera_catalog():
            if camera.camera_id == wanted:
                return camera
        raise CompanionCaptureError(f"unknown companion camera {wanted!r}")

    def _resolve_source(self, camera: CompanionCamera) -> tuple[str, Any]:
        try:
            # The canonical helper binds the selected source to the currently
            # running DS9 process and keeps the private URI on its authority
            # object only.  Do not fall back to repository defaults here.
            authority = self.source_resolver(camera.camera_id, require_process=True)
        except (StaticCaptureSourceError, OSError, RuntimeError) as exc:
            raise CompanionCaptureError(
                redact_runtime_secrets(f"active static source authority failed: {exc}")[:500]
            ) from exc
        source_uri = str(authority.private_uri or "").strip()
        if not source_uri.lower().startswith(("rtsp://", "rtsps://")):
            raise CompanionCaptureError("selected camera has no active RTSP authority")
        public = authority.public_snapshot()
        if not isinstance(public, Mapping):
            raise CompanionCaptureError("selected camera source authority has no public snapshot")
        return source_uri, authority

    def _make_observer(
        self,
        camera: CompanionCamera,
        output_dir: Path,
        authority: Any | None = None,
    ) -> TrackingObserverProtocol:
        if self.tracking_observer_factory is not None:
            factory = self.tracking_observer_factory
            if isinstance(authority, StaticSourceAuthority):
                try:
                    return factory(authority, output_dir, self.limits.max_tracking_records)
                except TypeError:
                    try:
                        return factory(
                            authority=authority,
                            output_dir=output_dir,
                            max_records=self.limits.max_tracking_records,
                        )
                    except TypeError:
                        pass
            try:
                return factory(camera.camera_id, output_dir, self.limits.max_tracking_records)
            except TypeError:
                try:
                    return factory(camera_id=camera.camera_id, output_dir=output_dir, max_records=self.limits.max_tracking_records)
                except TypeError:
                    return factory(camera.camera_id, output_dir)
        if authority is not None:
            try:
                return CanonicalTrackingRecorder(
                    authority,
                    output_dir,
                    self.limits.max_tracking_records,
                )
            except (TypeError, ValueError):
                # Focused/injected source resolvers can provide only a private
                # URI.  They remain usable with the bounded observer below;
                # production resolution always returns StaticSourceAuthority
                # and therefore takes the canonical authenticated recorder.
                pass
        try:
            initial_provenance = camera.source_authority
            if authority is not None:
                public_snapshot = getattr(authority, "public_snapshot", None)
                if callable(public_snapshot):
                    try:
                        candidate = public_snapshot()
                        if isinstance(candidate, Mapping):
                            initial_provenance = _public_authority(candidate)
                    except Exception:
                        pass
            return StaticCaptureObserver(
                output_dir / "tracking.jsonl",
                selected_source_id=camera.source_id,
                initial_provenance=initial_provenance,
            )
        except Exception as exc:
            return _UnavailableTrackingObserver(
                f"canonical tracking observer construction failed: {exc}"
            )

    @staticmethod
    def _observer_snapshot(observer: TrackingObserverProtocol) -> dict[str, Any]:
        snapshot = getattr(observer, "snapshot", None)
        if not callable(snapshot):
            return {}
        try:
            value = snapshot()
        except Exception:
            return {}
        if not isinstance(value, Mapping):
            return {}
        try:
            return _safe_json(dict(value), max_bytes=MAX_MARKER_BYTES * 4)
        except CompanionCaptureError:
            return {}

    @staticmethod
    def _ready(status: Mapping[str, Any], *, tracking: bool, selected_source_id: int | None = None) -> bool:
        state = str(status.get("status") or status.get("state") or "").lower()
        if state not in {"recording", "active", "ready", "started", "healthy", "ok"}:
            return False
        if status.get("error") not in (None, "") or status.get("writer_error") not in (None, ""):
            return False
        if tracking:
            if status.get("tracking_ready") is not True:
                return False
            observed_source_id = status.get("selected_source_id")
            if observed_source_id is None:
                observed_source_id = status.get("source_id")
            if selected_source_id is not None and observed_source_id is not None:
                try:
                    if int(observed_source_id) != int(selected_source_id):
                        return False
                except (TypeError, ValueError):
                    return False
        elif status.get("encoded_ready") is not True and status.get("ready") is not True:
            return False
        count = status.get(
            "records",
            status.get("record_count", status.get("cohort_count", status.get("packet_count", 0))),
        )
        try:
            return int(count) > 0
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _active_component_failure(state: Mapping[str, Any]) -> str | None:
        """Return the first component failure while a session is live.

        Runtime component status is written below ``readiness`` while a
        session is active and is copied to the top-level record only during
        finalization.  Keep this check against both shapes so a failed
        recorder/tracking stream cannot be hidden by a stale browser status.
        """

        if state.get("status") not in {"starting", "recording"}:
            return None
        readiness = state.get("readiness") if isinstance(state.get("readiness"), Mapping) else {}
        for label in ("recorder", "tracking"):
            value = state.get(label)
            component = value if isinstance(value, Mapping) and value else readiness.get(label)
            if not isinstance(component, Mapping):
                continue
            error = component.get("error") or component.get("writer_error")
            if error not in (None, ""):
                return f"{label}:{str(error)[:500]}"
            if component.get("partial") is True:
                reasons = component.get("partial_reasons")
                reason = reasons[0] if isinstance(reasons, list) and reasons else "partial"
                return f"{label}:{str(reason)[:500]}"
            status_value = str(component.get("status") or component.get("state") or "").lower()
            # Recorder/tracking admission is intentionally two-phase.  The
            # lease watchdog may observe these not-yet-ready states while the
            # session is still starting; readiness_timeout owns that boundary.
            # Explicit error/partial/failed states above still fail closed.
            if state.get("status") == "starting" and status_value in {"created", "idle", "starting"}:
                continue
            if status_value in {"failed", "error", "stopped", "idle"}:
                return f"{label}:component_{status_value}"
            if status_value and status_value not in {
                "recording",
                "active",
                "ready",
                "started",
                "healthy",
                "ok",
            }:
                return f"{label}:component_unhealthy"
        return None

    @staticmethod
    def _static_capture_healthy(state: Mapping[str, Any]) -> bool:
        """Whether finalized static evidence is healthy despite phone upload state."""

        if state.get("error") not in (None, ""):
            return False
        artifacts = state.get("artifacts")
        if not isinstance(artifacts, Mapping) or not artifacts.get("static_video") or not artifacts.get("tracking"):
            return False
        for label in ("recorder", "tracking"):
            component = state.get(label)
            if not isinstance(component, Mapping):
                return False
            if component.get("error") not in (None, "") or component.get("writer_error") not in (None, ""):
                return False
            if component.get("partial") is True:
                return False
            component_status = str(component.get("status") or component.get("state") or "").lower()
            if component_status not in {"stopped", "complete"}:
                return False
        return True

    def _await_readiness(self, session_id: str, ready_event: threading.Event) -> None:
        deadline = time.monotonic() + self.limits.readiness_timeout_s
        while time.monotonic() < deadline and not ready_event.is_set():
            with self._lock:
                runtime = self._runtime.get(session_id)
                if runtime is None:
                    return
                recorder, observer, _ = runtime
                # Keep the complete read-modify-write under the manager lock.
                # Otherwise a concurrent heartbeat/stop can be overwritten by
                # this worker's stale readiness snapshot.
                try:
                    state = self._read(session_id)
                    if state.get("status") != "starting":
                        return
                    recorder_status = _status_value(recorder)
                    tracking_status = _status_value(observer)
                    recorder_ready = self._ready(recorder_status, tracking=False)
                    camera_source_id = state.get("camera", {}).get("source_id")
                    tracking_ready = self._ready(
                        tracking_status,
                        tracking=True,
                        selected_source_id=camera_source_id,
                    )
                    state["readiness"].update(
                        {
                            "recorder_ready": recorder_ready,
                            "tracking_ready": tracking_ready,
                            "recorder": recorder_status,
                            "tracking": tracking_status,
                        }
                    )
                    if recorder_ready and tracking_ready:
                        state["status"] = "recording"
                        state["started_at"] = _utc_now()
                        state["readiness"]["ready_at"] = _utc_now()
                        state["lease"]["expires_monotonic_s"] = time.monotonic() + self.limits.lease_s
                        ready_event.set()
                    self._write(session_id, state)
                except CompanionCaptureError:
                    return
            if ready_event.is_set():
                return
            time.sleep(0.05)
        try:
            self.stop_session(session_id, reason="readiness_timeout")
        except Exception:
            pass

    def start_session(
        self,
        camera_id: str,
        *,
        client_request_id: str | None = None,
        phone_capture_id: str | None = None,
        clock_probes: Any | None = None,
        await_ready: bool = True,
    ) -> dict[str, Any]:
        camera = self._camera(camera_id)
        request_id = str(client_request_id or "").strip() or None
        if request_id and len(request_id) > 160:
            raise CompanionCaptureError("client_request_id is too long")
        capture_id = str(phone_capture_id or "").strip() or None
        if capture_id and COMPANION_CAPTURE_ID_PATTERN.fullmatch(capture_id) is None:
            raise CompanionCaptureError("phone capture ID is invalid")
        raw_clock_probes = self._validated_clock_probes(clock_probes)
        with self._lock:
            self._expire_if_needed_locked()
            if request_id:
                for session_path in self.session_root.glob("*/session.json"):
                    try:
                        prior = self._read(session_path.parent.name)
                    except CompanionCaptureError:
                        continue
                    if prior.get("client_request_id") == request_id:
                        prior_camera = str((prior.get("camera") or {}).get("camera_id") or "")
                        prior_phone = str(
                            prior.get("phone_capture_id")
                            or (prior.get("phone") or {}).get("capture_id")
                            or ""
                        )
                        if prior_camera != camera.camera_id or (
                            capture_id and prior_phone and prior_phone != capture_id
                        ):
                            raise CompanionCaptureConflict(
                                "client request ID conflicts with the existing companion session"
                            )
                        return deepcopy(prior)
            if self._active_session_id is not None:
                active = self._read(self._active_session_id)
                raise CompanionCaptureBusy(f"companion session {active['session_id']} is already active")
            source_uri, authority_object = self._resolve_source(camera)
            authority = _public_authority(authority_object.public_snapshot())
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            session_id = f"companion-{stamp}-{uuid.uuid4().hex[:8]}"
            output_dir = self.session_root / session_id
            output_dir.mkdir(parents=True, exist_ok=False)
            (output_dir / "runtime_provenance.json").write_text(
                json.dumps(
                    {
                        "schema": "noesis.companion_capture.runtime_provenance.v1",
                        "camera_id": camera.camera_id,
                        "source_reference": camera.source_reference,
                        "authority": authority,
                        "captured_at": _utc_now(),
                        "source_clock": "rtsp_source_clock_unverified",
                        "phone_server_clock_exchange_verified": False,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            try:
                recorder = self.recorder_factory(
                    source_uri,
                    output_dir,
                    max_duration_s=self.limits.max_duration_s,
                    max_bytes=self.limits.max_session_bytes,
                    max_packet_records=self.limits.max_packet_records,
                )
            except TypeError:
                recorder = self.recorder_factory(source_uri, output_dir)
            observer = self._make_observer(camera, output_dir, authority_object)
            observer_snapshot = self._observer_snapshot(observer)
            if observer_snapshot:
                try:
                    (output_dir / "tracking_provenance.json").write_text(
                        json.dumps(observer_snapshot, indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
                except OSError as exc:
                    raise CompanionCaptureError(
                        "tracking provenance artifact could not be written"
                    ) from exc
            ready_event = threading.Event()
            now_mono = time.monotonic()
            state: dict[str, Any] = {
                "schema": COMPANION_SCHEMA,
                "session_id": session_id,
                "camera": {
                    "camera_id": camera.camera_id,
                    "source_id": camera.source_id,
                    "label": camera.label,
                    "source_reference": camera.source_reference,
                    "authority": authority,
                },
                "client_request_id": request_id,
                "phone_capture_id": capture_id,
                "status": "starting",
                "created_at": _utc_now(),
                "started_at": None,
                "started_monotonic_s": now_mono,
                "lease": {
                    "duration_s": self.limits.lease_s,
                    "last_heartbeat_at": _utc_now(),
                    "expires_monotonic_s": now_mono + self.limits.readiness_timeout_s,
                },
                "limits": {
                    "max_duration_s": self.limits.max_duration_s,
                    "max_session_bytes": self.limits.max_session_bytes,
                    "max_tracking_records": self.limits.max_tracking_records,
                    "max_packet_records": self.limits.max_packet_records,
                },
                "readiness": {
                    "deadline_s": self.limits.readiness_timeout_s,
                    "recorder_ready": False,
                    "tracking_ready": False,
                },
                "markers": [],
                "clock_exchanges": [],
                "artifacts": {
                    "runtime_provenance": "runtime_provenance.json",
                    **({"tracking_provenance": "tracking_provenance.json"} if observer_snapshot else {}),
                },
                "provenance": {
                    "schema": "noesis.companion_capture.provenance.v1",
                    "camera_source": authority,
                    "runtime_artifact": "runtime_provenance.json",
                    "tracking_artifact": "tracking_provenance.json" if observer_snapshot else None,
                    "tracking_observer": observer_snapshot,
                    "timing": {
                        "static_source_clock": "rtsp_source_clock_unverified",
                        "phone_server_clock_exchange_verified": False,
                        "acquisition_alignment": "unverified",
                    },
                },
                "phone": {"capture_id": capture_id, "scan_id": None, "archive_sha256": None},
                "error": None,
            }
            for probe in raw_clock_probes:
                self._append_clock_probe_locked(state, probe)
            self._write(session_id, state)
            self._runtime[session_id] = (recorder, observer, ready_event)
            self._active_session_id = session_id
        try:
            recorder.start()
            observer.start()
        except Exception as exc:
            # Both starts can fail after allocating native resources.  Stop
            # every participant before releasing the runtime slot so a failed
            # admission cannot leave an RTSP pipeline or writer orphaned.
            try:
                observer.stop("start_failed")
            except Exception:
                pass
            try:
                recorder.stop("start_failed")
            except Exception:
                pass
            with self._lock:
                state = self._read(session_id)
                state.update({
                    "status": "failed",
                    "error": redact_runtime_secrets(
                        f"start:{type(exc).__name__}: {exc}"
                    )[:500],
                })
                self._write(session_id, state)
                self._active_session_id = None
                self._runtime.pop(session_id, None)
            raise CompanionCaptureError("paired capture could not start") from exc
        worker = threading.Thread(target=self._await_readiness, args=(session_id, ready_event), name="CompanionReadiness", daemon=True)
        worker.start()
        if await_ready:
            ready_event.wait(self.limits.readiness_timeout_s + 0.25)
        return self.status(session_id)

    def _expire_if_needed_locked(self) -> None:
        if not self._active_session_id:
            return
        try:
            state = self._read(self._active_session_id)
        except CompanionCaptureError:
            self._active_session_id = None
            return
        if state.get("status") in {"starting", "recording"}:
            expires = float(state.get("lease", {}).get("expires_monotonic_s") or 0.0)
            if time.monotonic() >= expires:
                self.stop_session(self._active_session_id, reason="lease_expired")

    def _refresh_runtime(self, session_id: str) -> None:
        with self._lock:
            runtime = self._runtime.get(session_id)
            if runtime is None:
                return
            recorder, observer, _ = runtime
            state = self._read(session_id)
            readiness = state.setdefault("readiness", {})
            readiness["recorder"] = _status_value(recorder)
            readiness["tracking"] = _status_value(observer)
            self._write(session_id, state)

    @staticmethod
    def _validated_clock_probes(clock_probes: Any) -> list[dict[str, Any]]:
        if clock_probes is None:
            return []
        if not isinstance(clock_probes, (list, tuple)):
            raise CompanionCaptureError("clock_probes must be a JSON array")
        if len(clock_probes) > MAX_CLOCK_EXCHANGES:
            raise CompanionCaptureError("clock_probes exceeds its size limit")
        result: list[dict[str, Any]] = []
        for probe in clock_probes:
            if not isinstance(probe, Mapping):
                raise CompanionCaptureError("clock probe must be a JSON object")
            checked = _safe_json(dict(probe), max_bytes=MAX_MARKER_BYTES)
            if not isinstance(checked, dict):
                raise CompanionCaptureError("clock probe must be a JSON object")
            result.append(checked)
        return result

    @staticmethod
    def _append_clock_probe_locked(
        state: dict[str, Any],
        probe: Mapping[str, Any],
        *,
        request_id: str | None = None,
    ) -> None:
        exchanges = state.setdefault("clock_exchanges", [])
        probe_id = str(
            probe.get("request_id")
            or probe.get("client_request_id")
            or request_id
            or ""
        ).strip() or None
        if probe_id and any(row.get("client_request_id") == probe_id for row in exchanges):
            return
        now_mono = time.monotonic_ns()
        now_utc = time.time_ns()
        exchange = {
            "client_request_id": probe_id,
            "server_monotonic_ns": now_mono,
            "server_utc_ns": now_utc,
            "server_received_monotonic_ns": now_mono,
            "server_received_unix_ns": now_utc,
            "server_sent_monotonic_ns": time.monotonic_ns(),
            "server_sent_unix_ns": time.time_ns(),
            "client_probe": dict(probe),
            "synchronization_verified": False,
            "uncertainty_ms": None,
        }
        _safe_json(exchange, max_bytes=MAX_MARKER_BYTES * 2)
        if len(exchanges) >= MAX_CLOCK_EXCHANGES:
            exchanges.pop(0)
        exchanges.append(exchange)

    def status(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            self._expire_if_needed_locked()
            self._refresh_runtime(session_id)
            return deepcopy(self._read(session_id))

    def heartbeat(
        self,
        session_id: str,
        *,
        phone_capture_id: str | None = None,
        client_request_id: str | None = None,
        client_time_ms: float | None = None,
        client_send_time_ms: float | None = None,
        client_receive_time_ms: float | None = None,
        clock_probes: Any | None = None,
    ) -> dict[str, Any]:
        failure_reason: str | None = None
        with self._lock:
            state = self._read(session_id)
            self._refresh_runtime(session_id)
            state = self._read(session_id)
            bound_capture_id = str(
                state.get("phone_capture_id")
                or (state.get("phone") or {}).get("capture_id")
                or ""
            )
            supplied_capture_id = str(phone_capture_id or "").strip()
            if supplied_capture_id and bound_capture_id and supplied_capture_id != bound_capture_id:
                raise CompanionCaptureConflict("phone capture ID conflicts with this companion session")
            if state.get("status") not in {"starting", "recording"}:
                return deepcopy(state)
            failure_reason = self._active_component_failure(state)
            if failure_reason:
                # Stop outside this lock after the current state snapshot is
                # released.  This preserves the original component reason
                # while allowing the recorder and tracking wrapper to flush.
                pass
            else:
                exchanges = state.setdefault("clock_exchanges", [])
                request_id = str(client_request_id or "").strip() or None
                if request_id and any(row.get("client_request_id") == request_id for row in exchanges):
                    return deepcopy(state)
                raw_probes = self._validated_clock_probes(clock_probes)
                exchange = {
                    "client_request_id": request_id,
                    "server_monotonic_ns": time.monotonic_ns(),
                    "server_utc_ns": time.time_ns(),
                    "client_time_ms": client_time_ms,
                    "client_send_time_ms": client_send_time_ms,
                    "client_receive_time_ms": client_receive_time_ms,
                    "synchronization_verified": False,
                    "uncertainty_ms": None,
                }
                if raw_probes:
                    for probe in raw_probes:
                        self._append_clock_probe_locked(state, probe, request_id=request_id)
                else:
                    _safe_json(exchange)
                    if len(exchanges) >= MAX_CLOCK_EXCHANGES:
                        exchanges.pop(0)
                    exchanges.append(exchange)
                state["lease"].update({"last_heartbeat_at": _utc_now(), "expires_monotonic_s": time.monotonic() + self.limits.lease_s})
                self._write(session_id, state)
                return deepcopy(state)
        if failure_reason:
            return self.stop_session(session_id, reason=failure_reason)
        return state

    def add_marker(self, session_id: str, marker: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            state = self._read(session_id)
            supplied_capture_id = str(marker.get("phone_capture_id") or "").strip()
            bound_capture_id = str(
                state.get("phone_capture_id")
                or (state.get("phone") or {}).get("capture_id")
                or ""
            )
            if supplied_capture_id and bound_capture_id and supplied_capture_id != bound_capture_id:
                raise CompanionCaptureConflict("phone capture ID conflicts with this companion session")
            marker_id = str(
                marker.get("client_marker_id")
                or marker.get("marker_id")
                or marker.get("id")
                or ""
            ).strip()
            if not marker_id or len(marker_id) > 160:
                raise CompanionCaptureError("client_marker_id is required")
            existing = next((row for row in state.get("markers", []) if row.get("client_marker_id") == marker_id), None)
            if existing is not None:
                state["last_marker_receipt_clock"] = {
                    "server_monotonic_ns": existing.get("server_monotonic_ns"),
                    "server_utc_ns": existing.get("server_utc_ns"),
                }
                return deepcopy(state)
            rows = state.setdefault("markers", [])
            if len(rows) >= self.limits.max_markers:
                raise CompanionCaptureError("marker limit reached")
            label = str(marker.get("label") or marker.get("type") or "marker").strip()
            if not label or len(label) > 160:
                raise CompanionCaptureError("marker label is invalid")
            row = {
                "client_marker_id": marker_id,
                "marker_id": marker_id,
                "label": label,
                "client_time_ms": marker.get("client_time_ms")
                if marker.get("client_time_ms") is not None
                else marker.get("client_monotonic_ms"),
                "client_monotonic_ms": marker.get("client_monotonic_ms"),
                "client_epoch_ms": marker.get("client_epoch_ms"),
                "server_monotonic_ns": time.monotonic_ns(),
                "server_utc_ns": time.time_ns(),
                "data": _safe_json(marker.get("data"), max_bytes=MAX_MARKER_BYTES) if marker.get("data") is not None else None,
            }
            rows.append(row)
            marker_path = self.session_root / session_id / "markers.jsonl"
            with marker_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            state["last_marker_receipt_clock"] = {
                "server_monotonic_ns": row["server_monotonic_ns"],
                "server_utc_ns": row["server_utc_ns"],
            }
            self._write(session_id, state)
            return deepcopy(state)

    def stop_session(
        self,
        session_id: str,
        *,
        reason: str = "user",
        phone_capture_id: str | None = None,
        clock_probes: Any | None = None,
        markers: Any | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            state = self._read(session_id)
            bound_capture_id = str(
                state.get("phone_capture_id")
                or (state.get("phone") or {}).get("capture_id")
                or ""
            )
            supplied_capture_id = str(phone_capture_id or "").strip()
            if supplied_capture_id and bound_capture_id and supplied_capture_id != bound_capture_id:
                raise CompanionCaptureConflict("phone capture ID conflicts with this companion session")
            if state.get("status") in {"stopped", "complete", "partial", "failed", "expired", "interrupted"}:
                return deepcopy(state)
            # The browser sends final clock probes and any marker receipts
            # that were pending when it stopped.  Preserve both before the
            # lifecycle state moves to stopping; each operation is bounded and
            # idempotent by its client identifier.
            raw_probes = self._validated_clock_probes(clock_probes)
            for probe in raw_probes:
                self._append_clock_probe_locked(state, probe)
            if raw_probes:
                self._write(session_id, state)
            if markers is not None:
                if not isinstance(markers, (list, tuple)):
                    raise CompanionCaptureError("markers must be a JSON array")
                if len(markers) > self.limits.max_markers:
                    raise CompanionCaptureError("markers exceeds its size limit")
                for marker in markers:
                    if not isinstance(marker, Mapping):
                        raise CompanionCaptureError("marker must be a JSON object")
                    marker_payload = dict(marker)
                    if supplied_capture_id:
                        marker_payload.setdefault("phone_capture_id", supplied_capture_id)
                    # add_marker persists its own state and receipt.  The
                    # reentrant manager lock keeps this stop request ordered.
                    state = self.add_marker(session_id, marker_payload)
            state["status"] = "stopping"
            state["stop_reason"] = str(reason)
            state["stopped_at"] = _utc_now()
            self._write(session_id, state)
            runtime = self._runtime.get(session_id)
        errors: list[str] = []
        recorder_result: Mapping[str, Any] = {}
        tracking_result: Mapping[str, Any] = {}
        if runtime is not None:
            recorder, observer, _ = runtime
            if str(reason) != "user":
                mark_partial = getattr(observer, "mark_partial", None)
                if callable(mark_partial):
                    try:
                        mark_partial(str(reason))
                    except Exception as exc:
                        errors.append(f"tracking_partial:{type(exc).__name__}: {str(exc)[:300]}")
            try:
                recorder_result = _status_value(recorder.stop(reason))
            except Exception as exc:
                errors.append(f"recorder:{type(exc).__name__}: {str(exc)[:500]}")
            try:
                tracking_result = _status_value(observer.stop(reason))
            except Exception as exc:
                errors.append(f"tracking:{type(exc).__name__}: {str(exc)[:500]}")
        for label, result in (("recorder", recorder_result), ("tracking", tracking_result)):
            if not isinstance(result, Mapping):
                continue
            result_error = result.get("error")
            if result_error not in (None, ""):
                errors.append(f"{label}:{str(result_error)[:500]}")
            if str(result.get("status") or result.get("state") or "").lower() in {"failed", "error"}:
                errors.append(f"{label}:failed")
            if result.get("partial") is True:
                errors.append(f"{label}:partial")
        with self._lock:
            state = self._read(session_id)
            session_dir = self.session_root / session_id
            artifacts = state.setdefault("artifacts", {})
            for key, filename in (
                ("static_video", "static_camera.mkv"),
                ("packet_timing", "packet_timing.jsonl"),
                ("tracking", "tracking.jsonl"),
                ("markers", "markers.jsonl"),
            ):
                if (session_dir / filename).is_file():
                    artifacts[key] = filename
            for key, result, candidates in (
                ("static_video", recorder_result, ("video_path", "static_video_path")),
                ("packet_timing", recorder_result, ("packet_timing_path", "packet_path")),
                ("tracking", tracking_result, ("tracking_path", "record_path", "artifact")),
            ):
                if key in artifacts:
                    continue
                for candidate in candidates:
                    raw = result.get(candidate) if isinstance(result, Mapping) else None
                    if not isinstance(raw, str):
                        continue
                    path = Path(raw)
                    if path.is_absolute():
                        try:
                            path = path.resolve().relative_to(session_dir.resolve())
                        except ValueError:
                            continue
                    if ".." in path.parts:
                        continue
                    if (session_dir / path).is_file():
                        artifacts[key] = path.as_posix()
                        break
            # CanonicalTrackingRecorder reports its bounded tracking and REST
            # evidence under an artifacts map.  Import only relative files
            # that actually exist inside this session directory.
            for source_label, result in (("recorder", recorder_result), ("tracking", tracking_result)):
                reported = result.get("artifacts") if isinstance(result, Mapping) else None
                if not isinstance(reported, Mapping):
                    continue
                for key, raw in reported.items():
                    if not isinstance(raw, str):
                        continue
                    path = Path(raw)
                    if path.is_absolute():
                        try:
                            path = path.resolve().relative_to(session_dir.resolve())
                        except ValueError:
                            continue
                    if ".." in path.parts or not (session_dir / path).is_file():
                        continue
                    artifact_key = str(key)
                    artifact_key = {
                        "video": "static_video",
                        "static_video": "static_video",
                        "packet_timing": "packet_timing",
                        "tracking": "tracking",
                        "markers": "markers",
                    }.get(artifact_key, artifact_key)
                    if artifact_key in {"static_video", "packet_timing", "tracking", "markers"}:
                        artifacts[artifact_key] = path.as_posix()
                    else:
                        prefix = "tracking_" if source_label == "tracking" else "recorder_"
                        artifacts.setdefault(f"{prefix}{artifact_key}", path.as_posix())
            if recorder_result:
                state["recorder"] = recorder_result
            if tracking_result:
                state["tracking"] = tracking_result
            state["artifacts"] = artifacts
            state["finalized_at"] = _utc_now()
            state["error"] = "; ".join(errors) if errors else state.get("error")
            forced_failure = str(reason) not in {"user", "completed", "stopped", "user_stop"}
            state["status"] = "partial" if forced_failure or errors or not artifacts.get("static_video") or not artifacts.get("tracking") else "stopped"
            if state.get("phone", {}).get("scan_id"):
                state["status"] = "complete" if not errors else "partial"
            self._write(session_id, state)
            if self._active_session_id == session_id:
                self._active_session_id = None
            self._runtime.pop(session_id, None)
            return deepcopy(state)

    def recover_orphans(self) -> None:
        with self._lock:
            for path in self.session_root.glob("*/session.json"):
                session_id = path.parent.name
                try:
                    state = self._read(session_id)
                except CompanionCaptureError:
                    continue
                if state.get("status") in {"starting", "recording", "stopping"}:
                    state.update({"status": "interrupted", "stop_reason": "service_restart", "error": "companion capture was interrupted by service restart", "finalized_at": _utc_now()})
                    self._write(session_id, state)

    def reserve_phone_scan_id(self, session_id: str, capture_id: str) -> str:
        if COMPANION_CAPTURE_ID_PATTERN.fullmatch(str(capture_id or "")) is None:
            raise CompanionCaptureError("phone capture ID is invalid")
        with self._lock:
            state = self._read(session_id)
            prior = state.get("phone") or {}
            prior_capture = str(prior.get("capture_id") or state.get("phone_capture_id") or "")
            if prior_capture and prior_capture != capture_id:
                raise CompanionCaptureConflict("companion session is already bound to another phone capture")
            if prior.get("scan_id"):
                return str(prior["scan_id"])
            # The reservation is stable for retries.  The phone service's
            # normal scan-ID format remains intact.
            reserved = f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
            state["phone_capture_id"] = capture_id
            state["phone"] = {"capture_id": capture_id, "scan_id": reserved, "archive_sha256": None}
            self._write(session_id, state)
            return reserved

    def check_phone_archive(self, session_id: str, capture_id: str, archive_sha256: str) -> dict[str, Any] | None:
        with self._lock:
            state = self._read(session_id)
            phone = state.get("phone") or {}
            bound = str(phone.get("capture_id") or state.get("phone_capture_id") or "")
            if bound and bound != capture_id:
                raise CompanionCaptureConflict("phone capture ID conflicts with this companion session")
            prior_hash = str(phone.get("archive_sha256") or "")
            if prior_hash and prior_hash != archive_sha256:
                raise CompanionCaptureConflict("phone archive conflicts with the already associated archive")
            if prior_hash == archive_sha256 and phone.get("scan_id"):
                return deepcopy(state)
            return None

    def associate_phone_bundle(
        self,
        session_id: str,
        *,
        capture_id: str,
        archive_sha256: str,
        scan_id: str,
    ) -> dict[str, Any]:
        if not re.fullmatch(r"[a-f0-9]{64}", str(archive_sha256 or "")):
            raise CompanionCaptureError("phone archive SHA-256 is invalid")
        with self._lock:
            state = self._read(session_id)
            phone = state.setdefault("phone", {})
            bound = str(phone.get("capture_id") or state.get("phone_capture_id") or "")
            if bound and bound != capture_id:
                raise CompanionCaptureConflict("phone capture ID conflicts with this companion session")
            if phone.get("archive_sha256") and phone.get("archive_sha256") != archive_sha256:
                raise CompanionCaptureConflict("phone archive conflicts with the already associated archive")
            previous_failed_path = phone.pop("failed_archive_path", None)
            phone.pop("upload_error", None)
            if previous_failed_path:
                phone["previous_failed_archive_path"] = previous_failed_path
            phone.update({"capture_id": capture_id, "archive_sha256": archive_sha256, "scan_id": scan_id, "associated_at": _utc_now()})
            state["phone_capture_id"] = capture_id
            state["phone"] = phone
            # An upload failure is phone-side state.  Recover a partial session
            # only when the finalized static recorder/tracking evidence itself
            # remains complete and error-free.
            if state.get("status") == "stopped" or (
                state.get("status") == "partial" and self._static_capture_healthy(state)
            ):
                state["status"] = "complete"
            self._write(session_id, state)
            return deepcopy(state)

    def record_upload_failure(self, session_id: str, capture_id: str, temporary_archive: Path, error: str) -> dict[str, Any]:
        with self._lock:
            state = self._read(session_id)
            failure_dir = self.session_root / session_id / "phone_uploads"
            failure_dir.mkdir(parents=True, exist_ok=True)
            destination = failure_dir / f"{capture_id}.partial"
            try:
                os.replace(temporary_archive, destination)
            except OSError:
                # The normal upload temp path is on the same storage root, so
                # rename is atomic.  Preserve the archive across a rare
                # cross-device or transient rename failure with a bounded
                # streaming copy rather than dropping the phone evidence.
                try:
                    shutil.copyfile(temporary_archive, destination)
                    Path(temporary_archive).unlink(missing_ok=True)
                except OSError:
                    # The original temp remains available for a later retry;
                    # do not publish an absolute machine path in session state.
                    destination = None
            phone = state.setdefault("phone", {})
            update: dict[str, Any] = {
                "capture_id": capture_id,
                "upload_error": str(error)[:500],
            }
            if destination is not None:
                update["failed_archive_path"] = destination.relative_to(
                    self.session_root / session_id
                ).as_posix()
            phone.update(update)
            state["phone"] = phone
            # Keep the static capture lifecycle intact.  A phone archive
            # failure must not turn a healthy stopped/complete static session
            # into a static partial or failed result.
            self._write(session_id, state)
            return deepcopy(state)

    def public_state(self, session_id: str) -> dict[str, Any]:
        state = self.status(session_id)
        failure_reason = self._active_component_failure(state)
        if failure_reason:
            # A GET/heartbeat projection must not continue advertising a live
            # session after either component has failed.  Stop the owned
            # participants before returning the failed lifecycle state.
            state = self.stop_session(session_id, reason=failure_reason)
        public = deepcopy(state)
        public.pop("client_request_id", None)
        public.pop("started_monotonic_s", None)
        lease = public.get("lease")
        if isinstance(lease, dict):
            lease.pop("expires_monotonic_s", None)
        camera = public.get("camera") if isinstance(public.get("camera"), Mapping) else {}
        readiness = public.get("readiness") if isinstance(public.get("readiness"), Mapping) else {}
        recorder = public.get("recorder") if isinstance(public.get("recorder"), Mapping) else None
        tracking = public.get("tracking") if isinstance(public.get("tracking"), Mapping) else None
        # Active component snapshots live below readiness; finalization copies
        # the same snapshots to the top level.  Prefer the finalized copy.
        if not recorder:
            recorder = readiness.get("recorder") if isinstance(readiness.get("recorder"), Mapping) else {}
        if not tracking:
            tracking = readiness.get("tracking") if isinstance(readiness.get("tracking"), Mapping) else {}
        recorder = dict(recorder or {})
        tracking = dict(tracking or {})
        raw_status = str(public.get("status") or "").lower()
        lifecycle_status = {
            "partial": "failed",
            "expired": "failed",
            "interrupted": "failed",
            "error": "failed",
            "stopping": "finalizing",
            "complete": "stopped",
        }.get(raw_status, raw_status)
        public.update(
            {
                "status": lifecycle_status,
                "camera_id": camera.get("camera_id"),
                "phone_capture_id": public.get("phone_capture_id") or (public.get("phone") or {}).get("capture_id"),
                "video_status": "recording" if public.get("status") in {"starting", "recording"} and recorder.get("error") in (None, "") else str(recorder.get("status") or "failed"),
                "tracking_status": "recording"
                if public.get("status") in {"starting", "recording"} and tracking.get("partial") is not True
                else (
                    "partial"
                    if tracking.get("partial")
                    else str(tracking.get("state") or tracking.get("status") or "failed")
                ),
                "tracking_sequence": tracking.get("last_tracking_publication_sequence")
                if tracking.get("last_tracking_publication_sequence") is not None
                else tracking.get("tracking_sequence", tracking.get("records")),
                "static_timestamp": recorder.get("first_packet_utc_ns") or recorder.get("started_utc_ns"),
                "limits": public.get("limits") or {},
                "provenance": public.get("provenance") or camera.get("authority") or {},
                "clock_probes": public.get("clock_exchanges") or [],
            }
        )
        artifacts = public.get("artifacts")
        if isinstance(artifacts, Mapping):
            public["artifact_urls"] = {
                key: f"/companion-assets/{session_id}/{value}"
                for key, value in artifacts.items()
                if isinstance(value, str)
                and ".." not in Path(value).parts
                and not Path(value).is_absolute()
                and (self.session_root / session_id / value).is_file()
            }
        return public

    def list_sessions(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with self._lock:
            self._expire_if_needed_locked()
            for path in self.session_root.glob("*/session.json"):
                try:
                    rows.append(self.public_state(path.parent.name))
                except CompanionCaptureError:
                    continue
        rows.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)
        return rows

    def finalize_session(self, session_id: str, *, reason: str = "finalize") -> dict[str, Any]:
        """Close an active session and make its current evidence readable."""

        state = self._read(session_id)
        if state.get("status") in {"starting", "recording", "stopping"}:
            state = self.stop_session(session_id, reason=reason)
        return self.public_state(session_id)


__all__ = [
    "COMPANION_SCHEMA",
    "COMPANION_SESSION_PATTERN",
    "CompanionCamera",
    "CompanionCaptureBusy",
    "CompanionCaptureConflict",
    "CompanionCaptureError",
    "CompanionCaptureLimits",
    "CompanionCaptureManager",
    "GstCompressedRecorder",
]
