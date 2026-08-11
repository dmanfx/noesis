"""Fail-closed ownership for exact and manual live depth capture windows.

One controller instance must be shared by the depth and floorplan providers in
a runtime.  It serializes aliases onto one non-blocking admission lock per
canonical camera, fences MapAnything and durable storage on both sides of the
burst, and permits exactly one configured fusion after the closing fence.  The
same controller also owns the process-wide manual refresh window used by REST;
that window has one non-daemon owner through gate close and both closing
barriers.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from noesis.depth_capture_event import DepthStorageFlushEvidence
from noesis_core.capture_event_fusion import (
    CaptureEventFusionCoordinator,
    CaptureEventFusionError,
    CaptureEventFusionRequest,
    CaptureEventStore,
    FusedDepthSnapshot,
    TimestampedRgbProvider,
    canonical_json_sha256,
    freeze_json_evidence,
    thaw_json_evidence,
)
from noesis_core.mapanything_lifecycle import MapAnythingIdleReceipt

_REQUEST_KINDS = frozenset({"depth", "floorplan"})
_MAX_COUNTER = (1 << 63) - 1
_CONTROLLER_ERROR_CODES = frozenset(
    {
        "camera_required",
        "unknown_camera_alias",
        "cache_only_forbids_capture_event",
        "capture_event_busy",
        "capture_event_cancelled",
        "depth_gate_transition_failed",
        "depth_gate_state_mismatch",
        "mapanything_idle_failed",
        "mapanything_idle_receipt_invalid",
        "depth_storage_flush_failed",
        "depth_storage_frontier_unclean",
        "depth_raw_baseline_failed",
        "capture_burst_wait_failed",
        "fused_snapshot_validation_failed",
    }
)
_GLOBAL_COUNTER_KEYS = (
    "requests_total",
    "admitted_total",
    "completed_total",
    "failed_total",
    "busy_total",
    "cache_only_rejected_total",
    "fusion_total",
    "fatal_barrier_failures_total",
)
_MANUAL_REFRESH_COUNTER_KEYS = (
    "requests_total",
    "admitted_total",
    "extended_total",
    "completed_total",
    "failed_total",
    "busy_total",
    "shutdown_total",
)
_MANUAL_REFRESH_CAMERA = "manual-refresh"
_DEFAULT_MANUAL_REFRESH_DRAIN_TIMEOUT_S = 35.0
_FATAL_ERROR_CODES = frozenset(
    {
        "depth_gate_transition_failed",
        "depth_gate_state_mismatch",
        "mapanything_idle_failed",
        "mapanything_idle_receipt_invalid",
        "depth_storage_flush_failed",
        "depth_storage_frontier_unclean",
        "depth_raw_baseline_failed",
        "capture_burst_wait_failed",
        "raw_snapshot_listing_failed",
        "untyped_raw_snapshot",
        "raw_snapshot_scope_mismatch",
        "raw_snapshot_precedes_baseline",
        "duplicate_raw_snapshot_evidence",
        "duplicate_raw_source_frame_identity",
        "capture_event_fusion_failed",
        "untyped_fused_snapshot",
        "fused_snapshot_evidence_mismatch",
        "fused_snapshot_validation_failed",
        "rgb_provider_failed",
        "rgb_depth_identity_unavailable",
        "rgb_camera_mismatch",
        "rgb_frame_identity_mismatch",
        "rgb_frame_identity_ambiguous",
    }
)


def capture_event_error_is_fatal(code: object) -> bool:
    return str(code or "").strip() in _FATAL_ERROR_CODES


class CaptureEventControllerError(CaptureEventFusionError):
    """Stable public controller failure; exception text is always its code."""

    def __init__(self, code: str, *, details: Mapping[str, Any] | None = None) -> None:
        if code not in _CONTROLLER_ERROR_CODES:
            raise ValueError(f"unknown capture-event controller error code: {code}")
        super().__init__(code, details=details)


class CanonicalCameraAliases:
    """Immutable alias table binding source indexes to canonical storage ids."""

    def __init__(
        self,
        camera_labels: Mapping[int, str],
        *,
        extra_aliases: Mapping[str, str] | None = None,
    ) -> None:
        canonical_by_source: dict[int, str] = {}
        for raw_source, raw_camera in camera_labels.items():
            if type(raw_source) is not int or raw_source < 0:  # noqa: E721
                raise ValueError("camera source ids must be non-negative integers")
            camera = str(raw_camera or "").strip()
            if not camera:
                raise ValueError("canonical camera ids cannot be empty")
            if camera.isdigit():
                raise ValueError("canonical camera ids cannot be numeric aliases")
            if raw_source in canonical_by_source:
                raise ValueError(f"duplicate camera source id: {raw_source}")
            canonical_by_source[raw_source] = camera
        if not canonical_by_source:
            raise ValueError("at least one canonical camera is required")
        canonical = tuple(sorted(set(canonical_by_source.values())))
        if len(canonical) != len(canonical_by_source):
            raise ValueError("canonical camera ids must be unique")

        aliases: dict[str, str] = {}

        def bind(alias: object, target: str) -> None:
            key = str(alias or "").strip()
            if not key:
                raise ValueError("camera aliases cannot be empty")
            prior = aliases.get(key)
            if prior is not None and prior != target:
                raise ValueError(
                    f"camera alias {key!r} is ambiguous between {prior!r} and {target!r}"
                )
            aliases[key] = target

        for source_id, camera in sorted(canonical_by_source.items()):
            bind(camera, camera)
            bind(str(source_id), camera)
        for raw_alias, raw_target in (extra_aliases or {}).items():
            target = str(raw_target or "").strip()
            if target not in canonical:
                raise ValueError(f"camera alias target is not canonical: {target!r}")
            bind(raw_alias, target)

        self._aliases = aliases
        self._canonical = canonical

    @property
    def canonical_cameras(self) -> tuple[str, ...]:
        return self._canonical

    @property
    def alias_count(self) -> int:
        return len(self._aliases)

    def canonicalize(self, camera_alias: object) -> str:
        alias = str(camera_alias or "").strip()
        if not alias:
            raise CaptureEventControllerError("camera_required")
        canonical = self._aliases.get(alias)
        if canonical is None:
            raise CaptureEventControllerError(
                "unknown_camera_alias",
                details={"camera_alias": alias},
            )
        return canonical


class CaptureEventControllerStorage(CaptureEventStore, Protocol):
    def flush_capture_frontier(
        self,
        *,
        timeout_s: float,
    ) -> DepthStorageFlushEvidence: ...

    def max_raw_timestamp(self, camera_id: str) -> int: ...

    def validate_fused_snapshot(
        self,
        snapshot: FusedDepthSnapshot,
    ) -> FusedDepthSnapshot: ...


class MapAnythingIdleBarrier(Protocol):
    def wait_idle(self, *, timeout_s: float = 5.0) -> MapAnythingIdleReceipt: ...


class CaptureEventRgbProvider(TimestampedRgbProvider, Protocol):
    def arm(self, camera_id: str) -> object: ...

    def disarm(self, arm: object) -> None: ...


@dataclass(frozen=True)
class CaptureEventRequest:
    camera_alias: str
    request_kind: str
    cache_only: bool = False
    burst_seconds: float = 20.0
    drain_timeout_s: float = 8.0
    raw_limit: int = 24
    min_observations: int = 3
    depth_agreement_m: float = 0.18
    max_cohort_span_us: int = 20_000_000

    def __post_init__(self) -> None:
        camera_alias = str(self.camera_alias or "").strip()
        request_kind = str(self.request_kind or "").strip().lower()
        object.__setattr__(self, "camera_alias", camera_alias)
        object.__setattr__(self, "request_kind", request_kind)
        if request_kind not in _REQUEST_KINDS:
            raise ValueError(f"request_kind must be one of {sorted(_REQUEST_KINDS)}")
        if type(self.cache_only) is not bool:  # noqa: E721
            raise ValueError("cache_only must be a boolean")
        burst_seconds = float(self.burst_seconds)
        drain_timeout_s = float(self.drain_timeout_s)
        depth_agreement_m = float(self.depth_agreement_m)
        if not math.isfinite(burst_seconds) or not 0.01 <= burst_seconds <= 30.0:
            raise ValueError("burst_seconds must be in [0.01, 30]")
        if not math.isfinite(drain_timeout_s) or not 0.1 <= drain_timeout_s <= 60.0:
            raise ValueError("drain_timeout_s must be in [0.1, 60]")
        if not math.isfinite(depth_agreement_m) or depth_agreement_m <= 0.0:
            raise ValueError("depth_agreement_m must be finite and positive")
        object.__setattr__(self, "burst_seconds", burst_seconds)
        object.__setattr__(self, "drain_timeout_s", drain_timeout_s)
        object.__setattr__(self, "depth_agreement_m", depth_agreement_m)
        if type(self.raw_limit) is not int or not 1 <= self.raw_limit <= 256:  # noqa: E721
            raise ValueError("raw_limit must be in [1, 256]")
        if (  # noqa: E721
            type(self.min_observations) is not int
            or not 1 <= self.min_observations <= self.raw_limit
        ):
            raise ValueError("min_observations must be in [1, raw_limit]")
        if (  # noqa: E721
            type(self.max_cohort_span_us) is not int or self.max_cohort_span_us <= 0
        ):
            raise ValueError("max_cohort_span_us must be a positive integer")


@dataclass(frozen=True)
class CaptureEventOutcome:
    canonical_camera: str
    request_kind: str
    fused_snapshot: FusedDepthSnapshot
    fusion_evidence_sha256: str
    compact_evidence: Mapping[str, Any]
    compact_evidence_sha256: str

    def __post_init__(self) -> None:
        frozen = freeze_json_evidence(self.compact_evidence)
        if not isinstance(frozen, Mapping):
            raise ValueError("compact_evidence must be a mapping")
        object.__setattr__(self, "compact_evidence", frozen)
        if canonical_json_sha256(frozen) != self.compact_evidence_sha256:
            raise ValueError("compact_evidence_sha256 does not match evidence")

    def compact_evidence_payload(self) -> dict[str, Any]:
        return thaw_json_evidence(self.compact_evidence)


def _default_burst_waiter(seconds: float) -> bool:
    time.sleep(seconds)
    return False


def _fused_evidence(snapshot: FusedDepthSnapshot) -> dict[str, Any]:
    return {
        "camera_id": snapshot.camera_id,
        "storage_key": snapshot.storage_key,
        "timestamp_us": snapshot.timestamp_us,
        "snapshot_id": snapshot.snapshot_id,
        "artifact_ref": snapshot.artifact_ref,
        "content_sha256": snapshot.content_sha256,
        "sequence": snapshot.sequence,
        "manifest_sha256": snapshot.manifest_sha256,
        "event_id": snapshot.event_id,
        "source_snapshot_ids": list(snapshot.source_snapshot_ids),
        "snapshot_role": snapshot.snapshot_role,
        "fusion_level": snapshot.fusion_level,
    }


def _idle_evidence(receipt: MapAnythingIdleReceipt) -> dict[str, Any]:
    return {
        "active_captures": receipt.active_captures,
        "unfinished_tasks": receipt.unfinished_tasks,
        "worker_started": receipt.worker_started,
        "worker_alive": receipt.worker_alive,
        "accepting": receipt.accepting,
    }


def _flush_evidence(receipt: DepthStorageFlushEvidence) -> dict[str, Any]:
    return {
        "frontier_sequence": receipt.frontier_sequence,
        "completed": receipt.completed,
        "timed_out": receipt.timed_out,
        "pending_sequences": list(receipt.pending_sequences),
        "failed_sequences": list(receipt.failed_sequences),
        "poisoned": receipt.poisoned,
    }


class CaptureEventController:
    """Own per-camera capture admission, barriers, and one exact fusion."""

    def __init__(
        self,
        *,
        aliases: CanonicalCameraAliases | Mapping[int, str],
        storage: CaptureEventControllerStorage,
        mapanything: MapAnythingIdleBarrier,
        set_depth_gate: Callable[[bool], None],
        depth_gate_is_open: Callable[[], bool],
        burst_waiter: Callable[[float], bool] | None = None,
        stop_requested: Callable[[], bool] | None = None,
        failure_callback: Callable[[BaseException], None] | None = None,
        manual_drain_timeout_s: float = _DEFAULT_MANUAL_REFRESH_DRAIN_TIMEOUT_S,
        rgb_provider: CaptureEventRgbProvider | None = None,
        require_rgb: bool = False,
    ) -> None:
        self._aliases = (
            aliases
            if isinstance(aliases, CanonicalCameraAliases)
            else CanonicalCameraAliases(aliases)
        )
        if not callable(set_depth_gate) or not callable(depth_gate_is_open):
            raise TypeError("depth gate setter and state reader must be callable")
        if burst_waiter is not None and not callable(burst_waiter):
            raise TypeError("burst_waiter must be callable")
        if stop_requested is not None and not callable(stop_requested):
            raise TypeError("stop_requested must be callable")
        if failure_callback is not None and not callable(failure_callback):
            raise TypeError("failure_callback must be callable")
        if type(require_rgb) is not bool:  # noqa: E721
            raise TypeError("require_rgb must be a boolean")
        if require_rgb and rgb_provider is None:
            raise ValueError("require_rgb needs a capture-event RGB provider")
        manual_drain_timeout = float(manual_drain_timeout_s)
        if not math.isfinite(manual_drain_timeout) or not (
            0.1 <= manual_drain_timeout <= 60.0
        ):
            raise ValueError("manual_drain_timeout_s must be in [0.1, 60]")
        self._storage = storage
        self._mapanything = mapanything
        self._set_depth_gate = set_depth_gate
        self._depth_gate_is_open = depth_gate_is_open
        self._burst_waiter = burst_waiter or _default_burst_waiter
        self._stop_requested = stop_requested or (lambda: False)
        self._failure_callback = failure_callback
        self._manual_drain_timeout_s = manual_drain_timeout
        self._rgb_provider = rgb_provider
        self._require_rgb = require_rgb
        self._fusion = CaptureEventFusionCoordinator(
            storage,
            rgb_provider=rgb_provider,
        )
        self._admission = {
            camera: threading.Lock() for camera in self._aliases.canonical_cameras
        }
        # The current graph has one process-wide MapAnything valve.  Keep the
        # per-camera locks for alias ownership, then serialize that shared valve
        # across different cameras as a second non-blocking admission boundary.
        self._gate_admission = threading.Lock()
        self._admission_closed = threading.Event()
        self._manual_condition = threading.Condition()
        self._shutdown_started = False
        self._shutdown_complete = False
        self._manual_refresh_active = False
        self._manual_refresh_phase = "idle"
        self._manual_refresh_thread: threading.Thread | None = None
        self._manual_refresh_deadline = 0.0
        self._manual_refresh_started_at: int | None = None
        self._manual_refresh_will_disable_at: int | None = None
        self._manual_refresh_seconds: int | None = None
        self._manual_refresh_last_error_code: str | None = None
        self._manual_refresh_counters = {
            key: 0 for key in _MANUAL_REFRESH_COUNTER_KEYS
        }
        self._async_failure_notified = False
        self._health_lock = threading.Lock()
        self._counters = {key: 0 for key in _GLOBAL_COUNTER_KEYS}
        self._last_fatal_error_code: str | None = None
        self._camera_health: dict[str, dict[str, Any]] = {
            camera: {
                "active": False,
                "active_request_kind": None,
                "requests_total": 0,
                "completed_total": 0,
                "failed_total": 0,
                "busy_total": 0,
                "last_error_code": None,
                "last_fused_timestamp_us": 0,
                "last_fused_snapshot_id": None,
            }
            for camera in self._aliases.canonical_cameras
        }

    @staticmethod
    def _bounded_increment(value: int) -> int:
        return min(_MAX_COUNTER, max(0, int(value)) + 1)

    def _increment_global(self, key: str) -> None:
        with self._health_lock:
            self._counters[key] = self._bounded_increment(self._counters[key])

    def _increment_manual_locked(self, key: str) -> None:
        self._manual_refresh_counters[key] = self._bounded_increment(
            self._manual_refresh_counters[key]
        )

    def _mark_camera_request(self, camera: str) -> None:
        with self._health_lock:
            state = self._camera_health[camera]
            state["requests_total"] = self._bounded_increment(state["requests_total"])

    def _mark_active(self, camera: str, request_kind: str, active: bool) -> None:
        with self._health_lock:
            state = self._camera_health[camera]
            state["active"] = bool(active)
            state["active_request_kind"] = request_kind if active else None

    def _mark_failure(self, camera: str | None, code: str) -> None:
        self._increment_global("failed_total")
        if capture_event_error_is_fatal(code):
            with self._health_lock:
                self._counters["fatal_barrier_failures_total"] = (
                    self._bounded_increment(
                        self._counters["fatal_barrier_failures_total"]
                    )
                )
                self._last_fatal_error_code = str(code)
        if camera is None:
            return
        with self._health_lock:
            state = self._camera_health[camera]
            state["failed_total"] = self._bounded_increment(state["failed_total"])
            state["last_error_code"] = str(code)

    def _mark_busy(self, camera: str) -> None:
        self._increment_global("busy_total")
        with self._health_lock:
            state = self._camera_health[camera]
            state["busy_total"] = self._bounded_increment(state["busy_total"])
            state["last_error_code"] = "capture_event_busy"

    def _mark_success(self, camera: str, snapshot: FusedDepthSnapshot) -> None:
        self._increment_global("completed_total")
        self._increment_global("fusion_total")
        with self._health_lock:
            state = self._camera_health[camera]
            state["completed_total"] = self._bounded_increment(state["completed_total"])
            state["last_error_code"] = None
            state["last_fused_timestamp_us"] = snapshot.timestamp_us
            state["last_fused_snapshot_id"] = snapshot.snapshot_id
            self._last_fatal_error_code = None

    def canonicalize(self, camera_alias: object) -> str:
        return self._aliases.canonicalize(camera_alias)

    @property
    def manual_drain_timeout_s(self) -> float:
        return self._manual_drain_timeout_s

    @staticmethod
    def _error(code: str, **details: Any) -> CaptureEventControllerError:
        return CaptureEventControllerError(code, details=details)

    def _check_cancelled(self, *, stage: str, camera: str) -> None:
        with self._manual_condition:
            controller_shutdown = (
                self._shutdown_started or self._admission_closed.is_set()
            )
        if controller_shutdown:
            raise self._error(
                "capture_event_cancelled",
                camera_id=camera,
                stage=stage,
                controller_shutdown=True,
            )
        self._check_stop_requested(stage=stage, camera=camera)

    def _check_stop_requested(self, *, stage: str, camera: str) -> None:
        try:
            cancelled = self._stop_requested()
        except Exception as exc:
            raise self._error(
                "capture_event_cancelled",
                camera_id=camera,
                stage=stage,
                stop_check_exception_type=type(exc).__name__,
            ) from exc
        if type(cancelled) is not bool:  # noqa: E721
            raise self._error(
                "capture_event_cancelled",
                camera_id=camera,
                stage=stage,
                stop_check_invalid=True,
            )
        if cancelled:
            raise self._error(
                "capture_event_cancelled",
                camera_id=camera,
                stage=stage,
            )

    def _transition_gate(self, opened: bool, *, stage: str, camera: str) -> None:
        try:
            self._set_depth_gate(opened)
        except Exception as exc:
            raise self._error(
                "depth_gate_transition_failed",
                camera_id=camera,
                stage=stage,
                target_open=opened,
                exception_type=type(exc).__name__,
            ) from exc
        try:
            observed = self._depth_gate_is_open()
        except Exception as exc:
            raise self._error(
                "depth_gate_state_mismatch",
                camera_id=camera,
                stage=stage,
                target_open=opened,
                exception_type=type(exc).__name__,
            ) from exc
        if type(observed) is not bool or observed is not opened:  # noqa: E721
            raise self._error(
                "depth_gate_state_mismatch",
                camera_id=camera,
                stage=stage,
                target_open=opened,
                observed_open=observed if type(observed) is bool else None,  # noqa: E721
            )

    def _wait_idle(
        self,
        *,
        timeout_s: float,
        stage: str,
        camera: str,
    ) -> MapAnythingIdleReceipt:
        try:
            receipt = self._mapanything.wait_idle(timeout_s=timeout_s)
        except Exception as exc:
            raise self._error(
                "mapanything_idle_failed",
                camera_id=camera,
                stage=stage,
                exception_type=type(exc).__name__,
            ) from exc
        if not isinstance(receipt, MapAnythingIdleReceipt):
            raise self._error(
                "mapanything_idle_receipt_invalid",
                camera_id=camera,
                stage=stage,
                observed_type=type(receipt).__name__,
            )
        return receipt

    def _flush(
        self,
        *,
        timeout_s: float,
        stage: str,
        camera: str,
    ) -> DepthStorageFlushEvidence:
        try:
            receipt = self._storage.flush_capture_frontier(timeout_s=timeout_s)
        except Exception as exc:
            raise self._error(
                "depth_storage_flush_failed",
                camera_id=camera,
                stage=stage,
                exception_type=type(exc).__name__,
            ) from exc
        if not isinstance(receipt, DepthStorageFlushEvidence) or not receipt.clean:
            details: dict[str, Any] = {
                "camera_id": camera,
                "stage": stage,
                "receipt_valid": isinstance(receipt, DepthStorageFlushEvidence),
            }
            if isinstance(receipt, DepthStorageFlushEvidence):
                details.update(_flush_evidence(receipt))
            raise self._error("depth_storage_frontier_unclean", **details)
        return receipt

    def _baseline(
        self,
        *,
        camera: str,
        timeout_s: float,
    ) -> tuple[MapAnythingIdleReceipt, DepthStorageFlushEvidence, int]:
        deadline = time.monotonic() + timeout_s
        self._transition_gate(False, stage="baseline_gate_close", camera=camera)
        idle = self._wait_idle(
            timeout_s=max(0.000001, deadline - time.monotonic()),
            stage="baseline_mapanything_idle",
            camera=camera,
        )
        flush = self._flush(
            timeout_s=max(0.000001, deadline - time.monotonic()),
            stage="baseline_storage_flush",
            camera=camera,
        )
        if time.monotonic() >= deadline:
            raise self._error(
                "depth_storage_flush_failed",
                camera_id=camera,
                stage="baseline_deadline_exhausted",
            )
        try:
            timestamp_us = self._storage.max_raw_timestamp(camera)
        except Exception as exc:
            raise self._error(
                "depth_raw_baseline_failed",
                camera_id=camera,
                exception_type=type(exc).__name__,
            ) from exc
        if type(timestamp_us) is not int or timestamp_us < 0:  # noqa: E721
            raise self._error(
                "depth_raw_baseline_failed",
                camera_id=camera,
                invalid_timestamp=True,
            )
        return idle, flush, timestamp_us

    def _postburst_barrier(
        self,
        *,
        camera: str,
        timeout_s: float,
    ) -> tuple[MapAnythingIdleReceipt, DepthStorageFlushEvidence]:
        deadline = time.monotonic() + timeout_s
        first_error: CaptureEventControllerError | None = None
        try:
            self._transition_gate(False, stage="postburst_gate_close", camera=camera)
        except CaptureEventControllerError as exc:
            first_error = exc
        idle: MapAnythingIdleReceipt | None = None
        try:
            idle = self._wait_idle(
                timeout_s=max(0.000001, deadline - time.monotonic()),
                stage="postburst_mapanything_idle",
                camera=camera,
            )
        except CaptureEventControllerError as exc:
            if first_error is None:
                first_error = exc
        flush: DepthStorageFlushEvidence | None = None
        try:
            flush = self._flush(
                timeout_s=max(0.000001, deadline - time.monotonic()),
                stage="postburst_storage_flush",
                camera=camera,
            )
        except CaptureEventControllerError as exc:
            if first_error is None:
                first_error = exc
        if time.monotonic() >= deadline and first_error is None:
            first_error = self._error(
                "depth_storage_flush_failed",
                camera_id=camera,
                stage="postburst_deadline_exhausted",
            )
        if first_error is not None:
            raise first_error
        if idle is None:
            raise self._error(
                "mapanything_idle_receipt_invalid",
                camera_id=camera,
                stage="postburst_mapanything_idle",
                observed_type="NoneType",
            )
        if flush is None:
            raise self._error(
                "depth_storage_frontier_unclean",
                camera_id=camera,
                stage="postburst_storage_flush",
                receipt_valid=False,
            )
        return idle, flush

    @staticmethod
    def _validate_refresh_seconds(seconds: object) -> int:
        if isinstance(seconds, bool):
            raise ValueError("seconds must be an integer in [1, 300]")
        try:
            numeric = float(seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("seconds must be an integer in [1, 300]") from exc
        if (
            not math.isfinite(numeric)
            or not numeric.is_integer()
            or not 1.0 <= numeric <= 300.0
        ):
            raise ValueError("seconds must be an integer in [1, 300]")
        return int(numeric)

    @staticmethod
    def _refresh_payload(
        *,
        started_at: int,
        will_disable_at: int,
        seconds: int,
    ) -> dict[str, Any]:
        return {
            "started_at": int(started_at),
            "will_disable_at": int(will_disable_at),
            "enabled": True,
            "seconds": int(seconds),
        }

    def _finish_manual_refresh(
        self,
        error: CaptureEventControllerError | None,
    ) -> None:
        notify_failure = False
        callback = self._failure_callback
        with self._manual_condition:
            try:
                self._gate_admission.release()
            except RuntimeError as exc:  # pragma: no cover - internal invariant
                if error is None:
                    error = self._error(
                        "depth_gate_state_mismatch",
                        camera_id=_MANUAL_REFRESH_CAMERA,
                        stage="manual_refresh_gate_release",
                        exception_type=type(exc).__name__,
                    )
            fatal_error = bool(
                error is not None and capture_event_error_is_fatal(error.code)
            )
            if fatal_error and error is not None:
                self._latch_fatal_code_locked(error.code)
            self._manual_refresh_active = False
            self._manual_refresh_deadline = 0.0
            self._manual_refresh_started_at = None
            self._manual_refresh_will_disable_at = None
            self._manual_refresh_seconds = None
            if error is None:
                self._increment_manual_locked("completed_total")
                self._manual_refresh_last_error_code = None
            else:
                self._increment_manual_locked("failed_total")
                self._manual_refresh_last_error_code = error.code
                if fatal_error and not self._async_failure_notified:
                    self._async_failure_notified = True
                    notify_failure = True
            self._manual_refresh_phase = (
                "shutdown" if self._shutdown_started else "idle"
            )
            self._manual_condition.notify_all()

        if error is not None:
            self._mark_failure(None, error.code)
        if notify_failure and callback is not None and error is not None:
            try:
                callback(error)
            except BaseException:
                # Runtime failure callbacks are notification boundaries.  The
                # controller has already closed the gate, released admission,
                # and recorded the original fatal barrier error.
                pass

    def _manual_refresh_owner(self) -> None:
        with self._manual_condition:
            while self._manual_refresh_active:
                if self._shutdown_started or self._admission_closed.is_set():
                    break
                remaining = self._manual_refresh_deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._manual_condition.wait(timeout=remaining)
            self._manual_refresh_phase = "draining"

        error: CaptureEventControllerError | None = None
        try:
            self._postburst_barrier(
                camera=_MANUAL_REFRESH_CAMERA,
                timeout_s=self._manual_drain_timeout_s,
            )
        except CaptureEventControllerError as exc:
            error = exc
        self._finish_manual_refresh(error)

    def _notify_synchronous_manual_failure(
        self,
        error: CaptureEventControllerError,
    ) -> None:
        """Notify one fatal startup failure only after gate cleanup returned."""

        callback: Callable[[BaseException], None] | None = None
        with self._manual_condition:
            if not capture_event_error_is_fatal(error.code):
                return
            self._latch_fatal_code_locked(error.code)
            if not self._async_failure_notified:
                self._async_failure_notified = True
                callback = self._failure_callback
        if callback is not None:
            try:
                callback(error)
            except BaseException:
                # Runtime failure callbacks are notification boundaries. The
                # synchronous start path has already closed the gate, drained
                # owned work, flushed storage, and released admission.
                pass

    def _latch_fatal_code_locked(
        self,
        code: str,
    ) -> None:
        """Close admission atomically before exposing a fatal start failure."""

        if not capture_event_error_is_fatal(code):
            return
        self._admission_closed.set()
        if not self._shutdown_started:
            self._shutdown_started = True
            self._increment_manual_locked("shutdown_total")
        self._manual_refresh_phase = "shutdown"
        self._manual_condition.notify_all()

    def start_refresh(self, seconds: object) -> dict[str, Any]:
        """Open or extend the one process-wide asynchronous depth window."""

        try:
            return self._start_refresh(seconds)
        except CaptureEventControllerError as error:
            self._notify_synchronous_manual_failure(error)
            raise

    def _start_refresh(self, seconds: object) -> dict[str, Any]:
        """Implement manual refresh admission beneath the failure boundary."""

        interval = self._validate_refresh_seconds(seconds)
        wall_now = time.time()
        monotonic_now = time.monotonic()
        with self._manual_condition:
            self._increment_manual_locked("requests_total")
            if self._shutdown_started or self._admission_closed.is_set():
                raise self._error(
                    "capture_event_cancelled",
                    stage="manual_refresh_admission",
                    controller_shutdown=True,
                )
            self._check_stop_requested(
                stage="manual_refresh_admission",
                camera=_MANUAL_REFRESH_CAMERA,
            )

            if self._manual_refresh_active:
                if self._manual_refresh_phase != "open":
                    self._increment_manual_locked("busy_total")
                    raise self._error(
                        "capture_event_busy",
                        request_kind="manual_refresh",
                        shared_gate_owned=True,
                        manual_refresh_phase=self._manual_refresh_phase,
                    )
                self._manual_refresh_deadline = max(
                    self._manual_refresh_deadline,
                    monotonic_now + interval,
                )
                will_disable_at = int(
                    wall_now
                    + max(0.0, self._manual_refresh_deadline - monotonic_now)
                )
                self._manual_refresh_will_disable_at = will_disable_at
                self._manual_refresh_seconds = interval
                self._increment_manual_locked("extended_total")
                self._manual_condition.notify_all()
                return self._refresh_payload(
                    started_at=int(wall_now),
                    will_disable_at=will_disable_at,
                    seconds=interval,
                )

            if not self._gate_admission.acquire(blocking=False):
                self._increment_manual_locked("busy_total")
                raise self._error(
                    "capture_event_busy",
                    request_kind="manual_refresh",
                    shared_gate_owned=True,
                )
            if self._admission_closed.is_set():
                self._gate_admission.release()
                raise self._error(
                    "capture_event_cancelled",
                    stage="manual_refresh_gate_admission",
                    controller_shutdown=True,
                )
            try:
                self._check_stop_requested(
                    stage="manual_refresh_gate_admission",
                    camera=_MANUAL_REFRESH_CAMERA,
                )
            except CaptureEventControllerError:
                self._gate_admission.release()
                raise

            started_at = int(wall_now)
            will_disable_at = int(wall_now + interval)
            self._manual_refresh_active = True
            self._manual_refresh_phase = "starting"
            self._manual_refresh_deadline = monotonic_now + interval
            self._manual_refresh_started_at = started_at
            self._manual_refresh_will_disable_at = will_disable_at
            self._manual_refresh_seconds = interval
            self._manual_refresh_last_error_code = None
            self._increment_manual_locked("admitted_total")

            try:
                self._transition_gate(
                    True,
                    stage="manual_refresh_gate_open",
                    camera=_MANUAL_REFRESH_CAMERA,
                )
                owner = threading.Thread(
                    target=self._manual_refresh_owner,
                    name="capture-event-manual-refresh",
                    daemon=False,
                )
                self._manual_refresh_thread = owner
                self._manual_refresh_phase = "open"
                owner.start()
            except CaptureEventControllerError as exc:
                failure = exc
                try:
                    self._postburst_barrier(
                        camera=_MANUAL_REFRESH_CAMERA,
                        timeout_s=self._manual_drain_timeout_s,
                    )
                except CaptureEventControllerError as barrier_exc:
                    failure = barrier_exc
                self._manual_refresh_active = False
                self._manual_refresh_thread = None
                self._manual_refresh_phase = "idle"
                self._manual_refresh_deadline = 0.0
                self._manual_refresh_started_at = None
                self._manual_refresh_will_disable_at = None
                self._manual_refresh_seconds = None
                self._manual_refresh_last_error_code = failure.code
                self._increment_manual_locked("failed_total")
                self._gate_admission.release()
                self._mark_failure(None, failure.code)
                self._latch_fatal_code_locked(failure.code)
                raise failure
            except Exception as exc:  # pragma: no cover - thread start failure
                failure = self._error(
                    "capture_burst_wait_failed",
                    camera_id=_MANUAL_REFRESH_CAMERA,
                    stage="manual_refresh_thread_start",
                    exception_type=type(exc).__name__,
                )
                try:
                    self._postburst_barrier(
                        camera=_MANUAL_REFRESH_CAMERA,
                        timeout_s=self._manual_drain_timeout_s,
                    )
                except CaptureEventControllerError as barrier_exc:
                    failure = barrier_exc
                finally:
                    self._manual_refresh_active = False
                    self._manual_refresh_thread = None
                    self._manual_refresh_phase = "idle"
                    self._manual_refresh_deadline = 0.0
                    self._manual_refresh_started_at = None
                    self._manual_refresh_will_disable_at = None
                    self._manual_refresh_seconds = None
                    self._manual_refresh_last_error_code = failure.code
                    self._increment_manual_locked("failed_total")
                    self._gate_admission.release()
                    self._mark_failure(None, failure.code)
                    self._latch_fatal_code_locked(failure.code)
                raise failure from exc

            return self._refresh_payload(
                started_at=started_at,
                will_disable_at=will_disable_at,
                seconds=interval,
            )

    def shutdown(self, timeout_s: float = 10.0) -> None:
        """Close admission and join the manual owner before returning."""

        timeout = float(timeout_s)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout_s must be finite and positive")
        deadline = time.monotonic() + timeout

        self._admission_closed.set()

        with self._manual_condition:
            if self._shutdown_complete:
                return
            if not self._shutdown_started:
                self._shutdown_started = True
                self._increment_manual_locked("shutdown_total")
            self._manual_condition.notify_all()
            owner = self._manual_refresh_thread

        if owner is not None and owner is not threading.current_thread():
            owner.join(timeout=max(0.0, deadline - time.monotonic()))
            if owner.is_alive():
                raise self._error(
                    "capture_event_busy",
                    stage="manual_refresh_shutdown_join",
                    thread_alive=True,
                )

        remaining = max(0.0, deadline - time.monotonic())
        if not self._gate_admission.acquire(timeout=remaining):
            raise self._error(
                "capture_event_busy",
                stage="capture_event_shutdown_admission",
                shared_gate_owned=True,
            )
        try:
            self._postburst_barrier(
                camera=_MANUAL_REFRESH_CAMERA,
                timeout_s=min(
                    self._manual_drain_timeout_s,
                    max(0.000001, deadline - time.monotonic()),
                ),
            )
        finally:
            self._gate_admission.release()

        with self._manual_condition:
            owner = self._manual_refresh_thread
            if owner is not None and owner.is_alive():
                raise self._error(
                    "capture_event_busy",
                    stage="manual_refresh_shutdown_verify",
                    thread_alive=True,
                )
            self._manual_refresh_active = False
            self._manual_refresh_phase = "shutdown"
            self._shutdown_complete = True
            self._manual_condition.notify_all()

    def close_admission(self) -> None:
        """Reject new capture ownership immediately without waiting for drain."""

        self._admission_closed.set()
        if self._manual_condition.acquire(blocking=False):
            try:
                self._manual_condition.notify_all()
            finally:
                self._manual_condition.release()

    def capture(self, request: CaptureEventRequest) -> CaptureEventOutcome:
        if not isinstance(request, CaptureEventRequest):
            raise TypeError("request must be CaptureEventRequest")
        self._increment_global("requests_total")
        if request.cache_only:
            self._increment_global("cache_only_rejected_total")
            self._mark_failure(None, "cache_only_forbids_capture_event")
            raise self._error(
                "cache_only_forbids_capture_event",
                request_kind=request.request_kind,
            )

        try:
            camera = self.canonicalize(request.camera_alias)
        except CaptureEventControllerError as exc:
            self._mark_failure(None, exc.code)
            raise
        self._mark_camera_request(camera)
        admission = self._admission[camera]
        if not admission.acquire(blocking=False):
            self._mark_busy(camera)
            raise self._error(
                "capture_event_busy",
                camera_id=camera,
                request_kind=request.request_kind,
            )
        with self._manual_condition:
            if self._shutdown_started or self._admission_closed.is_set():
                admission.release()
                self._mark_failure(camera, "capture_event_cancelled")
                raise self._error(
                    "capture_event_cancelled",
                    camera_id=camera,
                    stage="capture_event_admission",
                    controller_shutdown=True,
                )
            gate_acquired = self._gate_admission.acquire(blocking=False)
        if not gate_acquired:
            admission.release()
            self._mark_busy(camera)
            raise self._error(
                "capture_event_busy",
                camera_id=camera,
                request_kind=request.request_kind,
                shared_gate_owned=True,
            )

        self._increment_global("admitted_total")
        self._mark_active(camera, request.request_kind, True)
        rgb_arm: object | None = None
        operation_error: CaptureEventFusionError | None = None
        try:
            self._check_cancelled(stage="before_baseline", camera=camera)
            baseline_idle, baseline_flush, baseline_ts = self._baseline(
                camera=camera,
                timeout_s=request.drain_timeout_s,
            )
            self._check_cancelled(stage="before_gate_open", camera=camera)
            if self._require_rgb:
                try:
                    rgb_arm = self._rgb_provider.arm(camera)  # type: ignore[union-attr]
                except Exception as exc:
                    raise CaptureEventFusionError(
                        "rgb_provider_failed",
                        details={
                            "camera_id": camera,
                            "stage": "arm",
                            "exception_type": type(exc).__name__,
                        },
                    ) from exc

            primary_error: CaptureEventControllerError | None = None
            try:
                self._transition_gate(True, stage="burst_gate_open", camera=camera)
                try:
                    interrupted = self._burst_waiter(request.burst_seconds)
                except Exception as exc:
                    raise self._error(
                        "capture_burst_wait_failed",
                        camera_id=camera,
                        exception_type=type(exc).__name__,
                    ) from exc
                if type(interrupted) is not bool:  # noqa: E721
                    raise self._error(
                        "capture_burst_wait_failed",
                        camera_id=camera,
                        invalid_result=True,
                    )
                if interrupted:
                    raise self._error(
                        "capture_event_cancelled",
                        camera_id=camera,
                        stage="burst_wait",
                    )
                self._check_cancelled(stage="after_burst_wait", camera=camera)
            except CaptureEventControllerError as exc:
                primary_error = exc

            post_idle, post_flush = self._postburst_barrier(
                camera=camera,
                timeout_s=request.drain_timeout_s,
            )
            if primary_error is not None:
                raise primary_error

            fusion = self._fusion.fuse(
                CaptureEventFusionRequest(
                    camera_id=camera,
                    storage_keys=(camera,),
                    baseline_timestamp_us={camera: baseline_ts},
                    cache_only=False,
                    require_rgb=self._require_rgb,
                    raw_limit=request.raw_limit,
                    min_observations=request.min_observations,
                    depth_agreement_m=request.depth_agreement_m,
                    max_cohort_span_us=request.max_cohort_span_us,
                    max_rgb_skew_us=0,
                )
            )
            try:
                validated = self._storage.validate_fused_snapshot(fusion.fused_snapshot)
            except Exception as exc:
                raise self._error(
                    "fused_snapshot_validation_failed",
                    camera_id=camera,
                    exception_type=type(exc).__name__,
                ) from exc
            if validated != fusion.fused_snapshot:
                raise self._error(
                    "fused_snapshot_validation_failed",
                    camera_id=camera,
                    substituted_descriptor=True,
                )

            compact = {
                "contract": "noesis.capture_event_controller",
                "contract_version": 1,
                "camera_id": camera,
                "request_kind": request.request_kind,
                "capture_mode": (
                    "depth_rgb_exact" if self._require_rgb else "depth_only"
                ),
                "baseline_raw_timestamp_us": baseline_ts,
                "baseline_mapanything_idle": _idle_evidence(baseline_idle),
                "baseline_storage_flush": _flush_evidence(baseline_flush),
                "postburst_mapanything_idle": _idle_evidence(post_idle),
                "postburst_storage_flush": _flush_evidence(post_flush),
                "parameters": {
                    "burst_seconds": request.burst_seconds,
                    "raw_limit": request.raw_limit,
                    "min_observations": request.min_observations,
                    "depth_agreement_m": request.depth_agreement_m,
                    "max_cohort_span_us": request.max_cohort_span_us,
                },
                "fusion_evidence_sha256": fusion.evidence_sha256,
                "raw_snapshot_count": fusion.evidence["raw_snapshot_count"],
                "rgb": fusion.evidence["rgb"],
                "fusion_quality": thaw_json_evidence(
                    validated.quality_evidence
                ),
                "fused_snapshot": _fused_evidence(validated),
            }
            compact_sha256 = canonical_json_sha256(compact)
            outcome = CaptureEventOutcome(
                canonical_camera=camera,
                request_kind=request.request_kind,
                fused_snapshot=validated,
                fusion_evidence_sha256=fusion.evidence_sha256,
                compact_evidence=compact,
                compact_evidence_sha256=compact_sha256,
            )
            self._mark_success(camera, validated)
            return outcome
        except CaptureEventFusionError as exc:
            operation_error = exc
            self._mark_failure(camera, exc.code)
            if capture_event_error_is_fatal(exc.code):
                with self._manual_condition:
                    self._latch_fatal_code_locked(exc.code)
            raise
        finally:
            disarm_error: CaptureEventFusionError | None = None
            if rgb_arm is not None:
                try:
                    self._rgb_provider.disarm(rgb_arm)  # type: ignore[union-attr]
                except Exception as exc:
                    disarm_error = CaptureEventFusionError(
                        "rgb_provider_failed",
                        details={
                            "camera_id": camera,
                            "stage": "disarm",
                            "exception_type": type(exc).__name__,
                        },
                    )
            self._mark_active(camera, request.request_kind, False)
            self._gate_admission.release()
            admission.release()
            if disarm_error is not None:
                if operation_error is None:
                    self._mark_failure(camera, disarm_error.code)
                    with self._manual_condition:
                        self._latch_fatal_code_locked(disarm_error.code)
                    raise disarm_error
                try:
                    if self._failure_callback is not None:
                        self._failure_callback(disarm_error)
                except Exception:
                    pass

    def health_snapshot(self) -> dict[str, Any]:
        """Return a fixed-schema copy; no request history or dynamic keys grow."""

        with self._health_lock:
            counters = dict(self._counters)
            cameras = {
                camera: dict(state) for camera, state in self._camera_health.items()
            }
            last_fatal_error_code = self._last_fatal_error_code
            camera_active = any(bool(state["active"]) for state in cameras.values())
        shared_gate_owned = self._gate_admission.locked()
        with self._manual_condition:
            manual_thread = self._manual_refresh_thread
            manual_refresh = {
                "active": self._manual_refresh_active,
                "phase": self._manual_refresh_phase,
                "admission_open": not self._admission_closed.is_set(),
                "thread_alive": bool(
                    manual_thread is not None and manual_thread.is_alive()
                ),
                "started_at": self._manual_refresh_started_at,
                "will_disable_at": self._manual_refresh_will_disable_at,
                "seconds": self._manual_refresh_seconds,
                "last_error_code": self._manual_refresh_last_error_code,
                "counters": dict(self._manual_refresh_counters),
            }
        return {
            "contract": "noesis.capture_event_controller_health",
            "contract_version": 1,
            "canonical_camera_count": len(self._aliases.canonical_cameras),
            "alias_count": self._aliases.alias_count,
            "request_kinds": sorted(_REQUEST_KINDS),
            "shared_gate_scope": "process",
            "shared_gate_owned": shared_gate_owned,
            "healthy": bool(
                not camera_active
                and not shared_gate_owned
                and not self._admission_closed.is_set()
                and last_fatal_error_code is None
            ),
            "last_fatal_error_code": last_fatal_error_code,
            "counters": counters,
            "manual_refresh": manual_refresh,
            "cameras": cameras,
        }


__all__ = [
    "CanonicalCameraAliases",
    "CaptureEventController",
    "CaptureEventControllerError",
    "CaptureEventControllerStorage",
    "CaptureEventOutcome",
    "CaptureEventRequest",
    "MapAnythingIdleBarrier",
    "capture_event_error_is_fatal",
]
