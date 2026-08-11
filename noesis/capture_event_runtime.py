"""Shared DS8/V3DT/DS9 RPC adapter for exact capture-event artifacts."""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.parse import parse_qs, unquote, urlsplit

from geometry.depth_source import resolve_depth_store_commit_timeout_s
from noesis.capture_event_controller import (
    CanonicalCameraAliases,
    CaptureEventController,
    CaptureEventOutcome,
    CaptureEventRequest,
    capture_event_error_is_fatal,
)
from noesis_core.capture_event_fusion import (
    CaptureEventFusionError,
    FusedDepthSnapshot,
    canonical_json_sha256,
)
from noesis_core.depth_bulk import (
    DEPTH_BULK_MAX_COMPONENT_BYTES,
    DEPTH_BULK_MAX_PIXELS,
    DEPTH_BULK_MAX_SNAPSHOT_BYTES,
)

_DEPTH_ARTIFACT_PREFIX = "depth-zarr:"
_JAVASCRIPT_MAX_SAFE_INTEGER = (1 << 53) - 1
_FLOORPLAN_INTEGRITY_ERRORS = frozenset(
    {"load_failed", "invalid_snapshot", "shape_mismatch", "snapshot_identity_mismatch"}
)
_RUNTIME_FATAL_CODES = frozenset(
    {
        "capture_event_controller_unavailable",
        "capture_event_configuration_invalid",
        "depth_cache_read_failed",
        "invalid_cached_depth",
        "exact_depth_load_failed",
        "exact_depth_identity_mismatch",
        "exact_depth_invalid",
        "floorplan_generation_failed",
        "floorplan_snapshot_integrity_failed",
        "active_floorplan_contract_failed",
    }
)


class CaptureEventDepthStorage(Protocol):
    def describe_latest_depth_bulk(
        self,
        camera_id: str,
        ts_max_us: int | None = None,
    ) -> dict[str, Any] | None: ...

    def describe_depth_snapshot_bulk_exact(
        self,
        *,
        camera_id: str,
        storage_ref: str,
        snapshot_id: str,
        content_sha256: str,
    ) -> dict[str, Any]: ...

    def generate_topdown_floorplan(
        self,
        camera_id: str,
        max_age_sec: float = 60.0,
        grid_res_m: float = 0.15,
        max_extent_m: float = 20.0,
        cache_only: bool = False,
        *,
        snapshot_ref: str | None = None,
        snapshot_id: str | None = None,
        snapshot_content_sha256: str | None = None,
    ) -> dict[str, Any]: ...


class CaptureEventRuntimeError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = str(code)
        super().__init__(self.code)


def capture_event_runtime_error_is_fatal(code: object) -> bool:
    text = str(code or "").strip()
    return text in _RUNTIME_FATAL_CODES or capture_event_error_is_fatal(text)


def _read_float(
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    raw = os.environ.get(name)
    if raw is None:
        value = float(default)
    else:
        try:
            value = float(raw.strip())
        except Exception as exc:
            raise CaptureEventRuntimeError(
                "capture_event_configuration_invalid"
            ) from exc
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise CaptureEventRuntimeError("capture_event_configuration_invalid")
    return value


def _read_int(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = os.environ.get(name)
    if raw is None:
        value = int(default)
    else:
        try:
            value = int(raw.strip())
        except Exception as exc:
            raise CaptureEventRuntimeError(
                "capture_event_configuration_invalid"
            ) from exc
    if not minimum <= value <= maximum:
        raise CaptureEventRuntimeError("capture_event_configuration_invalid")
    return value


def _burst_seconds(request_kind: str) -> float:
    if (
        request_kind == "floorplan"
        and "NOESIS_FLOORPLAN_DEPTH_ENABLE_SECONDS" in os.environ
    ):
        return _read_float(
            "NOESIS_FLOORPLAN_DEPTH_ENABLE_SECONDS",
            20.0,
            minimum=0.01,
            maximum=20.0,
        )
    return _read_float(
        "NOESIS_DEPTH_RPC_ENABLE_SECONDS",
        20.0,
        minimum=0.01,
        maximum=20.0,
    )


def resolve_capture_event_drain_timeout_s(
    *,
    burst_seconds: float = 0.0,
) -> float:
    burst = float(burst_seconds)
    if not math.isfinite(burst) or burst < 0.0:
        raise CaptureEventRuntimeError("capture_event_configuration_invalid")
    try:
        commit_timeout = resolve_depth_store_commit_timeout_s()
    except (TypeError, ValueError) as exc:
        raise CaptureEventRuntimeError(
            "capture_event_configuration_invalid"
        ) from exc
    minimum_safe_timeout = min(
        60.0,
        max(35.0, burst + 5.0, commit_timeout + 5.0),
    )
    configured = _read_float(
        "NOESIS_CAPTURE_EVENT_DRAIN_TIMEOUT_SECONDS",
        minimum_safe_timeout,
        minimum=0.1,
        maximum=60.0,
    )
    if configured < minimum_safe_timeout:
        raise CaptureEventRuntimeError("capture_event_configuration_invalid")
    return configured


def build_capture_event_request(
    camera_alias: str,
    *,
    request_kind: str,
    burst_seconds: float | None = None,
) -> CaptureEventRequest:
    burst = (
        _burst_seconds(request_kind) if burst_seconds is None else float(burst_seconds)
    )
    raw_limit = _read_int(
        "NOESIS_CAPTURE_EVENT_RAW_LIMIT",
        24,
        minimum=1,
        maximum=256,
    )
    min_observations = _read_int(
        "NOESIS_CAPTURE_EVENT_MIN_OBSERVATIONS",
        3,
        minimum=1,
        maximum=256,
    )
    if min_observations > raw_limit:
        raise CaptureEventRuntimeError("capture_event_configuration_invalid")
    drain_timeout = resolve_capture_event_drain_timeout_s(
        burst_seconds=burst,
    )
    agreement = _read_float(
        "NOESIS_CAPTURE_EVENT_DEPTH_AGREEMENT_M",
        0.18,
        minimum=0.000001,
        maximum=10.0,
    )
    cohort_span_s = _read_float(
        "NOESIS_CAPTURE_EVENT_MAX_COHORT_SPAN_SECONDS",
        max(20.0, burst + 5.0),
        minimum=0.001,
        maximum=60.0,
    )
    try:
        return CaptureEventRequest(
            camera_alias=str(camera_alias),
            request_kind=request_kind,
            cache_only=False,
            burst_seconds=burst,
            drain_timeout_s=drain_timeout,
            raw_limit=raw_limit,
            min_observations=min_observations,
            depth_agreement_m=agreement,
            max_cohort_span_us=int(cohort_span_s * 1_000_000),
        )
    except (TypeError, ValueError) as exc:
        raise CaptureEventRuntimeError("capture_event_configuration_invalid") from exc


def _normalize_ts_max_us(value: object | None) -> int | None:
    if value is None:
        return None
    try:
        if type(value) is bool:  # noqa: E721
            raise ValueError("boolean timestamp")
        if isinstance(value, str):
            text = value.strip()
            if not text or not text.isdigit():
                raise ValueError("timestamp must be an unsigned decimal integer")
            raw = int(text, 10)
        elif type(value) is int:  # noqa: E721
            raw = value
        else:
            raise ValueError("timestamp must be an integer")
    except Exception as exc:
        raise CaptureEventRuntimeError("invalid_ts_max_us") from exc
    if raw < 0:
        raise CaptureEventRuntimeError("invalid_ts_max_us")
    if raw < 1_000_000_000:
        return raw * 1_000_000
    if raw < 1_000_000_000_000:
        return raw * 1_000
    return raw


def _contains_encoded_tensor_key(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).endswith("_b64") or _contains_encoded_tensor_key(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_encoded_tensor_key(item) for item in value)
    return False


def _browser_safe_json_domain(value: object) -> bool:
    if value is None or type(value) in {bool, str}:  # noqa: E721
        return True
    if type(value) is int:  # noqa: E721
        return abs(value) <= _JAVASCRIPT_MAX_SAFE_INTEGER
    if type(value) is float:  # noqa: E721
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(
            type(key) is str  # noqa: E721
            and bool(key)
            and _browser_safe_json_domain(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return all(_browser_safe_json_domain(item) for item in value)
    return False


def validate_depth_bulk_snapshot_descriptor(
    payload: Mapping[str, Any],
    *,
    camera_id: str | None = None,
) -> bool:
    """Strictly validate the compact dense-depth wire descriptor."""

    if (
        payload.get("contract") != "noesis.depth.bulk_snapshot"
        or type(payload.get("contract_version")) is not int  # noqa: E721
        or payload.get("contract_version") != 1
    ):
        return False
    required_keys = {
        "contract",
        "contract_version",
        "ts",
        "shape",
        "snapshot_id",
        "snapshot_ref",
        "content_sha256",
        "role",
        "fusion_level",
        "components",
        "normals",
    }
    allowed_keys = required_keys | {
        "capture_event",
        "capture_event_evidence_sha256",
    }
    if (
        not required_keys.issubset(payload)
        or not set(payload).issubset(allowed_keys)
        or _contains_encoded_tensor_key(payload)
    ):
        return False
    shape = payload.get("shape")
    if not (isinstance(shape, list) and len(shape) == 2):
        return False
    if any(
        type(dim) is not int  # noqa: E721
        or dim <= 0
        or dim > _JAVASCRIPT_MAX_SAFE_INTEGER
        for dim in shape
    ):
        return False
    height, width = shape
    if height > _JAVASCRIPT_MAX_SAFE_INTEGER // width:
        return False
    if height * width > DEPTH_BULK_MAX_PIXELS:
        return False

    snapshot_id = payload.get("snapshot_id")
    snapshot_ref = payload.get("snapshot_ref")
    content_sha256 = payload.get("content_sha256")
    timestamp_us = payload.get("ts")
    if (
        type(timestamp_us) is not int  # noqa: E721
        or not 0 < timestamp_us <= _JAVASCRIPT_MAX_SAFE_INTEGER
        or type(snapshot_id) is not str  # noqa: E721
        or not snapshot_id
        or len(snapshot_id) > 160
        or type(snapshot_ref) is not str  # noqa: E721
        or not snapshot_ref
        or len(snapshot_ref) > 512
        or snapshot_ref.startswith("/")
        or "\\" in snapshot_ref
        or any(part in {"", ".", ".."} for part in snapshot_ref.split("/"))
        or type(content_sha256) is not str  # noqa: E721
        or len(content_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in content_sha256)
        or payload.get("role") != "capture_event_fused"
        or payload.get("fusion_level") != "intra_capture"
    ):
        return False

    components = payload.get("components")
    if not isinstance(components, Mapping):
        return False
    component_names = set(components)
    if not {"depth", "conf", "mask"}.issubset(component_names):
        return False
    if not component_names.issubset({"depth", "conf", "mask", "rgb"}):
        return False
    transfer_bytes = 0
    for component in ("depth", "conf", "mask", "rgb"):
        if component not in components:
            continue
        row = components.get(component)
        if not isinstance(row, Mapping) or row.get("component") != component:
            return False
        expected_dtype = "<f4" if component in {"depth", "conf"} else "|u1"
        expected_shape = (
            [height, width, 3]
            if component == "rgb"
            else [height, width]
        )
        item_size = 4 if expected_dtype == "<f4" else 1
        expected_bytes = (
            height
            * width
            * (3 if component == "rgb" else 1)
            * item_size
        )
        digest = row.get("sha256")
        url = row.get("url")
        row_shape = row.get("shape")
        byte_count = row.get("byte_count")
        if (
            not isinstance(row_shape, list)
            or len(row_shape) != len(expected_shape)
            or any(
                type(dim) is not int  # noqa: E721
                or dim <= 0
                or dim > _JAVASCRIPT_MAX_SAFE_INTEGER
                for dim in row_shape
            )
            or type(byte_count) is not int  # noqa: E721
        ):
            return False
        if (
            set(row)
            != {"component", "dtype", "shape", "byte_count", "sha256", "url"}
            or row.get("dtype") != expected_dtype
            or row_shape != expected_shape
            or byte_count != expected_bytes
            or byte_count <= 0
            or byte_count > _JAVASCRIPT_MAX_SAFE_INTEGER
            or byte_count > DEPTH_BULK_MAX_COMPONENT_BYTES
            or type(digest) is not str  # noqa: E721
            or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)
            or type(url) is not str  # noqa: E721
            or not url.startswith("/api/v1/depth/snapshots/")
            or url.startswith("//")
            or "://" in url
            or len(url) > 2048
            or "\\" in url
            or any(ord(ch) < 32 or ord(ch) == 127 for ch in url)
        ):
            return False
        try:
            parsed = urlsplit(url)
            query = parse_qs(
                parsed.query,
                keep_blank_values=True,
                strict_parsing=True,
            )
        except Exception:
            return False
        path_parts = unquote(parsed.path).split("/")
        if (
            len(path_parts) != 9
            or path_parts[:5] != ["", "api", "v1", "depth", "snapshots"]
            or not path_parts[5]
            or (camera_id is not None and path_parts[5] != str(camera_id))
            or path_parts[6] != snapshot_id
            or path_parts[7:] != ["components", component]
            or parsed.scheme
            or parsed.netloc
            or parsed.fragment
            or query
            != {
                "snapshot_ref": [snapshot_ref],
                "content_sha256": [content_sha256],
            }
        ):
            return False
        transfer_bytes += byte_count
    if (
        transfer_bytes > _JAVASCRIPT_MAX_SAFE_INTEGER
        or transfer_bytes > DEPTH_BULK_MAX_SNAPSHOT_BYTES
    ):
        return False
    if payload.get("normals") != {
        "mode": "client_derived_depth_gradient_v1",
        "space": "camera",
        "dtype": "float32",
    }:
        return False

    has_capture_event = "capture_event" in payload
    has_capture_digest = "capture_event_evidence_sha256" in payload
    if has_capture_event != has_capture_digest:
        return False
    if has_capture_event:
        evidence = payload.get("capture_event")
        evidence_digest = payload.get("capture_event_evidence_sha256")
        if (
            not isinstance(evidence, Mapping)
            or not _browser_safe_json_domain(evidence)
            or type(evidence_digest) is not str  # noqa: E721
            or len(evidence_digest) != 64
            or any(ch not in "0123456789abcdef" for ch in evidence_digest)
        ):
            return False
        try:
            if canonical_json_sha256(evidence) != evidence_digest:
                return False
        except (TypeError, ValueError):
            return False
    return True


def validate_depth_bulk_component_bytes(
    descriptor: Mapping[str, Any],
    payload: bytes | bytearray | memoryview,
) -> bool:
    """Validate one fetched component body against its descriptor row."""

    try:
        body = bytes(payload)
        byte_count = descriptor.get("byte_count")
        digest = descriptor.get("sha256")
    except Exception:
        return False
    return bool(
        type(byte_count) is int  # noqa: E721
        and 0 < byte_count <= _JAVASCRIPT_MAX_SAFE_INTEGER
        and len(body) == byte_count
        and type(digest) is str  # noqa: E721
        and len(digest) == 64
        and hashlib.sha256(body).hexdigest() == digest
    )


def _storage_ref(snapshot: FusedDepthSnapshot) -> str:
    artifact_ref = str(snapshot.artifact_ref)
    if not artifact_ref.startswith(_DEPTH_ARTIFACT_PREFIX):
        raise CaptureEventRuntimeError("exact_depth_identity_mismatch")
    storage_ref = artifact_ref[len(_DEPTH_ARTIFACT_PREFIX) :]
    if not storage_ref or storage_ref.startswith("/") or ".." in storage_ref.split("/"):
        raise CaptureEventRuntimeError("exact_depth_identity_mismatch")
    return storage_ref


def _capture_event_evidence(
    outcome: CaptureEventOutcome,
) -> tuple[dict[str, Any], str]:
    evidence = outcome.compact_evidence_payload()
    try:
        recomputed = canonical_json_sha256(evidence)
    except (TypeError, ValueError) as exc:
        raise CaptureEventRuntimeError("exact_depth_invalid") from exc
    if recomputed != outcome.compact_evidence_sha256:
        raise CaptureEventRuntimeError("exact_depth_invalid")
    return evidence, recomputed


@dataclass
class CaptureEventRuntimeProviders:
    aliases: CanonicalCameraAliases
    storage: CaptureEventDepthStorage
    controller_getter: Callable[[], CaptureEventController | None]
    depth_branch_available: Callable[[], bool]
    shutdown_requested: Callable[[], bool]
    runtime_failure: Callable[[str, BaseException], None]
    record_floorplan: Callable[[str, Mapping[str, Any]], bool]

    def _escalate(self, error: BaseException, code: str) -> None:
        if capture_event_runtime_error_is_fatal(code):
            self.runtime_failure("capture_event", error)

    def _controller(self) -> CaptureEventController:
        controller = self.controller_getter()
        if not isinstance(controller, CaptureEventController):
            raise CaptureEventRuntimeError("capture_event_controller_unavailable")
        return controller

    def _canonicalize(self, camera_alias: object) -> str:
        return self.aliases.canonicalize(camera_alias)

    @staticmethod
    def _depth_response(
        *,
        camera: str,
        cache_only: bool,
        served_from_cache: bool,
        request_id: str | None,
        payload: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        response: dict[str, Any] = {
            "type": "ma_depth_response",
            "camera": camera,
            "cache_only": bool(cache_only),
            "served_from_cache": bool(served_from_cache),
            "ts_us": int((payload or {}).get("ts", 0) or 0),
            "ok": error is None,
        }
        if request_id:
            response["request_id"] = request_id
        if payload is not None:
            response["payload"] = dict(payload)
        if error is not None:
            response["error"] = str(error)
        return response

    def _load_cached_depth(
        self,
        camera: str,
        ts_max_us: int | None,
    ) -> dict[str, Any] | None:
        try:
            payload = self.storage.describe_latest_depth_bulk(
                camera,
                ts_max_us,
            )
        except Exception as exc:
            if (
                str(getattr(exc, "code", "") or "")
                == "bulk_component_manifest_missing"
            ):
                raise CaptureEventRuntimeError(
                    "bulk_component_manifest_missing"
                ) from exc
            raise CaptureEventRuntimeError("depth_cache_read_failed") from exc
        if payload is None:
            return None
        if (
            not isinstance(payload, dict)
            or not validate_depth_bulk_snapshot_descriptor(
                payload,
                camera_id=camera,
            )
        ):
            raise CaptureEventRuntimeError("invalid_cached_depth")
        return dict(payload)

    def _load_exact_depth(
        self,
        outcome: CaptureEventOutcome,
    ) -> dict[str, Any]:
        snapshot = outcome.fused_snapshot
        storage_ref = _storage_ref(snapshot)
        try:
            payload = self.storage.describe_depth_snapshot_bulk_exact(
                camera_id=snapshot.camera_id,
                storage_ref=storage_ref,
                snapshot_id=snapshot.snapshot_id,
                content_sha256=snapshot.content_sha256,
            )
        except Exception as exc:
            if (
                str(getattr(exc, "code", "") or "")
                == "bulk_component_manifest_missing"
            ):
                raise CaptureEventRuntimeError(
                    "bulk_component_manifest_missing"
                ) from exc
            raise CaptureEventRuntimeError("exact_depth_load_failed") from exc
        if not isinstance(payload, dict):
            raise CaptureEventRuntimeError("exact_depth_load_failed")
        try:
            payload_timestamp_us = int(payload.get("ts", 0) or 0)
        except Exception:
            payload_timestamp_us = -1
        if (
            str(payload.get("snapshot_ref") or "") != storage_ref
            or str(payload.get("snapshot_id") or "") != snapshot.snapshot_id
            or str(payload.get("content_sha256") or "") != snapshot.content_sha256
            or payload_timestamp_us != snapshot.timestamp_us
            or str(payload.get("role") or "") != snapshot.snapshot_role
            or str(payload.get("fusion_level") or "") != snapshot.fusion_level
        ):
            raise CaptureEventRuntimeError("exact_depth_identity_mismatch")
        if not validate_depth_bulk_snapshot_descriptor(
            payload,
            camera_id=snapshot.camera_id,
        ):
            raise CaptureEventRuntimeError("exact_depth_invalid")
        evidence, evidence_sha256 = _capture_event_evidence(outcome)
        payload["capture_event"] = evidence
        payload["capture_event_evidence_sha256"] = evidence_sha256
        if not validate_depth_bulk_snapshot_descriptor(
            payload,
            camera_id=snapshot.camera_id,
        ):
            raise CaptureEventRuntimeError("exact_depth_invalid")
        return payload

    def depth_provider(
        self,
        cam_id: str,
        ts_max_us: object | None = None,
        request_id: str | None = None,
        cache_only: bool = False,
        **_ignored: object,
    ) -> dict[str, Any]:
        request_camera = str(cam_id or "").strip()
        try:
            camera = self._canonicalize(request_camera)
        except CaptureEventFusionError as exc:
            return self._depth_response(
                camera=request_camera,
                cache_only=cache_only,
                served_from_cache=False,
                request_id=request_id,
                error=exc.code,
            )
        try:
            cutoff = _normalize_ts_max_us(ts_max_us)
        except CaptureEventRuntimeError as exc:
            return self._depth_response(
                camera=camera,
                cache_only=cache_only,
                served_from_cache=False,
                request_id=request_id,
                error=exc.code,
            )
        if self.shutdown_requested():
            return self._depth_response(
                camera=camera,
                cache_only=cache_only,
                served_from_cache=False,
                request_id=request_id,
                error="shutting_down",
            )

        cached: dict[str, Any] | None = None
        if cutoff is not None or cache_only:
            try:
                cached = self._load_cached_depth(camera, cutoff)
            except CaptureEventRuntimeError as exc:
                self._escalate(exc, exc.code)
                return self._depth_response(
                    camera=camera,
                    cache_only=cache_only,
                    served_from_cache=False,
                    request_id=request_id,
                    error=exc.code,
                )
        if cached is not None:
            return self._depth_response(
                camera=camera,
                cache_only=cache_only,
                served_from_cache=True,
                request_id=request_id,
                payload=cached,
            )
        if cache_only:
            return self._depth_response(
                camera=camera,
                cache_only=True,
                served_from_cache=False,
                request_id=request_id,
                error="no_cached_depth",
            )
        if not self.depth_branch_available():
            return self._depth_response(
                camera=camera,
                cache_only=False,
                served_from_cache=False,
                request_id=request_id,
                error="depth_branch_unavailable",
            )
        try:
            controller = self._controller()
            outcome = controller.capture(
                build_capture_event_request(camera, request_kind="depth")
            )
            payload = self._load_exact_depth(outcome)
        except (CaptureEventFusionError, CaptureEventRuntimeError) as exc:
            code = exc.code
            self._escalate(exc, code)
            return self._depth_response(
                camera=camera,
                cache_only=False,
                served_from_cache=False,
                request_id=request_id,
                error=code,
            )
        return self._depth_response(
            camera=camera,
            cache_only=False,
            served_from_cache=False,
            request_id=request_id,
            payload=payload,
        )

    def _record_floorplan(
        self,
        camera: str,
        payload: Mapping[str, Any],
    ) -> None:
        try:
            recorded = self.record_floorplan(camera, payload)
        except Exception as exc:
            raise CaptureEventRuntimeError("active_floorplan_contract_failed") from exc
        if recorded is not True:
            raise CaptureEventRuntimeError("stale_floorplan_version")

    def _generate_floorplan(
        self,
        camera: str,
        *,
        max_age_sec: float,
        grid_res_m: float,
        max_extent_m: float,
        cache_only: bool,
        snapshot: FusedDepthSnapshot | None = None,
        snapshot_identity: tuple[str, str, str] | None = None,
    ) -> dict[str, Any]:
        if snapshot is not None and snapshot_identity is not None:
            raise CaptureEventRuntimeError("capture_event_configuration_invalid")
        kwargs: dict[str, Any] = {}
        if snapshot is not None:
            kwargs = {
                "snapshot_ref": _storage_ref(snapshot),
                "snapshot_id": snapshot.snapshot_id,
                "snapshot_content_sha256": snapshot.content_sha256,
            }
        elif snapshot_identity is not None:
            kwargs = {
                "snapshot_ref": snapshot_identity[0],
                "snapshot_id": snapshot_identity[1],
                "snapshot_content_sha256": snapshot_identity[2],
            }
        try:
            payload = self.storage.generate_topdown_floorplan(
                camera,
                max_age_sec=float(max_age_sec),
                grid_res_m=float(grid_res_m),
                max_extent_m=float(max_extent_m),
                cache_only=bool(cache_only),
                **kwargs,
            )
        except Exception as exc:
            raise CaptureEventRuntimeError("floorplan_generation_failed") from exc
        if not isinstance(payload, dict):
            raise CaptureEventRuntimeError("floorplan_generation_failed")
        return dict(payload)

    def floorplan_provider(
        self,
        camera: str | None = None,
        max_age_sec: float = 60.0,
        grid_res_m: float = 0.5,
        max_extent_m: float = 20.0,
        cache_only: bool = False,
        snapshot_ref: str | None = None,
        snapshot_id: str | None = None,
        snapshot_content_sha256: str | None = None,
        **_ignored: object,
    ) -> dict[str, Any]:
        request_camera = str(camera or "").strip()
        try:
            canonical = self._canonicalize(request_camera)
        except CaptureEventFusionError as exc:
            return {"error": exc.code, "camera_id": request_camera}
        if (
            type(max_age_sec) is bool  # noqa: E721
            or type(grid_res_m) is bool  # noqa: E721
            or type(max_extent_m) is bool  # noqa: E721
        ):
            return {"error": "invalid_floorplan_request", "camera_id": canonical}
        try:
            max_age = float(max_age_sec)
            grid_resolution = float(grid_res_m)
            max_extent = float(max_extent_m)
        except Exception:
            return {"error": "invalid_floorplan_request", "camera_id": canonical}
        if (
            type(cache_only) is not bool  # noqa: E721
            or not math.isfinite(max_age)
            or max_age < 0.0
            or not math.isfinite(grid_resolution)
            or grid_resolution <= 0.0
            or not math.isfinite(max_extent)
            or max_extent <= 0.0
        ):
            return {"error": "invalid_floorplan_request", "camera_id": canonical}
        exact_values = (snapshot_ref, snapshot_id, snapshot_content_sha256)
        exact_requested = any(value is not None for value in exact_values)
        if exact_requested and not all(
            isinstance(value, str) and bool(value.strip()) for value in exact_values
        ):
            return {"error": "invalid_snapshot_identity", "camera_id": canonical}
        if cache_only and exact_requested:
            return {"error": "invalid_floorplan_request", "camera_id": canonical}
        if self.shutdown_requested():
            return {"error": "shutting_down", "camera_id": canonical}
        if cache_only:
            try:
                cached = self._generate_floorplan(
                    canonical,
                    max_age_sec=max_age,
                    grid_res_m=grid_resolution,
                    max_extent_m=max_extent,
                    cache_only=True,
                )
            except CaptureEventRuntimeError as exc:
                self._escalate(exc, exc.code)
                return {"error": exc.code, "camera_id": canonical}
            cached_error = str(cached.get("error") or "")
            if cached_error:
                if cached_error in _FLOORPLAN_INTEGRITY_ERRORS:
                    error = CaptureEventRuntimeError(
                        "floorplan_snapshot_integrity_failed"
                    )
                    self._escalate(error, error.code)
                    return {"error": error.code, "camera_id": canonical}
                return cached
            try:
                self._record_floorplan(canonical, cached)
            except CaptureEventRuntimeError as exc:
                self._escalate(exc, exc.code)
                return {"error": exc.code, "camera_id": canonical}
            return cached

        # A successful manual depth refresh already produced one immutable fused
        # snapshot. Derive its floorplan directly when the caller supplies that
        # exact identity; do not reopen the valve or select a second cohort.
        if exact_requested:
            identity = (
                str(snapshot_ref).strip(),
                str(snapshot_id).strip(),
                str(snapshot_content_sha256).strip(),
            )
            try:
                exact = self._generate_floorplan(
                    canonical,
                    max_age_sec=max_age,
                    grid_res_m=grid_resolution,
                    max_extent_m=max_extent,
                    cache_only=False,
                    snapshot_identity=identity,
                )
            except CaptureEventRuntimeError as exc:
                self._escalate(exc, exc.code)
                return {"error": exc.code, "camera_id": canonical}
            exact_error = str(exact.get("error") or "")
            if exact_error:
                # Client-provided identity mismatch is a rejected request, not a
                # reason to poison the runtime. Integrity failures after a
                # descriptor was successfully resolved remain fatal below.
                if exact_error in {"invalid_snapshot_identity", "snapshot_identity_mismatch"}:
                    return {"error": exact_error, "camera_id": canonical}
                if exact_error in _FLOORPLAN_INTEGRITY_ERRORS:
                    error = CaptureEventRuntimeError(
                        "floorplan_snapshot_integrity_failed"
                    )
                    self._escalate(error, error.code)
                    return {"error": error.code, "camera_id": canonical}
                return {"error": exact_error, "camera_id": canonical}
            if (
                str(exact.get("snapshot_ref") or "") != identity[0]
                or str(exact.get("snapshot_id") or "") != identity[1]
                or str(exact.get("snapshot_content_sha256") or "") != identity[2]
            ):
                error = CaptureEventRuntimeError("floorplan_snapshot_integrity_failed")
                self._escalate(error, error.code)
                return {"error": error.code, "camera_id": canonical}
            exact["exact_snapshot_reused"] = True
            exact["depth_burst_triggered"] = False
            exact["depth_burst_fresh"] = False
            try:
                self._record_floorplan(canonical, exact)
            except CaptureEventRuntimeError as exc:
                self._escalate(exc, exc.code)
                return {"error": exc.code, "camera_id": canonical}
            return exact

        # A non-cache request means "capture now".  It must never be satisfied
        # by a generic latest/raw snapshot because doing so bypasses the shared
        # valve, exact fusion descriptor, and capture-event evidence.
        if not self.depth_branch_available():
            return {"error": "depth_branch_unavailable", "camera_id": canonical}
        try:
            controller = self._controller()
            outcome = controller.capture(
                build_capture_event_request(canonical, request_kind="floorplan")
            )
            exact = self._generate_floorplan(
                canonical,
                max_age_sec=max_age,
                grid_res_m=grid_resolution,
                max_extent_m=max_extent,
                cache_only=False,
                snapshot=outcome.fused_snapshot,
            )
        except (CaptureEventFusionError, CaptureEventRuntimeError) as exc:
            code = exc.code
            self._escalate(exc, code)
            return {"error": code, "camera_id": canonical}
        exact_error = str(exact.get("error") or "")
        if exact_error:
            if exact_error in _FLOORPLAN_INTEGRITY_ERRORS:
                error = CaptureEventRuntimeError("floorplan_snapshot_integrity_failed")
                self._escalate(error, error.code)
                return {"error": error.code, "camera_id": canonical}
            return {"error": exact_error, "camera_id": canonical}
        storage_ref = _storage_ref(outcome.fused_snapshot)
        try:
            exact_snapshot_ts = int(exact.get("snapshot_ts", 0) or 0)
        except Exception:
            exact_snapshot_ts = -1
        if (
            str(exact.get("snapshot_ref") or "") != storage_ref
            or str(exact.get("snapshot_id") or "") != outcome.fused_snapshot.snapshot_id
            or str(exact.get("snapshot_content_sha256") or "")
            != outcome.fused_snapshot.content_sha256
            or exact_snapshot_ts != outcome.fused_snapshot.timestamp_us
        ):
            error = CaptureEventRuntimeError("floorplan_snapshot_integrity_failed")
            self._escalate(error, error.code)
            return {"error": error.code, "camera_id": canonical}
        exact["depth_burst_triggered"] = True
        exact["depth_burst_fresh"] = True
        try:
            evidence, evidence_sha256 = _capture_event_evidence(outcome)
        except CaptureEventRuntimeError as exc:
            self._escalate(exc, exc.code)
            return {"error": exc.code, "camera_id": canonical}
        exact["capture_event"] = evidence
        exact["capture_event_evidence_sha256"] = evidence_sha256
        try:
            self._record_floorplan(canonical, exact)
        except CaptureEventRuntimeError as exc:
            self._escalate(exc, exc.code)
            return {"error": exc.code, "camera_id": canonical}
        return exact

    def capture_for_auto_calibration(
        self,
        cameras: Sequence[str],
        *,
        burst_seconds: float,
    ) -> tuple[CaptureEventOutcome, ...]:
        try:
            controller = self._controller()
        except CaptureEventRuntimeError as exc:
            self._escalate(exc, exc.code)
            raise
        outcomes: list[CaptureEventOutcome] = []
        for camera in cameras:
            try:
                outcomes.append(
                    controller.capture(
                        build_capture_event_request(
                            self._canonicalize(camera),
                            request_kind="depth",
                            burst_seconds=burst_seconds,
                        )
                    )
                )
            except (CaptureEventFusionError, CaptureEventRuntimeError) as exc:
                self._escalate(exc, exc.code)
                raise
        return tuple(outcomes)


__all__ = [
    "CaptureEventDepthStorage",
    "CaptureEventRuntimeError",
    "CaptureEventRuntimeProviders",
    "build_capture_event_request",
    "capture_event_runtime_error_is_fatal",
    "resolve_capture_event_drain_timeout_s",
]
