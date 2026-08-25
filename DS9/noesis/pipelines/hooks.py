from __future__ import annotations

import configparser
import colorsys
import hashlib
import json
import logging
import math
import os
import queue
import re
import time
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from geometry.depth_source import (
    DepthStorageManager,
    resolve_depth_store_commit_timeout_s,
)
from geometry.dewarper_validity import (
    build_dewarper_fov_mask as _build_dewarper_fov_mask,
    load_dewarper_fov_spec as _load_dewarper_fov_spec,
)
from geometry.homography import (
    Plane,
    estimate_upright_height_from_top_and_foot,
    intersect_plane,
    parse_extrinsics,
    project_world_to_image,
    ray_from_pixel,
)
from noesis.calibration.depth_registration import DepthRegistrationManager
from noesis.calibration.world_fusion_policy import WorldFusionPolicy
from noesis.calibration.geometry import pixel_to_world
from noesis.metadata import intrinsics as intrinsics_module
from noesis.metadata.depth_result import DepthResult
from noesis.metadata.object_depth import ObjectDepthResult
from noesis.metadata.pose_features import PoseFeatureResult
from noesis.mapanything_profiles import (
    get_mapanything_profile,
    resolve_runtime_mapanything_profile,
)
from noesis.capture_event_rgb_provider import PipelineRgbFrameProvider
from noesis_core.capture_event_fusion import TimestampedRgbFrame
from noesis.identity_v2_service import IdentityFramePrimitive
from noesis.reid_swin_profile import (
    REID_SWIN_EMBEDDING_DIM,
    REID_SWIN_OUTPUT_LAYER,
)
from noesis.identity_v2_osd import (
    IdentityV2PostResolutionOsdOperator as _SharedIdentityV2OsdOperator,
    IdentityV2PostResolutionOsdProcessor,
)
from noesis_core.v3dt_validation import (
    V3DTAxisMap,
    V3DTAxisMapError,
    v3dt_bbox3d_world_foot,
)
from noesis_core.mapanything_lifecycle import MapAnythingIdleReceipt
from noesis_core.servicemaker_shutdown import is_synthetic_stub_pipeline
from noesis_core.depth_contract import usable_registered_depth_m
from noesis_core.scene_prior import ScenePriorError, ScenePriorSet
from noesis_core.analytics_zones import resolve_authoritative_analytics_zone
from noesis_core.contracts.base import Matrix3, Vector3
from noesis_core.contracts.world_measurement import (
    ResolvedGroundMeasurement,
    WorldMeasurementCohort,
    WorldMeasurementHypothesis,
    WorldMeasurementSet,
    WorldPriorEvidence,
)
from noesis_core.world.resolver import (
    UniversalWorldMeasurementResolver,
    WorldMeasurementResolverConfig,
)
from noesis_core.runtime_publication import RuntimePublicationGate
from noesis_core.tracking_continuity import (
    TrackingContinuityUpdate,
    TrackingLifecycleRegistry,
    pair_safe_publication_interval_s,
)
from noesis.telemetry.bev import (
    BevPublicationReceipt,
    CalibrationSnapshot,
    Footpoint,
)
from noesis.telemetry.publishers import TrackingPublicationReceipt
from noesis.telemetry.person_ground_state import (
    HumanGroundConfig,
    PersonGroundState,
    PoseAnchorCandidate,
    admit_human_ground_output,
    assess_lower_body_occlusion,
    advance_human_cv_prediction,
    bind_world_frame,
    begin_source_admission,
    classify_posture,
    commit_image_path_point,
    complete_source_admission,
    mark_world_measurement_unavailable,
    observe_bbox_stationarity,
    observe_coherent_image_motion,
    record_accepted_image_geometry,
    resolve_pose_floor_anchor,
    source_score,
    transport_accepted_image_foot,
    integrate_projective_ground_observation,
    update_human_cv_filter,
    update_motion_mode,
    world_frame_binding_from_calibration,
    world_frame_matches_calibration,
)
from noesis.telemetry.world_contract_adapter import (
    frame_temporal_contract as _frame_temporal_contract,
    stable_identity_contract as _stable_identity_contract,
)


def _require_tracking_publication_receipt(
    value: Any,
    *,
    source_id: int,
    frame_id: int,
    observed_at_us: int,
) -> TrackingPublicationReceipt:
    if not isinstance(value, TrackingPublicationReceipt):
        raise RuntimeError("tracking publisher returned no typed admission receipt")
    if (
        value.source_id != int(source_id)
        or value.frame_id != int(frame_id)
        or value.observed_at_us != int(observed_at_us)
    ):
        raise RuntimeError("tracking publication receipt cohort mismatch")
    return value

try:  # DeepStream imports are optional during unit tests
    from pyservicemaker import (  # type: ignore
        BatchMetadataOperator,
        BufferOperator,
        Probe,
        osd as ds_osd,
    )
except Exception:  # pragma: no cover - exercised only in DS runtime
    BatchMetadataOperator = None  # type: ignore
    BufferOperator = None  # type: ignore
    Probe = None  # type: ignore
    ds_osd = None  # type: ignore


class _UnavailableBatchMetadataOperator:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("pyservicemaker.BatchMetadataOperator is unavailable")


_BatchMetadataOperatorBase = (
    BatchMetadataOperator
    if BatchMetadataOperator is not None
    else _UnavailableBatchMetadataOperator
)


class _UnavailableBufferOperator:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("pyservicemaker.BufferOperator is unavailable")


_BufferOperatorBase = (
    BufferOperator if BufferOperator is not None else _UnavailableBufferOperator
)

try:  # pragma: no cover - DeepStream bindings are optional during unit tests
    import pyds  # type: ignore
except Exception:  # pragma: no cover - handled gracefully when absent
    pyds = None  # type: ignore

try:  # pragma: no cover - optional native bridge for V3DT meta
    import noesis_v3dt_meta_ext  # type: ignore
except Exception:  # pragma: no cover - extension unavailable in tests
    noesis_v3dt_meta_ext = None  # type: ignore

try:  # pragma: no cover - optional native bridge for pose meta
    import noesis_pose_meta_ext  # type: ignore
except Exception:  # pragma: no cover - extension unavailable in tests
    noesis_pose_meta_ext = None  # type: ignore

try:  # pragma: no cover - required DS9 bridge for analytics object meta
    import noesis_analytics_meta_ext  # type: ignore
except Exception:  # pragma: no cover - extension unavailable in tests
    noesis_analytics_meta_ext = None  # type: ignore

try:  # pragma: no cover - optional native bridge for ReID tensor extraction
    import noesis_reid_meta_ext  # type: ignore
except Exception:  # pragma: no cover - extension unavailable in tests
    noesis_reid_meta_ext = None  # type: ignore

try:  # pragma: no cover - optional native bridge for object depth meta
    import noesis_depth_meta_ext  # type: ignore
except Exception:  # pragma: no cover - extension unavailable in tests
    noesis_depth_meta_ext = None  # type: ignore

try:  # pragma: no cover - optional native bridge for baseline depth tensor extraction
    import noesis_depth_tracking_tensor_ext  # type: ignore
except Exception:  # pragma: no cover - extension unavailable in tests
    noesis_depth_tracking_tensor_ext = None  # type: ignore

try:  # pragma: no cover - diagnostics optional in tests
    from noesis.diagnostics.telemetry_log import (
        TrackingDiagnosticsLogger,
        build_v3dt_session_start_payload,
    )
except Exception:  # pragma: no cover - fallback when diagnostics are absent
    TrackingDiagnosticsLogger = None  # type: ignore
    build_v3dt_session_start_payload = None  # type: ignore

logger = logging.getLogger(__name__)
_REID_NATIVE_MISSING_LOGGED = False
_DLPACK_HOST_READ_LOCK = threading.Lock()
_POSE_META_MAX_JSON_BYTES = 65536
_OBJECT_DEPTH_EXACT_FRAME_WAIT_DEFAULT_MS = 0.0
_OBJECT_DEPTH_EXACT_FRAME_WAIT_MAX_MS = 250.0
_OSD_LABEL_DEPTH_RE = re.compile(
    r"\s+(?:depth|z)=(?:n/a|[-+]?\d+(?:\.\d+)?m)\s*$",
    re.IGNORECASE,
)
_OSD_LABEL_CONF_RE = re.compile(r"\s+[-+]?\d+(?:\.\d+)?\s*$")
_OSD_LABEL_ID_RE = re.compile(r"\s+(?:\[[^\]]+\]\s*\|\s*\[[^\]]+\]|XX|\d+)\s*$")
_ENV_TRUE_VALUES = {"1", "true", "yes", "on", "y"}
_PYDS_COMPAT_ENV = "NOESIS_DS9_ALLOW_PYDS_COMPAT"
_INTRINSICS_USER_META_ENV = "NOESIS_DS9_ENABLE_INTRINSICS_USER_META"
_MAPANYTHING_EXACT_CAPTURE_NAME = "capture_mapanything_tensor_layers_exact"
_MAPANYTHING_LEGACY_CAPTURE_NAME = "capture_tensor_layers"
_MAPANYTHING_EXACT_LAYERS = frozenset({"depth", "conf", "mask"})
_MAPANYTHING_CANONICAL_PROFILE = get_mapanything_profile()
_MAPANYTHING_OUTPUT_HEIGHT = _MAPANYTHING_CANONICAL_PROFILE.input_height
_MAPANYTHING_OUTPUT_WIDTH = _MAPANYTHING_CANONICAL_PROFILE.input_width
_MAPANYTHING_HOST_PAYLOAD_BYTES = (
    _MAPANYTHING_CANONICAL_PROFILE.output_bytes_per_frame
)
_MAPANYTHING_ASYNC_STOP = object()
_MAPANYTHING_AMBIGUOUS_CONFIDENCE_SCALE = 0.50
_MAPANYTHING_OUTSIDE_CALIBRATED_FOV_CONFIDENCE_SCALE = 0.25
_MANUAL_DEPTH_BACKENDS = frozenset({"mapanything", "da3metric-large"})
_DA3METRIC_LARGE_FOCAL_DENOMINATOR = 300.0
_WORLD_ESTIMATOR_DIAGNOSTIC_FIELDS = (
    "world_estimator_evaluated",
    "world_floor_candidate",
    "world_floor_range_m",
    "world_floor_range_limit_m",
    "world_floor_incidence_sin",
    "world_floor_admitted",
    "world_floor_rejection_reason",
    "world_floor_contact_plausible",
    "world_floor_contact_gap_px",
    "world_floor_contact_gap_ratio",
    "world_floor_contact_rejection_reason",
    "world_floor_bbox_bottom_range_m",
    "world_floor_contact_range_delta_m",
    "world_floor_contact_range_tolerance_m",
    "world_depth_candidate",
    "world_depth_rejection_reason",
    "world_observation_range_m",
    "world_observation_range_limit_m",
    "world_observation_range_admitted",
    "world_observation_range_rejection_reason",
    "world_prefilter_measurement",
    "world_filter_prediction",
    "world_prediction_provenance",
    "world_prediction_image_foot",
    "world_projective_continuation_allowed",
    "world_projective_authoritative_allowed",
    "world_measurement_accepted",
    "world_rejection_reason",
    "world_innovation_m",
    "world_innovation_limit_m",
    "world_reacquire_count",
    "world_reacquired",
    "world_contact_basis",
    "world_image_motion_supported",
    "world_image_motion_streak",
    "trail_break_required",
    "trail_segment_id",
    "world_fusion_policy_id",
    "world_floor_weight_scale",
    "world_depth_weight_scale",
    "world_floor_weight_effective",
    "world_depth_weight_effective",
    "motion_mode",
    "posture",
    "trail_append_allowed",
    "idle_jitter_m",
    "source_switch_count",
    "sticky_world_source",
    "scene_prior",
)


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in _ENV_TRUE_VALUES


def _allow_raw_pyds_compat() -> bool:
    return _env_flag(_PYDS_COMPAT_ENV, default=False)


def _record_quarantined_compat_path(path: str, env_name: str, *, level: int = logging.DEBUG) -> int:
    count = _increment_core_counter(f"ds9_compat_quarantined_total.{path}")
    if count <= 1 or (count % 250) == 0:
        logger.log(
            level,
            "DS9 compatibility path %s is disabled by default; set %s=1 to allow it",
            path,
            env_name,
        )
    return count


def _require_ds9_mapanything_native_capture() -> Callable[..., Any]:
    """Return the one DS9-owned MapAnything tensor capture entrypoint.

    Runtime import-path validation proves the module binary is owned by the
    active DS9 release.  This check additionally prevents an old generic
    capture surface from coexisting with the exact UID/layer/batch contract.
    """

    extension = noesis_depth_tracking_tensor_ext
    if extension is None:
        raise RuntimeError(
            "DS9 MapAnything requires the owned noesis_depth_tracking_tensor_ext"
        )
    capture = getattr(extension, _MAPANYTHING_EXACT_CAPTURE_NAME, None)
    if not callable(capture):
        raise RuntimeError(
            "DS9 MapAnything native extension is missing exact capture entrypoint "
            f"{_MAPANYTHING_EXACT_CAPTURE_NAME}"
        )
    if callable(getattr(extension, _MAPANYTHING_LEGACY_CAPTURE_NAME, None)):
        raise RuntimeError(
            "DS9 MapAnything native extension exposes the retired generic tensor "
            "capture alongside the exact capture contract"
        )
    return capture


@dataclass
class _CorePathInstrumentation:
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    counters: Dict[str, int] = field(default_factory=dict, init=False, repr=False)
    serialization_prep: Dict[str, Dict[str, int]] = field(default_factory=dict, init=False, repr=False)
    stage_timings: Dict[str, Dict[str, int]] = field(default_factory=dict, init=False, repr=False)
    events: deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=512), init=False, repr=False)

    def _inc_locked(self, key: str, delta: int = 1) -> int:
        value = int(self.counters.get(key, 0)) + int(delta)
        self.counters[key] = value
        return value

    def record_cpu_copy_violation(
        self,
        *,
        location: str,
        reason: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        now_ns = time.time_ns()
        with self._lock:
            total = self._inc_locked("core_path.cpu_copy_violation.total")
            per_loc = self._inc_locked(f"core_path.cpu_copy_violation.{location}")
            event: Dict[str, Any] = {
                "type": "core_path_cpu_copy_violation",
                "ts_ns": int(now_ns),
                "location": str(location),
                "reason": str(reason),
                "count": int(per_loc),
                "total": int(total),
            }
            if details:
                event["details"] = dict(details)
            if per_loc <= 3 or (per_loc % 250) == 0:
                self.events.append(event)

    def record_boundary_serialization_prep(
        self,
        *,
        metric: str,
        duration_ns: int,
        payload_bytes: int | None = None,
    ) -> None:
        now_ns = time.time_ns()
        metric_key = str(metric)
        with self._lock:
            bucket = self.serialization_prep.get(metric_key)
            if bucket is None:
                bucket = {
                    "count": 0,
                    "total_ns": 0,
                    "max_ns": 0,
                    "last_ns": 0,
                    "total_bytes": 0,
                    "last_payload_bytes": 0,
                }
                self.serialization_prep[metric_key] = bucket
            bucket["count"] = int(bucket.get("count", 0)) + 1
            bucket["total_ns"] = int(bucket.get("total_ns", 0)) + max(0, int(duration_ns))
            bucket["max_ns"] = max(int(bucket.get("max_ns", 0)), max(0, int(duration_ns)))
            bucket["last_ns"] = max(0, int(duration_ns))
            if payload_bytes is not None:
                bucket["total_bytes"] = int(bucket.get("total_bytes", 0)) + max(0, int(payload_bytes))
                bucket["last_payload_bytes"] = max(0, int(payload_bytes))
            count = int(bucket["count"])
            self._inc_locked("boundary_serialization_prep.total")
            self._inc_locked(f"boundary_serialization_prep.{metric_key}")
            if count <= 3 or (count % 250) == 0:
                event: Dict[str, Any] = {
                    "type": "boundary_serialization_prep",
                    "ts_ns": int(now_ns),
                    "metric": metric_key,
                    "duration_ns": max(0, int(duration_ns)),
                    "count": count,
                }
                if payload_bytes is not None:
                    event["payload_bytes"] = max(0, int(payload_bytes))
                self.events.append(event)

    def record_stage_timing(
        self,
        *,
        metric: str,
        duration_ns: int,
        item_count: int | None = None,
    ) -> None:
        now_ns = time.time_ns()
        metric_key = str(metric)
        elapsed_ns = max(0, int(duration_ns))
        with self._lock:
            bucket = self.stage_timings.get(metric_key)
            if bucket is None:
                bucket = {
                    "count": 0,
                    "total_ns": 0,
                    "max_ns": 0,
                    "last_ns": 0,
                    "total_items": 0,
                    "last_items": 0,
                }
                self.stage_timings[metric_key] = bucket
            bucket["count"] = int(bucket.get("count", 0)) + 1
            bucket["total_ns"] = int(bucket.get("total_ns", 0)) + elapsed_ns
            bucket["max_ns"] = max(int(bucket.get("max_ns", 0)), elapsed_ns)
            bucket["last_ns"] = elapsed_ns
            if item_count is not None:
                items = max(0, int(item_count))
                bucket["total_items"] = int(bucket.get("total_items", 0)) + items
                bucket["last_items"] = items
            count = int(bucket["count"])
            self._inc_locked("detection_wake.stage_timing.total")
            self._inc_locked(f"detection_wake.stage_timing.{metric_key}")
            if count <= 3 or (count % 250) == 0:
                event: Dict[str, Any] = {
                    "type": "detection_wake_stage_timing",
                    "ts_ns": int(now_ns),
                    "metric": metric_key,
                    "duration_ns": elapsed_ns,
                    "count": count,
                }
                if item_count is not None:
                    event["item_count"] = max(0, int(item_count))
                self.events.append(event)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "counters": dict(self.counters),
                "serialization_prep": {k: dict(v) for k, v in self.serialization_prep.items()},
                "stage_timings": {k: dict(v) for k, v in self.stage_timings.items()},
                "events": list(self.events),
            }

    def reset(self) -> None:
        with self._lock:
            self.counters.clear()
            self.serialization_prep.clear()
            self.stage_timings.clear()
            self.events.clear()


_CORE_PATH_INSTRUMENTATION = _CorePathInstrumentation()


def get_core_path_instrumentation_snapshot() -> Dict[str, Any]:
    """Return a thread-safe snapshot of core-path conversion counters/events."""
    return _CORE_PATH_INSTRUMENTATION.snapshot()


def reset_core_path_instrumentation() -> None:
    """Reset core-path conversion counters/events."""
    _CORE_PATH_INSTRUMENTATION.reset()


def _increment_core_counter(metric: str, delta: int = 1) -> int:
    """Increment a custom core instrumentation counter."""
    key = str(metric)
    with _CORE_PATH_INSTRUMENTATION._lock:
        current = int(_CORE_PATH_INSTRUMENTATION.counters.get(key, 0))
        updated = current + int(delta)
        _CORE_PATH_INSTRUMENTATION.counters[key] = updated
        return updated


def _set_core_counter(metric: str, value: int) -> int:
    """Set an instrumentation gauge stored alongside monotonic counters."""

    key = str(metric)
    normalized = max(0, int(value))
    with _CORE_PATH_INSTRUMENTATION._lock:
        _CORE_PATH_INSTRUMENTATION.counters[key] = normalized
    return normalized


def _max_core_counter(metric: str, value: int) -> int:
    """Raise a monotonic instrumentation high-watermark."""

    key = str(metric)
    normalized = max(0, int(value))
    with _CORE_PATH_INSTRUMENTATION._lock:
        current = int(_CORE_PATH_INSTRUMENTATION.counters.get(key, 0))
        updated = max(current, normalized)
        _CORE_PATH_INSTRUMENTATION.counters[key] = updated
        return updated


def _record_core_stage_timing(metric: str, start_ns: int, *, item_count: int | None = None) -> None:
    _CORE_PATH_INSTRUMENTATION.record_stage_timing(
        metric=str(metric),
        duration_ns=time.perf_counter_ns() - int(start_ns),
        item_count=item_count,
    )


def _read_env_int(name: str, default: int, *, min_value: int = 0) -> int:
    try:
        value = int(str(os.environ.get(name, str(default))).strip() or str(default))
    except Exception:
        value = int(default)
    return max(int(min_value), int(value))


def _read_env_float(name: str, default: float, *, min_value: float = 0.0) -> float:
    try:
        value = float(str(os.environ.get(name, str(default))).strip() or str(default))
    except Exception:
        value = float(default)
    return max(float(min_value), float(value))


def _bounded_object_depth_wait_ms(value: Any, *, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default)
    if not math.isfinite(parsed):
        parsed = float(default)
    return min(
        float(_OBJECT_DEPTH_EXACT_FRAME_WAIT_MAX_MS),
        max(0.0, float(parsed)),
    )


def _serialize_compact_json_with_metrics(payload: Mapping[str, Any], *, metric: str) -> str:
    start_ns = time.perf_counter_ns()
    encoded = ""
    try:
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=False)
        return encoded
    finally:
        _CORE_PATH_INSTRUMENTATION.record_boundary_serialization_prep(
            metric=metric,
            duration_ns=time.perf_counter_ns() - start_ns,
            payload_bytes=len(encoded.encode("utf-8")) if encoded else 0,
        )


def _pose_meta_payload_limit_bytes() -> int:
    raw = str(os.environ.get("NOESIS_POSE_META_MAX_JSON_BYTES", _POSE_META_MAX_JSON_BYTES) or _POSE_META_MAX_JSON_BYTES).strip()
    try:
        parsed = int(raw)
    except Exception:
        parsed = int(_POSE_META_MAX_JSON_BYTES)
    return max(1024, parsed)


def _read_nvinfer_property_int(config_path: Any, key: str) -> Optional[int]:
    if not config_path:
        return None
    try:
        path = Path(str(config_path))
        if not path.exists():
            return None
        parser = configparser.ConfigParser()
        parser.optionxform = str
        if not parser.read(str(path), encoding="utf-8"):
            return None
        if not parser.has_section("property"):
            return None
        raw = parser.get("property", str(key), fallback=None)
        if raw is None:
            return None
        return int(str(raw).strip())
    except Exception:
        return None


def _frame_pts_key_us(frame_meta: Any) -> int:
    raw = int(_meta_lookup(frame_meta, "buf_pts", "buffer_pts", default=0) or 0)
    if raw > 0:
        return raw // 1000
    return int(time.time_ns() // 1000)


def _depth_frame_key(frame_meta: Any) -> FrameKey:
    return (
        int(_meta_lookup(frame_meta, "source_id", "pad_index", default=0) or 0),
        int(_meta_lookup(frame_meta, "frame_number", "frame_num", default=0) or 0),
        _frame_pts_key_us(frame_meta),
    )


def _canonical_frame_size(frame_meta: Any, fallback_size: Tuple[int, int]) -> Tuple[int, int]:
    frame_w = int(_meta_lookup(frame_meta, "frame_width", "width", default=0) or 0)
    frame_h = int(_meta_lookup(frame_meta, "frame_height", "height", default=0) or 0)
    if frame_w > 0 and frame_h > 0:
        return frame_w, frame_h
    return max(0, int(fallback_size[0] or 0)), max(0, int(fallback_size[1] or 0))


def _tensor_to_numpy_cpu(tensor: Any) -> Optional[np.ndarray]:
    if isinstance(tensor, np.ndarray):
        return np.asarray(tensor)
    dlpack_fn = getattr(tensor, "__dlpack__", None)
    if callable(dlpack_fn):
        try:
            with _DLPACK_HOST_READ_LOCK:
                import torch
                import torch.utils.dlpack as torch_dlpack

                stream = 0
                try:
                    if torch.cuda.is_available():
                        stream = int(torch.cuda.current_stream().cuda_stream)
                except Exception:
                    stream = 0
                capsule = dlpack_fn(stream)
                return torch_dlpack.from_dlpack(capsule).detach().cpu().numpy()
        except Exception:
            logger.debug("Tensor DLPack conversion failed", exc_info=True)
    return None


def _iter_frame_tensor_meta(frame_meta: Any, *, unique_id: int) -> Iterable[Any]:
    if not _allow_raw_pyds_compat():
        _record_quarantined_compat_path("_iter_frame_tensor_meta", _PYDS_COMPAT_ENV)
        return
    if pyds is None:
        return
    meta_list = getattr(frame_meta, "frame_user_meta_list", None)
    if meta_list is None:
        return
    user_meta_cast = _resolve_pyds_cast("NvDsUserMeta")
    tensor_meta_cast = _resolve_pyds_cast("NvDsInferTensorMeta")
    target_meta_type = _resolve_pyds_attr("NVDSINFER_TENSOR_OUTPUT_META")
    if target_meta_type is None:
        meta_enum = _resolve_pyds_attr("NvDsMetaType")
        target_meta_type = getattr(meta_enum, "NVDSINFER_TENSOR_OUTPUT_META", None)
    for user_meta in _iter_meta_entries(meta_list, user_meta_cast):
        if user_meta is None:
            continue
        base_meta = getattr(user_meta, "base_meta", None)
        current_type = getattr(base_meta, "meta_type", getattr(user_meta, "meta_type", None))
        if target_meta_type is not None and current_type != target_meta_type:
            continue
        payload = getattr(user_meta, "user_meta_data", None)
        if payload is None:
            continue
        if tensor_meta_cast is not None:
            try:
                payload = tensor_meta_cast(payload)
            except Exception:
                continue
        try:
            current_uid = int(getattr(payload, "unique_id", -1))
        except Exception:
            current_uid = -1
        if current_uid != int(unique_id):
            continue
        yield payload


def _select_depth_layer(layers: Mapping[str, Any]) -> Any:
    for name in ("depth", "pred", "output"):
        if name in layers and layers.get(name) is not None:
            return layers[name]
    return next(iter(layers.values()))


def _erode_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    if mask.size <= 0 or not np.any(mask):
        return np.asarray(mask, dtype=bool)
    kernel = np.ones((max(1, int(kernel_size)), max(1, int(kernel_size))), dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8, copy=False), kernel, iterations=1)
    return np.asarray(eroded > 0, dtype=bool)


def _band_mask(mask: np.ndarray, *, y0_ratio: float, y1_ratio: float, center_width_ratio: float) -> np.ndarray:
    height, width = mask.shape[:2]
    if height <= 0 or width <= 0:
        return np.zeros_like(mask, dtype=bool)
    y0 = max(0, min(height, int(math.floor(height * float(y0_ratio)))))
    y1 = max(y0 + 1, min(height, int(math.ceil(height * float(y1_ratio)))))
    band_width = max(1, min(width, int(round(width * float(center_width_ratio)))))
    center_x = width * 0.5
    x0 = max(0, min(width, int(round(center_x - (band_width * 0.5)))))
    x1 = max(x0 + 1, min(width, int(round(center_x + (band_width * 0.5)))))
    band = np.zeros_like(mask, dtype=bool)
    band[y0:y1, x0:x1] = True
    return np.logical_and(mask, band)


def _bounded_depth_stat_values(values: np.ndarray, *, env_name: str = "NOESIS_OBJECT_DEPTH_MAX_STAT_SAMPLES") -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size <= 0:
        return arr
    max_samples = _read_env_int(env_name, 4096, min_value=128)
    if arr.size <= max_samples:
        return arr
    stride = max(1, int(math.ceil(float(arr.size) / float(max_samples))))
    return np.asarray(arr[::stride][:max_samples], dtype=np.float32)


def _depth_spread_from_bounds(
    depth_p10: Optional[float],
    depth_p90: Optional[float],
) -> Optional[float]:
    if depth_p10 is None or depth_p90 is None:
        return None
    try:
        spread = float(depth_p90) - float(depth_p10)
    except Exception:
        return None
    if not math.isfinite(spread) or spread < 0.0:
        return None
    return float(spread)


def _depth_spread_limit_m(depth_m: Optional[float], *, strict: bool) -> float:
    try:
        center = abs(float(depth_m)) if depth_m is not None else 0.0
    except Exception:
        center = 0.0
    relative = center * (0.10 if strict else 0.14)
    floor = 0.25 if strict else 0.35
    ceiling = 0.55 if strict else 0.75
    return float(min(ceiling, max(floor, relative)))


def _depth_spread_is_supported(
    depth_m: Optional[float],
    depth_spread_m: Optional[float],
    *,
    strict: bool,
) -> bool:
    if depth_m is None or depth_spread_m is None:
        return False
    try:
        center = float(depth_m)
        spread = float(depth_spread_m)
    except Exception:
        return False
    return bool(
        math.isfinite(center)
        and center > 0.0
        and math.isfinite(spread)
        and 0.0 <= spread <= _depth_spread_limit_m(center, strict=strict)
    )


def _depth_evidence_rejection_reason(
    *,
    sample_count: int,
    valid_fraction: float,
    depth_m: Optional[float],
    depth_spread_m: Optional[float],
    min_sample_count: int,
    min_valid_fraction: float,
    strict_spread: bool,
) -> Optional[str]:
    if int(sample_count) < int(min_sample_count):
        return "depth_support_count_low"
    if float(valid_fraction) < float(min_valid_fraction):
        return "depth_support_fraction_low"
    if depth_spread_m is None:
        return "depth_spread_unavailable"
    if not _depth_spread_is_supported(
        depth_m,
        depth_spread_m,
        strict=bool(strict_spread),
    ):
        return "depth_spread_exceeded"
    return None


def _extract_person_depth_anchor(
    mask: np.ndarray,
    depth_crop: np.ndarray,
    *,
    frame_origin: Tuple[int, int],
) -> _DepthAnchorSample:
    if mask.size <= 0 or depth_crop.size <= 0:
        return _DepthAnchorSample(
            foot_uv=None,
            anchor_source=None,
            anchor_depth_m=None,
            anchor_sample_count=0,
            anchor_valid_fraction=0.0,
            anchor_depth_spread_m=None,
            anchor_rejection_reason="empty_person_support",
            lower_body_sample_count=0,
            lower_body_valid_fraction=0.0,
            torso_sample_count=0,
            torso_valid_fraction=0.0,
        )

    origin_x, origin_y = int(frame_origin[0]), int(frame_origin[1])
    eroded_mask = _erode_mask(mask, kernel_size=3)

    foot_uv: Optional[Point2] = None
    lower_rows = np.nonzero(mask)[0]
    if lower_rows.size > 0:
        max_row = int(np.max(lower_rows))
        band_top = max(0, max_row - max(1, int(round(mask.shape[0] * 0.12))))
        foot_band = np.zeros_like(mask, dtype=bool)
        foot_band[band_top : max_row + 1, :] = True
        foot_band = np.logical_and(foot_band, _band_mask(mask, y0_ratio=0.0, y1_ratio=1.0, center_width_ratio=0.35))
        points = np.argwhere(foot_band)
        if points.size > 0:
            foot_y = int(np.max(points[:, 0]))
            foot_x = int(np.median(points[points[:, 0] == foot_y][:, 1]))
            foot_uv = (float(origin_x + foot_x), float(origin_y + foot_y))

    def _anchor_support_requirements(area_px: int, anchor_source: str) -> Tuple[int, float]:
        area_px = max(0, int(area_px))
        if anchor_source == "lower_body_band":
            base_count = 24
            floor_count = 16
            min_valid_fraction = 0.40
        else:
            base_count = 32
            floor_count = 20
            min_valid_fraction = 0.45
        adaptive_count = int(math.ceil(float(area_px) * 0.25))
        min_count = max(floor_count, min(base_count, adaptive_count or base_count))
        return min_count, min_valid_fraction

    lower_body_mask = _band_mask(eroded_mask, y0_ratio=0.88, y1_ratio=1.0, center_width_ratio=0.35)
    lower_values_full = np.asarray(depth_crop[np.logical_and(lower_body_mask, np.isfinite(depth_crop))], dtype=np.float32)
    lower_values = _bounded_depth_stat_values(lower_values_full, env_name="NOESIS_OBJECT_DEPTH_MAX_ANCHOR_SAMPLES")
    lower_count = int(lower_values_full.size)
    lower_area = int(np.count_nonzero(lower_body_mask))
    lower_valid_fraction = float(lower_count) / float(lower_area or 1)
    lower_min_count, lower_min_valid_fraction = _anchor_support_requirements(lower_area, "lower_body_band")
    lower_median = float(np.median(lower_values)) if lower_values.size > 0 else None
    lower_p10 = float(np.percentile(lower_values, 10.0)) if lower_values.size > 0 else None
    lower_p90 = float(np.percentile(lower_values, 90.0)) if lower_values.size > 0 else None
    lower_spread = _depth_spread_from_bounds(lower_p10, lower_p90)
    lower_support_ok = bool(
        lower_count >= lower_min_count
        and lower_valid_fraction >= lower_min_valid_fraction
    )
    if lower_support_ok and _depth_spread_is_supported(
        lower_median,
        lower_spread,
        strict=False,
    ):
        return _DepthAnchorSample(
            foot_uv=foot_uv,
            anchor_source="lower_body_band",
            anchor_depth_m=lower_median,
            anchor_sample_count=lower_count,
            anchor_valid_fraction=lower_valid_fraction,
            anchor_depth_spread_m=lower_spread,
            anchor_rejection_reason=None,
            lower_body_sample_count=lower_count,
            lower_body_valid_fraction=lower_valid_fraction,
            torso_sample_count=0,
            torso_valid_fraction=0.0,
        )

    torso_mask = _band_mask(eroded_mask, y0_ratio=0.35, y1_ratio=0.70, center_width_ratio=0.50)
    torso_values_full = np.asarray(depth_crop[np.logical_and(torso_mask, np.isfinite(depth_crop))], dtype=np.float32)
    torso_values = _bounded_depth_stat_values(torso_values_full, env_name="NOESIS_OBJECT_DEPTH_MAX_ANCHOR_SAMPLES")
    torso_count = int(torso_values_full.size)
    torso_area = int(np.count_nonzero(torso_mask))
    torso_valid_fraction = float(torso_count) / float(torso_area or 1)
    torso_min_count, torso_min_valid_fraction = _anchor_support_requirements(torso_area, "torso_core")
    torso_median = float(np.median(torso_values)) if torso_values.size > 0 else None
    torso_p10 = float(np.percentile(torso_values, 10.0)) if torso_values.size > 0 else None
    torso_p90 = float(np.percentile(torso_values, 90.0)) if torso_values.size > 0 else None
    torso_spread = _depth_spread_from_bounds(torso_p10, torso_p90)
    torso_support_ok = bool(
        torso_count >= torso_min_count
        and torso_valid_fraction >= torso_min_valid_fraction
    )
    anchor_depth = (
        torso_median
        if torso_support_ok
        and _depth_spread_is_supported(torso_median, torso_spread, strict=False)
        else None
    )
    anchor_source = "torso_core" if anchor_depth is not None else None
    if anchor_depth is not None:
        rejection_reason = None
        anchor_spread = torso_spread
    elif lower_support_ok and lower_spread is not None:
        rejection_reason = "lower_body_depth_spread_exceeded"
        anchor_spread = lower_spread
    elif torso_support_ok and torso_spread is not None:
        rejection_reason = "torso_depth_spread_exceeded"
        anchor_spread = torso_spread
    else:
        rejection_reason = "person_depth_support_low"
        anchor_spread = torso_spread if torso_spread is not None else lower_spread
    return _DepthAnchorSample(
        foot_uv=foot_uv,
        anchor_source=anchor_source,
        anchor_depth_m=anchor_depth,
        anchor_sample_count=torso_count if anchor_depth is not None else 0,
        anchor_valid_fraction=torso_valid_fraction if anchor_depth is not None else 0.0,
        anchor_depth_spread_m=anchor_spread,
        anchor_rejection_reason=rejection_reason,
        lower_body_sample_count=lower_count,
        lower_body_valid_fraction=lower_valid_fraction,
        torso_sample_count=torso_count,
        torso_valid_fraction=torso_valid_fraction,
    )


def _extract_object_depth_result_from_meta(obj_meta: Any) -> Optional[ObjectDepthResult]:
    if obj_meta is None or noesis_depth_meta_ext is None:
        return None
    extract_fn = getattr(noesis_depth_meta_ext, "extract_object_depth", None)
    if not callable(extract_fn):
        return None
    try:
        raw = extract_fn(obj_meta)
    except Exception:
        return None
    if raw in (None, ""):
        return None
    try:
        if isinstance(raw, Mapping):
            return ObjectDepthResult.from_dict(raw)
        return ObjectDepthResult.from_json(str(raw))
    except Exception:
        logger.debug("Failed to decode NOESIS.OBJECT_DEPTH payload", exc_info=True)
        return None


def _depth_used_m(depth_result: Optional[ObjectDepthResult]) -> Optional[float]:
    if depth_result is None:
        return None
    if str(depth_result.status) != "ok":
        return None
    if str(depth_result.evidence_quality or "").strip().lower() == "rejected":
        return None
    depth_m = depth_result.anchor_depth_m
    if depth_m is None:
        return None
    try:
        depth_val = float(depth_m)
    except Exception:
        return None
    if not math.isfinite(depth_val) or depth_val <= 0.0:
        return None
    return depth_val


def _depth_anchor_uv(depth_result: Optional[ObjectDepthResult]) -> Optional[Tuple[float, float]]:
    if depth_result is None:
        return None
    anchor_uv = depth_result.anchor_uv
    if not isinstance(anchor_uv, (list, tuple)) or len(anchor_uv) < 2:
        return None
    try:
        u = float(anchor_uv[0])
        v = float(anchor_uv[1])
    except Exception:
        return None
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    return float(u), float(v)


def _format_depth_label_fragment(depth_result: Optional[ObjectDepthResult], *, decimals: int) -> Optional[str]:
    if depth_result is None:
        return None
    depth_used = _depth_used_m(depth_result)
    if depth_used is not None:
        return f"depth={depth_used:.{max(0, int(decimals))}f}m"
    return "depth=n/a"


def _clean_osd_base_label(label: str) -> str:
    cleaned = str(label or "").strip()
    if not cleaned:
        return ""
    for pattern in (_OSD_LABEL_DEPTH_RE, _OSD_LABEL_CONF_RE, _OSD_LABEL_ID_RE):
        cleaned = pattern.sub("", cleaned).strip()
    return cleaned or str(label or "").strip()


def attach_intrinsics_hook(
    pipeline: "DS8Pipeline",
    *,
    config_path: str | Path | None = None,
) -> None:
    """Attach a per-frame intrinsics hook to the DS8 pipeline."""
    loader = _resolve_intrinsics_loader(config_path)
    component = pipeline.components.get("streammux")
    if component is None:
        raise KeyError("streammux component missing; cannot attach intrinsics hook")

    processor = _IntrinsicsProcessor(loader=loader, pipeline=pipeline)
    component.config["_intrinsics_processor"] = processor

    if pipeline.ds_pipeline is None or BatchMetadataOperator is None or Probe is None:
        logger.debug("Intrinsics hook stored for lazy execution (pyservicemaker unavailable)")
        return

    try:
        probe = Probe("intrinsics_hook", _IntrinsicsOperator(processor))
        pipeline.ds_pipeline.attach(component.name, probe)
        logger.info("Attached intrinsics probe to %s", component.name)
    except Exception:  # pragma: no cover - depends on DS runtime availability
        logger.exception("Failed to attach intrinsics probe; intrinsics will not stream")


def attach_mapanything_postprocess_hook(
    pipeline: "DS8Pipeline",
    *,
    storage: DepthStorageManager,
    depth_pub: "DepthTelemetryPublisher" | None = None,
    camera_labels: Optional[Mapping[int, str]] = None,
    failure_callback: Optional[Callable[[BaseException], None]] = None,
    rgb_provider: PipelineRgbFrameProvider | None = None,
) -> None:
    """Attach the MapAnything post-process hook to decode full-frame tensor meta.

    This operator expects the MapAnything model to run as a full-frame nvinfer
    branch (parallel to the main PGIE/tracker chain), not as a per-object SGIE.
    """
    ma_cfg = (pipeline.config.get("models") or {}).get("mapanything") or {}
    gie_id = int(ma_cfg.get("gie_id", 2))
    batch_size = int(ma_cfg.get("batch_size", 0) or 0)
    if gie_id <= 0:
        raise ValueError("models.mapanything.gie_id must be positive")
    if batch_size <= 0:
        raise ValueError("models.mapanything.batch_size must be positive")
    backend = str(ma_cfg.get("backend") or "mapanything").strip().lower()
    if backend not in _MANUAL_DEPTH_BACKENDS:
        raise ValueError(
            "models.mapanything.backend must be mapanything or da3metric-large"
        )
    if backend == "mapanything":
        profile = resolve_runtime_mapanything_profile(ma_cfg)
        profile_name = profile.name
        output_height = profile.input_height
        output_width = profile.input_width
    else:
        profile = None
        profile_name = backend
        output_height = 294
        output_width = 518
    ma_name = ma_cfg.get("name", "mapanything_fullframe")
    component = pipeline.components.get(ma_name)
    if component is None:
        raise KeyError(f"mapanything component '{ma_name}' missing in pipeline graph")
    capture_component = component
    streammux = pipeline.components.get("streammux")
    rgb_width = 0
    rgb_height = 0
    if rgb_provider is not None:
        capture_component = pipeline.components.get("mapanything_rgb_caps")
        if capture_component is None:
            raise KeyError(
                "mapanything_rgb_caps component missing; exact RGB capture "
                "cannot bind the inference tensors and RGB surface"
            )
        if streammux is None:
            raise KeyError("streammux component missing; exact RGB shape is unknown")
        rgb_width = int(streammux.config.get("width", 0) or 0)
        rgb_height = int(streammux.config.get("height", 0) or 0)
        if rgb_width <= 0 or rgb_height <= 0:
            raise ValueError("streammux dimensions must define exact RGB HxW")

    processor = MapAnythingProcessor(
        pipeline=pipeline,
        storage=storage,
        depth_pub=depth_pub,
        gie_id=gie_id,
        batch_size=batch_size,
        profile_name=profile_name,
        output_height=output_height,
        output_width=output_width,
        depth_backend=backend,
        metric_focal_denominator=float(
            ma_cfg.get(
                "metric_focal_denominator",
                _DA3METRIC_LARGE_FOCAL_DENOMINATOR,
            )
        ),
        camera_labels=camera_labels or getattr(pipeline, "camera_labels", {}) or {},
        failure_callback=failure_callback,
        rgb_provider=rgb_provider,
        rgb_width=rgb_width,
        rgb_height=rgb_height,
    )
    component.config["_mapanything_processor"] = processor
    pipeline.mapanything_processor = processor

    if pipeline.ds_pipeline is None or is_synthetic_stub_pipeline(pipeline):
        logger.debug("Stored MapAnything processor for lazy execution (pyservicemaker unavailable)")
        return
    if BufferOperator is None or Probe is None:
        raise RuntimeError(
            "DS9 MapAnything exact native post-process requires "
            "pyservicemaker.BufferOperator and Probe"
        )

    try:
        processor.native_capture = _require_ds9_mapanything_native_capture()
        probe = Probe(
            "mapanything_postprocess",
            _MapAnythingBufferOperator(processor),
        )
        pipeline.ds_pipeline.attach(capture_component.name, probe)
        logger.info(
            "Attached MapAnything post-process probe to %s "
            "(backend=%s profile=%s shape=%sx%s payload_bytes=%s)",
            capture_component.name,
            backend,
            profile_name,
            output_height,
            output_width,
            processor.host_payload_bytes,
        )
    except Exception as exc:  # pragma: no cover - depends on DS runtime availability
        logger.exception("Failed to attach MapAnything post-process probe")
        raise RuntimeError(
            "DS9 MapAnything exact native post-process attachment failed"
        ) from exc


def attach_pose_feature_hook(
    pipeline: "DS8Pipeline",
    *,
    camera_labels: Optional[Mapping[int, str]] = None,
) -> None:
    """Attach the reviewed SGIE or RF-DETR PGIE pose feature hook."""

    models_cfg = (pipeline.config.get("models") or {})
    if not isinstance(models_cfg, Mapping):
        models_cfg = {}
    pose_cfg = models_cfg.get("pose") or {}
    if not isinstance(pose_cfg, Mapping):
        pose_cfg = {}
    pose_enabled = bool(pose_cfg.get("enable", True)) and any(
        key in pose_cfg for key in ("config-file-path", "engine", "name")
    )
    rfdetr_cfg = models_cfg.get("rfdetr_keypoint") or {}
    if not isinstance(rfdetr_cfg, Mapping):
        rfdetr_cfg = {}
    rfdetr_enabled = bool(rfdetr_cfg.get("enable", False))
    if pose_enabled and rfdetr_enabled:
        raise ValueError(
            "RF-DETR keypoint metadata cannot run with the YOLO pose SGIE"
        )
    if not pose_enabled and not rfdetr_enabled:
        logger.info("Pose SGIE disabled or missing; skipping pose feature hook")
        return

    env_flag = os.environ.get("NOESIS_POSE_FEATURES_ENABLED", "1")
    if str(env_flag).strip().lower() not in ("1", "true", "yes", "on"):
        if rfdetr_enabled:
            raise RuntimeError(
                "RF-DETR keypoint profile requires "
                "NOESIS_POSE_FEATURES_ENABLED=1"
            )
        logger.info("Pose features disabled (NOESIS_POSE_FEATURES_ENABLED=%s)", env_flag)
        return

    active_cfg = rfdetr_cfg if rfdetr_enabled else pose_cfg
    if rfdetr_enabled:
        tensor_source = str(
            active_cfg.get("tensor_source") or ""
        ).strip()
        if tensor_source != "rfdetr_pgie_frame":
            raise ValueError(
                "RF-DETR keypoint tensor_source must be rfdetr_pgie_frame"
            )
        gie_id = int(
            active_cfg.get("gie_id", active_cfg.get("gie-id", 0) or 0)
        )
        if gie_id != 1:
            raise ValueError("RF-DETR keypoint PGIE gie_id must be 1")
        component_name = str(
            active_cfg.get("attach_component") or ""
        ).strip()
        if component_name != "world_observation_stage":
            raise ValueError(
                "RF-DETR keypoint attach_component must be "
                "world_observation_stage"
            )
        model_label = "rfdetr-keypoint-preview-1.8.3"
    else:
        tensor_source = "object_sgie"
        gie_id = int(
            active_cfg.get("gie_id", active_cfg.get("gie-id", 4) or 4)
        )
        component_name = (
            str(active_cfg.get("name") or "yolo26_pose").strip()
            or "yolo26_pose"
        )
        model_label = "yolo26-pose"
    component = pipeline.components.get(component_name)
    if component is None:
        raise KeyError(
            f"pose hook component '{component_name}' missing in pipeline graph"
        )

    model_size = active_cfg.get("model_size") or active_cfg.get("input_size")
    model_w, model_h = 640, 640
    try:
        if isinstance(model_size, (list, tuple)) and len(model_size) >= 2:
            model_w = int(model_size[0])
            model_h = int(model_size[1])
        elif isinstance(model_size, str) and "x" in model_size:
            parts = model_size.lower().split("x")
            if len(parts) >= 2:
                model_w = int(parts[0].strip())
                model_h = int(parts[1].strip())
    except Exception:
        model_w, model_h = 640, 640
    if rfdetr_enabled and (model_w, model_h) != (576, 576):
        raise ValueError(
            "RF-DETR keypoint preview model_size must be [576, 576]"
        )

    score_threshold_raw = active_cfg.get("score_threshold")
    score_threshold = float(
        0.25 if score_threshold_raw is None else score_threshold_raw
    )
    kpt_threshold_raw = active_cfg.get("kpt_threshold")
    kpt_threshold = float(
        0.35 if kpt_threshold_raw is None else kpt_threshold_raw
    )
    if not math.isfinite(score_threshold) or not 0.0 <= score_threshold <= 1.0:
        raise ValueError("pose score_threshold must be finite and in [0,1]")
    if not math.isfinite(kpt_threshold) or not 0.0 <= kpt_threshold <= 1.0:
        raise ValueError("pose kpt_threshold must be finite and in [0,1]")
    if rfdetr_enabled and abs(score_threshold - 0.4) > 1e-9:
        raise ValueError(
            "RF-DETR keypoint score_threshold must match "
            "pre-cluster-threshold=0.4"
        )
    if rfdetr_enabled and abs(kpt_threshold - 0.35) > 1e-9:
        raise ValueError("RF-DETR keypoint kpt_threshold must be 0.35")
    letterbox = bool(active_cfg.get("letterbox", True))
    if rfdetr_enabled and letterbox:
        raise ValueError(
            "RF-DETR keypoint direct-square contract requires letterbox=false"
        )
    match_min_iou_raw = active_cfg.get("match_min_iou")
    match_min_iou = float(
        0.7 if match_min_iou_raw is None else match_min_iou_raw
    )
    match_ambiguity_margin_raw = active_cfg.get(
        "match_ambiguity_margin"
    )
    match_ambiguity_margin = float(
        0.05
        if match_ambiguity_margin_raw is None
        else match_ambiguity_margin_raw
    )
    if not 0.0 < match_min_iou <= 1.0:
        raise ValueError("RF-DETR keypoint match_min_iou must be in (0,1]")
    if not 0.0 <= match_ambiguity_margin < 1.0:
        raise ValueError(
            "RF-DETR keypoint match_ambiguity_margin must be in [0,1)"
        )
    cache_max_age_frames = 0 if rfdetr_enabled else 6
    cache_age_cfg = active_cfg.get(
        "pose_cache_max_age_frames",
        active_cfg.get("cache_max_age_frames"),
    )
    if cache_age_cfg is None:
        reinfer_interval = _read_nvinfer_property_int(
            active_cfg.get("config-file-path") or active_cfg.get("config-file"),
            "secondary-reinfer-interval",
        )
        if reinfer_interval is not None:
            cache_age_cfg = reinfer_interval
    try:
        if cache_age_cfg is not None:
            cache_max_age_frames = max(0, int(cache_age_cfg))
    except Exception:
        cache_max_age_frames = 6
    if rfdetr_enabled and cache_max_age_frames != 0:
        raise ValueError(
            "RF-DETR keypoint pose_cache_max_age_frames must be 0"
        )
    cache_max_bbox_shift = 0.35
    cache_shift_cfg = active_cfg.get(
        "pose_cache_max_bbox_shift",
        active_cfg.get("cache_max_bbox_shift"),
    )
    try:
        if cache_shift_cfg is not None:
            cache_max_bbox_shift = max(0.0, float(cache_shift_cfg))
    except Exception:
        cache_max_bbox_shift = 0.35

    processor = PoseFeatureProcessor(
        pipeline=pipeline,
        gie_id=gie_id,
        model_size=(model_w, model_h),
        score_threshold=score_threshold,
        kpt_threshold=kpt_threshold,
        letterbox=letterbox,
        tensor_source=tensor_source,
        model_label=model_label,
        match_min_iou=match_min_iou,
        match_ambiguity_margin=match_ambiguity_margin,
        camera_labels=camera_labels or {},
        cache_max_age_frames=cache_max_age_frames,
        cache_max_bbox_shift=cache_max_bbox_shift,
    )
    component.config["_pose_feature_processor"] = processor

    if pipeline.ds_pipeline is None or BatchMetadataOperator is None or Probe is None:
        logger.debug("Stored pose feature processor for lazy execution (pyservicemaker unavailable)")
        return

    try:
        probe = Probe("pose_features", _PoseFeatureOperator(processor))
        pipeline.ds_pipeline.attach(component.name, probe)
        logger.info(
            "Attached pose feature probe to %s (tensor_source=%s)",
            component.name,
            tensor_source,
        )
    except Exception as exc:  # pragma: no cover - depends on DS runtime availability
        logger.exception("Failed to attach pose feature probe")
        if rfdetr_enabled:
            raise RuntimeError(
                "RF-DETR keypoint pose feature probe attachment failed"
            ) from exc


def attach_object_depth_fusion_hook(
    pipeline: "DS8Pipeline",
    *,
    camera_labels: Optional[Mapping[int, str]] = None,
    calibration_resolver: Any | None = None,
    depth_every_n_frames: int = 2,
) -> None:
    """Attach the baseline DS8 object-depth fusion path.

    This is the canonical baseline-only DAv2 lane used to provide concurrent
    range observations for the pose-first world estimator.
    """
    if noesis_depth_meta_ext is None:
        raise RuntimeError("noesis_depth_meta_ext is required for baseline depth tracking")
    if noesis_depth_tracking_tensor_ext is None:
        raise RuntimeError("noesis_depth_tracking_tensor_ext is required for baseline depth tracking")

    models_cfg = (pipeline.config.get("models") or {}) if isinstance(pipeline.config, Mapping) else {}
    depth_cfg = models_cfg.get("depth_tracking") or {}
    if not isinstance(depth_cfg, Mapping) or not bool(depth_cfg.get("enable", False)):
        raise KeyError("models.depth_tracking.enable=true is required for baseline depth tracking")

    depth_name = str(depth_cfg.get("name") or "depth_tracking_fullframe").strip() or "depth_tracking_fullframe"
    depth_component = pipeline.components.get(depth_name)
    if depth_component is None:
        raise KeyError(f"depth tracking component '{depth_name}' missing in pipeline graph")
    fusion_component = pipeline.components.get("world_observation_stage")
    if fusion_component is None:
        raise KeyError("world_observation_stage component missing in pipeline graph")

    depth_store = _AlignedDepthFrameStore(max_entries=24)
    frame_size = getattr(pipeline, "frame_size", (0, 0))
    depth_processor = _DepthTrackingFrameProcessor(
        depth_store=depth_store,
        depth_gie_id=int(depth_cfg.get("gie_id", depth_cfg.get("gie-id", 5) or 5)),
        fallback_frame_size=(int(frame_size[0] or 0), int(frame_size[1] or 0)),
        depth_model_name="depth-anything-v2-metric-hypersim-vits",
        depth_unit="m",
        depth_is_metric=True,
    )
    fusion_processor = _ObjectDepthFusionProcessor(
        depth_store=depth_store,
        fallback_frame_size=(int(frame_size[0] or 0), int(frame_size[1] or 0)),
        depth_model_name="depth-anything-v2-metric-hypersim-vits",
        depth_unit="m",
        depth_is_metric=True,
        depth_every_n_frames=max(1, int(depth_every_n_frames)),
        calibration_resolver=calibration_resolver,
        camera_labels=camera_labels or {},
    )
    depth_component.config["_depth_tracking_frame_processor"] = depth_processor
    fusion_component.config["_object_depth_fusion_processor"] = fusion_processor
    setattr(pipeline, "depth_tracking_frame_processor", depth_processor)
    setattr(pipeline, "object_depth_fusion_processor", fusion_processor)

    if pipeline.ds_pipeline is None or BatchMetadataOperator is None or Probe is None:
        logger.debug("Stored depth tracking processors for lazy execution (pyservicemaker unavailable)")
        return

    try:
        pipeline.ds_pipeline.attach(depth_component.name, Probe("depth_tracking_capture", _DepthTrackingFrameOperator(depth_processor)))
        pipeline.ds_pipeline.attach(fusion_component.name, Probe("object_depth_fusion", _ObjectDepthFusionOperator(fusion_processor)))
        logger.info(
            "Attached baseline depth-tracking probes to %s and %s",
            depth_component.name,
            fusion_component.name,
        )
    except Exception:
        logger.exception("Failed to attach baseline depth-tracking probes")
        raise


def attach_analytics_telemetry_hook(
    pipeline: "DS8Pipeline",
    *,
    tracking_pub: "TrackingTelemetryPublisher",
    tracking_mode: Optional[str] = None,
    camera_labels: Optional[Mapping[int, str]] = None,
    sensor_id_map: Optional[Mapping[int, int]] = None,
    bev_renderer: Any | None = None,
    bev_calibration: Any | None = None,
    depth_registration: DepthRegistrationManager | None = None,
    world_fusion_policy: WorldFusionPolicy | None = None,
    legacy_world_fusion_policy: WorldFusionPolicy | None = None,
    scene_priors: ScenePriorSet | None = None,
    diagnostics_logger: "TrackingDiagnosticsLogger" | None = None,
    publication_gate: RuntimePublicationGate,
) -> None:
    """Attach a BatchMetadataOperator that extracts analytics telemetry."""
    if tracking_pub is None:
        raise ValueError("tracking_pub must be provided for analytics telemetry")
    if noesis_analytics_meta_ext is None or not callable(
        getattr(noesis_analytics_meta_ext, "extract_analytics", None)
    ):
        raise RuntimeError(
            "noesis_analytics_meta_ext.extract_analytics is required for DS9 analytics telemetry"
        )

    analytics_component = pipeline.components.get("analytics")
    if analytics_component is None:
        raise KeyError("analytics component missing; cannot attach telemetry hook")

    processor = _AnalyticsTelemetryProcessor(
        pipeline=pipeline,
        tracking_pub=tracking_pub,
        tracking_mode=tracking_mode,
        camera_labels=camera_labels or {},
        sensor_id_map=sensor_id_map or {},
        bev_renderer=bev_renderer,
        bev_calibration=bev_calibration,
        depth_registration=depth_registration,
        world_fusion_policy=world_fusion_policy,
        legacy_world_fusion_policy=legacy_world_fusion_policy,
        scene_priors=scene_priors,
        diagnostics_logger=diagnostics_logger,
        publication_gate=publication_gate,
    )
    analytics_component.config["_analytics_processor"] = processor
    setattr(pipeline, "analytics_telemetry_processor", processor)

    # OSD label stamping is handled inside the telemetry hook so per-source IDs remain intact
    # (tiler can collapse source_id in downstream metadata). This keeps mosaic labels aligned
    # with BEV + active-tracks stable IDs.
    try:
        osd_label_processor = _OsdLabelProcessor.from_pipeline_config(getattr(pipeline, "config", {}) or {})
        osd_label_processor.stable_id_mgr = getattr(pipeline, "stable_id_mgr", None)
        setattr(pipeline, "osd_label_processor", osd_label_processor)
        processor.osd_label_processor = osd_label_processor
    except Exception:
        logger.exception("Failed to initialize OSD label processor; mosaic labels may be missing")

    attach_identity_v2_post_resolution_osd_hook(
        pipeline,
        camera_labels=camera_labels or {},
        sensor_id_map=sensor_id_map or {},
    )

    attach_component = pipeline.components.get("tracking_telemetry_stage") or analytics_component

    if pipeline.ds_pipeline is None or BatchMetadataOperator is None or Probe is None:
        logger.debug("Stored analytics telemetry processor for lazy execution (pyservicemaker unavailable)")
        return

    try:
        probe = Probe("analytics_telemetry", _AnalyticsTelemetryOperator(processor))
        pipeline.ds_pipeline.attach(attach_component.name, probe)
        logger.info("Attached analytics telemetry probe to %s", attach_component.name)
    except Exception:  # pragma: no cover - depends on DS runtime availability
        logger.exception("Failed to attach analytics telemetry probe")


def attach_identity_v2_post_resolution_osd_hook(
    pipeline: "DS8Pipeline",
    *,
    camera_labels: Mapping[int, str],
    sensor_id_map: Mapping[int, int],
) -> None:
    """Stamp authoritative v2 labels on fresh metadata at the tiler sink."""

    service = getattr(pipeline, "identity_v2_service", None)
    if service is None or not bool(getattr(service, "authoritative", False)):
        return
    tiler = pipeline.components.get("tiler")
    if tiler is None:
        raise KeyError(
            "tiler component missing; authoritative identity OSD cannot be attached"
        )
    label_processor = getattr(pipeline, "osd_label_processor", None)
    decimals = int(getattr(label_processor, "decimals", 2) or 0)
    processor = IdentityV2PostResolutionOsdProcessor(
        pipeline=pipeline,
        camera_labels=dict(camera_labels),
        sensor_id_map=dict(sensor_id_map),
        decimals=decimals,
    )
    tiler.config["_identity_v2_post_resolution_osd_processor"] = processor
    setattr(pipeline, "identity_v2_post_resolution_osd_processor", processor)
    if pipeline.ds_pipeline is None or BatchMetadataOperator is None or Probe is None:
        logger.debug(
            "Stored Identity v2 post-resolution OSD processor for lazy execution"
        )
        return
    try:
        probe = Probe(
            "identity_v2_post_resolution_osd",
            _IdentityV2PostResolutionOsdOperator(processor),
        )
        pipeline.ds_pipeline.attach(tiler.name, probe, tips="sink")
        logger.info(
            "Attached authoritative Identity v2 OSD probe to %s sink", tiler.name
        )
    except Exception:
        logger.exception("Failed to attach authoritative Identity v2 OSD probe")
        raise


def attach_trail_overlay_hook(
    pipeline: "DS8Pipeline",
    *,
    config: Mapping[str, Any] | None = None,
) -> None:
    """Attach a DS8 trail overlay hook for mosaic trails.

    This renders per-person motion trails (time-window history) into the mosaic
    stream using NvDsDisplayMeta line primitives.
    """
    osd_component = pipeline.components.get("osd")
    if osd_component is None:
        raise KeyError("osd component missing; cannot attach trail overlay hook")

    # IMPORTANT: The DisplayMeta must be appended *before* nvdsosd runs; attaching
    # the probe to `osd` can be too late depending on probe placement semantics.
    # Attaching at `tiler` ensures the overlay metadata is present when nvdsosd
    # renders the mosaic.
    attach_component = pipeline.components.get("tiler") or osd_component

    trails_cfg: Mapping[str, Any] = {}
    if config is not None:
        trails_cfg = config
    else:
        vis_cfg = pipeline.config.get("visualization") or {}
        if isinstance(vis_cfg, Mapping):
            raw_trails = vis_cfg.get("trails") or {}
            if isinstance(raw_trails, Mapping):
                trails_cfg = raw_trails

    settings = TrailOverlayConfig.from_mapping(trails_cfg)
    processor = TrailOverlayProcessor(pipeline=pipeline, config=settings)
    attach_component.config["_trail_overlay_processor"] = processor
    setattr(pipeline, "trail_overlay_processor", processor)

    if pipeline.ds_pipeline is None or BatchMetadataOperator is None or Probe is None:
        logger.debug("Stored trail overlay processor for lazy execution (pyservicemaker unavailable)")
        return

    try:
        probe = Probe("trail_overlay", _TrailOverlayOperator(processor))
        pipeline.ds_pipeline.attach(attach_component.name, probe)
        logger.info("Attached trail overlay probe to %s", attach_component.name)
    except Exception:  # pragma: no cover - depends on DS runtime availability
        logger.exception("Failed to attach trail overlay probe")


def attach_v3dt_cuboid_overlay_hook(
    pipeline: "DS8Pipeline", *, tracking_mode: str
) -> None:
    """Replace nvtracker's debug projection with a person-base anchored cuboid."""
    vis_cfg = pipeline.config.get("visualization") or {}
    raw_cfg = vis_cfg.get("v3dt_cuboid") if isinstance(vis_cfg, Mapping) else None
    settings = V3DTCuboidOverlayConfig.from_mapping(
        raw_cfg if isinstance(raw_cfg, Mapping) else {}
    )
    if not settings.enabled:
        logger.info("V3DT cuboid correction disabled")
        return

    mode = str(tracking_mode or "").strip().lower()
    if mode not in ("v3dt", "sv3dt", "mv3dt"):
        raise ValueError(
            "visualization.v3dt_cuboid is only valid for a V3DT tracking mode"
        )
    attach_component = pipeline.components.get("tracking_telemetry_stage")
    if attach_component is None:
        raise KeyError(
            "tracking_telemetry_stage missing; cannot attach V3DT cuboid correction"
        )
    if pipeline.ds_pipeline is None or BufferOperator is None or Probe is None:
        logger.debug("Stored V3DT cuboid correction for lazy execution")
        return
    if noesis_v3dt_meta_ext is None:
        raise RuntimeError("V3DT cuboid correction requires noesis_v3dt_meta_ext")
    scrub = getattr(
        noesis_v3dt_meta_ext, "scrub_tracker_projection_display_meta", None
    )
    extract_base = getattr(noesis_v3dt_meta_ext, "extract_person_base", None)
    if not callable(scrub) or not callable(extract_base):
        raise RuntimeError(
            "noesis_v3dt_meta_ext lacks the V3DT cuboid correction API"
        )

    processor = V3DTCuboidOverlayProcessor(pipeline=pipeline, config=settings)
    attach_component.config["_v3dt_cuboid_overlay_processor"] = processor
    setattr(pipeline, "v3dt_cuboid_overlay_processor", processor)
    probe = Probe("v3dt_cuboid_correction", _V3DTCuboidBufferOperator(processor))
    pipeline.ds_pipeline.attach(attach_component.name, probe)
    logger.info(
        "Attached V3DT person-base cuboid correction to %s",
        attach_component.name,
    )


def attach_pose_keypoint_overlay_hook(pipeline: "DS8Pipeline") -> None:
    """Attach a DS8 pose keypoint overlay hook (draws skeletons on the mosaic)."""
    vis_cfg = pipeline.config.get("visualization") or {}
    enabled = False
    if isinstance(vis_cfg, Mapping):
        enabled = bool(vis_cfg.get("display_keypoints", False))
    if not enabled:
        logger.info("Pose keypoint overlay disabled (visualization.display_keypoints=false)")
        return

    models_cfg = (pipeline.config.get("models") or {})
    if not isinstance(models_cfg, Mapping):
        models_cfg = {}
    pose_cfg = models_cfg.get("pose") or {}
    if not isinstance(pose_cfg, Mapping):
        pose_cfg = {}
    pose_enabled = bool(pose_cfg.get("enable", True)) and any(
        key in pose_cfg for key in ("config-file-path", "engine", "name")
    )
    rfdetr_cfg = models_cfg.get("rfdetr_keypoint") or {}
    if not isinstance(rfdetr_cfg, Mapping):
        rfdetr_cfg = {}
    rfdetr_enabled = bool(rfdetr_cfg.get("enable", False))
    if pose_enabled and rfdetr_enabled:
        raise ValueError(
            "RF-DETR keypoint overlay cannot run with the YOLO pose SGIE"
        )
    if not pose_enabled and not rfdetr_enabled:
        logger.info("Pose SGIE disabled or missing; skipping pose keypoint overlay")
        return
    active_cfg = rfdetr_cfg if rfdetr_enabled else pose_cfg

    osd_component = pipeline.components.get("osd")
    if osd_component is None:
        raise KeyError("osd component missing; cannot attach pose keypoint overlay")

    attach_component = pipeline.components.get("tiler") or osd_component

    gie_id = int(
        active_cfg.get(
            "gie_id",
            active_cfg.get("gie-id", 1 if rfdetr_enabled else 4),
        )
        or (1 if rfdetr_enabled else 4)
    )
    model_size = active_cfg.get("model_size") or active_cfg.get("input_size")
    model_w, model_h = 640, 640
    try:
        if isinstance(model_size, (list, tuple)) and len(model_size) >= 2:
            model_w = int(model_size[0])
            model_h = int(model_size[1])
        elif isinstance(model_size, str) and "x" in model_size:
            parts = model_size.lower().split("x")
            if len(parts) >= 2:
                model_w = int(parts[0].strip())
                model_h = int(parts[1].strip())
    except Exception:
        model_w, model_h = 640, 640

    score_threshold = float(
        active_cfg.get("score_threshold", 0.25) or 0.25
    )
    kpt_threshold = float(
        active_cfg.get("kpt_threshold", 0.35) or 0.35
    )
    letterbox = bool(active_cfg.get("letterbox", True))

    processor = PoseKeypointOverlayProcessor(
        pipeline=pipeline,
        gie_id=gie_id,
        model_size=(model_w, model_h),
        score_threshold=score_threshold,
        kpt_threshold=kpt_threshold,
        letterbox=letterbox,
    )
    attach_component.config["_pose_keypoint_overlay_processor"] = processor
    setattr(pipeline, "pose_keypoint_overlay_processor", processor)

    if pipeline.ds_pipeline is None or BatchMetadataOperator is None or Probe is None:
        logger.debug("Stored pose keypoint overlay processor for lazy execution (pyservicemaker unavailable)")
        return

    try:
        probe = Probe("pose_keypoints", _PoseKeypointOverlayOperator(processor))
        pipeline.ds_pipeline.attach(attach_component.name, probe)
        logger.info("Attached pose keypoint overlay probe to %s", attach_component.name)
    except Exception:  # pragma: no cover - depends on DS runtime availability
        logger.exception("Failed to attach pose keypoint overlay probe")


def attach_osd_label_hook(pipeline: "DS8Pipeline") -> None:
    """Attach a DS8 hook that stamps object labels with detection confidence."""
    osd_component = pipeline.components.get("osd")
    if osd_component is None:
        raise KeyError("osd component missing; cannot attach OSD label hook")

    # Attach upstream of nvdsosd so the text is rendered in the mosaic overlay.
    attach_component = pipeline.components.get("tiler") or osd_component
    processor = _OsdLabelProcessor.from_pipeline_config(getattr(pipeline, "config", {}) or {})
    processor.stable_id_mgr = getattr(pipeline, "stable_id_mgr", None)
    attach_component.config["_osd_label_processor"] = processor

    if pipeline.ds_pipeline is None or BatchMetadataOperator is None or Probe is None:
        logger.debug("Stored OSD label processor for lazy execution (pyservicemaker unavailable)")
        return

    try:
        probe = Probe("osd_labels", _OsdLabelOperator(processor))
        pipeline.ds_pipeline.attach(attach_component.name, probe)
        logger.info("Attached OSD label probe to %s", attach_component.name)
    except Exception:  # pragma: no cover - depends on DS runtime availability
        logger.exception("Failed to attach OSD label probe")


class AnalyticsReloadStateAmbiguous(RuntimeError):
    """Raised when a native reload may be active but cannot be fully acknowledged."""

    native_state_ambiguous = True


def attach_analytics_reload_bridge(
    pipeline: "DS8Pipeline",
    *,
    stage_name: str = "exclude",
) -> None:
    """Register a live analytics reload bridge for REST-triggered ROI updates."""
    from noesis.server import analytics_api  # imported lazily to avoid cycles

    component = pipeline.components.get("analytics")
    if component is None:
        raise KeyError("analytics component missing; cannot attach reload bridge")

    lock = threading.Lock()
    pipeline.analytics_reload_count = getattr(pipeline, "analytics_reload_count", 0)
    exclude_component = pipeline.components.get("analytics_exclude")
    if exclude_component is None:
        raise KeyError("analytics_exclude component missing; cannot attach reload bridge")
    if pipeline.ds_pipeline is not None and type(pipeline.ds_pipeline).__name__ != "_NoopDSPipeline":
        node = pipeline.ds_pipeline[exclude_component.name]
        expected_types = {
            "config-file": str,
            "reload-request-sequence": int,
            "reload-accepted-sequence": int,
            "reload-failed-sequence": int,
            "last-reload-ok": bool,
            "expected-config-sha256": str,
            "active-config-sha256": str,
            "reload-error-count": int,
            "objects-removed-count": int,
            "last-reload-error": str,
        }
        for property_name, expected_type in expected_types.items():
            value = node.get(property_name)
            if not isinstance(value, expected_type):
                raise RuntimeError(
                    f"Native analytics reload property {property_name!r} is unavailable or has "
                    f"the wrong type: {type(value).__name__}"
                )

    def _apply(
        stage: str,
        cfg: Dict[str, Any],
        reload_context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del cfg
        if stage != stage_name:
            raise RuntimeError(f"Native analytics reload does not support stage {stage!r}")
        exclude_component = pipeline.components.get("analytics_exclude")
        if exclude_component is None or pipeline.ds_pipeline is None:
            raise RuntimeError("Native analytics exclusion node is unavailable")
        config_path = Path(str(reload_context.get("config_path") or ""))
        expected_sha256 = str(reload_context.get("config_sha256") or "")
        component_path = Path(str(exclude_component.config.get("config-file") or "")).absolute()
        if component_path != config_path:
            raise RuntimeError(
                "Analytics exclusion component path does not match the transaction path "
                f"({component_path} != {config_path})"
            )

        with lock:
            node = pipeline.ds_pipeline[exclude_component.name]
            active_config_path = Path(str(node.get("config-file") or "")).absolute()
            if active_config_path != config_path:
                raise RuntimeError(
                    "Native analytics exclusion path does not match the writable state path "
                    f"({active_config_path} != {config_path})"
                )
            request_sequence = int(node.get("reload-request-sequence"))
            accepted_sequence = int(node.get("reload-accepted-sequence"))
            failed_sequence = int(node.get("reload-failed-sequence"))
            next_sequence = max(request_sequence, accepted_sequence, failed_sequence) + 1
            if next_sequence > 0xFFFFFFFF:
                raise RuntimeError("Analytics reload sequence is exhausted")
            error_count_before = int(node.get("reload-error-count"))
            node.set({"expected-config-sha256": expected_sha256})
            try:
                node.set({"reload-request-sequence": next_sequence})
                observed_request = int(node.get("reload-request-sequence"))
                observed_accepted = int(node.get("reload-accepted-sequence"))
                observed_failed = int(node.get("reload-failed-sequence"))
                last_reload_ok = bool(node.get("last-reload-ok"))
                active_sha256 = str(node.get("active-config-sha256") or "")
                error_count_after = int(node.get("reload-error-count"))
                objects_removed_count = int(node.get("objects-removed-count"))
                last_error = str(node.get("last-reload-error") or "")[:512]
            except Exception as exc:
                raise AnalyticsReloadStateAmbiguous(
                    "Native analytics reload receipt could not be read after dispatch"
                ) from exc
            native_candidate_active = (
                observed_request == next_sequence
                and observed_accepted == next_sequence
                and active_sha256 == expected_sha256
                and last_reload_ok
            )
            if not (
                native_candidate_active
                and observed_failed != next_sequence
                and error_count_after == error_count_before
                and last_error == ""
            ):
                error_type = (
                    AnalyticsReloadStateAmbiguous
                    if native_candidate_active
                    else RuntimeError
                )
                raise error_type(
                    "Native analytics reload was rejected or unacknowledged "
                    f"(request={observed_request}, accepted={observed_accepted}, "
                    f"failed={observed_failed}, active_sha256={active_sha256!r}, "
                    f"error_count={error_count_after}, error={last_error!r})"
                )
            receipt = {
                "request_sequence": observed_request,
                "accepted_sequence": observed_accepted,
                "failed_sequence": observed_failed,
                "active_config_sha256": active_sha256,
                "reload_error_count": error_count_after,
                "objects_removed_count": objects_removed_count,
            }
            try:
                pipeline.analytics_reload_count = (
                    getattr(pipeline, "analytics_reload_count", 0) + 1
                )
                pipeline.analytics_reload_receipt = receipt
            except Exception as exc:
                raise AnalyticsReloadStateAmbiguous(
                    "Native analytics reload committed but hook publication failed"
                ) from exc
            return receipt

    analytics_api.register_reload_hook(_apply)
    logger.info("Registered analytics reload bridge for stage '%s'", stage_name)


def verify_analytics_exclusion_initial_receipt(
    pipeline: "DS8Pipeline",
    reload_context: Mapping[str, Any],
) -> Dict[str, Any]:
    """Require the native exclusion element to acknowledge the prepared startup INI."""
    component = pipeline.components.get("analytics_exclude")
    if component is None or pipeline.ds_pipeline is None:
        raise RuntimeError("Native analytics exclusion node is unavailable")
    expected_path = Path(str(reload_context.get("config_path") or ""))
    expected_sha256 = str(reload_context.get("config_sha256") or "")
    node = pipeline.ds_pipeline[component.name]
    active_path = Path(str(node.get("config-file") or "")).absolute()
    receipt = {
        "request_sequence": int(node.get("reload-request-sequence")),
        "accepted_sequence": int(node.get("reload-accepted-sequence")),
        "failed_sequence": int(node.get("reload-failed-sequence")),
        "active_config_sha256": str(node.get("active-config-sha256") or ""),
        "reload_error_count": int(node.get("reload-error-count")),
        "objects_removed_count": int(node.get("objects-removed-count")),
        "last_reload_ok": bool(node.get("last-reload-ok")),
        "last_reload_error": str(node.get("last-reload-error") or "")[:512],
    }
    if not (
        active_path == expected_path
        and receipt["active_config_sha256"] == expected_sha256
        and receipt["reload_error_count"] == 0
        and receipt["last_reload_ok"]
        and receipt["last_reload_error"] == ""
    ):
        raise RuntimeError(f"Native analytics initial receipt mismatch: {receipt}")
    pipeline.analytics_initial_receipt = receipt
    return receipt


def _resolve_intrinsics_loader(config_path: str | Path | None) -> intrinsics_module.CameraConfigLoader:
    if config_path is not None:
        path = Path(config_path)
    else:
        path = intrinsics_module._DEFAULT_CONFIG_PATH
    return intrinsics_module.CameraConfigLoader(path)


def _normalise_dims(dims: Sequence[int]) -> Tuple[int, ...]:
    normalised = []
    for value in dims:
        val = int(value)
        if val <= 0:
            val = 1
        normalised.append(val)
    return tuple(normalised)


def _flatten_layer_dims(layer: Any) -> Tuple[int, ...]:
    dims = getattr(layer, "dims", None) or getattr(layer, "inferDims", None)
    if dims is None:
        raise AttributeError("Layer is missing dims information")
    values: Iterable[int]
    if hasattr(dims, "d"):
        values = [dims.d[i] for i in range(getattr(dims, "numDims", 0))]
    else:
        values = list(dims)
    return _normalise_dims(values)


def _layer_dtype(layer: Any) -> np.dtype:
    data_type = getattr(layer, "dataType", getattr(layer, "data_type", None))
    if isinstance(data_type, str):
        key = data_type.lower()
        if key in ("float", "float32", "fp32"):
            return np.float32
        if key in ("half", "float16", "fp16"):
            return np.float16
        if key in ("int32", "sint32"):
            return np.int32
        if key in ("uint8", "uchar", "uchar8"):
            return np.uint8
    if pyds is not None and _allow_raw_pyds_compat():
        try:
            from pyds import NvDsInferDataType  # type: ignore

            mapping = {
                NvDsInferDataType.NVDSINFER_TENSOR_FLOAT32: np.float32,
                NvDsInferDataType.NVDSINFER_TENSOR_FLOAT16: np.float16,
                NvDsInferDataType.NVDSINFER_TENSOR_INT32: np.int32,
                NvDsInferDataType.NVDSINFER_TENSOR_INT8: np.int8,
                NvDsInferDataType.NVDSINFER_TENSOR_UINT8: np.uint8,
            }
            return mapping.get(data_type, np.float32)
        except Exception:  # pragma: no cover - defensive
            pass
    elif pyds is not None:
        _record_quarantined_compat_path("_layer_dtype.pyds", _PYDS_COMPAT_ENV)
    return np.float32


def _numpy_from_layer(layer: Any) -> np.ndarray:
    shape = _flatten_layer_dims(layer)
    dtype = _layer_dtype(layer)
    numel = math.prod(shape) if shape else 0
    if numel <= 0:
        return np.empty(0, dtype=dtype)

    buffer_obj = getattr(layer, "buffer", None)
    if buffer_obj is None:
        raise AttributeError("Layer has no buffer pointer")

    if pyds is not None and _allow_raw_pyds_compat():
        try:  # pragma: no cover - requires DeepStream runtime
            import ctypes

            ptr_val = pyds.get_ptr(buffer_obj)  # type: ignore[attr-defined]
            ctype_map = {
                np.float32: ctypes.c_float,
                np.float16: ctypes.c_float,
                np.int32: ctypes.c_int32,
                np.int8: ctypes.c_int8,
                np.uint8: ctypes.c_uint8,
            }
            ctype = ctype_map.get(dtype, ctypes.c_float)
            ptr = ctypes.cast(ptr_val, ctypes.POINTER(ctype))
            flat = np.ctypeslib.as_array(ptr, shape=(numel,))  # type: ignore[arg-type]
            return np.array(flat, dtype=np.float32 if dtype == np.float16 else dtype, copy=True).reshape(shape)
        except Exception:
            logger.exception("Failed to map tensor layer buffer; falling back to numpy conversion")
    elif pyds is not None:
        _record_quarantined_compat_path("_numpy_from_layer.pyds_get_ptr", _PYDS_COMPAT_ENV)

    if isinstance(buffer_obj, np.ndarray):
        return np.array(buffer_obj, dtype=dtype, copy=True).reshape(shape)

    if isinstance(buffer_obj, (bytes, bytearray)):
        return np.frombuffer(buffer_obj, dtype=dtype).copy().reshape(shape)

    raise TypeError(f"Unsupported tensor buffer type: {type(buffer_obj)!r}")


def _classify_layer_name(name: str) -> str:
    lowered = name.lower()
    if "depth" in lowered or "disp" in lowered:
        return "depth"
    if "conf" in lowered:
        return "confidence"
    if "mask" in lowered or "valid" in lowered:
        return "mask"
    return name


def _extract_tensor_layers(tensor_meta: Any) -> Dict[str, np.ndarray]:
    tensors: Dict[str, np.ndarray] = {}
    count = (
        getattr(tensor_meta, "num_layers", None)
        or getattr(tensor_meta, "num_output_layers", None)
        or getattr(tensor_meta, "num_out_layers", None)
        or 0
    )

    layers_seq: list[Any]
    if hasattr(tensor_meta, "output_layers_info"):
        try:
            layers_seq = list(tensor_meta.output_layers_info)
        except Exception:  # pragma: no cover - attribute access may fail
            layers_seq = []
    else:
        layers_seq = []

    if count and not layers_seq and pyds is not None and _allow_raw_pyds_compat():
        try:  # pragma: no cover - requires DeepStream runtime
            layers_seq = [pyds.get_nvds_LayerInfo(tensor_meta, i) for i in range(count)]  # type: ignore[attr-defined]
        except Exception:
            layers_seq = []
    elif count and not layers_seq and pyds is not None:
        _record_quarantined_compat_path("_extract_tensor_layers.get_nvds_LayerInfo", _PYDS_COMPAT_ENV)

    if not count and layers_seq:
        count = len(layers_seq)

    if not layers_seq:
        return tensors

    for idx, layer in enumerate(layers_seq):
        try:
            array = _numpy_from_layer(layer)
        except Exception:
            logger.exception("Failed to decode tensor layer index %s", idx)
            continue
        raw_name = (
            getattr(layer, "layerName", None)
            or getattr(layer, "name", None)
            or f"layer_{idx}"
        )
        tensors[_classify_layer_name(str(raw_name))] = array
    return tensors


@dataclass
class MapAnythingProcessor:
    pipeline: "DS8Pipeline"
    storage: DepthStorageManager
    depth_pub: "DepthTelemetryPublisher" | None
    gie_id: int
    batch_size: int = 3
    profile_name: str = _MAPANYTHING_CANONICAL_PROFILE.name
    output_height: int = _MAPANYTHING_OUTPUT_HEIGHT
    output_width: int = _MAPANYTHING_OUTPUT_WIDTH
    depth_backend: str = "mapanything"
    metric_focal_denominator: float = _DA3METRIC_LARGE_FOCAL_DENOMINATOR
    camera_labels: Mapping[int, str] = field(default_factory=dict)
    rgb_provider: PipelineRgbFrameProvider | None = field(default=None, repr=False)
    rgb_width: int = 0
    rgb_height: int = 0
    tensor_samples: int = 0
    native_capture: Optional[Callable[..., Any]] = field(default=None, repr=False)
    failure_callback: Optional[Callable[[BaseException], None]] = field(
        default=None, repr=False
    )
    _async_enabled: bool = field(default=True, init=False, repr=False)
    _async_queue: "queue.Queue[Any]" = field(
        default_factory=lambda: queue.Queue(maxsize=32), init=False, repr=False
    )
    _async_thread: Optional[threading.Thread] = field(
        default=None, init=False, repr=False
    )
    _async_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _async_idle: threading.Condition = field(init=False, repr=False)
    _async_failure: Optional[str] = field(default=None, init=False, repr=False)
    _async_accepting: bool = field(default=True, init=False, repr=False)
    _async_active_captures: int = field(default=0, init=False, repr=False)
    _async_stop_enqueued: bool = field(default=False, init=False, repr=False)
    _async_stopped: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )
    _async_shutdown_complete: bool = field(default=False, init=False, repr=False)
    _dewarper_fov_masks: Dict[Tuple[int, int, int], np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _storage_commit_timeout_s: float = field(default=30.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if int(self.gie_id) <= 0:
            raise ValueError("MapAnything gie_id must be positive")
        if int(self.batch_size) <= 0:
            raise ValueError("MapAnything batch_size must be positive")
        if int(self.output_height) <= 0 or int(self.output_width) <= 0:
            raise ValueError("MapAnything output dimensions must be positive")
        self.depth_backend = str(self.depth_backend or "mapanything").strip().lower()
        if self.depth_backend not in _MANUAL_DEPTH_BACKENDS:
            raise ValueError(
                "manual depth backend must be mapanything or da3metric-large"
            )
        self.metric_focal_denominator = float(self.metric_focal_denominator)
        if (
            not math.isfinite(self.metric_focal_denominator)
            or self.metric_focal_denominator <= 0.0
        ):
            raise ValueError("metric focal denominator must be finite and positive")
        if (
            self.rgb_provider is not None
            and (int(self.rgb_width) <= 0 or int(self.rgb_height) <= 0)
        ):
            frame_size = getattr(self.pipeline, "frame_size", (0, 0))
            try:
                self.rgb_width = int(frame_size[0])
                self.rgb_height = int(frame_size[1])
            except Exception as exc:
                raise ValueError(
                    "MapAnything exact RGB dimensions must be positive"
                ) from exc
        if (
            self.rgb_provider is not None
            and (int(self.rgb_width) <= 0 or int(self.rgb_height) <= 0)
        ):
            raise ValueError("MapAnything exact RGB dimensions must be positive")
        self._async_idle = threading.Condition(self._async_lock)
        self._storage_commit_timeout_s = resolve_depth_store_commit_timeout_s()

    @property
    def host_payload_bytes(self) -> int:
        return (
            len(_MAPANYTHING_EXACT_LAYERS)
            * int(self.output_height)
            * int(self.output_width)
            * np.dtype(np.float32).itemsize
        )

    def _dewarper_validity_mask(self, source_id: int, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
        target_w, target_h = int(target_size[0]), int(target_size[1])
        key = (int(source_id), target_w, target_h)
        if key in self._dewarper_fov_masks:
            return self._dewarper_fov_masks[key]
        config = getattr(self.pipeline, "config", {}) or {}
        masks = config.get("dewarper_validity_masks") if isinstance(config, Mapping) else None
        if not isinstance(masks, Mapping):
            return None
        sources = masks.get("sources") if isinstance(masks.get("sources"), Mapping) else masks
        mask_cfg = sources.get(str(int(source_id)), sources.get(int(source_id))) if isinstance(sources, Mapping) else None
        if not isinstance(mask_cfg, Mapping):
            return None
        source_configs = config.get("sources") if isinstance(config, Mapping) else None
        if (
            not isinstance(source_configs, Sequence)
            or int(source_id) < 0
            or int(source_id) >= len(source_configs)
        ):
            raise RuntimeError(f"dewarper validity source {source_id} is absent from pipeline sources")
        source_cfg = source_configs[int(source_id)]
        if not isinstance(source_cfg, Mapping):
            raise RuntimeError(f"pipeline source {source_id} is not a mapping")
        spec = _load_dewarper_fov_spec(
            source_cfg=source_cfg,
            pipeline_yaml_path=getattr(self.pipeline, "yaml_path", None),
            mask_cfg=mask_cfg,
            repo_root=Path(__file__).resolve().parents[3],
        )
        if spec is None:
            return None
        mask = _build_dewarper_fov_mask(
            spec,
            target_size=(target_w, target_h),
            erode_px=int(mask_cfg.get("erode-px", mask_cfg.get("erode_px", 1)) or 0),
        )
        self._dewarper_fov_masks[key] = mask
        return mask

    def _target_frame_shape(self, frame_meta: Any, depth_shape: Tuple[int, int]) -> Tuple[int, int]:
        frame_w = 0
        frame_h = 0
        try:
            frame_w = int(_meta_lookup(frame_meta, "frame_width", "width", "source_frame_width", default=0) or 0)
            frame_h = int(_meta_lookup(frame_meta, "frame_height", "height", "source_frame_height", default=0) or 0)
        except Exception:
            frame_w = 0
            frame_h = 0
        if frame_w <= 0 or frame_h <= 0:
            try:
                frame_w, frame_h = getattr(self.pipeline, "frame_size", (0, 0))
            except Exception:
                frame_w, frame_h = 0, 0
        if frame_w <= 0 or frame_h <= 0:
            try:
                frame_h, frame_w = int(depth_shape[0]), int(depth_shape[1])
            except Exception:
                frame_w = 0
                frame_h = 0
        return max(0, int(frame_w or 0)), max(0, int(frame_h or 0))

    def _da3_metric_scale(
        self,
        *,
        source_id: int,
        camera_id: str,
        target_size: Tuple[int, int],
        model_size: Tuple[int, int],
    ) -> Tuple[float, float]:
        provider = getattr(self.pipeline, "bev_calibration", None)
        snapshot_fn = getattr(provider, "snapshot", None)
        if not callable(snapshot_fn):
            raise RuntimeError(
                "DA3Metric-Large requires the DS9 calibration provider for "
                "camera focal scaling"
            )
        snapshot = snapshot_fn(int(source_id), str(camera_id))
        if snapshot is None:
            raise RuntimeError(
                "DA3Metric-Large calibration is unavailable for "
                f"source_id={source_id} camera_id={camera_id}"
            )
        intrinsics = np.asarray(
            getattr(snapshot, "intrinsics", None), dtype=np.float64
        )
        if intrinsics.shape != (3, 3):
            raise RuntimeError(
                "DA3Metric-Large calibration intrinsics must be a 3x3 matrix"
            )
        target_w, target_h = (int(target_size[0]), int(target_size[1]))
        model_w, model_h = (int(model_size[0]), int(model_size[1]))
        try:
            calibration_w, calibration_h = (
                int(value) for value in snapshot.image_size
            )
        except Exception as exc:
            raise RuntimeError(
                "DA3Metric-Large calibration image_size must be width,height"
            ) from exc
        if min(target_w, target_h, model_w, model_h, calibration_w, calibration_h) <= 0:
            raise RuntimeError(
                "DA3Metric-Large focal scaling requires positive calibration, "
                "frame, and model dimensions"
            )
        fx_target = float(intrinsics[0, 0]) * target_w / calibration_w
        fy_target = float(intrinsics[1, 1]) * target_h / calibration_h
        resize_scale = min(model_w / target_w, model_h / target_h)
        model_input_focal_px = 0.5 * (fx_target + fy_target) * resize_scale
        if not math.isfinite(model_input_focal_px) or model_input_focal_px <= 0.0:
            raise RuntimeError(
                "DA3Metric-Large resolved model-input focal length is invalid"
            )
        return (
            model_input_focal_px / self.metric_focal_denominator,
            model_input_focal_px,
        )

    def _align_to_frame(
        self,
        depth: np.ndarray,
        conf: Optional[np.ndarray],
        mask: Optional[np.ndarray],
        target_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        target_w, target_h = target_size
        model_h, model_w = depth.shape[:2]
        if target_w <= 0 or target_h <= 0 or (model_w == target_w and model_h == target_h):
            return depth, conf, mask
        if model_w <= 0 or model_h <= 0:
            return depth, conf, mask

        scale_w = model_w / float(target_w) if target_w else 0.0
        scale_h = model_h / float(target_h) if target_h else 0.0
        scale = min(scale_w, scale_h)
        if scale <= 0.0:
            return depth, conf, mask

        resized_w = max(1, min(model_w, int(round(float(target_w) * scale))))
        resized_h = max(1, min(model_h, int(round(float(target_h) * scale))))
        pad_left = max(0, int(math.floor((model_w - resized_w) * 0.5)))
        pad_top = max(0, int(math.floor((model_h - resized_h) * 0.5)))
        pad_right = max(0, model_w - resized_w - pad_left)
        pad_bottom = max(0, model_h - resized_h - pad_top)
        x0 = pad_left
        x1 = model_w - pad_right
        y0 = pad_top
        y1 = model_h - pad_bottom
        try:
            depth_cropped = depth[y0:y1, x0:x1]
            conf_cropped = conf[y0:y1, x0:x1] if conf is not None else None
            mask_cropped = mask[y0:y1, x0:x1] if mask is not None else None
        except Exception:
            return depth, conf, mask

        if depth_cropped.size == 0 or depth_cropped.shape[0] <= 0 or depth_cropped.shape[1] <= 0:
            return depth, conf, mask

        depth_resized = cv2.resize(depth_cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        conf_resized = (
            cv2.resize(conf_cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            if conf_cropped is not None
            else None
        )
        mask_resized = None
        if mask_cropped is not None:
            mask_resized = cv2.resize(
                mask_cropped.astype(np.uint8, copy=False),
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        return depth_resized, conf_resized, mask_resized

    def handle_numpy_arrays(
        self,
        *,
        source_id: int,
        frame_id: int,
        pts_ns: int,
        tensors: Mapping[str, np.ndarray],
        captured_while_enabled: bool = False,
        captured_at_us: int | None = None,
    ) -> Optional[DepthResult]:
        frame_meta = {
            "pad_index": source_id,
            "frame_num": frame_id,
            "buf_pts": pts_ns,
            "captured_at_us": captured_at_us,
        }
        self.tensor_samples += 1
        return self._emit_from_tensors(
            frame_meta,
            tensors,
            captured_while_enabled=bool(captured_while_enabled),
        )

    def _raise_async_failure(self) -> None:
        with self._async_lock:
            failure = self._async_failure
        if failure is not None:
            raise RuntimeError(
                f"DS9 MapAnything async postprocess is poisoned: {failure}"
            )

    def _async_unfinished_task_count(self) -> int:
        with self._async_queue.mutex:
            return int(self._async_queue.unfinished_tasks)

    def _poison_async_worker(self, exc: BaseException) -> None:
        failure = f"{type(exc).__name__}: {exc}"
        with self._async_idle:
            first_failure = self._async_failure is None
            if first_failure:
                self._async_failure = failure
            self._async_idle.notify_all()
        if first_failure:
            _increment_core_counter("mapanything_async_postprocess_failures_total")
            logger.exception("DS9 MapAnything async postprocess failed")
            callback = self.failure_callback
            if callable(callback):
                try:
                    callback(exc)
                except Exception:
                    logger.exception("DS9 MapAnything runtime failure callback failed")

    def report_capture_failure(self, exc: BaseException) -> None:
        """Poison the worker without allowing probe failures across pybind."""

        self._poison_async_worker(exc)

    def _start_async_worker(self) -> None:
        with self._async_idle:
            if self._async_failure is not None:
                raise RuntimeError(
                    "DS9 MapAnything async postprocess is poisoned: "
                    f"{self._async_failure}"
                )
            if not self._async_accepting and self._async_active_captures <= 0:
                raise RuntimeError(
                    "DS9 MapAnything async postprocess is shutting down"
                )
            if self._async_thread is not None and self._async_thread.is_alive():
                return
            if self._async_thread is not None:
                raise RuntimeError(
                    "DS9 MapAnything async postprocess worker cannot be restarted"
                )

            def _loop() -> None:
                try:
                    while True:
                        job = self._async_queue.get()
                        try:
                            if job is _MAPANYTHING_ASYNC_STOP:
                                return
                            with self._async_lock:
                                poisoned = self._async_failure is not None
                            if poisoned:
                                _increment_core_counter(
                                    "mapanything_async_poisoned_jobs_total"
                                )
                                continue
                            try:
                                result = self.handle_numpy_arrays(
                                    source_id=int(job.source_id),
                                    frame_id=int(job.frame_id),
                                    pts_ns=int(job.pts_ns),
                                    tensors=job.tensors,
                                    captured_while_enabled=True,
                                    captured_at_us=int(job.captured_at_us),
                                )
                                if result is None:
                                    raise RuntimeError(
                                        "exact tensor capture produced no valid depth result"
                                    )
                            except Exception as exc:
                                self._poison_async_worker(exc)
                        finally:
                            self._async_queue.task_done()
                            with self._async_idle:
                                self._async_idle.notify_all()
                except BaseException as exc:
                    self._poison_async_worker(exc)
                finally:
                    self._async_stopped.set()
                    with self._async_idle:
                        self._async_idle.notify_all()

            self._async_stopped.clear()
            self._async_thread = threading.Thread(
                target=_loop,
                name="DS9MapAnythingPostprocess",
                daemon=False,
            )
            self._async_thread.start()

    def _begin_async_capture(self) -> None:
        with self._async_idle:
            if self._async_failure is not None:
                raise RuntimeError(
                    "DS9 MapAnything async postprocess is poisoned: "
                    f"{self._async_failure}"
                )
            if not self._async_accepting:
                raise RuntimeError(
                    "DS9 MapAnything async postprocess is shutting down"
                )
            self._async_active_captures += 1

    def _end_async_capture(self) -> None:
        with self._async_idle:
            if self._async_active_captures <= 0:
                raise RuntimeError(
                    "DS9 MapAnything async capture ownership underflow"
                )
            self._async_active_captures -= 1
            self._async_idle.notify_all()

    def _enqueue_async_job(self, job: "_MapAnythingNativeJob") -> None:
        # A capture that acquired its lease before shutdown is allowed to enqueue
        # even after accepting flips false.  Shutdown waits for every such lease
        # before appending the FIFO stop sentinel, so it cannot overtake this job.
        self._start_async_worker()
        with self._async_idle:
            if self._async_failure is not None:
                raise RuntimeError(
                    "DS9 MapAnything async postprocess is poisoned: "
                    f"{self._async_failure}"
                )
            try:
                self._async_queue.put_nowait(job)
            except queue.Full as exc:
                _increment_core_counter("mapanything_async_queue_full_total")
                raise RuntimeError(
                    "DS9 MapAnything bounded async postprocess queue is full"
                ) from exc

    def async_shutdown_quiesced(self) -> bool:
        with self._async_lock:
            thread = self._async_thread
            active_captures = int(self._async_active_captures)
            stopped = bool(self._async_stopped.is_set())
        unfinished = self._async_unfinished_task_count()
        thread_quiesced = thread is None or (
            not thread.is_alive() and stopped
        )
        return bool(
            active_captures == 0
            and unfinished == 0
            and thread_quiesced
        )

    def wait_idle(self, *, timeout_s: float = 5.0) -> MapAnythingIdleReceipt:
        """Wait for admitted captures and queued jobs without closing admission."""

        timeout = float(timeout_s)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("DS9 MapAnything idle timeout must be positive")
        deadline = time.monotonic() + timeout
        with self._async_idle:
            while True:
                if self._async_failure is not None:
                    raise RuntimeError(
                        "DS9 MapAnything async postprocess is poisoned: "
                        f"{self._async_failure}"
                    )
                if not self._async_accepting or self._async_shutdown_complete:
                    raise RuntimeError(
                        "DS9 MapAnything async postprocess is shutting down"
                    )
                active_captures = int(self._async_active_captures)
                unfinished = self._async_unfinished_task_count()
                thread = self._async_thread
                if active_captures == 0 and unfinished == 0:
                    return MapAnythingIdleReceipt(
                        active_captures=0,
                        unfinished_tasks=0,
                        worker_started=thread is not None,
                        worker_alive=bool(thread is not None and thread.is_alive()),
                        accepting=True,
                    )
                if unfinished > 0 and (
                    thread is None or not thread.is_alive()
                ):
                    raise RuntimeError(
                        "DS9 MapAnything has queued work without a live owned worker"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        "DS9 MapAnything did not become idle before timeout "
                        f"(active_captures={active_captures} unfinished={unfinished})"
                    )
                self._async_idle.wait(timeout=min(remaining, 0.05))

    def shutdown(self, *, wait: bool = True, timeout_s: float = 5.0) -> None:
        """Stop accepting captures, drain accepted jobs, and join the worker.

        The stop sentinel is enqueued only after all probe-local capture leases
        have completed.  FIFO ordering therefore proves that every accepted job
        was either processed or explicitly drained after a poison before return.
        """

        if not wait:
            raise ValueError("DS9 MapAnything shutdown requires wait=True")
        timeout = float(timeout_s)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("DS9 MapAnything shutdown timeout must be positive")
        deadline = time.monotonic() + timeout

        with self._async_idle:
            self._async_accepting = False
            if self._async_shutdown_complete:
                completed = True
            else:
                completed = False
                while self._async_active_captures > 0:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        _increment_core_counter(
                            "mapanything_async_shutdown_failures_total"
                        )
                        raise RuntimeError(
                            "DS9 MapAnything active tensor capture did not quiesce "
                            "before shutdown timeout"
                        )
                    self._async_idle.wait(timeout=remaining)
            thread = self._async_thread

        if completed:
            self._raise_async_failure()
            return

        if thread is None:
            if self._async_unfinished_task_count() != 0:
                _increment_core_counter(
                    "mapanything_async_shutdown_failures_total"
                )
                raise RuntimeError(
                    "DS9 MapAnything has queued work without an owned worker"
                )
            self._async_stopped.set()
            with self._async_idle:
                self._async_shutdown_complete = True
                self._async_idle.notify_all()
            _increment_core_counter("mapanything_async_shutdown_total")
            self._raise_async_failure()
            return

        with self._async_idle:
            enqueue_stop = not self._async_stop_enqueued
            if enqueue_stop:
                self._async_stop_enqueued = True
        if enqueue_stop:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                with self._async_idle:
                    self._async_stop_enqueued = False
                _increment_core_counter(
                    "mapanything_async_shutdown_failures_total"
                )
                raise RuntimeError(
                    "DS9 MapAnything shutdown timed out before stop enqueue"
                )
            try:
                self._async_queue.put(
                    _MAPANYTHING_ASYNC_STOP,
                    timeout=remaining,
                )
            except queue.Full as exc:
                with self._async_idle:
                    self._async_stop_enqueued = False
                _increment_core_counter(
                    "mapanything_async_shutdown_failures_total"
                )
                raise RuntimeError(
                    "DS9 MapAnything stop sentinel could not enter the bounded queue"
                ) from exc

        remaining = max(0.0, deadline - time.monotonic())
        thread.join(timeout=remaining)
        unfinished = self._async_unfinished_task_count()
        if thread.is_alive() or not self._async_stopped.is_set() or unfinished != 0:
            _increment_core_counter("mapanything_async_shutdown_failures_total")
            raise RuntimeError(
                "DS9 MapAnything async worker shutdown is unresolved "
                f"(alive={thread.is_alive()} stopped={self._async_stopped.is_set()} "
                f"unfinished={unfinished})"
            )

        with self._async_idle:
            self._async_shutdown_complete = True
            self._async_idle.notify_all()
        _increment_core_counter("mapanything_async_shutdown_total")
        self._raise_async_failure()

    def handle_native_frame_ds9(self, frame_meta: Any) -> bool:
        """Compatibility entrypoint for source-only exact tensor tests."""

        return self._handle_native_frame_ds9(frame_meta, buffer=None)

    def handle_native_buffer_frame_ds9(
        self,
        buffer: Any,
        frame_meta: Any,
    ) -> bool:
        """Bind the exact tensor metadata and batch surface from one buffer."""

        if buffer is None:
            raise RuntimeError("DS9 MapAnything exact RGB buffer is unavailable")
        return self._handle_native_frame_ds9(frame_meta, buffer=buffer)

    def _handle_native_frame_ds9(
        self,
        frame_meta: Any,
        *,
        buffer: Any | None,
    ) -> bool:
        """Capture one exact DS9 MapAnything UID/layer/batch contract."""

        self._raise_async_failure()
        if not self.pipeline.depth_enabled:
            raise RuntimeError(
                "DS9 MapAnything tensor capture was invoked while its gate was closed"
            )
        batch_id_raw = _meta_lookup(
            frame_meta,
            "batch_id",
            "batchId",
            "batch_index",
            default=None,
        )
        if batch_id_raw is None:
            raise RuntimeError("DS9 MapAnything frame is missing its exact batch_id")
        try:
            batch_id = int(batch_id_raw)
        except Exception as exc:
            raise RuntimeError("DS9 MapAnything frame batch_id is not an integer") from exc
        if batch_id < 0 or batch_id >= int(self.batch_size):
            raise RuntimeError(
                "DS9 MapAnything frame batch_id is outside the configured batch"
            )
        source_id_raw = _meta_lookup(
            frame_meta, "pad_index", "source_id", default=None
        )
        if source_id_raw is None:
            raise RuntimeError("DS9 MapAnything frame is missing its source identity")
        try:
            source_id = int(source_id_raw)
        except Exception as exc:
            raise RuntimeError(
                "DS9 MapAnything frame source identity is not an integer"
            ) from exc
        if source_id < 0 or (
            self.camera_labels and source_id not in self.camera_labels
        ):
            raise RuntimeError(
                "DS9 MapAnything frame source is outside the configured camera inventory"
            )
        frame_id_raw = _meta_lookup(
            frame_meta,
            "frame_num",
            "frame_number",
            default=None,
        )
        if frame_id_raw is None:
            raise RuntimeError(
                "DS9 MapAnything frame is missing its exact source frame number"
            )
        try:
            frame_id = int(frame_id_raw)
        except Exception as exc:
            raise RuntimeError(
                "DS9 MapAnything source frame number is not an integer"
            ) from exc
        if frame_id < 0:
            raise RuntimeError(
                "DS9 MapAnything source frame number must be non-negative"
            )
        pts_ns_raw = _meta_lookup(
            frame_meta,
            "buf_pts",
            "buffer_pts",
            default=None,
        )
        if pts_ns_raw is None:
            raise RuntimeError(
                "DS9 MapAnything frame is missing its exact source media PTS"
            )
        try:
            pts_ns = int(pts_ns_raw)
        except Exception as exc:
            raise RuntimeError(
                "DS9 MapAnything source media PTS is not an integer"
            ) from exc
        if pts_ns < 0 or pts_ns == (1 << 64) - 1:
            raise RuntimeError(
                "DS9 MapAnything source media PTS must be a valid non-negative "
                "GStreamer timestamp"
            )

        self._begin_async_capture()
        captured_at_us = max(1, int(time.time_ns() // 1_000))
        try:
            capture = self.native_capture or _require_ds9_mapanything_native_capture()
            capture_started_ns = time.perf_counter_ns()
            layers = capture(
                frame_meta,
                int(self.gie_id),
                int(self.output_height),
                int(self.output_width),
            )
            if layers is None:
                # nvinfer interval batches legitimately carry the video surface
                # without output tensor metadata.  They are not inference
                # results and must not trigger RGB conversion or D2H.
                return False
            try:
                raw_tensors = dict(layers)
            except Exception as exc:
                raise RuntimeError(
                    "DS9 MapAnything native tensor payload is not a mapping"
                ) from exc
            layer_names = {str(key) for key in raw_tensors}
            if layer_names != _MAPANYTHING_EXACT_LAYERS:
                raise RuntimeError(
                    "DS9 MapAnything native tensor layers must be exactly "
                    f"depth/conf/mask; found {sorted(layer_names)}"
                )

            tensors: Dict[str, np.ndarray] = {}
            expected_shape = (
                int(self.output_height),
                int(self.output_width),
            )
            for name in ("depth", "conf", "mask"):
                value = raw_tensors[name]
                array = np.asarray(value)
                if array.dtype != np.float32:
                    raise RuntimeError(
                        f"DS9 MapAnything layer {name} must be native float32 "
                        "after capture"
                    )
                array = np.squeeze(array)
                if (
                    array.ndim != 2
                    or int(array.shape[0]) <= 0
                    or int(array.shape[1]) <= 0
                ):
                    raise RuntimeError(
                        f"DS9 MapAnything layer {name} is not one non-empty "
                        "2-D batch slice"
                    )
                shape = (int(array.shape[0]), int(array.shape[1]))
                if shape != expected_shape:
                    raise RuntimeError(
                        "DS9 MapAnything depth/conf/mask tensors must match "
                        f"the exact per-frame shape {expected_shape}"
                    )
                if not array.flags.c_contiguous:
                    raise RuntimeError(
                        f"DS9 MapAnything layer {name} must be C-contiguous"
                    )
                tensors[name] = array

            payload_bytes = sum(int(array.nbytes) for array in tensors.values())
            if payload_bytes != self.host_payload_bytes:
                raise RuntimeError(
                    "DS9 MapAnything probe-local tensor copy exceeded its exact "
                    f"payload contract ({payload_bytes} != "
                    f"{self.host_payload_bytes})"
                )
            for _name in tensors:
                _increment_core_counter("tensor_host_copies_total.mapanything")
            _increment_core_counter(
                "tensor_boundary_copy_bytes_total.mapanything", payload_bytes
            )
            _increment_core_counter(
                "mapanything_probe_local_d2h_frames_total"
            )
            _increment_core_counter(
                "mapanything_probe_local_d2h_bytes_total", payload_bytes
            )
            _CORE_PATH_INSTRUMENTATION.record_boundary_serialization_prep(
                metric="mapanything.exact_native_tensor_to_host",
                duration_ns=time.perf_counter_ns() - capture_started_ns,
                payload_bytes=payload_bytes,
            )

            if buffer is not None and self.rgb_provider is not None:
                self._capture_armed_rgb(
                    buffer=buffer,
                    batch_id=batch_id,
                    source_id=source_id,
                    frame_id=frame_id,
                    source_media_pts_ns=pts_ns,
                )

            job = _MapAnythingNativeJob(
                source_id=source_id,
                frame_id=frame_id,
                pts_ns=pts_ns,
                captured_at_us=captured_at_us,
                tensors=tensors,
            )
            self._enqueue_async_job(job)
            return True
        finally:
            self._end_async_capture()

    def _capture_armed_rgb(
        self,
        *,
        buffer: Any,
        batch_id: int,
        source_id: int,
        frame_id: int,
        source_media_pts_ns: int,
    ) -> None:
        """Copy one exact GPU RGB surface to host only for a live arm."""

        camera_id = str(self.camera_labels.get(source_id, source_id))
        arm = self.rgb_provider.capture_arm(  # type: ignore[union-attr]
            source_id=source_id,
            camera_id=camera_id,
        )
        if arm is None:
            return

        try:
            import torch
            import torch.utils.dlpack as torch_dlpack
        except Exception as exc:
            raise RuntimeError(
                "DS9 exact RGB capture requires the pinned Torch runtime"
            ) from exc

        try:
            surface = buffer.extract(batch_id)
        except Exception as exc:
            raise RuntimeError(
                "DS9 exact RGB Buffer.extract(batch_id) failed"
            ) from exc
        if surface is None:
            raise RuntimeError("DS9 exact RGB buffer surface is unavailable")
        try:
            rgb = torch_dlpack.from_dlpack(surface)
        except Exception as exc:
            raise RuntimeError(
                "DS9 exact RGB buffer surface does not expose DLPack"
            ) from exc
        if not bool(getattr(rgb, "is_cuda", False)):
            raise RuntimeError("DS9 exact RGB buffer surface is not CUDA-backed")
        if rgb.dtype != torch.uint8:
            raise RuntimeError("DS9 exact RGB buffer surface must be uint8")
        expected_shape = (
            int(self.rgb_height),
            int(self.rgb_width),
            3,
        )
        if tuple(int(value) for value in rgb.shape) != expected_shape:
            raise RuntimeError(
                "DS9 exact RGB buffer must match the explicit RGB HxWx3 caps "
                f"{expected_shape}"
            )
        strides = tuple(int(value) for value in rgb.stride())
        if (
            len(strides) != 3
            or strides[2] != 1
            or strides[1] != 3
            or strides[0] < int(self.rgb_width) * 3
        ):
            raise RuntimeError(
                "DS9 exact RGB buffer has an unsupported RGB pitch/stride contract"
            )

        capture_started_ns = time.perf_counter_ns()
        rgb_cpu = rgb.contiguous().detach().cpu()
        pixels = rgb_cpu.numpy()
        if (
            pixels.dtype != np.uint8
            or pixels.shape
            != (int(self.rgb_height), int(self.rgb_width), 3)
            or not pixels.flags.c_contiguous
        ):
            raise RuntimeError("DS9 exact RGB D2H result is not contiguous rgb8")
        digest = hashlib.sha256(pixels.tobytes(order="C")).hexdigest()
        byte_count = int(pixels.nbytes)
        self.rgb_provider.offer(  # type: ignore[union-attr]
            TimestampedRgbFrame(
                camera_id=camera_id,
                source_id=source_id,
                batch_id=batch_id,
                captured_at_us=time.time_ns() // 1_000,
                frame_id=frame_id,
                source_media_pts_ns=source_media_pts_ns,
                width=int(self.rgb_width),
                height=int(self.rgb_height),
                content_sha256=digest,
                pixels=pixels,
            ),
            arm=arm,
        )
        _increment_core_counter("mapanything_exact_rgb_d2h_frames_total")
        _increment_core_counter("mapanything_exact_rgb_d2h_bytes_total", byte_count)
        _CORE_PATH_INSTRUMENTATION.record_boundary_serialization_prep(
            metric="mapanything.exact_buffer_rgb_to_host",
            duration_ns=time.perf_counter_ns() - capture_started_ns,
            payload_bytes=byte_count,
        )

    def _emit_from_tensors(
        self,
        frame_meta: Any,
        tensors: Mapping[str, np.ndarray],
        *,
        captured_while_enabled: bool = False,
    ) -> Optional[DepthResult]:
        if not self.pipeline.depth_enabled and not captured_while_enabled:
            logger.debug("Depth disabled; dropping MapAnything tensors")
            return None

        depth = _select_tensor(tensors, ("depth", "depth_z", "disp"))
        confidence = _select_tensor(tensors, ("confidence", "conf"))
        mask = _select_tensor(tensors, ("mask", "valid"))

        if depth is None and len(tensors) == 1:
            only_name, only_tensor = next(iter(tensors.items()))
            depth = only_tensor
            logger.debug("MapAnything tensors missing named depth; using sole tensor '%s' as depth", only_name)

        if depth is None:
            logger.debug("MapAnything tensors missing depth layer; skipping frame (keys=%s)", list(tensors.keys()))
            return None

        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3 and depth.shape[0] in (1, 3):
            depth = depth[0]
        if depth.ndim != 2:
            logger.debug("Unexpected depth tensor shape %s; expected 2-D map", depth.shape)
            return None
        original_shape = depth.shape

        if confidence is not None:
            confidence = np.asarray(confidence, dtype=np.float32)
            if confidence.shape != depth.shape:
                confidence = np.squeeze(confidence)
                if confidence.shape != depth.shape:
                    logger.debug("Confidence tensor shape %s does not match depth %s", confidence.shape, depth.shape)
                    confidence = None

        if mask is not None:
            mask = np.asarray(mask).astype(bool, copy=False)
            if mask.shape != depth.shape:
                mask = np.squeeze(mask)
                if mask.shape != depth.shape:
                    logger.debug("Mask tensor shape %s does not match depth %s", mask.shape, depth.shape)
                    mask = None

        if mask is None and confidence is not None:
            mask = confidence >= 0.5
        if mask is None:
            mask = np.ones_like(depth, dtype=bool)
        mask = np.logical_and(np.asarray(mask, dtype=bool), np.isfinite(depth))

        target_w, target_h = self._target_frame_shape(frame_meta, depth.shape)
        source_id = int(
            _meta_lookup(frame_meta, "pad_index", "source_id", default=0)
        )
        camera_id = str(self.camera_labels.get(source_id, source_id))
        metric_scale = 1.0
        model_input_focal_px: Optional[float] = None
        if self.depth_backend == "da3metric-large":
            metric_scale, model_input_focal_px = self._da3_metric_scale(
                source_id=source_id,
                camera_id=camera_id,
                target_size=(target_w, target_h),
                model_size=(int(depth.shape[1]), int(depth.shape[0])),
            )
            depth = np.asarray(depth * metric_scale, dtype=np.float32)
        depth, confidence, mask = self._align_to_frame(depth, confidence, mask, (target_w, target_h))
        if self.tensor_samples <= 5 or (self.tensor_samples % 50) == 0:
            if (depth.shape[1], depth.shape[0]) != (original_shape[1], original_shape[0]):
                logger.debug(
                    "Aligned MapAnything depth from %sx%s to %sx%s (frame coords)",
                    original_shape[1],
                    original_shape[0],
                    depth.shape[1],
                    depth.shape[0],
                )

        depth = np.asarray(depth, dtype=np.float32)
        if confidence is not None:
            confidence = np.asarray(confidence, dtype=np.float32)
            if confidence.shape != depth.shape:
                logger.debug("Confidence tensor shape %s does not match aligned depth %s", confidence.shape, depth.shape)
                confidence = None
        model_mask = (
            np.asarray(mask, dtype=bool)
            if mask is not None
            else np.ones_like(depth, dtype=bool)
        )
        fov_mask = self._dewarper_validity_mask(source_id, (depth.shape[1], depth.shape[0]))
        finite_depth = np.isfinite(depth) & (depth > 0.0)
        if fov_mask is not None:
            fov_mask = np.asarray(fov_mask, dtype=bool)
            if fov_mask.shape != depth.shape:
                raise RuntimeError(
                    "calibrated MapAnything FoV mask shape "
                    f"{fov_mask.shape} does not match depth {depth.shape}"
                )
        strict_mask = finite_depth & model_mask
        if fov_mask is not None:
            strict_mask &= fov_mask

        # MapAnything calls this output the non-ambiguous mask. It is useful
        # quality evidence, but it is not a geometric validity mask. Likewise,
        # the calibrated dewarper footprint identifies pixels outside the
        # source lens, but the model still produces finite dense depth there.
        # Hard-dropping either signal erased furniture and 23-38% of the image
        # in real captures. Preserve every finite positive prediction for the
        # manual quality path and encode both signals as confidence penalties.
        mask = finite_depth

        conf_array = confidence.astype(np.float32, copy=False) if confidence is not None else np.zeros_like(depth, dtype=np.float32)
        conf_array = np.array(conf_array, dtype=np.float32, copy=True)
        conf_array[~model_mask & finite_depth] *= float(
            _MAPANYTHING_AMBIGUOUS_CONFIDENCE_SCALE
        )
        if fov_mask is not None:
            conf_array[~fov_mask & finite_depth] *= float(
                _MAPANYTHING_OUTSIDE_CALIBRATED_FOV_CONFIDENCE_SCALE
            )
        conf_array[~finite_depth] = 0.0

        valid = depth[mask]
        if self.tensor_samples <= 5 or (self.tensor_samples % 50) == 0:
            min_val = float(valid.min()) if valid.size else float("nan")
            max_val = float(valid.max()) if valid.size else float("nan")
            logger.debug(
                "MapAnything depth stats: shape=%s finite=%s valid=%s depth[min,max]=(%s,%s)",
                depth.shape,
                np.isfinite(depth).sum(),
                valid.size,
                min_val,
                max_val,
            )
        if valid.size == 0:
            logger.debug("No finite positive MapAnything depth pixels; skipping frame")
            return None

        depth = np.array(depth, dtype=np.float32, copy=True)
        depth[~mask] = np.nan
        frame_id = int(_meta_lookup(frame_meta, "frame_num", "frame_number", default=0))
        source_pts_ns = int(
            _meta_lookup(frame_meta, "buf_pts", "buffer_pts", default=0) or 0
        )
        captured_at_us = int(
            _meta_lookup(frame_meta, "captured_at_us", default=0) or 0
        )
        # Callback-time wall clock identifies capture time. Keep the raw media
        # PTS as source evidence without letting async processing delay make the
        # snapshot appear newly captured.
        if captured_at_us > 0:
            ts_us = captured_at_us
            timestamp_basis = "capture_wall_clock"
        elif source_pts_ns >= 100_000_000_000_000_000:
            ts_us = max(0, source_pts_ns // 1_000)
            timestamp_basis = "source_epoch_pts"
        else:
            ts_us = max(0, time.time_ns() // 1_000)
            timestamp_basis = "wall_clock_fallback"
        ts_s = max(0, ts_us // 1_000_000)
        if self.tensor_samples <= 5 or (self.tensor_samples % 50) == 0:
            logger.debug(
                "MapAnything tensor meta timestamps: source=%s frame=%s pts_ns=%s ts_us=%s",
                source_id,
                frame_id,
                source_pts_ns,
                ts_us,
            )

        mask_u8 = mask.astype(np.uint8, copy=False)

        storage_attrs: Dict[str, Any] = {
            "source_frame_contract": "noesis.mapanything.source_frame.v1",
            "source_id": int(source_id),
            "source_frame_number": int(frame_id),
            "source_media_pts_ns": max(0, int(source_pts_ns)),
            "storage_timestamp_basis": timestamp_basis,
            "stored_mask_policy": "finite_positive_prediction",
            "strict_calibration_mask_policy": (
                "finite_positive_model_non_ambiguous_calibrated_fov"
            ),
            "strict_calibration_valid_fraction": float(
                np.count_nonzero(strict_mask) / max(1, strict_mask.size)
            ),
            "model_ambiguous_confidence_scale": float(
                _MAPANYTHING_AMBIGUOUS_CONFIDENCE_SCALE
            ),
            "outside_calibrated_fov_confidence_scale": float(
                _MAPANYTHING_OUTSIDE_CALIBRATED_FOV_CONFIDENCE_SCALE
            ),
        }
        if self.depth_backend == "da3metric-large":
            storage_attrs.update(
                {
                    "manual_depth_backend": self.depth_backend,
                    "depth_units": "meters",
                    "metric_scale": float(metric_scale),
                    "model_input_focal_px": float(model_input_focal_px),
                    "metric_focal_denominator": float(
                        self.metric_focal_denominator
                    ),
                }
            )
        write_handle = self.storage.store(
            camera_id,
            ts_us,
            depth,
            conf_array,
            mask_u8,
            attrs=storage_attrs,
        )
        try:
            commit_receipt = write_handle.wait(
                timeout=self._storage_commit_timeout_s
            )
        except Exception:
            _increment_core_counter(
                "mapanything_depth_store_commit_failures_total"
            )
            raise
        _increment_core_counter("mapanything_depth_store_commits_total")
        depth_ref = str(commit_receipt.path)
        self.pipeline.record_depth_frame(time.time())

        result = DepthResult(
            source_id=source_id,
            frame_id=frame_id,
            ts=ts_s,
            width=depth.shape[1],
            height=depth.shape[0],
            depth_map_ref=depth_ref,
            minmax=(float(np.nanmin(valid)), float(np.nanmax(valid))),
        )

        if self.depth_pub is not None:
            try:
                self.depth_pub.publish(result)
            except Exception:  # pragma: no cover - telemetry failures shouldn't break processing
                logger.exception("Depth telemetry publish failed for source %s frame %s", source_id, frame_id)

        return result


@dataclass(frozen=True)
class _MapAnythingNativeJob:
    source_id: int
    frame_id: int
    pts_ns: int
    captured_at_us: int
    tensors: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class V3DTCuboidOverlayConfig:
    enabled: bool = False
    class_ids: frozenset[int] = field(default_factory=lambda: frozenset({0}))
    line_width: int = 3
    color: Tuple[float, float, float] = (0.05, 0.70, 1.0)
    alpha: float = 1.0
    depth_scale: float = 0.22
    vanishing_point: Tuple[float, float] = (0.50, 0.18)

    def __post_init__(self) -> None:
        object.__setattr__(self, "line_width", max(1, int(self.line_width)))
        object.__setattr__(self, "alpha", min(1.0, max(0.0, float(self.alpha))))
        object.__setattr__(
            self, "depth_scale", min(0.5, max(0.02, float(self.depth_scale)))
        )
        color = tuple(min(1.0, max(0.0, float(value))) for value in self.color[:3])
        if len(color) != 3:
            color = (0.05, 0.70, 1.0)
        object.__setattr__(self, "color", color)
        vp = tuple(float(value) for value in self.vanishing_point[:2])
        if len(vp) != 2 or not all(math.isfinite(value) for value in vp):
            vp = (0.50, 0.18)
        object.__setattr__(self, "vanishing_point", vp)
        class_ids: set[int] = set()
        for value in self.class_ids:
            try:
                class_ids.add(int(value))
            except Exception:
                continue
        object.__setattr__(self, "class_ids", frozenset(class_ids or {0}))

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> "V3DTCuboidOverlayConfig":
        def _bool(value: Any, default: bool) -> bool:
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            if text in _ENV_TRUE_VALUES:
                return True
            if text in ("0", "false", "no", "off", "n"):
                return False
            return default

        def _tuple(raw: Any, length: int, default: Tuple[float, ...]) -> Tuple[float, ...]:
            if not isinstance(raw, (list, tuple)) or len(raw) < length:
                return default
            try:
                return tuple(float(value) for value in raw[:length])
            except Exception:
                return default

        raw_ids = cfg.get("class_ids", [0])
        if not isinstance(raw_ids, (list, tuple, set, frozenset)):
            raw_ids = [raw_ids]
        class_ids: set[int] = set()
        for value in raw_ids:
            try:
                class_ids.add(int(value))
            except Exception:
                continue
        try:
            line_width = int(cfg.get("line_width", 3) or 3)
        except Exception:
            line_width = 3
        try:
            alpha = float(cfg.get("alpha", 1.0) or 1.0)
        except Exception:
            alpha = 1.0
        try:
            depth_scale = float(cfg.get("depth_scale", 0.22) or 0.22)
        except Exception:
            depth_scale = 0.22
        return cls(
            enabled=_bool(cfg.get("enabled"), False),
            class_ids=frozenset(class_ids or {0}),
            line_width=line_width,
            color=_tuple(cfg.get("color"), 3, (0.05, 0.70, 1.0)),
            alpha=alpha,
            depth_scale=depth_scale,
            vanishing_point=_tuple(
                cfg.get("vanishing_point"), 2, (0.50, 0.18)
            ),
        )


def _anchored_cuboid_segments(
    bbox: Tuple[float, float, float, float],
    anchor: Tuple[float, float],
    depth_vector: Tuple[float, float],
) -> Tuple[Tuple[Point2, Point2], ...]:
    """Build cuboid edges with the bottom-face centroid exactly at anchor."""
    _left, _top, width, height = bbox
    anchor_x, anchor_y = anchor
    depth_x, depth_y = depth_vector
    half_width = max(1.0, float(width)) * 0.5
    box_height = max(1.0, float(height))

    front_bottom_mid = (
        float(anchor_x) - (0.5 * float(depth_x)),
        float(anchor_y) - (0.5 * float(depth_y)),
    )
    back_bottom_mid = (
        float(anchor_x) + (0.5 * float(depth_x)),
        float(anchor_y) + (0.5 * float(depth_y)),
    )
    front_bottom_left = (front_bottom_mid[0] - half_width, front_bottom_mid[1])
    front_bottom_right = (front_bottom_mid[0] + half_width, front_bottom_mid[1])
    back_bottom_left = (back_bottom_mid[0] - half_width, back_bottom_mid[1])
    back_bottom_right = (back_bottom_mid[0] + half_width, back_bottom_mid[1])
    front_top_left = (front_bottom_left[0], front_bottom_left[1] - box_height)
    front_top_right = (front_bottom_right[0], front_bottom_right[1] - box_height)
    back_top_left = (back_bottom_left[0], back_bottom_left[1] - box_height)
    back_top_right = (back_bottom_right[0], back_bottom_right[1] - box_height)

    return (
        (front_top_left, front_top_right),
        (front_top_right, front_bottom_right),
        (front_bottom_right, front_bottom_left),
        (front_bottom_left, front_top_left),
        (back_top_left, back_top_right),
        (back_top_right, back_bottom_right),
        (back_bottom_right, back_bottom_left),
        (back_bottom_left, back_top_left),
        (front_top_left, back_top_left),
        (front_top_right, back_top_right),
        (front_bottom_right, back_bottom_right),
        (front_bottom_left, back_bottom_left),
    )


@dataclass
class V3DTCuboidOverlayProcessor:
    pipeline: "DS8Pipeline"
    config: V3DTCuboidOverlayConfig
    _last_log_ts: float = field(default=0.0, init=False, repr=False)
    _frames: int = field(default=0, init=False, repr=False)
    _objects: int = field(default=0, init=False, repr=False)
    _mask_anchors: int = field(default=0, init=False, repr=False)
    _bbox_anchors: int = field(default=0, init=False, repr=False)
    _vendor_boxes_removed: int = field(default=0, init=False, repr=False)
    _vendor_feet_removed: int = field(default=0, init=False, repr=False)

    def _frame_size(self, frame_meta: Any) -> Tuple[int, int]:
        frame_w = int(
            _meta_lookup(
                frame_meta, "frame_width", "width", "source_frame_width", default=0
            )
            or 0
        )
        frame_h = int(
            _meta_lookup(
                frame_meta, "frame_height", "height", "source_frame_height", default=0
            )
            or 0
        )
        if frame_w <= 0 or frame_h <= 0:
            try:
                frame_w, frame_h = tuple(getattr(self.pipeline, "frame_size", (0, 0)))
                frame_w, frame_h = int(frame_w), int(frame_h)
            except Exception:
                frame_w, frame_h = 0, 0
        return frame_w, frame_h

    def _depth_vector(
        self,
        *,
        anchor: Tuple[float, float],
        bbox: Tuple[float, float, float, float],
        frame_size: Tuple[int, int],
    ) -> Tuple[float, float]:
        frame_w, frame_h = frame_size
        vp_x = float(self.config.vanishing_point[0]) * float(max(1, frame_w))
        vp_y = float(self.config.vanishing_point[1]) * float(max(1, frame_h))
        direction_x = vp_x - float(anchor[0])
        direction_y = vp_y - float(anchor[1])
        norm = math.hypot(direction_x, direction_y)
        if norm <= 1e-6:
            direction_x, direction_y, norm = 0.0, -1.0, 1.0
        magnitude = max(
            4.0,
            min(float(bbox[2]), float(bbox[3])) * float(self.config.depth_scale),
        )
        return (
            (direction_x / norm) * magnitude,
            (direction_y / norm) * magnitude,
        )

    def scrub_vendor_overlay(self, buffer: Any) -> None:
        stats = noesis_v3dt_meta_ext.scrub_tracker_projection_display_meta(buffer)
        if isinstance(stats, Mapping):
            self._vendor_boxes_removed += int(stats.get("bbox3d_removed", 0) or 0)
            self._vendor_feet_removed += int(stats.get("foot_removed", 0) or 0)

    def render_batch(self, batch_meta: Any) -> None:
        if ds_osd is None or batch_meta is None:
            return
        acquire_display_meta = getattr(batch_meta, "acquire_display_meta", None)
        frame_items = getattr(batch_meta, "frame_items", None)
        if not callable(acquire_display_meta) or frame_items is None:
            return
        for frame_meta in frame_items:
            self._frames += 1
            append_meta = getattr(frame_meta, "append", None)
            if not callable(append_meta):
                continue
            frame_size = self._frame_size(frame_meta)
            for obj_meta in getattr(frame_meta, "object_items", None) or []:
                try:
                    class_id = int(getattr(obj_meta, "class_id", -1))
                except Exception:
                    class_id = -1
                if class_id not in self.config.class_ids:
                    continue
                bbox = _rect_to_bbox(getattr(obj_meta, "rect_params", None))
                if bbox is None or bbox[2] <= 1.0 or bbox[3] <= 1.0:
                    continue
                raw_anchor = noesis_v3dt_meta_ext.extract_person_base(obj_meta)
                if not isinstance(raw_anchor, Mapping):
                    continue
                try:
                    anchor = (float(raw_anchor["x"]), float(raw_anchor["y"]))
                except Exception:
                    continue
                if not all(math.isfinite(value) for value in anchor):
                    continue
                source = str(raw_anchor.get("source") or "bbox_bottom")
                if source == "instance_mask_base":
                    self._mask_anchors += 1
                else:
                    self._bbox_anchors += 1
                depth_vector = self._depth_vector(
                    anchor=anchor, bbox=bbox, frame_size=frame_size
                )
                segments = _anchored_cuboid_segments(bbox, anchor, depth_vector)
                display_meta = acquire_display_meta()
                if not display_meta:
                    continue
                line = ds_osd.Line()
                line.width = int(self.config.line_width)
                line.color.r = float(self.config.color[0])
                line.color.g = float(self.config.color[1])
                line.color.b = float(self.config.color[2])
                line.color.a = float(self.config.alpha)
                frame_w, frame_h = frame_size
                for start, end in segments:
                    x1, y1 = start
                    x2, y2 = end
                    if frame_w > 0:
                        x1 = min(float(frame_w - 1), max(0.0, x1))
                        x2 = min(float(frame_w - 1), max(0.0, x2))
                    if frame_h > 0:
                        y1 = min(float(frame_h - 1), max(0.0, y1))
                        y2 = min(float(frame_h - 1), max(0.0, y2))
                    line.x1, line.y1 = int(round(x1)), int(round(y1))
                    line.x2, line.y2 = int(round(x2)), int(round(y2))
                    display_meta.add_line(line)
                append_meta(display_meta)
                self._objects += 1

        now = time.time()
        if now - self._last_log_ts >= 10.0:
            logger.info(
                "V3DT cuboid correction: frames=%d objects=%d mask_anchors=%d "
                "bbox_anchors=%d vendor_boxes_removed=%d vendor_feet_removed=%d",
                self._frames,
                self._objects,
                self._mask_anchors,
                self._bbox_anchors,
                self._vendor_boxes_removed,
                self._vendor_feet_removed,
            )
            self._last_log_ts = now


@dataclass(frozen=True)
class TrailOverlayConfig:
    enabled: bool = True
    class_ids: frozenset[int] = field(default_factory=lambda: frozenset({0}))
    anchor_mode: str = "bbox_bottom"
    window_s: float = 8.0
    draw_stride: int = 2
    min_step_px: float = 2.0
    min_dt_s: float = 0.08
    smooth_tau_s: float = 0.25
    max_speed_px_per_s: float = 600.0
    gap_predict_ttl_s: float = 0.0
    gap_predict_decay_tau_s: float = 0.75
    predicted_alpha_scale: float = 0.65
    height_peak_up_alpha: float = 0.35
    height_peak_down_alpha: float = 0.05
    height_good_frame_ratio: float = 0.90
    # Maximum history length stored per track (points, not segments).
    # Effective history is bounded by BOTH `window_s` and this point cap.
    max_points_per_track: int = 129
    # Maximum number of line segments per track rendered each frame.
    # This is the primary CPU control knob in DS8 (each segment is a Python→C call).
    max_segments_per_track: int = 64
    # Maximum number of tracks rendered per tile/frame.
    max_tracks: int = 8
    max_lines: int = 800
    max_display_metas: int = 32
    line_width: int = 3
    min_alpha: float = 0.15
    show_labels: bool = False
    color_key: str = "stable_id"

    def __post_init__(self) -> None:
        object.__setattr__(self, "window_s", max(0.1, float(self.window_s)))
        object.__setattr__(self, "draw_stride", max(1, int(self.draw_stride)))
        object.__setattr__(self, "min_step_px", max(0.0, float(self.min_step_px)))
        object.__setattr__(self, "min_dt_s", max(0.0, float(self.min_dt_s)))
        object.__setattr__(self, "smooth_tau_s", max(0.0, float(self.smooth_tau_s)))
        object.__setattr__(self, "max_speed_px_per_s", max(0.0, float(self.max_speed_px_per_s)))
        object.__setattr__(self, "gap_predict_ttl_s", max(0.0, float(self.gap_predict_ttl_s)))
        object.__setattr__(self, "gap_predict_decay_tau_s", max(0.0, float(self.gap_predict_decay_tau_s)))
        object.__setattr__(
            self,
            "predicted_alpha_scale",
            float(min(1.0, max(0.0, float(self.predicted_alpha_scale)))),
        )
        object.__setattr__(
            self,
            "height_peak_up_alpha",
            float(min(1.0, max(0.0, float(self.height_peak_up_alpha)))),
        )
        object.__setattr__(
            self,
            "height_peak_down_alpha",
            float(min(1.0, max(0.0, float(self.height_peak_down_alpha)))),
        )
        object.__setattr__(
            self,
            "height_good_frame_ratio",
            float(min(1.0, max(0.0, float(self.height_good_frame_ratio)))),
        )
        object.__setattr__(self, "max_points_per_track", max(2, int(self.max_points_per_track)))
        object.__setattr__(self, "max_segments_per_track", max(1, int(self.max_segments_per_track)))
        object.__setattr__(self, "max_tracks", max(1, int(self.max_tracks)))
        object.__setattr__(self, "max_lines", max(1, int(self.max_lines)))
        object.__setattr__(self, "max_display_metas", max(1, int(self.max_display_metas)))
        object.__setattr__(self, "line_width", max(1, int(self.line_width)))
        object.__setattr__(self, "min_alpha", float(min(1.0, max(0.0, float(self.min_alpha)))))
        anchor_mode = str(self.anchor_mode or "bbox_bottom").strip().lower()
        if anchor_mode not in ("bbox_bottom", "floor_plane_gravity_drop"):
            anchor_mode = "bbox_bottom"
        object.__setattr__(self, "anchor_mode", anchor_mode)
        color_key = str(self.color_key or "stable_id").strip().lower()
        if color_key not in ("stable_id", "track_id"):
            color_key = "stable_id"
        object.__setattr__(self, "color_key", color_key)

        cls_ids = set()
        for item in self.class_ids:
            try:
                cls_ids.add(int(item))
            except Exception:
                continue
        if not cls_ids:
            cls_ids = {0}
        object.__setattr__(self, "class_ids", frozenset(cls_ids))

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> "TrailOverlayConfig":
        if not isinstance(cfg, Mapping):
            return cls()

        def _bool(value: Any, default: bool) -> bool:
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            if text in ("1", "true", "yes", "y", "on"):
                return True
            if text in ("0", "false", "no", "n", "off"):
                return False
            return default

        def _float(value: Any, default: float) -> float:
            try:
                return float(value)
            except Exception:
                return default

        def _int(value: Any, default: int) -> int:
            try:
                return int(value)
            except Exception:
                return default

        class_ids_raw = cfg.get("class_ids")
        cls_ids: set[int] = set()
        if isinstance(class_ids_raw, (list, tuple, set)):
            for item in class_ids_raw:
                try:
                    cls_ids.add(int(item))
                except Exception:
                    continue
        elif class_ids_raw is not None:
            try:
                cls_ids.add(int(class_ids_raw))
            except Exception:
                pass

        if not cls_ids:
            cls_ids = {0}

        max_points_value = _int(cfg.get("max_points_per_track"), 129)
        max_segments_value = _int(cfg.get("max_segments_per_track"), 64)
        if max_segments_value > (max_points_value - 1):
            max_segments_value = max(1, int(max_points_value) - 1)

        return cls(
            enabled=_bool(cfg.get("enabled"), True),
            class_ids=frozenset(cls_ids),
            anchor_mode=str(cfg.get("anchor_mode") or "bbox_bottom"),
            window_s=_float(cfg.get("window_s"), 8.0),
            draw_stride=_int(cfg.get("draw_stride"), 2),
            min_step_px=_float(cfg.get("min_step_px"), 2.0),
            min_dt_s=_float(cfg.get("min_dt_s"), 0.08),
            smooth_tau_s=_float(cfg.get("smooth_tau_s"), 0.25),
            max_speed_px_per_s=_float(cfg.get("max_speed_px_per_s"), 600.0),
            gap_predict_ttl_s=_float(cfg.get("gap_predict_ttl_s"), 0.0),
            gap_predict_decay_tau_s=_float(cfg.get("gap_predict_decay_tau_s"), 0.75),
            predicted_alpha_scale=_float(cfg.get("predicted_alpha_scale"), 0.65),
            height_peak_up_alpha=_float(cfg.get("height_peak_up_alpha"), 0.35),
            height_peak_down_alpha=_float(cfg.get("height_peak_down_alpha"), 0.05),
            height_good_frame_ratio=_float(cfg.get("height_good_frame_ratio"), 0.90),
            max_points_per_track=max_points_value,
            max_segments_per_track=max_segments_value,
            max_tracks=_int(cfg.get("max_tracks"), 8),
            max_lines=_int(cfg.get("max_lines"), 800),
            max_display_metas=_int(cfg.get("max_display_metas"), 32),
            line_width=_int(cfg.get("line_width"), 3),
            min_alpha=_float(cfg.get("min_alpha"), 0.15),
            show_labels=_bool(cfg.get("show_labels"), False),
            color_key=str(cfg.get("color_key") or "stable_id"),
        )


@dataclass
class _TrailPoint:
    ts: float
    x: float
    y: float
    predicted: bool = False


@dataclass
class _TrailTrackState:
    points: "deque[_TrailPoint]" = field(default_factory=deque)
    last_seen_ts: float = 0.0
    ema_x: Optional[float] = None
    ema_y: Optional[float] = None
    ema_ts: float = 0.0
    height_ref_scene: Optional[float] = None
    height_peak_px: Optional[float] = None
    last_measure_world_x: Optional[float] = None
    last_measure_world_z: Optional[float] = None
    last_measure_ts: float = 0.0
    vel_world_x: float = 0.0
    vel_world_z: float = 0.0
    last_measure_speed: float = 0.0
    trail_segment_id: Optional[int] = None
    tracker_lifecycle_generation: Optional[int] = None
    anchor_basis: Optional[Tuple[Any, ...]] = None


@dataclass
class _WorldClockState:
    """Per-source world-filter clock anchored to stream media PTS.

    ``buf_pts`` is stream-relative, so using it directly as an epoch timestamp
    would make TTLs and ghost gaps incomparable across cameras.  The first
    usable PTS is therefore anchored to the observed wall clock for that
    source.  Subsequent frames advance in the media domain while retaining a
    wall-clock-shaped value for cross-source housekeeping.
    """

    source_epoch: int = 0
    base_media_pts_ns: Optional[int] = None
    base_observed_ts: Optional[float] = None
    last_media_pts_ns: Optional[int] = None
    logical_ts: Optional[float] = None
    last_observed_ts: Optional[float] = None
    basis: str = "observed_clock"


@dataclass(frozen=True)
class _WorldOutputWatermark:
    """Last canonical point emitted for one exact tracker lifecycle."""

    world_x: float
    world_z: float
    media_pts_ns: Optional[int]
    filter_ts: float
    trail_segment_id: int


# Product-owned person-ground state and scoring stay single-source. DS9 only
# adapts transient Service Maker metadata into these SDK-neutral types.
_PoseAnchorCandidate = PoseAnchorCandidate
_WorldAnchorState = PersonGroundState


FrameKey = Tuple[int, int, int]
Point2 = Tuple[float, float]
Point3 = Tuple[float, float, float]


@dataclass(frozen=True)
class _DepthAnchorSample:
    foot_uv: Optional[Point2]
    anchor_source: Optional[str]
    anchor_depth_m: Optional[float]
    anchor_sample_count: int
    anchor_valid_fraction: float
    anchor_depth_spread_m: Optional[float]
    anchor_rejection_reason: Optional[str]
    lower_body_sample_count: int
    lower_body_valid_fraction: float
    torso_sample_count: int
    torso_valid_fraction: float


@dataclass(frozen=True)
class _DepthObservationResult:
    world_point: Optional[np.ndarray]
    weight: float
    reason: str
    raw_depth_m: Optional[float] = None
    registered_depth_m: Optional[float] = None
    registration_status: Optional[str] = None
    registration_id: Optional[str] = None


@dataclass
class _AlignedDepthFrame:
    key: FrameKey
    source_id: int
    frame_id: int
    pts_us: int
    depth_map: Optional[np.ndarray]
    valid_mask: Optional[np.ndarray]
    frame_w: int
    frame_h: int
    depth_w: int
    depth_h: int
    unit: str
    is_metric: bool
    model_name: str
    depth_device_frame: Any | None = None


def _aligned_depth_frame_is_ready(frame: _AlignedDepthFrame) -> bool:
    """Return readiness without ever synchronizing a device depth frame.

    Host-backed depth maps are complete when inserted.  The native DAv2 frame
    exposes ``is_ready()`` as a query-only CUDA event check; a missing method
    is retained as ready for test doubles and legacy host-backed providers.
    A failed readiness query is fail-closed so the media callback cannot turn
    a secondary depth error into a blocking wait.
    """
    device_frame = getattr(frame, "depth_device_frame", None)
    if device_frame is None:
        return True
    query = getattr(device_frame, "is_ready", None)
    if not callable(query):
        return True
    try:
        return bool(query())
    except Exception:
        _increment_core_counter("depth_bridge_readiness_error_total")
        logger.debug("Aligned depth readiness query failed", exc_info=True)
        return False


class _AlignedDepthFrameStore:
    def __init__(self, max_entries: int = 16) -> None:
        self._max_entries = max(2, int(max_entries))
        self._entries: "OrderedDict[FrameKey, _AlignedDepthFrame]" = OrderedDict()
        self._condition = threading.Condition()

    def put(self, frame: _AlignedDepthFrame) -> None:
        with self._condition:
            self._entries[frame.key] = frame
            self._entries.move_to_end(frame.key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
            self._condition.notify_all()
        _increment_core_counter("depth_bridge_put_total")

    def resolve(
        self,
        *,
        source_id: int,
        frame_id: int,
        pts_us: int,
        max_age_frames: int,
        wait_ms: float = 0.0,
    ) -> Tuple[Optional[_AlignedDepthFrame], int, float]:
        source_key = int(source_id)
        frame_key = int(frame_id)
        pts_key = int(pts_us)
        exact_key = (source_key, frame_key, pts_key)
        max_age = max(0, int(max_age_frames))
        # ``wait_ms`` remains in the contract for callers/config compatibility,
        # but depth is an optional secondary input on the media callback.  A
        # wait here serializes encode/analytics behind the private CUDA stream,
        # so readiness is query-only and the value is deliberately ignored.
        with self._condition:
            exact = self._entries.get(exact_key)
            if exact is not None and _aligned_depth_frame_is_ready(exact):
                _increment_core_counter("depth_bridge_exact_resolve_total")
                return exact, 0, 0.0
            if exact is not None:
                _increment_core_counter("depth_bridge_pending_exact_total")
            if float(wait_ms or 0.0) > 0.0:
                _increment_core_counter("depth_bridge_wait_bypassed_total")

            if max_age > 0:
                ready_lagged: list[tuple[int, int, _AlignedDepthFrame]] = []
                for candidate in self._entries.values():
                    if int(candidate.source_id) != source_key:
                        continue
                    age_frames = frame_key - int(candidate.frame_id)
                    if age_frames <= 0 or age_frames > max_age:
                        continue
                    age_us = pts_key - int(candidate.pts_us)
                    if age_us < 0:
                        continue
                    if not _aligned_depth_frame_is_ready(candidate):
                        _increment_core_counter(
                            "depth_bridge_pending_lagged_skipped_total"
                        )
                        continue
                    ready_lagged.append((int(candidate.frame_id), int(candidate.pts_us), candidate))
                if ready_lagged:
                    _candidate_frame_id, _candidate_pts_us, candidate = max(
                        ready_lagged,
                        key=lambda item: (item[0], item[1]),
                    )
                    age_frames = frame_key - int(candidate.frame_id)
                    age_us = pts_key - int(candidate.pts_us)
                    age_ms = float(age_us) / 1000.0
                    _increment_core_counter("depth_bridge_lagged_resolve_total")
                    _increment_core_counter(
                        "depth_bridge_lagged_age_frames_total",
                        age_frames,
                    )
                    _increment_core_counter(
                        "depth_bridge_lagged_age_us_total",
                        age_us,
                    )
                    return candidate, age_frames, age_ms
        _increment_core_counter("depth_bridge_miss_total")
        return None, 0, 0.0


@dataclass
class TrailOverlayProcessor:
    pipeline: "DS8Pipeline"
    config: TrailOverlayConfig
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _enabled: bool = field(default=True, init=False)
    _tracks: Dict[int, Dict[int, _TrailTrackState]] = field(default_factory=dict, init=False)
    _frame_counts: Dict[int, int] = field(default_factory=dict, init=False)
    _color_cache: Dict[int, Tuple[float, float, float]] = field(default_factory=dict, init=False)
    _logged_priors: bool = field(default=False, init=False)
    _mosaic_size: Tuple[int, int] = field(default_factory=lambda: (0, 0), init=False)
    _debug_last_log_ts: float = field(default=0.0, init=False, repr=False)
    _debug_frames: int = field(default=0, init=False, repr=False)
    _debug_lines: int = field(default=0, init=False, repr=False)
    _debug_tracks_seen: int = field(default=0, init=False, repr=False)
    _debug_tracks_drawn: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._enabled = bool(self.config.enabled)
        self._mosaic_size = self._resolve_mosaic_size()

    def enabled(self) -> bool:
        with self._lock:
            return bool(self._enabled)

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = bool(enabled)
            if not self._enabled:
                self._tracks.clear()
                self._frame_counts.clear()

    def _resolve_mosaic_size(self) -> Tuple[int, int]:
        tiler = self.pipeline.components.get("tiler")
        if tiler is not None and isinstance(tiler.config, dict):
            try:
                w = int(tiler.config.get("width", 0) or 0)
                h = int(tiler.config.get("height", 0) or 0)
                if w > 0 and h > 0:
                    return w, h
            except Exception:
                pass
            try:
                cols = int(tiler.config.get("columns", 0) or 0)
                rows = int(tiler.config.get("rows", 0) or 0)
            except Exception:
                cols = 0
                rows = 0
        else:
            cols = 0
            rows = 0

        try:
            src_w, src_h = getattr(self.pipeline, "frame_size", (0, 0))
            src_w = int(src_w or 0)
            src_h = int(src_h or 0)
        except Exception:
            src_w = 0
            src_h = 0

        if src_w > 0 and src_h > 0 and cols > 0 and rows > 0:
            return int(src_w * cols), int(src_h * rows)
        return 0, 0

    def _hsl_to_rgb(self, h: float, s: float, lightness: float) -> Tuple[float, float, float]:
        h = h % 360.0
        s = max(0.0, min(1.0, s))
        lightness = max(0.0, min(1.0, lightness))
        c = (1.0 - abs(2.0 * lightness - 1.0)) * s
        x = c * (1.0 - abs(((h / 60.0) % 2.0) - 1.0))
        m = lightness - c / 2.0
        rp = gp = bp = 0.0
        if 0.0 <= h < 60.0:
            rp, gp, bp = c, x, 0.0
        elif 60.0 <= h < 120.0:
            rp, gp, bp = x, c, 0.0
        elif 120.0 <= h < 180.0:
            rp, gp, bp = 0.0, c, x
        elif 180.0 <= h < 240.0:
            rp, gp, bp = 0.0, x, c
        elif 240.0 <= h < 300.0:
            rp, gp, bp = x, 0.0, c
        else:
            rp, gp, bp = c, 0.0, x
        r, g, b = rp + m, gp + m, bp + m
        return (max(0.0, min(1.0, r)), max(0.0, min(1.0, g)), max(0.0, min(1.0, b)))

    def _color_for_key(self, key: int) -> Tuple[float, float, float]:
        cached = self._color_cache.get(key)
        if cached is not None:
            return cached
        hue = float((int(key) * 47) % 360)
        rgb = self._hsl_to_rgb(hue, 0.80, 0.60)
        self._color_cache[key] = rgb
        return rgb

    def _frame_source_id(self, frame_meta: Any) -> int:
        for attr in ("source_id", "pad_index", "camera_id"):
            value = getattr(frame_meta, attr, None)
            if value is None:
                continue
            try:
                return int(value)
            except Exception:
                continue
        return 0

    def _camera_id_for_sensor(self, sensor_id: int) -> str:
        labels = getattr(self.pipeline, "camera_labels", {}) or {}
        try:
            camera_id = labels.get(int(sensor_id))
        except Exception:
            camera_id = None
        return str(camera_id or f"camera_{int(sensor_id)}")

    @staticmethod
    def _frame_id(frame_meta: Any) -> Optional[int]:
        value = _meta_lookup(frame_meta, "frame_number", "frame_num", default=None)
        try:
            parsed = int(value)
        except Exception:
            return None
        return parsed if parsed >= 0 else None

    def _analytics_track_map(
        self,
        sensor_id: int,
        frame_id: Optional[int] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """Return only the analytics rows belonging to this exact video frame.

        The analytics processor owns a latest-per-source cache.  A latest cache
        is useful for diagnostics but is not a valid join key for OSD metadata:
        the tiler callback can observe frame N while the cache already contains
        frame N+1.  In floor mode that would make a trail jump to a different
        person's/current-frame anchor.  Keep the join exact and fail closed.
        """
        processor = getattr(self.pipeline, "analytics_telemetry_processor", None)
        getter = getattr(processor, "get_active_track_map", None)
        if not callable(getter):
            return {}
        try:
            # Production processor exposes an exact-frame ring.  Test doubles
            # may still implement the historical one-argument getter; their
            # returned rows are filtered below, but production never falls
            # back from a missing frame.
            try:
                track_map = getter(int(sensor_id), frame_id) if frame_id is not None else getter(int(sensor_id))
            except TypeError:
                track_map = getter(int(sensor_id))
        except Exception:
            return {}
        if not isinstance(track_map, Mapping):
            return {}

        indexed: Dict[int, Dict[str, Any]] = {}
        for key, value in track_map.items():
            if not isinstance(value, Mapping):
                continue
            if frame_id is not None:
                try:
                    track_frame_id = int(value.get("frame_id"))
                except Exception:
                    continue
                if track_frame_id != int(frame_id):
                    continue
            tracker_id = value.get("tracker_id", value.get("track_id", key))
            try:
                tracker_id_int = int(tracker_id)
            except Exception:
                continue
            if tracker_id_int < 0:
                continue
            indexed[tracker_id_int] = dict(value)
        return indexed

    def _resolve_calibration(self, sensor_id: int, camera_id: str) -> Any:
        provider = getattr(self.pipeline, "bev_calibration", None)
        world_snapshot = getattr(provider, "world_snapshot", None)
        if not callable(world_snapshot):
            return None
        try:
            return world_snapshot(int(sensor_id), str(camera_id))
        except Exception:
            return None

    def _frame_source_size(self, frame_meta: Any, calib: Any | None = None) -> Tuple[int, int]:
        try:
            frame_w = int(
                _meta_lookup(
                    frame_meta,
                    "frame_width",
                    "width",
                    "source_width",
                    "source_frame_width",
                    default=0,
                )
                or 0
            )
            frame_h = int(
                _meta_lookup(
                    frame_meta,
                    "frame_height",
                    "height",
                    "source_height",
                    "source_frame_height",
                    default=0,
                )
                or 0
            )
        except Exception:
            frame_w = 0
            frame_h = 0
        if (frame_w <= 0 or frame_h <= 0) and calib is not None:
            try:
                frame_w, frame_h = calib.image_size
                frame_w = int(frame_w or 0)
                frame_h = int(frame_h or 0)
            except Exception:
                frame_w = 0
                frame_h = 0
        if frame_w <= 0 or frame_h <= 0:
            try:
                frame_w, frame_h = getattr(self.pipeline, "frame_size", (0, 0))
                frame_w = int(frame_w or 0)
                frame_h = int(frame_h or 0)
            except Exception:
                frame_w = 0
                frame_h = 0
        return max(0, int(frame_w)), max(0, int(frame_h))

    def _frame_compositor_rect(self, frame_meta: Any) -> Optional[Tuple[float, float, float, float]]:
        rect = _meta_lookup(frame_meta, "compositor_rect", default=None)
        if rect is None:
            return None
        try:
            left = float(_meta_lookup(rect, "left", "x", default=0.0) or 0.0)
            top = float(_meta_lookup(rect, "top", "y", default=0.0) or 0.0)
            width = float(_meta_lookup(rect, "width", "w", default=0.0) or 0.0)
            height = float(_meta_lookup(rect, "height", "h", default=0.0) or 0.0)
        except Exception:
            return None
        if width <= 0.0 or height <= 0.0:
            return None
        return float(left), float(top), float(width), float(height)

    @staticmethod
    def _normalize_image_size(value: Any) -> Optional[Tuple[int, int]]:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return None
        try:
            width = int(value[0])
            height = int(value[1])
        except Exception:
            return None
        if width <= 0 or height <= 0:
            return None
        return int(width), int(height)

    def _tiler_grid(self) -> Optional[Tuple[int, int]]:
        tiler = self.pipeline.components.get("tiler")
        cfg = tiler.config if tiler is not None and isinstance(tiler.config, Mapping) else {}
        try:
            cols = int(cfg.get("columns", 0) or 0)
            rows = int(cfg.get("rows", 0) or 0)
        except Exception:
            cols = 0
            rows = 0
        if cols <= 0 or rows <= 0:
            # The canonical three-camera graph is source-id ordered and uses a
            # 3x1 layout.  Retain a useful deterministic fallback for tests or
            # metadata paths that omit the plugin's layout fields.
            try:
                source_count = max(
                    len(getattr(self.pipeline, "camera_labels", {}) or {}),
                    int(getattr(self.pipeline, "batch_size", 0) or 0),
                )
            except Exception:
                source_count = 0
            if source_count <= 1:
                cols, rows = 1, 1
            else:
                cols = 3
                rows = max(1, int(math.ceil(float(source_count) / float(cols))))
        return int(cols), int(rows)

    def _source_to_mosaic(
        self,
        frame_meta: Any,
        u: float,
        v: float,
        source_size: Tuple[int, int],
    ) -> Optional[Tuple[float, float]]:
        """Map a source-image pixel into the post-tiler mosaic.

        Source validity is checked before applying either an explicit compositor
        rectangle or the configured source-id tile.  Invalid coordinates must
        be rejected, never clipped to an apparently plausible mosaic edge.
        """
        try:
            src_w, src_h = int(source_size[0]), int(source_size[1])
            u_f, v_f = float(u), float(v)
        except Exception:
            return None
        if src_w <= 0 or src_h <= 0 or not math.isfinite(u_f) or not math.isfinite(v_f):
            return None
        if u_f < 0.0 or u_f >= float(src_w) or v_f < 0.0 or v_f >= float(src_h):
            return None

        comp = self._frame_compositor_rect(frame_meta)
        if comp is not None:
            left, top, width, height = comp
            x = float(left) + u_f * (float(width) / float(src_w))
            y = float(top) + v_f * (float(height) / float(src_h))
        else:
            mosaic_w, mosaic_h = self._mosaic_size
            grid = self._tiler_grid()
            if grid is None or mosaic_w <= 0 or mosaic_h <= 0:
                return float(u_f), float(v_f)
            cols, rows = grid
            source_id = self._frame_source_id(frame_meta)
            if source_id < 0 or source_id >= cols * rows:
                return None
            tile_w = float(mosaic_w) / float(cols)
            tile_h = float(mosaic_h) / float(rows)
            x = float(source_id % cols) * tile_w + u_f * (tile_w / float(src_w))
            y = float(source_id // cols) * tile_h + v_f * (tile_h / float(src_h))

        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        mosaic_w, mosaic_h = self._mosaic_size
        if mosaic_w > 0 and (x < 0.0 or x >= float(mosaic_w)):
            return None
        if mosaic_h > 0 and (y < 0.0 or y >= float(mosaic_h)):
            return None
        return float(x), float(y)

    @staticmethod
    def _apply_image_flip(
        u: float,
        v: float,
        width: int,
        height: int,
        flip_u: bool,
        flip_v: bool,
    ) -> Tuple[float, float]:
        if flip_u:
            u = float(max(0, int(width) - 1)) - float(u)
        if flip_v:
            v = float(max(0, int(height) - 1)) - float(v)
        return float(u), float(v)

    def _infer_image_flips(self, camera_id: str, calib: Any) -> Tuple[bool, bool]:
        return False, False

    def _ray_floor_hit(
        self,
        calib: Any,
        u: float,
        v: float,
        *,
        flip_u: bool,
        flip_v: bool,
    ) -> Optional[np.ndarray]:
        try:
            width_src, height_src = calib.image_size
            u_ray, v_ray = self._apply_image_flip(u, v, int(width_src), int(height_src), bool(flip_u), bool(flip_v))
            R_wc, C_world = parse_extrinsics(calib.extrinsics_col_major)
            scene_per_m = 1.0
            try:
                s_obj_to_m = float(calib.unit_scale or 1.0)
                if math.isfinite(s_obj_to_m) and s_obj_to_m > 1e-6:
                    scene_per_m = 1.0 / s_obj_to_m
            except Exception:
                scene_per_m = 1.0
            C_world = C_world * scene_per_m
            plane = Plane.horizontal(float(calib.floor_y))
            origin, direction = ray_from_pixel(u_ray, v_ray, calib.intrinsics, R_wc, C_world)
            hit = intersect_plane(origin, direction, plane)
            if hit is None:
                return None
            return np.asarray(hit, dtype=np.float64)
        except Exception:
            return None

    def _bbox_bottom_world(
        self,
        calib: Any,
        bbox: Sequence[float],
        *,
        flip_u: bool,
        flip_v: bool,
    ) -> Optional[np.ndarray]:
        if len(bbox) < 4:
            return None
        try:
            left, top, width, height = [float(x) for x in bbox[:4]]
        except Exception:
            return None
        if width <= 0.0 or height <= 0.0:
            return None
        return self._ray_floor_hit(
            calib,
            float(left) + float(width) * 0.5,
            float(top) + float(height),
            flip_u=flip_u,
            flip_v=flip_v,
        )

    def _update_height_peak(self, state: _TrailTrackState, height_px: float) -> None:
        height_px = float(max(0.0, height_px))
        if height_px <= 0.0:
            return
        if state.height_peak_px is None:
            state.height_peak_px = float(height_px)
            return
        alpha = float(self.config.height_peak_up_alpha) if height_px >= float(state.height_peak_px) else float(self.config.height_peak_down_alpha)
        state.height_peak_px = float(state.height_peak_px + alpha * (height_px - float(state.height_peak_px)))

    def _maybe_update_height_reference(
        self,
        state: _TrailTrackState,
        calib: Any,
        bbox: Sequence[float],
        foot_world: np.ndarray,
        *,
        flip_u: bool,
        flip_v: bool,
    ) -> None:
        if len(bbox) < 4:
            return
        try:
            left, top, width, height = [float(x) for x in bbox[:4]]
        except Exception:
            return
        if width <= 0.0 or height <= 0.0:
            return
        self._update_height_peak(state, height)
        peak = float(state.height_peak_px or 0.0)
        if peak <= 0.0:
            return
        if float(height) < (float(self.config.height_good_frame_ratio) * peak):
            return
        u_top = float(left) + float(width) * 0.5
        v_top = float(top)
        try:
            est_height = estimate_upright_height_from_top_and_foot(
                u_top,
                v_top,
                foot_world,
                calib.intrinsics,
                calib.extrinsics_col_major,
                float(calib.floor_y),
                tuple(int(x) for x in calib.image_size),
                unit_scale=1.0,
                flip_u=bool(flip_u),
                flip_v=bool(flip_v),
            )
        except Exception:
            est_height = None
        if est_height is None or not math.isfinite(float(est_height)) or float(est_height) <= 0.0:
            return
        if state.height_ref_scene is None:
            state.height_ref_scene = float(est_height)
            return
        # Good frames slowly re-lock the detector-height estimate without chasing jitter.
        state.height_ref_scene = float(state.height_ref_scene + 0.20 * (float(est_height) - float(state.height_ref_scene)))

    def _gravity_drop_world(
        self,
        calib: Any,
        bbox: Sequence[float],
        height_ref_scene: float,
        *,
        flip_u: bool,
        flip_v: bool,
    ) -> Optional[np.ndarray]:
        if len(bbox) < 4:
            return None
        try:
            left, top, width, _height = [float(x) for x in bbox[:4]]
        except Exception:
            return None
        if width <= 0.0 or float(height_ref_scene) <= 0.0:
            return None
        try:
            width_src, height_src = calib.image_size
            u_top = float(left) + float(width) * 0.5
            v_top = float(top)
            u_ray, v_ray = self._apply_image_flip(u_top, v_top, int(width_src), int(height_src), bool(flip_u), bool(flip_v))
            R_wc, C_world = parse_extrinsics(calib.extrinsics_col_major)
            scene_per_m = 1.0
            try:
                s_obj_to_m = float(calib.unit_scale or 1.0)
                if math.isfinite(s_obj_to_m) and s_obj_to_m > 1e-6:
                    scene_per_m = 1.0 / s_obj_to_m
            except Exception:
                scene_per_m = 1.0
            C_world = C_world * scene_per_m
            origin, direction = ray_from_pixel(u_ray, v_ray, calib.intrinsics, R_wc, C_world)
            denom = float(direction[1])
            if abs(denom) < 1e-9:
                return None
            plane_y = float(calib.floor_y) + float(height_ref_scene)
            t = (plane_y - float(origin[1])) / denom
            if not math.isfinite(t) or t <= 0.0:
                return None
            head = origin + (direction * t)
            return np.array([float(head[0]), float(calib.floor_y), float(head[2])], dtype=np.float64)
        except Exception:
            return None

    def _update_world_measurement(self, state: _TrailTrackState, world_x: float, world_z: float, now: float) -> None:
        prev_ts = float(state.last_measure_ts or 0.0)
        prev_x = state.last_measure_world_x
        prev_z = state.last_measure_world_z
        if prev_x is not None and prev_z is not None and prev_ts > 0.0 and float(now) > prev_ts:
            dt = float(now) - prev_ts
            if dt > 1e-6:
                vx = (float(world_x) - float(prev_x)) / dt
                vz = (float(world_z) - float(prev_z)) / dt
                state.vel_world_x = float(vx)
                state.vel_world_z = float(vz)
                state.last_measure_speed = float(math.hypot(vx, vz))
        state.last_measure_world_x = float(world_x)
        state.last_measure_world_z = float(world_z)
        state.last_measure_ts = float(now)

    def _predict_gap_anchor(
        self,
        sensor_id: int,
        frame_meta: Any,
        state: _TrailTrackState,
        now: float,
    ) -> Optional[Tuple[float, float]]:
        if self.config.anchor_mode != "floor_plane_gravity_drop":
            return None
        last_ts = float(state.last_measure_ts or 0.0)
        if last_ts <= 0.0:
            return None
        dt = float(now) - last_ts
        if dt <= 0.0 or dt > float(self.config.gap_predict_ttl_s):
            return None
        world_x = state.last_measure_world_x
        world_z = state.last_measure_world_z
        if world_x is None or world_z is None:
            return None
        camera_id = self._camera_id_for_sensor(sensor_id)
        calib = self._resolve_calibration(sensor_id, camera_id)
        if calib is None or getattr(calib, "intrinsics", None) is None or getattr(calib, "extrinsics_col_major", None) is None:
            return None
        tau = float(self.config.gap_predict_decay_tau_s)
        travel_scale = dt
        if tau > 1e-6:
            travel_scale = float(tau * (1.0 - math.exp(-dt / tau)))
        pred_world = np.array(
            [
                float(world_x) + float(state.vel_world_x) * travel_scale,
                float(calib.floor_y),
                float(world_z) + float(state.vel_world_z) * travel_scale,
            ],
            dtype=np.float64,
        )
        flip_u, flip_v = self._infer_image_flips(camera_id, calib)
        uv = project_world_to_image(
            pred_world,
            calib.intrinsics,
            calib.extrinsics_col_major,
            tuple(int(x) for x in calib.image_size),
            unit_scale=1.0,
            flip_u=bool(flip_u),
            flip_v=bool(flip_v),
        )
        if uv is None:
            return None
        source_size = self._frame_source_size(frame_meta, calib)
        calib_size = self._normalize_image_size(getattr(calib, "image_size", None))
        if calib_size is not None and source_size[0] > 0 and source_size[1] > 0:
            uv = (
                float(uv[0]) * float(source_size[0]) / float(calib_size[0]),
                float(uv[1]) * float(source_size[1]) / float(calib_size[1]),
            )
        return self._source_to_mosaic(frame_meta, float(uv[0]), float(uv[1]), source_size)

    @staticmethod
    def _anchor_basis_token(
        track: Mapping[str, Any],
        source_size: Tuple[int, int],
        anchor_key: str,
        *,
        reprojected: bool,
    ) -> Tuple[Any, ...]:
        explicit_basis = None
        for key in (
            "anchor_coordinate_basis",
            "image_coordinate_basis",
            "image_basis",
            "coordinate_basis",
        ):
            if track.get(key) is not None:
                explicit_basis = track.get(key)
                break
        if isinstance(explicit_basis, Mapping):
            explicit_basis = tuple(
                sorted((str(key), repr(value)) for key, value in explicit_basis.items())
            )
        elif explicit_basis is not None:
            explicit_basis = str(explicit_basis)
        return (
            str(anchor_key),
            explicit_basis,
            (int(source_size[0]), int(source_size[1])),
            str(track.get("world_frame") or ""),
            str(track.get("world_frame_revision") or ""),
            "world_snapshot" if reprojected else "track_image",
        )

    @staticmethod
    def _reset_trail_state(state: _TrailTrackState) -> None:
        state.points.clear()
        state.ema_x = None
        state.ema_y = None
        state.ema_ts = 0.0
        state.last_measure_world_x = None
        state.last_measure_world_z = None
        state.last_measure_ts = 0.0
        state.vel_world_x = 0.0
        state.vel_world_z = 0.0
        state.last_measure_speed = 0.0
        state.anchor_basis = None

    def _resolve_active_anchor(
        self,
        sensor_id: int,
        frame_meta: Any,
        obj_meta: Any,
        track_id: int,
        state: _TrailTrackState,
        now: float,
        *,
        track_info: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Tuple[float, float, bool]]:
        rect = getattr(obj_meta, "rect_params", None)
        if rect is None:
            return None
        left = float(getattr(rect, "left", 0.0) or 0.0)
        top = float(getattr(rect, "top", 0.0) or 0.0)
        width = float(getattr(rect, "width", 0.0) or 0.0)
        height = float(getattr(rect, "height", 0.0) or 0.0)
        if not all(math.isfinite(value) for value in (left, top, width, height)):
            return None
        if width <= 1.0 or height <= 1.0:
            return None
        x_bbox = float(left) + float(width) * 0.5
        y_bbox = float(top) + float(height)
        if self.config.anchor_mode != "floor_plane_gravity_drop":
            source_size = self._frame_source_size(frame_meta)
            mapped = self._source_to_mosaic(frame_meta, x_bbox, y_bbox, source_size)
            if mapped is None:
                return None
            return float(mapped[0]), float(mapped[1]), False

        # Floor trails are a projection of the exact canonical tracking row.
        # A missing row (or a row from another frame) is not permission to
        # switch to a bbox-bottom anchor: that would create a second estimator
        # and is the source of the visible lines-to-frame-edge failure.
        track = track_info
        if not isinstance(track, Mapping):
            return None

        measured_world = track.get("world")
        if not isinstance(measured_world, (list, tuple)) or len(measured_world) < 3 or track.get("world_valid") is not True:
            return None
        try:
            measured_world_arr = np.asarray(
                [float(measured_world[0]), float(measured_world[1]), float(measured_world[2])],
                dtype=np.float64,
            )
        except Exception:
            return None
        if not np.all(np.isfinite(measured_world_arr)):
            return None

        # The world row is the sole floor-mode render authority.  In
        # particular, do not prefer ``image_base``/``image_foot`` here: those
        # values can be produced by a different detector/coordinate basis and
        # would make the OSD trail disagree with the BEV point for the same
        # exact tracking cohort.  They remain useful diagnostic metadata, but
        # the active world snapshot must be reprojected for display.
        camera_id = self._camera_id_for_sensor(sensor_id)
        calib = self._resolve_calibration(sensor_id, camera_id)
        if calib is None or getattr(calib, "intrinsics", None) is None or getattr(calib, "extrinsics_col_major", None) is None:
            return None
        # The canonical world row is only meaningful under the active
        # calibration snapshot. A revisioned reload must not reproject an
        # older row, and must not retain a prior trail segment while waiting
        # for a compatible row.
        if not world_frame_matches_calibration(
            track,
            calib,
            default_frame_id="backend_world_m",
        ):
            self._reset_trail_state(state)
            return None
        try:
            calib_image_size = tuple(int(x) for x in calib.image_size)
        except Exception:
            return None
        if len(calib_image_size) < 2 or calib_image_size[0] <= 0 or calib_image_size[1] <= 0:
            return None
        flip_u, flip_v = self._infer_image_flips(camera_id, calib)
        uv = project_world_to_image(
            measured_world_arr,
            calib.intrinsics,
            calib.extrinsics_col_major,
            calib_image_size,
            unit_scale=1.0,
            flip_u=bool(flip_u),
            flip_v=bool(flip_v),
        )
        if uv is None:
            return None
        source_uv = (float(uv[0]), float(uv[1]))
        source_key = "world_reprojection"
        reprojected = True

        source_size = self._normalize_image_size(track.get("image_size") or track.get("frame_size"))
        if source_size is None:
            source_size = self._frame_source_size(frame_meta)
        if source_size[0] <= 0 or source_size[1] <= 0:
            return None

        calib_size = self._normalize_image_size(getattr(calib, "image_size", None))
        if calib_size is None or calib_size[0] <= 0 or calib_size[1] <= 0:
            return None
        source_uv = (
            float(source_uv[0]) * float(source_size[0]) / float(calib_size[0]),
            float(source_uv[1]) * float(source_size[1]) / float(calib_size[1]),
        )

        basis = self._anchor_basis_token(
            track,
            source_size,
            str(source_key or "canonical"),
            reprojected=bool(reprojected),
        )
        if state.anchor_basis is not None and state.anchor_basis != basis:
            self._reset_trail_state(state)
        state.anchor_basis = basis

        self._update_world_measurement(state, float(measured_world_arr[0]), float(measured_world_arr[2]), float(now))
        mapped = self._source_to_mosaic(frame_meta, float(source_uv[0]), float(source_uv[1]), source_size)
        if mapped is None:
            return None
        return float(mapped[0]), float(mapped[1]), False

    def _commit_point(
        self,
        state: _TrailTrackState,
        now: float,
        x: float,
        y: float,
        *,
        predicted: bool,
        append_allowed: bool = True,
        canonical_floor: bool = False,
    ) -> None:
        # Canonical floor anchors are already filtered by PersonGroundState and
        # projected through the active world calibration.  Applying another
        # speed limiter/EMA here makes OSD a lagging second estimator and can
        # draw a long synthetic segment when the exact frame join changes.
        if state.points and not canonical_floor:
            prev = state.points[-1]
            dt = max(0.0, float(now) - float(prev.ts))
            if dt > 0.0 and self.config.max_speed_px_per_s > 0.0:
                dx = float(x) - float(prev.x)
                dy = float(y) - float(prev.y)
                dist = math.hypot(dx, dy)
                max_step = float(self.config.max_speed_px_per_s) * dt
                if max_step > 0.0 and dist > max_step:
                    scale = max_step / dist
                    x = float(prev.x) + dx * scale
                    y = float(prev.y) + dy * scale

        if self.config.smooth_tau_s > 0.0 and not canonical_floor:
            if state.ema_x is None or state.ema_y is None:
                state.ema_x, state.ema_y = float(x), float(y)
                state.ema_ts = float(now)
            else:
                dt_ema = max(0.0, float(now) - float(state.ema_ts))
                tau = float(self.config.smooth_tau_s)
                alpha = 1.0 - math.exp(-dt_ema / tau) if (tau > 0.0 and dt_ema > 0.0) else 1.0
                state.ema_x = float(state.ema_x + alpha * (float(x) - float(state.ema_x)))
                state.ema_y = float(state.ema_y + alpha * (float(y) - float(state.ema_y)))
                state.ema_ts = float(now)
            x, y = float(state.ema_x), float(state.ema_y)

        # Phase 1: stationary people do not grow OSD trail history.
        if not append_allowed:
            if state.points:
                prev = state.points[-1]
                state.points[-1] = _TrailPoint(
                    ts=float(prev.ts),
                    x=float(x),
                    y=float(y),
                    predicted=bool(predicted),
                )
            return

        # Phase 6: image-space path commit with min-step + optional RDP.
        def _factory(*, ts: float, x: float, y: float) -> _TrailPoint:
            return _TrailPoint(ts=float(ts), x=float(x), y=float(y), predicted=bool(predicted))

        if state.points:
            prev = state.points[-1]
            dt = max(0.0, float(now) - float(prev.ts))
            if dt < float(self.config.min_dt_s):
                commit_image_path_point(
                    state.points,
                    ts=float(prev.ts),
                    x=float(x),
                    y=float(y),
                    min_step_px=float(self.config.min_step_px),
                    simplify_epsilon_px=max(1.0, float(self.config.min_step_px) * 1.5),
                    max_points=max(2, int(self.config.max_points_per_track)),
                    append_allowed=True,
                    point_factory=_factory,
                )
                return

        commit_image_path_point(
            state.points,
            ts=float(now),
            x=float(x),
            y=float(y),
            min_step_px=float(self.config.min_step_px),
            simplify_epsilon_px=max(1.0, float(self.config.min_step_px) * 1.5),
            max_points=max(2, int(self.config.max_points_per_track)),
            append_allowed=True,
            point_factory=_factory,
        )

    def handle_batch_ds8(self, batch_meta: Any) -> None:
        if ds_osd is None:
            return

        frame_items = getattr(batch_meta, "frame_items", None)
        if frame_items is None:
            return

        now = time.time()
        debug = os.environ.get("NOESIS_TRAILS_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")
        with self._lock:
            if not self._enabled:
                return

            for frame_meta in frame_items:
                if not self._logged_priors:
                    try:
                        logger.debug(
                            "Trail overlay priors: frame_meta=%s has_batch_meta=%s has_compositor_rect=%s has_object_items=%s",
                            type(frame_meta).__name__,
                            hasattr(frame_meta, "batch_meta"),
                            hasattr(frame_meta, "compositor_rect"),
                            hasattr(frame_meta, "object_items"),
                        )
                    except Exception:
                        pass
                self._logged_priors = True
                lines, tracks_seen, tracks_drawn = self._handle_frame(
                    frame_meta=frame_meta,
                    batch_meta=batch_meta,
                    now=now,
                    debug=debug,
                )
                if debug:
                    self._debug_frames += 1
                    self._debug_lines += int(lines)
                    self._debug_tracks_seen += int(tracks_seen)
                    self._debug_tracks_drawn += int(tracks_drawn)
                    if (now - float(self._debug_last_log_ts)) >= 1.0:
                        logger.info(
                            "TrailOverlay: frames=%d lines=%d tracks_seen=%d tracks_drawn=%d enabled=%s window_s=%.2f stride=%d",
                            self._debug_frames,
                            self._debug_lines,
                            self._debug_tracks_seen,
                            self._debug_tracks_drawn,
                            self._enabled,
                            float(self.config.window_s),
                            int(self.config.draw_stride),
                        )
                        self._debug_frames = 0
                        self._debug_lines = 0
                        self._debug_tracks_seen = 0
                        self._debug_tracks_drawn = 0
                        self._debug_last_log_ts = float(now)

    def _handle_frame(self, frame_meta: Any, batch_meta: Any, now: float, *, debug: bool = False) -> Tuple[int, int, int]:
        sensor_id = self._frame_source_id(frame_meta)

        # Update per-sensor frame counter for stride decisions.
        frame_idx = int(self._frame_counts.get(sensor_id, 0) + 1)
        self._frame_counts[sensor_id] = frame_idx
        do_sample = (frame_idx % int(self.config.draw_stride)) == 0

        object_items = getattr(frame_meta, "object_items", None) or []
        stable_mgr = getattr(self.pipeline, "stable_id_mgr", None)

        sensor_tracks = self._tracks.setdefault(sensor_id, {})

        mosaic_w, mosaic_h = self._mosaic_size
        max_points_per_track = max(2, int(self.config.max_points_per_track))
        tracks_seen = 0
        present_track_ids: set[int] = set()
        frame_id = self._frame_id(frame_meta)
        # Fetch the latest cache once.  The helper discards every row whose
        # publication frame is not exactly this frame.
        track_map = (
            self._analytics_track_map(sensor_id, frame_id)
            if frame_id is not None
            else {}
        )

        # Update trail state for all configured objects.
        for obj_meta in object_items:
            try:
                class_id = int(getattr(obj_meta, "class_id", -1))
            except Exception:
                class_id = -1
            if class_id not in self.config.class_ids:
                continue

            try:
                track_id = int(getattr(obj_meta, "object_id", -1))
            except Exception:
                track_id = -1
            if track_id < 0:
                continue
            tracks_seen += 1
            present_track_ids.add(int(track_id))

            rect = getattr(obj_meta, "rect_params", None)
            if rect is None:
                continue
            try:
                left = float(getattr(rect, "left", 0.0) or 0.0)
                top = float(getattr(rect, "top", 0.0) or 0.0)
                width = float(getattr(rect, "width", 0.0) or 0.0)
                height = float(getattr(rect, "height", 0.0) or 0.0)
            except Exception:
                continue
            if width <= 1.0 or height <= 1.0:
                continue

            track_info = track_map.get(int(track_id))
            if not isinstance(track_info, Mapping):
                # A detector/SDK object without the exact canonical analytics
                # row is not drawable trail evidence.  Clear any prior points
                # immediately and, crucially, do not refresh liveness; doing
                # so kept stale trails visible and connected their return to a
                # new bbox near the bottom of the mosaic.
                stale_state = sensor_tracks.get(track_id)
                if stale_state is not None:
                    self._reset_trail_state(stale_state)
                    stale_state.trail_segment_id = None
                continue

            state = sensor_tracks.get(track_id)
            if state is None:
                state = _TrailTrackState(points=deque(maxlen=max_points_per_track))
                sensor_tracks[track_id] = state

            generation = track_info.get("tracker_lifecycle_generation")
            try:
                generation_int = int(generation) if generation is not None else None
            except Exception:
                generation_int = None
            if (
                generation_int is not None
                and state.tracker_lifecycle_generation is not None
                and generation_int != int(state.tracker_lifecycle_generation)
            ):
                self._reset_trail_state(state)
                state.trail_segment_id = None
            if generation_int is not None:
                state.tracker_lifecycle_generation = generation_int

            # A non-sampled frame still gets a cheap validity check.  If the
            # exact canonical row is already invalid, stale points must break
            # immediately; otherwise a later valid row reconnects to a bad
            # segment.  Full projection remains decimated by draw_stride.
            if not do_sample:
                if self.config.anchor_mode == "floor_plane_gravity_drop":
                    world = track_info.get("world")
                    invalid_world = track_info.get("world_valid") is not True
                    if isinstance(world, (list, tuple)) and len(world) >= 3:
                        try:
                            invalid_world = invalid_world or not all(
                                math.isfinite(float(world[index]))
                                for index in range(3)
                            )
                        except (TypeError, ValueError):
                            invalid_world = True
                    if invalid_world:
                        self._reset_trail_state(state)
                        state.trail_segment_id = None
                continue

            # Always prune old samples so disappeared tracks naturally fade out.
            while state.points and (now - float(state.points[0].ts)) > float(self.config.window_s):
                state.points.popleft()

            if not do_sample:
                continue

            resolved = self._resolve_active_anchor(
                sensor_id,
                frame_meta,
                obj_meta,
                int(track_id),
                state,
                float(now),
                track_info=track_info,
            )
            # No valid exact-frame canonical anchor means no new OSD point.
            # In particular, never substitute the current bbox in floor mode.
            if resolved is None:
                self._reset_trail_state(state)
                state.trail_segment_id = None
                continue
            # Liveness is refreshed only after an exact canonical anchor has
            # resolved.  Missing/invalid rows therefore cannot keep stale
            # trail state alive.
            state.last_seen_ts = float(now)
            x, y, predicted = resolved
            append_allowed = True
            if isinstance(track_info, Mapping):
                segment_id = track_info.get("trail_segment_id")
                segment_changed = (
                    segment_id is not None
                    and state.trail_segment_id is not None
                    and int(segment_id) != int(state.trail_segment_id)
                )
                if bool(track_info.get("trail_break_required", False)) or segment_changed:
                    state.points.clear()
                    state.ema_x = None
                    state.ema_y = None
                    state.ema_ts = 0.0
                if segment_id is not None:
                    state.trail_segment_id = int(segment_id)
                if "trail_append_allowed" in track_info:
                    append_allowed = bool(track_info.get("trail_append_allowed"))
                if str(track_info.get("motion_mode") or "").strip().lower() in ("idle", "sit", "lie"):
                    append_allowed = False
            self._commit_point(
                state,
                float(now),
                float(x),
                float(y),
                predicted=bool(predicted),
                append_allowed=append_allowed,
                canonical_floor=bool(
                    self.config.anchor_mode == "floor_plane_gravity_drop"
                    and not predicted
                ),
            )

        # An exact canonical snapshot is also authoritative when the SDK no
        # longer carries an object row.  Break retained history for every
        # lifecycle absent from this frame before gap prediction or rendering;
        # otherwise a compatible same-generation return would reconnect to the
        # last point on the far side of the exact disappearance cohort.
        exact_track_ids = {int(track_id) for track_id in track_map}
        for track_id, state in sensor_tracks.items():
            if int(track_id) in exact_track_ids:
                continue
            self._reset_trail_state(state)
            state.trail_segment_id = None

        if do_sample and self.config.anchor_mode == "floor_plane_gravity_drop":
            for track_id, state in sensor_tracks.items():
                if int(track_id) in present_track_ids:
                    continue
                x_y = self._predict_gap_anchor(sensor_id, frame_meta, state, float(now))
                if x_y is None:
                    continue
                x, y = x_y
                self._commit_point(state, float(now), float(x), float(y), predicted=True)

        if os.environ.get("NOESIS_TRAILS_RENDER", "1").strip().lower() in ("0", "false", "no", "off"):
            return 0, tracks_seen, 0

        # Render trails for this sensor into the mosaic using display meta.
        acquire_display_meta = getattr(batch_meta, "acquire_display_meta", None)
        if not callable(acquire_display_meta):
            return 0, tracks_seen, 0
        append_meta = getattr(frame_meta, "append", None)
        if not callable(append_meta):
            return 0, tracks_seen, 0

        # Remove fully expired tracks to keep memory bounded.
        expired: List[int] = []
        for track_id, state in sensor_tracks.items():
            while state.points and (now - float(state.points[0].ts)) > float(self.config.window_s):
                state.points.popleft()
            if not state.points and (now - float(state.last_seen_ts)) > float(self.config.window_s):
                expired.append(track_id)
        for track_id in expired:
            sensor_tracks.pop(track_id, None)

        if not sensor_tracks:
            return 0, tracks_seen, 0

        # Build one or more DisplayMeta blocks as needed to fit the line budget.
        max_lines_total = int(self.config.max_lines)
        max_metas = int(self.config.max_display_metas)
        line_width = int(self.config.line_width)
        min_alpha = float(self.config.min_alpha)
        show_labels = bool(self.config.show_labels)
        label_budget = 16

        display_metas: List[Any] = []
        allocated_ids: set[int] = set()
        appended_ids: set[int] = set()
        current: Any = None
        lines_used = 0
        metas_exhausted = False
        tracks_drawn = 0

        def _append_display_meta(dm: Any) -> None:
            dm_id = id(dm)
            if dm_id in appended_ids:
                return
            try:
                append_meta(dm)
            except Exception:
                return
            appended_ids.add(dm_id)

        def _alloc() -> Optional[Any]:
            try:
                dm = acquire_display_meta()
            except Exception:
                return None
            if not dm:
                return None
            dm_id = id(dm)
            if dm_id in allocated_ids:
                return None
            allocated_ids.add(dm_id)
            display_metas.append(dm)
            return dm

        def _ensure_meta() -> Optional[Any]:
            nonlocal current
            if current is None:
                current = _alloc()
            return current

        def _line_count(dm: Any) -> Optional[int]:
            try:
                return int(getattr(dm, "n_lines"))
            except Exception:
                return None

        def _label_count(dm: Any) -> Optional[int]:
            try:
                return int(getattr(dm, "n_labels"))
            except Exception:
                return None

        def _add_line(dm: Any, line: Any) -> bool:
            before = _line_count(dm)
            try:
                dm.add_line(line)
            except Exception:
                return False
            if before is None:
                return True
            after = _line_count(dm)
            if after is None:
                return True
            return after > before

        def _add_text(dm: Any, text: Any) -> bool:
            before = _label_count(dm)
            try:
                dm.add_text(text)
            except Exception:
                return False
            if before is None:
                return True
            after = _label_count(dm)
            if after is None:
                return True
            return after > before

        def _resample_points(points: List[_TrailPoint], segments_budget: int, *, bias: float = 2.0) -> List[_TrailPoint]:
            if segments_budget <= 0:
                return []
            segments_available = len(points) - 1
            if segments_available <= segments_budget:
                return points
            # Select (segments_budget + 1) indices, biased towards the newest points.
            selected: List[_TrailPoint] = []
            last_idx = -1
            for j in range(segments_budget + 1):
                t = 0.0 if segments_budget == 0 else (float(j) / float(segments_budget))
                raw = int(round((t**bias) * float(segments_available)))
                # Clamp so we always have enough remaining room for strictly increasing indices.
                min_idx = j
                max_idx = segments_available - (segments_budget - j)
                idx = max(min_idx, min(max_idx, raw))
                if idx <= last_idx:
                    idx = min(max_idx, last_idx + 1)
                selected.append(points[idx])
                last_idx = idx
            return selected

        inv_window = 1.0 / max(0.1, float(self.config.window_s))

        # Draw newest→older segments so we never "freeze" the head when we hit
        # display meta capacity; newest motion always wins.
        track_items = list(sensor_tracks.items())
        track_items.sort(
            key=lambda item: float(item[1].points[-1].ts if item[1].points else getattr(item[1], "last_seen_ts", 0.0)),
            reverse=True,
        )
        max_tracks = max(1, int(self.config.max_tracks))
        track_items = track_items[:max_tracks]
        remaining_tracks = len(track_items)
        for track_id, state in track_items:
            pts = list(state.points)
            if len(pts) < 2 or lines_used >= max_lines_total:
                remaining_tracks = max(0, remaining_tracks - 1)
                continue
            segments_available = len(pts) - 1
            remaining_global = max(0, max_lines_total - lines_used)
            per_track_cap = int(self.config.max_segments_per_track)
            if remaining_tracks > 0:
                per_track_cap = min(per_track_cap, max(1, remaining_global // remaining_tracks))
            segments_budget = min(segments_available, per_track_cap, remaining_global)
            if segments_budget <= 0:
                remaining_tracks = max(0, remaining_tracks - 1)
                continue
            pts = _resample_points(pts, segments_budget)

            stable_id = None
            if stable_mgr is not None:
                try:
                    stable_id = stable_mgr.active_tracks.get((int(sensor_id), int(track_id)), {}).get("stable_id")
                except Exception:
                    stable_id = None

            stable_id_int = None
            try:
                stable_id_int = int(stable_id)
            except Exception:
                stable_id_int = None

            key_id = int(track_id)
            if self.config.color_key == "stable_id":
                if stable_id_int is not None and stable_id_int > 0:
                    key_id = int(stable_id_int)
                else:
                    key_id = ((int(sensor_id) + 1) << 32) + int(track_id)
            r, g, b = self._color_for_key(key_id)

            line = ds_osd.Line()
            line.width = line_width
            line.color.r = float(r)
            line.color.g = float(g)
            line.color.b = float(b)

            drawn_for_track = 0
            for idx in range(len(pts) - 2, -1, -1):
                if drawn_for_track >= segments_budget or lines_used >= max_lines_total or metas_exhausted:
                    break
                dm = _ensure_meta()
                if dm is None:
                    metas_exhausted = True
                    break

                pt0 = pts[idx]
                pt1 = pts[idx + 1]
                age = max(0.0, float(now) - float(pt0.ts))
                t = 1.0 - min(1.0, age * inv_window)
                alpha = min_alpha + (1.0 - min_alpha) * max(0.0, min(1.0, t))
                if pt0.predicted or pt1.predicted:
                    alpha *= float(self.config.predicted_alpha_scale)

                line.x1, line.y1 = int(pt0.x), int(pt0.y)
                line.x2, line.y2 = int(pt1.x), int(pt1.y)
                line.color.a = float(alpha)
                ok = _add_line(dm, line)
                if not ok:
                    _append_display_meta(dm)
                    if len(display_metas) >= max_metas:
                        metas_exhausted = True
                        break
                    current = None
                    dm = _ensure_meta()
                    if dm is None:
                        metas_exhausted = True
                        break
                    ok = _add_line(dm, line)
                if not ok:
                    if dm is not None:
                        _append_display_meta(dm)
                    metas_exhausted = True
                    break
                lines_used += 1
                drawn_for_track += 1
            if drawn_for_track > 0:
                tracks_drawn += 1
            remaining_tracks = max(0, remaining_tracks - 1)

            if not show_labels or label_budget <= 0 or not pts:
                continue
            if metas_exhausted:
                break
            dm = _ensure_meta()
            if dm is None:
                continue

            label_budget -= 1
            last_pt = pts[-1]
            text = ds_osd.Text()
            if stable_id_int is not None and stable_id_int > 0:
                text.display_text = f"sid {stable_id_int}"
            else:
                text.display_text = "sid XX"
            text.x_offset = int(last_pt.x)
            text.y_offset = int(last_pt.y)
            try:
                text.font.name = ds_osd.FontFamily.Serif
                text.font.size = 12
                text.font.color.r = float(r)
                text.font.color.g = float(g)
                text.font.color.b = float(b)
                text.font.color.a = 1.0
            except Exception:
                pass
            try:
                text.set_bg_color = False
            except Exception:
                pass
            try:
                ok = _add_text(dm, text)
            except Exception:
                ok = False
            if not ok:
                _append_display_meta(dm)
                if len(display_metas) >= max_metas:
                    metas_exhausted = True
                    break
                current = None
                dm = _ensure_meta()
                if dm is None:
                    metas_exhausted = True
                    break
                ok = _add_text(dm, text)
                if not ok:
                    if dm is not None:
                        _append_display_meta(dm)
                    metas_exhausted = True
                    break

        # Best-effort fallback: if append-on-alloc failed (API oddities), try again at end.
        for dm in display_metas:
            _append_display_meta(dm)
        return lines_used, tracks_seen, tracks_drawn


_POSE_KPT_INDEX = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


_POSE_SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


@dataclass
class PoseFeatureProcessor:
    pipeline: "DS8Pipeline"
    gie_id: int
    model_size: Tuple[int, int] = (640, 640)
    score_threshold: float = 0.25
    kpt_threshold: float = 0.35
    letterbox: bool = True
    tensor_source: str = "object_sgie"
    model_label: str = "yolo26-pose"
    match_min_iou: float = 0.7
    match_ambiguity_margin: float = 0.05
    camera_labels: Mapping[int, str] = field(default_factory=dict)
    cache_max_age_frames: int = 6
    cache_max_bbox_shift: float = 0.35
    _missing_native_logged: bool = field(default=False, init=False, repr=False)
    _debug_last_log: float = field(default=0.0, init=False, repr=False)
    _debug_frames: int = field(default=0, init=False, repr=False)
    _debug_objects: int = field(default=0, init=False, repr=False)
    _debug_attached: int = field(default=0, init=False, repr=False)
    _debug_missing: int = field(default=0, init=False, repr=False)
    _rfdetr_diag_last_log: float = field(default=0.0, init=False, repr=False)
    _pose_cache: Dict[Tuple[int, int], Dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

    def _features_per_frame_max(self) -> int:
        return _read_env_int("NOESIS_POSE_FEATURES_PER_FRAME_MAX", 2, min_value=0)

    def _frame_source_id(self, frame_meta: Any) -> int:
        for attr in ("source_id", "pad_index", "camera_id"):
            value = getattr(frame_meta, attr, None)
            if value is None:
                continue
            try:
                return int(value)
            except Exception:
                continue
        return 0

    def _frame_id(self, frame_meta: Any) -> int:
        value = _meta_lookup(frame_meta, "frame_number", "frame_num", default=0)
        try:
            return int(value or 0)
        except Exception:
            return 0

    def _frame_timestamp_us(self, frame_meta: Any) -> int:
        pts_ns = int(_meta_lookup(frame_meta, "buf_pts", "buffer_pts", "pts", default=0) or 0)
        if pts_ns <= 0:
            pts_ns = int(time.time() * 1_000_000_000)
        return max(0, pts_ns // 1_000)

    def _pose_cache_max_age_frames(self) -> int:
        raw = os.environ.get("NOESIS_POSE_FEATURE_CACHE_MAX_AGE_FRAMES")
        try:
            if raw is not None and str(raw).strip():
                return max(0, int(str(raw).strip()))
        except Exception:
            pass
        try:
            return max(0, int(self.cache_max_age_frames))
        except Exception:
            return 6

    def _pose_cache_max_bbox_shift(self) -> float:
        raw = os.environ.get("NOESIS_POSE_FEATURE_CACHE_MAX_BBOX_SHIFT")
        try:
            if raw is not None and str(raw).strip():
                return max(0.0, float(str(raw).strip()))
        except Exception:
            pass
        try:
            return max(0.0, float(self.cache_max_bbox_shift))
        except Exception:
            return 0.35

    def _pose_cache_key(self, source_id: int, track_id: int) -> Optional[Tuple[int, int]]:
        if int(track_id) < 0:
            return None
        return int(source_id), int(track_id)

    @staticmethod
    def _bbox_shift_ratio(old_bbox: Sequence[float], new_bbox: Sequence[float]) -> float:
        try:
            old_left, old_top, old_w, old_h = [float(x) for x in old_bbox[:4]]
            new_left, new_top, new_w, new_h = [float(x) for x in new_bbox[:4]]
        except Exception:
            return float("inf")
        diag = math.hypot(max(1.0, old_w), max(1.0, old_h))
        old_cx = old_left + old_w * 0.5
        old_cy = old_top + old_h * 0.5
        new_cx = new_left + new_w * 0.5
        new_cy = new_top + new_h * 0.5
        return float(math.hypot(new_cx - old_cx, new_cy - old_cy) / max(1.0, diag))

    def _stable_id_for_track(self, source_id: int, track_id: int) -> Optional[int]:
        try:
            mgr = getattr(self.pipeline, "stable_id_mgr", None)
            if mgr is not None:
                rec = mgr.active_tracks.get((int(source_id), int(track_id)))
                if rec and rec.get("stable_id") is not None:
                    return int(rec.get("stable_id"))
        except Exception:
            return None
        return None

    def _cache_pose_payload(
        self,
        source_id: int,
        track_id: int,
        bbox: Sequence[float],
        frame_id: int,
        payload: Mapping[str, Any],
    ) -> None:
        key = self._pose_cache_key(source_id, track_id)
        if key is None:
            return
        self._pose_cache[key] = {
            "frame_id": int(frame_id),
            "bbox": [float(x) for x in bbox[:4]],
            "payload": dict(payload),
        }

    def _cached_pose_payload(
        self,
        source_id: int,
        track_id: int,
        bbox: Sequence[float],
        frame_id: int,
        ts_us: int,
        stable_id: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        key = self._pose_cache_key(source_id, track_id)
        if key is None:
            return None
        entry = self._pose_cache.get(key)
        if not isinstance(entry, dict):
            return None
        try:
            cached_frame_id = int(entry.get("frame_id", 0))
            age_frames = int(frame_id) - cached_frame_id
        except Exception:
            age_frames = self._pose_cache_max_age_frames() + 1
        if age_frames < 0 or age_frames > self._pose_cache_max_age_frames():
            self._pose_cache.pop(key, None)
            return None

        old_bbox = entry.get("bbox")
        if not isinstance(old_bbox, (list, tuple)) or len(old_bbox) < 4:
            self._pose_cache.pop(key, None)
            return None
        if self._bbox_shift_ratio(old_bbox, bbox) > self._pose_cache_max_bbox_shift():
            self._pose_cache.pop(key, None)
            return None

        payload = entry.get("payload")
        if not isinstance(payload, Mapping):
            self._pose_cache.pop(key, None)
            return None
        clone: Dict[str, Any] = dict(payload)
        old_left, old_top, old_w, old_h = [float(x) for x in old_bbox[:4]]
        new_left, new_top, new_w, new_h = [float(x) for x in bbox[:4]]
        sx = float(new_w / old_w) if old_w > 1e-6 else 1.0
        sy = float(new_h / old_h) if old_h > 1e-6 else 1.0

        raw_roi = clone.get("keypoints_roi")
        if isinstance(raw_roi, (list, tuple)) and len(raw_roi) >= 17:
            roi_rows: List[List[float]] = []
            abs_rows: List[List[float]] = []
            for item in raw_roi[:17]:
                if not isinstance(item, (list, tuple)) or len(item) < 3:
                    return None
                try:
                    x = float(item[0]) * sx
                    y = float(item[1]) * sy
                    c = float(item[2])
                except Exception:
                    return None
                roi_rows.append([x, y, c])
                abs_rows.append([new_left + x, new_top + y, c])
            clone["keypoints_roi"] = roi_rows
            clone["keypoints_abs"] = abs_rows

        clone["source_id"] = int(source_id)
        clone["frame_id"] = int(frame_id)
        clone["object_id"] = int(track_id)
        clone["bbox"] = [new_left, new_top, new_w, new_h]
        clone["ts_us"] = int(ts_us)
        clone["pose_cache_reused"] = True
        clone["pose_cache_age_frames"] = int(age_frames)
        if stable_id is not None:
            clone["stable_id"] = int(stable_id)
        else:
            clone.pop("stable_id", None)
        return clone

    def _attach_pose_payload(
        self,
        batch_meta: Any,
        attach_obj: Any,
        obj_meta: Any,
        payload: Mapping[str, Any],
    ) -> bool:
        strict = self.tensor_source == "rfdetr_pgie_frame"
        if attach_obj is None:
            if strict:
                raise RuntimeError(
                    "RF-DETR keypoint metadata attach symbol is unavailable"
                )
            return False
        try:
            payload_json = _serialize_compact_json_with_metrics(
                dict(payload),
                metric="pose_features.user_meta_json",
            )
            payload_bytes = len(payload_json.encode("utf-8"))
            if payload_bytes > _pose_meta_payload_limit_bytes():
                message = (
                    "Pose metadata payload exceeds the configured limit "
                    f"({payload_bytes} bytes)"
                )
                if strict:
                    raise RuntimeError(
                        "RF-DETR keypoint " + message.lower()
                    )
                logger.debug("%s", message)
                return False
            _increment_core_counter(
                "tensor_boundary_copy_bytes_total.pose_meta",
                payload_bytes,
            )
            attached = bool(
                attach_obj(
                    batch_meta,
                    obj_meta,
                    payload_json,
                    True,
                )
            )
            if strict and not attached:
                raise RuntimeError(
                    "RF-DETR keypoint native metadata attach returned false"
                )
            return attached
        except Exception as exc:
            if strict:
                if isinstance(exc, RuntimeError):
                    raise
                raise RuntimeError(
                    "RF-DETR keypoint metadata serialization or attach failed"
                ) from exc
            return False

    def _point(self, kpts: np.ndarray, idx: int) -> Optional[Tuple[float, float]]:
        if idx < 0 or idx >= kpts.shape[0]:
            return None
        conf = float(kpts[idx, 2])
        if conf < float(self.kpt_threshold):
            return None
        return float(kpts[idx, 0]), float(kpts[idx, 1])

    @staticmethod
    def _dist(a: Optional[Tuple[float, float]], b: Optional[Tuple[float, float]]) -> Optional[float]:
        if a is None or b is None:
            return None
        return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))

    @staticmethod
    def _mid(a: Optional[Tuple[float, float]], b: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        if a is None or b is None:
            return None
        return (float(a[0] + b[0]) * 0.5, float(a[1] + b[1]) * 0.5)

    @staticmethod
    def _ratio(a: Optional[float], b: Optional[float], eps: float = 1e-6) -> Optional[float]:
        if a is None or b is None:
            return None
        if abs(float(b)) < eps:
            return None
        return float(a) / float(b)

    @staticmethod
    def _symmetry(a: Optional[float], b: Optional[float], eps: float = 1e-6) -> Optional[float]:
        if a is None or b is None:
            return None
        denom = max(float(a), float(b), eps)
        return abs(float(a) - float(b)) / denom

    def _compute_features(
        self,
        kpts: np.ndarray,
        *,
        roi_w: float,
        roi_h: float,
    ) -> Tuple[Dict[str, float], Tuple[float, float, float]]:
        conf = kpts[:, 2].astype(np.float32, copy=False)
        mean_conf = float(np.mean(conf)) if conf.size else 0.0
        min_conf = float(np.min(conf)) if conf.size else 0.0
        valid_frac = float(np.mean(conf >= float(self.kpt_threshold))) if conf.size else 0.0

        def p(name: str) -> Optional[Tuple[float, float]]:
            return self._point(kpts, _POSE_KPT_INDEX[name])

        left_shoulder = p("left_shoulder")
        right_shoulder = p("right_shoulder")
        left_hip = p("left_hip")
        right_hip = p("right_hip")
        left_elbow = p("left_elbow")
        right_elbow = p("right_elbow")
        left_wrist = p("left_wrist")
        right_wrist = p("right_wrist")
        left_knee = p("left_knee")
        right_knee = p("right_knee")
        left_ankle = p("left_ankle")
        right_ankle = p("right_ankle")

        shoulder_mid = self._mid(left_shoulder, right_shoulder)
        hip_mid = self._mid(left_hip, right_hip)
        ankle_mid = self._mid(left_ankle, right_ankle)

        torso_len = self._dist(shoulder_mid, hip_mid)
        leg_len = self._dist(hip_mid, ankle_mid)
        shoulder_to_ankle = self._dist(shoulder_mid, ankle_mid)
        height_proxy = shoulder_to_ankle or leg_len or torso_len

        shoulder_width = self._dist(left_shoulder, right_shoulder)
        hip_width = self._dist(left_hip, right_hip)

        left_upper_arm = self._dist(left_shoulder, left_elbow)
        left_lower_arm = self._dist(left_elbow, left_wrist)
        right_upper_arm = self._dist(right_shoulder, right_elbow)
        right_lower_arm = self._dist(right_elbow, right_wrist)

        left_upper_leg = self._dist(left_hip, left_knee)
        left_lower_leg = self._dist(left_knee, left_ankle)
        right_upper_leg = self._dist(right_hip, right_knee)
        right_lower_leg = self._dist(right_knee, right_ankle)

        torso_leg_ratio = self._ratio(torso_len, leg_len)
        leg_height_ratio = self._ratio(leg_len, height_proxy)
        left_arm_ratio = self._ratio(left_upper_arm, left_lower_arm)
        right_arm_ratio = self._ratio(right_upper_arm, right_lower_arm)
        left_leg_ratio = self._ratio(left_upper_leg, left_lower_leg)
        right_leg_ratio = self._ratio(right_upper_leg, right_lower_leg)

        arm_sym = self._symmetry(left_upper_arm, right_upper_arm)
        leg_sym = self._symmetry(left_upper_leg, right_upper_leg)

        norm = float(roi_h) if roi_h and roi_h > 0 else 1.0

        features: Dict[str, float] = {}

        def add(name: str, val: Optional[float]) -> None:
            if val is None:
                return
            if not math.isfinite(float(val)):
                return
            features[name] = float(val)

        add("height_proxy", height_proxy)
        add("height_proxy_norm", self._ratio(height_proxy, norm))
        add("torso_len", torso_len)
        add("torso_len_norm", self._ratio(torso_len, norm))
        add("leg_len", leg_len)
        add("leg_len_norm", self._ratio(leg_len, norm))
        add("torso_leg_ratio", torso_leg_ratio)
        add("leg_height_ratio", leg_height_ratio)
        add("shoulder_width", shoulder_width)
        add("shoulder_width_norm", self._ratio(shoulder_width, norm))
        add("hip_width", hip_width)
        add("hip_width_norm", self._ratio(hip_width, norm))
        add("chest_width", shoulder_width)
        add("chest_width_norm", self._ratio(shoulder_width, norm))
        add("pelvis_width", hip_width)
        add("pelvis_width_norm", self._ratio(hip_width, norm))

        add("left_upper_arm", left_upper_arm)
        add("left_upper_arm_norm", self._ratio(left_upper_arm, norm))
        add("left_lower_arm", left_lower_arm)
        add("left_lower_arm_norm", self._ratio(left_lower_arm, norm))
        add("right_upper_arm", right_upper_arm)
        add("right_upper_arm_norm", self._ratio(right_upper_arm, norm))
        add("right_lower_arm", right_lower_arm)
        add("right_lower_arm_norm", self._ratio(right_lower_arm, norm))

        add("left_upper_leg", left_upper_leg)
        add("left_upper_leg_norm", self._ratio(left_upper_leg, norm))
        add("left_lower_leg", left_lower_leg)
        add("left_lower_leg_norm", self._ratio(left_lower_leg, norm))
        add("right_upper_leg", right_upper_leg)
        add("right_upper_leg_norm", self._ratio(right_upper_leg, norm))
        add("right_lower_leg", right_lower_leg)
        add("right_lower_leg_norm", self._ratio(right_lower_leg, norm))

        add("left_arm_ratio", left_arm_ratio)
        add("right_arm_ratio", right_arm_ratio)
        add("left_leg_ratio", left_leg_ratio)
        add("right_leg_ratio", right_leg_ratio)
        add("arm_symmetry", arm_sym)
        add("leg_symmetry", leg_sym)

        return features, (mean_conf, min_conf, valid_frac)

    @staticmethod
    def _decode_native_pose_payload(
        payload: Any,
    ) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
        if payload is None:
            return None
        try:
            score = float(payload.get("score", 0.0))
            raw_roi = payload.get("keypoints_roi")
            raw_abs = payload.get("keypoints_abs")
        except Exception:
            return None
        if not isinstance(raw_roi, (list, tuple)) or len(raw_roi) < 17:
            return None
        if not isinstance(raw_abs, (list, tuple)) or len(raw_abs) < 17:
            return None
        rows_roi: List[List[float]] = []
        rows_abs: List[List[float]] = []
        for item in raw_roi[:17]:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                return None
            try:
                rows_roi.append([float(item[0]), float(item[1]), float(item[2])])
            except Exception:
                return None
        for item in raw_abs[:17]:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                return None
            try:
                rows_abs.append([float(item[0]), float(item[1]), float(item[2])])
            except Exception:
                return None
        try:
            arr_roi = np.asarray(rows_roi, dtype=np.float32)
            arr_abs = np.asarray(rows_abs, dtype=np.float32)
        except Exception:
            return None
        if arr_roi.shape != (17, 3) or arr_abs.shape != (17, 3):
            return None
        if (
            not math.isfinite(score)
            or score < 0.0
            or not np.isfinite(arr_roi).all()
            or not np.isfinite(arr_abs).all()
            or np.any(arr_roi[:, 2] < 0.0)
            or np.any(arr_roi[:, 2] > 1.0)
            or np.any(arr_abs[:, 2] < 0.0)
            or np.any(arr_abs[:, 2] > 1.0)
        ):
            return None
        _increment_core_counter("tensor_host_copies_total.pose")
        return score, arr_roi, arr_abs

    def _extract_pose_native(
        self,
        obj_meta: Any,
    ) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
        if noesis_pose_meta_ext is None:
            return None
        extract_obj = getattr(noesis_pose_meta_ext, "extract_pose_keypoints", None)
        if extract_obj is None or not callable(extract_obj):
            return None
        try:
            payload = extract_obj(
                obj_meta,
                int(self.gie_id),
                int(self.model_size[0]),
                int(self.model_size[1]),
                float(self.score_threshold),
                bool(self.letterbox),
            )
        except Exception:
            return None
        return self._decode_native_pose_payload(payload)

    def _extract_rfdetr_frame_native(
        self,
        frame_meta: Any,
        object_items: Sequence[Any],
    ) -> Dict[int, Tuple[float, np.ndarray, np.ndarray]]:
        if noesis_pose_meta_ext is None:
            raise RuntimeError(
                "noesis_pose_meta_ext is required for RF-DETR keypoint metadata"
            )
        extract_frame = getattr(
            noesis_pose_meta_ext,
            "extract_rfdetr_keypoint_matches",
            None,
        )
        if extract_frame is None or not callable(extract_frame):
            raise RuntimeError(
                "noesis_pose_meta_ext is missing extract_rfdetr_keypoint_matches"
            )
        result = extract_frame(
            frame_meta,
            int(self.gie_id),
            float(self.score_threshold),
            float(self.match_min_iou),
            float(self.match_ambiguity_margin),
        )
        if not isinstance(result, Mapping):
            raise RuntimeError(
                "RF-DETR keypoint native matcher returned no result mapping"
            )
        raw_matches = result.get("matches")
        diagnostics = result.get("diagnostics")
        if not isinstance(raw_matches, (list, tuple)) or not isinstance(
            diagnostics, Mapping
        ):
            raise RuntimeError(
                "RF-DETR keypoint native matcher result contract is invalid"
            )

        matches: Dict[int, Tuple[float, np.ndarray, np.ndarray]] = {}
        seen_query_indices: set[int] = set()
        for raw_match in raw_matches:
            if not isinstance(raw_match, Mapping):
                raise RuntimeError(
                    "RF-DETR keypoint native match row is invalid"
                )
            try:
                object_index = int(raw_match["object_index"])
                query_index = int(raw_match["query_index"])
                base_score = float(raw_match["base_score"])
                fused_score = float(raw_match["score"])
                match_iou = float(raw_match["match_iou"])
            except Exception as exc:
                raise RuntimeError(
                    "RF-DETR keypoint native match lacks required scalar fields"
                ) from exc
            if (
                object_index < 0
                or object_index >= len(object_items)
                or object_index in matches
                or query_index < 0
                or query_index >= 100
                or query_index in seen_query_indices
            ):
                raise RuntimeError(
                    "RF-DETR keypoint native match object/query index is "
                    "invalid or reused"
                )
            if (
                not math.isfinite(base_score)
                or not 0.0 <= base_score <= 1.0
                or not math.isfinite(fused_score)
                or fused_score < float(self.score_threshold)
                or not math.isfinite(match_iou)
                or not float(self.match_min_iou) <= match_iou <= 1.0
            ):
                raise RuntimeError(
                    "RF-DETR keypoint native match score/IoU contract is invalid"
                )
            seen_query_indices.add(query_index)
            obj_meta = object_items[object_index]
            try:
                native_object_id = int(raw_match["object_id"])
                python_object_id = int(getattr(obj_meta, "object_id", -1))
            except Exception as exc:
                raise RuntimeError(
                    "RF-DETR keypoint object identity contract is invalid"
                ) from exc
            if native_object_id != python_object_id:
                raise RuntimeError(
                    "RF-DETR keypoint native/Python object ordering drifted"
                )
            python_bbox = _rect_to_bbox(getattr(obj_meta, "rect_params", None))
            native_bbox = raw_match.get("bbox")
            if (
                python_bbox is None
                or not isinstance(native_bbox, (list, tuple))
                or len(native_bbox) < 4
            ):
                raise RuntimeError(
                    "RF-DETR keypoint match bbox contract is invalid"
                )
            if any(
                abs(float(python_bbox[index]) - float(native_bbox[index]))
                > 1e-3
                for index in range(4)
            ):
                raise RuntimeError(
                    "RF-DETR keypoint native/Python object bbox ordering drifted"
                )
            decoded = self._decode_native_pose_payload(raw_match)
            if decoded is None:
                raise RuntimeError(
                    "RF-DETR keypoint native match payload is invalid"
                )
            matches[object_index] = decoded

        try:
            ambiguous = int(diagnostics.get("ambiguous_objects", 0) or 0)
            unmatched = int(diagnostics.get("unmatched_objects", 0) or 0)
            matched = int(diagnostics.get("matched_objects", 0) or 0)
            person_objects = int(diagnostics.get("person_objects", 0) or 0)
            person_queries = int(diagnostics.get("person_queries", 0) or 0)
        except Exception as exc:
            raise RuntimeError(
                "RF-DETR keypoint matcher diagnostics are invalid"
            ) from exc
        if (
            min(
                ambiguous,
                unmatched,
                matched,
                person_objects,
                person_queries,
            )
            < 0
            or matched != len(matches)
            or matched + ambiguous + unmatched != person_objects
            or person_queries < matched
        ):
            raise RuntimeError(
                "RF-DETR keypoint matcher diagnostics/count contract drifted"
            )
        if ambiguous > 0:
            _increment_core_counter(
                "rfdetr_keypoint_match_ambiguous_total",
                ambiguous,
            )
        if unmatched > 0:
            _increment_core_counter(
                "rfdetr_keypoint_match_unmatched_total",
                unmatched,
            )
        _increment_core_counter(
            "rfdetr_keypoint_match_attached_candidates_total",
            matched,
        )
        if ambiguous > 0 or unmatched > 0:
            now = time.time()
            if now - float(self._rfdetr_diag_last_log) >= 1.0:
                logger.warning(
                    "RF-DETR keypoint strict association: matched=%d "
                    "ambiguous=%d unmatched=%d person_objects=%s "
                    "person_queries=%s min_iou=%.3f margin=%.3f",
                    matched,
                    ambiguous,
                    unmatched,
                    diagnostics.get("person_objects"),
                    diagnostics.get("person_queries"),
                    float(self.match_min_iou),
                    float(self.match_ambiguity_margin),
                )
                self._rfdetr_diag_last_log = now
        return matches

    def handle_frame_ds8(self, batch_meta: Any, frame_meta: Any) -> None:
        object_items = getattr(frame_meta, "object_items", None) or []
        rfdetr_matches: Dict[
            int, Tuple[float, np.ndarray, np.ndarray]
        ] = {}
        if self.tensor_source == "rfdetr_pgie_frame":
            rfdetr_matches = self._extract_rfdetr_frame_native(
                frame_meta,
                object_items,
            )
        elif self.tensor_source != "object_sgie":
            raise RuntimeError(
                f"unsupported pose tensor source: {self.tensor_source}"
            )
        source_id = self._frame_source_id(frame_meta)
        frame_id = self._frame_id(frame_meta)
        ts_us = self._frame_timestamp_us(frame_meta)
        debug = str(os.environ.get("NOESIS_POSE_FEATURE_DEBUG", "")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if debug:
            self._debug_frames += 1
        attach_obj = None
        if noesis_pose_meta_ext is not None:
            attach_obj = getattr(noesis_pose_meta_ext, "attach_pose_features", None)
        if attach_obj is None or not callable(attach_obj):
            attach_obj = None
            if self.tensor_source == "rfdetr_pgie_frame":
                raise RuntimeError(
                    "RF-DETR keypoint metadata requires callable "
                    "noesis_pose_meta_ext.attach_pose_features"
                )
            if not self._missing_native_logged:
                logger.warning(
                    "Pose meta attach skipped; noesis_pose_meta_ext is unavailable or missing attach_pose_features (build scripts/build_noesis_pose_meta_ext.sh)"
                )
                self._missing_native_logged = True
        pose_budget = self._features_per_frame_max()
        for object_index, obj_meta in enumerate(object_items):
            if debug:
                self._debug_objects += 1
            try:
                class_id = int(getattr(obj_meta, "class_id", -1))
            except Exception:
                class_id = -1
            if class_id != 0:
                continue

            rect = getattr(obj_meta, "rect_params", None)
            bbox = _rect_to_bbox(rect)
            if bbox is None:
                continue
            try:
                track_id = int(getattr(obj_meta, "object_id", -1))
            except Exception:
                track_id = -1
            roi_w = float(bbox[2])
            roi_h = float(bbox[3])
            stable_id = self._stable_id_for_track(source_id, track_id)
            cached_payload = self._cached_pose_payload(
                source_id,
                track_id,
                bbox,
                frame_id,
                ts_us,
                stable_id,
            )
            if cached_payload is not None and self._attach_pose_payload(
                batch_meta,
                attach_obj,
                obj_meta,
                cached_payload,
            ):
                _increment_core_counter("detection_wake.pose_feature_cache_hit")
                if debug:
                    self._debug_attached += 1
                continue
            score = 0.0
            kpts_abs: Optional[np.ndarray] = None
            kpts_for_features: Optional[np.ndarray] = None
            kpts_roi: Optional[np.ndarray] = None
            native: Optional[Tuple[float, np.ndarray, np.ndarray]] = None
            native_start_ns = time.perf_counter_ns()
            if self.tensor_source == "rfdetr_pgie_frame":
                native = rfdetr_matches.get(object_index)
            else:
                if pose_budget <= 0:
                    _increment_core_counter(
                        "detection_wake.pose_feature_budget_skipped"
                    )
                    if debug:
                        self._debug_missing += 1
                    continue
                pose_budget -= 1
                native = self._extract_pose_native(obj_meta)
            _record_core_stage_timing("pose_feature.native_extract", native_start_ns)
            if native is not None:
                _increment_core_counter("detection_wake.pose_feature_native_extract")
                score, kpts_roi, kpts_abs = native
                kpts_for_features = kpts_roi
            else:
                if debug:
                    self._debug_missing += 1
                continue

            if kpts_abs is None or kpts_for_features is None:
                if debug:
                    self._debug_missing += 1
                continue
            if kpts_roi is None:
                kpts_roi = np.array(kpts_for_features, dtype=np.float32, copy=True)

            features, quality = self._compute_features(kpts_for_features, roi_w=roi_w, roi_h=roi_h)
            mean_conf, min_conf, valid_frac = quality
            if not features:
                if debug:
                    self._debug_missing += 1
                continue

            stable_id = None
            try:
                mgr = getattr(self.pipeline, "stable_id_mgr", None)
                if mgr is not None:
                    rec = mgr.active_tracks.get((int(source_id), int(track_id)))
                    if rec and rec.get("stable_id") is not None:
                        stable_id = int(rec.get("stable_id"))
            except Exception:
                stable_id = None

            payload = PoseFeatureResult(
                source_id=int(source_id),
                frame_id=int(frame_id),
                object_id=int(track_id),
                class_id=int(class_id),
                bbox=bbox,
                score=float(score),
                kpt_mean_conf=float(mean_conf),
                kpt_min_conf=float(min_conf),
                kpt_valid_frac=float(valid_frac),
                features=features,
                keypoints_roi=kpts_roi.tolist(),
                keypoints_abs=kpts_abs.tolist(),
                stable_id=stable_id,
                model=str(self.model_label),
                ts_us=int(ts_us),
            ).to_dict()
            ok = self._attach_pose_payload(
                batch_meta,
                attach_obj,
                obj_meta,
                payload,
            )
            if ok:
                if debug:
                    self._debug_attached += 1
            else:
                if debug:
                    self._debug_missing += 1

        if debug:
            now = time.time()
            if (now - float(self._debug_last_log)) >= 1.0:
                logger.info(
                    "Pose features debug: frames=%d objects=%d attached=%d missing=%d",
                    int(self._debug_frames),
                    int(self._debug_objects),
                    int(self._debug_attached),
                    int(self._debug_missing),
                )
                self._debug_frames = 0
                self._debug_objects = 0
                self._debug_attached = 0
                self._debug_missing = 0
                self._debug_last_log = float(now)


@dataclass
class PoseKeypointOverlayProcessor:
    pipeline: "DS8Pipeline"
    gie_id: int
    model_size: Tuple[int, int] = (640, 640)
    score_threshold: float = 0.25
    kpt_threshold: float = 0.35
    letterbox: bool = True
    line_width: int = 2
    point_radius: int = 3
    max_display_metas: int = 12
    _missing_native_logged: bool = field(default=False, init=False, repr=False)
    _debug_last_log: float = field(default=0.0, init=False, repr=False)
    _debug_frames: int = field(default=0, init=False, repr=False)
    _debug_objects: int = field(default=0, init=False, repr=False)
    _debug_drawn: int = field(default=0, init=False, repr=False)
    _debug_missing: int = field(default=0, init=False, repr=False)

    def _frame_source_id(self, frame_meta: Any) -> int:
        for attr in ("source_id", "pad_index", "camera_id"):
            value = getattr(frame_meta, attr, None)
            if value is None:
                continue
            try:
                return int(value)
            except Exception:
                continue
        return 0

    def _extract_pose_payload(self, obj_meta: Any) -> Optional[Dict[str, Any]]:
        if noesis_pose_meta_ext is None:
            return None
        extract_obj = getattr(noesis_pose_meta_ext, "extract_pose_features", None)
        if extract_obj is None or not callable(extract_obj):
            return None
        try:
            raw = extract_obj(obj_meta)
        except Exception:
            return None
        if raw is None:
            return None
        try:
            text = str(raw)
            if not text:
                return None
            payload = json.loads(text)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _keypoints_from_payload(self, payload: Mapping[str, Any], bbox: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        src_bbox = payload.get("bbox")
        src_w = float(bbox[2])
        src_h = float(bbox[3])
        if isinstance(src_bbox, (list, tuple)) and len(src_bbox) >= 4:
            try:
                src_w = float(src_bbox[2])
                src_h = float(src_bbox[3])
            except Exception:
                src_w = float(bbox[2])
                src_h = float(bbox[3])
        dst_w = float(bbox[2])
        dst_h = float(bbox[3])
        sx = float(dst_w / src_w) if src_w > 1e-6 else 1.0
        sy = float(dst_h / src_h) if src_h > 1e-6 else 1.0

        raw_roi = payload.get("keypoints_roi")
        if isinstance(raw_roi, (list, tuple)) and len(raw_roi) >= 17:
            rows: List[List[float]] = []
            for item in raw_roi[:17]:
                if not isinstance(item, (list, tuple)) or len(item) < 3:
                    return None
                try:
                    x = float(item[0]) * sx + float(bbox[0])
                    y = float(item[1]) * sy + float(bbox[1])
                    c = float(item[2])
                    rows.append([x, y, c])
                except Exception:
                    return None
            try:
                arr = np.asarray(rows, dtype=np.float32)
            except Exception:
                return None
            if arr.shape == (17, 3):
                return arr

        raw_abs = payload.get("keypoints_abs")
        if not isinstance(raw_abs, (list, tuple)) or len(raw_abs) < 17:
            return None
        rows_abs: List[List[float]] = []
        for item in raw_abs[:17]:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                return None
            try:
                rows_abs.append([float(item[0]), float(item[1]), float(item[2])])
            except Exception:
                return None
        try:
            arr_abs = np.asarray(rows_abs, dtype=np.float32)
        except Exception:
            return None
        if arr_abs.shape != (17, 3):
            return None
        return arr_abs


    def _lookup_stable_id(self, source_id: int, track_id: int) -> Optional[int]:
        mgr = getattr(self.pipeline, "stable_id_mgr", None)
        if mgr is None:
            return None
        key = (int(source_id), int(track_id))
        rec = None
        lock = getattr(mgr, "_lock", None)
        if lock is not None:
            try:
                with lock:
                    rec = getattr(mgr, "active_tracks", {}).get(key)
            except Exception:
                rec = None
        else:
            try:
                rec = getattr(mgr, "active_tracks", {}).get(key)
            except Exception:
                rec = None
        if not isinstance(rec, dict):
            return None
        stable_id = rec.get("stable_id")
        try:
            stable_id_int = int(stable_id)
        except Exception:
            return None
        if stable_id_int <= 0:
            return None
        return stable_id_int

    @staticmethod
    def _color_for_key(key: int) -> Tuple[float, float, float]:
        hue = float((int(key) * 47) % 360)
        r, g, b = colorsys.hls_to_rgb(hue / 360.0, 0.60, 0.80)
        return float(r), float(g), float(b)

    def handle_batch_ds8(self, batch_meta: Any) -> None:
        if ds_osd is None:
            return
        frame_items = getattr(batch_meta, "frame_items", None)
        if frame_items is None:
            return

        debug = str(os.environ.get("NOESIS_POSE_KEYPOINT_DEBUG", "")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        now = time.time()
        for frame_meta in frame_items:
            object_items = getattr(frame_meta, "object_items", None) or []
            if debug:
                self._debug_frames += 1

            acquire_display_meta = getattr(batch_meta, "acquire_display_meta", None)
            append_meta = getattr(frame_meta, "append", None)
            if not callable(acquire_display_meta) or not callable(append_meta):
                continue

            display_metas: List[Any] = []
            appended_ids: set[int] = set()
            current: Any = None
            lines_used = 0
            circles_used = 0
            max_lines_per_meta = 16
            max_circles_per_meta = 16
            max_metas = int(self.max_display_metas)

            def _append_display_meta(dm: Any) -> None:
                dm_id = id(dm)
                if dm_id in appended_ids:
                    return
                try:
                    append_meta(dm)
                except Exception:
                    return
                appended_ids.add(dm_id)

            def _alloc_meta() -> Optional[Any]:
                try:
                    dm = acquire_display_meta()
                except Exception:
                    return None
                if not dm:
                    return None
                display_metas.append(dm)
                return dm

            def _ensure_meta() -> Optional[Any]:
                nonlocal current, lines_used, circles_used
                if current is None or lines_used >= max_lines_per_meta or circles_used >= max_circles_per_meta:
                    if len(display_metas) >= max_metas:
                        return None
                    current = _alloc_meta()
                    lines_used = 0
                    circles_used = 0
                return current

            drawn_this_frame = False
            for obj_meta in object_items:
                if debug:
                    self._debug_objects += 1
                try:
                    class_id = int(getattr(obj_meta, "class_id", -1))
                except Exception:
                    class_id = -1
                if class_id != 0:
                    continue
                rect = getattr(obj_meta, "rect_params", None)
                bbox = _rect_to_bbox(rect)
                if bbox is None:
                    continue
                try:
                    track_id = int(getattr(obj_meta, "object_id", -1))
                except Exception:
                    track_id = -1

                kpts = None
                payload = self._extract_pose_payload(obj_meta)
                if payload is not None:
                    kpts = self._keypoints_from_payload(payload, bbox)

                if kpts is None:
                    if payload is None and not self._missing_native_logged:
                        self._missing_native_logged = True
                        logger.info("Pose keypoint overlay missing native pose meta")
                    if debug:
                        self._debug_missing += 1
                    continue

                source_id = self._frame_source_id(frame_meta)
                stable_id = self._lookup_stable_id(source_id, track_id) if track_id >= 0 else None
                key_id = int(stable_id) if stable_id is not None else int(track_id if track_id >= 0 else class_id + 1)
                r, g, b = self._color_for_key(key_id)

                for i, j in _POSE_SKELETON:
                    if i >= kpts.shape[0] or j >= kpts.shape[0]:
                        continue
                    c1 = float(kpts[i, 2])
                    c2 = float(kpts[j, 2])
                    if c1 < float(self.kpt_threshold) or c2 < float(self.kpt_threshold):
                        continue
                    dm = _ensure_meta()
                    if dm is None:
                        break
                    line = ds_osd.Line()
                    line.x1 = int(kpts[i, 0])
                    line.y1 = int(kpts[i, 1])
                    line.x2 = int(kpts[j, 0])
                    line.y2 = int(kpts[j, 1])
                    line.width = int(self.line_width)
                    line.color.r = float(r)
                    line.color.g = float(g)
                    line.color.b = float(b)
                    line.color.a = 1.0
                    try:
                        dm.add_line(line)
                        lines_used += 1
                        drawn_this_frame = True
                    except Exception:
                        current = None
                        continue

                for xk, yk, ck in kpts:
                    if float(ck) < float(self.kpt_threshold):
                        continue
                    dm = _ensure_meta()
                    if dm is None:
                        break
                    circ = ds_osd.Circle()
                    circ.xc = int(xk)
                    circ.yc = int(yk)
                    circ.radius = int(self.point_radius)
                    circ.width = max(1, int(self.line_width))
                    circ.color.r = float(r)
                    circ.color.g = float(g)
                    circ.color.b = float(b)
                    circ.color.a = 1.0
                    try:
                        dm.add_circle(circ)
                        circles_used += 1
                        drawn_this_frame = True
                    except Exception:
                        current = None
                        continue

            for dm in display_metas:
                _append_display_meta(dm)

            if debug:
                if drawn_this_frame:
                    self._debug_drawn += 1
                now = time.time()
                if (now - float(self._debug_last_log)) >= 1.0:
                    logger.info(
                        "Pose keypoints debug: frames=%d objects=%d drawn=%d missing=%d",
                        self._debug_frames,
                        self._debug_objects,
                        self._debug_drawn,
                        self._debug_missing,
                    )
                    self._debug_frames = 0
                    self._debug_objects = 0
                    self._debug_drawn = 0
                    self._debug_missing = 0
                    self._debug_last_log = float(now)


@dataclass
class _DepthTrackingFrameProcessor:
    depth_store: _AlignedDepthFrameStore
    depth_gie_id: int
    fallback_frame_size: Tuple[int, int]
    depth_model_name: str
    depth_unit: str
    depth_is_metric: bool

    def _capture_depth_frame(self, frame_meta: Any, frame_w: int, frame_h: int) -> Any | None:
        if noesis_depth_tracking_tensor_ext is None:
            raise RuntimeError("noesis_depth_tracking_tensor_ext is required for baseline depth tracking")
        capture_fn = getattr(noesis_depth_tracking_tensor_ext, "capture_aligned_depth_frame", None)
        if not callable(capture_fn):
            raise RuntimeError("capture_aligned_depth_frame is required for baseline depth tracking")
        try:
            return capture_fn(frame_meta, int(self.depth_gie_id), int(frame_w), int(frame_h))
        except Exception:
            logger.exception("Native DAv2 GPU depth capture failed")
            return None

    def handle_frame_ds8(self, frame_meta: Any) -> None:
        frame_w, frame_h = _canonical_frame_size(frame_meta, self.fallback_frame_size)
        if frame_w <= 0 or frame_h <= 0:
            return
        depth_device_frame = self._capture_depth_frame(frame_meta, frame_w, frame_h)
        if depth_device_frame is None:
            return
        depth_w = int(getattr(depth_device_frame, "depth_width", frame_w) or frame_w)
        depth_h = int(getattr(depth_device_frame, "depth_height", frame_h) or frame_h)
        source_id = int(_meta_lookup(frame_meta, "source_id", "pad_index", default=0) or 0)
        frame_id = int(_meta_lookup(frame_meta, "frame_number", "frame_num", default=0) or 0)
        pts_us = _frame_pts_key_us(frame_meta)
        frame = _AlignedDepthFrame(
            key=(source_id, frame_id, pts_us),
            source_id=source_id,
            frame_id=frame_id,
            pts_us=pts_us,
            depth_map=None,
            valid_mask=None,
            frame_w=frame_w,
            frame_h=frame_h,
            depth_w=depth_w,
            depth_h=depth_h,
            unit=self.depth_unit,
            is_metric=self.depth_is_metric,
            model_name=self.depth_model_name,
            depth_device_frame=depth_device_frame,
        )
        self.depth_store.put(frame)
        _increment_core_counter("depth_tracking_device_frames_total")


@dataclass
class _ObjectDepthFusionProcessor:
    depth_store: _AlignedDepthFrameStore
    fallback_frame_size: Tuple[int, int]
    depth_model_name: str
    depth_unit: str
    depth_is_metric: bool
    depth_every_n_frames: int
    calibration_resolver: Any | None = None
    camera_labels: Mapping[int, str] = field(default_factory=dict)
    _result_cache: Dict[Tuple[int, int], Dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _depth_retry_tracks: set[Tuple[int, int]] = field(default_factory=set, init=False, repr=False)

    def _max_objects_per_frame(self) -> int:
        return _read_env_int("NOESIS_OBJECT_DEPTH_MAX_OBJECTS_PER_FRAME", 2, min_value=0)

    def _max_hz_per_track(self) -> float:
        return _read_env_float("NOESIS_OBJECT_DEPTH_MAX_HZ_PER_TRACK", 5.0, min_value=0.0)

    def _cache_max_age_us(self) -> int:
        max_age_ms = _read_env_float("NOESIS_OBJECT_DEPTH_CACHE_MAX_AGE_MS", 500.0, min_value=0.0)
        return int(max_age_ms * 1000.0)

    def _cache_max_bbox_shift(self) -> float:
        return _read_env_float("NOESIS_OBJECT_DEPTH_CACHE_MAX_BBOX_SHIFT", 0.25, min_value=0.0)

    def _exact_frame_wait_ms(self) -> float:
        raw = os.environ.get(
            "NOESIS_OBJECT_DEPTH_EXACT_FRAME_WAIT_MS",
            str(_OBJECT_DEPTH_EXACT_FRAME_WAIT_DEFAULT_MS),
        )
        return _bounded_object_depth_wait_ms(
            raw,
            default=_OBJECT_DEPTH_EXACT_FRAME_WAIT_DEFAULT_MS,
        )

    def _camera_id_for_source(self, source_id: int) -> Optional[str]:
        camera_id = self.camera_labels.get(int(source_id))
        if isinstance(camera_id, str) and camera_id.strip():
            return camera_id.strip()
        return None

    def _track_key(self, source_id: int, obj_meta: Any) -> Optional[Tuple[int, int]]:
        try:
            track_id = int(getattr(obj_meta, "object_id", -1))
        except Exception:
            track_id = -1
        if track_id < 0:
            return None
        return int(source_id), int(track_id)

    @staticmethod
    def _bbox_shift_ratio(old_bbox: Sequence[float], new_bbox: Sequence[float]) -> float:
        try:
            old_left, old_top, old_w, old_h = [float(x) for x in old_bbox[:4]]
            new_left, new_top, new_w, new_h = [float(x) for x in new_bbox[:4]]
        except Exception:
            return float("inf")
        diag = math.hypot(max(1.0, old_w), max(1.0, old_h))
        old_cx = old_left + old_w * 0.5
        old_cy = old_top + old_h * 0.5
        new_cx = new_left + new_w * 0.5
        new_cy = new_top + new_h * 0.5
        return float(math.hypot(new_cx - old_cx, new_cy - old_cy) / max(1.0, diag))

    def _cached_payload(
        self,
        *,
        source_id: int,
        frame_id: int,
        pts_us: int,
        obj_meta: Any,
        bbox: Sequence[float],
    ) -> Optional[Dict[str, Any]]:
        key = self._track_key(source_id, obj_meta)
        if key is None:
            return None
        entry = self._result_cache.get(key)
        if not isinstance(entry, dict):
            return None
        try:
            sample_ts_us = int(entry.get("_sample_ts_us", 0) or 0)
        except Exception:
            sample_ts_us = 0
        max_age_us = self._cache_max_age_us()
        if max_age_us > 0 and sample_ts_us > 0 and (int(pts_us) - sample_ts_us) > max_age_us:
            self._result_cache.pop(key, None)
            return None
        old_bbox = entry.get("bbox")
        if not isinstance(old_bbox, (list, tuple)) or len(old_bbox) < 4:
            self._result_cache.pop(key, None)
            return None
        if self._bbox_shift_ratio(old_bbox, bbox) > self._cache_max_bbox_shift():
            self._result_cache.pop(key, None)
            return None
        payload = {str(k): v for k, v in entry.items() if not str(k).startswith("_")}
        try:
            measurement_frame_id = int(
                payload.get("measurement_frame_id", payload.get("frame_id", frame_id))
            )
        except Exception:
            measurement_frame_id = int(frame_id)
        try:
            measurement_ts_us = int(
                payload.get("measurement_ts_us", payload.get("ts_us", pts_us))
            )
        except Exception:
            measurement_ts_us = int(pts_us)
        measurement_age_us = max(0, int(pts_us) - int(measurement_ts_us))
        try:
            object_id = int(getattr(obj_meta, "object_id", payload.get("object_id", -1)))
        except Exception:
            object_id = int(payload.get("object_id", -1) or -1)
        try:
            score = float(getattr(obj_meta, "confidence", payload.get("score", 0.0)))
        except Exception:
            score = float(payload.get("score", 0.0) or 0.0)
        payload.update(
            {
                "source_id": int(source_id),
                "frame_id": int(frame_id),
                "object_id": int(object_id),
                "bbox": [float(x) for x in bbox[:4]],
                "score": float(score),
                "ts_us": int(pts_us),
                "measurement_frame_id": int(measurement_frame_id),
                "measurement_ts_us": int(measurement_ts_us),
                "measurement_age_us": int(measurement_age_us),
                "measurement_cached": True,
            }
        )
        return payload

    def _sample_due(self, *, source_id: int, obj_meta: Any, pts_us: int) -> bool:
        key = self._track_key(source_id, obj_meta)
        if key is None:
            return True
        entry = self._result_cache.get(key)
        if not isinstance(entry, dict):
            return True
        max_hz = self._max_hz_per_track()
        if max_hz <= 0.0:
            return False
        # A previous sample attempt had no ready secondary depth frame.  Do
        # not let the old valid cache cadence hide the next completed DAv2
        # frame; retry once per incoming frame until sampling succeeds.
        if key in self._depth_retry_tracks:
            return True
        try:
            sample_ts_us = int(entry.get("_sample_ts_us", 0) or 0)
        except Exception:
            sample_ts_us = 0
        if sample_ts_us <= 0:
            return True
        min_interval_us = int(1_000_000.0 / max(1e-6, float(max_hz)))
        return (int(pts_us) - sample_ts_us) >= min_interval_us

    def _cache_result(self, *, source_id: int, pts_us: int, obj_meta: Any, result: ObjectDepthResult) -> None:
        key = self._track_key(source_id, obj_meta)
        if key is None:
            return
        payload = result.to_dict()
        payload["_sample_ts_us"] = int(pts_us)
        self._result_cache[key] = payload
        max_entries = max(8, _read_env_int("NOESIS_OBJECT_DEPTH_CACHE_MAX_TRACKS", 64, min_value=1))
        if len(self._result_cache) > max_entries:
            oldest_key = next(iter(self._result_cache.keys()))
            self._result_cache.pop(oldest_key, None)

    def _record_object_depth_attach(
        self,
        *,
        attached: bool,
        status: str | None,
        failure_reason: str | None = None,
    ) -> bool:
        if attached:
            _increment_core_counter("object_depth_attach_total")
            status_key = re.sub(
                r"[^a-z0-9_]+",
                "_",
                str(status or "unknown").strip().lower(),
            ).strip("_") or "unknown"
            _increment_core_counter(f"object_depth_status_total.{status_key}")
            return True
        reason_key = re.sub(
            r"[^a-z0-9_]+",
            "_",
            str(failure_reason or "unknown").strip().lower(),
        ).strip("_") or "unknown"
        count = _increment_core_counter("object_depth_attach_failure_total")
        _increment_core_counter(f"object_depth_attach_failure_total.{reason_key}")
        if count <= 3 or (count % 250) == 0:
            logger.warning(
                "NOESIS.OBJECT_DEPTH attachment failed (reason=%s, count=%d)",
                reason_key,
                count,
            )
        return False

    def _attach_object_depth_payload(
        self,
        batch_meta: Any,
        obj_meta: Any,
        payload: Mapping[str, Any],
    ) -> bool:
        payload_status = str(payload.get("status") or "unknown")
        if noesis_depth_meta_ext is None:
            return self._record_object_depth_attach(
                attached=False,
                status=payload_status,
                failure_reason="native_extension_unavailable",
            )
        attach_fn = getattr(noesis_depth_meta_ext, "attach_object_depth", None)
        if not callable(attach_fn):
            return self._record_object_depth_attach(
                attached=False,
                status=payload_status,
                failure_reason="attach_function_unavailable",
            )
        try:
            payload_json = json.dumps(dict(payload), separators=(",", ":"))
            attached = bool(attach_fn(batch_meta, obj_meta, payload_json, True))
        except Exception:
            logger.exception("Failed to attach NOESIS.OBJECT_DEPTH to object metadata")
            return self._record_object_depth_attach(
                attached=False,
                status=payload_status,
                failure_reason="native_exception",
            )
        return self._record_object_depth_attach(
            attached=attached,
            status=payload_status,
            failure_reason=None if attached else "native_rejected",
        )

    def _resolve_calibration_snapshot(self, source_id: int) -> Any | None:
        resolver = self.calibration_resolver
        if resolver is None:
            return None
        snapshot_fn = getattr(resolver, "snapshot", None)
        if not callable(snapshot_fn):
            return None
        camera_id = self._camera_id_for_source(source_id)
        try:
            return snapshot_fn(int(source_id), camera_id)
        except TypeError:
            try:
                return snapshot_fn(int(source_id))
            except Exception:
                return None
        except Exception:
            return None

    def _copy_depth_crop(self, depth_frame: _AlignedDepthFrame, x0: int, y0: int, x1: int, y1: int) -> Optional[np.ndarray]:
        width = int(x1) - int(x0)
        height = int(y1) - int(y0)
        if width <= 0 or height <= 0:
            return None
        depth_device_frame = getattr(depth_frame, "depth_device_frame", None)
        if depth_device_frame is None:
            logger.warning("Depth device frame is unavailable for object-depth fusion")
            return None
        copy_roi = getattr(depth_device_frame, "copy_roi_to_numpy", None)
        if not callable(copy_roi):
            logger.warning("Depth device frame is missing copy_roi_to_numpy")
            return None
        try:
            start_ns = time.perf_counter_ns()
            roi = np.asarray(copy_roi(int(x0), int(y0), int(width), int(height)), dtype=np.float32)
            _record_core_stage_timing(
                "object_depth.copy_roi_to_numpy",
                start_ns,
                item_count=int(width) * int(height),
            )
            _increment_core_counter("detection_wake.object_depth_roi_copy")
            _increment_core_counter("tensor_host_copies_total.object_depth_roi")
            _increment_core_counter("object_depth_gpu_roi_copies_total")
            return roi
        except Exception:
            logger.exception("GPU depth ROI copy failed")
            return None

    def _bbox_fallback_band_fraction(self) -> float:
        raw = os.environ.get("NOESIS_OBJECT_DEPTH_BBOX_BAND_FRACTION", "0.5")
        try:
            value = float(str(raw).strip())
        except Exception:
            value = 0.5
        return max(0.05, min(1.0, value))

    def _bbox_fallback_y0(self, y0: int, y1: int) -> int:
        height = max(1, int(y1) - int(y0))
        band_height = max(1, int(math.ceil(float(height) * self._bbox_fallback_band_fraction())))
        return max(int(y0), int(y1) - int(band_height))

    def _bbox_fallback_rect(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
    ) -> Tuple[int, int, int, int]:
        """Return a tight diagnostic ROI for detections without person support.

        Finite depth inside a detector rectangle is not evidence that the
        pixels belong to the person. Keep this small ROI for diagnostics only;
        callers must not publish its bottom edge as a floor contact.
        """

        width = max(1, int(x1) - int(x0))
        height = max(1, int(y1) - int(y0))
        width_fraction = _read_env_float(
            "NOESIS_OBJECT_DEPTH_BBOX_CORE_WIDTH_FRACTION",
            0.32,
            min_value=0.10,
        )
        height_fraction = _read_env_float(
            "NOESIS_OBJECT_DEPTH_BBOX_CORE_HEIGHT_FRACTION",
            0.24,
            min_value=0.08,
        )
        core_width = max(1, min(width, int(math.ceil(width * min(0.60, width_fraction)))))
        core_height = max(1, min(height, int(math.ceil(height * min(0.40, height_fraction)))))
        center_x = (int(x0) + int(x1)) // 2
        core_x0 = max(int(x0), min(int(x1) - 1, center_x - (core_width // 2)))
        core_x1 = min(int(x1), core_x0 + core_width)
        core_x0 = max(int(x0), core_x1 - core_width)
        core_y1 = int(y1)
        core_y0 = max(int(y0), core_y1 - core_height)
        return int(core_x0), int(core_y0), int(core_x1), int(core_y1)

    def _attached_pose_keypoints(
        self,
        obj_meta: Any,
        bbox: Tuple[float, float, float, float],
    ) -> Optional[np.ndarray]:
        if obj_meta is None or noesis_pose_meta_ext is None:
            return None
        extract_pose = getattr(noesis_pose_meta_ext, "extract_pose_features", None)
        if not callable(extract_pose):
            return None
        try:
            raw_payload = extract_pose(obj_meta)
        except Exception:
            return None
        if raw_payload is None:
            return None
        if isinstance(raw_payload, Mapping):
            payload = dict(raw_payload)
        else:
            try:
                payload = json.loads(str(raw_payload))
            except Exception:
                return None
        if not isinstance(payload, Mapping):
            return None

        src_bbox = payload.get("bbox")
        src_width = float(bbox[2])
        src_height = float(bbox[3])
        if isinstance(src_bbox, (list, tuple)) and len(src_bbox) >= 4:
            try:
                src_width = float(src_bbox[2])
                src_height = float(src_bbox[3])
            except Exception:
                src_width = float(bbox[2])
                src_height = float(bbox[3])
        scale_x = float(bbox[2]) / src_width if src_width > 1e-6 else 1.0
        scale_y = float(bbox[3]) / src_height if src_height > 1e-6 else 1.0

        raw_roi = payload.get("keypoints_roi")
        if isinstance(raw_roi, (list, tuple)) and len(raw_roi) >= 17:
            rows: List[List[float]] = []
            for item in raw_roi[:17]:
                if not isinstance(item, (list, tuple)) or len(item) < 3:
                    return None
                try:
                    rows.append(
                        [
                            float(bbox[0]) + (float(item[0]) * scale_x),
                            float(bbox[1]) + (float(item[1]) * scale_y),
                            float(item[2]),
                        ]
                    )
                except Exception:
                    return None
            points = np.asarray(rows, dtype=np.float32)
            if points.shape == (17, 3) and np.isfinite(points).all():
                return points

        raw_abs = payload.get("keypoints_abs")
        if not isinstance(raw_abs, (list, tuple)) or len(raw_abs) < 17:
            return None
        rows_abs: List[List[float]] = []
        for item in raw_abs[:17]:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                return None
            try:
                rows_abs.append([float(item[0]), float(item[1]), float(item[2])])
            except Exception:
                return None
        points_abs = np.asarray(rows_abs, dtype=np.float32)
        if points_abs.shape != (17, 3) or not np.isfinite(points_abs).all():
            return None
        return points_abs

    def _pose_capsule_masks(
        self,
        keypoints: np.ndarray,
        *,
        crop_rect: Tuple[int, int, int, int],
        bbox: Tuple[float, float, float, float],
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        Optional[List[float]],
        int,
        List[Tuple[float, float, float, float, float, float]],
    ]:
        x0, y0, x1, y1 = [int(value) for value in crop_rect]
        width = max(0, x1 - x0)
        height = max(0, y1 - y0)
        body_mask = np.zeros((height, width), dtype=np.uint8)
        contact_mask = np.zeros((height, width), dtype=np.uint8)
        contact_capsules: List[
            Tuple[float, float, float, float, float, float]
        ] = []
        if width <= 0 or height <= 0:
            return body_mask.astype(bool), contact_mask.astype(bool), None, 0, contact_capsules

        threshold = _read_env_float(
            "NOESIS_OBJECT_DEPTH_POSE_KPT_THRESHOLD",
            0.35,
            min_value=0.0,
        )
        scale = max(1.0, min(float(bbox[2]), float(bbox[3])))
        joint_radius = max(2, min(10, int(round(scale * 0.045))))
        limb_width = max(3, min(18, joint_radius * 2))
        contact_radius = max(3, min(14, int(round(scale * 0.065))))

        def _point(index: int) -> Optional[Tuple[int, int]]:
            if index < 0 or index >= int(keypoints.shape[0]):
                return None
            try:
                px = float(keypoints[index, 0])
                py = float(keypoints[index, 1])
                confidence = float(keypoints[index, 2])
            except Exception:
                return None
            if confidence < threshold or not (math.isfinite(px) and math.isfinite(py)):
                return None
            return int(round(px - x0)), int(round(py - y0))

        support_indices = (5, 6, 11, 12, 13, 14, 15, 16)
        support_segments = (
            (5, 6),
            (5, 11),
            (6, 12),
            (11, 12),
            (11, 13),
            (12, 14),
            (13, 15),
            (14, 16),
        )
        for first, second in support_segments:
            point_a = _point(first)
            point_b = _point(second)
            if point_a is not None and point_b is not None:
                cv2.line(body_mask, point_a, point_b, 1, thickness=limb_width)
        for index in support_indices:
            point = _point(index)
            if point is not None:
                cv2.circle(body_mask, point, joint_radius, 1, thickness=-1)

        contact_points: List[Tuple[int, int]] = []
        for knee_index, ankle_index in ((13, 15), (14, 16)):
            ankle = _point(ankle_index)
            if ankle is None:
                continue
            contact_points.append(ankle)
            knee = _point(knee_index)
            if knee is not None:
                lower_leg_start = (
                    int(round((0.35 * float(knee[0])) + (0.65 * float(ankle[0])))),
                    int(round((0.35 * float(knee[1])) + (0.65 * float(ankle[1])))),
                )
            else:
                lower_leg_start = ankle
            # OpenCV line thickness is a diameter while the ankle circle uses
            # a radius.  The former implementation passed ``contact_radius``
            # as a capsule radius and therefore made the native lower-leg band
            # nearly twice as wide as the host mask.  Carry both radii in one
            # compact compound primitive: a thin segment plus the ankle disk.
            line_radius = max(1.5, 0.5 * float(max(3, contact_radius)))
            contact_capsules.append(
                (
                    float(x0 + lower_leg_start[0]),
                    float(y0 + lower_leg_start[1]),
                    float(x0 + ankle[0]),
                    float(y0 + ankle[1]),
                    float(line_radius),
                    float(contact_radius),
                )
            )

        if not contact_points:
            return body_mask.astype(bool), contact_mask.astype(bool), None, 0, contact_capsules

        # Rasterize the diagnostic/fallback mask from the exact continuous
        # predicate used by the CUDA sampler.  Pixel centers are absolute
        # aligned-frame coordinates, matching the native +0.5 convention.
        grid_y, grid_x = np.ogrid[y0:y1, x0:x1]
        point_x = np.asarray(grid_x, dtype=np.float32) + 0.5
        point_y = np.asarray(grid_y, dtype=np.float32) + 0.5
        for ax, ay, bx, by, line_radius, ankle_radius in contact_capsules:
            dx = float(bx) - float(ax)
            dy = float(by) - float(ay)
            length_sq = (dx * dx) + (dy * dy)
            if length_sq > 1.0e-6:
                t = ((point_x - float(ax)) * dx + (point_y - float(ay)) * dy) / length_sq
                t = np.clip(t, 0.0, 1.0)
            else:
                t = np.zeros((height, width), dtype=np.float32)
            nearest_x = float(ax) + (t * dx)
            nearest_y = float(ay) + (t * dy)
            segment_active = (
                np.square(point_x - nearest_x) + np.square(point_y - nearest_y)
                <= float(line_radius) * float(line_radius)
            )
            ankle_active = (
                np.square(point_x - float(bx)) + np.square(point_y - float(by))
                <= float(ankle_radius) * float(ankle_radius)
            )
            contact_mask[np.asarray(segment_active | ankle_active)] = 1

        contact_uv = [
            float(x0) + (sum(float(point[0]) for point in contact_points) / len(contact_points)),
            float(y0) + (sum(float(point[1]) for point in contact_points) / len(contact_points)),
        ]
        return (
            body_mask.astype(bool),
            contact_mask.astype(bool),
            contact_uv,
            len(contact_points),
            contact_capsules,
        )

    def _extract_instance_mask_payload(self, obj_meta: Any) -> Optional[Mapping[str, Any]]:
        if noesis_depth_meta_ext is None:
            return None
        try:
            payload = noesis_depth_meta_ext.extract_object_mask(obj_meta)  # type: ignore[union-attr]
        except Exception:
            logger.debug("Native object-mask extraction failed", exc_info=True)
            return None
        if not payload:
            return None
        if not isinstance(payload, Mapping):
            return None
        return payload

    def _decode_instance_mask_payload(
        self,
        payload: Optional[Mapping[str, Any]],
        target_shape: Tuple[int, int],
    ) -> Tuple[Optional[np.ndarray], str]:
        if not payload:
            return None, "missing_mask"
        try:
            threshold = float(payload.get("threshold", 0.5) or 0.5)
            data = np.asarray(payload.get("data"), dtype=np.float32)
        except Exception:
            return None, "mask_decode_failed"
        if data.ndim != 2 or data.size <= 0:
            return None, "mask_decode_failed"
        mask = np.asarray(data > threshold, dtype=bool)
        if mask.shape != target_shape:
            mask = cv2.resize(
                mask.astype(np.uint8, copy=False),
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        return mask, "ok"

    def _decode_instance_mask(self, obj_meta: Any, target_shape: Tuple[int, int]) -> Tuple[Optional[np.ndarray], str]:
        payload = self._extract_instance_mask_payload(obj_meta)
        return self._decode_instance_mask_payload(payload, target_shape)

    def _build_result(
        self,
        frame_meta: Any,
        obj_meta: Any,
        *,
        bbox: Tuple[float, float, float, float],
        status: str,
        mask_area_px: int = 0,
        sample_count: int = 0,
        valid_fraction: float = 0.0,
        depth_center: Optional[float] = None,
        values: Optional[np.ndarray] = None,
        anchor_fields: Optional[Mapping[str, Any]] = None,
        sampling_mode: str = "instance_mask",
        evidence_quality: Optional[str] = None,
        evidence_reason: Optional[str] = None,
    ) -> ObjectDepthResult:
        try:
            object_id = int(getattr(obj_meta, "object_id", -1))
        except Exception:
            object_id = -1
        try:
            class_id = int(getattr(obj_meta, "class_id", -1))
        except Exception:
            class_id = -1
        try:
            score = float(getattr(obj_meta, "confidence", 0.0))
        except Exception:
            score = 0.0
        values_arr = np.asarray(values, dtype=np.float32) if values is not None else np.empty(0, dtype=np.float32)
        has_values = bool(values_arr.size)
        depth_p10 = float(np.percentile(values_arr, 10.0)) if has_values else None
        depth_p90 = float(np.percentile(values_arr, 90.0)) if has_values else None
        frame_id = int(_meta_lookup(frame_meta, "frame_number", "frame_num", default=0) or 0)
        ts_us = _frame_pts_key_us(frame_meta)
        payload: Dict[str, Any] = {
            "source_id": int(_meta_lookup(frame_meta, "source_id", "pad_index", default=0) or 0),
            "frame_id": frame_id,
            "object_id": object_id,
            "class_id": class_id,
            "bbox": bbox,
            "score": score,
            "sampling_mode": str(sampling_mode or "instance_mask"),
            "status": status,
            "unit": self.depth_unit,
            "is_metric": self.depth_is_metric,
            "sample_count": max(0, int(sample_count)),
            "valid_fraction": max(0.0, min(1.0, float(valid_fraction))),
            "depth_center": depth_center,
            "depth_median": float(np.median(values_arr)) if has_values else None,
            "depth_mean": float(np.mean(values_arr)) if has_values else None,
            "depth_p10": depth_p10,
            "depth_p90": depth_p90,
            "depth_min": float(np.min(values_arr)) if has_values else None,
            "depth_max": float(np.max(values_arr)) if has_values else None,
            "depth_spread_m": _depth_spread_from_bounds(depth_p10, depth_p90),
            "mask_area_px": max(0, int(mask_area_px)),
            "model": self.depth_model_name,
            "ts_us": ts_us,
            "measurement_frame_id": frame_id,
            "measurement_ts_us": ts_us,
            "measurement_age_us": 0,
            "measurement_cached": False,
            "evidence_quality": evidence_quality,
            "evidence_reason": evidence_reason,
        }
        if anchor_fields:
            payload.update({str(key): value for key, value in anchor_fields.items() if value is not None})
        return ObjectDepthResult(**payload)

    def _sample_bbox_band_result(
        self,
        frame_meta: Any,
        obj_meta: Any,
        *,
        bbox: Tuple[float, float, float, float],
        depth_crop: np.ndarray,
        crop_origin: Tuple[int, int],
        depth_center: Optional[float],
    ) -> ObjectDepthResult:
        mask_area = int(depth_crop.size)
        anchor_fields: Dict[str, Any] = {
            "spatial_class": "person",
        }
        values_full = np.asarray(depth_crop[np.isfinite(depth_crop)], dtype=np.float32)
        values = _bounded_depth_stat_values(values_full)
        sample_count = int(values_full.size)
        status = "no_ground_contact" if sample_count > 0 else "no_valid_depth"
        return self._build_result(
            frame_meta,
            obj_meta,
            bbox=bbox,
            status=status,
            mask_area_px=mask_area,
            sample_count=sample_count,
            valid_fraction=float(sample_count) / float(mask_area or 1),
            depth_center=depth_center,
            values=values,
            anchor_fields=anchor_fields,
            sampling_mode="bbox_core",
            evidence_quality="rejected",
            evidence_reason="bbox_only_without_person_contact_support",
        )

    def _sample_pose_capsule_result(
        self,
        frame_meta: Any,
        obj_meta: Any,
        *,
        bbox: Tuple[float, float, float, float],
        depth_crop: np.ndarray,
        body_mask: np.ndarray,
        contact_mask: np.ndarray,
        contact_uv: Optional[Sequence[float]],
        visible_ankles: int,
        depth_center: Optional[float],
    ) -> ObjectDepthResult:
        body_valid = np.logical_and(np.asarray(body_mask, dtype=bool), np.isfinite(depth_crop))
        body_values_full = np.asarray(depth_crop[body_valid], dtype=np.float32)
        body_values = _bounded_depth_stat_values(body_values_full)
        body_area = int(np.count_nonzero(body_mask))
        sample_count = int(body_values_full.size)
        valid_fraction = float(sample_count) / float(body_area or 1)

        anchor_source: Optional[str] = None
        anchor_depth_m: Optional[float] = None
        anchor_sample_count: Optional[int] = None
        anchor_valid_fraction: Optional[float] = None
        anchor_depth_spread_m: Optional[float] = None
        evidence_quality = "rejected"
        evidence_reason: Optional[str] = "pose_ankles_unavailable"
        status = "no_ground_contact" if sample_count > 0 else "no_valid_depth"

        contact_area = int(np.count_nonzero(contact_mask))
        if contact_uv is not None and visible_ankles > 0 and contact_area > 0:
            contact_valid = np.logical_and(
                np.asarray(contact_mask, dtype=bool),
                np.isfinite(depth_crop),
            )
            contact_values_full = np.asarray(depth_crop[contact_valid], dtype=np.float32)
            contact_values = _bounded_depth_stat_values(
                contact_values_full,
                env_name="NOESIS_OBJECT_DEPTH_MAX_ANCHOR_SAMPLES",
            )
            contact_count = int(contact_values_full.size)
            contact_fraction = float(contact_count) / float(contact_area or 1)
            contact_median = (
                float(np.median(contact_values)) if contact_values.size > 0 else None
            )
            contact_spread = _depth_spread_from_bounds(
                float(np.percentile(contact_values, 10.0)) if contact_values.size > 0 else None,
                float(np.percentile(contact_values, 90.0)) if contact_values.size > 0 else None,
            )
            min_contact_count = max(8, min(20, int(math.ceil(contact_area * 0.20))))
            evidence_reason = _depth_evidence_rejection_reason(
                sample_count=contact_count,
                valid_fraction=contact_fraction,
                depth_m=contact_median,
                depth_spread_m=contact_spread,
                min_sample_count=min_contact_count,
                min_valid_fraction=0.60,
                strict_spread=True,
            )
            anchor_depth_spread_m = contact_spread
            if evidence_reason is None:
                anchor_source = "pose_ankle_support"
                anchor_depth_m = contact_median
                anchor_sample_count = int(contact_count)
                anchor_valid_fraction = float(contact_fraction)
                evidence_quality = "good" if int(visible_ankles) >= 2 else "estimated"
                status = "ok"
            else:
                status = "ambiguous_depth"

        anchor_fields: Dict[str, Any] = {
            "spatial_class": "person",
            "anchor_uv": list(contact_uv) if anchor_source is not None and contact_uv is not None else None,
            "anchor_source": anchor_source,
            "anchor_depth_m": anchor_depth_m,
            "anchor_sample_count": anchor_sample_count,
            "anchor_valid_fraction": anchor_valid_fraction,
            "anchor_depth_spread_m": anchor_depth_spread_m,
        }
        return self._build_result(
            frame_meta,
            obj_meta,
            bbox=bbox,
            status=status,
            mask_area_px=body_area,
            sample_count=sample_count,
            valid_fraction=valid_fraction,
            depth_center=depth_center,
            values=body_values,
            anchor_fields=anchor_fields,
            sampling_mode="pose_capsule",
            evidence_quality=evidence_quality,
            evidence_reason=evidence_reason,
        )

    def _stats_float(self, stats: Mapping[str, Any], key: str) -> Optional[float]:
        value = stats.get(key)
        if value is None:
            return None
        try:
            parsed = float(value)
        except Exception:
            return None
        return parsed if math.isfinite(parsed) else None

    def _sample_bbox_band_result_native(
        self,
        frame_meta: Any,
        obj_meta: Any,
        *,
        bbox: Tuple[float, float, float, float],
        depth_frame: _AlignedDepthFrame,
        crop_rect: Tuple[int, int, int, int],
    ) -> Optional[ObjectDepthResult]:
        depth_device_frame = getattr(depth_frame, "depth_device_frame", None)
        if depth_device_frame is None:
            return None
        sample_stats = getattr(depth_device_frame, "sample_roi_stats", None)
        if not callable(sample_stats):
            return None
        x0, y0, x1, y1 = [int(v) for v in crop_rect]
        width = max(0, x1 - x0)
        height = max(0, y1 - y0)
        if width <= 0 or height <= 0:
            return None
        try:
            start_ns = time.perf_counter_ns()
            stats_raw = sample_stats(
                int(x0),
                int(y0),
                int(width),
                int(height),
                _read_env_int("NOESIS_OBJECT_DEPTH_NATIVE_MAX_STAT_SAMPLES", 4096, min_value=128),
            )
            _record_core_stage_timing("object_depth.native_roi_stats", start_ns, item_count=width * height)
        except Exception:
            logger.debug("Native object-depth ROI stats failed", exc_info=True)
            return None
        if not isinstance(stats_raw, Mapping):
            return None
        _increment_core_counter("object_depth_gpu_roi_copies_total")
        stats = dict(stats_raw)
        try:
            sample_count = int(stats.get("sample_count", 0) or 0)
        except Exception:
            sample_count = 0
        try:
            mask_area = int(stats.get("roi_area_px", width * height) or (width * height))
        except Exception:
            mask_area = int(width * height)
        valid_fraction = self._stats_float(stats, "valid_fraction") or 0.0
        depth_median = self._stats_float(stats, "depth_median")
        depth_p10 = self._stats_float(stats, "depth_p10")
        depth_p90 = self._stats_float(stats, "depth_p90")
        depth_spread = _depth_spread_from_bounds(depth_p10, depth_p90)
        status = "no_ground_contact" if sample_count > 0 and depth_median is not None else "no_valid_depth"
        try:
            object_id = int(getattr(obj_meta, "object_id", -1))
        except Exception:
            object_id = -1
        try:
            class_id = int(getattr(obj_meta, "class_id", -1))
        except Exception:
            class_id = -1
        try:
            score = float(getattr(obj_meta, "confidence", 0.0))
        except Exception:
            score = 0.0
        payload: Dict[str, Any] = {
            "source_id": int(_meta_lookup(frame_meta, "source_id", "pad_index", default=0) or 0),
            "frame_id": int(_meta_lookup(frame_meta, "frame_number", "frame_num", default=0) or 0),
            "object_id": object_id,
            "class_id": class_id,
            "bbox": bbox,
            "score": score,
            "sampling_mode": "bbox_core_native",
            "status": status,
            "unit": self.depth_unit,
            "is_metric": self.depth_is_metric,
            "sample_count": max(0, int(sample_count)),
            "valid_fraction": max(0.0, min(1.0, float(valid_fraction))),
            "depth_center": self._stats_float(stats, "depth_center"),
            "depth_median": depth_median,
            "depth_mean": self._stats_float(stats, "depth_mean"),
            "depth_p10": depth_p10,
            "depth_p90": depth_p90,
            "depth_min": self._stats_float(stats, "depth_min"),
            "depth_max": self._stats_float(stats, "depth_max"),
            "depth_spread_m": depth_spread,
            "mask_area_px": max(0, int(mask_area)),
            "model": self.depth_model_name,
            "ts_us": _frame_pts_key_us(frame_meta),
            "spatial_class": "person",
            "evidence_quality": "rejected",
            "evidence_reason": "bbox_only_without_person_contact_support",
            "measurement_frame_id": int(_meta_lookup(frame_meta, "frame_number", "frame_num", default=0) or 0),
            "measurement_ts_us": _frame_pts_key_us(frame_meta),
            "measurement_age_us": 0,
            "measurement_cached": False,
        }
        _increment_core_counter("detection_wake.object_depth_native_stats")
        return ObjectDepthResult(**payload)

    def _sample_mask_stats_native(
        self,
        depth_device_frame: Any,
        *,
        crop_rect: Tuple[int, int, int, int],
        mask: np.ndarray,
        max_samples_env: str = "NOESIS_OBJECT_DEPTH_NATIVE_MAX_STAT_SAMPLES",
        stage_name: str = "object_depth.native_mask_roi_stats",
    ) -> Optional[Dict[str, Any]]:
        sample_masked_stats = getattr(depth_device_frame, "sample_masked_roi_stats", None)
        if not callable(sample_masked_stats):
            return None
        x0, y0, x1, y1 = [int(v) for v in crop_rect]
        width = max(0, x1 - x0)
        height = max(0, y1 - y0)
        if width <= 0 or height <= 0:
            return None
        mask_arr = np.asarray(mask, dtype=np.float32)
        if mask_arr.shape != (height, width):
            return None
        mask_arr = np.ascontiguousarray(mask_arr)
        try:
            start_ns = time.perf_counter_ns()
            stats_raw = sample_masked_stats(
                int(x0),
                int(y0),
                int(width),
                int(height),
                mask_arr,
                0.5,
                _read_env_int(max_samples_env, 4096, min_value=128),
            )
            _record_core_stage_timing(stage_name, start_ns, item_count=width * height)
        except Exception:
            logger.debug("Native masked object-depth ROI stats failed", exc_info=True)
            return None
        if not isinstance(stats_raw, Mapping):
            return None
        return dict(stats_raw)

    def _sample_pose_capsule_result_native(
        self,
        frame_meta: Any,
        obj_meta: Any,
        *,
        bbox: Tuple[float, float, float, float],
        depth_frame: _AlignedDepthFrame,
        crop_rect: Tuple[int, int, int, int],
        body_mask: np.ndarray,
        contact_mask: np.ndarray,
        contact_uv: Optional[Sequence[float]],
        visible_ankles: int,
        contact_capsules: Optional[Sequence[Sequence[float]]] = None,
    ) -> Optional[ObjectDepthResult]:
        depth_device_frame = getattr(depth_frame, "depth_device_frame", None)
        if depth_device_frame is None:
            return None
        body_area = int(np.count_nonzero(body_mask))
        contact_area = int(np.count_nonzero(contact_mask))
        if body_area <= 0:
            return None

        # Position authority needs only compact ankle/contact support.  The
        # native method receives at most two scalar [ax, ay, bx, by, radius]
        # capsules, samples their union from the GPU-resident depth tensor,
        # and returns one compact statistic.  No image-sized host mask is
        # uploaded. With no observed ankle contact, fail closed without
        # touching the GPU at all.
        body_stats: Mapping[str, Any] = {}
        if contact_uv is not None and visible_ankles > 0 and contact_area > 0 and contact_capsules:
            sample_capsule_stats = getattr(
                depth_device_frame,
                "sample_pose_capsule_stats",
                None,
            )
            if not callable(sample_capsule_stats):
                return None
            try:
                start_ns = time.perf_counter_ns()
                stats_raw = sample_capsule_stats(
                    [list(capsule) for capsule in contact_capsules[:2]],
                    _read_env_int(
                        "NOESIS_OBJECT_DEPTH_MAX_ANCHOR_SAMPLES",
                        4096,
                        min_value=128,
                    ),
                )
                _record_core_stage_timing(
                    "object_depth.native_pose_contact_capsule_stats",
                    start_ns,
                    item_count=int(contact_area),
                )
            except Exception:
                logger.debug(
                    "Native pose-contact object-depth ROI stats failed",
                    exc_info=True,
                )
                return None
            if not isinstance(stats_raw, Mapping):
                return None
            body_stats = dict(stats_raw)
            body_area = int(
                body_stats.get("sampled_capsule_area_px", contact_area) or contact_area
            )
            _increment_core_counter("object_depth_gpu_roi_copies_total")

        try:
            sample_count = int(body_stats.get("sample_count", 0) or 0)
        except Exception:
            sample_count = 0
        valid_fraction = self._stats_float(body_stats, "valid_fraction") or 0.0
        depth_median = self._stats_float(body_stats, "depth_median")
        depth_p10 = self._stats_float(body_stats, "depth_p10")
        depth_p90 = self._stats_float(body_stats, "depth_p90")
        depth_spread = _depth_spread_from_bounds(depth_p10, depth_p90)

        anchor_source: Optional[str] = None
        anchor_depth_m: Optional[float] = None
        anchor_sample_count: Optional[int] = None
        anchor_valid_fraction: Optional[float] = None
        anchor_depth_spread_m: Optional[float] = None
        evidence_quality = "rejected"
        evidence_reason: Optional[str] = "pose_ankles_unavailable"
        status = "no_ground_contact"

        if contact_uv is not None and visible_ankles > 0 and contact_area > 0 and contact_capsules:
            contact_stats = body_stats
            if contact_stats is not None:
                try:
                    contact_count = int(contact_stats.get("sample_count", 0) or 0)
                except Exception:
                    contact_count = 0
                contact_fraction = self._stats_float(contact_stats, "valid_fraction") or 0.0
                contact_median = self._stats_float(contact_stats, "depth_median")
                contact_spread = _depth_spread_from_bounds(
                    self._stats_float(contact_stats, "depth_p10"),
                    self._stats_float(contact_stats, "depth_p90"),
                )
                min_contact_count = max(8, min(20, int(math.ceil(contact_area * 0.20))))
                evidence_reason = _depth_evidence_rejection_reason(
                    sample_count=contact_count,
                    valid_fraction=contact_fraction,
                    depth_m=contact_median,
                    depth_spread_m=contact_spread,
                    min_sample_count=min_contact_count,
                    min_valid_fraction=0.60,
                    strict_spread=True,
                )
                anchor_depth_spread_m = contact_spread
                if evidence_reason is None:
                    anchor_source = "pose_ankle_support"
                    anchor_depth_m = contact_median
                    anchor_sample_count = int(contact_count)
                    anchor_valid_fraction = float(contact_fraction)
                    evidence_quality = "good" if int(visible_ankles) >= 2 else "estimated"
                    status = "ok"
                else:
                    status = "ambiguous_depth"

        try:
            object_id = int(getattr(obj_meta, "object_id", -1))
        except Exception:
            object_id = -1
        try:
            class_id = int(getattr(obj_meta, "class_id", -1))
        except Exception:
            class_id = -1
        try:
            score = float(getattr(obj_meta, "confidence", 0.0))
        except Exception:
            score = 0.0
        frame_id = int(_meta_lookup(frame_meta, "frame_number", "frame_num", default=0) or 0)
        ts_us = _frame_pts_key_us(frame_meta)
        payload: Dict[str, Any] = {
            "source_id": int(_meta_lookup(frame_meta, "source_id", "pad_index", default=0) or 0),
            "frame_id": frame_id,
            "object_id": object_id,
            "class_id": class_id,
            "bbox": bbox,
            "score": score,
            "sampling_mode": "pose_capsule_native",
            "status": status,
            "unit": self.depth_unit,
            "is_metric": self.depth_is_metric,
            "sample_count": max(0, int(sample_count)),
            "valid_fraction": max(0.0, min(1.0, float(valid_fraction))),
            "depth_center": self._stats_float(body_stats, "depth_center"),
            "depth_median": depth_median,
            "depth_mean": self._stats_float(body_stats, "depth_mean"),
            "depth_p10": depth_p10,
            "depth_p90": depth_p90,
            "depth_min": self._stats_float(body_stats, "depth_min"),
            "depth_max": self._stats_float(body_stats, "depth_max"),
            "depth_spread_m": depth_spread,
            "mask_area_px": body_area,
            "model": self.depth_model_name,
            "ts_us": ts_us,
            "spatial_class": "person",
            "anchor_uv": list(contact_uv) if anchor_source is not None and contact_uv is not None else None,
            "anchor_source": anchor_source,
            "anchor_depth_m": anchor_depth_m,
            "anchor_sample_count": anchor_sample_count,
            "anchor_valid_fraction": anchor_valid_fraction,
            "anchor_depth_spread_m": anchor_depth_spread_m,
            "evidence_quality": evidence_quality,
            "evidence_reason": evidence_reason,
            "measurement_frame_id": frame_id,
            "measurement_ts_us": ts_us,
            "measurement_age_us": 0,
            "measurement_cached": False,
        }
        _increment_core_counter("detection_wake.object_depth_pose_capsule_stats")
        return ObjectDepthResult(**payload)

    def _sample_person_mask_stats_native(
        self,
        depth_device_frame: Any,
        *,
        crop_rect: Tuple[int, int, int, int],
        mask: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        sample_person_stats = getattr(depth_device_frame, "sample_masked_person_roi_stats", None)
        if not callable(sample_person_stats):
            return None
        x0, y0, x1, y1 = [int(v) for v in crop_rect]
        width = max(0, x1 - x0)
        height = max(0, y1 - y0)
        if width <= 0 or height <= 0:
            return None
        mask_arr = np.asarray(mask, dtype=np.float32)
        if mask_arr.shape != (height, width):
            return None
        mask_arr = np.ascontiguousarray(mask_arr)
        try:
            start_ns = time.perf_counter_ns()
            stats_raw = sample_person_stats(
                int(x0),
                int(y0),
                int(width),
                int(height),
                mask_arr,
                0.5,
                _read_env_int("NOESIS_OBJECT_DEPTH_NATIVE_MAX_STAT_SAMPLES", 4096, min_value=128),
            )
            _record_core_stage_timing("object_depth.native_mask_person_stats", start_ns, item_count=width * height)
        except Exception:
            logger.debug("Native masked person object-depth stats failed", exc_info=True)
            return None
        if not isinstance(stats_raw, Mapping):
            return None
        return dict(stats_raw)

    def _mask_foot_uv(self, mask: np.ndarray, *, frame_origin: Tuple[int, int]) -> Optional[List[float]]:
        if mask.size <= 0:
            return None
        lower_rows = np.nonzero(mask)[0]
        if lower_rows.size <= 0:
            return None
        max_row = int(np.max(lower_rows))
        band_top = max(0, max_row - max(1, int(round(mask.shape[0] * 0.12))))
        foot_band = np.zeros_like(mask, dtype=bool)
        foot_band[band_top : max_row + 1, :] = True
        foot_band = np.logical_and(foot_band, _band_mask(mask, y0_ratio=0.0, y1_ratio=1.0, center_width_ratio=0.35))
        points = np.argwhere(foot_band)
        if points.size <= 0:
            return None
        foot_y = int(np.max(points[:, 0]))
        foot_x = int(np.median(points[points[:, 0] == foot_y][:, 1]))
        return [float(int(frame_origin[0]) + foot_x), float(int(frame_origin[1]) + foot_y)]

    def _anchor_support_requirements(self, area_px: int, anchor_source: str) -> Tuple[int, float]:
        area_px = max(0, int(area_px))
        if anchor_source == "lower_body_band":
            base_count = 24
            floor_count = 16
            min_valid_fraction = 0.40
        else:
            base_count = 32
            floor_count = 20
            min_valid_fraction = 0.45
        adaptive_count = int(math.ceil(float(area_px) * 0.25))
        min_count = max(floor_count, min(base_count, adaptive_count or base_count))
        return min_count, min_valid_fraction

    def _sample_instance_mask_result_native(
        self,
        frame_meta: Any,
        obj_meta: Any,
        *,
        bbox: Tuple[float, float, float, float],
        depth_frame: _AlignedDepthFrame,
        crop_rect: Tuple[int, int, int, int],
        mask_payload: Mapping[str, Any],
    ) -> Optional[ObjectDepthResult]:
        depth_device_frame = getattr(depth_frame, "depth_device_frame", None)
        if depth_device_frame is None:
            return None
        sample_person_stats = getattr(depth_device_frame, "sample_masked_person_roi_stats", None)
        sample_masked_stats = getattr(depth_device_frame, "sample_masked_roi_stats", None)
        if not callable(sample_person_stats) and not callable(sample_masked_stats):
            return None
        x0, y0, x1, y1 = [int(v) for v in crop_rect]
        width = max(0, x1 - x0)
        height = max(0, y1 - y0)
        if width <= 0 or height <= 0:
            return None
        mask, _mask_status = self._decode_instance_mask_payload(mask_payload, (height, width))
        if mask is None or mask.size <= 0:
            return None
        mask = np.asarray(mask, dtype=bool)
        mask_area = int(np.count_nonzero(mask))
        if mask_area <= 0:
            return None

        stats = self._sample_person_mask_stats_native(
            depth_device_frame,
            crop_rect=(x0, y0, x1, y1),
            mask=mask,
        )
        used_combined_stats = stats is not None
        if stats is None:
            stats = self._sample_mask_stats_native(
                depth_device_frame,
                crop_rect=(x0, y0, x1, y1),
                mask=mask,
                stage_name="object_depth.native_mask_roi_stats",
            )
        if stats is None:
            return None
        _increment_core_counter("object_depth_gpu_roi_copies_total")

        def _int_stat(key: str, default: int = 0) -> int:
            try:
                return int(stats.get(key, default) or default)
            except Exception:
                return int(default)

        sample_count = _int_stat("sample_count")
        mask_area_native = _int_stat("mask_area_px", mask_area)
        valid_fraction = self._stats_float(stats, "valid_fraction") or 0.0
        depth_median = self._stats_float(stats, "depth_median")
        depth_p10 = self._stats_float(stats, "depth_p10")
        depth_p90 = self._stats_float(stats, "depth_p90")
        depth_spread = _depth_spread_from_bounds(depth_p10, depth_p90)
        status = "no_valid_depth"

        foot_uv = self._mask_foot_uv(mask, frame_origin=(x0, y0))
        anchor_source: Optional[str] = None
        anchor_depth_m: Optional[float] = None
        anchor_sample_count: Optional[int] = None
        anchor_valid_fraction: Optional[float] = None
        anchor_depth_spread_m: Optional[float] = None
        anchor_rejection_reason: Optional[str] = "person_depth_support_low"

        lower_count = 0
        lower_valid_fraction = 0.0
        torso_count = 0
        torso_valid_fraction = 0.0
        if used_combined_stats:
            lower_area = _int_stat("lower_mask_area_px", 0)
            lower_count = _int_stat("lower_sample_count", 0)
            lower_valid_fraction = self._stats_float(stats, "lower_valid_fraction") or 0.0
            lower_depth = self._stats_float(stats, "lower_depth_median")
            lower_spread = _depth_spread_from_bounds(
                self._stats_float(stats, "lower_depth_p10"),
                self._stats_float(stats, "lower_depth_p90"),
            )
            lower_min_count, lower_min_valid_fraction = self._anchor_support_requirements(lower_area, "lower_body_band")
            lower_support_ok = bool(
                lower_depth is not None
                and lower_count >= lower_min_count
                and lower_valid_fraction >= lower_min_valid_fraction
            )
            if lower_support_ok and _depth_spread_is_supported(
                lower_depth,
                lower_spread,
                strict=False,
            ):
                anchor_source = "lower_body_band"
                anchor_depth_m = lower_depth
                anchor_sample_count = int(lower_count)
                anchor_valid_fraction = float(lower_valid_fraction)
                anchor_depth_spread_m = lower_spread
                anchor_rejection_reason = None
            elif lower_support_ok:
                anchor_depth_spread_m = lower_spread
                anchor_rejection_reason = "lower_body_depth_spread_exceeded"
            if anchor_depth_m is None:
                torso_area = _int_stat("torso_mask_area_px", 0)
                torso_count = _int_stat("torso_sample_count", 0)
                torso_valid_fraction = self._stats_float(stats, "torso_valid_fraction") or 0.0
                torso_depth = self._stats_float(stats, "torso_depth_median")
                torso_spread = _depth_spread_from_bounds(
                    self._stats_float(stats, "torso_depth_p10"),
                    self._stats_float(stats, "torso_depth_p90"),
                )
                torso_min_count, torso_min_valid_fraction = self._anchor_support_requirements(torso_area, "torso_core")
                torso_support_ok = bool(
                    torso_depth is not None
                    and torso_count >= torso_min_count
                    and torso_valid_fraction >= torso_min_valid_fraction
                )
                if torso_support_ok and _depth_spread_is_supported(
                    torso_depth,
                    torso_spread,
                    strict=False,
                ):
                    anchor_source = "torso_core"
                    anchor_depth_m = torso_depth
                    anchor_sample_count = int(torso_count)
                    anchor_valid_fraction = float(torso_valid_fraction)
                    anchor_depth_spread_m = torso_spread
                    anchor_rejection_reason = None
                elif torso_support_ok:
                    anchor_depth_spread_m = torso_spread
                    anchor_rejection_reason = "torso_depth_spread_exceeded"
        else:
            eroded_mask = _erode_mask(mask, kernel_size=3)
            lower_body_mask = _band_mask(eroded_mask, y0_ratio=0.88, y1_ratio=1.0, center_width_ratio=0.35)
            lower_area = int(np.count_nonzero(lower_body_mask))
            lower_stats: Optional[Dict[str, Any]] = None
            if lower_area > 0:
                lower_stats = self._sample_mask_stats_native(
                    depth_device_frame,
                    crop_rect=(x0, y0, x1, y1),
                    mask=lower_body_mask,
                    max_samples_env="NOESIS_OBJECT_DEPTH_MAX_ANCHOR_SAMPLES",
                    stage_name="object_depth.native_mask_anchor_stats",
                )
            if lower_stats is not None:
                try:
                    lower_count = int(lower_stats.get("sample_count", 0) or 0)
                except Exception:
                    lower_count = 0
                lower_valid_fraction = self._stats_float(lower_stats, "valid_fraction") or 0.0
                lower_depth = self._stats_float(lower_stats, "depth_median")
                lower_spread = _depth_spread_from_bounds(
                    self._stats_float(lower_stats, "depth_p10"),
                    self._stats_float(lower_stats, "depth_p90"),
                )
                lower_min_count, lower_min_valid_fraction = self._anchor_support_requirements(lower_area, "lower_body_band")
                lower_support_ok = bool(
                    lower_depth is not None
                    and lower_count >= lower_min_count
                    and lower_valid_fraction >= lower_min_valid_fraction
                )
                if lower_support_ok and _depth_spread_is_supported(
                    lower_depth,
                    lower_spread,
                    strict=False,
                ):
                    anchor_source = "lower_body_band"
                    anchor_depth_m = lower_depth
                    anchor_sample_count = int(lower_count)
                    anchor_valid_fraction = float(lower_valid_fraction)
                    anchor_depth_spread_m = lower_spread
                    anchor_rejection_reason = None
                elif lower_support_ok:
                    anchor_depth_spread_m = lower_spread
                    anchor_rejection_reason = "lower_body_depth_spread_exceeded"

            if anchor_depth_m is None:
                torso_mask = _band_mask(eroded_mask, y0_ratio=0.35, y1_ratio=0.70, center_width_ratio=0.50)
                torso_area = int(np.count_nonzero(torso_mask))
                torso_stats: Optional[Dict[str, Any]] = None
                if torso_area > 0:
                    torso_stats = self._sample_mask_stats_native(
                        depth_device_frame,
                        crop_rect=(x0, y0, x1, y1),
                        mask=torso_mask,
                        max_samples_env="NOESIS_OBJECT_DEPTH_MAX_ANCHOR_SAMPLES",
                        stage_name="object_depth.native_mask_anchor_stats",
                    )
                if torso_stats is not None:
                    try:
                        torso_count = int(torso_stats.get("sample_count", 0) or 0)
                    except Exception:
                        torso_count = 0
                    torso_valid_fraction = self._stats_float(torso_stats, "valid_fraction") or 0.0
                    torso_depth = self._stats_float(torso_stats, "depth_median")
                    torso_spread = _depth_spread_from_bounds(
                        self._stats_float(torso_stats, "depth_p10"),
                        self._stats_float(torso_stats, "depth_p90"),
                    )
                    torso_min_count, torso_min_valid_fraction = self._anchor_support_requirements(torso_area, "torso_core")
                    torso_support_ok = bool(
                        torso_depth is not None
                        and torso_count >= torso_min_count
                        and torso_valid_fraction >= torso_min_valid_fraction
                    )
                    if torso_support_ok and _depth_spread_is_supported(
                        torso_depth,
                        torso_spread,
                        strict=False,
                    ):
                        anchor_source = "torso_core"
                        anchor_depth_m = torso_depth
                        anchor_sample_count = int(torso_count)
                        anchor_valid_fraction = float(torso_valid_fraction)
                        anchor_depth_spread_m = torso_spread
                        anchor_rejection_reason = None
                    elif torso_support_ok:
                        anchor_depth_spread_m = torso_spread
                        anchor_rejection_reason = "torso_depth_spread_exceeded"
        try:
            object_id = int(getattr(obj_meta, "object_id", -1))
        except Exception:
            object_id = -1
        try:
            class_id = int(getattr(obj_meta, "class_id", -1))
        except Exception:
            class_id = -1
        try:
            score = float(getattr(obj_meta, "confidence", 0.0))
        except Exception:
            score = 0.0

        if sample_count > 0 and depth_median is not None:
            status = "ok" if anchor_source is not None else "ambiguous_depth"
        frame_id = int(_meta_lookup(frame_meta, "frame_number", "frame_num", default=0) or 0)
        ts_us = _frame_pts_key_us(frame_meta)
        payload: Dict[str, Any] = {
            "source_id": int(_meta_lookup(frame_meta, "source_id", "pad_index", default=0) or 0),
            "frame_id": frame_id,
            "object_id": object_id,
            "class_id": class_id,
            "bbox": bbox,
            "score": score,
            "sampling_mode": "instance_mask",
            "status": status,
            "unit": self.depth_unit,
            "is_metric": self.depth_is_metric,
            "sample_count": max(0, int(sample_count)),
            "valid_fraction": max(0.0, min(1.0, float(valid_fraction))),
            "depth_center": self._stats_float(stats, "depth_center"),
            "depth_median": depth_median,
            "depth_mean": self._stats_float(stats, "depth_mean"),
            "depth_p10": depth_p10,
            "depth_p90": depth_p90,
            "depth_min": self._stats_float(stats, "depth_min"),
            "depth_max": self._stats_float(stats, "depth_max"),
            "depth_spread_m": depth_spread,
            "mask_area_px": max(0, int(mask_area_native)),
            "model": self.depth_model_name,
            "ts_us": ts_us,
            "spatial_class": "person",
            "anchor_uv": foot_uv if anchor_source is not None else None,
            "anchor_source": anchor_source,
            "anchor_depth_m": anchor_depth_m,
            "anchor_sample_count": anchor_sample_count,
            "anchor_valid_fraction": anchor_valid_fraction,
            "anchor_depth_spread_m": anchor_depth_spread_m,
            "evidence_quality": (
                "good"
                if anchor_source == "lower_body_band"
                else ("estimated" if anchor_source == "torso_core" else "rejected")
            ),
            "evidence_reason": anchor_rejection_reason,
            "measurement_frame_id": frame_id,
            "measurement_ts_us": ts_us,
            "measurement_age_us": 0,
            "measurement_cached": False,
        }
        _increment_core_counter("detection_wake.object_depth_native_mask_stats")
        return ObjectDepthResult(**payload)

    def _sample_person_result(
        self,
        frame_meta: Any,
        obj_meta: Any,
        depth_frame: _AlignedDepthFrame,
    ) -> Optional[ObjectDepthResult]:
        bbox = _rect_to_bbox(getattr(obj_meta, "rect_params", None))
        if bbox is None:
            return None
        frame_w = int(depth_frame.frame_w)
        frame_h = int(depth_frame.frame_h)
        left, top, width, height = bbox
        x0_raw = int(math.floor(left))
        y0_raw = int(math.floor(top))
        x1_raw = int(math.ceil(left + width))
        y1_raw = int(math.ceil(top + height))
        if x1_raw <= 0 or y1_raw <= 0 or x0_raw >= frame_w or y0_raw >= frame_h:
            return self._build_result(frame_meta, obj_meta, bbox=bbox, status="transform_mismatch")
        x0 = max(0, min(frame_w, x0_raw))
        y0 = max(0, min(frame_h, y0_raw))
        x1 = max(0, min(frame_w, x1_raw))
        y1 = max(0, min(frame_h, y1_raw))
        if x1 <= x0 or y1 <= y0:
            return self._build_result(frame_meta, obj_meta, bbox=bbox, status="transform_mismatch")

        mask_payload = self._extract_instance_mask_payload(obj_meta)
        crop_x0, crop_y0, crop_x1, crop_y1 = x0, y0, x1, y1
        pose_body_mask: Optional[np.ndarray] = None
        pose_contact_mask: Optional[np.ndarray] = None
        pose_contact_uv: Optional[List[float]] = None
        pose_visible_ankles = 0
        pose_contact_capsules: List[
            Tuple[float, float, float, float, float, float]
        ] = []
        if not mask_payload:
            pose_keypoints = self._attached_pose_keypoints(obj_meta, bbox)
            if pose_keypoints is not None:
                (
                    pose_body_mask,
                    pose_contact_mask,
                    pose_contact_uv,
                    pose_visible_ankles,
                    pose_contact_capsules,
                ) = self._pose_capsule_masks(
                    pose_keypoints,
                    crop_rect=(x0, y0, x1, y1),
                    bbox=bbox,
                )
                if int(np.count_nonzero(pose_body_mask)) <= 0:
                    pose_body_mask = None
                    pose_contact_mask = None
                    pose_contact_uv = None
                    pose_visible_ankles = 0
                    pose_contact_capsules = []

        if not mask_payload and pose_body_mask is not None and pose_contact_mask is not None:
            native_pose_result = self._sample_pose_capsule_result_native(
                frame_meta,
                obj_meta,
                bbox=bbox,
                depth_frame=depth_frame,
                crop_rect=(x0, y0, x1, y1),
                body_mask=pose_body_mask,
                contact_mask=pose_contact_mask,
                contact_uv=pose_contact_uv,
                visible_ankles=pose_visible_ankles,
                contact_capsules=pose_contact_capsules,
            )
            if native_pose_result is not None:
                return native_pose_result
        elif not mask_payload:
            crop_x0, crop_y0, crop_x1, crop_y1 = self._bbox_fallback_rect(
                x0,
                y0,
                x1,
                y1,
            )
            native_result = self._sample_bbox_band_result_native(
                frame_meta,
                obj_meta,
                bbox=bbox,
                depth_frame=depth_frame,
                crop_rect=(crop_x0, crop_y0, crop_x1, crop_y1),
            )
            if native_result is not None:
                return native_result
        else:
            native_mask_result = self._sample_instance_mask_result_native(
                frame_meta,
                obj_meta,
                bbox=bbox,
                depth_frame=depth_frame,
                crop_rect=(crop_x0, crop_y0, crop_x1, crop_y1),
                mask_payload=mask_payload,
            )
            if native_mask_result is not None:
                return native_mask_result
        depth_crop = self._copy_depth_crop(
            depth_frame,
            crop_x0,
            crop_y0,
            crop_x1,
            crop_y1,
        )
        if depth_crop is None:
            return self._build_result(frame_meta, obj_meta, bbox=bbox, status="depth_not_ready")
        if depth_crop.size <= 0:
            return self._build_result(frame_meta, obj_meta, bbox=bbox, status="transform_mismatch")

        cx = max(0, min(frame_w - 1, int(round(left + (width * 0.5)))))
        if mask_payload:
            cy = max(0, min(frame_h - 1, int(round(top + (height * 0.5)))))
        else:
            cy = max(0, min(frame_h - 1, int(round(crop_y0 + ((crop_y1 - crop_y0) * 0.5)))))
        local_cx = max(0, min(int(depth_crop.shape[1]) - 1, cx - crop_x0))
        local_cy = max(0, min(int(depth_crop.shape[0]) - 1, cy - crop_y0))
        center_sample = float(depth_crop[local_cy, local_cx])
        center_value = center_sample if np.isfinite(center_sample) else None

        if not mask_payload and pose_body_mask is not None and pose_contact_mask is not None:
            return self._sample_pose_capsule_result(
                frame_meta,
                obj_meta,
                bbox=bbox,
                depth_crop=depth_crop,
                body_mask=pose_body_mask,
                contact_mask=pose_contact_mask,
                contact_uv=pose_contact_uv,
                visible_ankles=pose_visible_ankles,
                depth_center=center_value,
            )

        mask, _mask_status = self._decode_instance_mask_payload(mask_payload, depth_crop.shape)
        if mask is None:
            return self._sample_bbox_band_result(
                frame_meta,
                obj_meta,
                bbox=bbox,
                depth_crop=depth_crop,
                crop_origin=(crop_x0, crop_y0),
                depth_center=center_value,
            )

        mask_area = int(np.count_nonzero(mask))

        if mask_area <= 0:
            return self._sample_bbox_band_result(
                frame_meta,
                obj_meta,
                bbox=bbox,
                depth_crop=depth_crop,
                crop_origin=(crop_x0, crop_y0),
                depth_center=center_value,
            )

        valid_mask = np.logical_and(mask, np.isfinite(depth_crop))
        values_full = np.asarray(depth_crop[valid_mask], dtype=np.float32)
        values = _bounded_depth_stat_values(values_full)
        anchor = _extract_person_depth_anchor(
            mask,
            depth_crop,
            frame_origin=(int(crop_x0), int(crop_y0)),
        )
        anchor_fields: Dict[str, Any] = {
            "spatial_class": "person",
            "anchor_uv": (
                list(anchor.foot_uv)
                if anchor.foot_uv is not None and anchor.anchor_source is not None
                else None
            ),
            "anchor_source": anchor.anchor_source,
            "anchor_depth_m": anchor.anchor_depth_m,
            "anchor_sample_count": int(anchor.anchor_sample_count) if anchor.anchor_sample_count > 0 else None,
            "anchor_valid_fraction": float(anchor.anchor_valid_fraction) if anchor.anchor_valid_fraction > 0.0 else None,
            "anchor_depth_spread_m": anchor.anchor_depth_spread_m,
        }
        sample_count = int(values_full.size)
        if sample_count <= 0:
            return self._build_result(
                frame_meta,
                obj_meta,
                bbox=bbox,
                status="no_valid_depth",
                mask_area_px=mask_area,
                depth_center=center_value,
                anchor_fields=anchor_fields,
                sampling_mode="instance_mask",
                evidence_quality="rejected",
                evidence_reason=anchor.anchor_rejection_reason or "person_depth_support_low",
            )
        return self._build_result(
            frame_meta,
            obj_meta,
            bbox=bbox,
            status="ok" if anchor.anchor_source is not None else "ambiguous_depth",
            mask_area_px=mask_area,
            sample_count=sample_count,
            valid_fraction=float(sample_count) / float(mask_area),
            depth_center=center_value,
            values=values,
            anchor_fields=anchor_fields,
            sampling_mode="instance_mask",
            evidence_quality=(
                "good"
                if anchor.anchor_source == "lower_body_band"
                else ("estimated" if anchor.anchor_source == "torso_core" else "rejected")
            ),
            evidence_reason=anchor.anchor_rejection_reason,
        )

    def handle_frame_ds8(self, batch_meta: Any, frame_meta: Any) -> None:
        source_id = int(_meta_lookup(frame_meta, "source_id", "pad_index", default=0) or 0)
        frame_id = int(_meta_lookup(frame_meta, "frame_number", "frame_num", default=0) or 0)
        pts_us = _frame_pts_key_us(frame_meta)
        depth_frame: Optional[_AlignedDepthFrame] = None
        depth_frame_resolved = False
        samples_remaining = self._max_objects_per_frame()
        for obj_meta in getattr(frame_meta, "object_items", None) or []:
            try:
                class_id = int(getattr(obj_meta, "class_id", -1))
            except Exception:
                class_id = -1
            if class_id != 0:
                continue
            bbox = _rect_to_bbox(getattr(obj_meta, "rect_params", None))
            if bbox is None:
                continue
            cached_payload = self._cached_payload(
                source_id=source_id,
                frame_id=frame_id,
                pts_us=pts_us,
                obj_meta=obj_meta,
                bbox=bbox,
            )
            if not self._sample_due(source_id=source_id, obj_meta=obj_meta, pts_us=pts_us):
                if cached_payload is not None and self._attach_object_depth_payload(
                    batch_meta,
                    obj_meta,
                    cached_payload,
                ):
                    _increment_core_counter("detection_wake.object_depth_cache_hit")
                else:
                    _increment_core_counter("detection_wake.object_depth_cadence_skipped")
                continue
            if samples_remaining <= 0:
                if cached_payload is not None and self._attach_object_depth_payload(
                    batch_meta,
                    obj_meta,
                    cached_payload,
                ):
                    _increment_core_counter("detection_wake.object_depth_budget_cache_hit")
                else:
                    _increment_core_counter("detection_wake.object_depth_budget_skipped")
                continue
            samples_remaining -= 1
            sample_start_ns = time.perf_counter_ns()
            if not depth_frame_resolved:
                depth_frame, depth_tensor_age_frames, _age_ms = self.depth_store.resolve(
                    source_id=source_id,
                    frame_id=frame_id,
                    pts_us=pts_us,
                    max_age_frames=max(0, int(self.depth_every_n_frames) - 1),
                    wait_ms=self._exact_frame_wait_ms(),
                )
                depth_frame_resolved = True
            if depth_frame is None:
                track_key = self._track_key(source_id, obj_meta)
                if track_key is not None:
                    self._depth_retry_tracks.add(track_key)
                _increment_core_counter("detection_wake.object_depth_pending_skip_total")
                # A not-ready secondary frame is not an observation.  Do not
                # cache a negative result or attach it to the live metadata:
                # the next frame must be free to consume the newly completed
                # DAv2 output without waiting or cadence suppression.  A
                # still-valid prior measurement remains useful and already
                # carries explicit cached/age provenance, so preserve it while
                # the retry stays armed instead of creating a one-frame depth
                # hole.
                if cached_payload is not None and self._attach_object_depth_payload(
                    batch_meta,
                    obj_meta,
                    cached_payload,
                ):
                    _increment_core_counter(
                        "detection_wake.object_depth_pending_cache_hit"
                    )
                continue
            else:
                result = self._sample_person_result(frame_meta, obj_meta, depth_frame)
            if result is None:
                continue
            depth_tensor_age_us = max(0, int(pts_us) - int(depth_frame.pts_us))
            result = replace(
                result,
                depth_tensor_frame_id=int(depth_frame.frame_id),
                depth_tensor_ts_us=int(depth_frame.pts_us),
                depth_tensor_age_frames=max(0, int(depth_tensor_age_frames)),
                depth_tensor_age_us=int(depth_tensor_age_us),
            )
            _record_core_stage_timing("object_depth.sample_person", sample_start_ns)
            _increment_core_counter("detection_wake.object_depth_sampled")
            track_key = self._track_key(source_id, obj_meta)
            if track_key is not None:
                self._depth_retry_tracks.discard(track_key)
            self._cache_result(source_id=source_id, pts_us=pts_us, obj_meta=obj_meta, result=result)
            self._attach_object_depth_payload(
                batch_meta,
                obj_meta,
                result.to_dict(),
            )


class _DepthTrackingFrameOperator(_BatchMetadataOperatorBase):  # pragma: no cover - requires DS runtime
    def __init__(self, processor: _DepthTrackingFrameProcessor) -> None:
        super().__init__()
        self.processor = processor

    def handle_metadata(self, batch_meta: Any) -> None:  # type: ignore[override]
        frame_items = getattr(batch_meta, "frame_items", None)
        if frame_items is None:
            return
        for frame_meta in frame_items:
            try:
                self.processor.handle_frame_ds8(frame_meta)
            except Exception:
                logger.exception("Failed to capture aligned DAv2 depth frame within batch metadata (DS8)")


class _ObjectDepthFusionOperator(_BatchMetadataOperatorBase):  # pragma: no cover - requires DS runtime
    def __init__(self, processor: _ObjectDepthFusionProcessor) -> None:
        super().__init__()
        self.processor = processor

    def handle_metadata(self, batch_meta: Any) -> None:  # type: ignore[override]
        frame_items = getattr(batch_meta, "frame_items", None)
        if frame_items is None:
            return
        for frame_meta in frame_items:
            try:
                self.processor.handle_frame_ds8(batch_meta, frame_meta)
            except Exception:
                logger.exception("Failed to fuse object depth within batch metadata (DS8)")


def _copy_public_scalar(value: Any) -> Any:
    """Copy the bounded scalar metadata allowed across the media boundary.

    The analytics callback must never hand DeepStream-owned metadata, surfaces,
    tensors, or Python objects with a runtime lifetime to the publication
    worker.  Public tracking payloads are deliberately JSON-shaped; recursively
    copy only that shape and normalize numpy scalar values at the boundary.
    """

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        try:
            return value.item()
        except Exception:
            return None
    if isinstance(value, Mapping):
        return {
            str(key): _copy_public_scalar(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_copy_public_scalar(item) for item in value]
    # There should be no opaque values in the public contract.  Returning None
    # is safer than retaining a borrowed SDK object in an asynchronous queue.
    return None


_IDENTITY_V2_SHADOW_MAX_PRIMITIVES_PER_FRAME = 64
_IDENTITY_V2_SHADOW_MAX_EMBEDDING_DIM = 4096
_IDENTITY_V2_SHADOW_MAX_PENDING_SOURCES = 64


def _identity_v2_tracker_id_for_epoch(
    tracker_id: Any,
    source_epoch: int,
) -> str:
    """Return the coordinator tracklet key for one physical source timeline."""

    tracker_key = str(tracker_id).strip()
    if not tracker_key:
        raise ValueError("identity-v2 tracker ID must be non-empty")
    epoch = int(source_epoch)
    if epoch < 0:
        raise ValueError("identity-v2 source epoch must be non-negative")
    if epoch == 0:
        return tracker_key
    return f"{tracker_key}@source_epoch:{epoch}"


def _copy_bounded_identity_sequence(
    value: Optional[Sequence[Any]],
    *,
    maximum: int,
    field_name: str,
) -> Optional[Tuple[float, ...]]:
    """Detach one small numeric identity field from callback-owned objects."""

    if value is None:
        return None
    limit = max(0, int(maximum))
    # Production ReID/bbox/world fields are owned one-dimensional ndarrays.
    # ``tolist`` performs the bounded native-to-Python copy in C instead of
    # holding the GIL once per 256-D element.  Other sequence types retain the
    # conservative iterator path below.
    if (
        isinstance(value, np.ndarray)
        and value.ndim == 1
        and value.dtype.kind in "biuf"
    ):
        return tuple(map(float, value[:limit].tolist()))
    try:
        iterator = iter(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be a numeric sequence") from exc
    copied: List[float] = []
    for index, raw in enumerate(iterator):
        if index >= limit:
            break
        try:
            copied.append(float(raw))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} contains a non-numeric value") from exc
    return tuple(copied)


def _copy_shadow_identity_primitives(
    *,
    primitives: Sequence[IdentityFramePrimitive],
    copied_track_by_original_id: Mapping[int, Dict[str, Any]],
    embedding_dim: int,
    source_epoch: int,
) -> Tuple[IdentityFramePrimitive, ...]:
    """Bind compact shadow inputs to private detached scalar track identities."""

    dimension = int(embedding_dim)
    if dimension <= 0 or dimension > _IDENTITY_V2_SHADOW_MAX_EMBEDDING_DIM:
        raise ValueError(
            "identity-v2 embedding dimension is outside the bounded worker contract"
        )
    epoch = int(source_epoch)
    if epoch < 0:
        raise ValueError("identity-v2 source epoch must be non-negative")

    primitive_by_track: Dict[int, IdentityFramePrimitive] = {}
    for index, primitive in enumerate(primitives):
        if index >= _IDENTITY_V2_SHADOW_MAX_PRIMITIVES_PER_FRAME:
            raise ValueError(
                "identity-v2 shadow source frame exceeds bounded primitive limit "
                f"{_IDENTITY_V2_SHADOW_MAX_PRIMITIVES_PER_FRAME}"
            )
        if not isinstance(primitive, IdentityFramePrimitive):
            raise TypeError("identity-v2 shadow primitives must be typed values")
        track_key = id(primitive.public_track)
        if track_key in primitive_by_track:
            raise ValueError(
                "identity-v2 shadow primitives contain a duplicate public track"
            )
        primitive_by_track[track_key] = primitive

    copied: List[IdentityFramePrimitive] = []
    for track_key, public_track in copied_track_by_original_id.items():
        primitive = primitive_by_track.pop(int(track_key), None)
        if primitive is None:
            continue
        primitive_tracker_id = str(primitive.tracker_id).strip()
        public_tracker_id = public_track.get(
            "tracker_id",
            public_track.get("track_id"),
        )
        expected_tracker_id = _identity_v2_tracker_id_for_epoch(
            public_tracker_id,
            epoch,
        )
        if primitive_tracker_id != expected_tracker_id:
            raise ValueError(
                "identity-v2 primitive tracker does not match its public track"
            )
        if str(public_track.get("camera_id") or "").strip() != str(
            primitive.camera_id
        ).strip():
            raise ValueError(
                "identity-v2 primitive camera does not match its public track"
            )
        try:
            public_frame_id = int(public_track.get("frame_id"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "identity-v2 public track has no valid frame identity"
            ) from exc
        if public_frame_id != int(primitive.frame_id):
            raise ValueError(
                "identity-v2 primitive frame does not match its public track"
            )
        try:
            public_source_epoch = int(public_track.get("source_epoch"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "identity-v2 public track has no valid source epoch"
            ) from exc
        if public_source_epoch != epoch:
            raise ValueError(
                "identity-v2 public track source epoch does not match its cohort"
            )
        copied.append(
            IdentityFramePrimitive(
                camera_id=str(primitive.camera_id),
                tracker_id=primitive_tracker_id,
                frame_id=int(primitive.frame_id),
                public_track=public_track,
                embedding=_copy_bounded_identity_sequence(
                    primitive.embedding,
                    # One extra value preserves the service's existing invalid-
                    # dimension rejection without copying an unbounded vector.
                    maximum=dimension + 1,
                    field_name="identity-v2 embedding",
                ),
                diagnostic_track=None,
                bbox=_copy_bounded_identity_sequence(
                    primitive.bbox,
                    maximum=4,
                    field_name="identity-v2 bbox",
                ),
                frame_size=_copy_bounded_identity_sequence(
                    primitive.frame_size,
                    maximum=2,
                    field_name="identity-v2 frame_size",
                ),
                detection_confidence=(
                    None
                    if primitive.detection_confidence is None
                    else float(primitive.detection_confidence)
                ),
                tracker_confidence=(
                    None
                    if primitive.tracker_confidence is None
                    else float(primitive.tracker_confidence)
                ),
                world_xyz=_copy_bounded_identity_sequence(
                    primitive.world_xyz,
                    maximum=3,
                    field_name="identity-v2 world_xyz",
                ),
                world_valid=bool(primitive.world_valid),
            )
        )
    if primitive_by_track:
        raise ValueError(
            "identity-v2 shadow primitive is not bound to a published track"
        )
    return tuple(copied)


@dataclass(frozen=True)
class _ScalarPublicationFrameMeta:
    """Frame metadata snapshot used by the worker; no SDK object is retained."""

    frame_number: int
    frame_num: int
    buf_pts: int
    source_id: int


@dataclass
class _TrackingPublicationWork:
    source_id: int
    camera_id: str
    frame_id: int
    observed_at_us: int
    now_ts: float
    timestamp_us: int
    tracks: List[Dict[str, Any]]
    footpoints: List[Footpoint]
    temporal_contract: Dict[str, Any]
    # Lifecycle stamping is performed in the media callback before its rows
    # are exposed to OSD.  The worker carries that exact continuity receipt;
    # it must never derive a second generation from a later snapshot.
    continuity: TrackingContinuityUpdate | None = None
    source_epoch: int = 0
    admitted_at_ns: int = field(default_factory=time.perf_counter_ns)


@dataclass(frozen=True)
class _ShadowIdentityWork:
    """Detached optional identity work; no canonical track row is retained."""

    source_id: int
    camera_id: str
    frame_id: int
    observed_at: float
    primitives: Tuple[IdentityFramePrimitive, ...]
    admission_sequence: int
    admitted_at_ns: int = field(default_factory=time.perf_counter_ns)


class _ShadowIdentityWorker:
    """Latest-per-source shadow scorer isolated from canonical publication.

    Shadow IdentityV2 performs diagnostics and may also mutate its private
    visitor gallery.  Those SQLite transactions have measured tails far above
    one tracking publication interval, so they cannot share the ordered
    tracking/world/BEV worker.  This worker retains at most one pending item
    per source and replaces stale pending work when it falls behind.  The
    current item is worker-owned; canonical rows and SDK metadata are never
    referenced or mutated.
    """

    def __init__(
        self,
        processor: "_AnalyticsTelemetryProcessor",
        *,
        failure_callback: Optional[Callable[[BaseException], None]] = None,
        max_pending_sources: int = _IDENTITY_V2_SHADOW_MAX_PENDING_SOURCES,
    ) -> None:
        self._processor = processor
        self._failure_callback = failure_callback
        self._max_pending_sources = max(1, int(max_pending_sources))
        self._condition = threading.Condition(threading.Lock())
        self._pending: Dict[int, _ShadowIdentityWork] = {}
        self._stopping = False
        self._terminal_failure: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="NoesisShadowIdentity",
            daemon=True,
        )
        self._thread.start()

    @property
    def terminal_failure(self) -> BaseException | None:
        with self._condition:
            return self._terminal_failure

    @property
    def pending_count(self) -> int:
        with self._condition:
            return len(self._pending)

    def enqueue(self, work: _ShadowIdentityWork) -> bool:
        source = int(work.source_id)
        with self._condition:
            if self._stopping or self._terminal_failure is not None:
                _increment_core_counter("identity_v2.shadow.dropped_unavailable_total")
                return False
            existing = self._pending.get(source)
            if existing is None and len(self._pending) >= self._max_pending_sources:
                _increment_core_counter("identity_v2.shadow.dropped_capacity_total")
                return False
            if existing is not None:
                # The older frame is optional comparison evidence. Replacing
                # it keeps this lane fresh without delaying its canonical
                # tracking/world/BEV cohort.
                _increment_core_counter("identity_v2.shadow.coalesced_total")
            self._pending[source] = work
            _set_core_counter(
                "identity_v2.shadow.pending_sources",
                len(self._pending),
            )
            _max_core_counter(
                "identity_v2.shadow.pending_sources_high_watermark",
                len(self._pending),
            )
            _increment_core_counter("identity_v2.shadow.enqueued_total")
            self._condition.notify()
            return True

    def _take(self) -> Optional[_ShadowIdentityWork]:
        with self._condition:
            yielded_for_backlog = False
            while not self._stopping:
                if not self._pending:
                    self._condition.wait()
                    continue
                # Keep optional work pending while canonical cohorts are
                # queued. Newer source work can still replace it, so this is
                # a bounded yield/coalescing point rather than an unbounded
                # shadow queue or a canonical drop.
                backlogged = getattr(
                    self._processor,
                    "_canonical_publication_backlogged",
                    None,
                )
                if callable(backlogged) and bool(backlogged()):
                    if not yielded_for_backlog:
                        _increment_core_counter(
                            "identity_v2.shadow.yielded_canonical_backlog_total"
                        )
                        yielded_for_backlog = True
                    self._condition.wait(timeout=0.005)
                    continue
                # The callback samples observed_at from one monotonic wall-clock
                # domain. Oldest-first selection preserves that global order when
                # camera-local pending work has been replaced.
                source, work = min(
                    self._pending.items(),
                    key=lambda item: (
                        float(item[1].observed_at),
                        int(item[1].admission_sequence),
                        int(item[0]),
                    ),
                )
                self._pending.pop(source, None)
                _set_core_counter(
                    "identity_v2.shadow.pending_sources",
                    len(self._pending),
                )
                return work
            return None

    def _fail_shadow(self, error: BaseException) -> None:
        callback = self._failure_callback
        with self._condition:
            if self._terminal_failure is None:
                self._terminal_failure = error
            discarded = len(self._pending)
            self._pending.clear()
            _set_core_counter("identity_v2.shadow.pending_sources", 0)
            self._stopping = True
            self._condition.notify_all()
        _increment_core_counter("identity_v2.shadow.worker_failures_total")
        if discarded:
            _increment_core_counter(
                "identity_v2.shadow.dropped_after_failure_total",
                discarded,
            )
        if callable(callback):
            try:
                callback(error)
            except Exception:
                logger.exception("Shadow identity degradation callback failed")

    def _run(self) -> None:  # pragma: no cover - exercised by focused worker tests
        while True:
            work = self._take()
            if work is None:
                return
            started_ns = time.perf_counter_ns()
            _record_core_stage_timing(
                "identity_v2.shadow_queue_wait",
                int(work.admitted_at_ns),
                item_count=len(work.primitives),
            )
            _set_core_counter("identity_v2.shadow.inflight", 1)
            try:
                self._processor._process_identity_v2_source_frame(
                    camera_id=str(work.camera_id),
                    frame_id=int(work.frame_id),
                    primitives=work.primitives,
                    observed_at=float(work.observed_at),
                    fatal=False,
                )
                _increment_core_counter("identity_v2.shadow.completed_total")
            except Exception as exc:
                logger.exception(
                    "Optional shadow identity worker failed for source %s frame %s; "
                    "canonical tracking remains active",
                    work.source_id,
                    work.frame_id,
                )
                self._fail_shadow(exc)
                return
            finally:
                _set_core_counter("identity_v2.shadow.inflight", 0)
                _record_core_stage_timing(
                    "identity_v2.shadow_process_source_frame",
                    started_ns,
                    item_count=len(work.primitives),
                )

    def shutdown(self, *, wait: bool = True, timeout_s: float = 5.0) -> None:
        with self._condition:
            discarded = len(self._pending)
            self._pending.clear()
            _set_core_counter("identity_v2.shadow.pending_sources", 0)
            self._stopping = True
            self._condition.notify_all()
        if discarded:
            _increment_core_counter(
                "identity_v2.shadow.dropped_shutdown_total",
                discarded,
            )
        if not wait:
            return
        timeout = max(0.001, float(timeout_s))
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise TimeoutError("shadow identity worker did not quiesce")


class _TrackingPublicationWorker:
    """One bounded FIFO worker for tracking + paired BEV output.

    The callback only extracts public scalar metadata and appends a bounded
    item for the source.  The worker fairly rotates sources while preserving
    exact per-source order.  Overflow is terminal rather than silently
    dropping a canonical tracking/world/BEV cohort.  A publisher failure is
    likewise terminal and is surfaced through the existing runtime failure
    callback.
    """

    def __init__(
        self,
        processor: "_AnalyticsTelemetryProcessor",
        *,
        failure_callback: Optional[Callable[[BaseException], None]] = None,
        max_pending_per_source: int = 16,
        max_pending_total: int = 64,
    ) -> None:
        self._processor = processor
        self._failure_callback = failure_callback
        self._condition = threading.Condition(threading.Lock())
        self._pending: Dict[int, deque[_TrackingPublicationWork]] = {}
        self._pending_total = 0
        self._max_pending_per_source = max(1, int(max_pending_per_source))
        self._max_pending_total = max(
            self._max_pending_per_source,
            int(max_pending_total),
        )
        self._ready_sources: deque[int] = deque()
        self._stopping = False
        self._terminal_failure: BaseException | None = None
        self._inflight = False
        self._last_frame_by_source: Dict[int, int] = {}
        self._last_epoch_by_source: Dict[int, int] = {}
        self._thread = threading.Thread(
            target=self._run,
            name="NoesisTrackingPublication",
            daemon=True,
        )
        self._thread.start()

    @property
    def terminal_failure(self) -> BaseException | None:
        with self._condition:
            return self._terminal_failure

    @property
    def busy(self) -> bool:
        """Whether canonical work is queued or currently being published."""

        with self._condition:
            return bool(self._pending_total or self._inflight)

    def enqueue(self, work: _TrackingPublicationWork) -> bool:
        source = int(work.source_id)
        callback: Optional[Callable[[BaseException], None]] = None
        error: BaseException | None = None
        with self._condition:
            if self._stopping or self._terminal_failure is not None:
                return False
            pending = self._pending.setdefault(source, deque())
            if (
                len(pending) >= self._max_pending_per_source
                or self._pending_total >= self._max_pending_total
            ):
                error = RuntimeError(
                    "tracking publication worker queue overflow for source "
                    f"{source} frame {work.frame_id}"
                )
                # Do not silently drop canonical cohorts.  Transition to the
                # terminal state while still under the condition, then invoke
                # the callback outside the lock.  Already-admitted work stays
                # in its per-source FIFO and drains before the worker exits;
                # only this newly rejected cohort is refused.
                self._terminal_failure = error
                self._stopping = True
                _increment_core_counter(
                    "tracking.publication_worker.overflow_total"
                )
                self._condition.notify_all()
                callback = self._failure_callback
            else:
                if not pending:
                    self._ready_sources.append(source)
                pending.append(work)
                self._pending_total += 1
                _set_core_counter(
                    "tracking.publication_worker.pending_total",
                    self._pending_total,
                )
                _max_core_counter(
                    "tracking.publication_worker.pending_high_watermark",
                    self._pending_total,
                )
                _increment_core_counter(
                    "tracking.publication_worker.enqueued_total"
                )
                self._condition.notify()
                return True
        if callback is not None and error is not None:
            try:
                callback(error)
            except Exception:
                logger.exception("Tracking publication overflow callback failed")
        return False

    def _take(self) -> Optional[_TrackingPublicationWork]:
        with self._condition:
            while not self._ready_sources and not self._stopping:
                self._condition.wait()
            if not self._ready_sources:
                return None
            source = self._ready_sources.popleft()
            pending = self._pending.get(source)
            if not pending:
                self._pending.pop(source, None)
                return None
            work = pending.popleft()
            self._pending_total = max(0, self._pending_total - 1)
            _set_core_counter(
                "tracking.publication_worker.pending_total",
                self._pending_total,
            )
            self._inflight = True
            if pending:
                self._ready_sources.append(source)
            else:
                self._pending.pop(source, None)
            return work

    def _fail_terminal(self, error: BaseException) -> None:
        _increment_core_counter("tracking.publication_worker.failures_total")
        with self._condition:
            if self._terminal_failure is None:
                self._terminal_failure = error
            self._pending.clear()
            self._ready_sources.clear()
            self._pending_total = 0
            _set_core_counter("tracking.publication_worker.pending_total", 0)
            self._stopping = True
            self._condition.notify_all()
        if callable(self._failure_callback):
            try:
                self._failure_callback(error)
            except Exception:
                logger.exception("Tracking publication failure callback failed")

    def _run(self) -> None:  # pragma: no cover - exercised by focused worker tests
        while True:
            work = self._take()
            if work is None:
                return
            item_started_ns = time.perf_counter_ns()
            _record_core_stage_timing(
                "tracking.publication_worker_queue_wait",
                int(work.admitted_at_ns),
                item_count=len(work.tracks),
            )
            _set_core_counter("tracking.publication_worker.inflight", 1)
            try:
                source = int(work.source_id)
                epoch = int(work.source_epoch)
                last_epoch = self._last_epoch_by_source.get(source)
                if last_epoch is not None and epoch < int(last_epoch):
                    raise RuntimeError(
                        "tracking publication worker received an older source epoch "
                        f"for source {source}: {epoch} < {last_epoch}"
                    )
                if last_epoch is not None and epoch > int(last_epoch):
                    self._last_frame_by_source.pop(source, None)
                self._last_epoch_by_source[source] = epoch
                last_frame = self._last_frame_by_source.get(source)
                if last_frame is not None and int(work.frame_id) <= int(last_frame):
                    raise RuntimeError(
                        "tracking publication worker received a non-advancing frame "
                        f"for source {work.source_id}: {work.frame_id} <= {last_frame}"
                    )
                self._processor._publish_tracking_work(work)
                self._last_frame_by_source[int(work.source_id)] = int(work.frame_id)
                _increment_core_counter(
                    "tracking.publication_worker.completed_total"
                )
            except Exception as exc:
                logger.exception(
                    "Canonical tracking publication worker failed for source %s frame %s",
                    work.source_id,
                    work.frame_id,
                )
                self._fail_terminal(exc)
                return
            finally:
                with self._condition:
                    self._inflight = False
                    self._condition.notify_all()
                _set_core_counter("tracking.publication_worker.inflight", 0)
                _record_core_stage_timing(
                    "tracking.publication_worker_item",
                    item_started_ns,
                    item_count=len(work.tracks),
                )

    def shutdown(self, *, wait: bool = True, timeout_s: float = 5.0) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        if not wait:
            return
        timeout = max(0.001, float(timeout_s))
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise TimeoutError("tracking publication worker did not quiesce")


@dataclass
class _AnalyticsTelemetryProcessor:
    pipeline: "DS8Pipeline"
    tracking_pub: "TrackingTelemetryPublisher"
    camera_labels: Mapping[int, str]
    sensor_id_map: Mapping[int, int]
    publication_gate: RuntimePublicationGate = field(
        default_factory=RuntimePublicationGate
    )
    tracking_mode: Optional[str] = None
    bev_renderer: Any = None
    bev_calibration: Any = None
    depth_registration: DepthRegistrationManager | None = None
    world_fusion_policy: WorldFusionPolicy | None = None
    # Retired room-specific selection policy retained only for the
    # request-gated dashboard comparison. Canonical resolution never consults
    # this field.
    legacy_world_fusion_policy: WorldFusionPolicy | None = None
    scene_priors: ScenePriorSet | None = None
    diagnostics_logger: Any = None
    osd_label_processor: Any = None
    _analytics_obj_meta_type: Any = field(default=None, init=False, repr=False)
    _analytics_unique_id: int | None = field(default=None, init=False, repr=False)
    _zone_state: Dict[int, Dict[int, Dict[str, Any]]] = field(default_factory=dict, init=False, repr=False)
    _occupancy_state: Dict[int, Dict[str, int]] = field(default_factory=dict, init=False, repr=False)
    _occupancy_last_seen: Dict[int, Dict[str, float]] = field(default_factory=dict, init=False, repr=False)
    _occupancy_grace_s: float = field(default=0.0, init=False, repr=False)
    _active_tracks: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict, init=False, repr=False)
    _active_tracks_lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False
    )
    # Exact scalar rows retained only long enough for the tiler/OSD callback
    # to join the same frame.  This is deliberately not a latest-only cache.
    _track_snapshot_ring: Dict[
        int, "OrderedDict[int, Dict[int, Dict[str, Any]]]"
    ] = field(default_factory=dict, init=False, repr=False)
    _track_snapshot_ring_size: int = field(default=8, init=False, repr=False)
    _transitions_state: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict, init=False, repr=False)
    _stable_id_enabled: bool = field(default=True, init=False, repr=False)
    _v3dt_reid_track_grace_s: float = field(default=0.0, init=False, repr=False)
    _v3dt_reid_last_seen_by_track: Dict[Tuple[int, int], float] = field(
        default_factory=dict, init=False, repr=False
    )
    _mv3dt_present_track_ids_by_sensor: Dict[int, set[int]] = field(
        default_factory=dict, init=False, repr=False
    )
    _bev_class_ids: frozenset[int] = field(default_factory=lambda: frozenset({0}), init=False, repr=False)
    _bev_class_ids_ready: bool = field(default=False, init=False, repr=False)
    _reid_unique_id: int = field(default=3, init=False, repr=False)
    _reid_layer_name: str = field(default=REID_SWIN_OUTPUT_LAYER, init=False, repr=False)
    _reid_embedding_dim: int = field(default=REID_SWIN_EMBEDDING_DIM, init=False, repr=False)
    _reid_diag_use_tracker_id: bool = field(default=False, init=False, repr=False)
    _mask_alpha: float = field(default=0.35, init=False, repr=False)
    _mask_alpha_ready: bool = field(default=False, init=False, repr=False)
    _reid_logged_shape: bool = field(default=False, init=False, repr=False)
    _reid_debug_last_log: float = field(default=0.0, init=False, repr=False)
    _reid_debug_frames: int = field(default=0, init=False, repr=False)
    _reid_debug_objects: int = field(default=0, init=False, repr=False)
    _reid_debug_people: int = field(default=0, init=False, repr=False)
    _reid_debug_emb_found: int = field(default=0, init=False, repr=False)
    _reid_debug_emb_missing: int = field(default=0, init=False, repr=False)
    _diag_logged: bool = field(default=False, init=False, repr=False)
    _tracking_mode: str = field(default="baseline", init=False, repr=False)
    _world_frame: str = field(default="backend_world_m", init=False, repr=False)
    _image_flip_by_key: Dict[str, Tuple[bool, bool]] = field(default_factory=dict, init=False, repr=False)
    _image_flip_logged: set[str] = field(default_factory=set, init=False, repr=False)
    _v3dt_meta_enabled: bool = field(default=True, init=False, repr=False)
    _v3dt_meta_logged_missing: bool = field(default=False, init=False, repr=False)
    _v3dt_caminfo_paths: Dict[int, Path] = field(default_factory=dict, init=False, repr=False)
    _v3dt_caminfo_cache: Dict[int, Tuple[str, List[List[float]]]] = field(default_factory=dict, init=False, repr=False)
    _v3dt_caminfo_logged_missing: bool = field(default=False, init=False, repr=False)
    _v3dt_axis_map: V3DTAxisMap | None = field(default=None, init=False, repr=False)
    _sid_metrics_log_enabled: bool = field(default=True, init=False, repr=False)
    _sid_metrics_log_interval_s: float = field(default=10.0, init=False, repr=False)
    _sid_metrics_last_log_by_sensor: Dict[int, float] = field(default_factory=dict, init=False, repr=False)
    _world_state_by_track: Dict[Tuple[int, int], _WorldAnchorState] = field(default_factory=dict, init=False, repr=False)
    _world_state_ghost_by_track: Dict[
        Tuple[int, int], Tuple[_WorldAnchorState, float]
    ] = field(default_factory=dict, init=False, repr=False)
    _world_output_watermarks: "OrderedDict[Tuple[int, int, int, str, str, str], _WorldOutputWatermark]" = field(
        default_factory=OrderedDict,
        init=False,
        repr=False,
    )
    _world_output_watermark_capacity: int = field(
        default=4096,
        init=False,
        repr=False,
    )
    _world_state_ghost_ttl_s: float = field(default=0.75, init=False, repr=False)
    _world_state_ttl_s: float = field(default=3.0, init=False, repr=False)
    _world_state_prune_interval_s: float = field(default=1.0, init=False, repr=False)
    _world_state_last_prune_ts: float = field(default=0.0, init=False, repr=False)
    _world_static_px_threshold: float = field(default=3.0, init=False, repr=False)
    _world_static_jump_scene: float = field(default=10.0, init=False, repr=False)
    _world_max_speed_scene_per_s: float = field(default=4.0, init=False, repr=False)
    _world_smooth_alpha_good: float = field(default=0.45, init=False, repr=False)
    _world_smooth_alpha_weak: float = field(default=0.20, init=False, repr=False)
    _human_ground_cfg: HumanGroundConfig = field(default_factory=HumanGroundConfig, init=False, repr=False)
    _pose_anchor_gie_id: int = field(default=4, init=False, repr=False)
    _pose_anchor_model_size: Tuple[int, int] = field(default=(640, 640), init=False, repr=False)
    _pose_anchor_score_threshold: float = field(default=0.25, init=False, repr=False)
    _pose_anchor_letterbox: bool = field(default=True, init=False, repr=False)
    _pose_anchor_kpt_threshold: float = field(default=0.35, init=False, repr=False)
    _world_height_update_alpha: float = field(default=0.20, init=False, repr=False)
    _world_height_min_m: float = field(default=0.60, init=False, repr=False)
    _world_height_max_m: float = field(default=2.40, init=False, repr=False)
    _world_anchor_hold_ttl_s: float = field(default=0.40, init=False, repr=False)
    _world_stationary_hold_ttl_s: float = field(default=2.0, init=False, repr=False)
    # The canonical DS9 config explicitly enables this resolver.  A processor
    # constructed without the canonical block remains a legacy fixture; the
    # active native baseline never relies on that omission.
    _world_resolver_enabled: bool = field(default=False, init=False, repr=False)
    _world_resolver_max_range_m: float = field(default=22.0, init=False, repr=False)
    _world_resolver_max_disagreement_m: float = field(default=1.25, init=False, repr=False)
    _world_resolver_diag_max_candidates: int = field(default=4, init=False, repr=False)
    # Rich candidate diagnostics are optional presentation work.  Canonical
    # resolution and the compact public summary remain enabled independently;
    # keeping this false avoids recursively copying and serializing the full
    # hypothesis tree on every exact tracking/BEV cohort.
    _world_resolver_diagnostics_enabled: bool = field(
        default=False,
        init=False,
        repr=False,
    )
    _world_resolver: UniversalWorldMeasurementResolver | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _reid_embeds_per_frame_max: int = field(default=2, init=False, repr=False)
    _pose_anchor_native_per_frame_max: int = field(default=1, init=False, repr=False)
    _pose_anchor_native_remaining: int = field(default=1, init=False, repr=False)
    _tracking_publish_interval_s: float = field(default=0.0, init=False, repr=False)
    _tracking_empty_publish_interval_s: float = field(default=0.5, init=False, repr=False)
    _bev_publish_interval_s: float = field(default=0.0, init=False, repr=False)
    _last_tracking_publish_ts_by_sensor: Dict[int, float] = field(default_factory=dict, init=False, repr=False)
    _last_bev_publish_ts_by_sensor: Dict[int, float] = field(default_factory=dict, init=False, repr=False)
    _last_tracking_count_by_sensor: Dict[int, int] = field(default_factory=dict, init=False, repr=False)
    _last_bev_count_by_sensor: Dict[int, int] = field(default_factory=dict, init=False, repr=False)
    _last_tracking_enqueue_ts_by_sensor: Dict[int, float] = field(default_factory=dict, init=False, repr=False)
    _last_tracking_enqueue_count_by_sensor: Dict[int, int] = field(default_factory=dict, init=False, repr=False)
    _last_observed_tracker_keys_by_sensor: Dict[int, frozenset[int]] = field(
        default_factory=dict, init=False, repr=False
    )
    _tracking_publication_worker: _TrackingPublicationWorker | None = field(
        default=None, init=False, repr=False
    )
    _shadow_identity_worker: _ShadowIdentityWorker | None = field(
        default=None, init=False, repr=False
    )
    _shadow_identity_admission_sequence: int = field(
        default=0, init=False, repr=False
    )
    _shadow_identity_publish_interval_s: float = field(
        default=0.5, init=False, repr=False
    )
    _last_shadow_identity_enqueue_ts_by_sensor: Dict[int, float] = field(
        default_factory=dict, init=False, repr=False
    )
    _tracking_lifecycle: TrackingLifecycleRegistry = field(
        default_factory=TrackingLifecycleRegistry, init=False, repr=False
    )
    _source_frame_lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False
    )
    _last_source_frame_by_sensor: Dict[int, Tuple[int, int]] = field(
        default_factory=dict, init=False, repr=False
    )
    _source_frame_inflight: Dict[int, Tuple[int, int, int]] = field(
        default_factory=dict, init=False, repr=False
    )
    _source_epoch_by_sensor: Dict[int, int] = field(
        default_factory=dict, init=False, repr=False
    )
    # World/filter time is source-local media time rebased onto the observed
    # wall-clock epoch.  It is deliberately separate from the publication
    # ordering clock, whose exact observed_at_us contract remains unchanged.
    _world_clock_by_sensor: Dict[int, _WorldClockState] = field(
        default_factory=dict, init=False, repr=False
    )
    _last_media_pts_ns_by_sensor: Dict[int, int] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        # Discover the ReID SGIE unique-id from the built pipeline config when present.
        try:
            models_cfg = getattr(self.pipeline, "config", {}).get("models", {}) or {}
            reid_cfg = models_cfg.get("reid") or {}
            if isinstance(reid_cfg, dict):
                gie_id = reid_cfg.get("gie_id", reid_cfg.get("gie-id", None))
                if gie_id is not None:
                    self._reid_unique_id = int(gie_id)
                layer_name = reid_cfg.get("layer")
                if layer_name:
                    self._reid_layer_name = str(layer_name)
                embedding_dim = reid_cfg.get("embedding_dim")
                if embedding_dim is not None:
                    self._reid_embedding_dim = int(embedding_dim)
        except Exception:
            self._reid_unique_id = 3

        try:
            analytics_cfg = getattr(self.pipeline, "config", {}).get("analytics", {}) or {}
            stages_path = Path(str(analytics_cfg.get("stages_config") or "").strip())
            if str(stages_path) not in {"", "."}:
                stages_payload = yaml.safe_load(stages_path.read_text(encoding="utf-8")) or {}
                unique_id = (
                    ((stages_payload.get("analytics") or {}).get("stages") or {})
                    .get("post", {})
                    .get("unique_id")
                )
                if (
                    isinstance(unique_id, int)
                    and not isinstance(unique_id, bool)
                    and unique_id > 0
                ):
                    self._analytics_unique_id = int(unique_id)
        except Exception:
            self._analytics_unique_id = None

        self._tracking_mode = self._resolve_tracking_mode(self.tracking_mode)
        v3dt_cfg = getattr(self.pipeline, "config", {}).get("v3dt", {}) or {}
        if self._tracking_mode_is_mv3dt():
            evaluation_requested = (
                str(os.environ.get("NOESIS_MV3DT_EVALUATION", "")).strip().lower()
                in {"1", "true", "yes", "y", "on"}
            )
            expected_activation = (
                "evaluation_only" if evaluation_requested else "ready_opt_in"
            )
            if (
                not isinstance(v3dt_cfg, Mapping)
                or v3dt_cfg.get("activation_state") != expected_activation
            ):
                raise ValueError(
                    "MV3DT runtime/profile mismatch: expected activation_state="
                    f"{expected_activation} for the current launch"
                )

        if self._tracking_mode_is_v3dt():
            if not isinstance(v3dt_cfg, Mapping):
                raise ValueError("V3DT runtime requires an explicit v3dt config mapping")
            if v3dt_cfg.get("world_frame") != "backend_world_m":
                raise ValueError(
                    "V3DT runtime requires world_frame=backend_world_m after camInfo axis restoration"
                )
            self._world_frame = "backend_world_m"
            self._v3dt_axis_map = V3DTAxisMap.parse(
                v3dt_cfg.get("caminfo_world_axes")
            )
        elif isinstance(v3dt_cfg, Mapping) and v3dt_cfg.get("world_frame"):
            self._world_frame = str(v3dt_cfg["world_frame"])

        if self._tracking_mode_is_v3dt():
            try:
                grace_v3dt_cfg = getattr(self.pipeline, "config", {}).get("v3dt", {}) or {}
                configured_grace_s = (
                    grace_v3dt_cfg.get("reid_track_grace_s", 0.0)
                    if isinstance(grace_v3dt_cfg, Mapping)
                    else 0.0
                )
                grace_raw = os.environ.get(
                    "NOESIS_V3DT_REID_TRACK_GRACE_S",
                    configured_grace_s,
                )
                grace_s = float(str(grace_raw).strip() or "0")
                if not math.isfinite(grace_s):
                    raise ValueError("grace must be finite")
                self._v3dt_reid_track_grace_s = min(5.0, max(0.0, grace_s))
            except (TypeError, ValueError):
                self._v3dt_reid_track_grace_s = 0.0

        flag = str(os.environ.get("NOESIS_V3DT_META_EXTRACT", "1") or "").strip().lower()
        self._v3dt_meta_enabled = flag in ("", "1", "true", "yes", "y", "on")
        if not self._tracking_mode_is_v3dt():
            self._v3dt_meta_enabled = False
        if self._tracking_mode_is_v3dt():
            self._ensure_v3dt_caminfo_paths()
        self._warn_on_tracking_mode_mismatch()
        try:
            grace_raw = os.environ.get("NOESIS_OCCUPANCY_GRACE_S", "0")
            self._occupancy_grace_s = max(0.0, float(str(grace_raw).strip() or "0"))
        except Exception:
            self._occupancy_grace_s = 0.0
        diag_raw = str(os.environ.get("NOESIS_REID_DIAG_USE_TRACKER_ID", "0") or "").strip().lower()
        self._reid_diag_use_tracker_id = diag_raw in ("1", "true", "yes", "on", "y")
        sid_metrics_env = str(os.environ.get("NOESIS_REID_METRICS_LOG_ENABLED", "1") or "").strip().lower()
        self._sid_metrics_log_enabled = sid_metrics_env in ("1", "true", "yes", "on", "y")
        try:
            interval_raw = os.environ.get("NOESIS_REID_METRICS_LOG_INTERVAL_S", "10")
            self._sid_metrics_log_interval_s = max(1.0, float(str(interval_raw).strip() or "10"))
        except Exception:
            self._sid_metrics_log_interval_s = 10.0
        try:
            models_cfg = getattr(self.pipeline, "config", {}).get("models", {}) or {}
            pose_cfg = models_cfg.get("pose") or {}
            if isinstance(pose_cfg, Mapping):
                self._pose_anchor_gie_id = int(pose_cfg.get("gie_id", pose_cfg.get("gie-id", 4) or 4))
                model_size = pose_cfg.get("model_size") or pose_cfg.get("input_size")
                if isinstance(model_size, (list, tuple)) and len(model_size) >= 2:
                    self._pose_anchor_model_size = (int(model_size[0]), int(model_size[1]))
                elif isinstance(model_size, str) and "x" in model_size:
                    parts = model_size.lower().split("x")
                    if len(parts) >= 2:
                        self._pose_anchor_model_size = (int(parts[0].strip()), int(parts[1].strip()))
                self._pose_anchor_score_threshold = float(pose_cfg.get("score_threshold", 0.25) or 0.25)
                self._pose_anchor_letterbox = bool(pose_cfg.get("letterbox", True))
                self._pose_anchor_kpt_threshold = max(
                    0.0,
                    float(pose_cfg.get("kpt_threshold", 0.35) or 0.35),
                )
        except Exception:
            self._pose_anchor_gie_id = 4
            self._pose_anchor_model_size = (640, 640)
            self._pose_anchor_score_threshold = 0.25
            self._pose_anchor_letterbox = True
            self._pose_anchor_kpt_threshold = 0.35
        try:
            self._world_state_ttl_s = max(0.25, float(str(os.environ.get("NOESIS_WORLD_STATE_TTL_S", "3.0")).strip() or "3.0"))
        except Exception:
            self._world_state_ttl_s = 3.0
        try:
            self._world_state_prune_interval_s = max(
                0.10, float(str(os.environ.get("NOESIS_WORLD_STATE_PRUNE_INTERVAL_S", "1.0")).strip() or "1.0")
            )
        except Exception:
            self._world_state_prune_interval_s = 1.0
        try:
            self._world_static_px_threshold = max(
                0.0, float(str(os.environ.get("NOESIS_WORLD_STATIC_PX_THRESHOLD", "3.0")).strip() or "3.0")
            )
        except Exception:
            self._world_static_px_threshold = 3.0
        try:
            self._world_static_jump_scene = max(
                0.0, float(str(os.environ.get("NOESIS_WORLD_STATIC_JUMP_SCENE", "10.0")).strip() or "10.0")
            )
        except Exception:
            self._world_static_jump_scene = 10.0
        try:
            self._world_max_speed_scene_per_s = max(
                0.0, float(str(os.environ.get("NOESIS_WORLD_MAX_SPEED_SCENE_PER_S", "4.0")).strip() or "4.0")
            )
        except Exception:
            self._world_max_speed_scene_per_s = 4.0
        try:
            self._world_smooth_alpha_good = float(
                str(os.environ.get("NOESIS_WORLD_SMOOTH_ALPHA_GOOD", "0.45")).strip() or "0.45"
            )
        except Exception:
            self._world_smooth_alpha_good = 0.45
        try:
            self._world_smooth_alpha_weak = float(
                str(os.environ.get("NOESIS_WORLD_SMOOTH_ALPHA_WEAK", "0.20")).strip() or "0.20"
            )
        except Exception:
            self._world_smooth_alpha_weak = 0.20
        self._world_smooth_alpha_good = float(max(0.0, min(1.0, self._world_smooth_alpha_good)))
        self._world_smooth_alpha_weak = float(max(0.0, min(1.0, self._world_smooth_alpha_weak)))
        try:
            self._world_anchor_hold_ttl_s = max(
                0.0, float(str(os.environ.get("NOESIS_WORLD_ANCHOR_HOLD_TTL_S", "0.40")).strip() or "0.40")
            )
        except Exception:
            self._world_anchor_hold_ttl_s = 0.40
        try:
            self._world_stationary_hold_ttl_s = max(
                self._world_anchor_hold_ttl_s,
                float(
                    str(
                        os.environ.get(
                            "NOESIS_WORLD_STATIONARY_HOLD_TTL_S",
                            "2.0",
                        )
                    ).strip()
                    or "2.0"
                ),
            )
        except Exception:
            self._world_stationary_hold_ttl_s = max(
                self._world_anchor_hold_ttl_s,
                2.0,
            )
        # Universal world measurement resolution is an explicit canonical
        # baseline capability.  It consumes one generic config block rather
        # than selecting behavior from camera or room names.  Environment
        # overrides are intentionally absent: changing estimator behavior is a
        # config/replay decision, not a hot-path process toggle.
        pipeline_cfg = getattr(self.pipeline, "config", {}) or {}
        canonical_cfg = (
            pipeline_cfg.get("canonical_world", {})
            if isinstance(pipeline_cfg, Mapping)
            else {}
        )
        resolver_cfg = (
            canonical_cfg.get("measurement_resolver")
            if isinstance(canonical_cfg, Mapping)
            else None
        )
        if resolver_cfg is not None:
            if not isinstance(resolver_cfg, Mapping):
                raise ValueError("canonical_world.measurement_resolver must be a mapping")
            try:
                required_resolver_keys = {
                    "enabled",
                    "max_range_m",
                    "max_disagreement_m",
                    "max_candidates",
                }
                allowed_resolver_keys = {
                    *required_resolver_keys,
                    "legacy_comparison_policy_path",
                }
                if (
                    not required_resolver_keys.issubset(resolver_cfg)
                    or not set(resolver_cfg).issubset(allowed_resolver_keys)
                ):
                    raise ValueError("resolver fields do not match the exact contract")
                enabled = resolver_cfg.get("enabled", False)
                raw_max_range = resolver_cfg.get("max_range_m", 22.0)
                raw_max_disagreement = resolver_cfg.get("max_disagreement_m", 1.25)
                raw_max_candidates = resolver_cfg.get("max_candidates", 4)
                legacy_comparison_policy_path = resolver_cfg.get(
                    "legacy_comparison_policy_path"
                )
                if not isinstance(enabled, bool):
                    raise ValueError("resolver enabled must be boolean")
                if isinstance(raw_max_range, bool) or isinstance(raw_max_disagreement, bool):
                    raise ValueError("resolver distance limits must be numeric")
                if isinstance(raw_max_candidates, bool) or not isinstance(raw_max_candidates, int):
                    raise ValueError("resolver max_candidates must be an integer")
                if (
                    legacy_comparison_policy_path is not None
                    and (
                        not isinstance(legacy_comparison_policy_path, str)
                        or not legacy_comparison_policy_path.strip()
                    )
                ):
                    raise ValueError(
                        "resolver legacy_comparison_policy_path must be a non-empty string"
                    )
                max_range = float(raw_max_range)
                max_disagreement = float(raw_max_disagreement)
                max_candidates = int(raw_max_candidates)
                if not math.isfinite(max_range) or not 1.0 <= max_range <= 100.0:
                    raise ValueError("resolver max_range_m is outside [1, 100]")
                if (
                    not math.isfinite(max_disagreement)
                    or not 0.05 <= max_disagreement <= 10.0
                ):
                    raise ValueError(
                        "resolver max_disagreement_m is outside [0.05, 10]"
                    )
                if not 1 <= max_candidates <= 4:
                    raise ValueError("resolver max_candidates is outside [1, 4]")
                self._world_resolver_enabled = enabled
                self._world_resolver_max_range_m = max_range
                self._world_resolver_max_disagreement_m = max_disagreement
                self._world_resolver_diag_max_candidates = max_candidates
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("canonical_world.measurement_resolver is invalid") from exc
        if self._world_resolver_enabled:
            # An enabled canonical resolver must fail construction loudly
            # rather than silently returning to legacy room-policy behavior.
            self._world_resolver = UniversalWorldMeasurementResolver(
                config=WorldMeasurementResolverConfig(
                    max_hypotheses=int(self._world_resolver_diag_max_candidates),
                    max_compatible_distance_m=float(
                        self._world_resolver_max_disagreement_m
                    ),
                )
            )
        self._human_ground_cfg = HumanGroundConfig(
            static_px_threshold=float(self._world_static_px_threshold),
            max_speed_mps=float(self._world_max_speed_scene_per_s),
            max_jump_m=0.75,
            alpha_good=float(self._world_smooth_alpha_good),
            alpha_weak=float(self._world_smooth_alpha_weak),
            kpt_conf_threshold=float(self._pose_anchor_kpt_threshold),
            rejected_prediction_horizon_s=float(self._world_anchor_hold_ttl_s),
        )
        self._reid_embeds_per_frame_max = _read_env_int("NOESIS_REID_EMBEDS_PER_FRAME_MAX", 2, min_value=0)
        self._pose_anchor_native_per_frame_max = _read_env_int(
            "NOESIS_POSE_ANCHOR_NATIVE_EXTRACTS_PER_FRAME_MAX",
            1,
            min_value=0,
        )
        tracking_max_hz = _read_env_float(
            "NOESIS_TRACKING_PUBLISH_MAX_HZ",
            _read_env_float("NOESIS_WS_TRACKING_MAX_HZ", 15.0, min_value=0.0),
            min_value=0.0,
        )
        bev_max_hz = _read_env_float(
            "NOESIS_BEV_PUBLISH_MAX_HZ",
            _read_env_float("NOESIS_WS_BEV_MAX_HZ", 12.0, min_value=0.0),
            min_value=0.0,
        )
        empty_tracking_hz = _read_env_float(
            "NOESIS_TRACKING_EMPTY_HEARTBEAT_HZ",
            2.0,
            min_value=0.0,
        )
        self._tracking_publish_interval_s = 0.0 if tracking_max_hz <= 0.0 else 1.0 / float(tracking_max_hz)
        self._tracking_empty_publish_interval_s = (
            0.0 if empty_tracking_hz <= 0.0 else 1.0 / float(empty_tracking_hz)
        )
        self._bev_publish_interval_s = 0.0 if bev_max_hz <= 0.0 else 1.0 / float(bev_max_hz)
        shadow_max_hz = _read_env_float(
            "NOESIS_IDENTITY_V2_SHADOW_MAX_HZ",
            2.0,
            min_value=0.1,
        )
        self._shadow_identity_publish_interval_s = 1.0 / float(shadow_max_hz)

    def set_world_resolver_diagnostics_enabled(self, enabled: bool) -> None:
        """Enable bounded resolver comparison details for presentation only."""

        self._world_resolver_diagnostics_enabled = bool(enabled)

    def _tracking_interval_for_frame(self, track_count: int) -> float:
        return pair_safe_publication_interval_s(
            track_count=int(track_count),
            tracking_interval_s=float(
                getattr(self, "_tracking_publish_interval_s", 0.0)
            ),
            empty_tracking_interval_s=float(
                getattr(self, "_tracking_empty_publish_interval_s", 0.0)
            ),
            bev_interval_s=float(getattr(self, "_bev_publish_interval_s", 0.0)),
            bev_active=getattr(self, "bev_renderer", None) is not None,
        )

    def _publish_gate_due(
        self,
        last_by_sensor: Dict[int, float],
        count_by_sensor: Dict[int, int],
        *,
        sensor_id: int,
        now_ts: float,
        count: int,
        interval_s: float,
        counter_prefix: str,
        update: bool = True,
        force: bool = False,
    ) -> bool:
        sid = int(sensor_id)
        current_count = int(count)
        previous_count = count_by_sensor.get(sid)
        last_ts = float(last_by_sensor.get(sid, 0.0))
        count_changed = previous_count is None or int(previous_count) != current_count
        due = bool(
            force
            or count_changed
            or interval_s <= 0.0
            or last_ts <= 0.0
            or (float(now_ts) - last_ts) >= float(interval_s)
        )
        if due:
            if update:
                last_by_sensor[sid] = float(now_ts)
                count_by_sensor[sid] = current_count
            _increment_core_counter(f"detection_wake.{counter_prefix}_publish_due")
            return True
        _increment_core_counter(f"detection_wake.{counter_prefix}_publish_skipped")
        return False

    def _observed_tracker_keys_changed(
        self,
        sensor_id: int,
        tracks: Sequence[Mapping[str, Any]],
    ) -> bool:
        current: set[int] = set()
        for track in tracks:
            if not isinstance(track, Mapping):
                continue
            try:
                tracker_id = int(track.get("tracker_id", track.get("track_id")))
            except (TypeError, ValueError):
                continue
            if tracker_id >= 0:
                current.add(tracker_id)
        current_keys = frozenset(current)
        previous = self._last_observed_tracker_keys_by_sensor.get(int(sensor_id))
        self._last_observed_tracker_keys_by_sensor[int(sensor_id)] = current_keys
        return previous is None or previous != current_keys

    def _ensure_tracking_publication_worker(self) -> _TrackingPublicationWorker:
        worker = self._tracking_publication_worker
        if worker is None:
            worker = _TrackingPublicationWorker(
                self,
                failure_callback=getattr(self.tracking_pub, "_report_failure", None),
            )
            self._tracking_publication_worker = worker
        return worker

    def _canonical_publication_backlogged(self) -> bool:
        """Return whether optional work must yield to canonical publication."""

        worker = self._tracking_publication_worker
        return worker is not None and worker.busy

    def _ensure_shadow_identity_worker(self) -> _ShadowIdentityWorker:
        worker = self._shadow_identity_worker
        if worker is None:
            worker = _ShadowIdentityWorker(
                self,
                failure_callback=getattr(
                    self.pipeline,
                    "identity_v2_shadow_failure_callback",
                    None,
                ),
            )
            self._shadow_identity_worker = worker
        return worker

    def _prepare_tracking_cohort(
        self,
        *,
        sensor_id: int,
        camera_id: str,
        frame_id: int,
        observed_at_us: int,
        tracks: Sequence[MutableMapping[str, Any]],
        footpoints: Sequence[Footpoint],
    ) -> TrackingContinuityUpdate:
        """Stamp one lifecycle on the rows shared by OSD, BEV, and worker.

        This runs after all public rows have been assembled but before
        ``_active_tracks`` and the scalar worker packet are created.  The
        worker therefore consumes the exact same generation-bearing mappings
        that the tiler/OSD exact-frame join observes.
        """

        source = int(sensor_id)
        frame = int(frame_id)
        observed = int(observed_at_us)
        with self._source_frame_lock:
            previous = self._last_source_frame_by_sensor.get(source)
            epoch = int(self._source_epoch_by_sensor.get(source, 0))
            prepared = self._source_frame_inflight.pop(source, None)
            if prepared is not None and prepared[:2] == (frame, observed):
                epoch = int(prepared[2])
            elif previous is not None:
                previous_frame, previous_observed = previous
                if frame <= int(previous_frame):
                    # DeepStream reconnects/file loops may restart the source
                    # frame counter.  Only a strictly newer processing clock
                    # can prove this is a new epoch; otherwise reject the
                    # reorder instead of silently publishing stale geometry.
                    if observed > int(previous_observed):
                        epoch = self._tracking_lifecycle.reset_source(source)
                        self._source_epoch_by_sensor[source] = int(epoch)
                        with self._active_tracks_lock:
                            self._track_snapshot_ring.pop(source, None)
                        _increment_core_counter(
                            "tracking.source_epoch_reconnect_total"
                        )
                    else:
                        raise RuntimeError(
                            "source frame is not advancing and timestamp did not "
                            f"prove reconnect: source={source} frame={frame} "
                            f"previous={previous_frame}"
                        )
                elif observed <= int(previous_observed):
                    raise RuntimeError(
                        "source observed_at_us reordered: "
                        f"source={source} observed={observed} "
                        f"previous={previous_observed}"
                    )

            continuity = self._tracking_lifecycle.update_frame(
                source_id=source,
                camera_id=str(camera_id),
                frame_id=frame,
                observed_at_us=observed,
                tracks=tracks,
            )
            self._source_epoch_by_sensor[source] = int(continuity.source_epoch)
            self._last_source_frame_by_sensor[source] = (frame, observed)

        self._stamp_footpoint_lifecycles(footpoints, tracks)
        for track in tracks:
            if isinstance(track, MutableMapping):
                track["source_epoch"] = int(continuity.source_epoch)
        return continuity

    @staticmethod
    def _valid_world_media_pts_ns(value: Any) -> Optional[int]:
        """Normalize one DeepStream PTS; zero/CLOCK_TIME_NONE is unavailable."""

        try:
            parsed = int(value or 0)
        except Exception:
            return None
        if parsed <= 0 or parsed >= (1 << 64) - 1:
            return None
        return parsed

    def _world_timestamp_for_frame(
        self,
        sensor_id: int,
        *,
        media_pts_ns: Any,
        observed_ts: float,
    ) -> float:
        """Return the filter clock for one source frame.

        Media PTS supplies the frame-to-frame delta whenever it is usable.
        The offset keeps the value in the observed wall-clock epoch so the
        existing bounded state/ghost TTLs remain meaningful across cameras.
        Missing or duplicated PTS falls back to a monotonic observed value;
        it never rewinds an already accepted filter timestamp.
        """

        source = int(sensor_id)
        try:
            observed = float(observed_ts)
        except Exception:
            observed = float(time.time())
        if not math.isfinite(observed) or observed <= 0.0:
            observed = float(time.time())
        observed = max(1e-9, observed)
        media = self._valid_world_media_pts_ns(media_pts_ns)
        with self._source_frame_lock:
            epoch = int(self._source_epoch_by_sensor.get(source, 0))
            clock = self._world_clock_by_sensor.get(source)
            if clock is None or int(clock.source_epoch) != epoch:
                clock = _WorldClockState(source_epoch=epoch)
                self._world_clock_by_sensor[source] = clock

            previous_logical = clock.logical_ts
            if media is not None:
                if clock.base_media_pts_ns is None:
                    clock.base_media_pts_ns = int(media)
                    clock.base_observed_ts = float(observed)
                    clock.logical_ts = float(observed)
                    clock.last_media_pts_ns = int(media)
                    clock.basis = "media_pts"
                elif (
                    clock.last_media_pts_ns is not None
                    and int(media) > int(clock.last_media_pts_ns)
                ):
                    base_pts = int(clock.base_media_pts_ns)
                    base_observed = float(clock.base_observed_ts or observed)
                    candidate = base_observed + (int(media) - base_pts) / 1e9
                    # A temporary fallback (for example, one unavailable PTS)
                    # must not make the filter time go backwards on recovery.
                    if previous_logical is not None and candidate < float(previous_logical):
                        clock.logical_ts = float(previous_logical)
                        clock.basis = "observed_clock_fallback"
                    else:
                        clock.logical_ts = float(candidate)
                        clock.basis = "media_pts"
                    clock.last_media_pts_ns = int(media)
                elif (
                    clock.last_media_pts_ns is not None
                    and int(media) == int(clock.last_media_pts_ns)
                ):
                    clock.logical_ts = max(float(previous_logical or 0.0), float(observed))
                    clock.basis = "observed_clock_fallback"
                else:
                    # A decreasing PTS is handled as a source reconnect by
                    # _begin_source_frame_timeline.  Keep this helper safe for
                    # direct callers and never feed a negative dt to the
                    # physical filter if metadata arrives out of order.
                    clock.logical_ts = max(float(previous_logical or 0.0), float(observed))
                    clock.basis = "observed_clock_fallback"
            else:
                clock.logical_ts = max(float(previous_logical or 0.0), float(observed))
                clock.basis = "observed_clock"
            clock.last_observed_ts = float(observed)
            return float(clock.logical_ts or observed)

    def _clear_world_source_state(self, source: int) -> None:
        """Discard world state and timebase at a physical source boundary."""

        source_id = int(source)
        self._world_clock_by_sensor.pop(source_id, None)
        for key in tuple(self._world_state_by_track):
            if int(key[0]) == source_id:
                self._world_state_by_track.pop(key, None)
        for key in tuple(self._world_state_ghost_by_track):
            if int(key[0]) == source_id:
                self._world_state_ghost_by_track.pop(key, None)
        output_watermarks = getattr(self, "_world_output_watermarks", {})
        for key in tuple(output_watermarks):
            if int(key[0]) == source_id:
                output_watermarks.pop(key, None)

    def _begin_source_frame_timeline(
        self,
        *,
        sensor_id: int,
        frame_id: int,
        observed_at_us: int,
        media_pts_ns: int = 0,
    ) -> int:
        """Validate/reset a source timeline before world estimation runs."""

        source = int(sensor_id)
        frame = int(frame_id)
        observed = int(observed_at_us)
        media = self._valid_world_media_pts_ns(media_pts_ns)
        with self._source_frame_lock:
            previous = self._last_source_frame_by_sensor.get(source)
            epoch = int(self._source_epoch_by_sensor.get(source, 0))
            epoch_reset = False
            if previous is not None:
                previous_frame, previous_observed = previous
                if frame <= int(previous_frame):
                    if observed > int(previous_observed):
                        epoch = self._tracking_lifecycle.reset_source(source)
                        epoch_reset = True
                        self._source_epoch_by_sensor[source] = int(epoch)
                        with self._active_tracks_lock:
                            self._track_snapshot_ring.pop(source, None)
                        # A reconnect is a hard physical timeline boundary;
                        # discard old filter/ghost state before this frame is
                        # allowed to request a generation.
                        self._clear_world_source_state(source)
                        _increment_core_counter(
                            "tracking.source_epoch_reconnect_total"
                        )
                    else:
                        raise RuntimeError(
                            "source frame is not advancing and timestamp did not "
                            f"prove reconnect: source={source} frame={frame} "
                            f"previous={previous_frame}"
                        )
                elif observed <= int(previous_observed):
                    raise RuntimeError(
                        "source observed_at_us reordered: "
                        f"source={source} observed={observed} "
                        f"previous={previous_observed}"
                    )
            previous_media = self._last_media_pts_ns_by_sensor.get(source)
            if (
                media is not None
                and previous_media is not None
                and int(media) < int(previous_media)
            ):
                # A source can restart while its frame counter continues. A
                # decreasing stream PTS is the only additional reconnect
                # signal needed here; equal PTS is tolerated as a duplicate
                # and uses the observed-clock fallback for that frame.
                if not epoch_reset:
                    epoch = self._tracking_lifecycle.reset_source(source)
                    epoch_reset = True
                    self._source_epoch_by_sensor[source] = int(epoch)
                    with self._active_tracks_lock:
                        self._track_snapshot_ring.pop(source, None)
                    self._clear_world_source_state(source)
                    _increment_core_counter(
                        "tracking.source_epoch_reconnect_total"
                    )
            self._source_epoch_by_sensor[source] = int(epoch)
            self._last_source_frame_by_sensor[source] = (frame, observed)
            if media is not None:
                self._last_media_pts_ns_by_sensor[source] = int(media)
            self._source_frame_inflight[source] = (frame, observed, int(epoch))
            return int(epoch)

    def _set_active_tracks_snapshot(
        self,
        sensor_id: int,
        frame_id: int,
        tracks: List[Dict[str, Any]],
    ) -> None:
        """Publish active rows and retain a bounded exact-frame scalar ring."""

        indexed: Dict[int, Dict[str, Any]] = {}
        for track in tracks:
            if not isinstance(track, Mapping):
                continue
            try:
                tracker_id = int(track.get("tracker_id", track.get("track_id")))
            except (TypeError, ValueError):
                continue
            if tracker_id >= 0:
                indexed[tracker_id] = track if isinstance(track, dict) else dict(track)
        source = int(sensor_id)
        with self._active_tracks_lock:
            self._active_tracks[source] = tracks
            ring = self._track_snapshot_ring.setdefault(source, OrderedDict())
            ring[int(frame_id)] = indexed
            ring.move_to_end(int(frame_id))
            while len(ring) > int(self._track_snapshot_ring_size):
                ring.popitem(last=False)

    def _publish_tracking_work(self, work: _TrackingPublicationWork) -> None:
        """Publish one exact tracking/BEV cohort outside the media callback."""

        publish_start_ns = time.perf_counter_ns()
        continuity = work.continuity
        # Compatibility for direct/unit callers that construct a work packet
        # without the media-side continuity receipt.  Native DS9 always carries
        # the receipt created before _active_tracks is published.
        if continuity is None:
            continuity = self._tracking_lifecycle.update_frame(
                source_id=int(work.source_id),
                camera_id=str(work.camera_id),
                frame_id=int(work.frame_id),
                observed_at_us=int(work.observed_at_us),
                tracks=work.tracks,
            )
        if (
            continuity.source_id != int(work.source_id)
            or continuity.frame_id != int(work.frame_id)
            or continuity.observed_at_us != int(work.observed_at_us)
        ):
            raise RuntimeError("tracking continuity receipt does not match work cohort")
        self._stamp_footpoint_lifecycles(work.footpoints, work.tracks)
        lease = self.publication_gate.acquire()
        if lease is None:
            raise RuntimeError(
                "runtime publication gate rejected an admitted tracking cohort"
            )
        with lease:
            # Media-side continuity rows can be several frames ahead of this
            # bounded worker. Bind tombstones against the last successfully
            # published lifecycle immediately before the tracking publish;
            # never publish the media-time preview carried by the cohort.
            continuity = self._tracking_lifecycle.prepare_publication(continuity)
            tracking_receipt = _require_tracking_publication_receipt(
                self.tracking_pub.publish(
                    int(work.source_id),
                    work.tracks,
                    frame_metadata={
                        "frame_id": int(work.frame_id),
                        "source_epoch": int(work.source_epoch),
                        "tracker_lifecycle_tombstones": list(continuity.tombstones),
                        "observed_at_us": int(work.observed_at_us),
                        **dict(work.temporal_contract),
                    },
                ),
                source_id=int(work.source_id),
                frame_id=int(work.frame_id),
                observed_at_us=int(work.observed_at_us),
            )
            # Tracking/world authority is committed immediately after its
            # typed admission receipt, matching the existing publisher
            # contract.  BEV is paired to that exact receipt.  A failed BEV
            # render is terminal for this canonical worker rather than being
            # hidden while later tracking rows continue without their paired
            # spatial publication.
            current_epoch = self._tracking_lifecycle.source_epoch(int(work.source_id))
            if int(continuity.source_epoch) > int(current_epoch):
                raise RuntimeError(
                    "tracking continuity receipt belongs to a future source epoch"
                )
            if int(continuity.source_epoch) == int(current_epoch):
                self._tracking_lifecycle.mark_published(continuity)
            else:
                # A source rewind can be observed by the media callback while
                # an older exact cohort is still draining FIFO.  Its tracking
                # and BEV rows remain ordered and are published before the new
                # epoch, but its lifecycle commit was intentionally superseded
                # by reset_source and must not mutate the new epoch's state.
                _increment_core_counter("tracking.superseded_epoch_commit_total")
            self._last_tracking_publish_ts_by_sensor[int(work.source_id)] = float(
                work.now_ts
            )
            self._last_tracking_count_by_sensor[int(work.source_id)] = len(work.tracks)
            _record_core_stage_timing(
                "tracking.publish",
                publish_start_ns,
                item_count=len(work.tracks),
            )

            # A processor used for tracking-only/unit paths has no paired BEV
            # authority.  Do not manufacture a BEV failure in that mode; the
            # terminal paired-failure rule applies only when a renderer is
            # actually configured.
            if self.bev_renderer is None:
                return

            bev_receipt = self._publish_bev(
                int(work.source_id),
                str(work.camera_id),
                _ScalarPublicationFrameMeta(
                    frame_number=int(work.frame_id),
                    frame_num=int(work.frame_id),
                    buf_pts=int(work.timestamp_us) * 1000,
                    source_id=int(work.source_id),
                ),
                work.footpoints,
                now_ts=float(work.now_ts),
                track_count=len(work.tracks),
                paired_with_tracking=True,
                tracking_receipt=tracking_receipt,
                observed_at_us=int(work.observed_at_us),
                timestamp_us=int(work.timestamp_us),
                tracker_lifecycle_tombstones=continuity.tombstones,
            )
            if bev_receipt.status == "failed":
                raise bev_receipt.failure or RuntimeError(
                    "BEV publication failed without a cause"
                )
            if bev_receipt.status == "startup_pending":
                logger.debug(
                    "BEV authority startup pending for sensor %s frame %s",
                    work.source_id,
                    work.frame_id,
                )

    def _enqueue_tracking_publication(
        self,
        *,
        sensor_id: int,
        camera_id: str,
        frame_meta: Any,
        tracks: Sequence[Mapping[str, Any]],
        footpoints: Sequence[Footpoint],
        now_ts: float,
        temporal_contract: Mapping[str, Any],
        continuity: TrackingContinuityUpdate,
        force: bool,
        identity_v2_primitives: Optional[Sequence[IdentityFramePrimitive]] = None,
    ) -> bool:
        frame_id = int(
            _meta_lookup(frame_meta, "frame_number", "frame_num", default=0) or 0
        )
        timestamp_us = int(self._frame_timestamp_us(frame_meta))
        observed_at_us = int(
            temporal_contract.get("observed_at_us")
            or max(1, int(float(now_ts) * 1_000_000))
        )
        count = len(tracks)
        interval = self._tracking_interval_for_frame(count)
        if not self._publish_gate_due(
            self._last_tracking_enqueue_ts_by_sensor,
            self._last_tracking_enqueue_count_by_sensor,
            sensor_id=int(sensor_id),
            now_ts=float(now_ts),
            count=count,
            interval_s=interval,
            counter_prefix="tracking",
            update=False,
            force=bool(force),
        ):
            return False

        # Keep the conversion explicit and typed for the canonical worker
        # packet. The gate runs first so rate-suppressed frames do not pay a
        # recursive public-contract copy. No borrowed DeepStream metadata is
        # retained beyond this callback.
        scalar_copy_start_ns = time.perf_counter_ns()
        copied_tracks: List[Dict[str, Any]] = []
        copied_track_by_original_id: Dict[int, Dict[str, Any]] = {}
        for track in tracks:
            # The rich resolver tree is a request-gated BEV presentation
            # diagnostic, never part of the canonical tracking publication.
            # Filter it before the recursive scalar copy so normal cohorts do
            # not pay to duplicate candidates, covariance matrices and PCF
            # evidence only to discard them later.
            public_track = {
                key: value
                for key, value in track.items()
                if key != "world_resolver"
            }
            copied = _copy_public_scalar(public_track)
            if isinstance(copied, Mapping):
                copied_track = dict(copied)
                copied_tracks.append(copied_track)
                copied_track_by_original_id[id(track)] = copied_track
        _record_core_stage_timing(
            "analytics.public_scalar_copy",
            scalar_copy_start_ns,
            item_count=len(copied_tracks),
        )

        work = _TrackingPublicationWork(
            source_id=int(sensor_id),
            camera_id=str(camera_id),
            frame_id=frame_id,
            observed_at_us=observed_at_us,
            now_ts=float(now_ts),
            timestamp_us=timestamp_us,
            tracks=copied_tracks,
            footpoints=list(footpoints),
            temporal_contract=dict(_copy_public_scalar(dict(temporal_contract)) or {}),
            continuity=continuity,
            source_epoch=int(continuity.source_epoch),
        )
        canonical_backlogged_before_enqueue = (
            self._canonical_publication_backlogged()
        )
        # Tests and synthetic processors without a native pipeline retain the
        # old deterministic synchronous behavior.  Native DS9 always has a
        # pipeline object and uses the non-blocking worker.
        if getattr(self.pipeline, "ds_pipeline", None) is None:
            self._publish_tracking_work(work)
        else:
            worker = self._ensure_tracking_publication_worker()
            if not worker.enqueue(work):
                error = worker.terminal_failure or RuntimeError(
                    "tracking publication worker is unavailable"
                )
                logger.error("Tracking publication enqueue failed: %s", error)
                return False
        self._last_tracking_enqueue_ts_by_sensor[int(sensor_id)] = float(now_ts)
        self._last_tracking_enqueue_count_by_sensor[int(sensor_id)] = count

        # Shadow identity is optional comparison/evidence work. Give it an
        # independent deep scalar snapshot only after the canonical cohort is
        # admitted. Its latest-per-source worker may coalesce stale work and
        # may degrade independently; neither copy nor scoring failure can
        # reject, delay, or mutate the exact tracking/world/BEV rows above.
        if identity_v2_primitives is not None:
            existing_shadow_worker = self._shadow_identity_worker
            if (
                existing_shadow_worker is not None
                and existing_shadow_worker.terminal_failure is not None
            ):
                # A failed optional lane stays disabled. Avoid repeatedly
                # copying embeddings on every later canonical frame merely to
                # discover the same terminal state at enqueue time.
                _increment_core_counter(
                    "identity_v2.shadow.dropped_unavailable_total"
                )
                return True
            # Shadow scoring is optional and can hold the GIL while it updates
            # its private gallery. Never add that work while canonical cohorts
            # are queued, and cap admitted comparison frames independently for
            # every source. The canonical cohort above remains admitted either
            # way; these counters describe diagnostic freshness only.
            if canonical_backlogged_before_enqueue:
                _increment_core_counter(
                    "identity_v2.shadow.dropped_canonical_backlog_total"
                )
                return True
            shadow_interval_s = float(
                getattr(self, "_shadow_identity_publish_interval_s", 0.5)
            )
            last_shadow_ts = self._last_shadow_identity_enqueue_ts_by_sensor.get(
                int(sensor_id)
            )
            if (
                shadow_interval_s > 0.0
                and last_shadow_ts is not None
                and float(now_ts) - float(last_shadow_ts) < shadow_interval_s
            ):
                _increment_core_counter("identity_v2.shadow.rate_limited_total")
                return True
            primitive_copy_start_ns = time.perf_counter_ns()
            try:
                service = getattr(self.pipeline, "identity_v2_service", None)
                if service is None or bool(
                    getattr(service, "authoritative", False)
                ):
                    raise RuntimeError(
                        "shadow identity-v2 primitives have no matching shadow service"
                    )
                shadow_track_by_original_id: Dict[int, Dict[str, Any]] = {}
                for track_key, copied_track in copied_track_by_original_id.items():
                    # The service needs only the scalar binding identity; the
                    # primitive already carries bbox/world/quality inputs. Do
                    # not duplicate the complete canonical row (pose, depth,
                    # analytics, etc.) merely to provide a private mutation
                    # target for shadow diagnostics.
                    shadow_binding: Dict[str, Any] = {}
                    for field_name in (
                        "camera_id",
                        "tracker_id",
                        "track_id",
                        "frame_id",
                        "source_epoch",
                    ):
                        if field_name in copied_track:
                            shadow_binding[field_name] = copied_track[field_name]
                    shadow_track_by_original_id[int(track_key)] = shadow_binding
                shadow_primitives = _copy_shadow_identity_primitives(
                    primitives=identity_v2_primitives,
                    copied_track_by_original_id=shadow_track_by_original_id,
                    embedding_dim=int(
                        getattr(service, "embedding_dim", self._reid_embedding_dim)
                    ),
                    source_epoch=int(continuity.source_epoch),
                )
                sequence = int(self._shadow_identity_admission_sequence)
                self._shadow_identity_admission_sequence = sequence + 1
                shadow_admitted = self._ensure_shadow_identity_worker().enqueue(
                    _ShadowIdentityWork(
                        source_id=int(sensor_id),
                        camera_id=str(camera_id),
                        frame_id=int(frame_id),
                        observed_at=float(now_ts),
                        primitives=shadow_primitives,
                        admission_sequence=sequence,
                    )
                )
                if shadow_admitted:
                    self._last_shadow_identity_enqueue_ts_by_sensor[
                        int(sensor_id)
                    ] = float(now_ts)
            except Exception as exc:
                _increment_core_counter("identity_v2.shadow.copy_failures_total")
                callback = getattr(
                    self.pipeline,
                    "identity_v2_shadow_failure_callback",
                    None,
                )
                if callable(callback):
                    try:
                        callback(exc)
                    except Exception:
                        logger.exception(
                            "Shadow identity copy degradation callback failed"
                        )
                logger.exception(
                    "Optional shadow identity work was rejected; canonical cohort "
                    "source=%s frame=%s remains admitted",
                    sensor_id,
                    frame_id,
                )
            finally:
                _record_core_stage_timing(
                    "identity_v2.shadow_primitive_copy",
                    primitive_copy_start_ns,
                    item_count=len(identity_v2_primitives),
                )
        return True

    def shutdown(self, *, wait: bool = True, timeout_s: float = 5.0) -> None:
        failures: List[BaseException] = []
        shadow_worker = self._shadow_identity_worker
        if shadow_worker is not None:
            try:
                shadow_worker.shutdown(wait=wait, timeout_s=timeout_s)
            except BaseException as exc:
                failures.append(exc)
        worker = self._tracking_publication_worker
        if worker is not None:
            try:
                worker.shutdown(wait=wait, timeout_s=timeout_s)
            except BaseException as exc:
                failures.append(exc)
        if failures:
            raise RuntimeError(
                "analytics telemetry workers did not quiesce: "
                + "; ".join(str(error) for error in failures)
            ) from failures[0]

    def get_active_track_map(
        self,
        sensor_id: int,
        frame_id: Optional[int] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """Return active rows or an exact frame from the bounded scalar ring.

        Supplying ``frame_id`` is an exact join: an absent frame returns an
        empty map and never falls back to the latest source state.  That
        distinction prevents a tiler callback for frame N from drawing the
        canonical anchor for frame N+1.
        """

        source = int(sensor_id)
        with self._active_tracks_lock:
            if frame_id is None:
                tracks = list(self._active_tracks.get(source) or [])
                indexed: Dict[int, Dict[str, Any]] = {}
                for track in tracks:
                    if not isinstance(track, Mapping):
                        continue
                    tracker_id = track.get("tracker_id", track.get("track_id"))
                    try:
                        tracker_id_int = int(tracker_id)
                    except Exception:
                        continue
                    if tracker_id_int < 0:
                        continue
                    indexed[int(tracker_id_int)] = dict(track)
                return indexed
            ring = self._track_snapshot_ring.get(source)
            snapshot = ring.get(int(frame_id)) if ring is not None else None
            if snapshot is None:
                return {}
            return {
                int(tracker_id): dict(track)
                for tracker_id, track in snapshot.items()
                if isinstance(track, Mapping)
            }

    def _maybe_log_stable_id_metrics(self, sensor_id: int, now_ts: float) -> None:
        if not self._sid_metrics_log_enabled or not self._stable_id_enabled:
            return
        sid_int = int(sensor_id)
        last_ts = float(self._sid_metrics_last_log_by_sensor.get(sid_int, 0.0))
        interval = float(self._sid_metrics_log_interval_s)
        if (float(now_ts) - last_ts) < interval:
            return
        mgr = getattr(self.pipeline, "stable_id_mgr", None)
        get_metrics = getattr(mgr, "get_sid_metrics", None)
        if not callable(get_metrics):
            return
        try:
            metrics = dict(get_metrics() or {})
        except Exception:
            logger.debug("StableID metrics fetch failed", exc_info=True)
            return
        self._sid_metrics_last_log_by_sensor[sid_int] = float(now_ts)
        logger.info(
            "StableID metrics sensor=%d canonical=%s active=%s next=%s merge_applied=%s merge_suppressed=%s candidates=%s blocked=%s support_gate=%s stall_streak=%s stall_reason=%s recycled=%s pool=%s sim_p50=%s sim_p95=%s new_alloc=%s remap=%s guard_reject=%s no_emb=%s fragmentation=%s pending_recycled=%s same_frame_conflicts=%s",
            sid_int,
            metrics.get("canonical_gallery_size"),
            metrics.get("active_unique"),
            metrics.get("next_sid"),
            metrics.get("auto_merge_applied"),
            metrics.get("auto_merge_suppressed"),
            metrics.get("auto_merge_last_candidate_count"),
            metrics.get("auto_merge_last_blocked_count"),
            metrics.get("auto_merge_last_support_gate"),
            metrics.get("auto_merge_zero_apply_streak"),
            metrics.get("auto_merge_last_reason"),
            metrics.get("auto_merge_last_pressure_recycled"),
            metrics.get("alias_candidate_pool_size"),
            metrics.get("alias_candidate_sim_p50"),
            metrics.get("alias_candidate_sim_p95"),
            metrics.get("sid_new_alloc_count"),
            metrics.get("sid_remap_count"),
            metrics.get("sid_guard_reject_count"),
            metrics.get("sid_no_embedding_count"),
            metrics.get("sid_fragmentation_events"),
            metrics.get("sid_pending_recycled_count"),
            metrics.get("sid_same_frame_conflict_count"),
        )

    def _tracking_mode_is_v3dt(self) -> bool:
        return str(self._tracking_mode or "").strip().lower() in {"v3dt", "mv3dt"}

    def _tracking_mode_is_mv3dt(self) -> bool:
        return str(self._tracking_mode or "").strip().lower() == "mv3dt"

    def _world_calibration_snapshot(
        self,
        sensor_id: int,
        camera_id: str,
    ) -> Any:
        """Return the calibration view owned by the active world producer.

        Every world admission check uses the same revision-bound target-frame
        view, including seeded V3DT/bbox3d observations.  Using the raw
        camera-calibration frame for only the seeded path makes a non-identity
        scene-prior transform appear as a range or frame mismatch later in the
        pipeline.
        """

        provider = self.bev_calibration
        if provider is None:
            return None
        world_snapshot = getattr(provider, "world_snapshot", None)
        if callable(world_snapshot):
            return world_snapshot(sensor_id, camera_id)
        return provider.snapshot(sensor_id, camera_id)

    def _resolve_tracking_mode(self, override: Optional[str] = None) -> str:
        if override is not None and str(override).strip():
            return self._normalize_tracking_mode(override)
        env_mode = str(os.environ.get("NOESIS_TRACKING_MODE", "") or "").strip()
        if env_mode:
            return self._normalize_tracking_mode(env_mode)
        try:
            cfg = getattr(self.pipeline, "config", {}) or {}
        except Exception:
            cfg = {}
        if isinstance(cfg, Mapping):
            raw_mode = cfg.get("tracking_mode")
            if raw_mode:
                return self._normalize_tracking_mode(raw_mode)
            v3dt_cfg = cfg.get("v3dt")
            if isinstance(v3dt_cfg, Mapping):
                raw_mode = (
                    v3dt_cfg.get("tracking_mode")
                    or v3dt_cfg.get("mode")
                    or v3dt_cfg.get("profile")
                )
                if raw_mode:
                    return self._normalize_tracking_mode(raw_mode)
            tracker_cfg = cfg.get("tracker")
            if isinstance(tracker_cfg, Mapping):
                cfg_path = tracker_cfg.get("config-file")
                if cfg_path and "config/v3dt/" in str(cfg_path):
                    return "v3dt"
        return "baseline"

    @staticmethod
    def _normalize_tracking_mode(value: Any) -> str:
        mode = str(value or "").strip().lower()
        if mode == "mv3dt":
            return "mv3dt"
        if mode in ("v3dt", "sv3dt", "3d"):
            return "v3dt"
        if mode in ("2d", "baseline", "standard", "default"):
            return "baseline"
        if not mode or mode == "auto":
            return "baseline"
        raise ValueError(
            "Unsupported DS9 tracking mode "
            f"{value!r}; expected baseline, v3dt, mv3dt, or auto"
        )

    def _warn_on_tracking_mode_mismatch(self) -> None:
        try:
            tracker_cfg = getattr(self.pipeline, "config", {}).get("tracker", {}) or {}
        except Exception:
            tracker_cfg = {}
        cfg_path = tracker_cfg.get("config-file") if isinstance(tracker_cfg, Mapping) else None
        if not cfg_path:
            return
        cfg_path = str(cfg_path)
        using_v3dt_tracker = "config/v3dt/" in cfg_path
        using_mv3dt_tracker = "nvtracker_mv3dt" in cfg_path
        if using_v3dt_tracker and not self._tracking_mode_is_v3dt():
            logger.warning(
                "Tracking mode '%s' with V3DT tracker config %s; V3DT meta/world will be ignored",
                self._tracking_mode,
                cfg_path,
            )
        if self._tracking_mode_is_mv3dt() and not using_mv3dt_tracker:
            logger.warning(
                "Tracking mode 'mv3dt' without the MV3DT tracker config (%s); "
                "multi-view association metadata cannot be present",
                cfg_path,
            )
        elif self._tracking_mode_is_v3dt() and not using_v3dt_tracker:
            logger.warning(
                "Tracking mode '%s' without V3DT tracker config (%s); V3DT meta may be absent",
                self._tracking_mode,
                cfg_path,
            )
        if not self._tracking_mode_is_mv3dt() and using_mv3dt_tracker:
            raise ValueError(
                f"Tracking mode {self._tracking_mode!r} cannot activate the "
                f"deferred MV3DT tracker config {cfg_path}"
            )

    def _ensure_v3dt_caminfo_paths(self) -> None:
        if not self._tracking_mode_is_v3dt():
            return
        if self._v3dt_caminfo_paths:
            return
        try:
            tracker_cfg = getattr(self.pipeline, "config", {}).get("tracker", {}) or {}
        except Exception:
            tracker_cfg = {}
        cfg_path = tracker_cfg.get("config-file") if isinstance(tracker_cfg, Mapping) else None
        if not cfg_path:
            return
        repo_root = Path(__file__).resolve().parents[2]
        cfg_path = Path(cfg_path)
        if not cfg_path.is_absolute():
            cfg_path = repo_root / cfg_path
        if not cfg_path.exists():
            if not self._v3dt_caminfo_logged_missing:
                logger.warning("V3DT camInfo config missing: %s", cfg_path)
                self._v3dt_caminfo_logged_missing = True
            return
        try:
            tracker_data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except Exception:
            return
        projection_cfg = tracker_data.get("ObjectModelProjection") if isinstance(tracker_data, Mapping) else None
        cam_paths = projection_cfg.get("cameraModelFilepath") if isinstance(projection_cfg, Mapping) else None
        if not isinstance(cam_paths, list):
            return
        for idx, item in enumerate(cam_paths):
            if not isinstance(item, str) or not item:
                continue
            path = Path(item)
            if not path.is_absolute():
                path = repo_root / path
            self._v3dt_caminfo_paths[int(idx)] = path

    def _v3dt_projection_for_source(self, source_id: int) -> Optional[Tuple[str, List[List[float]]]]:
        if not self._tracking_mode_is_v3dt():
            return None
        cached = self._v3dt_caminfo_cache.get(source_id)
        if cached is not None:
            return cached
        self._ensure_v3dt_caminfo_paths()
        caminfo_path = self._v3dt_caminfo_paths.get(int(source_id))
        if caminfo_path is None or not caminfo_path.exists():
            return None
        try:
            caminfo = yaml.safe_load(caminfo_path.read_text(encoding="utf-8")) or {}
        except Exception:
            return None
        key = "projectionMatrix_3x4_w2p"
        data = caminfo.get(key)
        if not isinstance(data, list) or len(data) != 12:
            key = "projectionMatrix_3x4"
            data = caminfo.get(key)
        if not isinstance(data, list) or len(data) != 12:
            return None
        P = [data[0:4], data[4:8], data[8:12]]
        cached = (key, P)
        self._v3dt_caminfo_cache[int(source_id)] = cached
        return cached

    @staticmethod
    def _project_point(P: Sequence[Sequence[float]], xyz: Sequence[float]) -> Optional[Tuple[float, float]]:
        try:
            x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        except Exception:
            return None
        try:
            w = (
                float(P[2][0]) * x
                + float(P[2][1]) * y
                + float(P[2][2]) * z
                + float(P[2][3])
            )
            if abs(w) < 1e-9:
                return None
            u = (
                float(P[0][0]) * x
                + float(P[0][1]) * y
                + float(P[0][2]) * z
                + float(P[0][3])
            ) / w
            v = (
                float(P[1][0]) * x
                + float(P[1][1]) * y
                + float(P[1][2]) * z
                + float(P[1][3])
            ) / w
            if not (math.isfinite(u) and math.isfinite(v)):
                return None
            return float(u), float(v)
        except Exception:
            return None

    def _image_base_from_bbox3d(
        self,
        source_id: int,
        bbox3d: Mapping[str, Any],
        frame_dims: Tuple[int, int],
    ) -> Optional[List[float]]:
        proj = self._v3dt_projection_for_source(source_id)
        if proj is None:
            return None
        key, P = proj
        try:
            x = float(bbox3d.get("xCentre"))
            y = float(bbox3d.get("yCentre"))
            z = float(bbox3d.get("zCentre"))
            z_len = float(bbox3d.get("zLen"))
        except Exception:
            return None
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z) and math.isfinite(z_len)):
            return None
        # The xzy camInfo lane places the tracker ground endpoint at
        # zCentre-0.5*zLen. NVIDIA's image-foot meta follows that model point,
        # while the opposite vertical endpoint aligns with the visible detector
        # base in the validated three-camera lane. Publish that projected point
        # as image_base; canonical world still uses the ground endpoint.
        z_image_base = z + 0.5 * z_len
        uv = self._project_point(P, (x, y, z_image_base))
        if uv is None:
            return None
        u, v = uv
        if key == "projectionMatrix_3x4":
            frame_w, frame_h = frame_dims
            if frame_w and frame_h:
                u += float(frame_w) * 0.5
                v += float(frame_h) * 0.5
        return [float(u), float(v)]

    def _log_diag_session_start(self) -> None:
        if not self.diagnostics_logger or self._diag_logged:
            return
        self._diag_logged = True
        try:
            if build_v3dt_session_start_payload is None:
                raise RuntimeError("V3DT diagnostic context builder is unavailable")
            payload = build_v3dt_session_start_payload(
                pipeline_config=getattr(self.pipeline, "config", {}),
                camera_labels=self.camera_labels or {},
                sensor_id_map=self.sensor_id_map or {},
            )
            self.diagnostics_logger.log_event(payload)
        except Exception:
            logger.debug("Failed to log V3DT diagnostics session start", exc_info=True)

    def _extract_v3dt_meta_ds8(self, obj_meta: Any) -> Optional[Dict[str, Any]]:
        if not self._v3dt_meta_enabled or not self._tracking_mode_is_v3dt():
            return None
        if noesis_v3dt_meta_ext is None:
            if not self._v3dt_meta_logged_missing:
                logger.warning("V3DT meta extraction disabled; noesis_v3dt_meta_ext is unavailable")
                self._v3dt_meta_logged_missing = True
            return None
        try:
            result = noesis_v3dt_meta_ext.extract_obj_3d_meta(obj_meta)
        except Exception:
            return None
        if result is None:
            return None
        if isinstance(result, dict):
            return result
        try:
            return dict(result)
        except Exception:
            return None

    def _world_from_bbox3d(self, bbox3d: Mapping[str, Any]) -> Optional[List[float]]:
        if self._v3dt_axis_map is None:
            raise V3DTAxisMapError("V3DT axis map was not initialized")
        try:
            world = v3dt_bbox3d_world_foot(bbox3d, self._v3dt_axis_map)
        except V3DTAxisMapError:
            return None
        return [float(world[0]), float(world[1]), float(world[2])]

    def _extract_reid_embedding_native(self, obj_meta: Any) -> Optional[np.ndarray]:
        if noesis_reid_meta_ext is None:
            return None
        extract_obj = getattr(noesis_reid_meta_ext, "extract_reid_embedding", None)
        if extract_obj is None or not callable(extract_obj):
            return None
        try:
            payload = extract_obj(
                obj_meta,
                int(self._reid_unique_id),
                str(self._reid_layer_name),
                int(self._reid_embedding_dim),
                True,
            )
        except Exception:
            return None
        if payload is None:
            return None
        try:
            emb = np.asarray(payload, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if emb.size < 1:
            return None
        if not np.all(np.isfinite(emb)):
            return None
        n = float(np.linalg.norm(emb) + 1e-12)
        if not math.isfinite(n) or n <= 0.0:
            return None
        _increment_core_counter("tensor_host_copies_total.reid")
        return (emb / n).astype(np.float32, copy=False)

    def _extract_reid_embedding_ds8(self, obj_meta: Any) -> Optional[np.ndarray]:
        """Extract the configured ReID embedding from object tensor metadata."""
        emb_native = self._extract_reid_embedding_native(obj_meta)
        if emb_native is not None:
            return emb_native
        global _REID_NATIVE_MISSING_LOGGED
        if not _REID_NATIVE_MISSING_LOGGED and noesis_reid_meta_ext is None:
            logger.warning("ReID native extraction unavailable; build scripts/build_noesis_reid_meta_ext.sh")
            _REID_NATIVE_MISSING_LOGGED = True
        return None

    def handle_frame_ds8(self, frame_meta: Any) -> None:
        lease = self.publication_gate.acquire()
        if lease is None:
            _increment_core_counter("runtime_publication_gate_rejected_frames_total")
            return
        with lease:
            self._handle_frame_ds8_admitted(frame_meta)

    def _handle_frame_ds8_admitted(self, frame_meta: Any) -> None:
        """Extract tracking telemetry for a single frame using DS8 pyservicemaker API."""
        frame_start_ns = time.perf_counter_ns()
        try:
            source_id = self._frame_source_id(frame_meta)
            sensor_id = self.sensor_id_map.get(source_id, source_id)
            camera_id = self.camera_labels.get(sensor_id, f"camera_{sensor_id}")
            now_ts = time.time()
            temporal_contract = _frame_temporal_contract(frame_meta, observed_at_s=now_ts)
            identity_v2_service = getattr(self.pipeline, "identity_v2_service", None)
            identity_v2_authoritative = bool(
                identity_v2_service is not None
                and getattr(identity_v2_service, "authoritative", False)
            )
            identity_v2_shadow = bool(
                identity_v2_service is not None and not identity_v2_authoritative
            )
            self._pose_anchor_native_remaining = int(self._pose_anchor_native_per_frame_max)
            reid_budget_remaining = int(self._reid_embeds_per_frame_max)
            _increment_core_counter("detection_wake.frames")
            self._log_diag_session_start()

            tracks: List[Dict[str, Any]] = []
            diagnostics_tracks: List[Dict[str, Any]] = []
            frame_id = int(getattr(frame_meta, "frame_number", -1))
            source_epoch = self._begin_source_frame_timeline(
                sensor_id=int(sensor_id),
                frame_id=int(frame_id),
                observed_at_us=int(temporal_contract["observed_at_us"]),
                media_pts_ns=int(temporal_contract.get("media_pts_ns", 0) or 0),
            )
            world_now_ts = self._world_timestamp_for_frame(
                int(sensor_id),
                media_pts_ns=temporal_contract.get("media_pts_ns", 0),
                observed_ts=float(now_ts),
            )
            occupancy_counts: Dict[str, int] = {}
            present_track_ids: set[int] = set()
            present_stable_ids: set[int] = set()
            footpoints: List[Footpoint] = []
            identity_v2_primitives: List[IdentityFramePrimitive] = []
            frame_dims = self._track_image_size(sensor_id, frame_meta)

            reid_debug = str(os.environ.get("NOESIS_REID_DEBUG", "")).strip().lower() in ("1", "true", "yes", "on")
            if reid_debug:
                self._reid_debug_frames += 1

            object_items = getattr(frame_meta, "object_items", None) or []
            for obj_meta in object_items:
                _increment_core_counter("detection_wake.objects_seen")
                track_build_start_ns = time.perf_counter_ns()
                raw = self._build_track_dict_ds8(obj_meta, camera_id)
                _record_core_stage_timing(
                    "analytics.track_build",
                    track_build_start_ns,
                )
                if raw is None:
                    continue
                if self._tracking_mode_is_v3dt():
                    bbox3d = raw.get("bbox3d")
                    if isinstance(bbox3d, dict):
                        image_base = self._image_base_from_bbox3d(int(source_id), bbox3d, frame_dims)
                        if image_base is not None:
                            raw["image_base"] = image_base
                diag_track = dict(raw)
                diag_track["frame_id"] = frame_id
                diag_track["source_id"] = int(source_id)
                if reid_debug:
                    self._reid_debug_objects += 1

                track_id = int(raw.get("track_id", -1))
                if track_id < 0:
                    continue
                present_track_ids.add(track_id)

                try:
                    class_id = int(raw.get("class_id", -1))
                except Exception:
                    class_id = -1

                zone = raw.get("zone")
                zone_source = raw.get("zone_source")
                zone_authoritative = raw.get("zone_authoritative") is True

                # People-only public identity. Do not show raw tracker IDs.
                if class_id != 0:
                    self._stamp_osd_label_ds8(obj_meta, sensor_id=sensor_id, stable_id=None)
                    diagnostics_tracks.append(diag_track)
                    continue

                _increment_core_counter("detection_wake.person_tracks")
                if not zone:
                    zone = _fallback_zone_from_camera(camera_id)
                    zone_source = "camera_default" if zone else None
                    zone_authoritative = False

                if reid_debug:
                    self._reid_debug_people += 1

                emb = None
                mgr = getattr(self.pipeline, "stable_id_mgr", None)
                manager_sensor_id = self._stable_id_manager_sensor_id(sensor_id)
                identity_v2_enabled = identity_v2_service is not None
                if identity_v2_enabled or (self._stable_id_enabled and mgr is not None):
                    need_emb = True
                    if (
                        self._stable_id_enabled
                        and mgr is not None
                        and not identity_v2_authoritative
                    ):
                        needs_fn = getattr(mgr, "needs_embedding", None)
                        if callable(needs_fn):
                            try:
                                need_emb = bool(
                                    needs_fn(
                                        manager_sensor_id,
                                        int(track_id),
                                        float(now_ts),
                                    )
                                )
                            except Exception:
                                need_emb = True
                        else:
                            try:
                                rec = mgr.active_tracks.get(
                                    (manager_sensor_id, int(track_id))
                                )
                                need_emb = rec is None or rec.get("emb") is None
                            except Exception:
                                need_emb = True

                    if need_emb or identity_v2_enabled:
                        _increment_core_counter("detection_wake.reid_emb_due")
                        if reid_budget_remaining <= 0 and not identity_v2_enabled:
                            _increment_core_counter("detection_wake.reid_emb_budget_skipped")
                        else:
                            reid_budget_remaining = max(0, reid_budget_remaining - 1)
                            reid_start_ns = time.perf_counter_ns()
                            emb = self._extract_reid_embedding_ds8(obj_meta)
                            _record_core_stage_timing("reid.extract_embedding", reid_start_ns)
                            if emb is None:
                                _increment_core_counter("detection_wake.reid_emb_missing")
                            else:
                                _increment_core_counter("detection_wake.reid_emb_extracted")
                        if reid_debug:
                            if emb is None:
                                self._reid_debug_emb_missing += 1
                            else:
                                self._reid_debug_emb_found += 1

                stable_id = None
                if not identity_v2_authoritative:
                    sid_start_ns = time.perf_counter_ns()
                    stable_id = self._maybe_assign_stable_id(
                        sensor_id=sensor_id,
                        track_id=track_id,
                        bbox=raw.get("bbox"),
                        zone=zone,
                        ts=now_ts,
                        frame_bgr=None,
                        embedding=emb,
                    )
                    _record_core_stage_timing("stable_id.update_track", sid_start_ns)
                if stable_id is None:
                    if not identity_v2_authoritative and not identity_v2_shadow:
                        self._stamp_osd_label_ds8(
                            obj_meta, sensor_id=sensor_id, stable_id=None
                        )
                        diagnostics_tracks.append(diag_track)
                        continue

                stable_id_int = int(stable_id) if stable_id is not None else 0
                if stable_id_int > 0:
                    present_stable_ids.add(stable_id_int)
                tracker_id_int = int(track_id)
                id_diag: Dict[str, Any] = {}
                mgr = getattr(self.pipeline, "stable_id_mgr", None)
                get_id_diag = (
                    getattr(mgr, "get_track_diagnostics", None)
                    if not identity_v2_authoritative
                    else None
                )
                if callable(get_id_diag):
                    try:
                        id_diag = dict(
                            get_id_diag(manager_sensor_id, int(track_id)) or {}
                        )
                    except Exception:
                        id_diag = {}
                identity_contract = _stable_identity_contract(
                    None if identity_v2_authoritative else mgr,
                    sensor_id=manager_sensor_id,
                    tracker_id=tracker_id_int,
                    diagnostics={} if identity_v2_authoritative else id_diag,
                )
                id_event = id_diag.get("id_event")
                id_reject_reason = id_diag.get("id_reject_reason")
                sid_candidate = id_diag.get("sid_candidate")
                embedding_present = bool(id_diag.get("embedding_present", emb is not None))
                pose_anchor_start_ns = time.perf_counter_ns()
                pose_kpts_abs = self._extract_pose_keypoints_for_anchor(obj_meta, raw.get("bbox") or [])
                _record_core_stage_timing("pose_anchor.extract_keypoints", pose_anchor_start_ns)
                depth_extract_start_ns = time.perf_counter_ns()
                depth_result = self._extract_object_depth_result(obj_meta)
                _record_core_stage_timing("object_depth.extract_meta", depth_extract_start_ns)
                pose_present = bool(id_diag.get("pose_present", False))
                if not pose_present:
                    pose_present = pose_kpts_abs is not None
                id_display = None
                if self._reid_diag_use_tracker_id and stable_id_int > 0:
                    id_display = f"[{tracker_id_int}] | [{stable_id_int}]"

                dwell = (
                    self._update_dwell_time(sensor_id, stable_id_int, zone, now_ts)
                    if stable_id_int > 0 and not identity_v2_authoritative
                    else None
                )

                analytics = raw.get("analytics")
                if analytics and "lcStatus" in analytics:
                    lc = analytics["lcStatus"]
                    if isinstance(lc, dict):
                        for line_name, status in lc.items():
                            if (
                                status == 1
                                and stable_id_int > 0
                                and not identity_v2_authoritative
                            ):
                                self._record_transition(
                                    sensor_id=sensor_id,
                                    stable_id=stable_id_int,
                                    line_name=line_name,
                                    ts=now_ts,
                                )

                if zone:
                    occupancy_counts[zone] = occupancy_counts.get(zone, 0) + 1

                public_track: Dict[str, Any] = {
                    "stable_id": stable_id_int if stable_id_int > 0 else None,
                    "tracker_id": tracker_id_int,
                    # Peek before world augmentation so a reused numeric
                    # tracker cannot inherit the prior lifecycle's filter.
                    "tracker_lifecycle_generation": self._tracking_lifecycle.peek_generation(
                        int(sensor_id),
                        int(tracker_id_int),
                        frame_id=int(frame_id),
                        observed_at_us=int(temporal_contract["observed_at_us"]),
                        bbox=raw.get("bbox"),
                    ),
                    "camera_id": camera_id,
                    "source_id": int(source_id),
                    "bbox": raw.get("bbox"),
                    "center": raw.get("center"),
                    "class_id": 0,
                    "confidence": raw.get("confidence"),
                    "tracker_confidence": raw.get("tracker_confidence"),
                    "analytics": raw.get("analytics"),
                    "zone": zone,
                    "zone_source": zone_source,
                    "zone_authoritative": zone_authoritative,
                    "frame_id": frame_id,
                    "dwell_time": dwell,
                    "id_event": id_event,
                    "id_reject_reason": id_reject_reason,
                    "embedding_present": bool(embedding_present),
                    "pose_present": bool(pose_present),
                    "sid_candidate": sid_candidate,
                    **temporal_contract,
                    **identity_contract,
                }
                frame_w, frame_h = frame_dims
                if frame_w > 8 and frame_h > 8:
                    public_track["image_size"] = [int(frame_w), int(frame_h)]
                if id_display:
                    public_track["id_display"] = str(id_display)
                for key in (
                    "bbox3d",
                    "velocity3d",
                    "visibility",
                    "image_foot",
                    "image_base",
                    "world",
                    "world_valid",
                    "world_quality",
                    "world_quality_reason",
                    "world_frame",
                    "world_frame_revision",
                    "world_transform_sha256",
                    "world_source",
                ):
                    if key in raw:
                        public_track[key] = raw.get(key)
                for key in _WORLD_ESTIMATOR_DIAGNOSTIC_FIELDS:
                    if key in raw:
                        public_track[key] = raw.get(key)

                world_augment_start_ns = time.perf_counter_ns()
                self._augment_track_with_world(
                    sensor_id,
                    camera_id,
                    public_track,
                    obj_meta=obj_meta,
                    pose_kpts_abs=pose_kpts_abs,
                    depth_result=depth_result,
                    world_now_ts=float(world_now_ts),
                )
                _record_core_stage_timing(
                    "analytics.world_augment",
                    world_augment_start_ns,
                )
                public_fields_start_ns = time.perf_counter_ns()
                self._apply_scene_prior_shadow(camera_id, public_track)
                self._apply_public_depth_fields(public_track, depth_result)
                _record_core_stage_timing(
                    "analytics.public_fields",
                    public_fields_start_ns,
                )
                try:
                    setattr(obj_meta, "_noesis_depth_used_m", public_track.get("depth_used_m"))
                except Exception:
                    pass
                # Stamp OSD after world/depth augmentation so ``depth=`` reflects
                # registered optical range without implying canonical world Z.
                osd_start_ns = time.perf_counter_ns()
                self._stamp_osd_label_ds8(
                    obj_meta,
                    sensor_id=sensor_id,
                    # Zero is an explicit neutral override.  None would ask the
                    # OSD processor to look up stale legacy StableID state.
                    stable_id=0 if identity_v2_authoritative else stable_id_int,
                )
                if not identity_v2_authoritative:
                    self._apply_instance_mask_color_ds8(obj_meta, stable_id=stable_id_int)
                _record_core_stage_timing(
                    "analytics.osd_object",
                    osd_start_ns,
                )
                diag_track.update(
                    {
                        "stable_id": stable_id_int,
                        "tracker_id": tracker_id_int,
                            **({"id_display": str(id_display)} if id_display else {}),
                            "zone": zone,
                            "zone_source": zone_source,
                            "zone_authoritative": zone_authoritative,
                            "dwell_time": dwell,
                        "world": public_track.get("world"),
                        "world_valid": public_track.get("world_valid"),
                        "world_quality": public_track.get("world_quality"),
                        "world_quality_reason": public_track.get("world_quality_reason"),
                        "world_frame": public_track.get("world_frame"),
                        "world_source": public_track.get("world_source"),
                        "depth_status": depth_result.status if depth_result is not None else None,
                        "depth_anchor_source": depth_result.anchor_source if depth_result is not None else None,
                        "depth_anchor_m": depth_result.anchor_depth_m if depth_result is not None else None,
                        "depth_registered_m": public_track.get("depth_registered_m"),
                        "depth_used_m": public_track.get("depth_used_m"),
                        "depth_registration_status": public_track.get("depth_registration_status"),
                        "depth_registration_id": public_track.get("depth_registration_id"),
                        "depth_samples": depth_result.sample_count if depth_result is not None else None,
                        "depth_valid_fraction": depth_result.valid_fraction if depth_result is not None else None,
                        "id_event": id_event,
                        "id_reject_reason": id_reject_reason,
                        "embedding_present": bool(embedding_present),
                        "pose_present": bool(pose_present),
                        "sid_candidate": sid_candidate,
                    }
                )
                tracks.append(public_track)
                identity_v2_primitives.append(
                    IdentityFramePrimitive(
                        camera_id=camera_id,
                        # Tracker IDs may be reused after a source reconnect.
                        # Scope the coordinator's tracklet key to the physical
                        # source timeline while preserving the raw diagnostic
                        # tracker_id in the public row.
                        tracker_id=_identity_v2_tracker_id_for_epoch(
                            tracker_id_int,
                            source_epoch,
                        ),
                        frame_id=frame_id,
                        public_track=public_track,
                        embedding=emb,
                        diagnostic_track=diag_track,
                        bbox=public_track.get("bbox"),
                        frame_size=frame_dims,
                        detection_confidence=public_track.get("confidence"),
                        tracker_confidence=public_track.get("tracker_confidence"),
                        world_xyz=public_track.get("world"),
                        world_valid=public_track.get("world_valid") is True,
                    )
                )
                diagnostics_tracks.append(diag_track)

                footpoint_start_ns = time.perf_counter_ns()
                fp = self._footpoint_from_track(
                    public_track,
                    frame_dims,
                    target_image_size=self._bev_target_image_size(
                        sensor_id,
                        camera_id,
                    ),
                )
                _record_core_stage_timing(
                    "analytics.footpoint",
                    footpoint_start_ns,
                )
                if fp is not None:
                    footpoints.append(fp)

            post_frame_start_ns = time.perf_counter_ns()
            if identity_v2_authoritative:
                identity_v2_start_ns = time.perf_counter_ns()
                try:
                    self._process_identity_v2_source_frame(
                        camera_id=camera_id,
                        frame_id=frame_id,
                        primitives=identity_v2_primitives,
                        observed_at=now_ts,
                    )
                finally:
                    _record_core_stage_timing(
                        "identity_v2.process_source_frame",
                        identity_v2_start_ns,
                        item_count=len(identity_v2_primitives),
                    )
            if identity_v2_authoritative:
                present_stable_ids = set()
                footpoints = []
                for track in tracks:
                    try:
                        resolved_sid = int(track.get("stable_id"))
                    except (TypeError, ValueError):
                        resolved_sid = 0
                    if resolved_sid > 0:
                        present_stable_ids.add(resolved_sid)
                        track["dwell_time"] = self._update_dwell_time(
                            sensor_id,
                            resolved_sid,
                            track.get("zone"),
                            now_ts,
                        )
                        analytics = track.get("analytics")
                        if isinstance(analytics, Mapping):
                            line_status = analytics.get("lcStatus")
                            if isinstance(line_status, Mapping):
                                for line_name, status in line_status.items():
                                    if status == 1:
                                        self._record_transition(
                                            sensor_id=sensor_id,
                                            stable_id=resolved_sid,
                                            line_name=str(line_name),
                                            ts=now_ts,
                                        )
                    else:
                        track["dwell_time"] = None
                    fp = self._footpoint_from_track(
                        track,
                        frame_dims,
                        target_image_size=self._bev_target_image_size(
                            sensor_id,
                            camera_id,
                        ),
                    )
                    if fp is not None:
                        footpoints.append(fp)
                for primitive in identity_v2_primitives:
                    if primitive.diagnostic_track is not None:
                        primitive.diagnostic_track["dwell_time"] = (
                            primitive.public_track.get("dwell_time")
                        )

            mgr = getattr(self.pipeline, "stable_id_mgr", None)
            observe_fn = getattr(mgr, "observe_copresence", None)
            if callable(observe_fn) and not identity_v2_authoritative:
                try:
                    observe_fn(sorted(present_stable_ids), float(now_ts))
                except Exception:
                    logger.debug("StableIDManager observe_copresence failed", exc_info=True)

            self._publish_occupancy(sensor_id, occupancy_counts)
            self._cleanup_zone_state(sensor_id, present_stable_ids)
            if not identity_v2_authoritative:
                self._maintain_stable_ids(sensor_id, present_track_ids, now_ts)
            if (
                not tracks
                and os.environ.get("NOESIS_REID_TEST_MODE") == "1"
                and not identity_v2_authoritative
            ):
                synthetic = {
                    "camera_id": camera_id,
                    "stable_id": 1,
                    "tracker_id": 1,
                    "bbox": (0.0, 0.0, 10.0, 10.0),
                    "image_size": [int(frame_dims[0]), int(frame_dims[1])],
                    "frame_id": frame_id,
                    "zone": None,
                    "class_id": 0,
                    "identity_kind": "provisional",
                    **temporal_contract,
                }
                tracks.append(synthetic)
                present_stable_ids.add(1)
                present_track_ids.add(1)
            self._clear_absent_world_state(
                sensor_id,
                (
                    track.get("tracker_id", track.get("track_id"))
                    for track in tracks
                    if isinstance(track, Mapping)
                ),
                now_ts=float(world_now_ts),
            )
            continuity = self._prepare_tracking_cohort(
                sensor_id=int(sensor_id),
                camera_id=str(camera_id),
                frame_id=int(frame_id),
                observed_at_us=int(temporal_contract["observed_at_us"]),
                tracks=tracks,
                footpoints=footpoints,
            )
            self._set_active_tracks_snapshot(sensor_id, frame_id, tracks)
            if not identity_v2_authoritative:
                self._maybe_log_stable_id_metrics(sensor_id, now_ts)

            if self.diagnostics_logger:
                bbox3d_count = 0
                world_count = 0
                people_count = 0
                for item in diagnostics_tracks:
                    if not isinstance(item, dict):
                        continue
                    if int(item.get("class_id", -1)) == 0:
                        people_count += 1
                    if isinstance(item.get("bbox3d"), dict):
                        bbox3d_count += 1
                    if isinstance(item.get("world"), list):
                        world_count += 1
                self.diagnostics_logger.log_frame(
                    {
                        "type": "v3dt_tracking_frame",
                        "ts": float(now_ts),
                        "ts_us": int(self._frame_timestamp_us(frame_meta)),
                        "source_id": int(source_id),
                        "camera_id": camera_id,
                        "frame_id": frame_id,
                        "tracks": diagnostics_tracks,
                        "counts": {
                            "tracks": len(diagnostics_tracks),
                            "people_tracks": people_count,
                            "bbox3d_tracks": bbox3d_count,
                            "world_tracks": world_count,
                        },
                    }
                )

            if reid_debug and (now_ts - float(self._reid_debug_last_log)) >= 1.0:
                logger.info(
                    "ReID debug: frames=%d tracks=%d people=%d emb_found=%d emb_missing=%d reid_unique_id=%d layer=%s",
                    int(self._reid_debug_frames),
                    int(self._reid_debug_objects),
                    int(self._reid_debug_people),
                    int(self._reid_debug_emb_found),
                    int(self._reid_debug_emb_missing),
                    int(self._reid_unique_id),
                    str(self._reid_layer_name),
                )
                self._reid_debug_last_log = float(now_ts)
                self._reid_debug_frames = 0
                self._reid_debug_objects = 0
                self._reid_debug_people = 0
                self._reid_debug_emb_found = 0
                self._reid_debug_emb_missing = 0

            tracker_keys_changed = self._observed_tracker_keys_changed(
                sensor_id,
                tracks,
            )
            self._enqueue_tracking_publication(
                sensor_id=int(sensor_id),
                camera_id=str(camera_id),
                frame_meta=frame_meta,
                tracks=tracks,
                footpoints=footpoints,
                now_ts=float(now_ts),
                temporal_contract=temporal_contract,
                continuity=continuity,
                force=tracker_keys_changed,
                identity_v2_primitives=(
                    identity_v2_primitives if identity_v2_shadow else None
                ),
            )
            _record_core_stage_timing(
                "analytics.post_frame",
                post_frame_start_ns,
                item_count=len(tracks),
            )
            _record_core_stage_timing("analytics.handle_frame_ds8", frame_start_ns, item_count=len(tracks))
        except Exception:  # pragma: no cover - defensive guardrail
            logger.exception("Failed to process analytics telemetry for frame (DS8)")
            _record_core_stage_timing("analytics.handle_frame_ds8", frame_start_ns)

    def handle_frame(self, frame_meta: Any) -> None:
        lease = self.publication_gate.acquire()
        if lease is None:
            _increment_core_counter("runtime_publication_gate_rejected_frames_total")
            return
        with lease:
            self._handle_frame_compat_admitted(frame_meta)

    def _handle_frame_compat_admitted(self, frame_meta: Any) -> None:
        """Extract tracking telemetry for a single frame and publish it."""
        if not _allow_raw_pyds_compat():
            _record_quarantined_compat_path("_AnalyticsTelemetryProcessor.handle_frame", _PYDS_COMPAT_ENV)
            return
        try:
            source_id = self._frame_source_id(frame_meta)
            sensor_id = self.sensor_id_map.get(source_id, source_id)
            camera_id = self.camera_labels.get(sensor_id, f"camera_{sensor_id}")
            now_ts = time.time()
            temporal_contract = _frame_temporal_contract(frame_meta, observed_at_s=now_ts)
            self._log_diag_session_start()

            tracks: List[Dict[str, Any]] = []
            diagnostics_tracks: List[Dict[str, Any]] = []
            frame_id = int(getattr(frame_meta, "frame_num", -1))
            self._begin_source_frame_timeline(
                sensor_id=int(sensor_id),
                frame_id=int(frame_id),
                observed_at_us=int(temporal_contract["observed_at_us"]),
                media_pts_ns=int(temporal_contract.get("media_pts_ns", 0) or 0),
            )
            world_now_ts = self._world_timestamp_for_frame(
                int(sensor_id),
                media_pts_ns=temporal_contract.get("media_pts_ns", 0),
                observed_ts=float(now_ts),
            )
            occupancy_counts: Dict[str, int] = {}
            present_track_ids: set[int] = set()
            present_stable_ids: set[int] = set()
            footpoints: List[Footpoint] = []
            frame_dims = self._track_image_size(sensor_id, frame_meta)

            for obj_meta in self._iter_object_meta(frame_meta):
                raw = self._build_track_dict(obj_meta, camera_id)
                if raw is None:
                    continue
                diag_track = dict(raw)
                diag_track["frame_id"] = frame_id
                diag_track["source_id"] = int(source_id)

                track_id = int(raw.get("track_id", -1))
                if track_id < 0:
                    continue
                present_track_ids.add(track_id)

                try:
                    class_id = int(raw.get("class_id", -1))
                except Exception:
                    class_id = -1
                zone = raw.get("zone")
                zone_source = raw.get("zone_source")
                zone_authoritative = raw.get("zone_authoritative") is True
                if class_id != 0:
                    self._stamp_osd_label(obj_meta, sensor_id=sensor_id, stable_id=None)
                    diagnostics_tracks.append(diag_track)
                    continue

                if not zone:
                    zone = _fallback_zone_from_camera(camera_id)
                    zone_source = "camera_default" if zone else None
                    zone_authoritative = False

                stable_id = self._maybe_assign_stable_id(
                    sensor_id=sensor_id,
                    track_id=track_id,
                    bbox=raw.get("bbox"),
                    zone=zone,
                    ts=now_ts,
                    frame_bgr=None,
                    embedding=None,
                )
                if stable_id is None:
                    self._stamp_osd_label(obj_meta, sensor_id=sensor_id, stable_id=None)
                    diagnostics_tracks.append(diag_track)
                    continue

                stable_id_int = int(stable_id)
                present_stable_ids.add(stable_id_int)
                tracker_id_int = int(track_id)
                mgr = getattr(self.pipeline, "stable_id_mgr", None)
                manager_sensor_id = self._stable_id_manager_sensor_id(sensor_id)
                identity_contract = _stable_identity_contract(
                    mgr,
                    sensor_id=manager_sensor_id,
                    tracker_id=tracker_id_int,
                )
                id_display = None
                if self._reid_diag_use_tracker_id:
                    id_display = f"[{tracker_id_int}] | [{stable_id_int}]"

                dwell = self._update_dwell_time(sensor_id, stable_id_int, zone, now_ts)

                analytics = raw.get("analytics")
                if analytics and "lcStatus" in analytics:
                    lc = analytics["lcStatus"]
                    if isinstance(lc, dict):
                        for line_name, status in lc.items():
                            if status == 1:
                                self._record_transition(
                                    sensor_id=sensor_id,
                                    stable_id=stable_id_int,
                                    line_name=line_name,
                                    ts=now_ts,
                                )

                if zone:
                    occupancy_counts[zone] = occupancy_counts.get(zone, 0) + 1

                public_track: Dict[str, Any] = {
                    "stable_id": stable_id_int,
                    "tracker_id": tracker_id_int,
                    "tracker_lifecycle_generation": self._tracking_lifecycle.peek_generation(
                        int(sensor_id),
                        int(tracker_id_int),
                        frame_id=int(frame_id),
                        observed_at_us=int(temporal_contract["observed_at_us"]),
                        bbox=raw.get("bbox"),
                    ),
                    "camera_id": camera_id,
                    "source_id": int(source_id),
                    "bbox": raw.get("bbox"),
                    "center": raw.get("center"),
                    "class_id": 0,
                    "confidence": raw.get("confidence"),
                    "tracker_confidence": raw.get("tracker_confidence"),
                    "analytics": raw.get("analytics"),
                    "zone": zone,
                    "zone_source": zone_source,
                    "zone_authoritative": zone_authoritative,
                    "frame_id": frame_id,
                    "dwell_time": dwell,
                    **temporal_contract,
                    **identity_contract,
                }
                frame_w, frame_h = frame_dims
                if frame_w > 8 and frame_h > 8:
                    public_track["image_size"] = [int(frame_w), int(frame_h)]
                if id_display:
                    public_track["id_display"] = str(id_display)
                for key in (
                    "bbox3d",
                    "velocity3d",
                    "visibility",
                    "image_foot",
                    "image_base",
                    "world",
                    "world_valid",
                    "world_quality",
                    "world_quality_reason",
                    "world_frame",
                    "world_frame_revision",
                    "world_transform_sha256",
                    "world_source",
                ):
                    if key in raw:
                        public_track[key] = raw.get(key)
                for key in _WORLD_ESTIMATOR_DIAGNOSTIC_FIELDS:
                    if key in raw:
                        public_track[key] = raw.get(key)

                pose_kpts_abs = self._extract_pose_keypoints_for_anchor(obj_meta, raw.get("bbox") or [])
                depth_result = self._extract_object_depth_result(obj_meta)
                self._augment_track_with_world(
                    sensor_id,
                    camera_id,
                    public_track,
                    obj_meta=obj_meta,
                    pose_kpts_abs=pose_kpts_abs,
                    depth_result=depth_result,
                    world_now_ts=float(world_now_ts),
                )
                self._apply_scene_prior_shadow(camera_id, public_track)
                self._apply_public_depth_fields(public_track, depth_result)
                try:
                    setattr(obj_meta, "_noesis_depth_used_m", public_track.get("depth_used_m"))
                except Exception:
                    pass
                self._stamp_osd_label(obj_meta, sensor_id=sensor_id, stable_id=stable_id_int)
                diag_track.update(
                    {
                        "stable_id": stable_id_int,
                        "tracker_id": tracker_id_int,
                            **({"id_display": str(id_display)} if id_display else {}),
                            "zone": zone,
                            "zone_source": zone_source,
                            "zone_authoritative": zone_authoritative,
                            "dwell_time": dwell,
                        "world": public_track.get("world"),
                        "world_valid": public_track.get("world_valid"),
                        "world_quality": public_track.get("world_quality"),
                        "world_quality_reason": public_track.get("world_quality_reason"),
                        "world_frame": public_track.get("world_frame"),
                        "world_source": public_track.get("world_source"),
                        "depth_status": depth_result.status if depth_result is not None else None,
                        "depth_anchor_source": depth_result.anchor_source if depth_result is not None else None,
                        "depth_anchor_m": depth_result.anchor_depth_m if depth_result is not None else None,
                        "depth_registered_m": public_track.get("depth_registered_m"),
                        "depth_used_m": public_track.get("depth_used_m"),
                        "depth_registration_status": public_track.get("depth_registration_status"),
                        "depth_registration_id": public_track.get("depth_registration_id"),
                        "depth_samples": depth_result.sample_count if depth_result is not None else None,
                        "depth_valid_fraction": depth_result.valid_fraction if depth_result is not None else None,
                    }
                )
                tracks.append(public_track)
                diagnostics_tracks.append(diag_track)

                fp = self._footpoint_from_track(
                    public_track,
                    frame_dims,
                    target_image_size=self._bev_target_image_size(
                        sensor_id,
                        camera_id,
                    ),
                )
                if fp is not None:
                    footpoints.append(fp)

            self._publish_occupancy(sensor_id, occupancy_counts)
            self._cleanup_zone_state(sensor_id, present_stable_ids)
            self._maintain_stable_ids(sensor_id, present_track_ids, now_ts)
            if not tracks and os.environ.get("NOESIS_REID_TEST_MODE") == "1":
                synthetic = {
                    "camera_id": camera_id,
                    "stable_id": 1,
                    "tracker_id": 1,
                    "bbox": (0.0, 0.0, 10.0, 10.0),
                    "image_size": [int(frame_dims[0]), int(frame_dims[1])],
                    "frame_id": frame_id,
                    "zone": None,
                    "class_id": 0,
                    "identity_kind": "provisional",
                    **temporal_contract,
                }
                tracks.append(synthetic)
                present_stable_ids.add(1)
                present_track_ids.add(1)
            self._clear_absent_world_state(
                sensor_id,
                (
                    track.get("tracker_id", track.get("track_id"))
                    for track in tracks
                    if isinstance(track, Mapping)
                ),
                now_ts=float(world_now_ts),
            )
            continuity = self._prepare_tracking_cohort(
                sensor_id=int(sensor_id),
                camera_id=str(camera_id),
                frame_id=int(frame_id),
                observed_at_us=int(temporal_contract["observed_at_us"]),
                tracks=tracks,
                footpoints=footpoints,
            )
            self._set_active_tracks_snapshot(sensor_id, frame_id, tracks)
            self._maybe_log_stable_id_metrics(sensor_id, now_ts)

            if self.diagnostics_logger:
                bbox3d_count = 0
                world_count = 0
                people_count = 0
                for item in diagnostics_tracks:
                    if not isinstance(item, dict):
                        continue
                    if int(item.get("class_id", -1)) == 0:
                        people_count += 1
                    if isinstance(item.get("bbox3d"), dict):
                        bbox3d_count += 1
                    if isinstance(item.get("world"), list):
                        world_count += 1
                self.diagnostics_logger.log_frame(
                    {
                        "type": "v3dt_tracking_frame",
                        "ts": float(now_ts),
                        "ts_us": int(self._frame_timestamp_us(frame_meta)),
                        "source_id": int(source_id),
                        "camera_id": camera_id,
                        "frame_id": frame_id,
                        "tracks": diagnostics_tracks,
                        "counts": {
                            "tracks": len(diagnostics_tracks),
                            "people_tracks": people_count,
                            "bbox3d_tracks": bbox3d_count,
                            "world_tracks": world_count,
                        },
                    }
                )

            tracker_keys_changed = self._observed_tracker_keys_changed(
                sensor_id,
                tracks,
            )
            self._enqueue_tracking_publication(
                sensor_id=int(sensor_id),
                camera_id=str(camera_id),
                frame_meta=frame_meta,
                tracks=tracks,
                footpoints=footpoints,
                now_ts=float(now_ts),
                temporal_contract=temporal_contract,
                continuity=continuity,
                force=tracker_keys_changed,
            )
        except Exception:  # pragma: no cover - defensive guardrail
            logger.exception("Failed to process analytics telemetry for frame")

    def _frame_source_id(self, frame_meta: Any) -> int:
        for attr in ("source_id", "pad_index", "camera_id"):
            value = getattr(frame_meta, attr, None)
            if value is None:
                continue
            try:
                return int(value)
            except Exception:
                continue
        return 0

    def _ensure_mask_alpha(self) -> float:
        if self._mask_alpha_ready:
            return float(self._mask_alpha)

        alpha = 0.35
        try:
            vis_cfg = (getattr(self.pipeline, "config", {}) or {}).get("visualization") or {}
            if isinstance(vis_cfg, Mapping):
                masks_cfg = vis_cfg.get("instance_masks") or vis_cfg.get("masks") or {}
                if isinstance(masks_cfg, Mapping):
                    raw = masks_cfg.get("alpha", masks_cfg.get("mask_alpha", alpha))
                    if raw is not None:
                        alpha = float(raw)
        except Exception:
            alpha = 0.35

        env = str(os.environ.get("NOESIS_INSTANCE_MASK_ALPHA", "")).strip()
        if env:
            try:
                alpha = float(env)
            except Exception:
                pass

        alpha = float(max(0.0, min(1.0, alpha)))
        self._mask_alpha = alpha
        self._mask_alpha_ready = True
        return alpha

    @staticmethod
    def _stable_id_color_rgb(stable_id: int) -> Tuple[float, float, float]:
        hue = float((int(stable_id) * 47) % 360)
        r, g, b = colorsys.hls_to_rgb(hue / 360.0, 0.60, 0.80)
        return float(r), float(g), float(b)

    def _apply_instance_mask_color_ds8(self, obj_meta: Any, *, stable_id: int) -> None:
        """Force instance mask color to be stable-id keyed (matches FE hue math)."""
        rect = getattr(obj_meta, "rect_params", None)
        if rect is None:
            return
        mask_params = getattr(obj_meta, "mask_params", None)
        if mask_params is None:
            return
        try:
            if getattr(mask_params, "data", None) is None:
                return
            size = int(getattr(mask_params, "size", 0) or 0)
            if size <= 0:
                return
        except Exception:
            return

        alpha = self._ensure_mask_alpha()
        r, g, b = self._stable_id_color_rgb(int(stable_id))

        try:
            border = getattr(rect, "border_color", None)
            if border is not None:
                setter = getattr(border, "set", None)
                if callable(setter):
                    setter(float(r), float(g), float(b), float(alpha))
                else:
                    setattr(border, "red", float(r))
                    setattr(border, "green", float(g))
                    setattr(border, "blue", float(b))
                    setattr(border, "alpha", float(alpha))
        except Exception:
            pass

        # Ensure masks aren't clipped by box border thickness (bboxes are disabled by default in DS8).
        try:
            setattr(rect, "border_width", 0)
        except Exception:
            pass

    def _frame_dims(self) -> Tuple[int, int]:
        try:
            width, height = getattr(self.pipeline, "frame_size", (0, 0))
            return int(width or 0), int(height or 0)
        except Exception:
            return 0, 0

    def _frame_source_size(self, frame_meta: Any) -> Tuple[int, int]:
        try:
            # Object metadata is expressed in the post-dewarp/post-mux raster.
            # Prefer that canonical frame size; source-native dimensions are
            # only a fallback for metadata producers that do not expose it.
            frame_w = int(_meta_lookup(frame_meta, "frame_width", "width", "source_frame_width", default=0) or 0)
            frame_h = int(_meta_lookup(frame_meta, "frame_height", "height", "source_frame_height", default=0) or 0)
        except Exception:
            frame_w = 0
            frame_h = 0
        if frame_w <= 0 or frame_h <= 0:
            return self._frame_dims()
        return max(0, int(frame_w)), max(0, int(frame_h))

    def _intrinsics_base_image_size(self, sensor_id: int) -> Optional[Tuple[int, int]]:
        provider = getattr(self, "bev_calibration", None)
        loader = getattr(provider, "_intrinsics_loader", None)
        getter = getattr(loader, "get", None)
        if not callable(getter):
            return None
        try:
            intr = getter(int(sensor_id))
        except Exception:
            intr = None
        if intr is None:
            return None
        try:
            width = int(getattr(intr, "width", 0) or 0)
            height = int(getattr(intr, "height", 0) or 0)
        except Exception:
            return None
        if width > 8 and height > 8:
            return width, height
        # A principal point is not required to be the image center. Inferring
        # dimensions as 2*cx/2*cy silently rescales dewarped detections whenever
        # the calibrated optical center is off-center. The canonical frame
        # metadata owns the raster whenever it declares one.
        return None

    def _track_image_size(self, sensor_id: int, frame_meta: Any) -> Tuple[int, int]:
        # The tracker bbox/pose raster is the canonical frame raster, not the
        # calibration loader's raw intrinsics resolution.  Using the latter
        # when the frame is already dewarped/muxed up (e.g. 1280 -> 1920)
        # causes a second 1.5x transform before projection.
        canonical_w, canonical_h = _canonical_frame_size(
            frame_meta,
            self._frame_dims(),
        )
        if canonical_w > 8 and canonical_h > 8:
            return canonical_w, canonical_h
        # If native metadata does not carry the raster, streammux's configured
        # output is the post-dewarp tracker coordinate space.  Do not fall
        # back to a source-native calibration size and apply a second scale.
        configured_w, configured_h = self._frame_dims()
        if configured_w > 8 and configured_h > 8:
            return configured_w, configured_h
        intrinsic_size = self._intrinsics_base_image_size(sensor_id)
        if intrinsic_size is not None:
            return intrinsic_size
        return self._frame_source_size(frame_meta)

    def _bev_target_image_size(
        self,
        sensor_id: int,
        camera_id: str,
    ) -> Optional[Tuple[int, int]]:
        resolver = self.bev_calibration
        if resolver is None:
            return None
        try:
            calib = resolver.snapshot(sensor_id, camera_id)
        except Exception:
            return None
        return self._normalize_image_size(getattr(calib, "image_size", None))

    def _footpoint_from_track(
        self,
        track: Mapping[str, Any],
        frame_dims: Tuple[int, int],
        *,
        target_image_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[Footpoint]:
        def _parse_uv(value: Any) -> Optional[Tuple[float, float]]:
            if not isinstance(value, (list, tuple)) or len(value) < 2:
                return None
            try:
                u = float(value[0])
                v = float(value[1])
            except Exception:
                return None
            if not math.isfinite(u) or not math.isfinite(v):
                return None
            return u, v

        def _clip_uv(
            u: float,
            v: float,
            *,
            vertical_overshoot_ratio: float = 0.01,
        ) -> Optional[Tuple[float, float]]:
            frame_w, frame_h = source_image_size_tuple or frame_dims
            if frame_h:
                margin = max(2.0, float(vertical_overshoot_ratio) * float(frame_h))
                if v < -margin or v > (frame_h + margin):
                    return None
                v = float(np.clip(v, 0.0, float(frame_h)))
            if frame_w:
                u = float(np.clip(u, 0.0, float(frame_w)))
            return u, v

        source_image_size_tuple = (
            self._normalize_image_size(track.get("image_size") or track.get("frame_size"))
            or self._normalize_image_size(frame_dims)
        )
        target_image_size_tuple = (
            self._normalize_image_size(target_image_size) or source_image_size_tuple
        )
        scale_x = 1.0
        scale_y = 1.0
        if source_image_size_tuple is not None and target_image_size_tuple is not None:
            src_w, src_h = source_image_size_tuple
            dst_w, dst_h = target_image_size_tuple
            if src_w > 0 and src_h > 0 and dst_w > 0 and dst_h > 0:
                scale_x = float(dst_w) / float(src_w)
                scale_y = float(dst_h) / float(src_h)

        def _scale_uv_to_target(u: float, v: float) -> Tuple[float, float]:
            return float(u) * float(scale_x), float(v) * float(scale_y)

        def _scale_bbox_to_target(
            bbox: Tuple[float, float, float, float],
        ) -> Tuple[float, float, float, float]:
            left, top, width, height = bbox
            return (
                float(left) * float(scale_x),
                float(top) * float(scale_y),
                float(width) * float(scale_x),
                float(height) * float(scale_y),
            )

        try:
            class_id = int(track.get("class_id", -1))
        except Exception:
            class_id = -1

        if not self._bev_class_ids_ready:
            cls_ids: set[int] = set()
            try:
                vis_cfg = self.pipeline.config.get("visualization") or {}
                trails_cfg = vis_cfg.get("trails") if isinstance(vis_cfg, Mapping) else None
                if isinstance(trails_cfg, Mapping):
                    raw = trails_cfg.get("class_ids")
                    if isinstance(raw, (list, tuple, set)):
                        for item in raw:
                            try:
                                cls_ids.add(int(item))
                            except Exception:
                                continue
                    elif raw is not None:
                        try:
                            cls_ids.add(int(raw))
                        except Exception:
                            pass
            except Exception:
                cls_ids = set()
            if not cls_ids:
                cls_ids = {0}
            self._bev_class_ids = frozenset(cls_ids)
            self._bev_class_ids_ready = True

        if class_id not in self._bev_class_ids:
            return None

        # A revision-bound canonical world point is already the spatial
        # authority consumed by BEV.  Do not make that transport contingent on
        # an image contact that may legitimately be absent while a tracked
        # person is partially outside the raster.  A BEV-compatible identity
        # remains mandatory so the renderer never creates anonymous history.
        canonical_world_transport = False
        if track.get("world_valid") is True:
            raw_world = track.get("world")
            try:
                world_x_value = float(raw_world[0])
                world_z_value = float(raw_world[2])
            except (IndexError, TypeError, ValueError, OverflowError):
                pass
            else:
                try:
                    tracker_value = int(
                        track.get("tracker_id", track.get("track_id"))
                    )
                except (TypeError, ValueError, OverflowError):
                    tracker_value = -1
                try:
                    stable_value = int(track.get("stable_id"))
                except (TypeError, ValueError, OverflowError):
                    stable_value = -1
                canonical_world_transport = bool(
                    (tracker_value >= 0 or stable_value > 0)
                    and math.isfinite(world_x_value)
                    and math.isfinite(world_z_value)
                    and str(track.get("world_frame") or "").strip()
                    and str(track.get("world_frame_revision") or "").strip()
                )

        u = v = None
        method = None
        bbox_source_tuple: Optional[Tuple[float, float, float, float]] = None
        bbox = track.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            try:
                left_b, top_b, width_b, height_b = [float(x) for x in bbox[:4]]
                if width_b > 0.0 and height_b > 0.0:
                    bbox_source_tuple = (
                        float(left_b),
                        float(top_b),
                        float(width_b),
                        float(height_b),
                    )
            except Exception:
                bbox_source_tuple = None

        image_anchor_keys = (("image_foot", "image_foot"), ("image_base", "image_base"))
        if track.get("world_source") == "bbox3d" or isinstance(track.get("bbox3d"), dict):
            image_anchor_keys = (("image_base", "image_base"), ("image_foot", "image_foot"))

        for key, label in image_anchor_keys:
            uv = _parse_uv(track.get(key))
            if uv is None:
                continue
            clipped = _clip_uv(*uv, vertical_overshoot_ratio=0.10)
            if clipped is None:
                continue
            u, v = clipped
            method = label
            break

        if (u is None or v is None) and bbox_source_tuple is not None:
            left, top, width, height = bbox_source_tuple
            if width > 0.0 and height > 0.0:
                clipped = _clip_uv(
                    float(left + width * 0.5),
                    float(top + height),
                )
                if clipped is not None:
                    u, v = clipped
                    method = "bbox"
        if u is None or v is None:
            if not canonical_world_transport:
                return None
            method = "world"
        else:
            u, v = _scale_uv_to_target(float(u), float(v))
        bbox_tuple = (
            _scale_bbox_to_target(bbox_source_tuple)
            if bbox_source_tuple is not None
            else None
        )

        stable_id = track.get("stable_id")
        try:
            stable_id_int = int(stable_id) if stable_id not in (None, "", -1) else None
        except Exception:
            stable_id_int = None
        if stable_id_int is not None and stable_id_int <= 0:
            stable_id_int = None
        tracker_id = track.get("tracker_id", track.get("track_id"))
        try:
            tracker_id_int = int(tracker_id) if tracker_id not in (None, "", -1) else None
        except Exception:
            tracker_id_int = None
        if tracker_id_int is not None and tracker_id_int < 0:
            tracker_id_int = None
        world_x = None
        world_z = None
        if track.get("world_valid") is True:
            world = track.get("world")
            if isinstance(world, (list, tuple)) and len(world) >= 3:
                try:
                    wx = float(world[0])
                    wz = float(world[2])
                    if math.isfinite(wx) and math.isfinite(wz):
                        world_x = float(wx)
                        world_z = float(wz)
                except Exception:
                    world_x = None
                    world_z = None
        depth_m = None
        depth_source = None
        depth_candidate = usable_registered_depth_m(track)
        if depth_candidate is not None and 0.05 < depth_candidate < 50.0:
            depth_m = float(depth_candidate)
            depth_source = "depth_registered_m"
        image_candidates: List[Dict[str, Any]] = []
        for idx, (key, label) in enumerate(
            (("image_foot", "image_foot"), ("image_base", "image_base"))
        ):
            uv = _parse_uv(track.get(key))
            if uv is None:
                continue
            cand_u, cand_v = _scale_uv_to_target(float(uv[0]), float(uv[1]))
            image_candidates.append(
                {
                    "name": str(label),
                    "source": str(key),
                    "priority": int(idx),
                    "u": float(cand_u),
                    "v": float(cand_v),
                }
            )
        if bbox_source_tuple is not None:
            left, top, width, height = bbox_source_tuple
            bottom_center = _scale_uv_to_target(
                float(left + width * 0.5), float(top + height)
            )
            lower_center = _scale_uv_to_target(
                float(left + width * 0.5), float(top + height * 0.95)
            )
            image_candidates.extend(
                [
                    {
                        "name": "bbox_bottom_center",
                        "source": "bbox",
                        "priority": 10,
                        "u": float(bottom_center[0]),
                        "v": float(bottom_center[1]),
                    },
                    {
                        "name": "bbox_lower_center_95",
                        "source": "bbox",
                        "priority": 11,
                        "u": float(lower_center[0]),
                        "v": float(lower_center[1]),
                    },
                ]
            )
        frame_id_value = None
        try:
            frame_id_value = (
                int(track.get("frame_id"))
                if track.get("frame_id") is not None
                else None
            )
        except Exception:
            frame_id_value = None
        trail_append = track.get("trail_append_allowed")
        trail_append_bool: Optional[bool]
        if trail_append is None:
            trail_append_bool = None
        else:
            trail_append_bool = bool(trail_append)
        trail_break = track.get("trail_break_required")
        trail_break_bool = bool(trail_break) if trail_break is not None else None
        trail_segment = track.get("trail_segment_id")
        try:
            trail_segment_int = int(trail_segment) if trail_segment is not None else None
        except Exception:
            trail_segment_int = None
        lifecycle_generation = track.get("tracker_lifecycle_generation")
        try:
            lifecycle_generation_int = (
                int(lifecycle_generation)
                if lifecycle_generation is not None
                else None
            )
        except Exception:
            lifecycle_generation_int = None
        track_key_value = str(track.get("track_key") or "").strip() or None
        if track_key_value is None and tracker_id_int is not None:
            try:
                source_value = int(track.get("source_id"))
            except (TypeError, ValueError):
                source_value = None
            if source_value is not None and lifecycle_generation_int is not None:
                track_key_value = (
                    f"{source_value}:{tracker_id_int}:{lifecycle_generation_int}"
                )
        world_transform_sha256 = str(
            track.get("world_transform_sha256") or ""
        ).strip() or None
        idle_jitter_raw = track.get("idle_jitter_m")
        try:
            idle_jitter = (
                float(idle_jitter_raw) if idle_jitter_raw is not None else None
            )
        except Exception:
            idle_jitter = None
        return Footpoint(
            u=u,
            v=v,
            method=method or "bbox",
            stable_id=stable_id_int,
            tracker_id=tracker_id_int,
            track_key=track_key_value,
            tracker_lifecycle_generation=lifecycle_generation_int,
            world_x=world_x,
            world_z=world_z,
            depth_m=depth_m,
            depth_source=depth_source,
            anchor_source=str(track.get("world_source")) if track.get("world_source") not in (None, "") else None,
            anchor_quality=str(track.get("world_quality")) if track.get("world_quality") not in (None, "") else None,
            anchor_reason=str(track.get("world_quality_reason")) if track.get("world_quality_reason") not in (None, "") else None,
            bbox=bbox_tuple,
            image_size=target_image_size_tuple or source_image_size_tuple,
            frame_id=frame_id_value,
            # Every live tracker footpoint is a projection consumer of the
            # canonical world estimator.  BEV may expose depth/ray candidates
            # for diagnostics, but it must not use them as display positions.
            canonical_world_required=True,
            world_frame=(
                str(track.get("world_frame"))
                if track.get("world_frame") not in (None, "")
                else None
            ),
            world_frame_revision=(
                str(track.get("world_frame_revision"))
                if track.get("world_frame_revision") not in (None, "")
                else None
            ),
            world_transform_sha256=world_transform_sha256,
            motion_mode=str(track.get("motion_mode")) if track.get("motion_mode") not in (None, "") else None,
            posture=str(track.get("posture")) if track.get("posture") not in (None, "") else None,
            trail_append_allowed=trail_append_bool,
            trail_break_required=trail_break_bool,
            trail_segment_id=trail_segment_int,
            idle_jitter_m=idle_jitter,
            debug={
                "image_candidates": image_candidates,
                "track_frame_id": frame_id_value,
                "track_image_size": (
                    list(source_image_size_tuple)
                    if source_image_size_tuple is not None
                    else None
                ),
                "bev_image_size": (
                    list(target_image_size_tuple)
                    if target_image_size_tuple is not None
                    else None
                ),
                "image_scale": [float(scale_x), float(scale_y)],
                "world": (
                    list(track.get("world"))
                    if isinstance(track.get("world"), (list, tuple))
                    else None
                ),
                "world_valid": (
                    bool(track.get("world_valid"))
                    if track.get("world_valid") is not None
                    else None
                ),
                "world_source": (
                    str(track.get("world_source"))
                    if track.get("world_source") not in (None, "")
                    else None
                ),
                "world_quality": (
                    str(track.get("world_quality"))
                    if track.get("world_quality") not in (None, "")
                    else None
                ),
                "motion_mode": (
                    str(track.get("motion_mode"))
                    if track.get("motion_mode") not in (None, "")
                    else None
                ),
                "posture": (
                    str(track.get("posture"))
                    if track.get("posture") not in (None, "")
                    else None
                ),
                "trail_append_allowed": trail_append_bool,
                "world_quality_reason": (
                    str(track.get("world_quality_reason"))
                    if track.get("world_quality_reason") not in (None, "")
                    else None
                ),
                "depth_status": (
                    str(track.get("depth_status"))
                    if track.get("depth_status") not in (None, "")
                    else None
                ),
                "depth_anchor_source": (
                    str(track.get("depth_anchor_source"))
                    if track.get("depth_anchor_source") not in (None, "")
                    else None
                ),
                "depth_anchor_m": track.get("depth_anchor_m"),
                "depth_used_m": track.get("depth_used_m"),
                "depth_registered_m": track.get("depth_registered_m"),
                "depth_registration_status": track.get("depth_registration_status"),
                "depth_registration_id": track.get("depth_registration_id"),
                "world_resolver": track.get("world_resolver"),
                "world_resolver_confidence": track.get("world_resolver_confidence"),
                "world_quantity": track.get("world_quantity"),
                "world_support_state": track.get("world_support_state"),
                "world_posture": track.get("world_posture"),
            },
        )

    @staticmethod
    def _stamp_footpoint_lifecycles(
        footpoints: Sequence[Footpoint],
        tracks: Sequence[Mapping[str, Any]],
    ) -> None:
        """Bind BEV trail identity to the lifecycle stamped for this frame."""

        generation_by_tracker: Dict[int, int] = {}
        generation_by_track_key: Dict[str, int] = {}
        for track in tracks:
            try:
                tracker_id = int(track.get("tracker_id", track.get("track_id")))
                generation = int(track.get("tracker_lifecycle_generation"))
            except (TypeError, ValueError):
                continue
            if tracker_id >= 0 and generation >= 0:
                generation_by_tracker[tracker_id] = generation
                key = str(track.get("track_key") or "").strip()
                if key:
                    generation_by_track_key[key] = generation
        for footpoint in footpoints:
            if footpoint.tracker_lifecycle_generation is not None:
                # Preserve the exact stamp copied from the public track.  A
                # tracker-id-only lookup can be ambiguous when a reconnect
                # reuses the numeric tracker ID in one cohort.
                continue
            track_key = str(footpoint.track_key or "").strip()
            if track_key and track_key in generation_by_track_key:
                footpoint.tracker_lifecycle_generation = generation_by_track_key[track_key]
                continue
            try:
                tracker_id = int(footpoint.tracker_id)
            except (TypeError, ValueError):
                continue
            footpoint.tracker_lifecycle_generation = generation_by_tracker.get(
                tracker_id
            )

    def _frame_timestamp_us(self, frame_meta: Any) -> int:
        pts_ns = int(_meta_lookup(frame_meta, "buf_pts", "buffer_pts", "pts", default=0) or 0)
        if pts_ns <= 0:
            pts_ns = int(time.time() * 1_000_000_000)
        return max(0, pts_ns // 1_000)

    @staticmethod
    def _apply_image_flip(
        u: float,
        v: float,
        width: int,
        height: int,
        flip_u: bool,
        flip_v: bool,
    ) -> Tuple[float, float]:
        if flip_u:
            u = float(max(0, width - 1)) - float(u)
        if flip_v:
            v = float(max(0, height - 1)) - float(v)
        return float(u), float(v)

    def _infer_image_flips(self, camera_id: str, calib: CalibrationSnapshot) -> Tuple[bool, bool]:
        return False, False

    def _world_track_key(self, sensor_id: int, track: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
        # Physical motion belongs to the camera-local tracker lifecycle.  A
        # semantic StableID may be reassigned or shared across cameras and must
        # never splice two people's kinematic state together.
        for key_name in ("tracker_id", "track_id"):
            raw = track.get(key_name)
            if raw is None:
                continue
            try:
                identity = int(raw)
            except Exception:
                continue
            if identity >= 0:
                return int(sensor_id), identity
        return None

    def _world_output_watermark_key(
        self,
        sensor_id: int,
        track: Mapping[str, Any],
        *,
        world_frame_id: Optional[str],
        world_frame_revision: Optional[str],
        world_transform_sha256: Optional[str],
    ) -> Optional[Tuple[int, int, int, str, str, str]]:
        """Return the exact public lifecycle/revision continuity key."""

        track_key = self._world_track_key(sensor_id, track)
        if track_key is None:
            return None
        try:
            generation = int(track.get("tracker_lifecycle_generation"))
        except (TypeError, ValueError, OverflowError):
            return None
        frame_id = str(world_frame_id or "").strip()
        revision = str(world_frame_revision or "").strip()
        transform = str(world_transform_sha256 or "").strip()
        if generation < 0 or not frame_id or not revision or not transform:
            return None
        return (
            int(track_key[0]),
            int(track_key[1]),
            generation,
            frame_id,
            revision,
            transform,
        )

    def _restore_world_output_watermark(
        self,
        state: _WorldAnchorState,
        key: Optional[Tuple[int, int, int, str, str, str]],
    ) -> bool:
        """Bind a recreated measurement state to its last public output."""

        if key is None:
            return False
        store = getattr(self, "_world_output_watermarks", None)
        if not isinstance(store, OrderedDict):
            return False
        watermark = store.get(key)
        if watermark is None:
            return False
        state_has_output = bool(
            state.last_output_world_x is not None
            and state.last_output_world_z is not None
        )
        if state_has_output:
            state_pts = state.last_output_media_pts_ns
            if (
                state_pts is not None
                and watermark.media_pts_ns is not None
                and int(state_pts) >= int(watermark.media_pts_ns)
            ):
                return False
            if float(state.last_output_filter_ts) >= float(watermark.filter_ts):
                return False
        store.move_to_end(key)
        state.last_output_world_x = float(watermark.world_x)
        state.last_output_world_z = float(watermark.world_z)
        state.last_output_media_pts_ns = watermark.media_pts_ns
        state.last_output_filter_ts = float(watermark.filter_ts)
        state.last_output_trail_segment_id = int(watermark.trail_segment_id)
        return True

    def _save_world_output_watermark(
        self,
        state: _WorldAnchorState,
        key: Optional[Tuple[int, int, int, str, str, str]],
    ) -> None:
        """Persist the final admission watermark with an explicit hard cap."""

        if key is None:
            return
        x = state.last_output_world_x
        z = state.last_output_world_z
        if x is None or z is None:
            return
        try:
            world_x = float(x)
            world_z = float(z)
            filter_ts = float(state.last_output_filter_ts)
            segment = int(state.last_output_trail_segment_id)
            media_pts = (
                int(state.last_output_media_pts_ns)
                if state.last_output_media_pts_ns is not None
                else None
            )
        except (TypeError, ValueError, OverflowError):
            return
        if not all(math.isfinite(value) for value in (world_x, world_z, filter_ts)):
            return
        store = getattr(self, "_world_output_watermarks", None)
        if not isinstance(store, OrderedDict):
            store = OrderedDict()
            self._world_output_watermarks = store
        store[key] = _WorldOutputWatermark(
            world_x=world_x,
            world_z=world_z,
            media_pts_ns=media_pts,
            filter_ts=filter_ts,
            trail_segment_id=segment,
        )
        store.move_to_end(key)
        capacity = max(
            1,
            int(getattr(self, "_world_output_watermark_capacity", 4096) or 4096),
        )
        while len(store) > capacity:
            store.popitem(last=False)

    def _maybe_prune_world_state(self, now_ts: float) -> None:
        if not self._world_state_by_track and not self._world_state_ghost_by_track:
            return
        if (float(now_ts) - float(self._world_state_last_prune_ts)) < float(self._world_state_prune_interval_s):
            return
        self._world_state_last_prune_ts = float(now_ts)
        ttl = float(self._world_state_ttl_s)
        if ttl <= 0.0:
            self._world_state_by_track.clear()
        else:
            expired: List[Tuple[int, int]] = []
            for key, state in self._world_state_by_track.items():
                ts = float(getattr(state, "ts", 0.0) or 0.0)
                if (float(now_ts) - ts) > ttl:
                    expired.append(key)
            for key in expired:
                self._world_state_by_track.pop(key, None)
        ghost_ttl = float(self._world_state_ghost_ttl_s)
        for key, (_state, absent_ts) in tuple(
            self._world_state_ghost_by_track.items()
        ):
            if ghost_ttl <= 0.0 or (float(now_ts) - float(absent_ts)) > ghost_ttl:
                self._world_state_ghost_by_track.pop(key, None)

    def _world_state_for_observation(
        self,
        key: Tuple[int, int],
        *,
        now_ts: float,
        bbox: Optional[Sequence[float]],
        lifecycle_generation: Optional[int] = None,
    ) -> Tuple[_WorldAnchorState, bool]:
        """Return active state or restore a visually continuous short ghost."""

        state = self._world_state_by_track.get(key)
        if state is not None and lifecycle_generation is not None:
            prior_generation = getattr(state, "tracker_lifecycle_generation", None)
            if prior_generation is not None and int(prior_generation) != int(lifecycle_generation):
                # A numeric tracker ID was reused.  Do not let its filter,
                # velocity, or seated lock cross the lifecycle boundary.
                self._world_state_by_track.pop(key, None)
                state = None
        if state is not None:
            if lifecycle_generation is not None:
                setattr(state, "tracker_lifecycle_generation", int(lifecycle_generation))
            return state, False
        ghost = self._world_state_ghost_by_track.pop(key, None)
        restored = False
        if ghost is not None:
            candidate, absent_ts = ghost
            prior_generation = getattr(candidate, "tracker_lifecycle_generation", None)
            if (
                lifecycle_generation is not None
                and prior_generation is not None
                and int(prior_generation) != int(lifecycle_generation)
            ):
                ghost = None
            if ghost is None:
                candidate = None
        if ghost is not None and candidate is not None:
            gap_s = float(now_ts) - float(absent_ts)
            lifecycle_proves_continuity = bool(
                lifecycle_generation is not None
                and prior_generation is not None
                and int(prior_generation) == int(lifecycle_generation)
                and 0.0 <= gap_s <= float(self._world_state_ghost_ttl_s)
            )
            if lifecycle_proves_continuity:
                # TrackingLifecycleRegistry already validated the return gap,
                # normalized bbox displacement, and size ratio before reusing
                # this generation.  Ground state must not impose a second,
                # stricter association rule: doing so starts the filter cold
                # under the same public trail identity and permits an
                # un-gated jump.  A generation change was rejected above.
                restored = True
            try:
                if not restored:
                    left, top, width, height = (
                        float(value) for value in bbox[:4]  # type: ignore[index]
                    )
                    prior_left, prior_top, prior_width, prior_height = (
                        float(value) for value in candidate.bbox_geometry  # type: ignore[union-attr]
                    )
                    if (
                        width <= 1.0
                        or height <= 1.0
                        or prior_width <= 1.0
                        or prior_height <= 1.0
                    ):
                        raise ValueError("bbox geometry is not usable")
                    inter_left = max(left, prior_left)
                    inter_top = max(top, prior_top)
                    inter_right = min(left + width, prior_left + prior_width)
                    inter_bottom = min(top + height, prior_top + prior_height)
                    inter_width = max(0.0, inter_right - inter_left)
                    inter_height = max(0.0, inter_bottom - inter_top)
                    intersection = inter_width * inter_height
                    union = (
                        (width * height)
                        + (prior_width * prior_height)
                        - intersection
                    )
                    iou = intersection / union if union > 1e-6 else 0.0
                    center_u = left + width * 0.5
                    center_v = top + height * 0.5
                    prior_center_u = prior_left + prior_width * 0.5
                    prior_center_v = prior_top + prior_height * 0.5
                    diagonal = max(
                        1.0,
                        math.hypot(width, height),
                        math.hypot(prior_width, prior_height),
                    )
                    center_ratio = math.hypot(
                        center_u - prior_center_u,
                        center_v - prior_center_v,
                    ) / diagonal
                    area_ratio = max(
                        (width * height) / (prior_width * prior_height),
                        (prior_width * prior_height) / (width * height),
                    )
                    restored = bool(
                        0.0 <= gap_s <= float(self._world_state_ghost_ttl_s)
                        and width > 1.0
                        and height > 1.0
                        and prior_width > 1.0
                        and prior_height > 1.0
                        and all(
                            math.isfinite(value)
                            for value in (
                                iou,
                                center_ratio,
                                area_ratio,
                            )
                        )
                        and iou >= 0.75
                        and center_ratio <= 0.25
                        and area_ratio <= 1.75
                    )
            except (TypeError, ValueError, IndexError, ZeroDivisionError):
                if not lifecycle_proves_continuity:
                    restored = False
            if restored:
                state = candidate
        if state is None:
            state = _WorldAnchorState()
        if lifecycle_generation is not None:
            setattr(state, "tracker_lifecycle_generation", int(lifecycle_generation))
        self._world_state_by_track[key] = state
        return state, restored

    def _clear_absent_world_state(
        self,
        sensor_id: int,
        present_track_ids: Iterable[int],
        *,
        now_ts: Optional[float] = None,
    ) -> None:
        """Quarantine kinematics when a tracker lifecycle disappears.

        ``_world_state_by_track`` is deliberately keyed by the physical
        ``(sensor_id, tracker_id)`` pair.  The tracker lifecycle registry can
        assign a new generation when NvDCF omits an object for one processed
        frame and later re-creates the same numeric tracker ID.  A TTL-only
        cache would then splice a reused ID into the old person's velocity,
        innovation gate, and stationary lock.  Dropping state immediately,
        however, makes a one-frame metadata/ReID gap restart a physically
        continuous person cold and causes the BEV dot to disappear.  Move the
        state into a short quarantine; `_world_state_for_observation` restores
        it only when the returning bbox matches the prior image position and
        scale.  Trail lifecycle remains separate and is never restored.

        The caller supplies the same canonical tracker rows that feed the
        lifecycle registry.  This matters when StableID cannot produce a
        public row for one frame: retaining the physical key would still
        splice the next public lifecycle into stale state.  Do not carry the
        state blindly across that boundary.  This is bounded scalar dictionary
        work over the existing small track set and does not touch image data or
        the media path.
        """
        present: set[int] = set()
        for track_id in present_track_ids:
            try:
                parsed = int(track_id)
            except Exception:
                continue
            if parsed >= 0:
                present.add(parsed)
        sensor = int(sensor_id)
        absent_ts = float(time.time() if now_ts is None else now_ts)
        for key in tuple(self._world_state_by_track):
            try:
                key_sensor, tracker_id = int(key[0]), int(key[1])
            except Exception:
                continue
            if key_sensor == sensor and tracker_id not in present:
                state = self._world_state_by_track.pop(key, None)
                if state is not None:
                    self._world_state_ghost_by_track[key] = (state, absent_ts)

    def _extract_pose_payload_for_anchor(self, obj_meta: Any) -> Optional[Dict[str, Any]]:
        if obj_meta is None or noesis_pose_meta_ext is None:
            return None
        extract_obj = getattr(noesis_pose_meta_ext, "extract_pose_features", None)
        if extract_obj is None or not callable(extract_obj):
            return None
        try:
            raw = extract_obj(obj_meta)
        except Exception:
            return None
        if raw is None:
            return None
        try:
            payload = json.loads(str(raw))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _extract_pose_keypoints_for_anchor(
        self,
        obj_meta: Any,
        bbox: Sequence[float],
    ) -> Optional[np.ndarray]:
        payload = self._extract_pose_payload_for_anchor(obj_meta)
        if payload is not None:
            keypoints_abs = self._keypoints_abs_from_pose_payload(payload, bbox)
            if keypoints_abs is not None:
                _increment_core_counter("detection_wake.pose_anchor_payload_hit")
                return keypoints_abs

        if obj_meta is not None and noesis_pose_meta_ext is not None:
            if int(self._pose_anchor_native_remaining) <= 0:
                _increment_core_counter("detection_wake.pose_anchor_native_budget_skipped")
                return None
            extract_obj = getattr(noesis_pose_meta_ext, "extract_pose_keypoints", None)
            if callable(extract_obj):
                self._pose_anchor_native_remaining = max(0, int(self._pose_anchor_native_remaining) - 1)
                try:
                    payload = extract_obj(
                        obj_meta,
                        int(self._pose_anchor_gie_id),
                        int(self._pose_anchor_model_size[0]),
                        int(self._pose_anchor_model_size[1]),
                        float(self._pose_anchor_score_threshold),
                        bool(self._pose_anchor_letterbox),
                    )
                except Exception:
                    payload = None
                if isinstance(payload, dict):
                    keypoints_abs = self._keypoints_abs_from_pose_payload(payload, bbox)
                    if keypoints_abs is not None:
                        _increment_core_counter("detection_wake.pose_anchor_native_extract")
                        return keypoints_abs

        _increment_core_counter("detection_wake.pose_anchor_missing")
        return None

    def _keypoints_abs_from_pose_payload(
        self,
        payload: Mapping[str, Any],
        bbox: Sequence[float],
    ) -> Optional[np.ndarray]:
        raw_abs = payload.get("keypoints_abs")
        if isinstance(raw_abs, (list, tuple)) and len(raw_abs) >= 17:
            rows_abs: List[List[float]] = []
            for item in raw_abs[:17]:
                if not isinstance(item, (list, tuple)) or len(item) < 3:
                    return None
                try:
                    rows_abs.append([float(item[0]), float(item[1]), float(item[2])])
                except Exception:
                    return None
            try:
                arr_abs = np.asarray(rows_abs, dtype=np.float32)
            except Exception:
                return None
            if arr_abs.shape == (17, 3):
                return arr_abs

        if len(bbox) < 4:
            return None
        try:
            dst_w = float(bbox[2])
            dst_h = float(bbox[3])
            dst_x = float(bbox[0])
            dst_y = float(bbox[1])
        except Exception:
            return None

        src_w = dst_w
        src_h = dst_h
        src_bbox = payload.get("bbox")
        if isinstance(src_bbox, (list, tuple)) and len(src_bbox) >= 4:
            try:
                src_w = float(src_bbox[2])
                src_h = float(src_bbox[3])
            except Exception:
                src_w = dst_w
                src_h = dst_h
        sx = float(dst_w / src_w) if src_w > 1e-6 else 1.0
        sy = float(dst_h / src_h) if src_h > 1e-6 else 1.0

        raw_roi = payload.get("keypoints_roi")
        if not isinstance(raw_roi, (list, tuple)) or len(raw_roi) < 17:
            return None
        rows_roi: List[List[float]] = []
        for item in raw_roi[:17]:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                return None
            try:
                x = float(item[0]) * sx + dst_x
                y = float(item[1]) * sy + dst_y
                c = float(item[2])
                rows_roi.append([x, y, c])
            except Exception:
                return None
        try:
            arr_roi = np.asarray(rows_roi, dtype=np.float32)
        except Exception:
            return None
        if arr_roi.shape != (17, 3):
            return None
        return arr_roi

    def _pose_point(self, kpts_abs: np.ndarray, name: str) -> Optional[Tuple[float, float]]:
        idx = _POSE_KPT_INDEX.get(name)
        if idx is None or idx < 0 or idx >= int(kpts_abs.shape[0]):
            return None
        try:
            x = float(kpts_abs[idx, 0])
            y = float(kpts_abs[idx, 1])
            conf = float(kpts_abs[idx, 2])
        except Exception:
            return None
        if conf < float(self._pose_anchor_kpt_threshold):
            return None
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        return float(x), float(y)

    def _estimate_ankle_from_leg(self, kpts_abs: np.ndarray, side: str) -> Optional[Tuple[float, float]]:
        hip = self._pose_point(kpts_abs, f"{side}_hip")
        knee = self._pose_point(kpts_abs, f"{side}_knee")
        if hip is None or knee is None:
            return None
        ankle_x = (2.0 * float(knee[0])) - float(hip[0])
        ankle_y = (2.0 * float(knee[1])) - float(hip[1])
        if not (math.isfinite(ankle_x) and math.isfinite(ankle_y)):
            return None
        return float(ankle_x), float(ankle_y)

    def _resolve_pose_floor_anchor(
        self,
        kpts_abs: np.ndarray,
        *,
        posture: str = "unknown",
    ) -> Optional[_PoseAnchorCandidate]:
        return resolve_pose_floor_anchor(
            kpts_abs,
            posture=str(posture or "unknown"),
            config=self._human_ground_cfg,
        )

    def _resolve_person_depth_anchor(self, depth_result: Optional[ObjectDepthResult]) -> Optional[_PoseAnchorCandidate]:
        if depth_result is None or str(depth_result.status) != "ok":
            return None
        sampling_mode = str(depth_result.sampling_mode or "").strip().lower()
        if sampling_mode.startswith("bbox_"):
            return None
        evidence_quality = str(depth_result.evidence_quality or "").strip().lower()
        if evidence_quality == "rejected":
            return None
        anchor_band = str(depth_result.anchor_source or "")
        if anchor_band not in ("lower_body_band", "torso_core", "pose_ankle_support"):
            return None
        anchor_depth_m = depth_result.anchor_depth_m
        if (
            anchor_depth_m is None
            or not math.isfinite(float(anchor_depth_m))
            or float(anchor_depth_m) <= 0.0
        ):
            return None
        anchor_uv = _depth_anchor_uv(depth_result)
        if anchor_uv is None:
            return None
        quality = (
            "good"
            if anchor_band in ("lower_body_band", "pose_ankle_support")
            and evidence_quality != "estimated"
            else "estimated"
        )
        quality_reason = (
            f"depth_anchor={anchor_band},sampling={sampling_mode or 'unknown'},"
            f"evidence={evidence_quality or 'legacy'}"
        )
        return _PoseAnchorCandidate(
            u=float(anchor_uv[0]),
            v=float(anchor_uv[1]),
            source="person_mask_floor",
            contact_basis=f"depth:{anchor_band}",
            quality=quality,
            quality_reason=quality_reason,
            height_lock_eligible=False,
        )

    @staticmethod
    def _anchor_is_verified_ground_contact(
        anchor: Optional[_PoseAnchorCandidate],
    ) -> bool:
        """Return whether an anchor is safe to use for a floor ray.

        A depth anchor can be a useful range observation without identifying
        the person's ground contact.  In particular, ``torso_core`` is a
        body-depth sample and must never be projected to the floor as if it
        were an ankle.  Only observed lower-body/ankle support (or the pose
        leg-floor construction, which has its own admission checks) may
        generate a floor-ray hypothesis.
        """

        if anchor is None:
            return False
        source = str(anchor.source or "").strip().lower()
        basis = str(anchor.contact_basis or "").strip().lower()
        if source in {
            "pose_ankle_floor",
            "pose_single_ankle_floor",
            "pose_ankle_support",
            "pose_leg_floor",
        }:
            return True
        if source != "person_mask_floor":
            return False
        return basis in {
            "depth:lower_body_band",
            "depth:pose_ankle_support",
        }

    @staticmethod
    def _depth_measurement_is_current(
        depth_result: Optional[ObjectDepthResult],
        *,
        track: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        """Return whether metric depth belongs to this exact object cohort.

        The depth result carries its own frame metadata, but that metadata is
        not sufficient when a cached object-depth row is attached to a new
        tracker row.  Require the source, tracker/object id, frame id, and
        source media timestamp to match the current track whenever those
        values are available.
        """
        if depth_result is None or depth_result.measurement_cached is True:
            return False
        if (
            depth_result.measurement_age_us is not None
            and int(depth_result.measurement_age_us) > 0
        ):
            return False
        if (
            depth_result.measurement_frame_id is not None
            and int(depth_result.measurement_frame_id) != int(depth_result.frame_id)
        ):
            return False
        if (
            depth_result.measurement_ts_us is not None
            and depth_result.ts_us is not None
            and int(depth_result.measurement_ts_us) != int(depth_result.ts_us)
        ):
            return False
        if (
            depth_result.depth_tensor_age_frames is not None
            and int(depth_result.depth_tensor_age_frames) > 0
        ):
            return False
        if (
            depth_result.depth_tensor_age_us is not None
            and int(depth_result.depth_tensor_age_us) > 0
        ):
            return False
        measurement_frame_id = (
            int(depth_result.measurement_frame_id)
            if depth_result.measurement_frame_id is not None
            else int(depth_result.frame_id)
        )
        if (
            depth_result.depth_tensor_frame_id is not None
            and int(depth_result.depth_tensor_frame_id) != measurement_frame_id
        ):
            return False
        measurement_ts_us = (
            int(depth_result.measurement_ts_us)
            if depth_result.measurement_ts_us is not None
            else (
                int(depth_result.ts_us)
                if depth_result.ts_us is not None
                else None
            )
        )
        if (
            depth_result.depth_tensor_ts_us is not None
            and measurement_ts_us is not None
            and int(depth_result.depth_tensor_ts_us) != measurement_ts_us
        ):
            return False
        if track is None:
            return True
        try:
            source_id = track.get("source_id")
            if source_id is not None and int(depth_result.source_id) != int(source_id):
                return False
            tracker_id = track.get("tracker_id", track.get("track_id"))
            if tracker_id is not None and int(depth_result.object_id) != int(tracker_id):
                return False
            frame_id = track.get("frame_id")
            if frame_id is not None and int(depth_result.frame_id) != int(frame_id):
                return False
            media_pts_ns = track.get("media_pts_ns")
            if media_pts_ns is not None and int(media_pts_ns) > 0:
                expected_ts_us = int(media_pts_ns) // 1_000
                if depth_result.ts_us is None or int(depth_result.ts_us) != expected_ts_us:
                    return False
                if (
                    depth_result.measurement_ts_us is None
                    or int(depth_result.measurement_ts_us) != expected_ts_us
                ):
                    return False
        except (TypeError, ValueError, OverflowError):
            return False
        return True

    @staticmethod
    def _scene_per_meter(calib: Any) -> float:
        return 1.0

    def _human_height_scene_bounds(self, calib: Any) -> Tuple[float, float]:
        scene_per_m = self._scene_per_meter(calib)
        return (
            float(self._world_height_min_m) * float(scene_per_m),
            float(self._world_height_max_m) * float(scene_per_m),
        )

    @staticmethod
    def _normalize_image_size(value: Any) -> Optional[Tuple[int, int]]:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return None
        try:
            width = int(value[0])
            height = int(value[1])
        except Exception:
            return None
        if width <= 8 or height <= 8:
            return None
        return width, height

    @staticmethod
    def _scale_uv_to_image_size(
        u: float,
        v: float,
        source_size: Optional[Tuple[int, int]],
        dest_size: Optional[Tuple[int, int]],
    ) -> Tuple[float, float]:
        if source_size is None or dest_size is None:
            return float(u), float(v)
        src_w, src_h = source_size
        dst_w, dst_h = dest_size
        if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
            return float(u), float(v)
        return float(u) * (float(dst_w) / float(src_w)), float(v) * (float(dst_h) / float(src_h))

    def _scale_bbox_to_image_size(
        self,
        bbox: Sequence[float],
        source_size: Optional[Tuple[int, int]],
        dest_size: Optional[Tuple[int, int]],
    ) -> Optional[List[float]]:
        if len(bbox) < 4:
            return None
        try:
            left, top, width, height = [float(x) for x in bbox[:4]]
        except Exception:
            return None
        if source_size is None or dest_size is None:
            return [left, top, width, height]
        src_w, src_h = source_size
        dst_w, dst_h = dest_size
        if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
            return [left, top, width, height]
        sx = float(dst_w) / float(src_w)
        sy = float(dst_h) / float(src_h)
        return [left * sx, top * sy, width * sx, height * sy]

    @staticmethod
    def _scale_pose_keypoints_to_image_size(
        keypoints: Optional[np.ndarray],
        source_size: Optional[Tuple[int, int]],
        dest_size: Optional[Tuple[int, int]],
    ) -> Optional[np.ndarray]:
        if keypoints is None:
            return None
        points = np.asarray(keypoints, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] < 2:
            return points
        if source_size is None or dest_size is None or source_size == dest_size:
            return points
        src_w, src_h = source_size
        dst_w, dst_h = dest_size
        if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
            return points
        scaled = points.copy()
        scaled[:, 0] *= float(dst_w) / float(src_w)
        scaled[:, 1] *= float(dst_h) / float(src_h)
        return scaled

    def _maybe_update_world_height_reference(
        self,
        state: _WorldAnchorState,
        calib: Any,
        bbox: Sequence[float],
        foot_world: Sequence[float],
        *,
        flip_u: bool,
        flip_v: bool,
        pose_kpts_abs: Optional[np.ndarray] = None,
    ) -> None:
        if len(bbox) < 4:
            return
        try:
            left, top, width, _height = [float(x) for x in bbox[:4]]
        except Exception:
            return
        if width <= 0.0:
            return
        u_top = float(left) + float(width) * 0.5
        v_top = float(top)
        try:
            est_height = estimate_upright_height_from_top_and_foot(
                u_top,
                v_top,
                foot_world,
                calib.intrinsics,
                calib.extrinsics_col_major,
                float(calib.floor_y),
                tuple(int(x) for x in calib.image_size),
                unit_scale=1.0,
                flip_u=bool(flip_u),
                flip_v=bool(flip_v),
            )
        except Exception:
            est_height = None
        if est_height is None or not math.isfinite(float(est_height)) or float(est_height) <= 0.0:
            return
        min_height, max_height = self._human_height_scene_bounds(calib)
        if float(est_height) < float(min_height) or float(est_height) > float(max_height):
            return
        if state.height_ref_scene is None:
            state.height_ref_scene = float(est_height)
        else:
            alpha = float(self._world_height_update_alpha)
            state.height_ref_scene = float(
                state.height_ref_scene
                + alpha * (float(est_height) - float(state.height_ref_scene))
            )
        self._maybe_update_body_plane_height_fractions(
            state,
            calib,
            pose_kpts_abs,
            foot_world,
            flip_u=flip_u,
            flip_v=flip_v,
        )

    def _maybe_update_body_plane_height_fractions(
        self,
        state: _WorldAnchorState,
        calib: Any,
        pose_kpts_abs: Optional[np.ndarray],
        foot_world: Sequence[float],
        *,
        flip_u: bool,
        flip_v: bool,
    ) -> None:
        """Learn this track's visible-body planes from trusted full-body frames."""
        if (
            pose_kpts_abs is None
            or state.height_ref_scene is None
            or float(state.height_ref_scene) <= 1e-6
            or len(foot_world) < 3
        ):
            return

        def _midpoint(
            left_name: str,
            right_name: str,
        ) -> Optional[Tuple[float, float]]:
            left_point = self._pose_point(pose_kpts_abs, left_name)
            right_point = self._pose_point(pose_kpts_abs, right_name)
            if left_point is None and right_point is None:
                return None
            if left_point is None:
                return right_point
            if right_point is None:
                return left_point
            return (
                (float(left_point[0]) + float(right_point[0])) * 0.5,
                (float(left_point[1]) + float(right_point[1])) * 0.5,
            )

        references: Tuple[
            Tuple[str, Optional[Tuple[float, float]], float, float],
            ...,
        ] = (
            ("nose", self._pose_point(pose_kpts_abs, "nose"), 0.72, 1.08),
            (
                "shoulders",
                _midpoint("left_shoulder", "right_shoulder"),
                0.55,
                0.96,
            ),
            ("hips", _midpoint("left_hip", "right_hip"), 0.30, 0.76),
        )
        try:
            width_src, height_src = calib.image_size
            R_wc, C_world = parse_extrinsics(calib.extrinsics_col_major)
            C_world = C_world * self._scene_per_meter(calib)
            foot_x = float(foot_world[0])
            foot_z = float(foot_world[2])
            floor_y = float(calib.floor_y)
            height_ref = float(state.height_ref_scene)
        except Exception:
            return

        alpha = float(self._human_ground_cfg.upright_reference_alpha)
        for label, uv, min_fraction, max_fraction in references:
            if uv is None:
                continue
            try:
                u_ray, v_ray = self._apply_image_flip(
                    float(uv[0]),
                    float(uv[1]),
                    int(width_src),
                    int(height_src),
                    bool(flip_u),
                    bool(flip_v),
                )
                origin, direction = ray_from_pixel(
                    u_ray,
                    v_ray,
                    calib.intrinsics,
                    R_wc,
                    C_world,
                )
                denom_xz = float(direction[0]) ** 2 + float(direction[2]) ** 2
                if denom_xz <= 1e-12:
                    continue
                ray_t = (
                    (foot_x - float(origin[0])) * float(direction[0])
                    + (foot_z - float(origin[2])) * float(direction[2])
                ) / denom_xz
                if not math.isfinite(ray_t) or ray_t <= 0.0:
                    continue
                point_y = float(origin[1]) + ray_t * float(direction[1])
                fraction = (point_y - floor_y) / height_ref
                if (
                    not math.isfinite(fraction)
                    or fraction < float(min_fraction)
                    or fraction > float(max_fraction)
                ):
                    continue
                previous = state.body_plane_height_fractions.get(label)
                if previous is None or not math.isfinite(float(previous)):
                    state.body_plane_height_fractions[label] = float(fraction)
                else:
                    state.body_plane_height_fractions[label] = float(
                        float(previous)
                        + alpha * (float(fraction) - float(previous))
                    )
            except Exception:
                continue

    def _reference_plane_floor_world(
        self,
        calib: Any,
        u: float,
        v: float,
        plane_height_scene: float,
        *,
        flip_u: bool,
        flip_v: bool,
    ) -> Optional[np.ndarray]:
        try:
            width_src, height_src = calib.image_size
            u_ray, v_ray = self._apply_image_flip(
                float(u),
                float(v),
                int(width_src),
                int(height_src),
                bool(flip_u),
                bool(flip_v),
            )
            R_wc, C_world = parse_extrinsics(calib.extrinsics_col_major)
            C_world = C_world * self._scene_per_meter(calib)
            origin, direction = ray_from_pixel(
                u_ray,
                v_ray,
                calib.intrinsics,
                R_wc,
                C_world,
            )
            denom = float(direction[1])
            if abs(denom) < 1e-9:
                return None
            plane_y = float(calib.floor_y) + float(plane_height_scene)
            ray_t = (plane_y - float(origin[1])) / denom
            if not math.isfinite(ray_t) or ray_t <= 0.0:
                return None
            reference_point = origin + (direction * ray_t)
            return np.array(
                [
                    float(reference_point[0]),
                    float(calib.floor_y),
                    float(reference_point[2]),
                ],
                dtype=np.float64,
            )
        except Exception:
            return None

    def _gravity_drop_world(
        self,
        calib: Any,
        bbox: Sequence[float],
        height_ref_scene: float,
        *,
        flip_u: bool,
        flip_v: bool,
        pose_kpts_abs: Optional[np.ndarray] = None,
        state: Optional[_WorldAnchorState] = None,
    ) -> Optional[np.ndarray]:
        if len(bbox) < 4:
            return None
        try:
            left, top, width, _height = [float(x) for x in bbox[:4]]
        except Exception:
            return None
        if width <= 0.0 or float(height_ref_scene) <= 0.0:
            return None
        candidates: List[Tuple[np.ndarray, float]] = []

        def _append_candidate(
            uv: Optional[Tuple[float, float]],
            *,
            height_fraction: float,
            weight: float,
        ) -> None:
            if uv is None:
                return
            point = self._reference_plane_floor_world(
                calib,
                float(uv[0]),
                float(uv[1]),
                float(height_ref_scene) * float(height_fraction),
                flip_u=flip_u,
                flip_v=flip_v,
            )
            if point is not None:
                candidates.append((point, float(weight)))

        _append_candidate(
            (float(left) + float(width) * 0.5, float(top)),
            height_fraction=1.0,
            weight=2.0,
        )

        if (
            pose_kpts_abs is not None
            and isinstance(pose_kpts_abs, np.ndarray)
            and pose_kpts_abs.shape[0] >= 17
        ):
            fractions = (
                state.body_plane_height_fractions
                if state is not None
                else {}
            )

            def _midpoint(
                left_name: str,
                right_name: str,
            ) -> Optional[Tuple[float, float]]:
                left_point = self._pose_point(pose_kpts_abs, left_name)
                right_point = self._pose_point(pose_kpts_abs, right_name)
                if left_point is None and right_point is None:
                    return None
                if left_point is None:
                    return right_point
                if right_point is None:
                    return left_point
                return (
                    (float(left_point[0]) + float(right_point[0])) * 0.5,
                    (float(left_point[1]) + float(right_point[1])) * 0.5,
                )

            _append_candidate(
                self._pose_point(pose_kpts_abs, "nose"),
                height_fraction=float(fractions.get("nose", 0.94)),
                weight=0.75,
            )
            _append_candidate(
                _midpoint("left_shoulder", "right_shoulder"),
                height_fraction=float(fractions.get("shoulders", 0.82)),
                weight=1.35,
            )
            _append_candidate(
                _midpoint("left_hip", "right_hip"),
                height_fraction=float(fractions.get("hips", 0.55)),
                weight=0.55,
            )

        if not candidates:
            return None
        if len(candidates) == 1:
            return np.asarray(candidates[0][0], dtype=np.float64)

        primary = np.asarray(candidates[0][0], dtype=np.float64)
        inliers: List[Tuple[np.ndarray, float]] = [candidates[0]]
        for point, weight in candidates[1:]:
            disagreement = math.hypot(
                float(point[0]) - float(primary[0]),
                float(point[2]) - float(primary[2]),
            )
            if math.isfinite(disagreement) and disagreement <= 0.75:
                inliers.append((point, weight))
        total_weight = sum(float(weight) for _point, weight in inliers)
        if total_weight <= 1e-9:
            return primary
        world_x = sum(
            float(point[0]) * float(weight) for point, weight in inliers
        ) / total_weight
        world_z = sum(
            float(point[2]) * float(weight) for point, weight in inliers
        ) / total_weight
        return np.array(
            [float(world_x), float(calib.floor_y), float(world_z)],
            dtype=np.float64,
        )

    def _fallback_quality_reason(
        self,
        pose_kpts_abs: Optional[np.ndarray],
        pose_anchor: Optional[_PoseAnchorCandidate],
        person_anchor: Optional[_PoseAnchorCandidate],
        depth_result: Optional[ObjectDepthResult],
        state: Optional[_WorldAnchorState],
    ) -> str:
        reasons: List[str] = []
        if noesis_pose_meta_ext is None:
            reasons.append("pose_meta_missing")
        elif pose_kpts_abs is None:
            reasons.append("pose_keypoints_unusable")
        elif pose_anchor is None:
            reasons.append("pose_anchor_unavailable")
        if person_anchor is None:
            if depth_result is None:
                reasons.append("depth_meta_missing")
            elif _depth_anchor_uv(depth_result) is None:
                reasons.append("depth_anchor_unavailable")
        if state is None or state.height_ref_scene is None:
            reasons.append("height_lock_missing")
        if not reasons:
            return "world_anchor_projection_failed"
        return ",".join(dict.fromkeys(str(reason) for reason in reasons if reason))

    def _extract_object_depth_result(self, obj_meta: Any) -> Optional[ObjectDepthResult]:
        return _extract_object_depth_result_from_meta(obj_meta)

    def _project_pixel_to_world_observation(
        self,
        calib: Any,
        u: float,
        v: float,
        *,
        depth_m: Optional[float],
        flip_u: bool,
        flip_v: bool,
    ) -> Optional[np.ndarray]:
        try:
            width_src, height_src = calib.image_size
            u_ray, v_ray = self._apply_image_flip(float(u), float(v), int(width_src), int(height_src), bool(flip_u), bool(flip_v))
            result = pixel_to_world(
                calib.intrinsics,
                calib.extrinsics_col_major,
                float(calib.floor_y),
                float(getattr(calib, "unit_scale", 1.0) or 1.0),
                float(u_ray),
                float(v_ray),
                depth_m=float(depth_m) if depth_m is not None else None,
            )
            if not bool(getattr(result, "ok", False)):
                return None
            point = getattr(result, "world_point", None)
            if not isinstance(point, Sequence) or len(point) < 3:
                return None
            return np.array([float(point[0]), float(point[1]), float(point[2])], dtype=np.float64)
        except Exception:
            return None

    def _project_pixel_to_floor_world(
        self,
        calib: Any,
        u: float,
        v: float,
        *,
        flip_u: bool,
        flip_v: bool,
    ) -> Optional[np.ndarray]:
        return self._project_pixel_to_world_observation(
            calib,
            u,
            v,
            depth_m=None,
            flip_u=flip_u,
            flip_v=flip_v,
        )

    def _depth_observation_from_anchor(
        self,
        *,
        calib: Any,
        anchor: _PoseAnchorCandidate,
        depth_result: Optional[ObjectDepthResult],
        flip_u: bool,
        flip_v: bool,
        track: Optional[Mapping[str, Any]] = None,
    ) -> _DepthObservationResult:
        if depth_result is None:
            return _DepthObservationResult(None, 0.0, "depth_meta_missing")
        if str(depth_result.status) != "ok":
            return _DepthObservationResult(None, 0.0, f"depth_status_{depth_result.status}")
        if not bool(depth_result.is_metric) or str(depth_result.unit) != "m":
            return _DepthObservationResult(None, 0.0, "depth_not_metric")
        anchor_depth_m = depth_result.anchor_depth_m
        raw_depth_value = (
            float(anchor_depth_m)
            if anchor_depth_m is not None
            and math.isfinite(float(anchor_depth_m))
            and float(anchor_depth_m) > 0.0
            else None
        )
        if not self._depth_measurement_is_current(depth_result, track=track):
            # Cached range remains available through the public depth
            # provenance fields, but it is not a current-frame metric
            # observation and must never update/reacquire the world filter.
            return _DepthObservationResult(
                None,
                0.0,
                "depth_measurement_not_current",
                raw_depth_m=raw_depth_value,
            )
        anchor_source = str(depth_result.anchor_source or "")
        if anchor_source == "lower_body_band":
            min_support_count = 16
            min_support_fraction = 0.40
            support_scale_denom = 96.0
            anchor_source_weight = 1.0
        elif anchor_source == "pose_ankle_support":
            min_support_count = 8
            min_support_fraction = 0.60
            support_scale_denom = 48.0
            anchor_source_weight = 1.0
        elif anchor_source == "torso_core":
            min_support_count = 20
            min_support_fraction = 0.45
            support_scale_denom = 128.0
            anchor_source_weight = 0.50
        else:
            return _DepthObservationResult(None, 0.0, "depth_anchor_source_invalid")
        if anchor_depth_m is None or not math.isfinite(float(anchor_depth_m)) or float(anchor_depth_m) <= 0.0:
            return _DepthObservationResult(None, 0.0, "depth_anchor_missing")
        support_count = int(
            depth_result.anchor_sample_count
            if depth_result.anchor_sample_count is not None
            else depth_result.sample_count
        )
        support_fraction = float(
            depth_result.anchor_valid_fraction
            if depth_result.anchor_valid_fraction is not None
            else depth_result.valid_fraction
        )
        if support_count < min_support_count or support_fraction < min_support_fraction:
            return _DepthObservationResult(
                None,
                0.0,
                "depth_support_low",
                raw_depth_m=float(anchor_depth_m),
            )
        anchor_spread_m = depth_result.anchor_depth_spread_m
        if anchor_spread_m is not None and not _depth_spread_is_supported(
            float(anchor_depth_m),
            float(anchor_spread_m),
            strict=anchor_source == "pose_ankle_support",
        ):
            return _DepthObservationResult(
                None,
                0.0,
                "depth_spread_high",
                raw_depth_m=float(anchor_depth_m),
            )
        raw_depth_value = float(anchor_depth_m)
        registered_depth_m = raw_depth_value
        registration_status = "raw_passthrough"
        registration_id: Optional[str] = None
        if self.depth_registration is not None:
            corrected_depth_m, reg_status, reg_id = self.depth_registration.apply(
                camera_id=str(getattr(calib, "camera_id", "") or ""),
                raw_depth_m=raw_depth_value,
            )
            registration_status = str(reg_status)
            registration_id = str(reg_id) if reg_id else None
            if corrected_depth_m is None:
                return _DepthObservationResult(
                    None,
                    0.0,
                    f"depth_registration_{registration_status}",
                    raw_depth_m=raw_depth_value,
                    registered_depth_m=None,
                    registration_status=registration_status,
                    registration_id=registration_id,
                )
            registered_depth_m = float(corrected_depth_m)
        depth_obs = self._project_pixel_to_world_observation(
            calib,
            float(anchor.u),
            float(anchor.v),
            depth_m=float(registered_depth_m),
            flip_u=flip_u,
            flip_v=flip_v,
        )
        if depth_obs is None:
            return _DepthObservationResult(
                None,
                0.0,
                "depth_projection_failed",
                raw_depth_m=raw_depth_value,
                registered_depth_m=registered_depth_m,
                registration_status=registration_status,
                registration_id=registration_id,
            )
        depth_weight = min(1.0, support_fraction) * min(1.0, float(support_count) / support_scale_denom)
        depth_weight *= anchor_source_weight
        if str(anchor.source) == "pose_leg_floor":
            depth_weight *= 0.85
        return _DepthObservationResult(
            depth_obs,
            max(0.0, min(1.0, depth_weight)),
            "ok",
            raw_depth_m=raw_depth_value,
            registered_depth_m=registered_depth_m,
            registration_status=registration_status,
            registration_id=registration_id,
        )

    @staticmethod
    def _covariance_matrix_from_projection(
        center: Optional[np.ndarray],
        perturbations: Sequence[Tuple[float, Optional[np.ndarray]]],
        *,
        floor_y_variance: float = 0.0025,
    ) -> Optional[Matrix3]:
        """Propagate bounded input uncertainty through a world projection.

        ``perturbations`` contains (input sigma, perturbed world point) pairs.
        The function is intentionally small and allocation-bounded: a floor
        hypothesis uses two UV perturbations and a depth hypothesis uses one
        additional range perturbation.  Failed finite differences simply add
        no information; the caller can then reject the candidate rather than
        inventing a position or covariance.
        """

        if center is None:
            return None
        try:
            center_arr = np.asarray(center, dtype=np.float64).reshape(-1)
        except Exception:
            return None
        if center_arr.size < 3 or not np.all(np.isfinite(center_arr[:3])):
            return None
        covariance = np.zeros((3, 3), dtype=np.float64)
        used = 0
        for sigma_raw, perturbed in perturbations:
            try:
                sigma = float(sigma_raw)
                point = np.asarray(perturbed, dtype=np.float64).reshape(-1)
            except Exception:
                continue
            if (
                not math.isfinite(sigma)
                or sigma <= 1e-6
                or point.size < 3
                or not np.all(np.isfinite(point[:3]))
            ):
                continue
            jacobian = (point[:3] - center_arr[:3]) / sigma
            covariance += np.outer(jacobian, jacobian) * (sigma * sigma)
            used += 1
        # The calibrated floor is not exact.  Keep a conservative bounded
        # vertical variance even though the resolver uses X/Z for display.
        covariance[1, 1] += max(1e-5, min(1.0, float(floor_y_variance)))
        if used == 0:
            return None
        covariance = 0.5 * (covariance + covariance.T)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            eigenvalues = np.clip(eigenvalues, 1e-4, 25.0)
            covariance = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        except Exception:
            return None
        values = tuple(float(value) for value in covariance.reshape(-1))
        try:
            return Matrix3(values=values)  # type: ignore[arg-type]
        except Exception:
            return None

    def _floor_candidate_covariance(
        self,
        calib: Any,
        *,
        anchor_uv: Sequence[float],
        floor_candidate: np.ndarray,
        incidence_sin: float,
        posture: str,
        occlusion_fraction: float,
        flip_u: bool,
        flip_v: bool,
    ) -> Optional[Matrix3]:
        """Estimate floor-ray covariance from image contact uncertainty."""

        try:
            u, v = float(anchor_uv[0]), float(anchor_uv[1])
            incidence = max(0.02, min(1.0, float(incidence_sin)))
        except (TypeError, ValueError, IndexError, OverflowError):
            return None
        sigma_px = 2.0 + 4.0 * (1.0 - incidence)
        sigma_px *= 1.0 + min(1.0, max(0.0, float(occlusion_fraction)))
        if str(posture) in ("sitting", "lying"):
            sigma_px *= 1.5
        sigma_px = max(1.5, min(24.0, float(sigma_px)))
        width, height = self._normalize_image_size(getattr(calib, "image_size", None)) or (0, 0)
        if width <= 0 or height <= 0:
            return None
        def _project(du: float, dv: float) -> Optional[np.ndarray]:
            pu = min(float(width - 1), max(0.0, float(u) + float(du)))
            pv = min(float(height - 1), max(0.0, float(v) + float(dv)))
            return self._project_pixel_to_floor_world(
                calib,
                pu,
                pv,
                flip_u=flip_u,
                flip_v=flip_v,
            )

        return self._covariance_matrix_from_projection(
            floor_candidate,
            ((sigma_px, _project(sigma_px, 0.0)), (sigma_px, _project(0.0, sigma_px))),
            floor_y_variance=0.01 + 0.04 * (1.0 - incidence),
        )

    def _depth_registration_sigma_m(self, calib: Any) -> float:
        """Read robust occupied-anchor residual evidence from registration."""

        manager = self.depth_registration
        if manager is None:
            return 0.15
        try:
            bundle = getattr(manager, "bundle", None)
            entries = getattr(bundle, "entries", None)
            entry = entries.get(str(getattr(calib, "camera_id", ""))) if isinstance(entries, Mapping) else None
            evidence = getattr(entry, "occupied_anchor_validation", None)
            if not isinstance(evidence, Mapping):
                return 0.15
            median = float(evidence.get("median_abs_error_m", 0.0) or 0.0)
            p95 = float(evidence.get("p95_abs_error_m", 0.0) or 0.0)
            if not math.isfinite(median) or not math.isfinite(p95):
                return 0.15
            # Convert robust absolute residuals to a bounded one-sigma floor;
            # retain p95 as a guard against overconfident registered depth.
            return max(0.05, min(2.5, median / 0.6745, p95 / 1.96))
        except Exception:
            return 0.15

    def _depth_candidate_covariance(
        self,
        calib: Any,
        *,
        anchor_uv: Sequence[float],
        registered_depth_m: float,
        depth_result: Optional[ObjectDepthResult],
        depth_candidate: np.ndarray,
        posture: str,
        occlusion_fraction: float,
        flip_u: bool,
        flip_v: bool,
    ) -> Optional[Matrix3]:
        """Propagate UV, depth-spread, and registration uncertainty."""

        try:
            u, v = float(anchor_uv[0]), float(anchor_uv[1])
            range_m = float(registered_depth_m)
        except (TypeError, ValueError, IndexError, OverflowError):
            return None
        if not math.isfinite(range_m) or range_m <= 0.0:
            return None
        support_fraction = float(
            depth_result.anchor_valid_fraction
            if depth_result is not None and depth_result.anchor_valid_fraction is not None
            else (depth_result.valid_fraction if depth_result is not None else 0.0)
        )
        spread = float(
            depth_result.anchor_depth_spread_m
            if depth_result is not None and depth_result.anchor_depth_spread_m is not None
            else 0.0
        )
        if not math.isfinite(support_fraction):
            support_fraction = 0.0
        if not math.isfinite(spread) or spread < 0.0:
            spread = 0.0
        sigma_depth = max(
            self._depth_registration_sigma_m(calib),
            0.05,
            spread / max(0.35, math.sqrt(max(0.01, support_fraction))),
        )
        sigma_depth *= 1.0 + min(1.0, max(0.0, float(occlusion_fraction)))
        if str(posture) in ("sitting", "lying"):
            sigma_depth *= 1.25
        sigma_depth = min(2.5, float(sigma_depth))
        sigma_px = max(1.5, min(16.0, 2.0 + 5.0 * (1.0 - max(0.0, min(1.0, support_fraction)))))
        width, height = self._normalize_image_size(getattr(calib, "image_size", None)) or (0, 0)
        if width <= 0 or height <= 0:
            return None
        def _project(du: float, dv: float, depth_delta: float = 0.0) -> Optional[np.ndarray]:
            pu = min(float(width - 1), max(0.0, float(u) + float(du)))
            pv = min(float(height - 1), max(0.0, float(v) + float(dv)))
            point = self._project_pixel_to_world_observation(
                calib,
                pu,
                pv,
                depth_m=max(0.01, range_m + float(depth_delta)),
                flip_u=flip_u,
                flip_v=flip_v,
            )
            if point is None:
                return None
            grounded = np.asarray(point, dtype=np.float64).copy()
            grounded[1] = float(calib.floor_y)
            return grounded

        return self._covariance_matrix_from_projection(
            depth_candidate,
            (
                (sigma_px, _project(sigma_px, 0.0)),
                (sigma_px, _project(0.0, sigma_px)),
                (sigma_depth, _project(0.0, 0.0, sigma_depth)),
            ),
            floor_y_variance=0.02 + 0.08 * (1.0 - max(0.0, min(1.0, support_fraction))),
        )

    def _append_universal_world_candidates(
        self,
        candidates: List[Dict[str, Any]],
        *,
        sensor_id: int,
        track: Mapping[str, Any],
        camera_id: str,
        calib: Any,
        floor_candidate: Optional[np.ndarray],
        floor_ray_admitted: bool,
        anchor_candidate: Optional[_PoseAnchorCandidate],
        pose_uv: Optional[Tuple[float, float]],
        contact_basis: Optional[str],
        posture: str,
        occlusion_fraction: float,
        image_motion_supported: bool,
        depth_obs: Optional[np.ndarray],
        depth_weight: float,
        depth_observation: _DepthObservationResult,
        depth_result: Optional[ObjectDepthResult],
        person_anchor: Optional[_PoseAnchorCandidate],
        reject_current_geometry: bool,
        flip_u: bool,
        flip_v: bool,
    ) -> None:
        """Append the bounded, camera-agnostic geometric hypotheses.

        This method is deliberately independent of the legacy policy.  Floor
        and registered-depth evidence are admitted separately, and each
        candidate carries its own support semantics and covariance.  A depth
        sample may come from a torso or body band, so its raw ray point is
        retained only as diagnostic data; the hypothesis quantity is always a
        ground footprint at the calibrated floor elevation.
        """

        floor_anchor = (
            anchor_candidate
            if self._anchor_is_verified_ground_contact(anchor_candidate)
            else None
        )
        if floor_candidate is not None and floor_ray_admitted and pose_uv is not None and floor_anchor is not None:
            floor_covariance = self._floor_candidate_covariance(
                calib,
                anchor_uv=pose_uv,
                floor_candidate=floor_candidate,
                incidence_sin=float(track.get("world_floor_incidence_sin", 0.0) or 0.0),
                posture=str(posture),
                occlusion_fraction=float(occlusion_fraction),
                flip_u=flip_u,
                flip_v=flip_v,
            )
            if floor_covariance is not None:
                floor_support_state = (
                    "unknown" if str(posture) in ("sitting", "lying") else "floor"
                )
                candidates.append(
                    {
                        "candidate_id": "floor_ray",
                        "kind": "floor_ray",
                        "position": floor_candidate,
                        "covariance": floor_covariance,
                        "anchor": str(floor_anchor.source),
                        "contact_basis": contact_basis,
                        "support_state": floor_support_state,
                        "confidence": 0.95 if floor_anchor.quality == "good" else 0.75,
                        "posture": str(posture),
                        "occlusion": float(occlusion_fraction),
                        "motion_consistency": 1.0 if image_motion_supported else 0.75,
                        "support_score": 0.95 if floor_anchor.height_lock_eligible else 0.72,
                        "posture_compatibility": 0.75 if posture in ("sitting", "lying") else 1.0,
                        "source_reliability": 1.0,
                        "ray_incidence_sin": float(track.get("world_floor_incidence_sin", 0.0) or 0.0),
                        "pixel_uncertainty_px": 2.0,
                        "correlation_group": f"floor_ray:{int(sensor_id)}:{int(track.get('tracker_id', -1))}",
                        "pcf": self._world_prior_evidence(camera_id, floor_candidate),
                    }
                )

        if (
            not reject_current_geometry
            and depth_obs is not None
            and float(depth_weight) > 0.0
            and person_anchor is not None
            and depth_observation.registered_depth_m is not None
        ):
            depth_position = np.asarray(depth_obs, dtype=np.float64).copy()
            try:
                depth_position[1] = float(calib.floor_y)
            except (TypeError, ValueError, IndexError, OverflowError):
                return
            depth_covariance = self._depth_candidate_covariance(
                calib,
                anchor_uv=(float(person_anchor.u), float(person_anchor.v)),
                registered_depth_m=float(depth_observation.registered_depth_m),
                depth_result=depth_result,
                depth_candidate=depth_position,
                posture=str(posture),
                occlusion_fraction=float(occlusion_fraction),
                flip_u=flip_u,
                flip_v=flip_v,
            )
            if depth_covariance is None:
                return
            depth_support_fraction = float(
                depth_result.anchor_valid_fraction
                if depth_result is not None and depth_result.anchor_valid_fraction is not None
                else (depth_result.valid_fraction if depth_result is not None else 0.0)
            )
            depth_support_fraction = max(0.0, min(1.0, depth_support_fraction))
            depth_support_state = (
                "floor"
                if str(posture) in ("standing", "unknown")
                and str(depth_observation.registration_status or "") in ("ok", "raw_passthrough")
                and str(depth_result.anchor_source if depth_result is not None else "")
                in ("lower_body_band", "pose_ankle_support")
                else "unknown"
            )
            candidates.append(
                {
                    "candidate_id": "registered_depth",
                    "kind": "registered_depth",
                    "position": depth_position,
                    "covariance": depth_covariance,
                    "anchor": str(person_anchor.source),
                    "contact_basis": str(person_anchor.contact_basis or person_anchor.source),
                    "support_state": depth_support_state,
                    "confidence": max(0.35, min(1.0, 0.45 + 0.55 * float(depth_weight))),
                    "posture": str(posture),
                    "occlusion": float(occlusion_fraction),
                    "motion_consistency": 1.0 if image_motion_supported else 0.75,
                    "support_score": depth_support_fraction,
                    "posture_compatibility": 0.80 if posture in ("sitting", "lying") else 1.0,
                    "source_reliability": 1.0,
                    "depth_support_fraction": depth_support_fraction,
                    # This is registration uncertainty, not the measured
                    # disagreement between the registered-depth and floor-ray
                    # hypotheses. Keep the two quantities distinct.
                    "depth_registration_sigma_m": self._depth_registration_sigma_m(calib),
                    "floor_depth_disagreement_m": (
                        float(
                            math.hypot(
                                float(depth_position[0]) - float(floor_candidate[0]),
                                float(depth_position[2]) - float(floor_candidate[2]),
                            )
                        )
                        if floor_candidate is not None
                        and np.asarray(floor_candidate).reshape(-1).size >= 3
                        and np.all(np.isfinite(np.asarray(floor_candidate).reshape(-1)[:3]))
                        else None
                    ),
                    "correlation_group": f"registered_depth:{int(sensor_id)}:{int(track.get('tracker_id', -1))}",
                    "pcf": self._world_prior_evidence(camera_id, depth_position),
                }
            )

    def _world_prior_evidence(
        self,
        camera_id: str,
        position: Sequence[float],
    ) -> Optional[WorldPriorEvidence]:
        """Convert revision-matched Scene Prior evidence into the contract."""

        priors = self.scene_priors
        if priors is None:
            return None
        revision = priors.revision_for_camera(str(camera_id))
        if revision is None:
            return None
        frame_binding = priors.frame_binding(str(camera_id))
        revision_id = str(
            getattr(frame_binding, "target_revision_id", None)
            or revision.manifest.prior_id
        ).strip()
        if not revision_id:
            return None
        try:
            diagnostic = priors.evaluate(str(camera_id), position)
        except ScenePriorError:
            return None
        if not isinstance(diagnostic, Mapping):
            return None
        try:
            observed_confidence = float(diagnostic.get("evidence_confidence", 0.0) or 0.0)
            boundary = diagnostic.get("boundary_signed_distance_m")
            extent_outside_distance = diagnostic.get("extent_outside_distance_m")
            floor_height = diagnostic.get("floor_height_m")
            obstacle_clearance = diagnostic.get("obstacle_signed_clearance_m")
            return WorldPriorEvidence(
                prior_id=str(diagnostic.get("prior_id") or revision.manifest.prior_id),
                revision_id=revision_id,
                status=str(diagnostic.get("status") or "unknown"),
                inside_extent=(
                    bool(diagnostic.get("inside_extent"))
                    if diagnostic.get("inside_extent") is not None
                    else None
                ),
                inside_authored_space=(
                    bool(diagnostic.get("inside_authored_space"))
                    if diagnostic.get("inside_authored_space") is not None
                    else None
                ),
                extent_outside_distance_m=(
                    float(extent_outside_distance)
                    if extent_outside_distance is not None
                    else None
                ),
                evidence_observed=bool(diagnostic.get("evidence_observed", False)),
                observed_confidence=max(0.0, min(1.0, observed_confidence)),
                boundary_signed_distance_m=(
                    float(boundary) if boundary is not None else None
                ),
                floor_height_m=(float(floor_height) if floor_height is not None else None),
                obstacle_clearance_m=(
                    float(obstacle_clearance) if obstacle_clearance is not None else None
                ),
                reasons=tuple(str(reason)[:160] for reason in (diagnostic.get("reasons") or ())[:8]),
            )
        except (TypeError, ValueError, OverflowError):
            return None

    def _world_measurement_cohort(
        self,
        sensor_id: int,
        camera_id: str,
        track: Mapping[str, Any],
        *,
        calib: Any,
        world_frame_revision: Optional[str],
        world_transform_sha256: Optional[str],
    ) -> Optional[WorldMeasurementCohort]:
        """Build one exact cohort identity for all current-frame hypotheses."""

        try:
            tracker_id = int(track.get("tracker_id", track.get("track_id")))
            frame_id = int(track.get("frame_id", -1))
        except (TypeError, ValueError):
            return None
        if tracker_id < 0 or frame_id < 0:
            return None
        world_revision = str(world_frame_revision or "").strip()
        calibration_revision = str(
            getattr(calib, "camera_calibration_sha256", None) or ""
        ).strip()
        transform_sha256 = str(world_transform_sha256 or "").strip()
        if not world_revision or not calibration_revision or not transform_sha256:
            return None
        try:
            observed_at_us = int(track.get("observed_at_us"))
            source_contract_id = int(track.get("source_id"))
        except (TypeError, ValueError):
            return None
        if observed_at_us <= 0 or source_contract_id < 0:
            return None
        raw_generation = track.get("tracker_lifecycle_generation")
        if (
            isinstance(raw_generation, bool)
            or not isinstance(raw_generation, int)
            or int(raw_generation) < 0
        ):
            # A canonical cohort must never synthesize lifecycle generation 0
            # for a row that did not carry the tracker registry stamp.
            return None
        generation = int(raw_generation)
        track_key_value = str(track.get("track_key") or "").strip()
        if not track_key_value:
            track_key_value = f"{source_contract_id}:{tracker_id}:{generation}"
        pcf_revision: Optional[str] = None
        if self.scene_priors is not None:
            revision = self.scene_priors.revision_for_camera(str(camera_id))
            binding = self.scene_priors.frame_binding(str(camera_id))
            pcf_revision = str(
                getattr(binding, "target_revision_id", None)
                or (revision.manifest.prior_id if revision is not None else "")
            ).strip() or None
        try:
            return WorldMeasurementCohort(
                track_key=track_key_value,
                tracker_lifecycle_generation=generation,
                camera_id=str(camera_id),
                source_id=source_contract_id,
                tracker_id=tracker_id,
                frame_id=frame_id,
                observed_at_us=observed_at_us,
                world_frame="backend_world_m",
                world_revision=world_revision,
                calibration_revision=calibration_revision,
                world_transform_sha256=transform_sha256,
                pcf_revision=pcf_revision,
            )
        except Exception:
            return None

    def _build_world_measurement_set(
        self,
        *,
        cohort: Optional[WorldMeasurementCohort],
        candidates: Sequence[Mapping[str, Any]],
        continuation: Any = None,
    ) -> Optional[WorldMeasurementSet]:
        """Validate and bound camera-local hypotheses before resolver entry."""

        if cohort is None:
            return None
        hypotheses: List[WorldMeasurementHypothesis] = []
        for candidate in candidates[: int(self._world_resolver_diag_max_candidates)]:
            position = candidate.get("position")
            covariance = candidate.get("covariance")
            try:
                position_values = np.asarray(position, dtype=np.float64).reshape(-1)
            except Exception:
                continue
            if position_values.size < 3 or not np.all(np.isfinite(position_values[:3])):
                continue
            if not isinstance(covariance, Matrix3):
                continue
            try:
                point = Vector3(
                    x=float(position_values[0]),
                    y=float(position_values[1]),
                    z=float(position_values[2]),
                )
                hypothesis = WorldMeasurementHypothesis(
                    candidate_id=str(candidate.get("candidate_id") or candidate.get("kind") or "candidate"),
                    cohort=cohort,
                    kind=str(candidate.get("kind") or "floor_ray"),  # type: ignore[arg-type]
                    position=point,
                    covariance=covariance,
                    anchor=str(candidate.get("anchor") or candidate.get("contact_basis") or "unknown"),
                    confidence=max(0.0, min(1.0, float(candidate.get("confidence", 0.5) or 0.5))),
                    posture=str(candidate.get("posture") or "unknown"),  # type: ignore[arg-type]
                    support_state=str(candidate.get("support_state") or "unknown"),  # type: ignore[arg-type]
                    occlusion=max(0.0, min(1.0, float(candidate.get("occlusion", 0.0) or 0.0))),
                    motion_consistency=max(0.0, min(1.0, float(candidate.get("motion_consistency", 1.0) or 1.0))),
                    support_score=max(0.0, min(1.0, float(candidate.get("support_score", 1.0) or 1.0))),
                    posture_compatibility=max(0.0, min(1.0, float(candidate.get("posture_compatibility", 1.0) or 1.0))),
                    source_reliability=max(0.0, min(1.0, float(candidate.get("source_reliability", 1.0) or 1.0))),
                    ray_incidence_sin=(
                        float(candidate["ray_incidence_sin"])
                        if candidate.get("ray_incidence_sin") is not None
                        else None
                    ),
                    depth_support_fraction=(
                        float(candidate["depth_support_fraction"])
                        if candidate.get("depth_support_fraction") is not None
                        else None
                    ),
                    depth_registration_sigma_m=(
                        float(candidate["depth_registration_sigma_m"])
                        if candidate.get("depth_registration_sigma_m") is not None
                        else None
                    ),
                    floor_depth_disagreement_m=(
                        float(candidate["floor_depth_disagreement_m"])
                        if candidate.get("floor_depth_disagreement_m") is not None
                        else None
                    ),
                    pixel_uncertainty_px=(
                        float(candidate["pixel_uncertainty_px"])
                        if candidate.get("pixel_uncertainty_px") is not None
                        else None
                    ),
                    correlation_group=(
                        str(candidate["correlation_group"])
                        if candidate.get("correlation_group")
                        else None
                    ),
                    pcf=(
                        candidate.get("pcf")
                        if isinstance(candidate.get("pcf"), WorldPriorEvidence)
                        else None
                    ),
                    valid=bool(candidate.get("valid", True)),
                    rejection_reason=(
                        str(candidate["rejection_reason"])
                        if candidate.get("rejection_reason")
                        else None
                    ),
                )
            except Exception:
                # Contract construction is the boundary: malformed evidence
                # cannot enter the resolver or become dashboard telemetry.
                continue
            hypotheses.append(hypothesis)
        try:
            return WorldMeasurementSet(
                cohort=cohort,
                hypotheses=tuple(hypotheses),
                continuation=continuation,
            )
        except Exception:
            return None

    @staticmethod
    def _world_vector_payload(value: Vector3) -> Dict[str, float]:
        payload = {
            "x": float(value.x),
            "y": float(value.y),
            "z": float(value.z),
        }
        if not all(math.isfinite(component) for component in payload.values()):
            raise RuntimeError("canonical world resolver returned a non-finite vector")
        return payload

    @staticmethod
    def _world_covariance_payload(value: Matrix3) -> Dict[str, List[float]]:
        try:
            values = [float(item) for item in value.values]
        except Exception as exc:
            raise RuntimeError("canonical world resolver returned invalid covariance") from exc
        if len(values) != 9 or not all(math.isfinite(item) for item in values):
            raise RuntimeError("canonical world resolver returned invalid covariance")
        return {"values": values}

    def _legacy_world_measurement_comparison(
        self,
        measurement_set: WorldMeasurementSet,
    ) -> Optional[Dict[str, Any]]:
        """Reconstruct the retired room-policy measurement without authority.

        The historical path selected the registered-depth X/Z wholesale when
        that camera's depth weight was nonzero, even when it called the result
        "fused"; otherwise it selected an admitted floor ray only where the
        room profile allowed floor-only output. This helper intentionally
        reproduces only that current-frame measurement decision. It does not
        run another temporal filter, mutate PersonGroundState, or feed any
        canonical consumer.
        """

        policy = self.legacy_world_fusion_policy
        if policy is None or not self._world_resolver_diagnostics_enabled:
            return None
        try:
            profile = policy.profile(str(measurement_set.cohort.camera_id))
        except Exception:
            return None
        floor_candidate = next(
            (
                candidate
                for candidate in measurement_set.hypotheses
                if str(candidate.kind) == "floor_ray" and candidate.valid
            ),
            None,
        )
        depth_candidate = next(
            (
                candidate
                for candidate in measurement_set.hypotheses
                if str(candidate.kind) == "registered_depth" and candidate.valid
            ),
            None,
        )
        selected: Optional[WorldMeasurementHypothesis] = None
        source: Optional[str] = None
        if depth_candidate is not None and float(profile.depth_weight_scale) > 0.0:
            selected = depth_candidate
            source = (
                "historical_depth_labeled_fused"
                if floor_candidate is not None
                and float(profile.floor_weight_scale) > 0.0
                else "historical_registered_depth_only"
            )
        elif (
            floor_candidate is not None
            and float(profile.floor_weight_scale) > 0.0
            and bool(profile.floor_only_allowed)
        ):
            selected = floor_candidate
            source = "historical_floor_ray_only"
        if selected is None or source is None:
            return None
        return {
            "id": "legacy_room_policy",
            "kind": "legacy_policy",
            "position": self._world_vector_payload(selected.position),
            "covariance": self._world_covariance_payload(selected.covariance),
            "source": source,
            "policy_id": str(policy.policy_id),
        }

    def _apply_resolved_world_measurement(
        self,
        track: Dict[str, Any],
        measurement_set: WorldMeasurementSet,
    ) -> Tuple[Optional[np.ndarray], Optional[ResolvedGroundMeasurement]]:
        """Resolve one exact cohort and publish bounded diagnostics."""

        resolver = self._world_resolver
        if resolver is None:
            raise RuntimeError("canonical world resolver is not constructed")
        # A track mapping can be reused by synthetic/replay callers.  Resolver
        # compact fields are current-cohort state, not durable identity state;
        # clear every one before resolution so a rejected/empty result cannot
        # inherit the prior frame's selected candidate or confidence.
        for field_name in (
            "world_quantity",
            "world_posture",
            "world_support_state",
            "world_resolver_confidence",
            "world_resolver_selected_id",
            "world_resolver_fused",
            "world_resolver_disagreement_m",
            "world_resolver_contact_basis",
            "world_resolver_source_continuity_match",
            "world_resolver",
            "world_covariance",
        ):
            track.pop(field_name, None)
        # A resolver covariance describes the raw current measurement.  It is
        # not valid for a later PGS prediction/hold or for a filtered point
        # displaced by the physical admission gate.  Clear any stale public
        # value here; the caller publishes an outer-error-inflated covariance
        # only after the current measurement is accepted.
        resolve_start_ns = time.perf_counter_ns()
        try:
            result = resolver.resolve(measurement_set)
        finally:
            _record_core_stage_timing(
                "world_resolver.resolve",
                resolve_start_ns,
                item_count=len(measurement_set.hypotheses),
            )
        track["world_quantity"] = "ground_footprint"
        track["world_posture"] = str(result.posture)
        track["world_support_state"] = str(result.support_state)
        diagnostics_enabled = bool(self._world_resolver_diagnostics_enabled)
        # Tracks are normally newly constructed, but explicit removal keeps a
        # reused synthetic/test mapping from leaking a previously requested
        # diagnostic into the compact path.
        selected_hypothesis = {
            candidate.candidate_id: candidate
            for candidate in measurement_set.hypotheses
        }.get(result.selected_candidate_id or "")
        candidates: List[Dict[str, Any]] = []
        diagnostic_by_id = {
            item.candidate_id: item for item in result.diagnostics
        }
        for candidate in (
            measurement_set.hypotheses[: int(self._world_resolver_diag_max_candidates)]
            if diagnostics_enabled
            else ()
        ):
            diagnostic = diagnostic_by_id.get(candidate.candidate_id)
            payload: Dict[str, Any] = {
                "id": str(candidate.candidate_id),
                "kind": str(candidate.kind),
                "position": self._world_vector_payload(candidate.position),
                "covariance": self._world_covariance_payload(candidate.covariance),
                "anchor": str(candidate.anchor),
                "contact_basis": str(candidate.anchor),
                "support_state": str(candidate.support_state),
                "posture": str(candidate.posture),
                "confidence": float(candidate.confidence),
                "support_score": float(candidate.support_score),
                "posture_compatibility": float(candidate.posture_compatibility),
                "occlusion": float(candidate.occlusion),
                "motion_consistency": float(candidate.motion_consistency),
                "ray_incidence_sin": candidate.ray_incidence_sin,
                "depth_support_fraction": candidate.depth_support_fraction,
                "depth_registration_sigma_m": candidate.depth_registration_sigma_m,
                "floor_depth_disagreement_m": candidate.floor_depth_disagreement_m,
                "pixel_uncertainty_px": candidate.pixel_uncertainty_px,
                "selected": bool(diagnostic.selected) if diagnostic is not None else False,
                "status": "selected" if diagnostic is not None and diagnostic.selected else (
                    "alternate" if diagnostic is not None and diagnostic.retained_as_alternate else "candidate"
                ),
            }
            if diagnostic is not None:
                payload.update(
                    {
                        "score": float(diagnostic.score),
                        "pcf_score": float(diagnostic.pcf_score),
                        "innovation_m": diagnostic.innovation_m,
                        "agreement_mahalanobis_sq": diagnostic.agreement_mahalanobis_sq,
                        "compatible_with_selected": bool(diagnostic.compatible_with_selected),
                        "retained_as_alternate": bool(diagnostic.retained_as_alternate),
                        "rejection_reason": diagnostic.rejection_reason,
                    }
                )
            if candidate.pcf is not None:
                payload["pcf"] = {
                    "prior_id": str(candidate.pcf.prior_id),
                    "revision_id": str(candidate.pcf.revision_id),
                    "status": str(candidate.pcf.status),
                    "inside_extent": candidate.pcf.inside_extent,
                    "inside_authored_space": candidate.pcf.inside_authored_space,
                    "extent_outside_distance_m": candidate.pcf.extent_outside_distance_m,
                    "evidence_observed": bool(candidate.pcf.evidence_observed),
                    "observed_confidence": float(candidate.pcf.observed_confidence),
                    "boundary_signed_distance_m": candidate.pcf.boundary_signed_distance_m,
                    "floor_height_m": candidate.pcf.floor_height_m,
                    "obstacle_clearance_m": candidate.pcf.obstacle_clearance_m,
                    "reasons": [str(reason)[:160] for reason in candidate.pcf.reasons[:8]],
                }
            candidates.append(payload)

        if diagnostics_enabled:
            legacy_comparison = self._legacy_world_measurement_comparison(
                measurement_set
            )
            diagnostic_payload: Dict[str, Any] = {
                "contract": "noesis.world_resolver_diagnostics",
                "version": 1,
                "camera_id": str(measurement_set.cohort.camera_id),
                "source_id": int(measurement_set.cohort.source_id),
                "tracker_id": int(measurement_set.cohort.tracker_id),
                "track_key": str(measurement_set.cohort.track_key),
                "tracker_lifecycle_generation": int(
                    measurement_set.cohort.tracker_lifecycle_generation
                ),
                "frame_id": int(measurement_set.cohort.frame_id),
                "observed_at_us": int(measurement_set.cohort.observed_at_us),
                "world_frame": str(measurement_set.cohort.world_frame),
                "world_frame_revision": str(measurement_set.cohort.world_revision),
                "calibration_revision": str(measurement_set.cohort.calibration_revision),
                "world_transform_sha256": (
                    str(measurement_set.cohort.world_transform_sha256)
                    if measurement_set.cohort.world_transform_sha256 is not None
                    else None
                ),
                "pcf_revision": measurement_set.cohort.pcf_revision,
                "selected_id": result.selected_candidate_id,
                "selected_kind": (
                    str(result.selected_kind)
                    if result.selected_kind is not None
                    else None
                ),
                "contributor_ids": [str(item) for item in result.contributor_ids[:4]],
                "alternate_id": result.alternate_candidate_id,
                "fused": bool(result.fused),
                "confidence": float(result.confidence),
                "pcf_score": result.pcf_score,
                "agreement_mahalanobis_sq": result.agreement_mahalanobis_sq,
                "decision": (
                    "weak"
                    if str(result.quality) == "weak"
                    else ("fused" if result.fused else str(result.quality))
                ),
                "reason": str(result.reason),
                "candidates": candidates,
            }
            if legacy_comparison is not None:
                diagnostic_payload["legacy"] = legacy_comparison
            if result.position is not None and result.covariance is not None:
                diagnostic_payload["resolved"] = {
                    "position": self._world_vector_payload(result.position),
                    "covariance": self._world_covariance_payload(result.covariance),
                }
            if result.disagreement_m is not None:
                diagnostic_payload["disagreement"] = {
                    "distance_m": float(result.disagreement_m),
                    "agreement_mahalanobis_sq": (
                        float(result.agreement_mahalanobis_sq)
                        if result.agreement_mahalanobis_sq is not None
                        else None
                    ),
                    "reason": "incompatible_alternate_preserved",
                }
            track["world_resolver"] = diagnostic_payload
        track["world_resolver_confidence"] = float(result.confidence)
        if result.status != "measured" or result.position is None:
            return None, result
        point = np.asarray(
            [float(result.position.x), float(result.position.y), float(result.position.z)],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(point)):
            raise RuntimeError("canonical world resolver returned a non-finite point")
        if result.covariance is None or len(result.covariance.values) != 9:
            raise RuntimeError("canonical world resolver returned no covariance")
        track["world_resolver_selected_id"] = str(result.selected_candidate_id or "")
        track["world_resolver_fused"] = bool(result.fused)
        track["world_resolver_disagreement_m"] = (
            float(result.disagreement_m) if result.disagreement_m is not None else None
        )
        if selected_hypothesis is not None:
            track["world_resolver_contact_basis"] = str(selected_hypothesis.anchor)
        return point, result

    def _predict_world_state(
        self,
        state: _WorldAnchorState,
        now_ts: float,
    ) -> Tuple[Optional[float], Optional[float], float]:
        if state.world_x is None or state.world_z is None or float(state.filtered_ts) < 0.0:
            return None, None, 0.0
        dt = max(0.0, float(now_ts) - float(state.filtered_ts))
        pred_x = float(state.world_x) + float(state.vel_world_x) * dt
        pred_z = float(state.world_z) + float(state.vel_world_z) * dt
        return pred_x, pred_z, dt

    @staticmethod
    def _publish_filtered_world_covariance(
        track: Dict[str, Any],
        *,
        resolved: Optional[ResolvedGroundMeasurement],
        resolved_point: Optional[np.ndarray],
        emitted_point: Optional[np.ndarray],
    ) -> None:
        """Publish covariance only for an accepted current measurement.

        PersonGroundState may reject a geometrically valid resolver result and
        emit a bounded prediction/hold instead.  The resolver covariance must
        not follow that continuation.  When the current sample is accepted,
        add the outer product of the resolver-to-emitted displacement so the
        published covariance describes the actual point consumed downstream.
        """

        track.pop("world_covariance", None)
        if resolved is None or resolved.status != "measured":
            return
        if resolved_point is None or emitted_point is None or resolved.covariance is None:
            raise RuntimeError("accepted canonical measurement has no covariance")
        try:
            center = np.asarray(resolved_point, dtype=np.float64).reshape(-1)[:3]
            emitted = np.asarray(emitted_point, dtype=np.float64).reshape(-1)[:3]
            covariance = np.asarray(resolved.covariance.values, dtype=np.float64).reshape(3, 3)
        except Exception as exc:
            raise RuntimeError("accepted canonical measurement covariance is invalid") from exc
        if (
            center.size != 3
            or emitted.size != 3
            or covariance.shape != (3, 3)
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(emitted))
            or not np.all(np.isfinite(covariance))
        ):
            raise RuntimeError("accepted canonical measurement covariance is non-finite")
        covariance = 0.5 * (covariance + covariance.T)
        displacement = emitted - center
        covariance += np.outer(displacement, displacement)
        covariance = 0.5 * (covariance + covariance.T)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        except Exception as exc:
            raise RuntimeError("accepted canonical measurement covariance decomposition failed") from exc
        if not np.all(np.isfinite(eigenvalues)) or not np.all(np.isfinite(eigenvectors)):
            raise RuntimeError("accepted canonical measurement covariance is invalid")
        covariance = eigenvectors @ np.diag(np.clip(eigenvalues, 1e-6, 1e6)) @ eigenvectors.T
        covariance = 0.5 * (covariance + covariance.T)
        values = [float(value) for value in covariance.reshape(-1)]
        if len(values) != 9 or not all(math.isfinite(value) for value in values):
            raise RuntimeError("accepted canonical measurement covariance is invalid")
        track["world_covariance"] = values

    def _update_world_state(
        self,
        state: Optional[_WorldAnchorState],
        *,
        measurement: np.ndarray,
        floor_y: float,
        now_ts: float,
        alpha: float,
        beta: float,
        quality: str = "good",
        force_accept: bool = False,
        contact_basis: Optional[str] = None,
        image_motion_supported: bool = False,
    ) -> np.ndarray:
        mx = float(measurement[0])
        mz = float(measurement[2])
        if state is None:
            return np.array([mx, float(floor_y), mz], dtype=np.float64)

        _ = (alpha, beta)
        return update_human_cv_filter(
            state,
            measurement=measurement,
            floor_y=float(floor_y),
            now_ts=float(now_ts),
            quality=str(quality or "good"),
            config=self._human_ground_cfg,
            force_accept=bool(force_accept),
            contact_basis=contact_basis,
            image_motion_supported=bool(image_motion_supported),
        )

    def _update_track_world_state(
        self,
        track: Dict[str, Any],
        state: Optional[_WorldAnchorState],
        *,
        measurement: np.ndarray,
        floor_y: float,
        now_ts: float,
        alpha: float,
        beta: float,
        quality: str = "good",
        force_accept: bool = False,
        contact_basis: Optional[str] = None,
        image_motion_supported: bool = False,
    ) -> np.ndarray:
        """Record pre-filter evidence and apply the shared physical filter."""
        track["world_prefilter_measurement"] = [
            float(measurement[0]),
            float(floor_y),
            float(measurement[2]),
        ]
        if state is not None:
            pred_x, pred_z, _dt = self._predict_world_state(state, now_ts)
            track["world_filter_prediction"] = (
                [float(pred_x), float(floor_y), float(pred_z)]
                if pred_x is not None and pred_z is not None
                else None
            )
        result = self._update_world_state(
            state,
            measurement=measurement,
            floor_y=float(floor_y),
            now_ts=float(now_ts),
            alpha=float(alpha),
            beta=float(beta),
            quality=str(quality or "good"),
            force_accept=bool(force_accept),
            contact_basis=contact_basis,
            image_motion_supported=bool(image_motion_supported),
        )
        if state is not None:
            # A physical reject advances only the bounded process model.  The
            # pre-filter prediction above is useful for accepted observations,
            # but after a long reject run it can be stale/unbounded diagnostic
            # evidence.  Publish the exact bounded filter output instead.
            if not bool(state.measurement_accepted):
                track["world_filter_prediction"] = [
                    float(result[0]),
                    float(floor_y),
                    float(result[2]),
                ]
            complete_source_admission(
                state,
                measurement_accepted=bool(state.measurement_accepted),
            )
        return result

    def _admit_live_world_source(
        self,
        state: Optional[_WorldAnchorState],
        *,
        candidate_source: str,
        quality: str,
        depth_weight: float,
        posture: str,
        authoritative: bool = False,
    ) -> bool:
        if state is None:
            return True
        return begin_source_admission(
            state,
            candidate_source=str(candidate_source),
            candidate_score=source_score(
                str(candidate_source),
                quality=str(quality),
                depth_weight=float(depth_weight),
                posture=str(posture),
            ),
            config=self._human_ground_cfg,
            authoritative=bool(authoritative),
        )

    def _observe_resolver_source_continuity(
        self,
        track: Dict[str, Any],
        state: Optional[_WorldAnchorState],
        *,
        candidate_source: str,
        quality: str,
        depth_weight: float,
        posture: str,
    ) -> bool:
        """Update PersonGroundState source continuity without vetoing resolution.

        The universal resolver has already compared every current geometric
        hypothesis.  The older source hysteresis remains useful as bounded
        continuity evidence, but applying it as a second admission authority
        makes intermittent depth cadence permanently suppress valid floor-ray
        measurements.  Record whether the resolved source matches that sticky
        preference, let the physical filter decide admission, and commit a
        pending source switch only when the physical measurement is accepted.
        """

        continuity_match = self._admit_live_world_source(
            state,
            candidate_source=str(candidate_source),
            quality=str(quality),
            depth_weight=float(depth_weight),
            posture=str(posture),
        )
        track["world_resolver_source_continuity_match"] = bool(continuity_match)
        diagnostics = track.get("world_resolver")
        if isinstance(diagnostics, MutableMapping):
            diagnostics["source_continuity_match"] = bool(continuity_match)
        return bool(continuity_match)

    def _world_fusion_weights(
        self,
        camera_id: str,
        *,
        floor_weight: float,
        depth_weight: float,
        track: Dict[str, Any],
    ) -> Tuple[float, float, bool]:
        if bool(getattr(self, "_world_resolver_enabled", False)):
            # Camera-specific policy weights are retained below for historical
            # comparison only.  The canonical resolver receives both
            # hypotheses and their evidence in every room.
            effective_floor = max(0.0, float(floor_weight))
            effective_depth = max(0.0, float(depth_weight))
            track["world_fusion_policy_id"] = "legacy_compatibility_only"
            track["world_floor_weight_scale"] = 1.0
            track["world_depth_weight_scale"] = 1.0
            track["world_floor_weight_effective"] = effective_floor
            track["world_depth_weight_effective"] = effective_depth
            return effective_floor, effective_depth, True
        policy = self.world_fusion_policy
        if policy is None:
            return float(floor_weight), float(depth_weight), True
        profile = policy.profile(str(camera_id))
        effective_floor = float(floor_weight) * float(profile.floor_weight_scale)
        effective_depth = float(depth_weight) * float(profile.depth_weight_scale)
        track["world_fusion_policy_id"] = str(policy.policy_id)
        track["world_floor_weight_scale"] = float(profile.floor_weight_scale)
        track["world_depth_weight_scale"] = float(profile.depth_weight_scale)
        track["world_floor_weight_effective"] = float(effective_floor)
        track["world_depth_weight_effective"] = float(effective_depth)
        return effective_floor, effective_depth, bool(profile.floor_only_allowed)

    def _projective_continuation_allowed(self, camera_id: str, track: Dict[str, Any]) -> bool:
        """Return the calibrated policy decision for weak floor continuation.

        A projective point is never promoted to ``last_good_world`` here, so a
        depth-required camera retains depth as its metric authority.  The
        policy is still consulted and published: an active profile is required
        for weak continuation, while ``floor_only_allowed`` records whether
        the same point could be authoritative on its own.
        """

        if bool(getattr(self, "_world_resolver_enabled", False)):
            track["world_projective_continuation_allowed"] = True
            track["world_projective_authoritative_allowed"] = True
            return True
        policy = self.world_fusion_policy
        if policy is None:
            allowed = True
            authoritative_allowed = True
        else:
            try:
                profile = policy.profile(str(camera_id))
                allowed = True
                authoritative_allowed = bool(profile.floor_only_allowed)
            except Exception:
                allowed = False
                authoritative_allowed = False
        track["world_projective_continuation_allowed"] = bool(allowed)
        track["world_projective_authoritative_allowed"] = bool(authoritative_allowed)
        return bool(allowed)

    @staticmethod
    def _floor_contact_bbox_gate(
        *,
        anchor_uv: Optional[Sequence[float]],
        bbox: Optional[Sequence[float]],
        incidence_sin: float,
        track: Dict[str, Any],
        anchor_range_m: Optional[float] = None,
        bbox_bottom_range_m: Optional[float] = None,
    ) -> bool:
        """Reject a near-horizon pose contact above its detector silhouette.

        A floor ray from a low-incidence pixel is extremely sensitive to a
        small vertical anchor error.  At the far edge of the Family Room
        replay, pose ankles can be hallucinated well above the detector box
        bottom; intersecting that point with the floor creates a plausible
        finite point many metres behind the visible person.  Keep this as an
        estimator-side plausibility gate: it does not clamp or consult the
        presentation floorplan.

        Missing image geometry is left to the existing ray/range checks.  A
        bbox gap is only rejected when the ray is also near the horizon, so
        modest detector padding or lower-body occlusion is not turned into a
        blanket loss of metric observations.
        """

        track["world_floor_contact_plausible"] = True
        track.pop("world_floor_contact_gap_px", None)
        track.pop("world_floor_contact_gap_ratio", None)
        track.pop("world_floor_contact_rejection_reason", None)
        if anchor_uv is None or bbox is None or len(anchor_uv) < 2 or len(bbox) < 4:
            return True
        try:
            anchor_u = float(anchor_uv[0])
            anchor_v = float(anchor_uv[1])
            _left, top, width, height = (float(value) for value in bbox[:4])
            incidence = float(incidence_sin)
        except (TypeError, ValueError, IndexError, OverflowError):
            return True
        if not all(
            math.isfinite(value)
            for value in (anchor_u, anchor_v, top, width, height, incidence)
        ) or width <= 1.0 or height <= 1.0:
            return True

        bbox_bottom = float(top) + float(height)
        gap_px = float(bbox_bottom) - float(anchor_v)
        gap_ratio = float(gap_px) / float(height)
        track["world_floor_contact_gap_px"] = float(gap_px)
        track["world_floor_contact_gap_ratio"] = float(gap_ratio)

        range_delta_m = math.nan
        if anchor_range_m is not None and bbox_bottom_range_m is not None:
            try:
                range_delta_m = float(anchor_range_m) - float(bbox_bottom_range_m)
                range_tolerance_m = max(
                    2.0,
                    0.20 * float(bbox_bottom_range_m),
                )
                if all(
                    math.isfinite(value)
                    for value in (
                        float(anchor_range_m),
                        float(bbox_bottom_range_m),
                        range_delta_m,
                        range_tolerance_m,
                    )
                ):
                    track["world_floor_bbox_bottom_range_m"] = float(
                        bbox_bottom_range_m
                    )
                    track["world_floor_contact_range_delta_m"] = float(
                        range_delta_m
                    )
                    track["world_floor_contact_range_tolerance_m"] = float(
                        range_tolerance_m
                    )
                else:
                    range_delta_m = math.nan
            except (TypeError, ValueError, OverflowError):
                range_delta_m = math.nan

        # Use a fractional tolerance for ordinary detector padding while
        # retaining a small absolute tolerance for compact, distant boxes.
        max_gap_px = max(16.0, 0.20 * float(height))
        near_horizon = float(incidence) < 0.10
        implausible_gap = float(gap_px) > max_gap_px
        implausible_range = bool(
            math.isfinite(float(range_delta_m))
            and float(range_delta_m)
            > float(track.get("world_floor_contact_range_tolerance_m", math.inf))
        )
        plausible = not (near_horizon and (implausible_gap or implausible_range))
        track["world_floor_contact_plausible"] = bool(plausible)
        if not plausible:
            track["world_floor_contact_rejection_reason"] = (
                "bbox_contact_range_disagreement_near_horizon"
                if implausible_range and not implausible_gap
                else "bbox_contact_gap_near_horizon"
            )
        return bool(plausible)

    def _admit_floor_ray_range(
        self,
        camera_id: str,
        *,
        calib: Any,
        floor_candidate: np.ndarray,
        track: Dict[str, Any],
        anchor_uv: Optional[Sequence[float]] = None,
        bbox: Optional[Sequence[float]] = None,
        flip_u: bool = False,
        flip_v: bool = False,
    ) -> bool:
        policy = self.world_fusion_policy
        if bool(getattr(self, "_world_resolver_enabled", False)):
            limit_m: Optional[float] = float(self._world_resolver_max_range_m)
        elif policy is None:
            limit_m: Optional[float] = None
        else:
            profile = policy.profile(str(camera_id))
            limit_m = float(profile.floor_ray_max_range_m)
        track["world_floor_range_limit_m"] = limit_m
        admitted = False
        rejection_reason = "floor_ray_geometry_invalid"
        try:
            _rotation, camera_world = parse_extrinsics(calib.extrinsics_col_major)
            unit_scale = float(getattr(calib, "unit_scale", 1.0) or 1.0)
            if not math.isfinite(unit_scale) or unit_scale <= 0.0:
                unit_scale = 1.0
            camera_world = np.asarray(camera_world, dtype=np.float64) * unit_scale
            candidate = np.asarray(floor_candidate, dtype=np.float64)
            delta = candidate - camera_world
            horizontal_range_m = float(math.hypot(float(delta[0]), float(delta[2])))
            ray_range_m = float(np.linalg.norm(delta))
            incidence_sin = (
                abs(float(delta[1])) / ray_range_m
                if math.isfinite(ray_range_m) and ray_range_m > 1e-9
                else 0.0
            )
            track["world_floor_range_m"] = horizontal_range_m
            track["world_floor_incidence_sin"] = incidence_sin
            bbox_bottom_range_m: Optional[float] = None
            if bbox is not None and len(bbox) >= 4:
                try:
                    bbox_left, bbox_top, bbox_width, bbox_height = (
                        float(value) for value in bbox[:4]
                    )
                    bbox_bottom_hit = self._project_pixel_to_floor_world(
                        calib,
                        bbox_left + 0.5 * bbox_width,
                        bbox_top + bbox_height,
                        flip_u=bool(flip_u),
                        flip_v=bool(flip_v),
                    )
                    if bbox_bottom_hit is not None:
                        bbox_bottom_delta = (
                            np.asarray(bbox_bottom_hit, dtype=np.float64)
                            - camera_world
                        )
                        bbox_bottom_range_m = float(
                            math.hypot(
                                float(bbox_bottom_delta[0]),
                                float(bbox_bottom_delta[2]),
                            )
                        )
                except (TypeError, ValueError, IndexError, OverflowError):
                    bbox_bottom_range_m = None
            if bbox_bottom_range_m is not None:
                track["world_floor_bbox_bottom_range_m"] = float(
                    bbox_bottom_range_m
                )
            range_admitted = bool(
                math.isfinite(horizontal_range_m)
                and (
                    limit_m is None
                    or horizontal_range_m <= float(limit_m)
                )
            )
            contact_admitted = self._floor_contact_bbox_gate(
                anchor_uv=anchor_uv,
                bbox=bbox,
                incidence_sin=incidence_sin,
                track=track,
                anchor_range_m=horizontal_range_m,
                bbox_bottom_range_m=bbox_bottom_range_m,
            )
            admitted = bool(range_admitted and contact_admitted)
            if not range_admitted:
                rejection_reason = "floor_ray_range_exceeded"
            elif not contact_admitted:
                rejection_reason = str(
                    track.get("world_floor_contact_rejection_reason")
                    or "floor_ray_contact_invalid"
                )
        except Exception:
            admitted = False
        track["world_floor_admitted"] = admitted
        if admitted:
            track.pop("world_floor_rejection_reason", None)
        else:
            track["world_floor_rejection_reason"] = rejection_reason
        return admitted

    def _admit_world_observation_range(
        self,
        camera_id: str,
        *,
        calib: Any,
        world_candidate: np.ndarray,
        track: Dict[str, Any],
    ) -> bool:
        """Admit any metric world candidate against the calibrated range envelope.

        ``floor_ray_max_range_m`` is the only calibrated per-camera metric
        envelope in the fusion policy.  It was originally applied only to the
        floor ray, which left registered-depth points on a separate path.  A
        depth point is still a camera-originated metric observation and must
        not be allowed to seed the canonical filter beyond that envelope.
        This is an estimator-side rejection, never a PCF/display crop.
        """
        policy = self.world_fusion_policy
        if bool(getattr(self, "_world_resolver_enabled", False)):
            limit_m = float(self._world_resolver_max_range_m)
        elif policy is None:
            return True
        else:
            profile = policy.profile(str(camera_id))
            limit_m = float(profile.floor_ray_max_range_m)
        track["world_observation_range_limit_m"] = limit_m
        admitted = False
        rejection_reason = "world_observation_range_invalid"
        try:
            _rotation, camera_world = parse_extrinsics(calib.extrinsics_col_major)
            unit_scale = float(getattr(calib, "unit_scale", 1.0) or 1.0)
            if not math.isfinite(unit_scale) or unit_scale <= 0.0:
                unit_scale = 1.0
            camera_world = np.asarray(camera_world, dtype=np.float64) * unit_scale
            candidate = np.asarray(world_candidate, dtype=np.float64)
            if candidate.shape[0] < 3 or not np.all(np.isfinite(candidate[:3])):
                raise ValueError("world candidate is not finite")
            delta = candidate[:3] - camera_world[:3]
            horizontal_range_m = float(math.hypot(float(delta[0]), float(delta[2])))
            track["world_observation_range_m"] = horizontal_range_m
            admitted = bool(
                math.isfinite(horizontal_range_m)
                and horizontal_range_m <= limit_m
            )
            if not admitted:
                rejection_reason = "world_observation_range_exceeded"
        except Exception:
            admitted = False
        track["world_observation_range_admitted"] = admitted
        if admitted:
            track.pop("world_observation_range_rejection_reason", None)
        else:
            track["world_observation_range_rejection_reason"] = rejection_reason
        return admitted

    @staticmethod
    def _invalidate_world_track(track: Dict[str, Any], reason: str) -> None:
        """Remove an untrusted world result at the canonical publication boundary."""

        track["world_valid"] = False
        track["world_quality"] = "invalid"
        track["world_quality_reason"] = str(reason or "world_observation_invalid")
        track.pop("world", None)
        track.pop("world_source", None)
        track.pop("world_frame", None)
        track.pop("world_frame_revision", None)
        track.pop("world_transform_sha256", None)

    def _set_track_image_base_from_world(
        self,
        track: Dict[str, Any],
        *,
        calib: Any,
        world_point: np.ndarray,
        flip_u: bool,
        flip_v: bool,
    ) -> None:
        try:
            uv = project_world_to_image(
                world_point,
                calib.intrinsics,
                calib.extrinsics_col_major,
                tuple(int(x) for x in calib.image_size),
                unit_scale=1.0,
                flip_u=bool(flip_u),
                flip_v=bool(flip_v),
            )
        except Exception:
            uv = None
        if uv is None:
            return
        try:
            track["image_base"] = [float(uv[0]), float(uv[1])]
        except Exception:
            return

    def _project_image_motion_prediction(
        self,
        *,
        camera_id: str,
        state: Optional[_WorldAnchorState],
        bbox_project: Sequence[float],
        calib: Any,
        now_ts: float,
        lifecycle_generation: Optional[int],
        flip_u: bool,
        flip_v: bool,
        track: Dict[str, Any],
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Project one bounded bbox-transported foot as weak observation.

        This candidate is downstream of metric admission and is integrated (if
        policy permits weak continuation) only by
        ``integrate_projective_ground_observation``. It never changes the
        accepted image origin: every frame is transported from the last
        physically accepted foot/bbox, so a run of rejected frames cannot
        integrate image-anchor drift.
        """

        if state is None or state.last_good_world is None:
            return None
        transported = transport_accepted_image_foot(
            state,
            bbox=bbox_project,
            now_ts=float(now_ts),
            lifecycle_generation=lifecycle_generation,
            ttl_s=float(self._world_anchor_hold_ttl_s),
            config=self._human_ground_cfg,
        )
        if transported is None:
            return None
        u, v, age_s, center_step_px, scale_ratio, transport_name = transported
        try:
            width, height = (int(calib.image_size[0]), int(calib.image_size[1]))
        except Exception:
            return None
        if not (0.0 <= float(u) < float(width) and 0.0 <= float(v) < float(height)):
            return None
        hit = self._project_pixel_to_floor_world(
            calib,
            float(u),
            float(v),
            flip_u=flip_u,
            flip_v=flip_v,
        )
        if hit is None:
            return None
        candidate = np.asarray(hit, dtype=np.float64)
        if candidate.shape[0] < 3 or not np.all(np.isfinite(candidate[:3])):
            return None
        # Reuse the exact per-camera ray envelope used for authoritative floor
        # observations.  This is a metric/ray gate, not a clamp.
        if not self._admit_floor_ray_range(
            camera_id,
            calib=calib,
            floor_candidate=candidate,
            track=track,
            anchor_uv=(float(u), float(v)),
            bbox=bbox_project,
            flip_u=flip_u,
            flip_v=flip_v,
        ):
            return None
        try:
            prior = np.asarray(state.last_good_world, dtype=np.float64)
            world_delta_m = math.hypot(
                float(candidate[0]) - float(prior[0]),
                float(candidate[2]) - float(prior[2]),
            )
        except Exception:
            return None
        allowed_world_delta_m = (
            float(self._human_ground_cfg.max_speed_mps) * max(0.0, float(age_s))
            + float(self._human_ground_cfg.projective_world_slack_m)
        )
        if not math.isfinite(world_delta_m) or world_delta_m > allowed_world_delta_m:
            return None
        provenance = {
            "type": "bbox_affine_floor_projection",
            "non_authoritative": True,
            "origin": "last_accepted_image_foot",
            "transport": str(transport_name),
            "age_s": float(age_s),
            "bbox_center_step_px": float(center_step_px),
            "bbox_scale_ratio": float(scale_ratio),
            "image_foot": [float(u), float(v)],
            "world_delta_m": float(world_delta_m),
            "world_delta_limit_m": float(allowed_world_delta_m),
        }
        return candidate, provenance

    def _refine_seeded_world_with_ground_state(
        self,
        sensor_id: int,
        camera_id: str,
        track: Dict[str, Any],
        *,
        world_source_label: str,
        world_now_ts: Optional[float] = None,
    ) -> None:
        """Adapt a pre-seeded SDK world observation into shared human state."""
        if self.bev_calibration is None:
            self._invalidate_world_track(track, "calibration_unavailable")
            return
        if track.get("world_valid") is not True:
            return
        world = track.get("world")
        if not isinstance(world, (list, tuple)) or len(world) < 3:
            self._invalidate_world_track(track, "seeded_world_invalid")
            return
        try:
            mx, my, mz = (float(world[0]), float(world[1]), float(world[2]))
        except Exception:
            self._invalidate_world_track(track, "seeded_world_invalid")
            return
        if not (math.isfinite(mx) and math.isfinite(my) and math.isfinite(mz)):
            self._invalidate_world_track(track, "seeded_world_nonfinite")
            return

        try:
            calib = self._world_calibration_snapshot(sensor_id, camera_id)
            if calib is None or calib.intrinsics is None or calib.extrinsics_col_major is None:
                self._invalidate_world_track(track, "calibration_unavailable")
                return

            world_frame_id, world_frame_revision, world_transform_sha256 = (
                world_frame_binding_from_calibration(
                    calib,
                    default_frame_id=self._world_frame,
                )
            )
            # Native V3DT bbox3d metadata is produced in the current backend
            # world frame, but older bridge payloads do not carry the active
            # revision digest. Bind that one trusted producer to the current
            # immutable snapshot; arbitrary pre-seeded world rows are handled
            # by _augment_track_with_world and remain fail-closed.
            if (
                str(world_source_label) == "bbox3d"
                and track.get("world_frame") == str(world_frame_id or self._world_frame)
            ):
                if world_frame_revision and not track.get("world_frame_revision"):
                    track["world_frame_revision"] = str(world_frame_revision)
                if world_transform_sha256 and not track.get("world_transform_sha256"):
                    track["world_transform_sha256"] = str(world_transform_sha256)
            if not world_frame_matches_calibration(track, calib, default_frame_id=self._world_frame):
                self._invalidate_world_track(track, "world_frame_mismatch")
                return

            now_ts = (
                float(world_now_ts)
                if world_now_ts is not None
                else self._world_timestamp_for_frame(
                    int(sensor_id),
                    media_pts_ns=track.get("media_pts_ns", 0),
                    observed_ts=float(time.time()),
                )
            )
            world_key = self._world_track_key(sensor_id, track)
            self._maybe_prune_world_state(now_ts)
            state: Optional[_WorldAnchorState] = None
            if world_key is not None:
                state, restored = self._world_state_for_observation(
                    world_key,
                    now_ts=now_ts,
                    bbox=track.get("bbox"),
                    lifecycle_generation=(
                        int(track.get("tracker_lifecycle_generation"))
                        if track.get("tracker_lifecycle_generation") is not None
                        else None
                    ),
                )
                if restored:
                    track["world_state_continuity"] = "restored_short_ghost"
                state.ts = now_ts
                if bind_world_frame(
                    state,
                    world_frame_id=world_frame_id,
                    world_frame_revision=world_frame_revision,
                    world_transform_sha256=world_transform_sha256,
                ):
                    # A calibration revision is a physical discontinuity even
                    # when the tracker ID survives. Never let a rejected
                    # seeded observation inherit the prior world's hold.
                    track["world_state_continuity"] = "reset_world_frame"
                    state.ts = now_ts

            metric_candidate = np.array([mx, float(calib.floor_y), mz], dtype=np.float64)
            if not self._admit_world_observation_range(
                camera_id,
                calib=calib,
                world_candidate=metric_candidate,
                track=track,
            ):
                rejection_reason = str(
                    track.get("world_observation_range_rejection_reason")
                    or "world_observation_range_exceeded"
                )
                if state is None:
                    track["world_valid"] = False
                    track["world_quality"] = "invalid"
                    track["world_quality_reason"] = rejection_reason
                    track.pop("world", None)
                    track.pop("world_source", None)
                    return
                mark_world_measurement_unavailable(state, reason=rejection_reason)
                hit = advance_human_cv_prediction(
                    state,
                    floor_y=float(calib.floor_y),
                    now_ts=now_ts,
                    config=self._human_ground_cfg,
                    reason=rejection_reason,
                )
            else:
                hit = self._update_track_world_state(
                    track,
                    state,
                    measurement=metric_candidate,
                    floor_y=float(calib.floor_y),
                    now_ts=now_ts,
                    alpha=float(self._world_smooth_alpha_good),
                    beta=max(0.0, min(1.0, float(self._world_smooth_alpha_good) * 0.25)),
                    quality="good",
                )

            if state is not None and not state.measurement_accepted:
                hold_age = float(now_ts) - float(state.last_good_ts or 0.0)
                if (
                    state.last_good_world is not None
                    and hold_age <= float(self._world_anchor_hold_ttl_s)
                ):
                    can_predict = bool(
                        str(state.motion_mode or "unknown")
                        not in ("idle", "sit", "lie")
                        and (
                            str(state.motion_mode or "unknown") == "walk"
                            or bool(state.image_motion_supported)
                        )
                    )
                    pred_x, pred_z, _prediction_age = self._predict_world_state(
                        state,
                        now_ts,
                    )
                    if can_predict and pred_x is not None and pred_z is not None:
                        hit = np.asarray(
                            [float(pred_x), float(calib.floor_y), float(pred_z)],
                            dtype=np.float64,
                        )
                        track["world_filter_prediction"] = [
                            float(pred_x),
                            float(calib.floor_y),
                            float(pred_z),
                        ]
                        world_source_label = "cv_prediction"
                    else:
                        hit = np.asarray(state.last_good_world, dtype=np.float64)
                        world_source_label = "anchor_hold"
                    track["world_quality"] = "estimated"
                    track["world_quality_reason"] = str(
                        state.measurement_rejection_reason or "physical_measurement_rejected"
                    )
                else:
                    track["world_estimator_evaluated"] = True
                    track["world_valid"] = False
                    track["world_quality"] = "invalid"
                    track["world_quality_reason"] = str(
                        state.measurement_rejection_reason or "physical_measurement_rejected"
                    )
                    track.pop("world", None)
                    track.pop("world_source", None)
                    state.trail_append_allowed = False
                    for key, value in state.as_public_fields().items():
                        if value is not None:
                            track[key] = value
                    return

            image_uv = None
            for key in ("image_base", "image_foot"):
                raw_uv = track.get(key)
                if isinstance(raw_uv, (list, tuple)) and len(raw_uv) >= 2:
                    try:
                        image_uv = (float(raw_uv[0]), float(raw_uv[1]))
                        break
                    except Exception:
                        image_uv = None
            if state is not None:
                update_motion_mode(
                    state,
                    now_ts=now_ts,
                    image_foot_uv=image_uv,
                    config=self._human_ground_cfg,
                )
                if (
                    world_source_label == "cv_prediction"
                    and state.motion_mode in ("idle", "sit", "lie")
                ):
                    world_source_label = "anchor_hold"
                if state.motion_mode in ("idle", "sit", "lie") and state.locked_world is not None:
                    hit = np.array(
                        [float(state.locked_world[0]), float(calib.floor_y), float(state.locked_world[1])],
                        dtype=np.float64,
                    )
                    state.world_x = float(state.locked_world[0])
                    state.world_z = float(state.locked_world[1])
                    state.vel_world_x = 0.0
                    state.vel_world_z = 0.0

            # Motion-mode locking can replace a rejected seeded posterior with
            # a held point, so validate once more after that canonical-state
            # operation and before publishing the refined row.
            if not self._admit_world_observation_range(
                camera_id,
                calib=calib,
                world_candidate=np.asarray(hit, dtype=np.float64),
                track=track,
            ):
                rejection_reason = str(
                    track.get("world_observation_range_rejection_reason")
                    or "world_observation_range_exceeded"
                )
                if state is not None:
                    mark_world_measurement_unavailable(
                        state,
                        reason=rejection_reason,
                    )
                    state.trail_append_allowed = False
                track["world_estimator_evaluated"] = True
                self._invalidate_world_track(track, rejection_reason)
                if state is not None:
                    for key, value in state.as_public_fields().items():
                        if value is not None:
                            track[key] = value
                return

            flip_u, flip_v = self._infer_image_flips(camera_id, calib)
            # V3DT already publishes ``image_base`` from the opposite cuboid
            # endpoint in the tracker/camInfo frame.  Preserve that diagnostic
            # while the canonical world point is filtered.
            if "image_base" not in track:
                self._set_track_image_base_from_world(
                    track,
                    calib=calib,
                    world_point=hit,
                    flip_u=flip_u,
                    flip_v=flip_v,
                )
            wx, wy, wz = (float(hit[0]), float(hit[1]), float(hit[2]))
            track["world"] = [wx, wy, wz]
            track["world_valid"] = True
            track.setdefault("world_quality", "good")
            track["world_frame"] = str(world_frame_id or self._world_frame)
            if world_frame_revision:
                track["world_frame_revision"] = str(world_frame_revision)
            else:
                track.pop("world_frame_revision", None)
            if world_transform_sha256:
                track["world_transform_sha256"] = str(world_transform_sha256)
            else:
                track.pop("world_transform_sha256", None)
            track["world_source"] = str(world_source_label)
            if state is not None:
                for key, value in state.as_public_fields().items():
                    if value is not None:
                        track[key] = value
                if world_source_label not in ("anchor_hold", "cv_prediction"):
                    state.last_good_world = (wx, wy, wz)
                    state.last_good_ts = now_ts
                    state.ts = now_ts
        except Exception:
            logger.debug("Seeded world refinement failed", exc_info=True)
            self._invalidate_world_track(track, "world_refinement_failed")

    def _apply_scene_prior_shadow(
        self,
        camera_id: str,
        track: Dict[str, Any],
    ) -> None:
        priors = self.scene_priors
        if priors is None:
            return
        revision = priors.revision_for_camera(camera_id)
        if revision is None:
            return
        frame_binding = priors.frame_binding(camera_id)
        expected_world_frame = (
            str(frame_binding.target_frame.frame_id)
            if frame_binding is not None
            else "backend_world_m"
        )
        expected_world_revision = (
            str(frame_binding.target_frame.revision)
            if frame_binding is not None
            else None
        )
        base = {
            "contract": "noesis.scene_prior.track_diagnostic",
            "contract_version": 1,
            "prior_id": revision.manifest.prior_id,
            "space_id": revision.manifest.space_id,
            "mode": "shadow",
            "coordinate_frame": expected_world_frame,
            "coordinate_frame_revision": expected_world_revision,
        }
        world = track.get("world")
        if (
            track.get("world_valid") is not True
            or not isinstance(world, (list, tuple))
            or len(world) < 3
        ):
            track["scene_prior"] = {
                **base,
                "status": "unknown",
                "inside_extent": False,
                "inside_authored_space": False,
                "evidence_observed": False,
                "evidence_confidence": 0.0,
                "reasons": ["world_position_unavailable"],
            }
            return
        if (
            track.get("world_frame") != expected_world_frame
            or (
                expected_world_revision is not None
                and track.get("world_frame_revision") != expected_world_revision
            )
        ):
            track["scene_prior"] = {
                **base,
                "status": "error",
                "inside_extent": False,
                "inside_authored_space": False,
                "evidence_observed": False,
                "evidence_confidence": 0.0,
                "reasons": ["world_frame_mismatch"],
            }
            return
        try:
            diagnostic = priors.evaluate(camera_id, world)
        except ScenePriorError as exc:
            track["scene_prior"] = {
                **base,
                "status": "error",
                "inside_extent": False,
                "inside_authored_space": False,
                "evidence_observed": False,
                "evidence_confidence": 0.0,
                "reasons": ["scene_prior_evaluation_error"],
                "error": str(exc),
            }
            return
        if diagnostic is not None:
            track["scene_prior"] = diagnostic

    def _augment_track_with_world(
        self,
        sensor_id: int,
        camera_id: str,
        track: Dict[str, Any],
        *,
        obj_meta: Any | None = None,
        pose_kpts_abs: Optional[np.ndarray] = None,
        depth_result: Optional[ObjectDepthResult] = None,
        world_now_ts: Optional[float] = None,
    ) -> None:
        """Calculate world coordinates for a track if calibration is available."""
        if self._tracking_mode_is_v3dt():
            has_bbox3d_world = bool(
                isinstance(track.get("bbox3d"), Mapping)
                and track.get("world_source") == "bbox3d"
                and track.get("world_valid") is True
                and isinstance(track.get("world"), (list, tuple))
            )
            if has_bbox3d_world:
                if self.bev_calibration is not None:
                    self._refine_seeded_world_with_ground_state(
                        sensor_id,
                        camera_id,
                        track,
                        world_source_label="bbox3d",
                        world_now_ts=world_now_ts,
                    )
                else:
                    self._invalidate_world_track(
                        track,
                        "calibration_unavailable",
                    )
                return
            # V3DT is its own source of shared metric world truth. Missing or
            # invalid bbox3d metadata must remain visibly absent rather than
            # being replaced by the baseline ray-plane/depth estimator.
            for field_name in (
                "world",
                "world_valid",
                "world_quality",
                "world_quality_reason",
                "world_frame",
                "world_source",
            ):
                track.pop(field_name, None)
            return
        if self.bev_calibration is None:
            if track.get("world") is not None or track.get("world_valid") is True:
                self._invalidate_world_track(track, "calibration_unavailable")
            return
        if track.get("world_estimator_evaluated") is True:
            return
        if track.get("world_source") == "bbox3d":
            if track.get("world_valid") is True:
                self._refine_seeded_world_with_ground_state(
                    sensor_id,
                    camera_id,
                    track,
                    world_source_label="bbox3d",
                    world_now_ts=world_now_ts,
                )
            return
        if track.get("world") is not None and track.get("world_valid") is True:
            # Do not trust an arbitrary pre-seeded row. Route it through the
            # same revision-bound calibration, range, and PersonGroundState
            # admission as bbox3d seeds.
            self._refine_seeded_world_with_ground_state(
                sensor_id,
                camera_id,
                track,
                world_source_label=str(track.get("world_source") or "preseeded_world"),
                world_now_ts=world_now_ts,
            )
            return

        try:
            calib = self._world_calibration_snapshot(sensor_id, camera_id)
            if calib is None or calib.intrinsics is None or calib.extrinsics_col_major is None:
                return
            world_frame_id, world_frame_revision, world_transform_sha256 = (
                world_frame_binding_from_calibration(
                    calib,
                    default_frame_id=self._world_frame,
                )
            )

            bbox = track.get("bbox")
            if not bbox or len(bbox) < 4:
                return
            track_image_size = self._normalize_image_size(track.get("image_size") or track.get("frame_size"))
            calib_image_size = self._normalize_image_size(getattr(calib, "image_size", None))
            bbox_project = self._scale_bbox_to_image_size(bbox, track_image_size, calib_image_size)
            if not bbox_project or len(bbox_project) < 4:
                return

            flip_u, flip_v = self._infer_image_flips(camera_id, calib)
            now_ts = (
                float(world_now_ts)
                if world_now_ts is not None
                else self._world_timestamp_for_frame(
                    int(sensor_id),
                    media_pts_ns=track.get("media_pts_ns", 0),
                    observed_ts=float(time.time()),
                )
            )
            world_key = self._world_track_key(sensor_id, track)
            output_watermark_key = self._world_output_watermark_key(
                sensor_id,
                track,
                world_frame_id=world_frame_id,
                world_frame_revision=world_frame_revision,
                world_transform_sha256=world_transform_sha256,
            )
            self._maybe_prune_world_state(now_ts)
            state: Optional[_WorldAnchorState] = None
            if world_key is not None:
                state, restored = self._world_state_for_observation(
                    world_key,
                    now_ts=now_ts,
                    bbox=bbox,
                    lifecycle_generation=(
                        int(track.get("tracker_lifecycle_generation"))
                        if track.get("tracker_lifecycle_generation") is not None
                        else None
                    ),
                )
                if restored:
                    track["world_state_continuity"] = "restored_short_ghost"
                state.ts = float(now_ts)
                if bind_world_frame(
                    state,
                    world_frame_id=world_frame_id,
                    world_frame_revision=world_frame_revision,
                    world_transform_sha256=world_transform_sha256,
                ):
                    # A calibration revision is a physical discontinuity even
                    # when the numeric tracker ID is unchanged. Clear all
                    # holds/predictions before evaluating this frame.
                    track["world_state_continuity"] = "reset_world_frame"
                    state.ts = float(now_ts)
                if self._restore_world_output_watermark(
                    state,
                    output_watermark_key,
                ):
                    track.setdefault(
                        "world_state_continuity",
                        "restored_lifecycle_output",
                    )

            try:
                current_frame_id = int(track.get("frame_id", -1))
            except Exception:
                current_frame_id = -1
            bbox_stationary_supported = bool(
                state is not None
                and observe_bbox_stationarity(
                    state,
                    frame_id=current_frame_id,
                    bbox=bbox,
                    config=self._human_ground_cfg,
                )
            )

            if pose_kpts_abs is None:
                pose_kpts_abs = self._extract_pose_keypoints_for_anchor(obj_meta, bbox)
            pose_kpts_project = self._scale_pose_keypoints_to_image_size(
                pose_kpts_abs,
                track_image_size,
                calib_image_size,
            )

            previous_posture = (
                str(state.posture or "unknown")
                if state is not None
                else "unknown"
            )
            previous_motion = (
                str(state.motion_mode or "unknown")
                if state is not None
                else "unknown"
            )
            posture = classify_posture(
                kpts_abs=pose_kpts_abs,
                bbox=bbox,
                height_ref_scene=state.height_ref_scene if state is not None else None,
                config=self._human_ground_cfg,
            )
            # Pose can briefly lose its compact seated geometry while the
            # tracker remains on the same person.  Do not let that one-frame
            # ``unknown`` state reopen the standing leg-extension heuristic:
            # the extrapolated hip/knee point is not a ground contact.
            if posture == "unknown" and state is not None:
                held_non_upright = bool(
                    previous_posture in ("sitting", "lying")
                    and float(state.last_non_upright_ts) >= 0.0
                    and float(now_ts) - float(state.last_non_upright_ts)
                    <= float(self._human_ground_cfg.occlusion_upright_memory_s)
                )
                if previous_motion == "sit" or held_non_upright:
                    posture = "sitting"
                elif previous_motion == "lie":
                    posture = "lying"
            occlusion_assessment = (
                assess_lower_body_occlusion(
                    state,
                    kpts_abs=pose_kpts_abs,
                    bbox=bbox,
                    posture=posture,
                    now_ts=float(now_ts),
                    config=self._human_ground_cfg,
                )
                if state is not None
                else None
            )
            force_occlusion_gravity = bool(
                occlusion_assessment is not None
                and occlusion_assessment.active
                and state is not None
                and state.height_ref_scene is not None
            )
            if force_occlusion_gravity:
                posture = "standing"
            if state is not None:
                state.posture = str(posture)
            stationary_hold_eligible = bool(
                state is not None
                and state.last_good_world is not None
                and bbox_stationary_supported
                and posture in ("sitting", "lying")
            )

            pose_anchor = (
                self._resolve_pose_floor_anchor(pose_kpts_abs, posture=posture)
                if pose_kpts_abs is not None
                else None
            )
            person_anchor = self._resolve_person_depth_anchor(depth_result)
            depth_reports_no_ground_contact = bool(
                depth_result is not None
                and str(depth_result.status or "") == "no_ground_contact"
            )
            if (
                pose_anchor is not None
                and str(pose_anchor.source) == "pose_leg_floor"
                and (
                    posture in ("sitting", "lying")
                    or depth_reports_no_ground_contact
                )
            ):
                # The source is an extrapolated ankle, not an observed contact.
                # Keep any independently supported person-mask/ankle depth
                # candidate available, but never floor-project this estimate
                # for a known non-upright person.
                pose_anchor = None
            if pose_anchor is None:
                anchor_candidate = person_anchor
            else:
                anchor_candidate = pose_anchor

            contact_basis = (
                str(anchor_candidate.contact_basis or anchor_candidate.source)
                if anchor_candidate is not None
                else None
            )
            image_motion_supported = False
            if state is not None:
                anchor_is_current = bool(
                    anchor_candidate is not None
                    and not force_occlusion_gravity
                    and (
                        pose_anchor is not None
                        or self._depth_measurement_is_current(depth_result, track=track)
                    )
                )
                try:
                    current_frame_id = int(track.get("frame_id", -1))
                except Exception:
                    current_frame_id = -1
                image_motion_supported = observe_coherent_image_motion(
                    state,
                    frame_id=current_frame_id,
                    image_foot_uv=(
                        (float(anchor_candidate.u), float(anchor_candidate.v))
                        if anchor_is_current and anchor_candidate is not None
                        else None
                    ),
                    bbox=bbox if anchor_is_current else None,
                    contact_basis=contact_basis if anchor_is_current else None,
                    config=self._human_ground_cfg,
                )

            hit: Optional[np.ndarray] = None
            floor_candidate: Optional[np.ndarray] = None
            quality = "invalid"
            quality_reason: Optional[str] = "no_floor_intersection"
            world_source: Optional[str] = None
            source_measurement_rejected = False
            reject_current_geometry = False
            floor_ray_admitted = True
            floor_ray_rejection_reason: Optional[str] = None
            depth_measurement_not_current = False
            depth_observation = _DepthObservationResult(
                None,
                0.0,
                "depth_anchor_unavailable",
            )
            depth_obs: Optional[np.ndarray] = None
            depth_weight = 0.0
            depth_reason = "depth_anchor_unavailable"
            pose_uv_for_candidate: Optional[Tuple[float, float]] = None
            universal_candidates: List[Dict[str, Any]] = []
            canonical_resolved: Optional[ResolvedGroundMeasurement] = None
            canonical_resolved_point: Optional[np.ndarray] = None
            occlusion_fraction = (
                0.75
                if force_occlusion_gravity
                else (
                    0.35
                    if occlusion_assessment is not None and occlusion_assessment.active
                    else 0.0
                )
            )

            # Lower-body occlusion disables floor-ray construction, but it
            # must not discard an independently valid registered-depth
            # observation.  The resolver can adjudicate that depth sample
            # against the separate gravity reconstruction hypothesis.
            if force_occlusion_gravity and person_anchor is not None:
                depth_observation = self._depth_observation_from_anchor(
                    calib=calib,
                    anchor=person_anchor,
                    depth_result=depth_result,
                    flip_u=flip_u,
                    flip_v=flip_v,
                    track=track,
                )
                depth_obs = depth_observation.world_point
                depth_weight = float(depth_observation.weight)
                depth_reason = str(depth_observation.reason)
                depth_measurement_not_current = (
                    depth_reason == "depth_measurement_not_current"
                )
                registration_status = depth_observation.registration_status
                reject_current_geometry = bool(
                    registration_status is not None
                    and registration_status not in ("ok", "raw_passthrough")
                )
                if depth_observation.raw_depth_m is not None:
                    track["depth_anchor_m"] = float(depth_observation.raw_depth_m)
                track["depth_registered_m"] = (
                    float(depth_observation.registered_depth_m)
                    if depth_observation.registered_depth_m is not None
                    else None
                )
                track["depth_used_m"] = track.get("depth_registered_m")
                track["depth_registration_status"] = depth_observation.registration_status
                track["depth_registration_id"] = depth_observation.registration_id
                if depth_obs is not None:
                    track["world_depth_candidate"] = [
                        float(depth_obs[0]),
                        float(depth_obs[1]),
                        float(depth_obs[2]),
                    ]
                    if not self._admit_world_observation_range(
                        camera_id,
                        calib=calib,
                        world_candidate=np.asarray(depth_obs, dtype=np.float64),
                        track=track,
                    ):
                        track["world_depth_rejection_reason"] = str(
                            track.get("world_observation_range_rejection_reason")
                            or "world_observation_range_exceeded"
                        )
                        depth_obs = None
                        depth_weight = 0.0
                        depth_reason = str(track["world_depth_rejection_reason"])

            if anchor_candidate is not None and not force_occlusion_gravity:
                track["image_foot"] = [float(anchor_candidate.u), float(anchor_candidate.v)]
                pose_u, pose_v = self._scale_uv_to_image_size(
                    float(anchor_candidate.u),
                    float(anchor_candidate.v),
                    track_image_size,
                    calib_image_size,
                )
                # Keep the accepted image foot in the calibration raster. It
                # is the predictor's physical image basis; ``track.image_foot``
                # remains in the source-track raster for OSD compatibility.
                track["_world_current_image_foot_calib"] = [
                    float(pose_u),
                    float(pose_v),
                ]
                pose_uv_for_candidate = (float(pose_u), float(pose_v))
                hit = (
                    self._project_pixel_to_floor_world(
                        calib,
                        float(pose_u),
                        float(pose_v),
                        flip_u=flip_u,
                        flip_v=flip_v,
                    )
                    if self._anchor_is_verified_ground_contact(anchor_candidate)
                    else None
                )
                # A registered range is valid only at the UV/support
                # location that produced that range.  Pose and depth
                # anchors are independent hypotheses; never project a
                # person-mask range through the pose ankle UV merely
                # because pose was selected as the floor anchor.  Extract
                # this candidate independently of floor-ray success so a
                # failed floor intersection cannot suppress valid depth.
                depth_observation = (
                    self._depth_observation_from_anchor(
                        calib=calib,
                        anchor=person_anchor,
                        depth_result=depth_result,
                        flip_u=flip_u,
                        flip_v=flip_v,
                        track=track,
                    )
                    if person_anchor is not None
                    else _DepthObservationResult(
                        None,
                        0.0,
                        "depth_anchor_unavailable",
                    )
                )
                depth_obs = depth_observation.world_point
                depth_weight = float(depth_observation.weight)
                depth_reason = str(depth_observation.reason)
                depth_measurement_not_current = (
                    depth_reason == "depth_measurement_not_current"
                )
                registration_status = depth_observation.registration_status
                reject_current_geometry = bool(
                    registration_status is not None
                    and registration_status not in ("ok", "raw_passthrough")
                )
                if depth_observation.raw_depth_m is not None:
                    track["depth_anchor_m"] = float(depth_observation.raw_depth_m)
                track["depth_registered_m"] = (
                    float(depth_observation.registered_depth_m)
                    if depth_observation.registered_depth_m is not None
                    else None
                )
                track["depth_used_m"] = track.get("depth_registered_m")
                track["depth_registration_status"] = depth_observation.registration_status
                track["depth_registration_id"] = depth_observation.registration_id
                if depth_obs is not None:
                    track["world_depth_candidate"] = [
                        float(depth_obs[0]),
                        float(depth_obs[1]),
                        float(depth_obs[2]),
                    ]
                    if not self._admit_world_observation_range(
                        camera_id,
                        calib=calib,
                        world_candidate=np.asarray(depth_obs, dtype=np.float64),
                        track=track,
                    ):
                        track["world_depth_rejection_reason"] = str(
                            track.get("world_observation_range_rejection_reason")
                            or "world_observation_range_exceeded"
                        )
                        depth_obs = None
                        depth_weight = 0.0
                        depth_reason = str(track["world_depth_rejection_reason"])
                if hit is not None:
                    floor_candidate = np.asarray(hit, dtype=np.float64).copy()
                    track["world_floor_candidate"] = [
                        float(floor_candidate[0]),
                        float(floor_candidate[1]),
                        float(floor_candidate[2]),
                    ]
                    base_floor_weight = 1.0 if anchor_candidate.quality == "good" else 0.75
                    floor_weight = 0.0
                    effective_depth_weight = float(depth_weight)
                    floor_only_allowed = False
                    if not self._world_resolver_enabled:
                        floor_weight, effective_depth_weight, floor_only_allowed = self._world_fusion_weights(
                            camera_id,
                            floor_weight=base_floor_weight,
                            depth_weight=depth_weight,
                            track=track,
                        )
                    floor_ray_admitted = self._admit_floor_ray_range(
                        camera_id,
                        calib=calib,
                        floor_candidate=floor_candidate,
                        track=track,
                        anchor_uv=(float(pose_u), float(pose_v)),
                        bbox=bbox_project,
                        flip_u=flip_u,
                        flip_v=flip_v,
                    )
                    if not floor_ray_admitted:
                        floor_ray_rejection_reason = str(
                            track.get("world_floor_rejection_reason")
                            or "floor_ray_geometry_invalid"
                        )
                        floor_weight = 0.0
                        floor_only_allowed = False
                        track["world_floor_weight_effective"] = 0.0
                    if reject_current_geometry:
                        # Reject the unusable metric range without discarding an
                        # independently permitted floor-ray observation.  The
                        # per-camera policy decides whether floor-only geometry
                        # is admissible; depth-required cameras still fail
                        # closed below.  Do not seed height-lock state from a
                        # sample whose metric registration was rejected.
                        quality_reason = (
                            f"anchor={anchor_candidate.source},depth={depth_reason},"
                            f"depth_registration={registration_status}"
                        )
                    elif (
                        state is not None
                        and floor_ray_admitted
                        and anchor_candidate.height_lock_eligible
                        and posture in ("standing", "unknown")
                    ):
                        self._maybe_update_world_height_reference(
                            state,
                            calib,
                            bbox_project,
                            hit,
                            flip_u=flip_u,
                            flip_v=flip_v,
                            pose_kpts_abs=pose_kpts_project,
                        )
                    if not self._world_resolver_enabled and not reject_current_geometry and depth_obs is not None and effective_depth_weight > 0.0:
                        fused = np.array(
                            [float(depth_obs[0]), float(calib.floor_y), float(depth_obs[2])],
                            dtype=np.float64,
                        )
                        depth_anchor_source = str(depth_result.anchor_source or "") if depth_result is not None else ""
                        alpha_boost = 0.10 if depth_anchor_source == "lower_body_band" else 0.05
                        if str(anchor_candidate.source) == "person_mask_floor":
                            alpha_boost = max(0.0, float(alpha_boost) - 0.02)
                        alpha = min(1.0, float(self._world_smooth_alpha_good) + float(alpha_boost))
                        beta = max(0.0, min(1.0, float(alpha) * 0.25))
                        quality = "good" if anchor_candidate.quality == "good" else "estimated"
                        if floor_weight <= 0.0:
                            world_source = "pose_depth_only" if pose_anchor is not None else "person_anchor_depth_only"
                        else:
                            world_source = "pose_depth_fused" if pose_anchor is not None else "person_anchor_depth_fused"
                        if self._admit_live_world_source(
                            state,
                            candidate_source=world_source,
                            quality=quality,
                            depth_weight=float(effective_depth_weight),
                            posture=str(posture),
                        ):
                            hit = self._update_track_world_state(
                                track,
                                state,
                                measurement=fused,
                                floor_y=float(calib.floor_y),
                                now_ts=float(now_ts),
                                alpha=alpha,
                                beta=beta,
                                quality=quality,
                                contact_basis=contact_basis,
                                image_motion_supported=image_motion_supported,
                            )
                        else:
                            hit = None
                            source_measurement_rejected = True
                        depth_support_count = int(
                            depth_result.anchor_sample_count
                            if depth_result is not None and depth_result.anchor_sample_count is not None
                            else (depth_result.sample_count if depth_result is not None else 0)
                        )
                        depth_support_fraction = float(
                            depth_result.anchor_valid_fraction
                            if depth_result is not None and depth_result.anchor_valid_fraction is not None
                            else (depth_result.valid_fraction if depth_result is not None else 0.0)
                        )
                        quality_reason = (
                            f"anchor={anchor_candidate.source},depth_anchor={depth_anchor_source or 'none'},"
                            f"depth_samples={depth_support_count},depth_valid={depth_support_fraction:.2f}"
                        )
                        registration_status = track.get("depth_registration_status")
                        if registration_status:
                            quality_reason = f"{quality_reason},depth_registration={registration_status}"
                    elif floor_weight > 0.0 and floor_only_allowed:
                        alpha = float(self._world_smooth_alpha_good if anchor_candidate.quality == "good" else self._world_smooth_alpha_weak)
                        beta = max(0.0, min(1.0, float(alpha) * 0.20))
                        quality = "good" if anchor_candidate.quality == "good" else "estimated"
                        world_source = "pose_floor_only" if pose_anchor is not None else "person_anchor_floor_only"
                        floor_measurement = np.asarray(hit, dtype=np.float64)
                        if self._admit_live_world_source(
                            state,
                            candidate_source=world_source,
                            quality=quality,
                            depth_weight=0.0,
                            posture=str(posture),
                        ):
                            hit = self._update_track_world_state(
                                track,
                                state,
                                measurement=floor_measurement,
                                floor_y=float(calib.floor_y),
                                now_ts=float(now_ts),
                                alpha=alpha,
                                beta=beta,
                                quality=quality,
                                contact_basis=contact_basis,
                                image_motion_supported=image_motion_supported,
                            )
                        else:
                            hit = None
                            source_measurement_rejected = True
                        quality_reason = f"anchor={anchor_candidate.source},depth={depth_reason}"
                        registration_status = track.get("depth_registration_status")
                        if registration_status:
                            quality_reason = f"{quality_reason},depth_registration={registration_status}"
                    else:
                        hit = None
                        source_measurement_rejected = True
                        quality = "invalid"
                        if not floor_ray_admitted:
                            quality_reason = floor_ray_rejection_reason
                        elif not reject_current_geometry:
                            quality_reason = (
                                "depth_measurement_not_current"
                                if depth_measurement_not_current
                                else (
                                    "world_observation_range_exceeded"
                                    if depth_reason == "world_observation_range_exceeded"
                                    else "fusion_policy_requires_registered_depth"
                                )
                            )
                        if depth_measurement_not_current and state is not None:
                            mark_world_measurement_unavailable(
                                state,
                                reason="depth_measurement_not_current",
                            )
                else:
                    hit = None

            if self._world_resolver_enabled:
                self._append_universal_world_candidates(
                    universal_candidates,
                    sensor_id=int(sensor_id),
                    track=track,
                    camera_id=camera_id,
                    calib=calib,
                    floor_candidate=floor_candidate,
                    floor_ray_admitted=bool(floor_ray_admitted),
                    anchor_candidate=(
                        anchor_candidate
                        if anchor_candidate is not None and not force_occlusion_gravity
                        else None
                    ),
                    pose_uv=pose_uv_for_candidate,
                    contact_basis=contact_basis,
                    posture=str(posture),
                    occlusion_fraction=float(occlusion_fraction),
                    image_motion_supported=bool(image_motion_supported),
                    depth_obs=depth_obs,
                    depth_weight=float(depth_weight),
                    depth_observation=depth_observation,
                    depth_result=depth_result,
                    person_anchor=person_anchor,
                    reject_current_geometry=bool(reject_current_geometry),
                    flip_u=flip_u,
                    flip_v=flip_v,
                )
                quality_reason = (
                    f"universal_candidates={len(universal_candidates)},"
                    f"depth={depth_reason}"
                )
                # Keep ``hit`` unset until the resolver has returned a typed
                # result.  PersonGroundState remains the only temporal filter
                # after that result is selected.
                hit = None
                # Gravity reconstruction is an additional, deliberately weak
                # hypothesis for upright lower-body occlusion or a missing
                # current metric contact.  It remains separate from the
                # process prediction/hold handled below.
                allow_gravity_hypothesis = posture not in ("sitting", "lying")
                if depth_reports_no_ground_contact:
                    # ``no_ground_contact`` rejects the depth capsule and any
                    # extrapolated leg-floor anchor.  It does not invalidate
                    # an independently learned upright body-scale hypothesis.
                    # Require positive upright/motion/occlusion evidence so an
                    # unknown seated newcomer can never be gravity-dropped to
                    # the floor merely because its ankles are unavailable.
                    allow_gravity_hypothesis = bool(
                        posture == "standing"
                        or force_occlusion_gravity
                        or (
                            state is not None
                            and str(state.motion_mode) == "walk"
                        )
                    )
                if state is not None and str(state.motion_mode) in ("sit", "lie"):
                    allow_gravity_hypothesis = False
                if (
                    allow_gravity_hypothesis
                    and state is not None
                    and state.height_ref_scene is not None
                    and (force_occlusion_gravity or not universal_candidates)
                ):
                    gravity_candidate = self._gravity_drop_world(
                        calib,
                        bbox_project,
                        float(state.height_ref_scene),
                        flip_u=flip_u,
                        flip_v=flip_v,
                        pose_kpts_abs=pose_kpts_project,
                        state=state,
                    )
                    if gravity_candidate is not None and self._admit_world_observation_range(
                        camera_id,
                        calib=calib,
                        world_candidate=np.asarray(gravity_candidate, dtype=np.float64),
                        track=track,
                    ):
                        try:
                            gravity_covariance = Matrix3(
                                values=(
                                    0.64,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.25,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.64,
                                )
                            )
                        except Exception:
                            gravity_covariance = None
                        if gravity_covariance is not None:
                            universal_candidates.append(
                                {
                                    "candidate_id": "gravity_reconstruction",
                                    "kind": "gravity_reconstruction",
                                    "position": np.asarray(gravity_candidate, dtype=np.float64),
                                    "covariance": gravity_covariance,
                                    "anchor": "learned_body_height",
                                    "contact_basis": "gravity_drop",
                                    "confidence": 0.45,
                                    "posture": str(posture),
                                    "occlusion": float(occlusion_fraction),
                                    "motion_consistency": 0.65,
                                    "support_score": 0.45,
                                    "posture_compatibility": 1.0,
                                    "source_reliability": 0.80,
                                    "correlation_group": f"gravity_reconstruction:{int(sensor_id)}:{int(track.get('tracker_id', -1))}",
                                    "pcf": self._world_prior_evidence(camera_id, gravity_candidate),
                                }
                            )

                cohort = self._world_measurement_cohort(
                    sensor_id,
                    camera_id,
                    track,
                    calib=calib,
                    world_frame_revision=world_frame_revision,
                    world_transform_sha256=world_transform_sha256,
                )
                measurement_set = self._build_world_measurement_set(
                    cohort=cohort,
                    candidates=universal_candidates,
                )
                if measurement_set is None:
                    raise RuntimeError("unable to construct exact world measurement cohort")
                resolved_hit, resolved = self._apply_resolved_world_measurement(
                    track,
                    measurement_set,
                )
                canonical_resolved = resolved
                canonical_resolved_point = resolved_hit
                if (
                    resolved_hit is not None
                    and resolved is not None
                    and str(resolved.quality) != "weak"
                ):
                    selected_kind = str(resolved.selected_kind or "")
                    contributor_kinds = {
                        str(candidate.kind)
                        for candidate in measurement_set.hypotheses
                        if candidate.candidate_id in set(resolved.contributor_ids)
                    }
                    if resolved.fused and {"floor_ray", "registered_depth"}.issubset(contributor_kinds):
                        world_source = "pose_depth_fused" if pose_anchor is not None else "person_anchor_depth_fused"
                    elif selected_kind == "registered_depth":
                        world_source = "person_anchor_depth_only"
                    elif selected_kind == "floor_ray":
                        world_source = "pose_floor_only" if pose_anchor is not None else "person_anchor_floor_only"
                    elif selected_kind == "gravity_reconstruction":
                        world_source = "gravity_drop"
                    else:
                        world_source = "pose_floor_only" if pose_anchor is not None else "person_anchor_floor_only"
                    quality = str(resolved.quality)
                    if quality not in ("good", "estimated"):
                        quality = "estimated"
                    depth_weight_for_admission = float(
                        next(
                            (
                                candidate.get("confidence", 0.0)
                                for candidate in universal_candidates
                                if candidate.get("kind") == "registered_depth"
                            ),
                            0.0,
                        )
                    )
                    self._observe_resolver_source_continuity(
                        track,
                        state,
                        candidate_source=world_source,
                        quality=quality,
                        depth_weight=depth_weight_for_admission,
                        posture=str(posture),
                    )
                    hit = self._update_track_world_state(
                        track,
                        state,
                        measurement=resolved_hit,
                        floor_y=float(calib.floor_y),
                        now_ts=float(now_ts),
                        alpha=(
                            float(self._world_smooth_alpha_good)
                            if quality == "good"
                            else float(self._world_smooth_alpha_weak)
                        ),
                        beta=0.15,
                        quality=quality,
                        contact_basis=(
                            str(track.get("world_resolver_contact_basis"))
                            if track.get("world_resolver_contact_basis")
                            else contact_basis
                        ),
                        image_motion_supported=image_motion_supported,
                    )
                    quality_reason = str(resolved.reason)
                elif resolved_hit is not None and resolved is not None:
                    # A weak resolver result remains visible in the bounded
                    # diagnostic payload, but it is not a current metric
                    # observation.  Leave PersonGroundState untouched so the
                    # continuation path below can emit its existing bounded
                    # prediction/hold instead of training on weak geometry.
                    hit = None
                    source_measurement_rejected = True
                    quality = "invalid"
                    quality_reason = "world_resolver_weak_measurement"
                else:
                    hit = None
                    source_measurement_rejected = True
                    quality = "invalid"
                    quality_reason = (
                        str(resolved.reason)
                        if resolved is not None
                        else "world_resolver_rejected_measurement"
                    )

            # Gravity-drop assumes upright height. Skip confirmed non-upright
            # postures/motion modes. A short box alone can be lower-body
            # occlusion of a standing person — height lock is exactly for that case.
            allow_gravity = posture not in ("sitting", "lying")
            if depth_reports_no_ground_contact:
                # Native body-depth samples without a contact band are not a
                # valid floor constraint; do not turn the bbox/height prior
                # into a second hallucinated floor anchor.
                allow_gravity = False
            if (
                not force_occlusion_gravity
                and state is not None
                and str(state.motion_mode) in ("sit", "lie")
            ):
                allow_gravity = False
            gravity_fallback_requested = bool(
                not source_measurement_rejected
                and hit is None
                and not reject_current_geometry
            )
            if (
                not self._world_resolver_enabled
                and (
                (force_occlusion_gravity or gravity_fallback_requested)
                and allow_gravity
                and state is not None
                and state.height_ref_scene is not None
                )
            ):
                gravity_hit = self._gravity_drop_world(
                    calib,
                    bbox_project,
                    float(state.height_ref_scene),
                    flip_u=flip_u,
                    flip_v=flip_v,
                    pose_kpts_abs=pose_kpts_project,
                    state=state,
                )
                if gravity_hit is not None:
                    gravity_admitted = self._admit_world_observation_range(
                        camera_id,
                        calib=calib,
                        world_candidate=np.asarray(gravity_hit, dtype=np.float64),
                        track=track,
                    )
                    if not gravity_admitted:
                        quality_reason = str(
                            track.get("world_observation_range_rejection_reason")
                            or "world_observation_range_exceeded"
                        )
                    if force_occlusion_gravity:
                        gravity_admitted = bool(
                            gravity_admitted
                            and self._admit_live_world_source(
                                state,
                                candidate_source="gravity_drop",
                                quality="estimated",
                                depth_weight=0.0,
                                posture="standing",
                                authoritative=True,
                            )
                        )
                    if gravity_admitted:
                        hit = self._update_track_world_state(
                            track,
                            state,
                            measurement=gravity_hit,
                            floor_y=float(calib.floor_y),
                            now_ts=float(now_ts),
                            alpha=float(self._world_smooth_alpha_weak),
                            beta=max(
                                0.0,
                                min(
                                    1.0,
                                    float(self._world_smooth_alpha_weak) * 0.15,
                                ),
                            ),
                            quality="estimated",
                            contact_basis="gravity_drop",
                            image_motion_supported=False,
                        )
                        world_source = "gravity_drop"
                        quality = "estimated"
                        if force_occlusion_gravity and occlusion_assessment is not None:
                            quality_reason = (
                                "lower_body_occlusion="
                                f"{occlusion_assessment.level}"
                            )
                            if occlusion_assessment.reason:
                                quality_reason = (
                                    f"{quality_reason},"
                                    f"evidence={occlusion_assessment.reason}"
                                )
                        else:
                            quality_reason = (
                                "current_anchor_unavailable"
                                if anchor_candidate is None
                                else "current_anchor_projection_failed"
                            )

            fallback_reason = self._fallback_quality_reason(pose_kpts_abs, pose_anchor, person_anchor, depth_result, state)
            if source_measurement_rejected:
                if self._world_resolver_enabled:
                    fallback_reason = str(
                        quality_reason or "world_resolver_rejected_measurement"
                    )
                else:
                    fallback_reason = (
                        quality_reason
                        if quality_reason in (
                            "fusion_policy_requires_registered_depth",
                            "depth_measurement_not_current",
                            "world_observation_range_exceeded",
                            "floor_ray_range_exceeded",
                            "floor_ray_geometry_invalid",
                            "bbox_contact_gap_near_horizon",
                            "bbox_contact_range_disagreement_near_horizon",
                        )
                        else "source_hysteresis_rejected_current_measurement"
                    )

            measurement_rejected = bool(
                state is not None
                and "world_prefilter_measurement" in track
                and not state.measurement_accepted
            )
            # Projective continuation is evaluated from the last accepted
            # separate material-motion bit so a failed transport cannot fall
            # through to a frozen anchor hold after a real bbox displacement.
            projective_prediction: Optional[Tuple[np.ndarray, Dict[str, Any]]] = None
            bbox_motion_material = False
            if state is not None and state.last_accepted_bbox_geometry is not None:
                try:
                    previous_bbox = state.last_accepted_bbox_geometry
                    current_left, current_top, current_width, current_height = (
                        float(value) for value in bbox_project[:4]
                    )
                    previous_center = (
                        float(previous_bbox[0]) + float(previous_bbox[2]) * 0.5,
                        float(previous_bbox[1]) + float(previous_bbox[3]) * 0.5,
                    )
                    current_center = (
                        current_left + current_width * 0.5,
                        current_top + current_height * 0.5,
                    )
                    bbox_motion_material = bool(
                        math.hypot(
                            current_center[0] - previous_center[0],
                            current_center[1] - previous_center[1],
                        )
                        > max(8.0, float(self._human_ground_cfg.static_px_threshold) * 2.0)
                    )
                except Exception:
                    bbox_motion_material = True
            if state is not None and (measurement_rejected or hit is None):
                projective_prediction = self._project_image_motion_prediction(
                    camera_id=camera_id,
                    state=state,
                    bbox_project=bbox_project,
                    calib=calib,
                    now_ts=float(now_ts),
                    lifecycle_generation=(
                        int(track.get("tracker_lifecycle_generation"))
                        if track.get("tracker_lifecycle_generation") is not None
                        else None
                    ),
                    flip_u=flip_u,
                    flip_v=flip_v,
                    track=track,
                )

            # A projective point is allowed to influence the canonical state
            # only when the calibrated camera policy permits weak continuation
            # and this frame has no current metric candidate or its candidate
            # was rejected.  It is still passed through PersonGroundState's
            # physical CV gate; the old display-only branch is gone.
            projective_integrated = False
            projective_rejection_reason: Optional[str] = None
            projective_origin_reason = (
                str(state.measurement_rejection_reason)
                if (
                    state is not None
                    and measurement_rejected
                    and state.measurement_rejection_reason
                )
                else None
            )
            projective_allowed = self._projective_continuation_allowed(
                camera_id,
                track,
            )
            if (
                state is not None
                and projective_prediction is not None
                and projective_allowed
                and (
                    anchor_candidate is None
                    or depth_measurement_not_current
                    or source_measurement_rejected
                    or measurement_rejected
                )
            ):
                projective_point, prediction_provenance = projective_prediction
                integrated = integrate_projective_ground_observation(
                    state,
                    measurement=projective_point,
                    floor_y=float(calib.floor_y),
                    now_ts=float(now_ts),
                    config=self._human_ground_cfg,
                )
                if integrated is not None:
                    hit = np.asarray(integrated, dtype=np.float64)
                    prediction_provenance = dict(prediction_provenance)
                    prediction_provenance["state_integrated"] = True
                    track["world_filter_prediction"] = [
                        float(hit[0]),
                        float(calib.floor_y),
                        float(hit[2]),
                    ]
                    track["world_prediction_provenance"] = prediction_provenance
                    track["world_prediction_image_foot"] = list(
                        prediction_provenance["image_foot"]
                    )
                    world_source = "image_motion_prediction"
                    quality = "estimated"
                    projective_reason = (
                        "depth_measurement_not_current"
                        if depth_measurement_not_current
                        else str(
                            projective_origin_reason
                            or fallback_reason
                            or "world_measurement_unavailable"
                        )
                    )
                    state.measurement_rejection_reason = projective_reason
                    quality_reason = (
                        f"{projective_reason},"
                        "predicted_from_accepted_image_motion"
                    )
                    projective_integrated = True
                else:
                    projective_rejection_reason = str(
                        state.measurement_rejection_reason
                        or "projective_observation_rejected"
                    )

            # A missing/filtered candidate must not inherit the previous
            # frame's ``measurement_accepted=True`` diagnostic.  Mark the
            # current frame explicitly unavailable before choosing the
            # bounded process-model fallback below.
            if state is not None and hit is None and not measurement_rejected:
                if projective_rejection_reason:
                    measurement_rejected = True
                else:
                    mark_world_measurement_unavailable(
                        state,
                        reason=(
                            "depth_measurement_not_current"
                            if depth_measurement_not_current
                            else str(fallback_reason or "world_measurement_unavailable")
                        ),
                    )
                    measurement_rejected = True

            if projective_integrated:
                # The canonical state/output has already been selected. Do
                # not reinterpret it as a held or display-only point below.
                measurement_rejected = False

            if measurement_rejected:
                quality_reason = str(
                    state.measurement_rejection_reason or "physical_measurement_rejected"
                )
                fallback_reason = quality_reason
                if stationary_hold_eligible:
                    quality_reason = f"{quality_reason},stationary_bbox_hold"
                hold_age = float(now_ts) - float(state.last_good_ts or 0.0)
                hold_ttl_s = (
                    float(self._world_stationary_hold_ttl_s)
                    if stationary_hold_eligible
                    else float(self._world_anchor_hold_ttl_s)
                )
                if (
                    state.last_good_world is not None
                    and hold_age <= hold_ttl_s
                ):
                    process_point = advance_human_cv_prediction(
                        state,
                        floor_y=float(calib.floor_y),
                        now_ts=float(now_ts),
                        config=self._human_ground_cfg,
                        reason=str(quality_reason),
                    )
                    if process_point is not None:
                        track["world_filter_prediction"] = [
                            float(process_point[0]),
                            float(calib.floor_y),
                            float(process_point[2]),
                        ]
                        if stationary_hold_eligible or (
                            int(current_frame_id) < 0 and not bbox_motion_material
                        ):
                            # ``advance_human_cv_prediction`` has already
                            # advanced the one canonical process state.  Do
                            # not replace that bounded posterior with the
                            # older ``last_good_world`` here: doing so makes
                            # a cv_prediction -> anchor_hold transition jump
                            # backwards by the entire rejected interval (and
                            # can exceed the physical speed bound).  The hold
                            # remains non-authoritative/non-appending through
                            # its source label and trail gate; only the
                            # accepted metric anchor remains unchanged.
                            hit = np.asarray(process_point, dtype=np.float64)
                            world_source = "anchor_hold"
                        else:
                            hit = np.asarray(process_point, dtype=np.float64)
                            world_source = "cv_prediction"
                        track["world_prediction_provenance"] = {
                            "type": "bounded_cv_process",
                            "non_authoritative": True,
                            "state_integrated": True,
                            "reason": str(quality_reason),
                        }
                        track["world_prediction_image_foot"] = list(
                            track.get("image_foot") or []
                        )
                    else:
                        hit = None
                        world_source = None
                    quality = "estimated"
                else:
                    hit = None
                    quality = "invalid"
                    world_source = None

            if (
                not measurement_rejected
                and hit is None
                and state is not None
                and state.last_good_world is not None
            ):
                hold_age = float(now_ts) - float(state.last_good_ts or 0.0)
                hold_ttl_s = (
                    float(self._world_stationary_hold_ttl_s)
                    if stationary_hold_eligible
                    else float(self._world_anchor_hold_ttl_s)
                )
                if hold_age <= hold_ttl_s:
                    process_point = advance_human_cv_prediction(
                        state,
                        floor_y=float(calib.floor_y),
                        now_ts=float(now_ts),
                        config=self._human_ground_cfg,
                        reason=(
                            "depth_measurement_not_current"
                            if depth_measurement_not_current
                            else str(fallback_reason or "world_measurement_unavailable")
                        ),
                    )
                    if process_point is not None:
                        track["world_filter_prediction"] = [
                            float(process_point[0]),
                            float(calib.floor_y),
                            float(process_point[2]),
                        ]
                        hit = np.asarray(process_point, dtype=np.float64)
                        world_source = "cv_prediction"
                        track["world_prediction_provenance"] = {
                            "type": "bounded_cv_process",
                            "non_authoritative": True,
                            "state_integrated": True,
                            "reason": str(
                                "depth_measurement_not_current"
                                if depth_measurement_not_current
                                else fallback_reason
                            ),
                        }
                        quality = "estimated"
                        if world_source == "anchor_hold" and not depth_measurement_not_current:
                            quality_reason = fallback_reason
                        if stationary_hold_eligible and world_source == "anchor_hold":
                            quality_reason = f"{quality_reason},stationary_bbox_hold"
                    else:
                        hit = None
                        world_source = None

            if hit is not None:
                # The final point may be a bounded CV/hold posterior rather
                # than the current metric sample. Re-admit the exact output
                # against the same active calibration before image reprojection
                # or BEV/trail publication. A prediction that leaves the
                # calibrated camera envelope is typed invalid; it is never
                # clipped, seeded as last-good, or appended to a trail.
                if not self._admit_world_observation_range(
                    camera_id,
                    calib=calib,
                    world_candidate=np.asarray(hit, dtype=np.float64),
                    track=track,
                ):
                    rejection_reason = str(
                        track.get("world_observation_range_rejection_reason")
                        or "world_observation_range_exceeded"
                    )
                    if state is not None:
                        mark_world_measurement_unavailable(
                            state,
                            reason=rejection_reason,
                        )
                        state.trail_append_allowed = False
                    track["world_estimator_evaluated"] = True
                    self._invalidate_world_track(track, rejection_reason)
                    if state is not None:
                        for key, value in state.as_public_fields().items():
                            if value is not None:
                                track[key] = value
                    return
                track["world_estimator_evaluated"] = True
                if self._world_resolver_enabled:
                    current_world_sources = {
                        "pose_floor_only",
                        "person_anchor_floor_only",
                        "pose_depth_fused",
                        "person_anchor_depth_fused",
                        "person_anchor_depth_only",
                        "gravity_drop",
                    }
                    accepted_current = bool(
                        canonical_resolved is not None
                        and canonical_resolved.status == "measured"
                        and world_source in current_world_sources
                        and (state is None or bool(state.measurement_accepted))
                    )
                else:
                    accepted_current = False
                if (
                    world_source == "gravity_drop"
                    and state is not None
                    and state.lower_body_occluded
                ):
                    self._set_track_image_base_from_world(
                        track,
                        calib=calib,
                        world_point=hit,
                        flip_u=flip_u,
                        flip_v=flip_v,
                    )
                    reconstructed_foot = track.get("image_base")
                    if (
                        isinstance(reconstructed_foot, (list, tuple))
                        and len(reconstructed_foot) >= 2
                    ):
                        track["image_foot"] = [
                            float(reconstructed_foot[0]),
                            float(reconstructed_foot[1]),
                        ]
                # Phase 1: motion mode / stationary lock using image foot + speed.
                image_uv = None
                raw_foot = track.get("image_foot")
                if isinstance(raw_foot, (list, tuple)) and len(raw_foot) >= 2:
                    try:
                        image_uv = (float(raw_foot[0]), float(raw_foot[1]))
                    except Exception:
                        image_uv = None
                if state is not None:
                    # Keep the bounded process posterior across the motion
                    # mode update.  ``update_motion_mode`` may enter/retain
                    # an idle-like mode and restore ``locked_world``; that
                    # lock can predate the current rejected-frame posterior.
                    # Saving this candidate lets the lock logic update its
                    # bookkeeping without replacing the point that is about
                    # to be emitted.
                    bounded_process_hit = (
                        np.asarray(hit, dtype=np.float64).copy()
                        if world_source
                        in ("cv_prediction", "image_motion_prediction", "anchor_hold")
                        else None
                    )
                    update_motion_mode(
                        state,
                        now_ts=float(now_ts),
                        image_foot_uv=image_uv,
                        config=self._human_ground_cfg,
                    )
                    if (
                        world_source == "cv_prediction"
                        and state.motion_mode in ("idle", "sit", "lie")
                    ):
                        # A prediction is only a continuity aid for an active
                        # track.  If current image evidence now classifies the
                        # person as stationary/seated/lying, downgrade to a
                        # true non-appending hold for this frame.
                        world_source = "anchor_hold"
                    if world_source == "anchor_hold":
                        state.trail_append_allowed = False
                    elif world_source in ("cv_prediction", "image_motion_prediction"):
                        state.trail_append_allowed = True
                    if bounded_process_hit is not None:
                        # A stale idle lock is a process/display state, not a
                        # new metric observation. Preserve the current bounded
                        # posterior and move the lock to that same point so a
                        # later frame cannot snap back to the old location.
                        hit = bounded_process_hit
                        state.world_x = float(hit[0])
                        state.world_z = float(hit[2])
                        if state.motion_mode in ("idle", "sit", "lie"):
                            state.locked_world = (
                                float(hit[0]),
                                float(hit[2]),
                            )
                    if state.motion_mode in ("idle", "sit", "lie") and state.locked_world is not None:
                        if bounded_process_hit is None:
                            hit = np.array(
                                [float(state.locked_world[0]), float(calib.floor_y), float(state.locked_world[1])],
                                dtype=np.float64,
                            )
                            state.world_x = float(state.locked_world[0])
                            state.world_z = float(state.locked_world[1])
                            state.vel_world_x = 0.0
                            state.vel_world_z = 0.0

                    hit, output_continuous = admit_human_ground_output(
                        state,
                        candidate=np.asarray(hit, dtype=np.float64),
                        floor_y=float(calib.floor_y),
                        now_ts=float(now_ts),
                        media_pts_ns=track.get("media_pts_ns", 0),
                        config=self._human_ground_cfg,
                    )
                    self._save_world_output_watermark(
                        state,
                        output_watermark_key,
                    )
                    if not output_continuous:
                        world_source = "anchor_hold"
                        quality = "held"
                        quality_reason = str(
                            state.measurement_rejection_reason
                            or "physical_output_continuity_exceeded"
                        )

                if world_source in (
                    "anchor_hold",
                    "cv_prediction",
                    "image_motion_prediction",
                ):
                    # Process continuation is a valid display point but not a
                    # current metric observation. Keep its quality explicit so
                    # downstream covariance fallback remains conservative.
                    quality = "held"
                if self._world_resolver_enabled:
                    accepted_current = bool(
                        canonical_resolved is not None
                        and canonical_resolved.status == "measured"
                        and world_source not in (
                            "anchor_hold",
                            "cv_prediction",
                            "image_motion_prediction",
                        )
                        and (state is None or bool(state.measurement_accepted))
                    )
                if accepted_current:
                    self._publish_filtered_world_covariance(
                        track,
                        resolved=canonical_resolved,
                        resolved_point=canonical_resolved_point,
                        emitted_point=np.asarray(hit, dtype=np.float64),
                    )
                else:
                    track.pop("world_covariance", None)
                self._set_track_image_base_from_world(track, calib=calib, world_point=hit, flip_u=flip_u, flip_v=flip_v)
                wx = float(hit[0])
                wy = float(hit[1])
                wz = float(hit[2])

                track["world"] = [float(wx), float(wy), float(wz)]
                track["world_valid"] = True
                track["world_quality"] = str(quality)
                if quality_reason:
                    track["world_quality_reason"] = str(quality_reason)
                else:
                    track.pop("world_quality_reason", None)
                track["world_frame"] = str(world_frame_id or self._world_frame)
                if world_frame_revision:
                    track["world_frame_revision"] = world_frame_revision
                else:
                    track.pop("world_frame_revision", None)
                if world_transform_sha256:
                    track["world_transform_sha256"] = world_transform_sha256
                else:
                    track.pop("world_transform_sha256", None)
                if world_source:
                    track["world_source"] = str(world_source)
                else:
                    track.pop("world_source", None)
                if state is not None:
                    for key, value in state.as_public_fields().items():
                        if value is not None:
                            track[key] = value
                    if world_source not in (
                        "anchor_hold",
                        "cv_prediction",
                        "image_motion_prediction",
                    ):
                        state.last_good_world = (float(wx), float(wy), float(wz))
                        state.last_good_ts = float(now_ts)
                        state.ts = float(now_ts)
                        # Store only the image basis that produced a physically
                        # accepted world point.  Predictions/holds never move
                        # this origin, which makes repeated rejected frames
                        # non-integrating by construction.
                        accepted_foot = track.get("_world_current_image_foot_calib")
                        if (
                            not isinstance(accepted_foot, (list, tuple))
                            or len(accepted_foot) < 2
                        ):
                            accepted_foot = track.get("image_base")
                        record_accepted_image_geometry(
                            state,
                            image_foot_uv=(
                                (float(accepted_foot[0]), float(accepted_foot[1]))
                                if isinstance(accepted_foot, (list, tuple))
                                and len(accepted_foot) >= 2
                                else None
                            ),
                            bbox=bbox_project,
                            now_ts=float(now_ts),
                            lifecycle_generation=(
                                int(track.get("tracker_lifecycle_generation"))
                                if track.get("tracker_lifecycle_generation") is not None
                                else None
                            ),
                        )
            else:
                track["world_estimator_evaluated"] = True
                track["world_valid"] = False
                track["world_quality"] = "invalid"
                track["world_quality_reason"] = str(fallback_reason or "no_floor_intersection")
                track.pop("world_source", None)
                track.pop("world_filter_prediction", None)
                track.pop("world_prediction_image_foot", None)
                track.pop("world_prediction_provenance", None)
                if state is not None:
                    state.trail_append_allowed = False
                    for key, value in state.as_public_fields().items():
                        if value is not None:
                            track[key] = value
        except Exception:
            # A calibration/refinement failure must not preserve arbitrary
            # pre-existing world_valid state or let a partial point cross the
            # publication boundary.
            logger.debug("World augmentation failed", exc_info=True)
            self._invalidate_world_track(track, "world_estimation_failed")
        finally:
            # Calibration-raster predictor state is an implementation detail,
            # not part of the public tracking contract.  Never let the
            # temporary underscore field cross the scalar publication boundary.
            track.pop("_world_current_image_foot_calib", None)

    def _publish_bev(
        self,
        sensor_id: int,
        camera_id: str,
        frame_meta: Any,
        footpoints: Sequence[Footpoint],
        *,
        now_ts: Optional[float] = None,
        track_count: Optional[int] = None,
        paired_with_tracking: bool = False,
        tracking_receipt: Optional[TrackingPublicationReceipt] = None,
        observed_at_us: Optional[int] = None,
        timestamp_us: Optional[int] = None,
        tracker_lifecycle_tombstones: Sequence[Mapping[str, Any]] = (),
    ) -> BevPublicationReceipt:
        if self.bev_renderer is None:
            raise RuntimeError("BEV publication requested without a renderer")
        if paired_with_tracking and not isinstance(
            tracking_receipt,
            TrackingPublicationReceipt,
        ):
            raise RuntimeError(
                "paired BEV publication requires a tracking admission receipt"
            )
        ts_now = float(now_ts) if now_ts is not None else time.time()
        ts_us = int(
            timestamp_us
            if timestamp_us is not None
            else self._frame_timestamp_us(frame_meta)
        )
        frame_id = int(
            _meta_lookup(
                frame_meta,
                "frame_number",
                "frame_num",
                default=0,
            )
            or 0
        )
        observed_at_us_value = max(1, int(ts_now * 1_000_000))
        if observed_at_us is not None:
            observed_at_us_value = max(1, int(observed_at_us))
        tracking_sequence = (
            tracking_receipt.tracking_publication_sequence
            if tracking_receipt is not None
            else None
        )
        tracking_submission_id = (
            tracking_receipt.outbound_submission_id
            if tracking_receipt is not None
            else None
        )
        fp_count = int(track_count) if track_count is not None else len(footpoints)
        if not paired_with_tracking:
            if not self._publish_gate_due(
                self._last_bev_publish_ts_by_sensor,
                self._last_bev_count_by_sensor,
                sensor_id=sensor_id,
                now_ts=ts_now,
                count=fp_count,
                interval_s=float(self._bev_publish_interval_s),
                counter_prefix="bev",
            ):
                raise RuntimeError(
                    "standalone BEV gate skipped a requested publication"
                )
        if self.bev_calibration is None:
            failure = self.bev_renderer.record_input_failure(
                camera_id,
                stage="calibration_provider",
                timestamp_us=ts_us,
                cause=RuntimeError("BEV calibration provider is unavailable"),
            )
            return BevPublicationReceipt(
                status="failed",
                camera_id=camera_id,
                source_id=int(sensor_id),
                frame_id=frame_id,
                observed_at_us=observed_at_us_value,
                tracking_publication_sequence=tracking_sequence,
                tracking_outbound_submission_id=tracking_submission_id,
                failure=failure,
            )
        try:
            calib = self._world_calibration_snapshot(sensor_id, camera_id)
        except Exception as exc:
            failure = self.bev_renderer.record_input_failure(
                camera_id,
                stage="calibration_snapshot",
                timestamp_us=ts_us,
                cause=exc,
            )
            return BevPublicationReceipt(
                status="failed",
                camera_id=camera_id,
                source_id=int(sensor_id),
                frame_id=frame_id,
                observed_at_us=observed_at_us_value,
                tracking_publication_sequence=tracking_sequence,
                tracking_outbound_submission_id=tracking_submission_id,
                failure=failure,
            )
        if calib is None:
            failure = self.bev_renderer.record_input_failure(
                camera_id,
                stage="calibration_snapshot",
                timestamp_us=ts_us,
                cause=LookupError("BEV calibration snapshot is unavailable"),
            )
            return BevPublicationReceipt(
                status="failed",
                camera_id=camera_id,
                source_id=int(sensor_id),
                frame_id=frame_id,
                observed_at_us=observed_at_us_value,
                tracking_publication_sequence=tracking_sequence,
                tracking_outbound_submission_id=tracking_submission_id,
                failure=failure,
            )
        try:
            publish_start_ns = time.perf_counter_ns()
            receipt = self.bev_renderer.render_and_publish(
                camera_id=camera_id,
                calib=calib,
                footpoints=list(footpoints),
                timestamp_us=ts_us,
                source_id=int(sensor_id),
                frame_id=frame_id,
                observed_at_us=observed_at_us_value,
                tracking_publication_sequence=tracking_sequence,
                tracking_outbound_submission_id=tracking_submission_id,
                tracker_lifecycle_tombstones=list(
                    tracker_lifecycle_tombstones
                ),
            )
            _record_core_stage_timing("bev.render_and_publish", publish_start_ns, item_count=len(footpoints))
        except Exception as exc:
            failure = self.bev_renderer.record_input_failure(
                camera_id,
                stage="render_call",
                timestamp_us=ts_us,
                cause=exc,
            )
            return BevPublicationReceipt(
                status="failed",
                camera_id=camera_id,
                source_id=int(sensor_id),
                frame_id=frame_id,
                observed_at_us=observed_at_us_value,
                tracking_publication_sequence=tracking_sequence,
                tracking_outbound_submission_id=tracking_submission_id,
                failure=failure,
            )
        if not isinstance(receipt, BevPublicationReceipt):
            failure = self.bev_renderer.record_input_failure(
                camera_id,
                stage="publication_receipt",
                timestamp_us=ts_us,
                cause=RuntimeError("BEV renderer returned no typed receipt"),
            )
            return BevPublicationReceipt(
                status="failed",
                camera_id=camera_id,
                source_id=int(sensor_id),
                frame_id=frame_id,
                observed_at_us=observed_at_us_value,
                tracking_publication_sequence=tracking_sequence,
                tracking_outbound_submission_id=tracking_submission_id,
                failure=failure,
            )
        if (
            receipt.source_id != int(sensor_id)
            or receipt.frame_id != frame_id
            or receipt.observed_at_us != observed_at_us_value
            or receipt.tracking_publication_sequence != tracking_sequence
            or receipt.tracking_outbound_submission_id
            != tracking_submission_id
        ):
            failure = self.bev_renderer.record_input_failure(
                camera_id,
                stage="publication_receipt",
                timestamp_us=ts_us,
                cause=RuntimeError("BEV publication receipt cohort mismatch"),
            )
            return BevPublicationReceipt(
                status="failed",
                camera_id=camera_id,
                source_id=int(sensor_id),
                frame_id=frame_id,
                observed_at_us=observed_at_us_value,
                tracking_publication_sequence=tracking_sequence,
                tracking_outbound_submission_id=tracking_submission_id,
                failure=failure,
            )
        return receipt

    def _iter_object_meta(self, frame_meta: Any) -> Iterable[Any]:
        if not _allow_raw_pyds_compat():
            _record_quarantined_compat_path("_iter_object_meta", _PYDS_COMPAT_ENV)
            return
        cast = _resolve_pyds_cast("NvDsObjectMeta")
        entries = _iter_meta_entries(getattr(frame_meta, "obj_meta_list", None), cast)
        for entry in entries:
            if entry is None:
                continue
            yield entry

    def _build_track_dict_ds8(self, obj_meta: Any, camera_id: str) -> Optional[Dict[str, Any]]:
        """Build track dictionary from DS8 pyservicemaker ObjectMetadata."""
        try:
            track_id = int(getattr(obj_meta, "object_id", -1))
        except Exception:
            track_id = -1
        try:
            class_id = int(getattr(obj_meta, "class_id", -1))
        except Exception:
            class_id = -1
        if class_id != 0:
            return None
        rect = getattr(obj_meta, "rect_params", None)
        bbox = _rect_to_bbox(rect)
        if bbox is None:
            return None
        try:
            confidence = float(getattr(obj_meta, "confidence", 0.0))
        except Exception:
            confidence = 0.0

        track: Dict[str, Any] = {
            "track_id": track_id,
            "camera_id": camera_id,
            "bbox": bbox,
            "class_id": class_id,
            "confidence": confidence,
            "center": _bbox_center(bbox),
        }

        if self._tracking_mode_is_v3dt():
            v3dt_meta = self._extract_v3dt_meta_ds8(obj_meta)
            if v3dt_meta:
                track.update(v3dt_meta)
                bbox3d = v3dt_meta.get("bbox3d")
                if isinstance(bbox3d, dict):
                    world = self._world_from_bbox3d(bbox3d)
                    if world is not None:
                        track["world"] = world
                        track["world_valid"] = True
                        track["world_frame"] = self._world_frame
                        track["world_source"] = "bbox3d"

        tracker_conf = getattr(obj_meta, "tracker_confidence", None)
        if tracker_conf is not None:
            try:
                track["tracker_confidence"] = float(tracker_conf)
            except Exception:
                pass

        analytics_items: list[Any]
        if noesis_analytics_meta_ext is not None:
            extract_analytics = getattr(noesis_analytics_meta_ext, "extract_analytics", None)
            if not callable(extract_analytics):
                raise RuntimeError("noesis_analytics_meta_ext.extract_analytics is required")
            extracted = extract_analytics(obj_meta)
            if not isinstance(extracted, list):
                raise RuntimeError("analytics native bridge returned a non-list payload")
            analytics_items = extracted
        else:
            # Compatibility for unit tests and non-DS9 import contexts only.
            analytics_items = list(getattr(obj_meta, "nvdsanalytics_obj_items", None) or [])

        selected_items: list[Any] = []
        if self._analytics_unique_id is not None:
            for analytics_info in analytics_items:
                raw_unique_id = (
                    analytics_info.get("unique_id")
                    if isinstance(analytics_info, Mapping)
                    else getattr(analytics_info, "unique_id", None)
                )
                if raw_unique_id == self._analytics_unique_id:
                    selected_items.append(analytics_info)
        elif len(analytics_items) == 1:
            selected_items = list(analytics_items)

        if not selected_items:
            logger.debug(f"DS8 track {track_id} has no analytics items")
        if len(selected_items) > 1:
            logger.warning(
                "DS9 track %s has duplicate post-analytics metadata for unique-id=%s",
                track_id,
                self._analytics_unique_id,
            )
            selected_items = []
        for analytics_info in selected_items:
            analytics_meta = self._extract_analytics_from_ds8(analytics_info)
            if analytics_meta:
                track["analytics"] = analytics_meta
                zone = _primary_zone_from_analytics(analytics_meta)
                if zone:
                    track["zone"] = zone
                    track["zone_source"] = "nvdsanalytics_roi"
                    track["zone_authoritative"] = True
                    logger.debug(f"DS8 track {track_id} assigned zone: {zone}")
                break  # Use first analytics item

        # Debug: if no zone but we have analytics, log it
        if not track.get("zone") and track.get("analytics"):
            logger.debug(f"DS8 track {track_id} has analytics but no zone: {track['analytics']}")

        return track

    def _extract_analytics_from_ds8(self, analytics_info: Any) -> Optional[Dict[str, Any]]:
        """Extract analytics info from DS8 AnalyticsObjInfo."""
        if analytics_info is None:
            return None
        try:
            result: Dict[str, Any] = {}
            # Common analytics fields
            for field in ("roiStatus", "ocStatus", "lcStatus", "dirStatus"):
                val = (
                    analytics_info.get(field)
                    if isinstance(analytics_info, Mapping)
                    else getattr(analytics_info, field, None)
                )
                if val is not None:
                    result[field] = val
            if not result:
                return None
            return result
        except Exception:
            return None

    def _build_track_dict(self, obj_meta: Any, camera_id: str) -> Optional[Dict[str, Any]]:
        rect = getattr(obj_meta, "rect_params", None)
        bbox = _rect_to_bbox(rect)
        if bbox is None:
            return None

        try:
            track_id = int(getattr(obj_meta, "object_id", -1))
        except Exception:
            track_id = -1
        try:
            class_id = int(getattr(obj_meta, "class_id", -1))
        except Exception:
            class_id = -1
        try:
            confidence = float(getattr(obj_meta, "confidence", 0.0))
        except Exception:
            confidence = 0.0

        track: Dict[str, Any] = {
            "track_id": track_id,
            "camera_id": camera_id,
            "bbox": bbox,
            "class_id": class_id,
            "confidence": confidence,
            "center": _bbox_center(bbox),
        }

        tracker_conf = getattr(obj_meta, "tracker_confidence", None)
        if tracker_conf is not None:
            try:
                track["tracker_confidence"] = float(tracker_conf)
            except Exception:
                pass

        analytics_meta = self._extract_analytics_obj_meta(obj_meta)
        if analytics_meta:
            track["analytics"] = analytics_meta
            zone = _primary_zone_from_analytics(analytics_meta)
            if zone:
                track["zone"] = zone
                track["zone_source"] = "nvdsanalytics_roi"
                track["zone_authoritative"] = True

        return track

    def _extract_analytics_obj_meta(self, obj_meta: Any) -> Optional[Dict[str, Any]]:
        if not _allow_raw_pyds_compat():
            _record_quarantined_compat_path("_extract_analytics_obj_meta", _PYDS_COMPAT_ENV)
            return None
        meta_list = getattr(obj_meta, "obj_user_meta_list", None)
        if meta_list is None:
            return None

        meta_type = self._resolve_analytics_obj_meta_type()

        cast = _resolve_pyds_cast("NvDsUserMeta")
        for user_meta in _iter_meta_entries(meta_list, cast):
            if user_meta is None:
                continue

            base_meta = getattr(user_meta, "base_meta", None)
            current_type = getattr(base_meta, "meta_type", getattr(user_meta, "meta_type", None))

            if meta_type is not None:
                if current_type != meta_type:
                    # Unit tests (and some wrappers) may expose meta_type as a string while
                    # DeepStream provides an integer enum value. Accept the canonical string
                    # sentinel as equivalent when the numeric type is unavailable on the object.
                    if str(current_type) != "NVIDIA.DSANALYTICSOBJ.USER_META":
                        continue
            else:
                if str(current_type) != "NVIDIA.DSANALYTICSOBJ.USER_META":
                    continue

            payload = getattr(user_meta, "user_meta_data", None)
            if payload is None:
                continue

            cast_payload = _resolve_pyds_cast("NvDsAnalyticsObjInfo")
            if cast_payload is not None:
                try:
                    payload = cast_payload(payload)
                except Exception:
                    pass

            data = {
                "dirStatus": getattr(payload, "dirStatus", None),
                "lcStatus": getattr(payload, "lcStatus", None),
                "ocStatus": getattr(payload, "ocStatus", None),
                "roiStatus": getattr(payload, "roiStatus", None),
            }
            return {key: value for key, value in data.items() if value is not None}

        return None

    def _resolve_analytics_obj_meta_type(self) -> Any:
        if self._analytics_obj_meta_type is not None:
            return self._analytics_obj_meta_type
        getter = _resolve_pyds_attr("nvds_get_user_meta_type")
        if getter is None:
            self._analytics_obj_meta_type = "NVIDIA.DSANALYTICSOBJ.USER_META"
            return self._analytics_obj_meta_type
        try:  # pragma: no cover - requires DeepStream runtime
            self._analytics_obj_meta_type = getter("NVIDIA.DSANALYTICSOBJ.USER_META")
        except Exception:
            self._analytics_obj_meta_type = None
        return self._analytics_obj_meta_type

    def _process_identity_v2_source_frame(
        self,
        *,
        camera_id: str,
        frame_id: int,
        primitives: Sequence[IdentityFramePrimitive],
        observed_at: float,
        fatal: bool = True,
    ) -> None:
        service = getattr(self.pipeline, "identity_v2_service", None)
        if service is None:
            return
        try:
            service.process_source_frame(
                camera_id=str(camera_id),
                frame_id=int(frame_id),
                primitives=tuple(primitives),
                observed_at=float(observed_at),
            )
        except Exception as exc:
            if fatal:
                callback = getattr(
                    self.pipeline,
                    "identity_v2_failure_callback",
                    None,
                )
                if callable(callback):
                    callback(exc)
            raise

    def _maybe_assign_stable_id(
        self,
        *,
        sensor_id: int,
        track_id: int,
        bbox: Optional[Sequence[float]],
        zone: Optional[str],
        ts: float,
        frame_bgr: Optional[np.ndarray],
        embedding: Optional[np.ndarray] = None,
    ) -> Optional[int]:
        """Return a positive stable_id for a tracked person from StableIDManager."""
        if track_id < 0:
            return None
        if bbox is None or len(bbox) < 4:
            return None
        safe_bbox = bbox
        if frame_bgr is not None:
            try:
                h, w = frame_bgr.shape[:2]
                safe_bbox = (
                    float(max(0.0, min(float(bbox[0]), float(w - 1)))),
                    float(max(0.0, min(float(bbox[1]), float(h - 1)))),
                    float(max(1.0, min(float(bbox[2]), float(w)))),
                    float(max(1.0, min(float(bbox[3]), float(h)))),
                )
            except Exception:
                safe_bbox = bbox
        mgr = getattr(self.pipeline, "stable_id_mgr", None)
        if self._stable_id_enabled and mgr is not None:
            try:
                manager_sensor_id = self._stable_id_manager_sensor_id(sensor_id)
                stable_id = mgr.update(
                    sensor_id=manager_sensor_id,
                    ds_obj_id=int(track_id),
                    bbox_ltrbwh=(float(safe_bbox[0]), float(safe_bbox[1]), float(safe_bbox[2]), float(safe_bbox[3])),
                    ts=float(ts),
                    zone=str(zone) if zone else None,
                    frame_bgr=frame_bgr,
                    embedding=embedding,
                )
                stable_id_int = int(stable_id)
                if stable_id_int > 0:
                    return stable_id_int
            except Exception:
                logger.exception("StableIDManager update failed for sensor %s track %s", sensor_id, track_id)
                self._stable_id_enabled = False

        return None

    def _stable_id_manager_sensor_id(self, sensor_id: int) -> int:
        """Use MV3DT's batch-global object ID as the legacy identity key.

        NvMultiObjectTracker allocates MV3DT IDs across the whole batched camera
        graph and propagates the winning ID to peer cameras.  Scoping that ID by
        camera recreates the split that MV3DT just resolved.  The sentinel is
        confined to the MV3DT hook; baseline and SV3DT retain their historical
        per-camera identity keys.
        """

        return -1 if self._tracking_mode_is_mv3dt() else int(sensor_id)

    def _reid_crop_from_track(
        self,
        track: Mapping[str, Any],
        frame_dims: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """Create a lightweight synthetic crop for StableIDManager (no full-frame appsink)."""
        try:
            class_id = int(track.get("class_id", -1))
        except Exception:
            class_id = -1
        if class_id not in (0,):  # limit to person-like class ids
            return None
        bbox = track.get("bbox")
        if not bbox or len(bbox) < 4:
            return None
        try:
            w = max(32, min(256, int(float(bbox[2]))))
            h = max(64, min(256, int(float(bbox[3]))))
        except Exception:
            w, h = 96, 128
        camera_id = str(track.get("camera_id", "")).strip()
        track_id = track.get("track_id", 0)
        try:
            base = int(track_id) if track_id is not None else 0
        except Exception:
            base = 0
        cam_hash = abs(hash(camera_id)) % 255 if camera_id else 127
        patch = np.zeros((h, w, 3), dtype=np.uint8)
        patch[:, :, 0] = base % 255
        patch[:, :, 1] = cam_hash
        patch[:, :, 2] = (base ^ cam_hash) % 255
        # Encode coarse bbox position to introduce variance without heavy image ops
        try:
            frame_w, frame_h = frame_dims
            u = float(bbox[0]) / max(1.0, float(frame_w))
            v = float(bbox[1]) / max(1.0, float(frame_h))
            intensity = int((u + v) * 127) % 255
            patch[: h // 4, : w // 4, :] = intensity
        except Exception:
            pass
        return patch

    def _maintain_stable_ids(
        self,
        sensor_id: int,
        present_track_ids: Iterable[int],
        ts: float,
    ) -> None:
        """Maintain StableIDManager state."""
        now_ts = float(ts)
        sensor_id_int = int(sensor_id)
        present_set = {int(tid) for tid in present_track_ids}

        mgr = getattr(self.pipeline, "stable_id_mgr", None)
        if self._stable_id_enabled and mgr is not None:
            try:
                manager_sensor_id = self._stable_id_manager_sensor_id(sensor_id_int)
                if self._tracking_mode_is_mv3dt():
                    self._mv3dt_present_track_ids_by_sensor[sensor_id_int] = set(
                        present_set
                    )
                    present_set = set().union(
                        *self._mv3dt_present_track_ids_by_sensor.values()
                    )
                if self._tracking_mode_is_v3dt() and self._v3dt_reid_track_grace_s > 0.0:
                    for track_id in present_set:
                        self._v3dt_reid_last_seen_by_track[(manager_sensor_id, track_id)] = now_ts
                    for key, last_seen_ts in list(self._v3dt_reid_last_seen_by_track.items()):
                        cached_sensor_id, cached_track_id = key
                        if cached_sensor_id != manager_sensor_id or cached_track_id in present_set:
                            continue
                        age_s = now_ts - last_seen_ts
                        if 0.0 <= age_s <= self._v3dt_reid_track_grace_s:
                            present_set.add(cached_track_id)
                        else:
                            self._v3dt_reid_last_seen_by_track.pop(key, None)
                mgr.remove_missing_tracks(manager_sensor_id, list(present_set), now_ts)
                mgr.prune_ghosts(now_ts)
            except Exception:
                logger.exception("StableIDManager maintenance failed for sensor %s", sensor_id_int)
                self._stable_id_enabled = False

    def _update_dwell_time(
        self,
        sensor_id: int,
        stable_id: int,
        zone: Optional[str],
        now_ts: float,
    ) -> Optional[float]:
        """Maintain per-track zone entry time to compute dwell seconds."""
        if stable_id <= 0:
            return None
        state = self._zone_state.setdefault(int(sensor_id), {})
        entry = state.get(int(stable_id), {})
        current_zone = entry.get("zone")
        entry_time = entry.get("entry")

        if not zone:
            state[int(stable_id)] = {"zone": None, "entry": None}
            return None

        if current_zone == zone:
            if entry_time is None:
                entry_time = float(now_ts)
            state[int(stable_id)] = {"zone": zone, "entry": float(entry_time)}
            return max(0.0, float(now_ts) - float(entry_time))

        # Zone change detected.
        if current_zone and current_zone != zone:
            self._record_transition(
                sensor_id=int(sensor_id),
                stable_id=int(stable_id),
                from_zone=str(current_zone),
                to_zone=str(zone),
                ts=float(now_ts),
            )

        state[int(stable_id)] = {"zone": zone, "entry": float(now_ts)}
        return 0.0

    def _record_transition(
        self,
        sensor_id: int,
        stable_id: int,
        ts: float,
        from_zone: Optional[str] = None,
        to_zone: Optional[str] = None,
        line_name: Optional[str] = None,
    ) -> None:
        """Store a zone transition or line crossing event."""
        trans_list = self._transitions_state.setdefault(int(sensor_id), [])
        camera_id = self.camera_labels.get(int(sensor_id), f"camera_{int(sensor_id)}")

        event: Dict[str, Any] = {
            "stable_id": int(stable_id),
            "camera_id": str(camera_id),
            "timestamp": float(ts),
        }
        if from_zone and to_zone:
            event["from_zone"] = from_zone
            event["to_zone"] = to_zone
        elif line_name:
            event["line_name"] = line_name
        else:
            return

        trans_list.append(event)
        if len(trans_list) > 100:
            self._transitions_state[int(sensor_id)] = trans_list[-100:]

    def _cleanup_zone_state(self, sensor_id: int, active_stable_ids: set[int]) -> None:
        """Remove stale zone entries for stable IDs no longer present."""
        state = self._zone_state.get(int(sensor_id))
        if not state:
            return
        for sid in list(state.keys()):
            if int(sid) not in active_stable_ids:
                state.pop(sid, None)

    def _publish_occupancy(self, sensor_id: int, occupancy_counts: Mapping[str, int]) -> None:
        publisher = getattr(self.pipeline, "occupancy_publisher", None)
        now_ts = time.time()
        previous = self._occupancy_state.get(sensor_id, {})
        merged_counts: Dict[str, int] = dict(occupancy_counts)

        grace_s = float(self._occupancy_grace_s)
        if grace_s > 0.0:
            last_seen_by_zone = self._occupancy_last_seen.setdefault(sensor_id, {})
            for zone, count in merged_counts.items():
                if int(count) > 0:
                    last_seen_by_zone[str(zone)] = float(now_ts)

            for zone, count in previous.items():
                zone_name = str(zone)
                if zone_name in merged_counts or int(count) <= 0:
                    continue
                last_seen = float(last_seen_by_zone.get(zone_name, 0.0) or 0.0)
                if (now_ts - last_seen) < grace_s:
                    merged_counts[zone_name] = int(count)
                else:
                    last_seen_by_zone.pop(zone_name, None)

            for zone in list(last_seen_by_zone.keys()):
                if zone in merged_counts:
                    continue
                last_seen = float(last_seen_by_zone.get(zone, 0.0) or 0.0)
                if (now_ts - last_seen) >= grace_s:
                    last_seen_by_zone.pop(zone, None)

        self._occupancy_state[sensor_id] = dict(merged_counts)
        if publisher is None:
            return

        now_ns = time.time_ns()
        try:
            for zone, count in merged_counts.items():
                room_id = str(zone).strip()
                if not room_id:
                    continue
                publisher.publish_state(
                    room_id=room_id,
                    occupied=int(count) > 0,
                    count=int(count),
                    ts_ns=now_ns,
                )
            # Emit vacate events for zones no longer present
            for zone in set(previous.keys()) - set(merged_counts.keys()):
                room_id = str(zone).strip()
                if not room_id:
                    continue
                publisher.publish_state(
                    room_id=room_id,
                    occupied=False,
                    count=0,
                    ts_ns=now_ns,
                )
        except Exception:
            logger.exception("Failed to publish occupancy for sensor %s", sensor_id)

    def get_tracking_stats(self, sensor_id: int) -> Dict[str, Any]:
        """Fetch latest tracking and occupancy for a specific sensor."""
        return {
            "occupancy": self._occupancy_state.get(sensor_id, {}),
            "active_tracks": self._active_tracks.get(sensor_id, []),
            "transitions": self._transitions_state.get(sensor_id, []),
        }

    def _apply_public_depth_fields(
        self,
        track: Dict[str, Any],
        depth_result: Optional[ObjectDepthResult],
    ) -> None:
        diagnostic_fields = (
            "depth_evidence_quality",
            "depth_evidence_reason",
            "depth_spread_m",
            "depth_anchor_spread_m",
            "depth_measurement_frame_id",
            "depth_measurement_ts_us",
            "depth_measurement_age_us",
            "depth_measurement_cached",
            "depth_tensor_frame_id",
            "depth_tensor_ts_us",
            "depth_tensor_age_frames",
            "depth_tensor_age_us",
        )
        if depth_result is None:
            track["depth_status"] = None
            track["depth_anchor_source"] = None
            track["depth_anchor_m"] = None
            track["depth_used_m"] = None
            track["depth_registered_m"] = None
            track["depth_registration_status"] = None
            track["depth_registration_id"] = None
            track["depth_center_m"] = None
            track["depth_median_m"] = None
            track["depth_sample_count"] = None
            track["depth_valid_fraction"] = None
            for field_name in diagnostic_fields:
                track[field_name] = None
            return

        def _bounded_text(value: Any, *, max_chars: int) -> Optional[str]:
            if value is None:
                return None
            text = str(value).strip()
            return text[:max_chars] if text else None

        def _nonnegative_float(value: Any) -> Optional[float]:
            if value is None:
                return None
            try:
                parsed = float(value)
            except Exception:
                return None
            if not math.isfinite(parsed) or parsed < 0.0:
                return None
            return parsed

        track["depth_status"] = str(depth_result.status)
        track["depth_anchor_source"] = str(depth_result.anchor_source) if depth_result.anchor_source else None
        if track.get("depth_anchor_m") is None:
            track["depth_anchor_m"] = float(depth_result.anchor_depth_m) if depth_result.anchor_depth_m is not None else None
        if track.get("depth_used_m") is None:
            track["depth_used_m"] = (
                _depth_used_m(depth_result)
                if self._depth_measurement_is_current(depth_result, track=track)
                else None
            )
        track["depth_registered_m"] = track.get("depth_registered_m")
        track["depth_registration_status"] = track.get("depth_registration_status")
        track["depth_registration_id"] = track.get("depth_registration_id")
        track["depth_center_m"] = float(depth_result.depth_center) if depth_result.depth_center is not None else None
        track["depth_median_m"] = float(depth_result.depth_median) if depth_result.depth_median is not None else None
        track["depth_sample_count"] = int(depth_result.sample_count)
        track["depth_valid_fraction"] = float(depth_result.valid_fraction)
        track["depth_evidence_quality"] = _bounded_text(
            depth_result.evidence_quality,
            max_chars=32,
        )
        track["depth_evidence_reason"] = _bounded_text(
            depth_result.evidence_reason,
            max_chars=160,
        )
        track["depth_spread_m"] = _nonnegative_float(depth_result.depth_spread_m)
        track["depth_anchor_spread_m"] = _nonnegative_float(
            depth_result.anchor_depth_spread_m
        )
        track["depth_measurement_frame_id"] = (
            max(0, int(depth_result.measurement_frame_id))
            if depth_result.measurement_frame_id is not None
            else None
        )
        track["depth_measurement_ts_us"] = (
            max(0, int(depth_result.measurement_ts_us))
            if depth_result.measurement_ts_us is not None
            else None
        )
        track["depth_measurement_age_us"] = (
            max(0, int(depth_result.measurement_age_us))
            if depth_result.measurement_age_us is not None
            else None
        )
        track["depth_measurement_cached"] = (
            bool(depth_result.measurement_cached)
            if depth_result.measurement_cached is not None
            else None
        )
        track["depth_tensor_frame_id"] = (
            max(0, int(depth_result.depth_tensor_frame_id))
            if depth_result.depth_tensor_frame_id is not None
            else None
        )
        track["depth_tensor_ts_us"] = (
            max(0, int(depth_result.depth_tensor_ts_us))
            if depth_result.depth_tensor_ts_us is not None
            else None
        )
        track["depth_tensor_age_frames"] = (
            max(0, int(depth_result.depth_tensor_age_frames))
            if depth_result.depth_tensor_age_frames is not None
            else None
        )
        track["depth_tensor_age_us"] = (
            max(0, int(depth_result.depth_tensor_age_us))
            if depth_result.depth_tensor_age_us is not None
            else None
        )

    def _stamp_osd_label_ds8(self, obj_meta: Any, *, sensor_id: int, stable_id: Optional[int]) -> None:
        proc = self.osd_label_processor
        if proc is None:
            proc = getattr(self.pipeline, "osd_label_processor", None)
        if proc is None:
            return
        try:
            proc._apply_label(obj_meta, sensor_id=int(sensor_id), stable_id_override=stable_id)
        except Exception:
            logger.debug("OSD label stamp failed (DS8)", exc_info=True)

    def _stamp_osd_label(self, obj_meta: Any, *, sensor_id: int, stable_id: Optional[int]) -> None:
        proc = self.osd_label_processor
        if proc is None:
            proc = getattr(self.pipeline, "osd_label_processor", None)
        if proc is None:
            return
        try:
            proc._apply_label(obj_meta, sensor_id=int(sensor_id), stable_id_override=stable_id)
        except Exception:
            logger.debug("OSD label stamp failed", exc_info=True)


@dataclass
class _OsdLabelProcessor:
    decimals: int = 2
    font_size: Optional[int] = 22
    font_name: Optional[str] = "Sans"
    show_both_ids: bool = False
    stable_id_mgr: Any = field(default=None, repr=False)

    @staticmethod
    def from_pipeline_config(pipeline_cfg: Mapping[str, Any]) -> "_OsdLabelProcessor":
        cfg: Mapping[str, Any] = {}
        vis_cfg = pipeline_cfg.get("visualization") or {}
        if isinstance(vis_cfg, Mapping):
            raw = vis_cfg.get("osd_labels") or {}
            if isinstance(raw, Mapping):
                cfg = raw

        def _int(value: Any, default: int) -> int:
            try:
                return int(value)
            except Exception:
                return int(default)

        def _str(value: Any) -> str:
            try:
                return str(value)
            except Exception:
                return ""

        decimals = _int(cfg.get("decimals", 2), 2)
        decimals = max(0, decimals)

        font_size_env = str(os.environ.get("NOESIS_OSD_LABEL_FONT_SIZE", "")).strip()
        font_size_raw = font_size_env if font_size_env else cfg.get("font_size", 22)
        font_size = _int(font_size_raw, 22)
        font_size = max(1, font_size)

        font_name_env = str(os.environ.get("NOESIS_OSD_LABEL_FONT_NAME", "")).strip()
        font_name_raw = font_name_env if font_name_env else cfg.get("font_name", "Sans")
        font_name = _str(font_name_raw).strip() or "Sans"
        diag_raw = str(os.environ.get("NOESIS_REID_DIAG_USE_TRACKER_ID", "0") or "").strip().lower()
        show_both = diag_raw in ("1", "true", "yes", "on", "y")

        return _OsdLabelProcessor(
            decimals=decimals,
            font_size=font_size,
            font_name=font_name,
            show_both_ids=show_both,
        )

    def handle_frame_ds8(self, frame_meta: Any) -> None:
        sensor_id = self._frame_source_id(frame_meta)
        object_items = getattr(frame_meta, "object_items", None) or []
        for obj_meta in object_items:
            self._apply_label(obj_meta, sensor_id=sensor_id, stable_id_override=None)

    def handle_frame(self, frame_meta: Any) -> None:
        if not _allow_raw_pyds_compat():
            _record_quarantined_compat_path("_OsdLabelProcessor.handle_frame", _PYDS_COMPAT_ENV)
            return
        sensor_id = self._frame_source_id(frame_meta)
        cast = _resolve_pyds_cast("NvDsObjectMeta")
        for obj_meta in _iter_meta_entries(getattr(frame_meta, "obj_meta_list", None), cast):
            if obj_meta is None:
                continue
            self._apply_label(obj_meta, sensor_id=sensor_id, stable_id_override=None)

    def _apply_label(self, obj_meta: Any, *, sensor_id: int, stable_id_override: Optional[int] = None) -> None:
        text_params = getattr(obj_meta, "text_params", None)
        if text_params is None or not hasattr(text_params, "display_text"):
            return
        label = self._format_label(obj_meta, text_params, sensor_id=sensor_id, stable_id_override=stable_id_override)
        if not label:
            return
        try:
            text_params.display_text = label
        except Exception:
            pass
        try:
            setattr(obj_meta, "obj_label", label)
        except Exception:
            pass
        self._apply_font(text_params)

    def _apply_font(self, text_params: Any) -> None:
        if self.font_size is None and not self.font_name:
            return
        font_params = getattr(text_params, "font_params", None)
        if font_params is None:
            return
        if self.font_name and ds_osd is not None:
            try:
                name = str(self.font_name).strip()
                family = None
                if name:
                    family = getattr(getattr(ds_osd, "FontFamily", None), name, None)
                    if family is None:
                        family = getattr(getattr(ds_osd, "FontFamily", None), name.capitalize(), None)
                if family is not None:
                    font_params.name = family
            except Exception:
                pass
        if self.font_size is not None:
            try:
                font_params.size = int(self.font_size)
            except Exception:
                pass

    def _frame_source_id(self, frame_meta: Any) -> int:
        for attr in ("source_id", "pad_index", "camera_id"):
            value = getattr(frame_meta, attr, None)
            if value is None:
                continue
            try:
                return int(value)
            except Exception:
                continue
        return 0

    def _lookup_stable_id(self, sensor_id: int, track_id: int) -> Optional[int]:
        mgr = self.stable_id_mgr
        if mgr is None:
            return None
        key = (int(sensor_id), int(track_id))
        rec = None
        lock = getattr(mgr, "_lock", None)
        if lock is not None:
            try:
                with lock:
                    rec = getattr(mgr, "active_tracks", {}).get(key)
            except Exception:
                rec = None
        else:
            try:
                rec = getattr(mgr, "active_tracks", {}).get(key)
            except Exception:
                rec = None
        if not isinstance(rec, dict):
            return None
        stable_id = rec.get("stable_id")
        try:
            stable_id_int = int(stable_id)
        except Exception:
            return None
        if stable_id_int <= 0:
            return None
        return stable_id_int

    def _format_label(
        self,
        obj_meta: Any,
        text_params: Any,
        *,
        sensor_id: int,
        stable_id_override: Optional[int] = None,
    ) -> str:
        label = ""
        for attr in ("label", "obj_label"):
            try:
                value = getattr(obj_meta, attr, None)
            except Exception:
                value = None
            if value:
                label = _clean_osd_base_label(str(value).strip())
            if label:
                break
        if not label:
            try:
                class_id = int(getattr(obj_meta, "class_id", -1))
            except Exception:
                class_id = -1
            label = f"class {class_id}" if class_id >= 0 else "class"

        try:
            class_id = int(getattr(obj_meta, "class_id", -1))
        except Exception:
            class_id = -1
        try:
            track_id = int(getattr(obj_meta, "object_id", -1))
        except Exception:
            track_id = -1

        parts: List[str] = [label]
        # Stable IDs are people-only; avoid showing raw tracker IDs for other classes.
        if class_id == 0 and track_id >= 0:
            stable_id = stable_id_override
            if stable_id is None:
                stable_id = self._lookup_stable_id(sensor_id, track_id)
            try:
                stable_id_int = int(stable_id) if stable_id is not None else None
            except Exception:
                stable_id_int = None
            if self.show_both_ids:
                tracker_text = str(track_id) if track_id >= 0 else "XX"
                stable_text = str(stable_id_int) if stable_id_int is not None and stable_id_int > 0 else "XX"
                parts.append(f"[{tracker_text}] | [{stable_text}]")
            else:
                if stable_id_int is not None and stable_id_int > 0:
                    parts.append(f"{stable_id_int}")
                else:
                    parts.append("XX")
        base_label = " ".join([p for p in parts if p]).strip()

        depth_text = None
        if class_id == 0:
            depth_override = getattr(obj_meta, "_noesis_depth_used_m", None)
            if depth_override is not None:
                try:
                    depth_val = float(depth_override)
                except Exception:
                    depth_val = float("nan")
                if math.isfinite(depth_val) and depth_val > 0.0:
                    depth_text = f"depth={depth_val:.{max(0, int(self.decimals))}f}m"
                else:
                    depth_text = "depth=n/a"
            else:
                depth_text = _format_depth_label_fragment(
                    _extract_object_depth_result_from_meta(obj_meta),
                    decimals=self.decimals,
                )
        if depth_text:
            if base_label:
                base_label = f"{base_label} {depth_text}"
            else:
                base_label = depth_text

        try:
            confidence = float(getattr(obj_meta, "confidence", float("nan")))
        except Exception:
            confidence = float("nan")
        if not math.isfinite(confidence) or confidence < 0.0:
            return base_label
        decimals = max(0, int(self.decimals))
        conf_text = f"{confidence:.{decimals}f}"
        if base_label.endswith(conf_text):
            return base_label
        if base_label:
            return f"{base_label} {conf_text}"
        return conf_text


class _OsdLabelOperator(_BatchMetadataOperatorBase):  # pragma: no cover - requires DeepStream runtime
    def __init__(self, processor: _OsdLabelProcessor) -> None:
        super().__init__()
        self._processor = processor

    def handle_metadata(self, batch_meta: Any) -> None:
        if batch_meta is None:
            return

        frame_items = getattr(batch_meta, "frame_items", None)
        if frame_items is None:
            return
        for frame_meta in frame_items:
            try:
                self._processor.handle_frame_ds8(frame_meta)
            except Exception:
                logger.exception("Failed to stamp OSD labels within batch metadata (DS8)")


class _IdentityV2PostResolutionOsdOperator(
    _BatchMetadataOperatorBase
):  # pragma: no cover - requires DeepStream runtime
    def __init__(self, processor: IdentityV2PostResolutionOsdProcessor) -> None:
        super().__init__()
        self._operator = _SharedIdentityV2OsdOperator(processor)

    def handle_metadata(self, batch_meta: Any) -> None:
        self._operator.handle_metadata(batch_meta)


class _AnalyticsTelemetryOperator(_BatchMetadataOperatorBase):  # pragma: no cover - requires DeepStream runtime
    def __init__(self, processor: _AnalyticsTelemetryProcessor) -> None:
        super().__init__()
        self._processor = processor

    def handle_metadata(self, batch_meta: Any) -> None:
        if batch_meta is None:
            return

        frame_items = getattr(batch_meta, "frame_items", None)
        if frame_items is None:
            return
        for frame_meta in frame_items:
            try:
                self._processor.handle_frame_ds8(frame_meta)
            except Exception:
                logger.exception("Failed to process analytics telemetry within batch metadata (DS8)")


class _TrailOverlayOperator(_BatchMetadataOperatorBase):  # pragma: no cover - requires DeepStream runtime
    def __init__(self, processor: TrailOverlayProcessor) -> None:
        super().__init__()
        self._processor = processor

    def handle_metadata(self, batch_meta: Any) -> None:
        if batch_meta is None:
            return
        frame_items = getattr(batch_meta, "frame_items", None)
        if frame_items is None:
            return
        try:
            self._processor.handle_batch_ds8(batch_meta)
        except Exception:
            logger.exception("Failed to render trail overlay within batch metadata (DS8)")


class _IntrinsicsProcessor:
    def __init__(self, loader: intrinsics_module.CameraConfigLoader, pipeline: "DS8Pipeline") -> None:
        self._loader = loader
        self._pipeline = pipeline

    def apply(self, frame_meta: Any) -> None:
        if not _env_flag(_INTRINSICS_USER_META_ENV, default=False):
            _record_quarantined_compat_path("_IntrinsicsProcessor.apply", _INTRINSICS_USER_META_ENV)
            return
        source_id = getattr(frame_meta, "pad_index", getattr(frame_meta, "source_id", None))
        if source_id is None:
            return
        try:
            intrinsics_module.attach_intrinsics(frame_meta, source_id, loader=self._loader)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to attach intrinsics for source %s", source_id)


class _IntrinsicsOperator(_BatchMetadataOperatorBase):  # pragma: no cover - requires DeepStream runtime
    def __init__(self, processor: _IntrinsicsProcessor) -> None:
        super().__init__()
        self._processor = processor

    def handle_metadata(self, batch_meta: Any) -> None:
        frame_items = getattr(batch_meta, "frame_items", None)
        if frame_items is None:
            return
        for frame_meta in frame_items:
            try:
                self._processor.apply(frame_meta)
            except Exception:
                logger.exception("Failed to apply intrinsics within batch metadata probe (DS8)")


class _MapAnythingBufferOperator(_BufferOperatorBase):  # pragma: no cover - requires DeepStream runtime
    """Use one Buffer object for exact tensor metadata and its batch surface."""

    def __init__(self, processor: MapAnythingProcessor) -> None:
        super().__init__()
        self._processor = processor
        self._frames_seen = 0
        self._matched_frames = 0

    def handle_buffer(self, buffer: Any) -> bool:
        frame_meta: Any = None
        try:
            if not bool(getattr(self._processor.pipeline, "depth_enabled", False)):
                return True
            batch_meta = getattr(buffer, "batch_meta", None)
            frame_items = getattr(batch_meta, "frame_items", None)
            if frame_items is None:
                raise RuntimeError(
                    "DS9 MapAnything exact buffer is missing batch frame metadata"
                )
            for frame_meta in frame_items:
                self._frames_seen += 1
                matched = self._processor.handle_native_buffer_frame_ds9(
                    buffer,
                    frame_meta,
                )
                if not matched:
                    _increment_core_counter(
                        "mapanything_non_inference_batches_total"
                    )
                    continue
                self._matched_frames += 1
                _increment_core_counter(
                    "mapanything_exact_native_capture_frames_total"
                )
        except BaseException as exc:
            try:
                _increment_core_counter(
                    "mapanything_exact_native_capture_failures_total"
                )
            except BaseException:
                pass
            try:
                logger.error(
                    "DS9 MapAnything exact buffer capture failed for gie_id=%s frame=%s",
                    self._processor.gie_id,
                    getattr(frame_meta, "frame_number", None),
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
            except BaseException:
                pass
            try:
                self._processor.report_capture_failure(exc)
            except BaseException:
                try:
                    logger.exception(
                        "DS9 MapAnything buffer failure reporting failed"
                    )
                except BaseException:
                    pass
        return True


class _MapAnythingOperator(_BatchMetadataOperatorBase):  # pragma: no cover - source-only test compatibility
    def __init__(self, processor: MapAnythingProcessor) -> None:
        super().__init__()
        self._processor = processor
        self._frames_seen = 0
        self._matched_frames = 0

    def handle_metadata(self, batch_meta: Any) -> None:
        frame_items = getattr(batch_meta, "frame_items", None)
        if frame_items is None:
            return
        for frame_meta in frame_items:
            self._frames_seen += 1
            if not bool(getattr(self._processor.pipeline, "depth_enabled", False)):
                continue
            try:
                matched = self._processor.handle_native_frame_ds9(frame_meta)
            except Exception:
                _increment_core_counter(
                    "mapanything_exact_native_capture_failures_total"
                )
                logger.exception(
                    "DS9 MapAnything exact native capture failed for gie_id=%s frame=%s",
                    self._processor.gie_id,
                    getattr(frame_meta, "frame_number", None),
                )
                raise
            if not matched:
                continue
            self._matched_frames += 1
            _increment_core_counter("mapanything_exact_native_capture_frames_total")


class _V3DTCuboidBufferOperator(_BufferOperatorBase):  # pragma: no cover - requires DeepStream runtime
    def __init__(self, processor: V3DTCuboidOverlayProcessor) -> None:
        super().__init__()
        self._processor = processor

    def handle_buffer(self, buffer: Any) -> bool:
        try:
            self._processor.scrub_vendor_overlay(buffer)
            self._processor.render_batch(getattr(buffer, "batch_meta", None))
            return True
        except Exception:
            logger.exception("V3DT cuboid correction failed")
            return False


class _PoseFeatureOperator(_BatchMetadataOperatorBase):  # pragma: no cover - requires DeepStream runtime
    def __init__(self, processor: PoseFeatureProcessor) -> None:
        super().__init__()
        self._processor = processor

    def handle_metadata(self, batch_meta: Any) -> None:
        if batch_meta is None:
            return
        frame_items = getattr(batch_meta, "frame_items", None)
        if frame_items is None:
            return
        for frame_meta in frame_items:
            try:
                self._processor.handle_frame_ds8(batch_meta, frame_meta)
            except Exception:
                logger.exception(
                    "Failed to compute pose features within batch metadata (DS8)"
                )
                if self._processor.tensor_source == "rfdetr_pgie_frame":
                    raise


class _PoseKeypointOverlayOperator(_BatchMetadataOperatorBase):  # pragma: no cover - requires DeepStream runtime
    def __init__(self, processor: PoseKeypointOverlayProcessor) -> None:
        super().__init__()
        self._processor = processor

    def handle_metadata(self, batch_meta: Any) -> None:
        if batch_meta is None:
            return
        frame_items = getattr(batch_meta, "frame_items", None)
        if frame_items is None:
            return
        try:
            self._processor.handle_batch_ds8(batch_meta)
        except Exception:
            logger.exception("Failed to render pose keypoints within batch metadata (DS8)")


def _select_tensor(tensors: Mapping[str, np.ndarray], keys: Sequence[str]) -> Optional[np.ndarray]:
    lower_map = {str(k).lower(): v for k, v in tensors.items()}
    for key in keys:
        if key in tensors:
            return tensors[key]
        key_lower = str(key).lower()
        if key_lower in lower_map:
            return lower_map[key_lower]
    return None


def _bbox_corners(bbox: Sequence[float]) -> List[Tuple[float, float]]:
    left, top, width, height = (float(bbox[i]) for i in range(4))
    right = left + width
    bottom = top + height
    return [
        (left, top),
        (right, top),
        (right, bottom),
        (left, bottom),
    ]


def _point_in_polygon(point: Tuple[float, float], polygon: Sequence[Tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            denom = (y2 - y1) if (y2 - y1) != 0 else 1e-9
            intersect = (x2 - x1) * (y - y1) / denom + x1
            if x < intersect:
                inside = not inside
    return inside


def _meta_lookup(meta: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if hasattr(meta, name):
            try:
                return getattr(meta, name)
            except Exception:
                pass
        if isinstance(meta, Mapping) and name in meta:
            return meta[name]
    return default


def _resolve_pyds_cast(name: str) -> Optional[Callable[[Any], Any]]:
    if not _allow_raw_pyds_compat():
        _record_quarantined_compat_path(f"_resolve_pyds_cast.{name}", _PYDS_COMPAT_ENV)
        return None
    if pyds is None:
        return None
    target = getattr(pyds, name, None)
    if target is None:
        return None
    cast = getattr(target, "cast", None)
    return cast if callable(cast) else None


def _resolve_pyds_attr(name: str) -> Any:
    if not _allow_raw_pyds_compat():
        _record_quarantined_compat_path(f"_resolve_pyds_attr.{name}", _PYDS_COMPAT_ENV)
        return None
    if pyds is None:
        return None
    return getattr(pyds, name, None)


def _iter_meta_entries(meta_list: Any, cast_fn: Optional[Callable[[Any], Any]] = None) -> Iterable[Any]:
    if meta_list is None:
        return
    if isinstance(meta_list, (list, tuple)):
        for item in meta_list:
            if cast_fn is not None and item is not None:
                try:
                    item = cast_fn(item)
                except Exception:
                    pass
            yield item
        return
    node = meta_list
    visited = 0
    while node is not None:
        data = getattr(node, "data", node)
        if cast_fn is not None and data is not None:
            try:
                data = cast_fn(data)
            except Exception:
                pass
        if data is not None:
            yield data
        visited += 1
        if visited > 4096:  # Safety guard to avoid infinite loops on malformed lists.
            logger.debug("Truncating metadata iteration after %s entries", visited)
            break
        try:
            node = node.next
        except StopIteration:
            break
        except Exception:
            break


def _rect_to_bbox(rect: Any) -> Optional[List[float]]:
    if rect is None:
        return None

    def _rect_attr(*names: str) -> Optional[float]:
        for name in names:
            try:
                value = getattr(rect, name)
            except AttributeError:
                continue
            except Exception:
                return None
            try:
                parsed = float(value)
            except Exception:
                return None
            if math.isfinite(parsed):
                return parsed
            return None
        return None

    try:
        left = _rect_attr("left", "x")
        top = _rect_attr("top", "y")
        width = _rect_attr("width", "w")
        height = _rect_attr("height", "h")
    except Exception:
        return None
    if left is None or top is None or width is None or height is None:
        return None
    if width <= 0.0 or height <= 0.0:
        return None
    return [left, top, width, height]


def _bbox_center(bbox: Sequence[float]) -> List[float]:
    try:
        left, top, width, height = map(float, bbox[:4])
        return [left + width / 2.0, top + height / 2.0]
    except Exception:
        return [0.0, 0.0]


def _primary_zone_from_analytics(analytics_meta: Mapping[str, Any]) -> Optional[str]:
    return resolve_authoritative_analytics_zone(analytics_meta)


def _fallback_zone_from_camera(camera_id: Any) -> Optional[str]:
    """Return a diagnostic camera zone when analytics membership is unavailable.

    The caller stamps this as non-authoritative `camera_default` evidence for
    occupancy, dwell, and operator diagnostics only. It never populates
    canonical `room_id`; authoritative membership requires exact `ocStatus` or
    ROI-only compatibility evidence from nvdsanalytics.
    """
    try:
        value = str(camera_id).strip()
    except Exception:
        return None
    if not value:
        return None
    value = value.replace("_", " ").replace("-", " ")
    value = " ".join(value.split())
    return value.title()


# Type checking imports (avoids circular at runtime)
from typing import TYPE_CHECKING  # noqa: E402  (import at end to satisfy linter)

if TYPE_CHECKING:  # pragma: no cover - used for type hints only
    from noesis.pipelines.ds8_pipeline import DS8Pipeline  # noqa: F401
    from noesis.telemetry.publishers import DepthTelemetryPublisher  # noqa: F401
    from noesis.telemetry.publishers import TrackingTelemetryPublisher  # noqa: F401
