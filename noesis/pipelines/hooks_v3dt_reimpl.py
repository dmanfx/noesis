from __future__ import annotations

import colorsys
import json
import logging
import math
import os
import queue
import re
import time
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
    v3dt_bbox3d_tracker_foot,
    v3dt_bbox3d_world_foot,
)
from noesis_core.mapanything_lifecycle import MapAnythingIdleReceipt
from noesis_core.depth_contract import usable_registered_depth_m
from noesis_core.analytics_zones import resolve_authoritative_analytics_zone
from noesis_core.tracking_continuity import (
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
    begin_source_admission,
    classify_posture,
    commit_image_path_point,
    complete_source_admission,
    resolve_pose_floor_anchor,
    source_score,
    update_human_cv_filter,
    update_motion_mode,
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
    from pyservicemaker import BatchMetadataOperator, Probe, osd as ds_osd  # type: ignore
except Exception:  # pragma: no cover - exercised only in DS runtime
    BatchMetadataOperator = None  # type: ignore
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
_OSD_LABEL_DEPTH_RE = re.compile(r"\s+z=(?:n/a|[-+]?\d+(?:\.\d+)?m)\s*$", re.IGNORECASE)
_OSD_LABEL_CONF_RE = re.compile(r"\s+[-+]?\d+(?:\.\d+)?\s*$")
_OSD_LABEL_ID_RE = re.compile(r"\s+(?:\[[^\]]+\]\s*\|\s*\[[^\]]+\]|XX|\d+)\s*$")
_MAPANYTHING_ASYNC_STOP = object()
_MAPANYTHING_AMBIGUOUS_CONFIDENCE_SCALE = 0.50
_MAPANYTHING_OUTSIDE_CALIBRATED_FOV_CONFIDENCE_SCALE = 0.25
_WORLD_ESTIMATOR_DIAGNOSTIC_FIELDS = (
    "world_estimator_evaluated",
    "world_floor_candidate",
    "world_floor_range_m",
    "world_floor_range_limit_m",
    "world_floor_incidence_sin",
    "world_floor_admitted",
    "world_floor_rejection_reason",
    "world_depth_candidate",
    "world_prefilter_measurement",
    "world_filter_prediction",
    "world_measurement_accepted",
    "world_rejection_reason",
    "world_innovation_m",
    "world_innovation_limit_m",
    "world_reacquire_count",
    "world_reacquired",
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
)
V3DT_WORLD_SOURCE_BBOX3D_FOOT = "v3dt_bbox3d_foot"


@dataclass
class _CorePathInstrumentation:
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    counters: Dict[str, int] = field(default_factory=dict, init=False, repr=False)
    serialization_prep: Dict[str, Dict[str, int]] = field(default_factory=dict, init=False, repr=False)
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

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "counters": dict(self.counters),
                "serialization_prep": {k: dict(v) for k, v in self.serialization_prep.items()},
                "events": list(self.events),
            }

    def reset(self) -> None:
        with self._lock:
            self.counters.clear()
            self.serialization_prep.clear()
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


def _frame_pts_key_us(frame_meta: Any) -> int:
    raw = int(_meta_lookup(frame_meta, "buf_pts", "buffer_pts", default=0) or 0)
    if raw > 0:
        return raw // 1000
    return int(time.time_ns() // 1000)


def _public_frame_timing(frame_meta: Any, observed_at_s: float) -> Dict[str, Any]:
    observed_at_us = max(1, int(float(observed_at_s) * 1_000_000))
    fields: Dict[str, Any] = {
        "captured_at_us": observed_at_us,
        "observed_at_us": observed_at_us,
        "capture_time_status": "estimated",
    }
    try:
        media_pts_ns = int(_meta_lookup(frame_meta, "buf_pts", "buffer_pts", "pts", default=-1))
    except Exception:
        media_pts_ns = -1
    if media_pts_ns >= 0:
        fields["media_pts_ns"] = media_pts_ns
    return fields


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
    lower_values = np.asarray(depth_crop[np.logical_and(lower_body_mask, np.isfinite(depth_crop))], dtype=np.float32)
    lower_count = int(lower_values.size)
    lower_area = int(np.count_nonzero(lower_body_mask))
    lower_valid_fraction = float(lower_count) / float(lower_area or 1)
    lower_min_count, lower_min_valid_fraction = _anchor_support_requirements(lower_area, "lower_body_band")
    if lower_count >= lower_min_count and lower_valid_fraction >= lower_min_valid_fraction:
        return _DepthAnchorSample(
            foot_uv=foot_uv,
            anchor_source="lower_body_band",
            anchor_depth_m=float(np.median(lower_values)),
            anchor_sample_count=lower_count,
            anchor_valid_fraction=lower_valid_fraction,
            lower_body_sample_count=lower_count,
            lower_body_valid_fraction=lower_valid_fraction,
            torso_sample_count=0,
            torso_valid_fraction=0.0,
        )

    torso_mask = _band_mask(eroded_mask, y0_ratio=0.35, y1_ratio=0.70, center_width_ratio=0.50)
    torso_values = np.asarray(depth_crop[np.logical_and(torso_mask, np.isfinite(depth_crop))], dtype=np.float32)
    torso_count = int(torso_values.size)
    torso_area = int(np.count_nonzero(torso_mask))
    torso_valid_fraction = float(torso_count) / float(torso_area or 1)
    torso_min_count, torso_min_valid_fraction = _anchor_support_requirements(torso_area, "torso_core")
    anchor_depth = (
        float(np.median(torso_values))
        if torso_count >= torso_min_count and torso_valid_fraction >= torso_min_valid_fraction
        else None
    )
    anchor_source = "torso_core" if anchor_depth is not None else None
    return _DepthAnchorSample(
        foot_uv=foot_uv,
        anchor_source=anchor_source,
        anchor_depth_m=anchor_depth,
        anchor_sample_count=torso_count if anchor_depth is not None else 0,
        anchor_valid_fraction=torso_valid_fraction if anchor_depth is not None else 0.0,
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
        return f"z={depth_used:.{max(0, int(decimals))}f}m"
    return "z=n/a"


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
) -> None:
    """Attach the MapAnything post-process hook to decode full-frame tensor meta.

    This operator expects the MapAnything model to run as a full-frame nvinfer
    branch (parallel to the main PGIE/tracker chain), not as a per-object SGIE.
    """
    ma_cfg = (pipeline.config.get("models") or {}).get("mapanything") or {}
    gie_id = int(ma_cfg.get("gie_id", 2))
    ma_name = ma_cfg.get("name", "mapanything_fullframe")
    component = pipeline.components.get(ma_name)
    if component is None:
        raise KeyError(f"mapanything component '{ma_name}' missing in pipeline graph")

    processor = MapAnythingProcessor(
        pipeline=pipeline,
        storage=storage,
        depth_pub=depth_pub,
        gie_id=gie_id,
        camera_labels=camera_labels or getattr(pipeline, "camera_labels", {}) or {},
        failure_callback=failure_callback,
    )
    component.config["_mapanything_processor"] = processor
    pipeline.mapanything_processor = processor

    if pipeline.ds_pipeline is None or BatchMetadataOperator is None or Probe is None:
        logger.debug("Stored MapAnything processor for lazy execution (pyservicemaker unavailable)")
        return

    try:
        probe = Probe("mapanything_postprocess", _MapAnythingOperator(processor))
        pipeline.ds_pipeline.attach(component.name, probe)
        logger.info("Attached MapAnything post-process probe to %s", component.name)
    except Exception:  # pragma: no cover - depends on DS runtime availability
        logger.exception("Failed to attach MapAnything post-process probe")


def attach_pose_feature_hook(
    pipeline: "DS8Pipeline",
    *,
    camera_labels: Optional[Mapping[int, str]] = None,
) -> None:
    """Attach the YOLO26 pose feature hook."""

    pose_cfg = (pipeline.config.get("models") or {}).get("pose") or {}
    if not isinstance(pose_cfg, Mapping):
        pose_cfg = {}
    enabled = bool(pose_cfg.get("enable", True)) and any(
        key in pose_cfg for key in ("config-file-path", "engine", "name")
    )
    if not enabled:
        logger.info("Pose SGIE disabled or missing; skipping pose feature hook")
        return

    env_flag = os.environ.get("NOESIS_POSE_FEATURES_ENABLED", "1")
    if str(env_flag).strip().lower() not in ("1", "true", "yes", "on"):
        logger.info("Pose features disabled (NOESIS_POSE_FEATURES_ENABLED=%s)", env_flag)
        return

    gie_id = int(pose_cfg.get("gie_id", pose_cfg.get("gie-id", 4) or 4))
    pose_name = str(pose_cfg.get("name") or "yolo26_pose").strip() or "yolo26_pose"
    component = pipeline.components.get(pose_name)
    if component is None:
        raise KeyError(f"pose component '{pose_name}' missing in pipeline graph")

    model_size = pose_cfg.get("model_size") or pose_cfg.get("input_size")
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

    score_threshold = float(pose_cfg.get("score_threshold", 0.25) or 0.25)
    kpt_threshold = float(pose_cfg.get("kpt_threshold", 0.35) or 0.35)
    letterbox = bool(pose_cfg.get("letterbox", True))

    processor = PoseFeatureProcessor(
        pipeline=pipeline,
        gie_id=gie_id,
        model_size=(model_w, model_h),
        score_threshold=score_threshold,
        kpt_threshold=kpt_threshold,
        letterbox=letterbox,
        camera_labels=camera_labels or {},
    )
    component.config["_pose_feature_processor"] = processor

    if pipeline.ds_pipeline is None or BatchMetadataOperator is None or Probe is None:
        logger.debug("Stored pose feature processor for lazy execution (pyservicemaker unavailable)")
        return

    try:
        probe = Probe("pose_features", _PoseFeatureOperator(processor))
        pipeline.ds_pipeline.attach(component.name, probe)
        logger.info("Attached pose feature probe to %s", component.name)
    except Exception:  # pragma: no cover - depends on DS runtime availability
        logger.exception("Failed to attach pose feature probe")


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
    diagnostics_logger: "TrackingDiagnosticsLogger" | None = None,
) -> None:
    """Attach a BatchMetadataOperator that extracts analytics telemetry."""
    if tracking_pub is None:
        raise ValueError("tracking_pub must be provided for analytics telemetry")

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
        diagnostics_logger=diagnostics_logger,
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


def attach_person_bbox_overlay_hook(
    pipeline: "DS8Pipeline",
    *,
    config: Mapping[str, Any] | None = None,
) -> None:
    """Attach a V3DT-only image-space bbox overlay for person masks."""
    osd_component = pipeline.components.get("osd")
    if osd_component is None:
        raise KeyError("osd component missing; cannot attach person bbox overlay hook")

    vis_cfg = pipeline.config.get("visualization") or {}
    bbox_cfg: Mapping[str, Any] = {}
    if config is not None:
        bbox_cfg = config
    elif isinstance(vis_cfg, Mapping):
        raw_bbox = vis_cfg.get("person_bbox_overlay") or vis_cfg.get("bbox_overlay") or {}
        if isinstance(raw_bbox, Mapping):
            bbox_cfg = raw_bbox

    settings = PersonBBoxOverlayConfig.from_mapping(bbox_cfg)
    if not settings.enabled:
        logger.info("Person bbox overlay disabled")
        return

    # Attach before nvmultistreamtiler so DisplayMeta remains in source-image coordinates;
    # the tiler will scale it into the mosaic exactly like tracker-generated display meta.
    attach_component = pipeline.components.get("tracking_telemetry_stage") or pipeline.components.get("tiler") or osd_component
    processor = PersonBBoxOverlayProcessor(pipeline=pipeline, config=settings)
    attach_component.config["_person_bbox_overlay_processor"] = processor
    setattr(pipeline, "person_bbox_overlay_processor", processor)

    if pipeline.ds_pipeline is None or BatchMetadataOperator is None or Probe is None:
        logger.debug("Stored person bbox overlay processor for lazy execution (pyservicemaker unavailable)")
        return

    try:
        probe = Probe("person_bbox_overlay", _PersonBBoxOverlayOperator(processor))
        pipeline.ds_pipeline.attach(attach_component.name, probe)
        logger.info("Attached person bbox overlay probe to %s", attach_component.name)
    except Exception:  # pragma: no cover - depends on DS runtime availability
        logger.exception("Failed to attach person bbox overlay probe")


def attach_pose_keypoint_overlay_hook(pipeline: "DS8Pipeline") -> None:
    """Attach a DS8 pose keypoint overlay hook (draws skeletons on the mosaic)."""
    vis_cfg = pipeline.config.get("visualization") or {}
    enabled = False
    if isinstance(vis_cfg, Mapping):
        enabled = bool(vis_cfg.get("display_keypoints", False))
    if not enabled:
        logger.info("Pose keypoint overlay disabled (visualization.display_keypoints=false)")
        return

    pose_cfg = (pipeline.config.get("models") or {}).get("pose") or {}
    if not isinstance(pose_cfg, Mapping):
        pose_cfg = {}
    pose_enabled = bool(pose_cfg.get("enable", True)) and any(
        key in pose_cfg for key in ("config-file-path", "engine", "name")
    )
    if not pose_enabled:
        logger.info("Pose SGIE disabled or missing; skipping pose keypoint overlay")
        return

    osd_component = pipeline.components.get("osd")
    if osd_component is None:
        raise KeyError("osd component missing; cannot attach pose keypoint overlay")

    attach_component = pipeline.components.get("tiler") or osd_component

    gie_id = int(pose_cfg.get("gie_id", pose_cfg.get("gie-id", 4) or 4))
    model_size = pose_cfg.get("model_size") or pose_cfg.get("input_size")
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

    score_threshold = float(pose_cfg.get("score_threshold", 0.25) or 0.25)
    kpt_threshold = float(pose_cfg.get("kpt_threshold", 0.35) or 0.35)
    letterbox = bool(pose_cfg.get("letterbox", True))

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
    if pyds is not None:
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

    if pyds is not None:
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

    if count and not layers_seq and pyds is not None:
        try:  # pragma: no cover - requires DeepStream runtime
            layers_seq = [pyds.get_nvds_LayerInfo(tensor_meta, i) for i in range(count)]  # type: ignore[attr-defined]
        except Exception:
            layers_seq = []

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
    camera_labels: Mapping[int, str] = field(default_factory=dict)
    failure_callback: Optional[Callable[[BaseException], None]] = field(
        default=None, repr=False
    )
    tensor_samples: int = 0
    _async_enabled: bool = field(default=True, init=False, repr=False)
    _async_queue: "queue.Queue[Any]" = field(default_factory=lambda: queue.Queue(maxsize=32), init=False, repr=False)
    _async_thread: Optional[threading.Thread] = field(default=None, init=False, repr=False)
    _async_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _async_idle: threading.Condition = field(init=False, repr=False)
    _async_failure: Optional[str] = field(default=None, init=False, repr=False)
    _async_accepting: bool = field(default=True, init=False, repr=False)
    _async_active_captures: int = field(default=0, init=False, repr=False)
    _async_stop_enqueued: bool = field(default=False, init=False, repr=False)
    _async_stopped: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _async_shutdown_complete: bool = field(default=False, init=False, repr=False)
    _dropped_jobs: int = field(default=0, init=False, repr=False)
    _last_drop_log: float = field(default=0.0, init=False, repr=False)
    _dewarper_fov_masks: Dict[Tuple[int, int, int], np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _storage_commit_timeout_s: float = field(default=30.0, init=False, repr=False)

    def __post_init__(self) -> None:
        # MapAnything host conversion must stay off the probe thread in DS8 production.
        self._async_enabled = True
        self._async_idle = threading.Condition(self._async_lock)
        self._storage_commit_timeout_s = resolve_depth_store_commit_timeout_s()

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
        if not isinstance(source_configs, Sequence) or int(source_id) >= len(source_configs):
            raise RuntimeError(f"dewarper validity source {source_id} is absent from pipeline sources")
        source_cfg = source_configs[int(source_id)]
        if not isinstance(source_cfg, Mapping):
            raise RuntimeError(f"pipeline source {source_id} is not a mapping")
        spec = _load_dewarper_fov_spec(
            source_cfg=source_cfg,
            pipeline_yaml_path=getattr(self.pipeline, "yaml_path", None),
            mask_cfg=mask_cfg,
            repo_root=Path(__file__).resolve().parents[2],
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

    def _to_numpy(self, tensor: Any) -> Optional[np.ndarray]:
        """Convert pyservicemaker Tensor to a CPU numpy array via DLPack.

        NOTE: Prefer torch's DLPack bridge over numpy.from_dlpack because some numpy
        builds will attempt to wrap GPU tensors directly, which can break downstream
        CPU numpy operations (e.g. np.isfinite) and stall the pipeline.
        """
        dlpack_fn = getattr(tensor, "__dlpack__", None)
        if callable(dlpack_fn):
            try:
                with _DLPACK_HOST_READ_LOCK:
                    import torch.utils.dlpack as torch_dlpack
                    import torch

                    start_ns = time.perf_counter_ns()
                    # DLPack expects the consumer to pass its CUDA stream handle.
                    # Use torch's current stream when available.
                    stream = 0
                    try:
                        if torch.cuda.is_available():
                            stream = int(torch.cuda.current_stream().cuda_stream)
                    except Exception:
                        stream = 0
                    capsule = dlpack_fn(stream)
                    torch_tensor = torch_dlpack.from_dlpack(capsule)
                    arr = torch_tensor.detach().cpu().numpy()
                    _increment_core_counter("tensor_host_copies_total.mapanything")
                    _CORE_PATH_INSTRUMENTATION.record_boundary_serialization_prep(
                        metric="mapanything.tensor_dlpack_to_host",
                        duration_ns=time.perf_counter_ns() - start_ns,
                        payload_bytes=int(getattr(arr, "nbytes", 0) or 0),
                    )
                    return arr
            except Exception as exc:
                logger.debug(
                    "Torch DLPack conversion failed for tensor (device=%s, dtype=%s, shape=%s): %s",
                    getattr(tensor, "device_type", None),
                    getattr(tensor, "dtype", None),
                    getattr(tensor, "shape", None),
                    exc,
                )
        return None

    def _target_frame_shape(self, frame_meta: Any, depth_shape: Tuple[int, int]) -> Tuple[int, int]:
        frame_w = 0
        frame_h = 0
        try:
            frame_w = int(_meta_lookup(frame_meta, "source_frame_width", "frame_width", "width", default=0) or 0)
            frame_h = int(_meta_lookup(frame_meta, "source_frame_height", "frame_height", "height", default=0) or 0)
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

    def handle_nvds_tensor_ds8(self, frame_meta: Any, tensor_meta: Any) -> Optional[DepthResult]:
        """Handle DS8 pyservicemaker TensorOutputUserMetadata."""
        if not self.pipeline.depth_enabled:
            logger.debug("Depth disabled; dropping MapAnything tensors before DS8 conversion")
            return None
        self._begin_async_capture()
        try:
            # DS8 API: tensor_meta.get_layers() returns dict[str, Tensor]
            layers = tensor_meta.get_layers() or {}
            if not layers:
                return None

            depth_tensor = layers.get("depth") or layers.get("depth_z") or layers.get("disp")
            conf_tensor = layers.get("confidence") or layers.get("conf")
            mask_tensor = layers.get("mask") or layers.get("valid")
            if depth_tensor is None and len(layers) == 1:
                depth_tensor = next(iter(layers.values()))
            if depth_tensor is None:
                return None

            try:
                depth_tensor = depth_tensor.clone()
            except Exception:
                return None
            if conf_tensor is not None:
                try:
                    conf_tensor = conf_tensor.clone()
                except Exception:
                    conf_tensor = None
            if mask_tensor is not None:
                try:
                    mask_tensor = mask_tensor.clone()
                except Exception:
                    mask_tensor = None

            source_id = int(_meta_lookup(frame_meta, "pad_index", "source_id", default=0))
            frame_id = int(_meta_lookup(frame_meta, "frame_num", "frame_number", default=0))
            pts_ns = int(_meta_lookup(frame_meta, "buf_pts", "buffer_pts", default=0))
            batch_id_raw = _meta_lookup(frame_meta, "batch_id", "batchId", "batch_index", default=None)
            batch_id: int | None
            try:
                batch_id = int(batch_id_raw) if batch_id_raw is not None else None
            except Exception:
                batch_id = None
            if batch_id is None and source_id >= 0:
                # Best-effort fallback: DS8 often uses pad/source id as the batch index.
                batch_id = int(source_id)
            job = _MapAnythingJob(
                source_id=source_id,
                frame_id=frame_id,
                pts_ns=pts_ns,
                batch_id=batch_id,
                depth=depth_tensor,
                confidence=conf_tensor,
                mask=mask_tensor,
            )

            self._enqueue_async_job(job)
            return None
        except Exception:
            logger.exception("Failed to extract tensor layers from DS8 TensorOutputUserMetadata")
            raise
        finally:
            self._end_async_capture()

    def _raise_async_failure(self) -> None:
        with self._async_lock:
            failure = self._async_failure
        if failure is not None:
            raise RuntimeError(
                f"MapAnything async postprocess is poisoned: {failure}"
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
            logger.exception("Async MapAnything postprocess failed")
            callback = self.failure_callback
            if callable(callback):
                try:
                    callback(exc)
                except Exception:
                    logger.exception("MapAnything runtime failure callback failed")

    @staticmethod
    def _slice_async_batch(arr: np.ndarray, batch_id: int | None) -> np.ndarray:
        if batch_id is None:
            return arr
        if arr.ndim == 4:
            batch_size = int(arr.shape[0] or 0)
            if batch_size > 0:
                index = batch_id if 0 <= batch_id < batch_size else 0
                return arr[index]
        if arr.ndim == 3 and arr.shape[0] > 1:
            batch_size = int(arr.shape[0] or 0)
            index = batch_id if 0 <= batch_id < batch_size else 0
            return arr[index]
        return arr

    def _async_worker_loop(self) -> None:
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
                        tensors: Dict[str, np.ndarray] = {}
                        depth_arr = self._to_numpy(job.depth)
                        if depth_arr is None:
                            continue
                        tensors["depth"] = self._slice_async_batch(
                            depth_arr, job.batch_id
                        )
                        if job.confidence is not None:
                            conf_arr = self._to_numpy(job.confidence)
                            if conf_arr is not None:
                                tensors["confidence"] = self._slice_async_batch(
                                    conf_arr, job.batch_id
                                )
                        if job.mask is not None:
                            mask_arr = self._to_numpy(job.mask)
                            if mask_arr is not None:
                                tensors["mask"] = self._slice_async_batch(
                                    mask_arr, job.batch_id
                                )
                        self.handle_numpy_arrays(
                            source_id=int(job.source_id),
                            frame_id=int(job.frame_id),
                            pts_ns=int(job.pts_ns),
                            tensors=tensors,
                            captured_while_enabled=True,
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

    def _start_async_worker_locked(self) -> None:
        if self._async_failure is not None:
            raise RuntimeError(
                "MapAnything async postprocess is poisoned: "
                f"{self._async_failure}"
            )
        if not self._async_accepting and self._async_active_captures <= 0:
            raise RuntimeError("MapAnything async postprocess is shutting down")
        if self._async_thread is not None and self._async_thread.is_alive():
            return
        if self._async_thread is not None:
            raise RuntimeError(
                "MapAnything async postprocess worker cannot be restarted"
            )
        self._async_stopped.clear()
        self._async_thread = threading.Thread(
            target=self._async_worker_loop,
            name="MapAnythingPostprocess",
            daemon=False,
        )
        self._async_thread.start()

    def _start_async_worker(self) -> None:
        with self._async_idle:
            self._start_async_worker_locked()

    def _begin_async_capture(self) -> None:
        with self._async_idle:
            if self._async_failure is not None:
                raise RuntimeError(
                    "MapAnything async postprocess is poisoned: "
                    f"{self._async_failure}"
                )
            if not self._async_accepting:
                raise RuntimeError("MapAnything async postprocess is shutting down")
            self._async_active_captures += 1

    def _end_async_capture(self) -> None:
        with self._async_idle:
            if self._async_active_captures <= 0:
                raise RuntimeError(
                    "MapAnything async capture ownership underflow"
                )
            self._async_active_captures -= 1
            self._async_idle.notify_all()

    def _enqueue_async_job(self, job: "_MapAnythingJob") -> bool:
        # An admitted capture may enqueue after shutdown closes admission.
        # Shutdown waits for every capture lease before appending the FIFO stop
        # sentinel, so the sentinel cannot overtake this accepted job.
        with self._async_idle:
            self._start_async_worker_locked()
            if self._async_failure is not None:
                raise RuntimeError(
                    "MapAnything async postprocess is poisoned: "
                    f"{self._async_failure}"
                )
            try:
                self._async_queue.put_nowait(job)
            except queue.Full as exc:
                self._dropped_jobs += 1
                _increment_core_counter("mapanything_async_queue_full_total")
                raise RuntimeError(
                    "MapAnything bounded async postprocess queue is full"
                ) from exc
        return True

    def async_shutdown_quiesced(self) -> bool:
        with self._async_lock:
            thread = self._async_thread
            active_captures = int(self._async_active_captures)
            stopped = bool(self._async_stopped.is_set())
        unfinished = self._async_unfinished_task_count()
        return bool(
            active_captures == 0
            and unfinished == 0
            and (
                thread is None
                or (not thread.is_alive() and stopped)
            )
        )

    def wait_idle(self, *, timeout_s: float = 5.0) -> MapAnythingIdleReceipt:
        """Wait for admitted captures and queued jobs without closing admission."""

        timeout = float(timeout_s)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("MapAnything idle timeout must be positive")
        deadline = time.monotonic() + timeout
        with self._async_idle:
            while True:
                if self._async_failure is not None:
                    raise RuntimeError(
                        "MapAnything async postprocess is poisoned: "
                        f"{self._async_failure}"
                    )
                if not self._async_accepting or self._async_shutdown_complete:
                    raise RuntimeError(
                        "MapAnything async postprocess is shutting down"
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
                        "MapAnything has queued work without a live owned worker"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        "MapAnything did not become idle before timeout "
                        f"(active_captures={active_captures} unfinished={unfinished})"
                    )
                self._async_idle.wait(timeout=min(remaining, 0.05))

    def shutdown(self, *, wait: bool = True, timeout_s: float = 5.0) -> None:
        """Close capture admission, drain accepted jobs, and join the worker.

        The stop sentinel is enqueued only after every probe-local capture
        lease completes. FIFO ordering then proves that no accepted job can be
        overtaken during teardown.
        """
        if not wait:
            raise ValueError("MapAnything shutdown requires wait=True")
        timeout = float(timeout_s)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("MapAnything shutdown timeout must be positive")
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
                            "MapAnything active tensor capture did not quiesce "
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
                    "MapAnything has queued work without an owned worker"
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
                    "MapAnything shutdown timed out before stop enqueue"
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
                    "MapAnything stop sentinel could not enter the bounded queue"
                ) from exc

        remaining = max(0.0, deadline - time.monotonic())
        thread.join(timeout=remaining)
        unfinished = self._async_unfinished_task_count()
        if thread.is_alive() or not self._async_stopped.is_set() or unfinished != 0:
            _increment_core_counter("mapanything_async_shutdown_failures_total")
            raise RuntimeError(
                "MapAnything async worker shutdown is unresolved "
                f"(alive={thread.is_alive()} stopped={self._async_stopped.is_set()} "
                f"unfinished={unfinished})"
            )

        with self._async_idle:
            self._async_shutdown_complete = True
            self._async_idle.notify_all()
        _increment_core_counter("mapanything_async_shutdown_total")
        self._raise_async_failure()

    def handle_nvds_tensor(self, frame_meta: Any, tensor_meta: Any) -> Optional[DepthResult]:
        if pyds is None:
            logger.debug("pyds unavailable; skipping NvDs tensor processing")
            return None
        try:
            tensors = _extract_tensor_layers(tensor_meta)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to extract tensor layers from NvDsInferTensorMeta")
            return None
        self.tensor_samples += 1
        return self._emit_from_tensors(frame_meta, tensors)

    def handle_numpy_arrays(
        self,
        *,
        source_id: int,
        frame_id: int,
        pts_ns: int,
        tensors: Mapping[str, np.ndarray],
        captured_while_enabled: bool = False,
    ) -> Optional[DepthResult]:
        frame_meta = {
            "pad_index": source_id,
            "frame_num": frame_id,
            "buf_pts": pts_ns,
        }
        self.tensor_samples += 1
        return self._emit_from_tensors(
            frame_meta,
            tensors,
            captured_while_enabled=bool(captured_while_enabled),
        )

    def handle_native_frame_ds8(self, frame_meta: Any) -> Optional[DepthResult]:
        """Read MapAnything tensors from raw DeepStream frame metadata.

        Some DS8 Service Maker builds omit full-frame multi-output tensors from
        frame_meta.tensor_items even though the underlying NvDsInferTensorMeta is
        present. The native helper keeps this on the canonical DS8 metadata path.
        """
        if not self.pipeline.depth_enabled:
            return None
        if noesis_depth_tracking_tensor_ext is None:
            return None
        capture_fn = getattr(noesis_depth_tracking_tensor_ext, "capture_tensor_layers", None)
        if not callable(capture_fn):
            return None
        self._begin_async_capture()
        try:
            try:
                layers = capture_fn(frame_meta, int(self.gie_id))
            except Exception:
                logger.debug("Native MapAnything tensor capture failed", exc_info=True)
                return None
            if not layers:
                return None
            try:
                tensors = {str(key): np.asarray(value) for key, value in dict(layers).items()}
            except Exception:
                logger.debug("Native MapAnything tensor payload was not array-like", exc_info=True)
                return None
            if not tensors:
                return None
            source_id = int(_meta_lookup(frame_meta, "pad_index", "source_id", default=0))
            frame_id = int(_meta_lookup(frame_meta, "frame_num", "frame_number", default=0))
            pts_ns = int(_meta_lookup(frame_meta, "buf_pts", "buffer_pts", default=0))
            return self.handle_numpy_arrays(
                source_id=source_id,
                frame_id=frame_id,
                pts_ns=pts_ns,
                tensors=tensors,
                captured_while_enabled=True,
            )
        finally:
            self._end_async_capture()

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
        source_id = int(_meta_lookup(frame_meta, "pad_index", "source_id", default=0))
        fov_mask = self._dewarper_validity_mask(source_id, (depth.shape[1], depth.shape[0]))
        finite_depth = np.isfinite(depth) & (depth > 0.0)
        if fov_mask is not None:
            fov_mask = np.asarray(fov_mask, dtype=bool)
            if fov_mask.shape != depth.shape:
                raise RuntimeError(
                    "calibrated MapAnything FoV mask shape "
                    f"{fov_mask.shape} does not match depth {depth.shape}"
                )
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
        pts_ns = int(_meta_lookup(frame_meta, "buf_pts", "buffer_pts", default=0))
        # Some DS8 bindings expose GStreamer PTS (relative) rather than epoch time.
        # Epoch ns is currently ~1e18; treat smaller values as non-epoch and fall back.
        if pts_ns <= 0 or pts_ns < 100_000_000_000_000_000:
            pts_ns = time.time_ns()
        ts_s = max(0, pts_ns // 1_000_000_000)
        ts_us = max(0, pts_ns // 1_000)
        if self.tensor_samples <= 5 or (self.tensor_samples % 50) == 0:
            logger.debug(
                "MapAnything tensor meta timestamps: source=%s frame=%s pts_ns=%s ts_us=%s",
                source_id,
                frame_id,
                pts_ns,
                ts_us,
            )

        camera_id = str(self.camera_labels.get(source_id, source_id))
        mask_u8 = mask.astype(np.uint8, copy=False)

        write_handle = self.storage.store(
            camera_id,
            ts_us,
            depth,
            conf_array,
            mask_u8,
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
class _MapAnythingJob:
    source_id: int
    frame_id: int
    pts_ns: int
    batch_id: int | None
    depth: Any
    confidence: Any | None
    mask: Any | None


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


@dataclass(frozen=True)
class PersonBBoxOverlayConfig:
    enabled: bool = True
    class_ids: frozenset[int] = field(default_factory=lambda: frozenset({0}))
    line_width: int = 3
    max_boxes: int = 24
    max_display_metas: int = 32
    color: Tuple[float, float, float] = (0.0, 0.92, 1.0)
    alpha: float = 1.0
    expand_px: float = 0.0
    shape: str = "cuboid"
    cuboid_geometry: str = "anchored"
    cuboid_depth_width_ratio: float = 0.25
    cuboid_depth_height_ratio: float = 0.12
    cuboid_min_depth_px: float = 24.0
    cuboid_max_depth_px: float = 96.0
    draw_anchor_rect: bool = True
    require_v3dt_meta: bool = True

    def __post_init__(self) -> None:
        cls_ids: set[int] = set()
        for item in self.class_ids:
            try:
                cls_ids.add(int(item))
            except Exception:
                continue
        if not cls_ids:
            cls_ids = {0}
        rgb = tuple(float(max(0.0, min(1.0, float(v)))) for v in self.color[:3])
        if len(rgb) != 3:
            rgb = (0.0, 0.92, 1.0)
        object.__setattr__(self, "class_ids", frozenset(cls_ids))
        object.__setattr__(self, "line_width", max(1, int(self.line_width)))
        object.__setattr__(self, "max_boxes", max(1, int(self.max_boxes)))
        object.__setattr__(self, "max_display_metas", max(1, int(self.max_display_metas)))
        object.__setattr__(self, "color", rgb)
        object.__setattr__(self, "alpha", float(max(0.0, min(1.0, float(self.alpha)))))
        object.__setattr__(self, "expand_px", max(0.0, float(self.expand_px)))
        shape = str(self.shape or "cuboid").strip().lower()
        if shape in ("2d", "bbox", "box", "rect", "rectangle"):
            shape = "rectangle"
        elif shape not in ("cuboid", "3d"):
            shape = "cuboid"
        if shape == "3d":
            shape = "cuboid"
        object.__setattr__(self, "shape", shape)
        geometry = str(self.cuboid_geometry or "anchored").strip().lower()
        if geometry in ("projected", "projection", "projected_normalized", "v3dt"):
            geometry = "projected_normalized"
        elif geometry not in ("anchored", "screen", "screen_space"):
            geometry = "anchored"
        if geometry in ("screen", "screen_space"):
            geometry = "anchored"
        object.__setattr__(self, "cuboid_geometry", geometry)
        object.__setattr__(self, "cuboid_depth_width_ratio", max(0.0, float(self.cuboid_depth_width_ratio)))
        object.__setattr__(self, "cuboid_depth_height_ratio", max(0.0, float(self.cuboid_depth_height_ratio)))
        object.__setattr__(self, "cuboid_min_depth_px", max(0.0, float(self.cuboid_min_depth_px)))
        object.__setattr__(self, "cuboid_max_depth_px", max(1.0, float(self.cuboid_max_depth_px)))
        object.__setattr__(self, "draw_anchor_rect", bool(self.draw_anchor_rect))
        object.__setattr__(self, "require_v3dt_meta", bool(self.require_v3dt_meta))

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> "PersonBBoxOverlayConfig":
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

        def _class_ids(value: Any) -> frozenset[int]:
            parsed: set[int] = set()
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    try:
                        parsed.add(int(item))
                    except Exception:
                        continue
            elif value is not None:
                try:
                    parsed.add(int(value))
                except Exception:
                    pass
            if not parsed:
                parsed = {0}
            return frozenset(parsed)

        def _color(value: Any) -> Tuple[float, float, float]:
            if isinstance(value, str):
                raw = value.strip().lower()
                named = {
                    "cyan": (0.0, 0.92, 1.0),
                    "blue": (0.1, 0.45, 1.0),
                    "white": (1.0, 1.0, 1.0),
                    "green": (0.0, 1.0, 0.2),
                    "yellow": (1.0, 0.9, 0.0),
                }
                if raw in named:
                    return named[raw]
                if raw.startswith("#") and len(raw) == 7:
                    try:
                        return (
                            int(raw[1:3], 16) / 255.0,
                            int(raw[3:5], 16) / 255.0,
                            int(raw[5:7], 16) / 255.0,
                        )
                    except Exception:
                        return (0.0, 0.92, 1.0)
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                try:
                    vals = [float(value[0]), float(value[1]), float(value[2])]
                    if any(v > 1.0 for v in vals):
                        vals = [v / 255.0 for v in vals]
                    return tuple(float(max(0.0, min(1.0, v))) for v in vals[:3])  # type: ignore[return-value]
                except Exception:
                    return (0.0, 0.92, 1.0)
            return (0.0, 0.92, 1.0)

        return cls(
            enabled=_bool(cfg.get("enabled"), True),
            class_ids=_class_ids(cfg.get("class_ids")),
            line_width=_int(cfg.get("line_width"), 3),
            max_boxes=_int(cfg.get("max_boxes"), 24),
            max_display_metas=_int(cfg.get("max_display_metas"), 32),
            color=_color(cfg.get("color", "cyan")),
            alpha=_float(cfg.get("alpha"), 1.0),
            expand_px=_float(cfg.get("expand_px"), 0.0),
            shape=str(cfg.get("shape") or cfg.get("mode") or "cuboid"),
            cuboid_geometry=str(cfg.get("cuboid_geometry") or "anchored"),
            cuboid_depth_width_ratio=_float(cfg.get("cuboid_depth_width_ratio"), 0.25),
            cuboid_depth_height_ratio=_float(cfg.get("cuboid_depth_height_ratio"), 0.12),
            cuboid_min_depth_px=_float(cfg.get("cuboid_min_depth_px"), 24.0),
            cuboid_max_depth_px=_float(cfg.get("cuboid_max_depth_px"), 96.0),
            draw_anchor_rect=_bool(cfg.get("draw_anchor_rect"), True),
            require_v3dt_meta=_bool(cfg.get("require_v3dt_meta"), True),
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


# Shared person-ground types (Phases 1–6). Historical names retained for tests/call sites.
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


class _AlignedDepthFrameStore:
    def __init__(self, max_entries: int = 16) -> None:
        self._max_entries = max(2, int(max_entries))
        self._entries: "OrderedDict[FrameKey, _AlignedDepthFrame]" = OrderedDict()

    def put(self, frame: _AlignedDepthFrame) -> None:
        self._entries[frame.key] = frame
        self._entries.move_to_end(frame.key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def resolve(
        self,
        *,
        source_id: int,
        frame_id: int,
        pts_us: int,
        max_age_frames: int,
    ) -> Tuple[Optional[_AlignedDepthFrame], int, float]:
        exact = self._entries.get((int(source_id), int(frame_id), int(pts_us)))
        if exact is not None:
            return exact, 0, 0.0
        max_age = max(0, int(max_age_frames))
        if max_age <= 0:
            return None, 0, 0.0
        for candidate in reversed(list(self._entries.values())):
            if int(candidate.source_id) != int(source_id):
                continue
            age_frames = int(frame_id) - int(candidate.frame_id)
            if age_frames < 0 or age_frames > max_age:
                continue
            age_ms = max(0.0, float(int(pts_us) - int(candidate.pts_us)) / 1000.0)
            return candidate, age_frames, age_ms
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

    def _analytics_track_map(self, sensor_id: int) -> Dict[int, Dict[str, Any]]:
        processor = getattr(self.pipeline, "analytics_telemetry_processor", None)
        getter = getattr(processor, "get_active_track_map", None)
        if not callable(getter):
            return {}
        try:
            track_map = getter(int(sensor_id)) or {}
        except Exception:
            return {}
        return dict(track_map) if isinstance(track_map, Mapping) else {}

    def _resolve_calibration(self, sensor_id: int, camera_id: str) -> Any:
        provider = getattr(self.pipeline, "bev_calibration", None)
        snapshot = getattr(provider, "snapshot", None)
        if not callable(snapshot):
            return None
        try:
            return snapshot(int(sensor_id), str(camera_id))
        except Exception:
            return None

    def _frame_source_size(self, frame_meta: Any, calib: Any | None = None) -> Tuple[int, int]:
        try:
            frame_w = int(_meta_lookup(frame_meta, "source_frame_width", "frame_width", "width", default=0) or 0)
            frame_h = int(_meta_lookup(frame_meta, "source_frame_height", "frame_height", "height", default=0) or 0)
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
        rect = getattr(frame_meta, "compositor_rect", None)
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

    def _configured_tile_rect(self, sensor_id: int) -> Optional[Tuple[float, float, float, float]]:
        tiler = self.pipeline.components.get("tiler")
        tiler_cfg = tiler.config if tiler is not None and isinstance(tiler.config, dict) else {}
        try:
            mosaic_w = float(tiler_cfg.get("width", 0) or 0)
            mosaic_h = float(tiler_cfg.get("height", 0) or 0)
            cols = int(tiler_cfg.get("columns", 0) or 0)
            rows = int(tiler_cfg.get("rows", 0) or 0)
        except Exception:
            return None
        if mosaic_w <= 0.0 or mosaic_h <= 0.0 or cols <= 0 or rows <= 0:
            return None
        try:
            source_count = max(1, len(getattr(self.pipeline, "camera_labels", {}) or {}))
        except Exception:
            source_count = cols * rows
        source_count = max(1, int(source_count))
        tile_index = int(sensor_id)
        if tile_index < 0 or tile_index >= source_count:
            return None
        col = tile_index % cols
        row = tile_index // cols
        if row >= rows:
            return None
        tile_w = mosaic_w / float(cols)
        tile_h = mosaic_h / float(rows)
        return float(col) * tile_w, float(row) * tile_h, tile_w, tile_h

    def _source_to_mosaic(self, frame_meta: Any, u: float, v: float, source_size: Tuple[int, int]) -> Tuple[float, float]:
        src_w, src_h = source_size
        comp = self._frame_compositor_rect(frame_meta)
        if comp is None:
            comp = self._configured_tile_rect(self._frame_source_id(frame_meta))
        if comp is None or src_w <= 0 or src_h <= 0:
            return float(u), float(v)
        left, top, width, height = comp
        x = float(left) + float(u) * (float(width) / float(src_w))
        y = float(top) + float(v) * (float(height) / float(src_h))
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
        return self._source_to_mosaic(frame_meta, float(uv[0]), float(uv[1]), source_size)

    def _resolve_active_anchor(
        self,
        sensor_id: int,
        frame_meta: Any,
        obj_meta: Any,
        track_id: int,
        state: _TrailTrackState,
        now: float,
    ) -> Tuple[float, float, bool]:
        rect = getattr(obj_meta, "rect_params", None)
        if rect is None:
            raise ValueError("rect_params required for trail anchor")
        left = float(getattr(rect, "left", 0.0) or 0.0)
        top = float(getattr(rect, "top", 0.0) or 0.0)
        width = float(getattr(rect, "width", 0.0) or 0.0)
        height = float(getattr(rect, "height", 0.0) or 0.0)
        x_bbox = float(left) + float(width) * 0.5
        y_bbox = float(top) + float(height)
        if self.config.anchor_mode != "floor_plane_gravity_drop":
            return float(x_bbox), float(y_bbox), False

        track_map = self._analytics_track_map(sensor_id)
        track = track_map.get(int(track_id))
        if not isinstance(track, Mapping):
            return float(x_bbox), float(y_bbox), False

        camera_id = self._camera_id_for_sensor(sensor_id)
        calib = self._resolve_calibration(sensor_id, camera_id)
        has_calib = calib is not None and getattr(calib, "intrinsics", None) is not None and getattr(calib, "extrinsics_col_major", None) is not None
        flip_u, flip_v = self._infer_image_flips(camera_id, calib) if has_calib else (False, False)

        def _world_to_mosaic(world_point: np.ndarray) -> Optional[Tuple[float, float]]:
            if not has_calib:
                return None
            uv = project_world_to_image(
                world_point,
                calib.intrinsics,
                calib.extrinsics_col_major,
                tuple(int(x) for x in calib.image_size),
                unit_scale=1.0,
                flip_u=bool(flip_u),
                flip_v=bool(flip_v),
            )
            if uv is None:
                return None
            return self._source_to_mosaic(
                frame_meta,
                float(uv[0]),
                float(uv[1]),
                self._frame_source_size(frame_meta, calib),
            )

        measured_world = track.get("world")
        if not isinstance(measured_world, (list, tuple)) or len(measured_world) < 3 or track.get("world_valid") is not True:
            if not has_calib:
                return float(x_bbox), float(y_bbox), False
            floor_world = None
            if state.height_ref_scene is not None:
                floor_world = self._gravity_drop_world(
                    calib,
                    [left, top, width, height],
                    float(state.height_ref_scene),
                    flip_u=bool(flip_u),
                    flip_v=bool(flip_v),
                )
            if floor_world is None:
                floor_world = self._bbox_bottom_world(
                    calib,
                    [left, top, width, height],
                    flip_u=bool(flip_u),
                    flip_v=bool(flip_v),
                )
            if floor_world is None:
                return float(x_bbox), float(y_bbox), False
            self._maybe_update_height_reference(
                state,
                calib,
                [left, top, width, height],
                floor_world,
                flip_u=bool(flip_u),
                flip_v=bool(flip_v),
            )
            self._update_world_measurement(state, float(floor_world[0]), float(floor_world[2]), float(now))
            mapped = _world_to_mosaic(floor_world)
            if mapped is None:
                return float(x_bbox), float(y_bbox), False
            return float(mapped[0]), float(mapped[1]), False
        try:
            measured_world_arr = np.asarray(
                [float(measured_world[0]), float(measured_world[1]), float(measured_world[2])],
                dtype=np.float64,
            )
        except Exception:
            return float(x_bbox), float(y_bbox), False
        if has_calib:
            self._maybe_update_height_reference(
                state,
                calib,
                [left, top, width, height],
                measured_world_arr,
                flip_u=bool(flip_u),
                flip_v=bool(flip_v),
            )

        source_uv = None
        # For the mosaic video overlay, keep the trail attached to the observed
        # person anchor. image_base can be a reprojected world point and may drift
        # when the floor/world estimate is still settling.
        for key in ("image_foot", "image_base"):
            uv = track.get(key)
            if isinstance(uv, (list, tuple)) and len(uv) >= 2:
                try:
                    source_uv = (float(uv[0]), float(uv[1]))
                except Exception:
                    source_uv = None
                if source_uv is not None:
                    break
        if source_uv is None:
            mapped = _world_to_mosaic(measured_world_arr)
            if mapped is None:
                return float(x_bbox), float(y_bbox), False
            source_uv = None

        self._update_world_measurement(state, float(measured_world_arr[0]), float(measured_world_arr[2]), float(now))
        if source_uv is None:
            x, y = mapped
        else:
            x, y = self._source_to_mosaic(
                frame_meta,
                float(source_uv[0]),
                float(source_uv[1]),
                self._frame_source_size(frame_meta, calib),
            )
        return float(x), float(y), False

    def _commit_point(
        self,
        state: _TrailTrackState,
        now: float,
        x: float,
        y: float,
        *,
        predicted: bool,
        append_allowed: bool = True,
    ) -> None:
        if state.points:
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

        if self.config.smooth_tau_s > 0.0:
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
        max_x = float(mosaic_w) if mosaic_w > 0 else None
        max_y = float(mosaic_h) if mosaic_h > 0 else None
        max_points_per_track = max(2, int(self.config.max_points_per_track))
        tracks_seen = 0
        present_track_ids: set[int] = set()

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

            x = left + width * 0.5
            y = top + height

            if max_x is not None:
                x = float(max(0.0, min(x, max_x)))
            if max_y is not None:
                y = float(max(0.0, min(y, max_y)))

            state = sensor_tracks.get(track_id)
            if state is None:
                state = _TrailTrackState(points=deque(maxlen=max_points_per_track))
                sensor_tracks[track_id] = state
            state.last_seen_ts = float(now)

            # Always prune old samples so disappeared tracks naturally fade out.
            while state.points and (now - float(state.points[0].ts)) > float(self.config.window_s):
                state.points.popleft()

            if not do_sample:
                continue

            x, y, predicted = self._resolve_active_anchor(
                sensor_id,
                frame_meta,
                obj_meta,
                int(track_id),
                state,
                float(now),
            )
            if max_x is not None:
                x = float(max(0.0, min(x, max_x)))
            if max_y is not None:
                y = float(max(0.0, min(y, max_y)))
            track_map = self._analytics_track_map(sensor_id)
            track_info = track_map.get(int(track_id)) if isinstance(track_map, Mapping) else None
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
                motion_mode = str(track_info.get("motion_mode") or "").strip().lower()
                if motion_mode in ("idle", "sit", "lie"):
                    append_allowed = False
            self._commit_point(
                state,
                float(now),
                float(x),
                float(y),
                predicted=bool(predicted),
                append_allowed=bool(append_allowed),
            )

        if do_sample and self.config.anchor_mode == "floor_plane_gravity_drop":
            for track_id, state in sensor_tracks.items():
                if int(track_id) in present_track_ids:
                    continue
                x_y = self._predict_gap_anchor(sensor_id, frame_meta, state, float(now))
                if x_y is None:
                    continue
                x, y = x_y
                if max_x is not None:
                    x = float(max(0.0, min(float(x), max_x)))
                if max_y is not None:
                    y = float(max(0.0, min(float(y), max_y)))
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


@dataclass
class PersonBBoxOverlayProcessor:
    pipeline: "DS8Pipeline"
    config: PersonBBoxOverlayConfig
    _mosaic_size: Tuple[int, int] = field(default_factory=lambda: (0, 0), init=False)
    _v3dt_projection_by_source: Dict[int, Tuple[np.ndarray, bool]] = field(default_factory=dict, init=False, repr=False)
    _v3dt_projection_warned: bool = field(default=False, init=False, repr=False)
    _debug_last_log_ts: float = field(default=0.0, init=False, repr=False)
    _debug_frames: int = field(default=0, init=False, repr=False)
    _debug_boxes: int = field(default=0, init=False, repr=False)
    _validate_last_log_ts: float = field(default=0.0, init=False, repr=False)
    _validate_objects: int = field(default=0, init=False, repr=False)
    _validate_ok: int = field(default=0, init=False, repr=False)
    _validate_coverages: List[float] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._mosaic_size = self._resolve_mosaic_size()
        self._v3dt_projection_by_source = self._load_v3dt_projection_matrices()

    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[2]

    @classmethod
    def _resolve_repo_path(cls, raw: Any) -> Optional[Path]:
        if raw in (None, ""):
            return None
        try:
            path = Path(str(raw))
        except Exception:
            return None
        if not path.is_absolute():
            path = cls._repo_root() / path
        return path

    def _load_v3dt_projection_matrices(self) -> Dict[int, Tuple[np.ndarray, bool]]:
        tracker_cfg = (getattr(self.pipeline, "config", {}) or {}).get("tracker") or {}
        if not isinstance(tracker_cfg, Mapping):
            return {}
        tracker_path = self._resolve_repo_path(tracker_cfg.get("config-file"))
        if tracker_path is None or not tracker_path.exists():
            return {}
        try:
            with tracker_path.open("r", encoding="utf-8") as handle:
                tracker_data = yaml.safe_load(handle) or {}
        except Exception:
            logger.debug("Unable to read V3DT tracker config for cuboid overlay: %s", tracker_path, exc_info=True)
            return {}
        projection_cfg = tracker_data.get("ObjectModelProjection") if isinstance(tracker_data, Mapping) else None
        if not isinstance(projection_cfg, Mapping):
            return {}
        caminfo_paths = projection_cfg.get("cameraModelFilepath") or []
        if isinstance(caminfo_paths, (str, Path)):
            caminfo_paths = [caminfo_paths]
        if not isinstance(caminfo_paths, (list, tuple)):
            return {}

        loaded: Dict[int, Tuple[np.ndarray, bool]] = {}
        for idx, raw_path in enumerate(caminfo_paths):
            caminfo_path = self._resolve_repo_path(raw_path)
            if caminfo_path is None or not caminfo_path.exists():
                continue
            try:
                with caminfo_path.open("r", encoding="utf-8") as handle:
                    caminfo = yaml.safe_load(handle) or {}
            except Exception:
                logger.debug("Unable to read V3DT camInfo for cuboid overlay: %s", caminfo_path, exc_info=True)
                continue
            key = None
            zero_centered = False
            if "projectionMatrix_3x4_w2p" in caminfo:
                key = "projectionMatrix_3x4_w2p"
            elif "projectionMatrix_3x4" in caminfo:
                key = "projectionMatrix_3x4"
                zero_centered = True
            if key is None:
                continue
            try:
                matrix = np.asarray(caminfo[key], dtype=np.float64).reshape((3, 4))
            except Exception:
                logger.debug("Invalid V3DT projection matrix in %s", caminfo_path, exc_info=True)
                continue
            if not np.all(np.isfinite(matrix)):
                continue
            loaded[int(idx)] = (matrix, zero_centered)
        if loaded:
            logger.info("Loaded %d V3DT projection matrix/matrices for person cuboid overlay", len(loaded))
        return loaded

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
        return 0, 0

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

    def _frame_source_size(self, frame_meta: Any) -> Tuple[int, int]:
        try:
            frame_w = int(_meta_lookup(frame_meta, "source_frame_width", "frame_width", "width", default=0) or 0)
            frame_h = int(_meta_lookup(frame_meta, "source_frame_height", "frame_height", default=0) or 0)
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
        rect = getattr(frame_meta, "compositor_rect", None)
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

    def _configured_tile_rect(self, sensor_id: int) -> Optional[Tuple[float, float, float, float]]:
        tiler = self.pipeline.components.get("tiler")
        tiler_cfg = tiler.config if tiler is not None and isinstance(tiler.config, dict) else {}
        try:
            mosaic_w = float(tiler_cfg.get("width", 0) or 0)
            mosaic_h = float(tiler_cfg.get("height", 0) or 0)
            cols = int(tiler_cfg.get("columns", 0) or 0)
            rows = int(tiler_cfg.get("rows", 0) or 0)
        except Exception:
            return None
        if mosaic_w <= 0.0 or mosaic_h <= 0.0 or cols <= 0 or rows <= 0:
            return None
        try:
            source_count = max(1, len(getattr(self.pipeline, "camera_labels", {}) or {}))
        except Exception:
            source_count = cols * rows
        tile_index = int(sensor_id)
        if tile_index < 0 or tile_index >= int(source_count):
            return None
        col = tile_index % cols
        row = tile_index // cols
        if row >= rows:
            return None
        tile_w = mosaic_w / float(cols)
        tile_h = mosaic_h / float(rows)
        return float(col) * tile_w, float(row) * tile_h, tile_w, tile_h

    def _source_to_mosaic(self, frame_meta: Any, u: float, v: float, source_size: Tuple[int, int]) -> Tuple[float, float]:
        src_w, src_h = source_size
        comp = self._frame_compositor_rect(frame_meta)
        if comp is None:
            comp = self._configured_tile_rect(self._frame_source_id(frame_meta))
        if comp is None or src_w <= 0 or src_h <= 0:
            return float(u), float(v)
        left, top, width, height = comp
        x = float(left) + float(u) * (float(width) / float(src_w))
        y = float(top) + float(v) * (float(height) / float(src_h))
        return float(x), float(y)

    @staticmethod
    def _line_count(dm: Any) -> Optional[int]:
        try:
            return int(getattr(dm, "n_lines"))
        except Exception:
            return None

    def _add_line(self, dm: Any, line: Any) -> bool:
        before = self._line_count(dm)
        try:
            dm.add_line(line)
        except Exception:
            return False
        if before is None:
            return True
        after = self._line_count(dm)
        if after is None:
            return True
        return after > before

    def _build_line(self, p0: Tuple[float, float], p1: Tuple[float, float]) -> Any:
        line = ds_osd.Line()
        line.width = int(self.config.line_width)
        line.x1 = int(round(p0[0]))
        line.y1 = int(round(p0[1]))
        line.x2 = int(round(p1[0]))
        line.y2 = int(round(p1[1]))
        r, g, b = self.config.color
        line.color.r = float(r)
        line.color.g = float(g)
        line.color.b = float(b)
        line.color.a = float(self.config.alpha)
        return line

    @staticmethod
    def _cuboid_vertex_specs() -> Tuple[Tuple[float, float, float], ...]:
        return (
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
        )

    @staticmethod
    def _cuboid_edges() -> Tuple[Tuple[int, int], ...]:
        return (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        )

    def _extract_bbox3d(self, obj_meta: Any) -> Optional[Mapping[str, Any]]:
        if noesis_v3dt_meta_ext is None:
            return None
        try:
            result = noesis_v3dt_meta_ext.extract_obj_3d_meta(obj_meta)  # type: ignore[union-attr]
        except Exception:
            return None
        if not isinstance(result, Mapping):
            return None
        bbox3d = result.get("bbox3d")
        return bbox3d if isinstance(bbox3d, Mapping) else None

    def _project_v3dt_cuboid(
        self,
        obj_meta: Any,
        frame_meta: Any,
        source_size: Tuple[int, int],
        bbox: Tuple[float, float, float, float],
    ) -> Optional[List[Tuple[float, float]]]:
        source_id = self._frame_source_id(frame_meta)
        projection = self._v3dt_projection_by_source.get(int(source_id))
        if projection is None:
            if not self._v3dt_projection_warned:
                logger.warning("Person cuboid overlay cannot find V3DT projection for source %s", source_id)
                self._v3dt_projection_warned = True
            return None
        bbox3d = self._extract_bbox3d(obj_meta)
        if not isinstance(bbox3d, Mapping):
            return None
        try:
            center = np.asarray(
                [
                    float(bbox3d.get("xCentre")),
                    float(bbox3d.get("yCentre")),
                    float(bbox3d.get("zCentre")),
                ],
                dtype=np.float64,
            )
            half = 0.5 * np.asarray(
                [
                    float(bbox3d.get("xLen")),
                    float(bbox3d.get("yLen")),
                    float(bbox3d.get("zLen")),
                ],
                dtype=np.float64,
            )
        except Exception:
            return None
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(half)) or np.any(half <= 0.0):
            return None

        matrix, zero_centered = projection
        src_w, src_h = source_size
        projected: List[Tuple[float, float]] = []
        for sx, sy, sz in self._cuboid_vertex_specs():
            world = center + np.asarray([sx, sy, sz], dtype=np.float64) * half
            homogeneous = np.asarray([world[0], world[1], world[2], 1.0], dtype=np.float64)
            uvw = matrix @ homogeneous
            z = float(uvw[2])
            if not np.isfinite(z) or abs(z) < 1.0e-6:
                return None
            u = float(uvw[0] / z)
            v = float(uvw[1] / z)
            if zero_centered:
                u += float(src_w) * 0.5
                v += float(src_h) * 0.5
            if not np.isfinite(u) or not np.isfinite(v):
                return None
            projected.append((u, v))
        if len(projected) != 8:
            return None

        xs = np.asarray([p[0] for p in projected], dtype=np.float64)
        ys = np.asarray([p[1] for p in projected], dtype=np.float64)
        min_x, max_x = float(np.min(xs)), float(np.max(xs))
        min_y, max_y = float(np.min(ys)), float(np.max(ys))
        span_x = max_x - min_x
        span_y = max_y - min_y
        if span_x <= 1.0 or span_y <= 1.0:
            return None

        left, top, width, height = [float(v) for v in bbox]
        # Keep the V3DT cuboid's perspective/orientation, but normalize the visible
        # envelope to the mask-aligned person rect. This prevents the raw tracker
        # projection from reintroducing position-dependent scale drift.
        normalized: List[Tuple[float, float]] = []
        for u, v in projected:
            x = left + ((float(u) - min_x) / span_x) * width
            y = top + ((float(v) - min_y) / span_y) * height
            normalized.append((float(x), float(y)))
        return normalized

    def _anchored_cuboid_points(
        self,
        frame_meta: Any,
        source_size: Tuple[int, int],
        bbox: Tuple[float, float, float, float],
    ) -> List[Tuple[float, float]]:
        left, top, width, height = [float(v) for v in bbox]
        src_w, src_h = source_size
        front = [
            (left, top),
            (left + width, top),
            (left + width, top + height),
            (left, top + height),
        ]
        center_x = left + (0.5 * width)
        foot_y = top + height
        vanishing_x = float(src_w) * 0.5 if src_w > 0 else center_x
        vanishing_y = float(src_h) * 0.08 if src_h > 0 else top - height
        direction = np.asarray([vanishing_x - center_x, vanishing_y - foot_y], dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= 1.0:
            direction = np.asarray([0.25 * width, -0.12 * height], dtype=np.float64)
            norm = float(np.linalg.norm(direction))
        if norm <= 1.0:
            direction = np.asarray([1.0, -1.0], dtype=np.float64)
            norm = float(np.linalg.norm(direction))
        direction /= norm

        requested = min(
            float(width) * float(self.config.cuboid_depth_width_ratio),
            float(height) * float(self.config.cuboid_depth_height_ratio),
        )
        depth = max(float(self.config.cuboid_min_depth_px), requested)
        depth = min(float(self.config.cuboid_max_depth_px), depth)
        offset = direction * float(depth)
        back = [(float(x + offset[0]), float(y + offset[1])) for x, y in front]
        return front + back

    def _cuboid_segments(
        self,
        obj_meta: Any,
        frame_meta: Any,
        source_size: Tuple[int, int],
        bbox: Tuple[float, float, float, float],
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        left, top, width, height = [float(v) for v in bbox]
        anchor_corners = [
            (left, top),
            (left + width, top),
            (left + width, top + height),
            (left, top + height),
        ]
        segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        if self.config.draw_anchor_rect:
            segments.extend(
                [
                    (anchor_corners[0], anchor_corners[1]),
                    (anchor_corners[1], anchor_corners[2]),
                    (anchor_corners[2], anchor_corners[3]),
                    (anchor_corners[3], anchor_corners[0]),
                ]
            )
        if self.config.shape == "rectangle":
            return segments
        if self.config.require_v3dt_meta and self._extract_bbox3d(obj_meta) is None:
            return []
        if self.config.cuboid_geometry == "projected_normalized":
            points = self._project_v3dt_cuboid(obj_meta, frame_meta, source_size, bbox)
            if points is None:
                return []
        else:
            points = self._anchored_cuboid_points(frame_meta, source_size, bbox)
        for i0, i1 in self._cuboid_edges():
            segments.append((points[i0], points[i1]))
        return segments

    def _validate_mask_alignment(self, obj_meta: Any) -> Optional[Tuple[bool, float]]:
        if noesis_depth_meta_ext is None:
            return None
        try:
            payload = noesis_depth_meta_ext.extract_object_mask(obj_meta)  # type: ignore[union-attr]
        except Exception:
            return None
        if not isinstance(payload, Mapping):
            return None
        try:
            width = int(payload.get("width", 0) or 0)
            height = int(payload.get("height", 0) or 0)
            data = np.asarray(payload.get("data"), dtype=np.float32).reshape((height, width))
            threshold = float(payload.get("threshold", 0.5) or 0.5)
        except Exception:
            return None
        if width <= 2 or height <= 2 or data.size <= 0:
            return None
        mask = np.asarray(data > threshold, dtype=np.uint8)
        count = int(np.count_nonzero(mask))
        if count < 20:
            return None

        num, labels, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
        if num <= 1:
            return None
        best_idx = 1
        best_area = 0
        for idx in range(1, num):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area > best_area:
                best_area = area
                best_idx = idx
        if best_area < 20:
            return None

        w = int(stats[best_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[best_idx, cv2.CC_STAT_HEIGHT])
        if w <= 0 or h <= 0:
            return None
        area_frac = float(best_area) / float(max(1, width * height))
        span_w = float(w) / float(max(1, width))
        span_h = float(h) / float(max(1, height))

        # Same-frame metadata check: the instance mask is stored in the rect_params ROI.
        # Passing here means the visible bbox is driven by the same axis-aligned image ROI
        # that contains the segmented person, not by V3DT's projected 3D debug cuboid.
        ok = (area_frac >= 0.03) and (span_w >= 0.12) and (span_h >= 0.20)
        return bool(ok), float(area_frac)

    def handle_batch_ds8(self, batch_meta: Any) -> None:
        if ds_osd is None or not self.config.enabled:
            return
        frame_items = getattr(batch_meta, "frame_items", None)
        if frame_items is None:
            return
        debug = str(os.environ.get("NOESIS_PERSON_BBOX_OVERLAY_DEBUG", "")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        validate = str(os.environ.get("NOESIS_PERSON_BBOX_OVERLAY_VALIDATE", "")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        boxes = 0
        for frame_meta in frame_items:
            try:
                boxes += self._handle_frame(frame_meta, batch_meta, validate=validate)
            except Exception:
                logger.exception("Failed to render person bbox overlay")
        if debug:
            now = time.time()
            self._debug_frames += len(list(frame_items)) if isinstance(frame_items, list) else 1
            self._debug_boxes += int(boxes)
            if (now - float(self._debug_last_log_ts)) >= 1.0:
                logger.info(
                    "PersonBBoxOverlay: frames=%d boxes=%d line_width=%d",
                    self._debug_frames,
                    self._debug_boxes,
                    int(self.config.line_width),
                )
                self._debug_frames = 0
                self._debug_boxes = 0
                self._debug_last_log_ts = float(now)
        if validate:
            self._log_validation()

    def _log_validation(self) -> None:
        now = time.time()
        if (now - float(self._validate_last_log_ts)) < 1.0:
            return
        total = int(self._validate_objects)
        ok = int(self._validate_ok)
        if total <= 0:
            return
        coverages = list(self._validate_coverages)
        median_coverage = float(np.median(np.asarray(coverages, dtype=np.float32))) if coverages else float("nan")
        pass_rate = float(ok) / float(total)
        logger.info(
            "PersonBBoxOverlayValidation: objects=%d pass=%d pass_rate=%.3f median_mask_roi_coverage=%.3f min_coverage=0.030",
            total,
            ok,
            pass_rate,
            median_coverage,
        )
        self._validate_objects = 0
        self._validate_ok = 0
        self._validate_coverages.clear()
        self._validate_last_log_ts = float(now)

    def _handle_frame(self, frame_meta: Any, batch_meta: Any, *, validate: bool = False) -> int:
        acquire_display_meta = getattr(batch_meta, "acquire_display_meta", None)
        append_meta = getattr(frame_meta, "append", None)
        if not callable(acquire_display_meta) or not callable(append_meta):
            return 0

        object_items = getattr(frame_meta, "object_items", None) or []
        if not object_items:
            return 0

        source_size = self._frame_source_size(frame_meta)
        if source_size[0] <= 0 or source_size[1] <= 0:
            return 0

        display_metas: List[Any] = []
        appended_ids: set[int] = set()
        current: Any = None
        boxes_drawn = 0
        max_boxes = int(self.config.max_boxes)
        max_metas = int(self.config.max_display_metas)

        def _append(dm: Any) -> None:
            dm_id = id(dm)
            if dm_id in appended_ids:
                return
            try:
                append_meta(dm)
            except Exception:
                return
            appended_ids.add(dm_id)

        def _alloc() -> Optional[Any]:
            if len(display_metas) >= max_metas:
                return None
            try:
                dm = acquire_display_meta()
            except Exception:
                return None
            if not dm:
                return None
            display_metas.append(dm)
            return dm

        def _ensure() -> Optional[Any]:
            nonlocal current
            if current is None:
                current = _alloc()
            return current

        for obj_meta in object_items:
            if boxes_drawn >= max_boxes:
                break
            try:
                class_id = int(getattr(obj_meta, "class_id", -1))
            except Exception:
                class_id = -1
            if class_id not in self.config.class_ids:
                continue
            bbox = _rect_to_bbox(getattr(obj_meta, "rect_params", None))
            if bbox is None:
                continue
            left, top, width, height = [float(v) for v in bbox[:4]]
            if width <= 1.0 or height <= 1.0:
                continue
            if validate:
                validation = self._validate_mask_alignment(obj_meta)
                if validation is not None:
                    ok, coverage = validation
                    self._validate_objects += 1
                    if ok:
                        self._validate_ok += 1
                    self._validate_coverages.append(float(coverage))
            expand = float(self.config.expand_px)
            left -= expand
            top -= expand
            width += 2.0 * expand
            height += 2.0 * expand
            draw_bbox = (float(left), float(top), float(width), float(height))
            segments = self._cuboid_segments(obj_meta, frame_meta, source_size, draw_bbox)
            if not segments:
                continue

            ok_box = True
            for p0, p1 in segments:
                dm = _ensure()
                if dm is None:
                    ok_box = False
                    break
                line = self._build_line(p0, p1)
                if self._add_line(dm, line):
                    continue
                _append(dm)
                current = None
                dm = _ensure()
                if dm is None or not self._add_line(dm, line):
                    ok_box = False
                    break
            if not ok_box:
                break
            boxes_drawn += 1

        for dm in display_metas:
            _append(dm)
        return int(boxes_drawn)


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
    camera_labels: Mapping[int, str] = field(default_factory=dict)
    _missing_native_logged: bool = field(default=False, init=False, repr=False)
    _debug_last_log: float = field(default=0.0, init=False, repr=False)
    _debug_frames: int = field(default=0, init=False, repr=False)
    _debug_objects: int = field(default=0, init=False, repr=False)
    _debug_attached: int = field(default=0, init=False, repr=False)
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
        _increment_core_counter("tensor_host_copies_total.pose")
        return score, arr_roi, arr_abs

    def handle_frame_ds8(self, frame_meta: Any) -> None:
        object_items = getattr(frame_meta, "object_items", None) or []
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
            if not self._missing_native_logged:
                logger.warning(
                    "Pose meta attach skipped; noesis_pose_meta_ext is unavailable or missing attach_pose_features (build scripts/build_noesis_pose_meta_ext.sh)"
                )
                self._missing_native_logged = True
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
            roi_w = float(bbox[2])
            roi_h = float(bbox[3])
            score = 0.0
            kpts_abs: Optional[np.ndarray] = None
            kpts_for_features: Optional[np.ndarray] = None
            kpts_roi: Optional[np.ndarray] = None
            native = self._extract_pose_native(obj_meta)
            if native is not None:
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
                model="yolo26-pose",
                ts_us=int(ts_us),
            ).to_dict()
            if attach_obj is None:
                if debug:
                    self._debug_missing += 1
            else:
                try:
                    payload_json = _serialize_compact_json_with_metrics(
                        payload,
                        metric="pose_features.user_meta_json",
                    )
                    payload_bytes = len(payload_json.encode("utf-8"))
                    if payload_bytes > _pose_meta_payload_limit_bytes():
                        if debug:
                            self._debug_missing += 1
                        logger.debug(
                            "Pose meta attach skipped: payload exceeds limit bytes=%d",
                            int(payload_bytes),
                        )
                        continue
                    _increment_core_counter(
                        "tensor_boundary_copy_bytes_total.pose_meta",
                        payload_bytes,
                    )
                    ok = bool(
                        attach_obj(
                            obj_meta,
                            payload_json,
                            True,
                        )
                    )
                except Exception:
                    ok = False
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
        frame = _AlignedDepthFrame(
            key=_depth_frame_key(frame_meta),
            source_id=int(_meta_lookup(frame_meta, "source_id", "pad_index", default=0) or 0),
            frame_id=int(_meta_lookup(frame_meta, "frame_number", "frame_num", default=0) or 0),
            pts_us=_frame_pts_key_us(frame_meta),
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

    def _camera_id_for_source(self, source_id: int) -> Optional[str]:
        camera_id = self.camera_labels.get(int(source_id))
        if isinstance(camera_id, str) and camera_id.strip():
            return camera_id.strip()
        return None

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
            return np.asarray(copy_roi(int(x0), int(y0), int(width), int(height)), dtype=np.float32)
        except Exception:
            logger.exception("GPU depth ROI copy failed")
            return None

    def _decode_instance_mask(self, obj_meta: Any, target_shape: Tuple[int, int]) -> Tuple[Optional[np.ndarray], str]:
        try:
            payload = noesis_depth_meta_ext.extract_object_mask(obj_meta)  # type: ignore[union-attr]
        except Exception:
            logger.debug("Native object-mask extraction failed", exc_info=True)
            return None, "mask_decode_failed"
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
        payload: Dict[str, Any] = {
            "source_id": int(_meta_lookup(frame_meta, "source_id", "pad_index", default=0) or 0),
            "frame_id": int(_meta_lookup(frame_meta, "frame_number", "frame_num", default=0) or 0),
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
            "depth_p10": float(np.percentile(values_arr, 10.0)) if has_values else None,
            "depth_p90": float(np.percentile(values_arr, 90.0)) if has_values else None,
            "depth_min": float(np.min(values_arr)) if has_values else None,
            "depth_max": float(np.max(values_arr)) if has_values else None,
            "mask_area_px": max(0, int(mask_area_px)),
            "model": self.depth_model_name,
            "ts_us": _frame_pts_key_us(frame_meta),
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
        synthetic_mask = np.ones(depth_crop.shape, dtype=bool)
        anchor = _extract_person_depth_anchor(
            synthetic_mask,
            depth_crop,
            frame_origin=(int(crop_origin[0]), int(crop_origin[1])),
        )
        anchor_fields: Dict[str, Any] = {
            "spatial_class": "person",
            "anchor_uv": list(anchor.foot_uv) if anchor.foot_uv is not None else None,
            "anchor_source": anchor.anchor_source,
            "anchor_depth_m": anchor.anchor_depth_m,
            "anchor_sample_count": int(anchor.anchor_sample_count) if anchor.anchor_sample_count > 0 else None,
            "anchor_valid_fraction": float(anchor.anchor_valid_fraction) if anchor.anchor_valid_fraction > 0.0 else None,
        }
        values = np.asarray(depth_crop[np.isfinite(depth_crop)], dtype=np.float32)
        sample_count = int(values.size)
        status = "ok" if sample_count > 0 and anchor.anchor_depth_m is not None else "no_valid_depth"
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
            sampling_mode="bbox_band",
        )

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

        depth_crop = self._copy_depth_crop(depth_frame, x0, y0, x1, y1)
        if depth_crop is None:
            return self._build_result(frame_meta, obj_meta, bbox=bbox, status="depth_not_ready")
        if depth_crop.size <= 0:
            return self._build_result(frame_meta, obj_meta, bbox=bbox, status="transform_mismatch")

        cx = max(0, min(frame_w - 1, int(round(left + (width * 0.5)))))
        cy = max(0, min(frame_h - 1, int(round(top + (height * 0.5)))))
        local_cx = max(0, min(int(depth_crop.shape[1]) - 1, cx - x0))
        local_cy = max(0, min(int(depth_crop.shape[0]) - 1, cy - y0))
        center_sample = float(depth_crop[local_cy, local_cx])
        center_value = center_sample if np.isfinite(center_sample) else None

        mask, _mask_status = self._decode_instance_mask(obj_meta, depth_crop.shape)
        if mask is None:
            return self._sample_bbox_band_result(
                frame_meta,
                obj_meta,
                bbox=bbox,
                depth_crop=depth_crop,
                crop_origin=(x0, y0),
                depth_center=center_value,
            )

        mask_area = int(np.count_nonzero(mask))

        if mask_area <= 0:
            return self._sample_bbox_band_result(
                frame_meta,
                obj_meta,
                bbox=bbox,
                depth_crop=depth_crop,
                crop_origin=(x0, y0),
                depth_center=center_value,
            )

        valid_mask = np.logical_and(mask, np.isfinite(depth_crop))
        values = np.asarray(depth_crop[valid_mask], dtype=np.float32)
        anchor = _extract_person_depth_anchor(
            mask,
            depth_crop,
            frame_origin=(int(math.floor(left)), int(math.floor(top))),
        )
        anchor_fields: Dict[str, Any] = {
            "spatial_class": "person",
            "anchor_uv": list(anchor.foot_uv) if anchor.foot_uv is not None else None,
            "anchor_source": anchor.anchor_source,
            "anchor_depth_m": anchor.anchor_depth_m,
            "anchor_sample_count": int(anchor.anchor_sample_count) if anchor.anchor_sample_count > 0 else None,
            "anchor_valid_fraction": float(anchor.anchor_valid_fraction) if anchor.anchor_valid_fraction > 0.0 else None,
        }
        sample_count = int(values.size)
        if sample_count <= 0:
            return self._build_result(
                frame_meta,
                obj_meta,
                bbox=bbox,
                status="no_valid_depth",
                mask_area_px=mask_area,
                depth_center=center_value,
                anchor_fields=anchor_fields,
            )
        return self._build_result(
            frame_meta,
            obj_meta,
            bbox=bbox,
            status="ok",
            mask_area_px=mask_area,
            sample_count=sample_count,
            valid_fraction=float(sample_count) / float(mask_area),
            depth_center=center_value,
            values=values,
            anchor_fields=anchor_fields,
            sampling_mode="instance_mask",
        )

    def handle_frame_ds8(self, frame_meta: Any) -> None:
        source_id = int(_meta_lookup(frame_meta, "source_id", "pad_index", default=0) or 0)
        frame_id = int(_meta_lookup(frame_meta, "frame_number", "frame_num", default=0) or 0)
        pts_us = _frame_pts_key_us(frame_meta)
        depth_frame, _age_frames, _age_ms = self.depth_store.resolve(
            source_id=source_id,
            frame_id=frame_id,
            pts_us=pts_us,
            max_age_frames=max(0, int(self.depth_every_n_frames) - 1),
        )
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
            if depth_frame is None:
                result = self._build_result(frame_meta, obj_meta, bbox=bbox, status="depth_not_ready")
            else:
                result = self._sample_person_result(frame_meta, obj_meta, depth_frame)
            if result is None:
                continue
            try:
                noesis_depth_meta_ext.attach_object_depth(obj_meta, result.to_json(), True)  # type: ignore[union-attr]
            except Exception:
                logger.exception("Failed to attach NOESIS.OBJECT_DEPTH to object metadata")


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
                self.processor.handle_frame_ds8(frame_meta)
            except Exception:
                logger.exception("Failed to fuse object depth within batch metadata (DS8)")


@dataclass
class _AnalyticsTelemetryProcessor:
    pipeline: "DS8Pipeline"
    tracking_pub: "TrackingTelemetryPublisher"
    camera_labels: Mapping[int, str]
    sensor_id_map: Mapping[int, int]
    tracking_mode: Optional[str] = None
    bev_renderer: Any = None
    bev_calibration: Any = None
    depth_registration: DepthRegistrationManager | None = None
    world_fusion_policy: WorldFusionPolicy | None = None
    diagnostics_logger: Any = None
    osd_label_processor: Any = None
    _analytics_obj_meta_type: Any = field(default=None, init=False, repr=False)
    _zone_state: Dict[int, Dict[int, Dict[str, Any]]] = field(default_factory=dict, init=False, repr=False)
    _occupancy_state: Dict[int, Dict[str, int]] = field(default_factory=dict, init=False, repr=False)
    _occupancy_last_seen: Dict[int, Dict[str, float]] = field(default_factory=dict, init=False, repr=False)
    _occupancy_grace_s: float = field(default=0.0, init=False, repr=False)
    _active_tracks: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict, init=False, repr=False)
    _transitions_state: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict, init=False, repr=False)
    _stable_id_enabled: bool = field(default=True, init=False, repr=False)
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
    _v3dt_caminfo_world_axes: str = field(default="xyz", init=False, repr=False)
    _v3dt_axis_map: V3DTAxisMap | None = field(default=None, init=False, repr=False)
    _sid_metrics_log_enabled: bool = field(default=True, init=False, repr=False)
    _sid_metrics_log_interval_s: float = field(default=10.0, init=False, repr=False)
    _sid_metrics_last_log_by_sensor: Dict[int, float] = field(default_factory=dict, init=False, repr=False)
    _stable_id_public_max_id: int = field(default=0, init=False, repr=False)
    _stable_id_public_reuse_window_s: float = field(default=8.0, init=False, repr=False)
    _stable_id_public_same_camera_sticky_window_s: float = field(default=0.0, init=False, repr=False)
    _stable_id_public_cross_camera_sticky_window_s: float = field(default=0.0, init=False, repr=False)
    _stable_id_public_single_person_mode: bool = field(default=False, init=False, repr=False)
    _stable_id_public_primary_min_observations: int = field(default=5, init=False, repr=False)
    _stable_id_public_support: Dict[int, int] = field(default_factory=dict, init=False, repr=False)
    _stable_id_public_primary_sid: Optional[int] = field(default=None, init=False, repr=False)
    _stable_id_public_primary_last_ts: float = field(default=0.0, init=False, repr=False)
    _stable_id_public_alias: Dict[int, int] = field(default_factory=dict, init=False, repr=False)
    _stable_id_public_last_seen: Dict[int, float] = field(default_factory=dict, init=False, repr=False)
    _stable_id_public_last_by_sensor: Dict[int, Tuple[int, float]] = field(default_factory=dict, init=False, repr=False)
    _stable_id_public_last_global: Optional[Tuple[int, float]] = field(default=None, init=False, repr=False)
    _world_state_by_track: Dict[Tuple[int, int], _WorldAnchorState] = field(default_factory=dict, init=False, repr=False)
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
    _reid_embeds_per_frame_max: int = field(default=2, init=False, repr=False)
    _stable_id_world_cache: Dict[Tuple[int, int], Tuple[float, float, bool, float]] = field(
        default_factory=dict, init=False, repr=False
    )
    _tracking_publish_interval_s: float = field(default=0.0, init=False, repr=False)
    _tracking_empty_publish_interval_s: float = field(default=0.5, init=False, repr=False)
    _bev_publish_interval_s: float = field(default=0.0, init=False, repr=False)
    _last_tracking_publish_ts_by_sensor: Dict[int, float] = field(default_factory=dict, init=False, repr=False)
    _last_bev_publish_ts_by_sensor: Dict[int, float] = field(default_factory=dict, init=False, repr=False)
    _last_tracking_count_by_sensor: Dict[int, int] = field(default_factory=dict, init=False, repr=False)
    _last_bev_count_by_sensor: Dict[int, int] = field(default_factory=dict, init=False, repr=False)
    _tracking_lifecycle: TrackingLifecycleRegistry = field(
        default_factory=TrackingLifecycleRegistry, init=False, repr=False
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

        self._tracking_mode = self._resolve_tracking_mode(self.tracking_mode)

        v3dt_cfg = getattr(self.pipeline, "config", {}).get("v3dt", {}) or {}
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
            self._v3dt_caminfo_world_axes = self._v3dt_axis_map.spec
        elif isinstance(v3dt_cfg, Mapping) and v3dt_cfg.get("world_frame"):
            self._world_frame = str(v3dt_cfg["world_frame"])

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
        if self._tracking_mode_is_v3dt():
            try:
                stable_id_cfg = getattr(self.pipeline, "config", {}).get("stable_id", {}) or {}
            except Exception:
                stable_id_cfg = {}
            if isinstance(stable_id_cfg, Mapping):
                try:
                    raw_max_id = stable_id_cfg.get("public_max_id", stable_id_cfg.get("max_total_ids", 0))
                    self._stable_id_public_max_id = max(0, int(raw_max_id or 0))
                except Exception:
                    self._stable_id_public_max_id = 0
                try:
                    from reid.household_state import is_household_identity_enabled  # type: ignore

                    if is_household_identity_enabled():
                        self._stable_id_public_max_id = 0
                except Exception:
                    pass
                try:
                    raw_window = stable_id_cfg.get("public_reuse_window_s", 8.0)
                    self._stable_id_public_reuse_window_s = max(0.5, float(raw_window or 8.0))
                except Exception:
                    self._stable_id_public_reuse_window_s = 8.0
                try:
                    raw_sticky = stable_id_cfg.get("public_same_camera_sticky_window_s", 0.0)
                    self._stable_id_public_same_camera_sticky_window_s = max(0.0, float(raw_sticky or 0.0))
                except Exception:
                    self._stable_id_public_same_camera_sticky_window_s = 0.0
                try:
                    raw_xcam = stable_id_cfg.get("public_cross_camera_sticky_window_s", 0.0)
                    self._stable_id_public_cross_camera_sticky_window_s = max(0.0, float(raw_xcam or 0.0))
                except Exception:
                    self._stable_id_public_cross_camera_sticky_window_s = 0.0
                try:
                    raw_single = stable_id_cfg.get("public_single_person_mode", False)
                    if isinstance(raw_single, str):
                        self._stable_id_public_single_person_mode = raw_single.strip().lower() in (
                            "1",
                            "true",
                            "yes",
                            "on",
                            "y",
                        )
                    else:
                        self._stable_id_public_single_person_mode = bool(raw_single)
                except Exception:
                    self._stable_id_public_single_person_mode = False
                try:
                    raw_min_obs = stable_id_cfg.get("public_primary_min_observations", 5)
                    self._stable_id_public_primary_min_observations = max(1, int(raw_min_obs or 5))
                except Exception:
                    self._stable_id_public_primary_min_observations = 5
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
        self._human_ground_cfg = HumanGroundConfig(
            static_px_threshold=float(self._world_static_px_threshold),
            max_speed_mps=float(self._world_max_speed_scene_per_s),
            max_jump_m=0.75,
            alpha_good=float(self._world_smooth_alpha_good),
            alpha_weak=float(self._world_smooth_alpha_weak),
            kpt_conf_threshold=float(self._pose_anchor_kpt_threshold),
        )
        try:
            raw_emb_max = os.environ.get("NOESIS_REID_EMBEDS_PER_FRAME_MAX", "2")
            self._reid_embeds_per_frame_max = max(0, int(str(raw_emb_max).strip() or "2"))
        except Exception:
            self._reid_embeds_per_frame_max = 2

        try:
            tracking_max_hz = max(
                0.0,
                float(
                    str(
                        os.environ.get(
                            "NOESIS_TRACKING_PUBLISH_MAX_HZ",
                            os.environ.get("NOESIS_WS_TRACKING_MAX_HZ", "15"),
                        )
                    ).strip()
                    or "15"
                ),
            )
        except Exception:
            tracking_max_hz = 15.0
        try:
            bev_max_hz = max(
                0.0,
                float(
                    str(
                        os.environ.get(
                            "NOESIS_BEV_PUBLISH_MAX_HZ",
                            os.environ.get("NOESIS_WS_BEV_MAX_HZ", "12"),
                        )
                    ).strip()
                    or "12"
                ),
            )
        except Exception:
            bev_max_hz = 12.0
        try:
            empty_tracking_hz = max(
                0.0,
                float(
                    str(
                        os.environ.get(
                            "NOESIS_TRACKING_EMPTY_HEARTBEAT_HZ",
                            "2",
                        )
                    ).strip()
                    or "2"
                ),
            )
        except Exception:
            empty_tracking_hz = 2.0
        self._tracking_publish_interval_s = (
            0.0 if tracking_max_hz <= 0.0 else 1.0 / tracking_max_hz
        )
        self._tracking_empty_publish_interval_s = (
            0.0 if empty_tracking_hz <= 0.0 else 1.0 / empty_tracking_hz
        )
        self._bev_publish_interval_s = (
            0.0 if bev_max_hz <= 0.0 else 1.0 / bev_max_hz
        )

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

    def _tracking_publish_due(
        self,
        *,
        sensor_id: int,
        now_ts: float,
        count: int,
        force: bool = False,
    ) -> bool:
        sid = int(sensor_id)
        current_count = int(count)
        previous_count = self._last_tracking_count_by_sensor.get(sid)
        last_ts = float(self._last_tracking_publish_ts_by_sensor.get(sid, 0.0))
        interval_s = self._tracking_interval_for_frame(current_count)
        due = bool(
            force
            or previous_count is None
            or int(previous_count) != current_count
            or interval_s <= 0.0
            or last_ts <= 0.0
            or (float(now_ts) - last_ts) >= interval_s
        )
        return due

    def _bev_publish_due(
        self,
        *,
        sensor_id: int,
        now_ts: float,
        count: int,
    ) -> bool:
        sid = int(sensor_id)
        current_count = int(count)
        previous_count = self._last_bev_count_by_sensor.get(sid)
        last_ts = float(self._last_bev_publish_ts_by_sensor.get(sid, 0.0))
        due = bool(
            previous_count is None
            or int(previous_count) != current_count
            or self._bev_publish_interval_s <= 0.0
            or last_ts <= 0.0
            or (float(now_ts) - last_ts) >= self._bev_publish_interval_s
        )
        if due:
            self._last_bev_publish_ts_by_sensor[sid] = float(now_ts)
            self._last_bev_count_by_sensor[sid] = current_count
        return due

    def get_active_track_map(self, sensor_id: int) -> Dict[int, Dict[str, Any]]:
        tracks = self._active_tracks.get(int(sensor_id)) or []
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
        return str(self._tracking_mode or "").strip().lower() == "v3dt"

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
                raw_mode = v3dt_cfg.get("tracking_mode") or v3dt_cfg.get("mode")
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
        if mode in ("v3dt", "sv3dt", "mv3dt", "3d"):
            return "v3dt"
        if mode in ("2d", "baseline", "standard", "default"):
            return "baseline"
        if not mode or mode == "auto":
            return "baseline"
        logger.warning("Unknown tracking_mode '%s'; defaulting to baseline", value)
        return "baseline"

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
        if using_v3dt_tracker and not self._tracking_mode_is_v3dt():
            logger.warning(
                "Tracking mode '%s' with V3DT tracker config %s; V3DT meta/world will be ignored",
                self._tracking_mode,
                cfg_path,
            )
        if self._tracking_mode_is_v3dt() and not using_v3dt_tracker:
            logger.warning(
                "Tracking mode 'v3dt' without V3DT tracker config (%s); V3DT meta may be absent",
                cfg_path,
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
        # In Noesis' xzy camInfo lane, the floor-plane footpoint is
        # zCentre - 0.5*zLen in tracker space, but NVIDIA's
        # NVDS_OBJ_IMAGE_FOOT_LOCATION lines up with that point rather than the
        # visible bottom of the 2D detector box. For image anchoring, project the
        # opposite vertical endpoint; live diagnostics show that this lands on
        # the detector's bottom edge and gives the frontend a usable base pixel.
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

    def _world_from_v3dt_image_base(
        self,
        sensor_id: int,
        camera_id: str,
        image_base: Sequence[float],
    ) -> Optional[List[float]]:
        if not isinstance(image_base, (list, tuple)) or len(image_base) < 2:
            return None
        try:
            u = float(image_base[0])
            v = float(image_base[1])
        except Exception:
            return None
        if not (math.isfinite(u) and math.isfinite(v)):
            return None

        provider = getattr(self, "bev_calibration", None)
        snapshot_fn = getattr(provider, "snapshot", None)
        if not callable(snapshot_fn):
            return None
        try:
            calib = snapshot_fn(int(sensor_id), str(camera_id))
        except Exception:
            calib = None
        if calib is None:
            return None

        try:
            result = pixel_to_world(
                calib.intrinsics,
                calib.extrinsics_col_major,
                float(calib.floor_y),
                float(calib.unit_scale or 1.0),
                u,
                v,
                depth_m=None,
            )
        except Exception:
            return None
        if not result.ok or not isinstance(result.world_point, list) or len(result.world_point) < 3:
            return None
        try:
            world = [float(result.world_point[0]), float(result.world_point[1]), float(result.world_point[2])]
        except Exception:
            return None
        if not all(math.isfinite(value) for value in world):
            return None
        return world

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

    def _v3dt_tracker_point_to_world(self, point: Sequence[float]) -> Optional[List[float]]:
        if self._v3dt_axis_map is None:
            raise V3DTAxisMapError("V3DT axis map was not initialized")
        try:
            world = self._v3dt_axis_map.tracker_to_world(point)
        except V3DTAxisMapError:
            return None
        return [float(world[0]), float(world[1]), float(world[2])]

    def _world_from_bbox3d(self, bbox3d: Mapping[str, Any]) -> Optional[List[float]]:
        if self._v3dt_axis_map is None:
            raise V3DTAxisMapError("V3DT axis map was not initialized")
        try:
            world = v3dt_bbox3d_world_foot(bbox3d, self._v3dt_axis_map)
        except V3DTAxisMapError:
            return None
        return [float(world[0]), float(world[1]), float(world[2])]

    def _v3dt_world_quality_reason(
        self,
        bbox3d: Mapping[str, Any],
        *,
        method: str = "projected_base_ray_floor",
    ) -> str:
        try:
            height = float(bbox3d.get("zLen"))
        except Exception:
            height = float("nan")
        parts = [
            "derived_from=bbox3d",
            f"point={method}",
            "tracker_vertical_axis=z",
            f"axis_map={self._v3dt_caminfo_world_axes}",
        ]
        if math.isfinite(height):
            parts.append(f"height_m={height:.3f}")
        return ",".join(parts)

    @staticmethod
    def _is_v3dt_world_source(source: Any) -> bool:
        return str(source or "").strip().lower() == V3DT_WORLD_SOURCE_BBOX3D_FOOT

    def _public_stable_id(
        self,
        stable_id: int,
        *,
        sensor_id: Optional[int] = None,
        sid_candidate: Any = None,
        ts: Optional[float] = None,
        claimed_public_ids: Optional[set[int]] = None,
    ) -> int:
        """Keep V3DT public IDs inside the configured small home identity range."""
        try:
            sid_int = int(stable_id)
        except Exception:
            return int(stable_id) if isinstance(stable_id, int) else 0
        max_id = int(self._stable_id_public_max_id or 0)
        if max_id <= 0 or sid_int <= 0:
            return sid_int
        now_ts = float(ts if ts is not None else time.time())
        claimed = {int(v) for v in (claimed_public_ids or set()) if int(v) > 0}

        def remember(public_sid: int) -> int:
            public_sid = max(1, min(max_id, int(public_sid)))
            self._stable_id_public_support[public_sid] = int(self._stable_id_public_support.get(public_sid, 0)) + 1
            self._stable_id_public_last_seen[public_sid] = now_ts
            if sensor_id is not None:
                try:
                    self._stable_id_public_last_by_sensor[int(sensor_id)] = (public_sid, now_ts)
                except Exception:
                    pass
            self._stable_id_public_last_global = (public_sid, now_ts)
            if self._stable_id_public_single_person_mode:
                primary = self._stable_id_public_primary_sid
                min_obs = int(max(1, self._stable_id_public_primary_min_observations))
                if primary is None:
                    best_sid = None
                    best_count = 0
                    for sid_key, count in self._stable_id_public_support.items():
                        if int(count) > best_count:
                            best_sid = int(sid_key)
                            best_count = int(count)
                    if best_sid is not None and best_count >= min_obs:
                        self._stable_id_public_primary_sid = int(best_sid)
                        self._stable_id_public_primary_last_ts = now_ts
                elif int(public_sid) == int(primary):
                    self._stable_id_public_primary_last_ts = now_ts
                else:
                    primary_count = int(self._stable_id_public_support.get(int(primary), 0))
                    public_count = int(self._stable_id_public_support.get(int(public_sid), 0))
                    primary_stale = (
                        self._stable_id_public_cross_camera_sticky_window_s > 0.0
                        and (now_ts - float(self._stable_id_public_primary_last_ts))
                        > float(self._stable_id_public_cross_camera_sticky_window_s)
                    )
                    if primary_stale and public_count >= max(min_obs, primary_count + min_obs):
                        self._stable_id_public_primary_sid = int(public_sid)
                        self._stable_id_public_primary_last_ts = now_ts
            return public_sid

        def single_person_primary_target() -> Optional[int]:
            if not self._stable_id_public_single_person_mode:
                return None
            primary = self._stable_id_public_primary_sid
            if primary is None:
                return None
            try:
                primary_int = int(primary)
            except Exception:
                return None
            if primary_int <= 0 or primary_int > max_id or primary_int in claimed:
                return None
            window = float(self._stable_id_public_cross_camera_sticky_window_s or 0.0)
            if window > 0.0 and (now_ts - float(self._stable_id_public_primary_last_ts)) > window:
                return None
            return primary_int

        def recent_sticky_target() -> Optional[int]:
            primary_target = single_person_primary_target()
            if primary_target is not None:
                return int(primary_target)
            if sensor_id is not None and self._stable_id_public_same_camera_sticky_window_s > 0.0:
                try:
                    last_sid, last_ts = self._stable_id_public_last_by_sensor.get(int(sensor_id), (0, 0.0))
                    if (
                        int(last_sid) > 0
                        and int(last_sid) <= max_id
                        and int(last_sid) not in claimed
                        and (now_ts - float(last_ts)) <= float(self._stable_id_public_same_camera_sticky_window_s)
                    ):
                        return int(last_sid)
                except Exception:
                    pass
            if self._stable_id_public_cross_camera_sticky_window_s > 0.0:
                try:
                    last_global = self._stable_id_public_last_global
                    if last_global is not None:
                        last_sid, last_ts = last_global
                        if (
                            int(last_sid) > 0
                            and int(last_sid) <= max_id
                            and int(last_sid) not in claimed
                            and (now_ts - float(last_ts)) <= float(self._stable_id_public_cross_camera_sticky_window_s)
                        ):
                            return int(last_sid)
                except Exception:
                    pass
            return None

        sticky_target = recent_sticky_target()
        if sid_int <= max_id:
            self._stable_id_public_alias.setdefault(sid_int, sid_int)
            if sticky_target is not None and sticky_target != sid_int:
                logger.info(
                    "V3DT public SID continuity remapped internal sid %s to recent public sid %s",
                    sid_int,
                    sticky_target,
                )
                return remember(sticky_target)
            return remember(sid_int)

        alias = self._stable_id_public_alias.get(sid_int)
        if alias is not None and 0 < int(alias) <= max_id:
            alias_int = int(alias)
            if sticky_target is not None and sticky_target != alias_int:
                alias_int = int(sticky_target)
                self._stable_id_public_alias[sid_int] = alias_int
            return remember(alias_int)

        target: Optional[int] = None
        try:
            cand_int = int(sid_candidate)
        except Exception:
            cand_int = 0
        if 0 < cand_int <= max_id:
            target = int(cand_int)
        if target is None:
            window_s = float(self._stable_id_public_reuse_window_s)
            recent = [
                (float(last_ts), int(public_sid))
                for public_sid, last_ts in self._stable_id_public_last_seen.items()
                if 0 < int(public_sid) <= max_id and (now_ts - float(last_ts)) <= window_s
            ]
            if recent:
                recent.sort(reverse=True)
                target = int(recent[0][1])
        if target is None:
            target = ((sid_int - 1) % max_id) + 1

        target = max(1, min(max_id, int(target)))
        if sticky_target is not None and sticky_target != target:
            target = int(sticky_target)
        self._stable_id_public_alias[sid_int] = target
        logger.info(
            "V3DT public SID cap remapped internal sid %s to public sid %s (max=%s, candidate=%s)",
            sid_int,
            target,
            max_id,
            sid_candidate,
        )
        return remember(target)

    @staticmethod
    def _track_quality_score(track: Mapping[str, Any]) -> float:
        """Rank split detections for the same public V3DT person track."""
        bbox = track.get("bbox")
        width = 0.0
        height = 0.0
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            try:
                width = max(0.0, float(bbox[2]))
                height = max(0.0, float(bbox[3]))
            except Exception:
                width = 0.0
                height = 0.0
        area = width * height

        def number(key: str) -> float:
            try:
                value = track.get(key)
                return float(value) if value is not None else 0.0
            except Exception:
                return 0.0

        confidence = max(0.0, number("confidence"))
        tracker_confidence = max(0.0, number("tracker_confidence"))
        embedding_bonus = 1.0 if track.get("embedding_present") else 0.0
        world_bonus = 1.0 if track.get("world_valid") is True else 0.0
        return (
            confidence * 1000.0
            + tracker_confidence * 100.0
            + min(area / 1000.0, 200.0)
            + min(height / 10.0, 100.0)
            + embedding_bonus * 25.0
            + world_bonus * 10.0
        )

    def _canonicalize_single_person_public_tracks(self, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        primary_sid = self._stable_id_public_primary_sid
        try:
            primary_sid_int = int(primary_sid) if primary_sid is not None else 0
        except Exception:
            primary_sid_int = 0
        if primary_sid_int <= 0:
            return tracks
        for track in tracks:
            if not isinstance(track, dict):
                continue
            try:
                sid = int(track.get("stable_id"))
            except Exception:
                sid = 0
            if sid <= 0 or sid == primary_sid_int:
                continue
            track["stable_id"] = int(primary_sid_int)
            id_display = track.get("id_display")
            if isinstance(id_display, str) and "|" in id_display:
                track["id_display"] = re.sub(r"\[\s*\d+\s*\]\s*$", f"[{primary_sid_int}]", id_display)
        return tracks

    def _dedupe_public_tracks(self, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Emit one canonical public track per camera/stable_id for V3DT."""
        if not self._tracking_mode_is_v3dt():
            return tracks

        if self._stable_id_public_single_person_mode and len(tracks) < 2:
            return self._canonicalize_single_person_public_tracks(tracks)

        if len(tracks) < 2:
            return tracks

        if self._stable_id_public_single_person_mode:
            best_by_camera: Dict[str, Dict[str, Any]] = {}
            camera_order: List[str] = []
            passthrough: List[Dict[str, Any]] = []
            for track in tracks:
                if not isinstance(track, dict):
                    passthrough.append(track)
                    continue
                camera_id = str(track.get("camera_id") or "")
                if not camera_id:
                    passthrough.append(track)
                    continue
                current = best_by_camera.get(camera_id)
                if current is None:
                    best_by_camera[camera_id] = track
                    camera_order.append(camera_id)
                    continue
                if self._track_quality_score(track) > self._track_quality_score(current):
                    best_by_camera[camera_id] = track
            deduped = passthrough + [best_by_camera[camera_id] for camera_id in camera_order]
            if len(deduped) != len(tracks):
                logger.debug(
                    "V3DT single-person public track dedupe reduced %d tracks to %d",
                    len(tracks),
                    len(deduped),
                )
            return self._canonicalize_single_person_public_tracks(deduped)

        best_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
        key_order: List[Tuple[str, int]] = []
        passthrough: List[Dict[str, Any]] = []
        for track in tracks:
            if not isinstance(track, dict):
                passthrough.append(track)
                continue
            try:
                sid = int(track.get("stable_id"))
            except Exception:
                sid = 0
            camera_id = str(track.get("camera_id") or "")
            if sid <= 0 or not camera_id:
                passthrough.append(track)
                continue

            key = (camera_id, sid)
            current = best_by_key.get(key)
            if current is None:
                best_by_key[key] = track
                key_order.append(key)
                continue
            if self._track_quality_score(track) > self._track_quality_score(current):
                best_by_key[key] = track

        deduped = passthrough + [best_by_key[key] for key in key_order]
        if len(deduped) != len(tracks):
            logger.debug(
                "V3DT public track dedupe reduced %d tracks to %d",
                len(tracks),
                len(deduped),
            )
        return deduped

    @staticmethod
    def _stable_ids_from_tracks(tracks: List[Mapping[str, Any]]) -> set[int]:
        stable_ids: set[int] = set()
        for track in tracks:
            if not isinstance(track, Mapping):
                continue
            try:
                sid = int(track.get("stable_id"))
            except Exception:
                continue
            if sid > 0:
                stable_ids.add(sid)
        return stable_ids

    def _footpoints_from_tracks(
        self,
        tracks: List[Mapping[str, Any]],
        frame_dims: Tuple[int, int],
        *,
        target_image_size: Optional[Tuple[int, int]] = None,
    ) -> List[Footpoint]:
        footpoints: List[Footpoint] = []
        for track in tracks:
            fp = self._footpoint_from_track(
                track,
                frame_dims,
                target_image_size=target_image_size,
            )
            if fp is not None:
                footpoints.append(fp)
        return footpoints

    @staticmethod
    def _occupancy_from_tracks(tracks: List[Mapping[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for track in tracks:
            if not isinstance(track, Mapping):
                continue
            zone = track.get("zone")
            if zone:
                zone_key = str(zone)
                counts[zone_key] = counts.get(zone_key, 0) + 1
        return counts

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
        """Extract tracking telemetry for a single frame using DS8 pyservicemaker API."""
        try:
            source_id = self._frame_source_id(frame_meta)
            sensor_id = self.sensor_id_map.get(source_id, source_id)
            camera_id = self.camera_labels.get(sensor_id, f"camera_{sensor_id}")
            now_ts = time.time()
            public_frame_timing = _public_frame_timing(frame_meta, now_ts)
            identity_v2_service = getattr(self.pipeline, "identity_v2_service", None)
            identity_v2_authoritative = bool(
                identity_v2_service is not None
                and getattr(identity_v2_service, "authoritative", False)
            )
            self._log_diag_session_start()
            reid_debug = str(os.environ.get("NOESIS_REID_DEBUG", "")).strip().lower() in ("1", "true", "yes", "on")
            if reid_debug:
                self._reid_debug_frames += 1
            reid_budget_remaining = int(self._reid_embeds_per_frame_max)

            tracks: List[Dict[str, Any]] = []
            diagnostics_tracks: List[Dict[str, Any]] = []
            frame_id = int(getattr(frame_meta, "frame_number", -1))
            occupancy_counts: Dict[str, int] = {}
            present_track_ids: set[int] = set()
            present_stable_ids: set[int] = set()
            footpoints: List[Footpoint] = []
            identity_v2_primitives: List[IdentityFramePrimitive] = []
            frame_dims = self._track_image_size(sensor_id, frame_meta)

            # Service Maker object_items is a one-shot iterator of transient
            # ObjectMetadata views. Do NOT list()/store wrappers for a second
            # pass — prior items become dangling and class_id access segfaults.
            object_items = getattr(frame_meta, "object_items", None) or []
            for obj_meta in object_items:
                raw = self._build_track_dict_ds8(obj_meta, camera_id)
                if raw is None:
                    continue
                if self._tracking_mode_is_v3dt():
                    bbox3d = raw.get("bbox3d")
                    if isinstance(bbox3d, dict):
                        image_base = self._image_base_from_bbox3d(int(source_id), bbox3d, frame_dims)
                        if image_base is not None:
                            raw["image_base"] = image_base
                            world = self._world_from_v3dt_image_base(int(sensor_id), camera_id, image_base)
                            if world is not None:
                                raw["world"] = world
                                raw["world_valid"] = True
                                raw["world_frame"] = self._world_frame
                                raw["world_source"] = V3DT_WORLD_SOURCE_BBOX3D_FOOT
                                raw["world_quality"] = "good"
                                raw["world_quality_reason"] = self._v3dt_world_quality_reason(bbox3d)
                            else:
                                for key in (
                                    "world",
                                    "world_valid",
                                    "world_frame",
                                    "world_source",
                                    "world_quality",
                                    "world_quality_reason",
                                ):
                                    raw.pop(key, None)
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

                if not zone:
                    zone = _fallback_zone_from_camera(camera_id)
                    zone_source = "camera_default" if zone else None
                    zone_authoritative = False

                if reid_debug:
                    self._reid_debug_people += 1

                emb = None
                mgr = getattr(self.pipeline, "stable_id_mgr", None)
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
                                need_emb = bool(needs_fn(int(sensor_id), int(track_id), float(now_ts)))
                            except Exception:
                                need_emb = True
                        else:
                            try:
                                rec = mgr.active_tracks.get((int(sensor_id), int(track_id)))
                                need_emb = rec is None or rec.get("emb") is None
                            except Exception:
                                need_emb = True
                    if need_emb or identity_v2_enabled:
                        if reid_budget_remaining <= 0 and not identity_v2_enabled:
                            pass
                        else:
                            reid_budget_remaining = max(0, reid_budget_remaining - 1)
                            emb = self._extract_reid_embedding_ds8(obj_meta)
                        if reid_debug:
                            if emb is None:
                                self._reid_debug_emb_missing += 1
                            else:
                                self._reid_debug_emb_found += 1

                if identity_v2_authoritative:
                    pose_features, pose_quality = None, None
                else:
                    pose_features, pose_quality = self._extract_stable_id_pose_inputs(
                        obj_meta,
                        mgr,
                        sensor_id=int(sensor_id),
                        track_id=int(track_id),
                        now_ts=float(now_ts),
                    )
                # bbox3d often seeds world above; still augment before StableID
                # when missing so overlap permits are not denied as missing_world.
                world_xy, world_valid, pose_kpts_abs, depth_result = (
                    self._ensure_world_before_stable_id(
                        sensor_id,
                        camera_id,
                        track_id,
                        raw,
                        obj_meta=obj_meta,
                        frame_dims=frame_dims,
                    )
                )

                stable_id = None
                if not identity_v2_authoritative:
                    stable_id = self._maybe_assign_stable_id(
                        sensor_id=sensor_id,
                        track_id=track_id,
                        bbox=raw.get("bbox"),
                        zone=zone,
                        ts=now_ts,
                        frame_bgr=None,
                        embedding=emb,
                        pose_features=pose_features,
                        pose_quality=pose_quality,
                        world_xy=world_xy,
                        world_valid=world_valid,
                    )
                if stable_id is None:
                    if not identity_v2_authoritative:
                        callback = getattr(
                            self.pipeline, "identity_v2_failure_callback", None
                        )
                        if callable(callback) and identity_v2_service is not None:
                            callback(
                                RuntimeError(
                                    "legacy identity unavailable while identity-v2 is shadowing"
                                )
                            )
                        self._stamp_osd_label_ds8(
                            obj_meta, sensor_id=sensor_id, stable_id=None
                        )
                        diagnostics_tracks.append(diag_track)
                        continue

                stable_id_int = int(stable_id) if stable_id is not None else 0
                tracker_id_int = int(track_id)
                id_diag: Dict[str, Any] = {}
                get_id_diag = (
                    getattr(mgr, "get_track_diagnostics", None)
                    if not identity_v2_authoritative
                    else None
                )
                if callable(get_id_diag):
                    try:
                        id_diag = dict(get_id_diag(int(sensor_id), int(track_id)) or {})
                    except Exception:
                        id_diag = {}
                id_event = id_diag.get("id_event")
                id_reject_reason = id_diag.get("id_reject_reason")
                sid_candidate = id_diag.get("sid_candidate")
                if stable_id_int > 0:
                    stable_id_int = self._public_stable_id(
                        stable_id_int,
                        sensor_id=sensor_id,
                        sid_candidate=sid_candidate,
                        ts=now_ts,
                        claimed_public_ids=present_stable_ids,
                    )
                    present_stable_ids.add(stable_id_int)
                embedding_present = bool(id_diag.get("embedding_present", emb is not None))
                if pose_kpts_abs is None:
                    pose_kpts_abs = self._extract_pose_keypoints_for_anchor(obj_meta, raw.get("bbox") or [])
                if depth_result is None:
                    depth_result = self._extract_object_depth_result(obj_meta)
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
                    "camera_id": camera_id,
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
                    **public_frame_timing,
                }
                self._apply_household_id_diag_fields(public_track, id_diag)
                frame_w, frame_h = frame_dims
                if frame_w > 8 and frame_h > 8:
                    public_track["image_size"] = [int(frame_w), int(frame_h)]
                if id_display:
                    public_track["id_display"] = str(id_display)
                for key in (
                    "image_foot",
                    "image_base",
                    "world",
                    "world_valid",
                    "world_quality",
                    "world_quality_reason",
                    "world_frame",
                    "world_source",
                ):
                    if key in raw:
                        public_track[key] = raw.get(key)
                for key in _WORLD_ESTIMATOR_DIAGNOSTIC_FIELDS:
                    if key in raw:
                        public_track[key] = raw.get(key)

                self._augment_track_with_world(
                    sensor_id,
                    camera_id,
                    public_track,
                    obj_meta=obj_meta,
                    pose_kpts_abs=pose_kpts_abs,
                    depth_result=depth_result,
                )
                self._record_stable_id_world_cache(sensor_id, track_id, public_track, now_ts)
                self._apply_public_depth_fields(public_track, depth_result)
                try:
                    setattr(obj_meta, "_noesis_depth_used_m", public_track.get("depth_used_m"))
                except Exception:
                    pass
                # Stamp OSD label after world/depth augmentation so z= reflects the
                # registered depth actually used by the estimator.
                self._stamp_osd_label_ds8(
                    obj_meta,
                    sensor_id=sensor_id,
                    # Zero is an explicit neutral override.  None would ask the
                    # OSD processor to look up stale legacy StableID state.
                    stable_id=0 if identity_v2_authoritative else stable_id_int,
                )
                if not identity_v2_authoritative:
                    self._apply_instance_mask_color_ds8(obj_meta, stable_id=stable_id_int)
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
                self._apply_household_id_diag_fields(diag_track, id_diag)
                tracks.append(public_track)
                identity_v2_primitives.append(
                    IdentityFramePrimitive(
                        camera_id=camera_id,
                        tracker_id=str(tracker_id_int),
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

            self._process_identity_v2_source_frame(
                camera_id=camera_id,
                frame_id=frame_id,
                primitives=identity_v2_primitives,
                observed_at=now_ts,
            )
            tracks = self._dedupe_public_tracks(tracks)
            if identity_v2_authoritative:
                for track in tracks:
                    try:
                        resolved_sid = int(track.get("stable_id"))
                    except (TypeError, ValueError):
                        resolved_sid = 0
                    if resolved_sid <= 0:
                        track["dwell_time"] = None
                        continue
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
                for primitive in identity_v2_primitives:
                    if primitive.diagnostic_track is not None:
                        primitive.diagnostic_track["dwell_time"] = (
                            primitive.public_track.get("dwell_time")
                        )
            present_stable_ids = self._stable_ids_from_tracks(tracks)
            occupancy_counts = self._occupancy_from_tracks(tracks)
            footpoints = self._footpoints_from_tracks(
                tracks,
                frame_dims,
                target_image_size=self._bev_target_image_size(
                    sensor_id,
                    camera_id,
                ),
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
            self._active_tracks[sensor_id] = tracks
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
                    "frame_id": frame_id,
                    "zone": None,
                    "class_id": 0,
                }
                tracks.append(synthetic)
                present_stable_ids.add(1)
                present_track_ids.add(1)

            continuity = self._tracking_lifecycle.update_frame(
                source_id=sensor_id,
                camera_id=camera_id,
                frame_id=frame_id,
                observed_at_us=int(public_frame_timing["observed_at_us"]),
                tracks=tracks,
            )

            # Publish tracking even on zero-track frames so Menon can clear
            # presence without waiting on client-side TTLs.
            if self._tracking_publish_due(
                sensor_id=sensor_id,
                now_ts=now_ts,
                count=len(tracks),
                force=continuity.tracker_keys_changed,
            ):
                tracking_published = False
                try:
                    tracking_receipt = _require_tracking_publication_receipt(
                        self.tracking_pub.publish(
                            sensor_id,
                            tracks,
                            frame_metadata={
                                "frame_id": frame_id,
                                "tracker_lifecycle_tombstones": list(
                                    continuity.tombstones
                                ),
                                **public_frame_timing,
                            },
                        ),
                        source_id=sensor_id,
                        frame_id=frame_id,
                        observed_at_us=int(
                            public_frame_timing["observed_at_us"]
                        ),
                    )
                    self._tracking_lifecycle.mark_published(continuity)
                    self._last_tracking_publish_ts_by_sensor[int(sensor_id)] = float(
                        now_ts
                    )
                    self._last_tracking_count_by_sensor[int(sensor_id)] = len(tracks)
                    tracking_published = True
                except Exception:  # pragma: no cover - telemetry should never break pipeline
                    logger.exception("Tracking telemetry publish failed for sensor %s", sensor_id)
                if tracking_published:
                    try:
                        bev_receipt = self._publish_bev(
                            sensor_id,
                            camera_id,
                            frame_meta,
                            footpoints,
                            now_ts=now_ts,
                            track_count=len(tracks),
                            paired_with_tracking=True,
                            tracking_receipt=tracking_receipt,
                        )
                        if bev_receipt.status == "failed":
                            raise bev_receipt.failure or RuntimeError(
                                "BEV publication failed without a cause"
                            )
                        if bev_receipt.status == "startup_pending":
                            logger.debug(
                                "BEV authority startup pending for sensor %s frame %s",
                                sensor_id,
                                frame_id,
                            )
                    except Exception:
                        logger.exception("BEV publish failed for sensor %s", sensor_id)
        except Exception:  # pragma: no cover - defensive guardrail
            logger.exception("Failed to process analytics telemetry for frame (DS8)")

    def handle_frame(self, frame_meta: Any) -> None:
        """Extract tracking telemetry for a single frame and publish it."""
        try:
            source_id = self._frame_source_id(frame_meta)
            sensor_id = self.sensor_id_map.get(source_id, source_id)
            camera_id = self.camera_labels.get(sensor_id, f"camera_{sensor_id}")
            now_ts = time.time()
            public_frame_timing = _public_frame_timing(frame_meta, now_ts)
            self._log_diag_session_start()

            tracks: List[Dict[str, Any]] = []
            diagnostics_tracks: List[Dict[str, Any]] = []
            frame_id = int(getattr(frame_meta, "frame_num", -1))
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

                mgr = getattr(self.pipeline, "stable_id_mgr", None)
                pose_features, pose_quality = self._extract_stable_id_pose_inputs(
                    obj_meta,
                    mgr,
                    sensor_id=int(sensor_id),
                    track_id=int(track_id),
                    now_ts=float(now_ts),
                )
                world_xy, world_valid, pose_kpts_abs, depth_result = (
                    self._ensure_world_before_stable_id(
                        sensor_id,
                        camera_id,
                        track_id,
                        raw,
                        obj_meta=obj_meta,
                        frame_dims=frame_dims,
                    )
                )
                stable_id = self._maybe_assign_stable_id(
                    sensor_id=sensor_id,
                    track_id=track_id,
                    bbox=raw.get("bbox"),
                    zone=zone,
                    ts=now_ts,
                    frame_bgr=None,
                    embedding=None,
                    pose_features=pose_features,
                    pose_quality=pose_quality,
                    world_xy=world_xy,
                    world_valid=world_valid,
                )
                if stable_id is None:
                    self._stamp_osd_label(obj_meta, sensor_id=sensor_id, stable_id=None)
                    diagnostics_tracks.append(diag_track)
                    continue

                stable_id_int = self._public_stable_id(
                    int(stable_id),
                    sensor_id=sensor_id,
                    ts=now_ts,
                    claimed_public_ids=present_stable_ids,
                )
                present_stable_ids.add(stable_id_int)
                tracker_id_int = int(track_id)
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
                    "camera_id": camera_id,
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
                    **public_frame_timing,
                }
                frame_w, frame_h = frame_dims
                if frame_w > 8 and frame_h > 8:
                    public_track["image_size"] = [int(frame_w), int(frame_h)]
                if id_display:
                    public_track["id_display"] = str(id_display)
                for key in (
                    "image_foot",
                    "image_base",
                    "world",
                    "world_valid",
                    "world_quality",
                    "world_quality_reason",
                    "world_frame",
                    "world_source",
                ):
                    if key in raw:
                        public_track[key] = raw.get(key)
                for key in _WORLD_ESTIMATOR_DIAGNOSTIC_FIELDS:
                    if key in raw:
                        public_track[key] = raw.get(key)

                if pose_kpts_abs is None:
                    pose_kpts_abs = self._extract_pose_keypoints_for_anchor(obj_meta, raw.get("bbox") or [])
                if depth_result is None:
                    depth_result = self._extract_object_depth_result(obj_meta)
                self._augment_track_with_world(
                    sensor_id,
                    camera_id,
                    public_track,
                    obj_meta=obj_meta,
                    pose_kpts_abs=pose_kpts_abs,
                    depth_result=depth_result,
                )
                self._record_stable_id_world_cache(sensor_id, track_id, public_track, now_ts)
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

            tracks = self._dedupe_public_tracks(tracks)
            present_stable_ids = self._stable_ids_from_tracks(tracks)
            occupancy_counts = self._occupancy_from_tracks(tracks)
            footpoints = self._footpoints_from_tracks(
                tracks,
                frame_dims,
                target_image_size=self._bev_target_image_size(
                    sensor_id,
                    camera_id,
                ),
            )

            self._publish_occupancy(sensor_id, occupancy_counts)
            self._cleanup_zone_state(sensor_id, present_stable_ids)
            self._maintain_stable_ids(sensor_id, present_track_ids, now_ts)
            self._active_tracks[sensor_id] = tracks
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

            if not tracks and os.environ.get("NOESIS_REID_TEST_MODE") == "1":
                synthetic = {
                    "camera_id": camera_id,
                    "stable_id": 1,
                    "tracker_id": 1,
                    "bbox": (0.0, 0.0, 10.0, 10.0),
                    "frame_id": frame_id,
                    "zone": None,
                    "class_id": 0,
                }
                tracks.append(synthetic)
                present_stable_ids.add(1)
                present_track_ids.add(1)

            continuity = self._tracking_lifecycle.update_frame(
                source_id=sensor_id,
                camera_id=camera_id,
                frame_id=frame_id,
                observed_at_us=int(public_frame_timing["observed_at_us"]),
                tracks=tracks,
            )

            # Publish tracking even on zero-track frames so Menon can clear
            # presence without waiting on client-side TTLs.
            if self._tracking_publish_due(
                sensor_id=sensor_id,
                now_ts=now_ts,
                count=len(tracks),
                force=continuity.tracker_keys_changed,
            ):
                tracking_published = False
                try:
                    tracking_receipt = _require_tracking_publication_receipt(
                        self.tracking_pub.publish(
                            sensor_id,
                            tracks,
                            frame_metadata={
                                "frame_id": frame_id,
                                "tracker_lifecycle_tombstones": list(
                                    continuity.tombstones
                                ),
                                **public_frame_timing,
                            },
                        ),
                        source_id=sensor_id,
                        frame_id=frame_id,
                        observed_at_us=int(
                            public_frame_timing["observed_at_us"]
                        ),
                    )
                    self._tracking_lifecycle.mark_published(continuity)
                    self._last_tracking_publish_ts_by_sensor[int(sensor_id)] = float(
                        now_ts
                    )
                    self._last_tracking_count_by_sensor[int(sensor_id)] = len(tracks)
                    tracking_published = True
                except Exception:  # pragma: no cover - telemetry should never break pipeline
                    logger.exception("Tracking telemetry publish failed for sensor %s", sensor_id)
                if tracking_published:
                    try:
                        bev_receipt = self._publish_bev(
                            sensor_id,
                            camera_id,
                            frame_meta,
                            footpoints,
                            now_ts=now_ts,
                            track_count=len(tracks),
                            paired_with_tracking=True,
                            tracking_receipt=tracking_receipt,
                        )
                        if bev_receipt.status == "failed":
                            raise bev_receipt.failure or RuntimeError(
                                "BEV publication failed without a cause"
                            )
                        if bev_receipt.status == "startup_pending":
                            logger.debug(
                                "BEV authority startup pending for sensor %s frame %s",
                                sensor_id,
                                frame_id,
                            )
                    except Exception:
                        logger.exception("BEV publish failed for sensor %s", sensor_id)
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
            frame_w = int(_meta_lookup(frame_meta, "source_frame_width", "frame_width", "width", default=0) or 0)
            frame_h = int(_meta_lookup(frame_meta, "source_frame_height", "frame_height", "height", default=0) or 0)
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
        # the calibrated optical center is off-center. Let frame metadata own
        # the size when calibration has no declared resolution.
        return None

    def _track_image_size(self, sensor_id: int, frame_meta: Any) -> Tuple[int, int]:
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

        def _clip_uv(u: float, v: float, *, vertical_overshoot_ratio: float = 0.01) -> Optional[Tuple[float, float]]:
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
        target_image_size_tuple = self._normalize_image_size(target_image_size) or source_image_size_tuple
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

        def _scale_bbox_to_target(bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
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
        u = v = None
        method = None
        bbox_source_tuple: Optional[Tuple[float, float, float, float]] = None
        bbox = track.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            try:
                left_b, top_b, width_b, height_b = [float(x) for x in bbox[:4]]
                if width_b > 0.0 and height_b > 0.0:
                    bbox_source_tuple = (float(left_b), float(top_b), float(width_b), float(height_b))
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

        if u is None or v is None:
            if bbox_source_tuple is None:
                return None
            try:
                left, top, width, height = [float(x) for x in bbox_source_tuple]
            except Exception:
                return None
            if width <= 0.0 or height <= 0.0:
                return None
            u = left + width * 0.5
            v = top + height
            clipped = _clip_uv(float(u), float(v))
            if clipped is None:
                return None
            u, v = clipped
            method = "bbox"
        u, v = _scale_uv_to_target(float(u), float(v))
        bbox_tuple = _scale_bbox_to_target(bbox_source_tuple) if bbox_source_tuple is not None else None

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
        for idx, (key, label) in enumerate((("image_foot", "image_foot"), ("image_base", "image_base"))):
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
            bottom_center = _scale_uv_to_target(float(left + width * 0.5), float(top + height))
            lower_center = _scale_uv_to_target(float(left + width * 0.5), float(top + height * 0.95))
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
            frame_id_value = int(track.get("frame_id")) if track.get("frame_id") is not None else None
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
        idle_jitter = track.get("idle_jitter_m")
        try:
            idle_jitter_f = float(idle_jitter) if idle_jitter is not None else None
        except Exception:
            idle_jitter_f = None
        return Footpoint(
            u=u,
            v=v,
            method=method or "bbox",
            stable_id=stable_id_int,
            tracker_id=tracker_id_int,
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
            motion_mode=str(track.get("motion_mode")) if track.get("motion_mode") not in (None, "") else None,
            posture=str(track.get("posture")) if track.get("posture") not in (None, "") else None,
            trail_append_allowed=trail_append_bool,
            trail_break_required=trail_break_bool,
            trail_segment_id=trail_segment_int,
            idle_jitter_m=idle_jitter_f,
            debug={
                "image_candidates": image_candidates,
                "track_frame_id": frame_id_value,
                "track_image_size": list(source_image_size_tuple) if source_image_size_tuple is not None else None,
                "bev_image_size": list(target_image_size_tuple) if target_image_size_tuple is not None else None,
                "image_scale": [float(scale_x), float(scale_y)],
                "world": list(track.get("world")) if isinstance(track.get("world"), (list, tuple)) else None,
                "world_valid": bool(track.get("world_valid")) if track.get("world_valid") is not None else None,
                "world_source": str(track.get("world_source")) if track.get("world_source") not in (None, "") else None,
                "world_quality": str(track.get("world_quality")) if track.get("world_quality") not in (None, "") else None,
                "motion_mode": str(track.get("motion_mode")) if track.get("motion_mode") not in (None, "") else None,
                "posture": str(track.get("posture")) if track.get("posture") not in (None, "") else None,
                "trail_append_allowed": trail_append_bool,
                "world_quality_reason": str(track.get("world_quality_reason")) if track.get("world_quality_reason") not in (None, "") else None,
                "depth_status": str(track.get("depth_status")) if track.get("depth_status") not in (None, "") else None,
                "depth_anchor_source": str(track.get("depth_anchor_source")) if track.get("depth_anchor_source") not in (None, "") else None,
                "depth_anchor_m": track.get("depth_anchor_m"),
                "depth_used_m": track.get("depth_used_m"),
                "depth_registered_m": track.get("depth_registered_m"),
                "depth_registration_status": track.get("depth_registration_status"),
                "depth_registration_id": track.get("depth_registration_id"),
            },
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
        # Physical motion belongs to the camera-local tracker lifecycle; StableID
        # is semantic association and must not splice different people's state.
        for key_name in ("tracker_id", "track_id"):
            raw = track.get(key_name)
            if raw is None:
                continue
            try:
                id_int = int(raw)
            except Exception:
                continue
            if id_int < 0:
                continue
            return int(sensor_id), int(id_int)
        return None

    def _maybe_prune_world_state(self, now_ts: float) -> None:
        if not self._world_state_by_track:
            return
        if (float(now_ts) - float(self._world_state_last_prune_ts)) < float(self._world_state_prune_interval_s):
            return
        self._world_state_last_prune_ts = float(now_ts)
        ttl = float(self._world_state_ttl_s)
        if ttl <= 0.0:
            self._world_state_by_track.clear()
            return
        expired: List[Tuple[int, int]] = []
        for key, state in self._world_state_by_track.items():
            ts = float(getattr(state, "ts", 0.0) or 0.0)
            if (float(now_ts) - ts) > ttl:
                expired.append(key)
        for key in expired:
            self._world_state_by_track.pop(key, None)

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
        if obj_meta is not None and noesis_pose_meta_ext is not None:
            extract_obj = getattr(noesis_pose_meta_ext, "extract_pose_keypoints", None)
            if callable(extract_obj):
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
                        return keypoints_abs

        payload = self._extract_pose_payload_for_anchor(obj_meta)
        if payload is None:
            return None
        return self._keypoints_abs_from_pose_payload(payload, bbox)

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
        anchor_uv = _depth_anchor_uv(depth_result)
        if anchor_uv is None:
            return None
        anchor_band = str(depth_result.anchor_source or "") if depth_result is not None else ""
        quality = "good" if anchor_band == "lower_body_band" else "estimated"
        quality_reason = f"mask_anchor={anchor_band or 'foot_uv'}"
        return _PoseAnchorCandidate(
            u=float(anchor_uv[0]),
            v=float(anchor_uv[1]),
            source="person_mask_floor",
            quality=quality,
            quality_reason=quality_reason,
            height_lock_eligible=False,
        )

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

    def _maybe_update_world_height_reference(
        self,
        state: _WorldAnchorState,
        calib: Any,
        bbox: Sequence[float],
        foot_world: Sequence[float],
        *,
        flip_u: bool,
        flip_v: bool,
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
            return
        alpha = float(self._world_height_update_alpha)
        state.height_ref_scene = float(state.height_ref_scene + alpha * (float(est_height) - float(state.height_ref_scene)))

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
            C_world = C_world * self._scene_per_meter(calib)
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
    ) -> _DepthObservationResult:
        if depth_result is None:
            return _DepthObservationResult(None, 0.0, "depth_meta_missing")
        if str(depth_result.status) != "ok":
            return _DepthObservationResult(None, 0.0, f"depth_status_{depth_result.status}")
        if not bool(depth_result.is_metric) or str(depth_result.unit) != "m":
            return _DepthObservationResult(None, 0.0, "depth_not_metric")
        anchor_source = str(depth_result.anchor_source or "")
        if anchor_source == "lower_body_band":
            min_support_count = 16
            min_support_fraction = 0.40
            support_scale_denom = 96.0
            anchor_source_weight = 1.0
        elif anchor_source == "torso_core":
            min_support_count = 20
            min_support_fraction = 0.45
            support_scale_denom = 128.0
            anchor_source_weight = 0.50
        else:
            return _DepthObservationResult(None, 0.0, "depth_anchor_source_invalid")
        anchor_depth_m = depth_result.anchor_depth_m
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

    def _predict_world_state(self, state: _WorldAnchorState, now_ts: float) -> Tuple[Optional[float], Optional[float], float]:
        if state.world_x is None or state.world_z is None or float(state.filtered_ts) < 0.0:
            return None, None, 0.0
        dt = max(0.0, float(now_ts) - float(state.filtered_ts))
        pred_x = float(state.world_x) + float(state.vel_world_x) * dt
        pred_z = float(state.world_z) + float(state.vel_world_z) * dt
        return pred_x, pred_z, dt

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
    ) -> np.ndarray:
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
        )
        if state is not None:
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
        )

    def _world_fusion_weights(
        self,
        camera_id: str,
        *,
        floor_weight: float,
        depth_weight: float,
        track: Dict[str, Any],
    ) -> Tuple[float, float, bool]:
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

    def _admit_floor_ray_range(
        self,
        camera_id: str,
        *,
        calib: Any,
        floor_candidate: np.ndarray,
        track: Dict[str, Any],
    ) -> bool:
        policy = self.world_fusion_policy
        if policy is None:
            return True
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
            admitted = bool(
                math.isfinite(horizontal_range_m)
                and horizontal_range_m <= limit_m
            )
            if not admitted:
                rejection_reason = "floor_ray_range_exceeded"
        except Exception:
            admitted = False
        track["world_floor_admitted"] = admitted
        if admitted:
            track.pop("world_floor_rejection_reason", None)
        else:
            track["world_floor_rejection_reason"] = rejection_reason
        return admitted

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

    def _refine_seeded_world_with_ground_state(
        self,
        sensor_id: int,
        camera_id: str,
        track: Dict[str, Any],
        *,
        world_source_label: str,
    ) -> None:
        """Smooth + idle-lock a pre-seeded world measurement (bbox3d / V3DT foot).

        Keeps the producer world_source label so BEV/overlap treat it as live
        tracking, while restoring PersonGroundState continuity that the early
        return previously skipped.
        """
        if self.bev_calibration is None:
            track.setdefault("world_quality", "good")
            track.setdefault("world_frame", self._world_frame)
            return
        if track.get("world_valid") is not True:
            return
        world = track.get("world")
        if not isinstance(world, (list, tuple)) or len(world) < 3:
            return
        try:
            mx = float(world[0])
            my = float(world[1])
            mz = float(world[2])
        except Exception:
            return
        if not (math.isfinite(mx) and math.isfinite(my) and math.isfinite(mz)):
            return

        try:
            calib = self.bev_calibration.snapshot(sensor_id, camera_id)
            if calib is None or calib.intrinsics is None or calib.extrinsics_col_major is None:
                track.setdefault("world_quality", "good")
                track.setdefault("world_frame", self._world_frame)
                return

            now_ts = float(time.time())
            world_key = self._world_track_key(sensor_id, track)
            self._maybe_prune_world_state(now_ts)
            state: Optional[_WorldAnchorState] = None
            if world_key is not None:
                state = self._world_state_by_track.get(world_key)
                if state is None:
                    state = _WorldAnchorState()
                    self._world_state_by_track[world_key] = state
                state.ts = float(now_ts)

            measurement = np.array([mx, float(calib.floor_y), mz], dtype=np.float64)
            hit = self._update_track_world_state(
                track,
                state,
                measurement=measurement,
                floor_y=float(calib.floor_y),
                now_ts=float(now_ts),
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
                    now_ts=float(now_ts),
                    image_foot_uv=image_uv,
                    config=self._human_ground_cfg,
                )
                if state.motion_mode in ("idle", "sit", "lie") and state.locked_world is not None:
                    hit = np.array(
                        [float(state.locked_world[0]), float(calib.floor_y), float(state.locked_world[1])],
                        dtype=np.float64,
                    )
                    state.world_x = float(state.locked_world[0])
                    state.world_z = float(state.locked_world[1])
                    state.vel_world_x = 0.0
                    state.vel_world_z = 0.0

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
            wx = float(hit[0])
            wy = float(hit[1])
            wz = float(hit[2])
            track["world"] = [float(wx), float(wy), float(wz)]
            track["world_valid"] = True
            track.setdefault("world_quality", "good")
            track["world_frame"] = self._world_frame
            track["world_source"] = str(world_source_label)
            if state is not None:
                for key, value in state.as_public_fields().items():
                    if value is not None:
                        track[key] = value
                if world_source_label != "anchor_hold":
                    state.last_good_world = (float(wx), float(wy), float(wz))
                    state.last_good_ts = float(now_ts)
                state.ts = float(now_ts)
        except Exception:
            track.setdefault("world_quality", "good")
            track.setdefault("world_frame", self._world_frame)

    def _augment_track_with_world(
        self,
        sensor_id: int,
        camera_id: str,
        track: Dict[str, Any],
        *,
        obj_meta: Any | None = None,
        pose_kpts_abs: Optional[np.ndarray] = None,
        depth_result: Optional[ObjectDepthResult] = None,
    ) -> None:
        """Calculate world coordinates for a track if calibration is available."""
        if self._tracking_mode_is_v3dt():
            has_v3dt_world = bool(
                isinstance(track.get("bbox3d"), Mapping)
                and self._is_v3dt_world_source(track.get("world_source"))
                and track.get("world_frame") == self._world_frame
                and track.get("world_valid") is True
                and isinstance(track.get("world"), (list, tuple))
            )
            if has_v3dt_world:
                if self.bev_calibration is not None:
                    self._refine_seeded_world_with_ground_state(
                        sensor_id,
                        camera_id,
                        track,
                        world_source_label=str(
                            track.get("world_source") or V3DT_WORLD_SOURCE_BBOX3D_FOOT
                        ),
                    )
                return
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
            return
        if track.get("world_estimator_evaluated") is True:
            return
        if getattr(self, "_is_v3dt_world_source", None) is not None and callable(self._is_v3dt_world_source) and self._is_v3dt_world_source(track.get("world_source")):
            if track.get("world_valid") is True:
                self._refine_seeded_world_with_ground_state(
                    sensor_id,
                    camera_id,
                    track,
                    world_source_label=str(track.get("world_source") or V3DT_WORLD_SOURCE_BBOX3D_FOOT),
                )
            return
        if track.get("world_source") == "bbox3d":
            if track.get("world_valid") is True:
                self._refine_seeded_world_with_ground_state(
                    sensor_id,
                    camera_id,
                    track,
                    world_source_label="bbox3d",
                )
            return
        if track.get("world") is not None and track.get("world_valid") is True:
            track.setdefault("world_quality", "good")
            track.setdefault("world_frame", self._world_frame)
            return

        try:
            calib = self.bev_calibration.snapshot(sensor_id, camera_id)
            if calib is None or calib.intrinsics is None or calib.extrinsics_col_major is None:
                return

            bbox = track.get("bbox")
            if not bbox or len(bbox) < 4:
                return
            track_image_size = self._normalize_image_size(track.get("image_size") or track.get("frame_size"))
            calib_image_size = self._normalize_image_size(getattr(calib, "image_size", None))
            bbox_project = self._scale_bbox_to_image_size(bbox, track_image_size, calib_image_size)
            if not bbox_project or len(bbox_project) < 4:
                return

            flip_u, flip_v = self._infer_image_flips(camera_id, calib)
            now_ts = float(time.time())
            world_key = self._world_track_key(sensor_id, track)
            self._maybe_prune_world_state(now_ts)
            state: Optional[_WorldAnchorState] = None
            if world_key is not None:
                state = self._world_state_by_track.get(world_key)
                if state is None:
                    state = _WorldAnchorState()
                    self._world_state_by_track[world_key] = state
                state.ts = float(now_ts)

            if pose_kpts_abs is None:
                pose_kpts_abs = self._extract_pose_keypoints_for_anchor(obj_meta, bbox)

            posture = classify_posture(
                kpts_abs=pose_kpts_abs,
                bbox=bbox_project,
                height_ref_scene=state.height_ref_scene if state is not None else None,
                config=self._human_ground_cfg,
            )
            if state is not None:
                state.posture = str(posture)

            pose_anchor = (
                self._resolve_pose_floor_anchor(pose_kpts_abs, posture=posture)
                if pose_kpts_abs is not None
                else None
            )
            # Prefer person-mask foot when pose is weak/bent and posture is non-upright.
            person_anchor = self._resolve_person_depth_anchor(depth_result)
            if pose_anchor is not None and person_anchor is not None and posture in ("sitting", "lying"):
                if str(pose_anchor.source) in ("pose_leg_floor", "pose_single_ankle_floor"):
                    pose_anchor = None
            if pose_anchor is None:
                anchor_candidate = person_anchor
            else:
                anchor_candidate = pose_anchor

            hit: Optional[np.ndarray] = None
            floor_candidate: Optional[np.ndarray] = None
            quality = "invalid"
            quality_reason: Optional[str] = "no_floor_intersection"
            world_source: Optional[str] = None
            source_measurement_rejected = False
            reject_current_geometry = False
            floor_ray_admitted = True
            floor_ray_rejection_reason: Optional[str] = None

            if anchor_candidate is not None:
                track["image_foot"] = [float(anchor_candidate.u), float(anchor_candidate.v)]
                pose_u, pose_v = self._scale_uv_to_image_size(
                    float(anchor_candidate.u),
                    float(anchor_candidate.v),
                    track_image_size,
                    calib_image_size,
                )
                hit = self._project_pixel_to_floor_world(
                    calib,
                    float(pose_u),
                    float(pose_v),
                    flip_u=flip_u,
                    flip_v=flip_v,
                )
                if hit is not None:
                    floor_candidate = np.asarray(hit, dtype=np.float64).copy()
                    track["world_floor_candidate"] = [
                        float(floor_candidate[0]),
                        float(floor_candidate[1]),
                        float(floor_candidate[2]),
                    ]
                    depth_observation = self._depth_observation_from_anchor(
                        calib=calib,
                        anchor=anchor_candidate,
                        depth_result=depth_result,
                        flip_u=flip_u,
                        flip_v=flip_v,
                    )
                    depth_obs = depth_observation.world_point
                    depth_weight = float(depth_observation.weight)
                    depth_reason = str(depth_observation.reason)
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
                    base_floor_weight = 1.0 if anchor_candidate.quality == "good" else 0.75
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
                    )
                    if not floor_ray_admitted:
                        floor_ray_rejection_reason = str(
                            track.get("world_floor_rejection_reason")
                            or "floor_ray_geometry_invalid"
                        )
                        floor_weight = 0.0
                        floor_only_allowed = False
                        track["world_floor_weight_effective"] = 0.0
                    registration_status = depth_observation.registration_status
                    reject_current_geometry = bool(
                        registration_status is not None
                        and registration_status not in ("ok", "raw_passthrough")
                    )
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
                        )
                    if not reject_current_geometry and depth_obs is not None and effective_depth_weight > 0.0:
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
                            quality_reason = "fusion_policy_requires_registered_depth"
                else:
                    hit = None

            # Gravity-drop assumes upright height. Skip only for confirmed non-upright
            # motion modes (or clear lying boxes). A short box alone can be lower-body
            # occlusion of a standing person — height lock is exactly for that case.
            allow_gravity = True
            if posture == "lying":
                allow_gravity = False
            if state is not None and str(state.motion_mode) in ("sit", "lie"):
                allow_gravity = False
            if (
                not source_measurement_rejected
                and hit is None
                and not reject_current_geometry
                and allow_gravity
                and state is not None
                and state.height_ref_scene is not None
            ):
                gravity_hit = self._gravity_drop_world(
                    calib,
                    bbox_project,
                    float(state.height_ref_scene),
                    flip_u=flip_u,
                    flip_v=flip_v,
                )
                if gravity_hit is not None:
                    hit = self._update_track_world_state(
                        track,
                        state,
                        measurement=gravity_hit,
                        floor_y=float(calib.floor_y),
                        now_ts=float(now_ts),
                        alpha=float(self._world_smooth_alpha_weak),
                        beta=max(0.0, min(1.0, float(self._world_smooth_alpha_weak) * 0.15)),
                        quality="estimated",
                    )
                    world_source = "gravity_drop"
                    quality = "estimated"
                    quality_reason = "current_anchor_unavailable" if anchor_candidate is None else "current_anchor_projection_failed"

            fallback_reason = self._fallback_quality_reason(pose_kpts_abs, pose_anchor, person_anchor, depth_result, state)

            if source_measurement_rejected:
                fallback_reason = (
                    quality_reason
                    if quality_reason in (
                        "fusion_policy_requires_registered_depth",
                        "floor_ray_range_exceeded",
                        "floor_ray_geometry_invalid",
                    )
                    else "source_hysteresis_rejected_current_measurement"
                )
            measurement_rejected = bool(
                state is not None
                and "world_prefilter_measurement" in track
                and not state.measurement_accepted
            )
            if measurement_rejected:
                quality_reason = str(
                    state.measurement_rejection_reason or "physical_measurement_rejected"
                )
                fallback_reason = quality_reason
                hold_age = float(now_ts) - float(state.last_good_ts or 0.0)
                if (
                    state.last_good_world is not None
                    and hold_age <= float(self._world_anchor_hold_ttl_s)
                ):
                    hit = np.asarray(state.last_good_world, dtype=np.float64)
                    quality = "estimated"
                    world_source = "anchor_hold"
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
                if hold_age <= float(self._world_anchor_hold_ttl_s):
                    hit = np.asarray(state.last_good_world, dtype=np.float64)
                    world_source = "anchor_hold"
                    quality = "estimated"
                    quality_reason = fallback_reason

            if hit is not None:
                track["world_estimator_evaluated"] = True
                # Phase 1: motion mode / stationary lock using image foot + speed.
                image_uv = None
                raw_foot = track.get("image_foot")
                if isinstance(raw_foot, (list, tuple)) and len(raw_foot) >= 2:
                    try:
                        image_uv = (float(raw_foot[0]), float(raw_foot[1]))
                    except Exception:
                        image_uv = None
                if state is not None:
                    update_motion_mode(
                        state,
                        now_ts=float(now_ts),
                        image_foot_uv=image_uv,
                        config=self._human_ground_cfg,
                    )
                    if state.motion_mode in ("idle", "sit", "lie") and state.locked_world is not None:
                        hit = np.array(
                            [float(state.locked_world[0]), float(calib.floor_y), float(state.locked_world[1])],
                            dtype=np.float64,
                        )
                        state.world_x = float(state.locked_world[0])
                        state.world_z = float(state.locked_world[1])
                        state.vel_world_x = 0.0
                        state.vel_world_z = 0.0

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
                track["world_frame"] = self._world_frame
                if world_source:
                    track["world_source"] = str(world_source)
                else:
                    track.pop("world_source", None)
                if state is not None:
                    for key, value in state.as_public_fields().items():
                        if value is not None:
                            track[key] = value
                    if world_source != "anchor_hold":
                        state.last_good_world = (float(wx), float(wy), float(wz))
                        state.last_good_ts = float(now_ts)
                        state.ts = float(now_ts)
            else:
                track["world_estimator_evaluated"] = True
                track["world_valid"] = False
                track["world_quality"] = "invalid"
                track["world_quality_reason"] = str(fallback_reason or "no_floor_intersection")
                track.pop("world_source", None)
                if state is not None:
                    for key, value in state.as_public_fields().items():
                        if value is not None:
                            track[key] = value
        except Exception:
            # Silently fail; world coordinates are best-effort
            pass

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
        ts_us = self._frame_timestamp_us(frame_meta)
        frame_id = int(
            _meta_lookup(
                frame_meta,
                "frame_number",
                "frame_num",
                default=0,
            )
            or 0
        )
        observed_at_us = max(1, int(ts_now * 1_000_000))
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
        if not paired_with_tracking and not self._bev_publish_due(
            sensor_id=sensor_id,
            now_ts=ts_now,
            count=fp_count,
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
                observed_at_us=observed_at_us,
                tracking_publication_sequence=tracking_sequence,
                tracking_outbound_submission_id=tracking_submission_id,
                failure=failure,
            )
        try:
            calib = self.bev_calibration.snapshot(sensor_id, camera_id)
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
                observed_at_us=observed_at_us,
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
                observed_at_us=observed_at_us,
                tracking_publication_sequence=tracking_sequence,
                tracking_outbound_submission_id=tracking_submission_id,
                failure=failure,
            )
        try:
            receipt = self.bev_renderer.render_and_publish(
                camera_id=camera_id,
                calib=calib,
                footpoints=list(footpoints),
                timestamp_us=ts_us,
                source_id=int(sensor_id),
                frame_id=frame_id,
                observed_at_us=observed_at_us,
                tracking_publication_sequence=tracking_sequence,
                tracking_outbound_submission_id=tracking_submission_id,
            )
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
                observed_at_us=observed_at_us,
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
                observed_at_us=observed_at_us,
                tracking_publication_sequence=tracking_sequence,
                tracking_outbound_submission_id=tracking_submission_id,
                failure=failure,
            )
        if (
            receipt.source_id != int(sensor_id)
            or receipt.frame_id != frame_id
            or receipt.observed_at_us != observed_at_us
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
                observed_at_us=observed_at_us,
                tracking_publication_sequence=tracking_sequence,
                tracking_outbound_submission_id=tracking_submission_id,
                failure=failure,
            )
        return receipt

    def _iter_object_meta(self, frame_meta: Any) -> Iterable[Any]:
        cast = _resolve_pyds_cast("NvDsObjectMeta")
        entries = _iter_meta_entries(getattr(frame_meta, "obj_meta_list", None), cast)
        for entry in entries:
            if entry is None:
                continue
            yield entry

    def _build_track_dict_ds8(self, obj_meta: Any, camera_id: str) -> Optional[Dict[str, Any]]:
        """Build track dictionary from DS8 pyservicemaker ObjectMetadata."""
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

        if self._tracking_mode_is_v3dt():
            v3dt_meta = self._extract_v3dt_meta_ds8(obj_meta)
            if v3dt_meta:
                track.update(v3dt_meta)
                bbox3d = v3dt_meta.get("bbox3d")
                if isinstance(bbox3d, dict):
                    try:
                        tracker_foot = v3dt_bbox3d_tracker_foot(bbox3d)
                        track["v3dt_world_raw"] = [float(value) for value in tracker_foot]
                        track["v3dt_world_axes"] = str(self._v3dt_caminfo_world_axes)
                    except V3DTAxisMapError:
                        pass
                    world = self._world_from_bbox3d(bbox3d)
                    if world is not None:
                        track["v3dt_world_axis_foot"] = world

        tracker_conf = getattr(obj_meta, "tracker_confidence", None)
        if tracker_conf is not None:
            try:
                track["tracker_confidence"] = float(tracker_conf)
            except Exception:
                pass

        # DS8 API: nvdsanalytics_obj_items is an iterable of analytics obj info
        analytics_items = getattr(obj_meta, "nvdsanalytics_obj_items", None) or []
        if not analytics_items:
            logger.debug(f"DS8 track {track_id} has no analytics items")
        for analytics_info in analytics_items:
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
                val = getattr(analytics_info, field, None)
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

    def _ensure_world_before_stable_id(
        self,
        sensor_id: int,
        camera_id: str,
        track_id: int,
        raw: Dict[str, Any],
        *,
        obj_meta: Any | None = None,
        frame_dims: Optional[Tuple[int, int]] = None,
    ) -> Tuple[
        Optional[Tuple[float, float]],
        bool,
        Optional[np.ndarray],
        Optional[ObjectDepthResult],
    ]:
        """Fill world on ``raw`` before StableID when bbox3d did not already seed it."""
        world_xy, world_valid = self._world_xy_for_stable_id(sensor_id, track_id, raw)
        if world_valid:
            return world_xy, True, None, None

        pose_kpts_abs = self._extract_pose_keypoints_for_anchor(obj_meta, raw.get("bbox") or [])
        depth_result = self._extract_object_depth_result(obj_meta)

        world_track: Dict[str, Any] = {
            "tracker_id": int(track_id),
            "track_id": int(track_id),
            "bbox": raw.get("bbox"),
            "center": raw.get("center"),
            "camera_id": camera_id,
        }
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
            "world_source",
            "image_size",
            "frame_size",
        ):
            if key in raw:
                world_track[key] = raw.get(key)
        if frame_dims is not None:
            frame_w, frame_h = frame_dims
            if frame_w > 8 and frame_h > 8 and "image_size" not in world_track:
                world_track["image_size"] = [int(frame_w), int(frame_h)]

        self._augment_track_with_world(
            sensor_id,
            camera_id,
            world_track,
            obj_meta=obj_meta,
            pose_kpts_abs=pose_kpts_abs,
            depth_result=depth_result,
        )
        for key in (
            "image_foot",
            "image_base",
            "world",
            "world_valid",
            "world_quality",
            "world_quality_reason",
            "world_frame",
            "world_source",
        ):
            if key in world_track:
                raw[key] = world_track.get(key)
        for key in _WORLD_ESTIMATOR_DIAGNOSTIC_FIELDS:
            if key in world_track:
                raw[key] = world_track.get(key)

        world_xy, world_valid = self._world_xy_for_stable_id(sensor_id, track_id, raw)
        return world_xy, world_valid, pose_kpts_abs, depth_result

    def _world_xy_for_stable_id(
        self,
        sensor_id: int,
        track_id: int,
        raw: Mapping[str, Any],
    ) -> Tuple[Optional[Tuple[float, float]], bool]:
        if raw.get("world_valid") is True:
            world = raw.get("world")
            if isinstance(world, (list, tuple)) and len(world) >= 3:
                try:
                    wx = float(world[0])
                    wz = float(world[2])
                    if math.isfinite(wx) and math.isfinite(wz):
                        return (wx, wz), True
                except Exception:
                    pass
        cached = self._stable_id_world_cache.get((int(sensor_id), int(track_id)))
        if cached is not None:
            try:
                wx, wz, valid, _ts = cached
                if bool(valid) and math.isfinite(float(wx)) and math.isfinite(float(wz)):
                    return (float(wx), float(wz)), True
            except Exception:
                pass
        return None, False

    def _record_stable_id_world_cache(
        self,
        sensor_id: int,
        track_id: int,
        track: Mapping[str, Any],
        ts: float,
    ) -> None:
        if track.get("world_valid") is not True:
            return
        world = track.get("world")
        if not isinstance(world, (list, tuple)) or len(world) < 3:
            return
        try:
            wx = float(world[0])
            wz = float(world[2])
            if not math.isfinite(wx) or not math.isfinite(wz):
                return
            self._stable_id_world_cache[(int(sensor_id), int(track_id))] = (
                float(wx),
                float(wz),
                True,
                float(ts),
            )
        except Exception:
            return

    def _extract_pose_payload(self, obj_meta: Any) -> Optional[Dict[str, Any]]:
        """Pose feature JSON for StableID (same source as floor-anchor payload)."""
        return self._extract_pose_payload_for_anchor(obj_meta)

    def _extract_stable_id_pose_inputs(
        self,
        obj_meta: Any,
        mgr: Any,
        *,
        sensor_id: Optional[int] = None,
        track_id: Optional[int] = None,
        now_ts: Optional[float] = None,
    ) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
        if mgr is None or not bool(getattr(mgr, "pose_enabled", False)):
            return None, None
        needs_fn = getattr(mgr, "needs_pose_update", None)
        if callable(needs_fn) and sensor_id is not None and track_id is not None:
            try:
                ts = float(now_ts if now_ts is not None else time.time())
                if not bool(needs_fn(int(sensor_id), int(track_id), ts)):
                    return None, None
            except Exception:
                pass
        try:
            payload = self._extract_pose_payload(obj_meta)
        except Exception:
            return None, None
        if payload is None:
            return None, None
        raw_features = payload.get("features")
        if not isinstance(raw_features, Mapping) or not raw_features:
            return None, None
        pose_features: Dict[str, float] = {}
        for key, value in raw_features.items():
            try:
                pose_features[str(key)] = float(value)
            except Exception:
                continue
        if not pose_features:
            return None, None
        pose_quality: Dict[str, float] = {}
        for key in ("kpt_mean_conf", "kpt_min_conf", "kpt_valid_frac"):
            if key not in payload:
                continue
            try:
                pose_quality[key] = float(payload[key])
            except Exception:
                continue
        return pose_features, (pose_quality or None)

    def _apply_household_id_diag_fields(
        self,
        target: Dict[str, Any],
        id_diag: Mapping[str, Any],
    ) -> None:
        for key in (
            "reid_confidence",
            "reid_required",
            "overlap_permit",
            "identity_state",
            "identity_kind",
            "resident_uuid",
            "display_name",
            "visitor_generation",
        ):
            if key in id_diag and id_diag.get(key) is not None:
                target[key] = id_diag.get(key)

    def _process_identity_v2_source_frame(
        self,
        *,
        camera_id: str,
        frame_id: int,
        primitives: Sequence[IdentityFramePrimitive],
        observed_at: float,
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
            callback = getattr(self.pipeline, "identity_v2_failure_callback", None)
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
        pose_features: Optional[Dict[str, float]] = None,
        pose_quality: Optional[Dict[str, float]] = None,
        world_xy: Optional[Tuple[float, float]] = None,
        world_valid: bool = False,
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
                stable_id = mgr.update(
                    sensor_id=int(sensor_id),
                    ds_obj_id=int(track_id),
                    bbox_ltrbwh=(float(safe_bbox[0]), float(safe_bbox[1]), float(safe_bbox[2]), float(safe_bbox[3])),
                    ts=float(ts),
                    zone=str(zone) if zone else None,
                    frame_bgr=frame_bgr,
                    embedding=embedding,
                    pose_features=pose_features,
                    pose_quality=pose_quality,
                    world_xy=world_xy,
                    world_valid=bool(world_valid),
                )
                stable_id_int = int(stable_id)
                if stable_id_int > 0:
                    return stable_id_int
            except Exception:
                # Do not permanently disable StableID on a single-frame failure;
                # that blanks tracking/BEV for the rest of the session.
                logger.exception("StableIDManager update failed for sensor %s track %s", sensor_id, track_id)

        return None

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
                mgr.remove_missing_tracks(sensor_id_int, list(present_set), now_ts)
                mgr.prune_ghosts(now_ts)
            except Exception:
                # Do not permanently disable StableID on maintenance failure.
                logger.exception("StableIDManager maintenance failed for sensor %s", sensor_id_int)

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
            return
        track["depth_status"] = str(depth_result.status)
        track["depth_anchor_source"] = str(depth_result.anchor_source) if depth_result.anchor_source else None
        if track.get("depth_anchor_m") is None:
            track["depth_anchor_m"] = float(depth_result.anchor_depth_m) if depth_result.anchor_depth_m is not None else None
        if track.get("depth_used_m") is None:
            track["depth_used_m"] = _depth_used_m(depth_result)
        track["depth_registered_m"] = track.get("depth_registered_m")
        track["depth_registration_status"] = track.get("depth_registration_status")
        track["depth_registration_id"] = track.get("depth_registration_id")
        track["depth_center_m"] = float(depth_result.depth_center) if depth_result.depth_center is not None else None
        track["depth_median_m"] = float(depth_result.depth_median) if depth_result.depth_median is not None else None
        track["depth_sample_count"] = int(depth_result.sample_count)
        track["depth_valid_fraction"] = float(depth_result.valid_fraction)

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
    font_size: Optional[int] = 11
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
        font_size_raw = font_size_env if font_size_env else cfg.get("font_size", 11)
        font_size = _int(font_size_raw, 11)
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
        try:
            class_id = int(getattr(obj_meta, "class_id", -1))
        except Exception:
            class_id = -1
        try:
            track_id = int(getattr(obj_meta, "object_id", -1))
        except Exception:
            track_id = -1

        # People: compact "#id confidence" (no class name / depth).
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
                id_text = f"[{tracker_text}]|[{stable_text}]"
            elif stable_id_int is not None and stable_id_int > 0:
                id_text = f"#{stable_id_int}"
            else:
                id_text = "#XX"
            base_label = id_text
        else:
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
                label = f"class {class_id}" if class_id >= 0 else "class"
            base_label = label

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


class _PersonBBoxOverlayOperator(_BatchMetadataOperatorBase):  # pragma: no cover - requires DeepStream runtime
    def __init__(self, processor: PersonBBoxOverlayProcessor) -> None:
        super().__init__()
        self._processor = processor

    def handle_metadata(self, batch_meta: Any) -> None:
        if batch_meta is None:
            return
        try:
            self._processor.handle_batch_ds8(batch_meta)
        except Exception:
            logger.exception("Failed to render person bbox overlay within batch metadata (DS8)")


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


class _MapAnythingOperator(_BatchMetadataOperatorBase):  # pragma: no cover - requires DeepStream runtime
    def __init__(self, processor: MapAnythingProcessor) -> None:
        super().__init__()
        self._processor = processor
        self._frames_seen = 0
        self._matched_frames = 0
        self._warned_no_tensors = False
        self._warned_no_match = False
        self._warned_native_probe = False

    def handle_metadata(self, batch_meta: Any) -> None:
        frame_items = getattr(batch_meta, "frame_items", None)
        if frame_items is None:
            return
        for frame_meta in frame_items:
            self._frames_seen += 1
            try:
                tensor_items = getattr(frame_meta, "tensor_items", None)
                if tensor_items is None:
                    if not self._warned_no_tensors:
                        logger.debug(
                            "MapAnything frame missing tensor_items (frame_number=%s)",
                            getattr(frame_meta, "frame_number", None),
                        )
                        self._warned_no_tensors = True
                    continue
                items_list = list(tensor_items)
                converted_items: List[Any] = []
                for item in items_list:
                    convert_fn = getattr(item, "as_tensor_output", None)
                    if callable(convert_fn):
                        try:
                            item = convert_fn()
                        except Exception:
                            logger.debug("Failed to convert tensor metadata via as_tensor_output")
                            continue
                    converted_items.append(item)
                if not converted_items:
                    if not self._warned_no_tensors:
                        logger.debug(
                            "MapAnything tensor_items present but none convertible to tensor output (frame_number=%s)",
                            getattr(frame_meta, "frame_number", None),
                        )
                        self._warned_no_tensors = True
                    continue
                matched = False
                for tensor_meta in converted_items:
                    unique_id = getattr(tensor_meta, "unique_id", -1)
                    if int(unique_id) == self._processor.gie_id:
                        matched = True
                        self._processor.handle_nvds_tensor_ds8(frame_meta, tensor_meta)
                if matched:
                    self._matched_frames += 1
                elif converted_items:
                    native_result = self._processor.handle_native_frame_ds8(frame_meta)
                    if native_result is not None:
                        self._matched_frames += 1
                        continue
                    ids = [getattr(item, "unique_id", None) for item in converted_items]
                    _increment_core_counter("tensor_gie_mismatch_drops_total.mapanything")
                    if not self._warned_no_match:
                        native_probe = None
                        if noesis_depth_tracking_tensor_ext is not None and not self._warned_native_probe:
                            self._warned_native_probe = True
                            try:
                                capture_fn = getattr(noesis_depth_tracking_tensor_ext, "capture_aligned_depth_frame", None)
                                if callable(capture_fn):
                                    frame_w = int(
                                        _meta_lookup(frame_meta, "source_frame_width", "frame_width", "width", default=0)
                                        or 0
                                    )
                                    frame_h = int(
                                        _meta_lookup(frame_meta, "source_frame_height", "frame_height", "height", default=0)
                                        or 0
                                    )
                                    if frame_w <= 0 or frame_h <= 0:
                                        frame_w, frame_h = getattr(self._processor.pipeline, "frame_size", (0, 0))
                                    probe_frame = capture_fn(
                                        frame_meta,
                                        int(self._processor.gie_id),
                                        int(frame_w or 0),
                                        int(frame_h or 0),
                                    )
                                    if probe_frame is not None:
                                        native_probe = {
                                            "found": True,
                                            "depth_width": int(getattr(probe_frame, "depth_width", 0) or 0),
                                            "depth_height": int(getattr(probe_frame, "depth_height", 0) or 0),
                                        }
                                    else:
                                        native_probe = {"found": False}
                            except Exception as exc:
                                native_probe = {"error": str(exc)}
                        logger.debug(
                            "MapAnything tensor_items present but no matching gie_id=%s (frame_number=%s, available_ids=%s, native_probe=%s)",
                            self._processor.gie_id,
                            getattr(frame_meta, "frame_number", None),
                            ids,
                            native_probe,
                        )
                        self._warned_no_match = True
            except Exception:
                logger.exception("Failed to process MapAnything tensors from batch metadata (DS8)")
                raise


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
                self._processor.handle_frame_ds8(frame_meta)
            except Exception:
                logger.exception("Failed to compute pose features within batch metadata (DS8)")


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
    if pyds is None:
        return None
    target = getattr(pyds, name, None)
    if target is None:
        return None
    cast = getattr(target, "cast", None)
    return cast if callable(cast) else None


def _resolve_pyds_attr(name: str) -> Any:
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
    try:
        left = float(getattr(rect, "left", getattr(rect, "x", 0.0)))
        top = float(getattr(rect, "top", getattr(rect, "y", 0.0)))
        width = float(getattr(rect, "width", getattr(rect, "w", 0.0)))
        height = float(getattr(rect, "height", getattr(rect, "h", 0.0)))
    except Exception:
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
