from __future__ import annotations

import colorsys
import json
import logging
import math
import os
import queue
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from geometry.depth_source import DepthStorageManager
from geometry.homography import Plane, parse_extrinsics, ray_from_pixel, intersect_plane
from noesis.metadata import intrinsics as intrinsics_module
from noesis.metadata.depth_result import DepthResult
from noesis.metadata.pose_features import PoseFeatureResult
from noesis.telemetry.bev import Footpoint

try:  # DeepStream imports are optional during unit tests
    from pyservicemaker import BatchMetadataOperator, Probe, osd as ds_osd  # type: ignore
except Exception:  # pragma: no cover - exercised only in DS runtime
    BatchMetadataOperator = None  # type: ignore
    Probe = None  # type: ignore
    ds_osd = None  # type: ignore

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

try:  # pragma: no cover - diagnostics optional in tests
    from noesis.diagnostics.telemetry_log import TrackingDiagnosticsLogger
except Exception:  # pragma: no cover - fallback when diagnostics are absent
    TrackingDiagnosticsLogger = None  # type: ignore

logger = logging.getLogger(__name__)
_REID_DLPACK_DEBUG_LOGGED = False
_POSE_DLPACK_TORCH_LOGGED = False
_CORE_FALLBACK_POLICY_ALLOW = "allow"
_CORE_FALLBACK_POLICY_GATE = "gate"
_CORE_FALLBACK_POLICY_FAIL = "fail"
_DLPACK_HOST_READ_LOCK = threading.Lock()


class _CorePathFallbackConversionError(RuntimeError):
    """Raised when fallback conversion policy requests fail-fast behavior."""


def _resolve_core_fallback_policy() -> str:
    raw = str(os.environ.get("NOESIS_CORE_PATH_FALLBACK_POLICY", _CORE_FALLBACK_POLICY_ALLOW) or "").strip().lower()
    if raw in ("fail", "fail-fast", "strict", "error"):
        return _CORE_FALLBACK_POLICY_FAIL
    if raw in ("gate", "block", "drop", "skip"):
        return _CORE_FALLBACK_POLICY_GATE
    return _CORE_FALLBACK_POLICY_ALLOW


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
        fallback: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> str:
        now_ns = time.time_ns()
        policy = _CORE_FALLBACK_POLICY_ALLOW
        with self._lock:
            total = self._inc_locked("core_path.cpu_copy_violation.total")
            per_loc = self._inc_locked(f"core_path.cpu_copy_violation.{location}")
            event: Dict[str, Any] = {
                "type": "core_path_cpu_copy_violation",
                "ts_ns": int(now_ns),
                "location": str(location),
                "reason": str(reason),
                "fallback": bool(fallback),
                "count": int(per_loc),
                "total": int(total),
            }
            if details:
                event["details"] = dict(details)
            if fallback or per_loc <= 3 or (per_loc % 250) == 0:
                self.events.append(event)
            if fallback:
                self._inc_locked("core_path.fallback_conversion.total")
                self._inc_locked(f"core_path.fallback_conversion.{location}")
                policy = _resolve_core_fallback_policy()
                if policy != _CORE_FALLBACK_POLICY_ALLOW:
                    self.events.append(
                        {
                            "type": "core_path_fallback_policy",
                            "ts_ns": int(now_ns),
                            "location": str(location),
                            "policy": str(policy),
                            "reason": str(reason),
                        }
                    )
        return policy

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


def _clone_tensor_for_host_read(tensor: Any, *, location: str) -> Any | None:
    """Best-effort clone before host conversion to avoid shared metadata ownership hazards."""
    if tensor is None:
        return None
    clone_fn = getattr(tensor, "clone", None)
    if not callable(clone_fn):
        return tensor
    try:
        return clone_fn()
    except Exception:
        logger.debug("Tensor clone failed before host read (%s)", location, exc_info=True)
        return None


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
    )
    component.config["_mapanything_processor"] = processor

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
    """Attach the YOLO26 pose feature hook to decode SGIE tensor meta."""
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


def attach_analytics_telemetry_hook(
    pipeline: "DS8Pipeline",
    *,
    tracking_pub: "TrackingTelemetryPublisher",
    tracking_mode: Optional[str] = None,
    camera_labels: Optional[Mapping[int, str]] = None,
    sensor_id_map: Optional[Mapping[int, int]] = None,
    bev_renderer: Any | None = None,
    bev_calibration: Any | None = None,
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
        diagnostics_logger=diagnostics_logger,
    )
    analytics_component.config["_analytics_processor"] = processor

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

    # Attach telemetry as far downstream as possible so tensor meta is still valid.
    attach_component = analytics_component
    try:
        models_cfg = getattr(pipeline, "config", {}).get("models", {}) or {}
        pose_cfg = models_cfg.get("pose") or {}
        reid_cfg = models_cfg.get("reid") or {}
        if isinstance(pose_cfg, dict) and bool(pose_cfg.get("enable", True)):
            pose_name = str(pose_cfg.get("name") or "yolo26_pose").strip() or "yolo26_pose"
            candidate = pipeline.components.get(pose_name)
            if candidate is not None:
                attach_component = candidate
        if attach_component is analytics_component and isinstance(reid_cfg, dict) and bool(reid_cfg.get("enable", True)):
            reid_name = str(reid_cfg.get("name") or "reid_osnet").strip() or "reid_osnet"
            candidate = pipeline.components.get(reid_name)
            if candidate is not None:
                attach_component = candidate
    except Exception:
        attach_component = analytics_component

    if pipeline.ds_pipeline is None or BatchMetadataOperator is None or Probe is None:
        logger.debug("Stored analytics telemetry processor for lazy execution (pyservicemaker unavailable)")
        return

    try:
        probe = Probe("analytics_telemetry", _AnalyticsTelemetryOperator(processor))
        pipeline.ds_pipeline.attach(attach_component.name, probe)
        logger.info("Attached analytics telemetry probe to %s", attach_component.name)
    except Exception:  # pragma: no cover - depends on DS runtime availability
        logger.exception("Failed to attach analytics telemetry probe")


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

    def _apply(stage: str, cfg: Dict[str, Any]) -> None:
        with lock:
            pipeline.analytics_reload_count = getattr(pipeline, "analytics_reload_count", 0) + 1
            runtime_updates = component.config.setdefault("runtime_updates", {})
            runtime_updates[stage] = cfg
            pipeline.config.setdefault("analytics", {}).setdefault("stages", {})[stage] = cfg
            refreshed_polygons: Optional[Dict[int, List[Tuple[str, List[Tuple[float, float]]]]]] = None
            if stage == stage_name:
                refreshed_polygons = _refresh_exclusion_polygons(pipeline, stage_name=stage)
            if pipeline.ds_pipeline is None:
                logger.info(
                    "Recorded analytics runtime update for stage %s (reload_count=%s)",
                    stage,
                    pipeline.analytics_reload_count,
                )
                return
            try:
                node = pipeline.ds_pipeline[component.name]
                node.set({"config-file": component.config.get("config-file", "config/nvdsanalytics.yaml")})
                logger.info(
                    "Applied analytics runtime update for stage %s (reload_count=%s)",
                    stage,
                    pipeline.analytics_reload_count,
                )
            except Exception:  # pragma: no cover - depends on DS runtime availability
                logger.exception("Failed to push analytics runtime update for stage %s", stage)
            if stage == stage_name:
                exclude_component = pipeline.components.get("analytics_exclude")
                if exclude_component is not None:
                    try:
                        node_excl = pipeline.ds_pipeline[exclude_component.name]
                        cfg_path = exclude_component.config.get("config-file", "config/config_nvdsanalytics_exclude.ini")
                        node_excl.set({"config-file": cfg_path})
                        logger.info(
                            "Applied exclusion runtime update for stage %s via %s (reload_count=%s)",
                            stage,
                            exclude_component.name,
                            pipeline.analytics_reload_count,
                        )
                    except Exception:
                        logger.exception("Failed to push exclusion runtime update for stage %s", stage)
                if refreshed_polygons is not None:
                    logger.info(
                        "Exclusion polygons refreshed for stage %s (%d stream(s))",
                        stage,
                        len(refreshed_polygons),
                    )

    analytics_api.register_reload_hook(_apply)
    logger.info("Registered analytics reload bridge for stage '%s'", stage_name)


def attach_exclude_prune_hook(
    pipeline: "DS8Pipeline",
    *,
    stage_name: str = "exclude",
) -> None:
    """Prune objects fully contained within exclusion ROIs after nvdsanalytics."""
    component = pipeline.components.get("analytics")
    if component is None:
        raise KeyError("analytics component missing; cannot attach exclusion prune hook")

    polygons = _extract_exclusion_polygons(pipeline, stage_name=stage_name)
    if not polygons:
        logger.info("No exclusion polygons configured; skip exclusion prune hook")
        return

    processor = _ExcludePruneProcessor(polygons=polygons)
    component.config["_exclude_prune_processor"] = processor

    if pipeline.ds_pipeline is None or BatchMetadataOperator is None or Probe is None:
        logger.debug("Stored exclusion prune processor for lazy execution (pyservicemaker unavailable)")
        return

    try:
        probe = Probe("analytics_exclude_prune", _ExcludePruneOperator(processor))
        pipeline.ds_pipeline.attach(component.name, probe)
        logger.info("Attached exclusion prune probe to %s", component.name)
    except Exception:  # pragma: no cover - depends on DS runtime availability
        logger.exception("Failed to attach exclusion prune probe")


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


def _extract_exclusion_polygons(
    pipeline: "DS8Pipeline",
    *,
    stage_name: str = "exclude",
) -> Dict[int, List[Tuple[str, List[Tuple[float, float]]]]]:
    analytics_cfg = pipeline.config.get("analytics") or {}
    stages_cfg = analytics_cfg.get("stages") or {}
    try:
        # Prefer live analytics YAML (keeps REST updates in sync with running hooks)
        from noesis.server import analytics_api

        live_cfg = analytics_api._load_config(force=True)  # type: ignore[attr-defined]
        live_stages = (live_cfg.get("analytics") or {}).get("stages") or {}
        if live_stages:
            stages_cfg = live_stages
    except Exception:
        pass

    stage_cfg = stages_cfg.get(stage_name) or {}
    streams_cfg = stage_cfg.get("streams") or {}

    polygons: Dict[int, List[Tuple[str, List[Tuple[float, float]]]]] = {}
    for stream_id, stream_cfg in streams_cfg.items():
        try:
            stream_idx = int(stream_id)
        except Exception:
            logger.debug("Skipping non-integer analytics stream id %s", stream_id)
            continue
        roi_filtering = stream_cfg.get("roi_filtering") or {}
        if not roi_filtering.get("enable", False):
            continue
        rois = roi_filtering.get("rois") or []
        for roi in rois:
            roi_id = str(roi.get("id") or roi.get("label") or "").strip() or "roi"
            points_raw = roi.get("points_px") or []
            points: List[Tuple[float, float]] = []
            for point in points_raw:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                try:
                    px = float(point[0])
                    py = float(point[1])
                except Exception:
                    continue
                points.append((px, py))
            if len(points) >= 3:
                polygons.setdefault(stream_idx, []).append((roi_id, points))
    if polygons:
        return polygons

    # Fall back to loading the external nvdsanalytics YAML if specified.
    analytics_component = pipeline.components.get("analytics_exclude") or pipeline.components.get("analytics")
    if not analytics_component:
        return polygons
    config_file = analytics_component.config.get("config-file") if isinstance(analytics_component.config, dict) else None
    if not config_file:
        return polygons

    try:
        cfg_path = Path(config_file)
        candidates: List[Path] = []
        if cfg_path.is_absolute():
            candidates.append(cfg_path)
        else:
            candidates.append((pipeline.yaml_path.parent / cfg_path).resolve())
            candidates.append((pipeline.yaml_path.parent.parent / cfg_path).resolve())
            candidates.append(Path.cwd() / cfg_path)

        resolved = next((cand for cand in candidates if cand.exists()), None)
        if resolved is None:
            logger.debug("Analytics config file %s does not exist; skipping exclusion polygon extraction", cfg_path)
            return polygons
        cfg_path = resolved
        with cfg_path.open("r", encoding="utf-8") as stream:
            analytics_yaml = yaml.safe_load(stream) or {}
    except Exception:
        logger.exception("Failed to load analytics config from %s", config_file)
        return polygons

    yaml_analytics = analytics_yaml.get("analytics") or {}
    yaml_stages = yaml_analytics.get("stages") or {}
    yaml_stage = yaml_stages.get(stage_name) or {}
    yaml_streams = yaml_stage.get("streams") or {}

    for stream_id, stream_cfg in yaml_streams.items():
        try:
            stream_idx = int(stream_id)
        except Exception:
            continue
        roi_filtering = stream_cfg.get("roi_filtering") or {}
        if not roi_filtering.get("enable", False):
            continue
        rois = roi_filtering.get("rois") or []
        for roi in rois:
            roi_id = str(roi.get("id") or roi.get("label") or "").strip() or "roi"
            points_raw = roi.get("points_px") or []
            points: List[Tuple[float, float]] = []
            for point in points_raw:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                try:
                    px = float(point[0])
                    py = float(point[1])
                except Exception:
                    continue
                points.append((px, py))
            if len(points) >= 3:
                polygons.setdefault(stream_idx, []).append((roi_id, points))
    return polygons


def _refresh_exclusion_polygons(
    pipeline: "DS8Pipeline",
    stage_name: str = "exclude",
) -> Dict[int, List[Tuple[str, List[Tuple[float, float]]]]]:
    """Recompute exclusion polygons and update the stored processor if present."""
    polygons = _extract_exclusion_polygons(pipeline, stage_name=stage_name)
    component = pipeline.components.get("analytics")
    processor = None
    if component and isinstance(component.config, dict):
        processor = component.config.get("_exclude_prune_processor")
    if processor is not None:
        try:
            processor.polygons = polygons
        except Exception:
            logger.debug("Failed to refresh exclusion processor polygons")
    return polygons


@dataclass
class MapAnythingProcessor:
    pipeline: "DS8Pipeline"
    storage: DepthStorageManager
    depth_pub: "DepthTelemetryPublisher" | None
    gie_id: int
    camera_labels: Mapping[int, str] = field(default_factory=dict)
    tensor_samples: int = 0
    _async_enabled: bool = field(default=True, init=False, repr=False)
    _async_queue: "queue.Queue[_MapAnythingJob]" = field(default_factory=lambda: queue.Queue(maxsize=32), init=False, repr=False)
    _async_thread: Optional[threading.Thread] = field(default=None, init=False, repr=False)
    _async_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _dropped_jobs: int = field(default=0, init=False, repr=False)
    _last_drop_log: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        flag = os.environ.get("NOESIS_MAPANYTHING_POSTPROCESS_ASYNC", "1")
        self._async_enabled = str(flag).strip().lower() in ("1", "true", "yes", "on")

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
        try:
            if not self.pipeline.depth_enabled:
                logger.debug("Depth disabled; dropping MapAnything tensors before DS8 conversion")
                return None
            # DS8 API: tensor_meta.get_layers() returns dict[str, Tensor]
            layers = tensor_meta.get_layers() or {}
            if not layers:
                return None

            if not self._async_enabled:
                tensors: Dict[str, np.ndarray] = {}
                layer_names: List[str] = []
                for name, tensor in layers.items():
                    key = str(name)
                    layer_names.append(key)
                    arr = self._to_numpy(tensor)
                    if arr is not None:
                        tensors[key] = arr
                    else:
                        logger.debug(
                            "Failed to convert tensor '%s' to numpy (device=%s, dtype=%s, shape=%s)",
                            key,
                            getattr(tensor, "device_type", None),
                            getattr(tensor, "dtype", None),
                            getattr(tensor, "shape", None),
                        )
                if not tensors:
                    logger.debug(
                        "MapAnything tensor meta had no convertible layers (unique_id=%s)",
                        getattr(tensor_meta, "unique_id", None),
                    )
                    return None
                self.tensor_samples += 1
                if self.tensor_samples <= 5 or (self.tensor_samples % 50) == 0:
                    logger.debug(
                        "MapAnything tensors received (unique_id=%s): %s",
                        getattr(tensor_meta, "unique_id", None),
                        ", ".join(layer_names) or "none",
                    )
                return self._emit_from_tensors(frame_meta, tensors)

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

            self._start_async_worker()
            try:
                self._async_queue.put_nowait(job)
            except queue.Full:
                self._dropped_jobs += 1
                now = time.time()
                if now - self._last_drop_log >= 2.0:
                    self._last_drop_log = now
                    logger.debug(
                        "MapAnything postprocess queue full; dropped %d jobs",
                        self._dropped_jobs,
                    )
            return None
        except Exception:
            logger.exception("Failed to extract tensor layers from DS8 TensorOutputUserMetadata")
            return None

    def _start_async_worker(self) -> None:
        if self._async_thread is not None and self._async_thread.is_alive():
            return
        with self._async_lock:
            if self._async_thread is not None and self._async_thread.is_alive():
                return

            def _loop() -> None:
                while True:
                    try:
                        job = self._async_queue.get()
                    except Exception:
                        time.sleep(0.01)
                        continue
                    try:
                        if not self.pipeline.depth_enabled:
                            continue
                        tensors: Dict[str, np.ndarray] = {}

                        def _slice_batch(arr: np.ndarray, batch_id: int | None) -> np.ndarray:
                            if batch_id is None:
                                return arr
                            if arr.ndim == 4:
                                b = int(arr.shape[0] or 0)
                                if b > 0:
                                    idx = batch_id if 0 <= batch_id < b else 0
                                    return arr[idx]
                            if arr.ndim == 3 and arr.shape[0] > 1:
                                b = int(arr.shape[0] or 0)
                                idx = batch_id if 0 <= batch_id < b else 0
                                return arr[idx]
                            return arr

                        depth_arr = self._to_numpy(job.depth)
                        if depth_arr is None:
                            continue
                        tensors["depth"] = _slice_batch(depth_arr, job.batch_id)
                        if job.confidence is not None:
                            conf_arr = self._to_numpy(job.confidence)
                            if conf_arr is not None:
                                tensors["confidence"] = _slice_batch(conf_arr, job.batch_id)
                        if job.mask is not None:
                            mask_arr = self._to_numpy(job.mask)
                            if mask_arr is not None:
                                tensors["mask"] = _slice_batch(mask_arr, job.batch_id)
                        self.handle_numpy_arrays(
                            source_id=int(job.source_id),
                            frame_id=int(job.frame_id),
                            pts_ns=int(job.pts_ns),
                            tensors=tensors,
                        )
                    except Exception:
                        logger.exception("Async MapAnything postprocess failed")
                    finally:
                        try:
                            self._async_queue.task_done()
                        except Exception:
                            pass

            self._async_thread = threading.Thread(
                target=_loop,
                name="MapAnythingPostprocess",
                daemon=True,
            )
            self._async_thread.start()

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
    ) -> Optional[DepthResult]:
        frame_meta = {
            "pad_index": source_id,
            "frame_num": frame_id,
            "buf_pts": pts_ns,
        }
        self.tensor_samples += 1
        return self._emit_from_tensors(frame_meta, tensors)

    def _emit_from_tensors(self, frame_meta: Any, tensors: Mapping[str, np.ndarray]) -> Optional[DepthResult]:
        if not self.pipeline.depth_enabled:
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
        mask = np.logical_and(np.asarray(mask, dtype=bool, copy=False), np.isfinite(depth))

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

        depth = np.asarray(depth, dtype=np.float32, copy=False)
        if confidence is not None:
            confidence = np.asarray(confidence, dtype=np.float32, copy=False)
            if confidence.shape != depth.shape:
                logger.debug("Confidence tensor shape %s does not match aligned depth %s", confidence.shape, depth.shape)
                confidence = None
        mask = np.asarray(mask, dtype=bool, copy=False) if mask is not None else np.ones_like(depth, dtype=bool)
        mask = np.logical_and(mask, np.isfinite(depth))

        conf_array = confidence.astype(np.float32, copy=False) if confidence is not None else np.zeros_like(depth, dtype=np.float32)

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
            if depth.size > 0:
                depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                mask = np.ones_like(depth, dtype=bool)
                if confidence is None:
                    conf_array = np.zeros_like(depth, dtype=np.float32)
                valid = depth[mask]
                logger.debug("Depth all-NaN/invalid; filled zeros to emit frame")
            else:
                logger.debug("No valid depth pixels after masking; skipping frame")
                return None

        source_id = int(_meta_lookup(frame_meta, "pad_index", "source_id", default=0))
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

        store_enabled_env = os.environ.get("NOESIS_DEPTH_STORE_ENABLED", "1")
        store_enabled = str(store_enabled_env).strip().lower() in ("1", "true", "yes", "on")
        if store_enabled:
            dest_path = self.storage.store(camera_id, ts_us, depth, conf_array, mask_u8)
            depth_ref = str(dest_path)
        else:
            depth_ref = f"memory://depth/{camera_id}/{ts_us}"
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
    window_s: float = 8.0
    draw_stride: int = 2
    min_step_px: float = 2.0
    min_dt_s: float = 0.08
    smooth_tau_s: float = 0.25
    max_speed_px_per_s: float = 600.0
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
        object.__setattr__(self, "max_points_per_track", max(2, int(self.max_points_per_track)))
        object.__setattr__(self, "max_segments_per_track", max(1, int(self.max_segments_per_track)))
        object.__setattr__(self, "max_tracks", max(1, int(self.max_tracks)))
        object.__setattr__(self, "max_lines", max(1, int(self.max_lines)))
        object.__setattr__(self, "max_display_metas", max(1, int(self.max_display_metas)))
        object.__setattr__(self, "line_width", max(1, int(self.line_width)))
        object.__setattr__(self, "min_alpha", float(min(1.0, max(0.0, float(self.min_alpha)))))
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
            window_s=_float(cfg.get("window_s"), 8.0),
            draw_stride=_int(cfg.get("draw_stride"), 2),
            min_step_px=_float(cfg.get("min_step_px"), 2.0),
            min_dt_s=_float(cfg.get("min_dt_s"), 0.08),
            smooth_tau_s=_float(cfg.get("smooth_tau_s"), 0.25),
            max_speed_px_per_s=_float(cfg.get("max_speed_px_per_s"), 600.0),
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
class _TrailTrackState:
    points: "deque[Tuple[float, float, float]]" = field(default_factory=deque)
    last_seen_ts: float = 0.0
    ema_x: Optional[float] = None
    ema_y: Optional[float] = None
    ema_ts: float = 0.0


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
        min_step_px = float(self.config.min_step_px)
        min_dt_s = float(self.config.min_dt_s)
        max_points_per_track = max(2, int(self.config.max_points_per_track))
        tracks_seen = 0

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
            while state.points and (now - float(state.points[0][0])) > float(self.config.window_s):
                state.points.popleft()

            if not do_sample:
                continue

            # Clamp spurious jumps based on max speed.
            if state.points:
                prev_ts, prev_x, prev_y = state.points[-1]
                dt = max(0.0, float(now) - float(prev_ts))
                if dt > 0.0 and self.config.max_speed_px_per_s > 0.0:
                    dx = x - float(prev_x)
                    dy = y - float(prev_y)
                    dist = math.hypot(dx, dy)
                    max_step = float(self.config.max_speed_px_per_s) * dt
                    if max_step > 0.0 and dist > max_step:
                        scale = max_step / dist
                        x = float(prev_x) + dx * scale
                        y = float(prev_y) + dy * scale

            # Smooth the footpoint directly (time-constant based EMA).
            if self.config.smooth_tau_s > 0.0:
                if state.ema_x is None or state.ema_y is None:
                    state.ema_x, state.ema_y = x, y
                    state.ema_ts = float(now)
                else:
                    dt_ema = max(0.0, float(now) - float(state.ema_ts))
                    tau = float(self.config.smooth_tau_s)
                    alpha = 1.0 - math.exp(-dt_ema / tau) if (tau > 0.0 and dt_ema > 0.0) else 1.0
                    state.ema_x = float(state.ema_x + alpha * (x - float(state.ema_x)))
                    state.ema_y = float(state.ema_y + alpha * (y - float(state.ema_y)))
                    state.ema_ts = float(now)
                x, y = float(state.ema_x), float(state.ema_y)

            # Decimation: enforce a time-window based sampling budget by updating the
            # most recent point when frames arrive faster than `min_dt_s`.
            if state.points:
                prev_ts, prev_x, prev_y = state.points[-1]
                dt = max(0.0, float(now) - float(prev_ts))
                dist = math.hypot(x - float(prev_x), y - float(prev_y))
                if dt < min_dt_s:
                    if dist >= min_step_px:
                        # Update the most recent point in-place without advancing its
                        # timestamp so `dt` can accumulate until the next commit.
                        state.points[-1] = (float(prev_ts), float(x), float(y))
                    continue
                if dist < min_step_px:
                    continue

            state.points.append((float(now), float(x), float(y)))

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
            while state.points and (now - float(state.points[0][0])) > float(self.config.window_s):
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

        def _resample_points(points: List[Tuple[float, float, float]], segments_budget: int, *, bias: float = 2.0) -> List[Tuple[float, float, float]]:
            if segments_budget <= 0:
                return []
            segments_available = len(points) - 1
            if segments_available <= segments_budget:
                return points
            # Select (segments_budget + 1) indices, biased towards the newest points.
            selected: List[Tuple[float, float, float]] = []
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
        track_items.sort(key=lambda item: float(getattr(item[1], "last_seen_ts", 0.0)), reverse=True)
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

                ts0, x1, y1 = pts[idx]
                _ts1, x2, y2 = pts[idx + 1]
                age = max(0.0, float(now) - float(ts0))
                t = 1.0 - min(1.0, age * inv_window)
                alpha = min_alpha + (1.0 - min_alpha) * max(0.0, min(1.0, t))

                line.x1, line.y1 = int(x1), int(y1)
                line.x2, line.y2 = int(x2), int(y2)
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
            _ts, last_x, last_y = pts[-1]
            text = ds_osd.Text()
            if stable_id_int is not None and stable_id_int > 0:
                text.display_text = f"sid {stable_id_int}"
            else:
                text.display_text = "sid XX"
            text.x_offset = int(last_x)
            text.y_offset = int(last_y)
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


def _dlpack_tensor_to_numpy(layer_tensor: Any) -> Optional[np.ndarray]:
    """Convert a Service Maker Tensor (dlpack) into a numpy array.

    Prefer torch's DLPack bridge when available to ensure the producer's
    deleter runs and GPU ownership is released safely.
    """
    if layer_tensor is None:
        return None
    dlpack_fn = getattr(layer_tensor, "__dlpack__", None)
    if not callable(dlpack_fn):
        return None

    use_torch = os.environ.get("NOESIS_POSE_DLPACK_TORCH", "1")
    fallback_reason = "torch_disabled"
    if str(use_torch).strip().lower() in ("1", "true", "yes", "on"):
        fallback_reason = "torch_dlpack_failed"
        try:
            import torch
            import torch.utils.dlpack as torch_dlpack

            start_ns = time.perf_counter_ns()
            stream = 0
            try:
                if torch.cuda.is_available():
                    stream = int(torch.cuda.current_stream().cuda_stream)
            except Exception:
                stream = 0
            capsule = dlpack_fn(stream)
            torch_tensor = torch_dlpack.from_dlpack(capsule)
            arr = torch_tensor.detach().cpu().numpy()
            _CORE_PATH_INSTRUMENTATION.record_cpu_copy_violation(
                location="pose.tensor_dlpack_to_numpy",
                reason="torch_dlpack_to_host",
                details={
                    "duration_ns": time.perf_counter_ns() - start_ns,
                    "shape": tuple(int(x) for x in getattr(arr, "shape", ())),
                },
            )
            return arr
        except Exception as exc:
            global _POSE_DLPACK_TORCH_LOGGED
            if not _POSE_DLPACK_TORCH_LOGGED:
                logger.debug(
                    "Pose torch DLPack conversion failed (device=%s, dtype=%s, shape=%s): %s",
                    getattr(layer_tensor, "device_type", None),
                    getattr(layer_tensor, "dtype", None),
                    getattr(layer_tensor, "shape", None),
                    exc,
                )
                _POSE_DLPACK_TORCH_LOGGED = True
            fallback_reason = f"torch_dlpack_failed:{type(exc).__name__}"
    policy = _CORE_PATH_INSTRUMENTATION.record_cpu_copy_violation(
        location="pose.tensor_dlpack_to_numpy.fallback",
        reason=fallback_reason,
        fallback=True,
        details={
            "device": getattr(layer_tensor, "device_type", None),
            "dtype": getattr(layer_tensor, "dtype", None),
            "shape": getattr(layer_tensor, "shape", None),
        },
    )
    if policy == _CORE_FALLBACK_POLICY_GATE:
        logger.warning("Blocked pose fallback tensor conversion due to NOESIS_CORE_PATH_FALLBACK_POLICY=gate")
        return None
    if policy == _CORE_FALLBACK_POLICY_FAIL:
        raise _CorePathFallbackConversionError(
            "Pose fallback tensor conversion blocked by NOESIS_CORE_PATH_FALLBACK_POLICY=fail"
        )
    try:
        import ctypes
        import ctypes.util

        def _get_capsule() -> Any:
            try:
                return dlpack_fn(None)
            except Exception:
                return dlpack_fn(0)

        class _DLDevice(ctypes.Structure):
            _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]

        class _DLDataType(ctypes.Structure):
            _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]

        class _DLTensor(ctypes.Structure):
            _fields_ = [
                ("data", ctypes.c_void_p),
                ("device", _DLDevice),
                ("ndim", ctypes.c_int),
                ("dtype", _DLDataType),
                ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)),
                ("byte_offset", ctypes.c_uint64),
            ]

        class _DLManagedTensor(ctypes.Structure):
            _fields_ = [("dl_tensor", _DLTensor), ("manager_ctx", ctypes.c_void_p), ("deleter", ctypes.c_void_p)]

        class _DLPackVersion(ctypes.Structure):
            _fields_ = [("major", ctypes.c_int32), ("minor", ctypes.c_int32)]

        class _DLManagedTensorVersioned(ctypes.Structure):
            _fields_ = [
                ("version", _DLPackVersion),
                ("dl_tensor", _DLTensor),
                ("manager_ctx", ctypes.c_void_p),
                ("deleter", ctypes.c_void_p),
            ]

        dlpack_capsule = _get_capsule()
        raw_name = None
        try:
            get_name = ctypes.pythonapi.PyCapsule_GetName
            get_name.restype = ctypes.c_char_p
            get_name.argtypes = [ctypes.py_object]
            raw_name = get_name(dlpack_capsule)
        except Exception:
            raw_name = None

        get_ptr = ctypes.pythonapi.PyCapsule_GetPointer
        get_ptr.restype = ctypes.c_void_p
        get_ptr.argtypes = [ctypes.py_object, ctypes.c_char_p]
        managed_ptr = get_ptr(dlpack_capsule, raw_name)
        if not managed_ptr:
            return None

        dl = None
        deleter_ptr = None
        try:
            managed_v = ctypes.cast(managed_ptr, ctypes.POINTER(_DLManagedTensorVersioned))
            ver = managed_v.contents.version
            dl_candidate = managed_v.contents.dl_tensor
            ndim_candidate = int(dl_candidate.ndim)
            dtype_bits_candidate = int(dl_candidate.dtype.bits)
            dtype_code_candidate = int(dl_candidate.dtype.code)
            dev_type_candidate = int(dl_candidate.device.device_type)
            plausible = (
                0 <= int(ver.major) <= 10
                and 0 <= int(ver.minor) <= 10
                and 1 <= ndim_candidate <= 8
                and dtype_bits_candidate in (8, 16, 32, 64)
                and 0 <= dtype_code_candidate <= 8
                and 1 <= dev_type_candidate <= 32
                and int(dl_candidate.data or 0) != 0
            )
            if plausible:
                dl = dl_candidate
                deleter_ptr = managed_v.contents.deleter
        except Exception:
            dl = None
            deleter_ptr = None

        if dl is None:
            managed = ctypes.cast(managed_ptr, ctypes.POINTER(_DLManagedTensor))
            dl = managed.contents.dl_tensor
            deleter_ptr = managed.contents.deleter

        ndim = int(dl.ndim)
        if ndim < 1:
            return None
        shape = [int(dl.shape[i]) for i in range(ndim)]
        total = 1
        for dim in shape:
            total *= max(1, int(dim))
        dtype_bits = int(dl.dtype.bits)
        dtype_code = int(dl.dtype.code)
        dtype_lanes = int(dl.dtype.lanes)
        if dtype_code != 2 or dtype_bits != 32 or dtype_lanes != 1:
            return None
        nbytes = int(total * (dtype_bits // 8) * dtype_lanes)
        if nbytes <= 0:
            return None

        out = np.empty((total,), dtype=np.float32)
        start_ns = time.perf_counter_ns()
        dst_ptr = ctypes.c_void_p(int(out.ctypes.data))
        src_ptr = ctypes.c_void_p(int(dl.data) + int(dl.byte_offset))
        dev_type = int(dl.device.device_type)

        if dev_type in (1, 3):  # kDLCPU / kDLCUDAHost
            ctypes.memmove(dst_ptr, src_ptr, nbytes)
        elif dev_type in (2, 13):  # kDLCUDA / kDLCUDAManaged
            cudart_path = ctypes.util.find_library("cudart")
            if not cudart_path:
                return None
            cudart = ctypes.CDLL(cudart_path)
            cuda_memcpy = cudart.cudaMemcpy
            cuda_memcpy.restype = ctypes.c_int
            cuda_memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            err = int(cuda_memcpy(dst_ptr, src_ptr, ctypes.c_size_t(nbytes), ctypes.c_int(2)))
            if err != 0:
                return None
        else:
            return None

        # Optional deleter call (disabled by default; matches reid handling).
        call_deleter = str(os.environ.get("NOESIS_POSE_DLPACK_CALL_DELETER", "")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if call_deleter and deleter_ptr:
            deleter = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(deleter_ptr)
            deleter(managed_ptr)

        _CORE_PATH_INSTRUMENTATION.record_cpu_copy_violation(
            location="pose.tensor_dlpack_to_numpy.fallback_copy",
            reason="ctypes_host_copy",
            details={
                "duration_ns": time.perf_counter_ns() - start_ns,
                "bytes": int(nbytes),
                "device_type": int(dev_type),
            },
        )
        return out.reshape(shape).astype(np.float32, copy=False)
    except Exception as exc:
        if isinstance(exc, _CorePathFallbackConversionError):
            raise
        logger.debug("Pose DLPack decode failed: %s", exc, exc_info=True)
        return None

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
    _logged_layers: bool = field(default=False, init=False, repr=False)
    _logged_shape: bool = field(default=False, init=False, repr=False)
    _missing_tensor_logged: bool = field(default=False, init=False, repr=False)
    _missing_native_logged: bool = field(default=False, init=False, repr=False)
    _debug_last_log: float = field(default=0.0, init=False, repr=False)
    _debug_frames: int = field(default=0, init=False, repr=False)
    _debug_objects: int = field(default=0, init=False, repr=False)
    _debug_attached: int = field(default=0, init=False, repr=False)
    _debug_missing: int = field(default=0, init=False, repr=False)

    def _to_numpy(self, tensor: Any) -> Optional[np.ndarray]:
        with _DLPACK_HOST_READ_LOCK:
            arr = _dlpack_tensor_to_numpy(tensor)
        if arr is None:
            logger.debug(
                "Pose tensor conversion failed (device=%s, dtype=%s, shape=%s)",
                getattr(tensor, "device_type", None),
                getattr(tensor, "dtype", None),
                getattr(tensor, "shape", None),
            )
        return arr

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

    def _extract_pose_output(self, obj_meta: Any) -> Optional[np.ndarray]:
        tensor_items_iter = getattr(obj_meta, "tensor_items", None)
        if tensor_items_iter is None:
            return None
        try:
            tensor_items = list(tensor_items_iter)
        except Exception:
            tensor_items = tensor_items_iter or []
        for item in tensor_items:
            try:
                tensor_output = item.as_tensor_output()
            except Exception:
                continue
            try:
                if int(getattr(tensor_output, "unique_id", -1)) != int(self.gie_id):
                    continue
            except Exception:
                continue
            try:
                layers = tensor_output.get_layers() or {}
            except Exception:
                continue
            if not layers:
                continue
            if not self._logged_layers:
                self._logged_layers = True
                logger.info("YOLO26 pose tensor layers: %s", list(layers.keys()))
            tensor = layers.get("output0")
            if tensor is None and len(layers) == 1:
                try:
                    tensor = next(iter(layers.values()))
                except Exception:
                    tensor = None
            if tensor is None:
                continue
            tensor_for_read = _clone_tensor_for_host_read(tensor, location="pose.feature.output")
            if tensor_for_read is None:
                continue
            arr = self._to_numpy(tensor_for_read)
            if arr is not None:
                return arr
        return None

    def _select_pose_row(self, output: np.ndarray) -> Optional[np.ndarray]:
        if output.ndim >= 3:
            try:
                output = output.reshape(-1, output.shape[-1])
            except Exception:
                output = output[0]
        if output.ndim != 2 or output.shape[1] < 6 + 17 * 3:
            return None
        scores = output[:, 4]
        if scores.size == 0:
            return None
        idx = int(np.argmax(scores))
        score = float(scores[idx])
        if score < float(self.score_threshold):
            return None
        if not self._logged_shape:
            self._logged_shape = True
            logger.info("YOLO26 pose output shape: %s", output.shape)
        return output[idx]

    def _letterbox_params(self, roi_w: float, roi_h: float) -> Tuple[float, float, float]:
        if roi_w <= 0 or roi_h <= 0:
            return 1.0, 0.0, 0.0
        model_w, model_h = float(self.model_size[0]), float(self.model_size[1])
        gain = min(model_w / roi_w, model_h / roi_h)
        new_w = roi_w * gain
        new_h = roi_h * gain
        pad_x = (model_w - new_w) / 2.0
        pad_y = (model_h - new_h) / 2.0
        return gain, pad_x, pad_y

    def _map_keypoints(
        self,
        kpts: np.ndarray,
        *,
        roi_w: float,
        roi_h: float,
        normalized: bool,
    ) -> np.ndarray:
        model_w, model_h = float(self.model_size[0]), float(self.model_size[1])
        x = kpts[:, 0].astype(np.float32, copy=False)
        y = kpts[:, 1].astype(np.float32, copy=False)
        c = kpts[:, 2].astype(np.float32, copy=False)
        if normalized:
            x = x * model_w
            y = y * model_h
        if self.letterbox:
            gain, pad_x, pad_y = self._letterbox_params(float(roi_w), float(roi_h))
            if gain > 0:
                x = (x - pad_x) / gain
                y = (y - pad_y) / gain
        else:
            if model_w > 0:
                x = x * (float(roi_w) / model_w)
            if model_h > 0:
                y = y * (float(roi_h) / model_h)
        if roi_w > 0:
            x = np.clip(x, 0.0, float(roi_w))
        if roi_h > 0:
            y = np.clip(y, 0.0, float(roi_h))
        return np.stack([x, y, c], axis=1)

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

            output = self._extract_pose_output(obj_meta)
            if output is None:
                if not self._missing_tensor_logged:
                    logger.debug("Pose SGIE missing tensor meta (unique_id=%s)", int(self.gie_id))
                    self._missing_tensor_logged = True
                if debug:
                    self._debug_missing += 1
                continue

            row = self._select_pose_row(output)
            if row is None:
                if debug:
                    self._debug_missing += 1
                continue

            try:
                score = float(row[4])
            except Exception:
                score = 0.0

            kpts_raw = np.asarray(row[6:], dtype=np.float32)
            if kpts_raw.size < 17 * 3:
                if debug:
                    self._debug_missing += 1
                continue
            kpts = kpts_raw[: 17 * 3].reshape(17, 3)
            normalized = float(np.max(row[:4])) <= 2.0

            roi_w = float(bbox[2])
            roi_h = float(bbox[3])
            kpts = self._map_keypoints(kpts, roi_w=roi_w, roi_h=roi_h, normalized=normalized)

            features, quality = self._compute_features(kpts, roi_w=roi_w, roi_h=roi_h)
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
    _logged_layers: bool = field(default=False, init=False, repr=False)
    _logged_shape: bool = field(default=False, init=False, repr=False)
    _missing_tensor_logged: bool = field(default=False, init=False, repr=False)
    _debug_last_log: float = field(default=0.0, init=False, repr=False)
    _debug_frames: int = field(default=0, init=False, repr=False)
    _debug_objects: int = field(default=0, init=False, repr=False)
    _debug_drawn: int = field(default=0, init=False, repr=False)
    _debug_missing: int = field(default=0, init=False, repr=False)

    def _to_numpy(self, tensor: Any) -> Optional[np.ndarray]:
        with _DLPACK_HOST_READ_LOCK:
            arr = _dlpack_tensor_to_numpy(tensor)
        if arr is None:
            logger.debug(
                "Pose keypoint tensor conversion failed (device=%s, dtype=%s, shape=%s)",
                getattr(tensor, "device_type", None),
                getattr(tensor, "dtype", None),
                getattr(tensor, "shape", None),
            )
        return arr

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

    def _extract_pose_output(self, obj_meta: Any) -> Optional[np.ndarray]:
        tensor_items_iter = getattr(obj_meta, "tensor_items", None)
        if tensor_items_iter is None:
            return None
        try:
            tensor_items = list(tensor_items_iter)
        except Exception:
            tensor_items = tensor_items_iter or []
        for item in tensor_items:
            try:
                tensor_output = item.as_tensor_output()
            except Exception:
                continue
            try:
                if int(getattr(tensor_output, "unique_id", -1)) != int(self.gie_id):
                    continue
            except Exception:
                continue
            try:
                layers = tensor_output.get_layers() or {}
            except Exception:
                continue
            if not layers:
                continue
            if not self._logged_layers:
                self._logged_layers = True
                logger.info("YOLO26 pose tensor layers: %s", list(layers.keys()))
            tensor = layers.get("output0")
            if tensor is None and len(layers) == 1:
                try:
                    tensor = next(iter(layers.values()))
                except Exception:
                    tensor = None
            if tensor is None:
                continue
            tensor_for_read = _clone_tensor_for_host_read(tensor, location="pose.overlay.output")
            if tensor_for_read is None:
                continue
            arr = self._to_numpy(tensor_for_read)
            if arr is not None:
                return arr
        return None

    def _select_pose_row(self, output: np.ndarray) -> Optional[np.ndarray]:
        if output.ndim >= 3:
            try:
                output = output.reshape(-1, output.shape[-1])
            except Exception:
                output = output[0]
        if output.ndim != 2 or output.shape[1] < 6 + 17 * 3:
            return None
        scores = output[:, 4]
        if scores.size == 0:
            return None
        idx = int(np.argmax(scores))
        score = float(scores[idx])
        if score < float(self.score_threshold):
            return None
        if not self._logged_shape:
            self._logged_shape = True
            logger.info("YOLO26 pose output shape: %s", output.shape)
        return output[idx]

    def _letterbox_params(self, roi_w: float, roi_h: float) -> Tuple[float, float, float]:
        if roi_w <= 0 or roi_h <= 0:
            return 1.0, 0.0, 0.0
        model_w, model_h = float(self.model_size[0]), float(self.model_size[1])
        gain = min(model_w / roi_w, model_h / roi_h)
        new_w = roi_w * gain
        new_h = roi_h * gain
        pad_x = (model_w - new_w) / 2.0
        pad_y = (model_h - new_h) / 2.0
        return gain, pad_x, pad_y

    def _map_keypoints(
        self,
        kpts: np.ndarray,
        *,
        roi_w: float,
        roi_h: float,
        normalized: bool,
    ) -> np.ndarray:
        model_w, model_h = float(self.model_size[0]), float(self.model_size[1])
        x = kpts[:, 0].astype(np.float32, copy=False)
        y = kpts[:, 1].astype(np.float32, copy=False)
        c = kpts[:, 2].astype(np.float32, copy=False)
        if normalized:
            x = x * model_w
            y = y * model_h
        if self.letterbox:
            gain, pad_x, pad_y = self._letterbox_params(float(roi_w), float(roi_h))
            if gain > 0:
                x = (x - pad_x) / gain
                y = (y - pad_y) / gain
        else:
            if model_w > 0:
                x = x * (float(roi_w) / model_w)
            if model_h > 0:
                y = y * (float(roi_h) / model_h)
        if roi_w > 0:
            x = np.clip(x, 0.0, float(roi_w))
        if roi_h > 0:
            y = np.clip(y, 0.0, float(roi_h))
        return np.stack([x, y, c], axis=1)

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

                output = self._extract_pose_output(obj_meta)
                if output is None:
                    if not self._missing_tensor_logged:
                        logger.debug("Pose SGIE missing tensor meta (unique_id=%s)", int(self.gie_id))
                        self._missing_tensor_logged = True
                    if debug:
                        self._debug_missing += 1
                    continue
                row = self._select_pose_row(output)
                if row is None:
                    if debug:
                        self._debug_missing += 1
                    continue

                kpts_raw = np.asarray(row[6:], dtype=np.float32)
                if kpts_raw.size < 17 * 3:
                    if debug:
                        self._debug_missing += 1
                    continue
                kpts = kpts_raw[: 17 * 3].reshape(17, 3)
                normalized = float(np.max(row[:4])) <= 2.0

                roi_w = float(bbox[2])
                roi_h = float(bbox[3])
                if roi_w <= 0 or roi_h <= 0:
                    continue
                kpts = self._map_keypoints(kpts, roi_w=roi_w, roi_h=roi_h, normalized=normalized)
                kpts[:, 0] += float(bbox[0])
                kpts[:, 1] += float(bbox[1])

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
class _AnalyticsTelemetryProcessor:
    pipeline: "DS8Pipeline"
    tracking_pub: "TrackingTelemetryPublisher"
    camera_labels: Mapping[int, str]
    sensor_id_map: Mapping[int, int]
    tracking_mode: Optional[str] = None
    bev_renderer: Any = None
    bev_calibration: Any = None
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
    _fallback_sid_by_key: Dict[Tuple[int, int], int] = field(default_factory=dict, init=False, repr=False)
    _fallback_sid_last_seen: Dict[Tuple[int, int], float] = field(default_factory=dict, init=False, repr=False)
    _fallback_sid_next: int = field(default=1, init=False, repr=False)
    _bev_class_ids: frozenset[int] = field(default_factory=lambda: frozenset({0}), init=False, repr=False)
    _bev_class_ids_ready: bool = field(default=False, init=False, repr=False)
    _reid_unique_id: int = field(default=3, init=False, repr=False)
    _reid_layer_name: str = field(default="features", init=False, repr=False)
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
    _world_frame: str = field(default="camera_local", init=False, repr=False)
    _v3dt_meta_enabled: bool = field(default=True, init=False, repr=False)
    _v3dt_meta_logged_missing: bool = field(default=False, init=False, repr=False)
    _v3dt_caminfo_paths: Dict[int, Path] = field(default_factory=dict, init=False, repr=False)
    _v3dt_caminfo_cache: Dict[int, Tuple[str, List[List[float]]]] = field(default_factory=dict, init=False, repr=False)
    _v3dt_caminfo_logged_missing: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        # Discover the ReID SGIE unique-id from the built pipeline config when present.
        try:
            models_cfg = getattr(self.pipeline, "config", {}).get("models", {}) or {}
            reid_cfg = models_cfg.get("reid") or {}
            if isinstance(reid_cfg, dict):
                gie_id = reid_cfg.get("gie_id", reid_cfg.get("gie-id", None))
                if gie_id is not None:
                    self._reid_unique_id = int(gie_id)
        except Exception:
            self._reid_unique_id = 3

        self._tracking_mode = self._resolve_tracking_mode(self.tracking_mode)

        try:
            v3dt_cfg = getattr(self.pipeline, "config", {}).get("v3dt", {}) or {}
            frame = v3dt_cfg.get("world_frame") if isinstance(v3dt_cfg, dict) else None
            if frame:
                self._world_frame = str(frame)
        except Exception:
            self._world_frame = "camera_local"

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
        z_base = z - 0.5 * z_len
        uv = self._project_point(P, (x, y, z_base))
        if uv is None:
            return None
        u, v = uv
        if key == "projectionMatrix_3x4":
            frame_w, frame_h = frame_dims
            if frame_w and frame_h:
                u += float(frame_w) * 0.5
                v += float(frame_h) * 0.5
        return [float(u), float(v)]

    @staticmethod
    def _tensor_to_embedding(layer_tensor: Any) -> Optional[np.ndarray]:
        """Convert a Service Maker Tensor (dlpack) into a 1D float32 embedding."""
        if layer_tensor is None:
            return None

        flag = str(os.environ.get("NOESIS_REID_NO_DLPACK", "")).strip().lower()
        if flag in ("1", "true", "yes", "on"):
            try:
                shape = getattr(layer_tensor, "shape", None)
                if shape is not None and hasattr(shape, "__len__") and len(shape) >= 1 and int(shape[-1]) == 512:
                    emb = np.zeros((512,), dtype=np.float32)
                    emb[0] = 1.0
                    return emb
            except Exception:
                return None
        dlpack_debug_enabled = str(os.environ.get("NOESIS_REID_DLPACK_DEBUG", "")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        fallback_reason = "torch_disabled"
        use_torch = os.environ.get("NOESIS_REID_DLPACK_TORCH", "1")
        if str(use_torch).strip().lower() in ("1", "true", "yes", "on"):
            fallback_reason = "torch_dlpack_failed"
            try:
                import torch
                import torch.utils.dlpack as torch_dlpack

                start_ns = time.perf_counter_ns()
                stream = 0
                try:
                    if torch.cuda.is_available():
                        stream = int(torch.cuda.current_stream().cuda_stream)
                except Exception:
                    stream = 0
                dlpack_capsule = layer_tensor.__dlpack__(stream)
                torch_tensor = torch_dlpack.from_dlpack(dlpack_capsule)
                emb = torch_tensor.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
                _CORE_PATH_INSTRUMENTATION.record_cpu_copy_violation(
                    location="reid.tensor_dlpack_to_embedding",
                    reason="torch_dlpack_to_host",
                    details={
                        "duration_ns": time.perf_counter_ns() - start_ns,
                        "length": int(emb.size),
                    },
                )
                if emb.size < 1:
                    return None
                n = float(np.linalg.norm(emb) + 1e-12)
                return (emb / n).astype(np.float32, copy=False)
            except Exception as exc:
                fallback_reason = f"torch_dlpack_failed:{type(exc).__name__}"
                if dlpack_debug_enabled:
                    logger.debug("ReID torch DLPack decode failed: %s", exc, exc_info=True)
        policy = _CORE_PATH_INSTRUMENTATION.record_cpu_copy_violation(
            location="reid.tensor_dlpack_to_embedding.fallback",
            reason=fallback_reason,
            fallback=True,
            details={
                "device": getattr(layer_tensor, "device_type", None),
                "dtype": getattr(layer_tensor, "dtype", None),
                "shape": getattr(layer_tensor, "shape", None),
            },
        )
        if policy == _CORE_FALLBACK_POLICY_GATE:
            logger.warning("Blocked ReID fallback embedding conversion due to NOESIS_CORE_PATH_FALLBACK_POLICY=gate")
            return None
        if policy == _CORE_FALLBACK_POLICY_FAIL:
            raise _CorePathFallbackConversionError(
                "ReID fallback embedding conversion blocked by NOESIS_CORE_PATH_FALLBACK_POLICY=fail"
            )
        try:
            import ctypes
            import ctypes.util

            capsule_name = None
            used_versioned = False
            ver_major = None
            ver_minor = None
            ndim = None
            shape = None
            dtype_bits = None
            dtype_code = None
            dtype_lanes = None
            dev_type = None

            def _fail(reason: str) -> Optional[np.ndarray]:
                global _REID_DLPACK_DEBUG_LOGGED
                if dlpack_debug_enabled and not _REID_DLPACK_DEBUG_LOGGED:
                    logger.info(
                        "ReID DLPack decode failed: reason=%s capsule=%s versioned=%s ver=%s.%s ndim=%s shape=%s dtype=(code=%s bits=%s lanes=%s) dev_type=%s",
                        reason,
                        capsule_name,
                        used_versioned,
                        ver_major,
                        ver_minor,
                        ndim,
                        shape,
                        dtype_code,
                        dtype_bits,
                        dtype_lanes,
                        dev_type,
                    )
                    _REID_DLPACK_DEBUG_LOGGED = True
                return None

            class _DLDevice(ctypes.Structure):
                _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]

            class _DLDataType(ctypes.Structure):
                _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]

            class _DLTensor(ctypes.Structure):
                _fields_ = [
                    ("data", ctypes.c_void_p),
                    ("device", _DLDevice),
                    ("ndim", ctypes.c_int),
                    ("dtype", _DLDataType),
                    ("shape", ctypes.POINTER(ctypes.c_int64)),
                    ("strides", ctypes.POINTER(ctypes.c_int64)),
                    ("byte_offset", ctypes.c_uint64),
                ]

            class _DLManagedTensor(ctypes.Structure):
                _fields_ = [("dl_tensor", _DLTensor), ("manager_ctx", ctypes.c_void_p), ("deleter", ctypes.c_void_p)]

            class _DLPackVersion(ctypes.Structure):
                _fields_ = [("major", ctypes.c_int32), ("minor", ctypes.c_int32)]

            class _DLManagedTensorVersioned(ctypes.Structure):
                _fields_ = [
                    ("version", _DLPackVersion),
                    ("dl_tensor", _DLTensor),
                    ("manager_ctx", ctypes.c_void_p),
                    ("deleter", ctypes.c_void_p),
                ]

            dlpack_capsule = layer_tensor.__dlpack__(None)
            raw_name = None
            try:
                get_name = ctypes.pythonapi.PyCapsule_GetName
                get_name.restype = ctypes.c_char_p
                get_name.argtypes = [ctypes.py_object]
                raw_name = get_name(dlpack_capsule)
                capsule_name = raw_name.decode("utf-8", "replace") if raw_name else None
            except Exception:
                capsule_name = None
            get_ptr = ctypes.pythonapi.PyCapsule_GetPointer
            get_ptr.restype = ctypes.c_void_p
            get_ptr.argtypes = [ctypes.py_object, ctypes.c_char_p]
            managed_ptr = get_ptr(dlpack_capsule, raw_name)
            if not managed_ptr:
                return _fail("capsule_get_pointer")

            dl = None
            deleter_ptr = None
            try:
                managed_v = ctypes.cast(managed_ptr, ctypes.POINTER(_DLManagedTensorVersioned))
                ver = managed_v.contents.version
                dl_candidate = managed_v.contents.dl_tensor
                ver_major = int(ver.major)
                ver_minor = int(ver.minor)
                ndim_candidate = int(dl_candidate.ndim)
                dtype_bits_candidate = int(dl_candidate.dtype.bits)
                dtype_code_candidate = int(dl_candidate.dtype.code)
                dev_type_candidate = int(dl_candidate.device.device_type)
                plausible = (
                    0 <= ver_major <= 10
                    and 0 <= ver_minor <= 10
                    and 1 <= ndim_candidate <= 8
                    and dtype_bits_candidate in (8, 16, 32, 64)
                    and 0 <= dtype_code_candidate <= 8
                    and 1 <= dev_type_candidate <= 32
                    and int(dl_candidate.data or 0) != 0
                )
                if plausible:
                    dl = dl_candidate
                    deleter_ptr = managed_v.contents.deleter
                    used_versioned = True
            except Exception:
                dl = None
                deleter_ptr = None
            if dl is None:
                managed = ctypes.cast(managed_ptr, ctypes.POINTER(_DLManagedTensor))
                dl = managed.contents.dl_tensor
                deleter_ptr = managed.contents.deleter

            ndim = int(dl.ndim)
            if ndim < 1:
                return _fail("ndim")
            shape = [int(dl.shape[i]) for i in range(ndim)]
            total = 1
            for dim in shape:
                total *= max(1, int(dim))
            dtype_bits = int(dl.dtype.bits)
            dtype_code = int(dl.dtype.code)
            dtype_lanes = int(dl.dtype.lanes)
            if dtype_code != 2 or dtype_bits != 32 or dtype_lanes != 1:
                return _fail("dtype")
            nbytes = int(total * (dtype_bits // 8) * dtype_lanes)
            if nbytes <= 0:
                return _fail("nbytes")

            out = np.empty((total,), dtype=np.float32)
            start_ns = time.perf_counter_ns()
            dst_ptr = ctypes.c_void_p(int(out.ctypes.data))
            src_ptr = ctypes.c_void_p(int(dl.data) + int(dl.byte_offset))
            dev_type = int(dl.device.device_type)

            if dev_type in (1, 3):  # kDLCPU / kDLCUDAHost
                ctypes.memmove(dst_ptr, src_ptr, nbytes)
            elif dev_type in (2, 13):  # kDLCUDA / kDLCUDAManaged
                cudart_path = ctypes.util.find_library("cudart")
                if not cudart_path:
                    return _fail("cudart")
                cudart = ctypes.CDLL(cudart_path)
                cuda_memcpy = cudart.cudaMemcpy
                cuda_memcpy.restype = ctypes.c_int
                cuda_memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
                err = int(cuda_memcpy(dst_ptr, src_ptr, ctypes.c_size_t(nbytes), ctypes.c_int(2)))
                if err != 0:
                    return _fail(f"cudaMemcpy:{err}")
            else:
                return _fail("device_type")

            call_deleter = str(os.environ.get("NOESIS_REID_DLPACK_CALL_DELETER", "")).strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            if call_deleter and deleter_ptr:
                deleter = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(deleter_ptr)
                deleter(managed_ptr)

            _CORE_PATH_INSTRUMENTATION.record_cpu_copy_violation(
                location="reid.tensor_dlpack_to_embedding.fallback_copy",
                reason="ctypes_host_copy",
                details={
                    "duration_ns": time.perf_counter_ns() - start_ns,
                    "bytes": int(nbytes),
                    "device_type": int(dev_type),
                },
            )
            emb = out.reshape(-1).astype(np.float32, copy=False)
            if emb.size < 1:
                return _fail("empty")
            n = float(np.linalg.norm(emb) + 1e-12)
            return (emb / n).astype(np.float32)
        except Exception as exc:
            if isinstance(exc, _CorePathFallbackConversionError):
                raise
            global _REID_DLPACK_DEBUG_LOGGED
            if dlpack_debug_enabled and not _REID_DLPACK_DEBUG_LOGGED:
                logger.info("ReID DLPack decode raised: %s", exc, exc_info=True)
                _REID_DLPACK_DEBUG_LOGGED = True
            return None

    def _log_diag_session_start(self) -> None:
        if not self.diagnostics_logger or self._diag_logged:
            return
        self._diag_logged = True
        try:
            payload = {
                "type": "v3dt_session_start",
                "ts": time.time(),
                "camera_labels": dict(self.camera_labels or {}),
                "sensor_id_map": dict(self.sensor_id_map or {}),
                "pipeline_config": getattr(self.pipeline, "config", {}),
                "env": {k: v for k, v in os.environ.items() if k.startswith("NOESIS_")},
            }
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

    @staticmethod
    def _world_from_bbox3d(bbox3d: Mapping[str, Any]) -> Optional[List[float]]:
        try:
            x = float(bbox3d.get("xCentre"))
            y = float(bbox3d.get("yCentre"))
            z = float(bbox3d.get("zCentre"))
            z_len = float(bbox3d.get("zLen"))
        except Exception:
            return None
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z) and math.isfinite(z_len)):
            return None
        # SV3DT uses Z-up; place the footpoint on the ground plane (Z = center - 0.5 * height).
        return [float(x), float(y), float(z - 0.5 * z_len)]

    def _extract_reid_embedding_ds8(self, obj_meta: Any) -> Optional[np.ndarray]:
        """Extract OSNet embedding from DS8 object tensor meta (SGIE output)."""
        tensor_items_iter = getattr(obj_meta, "tensor_items", None)
        if tensor_items_iter is None:
            return None
        # Snapshot iterator to avoid lifetime/iteration hazards in some DS builds.
        try:
            tensor_items = list(tensor_items_iter)
        except Exception:
            tensor_items = tensor_items_iter or []
        for item in tensor_items:
            try:
                if not item:
                    continue
            except Exception:
                pass
            try:
                tensor_output = item.as_tensor_output()
            except Exception:
                continue
            try:
                if not tensor_output:
                    continue
            except Exception:
                pass
            try:
                if int(getattr(tensor_output, "unique_id", -1)) != int(self._reid_unique_id):
                    continue
            except Exception:
                continue
            try:
                layers = tensor_output.get_layers()
            except Exception:
                continue
            if not isinstance(layers, dict) or not layers:
                continue
            layer_tensor = layers.get(self._reid_layer_name)
            if layer_tensor is None:
                # If the layer name is unknown, fall back to a likely embedding output.
                if len(layers) == 1:
                    try:
                        layer_tensor = next(iter(layers.values()))
                    except Exception:
                        layer_tensor = None
                else:
                    # Prefer a layer whose shape looks like a 512-D vector.
                    for candidate in layers.values():
                        try:
                            shape = getattr(candidate, "shape", None)
                            if shape is None:
                                continue
                            if hasattr(shape, "__len__") and len(shape) >= 1 and int(shape[-1]) == 512:
                                layer_tensor = candidate
                                break
                        except Exception:
                            continue
            layer_tensor = _clone_tensor_for_host_read(layer_tensor, location="reid.embedding")
            if layer_tensor is None:
                continue
            with _DLPACK_HOST_READ_LOCK:
                emb = self._tensor_to_embedding(layer_tensor)
            if emb is None:
                continue
            if not self._reid_logged_shape:
                try:
                    shape = getattr(layer_tensor, "shape", None)
                    dtype = getattr(layer_tensor, "dtype", None)
                    dev = getattr(layer_tensor, "device_type", None)
                    keys = list(layers.keys())
                    logger.info(
                        "ReID SGIE tensor observed: expected_unique_id=%s layers=%s shape=%s dtype=%s device=%s",
                        int(self._reid_unique_id),
                        keys,
                        shape,
                        dtype,
                        dev,
                    )
                except Exception:
                    pass
                self._reid_logged_shape = True
            return emb
        return None

    def handle_frame_ds8(self, frame_meta: Any) -> None:
        """Extract tracking telemetry for a single frame using DS8 pyservicemaker API."""
        try:
            source_id = self._frame_source_id(frame_meta)
            sensor_id = self.sensor_id_map.get(source_id, source_id)
            camera_id = self.camera_labels.get(sensor_id, f"camera_{sensor_id}")
            now_ts = time.time()
            self._log_diag_session_start()

            tracks: List[Dict[str, Any]] = []
            diagnostics_tracks: List[Dict[str, Any]] = []
            frame_id = int(getattr(frame_meta, "frame_number", -1))
            occupancy_counts: Dict[str, int] = {}
            present_track_ids: set[int] = set()
            present_stable_ids: set[int] = set()
            footpoints: List[Footpoint] = []
            frame_dims = self._frame_dims()

            reid_debug = str(os.environ.get("NOESIS_REID_DEBUG", "")).strip().lower() in ("1", "true", "yes", "on")
            if reid_debug:
                self._reid_debug_frames += 1

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

                # People-only public identity. Do not show raw tracker IDs.
                if class_id != 0:
                    self._stamp_osd_label_ds8(obj_meta, sensor_id=sensor_id, stable_id=None)
                    diagnostics_tracks.append(diag_track)
                    continue

                if not zone:
                    zone = _fallback_zone_from_camera(camera_id)

                if reid_debug:
                    self._reid_debug_people += 1

                emb = None
                if self._stable_id_enabled:
                    mgr = getattr(self.pipeline, "stable_id_mgr", None)
                    if mgr is not None:
                        need_emb = True
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

                        if need_emb:
                            emb = self._extract_reid_embedding_ds8(obj_meta)
                            if reid_debug:
                                if emb is None:
                                    self._reid_debug_emb_missing += 1
                                else:
                                    self._reid_debug_emb_found += 1

                stable_id = self._maybe_assign_stable_id(
                    sensor_id=sensor_id,
                    track_id=track_id,
                    bbox=raw.get("bbox"),
                    zone=zone,
                    ts=now_ts,
                    frame_bgr=None,
                    embedding=emb,
                )
                if stable_id is None:
                    # People should always have a stable_id; if we can't produce one, show placeholder.
                    self._stamp_osd_label_ds8(obj_meta, sensor_id=sensor_id, stable_id=None)
                    diagnostics_tracks.append(diag_track)
                    continue

                stable_id_int = int(stable_id)
                present_stable_ids.add(stable_id_int)

                # Stamp OSD label early so mosaic never falls back to tracker IDs.
                self._stamp_osd_label_ds8(obj_meta, sensor_id=sensor_id, stable_id=stable_id_int)
                self._apply_instance_mask_color_ds8(obj_meta, stable_id=stable_id_int)

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
                    "camera_id": camera_id,
                    "bbox": raw.get("bbox"),
                    "center": raw.get("center"),
                    "class_id": 0,
                    "confidence": raw.get("confidence"),
                    "tracker_confidence": raw.get("tracker_confidence"),
                    "analytics": raw.get("analytics"),
                    "zone": zone,
                    "frame_id": frame_id,
                    "dwell_time": dwell,
                }
                for key in (
                    "bbox3d",
                    "velocity3d",
                    "visibility",
                    "image_foot",
                    "image_base",
                    "world",
                    "world_valid",
                    "world_frame",
                    "world_source",
                ):
                    if key in raw:
                        public_track[key] = raw.get(key)

                self._augment_track_with_world(sensor_id, camera_id, public_track)
                diag_track.update(
                    {
                        "stable_id": stable_id_int,
                        "zone": zone,
                        "dwell_time": dwell,
                        "world": public_track.get("world"),
                        "world_valid": public_track.get("world_valid"),
                        "world_frame": public_track.get("world_frame"),
                        "world_source": public_track.get("world_source"),
                    }
                )
                tracks.append(public_track)
                diagnostics_tracks.append(diag_track)

                fp = self._footpoint_from_track(public_track, frame_dims)
                if fp is not None:
                    footpoints.append(fp)

            mgr = getattr(self.pipeline, "stable_id_mgr", None)
            observe_fn = getattr(mgr, "observe_copresence", None)
            if callable(observe_fn):
                try:
                    observe_fn(sorted(present_stable_ids), float(now_ts))
                except Exception:
                    logger.debug("StableIDManager observe_copresence failed", exc_info=True)

            self._publish_occupancy(sensor_id, occupancy_counts)
            self._cleanup_zone_state(sensor_id, present_stable_ids)
            self._maintain_stable_ids(sensor_id, present_track_ids, now_ts)
            self._active_tracks[sensor_id] = tracks

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

            if not tracks and os.environ.get("NOESIS_REID_TEST_MODE") == "1":
                synthetic = {
                    "camera_id": camera_id,
                    "stable_id": 1,
                    "bbox": (0.0, 0.0, 10.0, 10.0),
                    "frame_id": frame_id,
                    "zone": None,
                    "class_id": 0,
                }
                tracks.append(synthetic)
                present_stable_ids.add(1)
                present_track_ids.add(1)

            if not tracks:
                try:
                    self._publish_bev(sensor_id, camera_id, frame_meta, footpoints)
                except Exception:
                    logger.exception("BEV publish failed for sensor %s", sensor_id)
                return

            try:
                self.tracking_pub.publish(sensor_id, tracks)
            except Exception:  # pragma: no cover - telemetry should never break pipeline
                logger.exception("Tracking telemetry publish failed for sensor %s", sensor_id)
            try:
                self._publish_bev(sensor_id, camera_id, frame_meta, footpoints)
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
            self._log_diag_session_start()

            tracks: List[Dict[str, Any]] = []
            diagnostics_tracks: List[Dict[str, Any]] = []
            frame_id = int(getattr(frame_meta, "frame_num", -1))
            occupancy_counts: Dict[str, int] = {}
            present_track_ids: set[int] = set()
            present_stable_ids: set[int] = set()
            footpoints: List[Footpoint] = []
            frame_dims = self._frame_dims()

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
                if class_id != 0:
                    self._stamp_osd_label(obj_meta, sensor_id=sensor_id, stable_id=None)
                    diagnostics_tracks.append(diag_track)
                    continue

                if not zone:
                    zone = _fallback_zone_from_camera(camera_id)

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

                self._stamp_osd_label(obj_meta, sensor_id=sensor_id, stable_id=stable_id_int)
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
                    "camera_id": camera_id,
                    "bbox": raw.get("bbox"),
                    "center": raw.get("center"),
                    "class_id": 0,
                    "confidence": raw.get("confidence"),
                    "tracker_confidence": raw.get("tracker_confidence"),
                    "analytics": raw.get("analytics"),
                    "zone": zone,
                    "frame_id": frame_id,
                    "dwell_time": dwell,
                }
                for key in (
                    "bbox3d",
                    "velocity3d",
                    "visibility",
                    "image_foot",
                    "world",
                    "world_valid",
                    "world_frame",
                    "world_source",
                ):
                    if key in raw:
                        public_track[key] = raw.get(key)

                self._augment_track_with_world(sensor_id, camera_id, public_track)
                diag_track.update(
                    {
                        "stable_id": stable_id_int,
                        "zone": zone,
                        "dwell_time": dwell,
                        "world": public_track.get("world"),
                        "world_valid": public_track.get("world_valid"),
                        "world_frame": public_track.get("world_frame"),
                        "world_source": public_track.get("world_source"),
                    }
                )
                tracks.append(public_track)
                diagnostics_tracks.append(diag_track)

                fp = self._footpoint_from_track(public_track, frame_dims)
                if fp is not None:
                    footpoints.append(fp)

            self._publish_occupancy(sensor_id, occupancy_counts)
            self._cleanup_zone_state(sensor_id, present_stable_ids)
            self._maintain_stable_ids(sensor_id, present_track_ids, now_ts)
            self._active_tracks[sensor_id] = tracks

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
                    "bbox": (0.0, 0.0, 10.0, 10.0),
                    "frame_id": frame_id,
                    "zone": None,
                    "class_id": 0,
                }
                tracks.append(synthetic)
                present_stable_ids.add(1)
                present_track_ids.add(1)

            if not tracks:
                try:
                    self._publish_bev(sensor_id, camera_id, frame_meta, footpoints)
                except Exception:
                    logger.exception("BEV publish failed for sensor %s", sensor_id)
                return

            try:
                self.tracking_pub.publish(sensor_id, tracks)
            except Exception:  # pragma: no cover - telemetry should never break pipeline
                logger.exception("Tracking telemetry publish failed for sensor %s", sensor_id)
            try:
                self._publish_bev(sensor_id, camera_id, frame_meta, footpoints)
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

    def _footpoint_from_track(
        self, track: Mapping[str, Any], frame_dims: Tuple[int, int]
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

        def _clip_uv(u: float, v: float) -> Optional[Tuple[float, float]]:
            frame_w, frame_h = frame_dims
            if frame_h:
                margin = max(2.0, 0.01 * float(frame_h))
                if v < -margin or v > (frame_h + margin):
                    return None
                v = float(np.clip(v, 0.0, float(frame_h)))
            if frame_w:
                u = float(np.clip(u, 0.0, float(frame_w)))
            return u, v

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

        use_image_meta = self._tracking_mode_is_v3dt()
        if use_image_meta:
            if track.get("world_source") != "bbox3d" and not isinstance(track.get("bbox3d"), dict):
                use_image_meta = False

        if use_image_meta:
            for key, label in (("image_base", "image_base"), ("image_foot", "image_foot")):
                uv = _parse_uv(track.get(key))
                if uv is None:
                    continue
                clipped = _clip_uv(*uv)
                if clipped is None:
                    continue
                u, v = clipped
                method = label
                break

        if u is None or v is None:
            bbox = track.get("bbox")
            if not bbox or len(bbox) < 4:
                return None
            try:
                left, top, width, height = [float(x) for x in bbox[:4]]
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

        stable_id = track.get("stable_id")
        try:
            stable_id_int = int(stable_id) if stable_id not in (None, "", -1) else None
        except Exception:
            stable_id_int = None
        if stable_id_int is not None and stable_id_int <= 0:
            stable_id_int = None
        return Footpoint(u=u, v=v, method=method or "bbox", stable_id=stable_id_int)

    def _frame_timestamp_us(self, frame_meta: Any) -> int:
        pts_ns = int(_meta_lookup(frame_meta, "buf_pts", "buffer_pts", "pts", default=0) or 0)
        if pts_ns <= 0:
            pts_ns = int(time.time() * 1_000_000_000)
        return max(0, pts_ns // 1_000)

    def _augment_track_with_world(self, sensor_id: int, camera_id: str, track: Dict[str, Any]) -> None:
        """Calculate world coordinates for a track if calibration is available."""
        if self.bev_calibration is None:
            return
        if track.get("world_source") == "bbox3d":
            return
        if track.get("world") is not None and track.get("world_valid") is True:
            return
        
        try:
            calib = self.bev_calibration.snapshot(sensor_id, camera_id)
            if calib is None or calib.intrinsics is None or calib.extrinsics_col_major is None:
                return

            bbox = track.get("bbox")
            if not bbox or len(bbox) < 4:
                return

            # bbox is [left, top, width, height]
            # Use bottom center for footpoint
            u = float(bbox[0]) + float(bbox[2]) / 2.0
            v = float(bbox[1]) + float(bbox[3])

            R_wc, C_world = parse_extrinsics(calib.extrinsics_col_major)
            scale = float(calib.unit_scale or 1.0)
            C_world = C_world * scale
            plane = Plane.horizontal(float(calib.floor_y) * scale)

            origin, direction = ray_from_pixel(u, v, calib.intrinsics, R_wc, C_world)
            hit = intersect_plane(origin, direction, plane)

            if hit is not None:
                track["world"] = [float(hit[0]), float(hit[1]), float(hit[2])]
                track["world_valid"] = True
                track["world_frame"] = self._world_frame
                track["world_source"] = "ray"
            else:
                track["world_valid"] = False
        except Exception:
            # Silently fail; world coordinates are best-effort
            pass

    def _publish_bev(
        self,
        sensor_id: int,
        camera_id: str,
        frame_meta: Any,
        footpoints: Sequence[Footpoint],
    ) -> None:
        if self.bev_renderer is None or self.bev_calibration is None:
            return
        try:
            calib = self.bev_calibration.snapshot(sensor_id, camera_id)
        except Exception:
            calib = None
        if calib is None:
            return
        ts_us = self._frame_timestamp_us(frame_meta)
        try:
            self.bev_renderer.render_and_publish(
                camera_id=camera_id,
                calib=calib,
                footpoints=list(footpoints),
                timestamp_us=ts_us,
            )
        except Exception:
            logger.exception("BEV render failed for %s", camera_id)

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

    def _fallback_stable_id(self, sensor_id: int, track_id: int, ts: float) -> int:
        """Allocate a stable_id when the real StableIDManager is unavailable/unhealthy.

        This keeps user-facing payloads free of raw tracker IDs while ensuring every
        visible person always has a numeric stable_id.
        """
        key = (int(sensor_id), int(track_id))
        sid = self._fallback_sid_by_key.get(key)
        if sid is None:
            # Try to start after any already-allocated StableIDManager range to reduce
            # collisions if we fallback mid-run.
            if self._fallback_sid_next <= 1:
                mgr = getattr(self.pipeline, "stable_id_mgr", None)
                next_sid = getattr(mgr, "next_stable_id", None) if mgr is not None else None
                if next_sid is not None:
                    try:
                        self._fallback_sid_next = max(int(self._fallback_sid_next), int(next_sid))
                    except Exception:
                        pass
            sid = int(self._fallback_sid_next)
            self._fallback_sid_next = int(self._fallback_sid_next) + 1
            self._fallback_sid_by_key[key] = sid
        self._fallback_sid_last_seen[key] = float(ts)
        return int(sid)

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
        """Return a positive stable_id for a tracked person.

        Prefers StableIDManager (ReID) when healthy; falls back to an internal
        allocator so user-facing IDs never expose raw tracker IDs.
        """
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
                )
                stable_id_int = int(stable_id)
                if stable_id_int > 0:
                    return stable_id_int
            except Exception:
                logger.exception("StableIDManager update failed for sensor %s track %s", sensor_id, track_id)
                self._stable_id_enabled = False

        return self._fallback_stable_id(sensor_id, track_id, float(ts))

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
        """Maintain StableIDManager state and prune fallback IDs."""
        now_ts = float(ts)
        sensor_id_int = int(sensor_id)
        present_set = {int(tid) for tid in present_track_ids}

        mgr = getattr(self.pipeline, "stable_id_mgr", None)
        if self._stable_id_enabled and mgr is not None:
            try:
                mgr.remove_missing_tracks(sensor_id_int, list(present_set), now_ts)
                mgr.prune_ghosts(now_ts)
            except Exception:
                logger.exception("StableIDManager maintenance failed for sensor %s", sensor_id_int)
                self._stable_id_enabled = False

        # Maintain fallback stable IDs so they don't leak forever when the real ReID manager
        # is unavailable or returns invalid IDs.
        try:
            ttl_s = float(os.environ.get("NOESIS_FALLBACK_STABLE_ID_TTL_S", "15.0") or 15.0)
        except Exception:
            ttl_s = 15.0
        ttl_s = max(0.0, ttl_s)

        if not self._fallback_sid_by_key:
            return

        for key in list(self._fallback_sid_by_key.keys()):
            key_sensor, key_track = key
            if int(key_sensor) != sensor_id_int:
                continue
            if int(key_track) in present_set:
                continue
            last_seen = float(self._fallback_sid_last_seen.get(key, 0.0) or 0.0)
            if ttl_s <= 0.0 or (now_ts - last_seen) >= ttl_s:
                self._fallback_sid_by_key.pop(key, None)
                self._fallback_sid_last_seen.pop(key, None)

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

        return _OsdLabelProcessor(
            decimals=decimals,
            font_size=font_size,
            font_name=font_name,
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
                label = str(value).strip()
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
            if stable_id_int is not None and stable_id_int > 0:
                parts.append(f"{stable_id_int}")
            else:
                parts.append("XX")
        base_label = " ".join([p for p in parts if p]).strip()

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


class _OsdLabelOperator(BatchMetadataOperator):  # pragma: no cover - requires DeepStream runtime
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


class _AnalyticsTelemetryOperator(BatchMetadataOperator):  # pragma: no cover - requires DeepStream runtime
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


@dataclass
class _ExcludePruneProcessor:
    polygons: Mapping[int, Sequence[Tuple[str, Sequence[Tuple[float, float]]]]]
    _object_meta_cast: Optional[Callable[[Any], Any]] = field(default=None, init=False, repr=False)

    def handle_frame_ds8(self, frame_meta: Any) -> None:
        """Handle frame using DS8 pyservicemaker API."""
        source_id = int(getattr(frame_meta, "pad_index", getattr(frame_meta, "source_id", -1)))
        if source_id < 0:
            return
        poly_specs = self.polygons.get(source_id)
        if not poly_specs:
            return

        # DS8 API: object_items is read-only iterable; we can't remove items
        # Instead, we filter and log which objects would be pruned
        object_items = getattr(frame_meta, "object_items", None) or []
        pruned_count = 0
        for obj_meta in object_items:
            rect = getattr(obj_meta, "rect_params", None)
            bbox = _rect_to_bbox(rect)
            if bbox is None:
                continue
            corners = _bbox_corners(bbox)
            if self._inside_exclusion(corners, poly_specs):
                pruned_count += 1
                # Note: DS8 pyservicemaker doesn't expose object removal API
                # Objects are filtered downstream by analytics/hooks
        if pruned_count:
            logger.debug("DS8: Would prune %s object(s) from exclusion ROIs on source %s (read-only)", pruned_count, source_id)

    def handle_frame(self, frame_meta: Any) -> None:
        source_id = int(_meta_lookup(frame_meta, "source_id", "pad_index", default=-1))
        if source_id < 0:
            return
        poly_specs = self.polygons.get(source_id)
        if not poly_specs:
            return

        to_remove: List[Any] = []
        cast = self._resolve_object_meta_cast()
        obj_iter = _iter_meta_entries(getattr(frame_meta, "obj_meta_list", None), cast)
        for obj_meta in obj_iter:
            rect = getattr(obj_meta, "rect_params", None)
            bbox = _rect_to_bbox(rect)
            if bbox is None:
                continue
            corners = _bbox_corners(bbox)
            if self._inside_exclusion(corners, poly_specs):
                to_remove.append(obj_meta)

        if not to_remove:
            return

        removed = 0
        for obj_meta in to_remove:
            if self._remove_obj(frame_meta, obj_meta):
                removed += 1

        if removed:
            logger.debug("Pruned %s object(s) from exclusion ROIs on source %s", removed, source_id)

    def _inside_exclusion(
        self,
        corners: Sequence[Tuple[float, float]],
        poly_specs: Sequence[Tuple[str, Sequence[Tuple[float, float]]]],
    ) -> bool:
        for _, poly in poly_specs:
            if len(poly) < 3:
                continue
            if all(_point_in_polygon(point, poly) for point in corners):
                return True
        return False

    def _resolve_object_meta_cast(self) -> Optional[Callable[[Any], Any]]:
        if self._object_meta_cast is not None:
            return self._object_meta_cast
        self._object_meta_cast = _resolve_pyds_cast("NvDsObjectMeta")
        return self._object_meta_cast

    def _remove_obj(self, frame_meta: Any, obj_meta: Any) -> bool:
        remover = _resolve_pyds_attr("nvds_remove_obj_meta_from_frame")
        if callable(remover):
            try:  # pragma: no cover - requires DeepStream runtime
                remover(frame_meta, obj_meta)
                return True
            except Exception:
                logger.exception("Failed to remove NvDsObjectMeta via nvds_remove_obj_meta_from_frame")

        obj_list = getattr(frame_meta, "obj_meta_list", None)
        if isinstance(obj_list, list):
            try:
                obj_list.remove(obj_meta)
                return True
            except ValueError:
                return False

        # Fallback for simple single-link structures
        try:
            head = getattr(frame_meta, "obj_meta_list", None)
            prev = None
            node = head
            while node is not None:
                data = getattr(node, "data", node)
                if data is obj_meta:
                    nxt = getattr(node, "next", None)
                    if prev is None:
                        setattr(frame_meta, "obj_meta_list", nxt)
                    else:
                        setattr(prev, "next", nxt)
                    return True
                prev = node
                node = getattr(node, "next", None)
        except Exception:
            pass
        return False


class _ExcludePruneOperator(BatchMetadataOperator):  # pragma: no cover - requires DeepStream runtime
    def __init__(self, processor: _ExcludePruneProcessor) -> None:
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
                logger.exception("Failed to prune exclusion objects within batch metadata (DS8)")


class _TrailOverlayOperator(BatchMetadataOperator):  # pragma: no cover - requires DeepStream runtime
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


class _IntrinsicsOperator(BatchMetadataOperator):  # pragma: no cover - requires DeepStream runtime
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


class _MapAnythingOperator(BatchMetadataOperator):  # pragma: no cover - requires DeepStream runtime
    def __init__(self, processor: MapAnythingProcessor) -> None:
        super().__init__()
        self._processor = processor
        self._frames_seen = 0
        self._matched_frames = 0
        self._warned_no_tensors = False
        self._warned_no_match = False

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
                    ids = [getattr(item, "unique_id", None) for item in converted_items]
                    if not self._warned_no_match:
                        logger.debug(
                            "MapAnything tensor_items present but no matching gie_id=%s (frame_number=%s, available_ids=%s)",
                            self._processor.gie_id,
                            getattr(frame_meta, "frame_number", None),
                            ids,
                        )
                        self._warned_no_match = True
                    # Fallback: process the first tensor_meta when no matching gie_id is found.
                    try:
                        self._processor.handle_nvds_tensor_ds8(frame_meta, converted_items[0])
                        self._matched_frames += 1
                    except Exception:
                        logger.exception("Failed to process MapAnything tensors from batch metadata (DS8)")
            except Exception:
                logger.exception("Failed to process MapAnything tensors from batch metadata (DS8)")


class _PoseFeatureOperator(BatchMetadataOperator):  # pragma: no cover - requires DeepStream runtime
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


class _PoseKeypointOverlayOperator(BatchMetadataOperator):  # pragma: no cover - requires DeepStream runtime
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


def _iter_roi_labels(raw: Any) -> Iterable[str]:
    if raw is None:
        return
    if isinstance(raw, dict):
        for key, value in raw.items():
            status = str(value).strip().lower()
            if status in {"1", "true", "in", "inside", "present"}:
                yield str(key).strip()
    elif isinstance(raw, (list, tuple, set)):
        for item in raw:
            label = str(item).strip()
            if label:
                yield label
    elif isinstance(raw, str):
        for part in raw.split(","):
            label = part.strip()
            if label:
                yield label
    else:
        label = str(raw).strip()
        if label:
            yield label


def _primary_zone_from_analytics(analytics_meta: Mapping[str, Any]) -> Optional[str]:
    roi_status = analytics_meta.get("roiStatus")
    for label in _iter_roi_labels(roi_status):
        if label:
            return label
    return None


def _fallback_zone_from_camera(camera_id: Any) -> Optional[str]:
    """Fallback for Zone/Dwell/Occupancy when nvdsanalytics ROI labels are absent.

    Treat each camera/stream as its own room so UI occupancy and dwell timers remain useful
    even when the analytics config does not emit per-object `roiStatus`.
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
