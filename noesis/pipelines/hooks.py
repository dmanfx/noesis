from __future__ import annotations

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

logger = logging.getLogger(__name__)


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


def attach_analytics_telemetry_hook(
    pipeline: "DS8Pipeline",
    *,
    tracking_pub: "TrackingTelemetryPublisher",
    camera_labels: Optional[Mapping[int, str]] = None,
    sensor_id_map: Optional[Mapping[int, int]] = None,
    bev_renderer: Any | None = None,
    bev_calibration: Any | None = None,
) -> None:
    """Attach a BatchMetadataOperator that extracts analytics telemetry."""
    if tracking_pub is None:
        raise ValueError("tracking_pub must be provided for analytics telemetry")

    component = pipeline.components.get("analytics")
    if component is None:
        raise KeyError("analytics component missing; cannot attach telemetry hook")

    processor = _AnalyticsTelemetryProcessor(
        pipeline=pipeline,
        tracking_pub=tracking_pub,
        camera_labels=camera_labels or {},
        sensor_id_map=sensor_id_map or {},
        bev_renderer=bev_renderer,
        bev_calibration=bev_calibration,
    )
    component.config["_analytics_processor"] = processor

    if pipeline.ds_pipeline is None or BatchMetadataOperator is None or Probe is None:
        logger.debug("Stored analytics telemetry processor for lazy execution (pyservicemaker unavailable)")
        return

    try:
        probe = Probe("analytics_telemetry", _AnalyticsTelemetryOperator(processor))
        pipeline.ds_pipeline.attach(component.name, probe)
        logger.info("Attached analytics telemetry probe to %s", component.name)
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
                import torch.utils.dlpack as torch_dlpack
                import torch

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
                return torch_tensor.detach().cpu().numpy()
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

            key_id = int(track_id)
            if self.config.color_key == "stable_id" and stable_id not in (None, "", -1):
                try:
                    key_id = int(stable_id)
                except Exception:
                    key_id = int(track_id)
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
            if stable_id not in (None, "", -1):
                text.display_text = f"sid {stable_id}"
            else:
                text.display_text = f"id {track_id}"
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


@dataclass
class _AnalyticsTelemetryProcessor:
    pipeline: "DS8Pipeline"
    tracking_pub: "TrackingTelemetryPublisher"
    camera_labels: Mapping[int, str]
    sensor_id_map: Mapping[int, int]
    bev_renderer: Any = None
    bev_calibration: Any = None
    _analytics_obj_meta_type: Any = field(default=None, init=False, repr=False)
    _zone_state: Dict[int, Dict[int, Dict[str, Any]]] = field(default_factory=dict, init=False, repr=False)
    _occupancy_state: Dict[int, Dict[str, int]] = field(default_factory=dict, init=False, repr=False)
    _active_tracks: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict, init=False, repr=False)
    _transitions_state: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict, init=False, repr=False)
    _stable_id_enabled: bool = field(default=True, init=False, repr=False)
    _bev_class_ids: frozenset[int] = field(default_factory=lambda: frozenset({0}), init=False, repr=False)
    _bev_class_ids_ready: bool = field(default=False, init=False, repr=False)

    def handle_frame_ds8(self, frame_meta: Any) -> None:
        """Extract tracking telemetry for a single frame using DS8 pyservicemaker API."""
        try:
            source_id = self._frame_source_id(frame_meta)
            sensor_id = self.sensor_id_map.get(source_id, source_id)
            camera_id = self.camera_labels.get(sensor_id, f"camera_{sensor_id}")
            now_ts = time.time()

            tracks: List[Dict[str, Any]] = []
            frame_id = int(getattr(frame_meta, "frame_number", -1))
            occupancy_counts: Dict[str, int] = {}
            present_track_ids: set[int] = set()
            footpoints: List[Footpoint] = []
            frame_dims = self._frame_dims()

            # DS8 API: frame_meta.object_items is an iterable
            object_items = getattr(frame_meta, "object_items", None) or []
            for obj_meta in object_items:
                track = self._build_track_dict_ds8(obj_meta, camera_id)
                if track is None:
                    continue
                
                track_id = int(track.get("track_id", -1))
                present_track_ids.add(track_id)

                # Handle line crossings from analytics
                analytics = track.get("analytics")
                if analytics and "lcStatus" in analytics:
                    lc = analytics["lcStatus"]
                    if isinstance(lc, dict):
                        for line_name, status in lc.items():
                            if status == 1:
                                self._record_transition(
                                    sensor_id=sensor_id,
                                    track_id=track_id,
                                    line_name=line_name,
                                    ts=now_ts,
                                )

                if "stable_id" not in track:
                    track["stable_id"] = None
                
                zone = track.get("zone")
                dwell = self._update_dwell_time(sensor_id, track_id, zone, now_ts)
                track["dwell_time"] = dwell
                frame_crop = self._reid_crop_from_track(track, frame_dims)
                track["stable_id"] = self._maybe_assign_stable_id(
                    sensor_id=sensor_id,
                    track_id=track_id,
                    bbox=track.get("bbox"),
                    zone=zone,
                    ts=now_ts,
                    frame_bgr=frame_crop,
                )
                if zone:
                    occupancy_counts[zone] = occupancy_counts.get(zone, 0) + 1
                    logger.debug(f"Track {track_id} in zone {zone}, occupancy now: {occupancy_counts[zone]}")
                track["frame_id"] = frame_id
                
                # Augment track with world coordinates if calibration is available
                self._augment_track_with_world(sensor_id, camera_id, track)

                tracks.append(track)
                fp = self._footpoint_from_track(track, frame_dims)
                if fp is not None:
                    footpoints.append(fp)

            self._publish_occupancy(sensor_id, occupancy_counts)
            self._cleanup_zone_state(sensor_id, present_track_ids)
            self._maintain_stable_ids(sensor_id, present_track_ids, now_ts)
            self._active_tracks[sensor_id] = tracks

            if not tracks and os.environ.get("NOESIS_REID_TEST_MODE") == "1":
                synthetic = {
                    "camera_id": camera_id,
                    "track_id": 1,
                    "stable_id": 1,
                    "bbox": (0.0, 0.0, 10.0, 10.0),
                    "frame_id": frame_id,
                    "zone": None,
                    "class_id": 0,
                }
                tracks.append(synthetic)
                present_track_ids.add(1)

            if not tracks:
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

            tracks: List[Dict[str, Any]] = []
            frame_id = int(getattr(frame_meta, "frame_num", -1))
            occupancy_counts: Dict[str, int] = {}
            present_track_ids: set[int] = set()
            footpoints: List[Footpoint] = []
            frame_dims = self._frame_dims()

            for obj_meta in self._iter_object_meta(frame_meta):
                track = self._build_track_dict(obj_meta, camera_id)
                if track is None:
                    continue
                
                track_id = int(track.get("track_id", -1))
                present_track_ids.add(track_id)

                # Handle line crossings from analytics
                analytics = track.get("analytics")
                if analytics and "lcStatus" in analytics:
                    lc = analytics["lcStatus"]
                    if isinstance(lc, dict):
                        for line_name, status in lc.items():
                            if status == 1:
                                self._record_transition(
                                    sensor_id=sensor_id,
                                    track_id=track_id,
                                    line_name=line_name,
                                    ts=now_ts,
                                )

                if "stable_id" not in track:
                    track["stable_id"] = None
                
                zone = track.get("zone")
                dwell = self._update_dwell_time(sensor_id, track_id, zone, now_ts)
                track["dwell_time"] = dwell
                frame_crop = self._reid_crop_from_track(track, frame_dims)
                track["stable_id"] = self._maybe_assign_stable_id(
                    sensor_id=sensor_id,
                    track_id=track_id,
                    bbox=track.get("bbox"),
                    zone=zone,
                    ts=now_ts,
                    frame_bgr=frame_crop,
                )
                if zone:
                    occupancy_counts[zone] = occupancy_counts.get(zone, 0) + 1
                    logger.debug(f"Track {track_id} in zone {zone}, occupancy now: {occupancy_counts[zone]}")
                track["frame_id"] = frame_id
                
                # Augment track with world coordinates if calibration is available
                self._augment_track_with_world(sensor_id, camera_id, track)

                tracks.append(track)
                fp = self._footpoint_from_track(track, frame_dims)
                if fp is not None:
                    footpoints.append(fp)

            self._publish_occupancy(sensor_id, occupancy_counts)
            self._cleanup_zone_state(sensor_id, present_track_ids)
            self._maintain_stable_ids(sensor_id, present_track_ids, now_ts)
            self._active_tracks[sensor_id] = tracks

            if not tracks and os.environ.get("NOESIS_REID_TEST_MODE") == "1":
                synthetic = {
                    "camera_id": camera_id,
                    "track_id": 1,
                    "stable_id": 1,
                    "bbox": (0.0, 0.0, 10.0, 10.0),
                    "frame_id": frame_id,
                    "zone": None,
                    "class_id": 0,
                }
                tracks.append(synthetic)
                present_track_ids.add(1)

            if not tracks:
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

    def _frame_dims(self) -> Tuple[int, int]:
        try:
            width, height = getattr(self.pipeline, "frame_size", (0, 0))
            return int(width or 0), int(height or 0)
        except Exception:
            return 0, 0

    def _footpoint_from_track(
        self, track: Mapping[str, Any], frame_dims: Tuple[int, int]
    ) -> Optional[Footpoint]:
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
        frame_w, frame_h = frame_dims
        if frame_h:
            margin = max(2.0, 0.01 * float(frame_h))
            if v < -margin or v > (frame_h + margin):
                return None
            v = float(np.clip(v, 0.0, float(frame_h)))
        if frame_w:
            u = float(np.clip(u, 0.0, float(frame_w)))
        track_id = track.get("track_id")
        try:
            track_id = int(track_id)
        except Exception:
            pass
        stable_id = track.get("stable_id")
        try:
            stable_id = int(stable_id) if stable_id not in (None, "", -1) else None
        except Exception:
            stable_id = None
        return Footpoint(u=u, v=v, method="bbox", track_id=track_id, stable_id=stable_id)

    def _frame_timestamp_us(self, frame_meta: Any) -> int:
        pts_ns = int(_meta_lookup(frame_meta, "buf_pts", "buffer_pts", "pts", default=0) or 0)
        if pts_ns <= 0:
            pts_ns = int(time.time() * 1_000_000_000)
        return max(0, pts_ns // 1_000)

    def _augment_track_with_world(self, sensor_id: int, camera_id: str, track: Dict[str, Any]) -> None:
        """Calculate world coordinates for a track if calibration is available."""
        if self.bev_calibration is None:
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
        frame_w, frame_h = calib.image_size if calib.image_size else self._frame_dims()
        if frame_w <= 0 or frame_h <= 0:
            frame_w, frame_h = self._frame_dims()
        if frame_w <= 0 or frame_h <= 0:
            frame_w, frame_h = 1920, 1080
        try:
            frame_bgr = np.zeros((int(frame_h), int(frame_w), 3), dtype=np.uint8)
            self.bev_renderer.render_and_publish(
                camera_id=camera_id,
                calib=calib,
                frame_bgr=frame_bgr,
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
            # Provide snake_case aliases for compatibility with DS7 telemetry consumers.
            data["direction_status"] = data.get("dirStatus")
            data["line_crossing_status"] = data.get("lcStatus")
            data["overcrowding_status"] = data.get("ocStatus")
            data["roi_status"] = data.get("roiStatus")

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

    def _maybe_assign_stable_id(
        self,
        *,
        sensor_id: int,
        track_id: int,
        bbox: Optional[Sequence[float]],
        zone: Optional[str],
        ts: float,
        frame_bgr: Optional[np.ndarray],
    ) -> Optional[int]:
        """Bridge to StableIDManager if present on the pipeline."""
        if track_id < 0 or not self._stable_id_enabled:
            return None
        mgr = getattr(self.pipeline, "stable_id_mgr", None)
        if mgr is None:
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
        try:
            stable_id = mgr.update(
                sensor_id=int(sensor_id),
                ds_obj_id=int(track_id),
                bbox_ltrbwh=(float(safe_bbox[0]), float(safe_bbox[1]), float(safe_bbox[2]), float(safe_bbox[3])),
                ts=float(ts),
                zone=str(zone) if zone else None,
                frame_bgr=frame_bgr,
            )
            if stable_id is None:
                return int(track_id)
            return int(stable_id)
        except Exception:
            logger.exception("StableIDManager update failed for sensor %s track %s", sensor_id, track_id)
            self._stable_id_enabled = False
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
        """Clean up missing tracks and ghosts on the StableIDManager."""
        mgr = getattr(self.pipeline, "stable_id_mgr", None)
        if mgr is None:
            return
        try:
            mgr.remove_missing_tracks(int(sensor_id), list(present_track_ids), float(ts))
            mgr.prune_ghosts(float(ts))
        except Exception:
            logger.exception("StableIDManager maintenance failed for sensor %s", sensor_id)
            self._stable_id_enabled = False

    def _update_dwell_time(
        self,
        sensor_id: int,
        track_id: int,
        zone: Optional[str],
        now_ts: float,
    ) -> Optional[float]:
        """Maintain per-track zone entry time to compute dwell seconds."""
        if track_id < 0:
            return None
        state = self._zone_state.setdefault(sensor_id, {})
        entry = state.get(track_id, {})
        current_zone = entry.get("zone")
        entry_time = entry.get("entry")

        if not zone:
            state[track_id] = {"zone": None, "entry": None}
            return None

        if current_zone == zone:
            if entry_time is None:
                entry_time = now_ts
            state[track_id] = {"zone": zone, "entry": entry_time}
            return max(0.0, now_ts - float(entry_time))

        # Zone change detected
        if current_zone and zone and current_zone != zone:
            self._record_transition(
                sensor_id=sensor_id,
                track_id=track_id,
                from_zone=current_zone,
                to_zone=zone,
                ts=now_ts,
            )

        state[track_id] = {"zone": zone, "entry": now_ts}
        return 0.0

    def _record_transition(
        self,
        sensor_id: int,
        track_id: int,
        ts: float,
        from_zone: Optional[str] = None,
        to_zone: Optional[str] = None,
        line_name: Optional[str] = None,
    ) -> None:
        """Store a zone transition or line crossing event."""
        trans_list = self._transitions_state.setdefault(sensor_id, [])
        camera_id = self.camera_labels.get(sensor_id, f"camera_{sensor_id}")
        
        event: Dict[str, Any] = {
            "track_id": track_id,
            "camera_id": camera_id,
            "timestamp": ts,
        }
        if from_zone and to_zone:
            event["from_zone"] = from_zone
            event["to_zone"] = to_zone
        elif line_name:
            event["line_name"] = line_name
        else:
            return

        trans_list.append(event)
        # Keep buffer bounded (last 100 events per sensor)
        if len(trans_list) > 100:
            self._transitions_state[sensor_id] = trans_list[-100:]

    def _cleanup_zone_state(self, sensor_id: int, active_track_ids: set[int]) -> None:
        """Remove stale zone entries for tracks no longer present."""
        state = self._zone_state.get(sensor_id)
        if not state:
            return
        for tid in list(state.keys()):
            if tid not in active_track_ids:
                state.pop(tid, None)

    def _publish_occupancy(self, sensor_id: int, occupancy_counts: Mapping[str, int]) -> None:
        logger.debug(f"DS8 occupancy counts for sensor {sensor_id}: {dict(occupancy_counts)}")
        publisher = getattr(self.pipeline, "occupancy_publisher", None)
        previous = self._occupancy_state.get(sensor_id, {})
        self._occupancy_state[sensor_id] = dict(occupancy_counts)
        if publisher is None:
            logger.debug(f"No occupancy publisher for sensor {sensor_id}")
            return

        now_ns = time.time_ns()
        try:
            for zone, count in occupancy_counts.items():
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
            for zone in set(previous.keys()) - set(occupancy_counts.keys()):
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


class _AnalyticsTelemetryOperator(BatchMetadataOperator):  # pragma: no cover - requires DeepStream runtime
    def __init__(self, processor: _AnalyticsTelemetryProcessor) -> None:
        super().__init__()
        self._processor = processor

    def handle_metadata(self, batch_meta: Any) -> None:
        if batch_meta is None:
            return

        # DS8 pyservicemaker API: batch_meta.frame_items is an iterable
        frame_items = getattr(batch_meta, "frame_items", None)
        if frame_items is not None:
            for frame_meta in frame_items:
                try:
                    self._processor.handle_frame_ds8(frame_meta)
                except Exception:
                    logger.exception("Failed to process analytics telemetry within batch metadata (DS8)")
            return

        # Fallback to DS7 pyds linked-list iteration
        cast = _resolve_pyds_cast("NvDsFrameMeta")
        for frame_meta in _iter_meta_entries(getattr(batch_meta, "frame_meta_list", None), cast):
            if frame_meta is None:
                continue
            try:
                self._processor.handle_frame(frame_meta)
            except Exception:
                logger.exception("Failed to process analytics telemetry within batch metadata")


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

        # DS8 pyservicemaker API: batch_meta.frame_items is an iterable
        frame_items = getattr(batch_meta, "frame_items", None)
        if frame_items is not None:
            for frame_meta in frame_items:
                try:
                    self._processor.handle_frame_ds8(frame_meta)
                except Exception:
                    logger.exception("Failed to prune exclusion objects within batch metadata (DS8)")
            return

        # Fallback to DS7 pyds linked-list iteration
        cast = _resolve_pyds_cast("NvDsFrameMeta")
        for frame_meta in _iter_meta_entries(getattr(batch_meta, "frame_meta_list", None), cast):
            if frame_meta is None:
                continue
            try:
                self._processor.handle_frame(frame_meta)
            except Exception:
                logger.exception("Failed to prune exclusion objects within batch metadata")


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
        # DS8 pyservicemaker API: batch_meta.frame_items is an iterable
        frame_items = getattr(batch_meta, "frame_items", None)
        if frame_items is not None:
            for frame_meta in frame_items:
                try:
                    self._processor.apply(frame_meta)
                except Exception:
                    logger.exception("Failed to apply intrinsics within batch metadata probe (DS8)")
            return
        # Fallback to DS7 pyds linked-list iteration
        if pyds is None:
            return
        l_frame = getattr(batch_meta, "frame_meta_list", None)
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                self._processor.apply(frame_meta)
            except Exception:
                logger.exception("Failed to apply intrinsics within batch metadata probe")
            try:
                l_frame = l_frame.next
            except Exception:
                break


class _MapAnythingOperator(BatchMetadataOperator):  # pragma: no cover - requires DeepStream runtime
    def __init__(self, processor: MapAnythingProcessor) -> None:
        super().__init__()
        self._processor = processor
        self._frames_seen = 0
        self._matched_frames = 0
        self._warned_no_tensors = False
        self._warned_no_match = False

    def handle_metadata(self, batch_meta: Any) -> None:
        # DS8 pyservicemaker API: batch_meta.frame_items and frame_meta.tensor_items
        frame_items = getattr(batch_meta, "frame_items", None)
        if frame_items is not None:
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
                        self._processor.handle_nvds_tensor_ds8(frame_meta, converted_items[0])
                        self._matched_frames += 1
                except Exception:
                    logger.exception("Failed to process MapAnything tensors from batch metadata (DS8)")
            return

        # Fallback to DS7 pyds linked-list iteration
        if pyds is None:
            return

        l_frame = getattr(batch_meta, "frame_meta_list", None)
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                l_user = frame_meta.frame_user_meta_list
                while l_user is not None:
                    user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                    if user_meta.base_meta.meta_type == pyds.NVDSINFER_TENSOR_OUTPUT_META:  # type: ignore[attr-defined]
                        tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                        if int(getattr(tensor_meta, "unique_id", -1)) == self._processor.gie_id:
                            self._processor.handle_nvds_tensor(frame_meta, tensor_meta)
                    try:
                        l_user = l_user.next
                    except Exception:
                        break
            except Exception:
                logger.exception("Failed to process MapAnything tensors from batch metadata")
            try:
                l_frame = l_frame.next
            except Exception:
                break


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
    roi_status = analytics_meta.get("roiStatus", analytics_meta.get("roi_status"))
    for label in _iter_roi_labels(roi_status):
        if label:
            return label
    return None


# Type checking imports (avoids circular at runtime)
from typing import TYPE_CHECKING  # noqa: E402  (import at end to satisfy linter)

if TYPE_CHECKING:  # pragma: no cover - used for type hints only
    from noesis.pipelines.ds8_pipeline import DS8Pipeline  # noqa: F401
    from noesis.telemetry.publishers import DepthTelemetryPublisher  # noqa: F401
    from noesis.telemetry.publishers import TrackingTelemetryPublisher  # noqa: F401
