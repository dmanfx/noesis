from __future__ import annotations

import logging
import math
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from geometry.depth_source import DepthStorageManager
from noesis.metadata import intrinsics as intrinsics_module
from noesis.metadata.depth_result import DepthResult

try:  # DeepStream imports are optional during unit tests
    from pyservicemaker import BatchMetadataOperator, Probe  # type: ignore
except Exception:  # pragma: no cover - exercised only in DS runtime
    BatchMetadataOperator = None  # type: ignore
    Probe = None  # type: ignore

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
) -> None:
    """Attach the MapAnything post-process hook to decode SGIE tensor meta."""
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

    def _apply(stage: str, cfg: Dict[str, Any]) -> None:
        with lock:
            runtime_updates = component.config.setdefault("runtime_updates", {})
            runtime_updates[stage] = cfg
            if pipeline.ds_pipeline is None:
                logger.debug("Recorded analytics runtime update for stage %s", stage)
                return
            try:
                node = pipeline.ds_pipeline[component.name]
                node.set({"config-file": component.config.get("config-file", "config/nvdsanalytics.yaml")})
                logger.info("Applied analytics runtime update for stage %s", stage)
            except Exception:  # pragma: no cover - depends on DS runtime availability
                logger.exception("Failed to push analytics runtime update for stage %s", stage)

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
    analytics_component = pipeline.components.get("analytics")
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


@dataclass
class MapAnythingProcessor:
    pipeline: "DS8Pipeline"
    storage: DepthStorageManager
    depth_pub: "DepthTelemetryPublisher" | None
    gie_id: int

    def handle_nvds_tensor(self, frame_meta: Any, tensor_meta: Any) -> Optional[DepthResult]:
        if pyds is None:
            logger.debug("pyds unavailable; skipping NvDs tensor processing")
            return None
        try:
            tensors = _extract_tensor_layers(tensor_meta)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to extract tensor layers from NvDsInferTensorMeta")
            return None
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
        return self._emit_from_tensors(frame_meta, tensors)

    def _emit_from_tensors(self, frame_meta: Any, tensors: Mapping[str, np.ndarray]) -> Optional[DepthResult]:
        if not self.pipeline.depth_enabled:
            logger.debug("Depth disabled; dropping MapAnything tensors")
            return None

        depth = _select_tensor(tensors, ("depth", "depth_z", "disp"))
        confidence = _select_tensor(tensors, ("confidence", "conf"))
        mask = _select_tensor(tensors, ("mask", "valid"))

        if depth is None:
            logger.debug("MapAnything tensors missing depth layer; skipping frame")
            return None

        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3 and depth.shape[0] in (1, 3):
            depth = depth[0]
        if depth.ndim != 2:
            logger.debug("Unexpected depth tensor shape %s; expected 2-D map", depth.shape)
            return None

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

        valid = depth[mask]
        if valid.size == 0:
            logger.debug("No valid depth pixels after masking; skipping frame")
            return None

        source_id = int(_meta_lookup(frame_meta, "pad_index", "source_id", default=0))
        frame_id = int(_meta_lookup(frame_meta, "frame_num", default=0))
        pts_ns = int(_meta_lookup(frame_meta, "buf_pts", default=0))
        ts_s = max(0, pts_ns // 1_000_000_000)
        ts_us = max(0, pts_ns // 1_000)

        camera_id = str(source_id)
        conf_array = confidence.astype(np.float32, copy=False) if confidence is not None else np.zeros_like(depth, dtype=np.float32)
        mask_u8 = mask.astype(np.uint8, copy=False)

        dest_path = self.storage.store(camera_id, ts_us, depth, conf_array, mask_u8)
        self.pipeline.record_depth_frame(time.time())

        result = DepthResult(
            source_id=source_id,
            frame_id=frame_id,
            ts=ts_s,
            width=depth.shape[1],
            height=depth.shape[0],
            depth_map_ref=str(dest_path),
            minmax=(float(np.nanmin(valid)), float(np.nanmax(valid))),
        )

        if self.depth_pub is not None:
            try:
                self.depth_pub.publish(result)
            except Exception:  # pragma: no cover - telemetry failures shouldn't break processing
                logger.exception("Depth telemetry publish failed for source %s frame %s", source_id, frame_id)

        return result


@dataclass
class _AnalyticsTelemetryProcessor:
    pipeline: "DS8Pipeline"
    tracking_pub: "TrackingTelemetryPublisher"
    camera_labels: Mapping[int, str]
    sensor_id_map: Mapping[int, int]
    _analytics_obj_meta_type: Any = field(default=None, init=False, repr=False)

    def handle_frame(self, frame_meta: Any) -> None:
        """Extract tracking telemetry for a single frame and publish it."""
        try:
            source_id = self._frame_source_id(frame_meta)
            sensor_id = self.sensor_id_map.get(source_id, source_id)
            camera_id = self.camera_labels.get(sensor_id, f"camera_{sensor_id}")

            tracks: List[Dict[str, Any]] = []
            frame_id = int(getattr(frame_meta, "frame_num", -1))

            for obj_meta in self._iter_object_meta(frame_meta):
                track = self._build_track_dict(obj_meta, camera_id)
                if track is None:
                    continue
                track["frame_id"] = frame_id
                tracks.append(track)

            if not tracks:
                return

            try:
                self.tracking_pub.publish(sensor_id, tracks)
            except Exception:  # pragma: no cover - telemetry should never break pipeline
                logger.exception("Tracking telemetry publish failed for sensor %s", sensor_id)
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

    def _iter_object_meta(self, frame_meta: Any) -> Iterable[Any]:
        cast = _resolve_pyds_cast("NvDsObjectMeta")
        entries = _iter_meta_entries(getattr(frame_meta, "obj_meta_list", None), cast)
        for entry in entries:
            if entry is None:
                continue
            yield entry

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


class _AnalyticsTelemetryOperator(BatchMetadataOperator):  # pragma: no cover - requires DeepStream runtime
    def __init__(self, processor: _AnalyticsTelemetryProcessor) -> None:
        super().__init__()
        self._processor = processor

    def handle_metadata(self, batch_meta: Any) -> None:
        if batch_meta is None:
            return

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
        cast = _resolve_pyds_cast("NvDsFrameMeta")
        for frame_meta in _iter_meta_entries(getattr(batch_meta, "frame_meta_list", None), cast):
            if frame_meta is None:
                continue
            try:
                self._processor.handle_frame(frame_meta)
            except Exception:
                logger.exception("Failed to prune exclusion objects within batch metadata")


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
        if pyds is None:
            return
        l_frame = batch_meta.frame_meta_list
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

    def handle_metadata(self, batch_meta: Any) -> None:
        if pyds is None:
            return

        l_frame = batch_meta.frame_meta_list
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
    for key in keys:
        if key in tensors:
            return tensors[key]
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
