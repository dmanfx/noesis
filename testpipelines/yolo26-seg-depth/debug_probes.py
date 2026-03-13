"""Depth capture, object fusion, and overlay probes for the seg+depth DS8 prototype."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import importlib
import logging
import math
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from pyservicemaker import BatchMetadataOperator

try:
    import torch
    import torch.utils.dlpack as torch_dlpack
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    torch_dlpack = None  # type: ignore

from noesis.calibration.geometry import pixel_to_world
from noesis.metadata.object_depth import ObjectDepthResult

LOGGER = logging.getLogger(__name__)

FrameKey = Tuple[int, int, int]
ObjectSignature = Tuple[int, int, int, int, int, int]
Point2 = Tuple[float, float]
Point3 = Tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class PersonAnchorSample:
    foot_uv: Optional[Point2]
    anchor_source: Optional[str]
    anchor_depth_m: Optional[float]
    lower_body_sample_count: int
    lower_body_valid_fraction: float
    torso_sample_count: int
    torso_valid_fraction: float


@dataclass
class PrototypeRuntimeStats:
    depth_enabled: bool
    depth_every_n_frames: int
    start_monotonic: float = field(default_factory=time.monotonic)
    last_monotonic: float = field(default_factory=time.monotonic)
    frames_total: int = 0
    detections_total: int = 0
    depth_frames_total: int = 0
    depth_reused_frames: int = 0
    depth_missing_frames: int = 0
    tensor_extract_ms_total: float = 0.0
    align_ms_total: float = 0.0
    fusion_ms_total: float = 0.0

    def note_depth_frame(self, *, tensor_extract_ms: float, align_ms: float) -> None:
        self.depth_frames_total += 1
        self.tensor_extract_ms_total += max(0.0, float(tensor_extract_ms))
        self.align_ms_total += max(0.0, float(align_ms))
        self.last_monotonic = time.monotonic()

    def note_fusion_frame(
        self,
        *,
        detections: int,
        fusion_ms: float,
        depth_state: str,
    ) -> None:
        self.frames_total += 1
        self.detections_total += max(0, int(detections))
        self.fusion_ms_total += max(0.0, float(fusion_ms))
        if depth_state == "reused":
            self.depth_reused_frames += 1
        elif depth_state == "missing":
            self.depth_missing_frames += 1
        self.last_monotonic = time.monotonic()

    def summary_line(self) -> str:
        elapsed = max(1e-6, self.last_monotonic - self.start_monotonic)
        avg_fps = float(self.frames_total) / elapsed
        avg_tensor_ms = self.tensor_extract_ms_total / float(self.depth_frames_total or 1)
        avg_align_ms = self.align_ms_total / float(self.depth_frames_total or 1)
        avg_fusion_ms = self.fusion_ms_total / float(self.frames_total or 1)
        return (
            "seg_depth_summary fps=%.2f frames=%d dets=%d depth_enabled=%s depth_every_n_frames=%d "
            "depth_frames=%d depth_reused=%d depth_missing=%d tensor_extract_ms=%.2f align_ms=%.2f fusion_ms=%.2f"
            % (
                avg_fps,
                self.frames_total,
                self.detections_total,
                int(bool(self.depth_enabled)),
                max(1, int(self.depth_every_n_frames)),
                self.depth_frames_total,
                self.depth_reused_frames,
                self.depth_missing_frames,
                avg_tensor_ms,
                avg_align_ms,
                avg_fusion_ms,
            )
        )


def _meta_lookup(meta: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(meta, Mapping) and name in meta:
            value = meta.get(name)
            if value is not None:
                return value
        if hasattr(meta, name):
            value = getattr(meta, name)
            if value is not None:
                return value
    return default


def _timestamp_us(frame_meta: Any) -> int:
    raw = int(_meta_lookup(frame_meta, "buf_pts", "buffer_pts", default=0) or 0)
    if raw <= 0 or raw < 100_000_000_000_000_000:
        raw = time.time_ns()
    return raw // 1000


def _frame_pts_key_us(frame_meta: Any) -> int:
    raw = int(_meta_lookup(frame_meta, "buf_pts", "buffer_pts", default=0) or 0)
    if raw > 0:
        return raw // 1000
    return max(0, _timestamp_us(frame_meta))


def _frame_key(frame_meta: Any) -> FrameKey:
    return (
        int(_meta_lookup(frame_meta, "source_id", "pad_index", default=0) or 0),
        int(_meta_lookup(frame_meta, "frame_number", "frame_num", default=0) or 0),
        _frame_pts_key_us(frame_meta),
    )


def _source_frame_size(frame_meta: Any) -> Tuple[int, int]:
    return (
        int(_meta_lookup(frame_meta, "source_frame_width", default=0) or 0),
        int(_meta_lookup(frame_meta, "source_frame_height", default=0) or 0),
    )


def _canonical_frame_size(frame_meta: Any, fallback_size: Tuple[int, int]) -> Tuple[int, int]:
    frame_w = int(_meta_lookup(frame_meta, "frame_width", "width", default=0) or 0)
    frame_h = int(_meta_lookup(frame_meta, "frame_height", "height", default=0) or 0)
    if frame_w > 0 and frame_h > 0:
        return frame_w, frame_h
    return max(0, int(fallback_size[0] or 0)), max(0, int(fallback_size[1] or 0))


def _rect_to_bbox(rect: Any) -> Optional[Tuple[float, float, float, float]]:
    if rect is None:
        return None
    try:
        left = float(getattr(rect, "left"))
        top = float(getattr(rect, "top"))
        width = float(getattr(rect, "width"))
        height = float(getattr(rect, "height"))
    except Exception:
        return None
    if not all(math.isfinite(v) for v in (left, top, width, height)):
        return None
    if width <= 0.0 or height <= 0.0:
        return None
    return left, top, width, height


def _tensor_to_numpy(tensor: Any) -> Optional[np.ndarray]:
    if isinstance(tensor, np.ndarray):
        return np.asarray(tensor)
    dlpack_fn = getattr(tensor, "__dlpack__", None)
    if callable(dlpack_fn) and torch is not None and torch_dlpack is not None:
        try:
            stream = 0
            if torch.cuda.is_available():
                stream = int(torch.cuda.current_stream().cuda_stream)
            capsule = dlpack_fn(stream)
            return torch_dlpack.from_dlpack(capsule).detach().cpu().numpy()
        except Exception:
            LOGGER.debug("DLPack tensor conversion failed", exc_info=True)
    return None


def _align_to_frame(depth: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    target_w, target_h = target_size
    model_h, model_w = depth.shape[:2]
    if target_w <= 0 or target_h <= 0 or (model_w == target_w and model_h == target_h):
        return depth
    if model_w <= 0 or model_h <= 0:
        return depth
    return cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def _quadrant_stats(depth_map: np.ndarray, valid_mask: np.ndarray) -> Dict[str, Dict[str, float]]:
    frame_h, frame_w = depth_map.shape[:2]
    mid_x = max(1, frame_w // 2)
    mid_y = max(1, frame_h // 2)
    windows = {
        "left": (slice(None), slice(0, mid_x)),
        "right": (slice(None), slice(mid_x, None)),
        "top": (slice(0, mid_y), slice(None)),
        "bottom": (slice(mid_y, None), slice(None)),
    }
    stats: Dict[str, Dict[str, float]] = {}
    for name, window in windows.items():
        local_valid = valid_mask[window]
        finite_count = int(np.count_nonzero(local_valid))
        total = int(local_valid.size) or 1
        values = depth_map[window][local_valid]
        stats[name] = {
            "finite_fraction": float(finite_count) / float(total),
            "mean": float(np.mean(values)) if finite_count > 0 else float("nan"),
        }
    return stats


def _erode_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    if mask.size <= 0 or not np.any(mask):
        return np.asarray(mask, dtype=bool)
    kernel_size = max(1, int(kernel_size))
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
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
    x0 = max(0, min(width - 1, int(math.floor(center_x - (band_width * 0.5)))))
    x1 = max(x0 + 1, min(width, int(math.ceil(center_x + (band_width * 0.5)))))
    region = np.zeros_like(mask, dtype=bool)
    region[y0:y1, x0:x1] = True
    return np.logical_and(np.asarray(mask, dtype=bool), region)


def _sample_band_values(depth_crop: np.ndarray, sample_mask: np.ndarray) -> Tuple[np.ndarray, int, float]:
    mask_area = int(np.count_nonzero(sample_mask))
    if mask_area <= 0:
        return np.empty(0, dtype=np.float32), 0, 0.0
    valid_mask = np.logical_and(sample_mask, np.isfinite(depth_crop))
    values = np.asarray(depth_crop[valid_mask], dtype=np.float32)
    sample_count = int(values.size)
    valid_fraction = float(sample_count) / float(mask_area) if mask_area > 0 else 0.0
    return values, sample_count, valid_fraction


def _extract_person_anchor(mask: np.ndarray, depth_crop: np.ndarray, *, frame_origin: Tuple[int, int]) -> PersonAnchorSample:
    x0, y0 = int(frame_origin[0]), int(frame_origin[1])
    base_mask = np.asarray(mask, dtype=bool)
    if base_mask.size <= 0 or not np.any(base_mask):
        return PersonAnchorSample(
            foot_uv=None,
            anchor_source=None,
            anchor_depth_m=None,
            lower_body_sample_count=0,
            lower_body_valid_fraction=0.0,
            torso_sample_count=0,
            torso_valid_fraction=0.0,
        )

    lower_body_mask = _band_mask(base_mask, y0_ratio=0.88, y1_ratio=1.0, center_width_ratio=0.35)
    lower_coords = np.argwhere(lower_body_mask)
    foot_uv: Optional[Point2] = None
    if lower_coords.size > 0:
        max_row = int(np.max(lower_coords[:, 0]))
        foot_cols = lower_coords[lower_coords[:, 0] == max_row][:, 1]
        foot_uv = (float(x0) + float(np.median(foot_cols)) + 0.5, float(y0) + float(max_row) + 0.5)

    lower_values, lower_count, lower_valid_fraction = _sample_band_values(depth_crop, lower_body_mask)
    if lower_count >= 64 and lower_valid_fraction >= 0.5:
        return PersonAnchorSample(
            foot_uv=foot_uv,
            anchor_source="lower_body_band",
            anchor_depth_m=float(np.median(lower_values)),
            lower_body_sample_count=lower_count,
            lower_body_valid_fraction=lower_valid_fraction,
            torso_sample_count=0,
            torso_valid_fraction=0.0,
        )

    torso_mask = _band_mask(_erode_mask(base_mask), y0_ratio=0.35, y1_ratio=0.70, center_width_ratio=0.50)
    torso_values, torso_count, torso_valid_fraction = _sample_band_values(depth_crop, torso_mask)
    anchor_depth_m = float(np.median(torso_values)) if torso_count > 0 else None
    anchor_source = "torso_core" if anchor_depth_m is not None else None
    return PersonAnchorSample(
        foot_uv=foot_uv,
        anchor_source=anchor_source,
        anchor_depth_m=anchor_depth_m,
        lower_body_sample_count=lower_count,
        lower_body_valid_fraction=lower_valid_fraction,
        torso_sample_count=torso_count,
        torso_valid_fraction=torso_valid_fraction,
    )


def _ground_plane_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[2]) - float(b[2]))


def _object_signature(obj_meta: Any) -> ObjectSignature:
    bbox = _rect_to_bbox(getattr(obj_meta, "rect_params", None)) or (0.0, 0.0, 0.0, 0.0)
    try:
        object_id = int(getattr(obj_meta, "object_id", -1))
    except Exception:
        object_id = -1
    try:
        class_id = int(getattr(obj_meta, "class_id", -1))
    except Exception:
        class_id = -1
    return (
        object_id,
        class_id,
        int(round(bbox[0])),
        int(round(bbox[1])),
        int(round(bbox[2])),
        int(round(bbox[3])),
    )


def _depth_text(result: ObjectDepthResult) -> str:
    if result.status == "ok" and result.depth_median is not None:
        return f"z={result.depth_median:.2f}{result.unit}"
    return "z=n/a"


def _strip_depth_suffix(label: str) -> str:
    current = str(label or "").strip()
    if " z=" in current:
        current = current.split(" z=", 1)[0].strip()
    if " depth=" in current:
        current = current.split(" depth=", 1)[0].strip()
    return current


def _base_label(obj_meta: Any, text_params: Any, result: ObjectDepthResult) -> str:
    for attr in ("obj_label", "label"):
        try:
            current = _strip_depth_suffix(getattr(obj_meta, attr, "") or "")
        except Exception:
            current = ""
        if current:
            return current
    current = _strip_depth_suffix(getattr(text_params, "display_text", "") or "")
    if current:
        return current
    return f"cls {result.class_id} {result.score:.2f}"


def _text_offsets(text_params: Any, bbox: Tuple[float, float, float, float]) -> Tuple[int, int]:
    x = int(getattr(text_params, "x_offset", getattr(text_params, "xOffset", int(round(bbox[0])))) or int(round(bbox[0])))
    y = int(getattr(text_params, "y_offset", getattr(text_params, "yOffset", max(0, int(round(bbox[1])) - 4))) or max(0, int(round(bbox[1])) - 4))
    return x, y


def _set_text_offsets(text_params: Any, *, x: int, y: int) -> None:
    if hasattr(text_params, "x_offset"):
        setattr(text_params, "x_offset", int(x))
    elif hasattr(text_params, "xOffset"):
        setattr(text_params, "xOffset", int(x))
    if hasattr(text_params, "y_offset"):
        setattr(text_params, "y_offset", int(y))
    elif hasattr(text_params, "yOffset"):
        setattr(text_params, "yOffset", int(y))


def _font_size(text_params: Any) -> int:
    font_params = getattr(text_params, "font_params", None)
    if font_params is None:
        return 16
    return int(getattr(font_params, "font_size", getattr(font_params, "size", 16)) or 16)


def _apply_depth_text(obj_meta: Any, result: ObjectDepthResult, *, frame_size: Tuple[int, int]) -> None:
    text_params = getattr(obj_meta, "text_params", None)
    if text_params is None:
        return
    label = f"{_base_label(obj_meta, text_params, result)} {_depth_text(result)}".strip()
    setattr(text_params, "display_text", label)
    for attr in ("obj_label", "label"):
        try:
            if hasattr(obj_meta, attr):
                setattr(obj_meta, attr, label)
        except Exception:
            continue
    bbox = tuple(float(x) for x in result.bbox)
    x_offset, y_offset = _text_offsets(text_params, bbox)
    frame_w, frame_h = frame_size
    font_size = max(12, _font_size(text_params))
    estimated_width = int(max(font_size * 4, round(font_size * 0.58 * len(label))))
    x_offset = max(2, min(x_offset, max(2, frame_w - estimated_width - 4)))
    y_offset = max(font_size + 2, min(y_offset, max(font_size + 2, frame_h - 4)))
    _set_text_offsets(text_params, x=x_offset, y=y_offset)


@dataclass
class AlignedDepthFrame:
    key: FrameKey
    source_id: int
    frame_id: int
    pts_us: int
    depth_map: np.ndarray
    valid_mask: np.ndarray
    frame_w: int
    frame_h: int
    source_frame_w: int
    source_frame_h: int
    depth_w: int
    depth_h: int
    unit: str
    is_metric: bool
    model_name: str
    finite_fraction: float
    quadrant_stats: Dict[str, Dict[str, float]]
    transform_desc: str


class AlignedDepthFrameStore:
    def __init__(self, max_entries: int = 8) -> None:
        self._max_entries = max(2, int(max_entries))
        self._entries: "OrderedDict[FrameKey, AlignedDepthFrame]" = OrderedDict()

    def put(self, frame: AlignedDepthFrame) -> None:
        self._entries[frame.key] = frame
        self._entries.move_to_end(frame.key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def get(self, key: FrameKey) -> Optional[AlignedDepthFrame]:
        return self._entries.get(key)

    def resolve(
        self,
        *,
        source_id: int,
        frame_id: int,
        pts_us: int,
        max_age_frames: int,
    ) -> Tuple[Optional[AlignedDepthFrame], int, float]:
        exact = self.get((int(source_id), int(frame_id), int(pts_us)))
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


class OverlayStateStore:
    def __init__(self, max_entries: int = 8) -> None:
        self._max_entries = max(2, int(max_entries))
        self._entries: "OrderedDict[FrameKey, Dict[ObjectSignature, ObjectDepthResult]]" = OrderedDict()

    def put(self, key: FrameKey, payload: Dict[ObjectSignature, ObjectDepthResult]) -> None:
        self._entries[key] = dict(payload)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def pop(self, key: FrameKey) -> Dict[ObjectSignature, ObjectDepthResult]:
        return self._entries.pop(key, {})


class DepthFrameProbe(BatchMetadataOperator):
    """Capture and align full-frame depth into canonical frame coordinates."""

    def __init__(
        self,
        *,
        depth_store: AlignedDepthFrameStore,
        stats: PrototypeRuntimeStats,
        depth_gie_id: int,
        depth_model_name: str,
        depth_unit: str,
        depth_is_metric: bool,
        fallback_frame_size: Tuple[int, int],
    ) -> None:
        super().__init__()
        self._depth_store = depth_store
        self._stats = stats
        self._depth_gie_id = int(depth_gie_id)
        self._depth_model_name = str(depth_model_name)
        self._depth_unit = str(depth_unit)
        self._depth_is_metric = bool(depth_is_metric)
        self._fallback_frame_size = (int(fallback_frame_size[0]), int(fallback_frame_size[1]))
        self._warned_no_tensors = False

    def _extract_depth_tensor(self, frame_meta: Any) -> Optional[np.ndarray]:
        tensor_items = getattr(frame_meta, "tensor_items", None)
        if tensor_items is None:
            if not self._warned_no_tensors:
                LOGGER.warning("Depth frame probe could not find frame.tensor_items")
                self._warned_no_tensors = True
            return None
        for item in list(tensor_items):
            convert_fn = getattr(item, "as_tensor_output", None)
            if callable(convert_fn):
                try:
                    item = convert_fn()
                except Exception:
                    continue
            try:
                unique_id = int(getattr(item, "unique_id", -1))
            except Exception:
                unique_id = -1
            if unique_id != self._depth_gie_id:
                continue
            try:
                layers = item.get_layers() or {}
            except Exception:
                continue
            if not isinstance(layers, Mapping) or not layers:
                continue
            tensor = None
            for layer_name in ("depth", "pred", "output"):
                candidate = layers.get(layer_name)
                if candidate is not None:
                    tensor = candidate
                    break
            if tensor is None:
                tensor = next(iter(layers.values()))
            arr = _tensor_to_numpy(tensor)
            if arr is None:
                return None
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim != 2:
                LOGGER.warning("Depth frame probe got unexpected tensor shape: %s", getattr(arr, "shape", None))
                return None
            return arr
        return None

    def handle_metadata(self, batch_meta: Any) -> None:  # type: ignore[override]
        frame_items = getattr(batch_meta, "frame_items", None)
        if frame_items is None:
            return
        for frame_meta in frame_items:
            extract_start = time.perf_counter()
            depth_raw = self._extract_depth_tensor(frame_meta)
            extract_ms = (time.perf_counter() - extract_start) * 1000.0
            if depth_raw is None:
                continue
            frame_w, frame_h = _canonical_frame_size(frame_meta, self._fallback_frame_size)
            if frame_w <= 0 or frame_h <= 0:
                LOGGER.warning("Depth frame probe missing canonical frame size for key=%s", _frame_key(frame_meta))
                continue
            align_start = time.perf_counter()
            aligned = _align_to_frame(depth_raw, (frame_w, frame_h))
            align_ms = (time.perf_counter() - align_start) * 1000.0
            valid_mask = np.isfinite(aligned)
            source_frame_w, source_frame_h = _source_frame_size(frame_meta)
            source_id = int(_meta_lookup(frame_meta, "source_id", "pad_index", default=0) or 0)
            frame_id = int(_meta_lookup(frame_meta, "frame_number", "frame_num", default=0) or 0)
            pts_us = _frame_pts_key_us(frame_meta)
            frame = AlignedDepthFrame(
                key=_frame_key(frame_meta),
                source_id=source_id,
                frame_id=frame_id,
                pts_us=pts_us,
                depth_map=np.asarray(aligned, dtype=np.float32, copy=False),
                valid_mask=np.asarray(valid_mask, dtype=bool, copy=False),
                frame_w=frame_w,
                frame_h=frame_h,
                source_frame_w=source_frame_w,
                source_frame_h=source_frame_h,
                depth_w=int(depth_raw.shape[1]),
                depth_h=int(depth_raw.shape[0]),
                unit=self._depth_unit,
                is_metric=self._depth_is_metric,
                model_name=self._depth_model_name,
                finite_fraction=float(np.count_nonzero(valid_mask)) / float(valid_mask.size or 1),
                quadrant_stats=_quadrant_stats(np.asarray(aligned, dtype=np.float32, copy=False), valid_mask),
                transform_desc=f"resize({depth_raw.shape[1]}x{depth_raw.shape[0]}->{frame_w}x{frame_h})",
            )
            self._depth_store.put(frame)
            self._stats.note_depth_frame(tensor_extract_ms=extract_ms, align_ms=align_ms)


class ObjectDepthFusionProbe(BatchMetadataOperator):
    """Fuse aligned frame depth with segment masks and attach object depth meta."""

    def __init__(
        self,
        *,
        depth_store: Optional[AlignedDepthFrameStore],
        overlay_store: OverlayStateStore,
        stats: PrototypeRuntimeStats,
        depth_model_name: str,
        depth_unit: str,
        depth_is_metric: bool,
        depth_every_n_frames: int = 1,
        calibration_resolver: Any | None = None,
        report_interval_s: float = 1.0,
    ) -> None:
        super().__init__()
        self._depth_store = depth_store
        self._overlay_store = overlay_store
        self._stats = stats
        self._depth_model_name = str(depth_model_name)
        self._depth_unit = str(depth_unit)
        self._depth_is_metric = bool(depth_is_metric)
        self._depth_every_n_frames = max(1, int(depth_every_n_frames))
        self._calibration_resolver = calibration_resolver
        self._report_interval_s = max(0.2, float(report_interval_s))
        self._last_report = time.monotonic()
        self._frames_since_report = 0
        self._depth_meta_ext = importlib.import_module("noesis_depth_meta_ext")

    def _camera_id_for_source(self, source_id: int) -> Optional[str]:
        resolver = self._calibration_resolver
        if resolver is None:
            return None
        labels_fn = getattr(resolver, "camera_labels", None)
        if callable(labels_fn):
            try:
                labels = labels_fn() or {}
                camera_id = labels.get(int(source_id))
                if camera_id:
                    return str(camera_id)
            except Exception:
                LOGGER.debug("Calibration label lookup failed", exc_info=True)
        return None

    def _resolve_calibration_snapshot(self, source_id: int) -> Any | None:
        resolver = self._calibration_resolver
        if resolver is None:
            return None
        camera_id = self._camera_id_for_source(source_id)
        snapshot_fn = getattr(resolver, "snapshot", None)
        if not callable(snapshot_fn):
            return None
        try:
            return snapshot_fn(int(source_id), camera_id)
        except TypeError:
            try:
                return snapshot_fn(int(source_id))
            except Exception:
                LOGGER.debug("Calibration snapshot lookup failed", exc_info=True)
                return None
        except Exception:
            LOGGER.debug("Calibration snapshot lookup failed", exc_info=True)
            return None

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
        spatial_fields: Optional[Mapping[str, Any]] = None,
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
        values = np.asarray(values, dtype=np.float32) if values is not None else np.empty(0, dtype=np.float32)
        has_values = bool(values.size)
        payload: Dict[str, Any] = dict(
            source_id=int(_meta_lookup(frame_meta, "source_id", "pad_index", default=0) or 0),
            frame_id=int(_meta_lookup(frame_meta, "frame_number", "frame_num", default=0) or 0),
            object_id=object_id,
            class_id=class_id,
            bbox=bbox,
            score=score,
            sampling_mode="instance_mask",
            status=status,
            unit=self._depth_unit,
            is_metric=self._depth_is_metric,
            sample_count=max(0, int(sample_count)),
            valid_fraction=max(0.0, min(1.0, float(valid_fraction))),
            depth_center=depth_center,
            depth_median=float(np.median(values)) if has_values else None,
            depth_mean=float(np.mean(values)) if has_values else None,
            depth_p10=float(np.percentile(values, 10.0)) if has_values else None,
            depth_p90=float(np.percentile(values, 90.0)) if has_values else None,
            depth_min=float(np.min(values)) if has_values else None,
            depth_max=float(np.max(values)) if has_values else None,
            mask_area_px=max(0, int(mask_area_px)),
            model=self._depth_model_name,
            ts_us=max(0, _timestamp_us(frame_meta)),
        )
        if spatial_fields:
            payload.update({str(key): value for key, value in spatial_fields.items()})
        return ObjectDepthResult(**payload)

    def _build_person_spatial_fields(
        self,
        frame_meta: Any,
        obj_meta: Any,
        *,
        bbox: Tuple[float, float, float, float],
        depth_crop: np.ndarray,
        mask: np.ndarray,
    ) -> Dict[str, Any]:
        source_id = int(_meta_lookup(frame_meta, "source_id", "pad_index", default=0) or 0)
        snapshot = self._resolve_calibration_snapshot(source_id)
        if snapshot is None:
            return {
                "spatial_class": "person",
                "spatial_status": "geometry_unavailable",
            }

        left, top, _, _ = bbox
        anchor = _extract_person_anchor(
            mask,
            depth_crop,
            frame_origin=(int(math.floor(left)), int(math.floor(top))),
        )
        fields: Dict[str, Any] = {
            "spatial_class": "person",
            "spatial_status": "anchor_unavailable",
        }
        if anchor.foot_uv is not None:
            fields["anchor_uv"] = anchor.foot_uv
        if anchor.anchor_source is not None:
            fields["anchor_source"] = anchor.anchor_source
        if anchor.anchor_depth_m is not None:
            fields["anchor_depth_m"] = anchor.anchor_depth_m
        if anchor.foot_uv is None:
            return fields

        foot_u, foot_v = anchor.foot_uv
        floor_proj = pixel_to_world(
            snapshot.intrinsics,
            snapshot.extrinsics_col_major,
            snapshot.floor_y,
            snapshot.unit_scale,
            foot_u,
            foot_v,
            None,
        )
        world_point_floor: Optional[Point3] = None
        if bool(getattr(floor_proj, "ok", False)):
            floor_point = getattr(floor_proj, "world_point", None)
            if isinstance(floor_point, Sequence) and len(floor_point) == 3:
                world_point_floor = (float(floor_point[0]), float(floor_point[1]), float(floor_point[2]))
                fields["world_point_floor"] = world_point_floor

        world_point_depth: Optional[Point3] = None
        if anchor.anchor_depth_m is not None:
            depth_proj = pixel_to_world(
                snapshot.intrinsics,
                snapshot.extrinsics_col_major,
                snapshot.floor_y,
                snapshot.unit_scale,
                foot_u,
                foot_v,
                anchor.anchor_depth_m,
            )
            if bool(getattr(depth_proj, "ok", False)):
                depth_point = getattr(depth_proj, "world_point", None)
                if isinstance(depth_point, Sequence) and len(depth_point) == 3:
                    world_point_depth = (float(depth_point[0]), float(depth_point[1]), float(depth_point[2]))
                    fields["world_point_depth"] = world_point_depth

        if world_point_depth is not None and world_point_floor is not None:
            if _ground_plane_distance(world_point_depth, world_point_floor) <= 1.0:
                fields["world_point"] = world_point_depth
                fields["projection_method"] = "depth"
                fields["spatial_status"] = "ok"
            else:
                fields["world_point"] = world_point_floor
                fields["projection_method"] = "floor_guarded"
                fields["spatial_status"] = "ok"
            return fields
        if world_point_depth is not None:
            fields["world_point"] = world_point_depth
            fields["projection_method"] = "depth_only"
            fields["spatial_status"] = "ok"
            return fields
        if world_point_floor is not None:
            fields["world_point"] = world_point_floor
            fields["projection_method"] = "floor_only"
            fields["spatial_status"] = "ok"
            return fields
        fields["spatial_status"] = "projection_unavailable"
        return fields

    def _decode_instance_mask(self, obj_meta: Any, target_shape: Tuple[int, int]) -> Tuple[Optional[np.ndarray], str]:
        try:
            payload = self._depth_meta_ext.extract_object_mask(obj_meta)
        except Exception:
            LOGGER.debug("Native object-mask extraction failed", exc_info=True)
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

    def _sample_result(
        self,
        frame_meta: Any,
        obj_meta: Any,
        depth_frame: AlignedDepthFrame,
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

        depth_crop = np.asarray(depth_frame.depth_map[y0:y1, x0:x1], dtype=np.float32, copy=False)
        if depth_crop.size <= 0:
            return self._build_result(frame_meta, obj_meta, bbox=bbox, status="transform_mismatch")

        mask, mask_status = self._decode_instance_mask(obj_meta, depth_crop.shape)
        if mask is None:
            return self._build_result(frame_meta, obj_meta, bbox=bbox, status=mask_status)

        mask_area = int(np.count_nonzero(mask))
        try:
            class_id = int(getattr(obj_meta, "class_id", -1))
        except Exception:
            class_id = -1
        cx = max(0, min(frame_w - 1, int(round(left + (width * 0.5)))))
        cy = max(0, min(frame_h - 1, int(round(top + (height * 0.5)))))
        center_value = float(depth_frame.depth_map[cy, cx]) if np.isfinite(depth_frame.depth_map[cy, cx]) else None
        spatial_fields: Optional[Dict[str, Any]] = None
        if class_id == 0:
            spatial_fields = self._build_person_spatial_fields(
                frame_meta,
                obj_meta,
                bbox=bbox,
                depth_crop=depth_crop,
                mask=mask,
            )

        if mask_area <= 0:
            return self._build_result(
                frame_meta,
                obj_meta,
                bbox=bbox,
                status="missing_mask",
                mask_area_px=0,
                depth_center=center_value,
                spatial_fields=spatial_fields,
            )

        valid_mask = np.logical_and(mask, np.isfinite(depth_crop))
        values = depth_crop[valid_mask]
        sample_count = int(values.size)
        if sample_count <= 0:
            return self._build_result(
                frame_meta,
                obj_meta,
                bbox=bbox,
                status="no_valid_depth",
                mask_area_px=mask_area,
                depth_center=center_value,
                spatial_fields=spatial_fields,
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
            spatial_fields=spatial_fields,
        )

    def _emit_report(
        self,
        frame_meta: Any,
        *,
        detection_count: int,
        depth_frame: Optional[AlignedDepthFrame],
        results: Sequence[ObjectDepthResult],
        depth_state: str,
        depth_age_frames: int,
        depth_age_ms: float,
        fusion_ms: float,
    ) -> None:
        now = time.monotonic()
        self._frames_since_report += 1
        delta = now - self._last_report
        if delta < self._report_interval_s:
            return
        fps = float(self._frames_since_report) / delta if delta > 0.0 else 0.0
        if depth_frame is None:
            frame_summary = f"depth={depth_state}"
        else:
            q = depth_frame.quadrant_stats
            frame_summary = (
                "depth=finite%.2f left%.2f right%.2f top%.2f bottom%.2f xform=%s src=%sx%s frame=%sx%s state=%s agef=%d agems=%.1f"
                % (
                    depth_frame.finite_fraction,
                    q["left"]["finite_fraction"],
                    q["right"]["finite_fraction"],
                    q["top"]["finite_fraction"],
                    q["bottom"]["finite_fraction"],
                    depth_frame.transform_desc,
                    depth_frame.source_frame_w,
                    depth_frame.source_frame_h,
                    depth_frame.frame_w,
                    depth_frame.frame_h,
                    depth_state,
                    depth_age_frames,
                    depth_age_ms,
                )
            )
        parts: List[str] = []
        for result in results:
            world_point = result.world_point
            world_x = f"{world_point[0]:.2f}" if world_point is not None else "n/a"
            world_z = f"{world_point[2]:.2f}" if world_point is not None else "n/a"
            anchor_depth = f"{result.anchor_depth_m:.2f}m" if result.anchor_depth_m is not None else "n/a"
            projection_method = result.projection_method or "n/a"
            spatial_status = result.spatial_status or "n/a"
            parts.append(
                "cls=%s score=%.2f status=%s med=%s center=%s anchor=%s proj=%s spatial=%s x=%s z=%s mask=%s samples=%d bbox=(%.0f,%.0f,%.0f,%.0f)"
                % (
                    result.class_id,
                    result.score,
                    result.status,
                    f"{result.depth_median:.2f}{result.unit}" if result.depth_median is not None else "n/a",
                    f"{result.depth_center:.2f}{result.unit}" if result.depth_center is not None else "n/a",
                    anchor_depth,
                    projection_method,
                    spatial_status,
                    world_x,
                    world_z,
                    result.mask_area_px if result.mask_area_px is not None else "n/a",
                    result.sample_count,
                    result.bbox[0],
                    result.bbox[1],
                    result.bbox[2],
                    result.bbox[3],
                )
            )
        LOGGER.info(
            "seg_depth_debug fps=%.2f frame=%s source=%s dets=%d fusion_ms=%.2f %s %s",
            fps,
            int(_meta_lookup(frame_meta, "frame_number", "frame_num", default=0) or 0),
            int(_meta_lookup(frame_meta, "source_id", "pad_index", default=0) or 0),
            detection_count,
            fusion_ms,
            frame_summary,
            " | ".join(parts) if parts else "no detections",
        )
        self._frames_since_report = 0
        self._last_report = now

    def handle_metadata(self, batch_meta: Any) -> None:  # type: ignore[override]
        frame_items = getattr(batch_meta, "frame_items", None)
        if frame_items is None:
            return
        for frame_meta in frame_items:
            frame_start = time.perf_counter()
            source_id = int(_meta_lookup(frame_meta, "source_id", "pad_index", default=0) or 0)
            frame_id = int(_meta_lookup(frame_meta, "frame_number", "frame_num", default=0) or 0)
            pts_us = _frame_pts_key_us(frame_meta)
            depth_state = "disabled"
            depth_age_frames = 0
            depth_age_ms = 0.0
            depth_frame: Optional[AlignedDepthFrame] = None
            if self._depth_store is not None:
                depth_frame, depth_age_frames, depth_age_ms = self._depth_store.resolve(
                    source_id=source_id,
                    frame_id=frame_id,
                    pts_us=pts_us,
                    max_age_frames=max(0, self._depth_every_n_frames - 1),
                )
                if depth_frame is None:
                    depth_state = "missing"
                elif depth_age_frames > 0:
                    depth_state = "reused"
                else:
                    depth_state = "exact"
            key = _frame_key(frame_meta)
            overlay_payload: Dict[ObjectSignature, ObjectDepthResult] = {}
            results: List[ObjectDepthResult] = []
            detection_count = 0
            for obj_meta in getattr(frame_meta, "object_items", None) or []:
                bbox = _rect_to_bbox(getattr(obj_meta, "rect_params", None))
                if bbox is None:
                    continue
                detection_count += 1
                if self._depth_store is None:
                    continue
                if depth_frame is None:
                    result = self._build_result(frame_meta, obj_meta, bbox=bbox, status="depth_not_ready")
                else:
                    result = self._sample_result(frame_meta, obj_meta, depth_frame)
                if result is None:
                    continue
                self._depth_meta_ext.attach_object_depth(obj_meta, result.to_json(), True)
                if depth_frame is not None:
                    _apply_depth_text(obj_meta, result, frame_size=(depth_frame.frame_w, depth_frame.frame_h))
                overlay_payload[_object_signature(obj_meta)] = result
                results.append(result)
            self._overlay_store.put(key, overlay_payload)
            fusion_ms = (time.perf_counter() - frame_start) * 1000.0
            self._stats.note_fusion_frame(
                detections=detection_count,
                fusion_ms=fusion_ms,
                depth_state=depth_state,
            )
            self._emit_report(
                frame_meta,
                detection_count=detection_count,
                depth_frame=depth_frame,
                results=results,
                depth_state=depth_state,
                depth_age_frames=depth_age_frames,
                depth_age_ms=depth_age_ms,
                fusion_ms=fusion_ms,
            )
            if depth_frame is None and results:
                LOGGER.warning(
                    "Object depth fusion missing aligned depth frame for frame=%s source=%s",
                    int(_meta_lookup(frame_meta, "frame_number", "frame_num", default=0) or 0),
                    int(_meta_lookup(frame_meta, "source_id", "pad_index", default=0) or 0),
                )


class DepthOverlayProbe(BatchMetadataOperator):
    """Apply compact depth text after fusion and clamp it to the visible frame."""

    def __init__(
        self,
        *,
        overlay_store: OverlayStateStore,
        fallback_frame_size: Tuple[int, int],
    ) -> None:
        super().__init__()
        self._overlay_store = overlay_store
        self._fallback_frame_size = (int(fallback_frame_size[0]), int(fallback_frame_size[1]))

    def handle_metadata(self, batch_meta: Any) -> None:  # type: ignore[override]
        frame_items = getattr(batch_meta, "frame_items", None)
        if frame_items is None:
            return
        for frame_meta in frame_items:
            key = _frame_key(frame_meta)
            render_state = self._overlay_store.pop(key)
            if not render_state:
                continue
            frame_size = _canonical_frame_size(frame_meta, self._fallback_frame_size)
            for obj_meta in getattr(frame_meta, "object_items", None) or []:
                result = render_state.get(_object_signature(obj_meta))
                if result is None:
                    continue
                _apply_depth_text(obj_meta, result, frame_size=frame_size)
