"""
Utilities for decoding MapAnything SGIE tensor outputs into depth metadata.

The helpers in this module operate on plain NumPy arrays so they can be
unit-tested without the DeepStream Python bindings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import math
import numpy as np


@dataclass
class MapAnythingLayerBundle:
    depth: Optional[np.ndarray]
    confidence: Optional[np.ndarray]
    mask: Optional[np.ndarray]
    scale: Optional[float]
    pose: Optional[np.ndarray]
    extras: Dict[str, np.ndarray]


def select_layers(layers: Dict[str, np.ndarray]) -> MapAnythingLayerBundle:
    """
    Select depth/confidence/mask/scale/pose tensors from a dictionary of layers.

    Parameters
    ----------
    layers:
        Mapping of layer name → numpy array extracted from NvDsInferTensorMeta.

    Returns
    -------
    MapAnythingLayerBundle
        Structured view with any arrays that matched expected layer names.
    """

    depth = None
    confidence = None
    mask = None
    scale = None
    pose = None
    extras: Dict[str, np.ndarray] = {}

    for name, array in layers.items():
        lower = name.lower()
        if "depth" in lower:
            depth = np.asarray(array, dtype=np.float32)
        elif "conf" in lower:
            confidence = np.asarray(array, dtype=np.float32)
        elif "mask" in lower:
            mask = np.asarray(array)
        elif "scale" in lower:
            try:
                scale = float(np.asarray(array).reshape(-1)[0])
            except Exception:
                scale = None
        elif "pose" in lower:
            pose = np.asarray(array, dtype=np.float32).reshape(-1)
        else:
            extras[name] = np.asarray(array)

    return MapAnythingLayerBundle(depth, confidence, mask, scale, pose, extras)


def squeeze_hw(array: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Remove singleton batch/channel dimensions and return H×W views when possible."""
    if array is None:
        return None
    squeezed = np.squeeze(array)
    if squeezed.ndim == 3 and squeezed.shape[0] in (1, 3):
        # Handle NCHW / CHW layouts
        squeezed = squeezed[0]
    return squeezed


def compute_depth_summary(
    depth: np.ndarray,
    confidence: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    *,
    min_conf: float,
) -> Dict[str, float]:
    """
    Compute summary statistics for a depth map.

    The valid mask is derived from supplied mask/confidence thresholds.
    """
    depth = np.asarray(depth, dtype=np.float32)
    conf = np.asarray(confidence, dtype=np.float32) if confidence is not None else None
    mask_bool: Optional[np.ndarray] = None
    if mask is not None:
        mask_bool = np.asarray(mask).astype(bool)

    valid = np.isfinite(depth) & (depth > 0.0)
    if mask_bool is not None:
        valid &= mask_bool
    if conf is not None:
        valid &= conf >= float(min_conf)

    total = depth.size
    if total <= 0:
        return {
            "median": 0.0,
            "p10": 0.0,
            "p90": 0.0,
            "conf_mean": 0.0,
            "valid_ratio": 0.0,
            "sample_count": 0,
        }

    if not np.any(valid):
        conf_mean = float(np.mean(conf)) if conf is not None else 0.0
        return {
            "median": 0.0,
            "p10": 0.0,
            "p90": 0.0,
            "conf_mean": conf_mean,
            "valid_ratio": 0.0,
            "sample_count": 0,
        }

    valid_depth = depth[valid]
    conf_vals = conf[valid] if conf is not None else None
    conf_mean = float(np.mean(conf_vals)) if conf_vals is not None else 1.0

    return {
        "median": float(np.median(valid_depth)),
        "p10": float(np.percentile(valid_depth, 10)),
        "p90": float(np.percentile(valid_depth, 90)),
        "conf_mean": conf_mean,
        "valid_ratio": float(np.count_nonzero(valid) / total),
        "sample_count": int(np.count_nonzero(valid)),
    }


def anchor_to_depth_indices(
    anchor_xy: Tuple[float, float],
    rect_xywh: Tuple[float, float, float, float],
    depth_shape: Sequence[int],
) -> Tuple[float, float]:
    """
    Map an anchor in image coordinates to fractional indices in the ROI depth map.

    Returns (cx, cy) in depth-map coordinate space (floating point indices).
    """
    left, top, width, height = rect_xywh
    if width <= 0.0 or height <= 0.0:
        return 0.0, 0.0

    x_norm = (anchor_xy[0] - left) / width
    y_norm = (anchor_xy[1] - top) / height
    x_norm = float(np.clip(x_norm, 0.0, 1.0))
    y_norm = float(np.clip(y_norm, 0.0, 1.0))

    h = depth_shape[0]
    w = depth_shape[1] if len(depth_shape) > 1 else 1
    cx = x_norm * max(float(w - 1), 1.0)
    cy = y_norm * max(float(h - 1), 1.0)
    return cx, cy


def sample_depth_window(
    depth: np.ndarray,
    center_xy: Tuple[float, float],
    *,
    window_sizes: Iterable[int] = (7, 11),
    confidence: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    min_conf: float,
) -> Tuple[float, float, int]:
    """
    Sample concentric windows around the provided center to obtain a depth estimate.

    Returns (depth_m, conf_mean, sample_count). Depth is 0 when no valid samples exist.
    """
    h, w = depth.shape[:2]
    conf = np.asarray(confidence, dtype=np.float32) if confidence is not None else None
    mask_bool = mask.astype(bool) if mask is not None else None
    cx, cy = center_xy

    for window in window_sizes:
        if window <= 1:
            continue
        half = window // 2
        x0 = max(0, int(round(cx)) - half)
        y0 = max(0, int(round(cy)) - half)
        x1 = min(w, x0 + window)
        y1 = min(h, y0 + window)
        if x1 <= x0 or y1 <= y0:
            continue

        region_depth = depth[y0:y1, x0:x1]
        valid = np.isfinite(region_depth) & (region_depth > 0.0)
        if mask_bool is not None:
            region_mask = mask_bool[y0:y1, x0:x1]
            valid &= region_mask
        if conf is not None:
            region_conf = conf[y0:y1, x0:x1]
            valid &= region_conf >= float(min_conf)

        if not np.any(valid):
            continue

        valid_depth = region_depth[valid]
        depth_val = float(np.median(valid_depth))
        conf_val = (
            float(np.mean(region_conf[valid]))
            if conf is not None
            else 1.0
        )
        return depth_val, conf_val, int(np.count_nonzero(valid))

    return 0.0, 0.0, 0


def sanitize_pose(pose: Optional[np.ndarray]) -> Optional[Tuple[float, ...]]:
    """Convert pose array to a tuple of finite floats if available."""
    if pose is None:
        return None
    try:
        flat = np.asarray(pose, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if not np.all(np.isfinite(flat)):
        return None
    return tuple(float(v) for v in flat)


def sanitize_scale(scale: Optional[float]) -> Optional[float]:
    if scale is None:
        return None
    try:
        value = float(scale)
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return value
