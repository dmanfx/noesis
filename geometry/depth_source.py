"""HTTP client and storage utilities for MapAnything depth inference."""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import os
import queue
import shutil
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Tuple, Sequence

from concurrent.futures import Future

import cv2
import numpy as np
import requests
import scipy.ndimage as ndi
import zarr

from adapters.mapanything_adapter import ViewBuildResult, build_mono_view
from geometry.homography import parse_extrinsics
from mapanything_config import ServiceConfig, load_service_config
from utils.rate_limited_logger import RateLimitedLogger

try:  # zarr v3 codec shim
    from zarr.codecs import Blosc as _ZarrBlosc  # type: ignore
except Exception:
    _ZarrBlosc = None  # type: ignore

try:
    from numcodecs import Blosc as _NumcodecsBlosc  # type: ignore
except Exception:
    _NumcodecsBlosc = None  # type: ignore

_FLOORPLAN_FRAME = "camera_local_ground_m"
_FLOORPLAN_ORIENTATION = "camera_xz_forward"
# Floorplan grids are camera-local X/Z products. They must not inherit the
# image-axis flip heuristic used by BEV/world projection consumers.
_FLOORPLAN_CONTRACT_VERSION = 7
_FLOORPLAN_MIN_HALF_WIDTH_FRACTION = 0.0
_FLOORPLAN_MIN_FORWARD_FRACTION = 0.0


def _scene_units_per_meter_from_calibration_bundle(calib_bundle: Optional[Mapping[str, Any]]) -> Tuple[float, float]:
    """Return (scene_per_m, s_obj_to_m) from calibration bundle align units."""
    try:
        align = (calib_bundle or {}).get("align") if isinstance(calib_bundle, dict) else {}
        units = align.get("units") if isinstance(align, dict) else {}
        s_obj_to_m = float((units or {}).get("s_obj_to_m", 1.0) or 1.0)
        if not np.isfinite(s_obj_to_m) or s_obj_to_m <= 1e-9:
            return 1.0, 1.0
        return float(1.0 / s_obj_to_m), float(s_obj_to_m)
    except Exception:
        return 1.0, 1.0


def _expected_floorplan_units_from_calibration_bundle(calib_bundle: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Return expected floorplan units label for current calibration contract."""
    if not isinstance(calib_bundle, dict):
        return None
    try:
        meta = calib_bundle.get("meta")
        if isinstance(meta, dict):
            world_frame = str(meta.get("world_frame") or "").strip().lower()
            if world_frame == "backend_world_m":
                return "meters"
            meta_units = meta.get("units")
            if isinstance(meta_units, str):
                tag = meta_units.strip().lower()
                if tag in ("m", "meter", "meters"):
                    return "meters"
                if tag in ("obj_units", "scene", "scene_units", "scene_obj"):
                    return "scene"
            elif isinstance(meta_units, dict):
                coords = str(meta_units.get("coords") or "").strip().lower()
                if coords in ("backend_world_m", "world_m", "meters", "m"):
                    return "meters"
                if coords in ("scene", "obj_units", "scene_obj", "obj"):
                    return "scene"
        align = calib_bundle.get("align")
        units = align.get("units") if isinstance(align, dict) else None
        s_obj_to_m = float((units or {}).get("s_obj_to_m", 1.0) or 1.0) if isinstance(units, dict) else 1.0
        if isinstance(meta, dict):
            if str(meta.get("coord_space") or "").strip().lower() == "backend_world_m":
                return "meters"
        if np.isfinite(s_obj_to_m) and s_obj_to_m > 1e-9 and abs(s_obj_to_m - 1.0) > 1e-9:
            return "scene"
    except Exception:
        return None
    return None


def _floorplan_calibration_fingerprint(
    calib_bundle: Optional[Mapping[str, Any]],
    camera_id: str,
) -> Optional[str]:
    """Compact cache key for the calibration fields that shape floorplan grids."""
    if not isinstance(calib_bundle, Mapping) or not camera_id:
        return None
    try:
        cameras = calib_bundle.get("cameras")
        if not isinstance(cameras, Mapping):
            return None
        k_table = cameras.get("K")
        e_table = cameras.get("E")
        k_value = k_table.get(camera_id) if isinstance(k_table, Mapping) else None
        e_value = e_table.get(camera_id) if isinstance(e_table, Mapping) else None
        if k_value is None and e_value is None:
            return None
        align = calib_bundle.get("align")
        meta = calib_bundle.get("meta")
        payload = {
            "contract": "floorplan_calibration_v1",
            "camera_id": camera_id,
            "K": k_value,
            "E": e_value,
            "align": {
                "floor_y": align.get("floor_y") if isinstance(align, Mapping) else None,
                "units": align.get("units") if isinstance(align, Mapping) else None,
            },
            "meta": {
                "world_frame": meta.get("world_frame") if isinstance(meta, Mapping) else None,
                "coord_space": meta.get("coord_space") if isinstance(meta, Mapping) else None,
                "units": meta.get("units") if isinstance(meta, Mapping) else None,
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
    except Exception:
        return None


def _floorplan_cache_contract_matches(
    payload: Mapping[str, Any],
    *,
    expected_units: Optional[str] = None,
    expected_calibration_fingerprint: Optional[str] = None,
) -> bool:
    if payload.get("frame") != _FLOORPLAN_FRAME:
        return False
    if payload.get("orientation") != _FLOORPLAN_ORIENTATION:
        return False
    try:
        if int(payload.get("floorplan_contract_version", 0)) != int(_FLOORPLAN_CONTRACT_VERSION):
            return False
    except Exception:
        return False
    if expected_units:
        units_val = str(payload.get("units") or "").strip().lower()
        if units_val != str(expected_units).strip().lower():
            return False
    if expected_calibration_fingerprint:
        stored = str(payload.get("calibration_fingerprint") or "").strip()
        if stored != str(expected_calibration_fingerprint).strip():
            return False
    return True

# Clean floorplan layers (obstacle_height, walkable) shared by every camera.
_FLOORPLAN_CLEAN_FLOOR_SEED_PERCENTILE = 5.0
_FLOORPLAN_CLEAN_FLOOR_SEED_BAND_M = 0.06
# In camera coordinates the ground plane is typically *not* close to horizontal because the
# camera is pitched down. Allow a much larger slope here; we still clamp to avoid degenerate
# fits when depth is pathological.
_FLOORPLAN_CLEAN_PLANE_MAX_SLOPE = 0.6
_FLOORPLAN_CLEAN_PLANE_INLIER_TOL_M = 0.05
_FLOORPLAN_CLEAN_OBSTACLE_THRESH_M = 0.15
# Ignore very tall geometry (ceiling/walls) when computing obstacle footprint.
# This is a walkability map, not a full 3D reconstruction.
# Ignore very tall geometry (upper cabinets/ceiling) when computing obstacle footprint.
_FLOORPLAN_CLEAN_MAX_RELEVANT_H_M = 1.3
_FLOORPLAN_CLEAN_MAX_OBSTACLE_H_M = 1.8
_FLOORPLAN_CLEAN_MIN_SUPPORT = 2
_FLOORPLAN_CLEAN_MIN_OBSTACLE_SUPPORT = 1
# Support thresholds for classifying obstacle vs floor cells.
# These are separate from the plane-fit seed band; we want looser floor support (noise-tolerant)
# and a higher obstacle support threshold (ignore tiny above-floor noise).
_FLOORPLAN_CLEAN_FLOOR_SUPPORT_BAND_M = 0.12
_FLOORPLAN_CLEAN_OBSTACLE_SUPPORT_MIN_M = 0.35
_FLOORPLAN_CLEAN_OBSTACLE_RATIO_MIN = 0.20
_FLOORPLAN_CLEAN_TOP_BAND_M = 0.10
_FLOORPLAN_CLEAN_TOP_RATIO_MIN = 0.20
_FLOORPLAN_CLEAN_OBS_FLAT_MIN_POINTS = 3
_FLOORPLAN_CLEAN_OBS_STD_MAX_M = 0.25
# Kitchen BEV goal: highlight large, mostly-horizontal obstacles (island/counters/tabletop) and
# treat everything else inside the footprint as walkable floor (including occluded areas).
_FLOORPLAN_CLEAN_FLAT_OBS_MIN_H_M = 0.25
_FLOORPLAN_CLEAN_FLAT_OBS_MAX_DELTA_M = 0.20
_FLOORPLAN_CLEAN_MIN_COMPONENT_CELLS = 6
_FLOORPLAN_CLEAN_MORPH_SIZE = 3
_FLOORPLAN_CLEAN_BASELINE_MIN_ROW_SAMPLES = 8
_FLOORPLAN_CLEAN_BASELINE_SMOOTH_ROWS = 7
_FLOORPLAN_CLEAN_HEIGHT_SMOOTH = 3
_FLOORPLAN_CLEAN_OTSU_BINS = 128

# Height-above-ground (AGL) clean floorplan layers.
# These rely on estimating a floor Y from horizontal surfaces in each room.
_FLOORPLAN_AGL_HORIZ_DOT_THRESH = 0.85
_FLOORPLAN_AGL_HEIGHT_CLIP_M = 2.5
_FLOORPLAN_AGL_FLOOR_SEED_MAX_M = 0.08
_FLOORPLAN_AGL_OBSTACLE_THRESH_M = 0.25
_FLOORPLAN_AGL_FILL_RADIUS_M = 1.5
# Support-based classification (AGL).
# We count how many points in each BEV cell land near the floor vs above a threshold.
# This is materially more robust than using per-cell mean AGL, which gets dominated by
# vertical surfaces (walls/cabinets) and causes "everything is obstacle" failure modes.
_FLOORPLAN_AGL_FLOOR_SUPPORT_BAND_M = 0.12
_FLOORPLAN_AGL_OBSTACLE_SUPPORT_MIN_M = 0.35
_FLOORPLAN_AGL_OBSTACLE_RANGE_MIN_M = 0.25
_FLOORPLAN_AGL_FLOOR_SUPPORT_RATIO_MIN = 0.20
_FLOORPLAN_AGL_OBSTACLE_SUPPORT_RATIO_MIN = 0.25
_FLOORPLAN_AGL_MIN_SUPPORT_POINTS = 3
_FLOORPLAN_AGL_FLOOR_SUPPORT_MIN_POINTS = 2
_FLOORPLAN_AGL_OBSTACLE_SUPPORT_MIN_POINTS = 2
_FLOORPLAN_AGL_FLOOR_HIST_BINS = 256
_FLOORPLAN_AGL_FLOOR_SEGMENT_MIN_MASS_FRAC = 0.0005
_FLOORPLAN_AGL_FLOOR_SEGMENT_BIN_THRESH_FRAC = 0.0005
_FLOORPLAN_AGL_FLOOR_SEGMENT_TOTAL_THRESH_FRAC = 0.0003
_CORE_COUNTER_FN = None
_CORE_COUNTER_RESOLVED = False


def _increment_core_boundary_copy_bytes(path: str, payload_bytes: int) -> None:
    """Best-effort bridge into DS8 core counters without import-time cycles."""
    global _CORE_COUNTER_FN, _CORE_COUNTER_RESOLVED
    delta = max(0, int(payload_bytes))
    if delta <= 0:
        return
    if not _CORE_COUNTER_RESOLVED:
        _CORE_COUNTER_RESOLVED = True
        try:
            from noesis.pipelines import hooks as _hooks  # Local import to avoid circular import at module load.

            fn = getattr(_hooks, "_increment_core_counter", None)
            _CORE_COUNTER_FN = fn if callable(fn) else None
        except Exception:
            _CORE_COUNTER_FN = None
    if _CORE_COUNTER_FN is None:
        return
    try:
        _CORE_COUNTER_FN(f"tensor_boundary_copy_bytes_total.{str(path)}", delta)
    except Exception:
        pass


def _infer_image_flips_from_extrinsics(
    extrinsics_col_major: Sequence[float],
) -> Optional[Tuple[bool, bool]]:
    try:
        R_wc, _ = parse_extrinsics(extrinsics_col_major)
        forward = R_wc @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        f_norm = float(np.linalg.norm(forward))
        if f_norm > 1e-6:
            forward = forward / f_norm
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right_ref = np.cross(world_up, forward)
        r_norm = float(np.linalg.norm(right_ref))
        if r_norm <= 1e-6:
            right_ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            right_ref = right_ref / r_norm
        up_ref = np.cross(forward, right_ref)
        u_norm = float(np.linalg.norm(up_ref))
        if u_norm <= 1e-6:
            up_ref = world_up
        else:
            up_ref = up_ref / u_norm

        right = R_wc @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        up = R_wc @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
        r_actual = float(np.dot(right, right_ref))
        u_actual = float(np.dot(up, up_ref))

        flip_u = r_actual < 0.0
        flip_v = u_actual > 0.0
        return bool(flip_u), bool(flip_v)
    except Exception:
        return None


def _expected_floorplan_flip(
    calib_bundle: Any,
    camera_id: str,
) -> Optional[Tuple[bool, bool]]:
    if not isinstance(calib_bundle, dict):
        return None
    cameras_node = calib_bundle.get("cameras")
    if not isinstance(cameras_node, dict):
        return None
    e_table = cameras_node.get("E") if isinstance(cameras_node, dict) else None
    extr = e_table.get(camera_id) if isinstance(e_table, dict) else None
    if not isinstance(extr, (list, tuple)) or len(extr) != 16:
        return None
    return _infer_image_flips_from_extrinsics(extr)


def _floorplan_image_flip_payload(
    expected_flip: Optional[Tuple[bool, bool]],
) -> Dict[str, bool]:
    """Expose the inferred image flip as diagnostics only.

    Floorplan grids are already serialized in their final camera-local X/Z
    orientation, so consumers must not apply this hint back onto the raster.
    """
    return {
        "u": bool(expected_flip[0]) if expected_flip is not None else False,
        "v": bool(expected_flip[1]) if expected_flip is not None else False,
    }


def _floorplan_world_coordinate_grids(
    min_x: float,
    max_x: float,
    min_z: float,
    max_z: float,
    rows: int,
    cols: int,
) -> Tuple[np.ndarray, np.ndarray]:
    safe_rows = max(1, int(rows))
    safe_cols = max(1, int(cols))
    width_m = max(1e-6, float(max_x) - float(min_x))
    depth_m = max(1e-6, float(max_z) - float(min_z))
    x_step = width_m / safe_cols
    z_step = depth_m / safe_rows
    xs = float(min_x) + (np.arange(safe_cols, dtype=np.float32) + 0.5) * x_step
    zs = float(max_z) - (np.arange(safe_rows, dtype=np.float32) + 0.5) * z_step
    return np.meshgrid(xs, zs)


def _postprocess_floorplan_height_grid(
    camera_id: str,
    height_grid: np.ndarray,
    density_grid: Optional[np.ndarray],
    *,
    min_x: float,
    max_x: float,
    min_z: float,
    max_z: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    height = np.asarray(height_grid, dtype=np.float32).copy()
    valid_mask = np.isfinite(height)
    if not np.any(valid_mask):
        height.fill(0.0)
        return (
            height,
            {
                "mode": "empty_floor_fallback",
                "filled_cells": int(height.size),
                "floor_level_m": 0.0,
            },
        )

    # Legacy behavior: fill only truly empty cells (NaN) with the global minimum height.
    # Keep this behavior stable because the existing "Height (Inferno)" view relies on it.
    nan_mask = np.isnan(height)
    filled_cells = int(np.count_nonzero(nan_mask))
    if filled_cells:
        min_height = float(np.nanmin(height))
        height = np.nan_to_num(height, nan=min_height)
    if not np.isfinite(height).all():
        fallback = float(np.nanmin(height)) if np.any(np.isfinite(height)) else 0.0
        height = np.nan_to_num(height, nan=fallback, posinf=fallback, neginf=fallback)
    return (
        height,
        {
            "mode": "legacy_min_fill",
            "filled_cells": filled_cells,
            "floor_level_m": float(np.min(height)) if height.size else 0.0,
        },
    )


def _fit_floor_plane_from_points(
    x_cam_pts: np.ndarray,
    z_cam_pts: np.ndarray,
    y_world_pts: np.ndarray,
    pts_weight: Optional[np.ndarray],
) -> Tuple[float, float, float, Dict[str, Any]]:
    """Fit a global floor plane y ~= a*x + b*z + c from the lowest-percentile points.

    Returns (a, b, c, meta). Inputs are per-point values; x/z in camera frame, y in world frame.
    """
    x = np.asarray(x_cam_pts, dtype=np.float32).ravel()
    z = np.asarray(z_cam_pts, dtype=np.float32).ravel()
    y = np.asarray(y_world_pts, dtype=np.float32).ravel()
    if pts_weight is None:
        w = np.ones_like(y, dtype=np.float32)
    else:
        w = np.asarray(pts_weight, dtype=np.float32).ravel()

    n = min(x.size, z.size, y.size, w.size)
    if n <= 0:
        return 0.0, 0.0, 0.0, {"mode": "no_points", "seed_count": 0}
    x = x[:n]
    z = z[:n]
    y = y[:n]
    w = w[:n]

    finite = np.isfinite(x) & np.isfinite(z) & np.isfinite(y) & np.isfinite(w)
    if not np.any(finite):
        return 0.0, 0.0, 0.0, {"mode": "no_finite_points", "seed_count": 0}
    x = x[finite]
    z = z[finite]
    y = y[finite]
    w = w[finite]

    def _fit_from_seed(seed: np.ndarray, label: str, y_p: float) -> Optional[Tuple[float, float, float, Dict[str, Any]]]:
        seed_count = int(np.count_nonzero(seed))
        if seed_count < 3:
            return None

        sx = x[seed]
        sz = z[seed]
        sy = y[seed]
        sw = np.clip(w[seed], 1e-4, None)
        sqrtw = np.sqrt(sw).astype(np.float32, copy=False)
        design = np.column_stack((sx, sz, np.ones_like(sx, dtype=np.float32)))
        try:
            coeffs, _, _, _ = np.linalg.lstsq(design * sqrtw[:, None], sy * sqrtw, rcond=None)
            if not (isinstance(coeffs, np.ndarray) and coeffs.size == 3 and np.isfinite(coeffs).all()):
                raise ValueError("plane_fit_invalid")
        except Exception:
            return None

        a0 = float(coeffs[0])
        b0 = float(coeffs[1])
        c0 = float(coeffs[2])
        a = float(np.clip(a0, -_FLOORPLAN_CLEAN_PLANE_MAX_SLOPE, _FLOORPLAN_CLEAN_PLANE_MAX_SLOPE))
        b = float(np.clip(b0, -_FLOORPLAN_CLEAN_PLANE_MAX_SLOPE, _FLOORPLAN_CLEAN_PLANE_MAX_SLOPE))
        c = float(c0)

        residual = y - ((a * x) + (b * z) + c)
        tol = float(_FLOORPLAN_CLEAN_PLANE_INLIER_TOL_M)
        inlier = np.isfinite(residual) & (np.abs(residual) <= tol)
        score = float(np.sum(w[inlier])) if np.any(inlier) else 0.0
        return a, b, c, {
            "mode": "lstsq",
            "seed": label,
            "y_p": float(y_p),
            "seed_count": seed_count,
            "inlier_score": float(score),
            "slope_clamped": bool((a != a0) or (b != b0)),
        }

    p = float(_FLOORPLAN_CLEAN_FLOOR_SEED_PERCENTILE)
    p_hi = 100.0 - p
    try:
        y_p_low = float(np.percentile(y, p))
    except Exception:
        y_p_low = float(np.min(y)) if y.size else 0.0
    try:
        y_p_high = float(np.percentile(y, p_hi))
    except Exception:
        y_p_high = float(np.max(y)) if y.size else y_p_low

    low_seed = y <= (y_p_low + _FLOORPLAN_CLEAN_FLOOR_SEED_BAND_M)
    high_seed = y >= (y_p_high - _FLOORPLAN_CLEAN_FLOOR_SEED_BAND_M)

    cand_low = _fit_from_seed(low_seed, "low", y_p_low)
    cand_high = _fit_from_seed(high_seed, "high", y_p_high)

    if cand_low is None and cand_high is None:
        # Fallback to whichever extreme has more samples, to at least anchor to a surface.
        low_count = int(np.count_nonzero(low_seed))
        high_count = int(np.count_nonzero(high_seed))
        y_p = y_p_high if high_count >= low_count else y_p_low
        return 0.0, 0.0, float(y_p), {
            "mode": "percentile_fallback",
            "y_p_low": float(y_p_low),
            "y_p_high": float(y_p_high),
            "seed_count_low": low_count,
            "seed_count_high": high_count,
        }

    # For walkability we want the *floor* plane, not the largest planar surface (countertops/ceilings).
    # Prefer the lower-height seed (y-percentile low) when available; fall back to the high seed.
    if cand_low is not None:
        return cand_low
    return cand_high  # type: ignore[return-value]


def _remove_small_components(mask: np.ndarray, min_cells: int) -> np.ndarray:
    if min_cells <= 1:
        return mask
    labeled, num = ndi.label(mask)
    if num <= 0:
        return mask
    counts = np.bincount(labeled.ravel())
    remove = counts < int(min_cells)
    if remove.size:
        remove[0] = False
    return mask & ~remove[labeled]


def _otsu_threshold(values: np.ndarray, *, bins: int) -> float:
    """Return an Otsu threshold for non-negative float values."""
    vals = np.asarray(values, dtype=np.float32).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size < 32:
        return float(np.percentile(vals, 75)) if vals.size else 0.0
    vmax = float(np.percentile(vals, 99))
    if not np.isfinite(vmax) or vmax <= 1e-6:
        vmax = float(np.max(vals)) if vals.size else 0.0
    if not np.isfinite(vmax) or vmax <= 1e-6:
        return 0.0
    vals = np.clip(vals, 0.0, vmax)
    hist, edges = np.histogram(vals, bins=int(max(16, bins)), range=(0.0, vmax))
    hist = hist.astype(np.float64, copy=False)
    total = float(np.sum(hist))
    if total <= 0.0:
        return 0.0
    prob = hist / total
    omega = np.cumsum(prob)
    centers = (edges[:-1] + edges[1:]) * 0.5
    mu = np.cumsum(prob * centers)
    mu_t = float(mu[-1]) if mu.size else 0.0
    denom = omega * (1.0 - omega) + 1e-12
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    try:
        idx = int(np.nanargmax(sigma_b))
    except Exception:
        idx = 0
    idx = max(0, min(idx, centers.size - 1))
    return float(centers[idx]) if centers.size else 0.0


def _estimate_floor_y_from_horizontal_points(
    y_world: np.ndarray,
    normals_world: np.ndarray,
    weights: Optional[np.ndarray] = None,
    *,
    horiz_dot_thresh: float = _FLOORPLAN_AGL_HORIZ_DOT_THRESH,
) -> Tuple[float, Dict[str, Any]]:
    """Estimate a global floor Y from points that lie on horizontal surfaces.

    We intentionally pick the *lowest* horizontal surface mode so that countertop-dominant
    views still anchor to the floor when any floor is visible.
    """
    y = np.asarray(y_world, dtype=np.float32).ravel()
    nrm = np.asarray(normals_world, dtype=np.float32).reshape(-1, 3)
    n = min(y.size, nrm.shape[0])
    if n <= 0:
        return 0.0, {"mode": "no_points", "candidate_count": 0}
    y = y[:n]
    nrm = nrm[:n]

    if weights is None:
        w = np.ones((n,), dtype=np.float32)
    else:
        w = np.asarray(weights, dtype=np.float32).ravel()[:n]
        if w.size != n:
            w = np.ones((n,), dtype=np.float32)

    finite = np.isfinite(y) & np.isfinite(w) & np.isfinite(nrm).all(axis=1)
    if not np.any(finite):
        return 0.0, {"mode": "no_finite", "candidate_count": 0, "point_count": int(n)}

    # |dot(up, normal)| near 1 means a horizontal surface (floor/countertop/table/ceiling).
    dot_up = np.abs(nrm[:, 1]).astype(np.float32, copy=False)
    cand = finite & (dot_up >= float(horiz_dot_thresh))
    cand_count = int(np.count_nonzero(cand))
    meta: Dict[str, Any] = {
        "mode": "histogram",
        "point_count": int(n),
        "candidate_count": cand_count,
        "horiz_dot_thresh": float(horiz_dot_thresh),
    }
    if cand_count < 64:
        # Too few horizontal points; fall back to a low percentile of all points.
        y0 = y[finite]
        try:
            floor_y = float(np.percentile(y0, 2.0))
        except Exception:
            floor_y = float(np.min(y0)) if y0.size else 0.0
        meta["mode"] = "fallback_percentile"
        meta["floor_y"] = float(floor_y)
        return float(floor_y), meta

    y_cand = y[cand]
    w_cand = np.clip(w[cand], 0.0, None).astype(np.float32, copy=False)
    try:
        y_lo = float(np.percentile(y_cand, 0.5))
        y_hi = float(np.percentile(y_cand, 99.5))
    except Exception:
        y_lo = float(np.min(y_cand)) if y_cand.size else 0.0
        y_hi = float(np.max(y_cand)) if y_cand.size else y_lo + 1.0
    if not np.isfinite(y_lo) or not np.isfinite(y_hi) or y_hi <= y_lo + 1e-6:
        y_lo = float(np.min(y_cand)) if y_cand.size else 0.0
        y_hi = float(np.max(y_cand)) if y_cand.size else y_lo + 1.0
    if not np.isfinite(y_hi) or y_hi <= y_lo + 1e-6:
        floor_y = float(y_lo)
        meta["mode"] = "degenerate_range"
        meta["floor_y"] = float(floor_y)
        return float(floor_y), meta

    bins = int(max(32, _FLOORPLAN_AGL_FLOOR_HIST_BINS))
    hist, edges = np.histogram(y_cand, bins=bins, range=(y_lo, y_hi), weights=w_cand)
    hist = hist.astype(np.float64, copy=False)
    # Light smoothing to merge adjacent bins.
    hist_s = np.convolve(hist, np.array([1.0, 2.0, 1.0], dtype=np.float64), mode="same")
    total = float(np.sum(hist_s))
    peak = float(np.max(hist_s)) if hist_s.size else 0.0
    if not np.isfinite(total) or total <= 0.0 or not np.isfinite(peak) or peak <= 0.0:
        floor_y = float(np.min(y_cand)) if y_cand.size else 0.0
        meta["mode"] = "empty_hist"
        meta["floor_y"] = float(floor_y)
        return float(floor_y), meta

    centers = (edges[:-1] + edges[1:]) * 0.5
    meta.update(
        {
            "y_lo": float(y_lo),
            "y_hi": float(y_hi),
            "bins": int(bins),
            "hist_total": float(total),
            "hist_peak": float(peak),
        }
    )

    # Find local maxima and pick the *lowest* significant peak. We use a fixed bin window
    # around the peak to avoid the \"everything is one segment\" failure mode when a tall
    # surface (countertop) dominates the histogram.
    peaks = []
    if hist_s.size >= 3:
        mids = (hist_s[1:-1] > hist_s[:-2]) & (hist_s[1:-1] >= hist_s[2:])
        peaks = (np.flatnonzero(mids) + 1).tolist()
    if hist_s.size >= 2:
        if hist_s[0] >= hist_s[1]:
            peaks.insert(0, 0)
        if hist_s[-1] >= hist_s[-2]:
            peaks.append(int(hist_s.size - 1))
    if not peaks:
        peaks = [int(np.argmax(hist_s))]

    peaks_sorted = sorted(set(int(p) for p in peaks), key=lambda i: float(centers[i]) if 0 <= i < centers.size else 0.0)

    peak_thr = float(peak) * float(_FLOORPLAN_AGL_FLOOR_SEGMENT_BIN_THRESH_FRAC)
    min_mass = float(total) * float(_FLOORPLAN_AGL_FLOOR_SEGMENT_MIN_MASS_FRAC)
    radius = 2

    chosen = None
    chosen_window = None
    for idx in peaks_sorted:
        if idx < 0 or idx >= hist_s.size:
            continue
        if float(hist_s[idx]) < peak_thr:
            continue
        i0 = max(0, int(idx) - radius)
        i1 = min(int(hist_s.size - 1), int(idx) + radius)
        win_mass = float(np.sum(hist_s[i0 : i1 + 1]))
        if not np.isfinite(win_mass) or win_mass < min_mass:
            continue
        chosen = int(idx)
        chosen_window = (int(i0), int(i1), float(win_mass))
        break

    if chosen is None or chosen_window is None:
        try:
            floor_y = float(np.percentile(y_cand, 2.0))
        except Exception:
            floor_y = float(np.min(y_cand)) if y_cand.size else 0.0
        meta["mode"] = "fallback_low_percentile"
        meta["floor_y"] = float(floor_y)
        meta["peak_threshold"] = float(peak_thr)
        meta["window_min_mass"] = float(min_mass)
        return float(floor_y), meta

    i0, i1, win_mass = chosen_window
    y0 = float(edges[i0])
    y1 = float(edges[i1 + 1]) if (i1 + 1) < edges.size else float(edges[-1])
    in_band = (y_cand >= y0) & (y_cand <= y1)
    if int(np.count_nonzero(in_band)) >= 32:
        try:
            floor_y = float(np.average(y_cand[in_band], weights=w_cand[in_band]))
        except Exception:
            floor_y = float(np.median(y_cand[in_band]))
    else:
        floor_y = float(centers[chosen]) if centers.size else float(y0)

    meta.update(
        {
            "mode": "histogram_lowest_peak",
            "peak_threshold": float(peak_thr),
            "window_min_mass": float(min_mass),
            "peak_bin": int(chosen),
            "peak_center": float(centers[chosen]) if centers.size else float(y0),
            "peak_height": float(hist_s[chosen]),
            "window_bins": [int(i0), int(i1)],
            "window_mass": float(win_mass),
            "window_range": [float(y0), float(y1)],
            "floor_y": float(floor_y),
        }
    )
    return float(floor_y), meta


def _compute_inside_mask_from_observed(observed: np.ndarray) -> np.ndarray:
    """Best-effort room footprint from observed support cells.

    The goal is to bridge occluded floor behind big objects (kitchen island) without
    defaulting to a full rectangle. This is intentionally heuristic.
    """
    observed = np.asarray(observed, dtype=bool)
    h_px, w_px = observed.shape if hasattr(observed, "shape") else (0, 0)
    inside = np.zeros((h_px, w_px), dtype=bool)
    if h_px <= 0 or w_px <= 0:
        return inside
    if not np.any(observed):
        return inside

    min_row_support = 1
    left = np.full(h_px, -1, dtype=np.int32)
    right = np.full(h_px, -1, dtype=np.int32)
    for r in range(h_px):
        cols = np.flatnonzero(observed[r])
        if cols.size >= min_row_support:
            left[r] = int(cols[0])
            right[r] = int(cols[-1])
    valid_rows = np.flatnonzero(left >= 0)
    if valid_rows.size:
        first = int(valid_rows[0])
        last = int(valid_rows[-1])
        max_extend_rows = 4
        start = 0 if first <= max_extend_rows else first
        end = (h_px - 1) if ((h_px - 1) - last) <= max_extend_rows else last

        rows = np.arange(start, end + 1, dtype=np.float32)
        left_i = np.interp(rows, valid_rows.astype(np.float32), left[valid_rows].astype(np.float32))
        right_i = np.interp(rows, valid_rows.astype(np.float32), right[valid_rows].astype(np.float32))
        left_s = ndi.median_filter(left_i, size=7).astype(np.int32, copy=False)
        right_s = ndi.median_filter(right_i, size=7).astype(np.int32, copy=False)
        left_s = np.clip(left_s, 0, w_px - 1)
        right_s = np.clip(right_s, 0, w_px - 1)
        for idx, r in enumerate(range(int(start), int(end) + 1)):
            l = int(left_s[idx])
            rr = int(right_s[idx])
            if rr < l:
                continue
            inside[r, l: rr + 1] = True

        # Add convex hull to reduce concavity.
        try:
            oy, ox = np.where(observed)
            if oy.size >= 3:
                pts = np.stack([ox, oy], axis=1).astype(np.int32, copy=False)
                hull = cv2.convexHull(pts)
                hull_mask = np.zeros((h_px, w_px), dtype=np.uint8)
                cv2.fillConvexPoly(hull_mask, hull, 1)
                inside |= hull_mask.astype(bool)
        except Exception:
            pass

        # Fill small gaps/holes inside footprint.
        fp_struct = np.ones((7, 7), dtype=bool)
        fp_pad = int(fp_struct.shape[0] // 2)
        padded_fp = np.pad(inside, pad_width=fp_pad, mode="constant", constant_values=False)
        padded_fp = ndi.binary_closing(padded_fp, structure=fp_struct, border_value=0)
        inside = padded_fp[fp_pad:-fp_pad, fp_pad:-fp_pad]
        inside = ndi.binary_fill_holes(inside)
        labeled, num = ndi.label(inside)
        if num > 0:
            seed = (h_px - 1, w_px // 2)
            seed_label = int(labeled[seed])
            if seed_label == 0:
                counts = np.bincount(labeled.ravel())
                if counts.size:
                    counts[0] = 0
                    seed_label = int(np.argmax(counts))
            if seed_label > 0:
                inside = labeled == seed_label

    if not np.any(inside):
        inside = observed.copy()
    return inside


def _compute_kitchen_clean_floorplan_layers_from_grids(
    camera_id: str,
    *,
    height_grid: np.ndarray,
    support_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return (obstacle_height_grid, walkable_grid, meta) using only rasterized grids.

    This avoids relying on world-frame point heights being well-calibrated. Instead, we:
    - build an inside footprint from observed support,
    - estimate a per-row floor baseline (low percentile) to remove tilt/slope,
    - threshold height-above-baseline via Otsu to get obstacles,
    - mark unobserved-but-inside as walkable to "extend floor" through occlusions.
    """
    support = np.asarray(support_grid, dtype=np.uint32)
    h_px, w_px = support.shape if hasattr(support, "shape") else (0, 0)
    if h_px <= 0 or w_px <= 0:
        obstacle_height = np.zeros((1, 1), dtype=np.float32)
        walkable = np.ones((1, 1), dtype=np.float32)
        return obstacle_height, walkable, {"mode": "bad_support_grid"}

    height = np.asarray(height_grid, dtype=np.float32)
    if height.shape != (h_px, w_px):
        obstacle_height = np.zeros((h_px, w_px), dtype=np.float32)
        walkable = np.ones((h_px, w_px), dtype=np.float32)
        return obstacle_height, walkable, {"mode": "shape_mismatch", "height_shape": list(getattr(height, "shape", []))}

    observed = support > 0
    inside = _compute_inside_mask_from_observed(observed)
    finite_h = np.isfinite(height)
    observed_inside = observed & inside & finite_h

    # Estimate floor baseline per row from low percentile of observed heights.
    baseline = np.full(h_px, np.nan, dtype=np.float32)
    min_row = int(_FLOORPLAN_CLEAN_BASELINE_MIN_ROW_SAMPLES)
    pct = float(_FLOORPLAN_CLEAN_FLOOR_SEED_PERCENTILE)
    for r in range(h_px):
        row_mask = observed_inside[r]
        if int(np.count_nonzero(row_mask)) >= min_row:
            try:
                baseline[r] = float(np.percentile(height[r, row_mask], pct))
            except Exception:
                baseline[r] = np.nan
    valid_rows = np.flatnonzero(np.isfinite(baseline))
    if valid_rows.size:
        rows = np.arange(h_px, dtype=np.float32)
        baseline_i = np.interp(rows, valid_rows.astype(np.float32), baseline[valid_rows].astype(np.float32)).astype(np.float32)
    else:
        baseline_i = np.zeros(h_px, dtype=np.float32)
    smooth = int(_FLOORPLAN_CLEAN_BASELINE_SMOOTH_ROWS)
    baseline_s = ndi.median_filter(baseline_i, size=max(1, smooth)).astype(np.float32, copy=False) if smooth > 1 else baseline_i

    height_above = height - baseline_s[:, None]
    height_above = np.where(finite_h, height_above, 0.0).astype(np.float32, copy=False)
    height_above = np.clip(height_above, 0.0, None)
    if int(_FLOORPLAN_CLEAN_HEIGHT_SMOOTH) > 1:
        # Smooth small noise while keeping discontinuities.
        filtered = ndi.median_filter(height_above, size=int(_FLOORPLAN_CLEAN_HEIGHT_SMOOTH)).astype(np.float32, copy=False)
        height_above = np.where(inside, filtered, height_above).astype(np.float32, copy=False)

    samples = height_above[observed_inside]
    clip_max = float(np.percentile(samples, 99)) if samples.size else 0.0
    if not np.isfinite(clip_max) or clip_max <= 1e-6:
        clip_max = float(np.max(samples)) if samples.size else 0.0
    if not np.isfinite(clip_max) or clip_max <= 1e-6:
        clip_max = 0.0
    if clip_max > 0.0:
        samples = np.clip(samples, 0.0, clip_max)

    thr = _otsu_threshold(samples, bins=int(_FLOORPLAN_CLEAN_OTSU_BINS))
    # Prevent thresholds that collapse to ~0 when the histogram is dominated by floor cells.
    thr = max(float(thr), float(clip_max) * 0.05) if clip_max > 0.0 else float(thr)

    min_support = int(_FLOORPLAN_CLEAN_MIN_SUPPORT)
    obstacle_mask = inside & observed & (support >= min_support) & (height_above > float(thr))
    obstacle_mask = _remove_small_components(obstacle_mask, int(_FLOORPLAN_CLEAN_MIN_COMPONENT_CELLS))
    structure = np.ones((_FLOORPLAN_CLEAN_MORPH_SIZE, _FLOORPLAN_CLEAN_MORPH_SIZE), dtype=bool)
    pad = max(0, int(_FLOORPLAN_CLEAN_MORPH_SIZE) // 2)
    if pad > 0:
        padded = np.pad(obstacle_mask, pad_width=pad, mode="constant", constant_values=False)
        padded = ndi.binary_closing(padded, structure=structure, border_value=0)
        obstacle_mask = padded[pad:-pad, pad:-pad]
    else:
        obstacle_mask = ndi.binary_closing(obstacle_mask, structure=structure, border_value=0)
    obstacle_mask &= inside

    obstacle_height = np.zeros((h_px, w_px), dtype=np.float32)
    if np.any(obstacle_mask):
        max_obs = float(np.percentile(height_above[obstacle_mask], 95)) if np.any(obstacle_mask) else 0.0
        if not np.isfinite(max_obs) or max_obs <= 1e-6:
            max_obs = float(np.max(height_above[obstacle_mask])) if np.any(obstacle_mask) else 0.0
        if not np.isfinite(max_obs) or max_obs <= 1e-6:
            max_obs = float(clip_max)
        if not np.isfinite(max_obs) or max_obs <= 1e-6:
            max_obs = 1.0
        obstacle_height[obstacle_mask] = np.clip(height_above[obstacle_mask], 0.0, max_obs).astype(np.float32, copy=False)
    else:
        max_obs = 0.0

    walkable = np.zeros((h_px, w_px), dtype=np.float32)
    walkable[inside] = 1.0
    walkable[obstacle_mask] = 0.0

    meta: Dict[str, Any] = {
        "mode": "kitchen_clean_layers_grid",
        "baseline": {
            "percentile": float(pct),
            "min_row_samples": int(min_row),
            "smooth_rows": int(smooth),
        },
        "height_above": {
            "smooth_size": int(_FLOORPLAN_CLEAN_HEIGHT_SMOOTH),
            "clip_max_p99": float(clip_max),
            "otsu_bins": int(_FLOORPLAN_CLEAN_OTSU_BINS),
            "threshold": float(thr),
        },
        "cells": {
            "observed": int(np.count_nonzero(observed)),
            "inside": int(np.count_nonzero(inside)),
            "inside_unobserved": int(np.count_nonzero(inside & ~observed)),
            "obstacle": int(np.count_nonzero(obstacle_mask)),
            "walkable": int(np.count_nonzero(walkable > 0.5)),
        },
        "thresholds": {
            "min_support": int(min_support),
            "min_component_cells": int(_FLOORPLAN_CLEAN_MIN_COMPONENT_CELLS),
            "morph_size": int(_FLOORPLAN_CLEAN_MORPH_SIZE),
        },
        "camera_id": str(camera_id),
    }
    return obstacle_height, walkable, meta


def _compute_kitchen_clean_floorplan_layers_from_agl_grids(
    camera_id: str,
    *,
    height_agl_min_grid: np.ndarray,
    height_agl_max_grid: np.ndarray,
    support_grid: np.ndarray,
    floor_support_grid: np.ndarray,
    obstacle_support_grid: np.ndarray,
    grid_res_m: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return (obstacle_height_grid, walkable_grid, meta) from height-above-floor grids.

    The intent is a clean floorplan for BEV: large obstacles vs walkable floor, while
    also extending floor through short occlusions (behind an island) without painting
    the entire footprint as walkable.
    """
    support = np.asarray(support_grid, dtype=np.uint32)
    h_px, w_px = support.shape if hasattr(support, "shape") else (0, 0)
    if h_px <= 0 or w_px <= 0:
        obstacle_height = np.zeros((1, 1), dtype=np.float32)
        walkable = np.ones((1, 1), dtype=np.float32)
        return obstacle_height, walkable, {"mode": "bad_support_grid"}

    hmin = np.asarray(height_agl_min_grid, dtype=np.float32)
    hmax = np.asarray(height_agl_max_grid, dtype=np.float32)
    floor_support = np.asarray(floor_support_grid, dtype=np.uint32)
    obstacle_support = np.asarray(obstacle_support_grid, dtype=np.uint32)
    if (
        hmin.shape != (h_px, w_px)
        or hmax.shape != (h_px, w_px)
        or floor_support.shape != (h_px, w_px)
        or obstacle_support.shape != (h_px, w_px)
    ):
        obstacle_height = np.zeros((h_px, w_px), dtype=np.float32)
        walkable = np.ones((h_px, w_px), dtype=np.float32)
        return obstacle_height, walkable, {
            "mode": "shape_mismatch",
            "height_agl_min_shape": list(getattr(hmin, "shape", [])),
            "height_agl_max_shape": list(getattr(hmax, "shape", [])),
            "floor_support_shape": list(getattr(floor_support, "shape", [])),
            "obstacle_support_shape": list(getattr(obstacle_support, "shape", [])),
        }

    observed = support > 0
    inside = _compute_inside_mask_from_observed(observed)
    finite_min = np.isfinite(hmin)
    finite_max = np.isfinite(hmax)

    floor_band = float(_FLOORPLAN_AGL_FLOOR_SUPPORT_BAND_M)
    obstacle_band = float(_FLOORPLAN_AGL_OBSTACLE_SUPPORT_MIN_M)
    max_relevant_h = float(_FLOORPLAN_CLEAN_MAX_RELEVANT_H_M)
    min_support = max(int(_FLOORPLAN_CLEAN_MIN_SUPPORT), int(_FLOORPLAN_AGL_MIN_SUPPORT_POINTS))
    floor_min_pts = int(_FLOORPLAN_AGL_FLOOR_SUPPORT_MIN_POINTS)
    obs_min_pts = int(_FLOORPLAN_AGL_OBSTACLE_SUPPORT_MIN_POINTS)
    floor_ratio_min = float(_FLOORPLAN_AGL_FLOOR_SUPPORT_RATIO_MIN)
    obs_ratio_min = float(_FLOORPLAN_AGL_OBSTACLE_SUPPORT_RATIO_MIN)
    range_thresh = float(_FLOORPLAN_AGL_OBSTACLE_RANGE_MIN_M)

    # Ratios are more robust than per-cell mean AGL because vertical surfaces dump many points
    # into a single (x,z) cell, inflating the mean and destroying floor separation.
    denom = np.maximum(1.0, support.astype(np.float32))
    floor_ratio = floor_support.astype(np.float32) / denom
    obs_ratio = obstacle_support.astype(np.float32) / denom

    floor_seed = (
        inside
        & observed
        & finite_min
        & (support >= min_support)
        & (floor_support >= floor_min_pts)
        & (hmin <= floor_band)
    )

    # Obstacle detection: focus on *mostly-horizontal* surfaces (countertops/tabletops).
    # Vertical surfaces (walls/cabinet fronts) dump many points into one (x,z) cell and tend to
    # blanket the map if we include them; they are not helpful for the "furniture vs walkable"
    # floorplan goal here.
    finite = finite_min & finite_max
    h_range = np.zeros_like(hmax, dtype=np.float32)
    h_range[finite] = (hmax[finite] - hmin[finite]).astype(np.float32, copy=False)

    flat_obs_min_h = float(_FLOORPLAN_CLEAN_FLAT_OBS_MIN_H_M)
    flat_obs_max_delta = float(_FLOORPLAN_CLEAN_FLAT_OBS_MAX_DELTA_M)
    obstacle_mask = (
        inside
        & observed
        & finite
        & (support >= min_support)
        & (obstacle_support >= obs_min_pts)
        & (hmin >= flat_obs_min_h)
        & (h_range <= flat_obs_max_delta)
    )
    obstacle_mask = _remove_small_components(obstacle_mask, int(_FLOORPLAN_CLEAN_MIN_COMPONENT_CELLS))
    structure = np.ones((_FLOORPLAN_CLEAN_MORPH_SIZE, _FLOORPLAN_CLEAN_MORPH_SIZE), dtype=bool)
    pad = max(0, int(_FLOORPLAN_CLEAN_MORPH_SIZE) // 2)
    if pad > 0:
        padded = np.pad(obstacle_mask, pad_width=pad, mode="constant", constant_values=False)
        # Opening reduces thin spurious bridges between nearby obstacles (island vs counter)
        # while keeping large obstacle blobs intact.
        padded = ndi.binary_opening(padded, structure=structure, border_value=0)
        padded = ndi.binary_closing(padded, structure=structure, border_value=0)
        obstacle_mask = padded[pad:-pad, pad:-pad]
    else:
        obstacle_mask = ndi.binary_opening(obstacle_mask, structure=structure, border_value=0)
        obstacle_mask = ndi.binary_closing(obstacle_mask, structure=structure, border_value=0)
    obstacle_mask &= inside

    obstacle_height = np.zeros((h_px, w_px), dtype=np.float32)
    if np.any(obstacle_mask):
        max_obs = float(_FLOORPLAN_CLEAN_MAX_OBSTACLE_H_M)
        obstacle_height[obstacle_mask] = np.clip(hmax[obstacle_mask], 0.0, max_obs).astype(np.float32, copy=False)
    else:
        max_obs = 0.0

    # Walkable: any inside cell that is not classified as an obstacle. We intentionally include
    # unobserved-but-inside cells as walkable to extend floor through occlusions (island).
    walkable = (inside & (~obstacle_mask)).astype(np.float32, copy=False)

    meta: Dict[str, Any] = {
        "mode": "kitchen_clean_agl_v4_flat",
        "camera_id": str(camera_id),
        "thresholds": {
            "floor_support_band_m": float(floor_band),
            "obstacle_support_min_m": float(obstacle_band),
            "flat_obs_min_h_m": float(flat_obs_min_h),
            "flat_obs_max_delta_m": float(flat_obs_max_delta),
            "max_relevant_h_m": float(max_relevant_h),
            "min_support_points": int(min_support),
            "floor_support_min_points": int(floor_min_pts),
            "obstacle_support_min_points": int(obs_min_pts),
            "floor_support_ratio_min": float(floor_ratio_min),
            "obstacle_support_ratio_min": float(obs_ratio_min),
            "min_component_cells": int(_FLOORPLAN_CLEAN_MIN_COMPONENT_CELLS),
            "morph_size": int(_FLOORPLAN_CLEAN_MORPH_SIZE),
        },
        "cells": {
            "observed": int(np.count_nonzero(observed)),
            "inside": int(np.count_nonzero(inside)),
            "inside_unobserved": int(np.count_nonzero(inside & ~observed)),
            "floor_seed": int(np.count_nonzero(floor_seed)),
            "obstacle": int(np.count_nonzero(obstacle_mask)),
            "walkable": int(np.count_nonzero(walkable > 0.5)),
        },
        "grid_res_m": float(grid_res_m),
        "obstacle_height_value_max": float(max_obs),
    }
    return obstacle_height, walkable, meta


def _compute_kitchen_clean_floorplan_layers(
    camera_id: str,
    *,
    x_cam_pts: np.ndarray,
    z_cam_pts: np.ndarray,
    y_world_pts: np.ndarray,
    pts_weight: Optional[np.ndarray],
    x_idx: np.ndarray,
    z_idx: np.ndarray,
    support_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return (obstacle_height_grid, walkable_grid, meta) for kitchen cameras."""
    h_px, w_px = support_grid.shape if hasattr(support_grid, "shape") else (0, 0)
    if h_px <= 0 or w_px <= 0:
        obstacle_height = np.zeros((1, 1), dtype=np.float32)
        walkable = np.ones((1, 1), dtype=np.float32)
        return obstacle_height, walkable, {"mode": "bad_support_grid"}

    a, b, c, plane_meta = _fit_floor_plane_from_points(x_cam_pts, z_cam_pts, y_world_pts, pts_weight)
    x = np.asarray(x_cam_pts, dtype=np.float32).ravel()
    z = np.asarray(z_cam_pts, dtype=np.float32).ravel()
    y = np.asarray(y_world_pts, dtype=np.float32).ravel()
    xi = np.asarray(x_idx, dtype=np.int32).ravel()
    zi = np.asarray(z_idx, dtype=np.int32).ravel()
    n = min(x.size, z.size, y.size, xi.size, zi.size)
    if n <= 0:
        obstacle_height = np.zeros((h_px, w_px), dtype=np.float32)
        walkable = np.ones((h_px, w_px), dtype=np.float32)
        return obstacle_height, walkable, {"mode": "no_points", "plane": [a, b, c], "plane_meta": plane_meta}

    x = x[:n]
    z = z[:n]
    y = y[:n]
    xi = xi[:n]
    zi = zi[:n]

    residual_full = y - ((a * x) + (b * z) + c)
    finite = np.isfinite(residual_full)
    if not np.any(finite):
        obstacle_height = np.zeros((h_px, w_px), dtype=np.float32)
        walkable = np.zeros((h_px, w_px), dtype=np.float32)
        return obstacle_height, walkable, {"mode": "no_finite_residuals", "plane": [a, b, c], "plane_meta": plane_meta}

    x = x[finite]
    z = z[finite]
    y = y[finite]
    residual = residual_full[finite].astype(np.float32, copy=False)
    xi = xi[finite]
    zi = zi[finite]

    support = np.asarray(support_grid, dtype=np.uint32)
    observed = support > 0

    # Define an "inside" footprint in XZ by spanning between observed columns per row.
    # This keeps the clean floorplan from defaulting to a full rectangle while still
    # filling occluded gaps behind large obstacles (island/countertops).
    inside = np.zeros((h_px, w_px), dtype=bool)
    if np.any(observed):
        # Be permissive here: the footprint should include sparsely observed wall edges,
        # otherwise the floorplan collapses into a thin strip.
        min_row_support = 1
        left = np.full(h_px, -1, dtype=np.int32)
        right = np.full(h_px, -1, dtype=np.int32)
        for r in range(h_px):
            cols = np.flatnonzero(observed[r])
            if cols.size >= min_row_support:
                left[r] = int(cols[0])
                right[r] = int(cols[-1])
        valid_rows = np.flatnonzero(left >= 0)
        if valid_rows.size:
            first = int(valid_rows[0])
            last = int(valid_rows[-1])
            # Only extend to the image edge when the gap is small; otherwise we'd
            # incorrectly paint huge unobserved regions as "inside".
            max_extend_rows = 4
            start = 0 if first <= max_extend_rows else first
            end = (h_px - 1) if ((h_px - 1) - last) <= max_extend_rows else last

            rows = np.arange(start, end + 1, dtype=np.float32)
            left_i = np.interp(rows, valid_rows.astype(np.float32), left[valid_rows].astype(np.float32))
            right_i = np.interp(rows, valid_rows.astype(np.float32), right[valid_rows].astype(np.float32))
            # Smooth jaggies along Z a bit.
            left_s = ndi.median_filter(left_i, size=7).astype(np.int32, copy=False)
            right_s = ndi.median_filter(right_i, size=7).astype(np.int32, copy=False)
            left_s = np.clip(left_s, 0, w_px - 1)
            right_s = np.clip(right_s, 0, w_px - 1)
            for idx, r in enumerate(range(int(start), int(end) + 1)):
                l = int(left_s[idx])
                rr = int(right_s[idx])
                if rr < l:
                    continue
                inside[r, l: rr + 1] = True

            # Also add the convex hull of observed cells to reduce concavity in the footprint.
            # This makes the final BEV look more like a room floorplan instead of a thin strip.
            try:
                oy, ox = np.where(observed)
                if oy.size >= 3:
                    pts = np.stack([ox, oy], axis=1).astype(np.int32, copy=False)
                    hull = cv2.convexHull(pts)
                    hull_mask = np.zeros((h_px, w_px), dtype=np.uint8)
                    cv2.fillConvexPoly(hull_mask, hull, 1)
                    inside |= hull_mask.astype(bool)
            except Exception:
                pass
            # Fill small gaps/holes inside the footprint without bleeding past the image edges.
            fp_struct = np.ones((7, 7), dtype=bool)
            fp_pad = int(fp_struct.shape[0] // 2)
            padded_fp = np.pad(inside, pad_width=fp_pad, mode="constant", constant_values=False)
            padded_fp = ndi.binary_closing(padded_fp, structure=fp_struct, border_value=0)
            inside = padded_fp[fp_pad:-fp_pad, fp_pad:-fp_pad]
            inside = ndi.binary_fill_holes(inside)
            labeled, num = ndi.label(inside)
            if num > 0:
                seed = (h_px - 1, w_px // 2)
                seed_label = int(labeled[seed])
                if seed_label == 0:
                    counts = np.bincount(labeled.ravel())
                    if counts.size:
                        counts[0] = 0
                        seed_label = int(np.argmax(counts))
                if seed_label > 0:
                    inside = labeled == seed_label
    if not np.any(inside):
        inside = observed.copy()

    # Height-above-plane candidates (some calibrations effectively use +Y down).
    h_pos_raw = residual
    h_neg_raw = -residual
    max_relevant = float(_FLOORPLAN_CLEAN_MAX_RELEVANT_H_M)
    thresh = float(_FLOORPLAN_CLEAN_OBSTACLE_THRESH_M)
    floor_band = float(_FLOORPLAN_CLEAN_FLOOR_SEED_BAND_M)
    min_cells = int(_FLOORPLAN_CLEAN_MIN_COMPONENT_CELLS)

    def _side_counts(h_raw: np.ndarray) -> Tuple[int, int, float, float]:
        h_raw = np.asarray(h_raw, dtype=np.float32)
        finite_h = np.isfinite(h_raw)
        if not np.any(finite_h):
            return 0, 0, 0.0, 0.0
        h0 = h_raw[finite_h]
        # floor points cluster near 0 on either side (use abs), but obstacles should land on the chosen positive side.
        floor_cnt = int(np.count_nonzero(np.abs(h0) <= floor_band))
        obs = h0[h0 >= thresh]
        obs = obs[obs <= max_relevant]
        obs_cnt = int(obs.size)
        p95 = float(np.percentile(np.clip(obs, 0.0, max_relevant), 95)) if obs.size else 0.0
        med = float(np.median(np.clip(obs, 0.0, max_relevant))) if obs.size else 0.0
        return floor_cnt, obs_cnt, med, p95

    pos_floor_cnt, pos_obs_cnt, pos_med, pos_p95 = _side_counts(h_pos_raw)
    neg_floor_cnt, neg_obs_cnt, neg_med, neg_p95 = _side_counts(h_neg_raw)

    # Prefer the side that yields more above-floor obstacle evidence.
    if neg_obs_cnt > pos_obs_cnt:
        obstacle_side = "negative"
        h = h_neg_raw
    else:
        obstacle_side = "positive"
        h = h_pos_raw

    h = np.clip(np.asarray(h, dtype=np.float32), 0.0, None)

    # Ignore points that are too tall to matter for walkability to avoid ceiling artifacts.
    relevant = np.isfinite(h) & (h <= max_relevant)
    rel_h = h[relevant]
    rel_xi = xi[relevant]
    rel_zi = zi[relevant]

    relevant_support = np.zeros((h_px, w_px), dtype=np.uint32)
    if rel_h.size:
        np.add.at(relevant_support, (rel_zi, rel_xi), 1)

    floor_support = np.zeros((h_px, w_px), dtype=np.uint32)
    obstacle_support = np.zeros((h_px, w_px), dtype=np.uint32)
    # Use looser floor support and higher obstacle support thresholds so ratios are robust
    # to sparse floor observations and small above-floor noise.
    floor_support_band = float(_FLOORPLAN_CLEAN_FLOOR_SUPPORT_BAND_M)
    obs_support_min = float(_FLOORPLAN_CLEAN_OBSTACLE_SUPPORT_MIN_M)
    floor_points = rel_h <= float(floor_support_band)
    obstacle_points = rel_h >= thresh
    obstacle_support_points = rel_h >= float(obs_support_min)
    if np.any(floor_points):
        np.add.at(floor_support, (rel_zi[floor_points], rel_xi[floor_points]), 1)
    if np.any(obstacle_support_points):
        np.add.at(obstacle_support, (rel_zi[obstacle_support_points], rel_xi[obstacle_support_points]), 1)

    h_min = np.full((h_px, w_px), np.inf, dtype=np.float32)
    h_max = np.full((h_px, w_px), -np.inf, dtype=np.float32)
    if rel_h.size:
        rel_h_clip = np.clip(rel_h, 0.0, float(_FLOORPLAN_CLEAN_MAX_OBSTACLE_H_M)).astype(np.float32, copy=False)
        np.minimum.at(h_min, (rel_zi, rel_xi), rel_h_clip)
        np.maximum.at(h_max, (rel_zi, rel_xi), rel_h_clip)
    h_min[~np.isfinite(h_min)] = np.nan
    h_max[~np.isfinite(h_max)] = 0.0
    h_delta = np.nan_to_num(h_max - h_min, nan=np.inf, posinf=np.inf, neginf=np.inf).astype(np.float32, copy=False)

    # Obstacle classification: prefer cells that exhibit a strong, tight "top surface" mode.
    # This targets countertops/tabletops and avoids walls/cabinet-front clutter that tends to
    # blanket the grid when using per-cell mean/max alone.
    min_support = int(_FLOORPLAN_CLEAN_MIN_SUPPORT)
    obs_min_h = float(_FLOORPLAN_CLEAN_FLAT_OBS_MIN_H_M)
    top_band = float(_FLOORPLAN_CLEAN_TOP_BAND_M)
    top_ratio_min = float(_FLOORPLAN_CLEAN_TOP_RATIO_MIN)
    obs_flat_min_pts = int(_FLOORPLAN_CLEAN_OBS_FLAT_MIN_POINTS)
    obs_std_max = float(_FLOORPLAN_CLEAN_OBS_STD_MAX_M)

    # Top-band support: percentage of obstacle-support points that cluster near the cell maximum.
    top_support = np.zeros((h_px, w_px), dtype=np.uint32)
    if rel_h.size:
        cell_hmax = h_max[rel_zi, rel_xi]
        top_points = (rel_h >= (cell_hmax - top_band)) & (rel_h >= float(obs_support_min))
        if np.any(top_points):
            np.add.at(top_support, (rel_zi[top_points], rel_xi[top_points]), 1)
    denom_obs = np.maximum(1.0, obstacle_support.astype(np.float32))
    top_ratio = top_support.astype(np.float32) / denom_obs

    # Obstacle height variance (on obstacle-support points only). Flat tops have low stddev;
    # vertical faces have high stddev.
    obs_sum = np.zeros((h_px, w_px), dtype=np.float64)
    obs_sumsq = np.zeros((h_px, w_px), dtype=np.float64)
    if np.any(obstacle_support_points):
        h_obs = rel_h[obstacle_support_points].astype(np.float64, copy=False)
        zi_obs = rel_zi[obstacle_support_points]
        xi_obs = rel_xi[obstacle_support_points]
        np.add.at(obs_sum, (zi_obs, xi_obs), h_obs)
        np.add.at(obs_sumsq, (zi_obs, xi_obs), h_obs * h_obs)

    obs_mean = np.zeros((h_px, w_px), dtype=np.float32)
    obs_std = np.full((h_px, w_px), np.inf, dtype=np.float32)
    valid_stats = obstacle_support >= obs_flat_min_pts
    if np.any(valid_stats):
        cnt = obstacle_support[valid_stats].astype(np.float64, copy=False)
        mean = (obs_sum[valid_stats] / cnt)
        var = (obs_sumsq[valid_stats] / cnt) - (mean * mean)
        var = np.maximum(var, 0.0)
        obs_mean[valid_stats] = mean.astype(np.float32, copy=False)
        obs_std[valid_stats] = np.sqrt(var).astype(np.float32, copy=False)

    flat_top = (
        (relevant_support >= min_support)
        & (obstacle_support >= obs_flat_min_pts)
        & (h_max >= float(obs_support_min))
        & (obs_mean >= obs_min_h)
        & (obs_std <= obs_std_max)
        & (top_ratio >= top_ratio_min)
    )

    obstacle_mask = inside & flat_top
    structure = np.ones((_FLOORPLAN_CLEAN_MORPH_SIZE, _FLOORPLAN_CLEAN_MORPH_SIZE), dtype=bool)
    obstacle_mask = _remove_small_components(obstacle_mask, int(_FLOORPLAN_CLEAN_MIN_COMPONENT_CELLS))
    # Run closing on a padded grid so we don't accidentally erode real obstacles that touch the array edge.
    pad = max(0, int(_FLOORPLAN_CLEAN_MORPH_SIZE) // 2)
    if pad > 0:
        padded = np.pad(obstacle_mask, pad_width=pad, mode="constant", constant_values=False)
        padded = ndi.binary_closing(padded, structure=structure, border_value=0)
        obstacle_mask = padded[pad:-pad, pad:-pad]
    else:
        obstacle_mask = ndi.binary_closing(obstacle_mask, structure=structure, border_value=0)
    obstacle_mask &= inside

    obstacle_height = np.zeros((h_px, w_px), dtype=np.float32)
    obstacle_height[obstacle_mask] = h_max[obstacle_mask]

    # For visualization, fill tiny holes inside obstacle regions by propagating local maxima.
    if np.any(obstacle_mask):
        local_max = ndi.maximum_filter(obstacle_height, size=_FLOORPLAN_CLEAN_MORPH_SIZE)
        hole_mask = obstacle_mask & (obstacle_height < float(_FLOORPLAN_CLEAN_OBSTACLE_THRESH_M))
        if np.any(hole_mask):
            obstacle_height[hole_mask] = np.maximum(local_max[hole_mask], float(_FLOORPLAN_CLEAN_OBSTACLE_THRESH_M)).astype(np.float32, copy=False)

    walkable = np.zeros((h_px, w_px), dtype=np.float32)
    walkable[inside] = 1.0
    walkable[obstacle_mask] = 0.0

    meta: Dict[str, Any] = {
        "mode": "kitchen_clean_layers",
        "plane": [float(a), float(b), float(c)],
        "plane_meta": plane_meta,
        "obstacle_side": obstacle_side,
        "side_metrics": {
            "pos": {"median": float(pos_med), "p95": float(pos_p95), "floor_count": int(pos_floor_cnt), "obstacle_count": int(pos_obs_cnt)},
            "neg": {"median": float(neg_med), "p95": float(neg_p95), "floor_count": int(neg_floor_cnt), "obstacle_count": int(neg_obs_cnt)},
        },
        "thresholds": {
            "obstacle_m": float(_FLOORPLAN_CLEAN_OBSTACLE_THRESH_M),
            "max_relevant_m": float(_FLOORPLAN_CLEAN_MAX_RELEVANT_H_M),
            "max_obstacle_m": float(_FLOORPLAN_CLEAN_MAX_OBSTACLE_H_M),
            "min_support": int(_FLOORPLAN_CLEAN_MIN_SUPPORT),
            "min_component_cells": int(_FLOORPLAN_CLEAN_MIN_COMPONENT_CELLS),
            "morph_size": int(_FLOORPLAN_CLEAN_MORPH_SIZE),
            "floor_band_m": float(_FLOORPLAN_CLEAN_FLOOR_SEED_BAND_M),
            "floor_support_band_m": float(_FLOORPLAN_CLEAN_FLOOR_SUPPORT_BAND_M),
            "obstacle_support_min_m": float(_FLOORPLAN_CLEAN_OBSTACLE_SUPPORT_MIN_M),
            "top_band_m": float(_FLOORPLAN_CLEAN_TOP_BAND_M),
            "top_ratio_min": float(_FLOORPLAN_CLEAN_TOP_RATIO_MIN),
            "obs_flat_min_points": int(_FLOORPLAN_CLEAN_OBS_FLAT_MIN_POINTS),
            "obs_std_max_m": float(_FLOORPLAN_CLEAN_OBS_STD_MAX_M),
            "flat_obs_min_h_m": float(_FLOORPLAN_CLEAN_FLAT_OBS_MIN_H_M),
        },
        "cells": {
            "observed": int(np.count_nonzero(observed)),
            "inside": int(np.count_nonzero(inside)),
            "obstacle": int(np.count_nonzero(obstacle_mask)),
            "relevant_point_count": int(rel_h.size),
            "obstacle_point_count": int(np.count_nonzero(obstacle_points)),
            "floor_point_count": int(np.count_nonzero(floor_points)),
        },
    }
    return obstacle_height, walkable, meta


@dataclass(frozen=True)
class DepthSummary:
    median: float
    p10: float
    p90: float
    conf_mean: float
    valid_ratio: float
    sample_count: int


@dataclass(frozen=True)
class DepthResult:
    camera_id: str
    ts_us: int
    depth: np.ndarray
    conf: np.ndarray
    mask: np.ndarray
    intrinsics: Optional[np.ndarray]
    native_intrinsics: Optional[np.ndarray]
    summary: DepthSummary
    storage_path: Path


@dataclass(frozen=True)
class _SnapshotJob:
    camera_id: str
    ts_us: int
    depth: np.ndarray
    conf: np.ndarray
    mask: np.ndarray
    dest_path: Path


@dataclass(frozen=True)
class _BatchItem:
    camera_id: str
    timestamp_s: float
    view_result: ViewBuildResult
    view_payload: Dict[str, object]
    future: Future


class DepthStorageManager:
    """Persist depth outputs to Zarr for later consumption with retention enforcement."""

    def __init__(
        self,
        base_path: Path,
        max_snapshots_per_camera: int,
        retention_minutes: float,
        max_total_bytes: Optional[int] = None,
        *,
        enable_async: bool = True,
        max_queue_size: int = 32,
        worker_count: int = 1,
        max_worker_count: int = 0,
        # New tuning knobs
        enforce_async: bool = True,
        enforce_interval_s: float = 1.0,
        size_hysteresis_ratio: float = 0.9,
        zarr_clevel: int = 5,
        zarr_chunk_px: int = 128,
        min_conf: float = float("nan"),
        floorplan_store_dir: Optional[Path] = None,
        max_depth_cache_entries: int = 16,
        max_floorplan_cache_entries: int = 24,
        max_normals_cache_entries: int = 8,
    ) -> None:
        self.base_path = Path(base_path).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._locks: Dict[str, threading.Lock] = {}
        self._indices: Dict[str, Deque[Tuple[int, Path]]] = {}
        self._logger = logging.getLogger(__name__)
        self._max_snapshots = max(max_snapshots_per_camera, 0)
        self._retention_us = max(0, int(retention_minutes * 60.0 * 1_000_000))
        self._max_total_bytes = max_total_bytes if (max_total_bytes is not None and max_total_bytes > 0) else None
        self._async_enabled = bool(enable_async)
        self._max_queue_size = max(1, int(max_queue_size)) if self._async_enabled else 0
        self._initial_workers = 0
        self._max_worker_count = 0
        if self._async_enabled:
            requested_workers = int(worker_count)
            default_workers = self._default_worker_target()
            if requested_workers <= 0:
                requested_workers = default_workers
            self._initial_workers = max(1, requested_workers)
            requested_max = int(max_worker_count)
            if requested_max <= 0:
                requested_max = max(self._initial_workers, default_workers)
            self._max_worker_count = max(self._initial_workers, requested_max)
            self._max_queue_size = max(self._max_queue_size, self._initial_workers * 4)
        self._queue: Optional["queue.Queue[_SnapshotJob]"] = None
        self._stop_event: Optional[threading.Event] = None
        self._writer_threads: List[threading.Thread] = []
        self._worker_lock = threading.Lock()
        self._worker_name_counter = 0
        self._queue_put_timeout = 1.0
        self._last_queue_full_warning: float = 0.0
        # Retention/size enforcement tuning
        self._enforce_async = bool(enforce_async)
        self._enforce_interval_s = max(0.05, float(enforce_interval_s))
        self._size_hysteresis_ratio = min(1.0, max(0.1, float(size_hysteresis_ratio)))
        self._zarr_clevel = max(0, int(zarr_clevel))
        self._zarr_chunk_px = int(zarr_chunk_px)
        self._cache_lock = threading.Lock()
        self._depth_payload_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._floorplan_cache: "OrderedDict[Tuple[str, float, float], Dict[str, Any]]" = OrderedDict()
        self._normals_cache: "OrderedDict[Tuple[str, int, str, str], Dict[str, Any]]" = OrderedDict()
        self._floorplan_store_dir = Path(floorplan_store_dir).resolve() if floorplan_store_dir else (self.base_path / "floorplans")
        self._floorplan_store_dir.mkdir(parents=True, exist_ok=True)
        self._max_depth_cache_entries = max(1, int(max_depth_cache_entries))
        self._max_floorplan_cache_entries = max(1, int(max_floorplan_cache_entries))
        self._max_normals_cache_entries = max(1, int(max_normals_cache_entries))
        self.min_conf = float(min_conf)
        # Background enforcement thread
        self._enforce_thread: Optional[threading.Thread] = None
        self._enforce_stop = threading.Event()
        if self._max_total_bytes is not None and self._max_total_bytes < 10 * 1024 * 1024:
            self._logger.warning(
                "Configured max_total_bytes=%s is very small; increasing to 10MB minimum",
                self._max_total_bytes,
            )
            self._max_total_bytes = 10 * 1024 * 1024
        self._seed_existing_entries()
        if self._async_enabled:
            self._start_writer()
        # Start async enforcement if enabled
        if self._enforce_async:
            self._start_enforcer()

    def _default_worker_target(self) -> int:
        cpu_count = os.cpu_count() or 1
        return max(2, min(8, cpu_count))

    def _seed_existing_entries(self) -> None:
        """Populate in-memory indices from disk on startup and prune if needed."""
        for camera_dir in sorted(self.base_path.glob("*")):
            if not camera_dir.is_dir():
                continue
            camera_id = camera_dir.name
            index = self._indices.setdefault(camera_id, deque())
            zarr_paths = []
            for path in camera_dir.rglob("*.zarr"):
                try:
                    ts = int(path.stem)
                except ValueError:
                    continue
                zarr_paths.append((ts, path))
            if not zarr_paths:
                continue
            zarr_paths.sort(key=lambda item: item[0])
            index.extend(zarr_paths)
            self._enforce_limits(camera_id, index, now_ts=self._current_time_us())

    def _current_time_us(self) -> int:
        return int(time.time() * 1_000_000)

    def _get_lock(self, camera_id: str) -> threading.Lock:
        return self._locks.setdefault(camera_id, threading.Lock())

    def _get_index(self, camera_id: str) -> Deque[Tuple[int, Path]]:
        return self._indices.setdefault(camera_id, deque())

    def _start_writer(self) -> None:
        if self._queue is not None or self._initial_workers <= 0:
            return
        self._queue = queue.Queue(maxsize=self._max_queue_size)
        self._stop_event = threading.Event()
        with self._worker_lock:
            for _ in range(self._initial_workers):
                self._spawn_worker_locked()

    def _spawn_worker_locked(self) -> None:
        if self._queue is None or self._stop_event is None:
            return
        self._worker_name_counter += 1
        thread = threading.Thread(
            target=self._writer_loop,
            name=f"DepthSnapshotWriter-{self._worker_name_counter}",
            daemon=True,
        )
        self._writer_threads.append(thread)
        thread.start()

    def _writer_loop(self) -> None:
        # Defensive loop that tolerates shutdown races and empty queue timeouts.
        # This is intentionally conservative to avoid noisy thread exceptions
        # while retaining async snapshot functionality.
        try:
            q = self._queue
            stop = self._stop_event
        except Exception:
            q = None
            stop = None
        while True:
            try:
                # Refresh local refs each iteration in case shutdown mutated them
                if q is None or stop is None:
                    q = self._queue
                    stop = self._stop_event
                if q is None:
                    # Queue no longer available; exit quietly
                    break
                try:
                    job = q.get(timeout=0.2)
                except queue.Empty:
                    try:
                        if stop is None:
                            stop = self._stop_event
                        if stop is not None and stop.is_set():
                            break
                    except Exception:
                        # If stop flag is unavailable, exit defensively
                        break
                    continue
                try:
                    self._write_snapshot(job)
                except Exception as exc:
                    # Keep processing other jobs; log at error level
                    self._logger.error(
                        "Depth snapshot write failed for %s to %s: %s",
                        getattr(job, "camera_id", "unknown"),
                        getattr(job, "dest_path", None),
                        exc,
                    )
                finally:
                    try:
                        q.task_done()
                    except Exception:
                        pass
            except Exception:
                # Any unexpected error should not tear down the thread noisily.
                # Re-check shutdown and either continue or exit quietly.
                try:
                    stop = self._stop_event
                    if stop is not None and stop.is_set():
                        break
                except Exception:
                    break
                time.sleep(0.05)

    def _start_enforcer(self) -> None:
        if self._enforce_thread is not None:
            return
        def _loop() -> None:
            while not self._enforce_stop.is_set():
                try:
                    # Prune all cameras; uses per-camera locks internally
                    self.prune()
                except Exception:
                    pass
                # Sleep a bit to batch work and avoid thrash
                self._enforce_stop.wait(self._enforce_interval_s)
        self._enforce_thread = threading.Thread(target=_loop, name="DepthRetentionEnforcer", daemon=True)
        self._enforce_thread.start()

    def shutdown(self, *, wait: bool = True) -> None:
        # Stop background enforcer
        try:
            self._enforce_stop.set()
            if self._enforce_thread and wait:
                self._enforce_thread.join(timeout=2.0)
        except Exception:
            pass
        self._enforce_thread = None
        # Stop async writers
        if not self._async_enabled or self._stop_event is None:
            return
        self._stop_event.set()
        # Always give workers a small window to exit to avoid races
        join_timeout = 2.0 if wait else 0.5
        threads = self._collect_alive_threads()
        for thread in threads:
            try:
                thread.join(timeout=join_timeout)
            except Exception:
                pass
        with self._worker_lock:
            self._writer_threads.clear()
        # Only clear references after attempting joins to prevent attr races
        self._queue = None
        self._stop_event = None

    def flush(self, timeout: Optional[float] = None) -> None:
        if not self._async_enabled or self._queue is None:
            return
        if timeout is None:
            self._queue.join()
            return
        deadline = time.time() + max(0.0, timeout)
        while getattr(self._queue, "unfinished_tasks", 0) > 0:
            if time.time() >= deadline:
                break
            time.sleep(0.01)

    def _create_job(
        self,
        camera_id: str,
        ts_us: int,
        depth: np.ndarray,
        conf: np.ndarray,
        mask: np.ndarray,
        dest_path: Path,
    ) -> _SnapshotJob:
        depth_c = np.ascontiguousarray(depth, dtype=np.float32).copy()
        conf_c = np.ascontiguousarray(conf, dtype=np.float32).copy()
        mask_c = np.ascontiguousarray(mask, dtype=np.uint8).copy()
        _increment_core_boundary_copy_bytes(
            "depth_store",
            int(getattr(depth_c, "nbytes", 0) or 0)
            + int(getattr(conf_c, "nbytes", 0) or 0)
            + int(getattr(mask_c, "nbytes", 0) or 0),
        )
        return _SnapshotJob(camera_id, ts_us, depth_c, conf_c, mask_c, dest_path)

    def _register_snapshot(self, camera_id: str, ts_us: int, dest_path: Path) -> None:
        # Keep lock ordering consistent with load_latest_depth (cache_lock -> camera_lock)
        # to avoid deadlocks and to ensure we don't repopulate stale cached payloads
        # while a new snapshot is being registered.
        with self._cache_lock:
            lock = self._get_lock(camera_id)
            with lock:
                index = self._get_index(camera_id)
                index.append((ts_us, dest_path))
                # Defer enforcement to background thread if enabled
                if not self._enforce_async:
                    self._enforce_limits(camera_id, index)
            # Invalidate cached WS payload so subsequent RPCs see the newest snapshot.
            self._depth_payload_cache.pop(camera_id, None)

    def _make_blosc_compressor(self) -> Optional[Any]:
        if self._zarr_clevel <= 0:
            return None
        if _ZarrBlosc is not None:
            try:
                return _ZarrBlosc(cname="zstd", clevel=int(self._zarr_clevel))
            except Exception:
                self._logger.debug("zarr.codecs.Blosc unavailable; falling back to numcodecs")
        if _NumcodecsBlosc is not None:
            try:
                shuffle = getattr(_NumcodecsBlosc, "SHUFFLE", 1)
                return _NumcodecsBlosc(cname="zstd", clevel=int(self._zarr_clevel), shuffle=shuffle, blocksize=0)
            except Exception:
                self._logger.debug("numcodecs.Blosc unavailable; storing uncompressed")
        return None

    def _create_zarr_dataset(
        self,
        root: "zarr.hierarchy.Group",
        name: str,
        data: np.ndarray,
        chunk_shape: Tuple[int, int],
        compressor: Optional[Any],
    ) -> None:
        create_kwargs = {
            "shape": tuple(int(dim) for dim in data.shape),
            "data": data,
            "chunks": tuple(int(dim) for dim in chunk_shape),
            "overwrite": True,
        }
        if compressor is not None:
            create_kwargs["compressor"] = compressor
        try:
            root.create_dataset(name, **create_kwargs)
            return
        except TypeError as exc:
            # Retry using the zarr v3 compressors API, otherwise fall back to uncompressed.
            self._logger.debug("create_dataset fallback for %s due to %s", name, exc)
            create_kwargs.pop("compressor", None)
            if compressor is not None:
                try:
                    create_kwargs["compressors"] = [compressor]
                    root.create_dataset(name, **create_kwargs)
                    return
                except TypeError:
                    create_kwargs.pop("compressors", None)
        root.create_dataset(name, **create_kwargs)

    def _write_snapshot(self, job: _SnapshotJob) -> None:
        job.dest_path.parent.mkdir(parents=True, exist_ok=True)
        compressor = self._make_blosc_compressor()
        root = zarr.open_group(str(job.dest_path), mode="w")
        if self._zarr_chunk_px and self._zarr_chunk_px > 0:
            chunk_shape = (min(self._zarr_chunk_px, job.depth.shape[0]), min(self._zarr_chunk_px, job.depth.shape[1]))
        else:
            # Single-chunk per array to minimize file count
            chunk_shape = job.depth.shape
        self._create_zarr_dataset(root, "depth_z", job.depth, chunk_shape, compressor)
        self._create_zarr_dataset(root, "conf", job.conf, chunk_shape, compressor)
        self._create_zarr_dataset(root, "mask", job.mask, chunk_shape, compressor)
        root.attrs.update(
            camera_id=job.camera_id,
            timestamp_us=int(job.ts_us),
            stored_at=time.time(),
            shape=json.dumps(job.depth.shape),
        )
        self._register_snapshot(job.camera_id, job.ts_us, job.dest_path)

    def _remove_snapshot(self, path: Path) -> None:
        try:
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
                # Clean up empty parent directories up to camera root
                parent = path.parent
                for _ in range(2):
                    if parent == self.base_path or not parent.exists():
                        break
                    try:
                        next(parent.iterdir())
                    except StopIteration:
                        parent.rmdir()
                    parent = parent.parent
        except Exception as exc:
            self._logger.debug("Failed to remove snapshot %s: %s", path, exc)

    def _enforce_limits(
        self,
        camera_id: str,
        index: Deque[Tuple[int, Path]],
        now_ts: Optional[int] = None,
    ) -> None:
        if not index:
            return
        now_ts = now_ts if now_ts is not None else self._current_time_us()
        retention_cutoff = None
        if self._retention_us > 0:
            retention_cutoff = now_ts - self._retention_us

        removed = 0
        # Enforce retention duration first so age limit always wins
        if retention_cutoff is not None:
            while index and index[0][0] < retention_cutoff:
                _, path = index.popleft()
                self._remove_snapshot(path)
                removed += 1

        # Enforce max snapshot count with hysteresis to prevent thrash
        if self._max_snapshots > 0 and len(index) > self._max_snapshots:
            # determine floor based on 90% of max (at least 1)
            target_len = max(1, int(self._max_snapshots * 0.9))
            while len(index) > target_len:
                _, path = index.popleft()
                self._remove_snapshot(path)
                removed += 1

        if self._max_total_bytes is not None:
            removed += self._enforce_total_size(camera_id, index)

        if removed:
            self._logger.info(
                "Pruned %s depth snapshots for camera %s (max=%s, retention_us=%s)",
                removed,
                camera_id,
                self._max_snapshots,
                self._retention_us,
            )

    def _enforce_total_size(self, camera_id: str, index: Deque[Tuple[int, Path]]) -> int:
        if not index:
            return 0
        total_bytes = 0
        sizes: List[int] = []
        for _, path in index:
            try:
                size = sum(file.stat().st_size for file in path.rglob('*') if file.is_file())
            except FileNotFoundError:
                size = 0
            sizes.append(size)
            total_bytes += size

        removed = 0
        # Hysteresis: prune down to a target below the hard cap to avoid frequent rescans
        target_bytes = int(self._max_total_bytes * self._size_hysteresis_ratio)
        while total_bytes > target_bytes and index:
            ts, path = index.popleft()
            size = sizes.pop(0)
            total_bytes -= size
            self._remove_snapshot(path)
            removed += 1

        if removed and total_bytes > self._max_total_bytes:
            self._logger.warning(
                "Total depth storage still above quota after pruning (camera=%s remaining=%s max=%s)",
                camera_id,
                total_bytes,
                self._max_total_bytes,
            )
        return removed

    def prune(self, camera_id: Optional[str] = None) -> int:
        """Manual pruning entrypoint; returns number of snapshots removed."""
        total_removed = 0
        if camera_id:
            lock = self._get_lock(camera_id)
            with lock:
                index = self._get_index(camera_id)
                before = len(index)
                self._enforce_limits(camera_id, index)
                total_removed += max(0, before - len(index))
            return total_removed

        for cam_id in list(self._indices.keys()):
            total_removed += self.prune(cam_id)
        return total_removed

    def purge_all(self, camera_id: Optional[str] = None) -> int:
        """Remove all stored depth snapshots to start fresh."""
        if camera_id is not None:
            return self._purge_camera(camera_id)

        total = 0
        for cam_id in list(self._indices.keys()):
            total += self._purge_camera(cam_id)
        return total

    def _purge_camera(self, camera_id: str) -> int:
        removed = 0
        lock = self._get_lock(camera_id)
        with lock:
            index = self._get_index(camera_id)
            while index:
                _, path = index.popleft()
                self._remove_snapshot(path)
                removed += 1
        cam_dir = self.base_path / camera_id
        if cam_dir.exists():
            try:
                next(cam_dir.iterdir())
            except StopIteration:
                cam_dir.rmdir()
        return removed

    def store(
        self,
        camera_id: str,
        ts_us: int,
        depth: np.ndarray,
        conf: np.ndarray,
        mask: np.ndarray,
    ) -> Path:
        timestamp = datetime.utcfromtimestamp(ts_us / 1_000_000.0)
        date_dir = timestamp.strftime("%Y%m%d")
        hour_dir = timestamp.strftime("%H")
        dest_dir = self.base_path / camera_id / date_dir / hour_dir
        dest_path = dest_dir / f"{ts_us}.zarr"

        job = self._create_job(camera_id, ts_us, depth, conf, mask, dest_path)
        if self._async_enabled and self._queue is not None:
            try:
                self._queue.put(job, timeout=self._queue_put_timeout)
                return dest_path
            except queue.Full:
                if self._maybe_scale_workers():
                    try:
                        self._queue.put(job, timeout=self._queue_put_timeout)
                        return dest_path
                    except queue.Full:
                        pass
                now = time.time()
                if now - self._last_queue_full_warning >= 5.0:
                    self._logger.warning(
                        "Depth snapshot queue full; writing synchronously (size=%s)",
                        self._max_queue_size,
                    )
                    self._last_queue_full_warning = now

        self._write_snapshot(job)
        return dest_path

    def _collect_alive_threads(self) -> List[threading.Thread]:
        with self._worker_lock:
            alive = [thread for thread in self._writer_threads if thread.is_alive()]
            self._writer_threads = alive
            return list(alive)

    def _maybe_scale_workers(self) -> bool:
        if not self._async_enabled or self._queue is None:
            return False
        with self._worker_lock:
            alive = [thread for thread in self._writer_threads if thread.is_alive()]
            self._writer_threads = alive
            if len(alive) >= self._max_worker_count:
                return False
            self._spawn_worker_locked()
            return True

    def latest_entry(self, camera_id: str, ts_max: Optional[int]) -> Optional[Path]:
        lock = self._get_lock(camera_id)
        with lock:
            index = self._get_index(camera_id)
            if not index:
                return None
            # Drop missing files from the tail
            while index and not index[-1][1].exists():
                index.pop()
            if not index:
                return None
            if ts_max is None:
                return index[-1][1]
            # Find newest entry <= ts_max
            for ts, path in reversed(index):
                if ts <= ts_max and path.exists():
                    return path
            return None

    def all_cameras(self) -> Iterable[str]:
        return list(self._indices.keys())

    def load_datasets(self, path: Path) -> Optional[Dict[str, np.ndarray]]:
        try:
            group = zarr.open_group(str(path), mode='r')
            depth = np.array(group['depth_z'])
            conf = np.array(group['conf'])
            mask = np.array(group['mask'])
            return {'depth': depth, 'conf': conf, 'mask': mask}
        except Exception:
            return None

    def load_latest_depth(self, camera_id: str, ts_max_us: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Return the newest cached payload up to ts_max_us (microseconds)."""
        cache_key = camera_id
        ts_cutoff = None
        if ts_max_us is not None:
            try:
                ts_cutoff = int(ts_max_us)
            except Exception:
                ts_cutoff = None
        with self._cache_lock:
            cached = self._depth_payload_cache.get(cache_key)
            if cached:
                try:
                    cached_ts = int(cached.get('ts', 0) or 0)
                except Exception:
                    cached_ts = 0
                if ts_cutoff is None or cached_ts <= ts_cutoff:
                    return dict(cached)

        path = self.latest_entry(camera_id, ts_cutoff)
        if not path:
            return None
        datasets = self.load_datasets(path)
        if not datasets:
            return None
        depth = datasets['depth'].astype(np.float32, copy=False)
        conf = datasets['conf'].astype(np.float32, copy=False)
        mask = datasets['mask'].astype(np.uint8, copy=False)
        height, width = depth.shape[:2]
        ts_us = int(path.stem)
        payload = {
            'ts': ts_us,
            'depth_b64': base64.b64encode(depth.tobytes()).decode('ascii'),
            'conf_b64': base64.b64encode(conf.tobytes()).decode('ascii'),
            'mask_b64': base64.b64encode(mask.tobytes()).decode('ascii'),
            'shape': [int(height), int(width)],
        }
        with self._cache_lock:
            self._depth_payload_cache[cache_key] = dict(payload)
            self._depth_payload_cache.move_to_end(cache_key, last=True)
            while len(self._depth_payload_cache) > self._max_depth_cache_entries:
                self._depth_payload_cache.popitem(last=False)
        return payload

    def _resolve_intrinsics_for_depth(
        self,
        camera_id: str,
        _depth_shape: Tuple[int, int],
    ) -> Tuple[float, float, float, float]:
        calib_bundle = getattr(self, "calibration_bundle", None) or {}
        cameras_node = calib_bundle.get("cameras") if isinstance(calib_bundle, dict) else {}
        k_table = cameras_node.get("K") if isinstance(cameras_node, dict) else {}
        intr = k_table.get(camera_id) if isinstance(k_table, dict) else None

        if intr is None:
            raise ValueError("missing_calibration")

        intr_arr = np.asarray(intr, dtype=np.float32).reshape(-1)
        if intr_arr.size == 4:
            fx, fy, cx, cy = [float(v) for v in intr_arr]
        elif intr_arr.size == 9:
            k_mat = intr_arr.reshape(3, 3)
            fx = float(k_mat[0, 0])
            fy = float(k_mat[1, 1])
            cx = float(k_mat[0, 2])
            cy = float(k_mat[1, 2])
        else:
            raise ValueError("bad_intrinsics")

        if not all(np.isfinite([fx, fy, cx, cy])) or fx == 0.0 or fy == 0.0:
            raise ValueError("invalid_intrinsics")

        # DS8 MapAnything depth is aligned to frame coordinates before it is persisted.
        # Keep K as the runtime-calibrated canonical source and avoid a second letterbox
        # transform here, which introduces per-camera geometric drift in floorplans.
        return fx, fy, cx, cy

    def _resolve_extrinsics(self, camera_id: str) -> Optional[Sequence[float]]:
        calib_bundle = getattr(self, "calibration_bundle", None) or {}
        cameras_node = calib_bundle.get("cameras") if isinstance(calib_bundle, dict) else {}
        e_table = cameras_node.get("E") if isinstance(cameras_node, dict) else {}
        extr = e_table.get(camera_id) if isinstance(e_table, dict) else None
        if not isinstance(extr, (list, tuple)) or len(extr) != 16:
            return None
        return extr

    @staticmethod
    def _compute_normals(
        depth: np.ndarray,
        valid_mask: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> np.ndarray:
        height, width = depth.shape
        grid_u, grid_v = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
            indexing="xy",
        )
        x_cam = (grid_u - float(cx)) * depth / float(fx)
        y_cam = (grid_v - float(cy)) * depth / float(fy)
        z_cam = depth
        points = np.stack([x_cam, y_cam, z_cam], axis=-1).astype(np.float32, copy=False)
        invalid = ~valid_mask
        if np.any(invalid):
            points[invalid] = np.nan

        dPdx = np.zeros_like(points)
        dPdy = np.zeros_like(points)
        if width > 1:
            dPdx[:, 1:-1] = points[:, 2:] - points[:, :-2]
            dPdx[:, 0] = points[:, 1] - points[:, 0]
            dPdx[:, -1] = points[:, -1] - points[:, -2]
        if height > 1:
            dPdy[1:-1] = points[2:] - points[:-2]
            dPdy[0] = points[1] - points[0]
            dPdy[-1] = points[-1] - points[-2]

        normals = np.cross(dPdx, dPdy)
        norm = np.linalg.norm(normals, axis=-1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            normals = np.divide(normals, norm, out=np.zeros_like(normals), where=(norm > 1e-6))

        good = np.isfinite(normals).all(axis=-1) & valid_mask
        normals[~good] = 0.0

        flip = normals[..., 2] > 0
        normals[flip] *= -1.0
        return normals

    def attach_normals_to_payload(
        self,
        camera_id: str,
        payload: Dict[str, Any],
        *,
        space: str = "camera",
        dtype: str = "float16",
    ) -> None:
        if not isinstance(payload, dict):
            return
        if payload.get("normals_b64"):
            return

        depth_b64 = payload.get("depth_b64") or payload.get("depth_z_b64")
        shape = payload.get("shape")
        if not isinstance(depth_b64, str) or not depth_b64:
            payload["normals_error"] = "missing_depth"
            return
        if not (isinstance(shape, (list, tuple)) and len(shape) == 2):
            payload["normals_error"] = "bad_shape"
            return

        try:
            height = int(shape[0])
            width = int(shape[1])
        except Exception:
            payload["normals_error"] = "bad_shape"
            return
        if height <= 0 or width <= 0:
            payload["normals_error"] = "bad_shape"
            return

        ts_raw = payload.get("ts") or payload.get("ts_us") or 0
        try:
            ts_us = int(ts_raw)
        except Exception:
            ts_us = 0

        space_norm = str(space or "camera").strip().lower() or "camera"
        dtype_norm = str(dtype or "float16").strip().lower() or "float16"
        if space_norm not in ("camera", "world"):
            payload["normals_error"] = "unsupported_space"
            return
        if dtype_norm not in ("float16", "float32"):
            payload["normals_error"] = "unsupported_dtype"
            return

        cache_key = (str(camera_id), int(ts_us or 0), space_norm, dtype_norm)
        with self._cache_lock:
            cached = self._normals_cache.get(cache_key)
            if cached:
                payload.pop("normals_error", None)
                payload.update(cached)
                return

        try:
            raw = base64.b64decode(depth_b64)
            depth = np.frombuffer(raw, dtype=np.float32)
            needed = height * width
            if depth.size < needed:
                payload["normals_error"] = "depth_too_small"
                return
            depth = depth[:needed].reshape((height, width))
        except Exception:
            payload["normals_error"] = "depth_decode_failed"
            return

        mask = None
        mask_b64 = payload.get("mask_b64")
        if isinstance(mask_b64, str) and mask_b64:
            try:
                raw_mask = base64.b64decode(mask_b64)
                mask_arr = np.frombuffer(raw_mask, dtype=np.uint8)
                if mask_arr.size >= height * width:
                    mask = mask_arr[: height * width].reshape((height, width)) > 0
            except Exception:
                mask = None

        valid = np.isfinite(depth)
        valid &= depth > 0.1
        valid &= depth < 50.0
        if mask is not None:
            valid &= mask
        if not np.any(valid):
            payload["normals_error"] = "no_valid_depth"
            return

        try:
            fx, fy, cx, cy = self._resolve_intrinsics_for_depth(camera_id, depth.shape)
        except Exception as exc:
            payload["normals_error"] = str(exc) or "intrinsics_failed"
            return

        normals = self._compute_normals(depth.astype(np.float32, copy=False), valid, fx, fy, cx, cy)

        if space_norm == "world":
            extr = self._resolve_extrinsics(camera_id)
            if extr is None:
                payload["normals_error"] = "missing_extrinsics"
                return
            try:
                r_wc, _ = parse_extrinsics(extr)
                normals = (r_wc @ normals.reshape(-1, 3).T).T.reshape((height, width, 3))
                norm = np.linalg.norm(normals, axis=-1, keepdims=True)
                with np.errstate(invalid="ignore", divide="ignore"):
                    normals = np.divide(normals, norm, out=np.zeros_like(normals), where=(norm > 1e-6))
            except Exception:
                payload["normals_error"] = "extrinsics_failed"
                return

        normals = np.asarray(normals, dtype=np.float32)
        if dtype_norm == "float16":
            normals_out = normals.astype(np.float16)
        else:
            normals_out = normals.astype(np.float32)

        try:
            normals_b64 = base64.b64encode(normals_out.tobytes()).decode("ascii")
        except Exception:
            payload["normals_error"] = "normals_encode_failed"
            return

        normals_payload = {
            "normals_b64": normals_b64,
            "normals_shape": [int(height), int(width), 3],
            "normals_dtype": dtype_norm,
            "normals_space": space_norm,
        }
        payload.pop("normals_error", None)
        payload.update(normals_payload)
        with self._cache_lock:
            self._normals_cache[cache_key] = dict(normals_payload)
            self._normals_cache.move_to_end(cache_key, last=True)
            while len(self._normals_cache) > self._max_normals_cache_entries:
                self._normals_cache.popitem(last=False)

    @staticmethod
    def _sanitize_camera_id(camera_id: str) -> str:
        safe = ''.join(ch if ch.isalnum() or ch in {'-', '_', '.'} else '_' for ch in camera_id.strip())
        return safe or "camera"

    @staticmethod
    def _format_param(value: float) -> str:
        text = f"{float(value):.6f}".rstrip('0').rstrip('.')
        if not text:
            text = "0"
        if text.startswith('-'):
            text = 'neg' + text[1:]
        return text.replace('.', 'p')

    def _floorplan_path(self, camera_id: str, grid_res_m: float, max_extent_m: float) -> Path:
        safe_cam = self._sanitize_camera_id(camera_id)
        filename = f"grid{self._format_param(grid_res_m)}__ext{self._format_param(max_extent_m)}.json"
        return self._floorplan_store_dir / safe_cam / filename

    def _load_floorplan_from_disk(
        self,
        camera_id: str,
        grid_res_m: float,
        max_extent_m: float,
        expected_units: Optional[str] = None,
        expected_calibration_fingerprint: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        path = self._floorplan_path(camera_id, grid_res_m, max_extent_m)
        if not path.exists():
            return None
        try:
            with path.open('r', encoding='utf-8') as fh:
                payload = json.load(fh)
            if isinstance(payload, dict):
                if not _floorplan_cache_contract_matches(
                    payload,
                    expected_units=expected_units,
                    expected_calibration_fingerprint=expected_calibration_fingerprint,
                ):
                    return None
                payload.setdefault('camera_id', camera_id)
                return payload
        except Exception as exc:
            self._logger.debug(f"Failed to load cached floorplan for {camera_id}: {exc}")
        return None

    def _persist_floorplan_to_disk(
        self,
        camera_id: str,
        grid_res_m: float,
        max_extent_m: float,
        payload: Mapping[str, Any],
    ) -> None:
        path = self._floorplan_path(camera_id, grid_res_m, max_extent_m)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            to_store = dict(payload)
            to_store.setdefault("frame", _FLOORPLAN_FRAME)
            to_store.setdefault("orientation", _FLOORPLAN_ORIENTATION)
            to_store.setdefault("floorplan_contract_version", int(_FLOORPLAN_CONTRACT_VERSION))
            to_store.pop('served_from_cache', None)
            tmp_path = path.with_suffix(path.suffix + '.tmp')
            with tmp_path.open('w', encoding='utf-8') as fh:
                json.dump(to_store, fh, separators=(',', ':'))
            tmp_path.replace(path)
        except Exception as exc:
            self._logger.debug(f"Failed to persist floorplan for {camera_id}: {exc}")

    def update_depth_cache(self, result: DepthResult) -> None:
        try:
            depth_bytes = result.depth.astype(np.float32, copy=False).tobytes()
            conf_bytes = result.conf.astype(np.float32, copy=False).tobytes()
            mask_bytes = result.mask.astype(np.uint8, copy=False).tobytes()
        except Exception as exc:
            self._logger.debug(f"Depth cache serialization failed for {result.camera_id}: {exc}")
            return
        payload = {
            'ts': result.ts_us,
            'depth_b64': base64.b64encode(depth_bytes).decode('ascii'),
            'conf_b64': base64.b64encode(conf_bytes).decode('ascii'),
            'mask_b64': base64.b64encode(mask_bytes).decode('ascii'),
            'shape': [int(result.depth.shape[0]), int(result.depth.shape[1])],
        }
        with self._cache_lock:
            self._depth_payload_cache[result.camera_id] = payload
            self._depth_payload_cache.move_to_end(result.camera_id, last=True)
            while len(self._depth_payload_cache) > self._max_depth_cache_entries:
                self._depth_payload_cache.popitem(last=False)

    def precompute_for_cameras(
        self,
        camera_ids: Iterable[str],
        max_age_sec: float = 60.0,
        grid_res_m: float = 0.5,
        max_extent_m: float = 20.0,
    ) -> None:
        for cam_id in camera_ids:
            if not cam_id:
                continue
            try:
                self.load_latest_depth(cam_id)
            except Exception as exc:
                self._logger.debug(f"Depth cache warmup for {cam_id} failed: {exc}")
                continue
            try:
                self.generate_topdown_floorplan(
                    cam_id,
                    max_age_sec=max_age_sec,
                    grid_res_m=grid_res_m,
                    max_extent_m=max_extent_m,
                )
            except Exception as exc:
                self._logger.debug(f"Floorplan warmup for {cam_id} failed: {exc}")

    def generate_topdown_floorplan(
        self,
        camera_id: str,
        max_age_sec: float = 60.0,
        grid_res_m: float = 0.15,
        max_extent_m: float = 20.0,
        cache_only: bool = False,
    ) -> Dict[str, Any]:
        """Generate a per-camera top-down (XZ) blueprint view from the latest depth snapshot."""
        if not camera_id:
            return {'error': 'camera_required', 'ts': int(time.time() * 1_000_000)}

        if not cache_only:
            try:
                self.flush(timeout=1.5)
            except Exception:
                pass

        cache_key = (camera_id, float(grid_res_m), float(max_extent_m))
        calib_bundle = getattr(self, 'calibration_bundle', None) or {}
        expected_flip = _expected_floorplan_flip(calib_bundle, camera_id)
        expected_units = _expected_floorplan_units_from_calibration_bundle(calib_bundle)
        expected_calibration_fingerprint = _floorplan_calibration_fingerprint(calib_bundle, camera_id)
        now_us = int(time.time() * 1_000_000)
        with self._cache_lock:
            cached = self._floorplan_cache.get(cache_key)
            if cached:
                if not _floorplan_cache_contract_matches(
                    cached,
                    expected_units=expected_units,
                    expected_calibration_fingerprint=expected_calibration_fingerprint,
                ):
                    cached = None
            if cached:
                if cache_only:
                    payload = dict(cached)
                    payload['served_from_cache'] = True
                    return payload
                age_us = now_us - cached.get('snapshot_ts', cached.get('ts', 0))
                if age_us <= int(max(0.0, max_age_sec) * 1_000_000):
                    payload = dict(cached)
                    payload['served_from_cache'] = True
                    return payload

        disk_payload = self._load_floorplan_from_disk(
            camera_id,
            grid_res_m,
            max_extent_m,
            expected_units=expected_units,
            expected_calibration_fingerprint=expected_calibration_fingerprint,
        )
        if disk_payload:
            snapshot_ts = disk_payload.get('snapshot_ts', disk_payload.get('ts'))
            if isinstance(snapshot_ts, (int, float)):
                age_us = now_us - int(snapshot_ts)
            else:
                age_us = None
            if cache_only or age_us is None or age_us <= int(max(0.0, max_age_sec) * 1_000_000):
                with self._cache_lock:
                    self._floorplan_cache[cache_key] = dict(disk_payload)
                    self._floorplan_cache.move_to_end(cache_key, last=True)
                    while len(self._floorplan_cache) > self._max_floorplan_cache_entries:
                        self._floorplan_cache.popitem(last=False)
                payload = dict(disk_payload)
                payload['served_from_cache'] = True
                return payload

        max_age_us = int(max(0.0, max_age_sec) * 1_000_000)
        ts_cutoff = now_us - max_age_us if max_age_us > 0 else None

        path_entry = self.latest_entry(camera_id, now_us)
        if not path_entry:
            return {'error': 'no_depth', 'camera_id': camera_id, 'ts': now_us}

        if ts_cutoff is not None:
            try:
                snapshot_ts = int(path_entry.stem)
            except ValueError:
                snapshot_ts = None
            if snapshot_ts is None or snapshot_ts < ts_cutoff:
                return {'error': 'stale_depth', 'camera_id': camera_id, 'ts': now_us}

        datasets = self.load_datasets(path_entry)
        if not datasets:
            return {'error': 'load_failed', 'camera_id': camera_id, 'ts': now_us}

        depth = datasets.get('depth')
        conf = datasets.get('conf')
        mask = datasets.get('mask')
        if depth is None or conf is None or mask is None:
            return {'error': 'invalid_snapshot', 'camera_id': camera_id, 'ts': now_us}

        depth = np.asarray(depth, dtype=np.float32)
        conf = np.asarray(conf, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.uint8) > 0
        if depth.ndim != 2 or conf.shape != depth.shape or mask.shape != depth.shape:
            return {'error': 'shape_mismatch', 'camera_id': camera_id, 'ts': now_us}

        scene_per_m, s_obj_to_m = _scene_units_per_meter_from_calibration_bundle(calib_bundle)
        cameras_node = calib_bundle.get('cameras') if isinstance(calib_bundle, dict) else {}
        e_table = cameras_node.get('E') if isinstance(cameras_node, dict) else {}

        extr = None
        if isinstance(e_table, dict):
            extr = e_table.get(camera_id)

        if extr is None:
            return {'error': 'missing_calibration', 'camera_id': camera_id, 'ts': now_us}

        try:
            fx, fy, cx, cy = self._resolve_intrinsics_for_depth(camera_id, depth.shape)
        except ValueError as exc:
            err = str(exc or "intrinsics_failed")
            err_map = {
                'missing_calibration': 'missing_calibration',
                'bad_intrinsics': 'bad_intrinsics',
                'invalid_intrinsics': 'invalid_intrinsics',
            }
            return {'error': err_map.get(err, 'invalid_intrinsics'), 'camera_id': camera_id, 'ts': now_us}

        # PERMISSIVE validity: only reject truly invalid depth values
        # Do NOT hard-filter by mask or confidence - use them as soft weights instead
        valid = np.isfinite(depth)
        valid &= depth > 0.1
        valid &= depth < 50.0
        # Note: mask and conf are used as weights below, not hard filters

        if not np.any(valid):
            grid = np.zeros((1, 1), dtype=np.float32)
            bounds_m = {
                'min_x': float(-grid_res_m * 0.5),
                'max_x': float(grid_res_m * 0.5),
                'min_z': 0.0,
                'max_z': float(max(grid_res_m, 1.0)),
            }
            flip_payload = _floorplan_image_flip_payload(expected_flip)
            payload = {
                'camera_id': camera_id,
                'ts': now_us,
                'snapshot_ts': int(path_entry.stem) if path_entry.stem.isdigit() else None,
                'frame': _FLOORPLAN_FRAME,
                'bounds': bounds_m,
                'scale_m_per_px': float(grid_res_m),
                'scale_scene_per_px': float(grid_res_m * scene_per_m),
                'units': 'meters',
                's_obj_to_m': float(s_obj_to_m),
                'point_count': 0,
                'density': {
                    'grid_b64': base64.b64encode(grid.tobytes()).decode('ascii'),
                    'grid_shape': [1, 1],
                    'value_min': 0.0,
                    'value_max': 0.0,
                },
                'height': {
                    'grid_b64': base64.b64encode(grid.tobytes()).decode('ascii'),
                    'grid_shape': [1, 1],
                    'value_min': 0.0,
                    'value_max': 0.0,
                },
                'height_agl': {
                    'grid_b64': base64.b64encode(grid.tobytes()).decode('ascii'),
                    'grid_shape': [1, 1],
                    'value_min': 0.0,
                    'value_max': 0.0,
                },
                'distance': {
                    'grid_b64': base64.b64encode(grid.tobytes()).decode('ascii'),
                    'grid_shape': [1, 1],
                    'value_min': 0.0,
                    'value_max': 0.0,
                },
                'height_agl_meta': {'floor_y': 0.0, 'floor_offset_m': 0.0, 'floor_estimate': {'mode': 'no_valid_depth'}},
                'served_from_cache': False,
                'grid_res_m': float(grid_res_m),
                'grid_res_scene': float(grid_res_m * scene_per_m),
                'max_extent_m': float(max_extent_m),
                'max_extent_scene': float(max_extent_m * scene_per_m),
                'orientation': _FLOORPLAN_ORIENTATION,
                'floorplan_contract_version': int(_FLOORPLAN_CONTRACT_VERSION),
                'image_flip': flip_payload,
                'calibration_fingerprint': expected_calibration_fingerprint,
            }
            self._persist_floorplan_to_disk(camera_id, grid_res_m, max_extent_m, payload)
            with self._cache_lock:
                self._floorplan_cache[cache_key] = dict(payload)
                self._floorplan_cache.move_to_end(cache_key, last=True)
                while len(self._floorplan_cache) > self._max_floorplan_cache_entries:
                    self._floorplan_cache.popitem(last=False)
            return payload

        # Extract confidence values for valid points - use as weights, not filter
        # Combine mask (as 0/1) and conf into a single weight
        # Points inside mask with high conf get weight ~1.0
        # Points outside mask or low conf get lower weights but still contribute
        # Soften the mask: give masked-out points a small weight (0.15) instead of 0
        soft_mask = np.where(mask, 1.0, 0.15).astype(np.float32)
        # Clamp confidence to [0.05, 1.0] to avoid zero weights
        conf_clamped = np.clip(conf, 0.05, 1.0)
        # Combined weight = soft_mask * confidence
        combined_weight = soft_mask * conf_clamped
        # Extract weights for valid points
        pts_weight = combined_weight[valid].astype(np.float32)

        h_img, w_img = depth.shape
        grid_u, grid_v = np.meshgrid(
            np.arange(w_img, dtype=np.float32),
            np.arange(h_img, dtype=np.float32),
            indexing='xy'
        )

        # Floorplan grids stay anchored to the canonical camera-local X/Z frame.
        # Do not remap the image axes here using the BEV/world flip heuristic.
        x_cam = (grid_u - cx) * depth / fx
        y_cam = (grid_v - cy) * depth / fy
        z_cam = depth

        pts_cam = np.stack([x_cam[valid], y_cam[valid], z_cam[valid]], axis=1)

        e_arr = np.asarray(extr, dtype=np.float32)
        if e_arr.size == 16:
            e_mat = e_arr.reshape(4, 4, order='F')
        elif e_arr.shape == (3, 4):
            e_mat = np.eye(4, dtype=np.float32)
            e_mat[:3, :4] = e_arr
        elif e_arr.shape == (4, 4):
            e_mat = e_arr
        else:
            return {'error': 'bad_extrinsics', 'camera_id': camera_id, 'ts': now_us}

        try:
            twc = np.linalg.inv(e_mat)
        except np.linalg.LinAlgError:
            return {'error': 'extrinsics_singular', 'camera_id': camera_id, 'ts': now_us}

        pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1), dtype=np.float32)], axis=1)
        pts_world_h = pts_cam_h @ twc.T
        pts_world = pts_world_h[:, :3]

        pts_depth = pts_cam[:, 2]
        pts_y = pts_world[:, 1]
        x_cam_pts = pts_cam[:, 0]
        z_cam_pts = pts_cam[:, 2]

        # Estimate floor Y from horizontal surfaces and compute per-point height above floor (AGL).
        # This is intentionally global (single Y) for now; kitchen floors are close enough to planar.
        floor_y = 0.0
        floor_est_meta: Dict[str, Any] = {"mode": "uninitialized"}
        pts_y_agl = pts_y
        try:
            # Extrinsics assume camera +Y is up. Our depth unprojection uses +Y down (image V),
            # so for any world-frame height computation we flip the camera Y axis.
            twc_row_y = twc[1].astype(np.float32, copy=False)
            pts_y_agl = (
                (pts_cam[:, 0] * twc_row_y[0])
                + ((-pts_cam[:, 1]) * twc_row_y[1])
                + (pts_cam[:, 2] * twc_row_y[2])
                + twc_row_y[3]
            ).astype(np.float32, copy=False)

            valid_normals = valid & mask
            normals_cam = self._compute_normals(
                depth.astype(np.float32, copy=False),
                valid_normals,
                fx,
                fy,
                cx,
                cy,
            )
            # Convert normals into the same camera coordinate convention used by extrinsics (+Y up).
            normals_cam = np.asarray(normals_cam, dtype=np.float32)
            normals_cam[..., 1] *= -1.0
            normals_cam_flat = normals_cam[valid]
            r_wc = twc[:3, :3].astype(np.float32, copy=False)
            normals_world_flat = (r_wc @ normals_cam_flat.T).T
            good_point = mask[valid] & (conf_clamped[valid] >= 0.2)
            floor_w = np.where(good_point, pts_weight, 0.0).astype(np.float32, copy=False)
            floor_y, floor_est_meta = _estimate_floor_y_from_horizontal_points(
                pts_y_agl,
                normals_world_flat,
                floor_w,
                horiz_dot_thresh=float(_FLOORPLAN_AGL_HORIZ_DOT_THRESH),
            )
        except Exception as exc:
            try:
                floor_y = float(np.nanpercentile(pts_y_agl, 1.0)) if pts_y_agl.size else 0.0
            except Exception:
                floor_y = float(np.nanmin(pts_y_agl)) if pts_y_agl.size else 0.0
            floor_est_meta = {"mode": "error", "error": str(exc), "floor_y": float(floor_y)}

        height_agl_pts = (pts_y_agl - float(floor_y)).astype(np.float32, copy=False)
        height_agl_pts = np.clip(height_agl_pts, 0.0, float(_FLOORPLAN_AGL_HEIGHT_CLIP_M)).astype(np.float32, copy=False)

        # Normalize AGL by subtracting a low-percentile offset so the lowest observed surface
        # lands near 0m even when the floor estimator is biased low (heavy occlusions).
        agl_floor_offset_m = 0.0
        try:
            good_offset = mask[valid] & (conf_clamped[valid] >= 0.2)
            cand = height_agl_pts[good_offset] if np.any(good_offset) else height_agl_pts
            if cand.size:
                agl_floor_offset_m = float(np.percentile(cand, 2.0))
        except Exception:
            agl_floor_offset_m = 0.0
        if not np.isfinite(agl_floor_offset_m) or agl_floor_offset_m <= 1e-3:
            agl_floor_offset_m = 0.0
        if agl_floor_offset_m > 0.0:
            height_agl_pts = np.clip(
                height_agl_pts - float(agl_floor_offset_m), 0.0, float(_FLOORPLAN_AGL_HEIGHT_CLIP_M)
            ).astype(np.float32, copy=False)

        # If the floor estimator drifts low (common when the floor is heavily occluded),
        # all AGL values get offset upward and saturate the visualization. Normalize by
        # subtracting a low-percentile offset so the lowest observed surface is ~0.
        agl_floor_offset_m = 0.0
        try:
            good_offset = mask[valid] & (conf_clamped[valid] >= 0.2)
            cand = height_agl_pts[good_offset] if np.any(good_offset) else height_agl_pts
            if cand.size:
                agl_floor_offset_m = float(np.percentile(cand, 2.0))
        except Exception:
            agl_floor_offset_m = 0.0
        if not np.isfinite(agl_floor_offset_m) or agl_floor_offset_m <= 1e-3:
            agl_floor_offset_m = 0.0
        if agl_floor_offset_m > 0.0:
            height_agl_pts = np.clip(
                height_agl_pts - float(agl_floor_offset_m), 0.0, float(_FLOORPLAN_AGL_HEIGHT_CLIP_M)
            ).astype(np.float32, copy=False)

        if x_cam_pts.size == 0 or z_cam_pts.size == 0:
            return {'error': 'no_points', 'camera_id': camera_id, 'ts': now_us, 'point_count': 0}

        pad_x = max(0.5, grid_res_m * 2.0)
        pad_z = max(0.5, grid_res_m * 2.0)

        max_x_abs = float(np.max(np.abs(x_cam_pts))) if x_cam_pts.size else 0.0
        if not np.isfinite(max_x_abs):
            max_x_abs = 0.0
        forward_max = float(np.max(z_cam_pts)) if z_cam_pts.size else 0.0
        if not np.isfinite(forward_max):
            forward_max = 0.0

        half_width = max_x_abs + pad_x
        forward_extent = max(0.0, forward_max) + pad_z
        if max_extent_m > 0:
            max_extent = float(max_extent_m)
            min_half_width = max_extent * float(_FLOORPLAN_MIN_HALF_WIDTH_FRACTION)
            min_forward = max_extent * float(_FLOORPLAN_MIN_FORWARD_FRACTION)
            half_width = max(half_width, min_half_width)
            forward_extent = max(forward_extent, min_forward)
            half_width = min(half_width, max_extent)
            forward_extent = min(forward_extent, max_extent)

        half_width = max(half_width, grid_res_m * 0.5)
        forward_extent = max(forward_extent, grid_res_m)

        min_x = -half_width
        max_x = half_width
        min_z = 0.0
        max_z = forward_extent

        width_m = max_x - min_x
        height_m = max_z - min_z

        w_px = max(1, int(np.ceil(width_m / grid_res_m)))
        h_px = max(1, int(np.ceil(height_m / grid_res_m)))

        x_norm = np.clip((x_cam_pts - min_x) / width_m, 0.0, 0.999999)
        z_norm = np.clip((z_cam_pts - min_z) / height_m, 0.0, 0.999999)
        x_idx = np.clip(np.floor(x_norm * w_px).astype(np.int32), 0, w_px - 1)
        z_idx = np.clip(np.floor((1.0 - z_norm) * h_px).astype(np.int32), 0, h_px - 1)

        density_grid = np.zeros((h_px, w_px), dtype=np.float32)
        distance_sum = np.zeros((h_px, w_px), dtype=np.float32)
        distance_count = np.zeros((h_px, w_px), dtype=np.uint32)
        height_grid = np.full((h_px, w_px), -np.inf, dtype=np.float32)
        height_agl_max_grid = np.full((h_px, w_px), -np.inf, dtype=np.float32)
        height_agl_min_grid = np.full((h_px, w_px), np.inf, dtype=np.float32)

        # Confidence-weighted height aggregation
        weighted_height_sum = np.zeros((h_px, w_px), dtype=np.float64)
        weight_sum = np.zeros((h_px, w_px), dtype=np.float64)
        weighted_agl_sum = np.zeros((h_px, w_px), dtype=np.float64)
        agl_weight_sum = np.zeros((h_px, w_px), dtype=np.float64)
        agl_floor_support_grid = np.zeros((h_px, w_px), dtype=np.uint32)
        agl_obstacle_support_grid = np.zeros((h_px, w_px), dtype=np.uint32)

        indices = (z_idx, x_idx)
        # Density: count of points (unweighted for backward compat)
        np.add.at(density_grid, indices, 1.0)
        # Distance: weighted by confidence
        np.add.at(distance_sum, indices, (pts_depth * pts_weight).astype(np.float32, copy=False))
        np.add.at(distance_count, indices, 1)
        # Height: accumulate weighted sum and weights for weighted mean
        np.add.at(weighted_height_sum, indices, (pts_y * pts_weight).astype(np.float64))
        np.add.at(weight_sum, indices, pts_weight.astype(np.float64))
        np.add.at(weighted_agl_sum, indices, (height_agl_pts * pts_weight).astype(np.float64))
        np.add.at(agl_weight_sum, indices, pts_weight.astype(np.float64))
        floor_band = float(_FLOORPLAN_AGL_FLOOR_SUPPORT_BAND_M)
        obstacle_band = float(_FLOORPLAN_AGL_OBSTACLE_SUPPORT_MIN_M)
        max_relevant_h = float(_FLOORPLAN_CLEAN_MAX_RELEVANT_H_M)
        np.add.at(agl_floor_support_grid, indices, (height_agl_pts <= floor_band).astype(np.uint32))
        np.add.at(
            agl_obstacle_support_grid,
            indices,
            ((height_agl_pts >= obstacle_band) & (height_agl_pts <= max_relevant_h)).astype(np.uint32),
        )
        # Also track max height (still useful for some visualizations)
        np.maximum.at(height_grid, indices, np.asarray(pts_y, dtype=np.float32))
        np.maximum.at(height_agl_max_grid, indices, height_agl_pts.astype(np.float32, copy=False))
        np.minimum.at(height_agl_min_grid, indices, height_agl_pts.astype(np.float32, copy=False))

        # Compute confidence-weighted mean height
        # Use weighted mean where we have weights, otherwise fall back to max
        has_weight = weight_sum > 1e-9
        # Avoid invalid division warnings: np.where evaluates both branches eagerly.
        weighted_mean_height = height_grid.astype(np.float32, copy=True)
        np.divide(
            weighted_height_sum,
            weight_sum,
            out=weighted_mean_height,
            where=has_weight,
        )

        height_agl_mean_grid = height_agl_max_grid.astype(np.float32, copy=True)
        has_agl_weight = agl_weight_sum > 1e-9
        np.divide(
            weighted_agl_sum,
            agl_weight_sum,
            out=height_agl_mean_grid,
            where=has_agl_weight,
        )

        # For cells with no points at all, mark as NaN
        empty_cells = height_grid == -np.inf
        if np.any(empty_cells):
            weighted_mean_height[empty_cells] = np.nan
            height_grid[empty_cells] = np.nan
            height_agl_mean_grid[empty_cells] = np.nan
            height_agl_max_grid[empty_cells] = np.nan
            height_agl_min_grid[empty_cells] = np.nan

        # Use weighted mean as the primary height grid (preserves detail better than max)
        height_grid = weighted_mean_height
        clean_layers = None

        density_max = float(np.max(density_grid)) if density_grid.size else 0.0
        if density_max > 0.0:
            density_grid /= density_max

        height_grid, height_fill_meta = _postprocess_floorplan_height_grid(
            camera_id,
            height_grid,
            density_grid,
            min_x=min_x,
            max_x=max_x,
            min_z=min_z,
            max_z=max_z,
        )
        height_min = float(np.min(height_grid)) if height_grid.size else 0.0
        height_max = float(np.max(height_grid)) if height_grid.size else 0.0
        if height_grid.size:
            height_grid = height_grid - height_min
            height_min = 0.0
            height_max = float(np.max(height_grid)) if height_grid.size else 0.0

        # Compute clean BEV layers for every room. This classifies walkable floor
        # and obstacle surfaces in camera-local X/Z space so the overlay and the
        # visible floorplan share the same raster surface.
        try:
            y_up_pts = (-pts_cam[:, 1]).astype(np.float32, copy=False)
            obs_h, walk, clean_meta = _compute_kitchen_clean_floorplan_layers(
                camera_id,
                x_cam_pts=x_cam_pts,
                z_cam_pts=z_cam_pts,
                y_world_pts=y_up_pts,
                pts_weight=pts_weight,
                x_idx=x_idx,
                z_idx=z_idx,
                support_grid=distance_count,
            )
            if isinstance(clean_meta, dict):
                clean_meta = dict(clean_meta)
            else:
                clean_meta = {"mode": "clean_floorplan_layers"}
            clean_meta["mode"] = "clean_floorplan_layers"
            clean_meta["floor_estimate"] = floor_est_meta
            clean_layers = (obs_h, walk, clean_meta)
        except Exception:
            clean_layers = None

        # Compute height gradient magnitude for edge detection
        # First, fill empty/NaN cells with floor level so boundaries don't create false edges
        height_for_gradient = height_grid.copy()
        floor_level = float(np.nanmin(height_grid)) if np.any(np.isfinite(height_grid)) else 0.0
        height_for_gradient = np.nan_to_num(
            height_for_gradient,
            nan=floor_level,
            posinf=floor_level,
            neginf=floor_level,
        )

        # Also mask cells with zero density (no points) to floor level
        if density_grid is not None:
            empty_mask = density_grid < 1e-6
            height_for_gradient[empty_mask] = floor_level

        # Sobel filters capture directional derivatives
        gx = ndi.sobel(height_for_gradient, axis=1, mode='nearest')
        gz = ndi.sobel(height_for_gradient, axis=0, mode='nearest')
        gradient_mag = np.sqrt(gx ** 2 + gz ** 2)

        # Use 95th percentile normalization to prevent outliers from dominating
        gradient_flat = gradient_mag[np.isfinite(gradient_mag)]
        if gradient_flat.size > 0:
            gradient_p95 = float(np.percentile(gradient_flat, 95))
            if gradient_p95 > 1e-6:
                gradient_grid = np.clip(gradient_mag / gradient_p95, 0.0, 1.0).astype(np.float32)
            else:
                gradient_grid = np.zeros_like(height_grid, dtype=np.float32)
        else:
            gradient_grid = np.zeros_like(height_grid, dtype=np.float32)

        # Clean up any remaining NaN
        gradient_grid = np.nan_to_num(gradient_grid, nan=0.0, posinf=0.0, neginf=0.0)

        distance_grid = np.zeros((h_px, w_px), dtype=np.float32)
        nonzero_mask = distance_count > 0
        if np.any(nonzero_mask):
            distance_grid[nonzero_mask] = distance_sum[nonzero_mask] / distance_count[nonzero_mask]
            min_distance = float(np.min(distance_grid[nonzero_mask]))
            max_distance = float(np.max(distance_grid[nonzero_mask]))
        else:
            min_distance = 0.0
            max_distance = 0.0

        bounds = {
            'min_x': float(min_x),
            'max_x': float(max_x),
            'min_z': float(min_z),
            'max_z': float(max_z),
        }

        agl_max = 0.0
        try:
            finite_agl = np.isfinite(height_agl_mean_grid)
            if np.any(finite_agl):
                agl_max = float(np.percentile(height_agl_mean_grid[finite_agl], 99))
                if not np.isfinite(agl_max) or agl_max <= 1e-6:
                    agl_max = float(np.nanmax(height_agl_mean_grid[finite_agl]))
        except Exception:
            agl_max = 0.0
        if not np.isfinite(agl_max) or agl_max <= 1e-6:
            agl_max = float(_FLOORPLAN_AGL_HEIGHT_CLIP_M)

        payload: Dict[str, Any] = {
            'camera_id': camera_id,
            'ts': now_us,
            'snapshot_ts': int(path_entry.stem) if path_entry.stem.isdigit() else None,
            'frame': _FLOORPLAN_FRAME,
            'orientation': _FLOORPLAN_ORIENTATION,
            'floorplan_contract_version': int(_FLOORPLAN_CONTRACT_VERSION),
            'bounds': bounds,
            'scale_m_per_px': float(width_m / w_px if w_px else grid_res_m),
            'scale_scene_per_px': float((width_m / w_px if w_px else grid_res_m) * scene_per_m),
            'units': 'meters',
            's_obj_to_m': float(s_obj_to_m),
            'point_count': int(pts_cam.shape[0]),
            'density': {
                'grid_b64': base64.b64encode(density_grid.astype(np.float32, copy=False).ravel().tobytes()).decode('ascii'),
                'grid_shape': [int(h_px), int(w_px)],
                'value_min': 0.0,
                'value_max': 1.0,
            },
            'height': {
                'grid_b64': base64.b64encode(height_grid.astype(np.float32, copy=False).ravel().tobytes()).decode('ascii'),
                'grid_shape': [int(h_px), int(w_px)],
                'value_min': float(height_min),
                'value_max': float(height_max),
            },
            'height_agl': {
                'grid_b64': base64.b64encode(height_agl_mean_grid.astype(np.float32, copy=False).ravel().tobytes()).decode('ascii'),
                'grid_shape': [int(h_px), int(w_px)],
                'value_min': 0.0,
                'value_max': float(agl_max),
            },
            'distance': {
                'grid_b64': base64.b64encode(distance_grid.astype(np.float32, copy=False).ravel().tobytes()).decode('ascii'),
                'grid_shape': [int(h_px), int(w_px)],
                'value_min': float(min_distance),
                'value_max': float(max_distance),
            },
            'gradient': {
                'grid_b64': base64.b64encode(gradient_grid.astype(np.float32, copy=False).ravel().tobytes()).decode('ascii'),
                'grid_shape': [int(h_px), int(w_px)],
                'value_min': 0.0,
                'value_max': 1.0,
            },
            'height_fill_meta': height_fill_meta,
            'height_agl_meta': {
                'floor_y': float(floor_y),
                'floor_offset_m': float(agl_floor_offset_m),
                'floor_estimate': floor_est_meta,
            },
            'calibration_fingerprint': expected_calibration_fingerprint,
        }
        if clean_layers is not None:
            obstacle_height_grid, walkable_grid, clean_meta = clean_layers
            try:
                obs_max = float(np.nanmax(obstacle_height_grid)) if obstacle_height_grid.size else 0.0
            except Exception:
                obs_max = 0.0
            if not np.isfinite(obs_max) or obs_max <= 1e-6:
                obs_max = float(_FLOORPLAN_CLEAN_MAX_OBSTACLE_H_M)
            payload['obstacle_height'] = {
                'grid_b64': base64.b64encode(obstacle_height_grid.astype(np.float32, copy=False).ravel().tobytes()).decode('ascii'),
                'grid_shape': [int(h_px), int(w_px)],
                'value_min': 0.0,
                'value_max': float(obs_max),
            }
            payload['walkable'] = {
                'grid_b64': base64.b64encode(walkable_grid.astype(np.float32, copy=False).ravel().tobytes()).decode('ascii'),
                'grid_shape': [int(h_px), int(w_px)],
                'value_min': 0.0,
                'value_max': 1.0,
            }
            payload['clean_floorplan_meta'] = clean_meta
        payload['served_from_cache'] = False
        payload['grid_res_m'] = float(grid_res_m)
        payload['grid_res_scene'] = float(grid_res_m * scene_per_m)
        payload['max_extent_m'] = float(max_extent_m)
        payload['max_extent_scene'] = float(max_extent_m * scene_per_m)
        payload['image_flip'] = _floorplan_image_flip_payload(expected_flip)

        self._persist_floorplan_to_disk(camera_id, grid_res_m, max_extent_m, payload)

        with self._cache_lock:
            self._floorplan_cache[cache_key] = dict(payload)
            self._floorplan_cache.move_to_end(cache_key, last=True)
            while len(self._floorplan_cache) > self._max_floorplan_cache_entries:
                self._floorplan_cache.popitem(last=False)

        return payload

    def generate_clean_topdown_floorplan(
        self,
        camera_id: str,
        max_age_sec: float = 30.0,
        grid_res_m: float = 0.05,
        max_extent_m: float = 15.0,
    ) -> Optional[np.ndarray]:
        """Generate a simple, high-level floorplan RGB view (floor vs obstacles).

        This is intentionally a coarse, heuristic visualization designed for BEV review:
        - Project depth points into world XZ using the current calibration bundle.
        - Bin into a 2D grid and compute per-cell mean world Y.
        - Classify floor vs obstacles by relative height to an estimated floor Y and density.

        Returns an RGB uint8 image (H,W,3) or None when data/calibration is missing.
        """
        cam = str(camera_id).strip()
        if not cam:
            return None

        now_us = int(time.time() * 1_000_000)
        min_ts_us = None
        if max_age_sec is not None:
            try:
                max_age = float(max_age_sec)
            except Exception:
                max_age = 0.0
            if max_age > 0.0:
                min_ts_us = now_us - int(max_age * 1_000_000)

        path_entry = self.latest_entry(cam, None)
        if not path_entry:
            return None
        try:
            snapshot_ts = int(path_entry.stem)
        except Exception:
            snapshot_ts = None
        if min_ts_us is not None and snapshot_ts is not None and snapshot_ts < min_ts_us:
            return None

        datasets = self.load_datasets(path_entry)
        if not datasets:
            return None
        depth = np.asarray(datasets.get("depth"), dtype=np.float32)
        conf = np.asarray(datasets.get("conf"), dtype=np.float32)
        mask = np.asarray(datasets.get("mask"), dtype=np.uint8) > 0
        if depth.ndim != 2 or conf.shape != depth.shape or mask.shape != depth.shape:
            return None

        extr = self._resolve_extrinsics(cam)
        if extr is None:
            return None

        try:
            fx, fy, cx, cy = self._resolve_intrinsics_for_depth(cam, depth.shape)
        except Exception:
            return None

        # Basic validity: keep only finite, positive depth with decent confidence.
        valid = mask & np.isfinite(depth) & (depth > 0.2) & (depth < 50.0) & np.isfinite(conf) & (conf > 0.5)
        if not np.any(valid):
            return None

        rows, cols = np.nonzero(valid)
        d = depth[valid].astype(np.float32, copy=False)
        x_cam = (cols.astype(np.float32, copy=False) - float(cx)) * d / float(fx)
        # Depth unprojection uses +Y down (image V axis), but our extrinsics convention expects
        # camera +Y up. Flip Y so world-frame heights are meaningful.
        y_cam = -(rows.astype(np.float32, copy=False) - float(cy)) * d / float(fy)
        z_cam = d
        pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1).astype(np.float32, copy=False)

        e_arr = np.asarray(extr, dtype=np.float32).reshape(-1)
        if e_arr.size != 16:
            return None
        e_mat = e_arr.reshape(4, 4, order="F")
        try:
            twc = np.linalg.inv(e_mat).astype(np.float32, copy=False)
        except np.linalg.LinAlgError:
            return None

        pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1), dtype=np.float32)], axis=1)
        pts_world_h = pts_cam_h @ twc.T
        pts_world = pts_world_h[:, :3].astype(np.float32, copy=False)

        xw = pts_world[:, 0]
        yw = pts_world[:, 1]
        zw = pts_world[:, 2]
        finite = np.isfinite(xw) & np.isfinite(yw) & np.isfinite(zw)
        if not np.any(finite):
            return None
        xw = xw[finite]
        yw = yw[finite]
        zw = zw[finite]

        max_extent = float(max(0.1, float(max_extent_m)))
        in_range = (np.abs(xw) <= max_extent) & (np.abs(zw) <= max_extent)
        if not np.any(in_range):
            return None
        xw = xw[in_range]
        yw = yw[in_range]
        zw = zw[in_range]

        # Estimate floor_y as the lowest significant peak in world-Y. Median is a poor estimator
        # when countertops dominate the visible surfaces.
        floor_y = 0.0
        try:
            if yw.size:
                y_lo = float(np.percentile(yw, 0.5))
                y_hi = float(np.percentile(yw, 99.5))
                if np.isfinite(y_lo) and np.isfinite(y_hi) and (y_hi > (y_lo + 1e-6)):
                    hist, edges = np.histogram(yw, bins=256, range=(y_lo, y_hi))
                    centers = (edges[:-1] + edges[1:]) * 0.5
                    max_hist = float(hist.max()) if hist.size else 0.0
                    thresh = max(1.0, 0.03 * max_hist)
                    best: Optional[int] = None
                    for i in range(1, len(hist) - 1):
                        if hist[i] <= hist[i - 1] or hist[i] <= hist[i + 1]:
                            continue
                        if float(hist[i]) < thresh:
                            continue
                        if best is None or float(centers[i]) < float(centers[best]):
                            best = i
                    if best is not None:
                        floor_y = float(centers[best])
                    else:
                        floor_y = float(np.nanpercentile(yw, 1.0))
                else:
                    floor_y = float(np.nanmedian(yw))
        except Exception:
            try:
                floor_y = float(np.nanpercentile(yw, 1.0)) if yw.size else 0.0
            except Exception:
                floor_y = 0.0

        res = float(max(0.005, float(grid_res_m)))
        bins = int(max(16, math.ceil((2.0 * max_extent) / res)))
        bins = min(bins, 2048)

        # Build (z,x) grid so array rows correspond to Z and columns to X.
        hist_h, _, _ = np.histogram2d(
            zw,
            xw,
            bins=bins,
            range=[[-max_extent, max_extent], [-max_extent, max_extent]],
        )
        hist_y, _, _ = np.histogram2d(
            zw,
            xw,
            bins=bins,
            range=[[-max_extent, max_extent], [-max_extent, max_extent]],
            weights=yw,
        )

        nonzero = hist_h > 0
        counts = hist_h[nonzero]
        try:
            denom_h = float(np.percentile(counts, 95.0)) if counts.size else 1.0
        except Exception:
            denom_h = float(counts.max()) if counts.size else 1.0
        if not np.isfinite(denom_h) or denom_h <= 1e-6:
            denom_h = 1.0
        density = np.clip((hist_h / denom_h).astype(np.float32, copy=False), 0.0, 1.0)
        height_map = np.zeros_like(density, dtype=np.float32)
        np.divide(hist_y, hist_h, out=height_map, where=nonzero)

        floor_m = (np.abs(height_map - floor_y) < 0.15) & (density > 0.25) & nonzero
        obst_m = (height_map > (floor_y + 0.35)) & (density > 0.35) & nonzero

        img = np.zeros((bins, bins, 3), dtype=np.uint8)
        img[floor_m] = (90, 220, 90)  # floor
        img[obst_m] = (70, 70, 110)   # obstacles
        img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_NEAREST)
        return img

    def generate_clean_topdown_rgb(
        self,
        camera_id: str,
        max_age_sec: float = 30.0,
    ) -> np.ndarray | None:
        """Generate a coarse RGB topdown view (walkable floor vs large obstacles).

        This is intended as a pragmatic, occlusion-tolerant visualization:
        - Bin depth points into a fixed 0.05m/cell grid over [-12m, +12m] in camera X/Z.
        - Estimate a global floor height in world-Y.
        - Mark "obstacle" cells as sufficiently above the floor.
        - Treat the remaining filled room footprint as walkable floor, which extends through
          occluded regions (for example behind a kitchen island).
        """
        cam = str(camera_id).strip()
        if not cam:
            return None

        now_us = int(time.time() * 1_000_000)
        try:
            max_age = float(max_age_sec)
        except Exception:
            max_age = 0.0
        min_ts_us = now_us - int(max_age * 1_000_000) if max_age > 0.0 else None

        path_entry = self.latest_entry(cam, None)
        if not path_entry:
            return None
        try:
            snapshot_ts = int(path_entry.stem)
        except Exception:
            snapshot_ts = None
        if min_ts_us is not None and snapshot_ts is not None and snapshot_ts < min_ts_us:
            return None

        datasets = self.load_datasets(path_entry)
        if not datasets:
            return None
        depth = np.asarray(datasets.get("depth"), dtype=np.float32)
        conf = np.asarray(datasets.get("conf"), dtype=np.float32)
        mask = np.asarray(datasets.get("mask"), dtype=np.uint8) > 0
        if depth.ndim != 2 or conf.shape != depth.shape or mask.shape != depth.shape:
            return None

        extr = self._resolve_extrinsics(cam)
        if extr is None:
            return None
        try:
            fx, fy, cx, cy = self._resolve_intrinsics_for_depth(cam, depth.shape)
        except Exception:
            return None

        valid = (
            mask
            & np.isfinite(depth)
            & (depth > 0.2)
            & (depth < 50.0)
            & np.isfinite(conf)
            & (conf > 0.5)
        )
        if not np.any(valid):
            return None

        rows, cols = np.nonzero(valid)
        d = depth[valid].astype(np.float32, copy=False)
        x_cam = (cols.astype(np.float32, copy=False) - float(cx)) * d / float(fx)
        # Depth unprojection uses +Y down (image V axis), but extrinsics convention expects +Y up.
        y_cam = -(rows.astype(np.float32, copy=False) - float(cy)) * d / float(fy)
        z_cam = d
        pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1).astype(np.float32, copy=False)

        e_arr = np.asarray(extr, dtype=np.float32).reshape(-1)
        if e_arr.size != 16:
            return None
        e_mat = e_arr.reshape(4, 4, order="F")
        try:
            twc = np.linalg.inv(e_mat).astype(np.float32, copy=False)
        except np.linalg.LinAlgError:
            return None

        # Transform to world only to compute world-Y for height classification.
        pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1), dtype=np.float32)], axis=1)
        pts_world = (pts_cam_h @ twc.T)[:, :3].astype(np.float32, copy=False)
        y_world = pts_world[:, 1].astype(np.float32, copy=False)

        finite = np.isfinite(x_cam) & np.isfinite(z_cam) & np.isfinite(y_world)
        if not np.any(finite):
            return None
        x = x_cam[finite].astype(np.float32, copy=False)
        z = z_cam[finite].astype(np.float32, copy=False)
        y = y_world[finite].astype(np.float32, copy=False)

        bins = 480
        max_extent_m = 12.0
        in_range = (np.abs(x) <= float(max_extent_m)) & (np.abs(z) <= float(max_extent_m))
        if not np.any(in_range):
            return None
        x = x[in_range]
        z = z[in_range]
        y = y[in_range]

        try:
            floor_y = float(np.median(y)) if y.size else 0.0
        except Exception:
            floor_y = 0.0

        hist_h, _, _ = np.histogram2d(
            x,
            z,
            bins=bins,
            range=[[-max_extent_m, max_extent_m], [-max_extent_m, max_extent_m]],
        )
        hist_y, _, _ = np.histogram2d(
            x,
            z,
            bins=bins,
            range=[[-max_extent_m, max_extent_m], [-max_extent_m, max_extent_m]],
            weights=y,
        )

        nonzero = hist_h > 0
        counts = hist_h[nonzero]
        denom = 1.0
        if counts.size:
            try:
                denom = float(np.percentile(counts, 95.0))
            except Exception:
                denom = float(counts.max())
        if not np.isfinite(denom) or denom <= 1e-6:
            denom = 1.0
        density = np.clip((hist_h / denom).astype(np.float32, copy=False), 0.0, 1.0)

        height_map = np.zeros_like(density, dtype=np.float32)
        np.divide(hist_y, hist_h, out=height_map, where=nonzero)

        # Classification thresholds (tuned for kitchen: countertop/island vs floor).
        floor_seed = (np.abs(height_map - float(floor_y)) < 0.18) & (density > 0.22) & nonzero
        obst_m = (height_map > (float(floor_y) + 0.40)) & (density > 0.35) & nonzero

        # Construct a room footprint and fill holes to extend floor through occlusions.
        try:
            observed_u8 = nonzero.astype(np.uint8, copy=False) * 255
            kernel = np.ones((5, 5), dtype=np.uint8)
            observed_closed = cv2.morphologyEx(observed_u8, cv2.MORPH_CLOSE, kernel)
            room = observed_closed > 0
            room = ndi.binary_fill_holes(room)
        except Exception:
            room = nonzero

        # Suppress speckle obstacles: keep only larger connected components.
        try:
            labeled, num = ndi.label(obst_m)
            if int(num) > 0:
                comp_sizes = np.bincount(labeled.ravel())
                if comp_sizes.size:
                    comp_sizes[0] = 0
                keep = np.zeros_like(obst_m, dtype=bool)
                min_cells = 25
                for lbl, sz in enumerate(comp_sizes):
                    if lbl == 0:
                        continue
                    if int(sz) >= int(min_cells):
                        keep |= (labeled == lbl)
                obst_m = keep
        except Exception:
            pass

        floor_m = (room & ~obst_m) | floor_seed

        rgb = np.zeros((bins, bins, 3), np.uint8)
        rgb[floor_m] = (85, 210, 95)   # green walkable (fills under islands)
        rgb[obst_m] = (70, 70, 110)    # gray large obstacles
        rgb = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_NEAREST)
        return rgb


class MapAnythingDepthSource:
    """Client for invoking MapAnything inference service and persisting results."""

    def __init__(self, config: Optional[ServiceConfig] = None) -> None:
        self.config = config or load_service_config()
        self.session = requests.Session()
        self.logger = RateLimitedLogger(logging.getLogger(__name__), rate_limit_seconds=2.0)
        self.min_conf = float(self.config.performance.min_conf)
        self.storage = DepthStorageManager(
            Path(self.config.storage.depth_base),
            max_snapshots_per_camera=self.config.storage.max_snapshots_per_camera,
            retention_minutes=self.config.storage.snapshot_retention_minutes,
            max_total_bytes=getattr(self.config.storage, 'max_total_bytes', None),
            enable_async=getattr(self.config.storage, 'async_enabled', True),
            max_queue_size=max(1, int(getattr(self.config.storage, 'queue_size', 32))),
            worker_count=int(getattr(self.config.storage, 'async_workers', 0)),
            max_worker_count=int(getattr(self.config.storage, 'async_max_workers', 0)),
            enforce_async=bool(getattr(self.config.storage, 'enforce_async', True)),
            enforce_interval_s=float(getattr(self.config.storage, 'enforce_interval_s', 1.0)),
            size_hysteresis_ratio=float(getattr(self.config.storage, 'quota_hysteresis_ratio', 0.9)),
            zarr_clevel=int(getattr(self.config.storage, 'zarr_clevel', 5)),
            zarr_chunk_px=int(getattr(self.config.storage, 'zarr_chunk_px', 128)),
            min_conf=self.min_conf,
        )
        self.mono_interval = 1.0 / max(self.config.performance.mono_freq_hz, 1e-6)
        self.last_request_per_camera: Dict[str, float] = {}
        self.timeout = 5.0
        self._cache_lock = threading.Lock()
        self._depth_payload_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._floorplan_cache: "OrderedDict[Tuple[str, float, float], Dict[str, Any]]" = OrderedDict()
        self._floorplan_store_dir = Path(self.config.storage.depth_base) / "floorplans"
        self._floorplan_store_dir.mkdir(parents=True, exist_ok=True)
        self._max_depth_cache_entries = 16
        self._max_floorplan_cache_entries = 24
        self._batch_lock = threading.Lock()
        self._batch_condition = threading.Condition(self._batch_lock)
        self._batch_queue: Deque[_BatchItem] = deque()
        self._batch_worker: Optional[threading.Thread] = None
        self._batch_shutdown = False
        batch_size = max(1, int(getattr(self.config.performance, 'multi_batch_size', 1)))
        self._batch_size = batch_size
        # Flush quickly enough to avoid latency while still gathering a few cameras.
        candidate_flush = self.mono_interval * 0.25
        self._batch_flush_s = max(0.005, min(0.02, candidate_flush))
        self._multi_batches_attempted = 0
        self._multi_batches_succeeded = 0
        self._multi_batches_fallback = 0
        self._multi_scene_counter = 0

    def close(self) -> None:
        try:
            self._stop_batch_worker()
        except Exception:
            pass
        try:
            self.storage.flush(timeout=2.0)
            self.storage.shutdown(wait=False)
        except Exception:
            pass

    def should_infer(self, camera_id: str, timestamp_s: float) -> bool:
        last = self.last_request_per_camera.get(camera_id)
        if last is None:
            return True
        return (timestamp_s - last) >= self.mono_interval

    def maybe_infer_mono(
        self,
        camera_id: str,
        frame_bgr: np.ndarray,
        calib_bundle: Optional[Mapping[str, object]],
        timestamp_s: Optional[float] = None,
    ) -> Optional[DepthResult]:
        ts = timestamp_s or time.time()
        if not self.should_infer(camera_id, ts):
            return None
        try:
            view_result, view_payload = self._prepare_view(camera_id, frame_bgr, calib_bundle)
            if self._batch_size <= 1:
                result = self._run_single_request(camera_id, ts, view_result, view_payload)
            else:
                result = self._submit_batch_request(camera_id, ts, view_result, view_payload)
            self.last_request_per_camera[camera_id] = ts
            return result
        except Exception as exc:
            self.logger.error(f"Mono depth inference failed for {camera_id}: {exc}")
            return None

    def _prepare_view(
        self,
        camera_id: str,
        frame_bgr: np.ndarray,
        calib_bundle: Optional[Mapping[str, object]],
    ) -> Tuple[ViewBuildResult, Dict[str, object]]:
        view_result = build_mono_view(frame_bgr, camera_id, calib_bundle)
        payload = self._serialize_view_payload(camera_id, view_result)
        return view_result, payload

    def _serialize_view_payload(self, camera_id: str, view_result: ViewBuildResult) -> Dict[str, object]:
        view_payload = dict(view_result.payload)
        view_payload['cam_id'] = camera_id

        shape = view_payload.get('shape') or view_result.resized_shape
        view_payload['shape'] = [int(shape[0]), int(shape[1]), int(shape[2])]

        if 'img_b64' not in view_payload:
            frame_rgb = view_payload.get('img')
            if frame_rgb is None:
                raise ValueError("MapAnything view payload missing img_b64 data")
            img_bytes = np.ascontiguousarray(frame_rgb).tobytes()
            view_payload['img_b64'] = base64.b64encode(img_bytes).decode('ascii')
        view_payload.pop('img', None)

        if 'intrinsics' not in view_payload and view_result.intrinsics is not None:
            view_payload['intrinsics'] = view_result.intrinsics.tolist()

        return view_payload

    def _submit_batch_request(
        self,
        camera_id: str,
        timestamp_s: float,
        view_result: ViewBuildResult,
        view_payload: Dict[str, object],
    ) -> DepthResult:
        future: Future = Future()
        item = _BatchItem(
            camera_id=camera_id,
            timestamp_s=timestamp_s,
            view_result=view_result,
            view_payload=view_payload,
            future=future,
        )
        with self._batch_condition:
            self._start_batch_worker_locked()
            self._batch_queue.append(item)
            self._batch_condition.notify()
        return future.result()

    def _start_batch_worker_locked(self) -> None:
        if self._batch_worker is not None and self._batch_worker.is_alive():
            return
        self._batch_shutdown = False
        self._batch_worker = threading.Thread(target=self._batch_loop, name="MapAnythingBatcher", daemon=True)
        self._batch_worker.start()

    def _batch_loop(self) -> None:
        while True:
            with self._batch_condition:
                while not self._batch_queue and not self._batch_shutdown:
                    self._batch_condition.wait()
                if self._batch_shutdown and not self._batch_queue:
                    return
                if not self._batch_queue:
                    continue
                first = self._batch_queue.popleft()
                batch: List[_BatchItem] = [first]
                if self._batch_size > 1:
                    deadline = time.perf_counter() + self._batch_flush_s
                    while len(batch) < self._batch_size:
                        if self._batch_queue:
                            batch.append(self._batch_queue.popleft())
                            continue
                        if self._batch_shutdown:
                            break
                        remaining = deadline - time.perf_counter()
                        if remaining <= 0:
                            break
                        self._batch_condition.wait(timeout=remaining)
                    while self._batch_queue and len(batch) < self._batch_size:
                        batch.append(self._batch_queue.popleft())
            try:
                self._process_batch(batch)
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.error(f"Batch processing failed: {exc}")
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(exc)

    def _process_batch(self, batch: List[_BatchItem]) -> None:
        if not batch:
            return
        if len(batch) == 1:
            self._execute_single(batch[0])
            return
        self._multi_batches_attempted += 1
        try:
            parsed = self._invoke_multi(batch)
        except Exception as exc:
            self._multi_batches_fallback += 1
            self.logger.warning(f"/infer_multi failed ({exc}); falling back to individual requests")
            for item in batch:
                self._execute_single(item, suppress_error_log=True)
            return
        self._multi_batches_succeeded += 1
        for item in batch:
            entry = parsed.get(item.camera_id)
            if entry is None:
                self.logger.warning(f"/infer_multi response missing camera {item.camera_id}; retrying singly")
                self._execute_single(item, suppress_error_log=True)
                continue
            depth, conf, mask = entry
            self._resolve_item(item, depth, conf, mask)

    def _invoke_multi(self, batch: List[_BatchItem]) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        scene_id = self._next_scene_id()
        payload = {
            'scene_id': scene_id,
            'views': [dict(item.view_payload) for item in batch],
        }
        response_json = self._post_json('/infer_multi', payload)
        return self._parse_multi_response(response_json)

    def _execute_single(self, item: _BatchItem, *, suppress_error_log: bool = False) -> None:
        try:
            payload = {'view': dict(item.view_payload)}
            response_json = self._post_json('/infer_mono', payload)
            depth, conf, mask = self._parse_response(response_json)
            self._resolve_item(item, depth, conf, mask)
        except Exception as exc:
            if not suppress_error_log:
                self.logger.warning(f"/infer_mono fallback failed for {item.camera_id}: {exc}")
            if not item.future.done():
                item.future.set_exception(exc)

    def _resolve_item(
        self,
        item: _BatchItem,
        depth: np.ndarray,
        conf: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        try:
            result = self._finalize_depth_result(item.camera_id, item.timestamp_s, item.view_result, depth, conf, mask)
            if not item.future.done():
                item.future.set_result(result)
        except Exception as exc:
            if not item.future.done():
                item.future.set_exception(exc)

    def _finalize_depth_result(
        self,
        camera_id: str,
        timestamp_s: float,
        view_result: ViewBuildResult,
        depth: np.ndarray,
        conf: np.ndarray,
        mask: np.ndarray,
    ) -> DepthResult:
        depth_aligned, conf_aligned, mask_aligned = self._align_to_original_shape(depth, conf, mask, view_result)
        ts_us = int(timestamp_s * 1_000_000)
        storage_path = self.storage.store(camera_id, ts_us, depth_aligned, conf_aligned, mask_aligned)
        summary = self._compute_summary(depth_aligned, conf_aligned, mask_aligned)
        return DepthResult(
            camera_id=camera_id,
            ts_us=ts_us,
            depth=depth_aligned,
            conf=conf_aligned,
            mask=mask_aligned,
            intrinsics=view_result.native_intrinsics.copy() if view_result.native_intrinsics is not None else None,
            native_intrinsics=view_result.native_intrinsics.copy() if view_result.native_intrinsics is not None else None,
            summary=summary,
            storage_path=storage_path,
        )

    def _run_single_request(
        self,
        camera_id: str,
        timestamp_s: float,
        view_result: ViewBuildResult,
        view_payload: Dict[str, object],
    ) -> DepthResult:
        response_json = self._post_json('/infer_mono', {'view': dict(view_payload)})
        depth, conf, mask = self._parse_response(response_json)
        return self._finalize_depth_result(camera_id, timestamp_s, view_result, depth, conf, mask)

    def _parse_multi_response(
        self,
        response: Dict[str, object],
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        depth_map = response.get('depth_b64')
        conf_map = response.get('conf_b64')
        mask_map = response.get('mask_b64') or {}
        shapes_map = response.get('shapes')
        if not isinstance(depth_map, dict) or not isinstance(conf_map, dict) or not isinstance(shapes_map, dict):
            raise ValueError('Multi response missing depth/conf/shape dictionaries')
        results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for cam_id, depth_b64 in depth_map.items():
            shape = shapes_map.get(cam_id)
            if not (isinstance(shape, (list, tuple)) and len(shape) == 2):
                raise ValueError(f"Missing shape metadata for camera {cam_id}")
            height, width = int(shape[0]), int(shape[1])
            conf_b64 = conf_map.get(cam_id)
            if not isinstance(conf_b64, str):
                raise ValueError(f"Missing confidence tensor for camera {cam_id}")
            mask_b64 = mask_map.get(cam_id)
            try:
                depth = np.frombuffer(base64.b64decode(depth_b64), dtype=np.float32).reshape((height, width))
                conf = np.frombuffer(base64.b64decode(conf_b64), dtype=np.float32).reshape((height, width))
                if isinstance(mask_b64, str):
                    mask = np.frombuffer(base64.b64decode(mask_b64), dtype=np.uint8).reshape((height, width)).astype(bool)
                else:
                    mask = np.ones((height, width), dtype=bool)
            except Exception as exc:
                raise ValueError(f"Failed to decode multi response for camera {cam_id}: {exc}") from exc
            results[cam_id] = (depth, conf, mask)
        return results

    def _next_scene_id(self) -> str:
        self._multi_scene_counter = (self._multi_scene_counter + 1) % 1_000_000
        return f"mono-batch-{self._multi_scene_counter}"

    def _stop_batch_worker(self) -> None:
        with self._batch_condition:
            self._batch_shutdown = True
            self._batch_condition.notify_all()
        if self._batch_worker is not None and self._batch_worker.is_alive():
            self._batch_worker.join(timeout=1.0)
        pending: List[_BatchItem] = []
        with self._batch_condition:
            while self._batch_queue:
                pending.append(self._batch_queue.popleft())
        for item in pending:
            if not item.future.done():
                item.future.set_exception(RuntimeError('Batch worker stopped before completion'))

    def _post_json(self, endpoint: str, payload: Dict[str, object]) -> Dict[str, object]:
        url = f"{self.config.service.base_url}{endpoint}"
        headers = {"X-API-Key": self.config.service.api_key}
        backoff = 0.5
        for attempt in range(1, 4):
            try:
                response = self.session.post(url, json=payload, headers=headers, timeout=self.timeout)
                if response.status_code == 200:
                    return response.json()
                self.logger.warning(f"MapAnything service returned {response.status_code}: {response.text}")
            except requests.RequestException as exc:
                self.logger.warning(f"MapAnything request failed (attempt {attempt}): {exc}")
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 4.0)
        raise RuntimeError("MapAnything request failed after retries")

    def _parse_response(self, response: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        shape = response.get("shape")
        if not (isinstance(shape, (list, tuple)) and len(shape) == 2):
            raise ValueError(f"Depth response missing shape metadata (got {shape!r})")
        height, width = int(shape[0]), int(shape[1])

        depth_b64 = response.get("depth_b64") or response.get("depth_z_b64")
        conf_b64 = response.get("conf_b64")
        mask_b64 = response.get("mask_b64")
        missing = [name for name, value in (
            ("depth_b64", depth_b64),
            ("conf_b64", conf_b64),
            ("mask_b64", mask_b64),
        ) if not isinstance(value, str)]
        if missing:
            raise ValueError(f"Depth response missing encoded tensors: {missing}")

        try:
            depth = np.frombuffer(base64.b64decode(depth_b64), dtype=np.float32).reshape((height, width))
            conf = np.frombuffer(base64.b64decode(conf_b64), dtype=np.float32).reshape((height, width))
            mask = np.frombuffer(base64.b64decode(mask_b64), dtype=np.uint8).reshape((height, width)).astype(bool)
        except Exception as exc:
            raise ValueError(f"Failed to decode depth response: {exc}") from exc
        return depth, conf, mask

    def _align_to_original_shape(
        self,
        depth: np.ndarray,
        conf: np.ndarray,
        mask: np.ndarray,
        view_result: ViewBuildResult,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        target_h, target_w = view_result.original_shape[:2]
        current_h, current_w = depth.shape[:2]
        if (target_h, target_w) == (current_h, current_w):
            return depth, conf, mask
        depth_resized = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        conf_resized = cv2.resize(conf, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        mask_uint8 = cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        mask_resized = mask_uint8.astype(bool)
        return depth_resized, conf_resized, mask_resized

    def _compute_summary(self, depth: np.ndarray, conf: np.ndarray, mask: np.ndarray) -> DepthSummary:
        valid = mask & np.isfinite(depth) & (conf >= self.min_conf) & (depth > 0.0)
        total = depth.size
        if total == 0:
            return DepthSummary(median=0.0, p10=0.0, p90=0.0, conf_mean=0.0, valid_ratio=0.0, sample_count=0)
        if not np.any(valid):
            return DepthSummary(median=0.0, p10=0.0, p90=0.0, conf_mean=float(conf.mean()), valid_ratio=0.0, sample_count=0)
        valid_depth = depth[valid]
        return DepthSummary(
            median=float(np.median(valid_depth)),
            p10=float(np.percentile(valid_depth, 10)),
            p90=float(np.percentile(valid_depth, 90)),
            conf_mean=float(conf[valid].mean()),
            valid_ratio=float(valid.sum() / total),
            sample_count=int(valid.sum()),
        )

    def load_latest_depth(self, camera_id: str, ts_max_us: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Return the newest cached payload up to ts_max_us (microseconds)."""
        cache_key = camera_id
        ts_cutoff = None
        if ts_max_us is not None:
            try:
                ts_cutoff = int(ts_max_us)
            except Exception:
                ts_cutoff = None
        with self._cache_lock:
            cached = self._depth_payload_cache.get(cache_key)
            if cached:
                try:
                    cached_ts = int(cached.get('ts', 0) or 0)
                except Exception:
                    cached_ts = 0
                if ts_cutoff is None or cached_ts <= ts_cutoff:
                    return dict(cached)

        path = self.storage.latest_entry(camera_id, ts_cutoff)
        if not path:
            return None
        datasets = self.storage.load_datasets(path)
        if not datasets:
            return None
        depth = datasets['depth'].astype(np.float32, copy=False)
        conf = datasets['conf'].astype(np.float32, copy=False)
        mask = datasets['mask'].astype(np.uint8, copy=False)
        height, width = depth.shape[:2]
        ts_us = int(path.stem)
        payload = {
            'ts': ts_us,
            'depth_b64': base64.b64encode(depth.tobytes()).decode('ascii'),
            'conf_b64': base64.b64encode(conf.tobytes()).decode('ascii'),
            'mask_b64': base64.b64encode(mask.tobytes()).decode('ascii'),
            'shape': [int(height), int(width)],
        }
        with self._cache_lock:
            self._depth_payload_cache[cache_key] = dict(payload)
            self._depth_payload_cache.move_to_end(cache_key, last=True)
            while len(self._depth_payload_cache) > self._max_depth_cache_entries:
                self._depth_payload_cache.popitem(last=False)
        return payload

    @staticmethod
    def _sanitize_camera_id(camera_id: str) -> str:
        safe = ''.join(ch if ch.isalnum() or ch in {'-', '_', '.'} else '_' for ch in camera_id.strip())
        return safe or "camera"

    @staticmethod
    def _format_param(value: float) -> str:
        text = f"{float(value):.6f}".rstrip('0').rstrip('.')
        if not text:
            text = "0"
        if text.startswith('-'):
            text = 'neg' + text[1:]
        return text.replace('.', 'p')

    def _floorplan_path(self, camera_id: str, grid_res_m: float, max_extent_m: float) -> Path:
        safe_cam = self._sanitize_camera_id(camera_id)
        filename = f"grid{self._format_param(grid_res_m)}__ext{self._format_param(max_extent_m)}.json"
        return self._floorplan_store_dir / safe_cam / filename

    def _load_floorplan_from_disk(
        self,
        camera_id: str,
        grid_res_m: float,
        max_extent_m: float,
        expected_units: Optional[str] = None,
        expected_calibration_fingerprint: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        path = self._floorplan_path(camera_id, grid_res_m, max_extent_m)
        if not path.exists():
            return None
        try:
            with path.open('r', encoding='utf-8') as fh:
                payload = json.load(fh)
            if isinstance(payload, dict):
                if not _floorplan_cache_contract_matches(
                    payload,
                    expected_units=expected_units,
                    expected_calibration_fingerprint=expected_calibration_fingerprint,
                ):
                    return None
                payload.setdefault('camera_id', camera_id)
                return payload
        except Exception as exc:
            self.logger.debug(f"Failed to load cached floorplan for {camera_id}: {exc}")
        return None

    def _persist_floorplan_to_disk(
        self,
        camera_id: str,
        grid_res_m: float,
        max_extent_m: float,
        payload: Mapping[str, Any],
    ) -> None:
        path = self._floorplan_path(camera_id, grid_res_m, max_extent_m)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            to_store = dict(payload)
            to_store.setdefault("frame", _FLOORPLAN_FRAME)
            to_store.setdefault("orientation", _FLOORPLAN_ORIENTATION)
            to_store.setdefault("floorplan_contract_version", int(_FLOORPLAN_CONTRACT_VERSION))
            to_store.pop('served_from_cache', None)
            tmp_path = path.with_suffix(path.suffix + '.tmp')
            with tmp_path.open('w', encoding='utf-8') as fh:
                json.dump(to_store, fh, separators=(',', ':'))
            tmp_path.replace(path)
        except Exception as exc:
            self.logger.debug(f"Failed to persist floorplan for {camera_id}: {exc}")

    def update_depth_cache(self, result: DepthResult) -> None:
        try:
            depth_bytes = result.depth.astype(np.float32, copy=False).tobytes()
            conf_bytes = result.conf.astype(np.float32, copy=False).tobytes()
            mask_bytes = result.mask.astype(np.uint8, copy=False).tobytes()
        except Exception as exc:
            self.logger.debug(f"Depth cache serialization failed for {result.camera_id}: {exc}")
            return
        payload = {
            'ts': result.ts_us,
            'depth_b64': base64.b64encode(depth_bytes).decode('ascii'),
            'conf_b64': base64.b64encode(conf_bytes).decode('ascii'),
            'mask_b64': base64.b64encode(mask_bytes).decode('ascii'),
            'shape': [int(result.depth.shape[0]), int(result.depth.shape[1])],
        }
        with self._cache_lock:
            self._depth_payload_cache[result.camera_id] = payload
            self._depth_payload_cache.move_to_end(result.camera_id, last=True)
            while len(self._depth_payload_cache) > self._max_depth_cache_entries:
                self._depth_payload_cache.popitem(last=False)

    def precompute_for_cameras(
        self,
        camera_ids: Iterable[str],
        max_age_sec: float = 60.0,
        grid_res_m: float = 0.5,
        max_extent_m: float = 20.0,
    ) -> None:
        for cam_id in camera_ids:
            if not cam_id:
                continue
            try:
                self.load_latest_depth(cam_id)
            except Exception as exc:
                self.logger.debug(f"Depth cache warmup for {cam_id} failed: {exc}")
                continue
            try:
                self.generate_topdown_floorplan(
                    cam_id,
                    max_age_sec=max_age_sec,
                    grid_res_m=grid_res_m,
                    max_extent_m=max_extent_m,
                )
            except Exception as exc:
                self.logger.debug(f"Floorplan warmup for {cam_id} failed: {exc}")

    def generate_topdown_floorplan(
        self,
        camera_id: str,
        max_age_sec: float = 60.0,
        grid_res_m: float = 0.5,
        max_extent_m: float = 20.0,
        cache_only: bool = False,
    ) -> Dict[str, Any]:
        """Generate a per-camera top-down (XZ) blueprint view from the latest depth snapshot."""
        if not camera_id:
            return {'error': 'camera_required', 'ts': int(time.time() * 1_000_000)}

        # Ensure any recently queued depth snapshots are flushed to disk before attempting
        # to read for a regenerate request. This avoids a race where floorplan generation
        # runs immediately after inference but before the async writer persists the zarr.
        if not cache_only:
            try:
                self.storage.flush(timeout=1.5)
            except Exception:
                pass

        cache_key = (camera_id, float(grid_res_m), float(max_extent_m))
        calib_bundle = getattr(self, 'calibration_bundle', None) or {}
        expected_flip = _expected_floorplan_flip(calib_bundle, camera_id)
        expected_units = _expected_floorplan_units_from_calibration_bundle(calib_bundle)
        expected_calibration_fingerprint = _floorplan_calibration_fingerprint(calib_bundle, camera_id)
        now_us = int(time.time() * 1_000_000)
        with self._cache_lock:
            cached = self._floorplan_cache.get(cache_key)
            if cached:
                if not _floorplan_cache_contract_matches(
                    cached,
                    expected_units=expected_units,
                    expected_calibration_fingerprint=expected_calibration_fingerprint,
                ):
                    cached = None
            if cached:
                if cache_only:
                    payload = dict(cached)
                    payload['served_from_cache'] = True
                    return payload
                age_us = now_us - cached.get('snapshot_ts', cached.get('ts', 0))
                if age_us <= int(max(0.0, max_age_sec) * 1_000_000):
                    payload = dict(cached)
                    payload['served_from_cache'] = True
                    return payload
            # Fall through to load from disk or recompute when cache is empty
            # so callers receive a floorplan without additional interaction.

        disk_payload = self._load_floorplan_from_disk(
            camera_id,
            grid_res_m,
            max_extent_m,
            expected_units=expected_units,
            expected_calibration_fingerprint=expected_calibration_fingerprint,
        )
        if disk_payload:
            snapshot_ts = disk_payload.get('snapshot_ts', disk_payload.get('ts'))
            if isinstance(snapshot_ts, (int, float)):
                age_us = now_us - int(snapshot_ts)
            else:
                age_us = None
            if cache_only or age_us is None or age_us <= int(max(0.0, max_age_sec) * 1_000_000):
                with self._cache_lock:
                    self._floorplan_cache[cache_key] = dict(disk_payload)
                    self._floorplan_cache.move_to_end(cache_key, last=True)
                    while len(self._floorplan_cache) > self._max_floorplan_cache_entries:
                        self._floorplan_cache.popitem(last=False)
                payload = dict(disk_payload)
                payload['served_from_cache'] = True
                return payload

        max_age_us = int(max(0.0, max_age_sec) * 1_000_000)
        ts_cutoff = now_us - max_age_us if max_age_us > 0 else None

        path_entry = self.storage.latest_entry(camera_id, now_us)
        if not path_entry:
            return {'error': 'no_depth', 'camera_id': camera_id, 'ts': now_us}

        if ts_cutoff is not None:
            try:
                snapshot_ts = int(path_entry.stem)
            except ValueError:
                snapshot_ts = None
            if snapshot_ts is None or snapshot_ts < ts_cutoff:
                return {'error': 'stale_depth', 'camera_id': camera_id, 'ts': now_us}

        datasets = self.storage.load_datasets(path_entry)
        if not datasets:
            return {'error': 'load_failed', 'camera_id': camera_id, 'ts': now_us}

        depth = datasets.get('depth')
        conf = datasets.get('conf')
        mask = datasets.get('mask')
        if depth is None or conf is None or mask is None:
            return {'error': 'invalid_snapshot', 'camera_id': camera_id, 'ts': now_us}

        scene_per_m, s_obj_to_m = _scene_units_per_meter_from_calibration_bundle(calib_bundle)
        cameras_node = calib_bundle.get('cameras') if isinstance(calib_bundle, dict) else {}
        k_table = cameras_node.get('K') if isinstance(cameras_node, dict) else {}
        e_table = cameras_node.get('E') if isinstance(cameras_node, dict) else {}

        intr = None
        extr = None
        if isinstance(k_table, dict):
            intr = k_table.get(camera_id)
        if isinstance(e_table, dict):
            extr = e_table.get(camera_id)

        if intr is None or extr is None:
            return {'error': 'missing_calibration', 'camera_id': camera_id, 'ts': now_us}

        depth = np.asarray(depth, dtype=np.float32)
        conf = np.asarray(conf, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.uint8) > 0
        if depth.ndim != 2 or conf.shape != depth.shape or mask.shape != depth.shape:
            return {'error': 'shape_mismatch', 'camera_id': camera_id, 'ts': now_us}

        intr_arr = np.asarray(intr, dtype=np.float32).reshape(-1)
        if intr_arr.size == 4:
            fx, fy, cx, cy = [float(v) for v in intr_arr]
        elif intr_arr.size == 9:
            k_mat = intr_arr.reshape(3, 3)
            fx = float(k_mat[0, 0])
            fy = float(k_mat[1, 1])
            cx = float(k_mat[0, 2])
            cy = float(k_mat[1, 2])
        else:
            return {'error': 'bad_intrinsics', 'camera_id': camera_id, 'ts': now_us}

        if not all(np.isfinite([fx, fy, cx, cy])) or fx == 0.0 or fy == 0.0:
            return {'error': 'invalid_intrinsics', 'camera_id': camera_id, 'ts': now_us}

        # PERMISSIVE validity: only reject truly invalid depth values
        # Do NOT hard-filter by mask or confidence - use them as soft weights instead
        valid = np.isfinite(depth)
        valid &= depth > 0.1
        valid &= depth < 50.0
        # Note: mask and conf are used as weights below, not hard filters

        if not np.any(valid):
            return {'error': 'no_points', 'camera_id': camera_id, 'ts': now_us, 'point_count': 0}

        # Extract confidence values for valid points - use as weights, not filter
        # Combine mask (as 0/1) and conf into a single weight
        # Points inside mask with high conf get weight ~1.0
        # Points outside mask or low conf get lower weights but still contribute
        soft_mask = np.where(mask, 1.0, 0.15).astype(np.float32)
        # Clamp confidence to [0.05, 1.0] to avoid zero weights
        conf_clamped = np.clip(conf, 0.05, 1.0)
        # Combined weight = soft_mask * confidence
        combined_weight = soft_mask * conf_clamped
        # Extract weights for valid points
        pts_weight = combined_weight[valid].astype(np.float32)

        h_img, w_img = depth.shape
        grid_u, grid_v = np.meshgrid(
            np.arange(w_img, dtype=np.float32),
            np.arange(h_img, dtype=np.float32),
            indexing='xy'
        )

        # Floorplan grids stay anchored to the canonical camera-local X/Z frame.
        # Do not remap the image axes here using the BEV/world flip heuristic.
        x_cam = (grid_u - cx) * depth / fx
        y_cam = (grid_v - cy) * depth / fy
        z_cam = depth

        pts_cam = np.stack([x_cam[valid], y_cam[valid], z_cam[valid]], axis=1)

        e_arr = np.asarray(extr, dtype=np.float32)
        if e_arr.size == 16:
            e_mat = e_arr.reshape(4, 4, order='F')
        elif e_arr.shape == (3, 4):
            e_mat = np.eye(4, dtype=np.float32)
            e_mat[:3, :4] = e_arr
        elif e_arr.shape == (4, 4):
            e_mat = e_arr
        else:
            return {'error': 'bad_extrinsics', 'camera_id': camera_id, 'ts': now_us}

        try:
            twc = np.linalg.inv(e_mat)
        except np.linalg.LinAlgError:
            return {'error': 'extrinsics_singular', 'camera_id': camera_id, 'ts': now_us}

        pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1), dtype=np.float32)], axis=1)
        pts_world_h = pts_cam_h @ twc.T
        pts_world = pts_world_h[:, :3]

        pts_depth = pts_cam[:, 2]
        pts_y = pts_world[:, 1]
        x_cam_pts = pts_cam[:, 0]
        z_cam_pts = pts_cam[:, 2]

        # Estimate floor Y from horizontal surfaces and compute per-point height above floor (AGL).
        floor_y = 0.0
        floor_est_meta: Dict[str, Any] = {"mode": "uninitialized"}
        pts_y_agl = pts_y
        try:
            twc_row_y = twc[1].astype(np.float32, copy=False)
            pts_y_agl = (
                (pts_cam[:, 0] * twc_row_y[0])
                + ((-pts_cam[:, 1]) * twc_row_y[1])
                + (pts_cam[:, 2] * twc_row_y[2])
                + twc_row_y[3]
            ).astype(np.float32, copy=False)

            valid_normals = valid & mask
            normals_cam = DepthStorageManager._compute_normals(
                depth.astype(np.float32, copy=False),
                valid_normals,
                fx,
                fy,
                cx,
                cy,
            )
            normals_cam = np.asarray(normals_cam, dtype=np.float32)
            normals_cam[..., 1] *= -1.0
            normals_cam_flat = normals_cam[valid]
            r_wc = twc[:3, :3].astype(np.float32, copy=False)
            normals_world_flat = (r_wc @ normals_cam_flat.T).T
            good_point = mask[valid] & (conf_clamped[valid] >= 0.2)
            floor_w = np.where(good_point, pts_weight, 0.0).astype(np.float32, copy=False)
            floor_y, floor_est_meta = _estimate_floor_y_from_horizontal_points(
                pts_y_agl,
                normals_world_flat,
                floor_w,
                horiz_dot_thresh=float(_FLOORPLAN_AGL_HORIZ_DOT_THRESH),
            )
        except Exception as exc:
            try:
                floor_y = float(np.nanpercentile(pts_y_agl, 1.0)) if pts_y_agl.size else 0.0
            except Exception:
                floor_y = float(np.nanmin(pts_y_agl)) if pts_y_agl.size else 0.0
            floor_est_meta = {"mode": "error", "error": str(exc), "floor_y": float(floor_y)}

        height_agl_pts = (pts_y_agl - float(floor_y)).astype(np.float32, copy=False)
        height_agl_pts = np.clip(height_agl_pts, 0.0, float(_FLOORPLAN_AGL_HEIGHT_CLIP_M)).astype(np.float32, copy=False)

        if x_cam_pts.size == 0 or z_cam_pts.size == 0:
            return {'error': 'no_points', 'camera_id': camera_id, 'ts': now_us, 'point_count': 0}

        pad_x = max(0.5, grid_res_m * 2.0)
        pad_z = max(0.5, grid_res_m * 2.0)

        max_x_abs = float(np.max(np.abs(x_cam_pts))) if x_cam_pts.size else 0.0
        if not np.isfinite(max_x_abs):
            max_x_abs = 0.0
        forward_max = float(np.max(z_cam_pts)) if z_cam_pts.size else 0.0
        if not np.isfinite(forward_max):
            forward_max = 0.0

        half_width = max_x_abs + pad_x
        forward_extent = max(0.0, forward_max) + pad_z
        if max_extent_m > 0:
            max_extent = float(max_extent_m)
            min_half_width = max_extent * float(_FLOORPLAN_MIN_HALF_WIDTH_FRACTION)
            min_forward = max_extent * float(_FLOORPLAN_MIN_FORWARD_FRACTION)
            half_width = max(half_width, min_half_width)
            forward_extent = max(forward_extent, min_forward)
            half_width = min(half_width, max_extent)
            forward_extent = min(forward_extent, max_extent)

        half_width = max(half_width, grid_res_m * 0.5)
        forward_extent = max(forward_extent, grid_res_m)

        min_x = -half_width
        max_x = half_width
        min_z = 0.0
        max_z = forward_extent

        width_m = max_x - min_x
        height_m = max_z - min_z

        w_px = max(1, int(np.ceil(width_m / grid_res_m)))
        h_px = max(1, int(np.ceil(height_m / grid_res_m)))

        x_norm = np.clip((x_cam_pts - min_x) / width_m, 0.0, 0.999999)
        z_norm = np.clip((z_cam_pts - min_z) / height_m, 0.0, 0.999999)
        x_idx = np.clip(np.floor(x_norm * w_px).astype(np.int32), 0, w_px - 1)
        z_idx = np.clip(np.floor((1.0 - z_norm) * h_px).astype(np.int32), 0, h_px - 1)

        density_grid = np.zeros((h_px, w_px), dtype=np.float32)
        distance_sum = np.zeros((h_px, w_px), dtype=np.float32)
        distance_count = np.zeros((h_px, w_px), dtype=np.uint32)
        height_grid = np.full((h_px, w_px), -np.inf, dtype=np.float32)
        height_agl_max_grid = np.full((h_px, w_px), -np.inf, dtype=np.float32)
        height_agl_min_grid = np.full((h_px, w_px), np.inf, dtype=np.float32)

        # Confidence-weighted height aggregation
        weighted_height_sum = np.zeros((h_px, w_px), dtype=np.float64)
        weight_sum = np.zeros((h_px, w_px), dtype=np.float64)
        weighted_agl_sum = np.zeros((h_px, w_px), dtype=np.float64)
        agl_weight_sum = np.zeros((h_px, w_px), dtype=np.float64)
        agl_floor_support_grid = np.zeros((h_px, w_px), dtype=np.uint32)
        agl_obstacle_support_grid = np.zeros((h_px, w_px), dtype=np.uint32)

        indices = (z_idx, x_idx)
        # Density: count of points (unweighted for backward compat)
        np.add.at(density_grid, indices, 1.0)
        # Distance: weighted by confidence
        np.add.at(distance_sum, indices, (pts_depth * pts_weight).astype(np.float32, copy=False))
        np.add.at(distance_count, indices, 1)
        # Height: accumulate weighted sum and weights for weighted mean
        np.add.at(weighted_height_sum, indices, (pts_y * pts_weight).astype(np.float64))
        np.add.at(weight_sum, indices, pts_weight.astype(np.float64))
        np.add.at(weighted_agl_sum, indices, (height_agl_pts * pts_weight).astype(np.float64))
        np.add.at(agl_weight_sum, indices, pts_weight.astype(np.float64))
        floor_band = float(_FLOORPLAN_AGL_FLOOR_SUPPORT_BAND_M)
        obstacle_band = float(_FLOORPLAN_AGL_OBSTACLE_SUPPORT_MIN_M)
        max_relevant_h = float(_FLOORPLAN_CLEAN_MAX_RELEVANT_H_M)
        np.add.at(agl_floor_support_grid, indices, (height_agl_pts <= floor_band).astype(np.uint32))
        np.add.at(
            agl_obstacle_support_grid,
            indices,
            ((height_agl_pts >= obstacle_band) & (height_agl_pts <= max_relevant_h)).astype(np.uint32),
        )
        # Also track max height (still useful for some visualizations)
        np.maximum.at(height_grid, indices, np.asarray(pts_y, dtype=np.float32))
        np.maximum.at(height_agl_max_grid, indices, height_agl_pts.astype(np.float32, copy=False))
        np.minimum.at(height_agl_min_grid, indices, height_agl_pts.astype(np.float32, copy=False))

        # Compute confidence-weighted mean height
        # Use weighted mean where we have weights, otherwise fall back to max
        has_weight = weight_sum > 1e-9
        # Avoid invalid division warnings: np.where evaluates both branches eagerly.
        weighted_mean_height = height_grid.astype(np.float32, copy=True)
        np.divide(
            weighted_height_sum,
            weight_sum,
            out=weighted_mean_height,
            where=has_weight,
        )

        height_agl_mean_grid = height_agl_max_grid.astype(np.float32, copy=True)
        has_agl_weight = agl_weight_sum > 1e-9
        np.divide(
            weighted_agl_sum,
            agl_weight_sum,
            out=height_agl_mean_grid,
            where=has_agl_weight,
        )

        # For cells with no points at all, mark as NaN
        empty_cells = height_grid == -np.inf
        if np.any(empty_cells):
            weighted_mean_height[empty_cells] = np.nan
            height_grid[empty_cells] = np.nan
            height_agl_mean_grid[empty_cells] = np.nan
            height_agl_max_grid[empty_cells] = np.nan
            height_agl_min_grid[empty_cells] = np.nan

        # Use weighted mean as the primary height grid (preserves detail better than max)
        height_grid = weighted_mean_height

        clean_layers = None

        density_max = float(np.max(density_grid)) if density_grid.size else 0.0
        if density_max > 0.0:
            density_grid /= density_max

        height_grid, height_fill_meta = _postprocess_floorplan_height_grid(
            camera_id,
            height_grid,
            density_grid,
            min_x=min_x,
            max_x=max_x,
            min_z=min_z,
            max_z=max_z,
        )
        height_min = float(np.min(height_grid)) if height_grid.size else 0.0
        height_max = float(np.max(height_grid)) if height_grid.size else 0.0
        if height_grid.size:
            height_grid = height_grid - height_min
            height_min = 0.0
            height_max = float(np.max(height_grid)) if height_grid.size else 0.0

        # Compute clean BEV layers for every room. This classifies walkable floor
        # and obstacle surfaces in camera-local X/Z space so the overlay and the
        # visible floorplan share the same raster surface.
        try:
            y_up_pts = (-pts_cam[:, 1]).astype(np.float32, copy=False)
            obs_h, walk, clean_meta = _compute_kitchen_clean_floorplan_layers(
                camera_id,
                x_cam_pts=x_cam_pts,
                z_cam_pts=z_cam_pts,
                y_world_pts=y_up_pts,
                pts_weight=pts_weight,
                x_idx=x_idx,
                z_idx=z_idx,
                support_grid=distance_count,
            )
            if isinstance(clean_meta, dict):
                clean_meta = dict(clean_meta)
            else:
                clean_meta = {"mode": "clean_floorplan_layers"}
            clean_meta["mode"] = "clean_floorplan_layers"
            clean_meta["floor_estimate"] = floor_est_meta
            clean_layers = (obs_h, walk, clean_meta)
        except Exception:
            clean_layers = None

        # Compute height gradient magnitude for edge detection
        # First, fill empty/NaN cells with floor level so boundaries don't create false edges
        height_for_gradient = height_grid.copy()
        floor_level = float(np.nanmin(height_grid)) if np.any(np.isfinite(height_grid)) else 0.0
        height_for_gradient = np.nan_to_num(
            height_for_gradient,
            nan=floor_level,
            posinf=floor_level,
            neginf=floor_level,
        )

        # Also mask cells with zero density (no points) to floor level
        if density_grid is not None:
            empty_mask = density_grid < 1e-6
            height_for_gradient[empty_mask] = floor_level

        # Sobel filters capture directional derivatives
        gx = ndi.sobel(height_for_gradient, axis=1, mode='nearest')
        gz = ndi.sobel(height_for_gradient, axis=0, mode='nearest')
        gradient_mag = np.sqrt(gx ** 2 + gz ** 2)

        # Use 95th percentile normalization to prevent outliers from dominating
        gradient_flat = gradient_mag[np.isfinite(gradient_mag)]
        if gradient_flat.size > 0:
            gradient_p95 = float(np.percentile(gradient_flat, 95))
            if gradient_p95 > 1e-6:
                gradient_grid = np.clip(gradient_mag / gradient_p95, 0.0, 1.0).astype(np.float32)
            else:
                gradient_grid = np.zeros_like(height_grid, dtype=np.float32)
        else:
            gradient_grid = np.zeros_like(height_grid, dtype=np.float32)

        # Clean up any remaining NaN
        gradient_grid = np.nan_to_num(gradient_grid, nan=0.0, posinf=0.0, neginf=0.0)

        distance_grid = np.zeros((h_px, w_px), dtype=np.float32)
        nonzero_mask = distance_count > 0
        if np.any(nonzero_mask):
            distance_grid[nonzero_mask] = distance_sum[nonzero_mask] / distance_count[nonzero_mask]
            min_distance = float(np.min(distance_grid[nonzero_mask]))
            max_distance = float(np.max(distance_grid[nonzero_mask]))
        else:
            min_distance = 0.0
            max_distance = 0.0

        bounds = {
            'min_x': float(min_x),
            'max_x': float(max_x),
            'min_z': float(min_z),
            'max_z': float(max_z),
        }

        agl_max = 0.0
        try:
            finite_agl = np.isfinite(height_agl_mean_grid)
            if np.any(finite_agl):
                agl_max = float(np.percentile(height_agl_mean_grid[finite_agl], 99))
                if not np.isfinite(agl_max) or agl_max <= 1e-6:
                    agl_max = float(np.nanmax(height_agl_mean_grid[finite_agl]))
        except Exception:
            agl_max = 0.0
        if not np.isfinite(agl_max) or agl_max <= 1e-6:
            agl_max = float(_FLOORPLAN_AGL_HEIGHT_CLIP_M)

        payload: Dict[str, Any] = {
            'camera_id': camera_id,
            'ts': now_us,
            'snapshot_ts': int(path_entry.stem) if path_entry.stem.isdigit() else None,
            'frame': _FLOORPLAN_FRAME,
            'orientation': _FLOORPLAN_ORIENTATION,
            'floorplan_contract_version': int(_FLOORPLAN_CONTRACT_VERSION),
            'bounds': bounds,
            'scale_m_per_px': float(width_m / w_px if w_px else grid_res_m),
            'scale_scene_per_px': float((width_m / w_px if w_px else grid_res_m) * scene_per_m),
            'units': 'meters',
            's_obj_to_m': float(s_obj_to_m),
            'point_count': int(pts_cam.shape[0]),
            'density': {
                'grid_b64': base64.b64encode(density_grid.astype(np.float32, copy=False).ravel().tobytes()).decode('ascii'),
                'grid_shape': [int(h_px), int(w_px)],
                'value_min': 0.0,
                'value_max': 1.0,
            },
            'height': {
                'grid_b64': base64.b64encode(height_grid.astype(np.float32, copy=False).ravel().tobytes()).decode('ascii'),
                'grid_shape': [int(h_px), int(w_px)],
                'value_min': float(height_min),
                'value_max': float(height_max),
            },
            'height_agl': {
                'grid_b64': base64.b64encode(height_agl_mean_grid.astype(np.float32, copy=False).ravel().tobytes()).decode('ascii'),
                'grid_shape': [int(h_px), int(w_px)],
                'value_min': 0.0,
                'value_max': float(agl_max),
            },
            'distance': {
                'grid_b64': base64.b64encode(distance_grid.astype(np.float32, copy=False).ravel().tobytes()).decode('ascii'),
                'grid_shape': [int(h_px), int(w_px)],
                'value_min': float(min_distance),
                'value_max': float(max_distance),
            },
            'gradient': {
                'grid_b64': base64.b64encode(gradient_grid.astype(np.float32, copy=False).ravel().tobytes()).decode('ascii'),
                'grid_shape': [int(h_px), int(w_px)],
                'value_min': 0.0,
                'value_max': 1.0,
            },
            'height_fill_meta': height_fill_meta,
            'height_agl_meta': {
                'floor_y': float(floor_y),
                'floor_offset_m': float(agl_floor_offset_m),
                'floor_estimate': floor_est_meta,
            },
            'calibration_fingerprint': expected_calibration_fingerprint,
        }
        if clean_layers is not None:
            obstacle_height_grid, walkable_grid, clean_meta = clean_layers
            try:
                obs_max = float(np.nanmax(obstacle_height_grid)) if obstacle_height_grid.size else 0.0
            except Exception:
                obs_max = 0.0
            if not np.isfinite(obs_max) or obs_max <= 1e-6:
                obs_max = float(_FLOORPLAN_CLEAN_MAX_OBSTACLE_H_M)
            payload['obstacle_height'] = {
                'grid_b64': base64.b64encode(obstacle_height_grid.astype(np.float32, copy=False).ravel().tobytes()).decode('ascii'),
                'grid_shape': [int(h_px), int(w_px)],
                'value_min': 0.0,
                'value_max': float(obs_max),
            }
            payload['walkable'] = {
                'grid_b64': base64.b64encode(walkable_grid.astype(np.float32, copy=False).ravel().tobytes()).decode('ascii'),
                'grid_shape': [int(h_px), int(w_px)],
                'value_min': 0.0,
                'value_max': 1.0,
            }
            payload['clean_floorplan_meta'] = clean_meta
        payload['served_from_cache'] = False
        payload['grid_res_m'] = float(grid_res_m)
        payload['grid_res_scene'] = float(grid_res_m * scene_per_m)
        payload['max_extent_m'] = float(max_extent_m)
        payload['max_extent_scene'] = float(max_extent_m * scene_per_m)
        payload['image_flip'] = _floorplan_image_flip_payload(expected_flip)

        self._persist_floorplan_to_disk(camera_id, grid_res_m, max_extent_m, payload)

        with self._cache_lock:
            self._floorplan_cache[cache_key] = dict(payload)
            self._floorplan_cache.move_to_end(cache_key, last=True)
            while len(self._floorplan_cache) > self._max_floorplan_cache_entries:
                self._floorplan_cache.popitem(last=False)

        return payload


__all__ = [
    "DepthStorageManager",
    "MapAnythingDepthSource",
    "DepthResult",
    "DepthSummary",
]
