"""
Auto-calibrate camera extrinsics from the latest MapAnything depth cache.

This reads the newest cached depth snapshot per camera, fits a ground plane,
derives candidate camera poses (R, t) that point the optical axis toward +Z
with ground at Y=0, scores both normal directions using physical constraints
and world-space floor coverage, and optionally persists the selected
world→camera matrix into config/camera_calibration.json. Bad or implausible
solutions are rejected and never written.
"""
from __future__ import annotations

import base64
import json
import math
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from calibration_bundle import (
    load_intrinsics,
    save_extrinsics,
    _derive_k_from_intrinsics_model,  # type: ignore
    _resolution_from_spec,  # type: ignore
)
from config import AppConfig
from geometry.depth_source import MapAnythingDepthSource
from mapanything_config import load_service_config

# Defaults match the FE camera keys
DEFAULT_CAMERAS = ["living-room", "kitchen", "family-room"]
logger = logging.getLogger(__name__)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _load_camera_height_priors() -> Dict[str, float]:
    """
    Returns camera->expected_height_m from config/cameras.yaml if available.
    """
    cfg_path = ROOT / "config/cameras.yaml"
    if not cfg_path.exists():
        return {}
    try:
        data = yaml.safe_load(cfg_path.read_text()) or {}
    except Exception:
        return {}
    cams = data.get("cameras") or {}
    priors: Dict[str, float] = {}
    if isinstance(cams, dict):
        for entry in cams.values():
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            h = entry.get("height_m")
            if isinstance(name, str) and isinstance(h, (int, float)):
                priors[name] = float(h)
    return priors


def _load_camera_model_map() -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Returns (intrinsics_models, camera->model) from config/cameras.yaml
    """
    cfg_path = ROOT / "config/cameras.yaml"
    if not cfg_path.exists():
        return {}, {}
    data = yaml.safe_load(cfg_path.read_text()) or {}
    models = data.get("intrinsics_models") or {}
    cam_entries = data.get("cameras") or {}
    cam_to_model: Dict[str, str] = {}
    for entry in cam_entries.values():
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        model = entry.get("model") or entry.get("intrinsics_model")
        if isinstance(name, str) and isinstance(model, str):
            cam_to_model[name] = model
    return models, cam_to_model


def _intrinsics_from_model(model: str, shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Build a 3x3 K from the intrinsics model; scales fx/fy/cx/cy if shape differs.
    """
    models, _ = _load_camera_model_map()
    entry = models.get(model) if isinstance(models, dict) else None
    if not isinstance(entry, dict):
        return None
    intr = entry.get("intrinsics") or {}
    try:
        fx = float(intr.get("fx"))
        fy = float(intr.get("fy"))
        cx = float(intr.get("cx"))
        cy = float(intr.get("cy"))
    except Exception:
        return None

    # Optional: scale if a base resolution is specified
    base_w = None
    base_h = None
    res = intr.get("resolution") or entry.get("resolution")
    if isinstance(res, (list, tuple)) and len(res) >= 2:
        try:
            base_w = float(res[0])
            base_h = float(res[1])
        except Exception:
            base_w = base_h = None
    width, height = float(shape[1]), float(shape[0])
    if base_w and base_h and base_w > 0 and base_h > 0 and (base_w != width or base_h != height):
        sx = width / base_w
        sy = height / base_h
        fx *= sx
        cx *= sx
        fy *= sy
        cy *= sy

    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


@lru_cache(maxsize=1)
def _load_intrinsics_bundle() -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, Any]]:
    """
    Load canonical intrinsics models, camera->model map, and camera specs from CalibrationSettings.
    """
    cfg = AppConfig()
    intr_path = Path(cfg.calibration.INTRINSICS_PATH)
    if not intr_path.is_absolute():
        intr_path = ROOT / intr_path
    intrinsics_models = load_intrinsics(str(intr_path))
    model_map = dict(getattr(cfg.calibration, "CAMERA_INTRINSICS_MODEL_MAP", {}) or {})
    camera_specs = dict(getattr(cfg.calibration, "CAMERA_SPECS", {}) or {})
    return intrinsics_models, model_map, camera_specs


@lru_cache(maxsize=1)
def _resolve_extrinsics_path() -> Path:
    cfg = AppConfig()
    calib_path = Path(cfg.calibration.CAMERA_CALIBRATION_PATH)
    if not calib_path.is_absolute():
        calib_path = ROOT / calib_path
    return calib_path


_FALLBACK_WARNED: set[str] = set()


def _intrinsics_for_camera(cam_id: str, shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Build a 3x3 K using canonical intrinsics.json + CalibrationSettings, scaling to match depth shape.
    """
    intr_models, model_map, camera_specs = _load_intrinsics_bundle()
    model_key = model_map.get(cam_id)
    k_tuple = _derive_k_from_intrinsics_model(model_key, intr_models or {})
    if k_tuple is not None:
        fx, fy, cx, cy = k_tuple
        base_res = None
        spec = camera_specs.get(cam_id) if isinstance(camera_specs, dict) else None
        if spec:
            base_res = _resolution_from_spec(spec)  # type: ignore[arg-type]
        if base_res is None and model_key and intr_models:
            model_entry = intr_models.get(model_key)
            if isinstance(model_entry, dict):
                base_res = _resolution_from_spec(model_entry)  # type: ignore[arg-type]
                if base_res is None:
                    intr = model_entry.get("intrinsics")
                    if isinstance(intr, dict):
                        base_res = _resolution_from_spec(intr)  # type: ignore[arg-type]
        width = float(shape[1])
        height = float(shape[0])
        if base_res:
            base_w, base_h = base_res
            if base_w > 0 and base_h > 0 and (base_w != width or base_h != height):
                sx = width / float(base_w)
                sy = height / float(base_h)
                fx *= sx
                cx *= sx
                fy *= sy
                cy *= sy
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    # Fallback to legacy config/cameras.yaml intrinsics when canonical lookup fails.
    models, cam_to_model = _load_camera_model_map()
    legacy_model_key = cam_to_model.get(cam_id, "")
    K = _intrinsics_from_model(legacy_model_key, shape)
    if K is not None and cam_id not in _FALLBACK_WARNED:
        logger.warning(
            "Using legacy intrinsics from config/cameras.yaml for camera '%s' (model=%s)",
            cam_id,
            legacy_model_key or "unknown",
        )
        _FALLBACK_WARNED.add(cam_id)
    return K


def _decode_float32(b64: str, shape: Tuple[int, int]) -> np.ndarray:
    if not b64:
        return np.zeros(shape, dtype=np.float32)
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.float32)
    return arr.reshape(shape)


def _decode_uint8(b64: str, shape: Tuple[int, int]) -> np.ndarray:
    if not b64:
        return np.zeros(shape, dtype=np.uint8)
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return arr.reshape(shape)


def _fit_plane(points: np.ndarray, thresh: float = 0.05, iters: int = 200) -> Optional[Tuple[np.ndarray, float]]:
    """
    RANSAC plane fit. Returns (normal, d) with unit normal.
    """
    if points.shape[0] < 3:
        return None
    best_inliers = 0
    best_model: Optional[Tuple[np.ndarray, float]] = None
    n_pts = points.shape[0]
    rng = np.random.default_rng()
    for _ in range(iters):
        idx = rng.choice(n_pts, size=3, replace=False)
        a, b, c = points[idx]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-6:
            continue
        n = n / norm
        d = -float(np.dot(n, a))
        dist = np.abs(points @ n + d)
        inliers_mask = dist < thresh
        count = int(np.count_nonzero(inliers_mask))
        if count > best_inliers:
            best_inliers = count
            pts_in = points[inliers_mask]
            # Refine with SVD on inliers
            centroid = pts_in.mean(axis=0)
            cov = pts_in - centroid
            _, _, vh = np.linalg.svd(cov, full_matrices=False)
            n_refined = vh[-1]
            n_refined = n_refined / (np.linalg.norm(n_refined) + 1e-12)
            d_refined = -float(np.dot(n_refined, centroid))
            best_model = (n_refined, d_refined)
    return best_model


def _align_rotation(normal_cam: np.ndarray, *, force_camera_up: bool = True) -> np.ndarray:
    """
    Build rotation matrix R_wc that maps camera frame into world with up = +Y and forward = +Z.
    """
    n = normal_cam / (np.linalg.norm(normal_cam) + 1e-12)
    # Flip to make sure normal points upward in camera coords (camera Y points down)
    if force_camera_up and n[1] > 0:
        n = -n
    target_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    dot = float(np.clip(np.dot(n, target_up), -1.0, 1.0))
    if abs(dot - 1.0) < 1e-6:
        R_up = np.eye(3, dtype=np.float64)
    elif abs(dot + 1.0) < 1e-6:
        # 180 around X axis
        R_up = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
    else:
        axis = np.cross(n, target_up)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        angle = math.acos(dot)
        K = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
            dtype=np.float64,
        )
        R_up = np.eye(3, dtype=np.float64) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)

    forward_after = R_up @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    yaw = math.atan2(forward_after[0], forward_after[2])
    cy, sy = math.cos(-yaw), math.sin(-yaw)
    R_yaw = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    return R_yaw @ R_up


def _estimate_extrinsics(
    depth: np.ndarray,
    conf: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    unit_scale: float,
    min_conf: float = 0.3,
    camera_id: Optional[str] = None,
    expected_height_m: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    h, w = depth.shape
    valid = (
        np.isfinite(depth)
        & (depth > 0.1)
        & (depth < 50.0)
        & (conf >= min_conf)
        & (mask > 0)
    )
    if not np.any(valid):
        return None

    grid_u, grid_v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32), indexing="xy")
    z = depth[valid].astype(np.float64)
    x = (grid_u[valid] - K[0, 2]) * z / K[0, 0]
    y = (grid_v[valid] - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z], axis=1)

    # Light downsample for speed
    if pts.shape[0] > 200_000:
        idx = np.random.choice(pts.shape[0], size=200_000, replace=False)
        pts = pts[idx]

    median_depth = float(np.median(z))
    thresh = max(0.02, min(0.25, median_depth * 0.05))
    plane = _fit_plane(pts, thresh=thresh, iters=300)
    if not plane:
        return None
    n_cam, d = plane

    unit_s = float(unit_scale or 1.0)

    # Build candidate extrinsics for both normal directions and score them using
    # physical constraints and floor coverage in world Y.
    candidates: List[Dict[str, Any]] = []

    for sign in (1.0, -1.0):
        n_signed = n_cam * float(sign)
        d_signed = d * float(sign)

        R_wc = _align_rotation(n_signed, force_camera_up=False)

        height_raw = abs(d_signed)
        height_m = height_raw * unit_s
        C_world = np.array([0.0, height_m, 0.0], dtype=np.float64)

        R_cw = R_wc.T
        t_cw = -R_cw @ C_world

        E = np.eye(4, dtype=np.float64)
        E[:3, :3] = R_cw
        E[:3, 3] = t_cw
        e_list = [float(x) for x in E.flatten(order="F")]

        # Basic physical constraints on camera pose
        cam_height = float(C_world[1])
        forward_world = R_wc @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        forward_y = float(forward_world[1])
        forward_z = float(forward_world[2])

        # Approximate pitch from vertical component of forward vector.
        # Negative pitch means looking downward.
        try:
            fy_clamped = max(-1.0, min(1.0, forward_y))
            pitch_rad = math.asin(fy_clamped)
        except Exception:
            pitch_rad = 0.0
        pitch_deg = math.degrees(pitch_rad)

        # Roll is less critical here; approximate it from the world up vector in camera frame.
        # world_up in camera frame: u_cam = R_cw @ [0,1,0]^T
        u_cam = R_cw @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
        try:
            roll_rad = math.atan2(u_cam[0], u_cam[1])
        except Exception:
            roll_rad = 0.0
        roll_deg = math.degrees(roll_rad)

        constraints_ok = True
        reject_reason: Optional[str] = None

        # Height constraints: camera should be above the floor with a sane range.
        min_height = 0.3
        max_height = 5.0
        if cam_height < min_height or cam_height > max_height:
            constraints_ok = False
            reject_reason = "height_out_of_range"

        if expected_height_m is not None:
            if abs(cam_height - float(expected_height_m)) > 2.0:
                constraints_ok = False
                reject_reason = "height_mismatches_prior"

        # Camera must be pitched downward (forward vector has negative Y).
        if forward_y >= -0.01:
            constraints_ok = False
            reject_reason = "camera_not_pitched_down"

        # Enforce alignment with MA world +Z (camera should look “into the room”).
        if forward_z <= 0.0:
            constraints_ok = False
            reject_reason = "camera_forward_not_pos_z"

        # Keep pitch/roll within a sane envelope.
        if not (-80.0 <= pitch_deg <= -0.1):
            constraints_ok = False
            reject_reason = "pitch_out_of_range"
        roll_abs = abs(roll_deg)
        roll_to_180 = abs(roll_abs - 180.0)
        # Reject only if roll is far from both 0° and 180° (i.e., side-on).
        if min(roll_abs, roll_to_180) > 45.0:
            constraints_ok = False
            reject_reason = "roll_out_of_range"

        # Score floor coverage in world Y. Use the sampled points used for plane fitting.
        pts_world = (R_wc @ pts.T).T + C_world[None, :]
        y_world = pts_world[:, 1]
        if y_world.size == 0:
            floor_score = 0.0
            floor_ratio = 0.0
            floor_std = float("inf")
        else:
            # Robust floor height estimate: lower percentile of Y.
            h0 = float(np.percentile(y_world, 5.0))
            spread = float(np.percentile(y_world, 95.0) - h0)
            eps = max(0.05, 0.1 * max(spread, 0.0))
            floor_mask = y_world <= (h0 + eps)
            floor_count = int(np.count_nonzero(floor_mask))
            total_count = int(y_world.size)
            if floor_count > 0:
                floor_heights = y_world[floor_mask]
                floor_ratio = float(floor_count) / float(total_count)
                floor_std = float(np.std(floor_heights))
            else:
                floor_ratio = 0.0
                floor_std = float("inf")
            # Higher ratio and lower spread are better.
            floor_score = floor_ratio / (1.0 + max(floor_std, 0.0))

        # Enforce minimum floor support quality.
        if floor_ratio < 0.1 or not np.isfinite(floor_std) or floor_std > 0.3:
            constraints_ok = False
            if reject_reason is None:
                reject_reason = "floor_support_poor"

        candidates.append(
            {
                "E": e_list,
                "height_m": height_m,
                "plane_normal_cam": n_signed.tolist(),
                "inlier_thresh": thresh,
                "points_used": int(pts.shape[0]),
                "pitch_deg": pitch_deg,
                "roll_deg": roll_deg,
                "forward_y": forward_y,
                "forward_z": forward_z,
                "floor_ratio": floor_ratio,
                "floor_std": floor_std,
                "floor_score": floor_score,
                "constraints_ok": constraints_ok,
                "reject_reason": reject_reason,
                "cameraId": camera_id,
                "normal_sign": sign,
            }
        )

    if not candidates:
        return None

    # Prefer candidates that satisfy constraints; break ties by floor_score.
    valid = [c for c in candidates if c.get("constraints_ok")]
    pool = valid if valid else candidates
    best = max(pool, key=lambda c: c.get("floor_score", 0.0))

    # If no candidate passed constraints, treat as failure but surface diagnostics.
    if not best.get("constraints_ok"):
        best["ok"] = False
        return best

    best = dict(best)
    best["ok"] = True
    return best


def auto_calibrate_from_latest_depth(
    camera_ids: Optional[Iterable[str]] = None,
    *,
    persist: bool = False,
    unit_scale: Optional[float] = None,
) -> Dict[str, Any]:
    cfg = load_service_config()
    # Depth values are already in meters; stick to unit_scale=1 unless explicitly overridden.
    unit_s = float(unit_scale) if unit_scale is not None else 1.0

    cam_list = list(camera_ids) if camera_ids else list(DEFAULT_CAMERAS)
    height_priors = _load_camera_height_priors()

    depth_source = MapAnythingDepthSource(cfg)

    calib_path = _resolve_extrinsics_path()
    calib_data = _read_json(calib_path)
    calib_data.setdefault("cameras", {})

    results: List[Dict[str, Any]] = []
    for cam in cam_list:
        payload = depth_source.load_latest_depth(cam)
        if not payload:
            results.append({"cameraId": cam, "ok": False, "error": "no_depth"})
            continue
        shape = payload.get("shape") or []
        if not (isinstance(shape, list) and len(shape) == 2):
            results.append({"cameraId": cam, "ok": False, "error": "bad_shape"})
            continue
        h, w = int(shape[0]), int(shape[1])
        K = _intrinsics_for_camera(cam, (h, w))
        if K is None:
            results.append({"cameraId": cam, "ok": False, "error": "no_intrinsics"})
            continue
        depth = _decode_float32(payload["depth_b64"], (h, w))
        conf = _decode_float32(payload.get("conf_b64", ""), (h, w))
        mask = _decode_uint8(payload.get("mask_b64", ""), (h, w))
        expected_h = height_priors.get(cam)
        est = _estimate_extrinsics(depth, conf, mask, K, unit_s, camera_id=cam, expected_height_m=expected_h)
        if not est:
            results.append({"cameraId": cam, "ok": False, "error": "fit_failed"})
            continue
        if not est.get("ok", False):
            err = est.get("reject_reason") or "constraints_failed"
            debug = {
                "pitch_deg": est.get("pitch_deg"),
                "roll_deg": est.get("roll_deg"),
                "height_m": est.get("height_m"),
                "floor_ratio": est.get("floor_ratio"),
                "floor_std": est.get("floor_std"),
                "normal_sign": est.get("normal_sign"),
            }
            results.append(
                {
                    "cameraId": cam,
                    "ok": False,
                    "error": err,
                    "debug": debug,
                }
            )
            continue
        est["cameraId"] = cam
        results.append(est)
        if persist:
            ok = save_extrinsics(str(calib_path), cam, est["E"])
            est["persisted"] = bool(ok)

    summary = {"ok": any(r.get("ok") for r in results), "results": results}
    if persist:
        summary["calibration_path"] = str(calib_path)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Auto-calibrate poses from latest MapAnything depth.")
    parser.add_argument("--camera", action="append", help="Camera ID(s) to calibrate; defaults to all known")
    parser.add_argument("--no-write", dest="persist", action="store_false", help="Do not persist to camera_calibration.json")
    parser.set_defaults(persist=True)
    args = parser.parse_args()

    res = auto_calibrate_from_latest_depth(args.camera, persist=args.persist)
    print(json.dumps(res, indent=2))
