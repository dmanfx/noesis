"""
Auto-calibrate camera extrinsics from the latest MapAnything depth cache.

This reads the newest cached depth snapshot per camera, fits a ground plane,
derives a camera pose (R, t) that points the optical axis toward +Z with
ground at Y=0, and optionally persists the resulting world→camera matrix
into config/camera_calibration.json.
"""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from calibration_bundle import save_extrinsics
from geometry.depth_source import MapAnythingDepthSource
from mapanything_config import load_service_config

# Defaults match the FE camera keys
DEFAULT_CAMERAS = ["living-room", "kitchen", "family-room"]


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _load_camera_model_map() -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Returns (intrinsics_models, camera->model) from config/cameras.yaml
    """
    cfg_path = Path("config/cameras.yaml")
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


def _align_rotation(normal_cam: np.ndarray) -> np.ndarray:
    """
    Build rotation matrix R_wc that maps camera frame into world with up = +Y and forward = +Z.
    """
    n = normal_cam / (np.linalg.norm(normal_cam) + 1e-12)
    # Flip to make sure normal points upward in camera coords (camera Y points down)
    if n[1] > 0:
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

    R_wc = _align_rotation(n_cam)
    height_raw = abs(d)
    height_m = height_raw * float(unit_scale or 1.0)

    C_world = np.array([0.0, height_m, 0.0], dtype=np.float64)
    R_cw = R_wc.T
    t_cw = -R_cw @ C_world
    E = np.eye(4, dtype=np.float64)
    E[:3, :3] = R_cw
    E[:3, 3] = t_cw
    e_list = [float(x) for x in E.flatten(order="F")]
    return {
        "E": e_list,
        "height_m": height_m,
        "plane_normal_cam": n_cam.tolist(),
        "inlier_thresh": thresh,
        "points_used": int(pts.shape[0]),
    }


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
    models, cam_to_model = _load_camera_model_map()

    depth_source = MapAnythingDepthSource(cfg)

    calib_path = Path("config/camera_calibration.json")
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
        model_key = cam_to_model.get(cam, "")
        K = _intrinsics_from_model(model_key, (h, w))
        if K is None:
            results.append({"cameraId": cam, "ok": False, "error": f"no_intrinsics_for_model:{model_key}"})
            continue
        depth = _decode_float32(payload["depth_b64"], (h, w))
        conf = _decode_float32(payload.get("conf_b64", ""), (h, w))
        mask = _decode_uint8(payload.get("mask_b64", ""), (h, w))
        est = _estimate_extrinsics(depth, conf, mask, K, unit_s)
        if not est:
            results.append({"cameraId": cam, "ok": False, "error": "fit_failed"})
            continue
        est["cameraId"] = cam
        est["ok"] = True
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
