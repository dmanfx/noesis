"""Tilt-only calibration preview using the latest MapAnything depth snapshot.

This module fits a ground plane from depth, then adjusts camera pitch/roll
while preserving yaw and camera center. It writes to a separate preview
calibration file so production calibration can remain untouched.
"""
from __future__ import annotations

import base64
import copy
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from noesis.calibration.pose_v1 import E_col_major_to_pose_v1
import yaml

from calibration_bundle import (
    load_intrinsics,
    _derive_k_from_intrinsics_model,  # type: ignore
    _resolution_from_spec,  # type: ignore
)
from config import AppConfig
from geometry.depth_source import MapAnythingDepthSource
from mapanything_config import load_service_config

DEFAULT_CAMERAS = ["living-room", "kitchen", "family-room"]
logger = logging.getLogger(__name__)


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _load_camera_model_map(cameras_path: Optional[Path] = None) -> Tuple[Dict[str, Any], Dict[str, str]]:
    cfg_path = Path(cameras_path) if cameras_path is not None else Path("config/cameras.yaml")
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


def _intrinsics_from_model(
    model: str,
    shape: Tuple[int, int],
    cameras_path: Optional[Path] = None,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "model": model or "",
        "base_resolution": None,
        "scale_x": 1.0,
        "scale_y": 1.0,
    }
    models, _ = _load_camera_model_map(cameras_path)
    entry = models.get(model) if isinstance(models, dict) else None
    if not isinstance(entry, dict):
        return None, meta
    intr = entry.get("intrinsics") or {}
    try:
        fx = float(intr.get("fx"))
        fy = float(intr.get("fy"))
        cx = float(intr.get("cx"))
        cy = float(intr.get("cy"))
    except Exception:
        return None, meta

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
    if base_w and base_h and base_w > 0 and base_h > 0:
        meta["base_resolution"] = [int(base_w), int(base_h)]
        if base_w != width or base_h != height:
            sx = width / base_w
            sy = height / base_h
            fx *= sx
            cx *= sx
            fy *= sy
            cy *= sy
            meta["scale_x"] = float(sx)
            meta["scale_y"] = float(sy)

    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64), meta


_FALLBACK_WARNED: set[str] = set()


def _intrinsics_for_camera(
    cam_id: str,
    shape: Tuple[int, int],
    cameras_path: Optional[Path] = None,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    if cameras_path is not None:
        _, cam_to_model = _load_camera_model_map(cameras_path)
        model_key = cam_to_model.get(cam_id, "")
        K, meta = _intrinsics_from_model(model_key, shape, cameras_path)
        if K is not None:
            meta["source"] = "cameras_config"
            return K, meta

    intr_models, model_map, camera_specs = _load_intrinsics_bundle()
    model_key = model_map.get(cam_id)
    k_tuple = _derive_k_from_intrinsics_model(model_key, intr_models or {})
    if k_tuple is not None:
        fx, fy, cx, cy = k_tuple
        meta: Dict[str, Any] = {
            "source": "intrinsics_json",
            "model": model_key or "",
            "base_resolution": None,
            "scale_x": 1.0,
            "scale_y": 1.0,
        }
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
                meta["scale_x"] = float(sx)
                meta["scale_y"] = float(sy)
            meta["base_resolution"] = [int(base_w), int(base_h)]
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64), meta

    if cameras_path is None:
        models, cam_to_model = _load_camera_model_map(cameras_path)
        fallback_model_key = cam_to_model.get(cam_id, "")
        K, meta = _intrinsics_from_model(fallback_model_key, shape, cameras_path)
    else:
        K = None
        meta = {}
    if K is not None and cam_id not in _FALLBACK_WARNED:
        logger.warning(
            "Using fallback intrinsics from config/cameras.yaml for camera '%s' (model=%s)",
            cam_id,
            fallback_model_key or "unknown",
        )
        _FALLBACK_WARNED.add(cam_id)
    if K is not None:
        meta["source"] = "cameras_yaml_fallback"
        return K, meta
    return None, {}


def _load_intrinsics_bundle() -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, Any]]:
    cfg = AppConfig()
    intr_path = Path(cfg.calibration.INTRINSICS_PATH)
    if not intr_path.is_absolute():
        intr_path = Path(__file__).resolve().parents[2] / intr_path
    intrinsics_models = load_intrinsics(str(intr_path))
    model_map = dict(getattr(cfg.calibration, "CAMERA_INTRINSICS_MODEL_MAP", {}) or {})
    camera_specs = dict(getattr(cfg.calibration, "CAMERA_SPECS", {}) or {})
    return intrinsics_models, model_map, camera_specs


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


def _normalize(vec: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(vec))
    if not math.isfinite(norm) or norm < 1e-9:
        return None
    return vec / norm


def _rotation_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis_n = _normalize(axis)
    if axis_n is None:
        return np.eye(3, dtype=np.float64)
    ax = axis_n.astype(np.float64)
    K = np.array(
        [[0.0, -ax[2], ax[1]], [ax[2], 0.0, -ax[0]], [-ax[1], ax[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def _align_vector(src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
    src_n = _normalize(src)
    dst_n = _normalize(dst)
    if src_n is None or dst_n is None:
        return None
    dot = float(np.clip(np.dot(src_n, dst_n), -1.0, 1.0))
    if dot > 1.0 - 1e-8:
        return np.eye(3, dtype=np.float64)
    if dot < -1.0 + 1e-8:
        # 180-degree flip; pick any orthogonal axis.
        probe = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(src_n, probe))) > 0.9:
            probe = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        axis = np.cross(src_n, probe)
        return _rotation_from_axis_angle(axis, math.pi)
    axis = np.cross(src_n, dst_n)
    angle = math.acos(dot)
    return _rotation_from_axis_angle(axis, angle)


def _rotation_yaw(yaw_rad: float) -> np.ndarray:
    cy = math.cos(yaw_rad)
    sy = math.sin(yaw_rad)
    return np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)


def _yaw_pitch_roll(R_wc: np.ndarray) -> Tuple[float, float, float]:
    forward = R_wc @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    yaw = math.degrees(math.atan2(float(forward[0]), float(forward[2])))
    fy = max(-1.0, min(1.0, float(forward[1])))
    pitch = math.degrees(math.asin(fy))
    R_cw = R_wc.T
    up_cam = R_cw @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
    roll = math.degrees(math.atan2(float(up_cam[0]), float(up_cam[1])))
    return yaw, pitch, roll


def _camera_pose_from_E(E_col_major: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    E = np.array(E_col_major, dtype=np.float64).reshape((4, 4), order="F")
    R_cw = E[:3, :3]
    t_cw = E[:3, 3]
    R_wc = R_cw.T
    C_world = -R_wc @ t_cw
    return R_wc, C_world


def tilt_correct_extrinsics(
    E_col_major: Sequence[float],
    plane_normal_cam: Sequence[float],
    *,
    preserve_yaw: bool = True,
    force_world_up: bool = True,
    world_up: Optional[Sequence[float]] = None,
) -> Tuple[List[float], Dict[str, Any]]:
    R_wc, C_world = _camera_pose_from_E(E_col_major)
    n_cam = np.array(plane_normal_cam, dtype=np.float64)
    n_cam = _normalize(n_cam)
    if n_cam is None:
        raise ValueError("invalid_plane_normal")

    up = np.array(world_up or [0.0, 1.0, 0.0], dtype=np.float64)
    up = _normalize(up)
    if up is None:
        raise ValueError("invalid_world_up")

    yaw_before, pitch_before, roll_before = _yaw_pitch_roll(R_wc)
    yaw_rad = math.radians(yaw_before)
    n_world = R_wc @ n_cam
    normal_flipped = False
    if force_world_up and float(np.dot(n_world, up)) < 0.0:
        n_cam = -n_cam
        n_world = -n_world
        normal_flipped = True

    tilt_angle = math.degrees(math.acos(float(np.clip(np.dot(_normalize(n_world), up), -1.0, 1.0))))

    if preserve_yaw:
        R_yaw = _rotation_yaw(yaw_rad)
        n_world_yawfree = R_yaw.T @ n_world
        R_tilt = _align_vector(n_world_yawfree, up)
        if R_tilt is None:
            raise ValueError("tilt_alignment_failed")
        R_wc_new = R_yaw @ (R_tilt @ (R_yaw.T @ R_wc))
        yaw_after_tmp, _, _ = _yaw_pitch_roll(R_wc_new)
        yaw_delta = math.radians(float(yaw_before) - float(yaw_after_tmp))
        if abs(yaw_delta) > 1e-9:
            R_wc_new = _rotation_yaw(yaw_delta) @ R_wc_new
    else:
        R_tilt = _align_vector(n_world, up)
        if R_tilt is None:
            raise ValueError("tilt_alignment_failed")
        R_wc_new = R_tilt @ R_wc

    R_cw_new = R_wc_new.T
    t_cw_new = -R_cw_new @ C_world
    E_new = np.eye(4, dtype=np.float64)
    E_new[:3, :3] = R_cw_new
    E_new[:3, 3] = t_cw_new
    E_list = [float(x) for x in E_new.flatten(order="F")]

    yaw_after, pitch_after, roll_after = _yaw_pitch_roll(R_wc_new)
    n_world_after = R_wc_new @ n_cam
    _, C_world_after = _camera_pose_from_E(E_list)

    info = {
        "yaw_deg_before": float(yaw_before),
        "pitch_deg_before": float(pitch_before),
        "roll_deg_before": float(roll_before),
        "yaw_deg_after": float(yaw_after),
        "pitch_deg_after": float(pitch_after),
        "roll_deg_after": float(roll_after),
        "tilt_angle_deg": float(tilt_angle),
        "normal_cam": [float(x) for x in n_cam],
        "normal_world_before": [float(x) for x in n_world],
        "normal_world_after": [float(x) for x in n_world_after],
        "normal_flipped": bool(normal_flipped),
        "force_world_up": bool(force_world_up),
        "camera_center_before": [float(x) for x in C_world],
        "camera_center_after": [float(x) for x in C_world_after],
        "preserve_yaw": bool(preserve_yaw),
    }
    return E_list, info


def _score_tilt_candidate(info: Dict[str, Any]) -> float:
    pitch = float(info.get("pitch_deg_after", 0.0) or 0.0)
    roll = float(info.get("roll_deg_after", 0.0) or 0.0)
    tilt = float(info.get("tilt_angle_deg", 0.0) or 0.0)
    roll_abs = abs(roll)
    roll_metric = min(roll_abs, abs(roll_abs - 180.0))

    penalty = 0.0
    if pitch > -0.1:
        penalty += 1000.0 + abs(pitch)
    penalty += roll_metric * 0.25
    penalty += abs(pitch) * 0.05
    penalty += abs(tilt) * 0.01
    return penalty


def apply_preview_updates(
    base_data: Dict[str, Any],
    updates: Dict[str, Sequence[float]],
) -> Dict[str, Any]:
    data = copy.deepcopy(base_data) if base_data else {}
    cameras = data.get("cameras")
    if not isinstance(cameras, dict):
        cameras = {}
        data["cameras"] = cameras
    for cam_id, E in updates.items():
        entry = cameras.get(cam_id)
        if not isinstance(entry, dict):
            entry = {}
        else:
            entry = dict(entry)
        entry["E"] = [float(x) for x in E]
        existing_source = (entry.get("pose") or {}).get("source") if isinstance(entry.get("pose"), dict) else None
        pose = E_col_major_to_pose_v1(entry["E"], source=existing_source or "derived_from_E")
        if pose is not None:
            entry["pose"] = pose
        cameras[cam_id] = entry
    return data


def _fit_plane(
    points: np.ndarray,
    thresh: float = 0.05,
    iters: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> Optional[Tuple[np.ndarray, float]]:
    if points.shape[0] < 3:
        return None
    best_inliers = 0
    best_model: Optional[Tuple[np.ndarray, float]] = None
    n_pts = points.shape[0]
    rng = rng or np.random.default_rng()
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
            centroid = pts_in.mean(axis=0)
            cov = pts_in - centroid
            _, _, vh = np.linalg.svd(cov, full_matrices=False)
            n_refined = vh[-1]
            n_refined = n_refined / (np.linalg.norm(n_refined) + 1e-12)
            d_refined = -float(np.dot(n_refined, centroid))
            best_model = (n_refined, d_refined)
    return best_model


def _estimate_plane_normal(
    depth: np.ndarray,
    conf: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    min_conf: float,
    max_points: int,
    rng: np.random.Generator,
    *,
    flip_image_y: bool = False,
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
    if flip_image_y:
        y = -y
    pts = np.stack([x, y, z], axis=1)
    if pts.shape[0] > max_points:
        idx = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
    median_depth = float(np.median(z))
    thresh = max(0.02, min(0.25, median_depth * 0.05))
    plane = _fit_plane(pts, thresh=thresh, iters=300, rng=rng)
    if not plane:
        return None
    n_cam, d = plane
    dist = np.abs(pts @ n_cam + d)
    inlier_mask = dist < thresh
    inlier_count = int(np.count_nonzero(inlier_mask))
    inlier_ratio = float(inlier_count) / float(max(1, pts.shape[0]))
    valid_ratio = float(np.count_nonzero(valid)) / float(valid.size)
    return {
        "normal_cam": [float(x) for x in n_cam],
        "d": float(d),
        "inlier_thresh": float(thresh),
        "points_used": int(pts.shape[0]),
        "inlier_ratio": float(inlier_ratio),
        "valid_ratio": float(valid_ratio),
        "median_depth": float(median_depth),
        "flip_image_y": bool(flip_image_y),
    }


def _resolve_camera_ids(
    camera_ids: Optional[Iterable[str]],
    calib_data: Dict[str, Any],
    cameras_path: Optional[Path] = None,
) -> List[str]:
    if camera_ids:
        return list(camera_ids)
    cams = calib_data.get("cameras")
    if isinstance(cams, dict) and cams:
        return list(cams.keys())
    cfg_path = Path(cameras_path) if cameras_path is not None else Path("config/cameras.yaml")
    if cfg_path.exists():
        data = yaml.safe_load(cfg_path.read_text()) or {}
        cam_entries = data.get("cameras") or {}
        names = []
        if isinstance(cam_entries, dict):
            for entry in cam_entries.values():
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name")
                if isinstance(name, str):
                    names.append(name)
        if names:
            return names
    return list(DEFAULT_CAMERAS)


def tilt_preview_from_latest_depth(
    camera_ids: Optional[Iterable[str]] = None,
    *,
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    cameras_path: Optional[Path] = None,
    persist: bool = True,
    min_conf: float = 0.3,
    max_points: int = 200_000,
    seed: Optional[int] = None,
    preserve_yaw: bool = True,
    normal_signs: Optional[Dict[str, float]] = None,
    flip_image_y: bool = False,
) -> Dict[str, Any]:
    cfg = load_service_config()
    depth_source = MapAnythingDepthSource(cfg)

    base_path = Path(input_path or "config/camera_calibration.json")
    calib_data = _read_json(base_path)
    calib_data.setdefault("cameras", {})
    cam_list = _resolve_camera_ids(camera_ids, calib_data, cameras_path)
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    updates: Dict[str, List[float]] = {}
    results: List[Dict[str, Any]] = []
    for cam in cam_list:
        entry = (calib_data.get("cameras") or {}).get(cam)
        E = entry.get("E") if isinstance(entry, dict) else None
        if not (isinstance(E, list) and len(E) == 16):
            results.append({"cameraId": cam, "ok": False, "error": "missing_extrinsics"})
            continue
        payload = depth_source.load_latest_depth(cam)
        if not payload:
            results.append({"cameraId": cam, "ok": False, "error": "no_depth"})
            continue
        shape = payload.get("shape") or []
        if not (isinstance(shape, list) and len(shape) == 2):
            results.append({"cameraId": cam, "ok": False, "error": "bad_shape"})
            continue
        h, w = int(shape[0]), int(shape[1])
        K, K_meta = _intrinsics_for_camera(cam, (h, w), cameras_path)
        if K is None:
            results.append({"cameraId": cam, "ok": False, "error": "no_intrinsics"})
            continue
        depth_ts_us = None
        depth_age_s = None
        try:
            depth_ts_us = int(payload.get("ts", 0) or 0)
            if depth_ts_us > 0:
                depth_age_s = max(0.0, time.time() - (depth_ts_us / 1_000_000.0))
        except Exception:
            depth_ts_us = None
            depth_age_s = None
        depth = _decode_float32(payload.get("depth_b64", ""), (h, w))
        conf = _decode_float32(payload.get("conf_b64", ""), (h, w))
        mask = _decode_uint8(payload.get("mask_b64", ""), (h, w))
        plane = _estimate_plane_normal(
            depth,
            conf,
            mask,
            K,
            min_conf,
            max_points,
            rng,
            flip_image_y=flip_image_y,
        )
        if not plane:
            results.append({"cameraId": cam, "ok": False, "error": "plane_fit_failed"})
            continue
        n_cam = np.array(plane["normal_cam"], dtype=np.float64)
        candidates: List[Tuple[float, List[float], Dict[str, Any]]] = []
        sign_override = None
        if normal_signs and cam in normal_signs:
            sign_value = float(normal_signs[cam])
            if sign_value == 0.0 or not math.isfinite(sign_value):
                results.append({"cameraId": cam, "ok": False, "error": "invalid_normal_sign"})
                continue
            sign_override = 1.0 if sign_value > 0.0 else -1.0

        signs = [sign_override] if sign_override is not None else [1.0, -1.0]
        for sign in signs:
            try:
                E_new, info = tilt_correct_extrinsics(
                    E,
                    (n_cam * sign).tolist(),
                    preserve_yaw=preserve_yaw,
                    force_world_up=False,
                )
            except Exception:
                continue
            info["candidate_sign"] = float(sign)
            info["candidate_score"] = float(_score_tilt_candidate(info))
            candidates.append((info["candidate_score"], E_new, info))

        if not candidates:
            results.append({"cameraId": cam, "ok": False, "error": "tilt_failed"})
            continue

        candidates.sort(key=lambda item: item[0])
        _, E_new, info = candidates[0]

        updates[cam] = E_new
        intrinsics_meta = dict(K_meta or {})
        intrinsics_meta.update(
            {
                "fx": float(K[0, 0]),
                "fy": float(K[1, 1]),
                "cx": float(K[0, 2]),
                "cy": float(K[1, 2]),
            }
        )
        results.append(
            {
                "cameraId": cam,
                "ok": True,
                "plane_fit": plane,
                "tilt": info,
                "intrinsics": intrinsics_meta,
                "depth_snapshot": {
                    "ts_us": depth_ts_us,
                    "age_s": depth_age_s,
                    "shape": [h, w],
                },
            }
        )

    summary = {
        "ok": any(r.get("ok") for r in results),
        "results": results,
        "updated": sorted(updates.keys()),
    }
    if persist and updates:
        out_path = Path(output_path or "config/camera_calibration_preview.json")
        preview_data = apply_preview_updates(calib_data, updates)
        preview_data["preview_meta"] = {
            "method": "depth_tilt_preview",
            "generated_at": _now_stamp(),
            "source_path": str(base_path),
            "preserve_yaw": bool(preserve_yaw),
            "min_conf": float(min_conf),
            "max_points": int(max_points),
            "flip_image_y": bool(flip_image_y),
            "cameras": sorted(updates.keys()),
            "normal_sign_overrides": {k: float(v) for k, v in (normal_signs or {}).items()},
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(preview_data, indent=2))
        summary["preview_path"] = str(out_path)
    return summary
