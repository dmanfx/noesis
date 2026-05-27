"""PoseV1 conversion helpers for DS8 calibration.

PoseV1 can appear in either canonical backend world meters or legacy Menon
scene coordinates. The ray math in DS8 always expects backend world meters
with the OpenCV-style image-camera basis, so scene-space poses must be mapped
through alignment and unit conversion before extrinsics are synthesized.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np


POSE_V1_LOCAL_TO_OPENCV = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

POSE_V1_FRAME_BACKEND_WORLD_M = "backend_world_m"
POSE_V1_FRAME_MENON_SCENE = "menon_scene"
_IDENTITY_4 = np.eye(4, dtype=np.float64)


def _normalize_align_context(align_data: Optional[Dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, float]:
    align = align_data if isinstance(align_data, dict) else {}
    matrix_raw = align.get("matrix")
    if isinstance(matrix_raw, list) and len(matrix_raw) == 16:
        try:
            scene_from_world_obj = np.array(matrix_raw, dtype=np.float64).reshape((4, 4))
        except Exception:
            scene_from_world_obj = _IDENTITY_4.copy()
    else:
        scene_from_world_obj = _IDENTITY_4.copy()
    try:
        world_obj_from_scene = np.linalg.inv(scene_from_world_obj)
    except Exception:
        scene_from_world_obj = _IDENTITY_4.copy()
        world_obj_from_scene = _IDENTITY_4.copy()
    units = align.get("units") if isinstance(align, dict) else {}
    try:
        s_obj_to_m = float((units or {}).get("s_obj_to_m", 1.0) or 1.0)
    except Exception:
        s_obj_to_m = 1.0
    if not math.isfinite(s_obj_to_m) or s_obj_to_m <= 1e-9:
        s_obj_to_m = 1.0
    return scene_from_world_obj, world_obj_from_scene, float(s_obj_to_m)


def _transform_point(mat4: np.ndarray, point_xyz: List[float]) -> np.ndarray:
    vec = np.array([float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2]), 1.0], dtype=np.float64)
    out = mat4 @ vec
    return out[:3].copy()


def _scene_pose_to_backend_world_pose(pose: Dict[str, Any], align_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    scene_from_world_obj, world_obj_from_scene, s_obj_to_m = _normalize_align_context(align_data)
    yaw_deg, pitch_deg, roll_deg = pose["yaw_pitch_roll_deg"]
    rotation_scene = _rotation_yxz_from_ypr_deg(yaw_deg, pitch_deg, roll_deg)
    rotation_world = world_obj_from_scene[:3, :3] @ rotation_scene
    position_world_m = _transform_point(world_obj_from_scene, pose["position"]) * float(s_obj_to_m)
    ypr_world = _ypr_deg_from_rotation_yxz(rotation_world)
    out: Dict[str, Any] = {
        "position": [float(position_world_m[0]), float(position_world_m[1]), float(position_world_m[2])],
        "yaw_pitch_roll_deg": [float(ypr_world[0]), float(ypr_world[1]), float(ypr_world[2])],
        "rotation_order": "YXZ",
        "frame": POSE_V1_FRAME_BACKEND_WORLD_M,
    }
    source = pose.get("source")
    if isinstance(source, str) and source.strip():
        out["source"] = source.strip()
    return out


def _backend_world_pose_to_scene_pose(
    pose: Dict[str, Any],
    align_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    scene_from_world_obj, _world_obj_from_scene, s_obj_to_m = _normalize_align_context(align_data)
    yaw_deg, pitch_deg, roll_deg = pose["yaw_pitch_roll_deg"]
    rotation_world = _rotation_yxz_from_ypr_deg(yaw_deg, pitch_deg, roll_deg)
    rotation_scene = scene_from_world_obj[:3, :3] @ rotation_world
    position_world_obj = np.asarray(pose["position"], dtype=np.float64) / float(s_obj_to_m)
    position_scene = _transform_point(scene_from_world_obj, [float(position_world_obj[0]), float(position_world_obj[1]), float(position_world_obj[2])])
    ypr_scene = _ypr_deg_from_rotation_yxz(rotation_scene)
    out: Dict[str, Any] = {
        "position": [float(position_scene[0]), float(position_scene[1]), float(position_scene[2])],
        "yaw_pitch_roll_deg": [float(ypr_scene[0]), float(ypr_scene[1]), float(ypr_scene[2])],
        "rotation_order": "YXZ",
        "frame": POSE_V1_FRAME_MENON_SCENE,
    }
    source = pose.get("source")
    if isinstance(source, str) and source.strip():
        out["source"] = source.strip()
    return out


def normalize_pose_v1(raw_pose: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw_pose, dict):
        return None
    position = raw_pose.get("position")
    ypr = raw_pose.get("yaw_pitch_roll_deg")
    rotation_order = str(raw_pose.get("rotation_order") or "").strip().upper()
    frame = str(raw_pose.get("frame") or "").strip()
    if not (isinstance(position, list) and len(position) == 3):
        return None
    if not (isinstance(ypr, list) and len(ypr) == 3):
        return None
    try:
        position_f = [float(position[0]), float(position[1]), float(position[2])]
        ypr_f = [float(ypr[0]), float(ypr[1]), float(ypr[2])]
    except Exception:
        return None
    if not all(math.isfinite(v) for v in (position_f + ypr_f)):
        return None
    if rotation_order != "YXZ":
        return None
    if frame not in (POSE_V1_FRAME_BACKEND_WORLD_M, POSE_V1_FRAME_MENON_SCENE):
        return None
    out: Dict[str, Any] = {
        "position": position_f,
        "yaw_pitch_roll_deg": ypr_f,
        "rotation_order": "YXZ",
        "frame": frame,
    }
    source = raw_pose.get("source")
    if isinstance(source, str) and source.strip():
        out["source"] = source.strip()
    return out


def _rotation_yxz_from_ypr_deg(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    roll = math.radians(float(roll_deg))

    cy, sy = math.cos(yaw), math.sin(yaw)
    cx, sx = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(roll), math.sin(roll)

    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return ry @ rx @ rz


def _ypr_deg_from_rotation_yxz(rotation: np.ndarray) -> List[float]:
    pitch = math.asin(max(-1.0, min(1.0, -float(rotation[1, 2]))))
    cos_pitch = math.cos(pitch)
    if abs(cos_pitch) > 1e-8:
        yaw = math.atan2(float(rotation[0, 2]), float(rotation[2, 2]))
        roll = math.atan2(float(rotation[1, 0]), float(rotation[1, 1]))
    else:
        yaw = math.atan2(-float(rotation[2, 0]), float(rotation[0, 0]))
        roll = 0.0
    return [math.degrees(yaw), math.degrees(pitch), math.degrees(roll)]


def pose_v1_to_Twc(pose: Dict[str, Any], *, align_data: Optional[Dict[str, Any]] = None) -> Optional[np.ndarray]:
    norm = normalize_pose_v1(pose)
    if not norm:
        return None
    try:
        if str(norm.get("frame")) == POSE_V1_FRAME_MENON_SCENE:
            norm = _scene_pose_to_backend_world_pose(norm, align_data)
        yaw_deg, pitch_deg, roll_deg = norm["yaw_pitch_roll_deg"]
        rotation_pose = _rotation_yxz_from_ypr_deg(yaw_deg, pitch_deg, roll_deg)
        rotation_wc = rotation_pose @ POSE_V1_LOCAL_TO_OPENCV
        twc = np.eye(4, dtype=np.float64)
        twc[:3, :3] = rotation_wc
        twc[:3, 3] = np.array(norm["position"], dtype=np.float64)
        return twc
    except Exception:
        return None


def pose_to_E_col_major(pose: Dict[str, Any], *, align_data: Optional[Dict[str, Any]] = None) -> Optional[List[float]]:
    twc = pose_v1_to_Twc(pose, align_data=align_data)
    if twc is None:
        return None
    try:
        e_mat = np.linalg.inv(twc)
        return [float(x) for x in e_mat.flatten(order="F")]
    except Exception:
        return None


def E_col_major_to_pose_v1(
    E_col_major: List[float],
    *,
    source: Optional[str] = None,
    frame: str = POSE_V1_FRAME_BACKEND_WORLD_M,
    align_data: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if not (isinstance(E_col_major, list) and len(E_col_major) == 16):
        return None
    try:
        e_mat = np.array(E_col_major, dtype=np.float64).reshape((4, 4), order="F")
        twc = np.linalg.inv(e_mat)
        rotation_wc = twc[:3, :3]
        position = twc[:3, 3]
        rotation_pose = rotation_wc @ POSE_V1_LOCAL_TO_OPENCV.T
        ypr = _ypr_deg_from_rotation_yxz(rotation_pose)
        pose_world: Dict[str, Any] = {
            "position": [float(position[0]), float(position[1]), float(position[2])],
            "yaw_pitch_roll_deg": [float(ypr[0]), float(ypr[1]), float(ypr[2])],
            "rotation_order": "YXZ",
            "frame": POSE_V1_FRAME_BACKEND_WORLD_M,
        }
        if isinstance(source, str) and source.strip():
            pose_world["source"] = source.strip()
        target_frame = str(frame or POSE_V1_FRAME_BACKEND_WORLD_M).strip()
        if target_frame == POSE_V1_FRAME_MENON_SCENE:
            return _backend_world_pose_to_scene_pose(pose_world, align_data)
        if target_frame != POSE_V1_FRAME_BACKEND_WORLD_M:
            return None
        return pose_world
    except Exception:
        return None
