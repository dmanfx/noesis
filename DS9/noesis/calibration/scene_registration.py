from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


CAMERA_ANCHOR_STATE_CONTRACT = "noesis.menon.camera_anchor_state"
CAMERA_ANCHOR_STATE_CONTRACT_VERSION = 1


def _as_point3(values: Sequence[Any], *, name: str) -> np.ndarray:
    if not isinstance(values, Sequence) or len(values) != 3:
        raise ValueError(f"{name} must be a 3-vector")
    point = np.asarray([float(values[0]), float(values[1]), float(values[2])], dtype=np.float64)
    if not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be finite")
    return point


def solve_similarity_transform(source_points: np.ndarray, target_points: np.ndarray) -> tuple[np.ndarray, float, float]:
    src = np.asarray(source_points, dtype=np.float64)
    dst = np.asarray(target_points, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[0] < 2 or src.shape[1] != 3:
        raise ValueError("similarity solve requires Nx3 source/target points")

    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    covariance = (dst_centered.T @ src_centered) / float(src.shape[0])
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0.0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt

    src_var = float(np.mean(np.sum(src_centered * src_centered, axis=1)))
    if not math.isfinite(src_var) or src_var <= 1e-12:
        raise ValueError("degenerate source variance")

    scale = float(np.sum(singular_values * np.diag(correction)) / src_var)
    translation = dst_mean - scale * (rotation @ src_mean)

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation

    residuals = (src @ (scale * rotation).T) + translation - dst
    rmse = float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))))
    return matrix, scale, rmse


def solve_scene_similarity(
    correspondences: Sequence[Mapping[str, Any]],
    *,
    source: str = "camera_device_similarity",
    residual_units: str = "scene_units",
) -> dict[str, Any]:
    normalized = []
    source_points = []
    target_points = []

    for item in correspondences or []:
        if not isinstance(item, Mapping):
            continue
        camera_id = str(item.get("camera_id") or "").strip()
        if not camera_id:
            continue
        world_point = _as_point3(item.get("world_position_m") or (), name="world_position_m")
        scene_point = _as_point3(item.get("scene_position") or (), name="scene_position")
        normalized.append(
            {
                "camera_id": camera_id,
                **(
                    {"anchor_id": str(item.get("anchor_id")).strip()}
                    if str(item.get("anchor_id") or "").strip()
                    else {}
                ),
                "world_position_m": [float(x) for x in world_point],
                "scene_position": [float(x) for x in scene_point],
            }
        )
        source_points.append(world_point)
        target_points.append(scene_point)

    if len(source_points) < 2:
        raise ValueError("scene registration requires at least two camera correspondences")

    matrix, scale, rmse = solve_similarity_transform(
        np.stack(source_points, axis=0),
        np.stack(target_points, axis=0),
    )
    rotation = matrix[:3, :3] / max(scale, 1e-9)

    per_camera = []
    residual_values = []
    for item in normalized:
        source_point = np.asarray(item["world_position_m"], dtype=np.float64)
        target_point = np.asarray(item["scene_position"], dtype=np.float64)
        transformed = (matrix[:3, :3] @ source_point) + matrix[:3, 3]
        residual = float(np.linalg.norm(transformed - target_point))
        residual_values.append(residual)
        per_camera.append(
            {
                **item,
                "residual_scene_units": residual,
            }
        )

    mean_residual = float(np.mean(residual_values)) if residual_values else 0.0
    max_residual = float(np.max(residual_values)) if residual_values else 0.0

    return {
        "source": str(source or "camera_device_similarity"),
        "residual_units": str(residual_units or "scene_units"),
        "camera_count": len(per_camera),
        "scene_per_m": float(scale),
        "s_obj_to_m": float(1.0 / scale) if scale > 1e-9 else 1.0,
        "mean_residual": mean_residual,
        "max_residual": max_residual,
        "position_rmse_scene_units": float(rmse),
        "position_rmse_m": float(rmse / scale) if scale > 1e-9 else math.inf,
        "max_residual_m": float(max_residual / scale) if scale > 1e-9 else math.inf,
        "world_to_scene_col_major": [float(x) for x in matrix.flatten(order="F")],
        "matrix_row_major": [float(x) for x in matrix.reshape(-1)],
        "rotation_row_major": [float(x) for x in rotation.reshape(-1)],
        "correspondences": per_camera,
    }


def camera_anchor_state_payload(
    correspondences: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the exact, portable Menon camera-anchor state used by a fit."""

    anchors: list[dict[str, Any]] = []
    for item in correspondences or []:
        if not isinstance(item, Mapping):
            continue
        camera_id = str(item.get("camera_id") or "").strip()
        anchor_id = str(item.get("anchor_id") or "").strip()
        if not camera_id or not anchor_id:
            raise ValueError("camera anchor state requires camera_id and anchor_id")
        scene_point = _as_point3(item.get("scene_position") or (), name="scene_position")
        anchors.append(
            {
                "anchor_id": anchor_id,
                "camera_id": camera_id,
                "scene_position": [float(value) for value in scene_point],
            }
        )
    if not anchors:
        raise ValueError("camera anchor state requires at least one anchor")
    anchors.sort(key=lambda item: (item["camera_id"], item["anchor_id"]))
    camera_ids = [item["camera_id"] for item in anchors]
    if len(camera_ids) != len(set(camera_ids)):
        raise ValueError("camera anchor state contains duplicate camera IDs")
    return {
        "contract": CAMERA_ANCHOR_STATE_CONTRACT,
        "contract_version": CAMERA_ANCHOR_STATE_CONTRACT_VERSION,
        "anchors": anchors,
    }


def camera_anchor_state_sha256(
    correspondences: Sequence[Mapping[str, Any]],
) -> str:
    payload = camera_anchor_state_payload(correspondences)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "CAMERA_ANCHOR_STATE_CONTRACT",
    "CAMERA_ANCHOR_STATE_CONTRACT_VERSION",
    "camera_anchor_state_payload",
    "camera_anchor_state_sha256",
    "solve_scene_similarity",
    "solve_similarity_transform",
]
