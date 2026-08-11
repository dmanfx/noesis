#!/usr/bin/env python3
"""Unit tests for tilt-only calibration preview."""
from __future__ import annotations

import math
from typing import List

import numpy as np

from noesis.calibration.tilt_preview import apply_preview_updates, tilt_correct_extrinsics


def _rot_x(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _build_E(R_wc: np.ndarray, C_world: np.ndarray) -> List[float]:
    R_cw = R_wc.T
    t_cw = -R_cw @ C_world
    E = np.eye(4, dtype=np.float64)
    E[:3, :3] = R_cw
    E[:3, 3] = t_cw
    return [float(x) for x in E.flatten(order="F")]


def test_tilt_correct_extrinsics_preserves_yaw_and_center() -> None:
    yaw = math.radians(30.0)
    pitch = math.radians(15.0)
    roll = math.radians(-5.0)
    R_yaw = _rot_y(yaw)
    R_pitch = _rot_x(pitch)
    R_roll = _rot_z(roll)

    R_wc_true = R_yaw
    R_wc_bad = R_yaw @ R_pitch @ R_roll
    C_world = np.array([2.0, 1.5, -3.0], dtype=np.float64)

    E_bad = _build_E(R_wc_bad, C_world)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    plane_normal_cam = R_wc_true.T @ up

    _, info = tilt_correct_extrinsics(E_bad, plane_normal_cam, preserve_yaw=True)

    assert abs(info["yaw_deg_before"] - info["yaw_deg_after"]) < 1e-6

    center_before = np.array(info["camera_center_before"])
    center_after = np.array(info["camera_center_after"])
    assert np.allclose(center_before, center_after, atol=1e-6)

    n_world_after = np.array(info["normal_world_after"])
    n_world_after = n_world_after / np.linalg.norm(n_world_after)
    assert np.allclose(n_world_after, up, atol=1e-5)


def test_apply_preview_updates_preserves_fields() -> None:
    base = {
        "align": {"matrix": [1.0] * 16, "floor_y": 0.1},
        "cameras": {
            "cam1": {"E": [0.0] * 16, "invert_e": 1},
        },
        "meta": {"note": "keep"},
    }
    updates = {"cam1": [2.0] * 16, "cam2": [3.0] * 16}

    updated = apply_preview_updates(base, updates)

    assert updated["meta"] == base["meta"]
    assert updated["align"] == base["align"]
    assert updated["cameras"]["cam1"]["invert_e"] == 1
    assert updated["cameras"]["cam1"]["E"] == [2.0] * 16
    assert updated["cameras"]["cam2"]["E"] == [3.0] * 16
    assert base["cameras"]["cam1"]["E"] == [0.0] * 16
