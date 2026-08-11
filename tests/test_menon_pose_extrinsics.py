#!/usr/bin/env python3
"""Unit tests for Menon PoseV1 -> extrinsics conversion."""

from __future__ import annotations

import numpy as np

from calibration_bundle import pose_to_E_col_major
from noesis.calibration.geometry import pixel_to_world
from noesis.calibration.pose_v1 import E_col_major_to_pose_v1


BACKEND_FRAME = "backend_world_m"


def _reshape(E_col_major):
    return np.array(E_col_major, dtype=np.float64).reshape((4, 4), order="F")


def test_pose_to_e_identity() -> None:
    pose = {
        "position": [0.0, 0.0, 0.0],
        "yaw_pitch_roll_deg": [0.0, 0.0, 0.0],
        "rotation_order": "YXZ",
        "frame": BACKEND_FRAME,
    }
    E = pose_to_E_col_major(pose)
    assert isinstance(E, list) and len(E) == 16
    expected = np.eye(4, dtype=np.float64)
    expected[:3, :3] = np.diag([-1.0, -1.0, 1.0])
    np.testing.assert_allclose(_reshape(E), expected, atol=1e-9, rtol=1e-9)


def test_pose_to_e_translation_only() -> None:
    pose = {
        "position": [1.0, 2.0, 3.0],
        "yaw_pitch_roll_deg": [0.0, 0.0, 0.0],
        "rotation_order": "YXZ",
        "frame": BACKEND_FRAME,
    }
    E = pose_to_E_col_major(pose)
    assert isinstance(E, list) and len(E) == 16

    expected = np.eye(4, dtype=np.float64)
    expected[:3, :3] = np.diag([-1.0, -1.0, 1.0])
    expected[:3, 3] = np.array([1.0, 2.0, -3.0], dtype=np.float64)
    np.testing.assert_allclose(_reshape(E), expected, atol=1e-9, rtol=1e-9)


def test_pose_to_e_yaw_rotation() -> None:
    pose = {
        "position": [1.0, 2.0, 3.0],
        "yaw_pitch_roll_deg": [90.0, 0.0, 0.0],
        "rotation_order": "YXZ",
        "frame": BACKEND_FRAME,
    }
    E = pose_to_E_col_major(pose)
    assert isinstance(E, list) and len(E) == 16

    R_wc = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    Twc = np.eye(4, dtype=np.float64)
    Twc[:3, :3] = R_wc
    Twc[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    expected = np.linalg.inv(Twc)
    np.testing.assert_allclose(_reshape(E), expected, atol=1e-9, rtol=1e-9)


def test_pose_round_trip_from_extrinsics() -> None:
    pose = {
        "position": [6.5, 2.4, 0.15],
        "yaw_pitch_roll_deg": [-13.0, 15.0, 0.0],
        "rotation_order": "YXZ",
        "frame": BACKEND_FRAME,
        "source": "unit-test",
    }
    E = pose_to_E_col_major(pose)
    round_trip = E_col_major_to_pose_v1(E, source="unit-test")
    assert round_trip is not None
    np.testing.assert_allclose(round_trip["position"], pose["position"], atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(round_trip["yaw_pitch_roll_deg"], pose["yaw_pitch_roll_deg"], atol=1e-9, rtol=1e-9)
    assert round_trip["rotation_order"] == "YXZ"
    assert round_trip["frame"] == BACKEND_FRAME


def test_pose_to_e_makes_lower_pixels_descend_in_world_y() -> None:
    pose = {
        "position": [0.0, 0.0, 0.0],
        "yaw_pitch_roll_deg": [0.0, 0.0, 0.0],
        "rotation_order": "YXZ",
        "frame": BACKEND_FRAME,
    }
    E = pose_to_E_col_major(pose)
    K = np.array([[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    top = pixel_to_world(K, E, 0.0, 1.0, 500.0, 400.0, depth_m=1.0)
    bottom = pixel_to_world(K, E, 0.0, 1.0, 500.0, 600.0, depth_m=1.0)
    assert top.ok and bottom.ok
    assert float(bottom.world_point[1]) < float(top.world_point[1])


def test_pose_to_e_rejects_invalid_contract() -> None:
    bad_pose = {
        "position": [0.0, 0.0, 0.0],
        "yaw_pitch_roll_deg": [0.0, 0.0, 0.0],
        "rotation_order": "ZYX",
        "frame": "world",
    }
    assert pose_to_E_col_major(bad_pose) is None
