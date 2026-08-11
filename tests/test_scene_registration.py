from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from noesis.calibration.scene_registration import (
    camera_anchor_state_sha256,
    solve_scene_similarity,
    solve_similarity_transform,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_solve_similarity_transform_recovers_scale_and_translation() -> None:
    source = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [4.0, 2.0, 3.0],
            [1.0, 5.0, 3.0],
        ],
        dtype=np.float64,
    )
    target = source * 100.0
    matrix, scale, rmse = solve_similarity_transform(source, target)
    assert abs(scale - 100.0) < 1e-9
    assert rmse < 1e-9
    assert np.allclose(matrix, np.diag([100.0, 100.0, 100.0, 1.0]), atol=1e-9)


def test_solve_scene_similarity_reports_per_camera_residuals() -> None:
    payload = solve_scene_similarity(
        [
            {
                "camera_id": "living-room",
                "world_position_m": [1.0, 2.0, 3.0],
                "scene_position": [100.0, 200.0, 300.0],
            },
            {
                "camera_id": "kitchen",
                "world_position_m": [4.0, 2.0, 3.0],
                "scene_position": [400.0, 200.0, 300.0],
            },
        ]
    )
    assert payload["camera_count"] == 2
    assert abs(float(payload["scene_per_m"]) - 100.0) < 1e-9
    assert abs(float(payload["s_obj_to_m"]) - 0.01) < 1e-12
    assert payload["source"] == "camera_device_similarity"
    assert payload["residual_units"] == "scene_units"
    assert len(payload["world_to_scene_col_major"]) == 16
    assert len(payload["matrix_row_major"]) == 16
    assert payload["mean_residual"] < 1e-9
    assert payload["max_residual"] < 1e-9
    assert payload["position_rmse_m"] < 1e-9
    assert payload["max_residual_m"] < 1e-9
    assert payload["correspondences"][0]["camera_id"] == "living-room"
    assert payload["correspondences"][1]["camera_id"] == "kitchen"


def test_camera_anchor_state_digest_is_order_independent_and_binds_anchor_identity() -> None:
    rows = [
        {
            "anchor_id": "device-b",
            "camera_id": "living-room",
            "scene_position": [1.0, 2.0, 3.0],
        },
        {
            "anchor_id": "device-a",
            "camera_id": "kitchen",
            "scene_position": [4.0, 5.0, 6.0],
        },
    ]
    expected = camera_anchor_state_sha256(rows)
    assert camera_anchor_state_sha256(list(reversed(rows))) == expected
    changed = [dict(rows[0]), dict(rows[1])]
    changed[0]["anchor_id"] = "replacement-device"
    assert camera_anchor_state_sha256(changed) != expected


def test_promoted_scene_similarity_is_bound_and_within_five_centimeters() -> None:
    alignment = json.loads((REPO_ROOT / "config" / "ply_alignment.json").read_text())
    similarity = alignment["scene_similarity"]
    assert similarity["source"] == "menon_virtual_device_camera_similarity_v1"
    assert similarity["anchor_state_sha256"] == camera_anchor_state_sha256(
        similarity["correspondences"]
    )
    calibration_bytes = (REPO_ROOT / "config" / "camera_calibration.json").read_bytes()
    assert similarity["camera_calibration_sha256"] == hashlib.sha256(calibration_bytes).hexdigest()
    assert similarity["anchor_residual_limit_m"] == 0.05
    assert similarity["position_rmse_m"] <= 0.05
    assert similarity["max_residual_m"] <= 0.05

    scene_calibration = json.loads(
        (REPO_ROOT / "config" / "camera_calibration_menon_obj.json").read_text()
    )
    assert (
        scene_calibration["scene_anchor_binding"]["anchor_state_sha256"]
        == similarity["anchor_state_sha256"]
    )
    expected_anchors = {
        row["camera_id"]: np.asarray(row["scene_position"], dtype=np.float64)
        for row in similarity["correspondences"]
    }
    for camera_id, camera in scene_calibration["cameras"].items():
        world_to_camera = np.asarray(camera["E"], dtype=np.float64).reshape(
            (4, 4), order="F"
        )
        camera_center = np.linalg.inv(world_to_camera)[:3, 3]
        assert np.allclose(camera_center, expected_anchors[camera_id], rtol=0, atol=1e-9)
        assert camera["image_size"] == [1920, 1080]
