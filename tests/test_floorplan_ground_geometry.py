from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from geometry.depth_source import (
    DepthStorageManager,
    _camera_points_to_ground_frame,
    _condition_floorplan_metric_scale,
    _estimate_floor_y_from_horizontal_points,
    _resolve_floorplan_bounds,
    _world_y_from_camera_points,
)


def _pitched_camera_to_world(
    *,
    pitch_deg: float,
    camera_height_m: float = 2.0,
) -> np.ndarray:
    """Return an OpenCV camera pose pitched downward in a Y-up world."""
    pitch = np.deg2rad(float(pitch_deg))
    cosine = float(np.cos(pitch))
    sine = float(np.sin(pitch))
    transform = np.eye(4, dtype=np.float32)
    # Columns are camera-right, camera-down, and camera-forward in world axes.
    transform[:3, :3] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cosine, -sine],
            [0.0, sine, cosine],
        ],
        dtype=np.float32,
    )
    transform[1, 3] = float(camera_height_m)
    return transform


def _camera_to_world_y_up(*, camera_height_m: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[1, 1] = -1.0
    transform[1, 3] = float(camera_height_m)
    return transform


def test_ground_frame_does_not_smear_a_vertical_object_when_camera_is_pitched() -> None:
    camera_to_world = _pitched_camera_to_world(pitch_deg=24.0)
    world_points = np.column_stack(
        [
            np.full(19, 1.1, dtype=np.float32),
            np.linspace(0.0, 1.8, 19, dtype=np.float32),
            np.full(19, 4.2, dtype=np.float32),
        ]
    )
    points_camera = (
        world_points - camera_to_world[:3, 3]
    ) @ camera_to_world[:3, :3]

    x_ground, z_ground, y_world, meta = _camera_points_to_ground_frame(
        points_camera,
        camera_to_world,
    )

    # Raw optical Z changes with height under pitch and would streak the pole
    # across a top-down raster. The gravity-aligned ground coordinates do not.
    assert float(np.ptp(points_camera[:, 2])) > 0.7
    assert float(np.ptp(x_ground)) < 1e-5
    assert float(np.ptp(z_ground)) < 1e-5
    np.testing.assert_allclose(x_ground, 1.1, atol=1e-5, rtol=0.0)
    np.testing.assert_allclose(z_ground, 4.2, atol=1e-5, rtol=0.0)
    np.testing.assert_allclose(y_world, world_points[:, 1], atol=1e-5, rtol=0.0)
    assert meta["camera_pitch_deg"] == pytest.approx(24.0, abs=1e-5)


def test_floor_estimator_rejects_tiny_low_outlier_and_selects_real_floor_mode() -> None:
    rng = np.random.default_rng(109)
    y_world = np.concatenate(
        [
            rng.normal(-1.60, 0.003, 50),
            rng.normal(-0.84, 0.006, 400),
            rng.normal(0.20, 0.008, 4_000),
            rng.normal(2.40, 0.008, 5_550),
        ]
    ).astype(np.float32)
    normals_world = np.zeros((y_world.size, 3), dtype=np.float32)
    normals_world[:, 1] = 1.0

    floor_y, meta = _estimate_floor_y_from_horizontal_points(
        y_world,
        normals_world,
        np.ones(y_world.size, dtype=np.float32),
    )

    assert meta["quality"] == "ok"
    assert meta["mode"] == "histogram_lowest_coherent_peak"
    assert floor_y == pytest.approx(-0.84, abs=0.025)
    assert abs(floor_y - (-1.60)) > 0.70
    assert meta["window_mass_fraction"] >= 0.02


def test_metric_scale_recovers_calibrated_camera_height_and_object_heights() -> None:
    camera_height_m = 2.265
    camera_to_world = _camera_to_world_y_up(
        camera_height_m=camera_height_m
    )
    true_points_camera = np.array(
        [
            [0.0, camera_height_m, 3.0],
            [0.8, camera_height_m - 0.75, 4.0],
            [-1.1, camera_height_m - 1.20, 5.0],
        ],
        dtype=np.float32,
    )
    monocular_scale_error = 1.0 / 0.73
    observed_points_camera = (
        true_points_camera * monocular_scale_error
    ).astype(np.float32)
    observed_floor_y = float(
        _world_y_from_camera_points(
            observed_points_camera[:1],
            camera_to_world,
        )[0]
    )

    conditioned, corrected_floor_y, meta = _condition_floorplan_metric_scale(
        observed_points_camera,
        camera_to_world,
        observed_floor_y=observed_floor_y,
        calibrated_floor_y=0.0,
        floor_estimate_meta={
            "quality": "ok",
            "mode": "histogram_lowest_coherent_peak",
        },
    )

    assert meta["applied"] is True
    assert meta["reason"] == "calibrated_camera_height"
    assert meta["scale_factor"] == pytest.approx(0.73, abs=1e-6)
    assert corrected_floor_y == pytest.approx(0.0, abs=1e-6)
    assert meta["floor_residual_m"] == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(
        conditioned,
        true_points_camera,
        atol=1e-5,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        _world_y_from_camera_points(conditioned, camera_to_world),
        np.array([0.0, 0.75, 1.20], dtype=np.float32),
        atol=1e-5,
        rtol=0.0,
    )


def test_bounds_are_asymmetric_and_robust_to_non_authoritative_outliers() -> None:
    observed_x = np.linspace(-1.2, 3.4, 1_001, dtype=np.float32)
    observed_z = np.linspace(1.0, 5.0, 1_001, dtype=np.float32)
    x_ground = np.concatenate(
        [observed_x, np.array([-30.0, 30.0], dtype=np.float32)]
    )
    z_ground = np.concatenate(
        [observed_z, np.array([30.0, 30.0], dtype=np.float32)]
    )
    authoritative = np.concatenate(
        [
            np.ones(observed_x.size, dtype=bool),
            np.zeros(2, dtype=bool),
        ]
    )

    min_x, max_x, forward_extent, meta = _resolve_floorplan_bounds(
        x_ground=x_ground,
        z_ground=z_ground,
        authoritative_mask=authoritative,
        image_shape=(64, 96),
        intrinsics=(80.0, 80.0, 47.5, 31.5),
        camera_to_world=np.eye(4, dtype=np.float32),
        calibration_bundle={},
        grid_res_m=0.1,
        max_extent_m=40.0,
        pad_x_m=0.2,
        pad_z_m=0.2,
    )

    assert min_x == pytest.approx(-1.5)
    assert max_x == pytest.approx(4.0)
    assert forward_extent == pytest.approx(5.5)
    assert abs(min_x) != pytest.approx(max_x)
    assert meta["source"] == "strict_observed_ground_depth_quantized"
    assert meta["strict_input_point_count"] == observed_x.size
    assert meta["selected_point_count"] == observed_x.size
    assert meta["observed_x_low_m"] > -2.0
    assert meta["observed_x_high_m"] < 4.0


def test_exact_continuity_confidence_is_excluded_from_authoritative_bounds(
    tmp_path: Path,
) -> None:
    camera_id = "continuity-boundary"
    manager = DepthStorageManager(
        base_path=tmp_path / "depth",
        max_snapshots_per_camera=0,
        retention_minutes=0.0,
        max_total_bytes=None,
        enable_async=False,
        enforce_async=False,
        zarr_clevel=0,
        zarr_chunk_px=0,
        min_conf=0.0,
    )
    manager.calibration_bundle = {
        "cameras": {
            "K": {camera_id: [10.0, 10.0, 2.5, 2.5]},
            "E": {
                camera_id: np.eye(4, dtype=np.float32)
                .reshape(-1, order="F")
                .astype(float)
                .tolist()
            },
        }
    }
    depth = np.full((6, 6), 2.0, dtype=np.float32)
    confidence = np.ones_like(depth, dtype=np.float32)
    mask = np.ones_like(depth, dtype=np.uint8)
    continuity_pixels = ((0, 0), (0, 5), (5, 0), (5, 5))
    for row, column in continuity_pixels:
        depth[row, column] = 40.0
        confidence[row, column] = np.float32(0.20)

    try:
        manager.store(
            camera_id,
            int(time.time() * 1_000_000),
            depth,
            confidence,
            mask,
        )
        payload = manager.generate_topdown_floorplan(
            camera_id,
            max_age_sec=60.0,
            grid_res_m=0.1,
            max_extent_m=45.0,
        )
    finally:
        manager.shutdown(wait=True, timeout=2.0)

    assert payload.get("error") is None
    bounds_meta = payload["bounds_meta"]
    assert bounds_meta["source"] == "strict_observed_ground_depth_quantized"
    assert bounds_meta["strict_input_point_count"] == 32
    assert bounds_meta["selected_point_count"] == 32
    assert bounds_meta["omitted_outlier_point_count"] == 4
    assert payload["bounds"]["max_z"] < 3.0
