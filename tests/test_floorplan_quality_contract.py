from __future__ import annotations

import base64
import time
from pathlib import Path

import numpy as np
import pytest

import geometry.depth_source as depth_source
from geometry.depth_source import (
    DepthStorageManager,
    _fit_ray_to_floorplan_alignment,
    _world_y_from_camera_points,
)


def _decode_grid(layer: dict) -> np.ndarray:
    return np.frombuffer(
        base64.b64decode(layer["grid_b64"]),
        dtype=np.float32,
    ).reshape(layer["grid_shape"])


def _camera_to_world_y_up(*, camera_height_m: float = 2.0) -> np.ndarray:
    # OpenCV camera +Y points down, while canonical world +Y points up.
    transform = np.eye(4, dtype=np.float32)
    transform[1, 1] = -1.0
    transform[1, 3] = float(camera_height_m)
    return transform


def _manager(
    tmp_path: Path,
    camera_id: str,
    intrinsics: list[float],
) -> DepthStorageManager:
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
    camera_to_world = _camera_to_world_y_up()
    world_to_camera = np.linalg.inv(camera_to_world)
    manager.calibration_bundle = {
        "cameras": {
            "K": {camera_id: intrinsics},
            "E": {
                camera_id: world_to_camera.reshape(
                    -1,
                    order="F",
                ).astype(float).tolist()
            },
        }
    }
    return manager


def test_world_y_transform_consumes_opencv_camera_y_without_a_second_flip() -> None:
    camera_to_world = _camera_to_world_y_up()
    points_camera = np.array(
        [
            [0.0, 2.0, 2.0],  # floor, two metres below the camera
            [0.0, 1.0, 2.0],  # one-metre-high object
        ],
        dtype=np.float32,
    )

    world_y = _world_y_from_camera_points(points_camera, camera_to_world)

    np.testing.assert_allclose(world_y, np.array([0.0, 1.0], dtype=np.float32))


def test_ray_alignment_uses_only_depth_floor_contacts_between_distinct_planes() -> None:
    height = 80
    width = 96
    fx = 100.0
    fy = 100.0
    cx = 47.5
    cy = 30.0
    calibrated_floor_y = 0.0
    depth_floor_y = 0.5
    camera_to_world = _camera_to_world_y_up(camera_height_m=2.0).astype(
        np.float64
    )
    world_to_camera = np.linalg.inv(camera_to_world)
    extrinsics = world_to_camera.reshape(-1, order="F").tolist()

    image_rows, image_cols = np.indices((height, width), dtype=np.float64)
    ray_y_camera = (image_rows - cy) / fy
    depth = np.full((height, width), 3.0, dtype=np.float32)
    floor_pixels = image_rows >= 40
    depth[floor_pixels] = (
        (2.0 - depth_floor_y) / ray_y_camera[floor_pixels]
    ).astype(np.float32)
    x_cam = ((image_cols - cx) / fx) * depth
    z_cam = depth.copy()
    valid = np.isfinite(depth) & (depth > 0.1) & (depth < 50.0)

    result = _fit_ray_to_floorplan_alignment(
        camera_id="synthetic-distinct-floor-planes",
        intrinsics=np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        extrinsics_col_major=extrinsics,
        calibrated_floor_y=calibrated_floor_y,
        depth_floor_y=depth_floor_y,
        depth=depth,
        conf=np.ones_like(depth, dtype=np.float32),
        mask=np.ones_like(depth, dtype=np.uint8),
        valid=valid,
        x_cam=x_cam,
        z_cam=z_cam,
        bounds={
            "min_x": -20.0,
            "max_x": 20.0,
            "min_z": 0.0,
            "max_z": 20.0,
        },
        walkable_grid=None,
        obstacle_height_grid=None,
    )

    assert result["quality"] == "ok"
    assert result["sample_mode"] == "depth_floor_contact_valid_mask_conf"
    assert result["sample_count"] >= 64
    assert result["residual_m"]["p95"] < 1e-4
    np.testing.assert_allclose(
        np.asarray(result["matrix_2x3"], dtype=np.float64),
        np.array(
            [
                [0.75, 0.0, 0.0],
                [0.0, 0.75, 0.0],
            ],
            dtype=np.float64,
        ),
        atol=1e-4,
        rtol=0.0,
    )


def test_ray_alignment_fails_closed_without_an_authored_floor() -> None:
    scalar = np.ones((1, 1), dtype=np.float32)
    result = _fit_ray_to_floorplan_alignment(
        camera_id="missing-authored-floor",
        intrinsics=np.eye(3, dtype=np.float64),
        extrinsics_col_major=np.eye(4, dtype=np.float64)
        .reshape(-1, order="F")
        .tolist(),
        calibrated_floor_y=None,
        depth_floor_y=0.0,
        depth=scalar,
        conf=scalar,
        mask=np.ones((1, 1), dtype=np.uint8),
        valid=np.ones((1, 1), dtype=bool),
        x_cam=np.zeros((1, 1), dtype=np.float32),
        z_cam=scalar,
        bounds={
            "min_x": -1.0,
            "max_x": 1.0,
            "min_z": 0.0,
            "max_z": 2.0,
        },
        walkable_grid=None,
        obstacle_height_grid=None,
    )

    assert result["quality"] == "unavailable"
    assert result["reason"] == "invalid_calibrated_floor_y"
    assert "matrix_2x3" not in result


def test_floorplan_alignment_receives_authored_and_agl_corrected_floors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera_id = "alignment-floor-contract"
    manager = _manager(tmp_path, camera_id, [8.0, 8.0, 3.5, 2.5])
    manager.calibration_bundle["align"] = {"floor_y": 0.0}
    received: list[tuple[float | None, float]] = []

    monkeypatch.setattr(
        depth_source,
        "_estimate_floor_y_from_horizontal_points",
        lambda *_args, **_kwargs: (
            -1.0,
            {"mode": "synthetic", "floor_y": -1.0},
        ),
    )
    monkeypatch.setattr(
        depth_source,
        "_normalize_floorplan_agl_heights",
        lambda heights, _quality_mask: (
            np.asarray(heights, dtype=np.float32),
            0.25,
        ),
    )

    def _capture_alignment(**kwargs: object) -> dict[str, object]:
        received.append(
            (
                kwargs["calibrated_floor_y"],  # type: ignore[arg-type]
                float(kwargs["depth_floor_y"]),
            )
        )
        return {
            "version": 1,
            "quality": "unavailable",
            "reason": "test_probe",
            "source": "floorplan_depth_snapshot",
            "from": "calibrated_floor_contact_ray_camera_local_xz",
            "to": "floorplan_depth_camera_local_xz",
        }

    monkeypatch.setattr(
        depth_source,
        "_fit_ray_to_floorplan_alignment",
        _capture_alignment,
    )

    depth = np.full((6, 8), 2.0, dtype=np.float32)
    confidence = np.ones_like(depth)
    mask = np.ones_like(depth, dtype=np.uint8)

    try:
        manager.store(
            camera_id,
            int(time.time() * 1_000_000),
            depth,
            confidence,
            mask,
        ).wait(5.0)
        payload = manager.generate_topdown_floorplan(
            camera_id,
            max_age_sec=60.0,
            grid_res_m=0.25,
            max_extent_m=5.0,
        )

        assert payload.get("error") is None
        assert received == [(0.0, -0.75)]
    finally:
        manager.shutdown(wait=True, timeout=5.0)


def test_floorplan_height_and_agl_preserve_world_y_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera_id = "camera-y-test"
    manager = _manager(tmp_path, camera_id, [4.0, 4.0, 1.0, 1.0])
    monkeypatch.setattr(
        depth_source,
        "_estimate_floor_y_from_horizontal_points",
        lambda *_args, **_kwargs: (0.0, {"mode": "synthetic", "floor_y": 0.0}),
    )
    depth = np.zeros((6, 3), dtype=np.float32)
    confidence = np.zeros_like(depth)
    mask = np.zeros_like(depth, dtype=np.uint8)
    # Both points are at z=2 m. The lower image point is world floor Y=0;
    # the higher image point is an object at world Y=1.
    depth[5, 0] = 2.0
    confidence[5, 0] = 1.0
    mask[5, 0] = 1
    depth[3, 2] = 2.0
    confidence[3, 2] = 1.0
    mask[3, 2] = 1

    try:
        manager.store(
            camera_id,
            int(time.time() * 1_000_000),
            depth,
            confidence,
            mask,
        ).wait(5.0)
        payload = manager.generate_topdown_floorplan(
            camera_id,
            max_age_sec=60.0,
            grid_res_m=0.25,
            max_extent_m=5.0,
        )

        assert payload.get("error") is None
        observed = _decode_grid(payload["observed"]) > 0.5
        raw_height = _decode_grid(payload["height"])[observed]
        height_agl = _decode_grid(payload["height_agl"])[observed]
        assert raw_height.size == 2
        assert height_agl.size == 2
        assert float(np.min(raw_height)) == pytest.approx(0.0, abs=0.05)
        assert float(np.max(raw_height)) == pytest.approx(1.0, abs=0.05)
        assert float(np.min(height_agl)) == pytest.approx(0.0, abs=0.05)
        assert float(np.max(height_agl)) > 0.8
        assert payload["height_agl_meta"]["floor_y"] == pytest.approx(0.0)
    finally:
        manager.shutdown(wait=True, timeout=5.0)


def test_dense_synthetic_floor_estimates_canonical_world_y(
    tmp_path: Path,
) -> None:
    camera_id = "synthetic-floor"
    fx = 4.0
    fy = 4.0
    cx = 5.5
    cy = 2.0
    manager = _manager(tmp_path, camera_id, [fx, fy, cx, cy])
    height, width = 12, 12
    depth = np.zeros((height, width), dtype=np.float32)
    for row in range(3, height):
        depth[row, :] = (2.0 * fy) / (float(row) - cy)
    confidence = (depth > 0.0).astype(np.float32)
    mask = (depth > 0.0).astype(np.uint8)

    try:
        manager.store(
            camera_id,
            int(time.time() * 1_000_000),
            depth,
            confidence,
            mask,
        ).wait(5.0)
        payload = manager.generate_topdown_floorplan(
            camera_id,
            max_age_sec=60.0,
            grid_res_m=0.5,
            max_extent_m=20.0,
        )

        assert payload.get("error") is None
        assert payload["height_agl_meta"]["floor_y"] == pytest.approx(
            0.0,
            abs=0.05,
        )
        observed = _decode_grid(payload["observed"]) > 0.5
        height_agl = _decode_grid(payload["height_agl"])[observed]
        assert height_agl.size > 0
        assert float(np.nanpercentile(height_agl, 95.0)) < 0.05
    finally:
        manager.shutdown(wait=True, timeout=5.0)


def test_floorplan_distance_uses_weight_sum_and_marks_unknown_cells(
    tmp_path: Path,
) -> None:
    camera_id = "distance-test"
    manager = _manager(tmp_path, camera_id, [1000.0, 1000.0, 0.0, 0.0])
    depth = np.array([[1.0, 3.0]], dtype=np.float32)
    confidence = np.array([[1.0, 0.1]], dtype=np.float32)
    mask = np.ones_like(depth, dtype=np.uint8)

    try:
        manager.store(
            camera_id,
            int(time.time() * 1_000_000),
            depth,
            confidence,
            mask,
        ).wait(5.0)
        payload = manager.generate_topdown_floorplan(
            camera_id,
            max_age_sec=60.0,
            grid_res_m=10.0,
            max_extent_m=10.0,
        )

        assert payload.get("error") is None
        observed = _decode_grid(payload["observed"])
        unknown = _decode_grid(payload["unknown"])
        distance = _decode_grid(payload["distance"])
        raw_height = _decode_grid(payload["height"])
        height_agl = _decode_grid(payload["height_agl"])

        np.testing.assert_array_equal(observed + unknown, np.ones_like(observed))
        assert int(np.count_nonzero(observed)) == 1
        expected_weighted_distance = (1.0 * 1.0 + 3.0 * 0.1) / (1.0 + 0.1)
        assert distance[observed > 0.5].item() == pytest.approx(
            expected_weighted_distance,
            abs=1e-6,
        )
        # Unknown is a sideband: the raw diagnostic rasters remain present and
        # retain their existing finite-fill behavior instead of being blacked out.
        assert np.isfinite(raw_height).all()
        assert np.isfinite(height_agl[observed > 0.5]).all()
        assert payload["observation_meta"]["observed_cells"] == 1
        assert (
            payload["observation_meta"]["unknown_cells"]
            == int(np.count_nonzero(unknown))
        )
    finally:
        manager.shutdown(wait=True, timeout=5.0)


def test_calibrated_ground_projection_is_diagnostic_not_horizon_bounds(
    tmp_path: Path,
) -> None:
    camera_id = "stable-bounds"
    manager = _manager(
        tmp_path,
        camera_id,
        [8.0, 8.0, 3.5, 2.5],
    )
    manager.calibration_bundle["align"] = {"floor_y": 0.0}
    confidence = np.ones((6, 8), dtype=np.float32)
    mask = np.ones((6, 8), dtype=np.uint8)
    base_timestamp = int(time.time() * 1_000_000)

    try:
        descriptors = []
        for offset, depth_value in enumerate((2.0, 3.0)):
            receipt = manager.store(
                camera_id,
                base_timestamp + offset,
                np.full((6, 8), depth_value, dtype=np.float32),
                confidence,
                mask,
            ).wait(5.0)
            descriptors.append(manager.describe_snapshot(receipt.path))

        payloads = []
        for descriptor in descriptors:
            payloads.append(
                manager.generate_topdown_floorplan(
                    camera_id,
                    max_age_sec=0.0,
                    grid_res_m=0.15,
                    max_extent_m=10.0,
                    snapshot_ref=descriptor.storage_ref,
                    snapshot_id=descriptor.write_id,
                    snapshot_content_sha256=descriptor.content_sha256,
                )
            )

        assert all(payload.get("error") is None for payload in payloads)
        for payload in payloads:
            assert (
                payload["bounds_meta"]["source"]
                == "strict_observed_ground_depth_quantized"
            )
            assert payload["bounds_meta"]["contract"] == "noesis.floorplan.bounds.v3"
            assert (
                payload["bounds_meta"]["calibrated_ground_projection_policy"]
                == "diagnostic_only_never_expands_observed_depth"
            )
            projection = payload["bounds_meta"]["calibrated_ground_projection"]
            assert projection["sample_count"] >= 4
            assert (
                payload["bounds"]["max_x"] - payload["bounds"]["min_x"]
                < projection["projected_half_width_m"] * 2.0
            )
            assert (
                payload["bounds"]["max_z"]
                < projection["projected_forward_extent_m"]
            )
            assert payload["bounds_meta"]["omitted_outlier_point_count"] == 0
    finally:
        manager.shutdown(wait=True, timeout=5.0)
