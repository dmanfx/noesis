from __future__ import annotations

import numpy as np
import pytest

from geometry.depth_source import (
    _compute_floorplan_detail_layers,
    _compute_kitchen_clean_floorplan_layers,
    _compute_kitchen_clean_floorplan_layers_from_agl_grids,
    _compute_kitchen_clean_floorplan_layers_from_grids,
    _fit_floor_plane_from_points,
    _metric_morphology_size,
)


def test_detail_layers_separate_floor_furniture_walls_and_ceiling() -> None:
    rows, cols = 8, 8
    heights: list[float] = []
    normals: list[list[float]] = []
    weights: list[float] = []
    x_indices: list[int] = []
    z_indices: list[int] = []
    colors: list[list[int]] = []
    support = np.zeros((rows, cols), dtype=np.uint32)

    def add(
        row: int,
        col: int,
        height: float,
        normal: list[float],
        color: list[int],
    ) -> None:
        heights.append(height)
        normals.append(normal)
        weights.append(1.0)
        x_indices.append(col)
        z_indices.append(row)
        colors.append(color)
        support[row, col] += 1

    for row in range(rows):
        for col in range(cols):
            add(row, col, 0.0, [0.0, 1.0, 0.0], [80, 80, 80])
            # A ceiling must never become the visible furniture surface.
            add(row, col, 2.4, [0.0, -1.0, 0.0], [240, 240, 240])

    for row in range(2, 6):
        for col in range(2, 6):
            add(row, col, 0.8, [0.0, 1.0, 0.0], [180, 40, 30])
            add(row, col, 0.8, [0.0, 1.0, 0.0], [180, 40, 30])

    for row in range(rows):
        add(row, 0, 1.2, [1.0, 0.0, 0.0], [30, 70, 160])
        add(row, 0, 1.4, [1.0, 0.0, 0.0], [30, 70, 160])

    layers = _compute_floorplan_detail_layers(
        height_agl_pts=np.asarray(heights, dtype=np.float32),
        normals_world_pts=np.asarray(normals, dtype=np.float32),
        pts_weight=np.asarray(weights, dtype=np.float32),
        x_idx=np.asarray(x_indices, dtype=np.int32),
        z_idx=np.asarray(z_indices, dtype=np.int32),
        support_grid=support,
        rgb_pts=np.asarray(colors, dtype=np.uint8),
    )

    structural = layers["structural_height"]
    assert structural.shape == (rows, cols)
    assert float(np.nanmax(structural[2:6, 2:6])) == pytest.approx(
        0.825,
        abs=0.03,
    )
    assert float(structural[1, 1]) == 0.0
    assert float(np.max(layers["wall_support"][:, 0])) >= 0.5
    assert float(np.max(layers["wall_support"][:, 2:])) == 0.0
    assert np.all(layers["room_footprint"] > 0.5)
    assert layers["meta"]["cells"]["furniture_surface"] == 16
    assert layers["meta"]["cells"]["surface_rgb"] > 0


def test_fit_floor_plane_robust_to_outliers() -> None:
    rng = np.random.default_rng(0)
    n = 800
    x = rng.uniform(-2.0, 2.0, size=n).astype(np.float32)
    z = rng.uniform(0.0, 4.0, size=n).astype(np.float32)
    y_floor = (0.02 * x) + (-0.01 * z) + 1.2
    y = (y_floor + rng.normal(0.0, 0.004, size=n)).astype(np.float32)
    # Add tall outliers (furniture) that should not affect the floor fit.
    y[:80] += 0.8
    w = np.ones_like(y, dtype=np.float32)

    a, b, c, meta = _fit_floor_plane_from_points(x, z, y, w)

    assert abs(a - 0.02) < 0.02
    assert abs(b - (-0.01)) < 0.02
    assert abs(c - 1.2) < 0.05
    assert int(meta.get("seed_count") or 0) > 0


def test_floorplan_morphology_radius_is_converted_from_metres() -> None:
    assert _metric_morphology_size(0.15, 0.15) == 3
    assert _metric_morphology_size(0.15, 0.075) == 5
    assert _metric_morphology_size(0.15, 0.50) == 1


def test_grid_layer_helpers_use_resolution_aware_morphology() -> None:
    shape = (10, 10)
    support = np.ones(shape, dtype=np.uint32)

    _, _, grid_meta = _compute_kitchen_clean_floorplan_layers_from_grids(
        "kitchen_camera",
        height_grid=np.zeros(shape, dtype=np.float32),
        support_grid=support,
        grid_res_m=0.075,
    )
    assert grid_meta["thresholds"]["morph_size_cells"] == 5
    assert grid_meta["thresholds"]["grid_res_m"] == 0.075

    _, _, agl_meta = _compute_kitchen_clean_floorplan_layers_from_agl_grids(
        "kitchen_camera",
        height_agl_min_grid=np.zeros(shape, dtype=np.float32),
        height_agl_max_grid=np.zeros(shape, dtype=np.float32),
        support_grid=support,
        floor_support_grid=support,
        obstacle_support_grid=np.zeros(shape, dtype=np.uint32),
        grid_res_m=0.50,
    )
    assert agl_meta["thresholds"]["morph_size_cells"] == 1
    assert agl_meta["grid_res_m"] == 0.50


def test_kitchen_clean_layers_obstacles_and_unknown_walkable() -> None:
    h_px, w_px = 10, 10
    support = np.zeros((h_px, w_px), dtype=np.uint32)

    xs: list[float] = []
    zs: list[float] = []
    ys: list[float] = []
    x_idx: list[int] = []
    z_idx: list[int] = []

    # Provide floor points only in the bottom half; top half stays unobserved.
    for zi in range(5, h_px):
        for xi in range(w_px):
            xs.append((xi - (w_px / 2)) * 0.12)
            zs.append(zi * 0.12)
            ys.append(1.0)
            x_idx.append(xi)
            z_idx.append(zi)
            support[zi, xi] += 1

    # Large obstacle block (4x4) with repeated top-surface support so it survives cleanup.
    obs_rows = range(6, 10)
    obs_cols = range(3, 7)
    for zi in obs_rows:
        for xi in obs_cols:
            for _ in range(3):
                xs.append((xi - (w_px / 2)) * 0.12)
                zs.append(zi * 0.12)
                ys.append(1.5)
                x_idx.append(xi)
                z_idx.append(zi)
                support[zi, xi] += 1

    # Single-cell speckle that should be removed by morphology/component filtering.
    speckle = (8, 0)
    xs.append((speckle[1] - (w_px / 2)) * 0.12)
    zs.append(speckle[0] * 0.12)
    ys.append(1.5)
    x_idx.append(speckle[1])
    z_idx.append(speckle[0])
    support[speckle[0], speckle[1]] += 1

    obstacle_height, walkable, meta = _compute_kitchen_clean_floorplan_layers(
        "kitchen_camera",
        x_cam_pts=np.asarray(xs, dtype=np.float32),
        z_cam_pts=np.asarray(zs, dtype=np.float32),
        y_world_pts=np.asarray(ys, dtype=np.float32),
        pts_weight=None,
        x_idx=np.asarray(x_idx, dtype=np.int32),
        z_idx=np.asarray(z_idx, dtype=np.int32),
        support_grid=support,
    )

    assert obstacle_height.shape == (h_px, w_px)
    assert walkable.shape == (h_px, w_px)
    assert meta.get("mode") == "kitchen_clean_layers"
    assert meta["thresholds"]["morph_radius_m"] == 0.15
    assert meta["thresholds"]["morph_size_cells"] == 3

    # Unobserved/outside region defaults to non-walkable (outside the inferred footprint).
    assert float(np.max(walkable[:5, :])) == 0.0

    # Obstacle region should be detected and marked non-walkable.
    block_h = obstacle_height[np.ix_(list(obs_rows), list(obs_cols))]
    block_w = walkable[np.ix_(list(obs_rows), list(obs_cols))]
    assert float(np.min(block_h)) >= 0.14
    assert float(np.max(block_w)) == 0.0

    # Speckle should be removed.
    assert float(obstacle_height[speckle]) == 0.0
    assert float(walkable[speckle]) == 1.0


def test_kitchen_clean_layers_handles_inverted_height_axis() -> None:
    # DS8 conventions require +Y up in the "camera_local_ground" frame. If a deployment
    # violates this, floorplan layers may need additional sign/plane selection logic.
    # Keep this test as a placeholder for future regression coverage once a real
    # inverted-axis calibration is observed in the wild.
    pass
