from __future__ import annotations

import numpy as np

from geometry.floorplan_shadow_fill import (
    PhoneSurface,
    ShadowFillSettings,
    apply_floor_underlay,
    apply_shadow_phone_authority,
    build_floor_underlay_masks,
    build_phone_floor_region_completion,
    build_phone_floor_visibility_from_views,
    build_phone_surface,
    build_phone_surface_from_views,
    build_region_surface_completion,
    compute_radial_occlusion_artifact,
    compute_static_floor_occlusion_shadow,
    floorplan_grid_centers,
    phone_view_points_to_static_floorplan,
)


def test_floor_shadow_ends_after_camera_ray_clears_low_obstacle() -> None:
    shape = (12, 5)
    bounds = {"min_x": -1.25, "max_x": 1.25, "min_z": 0.0, "max_z": 6.0}
    x_grid, z_grid = floorplan_grid_centers(bounds, shape)
    footprint = np.zeros(shape, dtype=bool)
    footprint[:, 2] = True
    structural = np.zeros(shape, dtype=np.float32)
    surface = np.zeros(shape, dtype=bool)
    obstacle_row = int(np.argmin(np.abs(z_grid[:, 2] - 2.25)))
    structural[obstacle_row, 2] = 1.0
    surface[obstacle_row, 2] = True

    result = compute_static_floor_occlusion_shadow(
        structural_height_m=structural,
        surface_observed=surface,
        raw_height_agl_m=np.zeros(shape, dtype=np.float32),
        wall_support=np.zeros(shape, dtype=np.float32),
        room_footprint=footprint,
        x_grid_m=x_grid,
        z_grid_m=z_grid,
        camera_height_m=2.0,
        settings=ShadowFillSettings(
            occluder_clearance_m=0.0,
            occluder_min_component_cells=1,
            occluder_dilation_cells=0,
            angular_step_deg=0.1,
        ),
    )

    shadow_ranges = z_grid[:, 2][result.shadow[:, 2]]
    assert np.any((shadow_ranges > 2.25) & (shadow_ranges < 4.5))
    assert not np.any(shadow_ranges >= 4.5)
    assert not result.shadow[obstacle_row, 2]


def test_phone_surface_uses_multiview_local_height_mode() -> None:
    x_grid = np.asarray([[0.0]], dtype=np.float64)
    z_grid = np.asarray([[1.0]], dtype=np.float64)
    points = np.asarray(
        [
            [-0.02, 0.79, 1.00],
            [0.01, 0.81, 1.01],
            [0.03, 0.80, 0.98],
            [0.02, 0.02, 1.01],
            [-0.03, 0.03, 0.99],
            [0.00, 0.42, 1.00],  # rejected: one phone view
            [0.00, 2.20, 1.00],  # rejected: ceiling range
        ],
        dtype=np.float64,
    )
    surface = build_phone_surface(
        points_static_m=points,
        confidence=np.asarray([0.8, 0.8, 0.8, 0.2, 0.2, 1.0, 1.0]),
        provenance=np.asarray([2, 2, 2, 2, 2, 2, 2], dtype=np.uint8),
        view_support=np.asarray([2, 3, 2, 2, 2, 1, 3], dtype=np.uint16),
        room_footprint=np.ones((1, 1), dtype=bool),
        x_grid_m=x_grid,
        z_grid_m=z_grid,
        settings=ShadowFillSettings(phone_min_quality=0.2),
    )

    assert surface.good[0, 0]
    assert int(surface.support[0, 0]) == 3
    assert abs(float(surface.height_m[0, 0]) - 0.80) <= 0.02


def test_phone_authority_changes_only_good_static_shadow_cells() -> None:
    static_height = np.asarray([[0.1, 1.5, 0.4], [0.2, 0.8, 0.6]], dtype=np.float32)
    static_agl = static_height.copy()
    density = np.full_like(static_height, 0.2)
    observed = np.ones_like(static_height, dtype=bool)
    footprint = np.ones_like(static_height, dtype=bool)
    shadow = np.asarray([[False, True, True], [False, True, False]])
    phone_height = np.asarray([[0.7, 0.3, 0.2], [0.4, 0.5, 0.9]], dtype=np.float32)
    phone_good = np.asarray([[True, True, False], [False, True, True]])
    phone = PhoneSurface(
        height_m=phone_height,
        quality=np.full_like(static_height, 0.8),
        support=np.full(static_height.shape, 3, dtype=np.uint16),
        spread_m=np.zeros_like(static_height),
        good=phone_good,
        eligible_voxel_count=12,
    )
    admission = np.asarray([[True, True, True], [True, False, True]])

    result = apply_shadow_phone_authority(
        static_height=static_height,
        static_height_agl=static_agl,
        static_density=density,
        static_observed=observed,
        room_footprint=footprint,
        shadow=shadow,
        phone_surface=phone,
        admission_mask=admission,
    )

    expected_admitted = shadow & phone_good & admission
    assert np.array_equal(result.admitted, expected_admitted)
    assert np.allclose(result.height[expected_admitted], phone_height[expected_admitted])
    assert np.array_equal(
        result.height[~expected_admitted],
        static_height[~expected_admitted],
    )
    assert result.unresolved[0, 2]
    assert result.unresolved[1, 1]
    assert not result.admitted[0, 0]


def test_radial_artifact_selects_sparse_comb_not_dense_object() -> None:
    shape = (100, 100)
    bounds = {"min_x": -2.0, "max_x": 2.0, "min_z": 0.0, "max_z": 4.0}
    x_grid, z_grid = floorplan_grid_centers(bounds, shape)
    radius = np.hypot(x_grid, z_grid)
    angle = np.arctan2(x_grid, z_grid)
    footprint = np.ones(shape, dtype=bool)
    height = np.zeros(shape, dtype=np.float32)
    density = np.full(shape, 0.001, dtype=np.float32)

    comb = (
        (np.abs(angle) < 0.10)
        & (radius >= 1.8)
        & (radius <= 3.0)
    )
    height[comb] = 0.9 + (0.45 * np.sin(radius[comb] * (2.0 * np.pi / 0.16)))
    dense_object = (
        (x_grid >= 0.75)
        & (x_grid <= 1.25)
        & (z_grid >= 2.0)
        & (z_grid <= 2.7)
    )
    height[dense_object] = 0.9
    density[dense_object] = 0.05

    artifact = compute_radial_occlusion_artifact(
        raw_height_agl_m=height,
        structural_height_m=np.zeros(shape, dtype=np.float32),
        density=density,
        observed=np.ones(shape, dtype=bool),
        wall_support=np.zeros(shape, dtype=np.float32),
        room_footprint=footprint,
        floor_shadow=footprint,
        x_grid_m=x_grid,
        z_grid_m=z_grid,
        settings=ShadowFillSettings(
            artifact_energy_percentile=60.0,
            artifact_min_component_cells=2,
        ),
    )

    assert np.count_nonzero(artifact.region & comb) >= 20
    assert not np.any(artifact.region & dense_object)


def test_original_phone_surface_counts_views_not_dense_pixels() -> None:
    shape = (2, 2)
    x_grid, z_grid = floorplan_grid_centers(
        {"min_x": 0.0, "max_x": 0.08, "min_z": 0.0, "max_z": 0.08},
        shape,
    )
    x = float(x_grid[0, 0])
    z = float(z_grid[0, 0])
    points = np.asarray(
        [
            [x, 0.80, z],
            [x, 0.81, z],  # duplicate pixels in view zero
            [x, 0.79, z],
            [x, 0.80, z],  # same upper mode from view one
            [x, 0.02, z],
            [x, 0.03, z],  # floor also has two views
            [x, 1.40, z],  # one-view high outlier
        ],
        dtype=np.float64,
    )
    surface = build_phone_surface_from_views(
        points_static_m=points,
        view_indices=np.asarray([0, 0, 0, 1, 0, 1, 0]),
        confidence=np.ones(points.shape[0], dtype=np.float32),
        room_footprint=np.ones(shape, dtype=bool),
        x_grid_m=x_grid,
        z_grid_m=z_grid,
        settings=ShadowFillSettings(phone_min_quality=0.2),
    )

    assert surface.good[0, 0]
    assert int(surface.support[0, 0]) == 2
    assert abs(float(surface.height_m[0, 0]) - 0.80) <= 0.02


def test_original_phone_points_follow_admitted_pose_bridge() -> None:
    source_pose = np.eye(4, dtype=np.float64)
    source_pose[0, 3] = 1.0
    admitted_pose = np.eye(4, dtype=np.float64)
    admitted_pose[0, 3] = 2.0
    refinement = np.eye(4, dtype=np.float64)
    refinement[2, 3] = 1.0

    transformed = phone_view_points_to_static_floorplan(
        points_phone_world_m=np.asarray([[1.0, 0.0, 1.0]]),
        source_camera_to_world=source_pose,
        admitted_camera_to_backend_world=admitted_pose,
        phone_to_fixed_refinement=refinement,
        reference_camera_to_world=np.eye(4, dtype=np.float64),
        floor_y_m=0.0,
        scene_to_static_transform=np.eye(4, dtype=np.float64),
    )

    assert np.allclose(transformed, [[2.0, 0.0, 2.0]])


def test_region_completion_flattens_whole_confirmed_components() -> None:
    shape = (6, 12)
    artifact = np.zeros(shape, dtype=bool)
    artifact[0:3, 0:3] = True
    artifact[0:2, 5:8] = True
    artifact[4:6, 9:12] = True
    height = np.full(shape, np.nan, dtype=np.float32)
    good = np.zeros(shape, dtype=bool)
    support = np.zeros(shape, dtype=np.uint16)
    floor_samples = [(0, 0, 0.02), (0, 1, 0.04), (1, 0, 0.06), (1, 1, 0.08)]
    flat_samples = [(0, 5, 0.58), (0, 6, 0.61), (1, 5, 0.62)]
    weak_samples = [(4, 9, 0.03)]
    for row, column, value in floor_samples + flat_samples + weak_samples:
        height[row, column] = value
        good[row, column] = True
        support[row, column] = 3
    phone = PhoneSurface(
        height_m=height,
        quality=np.where(good, 0.9, 0.0).astype(np.float32),
        support=support,
        spread_m=np.where(good, 0.02, np.nan).astype(np.float32),
        good=good,
        eligible_voxel_count=8,
    )

    completion = build_region_surface_completion(
        artifact_region=artifact,
        phone_surface=phone,
    )

    assert completion.floor_component_count == 1
    assert completion.flat_component_count == 1
    assert np.all(completion.floor[0:3, 0:3])
    assert np.allclose(completion.height_m[0:3, 0:3], 0.0)
    assert np.all(completion.flat[0:2, 5:8])
    assert np.allclose(completion.height_m[0:2, 5:8], 0.61, atol=0.02)
    assert not np.any(completion.admitted[4:6, 9:12])


def test_phone_floor_evidence_becomes_a_continuous_bounded_region() -> None:
    shape = (5, 9)
    x_grid, z_grid = floorplan_grid_centers(
        {"min_x": 0.0, "max_x": 0.36, "min_z": 0.0, "max_z": 0.20},
        shape,
    )
    height = np.full(shape, np.nan, dtype=np.float32)
    good = np.zeros(shape, dtype=bool)
    support = np.zeros(shape, dtype=np.uint16)
    for row, column in [(1, 1), (2, 1), (3, 1), (2, 3)]:
        height[row, column] = 0.04
        good[row, column] = True
        support[row, column] = 3
    height[2, 6] = 0.75
    good[2, 6] = True
    support[2, 6] = 3
    phone = PhoneSurface(
        height_m=height,
        quality=np.where(good, 0.9, 0.0).astype(np.float32),
        support=support,
        spread_m=np.where(good, 0.02, np.nan).astype(np.float32),
        good=good,
        eligible_voxel_count=5,
    )

    completion = build_phone_floor_region_completion(
        shadow=np.ones(shape, dtype=bool),
        phone_surface=phone,
        x_grid_m=x_grid,
        z_grid_m=z_grid,
        settings=ShadowFillSettings(phone_floor_region_max_gap_m=0.08),
    )

    assert completion.component_count == 1
    assert completion.admitted[2, 2]
    assert completion.interpolated[2, 2]
    assert not completion.admitted[2, 6]
    assert not completion.admitted[2, 5]
    assert np.all(completion.support[completion.admitted] >= 3)


def test_aligned_floor_visibility_counts_views_without_spatial_growth() -> None:
    shape = (4, 4)
    x_grid, z_grid = floorplan_grid_centers(
        {"min_x": 0.0, "max_x": 0.16, "min_z": 0.0, "max_z": 0.16},
        shape,
    )
    floor_cell = (1, 1)
    weak_cell = (2, 2)
    floor_x = float(x_grid[floor_cell])
    floor_z = float(z_grid[floor_cell])
    weak_x = float(x_grid[weak_cell])
    weak_z = float(z_grid[weak_cell])
    points = np.asarray(
        [
            [floor_x, 0.02, floor_z],
            [floor_x, 0.03, floor_z],
            [floor_x, 0.04, floor_z],
            [floor_x, 0.05, floor_z],  # duplicate pixel from view zero
            [weak_x, 0.02, weak_z],
            [weak_x, 0.03, weak_z],
            [weak_x, 0.70, weak_z],  # raised point is not floor visibility
        ],
        dtype=np.float64,
    )
    visibility = build_phone_floor_visibility_from_views(
        points_static_m=points,
        view_indices=np.asarray([0, 1, 2, 0, 0, 1, 2]),
        confidence=np.ones(points.shape[0], dtype=np.float32),
        room_footprint=np.ones(shape, dtype=bool),
        x_grid_m=x_grid,
        z_grid_m=z_grid,
    )

    assert visibility.visible[floor_cell]
    assert int(visibility.view_support[floor_cell]) == 3
    assert not visibility.visible[weak_cell]
    assert int(visibility.view_support[weak_cell]) == 2
    neighbour = (floor_cell[0], floor_cell[1] + 1)
    assert not visibility.visible[neighbour]
    assert int(visibility.view_support[neighbour]) == 0


def test_complete_floor_is_layered_below_byte_preserved_furniture() -> None:
    shape = (30, 30)
    x_grid, z_grid = floorplan_grid_centers(
        {"min_x": -0.6, "max_x": 0.6, "min_z": 0.0, "max_z": 1.2},
        shape,
    )
    footprint = np.ones(shape, dtype=bool)
    left_couch = np.zeros(shape, dtype=bool)
    left_couch[10:22, 8:12] = True
    right_couch = np.zeros(shape, dtype=bool)
    right_couch[10:22, 19:23] = True
    comb = np.zeros(shape, dtype=bool)
    comb[8, 12:19] = True
    comb[8:21, 12] = True
    comb[8:21, 15] = True
    comb[8:21, 18] = True
    foreground_candidate = left_couch | right_couch | comb
    density = np.zeros(shape, dtype=np.float32)
    density[comb] = 0.001
    density[left_couch | right_couch] = 0.05
    wall_support = np.zeros(shape, dtype=np.float32)
    wall_support[left_couch | right_couch] = 0.2
    phone_floor = comb.copy()
    phone_floor[8, 12:14] = False

    masks = build_floor_underlay_masks(
        room_footprint=footprint,
        foreground_candidate=foreground_candidate,
        floor_shadow=comb,
        radial_artifact_region=comb,
        phone_floor_visibility=phone_floor,
        density=density,
        wall_support=wall_support,
        x_grid_m=x_grid,
        z_grid_m=z_grid,
        settings=ShadowFillSettings(
            floor_artifact_min_phone_cells=8,
            floor_artifact_min_phone_fraction=0.70,
            floor_artifact_min_foreground_cells=4,
            floor_artifact_expansion_m=0.08,
        ),
    )

    assert masks.accepted_component_count == 1
    assert np.all(masks.removed_artifact[comb])
    assert np.all(masks.foreground[left_couch | right_couch])
    assert not np.any(masks.floor & masks.foreground)
    assert np.array_equal(masks.floor | masks.foreground, footprint)

    static_height = np.arange(900, dtype=np.float32).reshape(shape) / 100.0
    static_agl = static_height + 0.25
    static_density = np.full(shape, 0.03, dtype=np.float32)
    static_observed = np.ones(shape, dtype=bool)
    result = apply_floor_underlay(
        static_height=static_height,
        static_height_agl=static_agl,
        static_density=static_density,
        static_observed=static_observed,
        room_footprint=footprint,
        masks=masks,
    )

    couches = left_couch | right_couch
    assert np.array_equal(result.height[couches], static_height[couches])
    assert np.array_equal(result.height_agl[couches], static_agl[couches])
    assert np.array_equal(result.density[couches], static_density[couches])
    assert np.all(result.height_agl[masks.floor] == 0.0)
    assert np.all(result.observed[masks.floor])


def test_floor_underlay_rejects_artifact_without_phone_floor_coverage() -> None:
    shape = (12, 16)
    x_grid, z_grid = floorplan_grid_centers(
        {"min_x": 0.0, "max_x": 0.64, "min_z": 0.0, "max_z": 0.48},
        shape,
    )
    accepted = np.zeros(shape, dtype=bool)
    accepted[2:5, 2:6] = True
    unsupported = np.zeros(shape, dtype=bool)
    unsupported[7:10, 10:14] = True
    artifact = accepted | unsupported
    phone_floor = accepted.copy()
    phone_floor[7, 10:12] = True

    masks = build_floor_underlay_masks(
        room_footprint=np.ones(shape, dtype=bool),
        foreground_candidate=artifact,
        floor_shadow=artifact,
        radial_artifact_region=artifact,
        phone_floor_visibility=phone_floor,
        density=np.full(shape, 0.001, dtype=np.float32),
        wall_support=np.zeros(shape, dtype=np.float32),
        x_grid_m=x_grid,
        z_grid_m=z_grid,
        settings=ShadowFillSettings(
            floor_artifact_min_phone_cells=4,
            floor_artifact_min_phone_fraction=0.70,
            floor_artifact_min_foreground_cells=4,
            floor_artifact_expansion_m=0.0,
            floor_artifact_max_island_cells=0,
        ),
    )

    assert masks.accepted_component_count == 1
    assert np.all(masks.removed_artifact[accepted])
    assert not np.any(masks.removed_artifact[unsupported])
    assert np.all(masks.foreground[unsupported])
