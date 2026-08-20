from __future__ import annotations

import math

import numpy as np

from tools.mapanything_phone_scan import pcf_local_overlap_refinement as subject


def _synthetic_corner(seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    floor = np.column_stack(
        [
            rng.uniform(0.0, 4.0, 2_000),
            np.zeros(2_000),
            rng.uniform(0.0, 3.0, 2_000),
        ]
    )
    wall_x = np.column_stack(
        [
            np.zeros(1_000),
            rng.uniform(0.0, 2.4, 1_000),
            rng.uniform(0.0, 3.0, 1_000),
        ]
    )
    wall_z = np.column_stack(
        [
            rng.uniform(0.0, 4.0, 1_000),
            rng.uniform(0.0, 2.4, 1_000),
            np.zeros(1_000),
        ]
    )
    return np.concatenate([floor, wall_x, wall_z])


def test_overlap_refinement_recovers_planar_rigid_transform_and_floor() -> None:
    moving = _synthetic_corner()
    truth = subject.yaw_transform(
        np.asarray([math.radians(5.0), 1.0, 0.0, 2.0])
    )
    fixed = subject.transform_points(moving, truth)
    initial = subject.yaw_transform(
        np.asarray([math.radians(5.45), 1.07, 0.18, 1.93])
    )
    solved, report = subject.refine_visual_overlap(
        moving,
        fixed,
        fixed[::20],
        initial,
        settings=subject.RefinementSettings(
            voxel_size_m=0.06,
            normal_radius_m=0.18,
            minimum_correspondences=100,
        ),
    )
    parameters = subject.transform_parameters(solved)
    assert abs(math.degrees(parameters[0]) - 5.0) < 0.15
    assert np.linalg.norm(parameters[[1, 3]] - [1.0, 2.0]) < 0.03
    assert abs(parameters[2]) < 0.005
    assert report["whole_room_icp_used"] is False
    assert report["floor_constraint"]["locked"] is True
    assert report["final"]["euclidean_p80_m"] < 0.04
