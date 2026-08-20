from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from tools.mapanything_phone_scan.pcf_multiroom_pose_graph import (
    PnPTransformObservation,
    PoseGraphSettings,
    solve_cross_session_pose_graph,
)


def _transform(yaw_deg: float, translation: list[float] | np.ndarray) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]
    )
    transform[:3, 3] = translation
    return transform


def _camera_poses() -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    moving = {
        index: _transform(
            index * 2.0,
            [index * 0.25, 1.5, 0.2 * math.sin(index * 0.5)],
        )
        for index in range(10)
    }
    fixed = {
        index: _transform(
            -10.0 + index * 8.0,
            [-1.0 + index * 0.55, 1.4, 2.0 + index * 0.15],
        )
        for index in range(5)
    }
    return moving, fixed


def _observations(*, contradictory_segments: bool = False) -> list[PnPTransformObservation]:
    generator = np.random.default_rng(3)
    observations: list[PnPTransformObservation] = []
    for order, moving_view in enumerate([1, 2, 7, 8]):
        if contradictory_segments:
            parameters = np.asarray(
                [
                    16.0 if moving_view < 5 else 20.0,
                    -4.5,
                    -0.12,
                    8.35 if moving_view < 5 else 9.45,
                ]
            )
        else:
            parameters = np.asarray(
                [
                    18.0 + 0.35 * math.sin(moving_view * 0.35),
                    -4.5 + 0.08 * math.sin(moving_view * 0.2),
                    -0.12 + 0.02 * math.cos(moving_view * 0.3),
                    8.9 + 0.08 * math.sin(moving_view * 0.25),
                ]
            )
        for fixed_view in [order % 5, (order + 2) % 5]:
            for direction in ["moving_to_fixed", "fixed_to_moving"]:
                noisy = parameters.copy()
                if not contradictory_segments:
                    noisy[0] += generator.normal(0.0, 0.08)
                    noisy[1:] += generator.normal(0.0, 0.012, 3)
                observations.append(
                    PnPTransformObservation(
                        moving_view=moving_view,
                        fixed_view=fixed_view,
                        direction=direction,
                        transform_moving_to_fixed=_transform(noisy[0], noisy[1:]),
                        inlier_count=30,
                        reprojection_median_px=1.2,
                        reprojection_p80_px=2.0,
                    )
                )
    return observations


def _settings() -> PoseGraphSettings:
    return PoseGraphSettings(temporal_segment_gap_views=2)


def test_smooth_two_segment_alignment_is_recovered_and_accepted() -> None:
    moving, fixed = _camera_poses()

    result = solve_cross_session_pose_graph(
        _observations(),
        moving,
        fixed,
        vertical_translation_m=-0.12,
        settings=_settings(),
    )

    assert result.accepted
    assert not result.reason_codes
    assert result.global_transform_moving_to_fixed is not None
    assert set(result.per_view_transforms_moving_to_fixed) == set(moving)
    assert set(result.corrected_moving_camera_poses) == set(moving)
    candidate = result.report["candidate"]
    assert abs(candidate["global_yaw_deg"] - 18.0) < 0.6
    np.testing.assert_allclose(
        candidate["global_translation_m"], [-4.5, -0.12, 8.9], atol=0.16
    )
    assert candidate["locked_vertical_translation_m"] == -0.12
    assert result.global_transform_moving_to_fixed[1, 3] == -0.12
    assert all(
        transform[1, 3] == -0.12
        for transform in result.per_view_transforms_moving_to_fixed.values()
    )
    segment_validation = candidate["leave_one_temporal_overlap_segment_out"]
    assert segment_validation["applicable"]
    assert segment_validation["segments"] == [[1, 2], [7, 8]]
    assert all(fold["passed"] for fold in segment_validation["folds"])


def test_contradictory_early_and_late_segments_fail_closed() -> None:
    moving, fixed = _camera_poses()

    result = solve_cross_session_pose_graph(
        _observations(contradictory_segments=True),
        moving,
        fixed,
        vertical_translation_m=-0.12,
        settings=_settings(),
    )

    assert not result.accepted
    assert "temporal_segment_holdout_failed" in result.reason_codes
    assert result.global_transform_moving_to_fixed is None
    assert result.per_view_transforms_moving_to_fixed == {}
    candidate = result.report["candidate"]
    # The full graph can fit both segments using per-view drift.  The segment
    # holdout is what reveals that one segment does not predict the other.
    assert candidate["training_observation_error"]["translation_p80_m"] < 0.20
    folds = candidate["leave_one_temporal_overlap_segment_out"]["folds"]
    assert not any(fold["passed"] for fold in folds)
    assert max(fold["translation_p80_m"] for fold in folds) > 0.90


def test_unidirectional_support_is_not_promoted() -> None:
    moving, fixed = _camera_poses()
    observations = [
        observation
        for observation in _observations()
        if observation.direction == "moving_to_fixed"
    ]

    result = solve_cross_session_pose_graph(
        observations,
        moving,
        fixed,
        settings=_settings(),
    )

    assert not result.accepted
    assert "insufficient_bidirectional_pairs" in result.reason_codes
    assert result.global_transform_moving_to_fixed is None


def test_scaled_pnp_transform_is_rejected_as_invalid_input() -> None:
    moving, fixed = _camera_poses()
    observations = _observations()
    invalid = observations[0].transform_moving_to_fixed.copy()
    invalid[:3, :3] *= 1.01
    observations[0] = PnPTransformObservation(
        moving_view=observations[0].moving_view,
        fixed_view=observations[0].fixed_view,
        direction=observations[0].direction,
        transform_moving_to_fixed=invalid,
        inlier_count=observations[0].inlier_count,
        reprojection_median_px=observations[0].reprojection_median_px,
        reprojection_p80_px=observations[0].reprojection_p80_px,
    )

    with pytest.raises(ValueError, match="scale or shear"):
        solve_cross_session_pose_graph(
            observations,
            moving,
            fixed,
            settings=_settings(),
        )


def test_canonical_pnp_quality_and_leaveout_defaults_fail_weak_evidence() -> None:
    settings = PoseGraphSettings()
    assert settings.minimum_inliers_per_observation == 20
    assert settings.maximum_observation_reprojection_median_px == 3.0
    assert settings.maximum_observation_reprojection_p80_px == 5.0
    assert settings.maximum_leaveout_session_translation_p80_m == 0.25
    assert settings.maximum_leaveout_session_yaw_p80_deg == 0.5
    moving, fixed = _camera_poses()
    weak = [replace(row, reprojection_p80_px=5.01) for row in _observations()]

    result = solve_cross_session_pose_graph(weak, moving, fixed)

    assert not result.accepted
    rejected = result.report["screening"]["quality_rejected"]
    assert len(rejected) == len(weak)
    assert all(
        "high_reprojection_p80_error" in row["reason_codes"]
        for row in rejected
    )
