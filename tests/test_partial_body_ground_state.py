"""Focused continuity checks for measured partial-body anatomy."""

import numpy as np

from noesis.telemetry.person_ground_state import (
    HumanGroundConfig, PersonGroundState, mark_world_measurement_unavailable,
    update_human_cv_filter,
)


def test_shoulder_cold_start_needs_current_same_basis_consensus_across_depth_gaps():
    state = PersonGroundState(posture="sitting")
    config = HumanGroundConfig()
    for index, timestamp in enumerate((1.0, 1.2, 1.4)):
        result = update_human_cv_filter(
            state, measurement=np.array([2.0, 0.0, 3.0]), floor_y=0.0,
            now_ts=timestamp, quality="weak", config=config,
            contact_basis="depth:pose_shoulder_support",
        )
        if index < 2:
            assert not state.measurement_accepted
            assert state.world_x is None
            # Cached/missing rows preserve only bounded consensus age. They
            # neither count as another sample nor refresh its timestamp.
            mark_world_measurement_unavailable(
                state, reason="depth_measurement_not_current",
                now_ts=timestamp + .1, config=config,
            )
            assert state.reacquire_count == index + 1
            assert state.reacquire_candidate_ts == timestamp
        else:
            assert state.measurement_accepted
            np.testing.assert_allclose(result, [2.0, 0.0, 3.0])
            assert state.accepted_metric_contact_basis == "depth:pose_shoulder_support"


def test_body_anatomy_switch_does_not_smuggle_a_stationary_dot_shift():
    state = PersonGroundState(posture="sitting")
    config = HumanGroundConfig()
    update_human_cv_filter(
        state, measurement=np.array([0.0, 0.0, 3.0]), floor_y=0.0,
        now_ts=1.0, quality="estimated", config=config,
        contact_basis="depth:pose_torso_support",
    )
    result = update_human_cv_filter(
        state, measurement=np.array([.15, 0.0, 3.0]), floor_y=0.0,
        now_ts=1.2, quality="estimated", config=config,
        contact_basis="depth:pose_shoulder_support",
    )
    assert not state.measurement_accepted
    assert state.measurement_rejection_reason == "body_depth_anatomical_basis_changed"
    np.testing.assert_allclose(result, [0.0, 0.0, 3.0])
    assert state.accepted_metric_contact_basis == "depth:pose_torso_support"
    # A supported relocation uses existing consensus and an explicit segment
    # break, never a continuous phantom walk between different body anchors.
    for timestamp in (1.4, 1.6, 1.8):
        result = update_human_cv_filter(
            state, measurement=np.array([.15, 0.0, 3.0]), floor_y=0.0,
            now_ts=timestamp, quality="estimated", config=config,
            contact_basis="depth:pose_shoulder_support", image_motion_supported=True,
        )
    assert state.measurement_accepted
    assert state.trail_break_required and state.trail_segment_id == 1
    np.testing.assert_allclose(result, [.15, 0.0, 3.0])
