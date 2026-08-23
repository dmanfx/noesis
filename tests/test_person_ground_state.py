"""Unit tests for human-realistic person ground-state pathing (Phases 1–6)."""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from noesis.telemetry.person_ground_state import (
    HumanGroundConfig,
    PersonGroundState,
    apply_source_hysteresis,
    assess_lower_body_occlusion,
    begin_source_admission,
    classify_posture,
    commit_path_point,
    complete_source_admission,
    legs_are_bent,
    rdp_simplify,
    resolve_pose_floor_anchor,
    source_score,
    update_human_cv_filter,
    update_motion_mode,
)


def _kpts_standing() -> np.ndarray:
    # COCO-17 style absolute keypoints: [x, y, conf]
    k = np.zeros((17, 3), dtype=np.float32)
    # shoulders / hips / knees / ankles roughly vertical
    k[5] = [100, 100, 0.9]  # L shoulder
    k[6] = [140, 100, 0.9]  # R shoulder
    k[11] = [105, 180, 0.9]  # L hip
    k[12] = [135, 180, 0.9]  # R hip
    k[13] = [105, 240, 0.9]  # L knee
    k[14] = [135, 240, 0.9]  # R knee
    k[15] = [105, 300, 0.9]  # L ankle
    k[16] = [135, 300, 0.9]  # R ankle
    return k


def _kpts_sitting() -> np.ndarray:
    k = np.zeros((17, 3), dtype=np.float32)
    k[5] = [100, 120, 0.9]
    k[6] = [140, 120, 0.9]
    k[11] = [110, 190, 0.9]
    k[12] = [130, 190, 0.9]
    # Bent legs: knees forward, ankles near hips vertically
    k[13] = [150, 210, 0.9]
    k[14] = [160, 210, 0.9]
    k[15] = [155, 215, 0.5]
    k[16] = [165, 215, 0.5]
    return k


def _occluded_pose(level: str) -> np.ndarray:
    keypoints = _kpts_standing()
    if level in ("feet_ankles", "knees", "waist_hips"):
        keypoints[15:17, 2] = 0.0
    if level in ("knees", "waist_hips"):
        keypoints[13:15, 2] = 0.0
    if level == "waist_hips":
        keypoints[11:13, 2] = 0.0
    return keypoints


def _seed_upright_occlusion_state(
    *,
    config: HumanGroundConfig,
) -> PersonGroundState:
    state = PersonGroundState(height_ref_scene=1.8)
    assessment = assess_lower_body_occlusion(
        state,
        kpts_abs=_kpts_standing(),
        bbox=[90.0, 80.0, 60.0, 240.0],
        posture="standing",
        now_ts=1.0,
        config=config,
    )
    assert assessment.active is False
    assert state.last_full_body_ts == pytest.approx(1.0)
    return state


def test_phase3_classify_posture_sitting_and_standing() -> None:
    cfg = HumanGroundConfig()
    assert classify_posture(kpts_abs=_kpts_standing(), bbox=[0, 0, 60, 220], height_ref_scene=1.8, config=cfg) in (
        "standing",
        "unknown",
    )
    assert classify_posture(kpts_abs=_kpts_sitting(), bbox=[0, 0, 90, 110], height_ref_scene=1.8, config=cfg) == "sitting"
    assert classify_posture(kpts_abs=None, bbox=[0, 0, 200, 80], height_ref_scene=None, config=cfg) == "lying"


@pytest.mark.parametrize(
    ("level", "expected_confidence"),
    (
        ("feet_ankles", 0.82),
        ("knees", 0.89),
        ("waist_hips", 0.96),
    ),
)
def test_lower_body_occlusion_covers_ankles_knees_and_hips(
    level: str,
    expected_confidence: float,
) -> None:
    cfg = HumanGroundConfig()
    state = _seed_upright_occlusion_state(config=cfg)

    assessment = assess_lower_body_occlusion(
        state,
        kpts_abs=_occluded_pose(level),
        bbox=[90.0, 80.0, 60.0, 115.0],
        posture="unknown",
        now_ts=1.1,
        config=cfg,
    )

    assert assessment.active is True
    assert assessment.level == level
    assert assessment.confidence == pytest.approx(expected_confidence)
    assert state.as_public_fields()["lower_body_occluded"] is True
    assert state.as_public_fields()["lower_body_occlusion_level"] == level


def test_lower_body_occlusion_detects_collapsed_bbox_despite_hallucinated_ankles() -> None:
    cfg = HumanGroundConfig()
    state = _seed_upright_occlusion_state(config=cfg)

    assessment = assess_lower_body_occlusion(
        state,
        kpts_abs=_kpts_standing(),
        bbox=[90.0, 80.0, 60.0, 100.0],
        posture="standing",
        now_ts=1.1,
        config=cfg,
    )

    assert assessment.active is True
    assert assessment.level == "waist_hips"
    assert "upright_bbox_collapsed" in str(assessment.reason)


def test_lower_body_occlusion_requires_stable_full_body_return_before_release() -> None:
    cfg = HumanGroundConfig(occlusion_exit_frames=3)
    state = _seed_upright_occlusion_state(config=cfg)
    assess_lower_body_occlusion(
        state,
        kpts_abs=_occluded_pose("waist_hips"),
        bbox=[90.0, 80.0, 60.0, 110.0],
        posture="unknown",
        now_ts=1.1,
        config=cfg,
    )

    for frame_index in range(2):
        assessment = assess_lower_body_occlusion(
            state,
            kpts_abs=_kpts_standing(),
            bbox=[90.0, 80.0, 60.0, 240.0],
            posture="standing",
            now_ts=1.2 + frame_index * 0.1,
            config=cfg,
        )
        assert assessment.active is True

    assessment = assess_lower_body_occlusion(
        state,
        kpts_abs=_kpts_standing(),
        bbox=[90.0, 80.0, 60.0, 240.0],
        posture="standing",
        now_ts=1.4,
        config=cfg,
    )
    assert assessment.active is False
    assert assessment.level == "none"


def test_explicit_sitting_pose_blocks_upright_occlusion_mode() -> None:
    cfg = HumanGroundConfig()
    state = _seed_upright_occlusion_state(config=cfg)

    assessment = assess_lower_body_occlusion(
        state,
        kpts_abs=_kpts_sitting(),
        bbox=[90.0, 80.0, 90.0, 120.0],
        posture="sitting",
        now_ts=1.1,
        config=cfg,
    )

    assert assessment.active is False
    assert state.last_non_upright_ts == pytest.approx(1.1)


def test_explicit_sitting_state_blocks_occlusion_without_lower_pose() -> None:
    cfg = HumanGroundConfig()
    state = _seed_upright_occlusion_state(config=cfg)

    assessment = assess_lower_body_occlusion(
        state,
        kpts_abs=None,
        bbox=[90.0, 80.0, 90.0, 120.0],
        posture="sitting",
        now_ts=1.1,
        config=cfg,
    )

    assert assessment.active is False
    assert state.last_non_upright_ts == pytest.approx(1.1)


def test_explicit_sitting_pose_immediately_releases_active_occlusion() -> None:
    cfg = HumanGroundConfig()
    state = _seed_upright_occlusion_state(config=cfg)
    assessment = assess_lower_body_occlusion(
        state,
        kpts_abs=_occluded_pose("waist_hips"),
        bbox=[90.0, 80.0, 60.0, 115.0],
        posture="unknown",
        now_ts=1.1,
        config=cfg,
    )
    assert assessment.active is True

    assessment = assess_lower_body_occlusion(
        state,
        kpts_abs=_kpts_sitting(),
        bbox=[90.0, 80.0, 90.0, 120.0],
        posture="sitting",
        now_ts=1.2,
        config=cfg,
    )

    assert assessment.active is False
    assert assessment.level == "none"


def test_height_lock_prevents_bbox_only_torso_from_becoming_sitting() -> None:
    cfg = HumanGroundConfig()
    assert (
        classify_posture(
            kpts_abs=None,
            bbox=[0.0, 0.0, 100.0, 110.0],
            height_ref_scene=1.8,
            config=cfg,
        )
        == "unknown"
    )


def test_phase2_bent_leg_without_ankles_has_no_floor_contact() -> None:
    cfg = HumanGroundConfig()
    assert legs_are_bent(_kpts_sitting(), config=cfg) is True
    # Zero ankle conf so only leg-extension path would fire for standing logic
    k = _kpts_sitting().copy()
    k[15, 2] = 0.0
    k[16, 2] = 0.0
    assert (
        classify_posture(
            kpts_abs=k,
            bbox=[0.0, 0.0, 90.0, 110.0],
            height_ref_scene=1.8,
            config=cfg,
        )
        == "sitting"
    )
    cand = resolve_pose_floor_anchor(k, posture="sitting", config=cfg)
    assert cand is None


def test_phase3_sitting_uses_observed_ankles_not_hips() -> None:
    cfg = HumanGroundConfig()
    cand = resolve_pose_floor_anchor(_kpts_sitting(), posture="sitting", config=cfg)
    assert cand is not None
    assert cand.source == "pose_ankle_floor"
    assert cand.u == pytest.approx(160.0)
    assert cand.v == pytest.approx(215.0)
    assert cand.height_lock_eligible is False


def test_phase3_standing_uses_ankle_mid() -> None:
    cfg = HumanGroundConfig()
    cand = resolve_pose_floor_anchor(_kpts_standing(), posture="standing", config=cfg)
    assert cand is not None
    assert cand.source == "pose_ankle_floor"
    assert cand.u == pytest.approx(120.0)
    assert cand.v == pytest.approx(300.0)


def test_phase2_source_hysteresis_sticky() -> None:
    cfg = HumanGroundConfig(
        source_hold_frames=5,
        source_replacement_frames=5,
        source_switch_min_score_gain=0.20,
    )
    state = PersonGroundState()
    src, switched = apply_source_hysteresis(
        state, candidate_source="pose_floor_only", candidate_score=0.95, config=cfg
    )
    assert src == "pose_floor_only" and switched is False
    # Weak alternate source within hold window should not switch.
    for _ in range(3):
        src, switched = apply_source_hysteresis(
            state, candidate_source="pose_leg_floor", candidate_score=0.55, config=cfg
        )
        assert src == "pose_floor_only"
        assert switched is False


def test_phase2_source_hysteresis_admits_sustained_lower_score_replacement() -> None:
    cfg = HumanGroundConfig(
        source_hold_frames=10,
        source_replacement_frames=3,
        source_switch_min_score_gain=0.20,
    )
    state = PersonGroundState(
        sticky_source="pose_floor_only",
        sticky_source_frames=20,
        sticky_source_score=1.0,
    )

    for _ in range(2):
        source, switched = apply_source_hysteresis(
            state,
            candidate_source="person_anchor_floor_only",
            candidate_score=0.78,
            config=cfg,
        )
        assert source == "pose_floor_only"
        assert switched is False

    source, switched = apply_source_hysteresis(
        state,
        candidate_source="person_anchor_floor_only",
        candidate_score=0.78,
        config=cfg,
    )
    assert source == "person_anchor_floor_only"
    assert switched is True
    assert state.source_candidate is None
    assert state.source_candidate_frames == 0


def test_phase2_source_hysteresis_requires_consecutive_replacement_samples() -> None:
    cfg = HumanGroundConfig(
        source_hold_frames=10,
        source_replacement_frames=3,
        source_switch_min_score_gain=0.20,
    )
    state = PersonGroundState(
        sticky_source="pose_floor_only",
        sticky_source_frames=20,
        sticky_source_score=1.0,
    )

    for candidate in (
        "person_anchor_floor_only",
        "pose_depth_only",
        "person_anchor_floor_only",
        "pose_depth_only",
    ):
        source, switched = apply_source_hysteresis(
            state,
            candidate_source=candidate,
            candidate_score=0.70,
            config=cfg,
        )
        assert source == "pose_floor_only"
        assert switched is False

    assert state.source_candidate_frames == 1


def test_source_admission_rolls_back_physically_rejected_switch() -> None:
    cfg = HumanGroundConfig(
        source_hold_frames=1,
        source_replacement_frames=1,
        source_switch_min_score_gain=0.10,
    )
    state = PersonGroundState(
        sticky_source="pose_floor_only",
        sticky_source_frames=6,
        sticky_source_score=0.60,
        source_switch_count=2,
    )
    previous = (
        state.sticky_source,
        state.sticky_source_frames,
        state.sticky_source_score,
        state.source_switch_count,
    )

    assert begin_source_admission(
        state,
        candidate_source="pose_depth_fused",
        candidate_score=0.95,
        config=cfg,
    ) is True
    assert state.sticky_source == "pose_depth_fused"
    complete_source_admission(state, measurement_accepted=False)

    assert (
        state.sticky_source,
        state.sticky_source_frames,
        state.sticky_source_score,
        state.source_switch_count,
    ) == previous
    assert state.pending_source_previous is None


def test_source_admission_commits_physically_accepted_switch() -> None:
    cfg = HumanGroundConfig(
        source_hold_frames=1,
        source_replacement_frames=1,
        source_switch_min_score_gain=0.10,
    )
    state = PersonGroundState(
        sticky_source="pose_floor_only",
        sticky_source_frames=6,
        sticky_source_score=0.60,
        source_switch_count=2,
    )

    assert begin_source_admission(
        state,
        candidate_source="pose_depth_fused",
        candidate_score=0.95,
        config=cfg,
    ) is True
    complete_source_admission(state, measurement_accepted=True)

    assert state.sticky_source == "pose_depth_fused"
    assert state.sticky_source_frames == 1
    assert state.sticky_source_score == pytest.approx(0.95)
    assert state.source_switch_count == 3
    assert state.pending_source_previous is None


def test_authoritative_occlusion_source_rolls_back_if_geometry_is_rejected() -> None:
    cfg = HumanGroundConfig()
    state = PersonGroundState(
        sticky_source="pose_depth_fused",
        sticky_source_frames=8,
        sticky_source_score=1.05,
        source_switch_count=4,
    )

    assert begin_source_admission(
        state,
        candidate_source="gravity_drop",
        candidate_score=0.35,
        config=cfg,
        authoritative=True,
    )
    assert state.sticky_source == "gravity_drop"
    complete_source_admission(state, measurement_accepted=False)

    assert state.sticky_source == "pose_depth_fused"
    assert state.sticky_source_frames == 8
    assert state.sticky_source_score == pytest.approx(1.05)
    assert state.source_switch_count == 4


def test_phase1_stationary_lock_and_no_trail_append() -> None:
    cfg = HumanGroundConfig(static_enter_s=0.2, static_speed_mps=0.15, static_px_threshold=5.0)
    state = PersonGroundState(world_x=1.0, world_z=2.0, vel_world_x=0.0, vel_world_z=0.0, filtered_ts=1.0)
    # Feed several static image foot observations.
    t = 1.0
    for i in range(8):
        t = 1.0 + i * 0.1
        state.world_x = 1.0 + (0.001 if i % 2 == 0 else -0.001)
        state.world_z = 2.0
        state.vel_world_x = 0.01
        state.vel_world_z = 0.0
        update_motion_mode(state, now_ts=t, image_foot_uv=(100.0 + 0.2 * (i % 2), 200.0), config=cfg)
    assert state.motion_mode in ("idle", "sit", "lie")
    assert state.trail_append_allowed is False
    assert state.locked_world is not None


def test_phase1_idle_deadzone_absorbs_jitter() -> None:
    cfg = HumanGroundConfig(idle_deadzone_m=0.30)
    state = PersonGroundState(
        world_x=0.0,
        world_z=0.0,
        vel_world_x=0.0,
        vel_world_z=0.0,
        filtered_ts=-1.0,
        motion_mode="idle",
        locked_world=(0.0, 0.0),
        posture="sitting",
    )
    # Seed filter at t=0
    update_human_cv_filter(
        state,
        measurement=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=0.0,
        quality="good",
        config=cfg,
    )
    state.motion_mode = "idle"
    state.locked_world = (0.0, 0.0)
    out = update_human_cv_filter(
        state,
        measurement=np.array([0.12, 0.0, 0.08], dtype=np.float64),
        floor_y=0.0,
        now_ts=0.1,
        quality="estimated",
        config=cfg,
    )
    assert out[0] == pytest.approx(0.0, abs=1e-6)
    assert out[2] == pytest.approx(0.0, abs=1e-6)


def test_phase4_cv_filter_rejects_impossible_innovation_without_moving() -> None:
    cfg = HumanGroundConfig(max_speed_mps=2.0, max_jump_m=0.5, alpha_good=0.5, process_noise_walk=0.8, meas_noise_good=0.05)
    state = PersonGroundState()
    p0 = update_human_cv_filter(
        state,
        measurement=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=0.0,
        quality="good",
        config=cfg,
    )
    assert np.allclose(p0, [0.0, 0.0, 0.0])
    state.motion_mode = "walk"
    p1 = update_human_cv_filter(
        state,
        measurement=np.array([10.0, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=0.1,
        quality="good",
        config=cfg,
    )
    assert np.allclose(p1, [0.0, 0.0, 0.0])
    assert state.measurement_accepted is False
    assert state.measurement_rejection_reason == "physical_innovation_exceeded"
    assert state.measurement_innovation_m == pytest.approx(10.0)
    assert state.measurement_allowed_m == pytest.approx(0.7)
    assert state.reacquire_count == 1
    assert state.trail_append_allowed is False


def test_phase4_gap_does_not_raw_reset_existing_lifecycle() -> None:
    cfg = HumanGroundConfig(
        max_speed_mps=2.0,
        max_jump_m=0.5,
        reset_after_s=1.25,
        reacquire_samples=3,
    )
    state = PersonGroundState()
    update_human_cv_filter(
        state,
        measurement=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=0.0,
        quality="good",
        config=cfg,
    )
    out = update_human_cv_filter(
        state,
        measurement=np.array([7.0, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=1.26,
        quality="good",
        config=cfg,
    )
    assert np.allclose(out, [0.0, 0.0, 0.0])
    assert state.measurement_accepted is False
    assert state.world_x == pytest.approx(0.0)
    assert state.world_z == pytest.approx(0.0)


def test_phase4_reacquisition_requires_consistent_samples_and_breaks_trail() -> None:
    cfg = HumanGroundConfig(
        max_speed_mps=2.0,
        max_jump_m=0.5,
        reset_after_s=1.25,
        reacquire_samples=3,
        reacquire_max_gap_s=0.5,
    )
    state = PersonGroundState()
    update_human_cv_filter(
        state,
        measurement=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=0.0,
        quality="good",
        config=cfg,
    )
    for ts, x, expected_count in ((2.0, 7.0, 1), (2.1, 7.1, 2)):
        out = update_human_cv_filter(
            state,
            measurement=np.array([x, 0.0, 0.0], dtype=np.float64),
            floor_y=0.0,
            now_ts=ts,
            quality="good",
            config=cfg,
        )
        assert np.allclose(out, [0.0, 0.0, 0.0])
        assert state.measurement_accepted is False
        assert state.reacquire_count == expected_count

    out = update_human_cv_filter(
        state,
        measurement=np.array([7.2, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=2.2,
        quality="good",
        config=cfg,
    )
    assert np.allclose(out, [7.2, 0.0, 0.0])
    assert state.measurement_accepted is True
    assert state.reacquired is True
    assert state.trail_break_required is True
    assert state.trail_segment_id == 1
    assert state.reacquire_count == 0

    update_human_cv_filter(
        state,
        measurement=np.array([7.3, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=2.3,
        quality="good",
        config=cfg,
    )
    assert state.reacquired is False
    assert state.trail_break_required is False
    assert state.trail_segment_id == 1
    assert state.as_public_fields()["trail_segment_id"] == 1


def test_phase4_inconsistent_reacquisition_restarts_confirmation() -> None:
    cfg = HumanGroundConfig(
        max_speed_mps=1.0,
        max_jump_m=0.25,
        reacquire_samples=3,
        reacquire_max_gap_s=0.5,
    )
    state = PersonGroundState()
    update_human_cv_filter(
        state,
        measurement=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=0.0,
        quality="good",
        config=cfg,
    )
    for ts, x in ((0.1, 5.0), (0.2, -5.0)):
        update_human_cv_filter(
            state,
            measurement=np.array([x, 0.0, 0.0], dtype=np.float64),
            floor_y=0.0,
            now_ts=ts,
            quality="good",
            config=cfg,
        )
    assert state.measurement_accepted is False
    assert state.reacquire_count == 1
    assert state.world_x == pytest.approx(0.0)


def test_phase6_rdp_and_commit_path_point() -> None:
    cfg = HumanGroundConfig(path_min_step_m=0.05, path_simplify_epsilon_m=0.04, path_max_points=64)
    pts: deque = deque()
    # Straight line with tiny noise — RDP should collapse
    for i in range(20):
        noise = 0.01 if i % 2 == 0 else -0.01
        commit_path_point(
            pts,
            ts=float(i) * 0.1,
            x=float(i) * 0.2,
            z=noise,
            config=cfg,
            append_allowed=True,
        )
    simplified = rdp_simplify(list(pts), epsilon=0.05)
    assert len(simplified) <= len(pts)
    assert len(simplified) < 12

    # Idle: no append
    before = len(pts)
    commit_path_point(pts, ts=10.0, x=100.0, z=100.0, config=cfg, append_allowed=False)
    assert len(pts) == before


def test_source_score_never_prefers_hip_when_sitting() -> None:
    ankle = source_score("pose_ankle_floor", quality="good", posture="sitting")
    hip = source_score("pose_hip_floor", quality="good", posture="sitting")
    assert ankle > hip
