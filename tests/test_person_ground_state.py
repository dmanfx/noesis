"""Unit tests for human-realistic person ground-state pathing (Phases 1–6)."""

from __future__ import annotations

import math
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from noesis.telemetry.person_ground_state import (
    HumanGroundConfig,
    PersonGroundState,
    admit_human_ground_output,
    align_inferred_ground_observation,
    apply_source_hysteresis,
    advance_human_cv_prediction,
    assess_lower_body_occlusion,
    begin_source_admission,
    classify_posture,
    clear_inferred_ground_continuity_anchor,
    commit_path_point,
    complete_source_admission,
    legs_are_bent,
    mark_image_motion_observation_unavailable,
    mark_world_measurement_unavailable,
    observe_bbox_stationarity,
    observe_coherent_image_motion,
    integrate_projective_ground_observation,
    rdp_simplify,
    resolve_pose_floor_anchor,
    source_score,
    update_human_cv_filter,
    update_motion_mode,
    world_frame_matches_calibration,
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


def test_world_frame_match_rejects_missing_observed_transform() -> None:
    transform = "a" * 64
    calibration = SimpleNamespace(
        world_frame_id="backend_world_m",
        world_frame_revision="world-r1",
        frame_transform_sha256=transform,
    )
    track = {
        "world_frame": "backend_world_m",
        "world_frame_revision": "world-r1",
    }

    assert world_frame_matches_calibration(track, calibration) is False


def test_world_frame_match_rejects_unexpected_observed_transform() -> None:
    transform = "a" * 64
    calibration = SimpleNamespace(
        world_frame_id="backend_world_m",
        world_frame_revision="world-r1",
    )
    track = {
        "world_frame": "backend_world_m",
        "world_frame_revision": "world-r1",
        "world_transform_sha256": transform,
    }

    assert world_frame_matches_calibration(track, calibration) is False


def test_world_frame_match_accepts_matching_transform() -> None:
    transform = "a" * 64
    calibration = SimpleNamespace(
        world_frame_id="backend_world_m",
        world_frame_revision="world-r1",
        frame_transform_sha256=transform,
    )
    track = {
        "world_frame": "backend_world_m",
        "world_frame_revision": "world-r1",
        "world_transform_sha256": transform,
    }

    assert world_frame_matches_calibration(track, calibration) is True


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
    assert assessment.level == "feet_ankles"
    assert "pose_contact_outside_detector_silhouette" in str(assessment.reason)


def test_uniformly_scaled_complete_pose_does_not_masquerade_as_occlusion() -> None:
    cfg = HumanGroundConfig()
    state = _seed_upright_occlusion_state(config=cfg)
    scaled_pose = _kpts_standing()
    visible = scaled_pose[:, 2] > 0.0
    scaled_pose[visible, 0] = 120.0 + 0.5 * (scaled_pose[visible, 0] - 120.0)
    scaled_pose[visible, 1] = 80.0 + 0.5 * (scaled_pose[visible, 1] - 80.0)

    assessment = assess_lower_body_occlusion(
        state,
        kpts_abs=scaled_pose,
        bbox=[105.0, 80.0, 30.0, 120.0],
        posture="standing",
        now_ts=1.1,
        config=cfg,
    )

    assert assessment.active is False
    assert assessment.level == "none"


def test_missing_lower_pose_without_silhouette_collapse_is_not_occlusion() -> None:
    cfg = HumanGroundConfig()
    state = _seed_upright_occlusion_state(config=cfg)

    assessment = assess_lower_body_occlusion(
        state,
        kpts_abs=_occluded_pose("feet_ankles"),
        bbox=[90.0, 82.0, 60.0, 235.0],
        posture="standing",
        now_ts=1.1,
        config=cfg,
    )

    assert assessment.active is False
    assert assessment.level == "none"
    assert state.lower_body_occluded is False


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


def test_pose_dropout_cannot_masquerade_as_full_body_occlusion_return() -> None:
    cfg = HumanGroundConfig(occlusion_exit_frames=3)
    state = _seed_upright_occlusion_state(config=cfg)
    assessment = assess_lower_body_occlusion(
        state,
        kpts_abs=_occluded_pose("waist_hips"),
        bbox=[90.0, 80.0, 60.0, 110.0],
        posture="unknown",
        now_ts=1.1,
        config=cfg,
    )
    assert assessment.active is True
    last_full_body_ts = state.last_full_body_ts

    for now_ts in (1.2, 1.3, 1.4, 1.5):
        assessment = assess_lower_body_occlusion(
            state,
            kpts_abs=None,
            bbox=[90.0, 80.0, 60.0, 240.0],
            posture="standing",
            now_ts=now_ts,
            config=cfg,
        )
        assert assessment.active is True
        assert state.lower_body_clear_frames == 0
        assert state.lower_body_clear_since_ts == pytest.approx(-1.0)

    assert state.last_full_body_ts == pytest.approx(last_full_body_ts)


def test_hidden_callback_count_cannot_end_occlusion_before_elapsed_exit_time() -> None:
    cfg = HumanGroundConfig(occlusion_exit_frames=3, occlusion_exit_s=0.20)
    state = _seed_upright_occlusion_state(config=cfg)
    assessment = assess_lower_body_occlusion(
        state,
        kpts_abs=_occluded_pose("waist_hips"),
        bbox=[90.0, 80.0, 60.0, 110.0],
        posture="unknown",
        now_ts=1.1,
        config=cfg,
    )
    assert assessment.active is True

    # Three 30 fps processing callbacks are not three queue-visible display
    # intervals.  A hidden callback burst therefore cannot consume the entire
    # occlusion-exit budget before the next published tracking cohort.
    for now_ts in (1.133, 1.166, 1.199):
        assessment = assess_lower_body_occlusion(
            state,
            kpts_abs=_kpts_standing(),
            bbox=[90.0, 80.0, 60.0, 240.0],
            posture="standing",
            now_ts=now_ts,
            config=cfg,
        )
        assert assessment.active is True

    assessment = assess_lower_body_occlusion(
        state,
        kpts_abs=_kpts_standing(),
        bbox=[90.0, 80.0, 60.0, 240.0],
        posture="standing",
        now_ts=1.333,
        config=cfg,
    )
    assert assessment.active is False


def test_occlusion_exit_uses_media_evidence_time_not_slow_processing_clock() -> None:
    cfg = HumanGroundConfig(occlusion_exit_frames=3, occlusion_exit_s=0.20)
    state = _seed_upright_occlusion_state(config=cfg)
    assessment = assess_lower_body_occlusion(
        state,
        kpts_abs=_occluded_pose("waist_hips"),
        bbox=[90.0, 80.0, 60.0, 110.0],
        posture="unknown",
        now_ts=1.1,
        evidence_ts=1.1,
        config=cfg,
    )
    assert assessment.active is True

    # Inference can take substantially longer than the media interval.  The
    # physical evidence window follows media PTS, not callback wall time.
    for callback_ts, media_ts in (
        (1.5, 1.133),
        (1.9, 1.166),
        (2.3, 1.199),
    ):
        assessment = assess_lower_body_occlusion(
            state,
            kpts_abs=_kpts_standing(),
            bbox=[90.0, 80.0, 60.0, 240.0],
            posture="standing",
            now_ts=callback_ts,
            evidence_ts=media_ts,
            config=cfg,
        )
        assert assessment.active is True

    assessment = assess_lower_body_occlusion(
        state,
        kpts_abs=_kpts_standing(),
        bbox=[90.0, 80.0, 60.0, 240.0],
        posture="standing",
        now_ts=2.7,
        evidence_ts=1.333,
        config=cfg,
    )
    assert assessment.active is False


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


def test_one_moving_sit_hallucination_does_not_end_standing_occlusion() -> None:
    cfg = HumanGroundConfig(occlusion_exit_frames=3, static_exit_frames=3)
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
    state.motion_mode = "walk"
    state.image_motion_supported = True
    state.image_motion_streak = 3

    assessment = assess_lower_body_occlusion(
        state,
        kpts_abs=_kpts_sitting(),
        bbox=[90.0, 80.0, 90.0, 120.0],
        posture="sitting",
        now_ts=1.2,
        config=cfg,
    )
    assert assessment.active is True
    assert state.lower_body_non_upright_frames == 1
    assert state.lower_body_clear_frames == 0

    # The next standing frames resume ordinary clear-evidence hysteresis;
    # neither a one-frame pose hallucination nor one clear row can erase the
    # established occlusion episode.
    for now_ts in (1.3, 1.4):
        assessment = assess_lower_body_occlusion(
            state,
            kpts_abs=_kpts_standing(),
            bbox=[90.0, 80.0, 60.0, 240.0],
            posture="standing",
            now_ts=now_ts,
            config=cfg,
        )
        assert assessment.active is True
    assessment = assess_lower_body_occlusion(
        state,
        kpts_abs=_kpts_standing(),
        bbox=[90.0, 80.0, 60.0, 240.0],
        posture="standing",
        now_ts=1.5,
        config=cfg,
    )
    assert assessment.active is False


def test_persistent_moving_non_upright_evidence_ends_occlusion() -> None:
    cfg = HumanGroundConfig(occlusion_exit_frames=3, static_exit_frames=3)
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
    state.motion_mode = "walk"
    state.image_motion_supported = True
    state.image_motion_streak = 3

    for index in range(2):
        assessment = assess_lower_body_occlusion(
            state,
            kpts_abs=_kpts_sitting(),
            bbox=[90.0, 80.0, 90.0, 120.0],
            posture="sitting",
            now_ts=1.2 + index * 0.1,
            config=cfg,
        )
        assert assessment.active is True
    assessment = assess_lower_body_occlusion(
        state,
        kpts_abs=_kpts_sitting(),
        bbox=[90.0, 80.0, 90.0, 120.0],
        posture="sitting",
        now_ts=1.4,
        config=cfg,
    )
    assert assessment.active is False
    assert state.lower_body_non_upright_frames == 0


def test_pending_non_upright_confirmation_survives_image_motion_dropout() -> None:
    cfg = HumanGroundConfig(
        occlusion_exit_frames=3,
        occlusion_exit_s=0.20,
        static_exit_frames=3,
    )
    state = _seed_upright_occlusion_state(config=cfg)
    assessment = assess_lower_body_occlusion(
        state,
        kpts_abs=_occluded_pose("waist_hips"),
        bbox=[90.0, 80.0, 60.0, 115.0],
        posture="unknown",
        now_ts=1.1,
        evidence_ts=1.1,
        config=cfg,
    )
    assert assessment.active is True
    state.motion_mode = "walk"
    state.image_motion_supported = True
    state.image_motion_streak = 3

    assessment = assess_lower_body_occlusion(
        state,
        kpts_abs=_kpts_sitting(),
        bbox=[90.0, 80.0, 90.0, 120.0],
        posture="sitting",
        now_ts=1.2,
        evidence_ts=1.2,
        config=cfg,
    )
    assert assessment.active is True

    # Once a moving track has entered bounded transition confirmation, one
    # failed image-motion sample cannot turn the next pose hallucination into
    # an immediate state change. Persistent seated evidence still clears on
    # the same frame-and-media-time budget.
    state.image_motion_supported = False
    state.image_motion_streak = 0
    for evidence_ts in (1.233, 1.266):
        assessment = assess_lower_body_occlusion(
            state,
            kpts_abs=_kpts_sitting(),
            bbox=[90.0, 80.0, 90.0, 120.0],
            posture="sitting",
            now_ts=evidence_ts,
            evidence_ts=evidence_ts,
            config=cfg,
        )
        assert assessment.active is True

    assessment = assess_lower_body_occlusion(
        state,
        kpts_abs=_kpts_sitting(),
        bbox=[90.0, 80.0, 90.0, 120.0],
        posture="sitting",
        now_ts=1.4,
        evidence_ts=1.4,
        config=cfg,
    )
    assert assessment.active is False


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
    assert state.reacquire_count == 0
    assert state.trail_append_allowed is False


@pytest.mark.parametrize(
    ("measurement", "floor_y", "now_ts"),
    (
        (np.array([np.nan, 0.0, 2.0]), 0.0, 0.0),
        (np.array([1.0, 0.0, np.inf]), 0.0, 0.0),
        (np.array([1.0, 0.0, 2.0]), np.nan, 0.0),
        (np.array([1.0, 0.0, 2.0]), 0.0, np.inf),
    ),
)
def test_first_measurement_rejects_nonfinite_inputs_without_seeding(
    measurement: np.ndarray,
    floor_y: float,
    now_ts: float,
) -> None:
    state = PersonGroundState()
    output = update_human_cv_filter(
        state,
        measurement=measurement,
        floor_y=floor_y,
        now_ts=now_ts,
        quality="good",
        config=HumanGroundConfig(),
    )

    assert np.all(np.isnan(output))
    assert state.world_x is None
    assert state.world_z is None
    assert state.filtered_ts == pytest.approx(-1.0)
    assert state.measurement_accepted is False
    assert state.measurement_rejection_reason == "nonfinite_measurement"
    assert state.trail_append_allowed is False


def test_weak_first_measurement_requires_consistent_contact_consensus() -> None:
    config = HumanGroundConfig(reacquire_samples=3, reacquire_max_gap_s=0.5)
    state = PersonGroundState()

    for index, x in enumerate((1.00, 1.02), start=1):
        output = update_human_cv_filter(
            state,
            measurement=np.array([x, 0.0, 2.0], dtype=np.float64),
            floor_y=0.0,
            now_ts=index / 30.0,
            quality="weak",
            config=config,
            contact_basis="bbox_bottom",
        )
        assert output == pytest.approx([x, 0.0, 2.0])
        assert state.measurement_accepted is False
        assert state.measurement_rejection_reason == "weak_measurement_bootstrap_pending"
        assert state.world_x is None
        assert state.world_z is None

    output = update_human_cv_filter(
        state,
        measurement=np.array([1.03, 0.0, 2.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=3.0 / 30.0,
        quality="weak",
        config=config,
        contact_basis="bbox_bottom",
    )

    assert output == pytest.approx([1.03, 0.0, 2.0])
    assert state.measurement_accepted is True
    assert state.world_x == pytest.approx(1.03)
    assert state.world_z == pytest.approx(2.0)
    assert state.reacquire_count == 0


def test_cold_upright_body_bootstrap_requires_verified_current_sample() -> None:
    config = HumanGroundConfig(reacquire_samples=3, reacquire_max_gap_s=0.5)
    state = PersonGroundState()

    for index, x in enumerate((1.00, 1.03, 1.06), start=1):
        output = update_human_cv_filter(
            state,
            measurement=np.array([x, 0.0, 2.0], dtype=np.float64),
            floor_y=0.0,
            now_ts=index / 10.0,
            quality="weak",
            config=config,
            contact_basis="pose:upright_body_planes",
            verified_reacquire_support=False,
        )
        assert output == pytest.approx([x, 0.0, 2.0])
        assert state.measurement_accepted is False
        assert state.measurement_rejection_reason == "weak_measurement_bootstrap_pending"
        assert state.world_x is None
        assert state.reacquire_count == index

    output = update_human_cv_filter(
        state,
        measurement=np.array([1.09, 0.0, 2.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=0.4,
        quality="weak",
        config=config,
        contact_basis="pose:upright_body_planes",
        verified_reacquire_support=True,
    )

    assert output == pytest.approx([1.09, 0.0, 2.0])
    assert state.measurement_accepted is True
    assert state.world_x == pytest.approx(1.09)
    assert state.world_z == pytest.approx(2.0)


def test_two_four_plane_body_samples_seed_across_bounded_unavailable_rows() -> None:
    config = HumanGroundConfig(reacquire_samples=3, reacquire_max_gap_s=0.75)
    state = PersonGroundState()

    first = update_human_cv_filter(
        state,
        measurement=np.array([11.56, 0.0, 7.32], dtype=np.float64),
        floor_y=0.0,
        now_ts=51.466,
        quality="weak",
        config=config,
        contact_basis="pose:upright_body_planes:four_plane",
    )

    assert first == pytest.approx([11.56, 0.0, 7.32])
    assert state.measurement_accepted is False
    assert state.world_x is None
    assert state.reacquire_candidate_basis == "pose:upright_body_planes"
    assert (
        state.reacquire_candidate_exact_basis
        == "pose:upright_body_planes:four_plane"
    )

    for now_ts in (51.566, 51.666, 51.866):
        mark_world_measurement_unavailable(
            state,
            reason="upright_body_temporarily_unavailable",
            now_ts=now_ts,
            config=config,
        )
        assert state.reacquire_count == 1
        assert state.reacquire_candidate_ts == pytest.approx(51.466)

    second = update_human_cv_filter(
        state,
        measurement=np.array([11.61, 0.0, 7.61], dtype=np.float64),
        floor_y=0.0,
        now_ts=52.066,
        quality="weak",
        config=config,
        contact_basis="pose:upright_body_planes:four_plane",
    )

    assert second == pytest.approx([11.61, 0.0, 7.61])
    assert state.measurement_accepted is True
    assert state.world_x == pytest.approx(11.61)
    assert state.world_z == pytest.approx(7.61)


def test_four_plane_body_consensus_expires_by_elapsed_time() -> None:
    config = HumanGroundConfig(reacquire_samples=3, reacquire_max_gap_s=0.75)
    state = PersonGroundState()

    update_human_cv_filter(
        state,
        measurement=np.array([1.0, 0.0, 2.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=10.0,
        quality="weak",
        config=config,
        contact_basis="pose:upright_body_planes:four_plane",
    )
    mark_world_measurement_unavailable(
        state,
        reason="upright_body_temporarily_unavailable",
        now_ts=10.76,
        config=config,
    )

    assert state.reacquire_count == 0
    assert state.reacquire_candidate_basis is None

    output = update_human_cv_filter(
        state,
        measurement=np.array([1.1, 0.0, 2.1], dtype=np.float64),
        floor_y=0.0,
        now_ts=10.80,
        quality="weak",
        config=config,
        contact_basis="pose:upright_body_planes:four_plane",
    )

    assert output == pytest.approx([1.1, 0.0, 2.1])
    assert state.measurement_accepted is False
    assert state.world_x is None
    assert state.reacquire_count == 1


def test_moving_torso_corroboration_allows_two_sample_pose_floor_bootstrap() -> None:
    config = HumanGroundConfig(
        reacquire_samples=3,
        reacquire_max_gap_s=0.5,
        static_px_threshold=3.0,
        static_exit_frames=3,
    )
    state = PersonGroundState()

    support = []
    for frame_id, observation_ts, px in (
        (100, 1.0, 100.0),
        (103, 1.1, 105.0),
        (106, 1.2, 110.0),
    ):
        state.bbox_stationary_streak = 3
        state.bbox_stationary_supported = True
        support.append(
            observe_coherent_image_motion(
                state,
                frame_id=frame_id,
                observation_ts=observation_ts,
                image_foot_uv=(px, 160.0),
                bbox=(px - 25.0, 100.0, 50.0, 100.0),
                contact_basis="pose:torso_motion",
                config=config,
            )
        )
    assert support == [False, False, True]
    assert state.bbox_stationary_streak == 0
    assert state.bbox_stationary_supported is False

    update_human_cv_filter(
        state,
        measurement=np.array([1.00, 0.0, 2.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=1.2,
        quality="weak",
        config=config,
        contact_basis="pose_single_ankle_floor",
        image_motion_supported=True,
    )
    assert state.measurement_accepted is False
    assert state.reacquire_count == 1

    supported = observe_coherent_image_motion(
        state,
        frame_id=109,
        observation_ts=1.3,
        image_foot_uv=(115.0, 160.0),
        bbox=(90.0, 100.0, 50.0, 100.0),
        contact_basis="pose:torso_motion",
        config=config,
    )
    assert supported is True
    output = update_human_cv_filter(
        state,
        measurement=np.array([1.04, 0.0, 2.02], dtype=np.float64),
        floor_y=0.0,
        now_ts=1.3,
        quality="weak",
        config=config,
        contact_basis="pose_ankle_floor",
        image_motion_supported=supported,
    )

    assert output == pytest.approx([1.04, 0.0, 2.02])
    assert state.measurement_accepted is True
    assert state.world_x == pytest.approx(1.04)
    assert state.world_z == pytest.approx(2.02)


def test_two_observed_ankle_pairs_seed_cold_track_without_motion_witness() -> None:
    config = HumanGroundConfig(reacquire_samples=3, reacquire_max_gap_s=0.5)
    state = PersonGroundState()

    for index, point in enumerate(((1.00, 2.00), (1.04, 2.02)), start=1):
        output = update_human_cv_filter(
            state,
            measurement=np.array([point[0], 0.0, point[1]], dtype=np.float64),
            floor_y=0.0,
            now_ts=0.1 * index,
            quality="weak",
            config=config,
            contact_basis="pose_ankle_floor",
            image_motion_supported=False,
        )
        if index == 1:
            assert state.measurement_accepted is False
            assert state.world_x is None

    assert output == pytest.approx([1.04, 0.0, 2.02])
    assert state.measurement_accepted is True
    assert state.world_x == pytest.approx(1.04)
    assert state.world_z == pytest.approx(2.02)


def test_cold_ankle_pair_consensus_spans_one_weaker_bbox_row() -> None:
    config = HumanGroundConfig(reacquire_samples=3, reacquire_max_gap_s=0.5)
    state = PersonGroundState()

    update_human_cv_filter(
        state,
        measurement=np.array([1.00, 0.0, 2.00], dtype=np.float64),
        floor_y=0.0,
        now_ts=0.1,
        quality="weak",
        config=config,
        contact_basis="pose_ankle_floor",
    )
    update_human_cv_filter(
        state,
        measurement=np.array([1.50, 0.0, 2.50], dtype=np.float64),
        floor_y=0.0,
        now_ts=0.2,
        quality="weak",
        config=config,
        contact_basis="bbox_bottom",
    )

    assert state.world_x is None
    assert state.reacquire_candidate_basis == "pose_floor"
    assert state.reacquire_candidate_exact_basis == "pose_ankle_floor"
    assert state.reacquire_count == 1
    assert state.reacquire_unavailable_rows == 1

    output = update_human_cv_filter(
        state,
        measurement=np.array([1.05, 0.0, 2.02], dtype=np.float64),
        floor_y=0.0,
        now_ts=0.3,
        quality="weak",
        config=config,
        contact_basis="pose_ankle_floor",
    )

    assert output == pytest.approx([1.05, 0.0, 2.02])
    assert state.measurement_accepted is True
    assert state.world_x == pytest.approx(1.05)


def test_slow_coherent_image_motion_does_not_unlock_two_sample_bootstrap() -> None:
    config = HumanGroundConfig(
        reacquire_samples=3,
        reacquire_max_gap_s=0.5,
        static_px_threshold=3.0,
        static_exit_frames=3,
    )
    state = PersonGroundState()

    for frame_id, observation_ts, px in (
        (100, 1.0, 100.0),
        (103, 1.1, 102.0),
        (106, 1.2, 104.0),
    ):
        observe_coherent_image_motion(
            state,
            frame_id=frame_id,
            observation_ts=observation_ts,
            image_foot_uv=(px, 160.0),
            bbox=(px - 25.0, 100.0, 50.0, 100.0),
            contact_basis="pose:torso_motion",
            config=config,
        )

    assert state.image_motion_supported is True
    assert state.image_motion_bootstrap_supported is False
    for index, x in enumerate((1.00, 1.04), start=1):
        update_human_cv_filter(
            state,
            measurement=np.array([x, 0.0, 2.0], dtype=np.float64),
            floor_y=0.0,
            now_ts=1.2 + 0.1 * index,
            quality="weak",
            config=config,
            contact_basis="pose_single_ankle_floor",
            image_motion_supported=True,
        )
        assert state.measurement_accepted is False
        assert state.reacquire_count == index


def test_strong_bootstrap_motion_latches_across_one_consistent_sample() -> None:
    config = HumanGroundConfig(
        reacquire_max_gap_s=0.5,
        static_px_threshold=3.0,
        static_exit_frames=3,
    )
    state = PersonGroundState()
    for frame_id, observation_ts, px in (
        (100, 1.0, 100.0),
        (103, 1.1, 105.0),
        (106, 1.2, 110.0),
    ):
        observe_coherent_image_motion(
            state,
            frame_id=frame_id,
            observation_ts=observation_ts,
            image_foot_uv=(px, 160.0),
            bbox=(px - 25.0, 100.0, 50.0, 100.0),
            contact_basis="pose:torso_motion",
            config=config,
        )
    assert state.image_motion_bootstrap_supported is True
    strong_ts = state.image_motion_bootstrap_ts

    # The next exact sample still belongs to the same physical consensus, but
    # its rolling-window displacement no longer exceeds the stronger 8 px
    # cold-start threshold. Preserve the prior proof without renewing it.
    observe_coherent_image_motion(
        state,
        frame_id=109,
        observation_ts=1.3,
        image_foot_uv=(111.0, 160.0),
        bbox=(86.0, 100.0, 50.0, 100.0),
        contact_basis="pose:torso_motion",
        config=config,
    )
    assert state.image_motion_supported is True
    assert state.image_motion_contact_distance_px == pytest.approx(6.0)
    assert state.image_motion_bootstrap_supported is True
    assert state.image_motion_bootstrap_ts == pytest.approx(strong_ts)

    observe_coherent_image_motion(
        state,
        frame_id=130,
        observation_ts=1.9,
        image_foot_uv=(112.0, 160.0),
        bbox=(87.0, 100.0, 50.0, 100.0),
        contact_basis="pose:torso_motion",
        config=config,
    )
    assert state.image_motion_bootstrap_supported is False


def test_missing_motion_witness_preserves_only_bounded_observed_consensus() -> None:
    config = HumanGroundConfig(
        reacquire_max_gap_s=0.5,
        static_px_threshold=3.0,
        static_exit_frames=3,
    )
    state = PersonGroundState()

    assert not observe_coherent_image_motion(
        state,
        frame_id=100,
        observation_ts=1.0,
        image_foot_uv=(100.0, 160.0),
        bbox=(75.0, 100.0, 50.0, 100.0),
        contact_basis="pose:torso_motion",
        config=config,
    )
    mark_image_motion_observation_unavailable(
        state,
        observation_ts=1.1,
        config=config,
    )
    assert state.image_motion_supported is False
    assert state.image_motion_bootstrap_supported is False
    assert len(state.image_motion_window) == 1

    assert not observe_coherent_image_motion(
        state,
        frame_id=106,
        observation_ts=1.2,
        image_foot_uv=(110.0, 160.0),
        bbox=(85.0, 100.0, 50.0, 100.0),
        contact_basis="pose:torso_motion",
        config=config,
    )
    assert observe_coherent_image_motion(
        state,
        frame_id=109,
        observation_ts=1.3,
        image_foot_uv=(120.0, 160.0),
        bbox=(95.0, 100.0, 50.0, 100.0),
        contact_basis="pose:torso_motion",
        config=config,
    )
    assert state.image_motion_bootstrap_supported is True

    mark_image_motion_observation_unavailable(
        state,
        observation_ts=1.9,
        config=config,
    )
    assert len(state.image_motion_window) == 0
    assert state.image_motion_contact_basis is None
    assert state.image_motion_bootstrap_supported is False
    assert state.image_motion_bootstrap_basis is None


@pytest.mark.parametrize(
    ("motion_basis", "bbox_stationary"),
    (("pose:ankle_pair", False), ("pose:torso_motion", True)),
)
def test_two_sample_pose_floor_bootstrap_rejects_wrong_motion_provenance(
    motion_basis: str,
    bbox_stationary: bool,
) -> None:
    config = HumanGroundConfig(reacquire_samples=3, reacquire_max_gap_s=0.5)
    state = PersonGroundState(
        image_motion_contact_basis=motion_basis,
        image_motion_streak=3,
        image_motion_supported=True,
        bbox_stationary_supported=bbox_stationary,
    )

    for index, basis in enumerate(
        ("pose_single_ankle_floor", "pose_ankle_floor"),
        start=1,
    ):
        update_human_cv_filter(
            state,
            measurement=np.array([1.0 + 0.02 * index, 0.0, 2.0], dtype=np.float64),
            floor_y=0.0,
            now_ts=0.1 * index,
            quality="weak",
            config=config,
            contact_basis=basis,
            image_motion_supported=True,
        )
        assert state.measurement_accepted is False
        assert state.world_x is None
        assert state.reacquire_count == index

    output = update_human_cv_filter(
        state,
        measurement=np.array([1.07, 0.0, 2.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=0.3,
        quality="weak",
        config=config,
        contact_basis="pose_ankle_floor",
        image_motion_supported=True,
    )
    assert output == pytest.approx([1.07, 0.0, 2.0])
    assert state.measurement_accepted is True


def test_weak_first_measurement_without_contact_basis_never_seeds() -> None:
    state = PersonGroundState()

    update_human_cv_filter(
        state,
        measurement=np.array([1.0, 0.0, 2.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=1.0,
        quality="weak",
        config=HumanGroundConfig(),
        contact_basis=None,
    )

    assert state.measurement_accepted is False
    assert state.measurement_rejection_reason == "weak_measurement_contact_basis_missing"
    assert state.world_x is None
    assert state.world_z is None


def test_phase4_same_segment_output_never_exceeds_human_speed() -> None:
    cfg = HumanGroundConfig(
        max_speed_mps=4.0,
        max_jump_m=0.75,
        alpha_good=0.45,
        process_noise_walk=0.55,
        meas_noise_good=0.08,
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
    state.motion_mode = "walk"

    # This observation is just inside the legacy innovation gate
    # (0.75 + 4/30 = 0.8833 m), but its proposed posterior would publish a
    # roughly 17 m/s step. It must be quarantined, not visually slewed.
    dt = 1.0 / 30.0
    out = update_human_cv_filter(
        state,
        measurement=np.array([0.88, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=dt,
        quality="good",
        config=cfg,
    )

    assert math.hypot(float(out[0]), float(out[2])) <= cfg.max_speed_mps * dt + 1e-9
    assert state.measurement_accepted is False
    assert state.measurement_rejection_reason == "physical_output_speed_exceeded"
    assert state.trail_append_allowed is False
    assert state.trail_break_required is False


def test_canonical_output_gate_quarantines_hidden_process_jump_in_media_time() -> None:
    cfg = HumanGroundConfig(max_speed_mps=4.0, reset_after_s=1.25)
    state = PersonGroundState(
        world_x=0.0,
        world_z=0.0,
        filtered_ts=10.0,
        motion_mode="walk",
    )
    first, admitted = admit_human_ground_output(
        state,
        candidate=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=10.0,
        media_pts_ns=1_000_000_000,
        config=cfg,
    )
    assert admitted is True
    assert first == pytest.approx([0.0, 0.0, 0.0])

    # Simulate an alternate process path drifting farther than a person can
    # travel over two 30fps media frames.  The boundary holds the prior point;
    # it never emits a post-hoc slew toward the divergent candidate.
    state.world_x = 0.64
    state.world_z = 0.0
    held, admitted = admit_human_ground_output(
        state,
        candidate=np.array([0.64, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=10.3,
        media_pts_ns=1_066_666_667,
        config=cfg,
    )
    assert admitted is False
    assert held == pytest.approx([0.0, 0.0, 0.0])
    assert state.world_x == pytest.approx(0.0)
    assert state.measurement_accepted is False
    assert state.measurement_rejection_reason == "physical_output_continuity_exceeded"
    assert state.measurement_allowed_m == pytest.approx(4.0 * 0.066666667)
    assert state.trail_append_allowed is False


def test_output_speed_rejection_restamps_exact_prior_output_as_hold() -> None:
    cfg = HumanGroundConfig(max_speed_mps=4.0, reset_after_s=1.25)
    state = PersonGroundState(
        world_x=0.10,
        world_z=0.0,
        filtered_ts=10.1,
        motion_mode="walk",
        measurement_accepted=False,
        measurement_rejection_reason="physical_output_speed_exceeded",
        last_output_world_x=0.0,
        last_output_world_z=0.0,
        last_output_media_pts_ns=1_000_000_000,
        last_output_filter_ts=10.0,
        last_output_trail_segment_id=0,
    )

    held, admitted = admit_human_ground_output(
        state,
        candidate=np.array([0.10, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=10.1,
        media_pts_ns=1_100_000_000,
        config=cfg,
        allow_segment_break=False,
        prior_output=(0.0, 0.0, 1_000_000_000, 10.0, 0),
    )

    assert admitted is False
    assert held == pytest.approx([0.0, 0.0, 0.0])
    assert state.last_output_world_x == pytest.approx(0.0)
    assert state.last_output_media_pts_ns == 1_100_000_000
    assert state.measurement_rejection_reason == "physical_output_speed_exceeded"
    assert state.trail_append_allowed is False


def test_canonical_output_gate_accepts_explicit_reanchor_with_trail_break() -> None:
    cfg = HumanGroundConfig(max_speed_mps=4.0)
    state = PersonGroundState(
        world_x=0.0,
        world_z=0.0,
        filtered_ts=5.0,
    )
    admit_human_ground_output(
        state,
        candidate=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=5.0,
        media_pts_ns=2_000_000_000,
        config=cfg,
    )
    state.trail_break_required = True
    state.trail_segment_id = 1
    relocated, admitted = admit_human_ground_output(
        state,
        candidate=np.array([3.0, 0.0, 2.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=5.033,
        media_pts_ns=2_033_333_333,
        config=cfg,
    )
    assert admitted is True
    assert relocated == pytest.approx([3.0, 0.0, 2.0])
    assert state.last_output_trail_segment_id == 1


def test_prediction_cannot_spend_a_hidden_metric_segment_break() -> None:
    cfg = HumanGroundConfig(max_speed_mps=4.0)
    state = PersonGroundState(
        world_x=0.0,
        world_z=0.0,
        filtered_ts=5.0,
    )
    admit_human_ground_output(
        state,
        candidate=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=5.0,
        media_pts_ns=2_000_000_000,
        config=cfg,
    )
    prior_output = (
        float(state.last_output_world_x),
        float(state.last_output_world_z),
        state.last_output_media_pts_ns,
        float(state.last_output_filter_ts),
        int(state.last_output_trail_segment_id),
    )
    # A metric candidate may have incremented the state segment and overwritten
    # the mutable output watermark before a later prediction wins source
    # selection. The prediction must be compared with the immutable output
    # visible at frame entry, not inherit the metric relocation exemption.
    state.trail_break_required = True
    state.trail_segment_id = 1
    state.last_output_world_x = 3.0
    state.last_output_world_z = 2.0
    state.last_output_media_pts_ns = 2_033_333_333
    state.last_output_filter_ts = 5.033
    state.last_output_trail_segment_id = 1
    held, admitted = admit_human_ground_output(
        state,
        candidate=np.array([3.0, 0.0, 2.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=5.033,
        media_pts_ns=2_033_333_333,
        config=cfg,
        allow_segment_break=False,
        prior_output=prior_output,
    )

    assert admitted is False
    assert held == pytest.approx([0.0, 0.0, 0.0])
    assert state.last_output_world_x == pytest.approx(0.0)
    assert state.last_output_world_z == pytest.approx(0.0)
    assert state.last_output_media_pts_ns == 2_033_333_333
    assert state.last_output_trail_segment_id == 0
    assert state.trail_segment_id == 1
    assert state.trail_break_required is True
    assert state.as_public_fields()["trail_segment_id"] == 0
    assert state.as_public_fields()["trail_break_required"] is False
    assert state.measurement_rejection_reason == (
        "physical_output_continuity_exceeded"
    )


def test_near_prediction_cannot_publish_a_hidden_metric_segment_break() -> None:
    cfg = HumanGroundConfig(max_speed_mps=4.0)
    state = PersonGroundState(
        world_x=0.0,
        world_z=0.0,
        filtered_ts=5.0,
    )
    admit_human_ground_output(
        state,
        candidate=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=5.0,
        media_pts_ns=2_000_000_000,
        config=cfg,
    )
    prior_output = (
        float(state.last_output_world_x),
        float(state.last_output_world_z),
        state.last_output_media_pts_ns,
        float(state.last_output_filter_ts),
        int(state.last_output_trail_segment_id),
    )
    state.trail_break_required = True
    state.trail_segment_id = 1

    emitted, admitted = admit_human_ground_output(
        state,
        candidate=np.array([0.10, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=5.033333333,
        media_pts_ns=2_033_333_333,
        config=cfg,
        allow_segment_break=False,
        prior_output=prior_output,
    )

    assert admitted is True
    assert emitted == pytest.approx([0.10, 0.0, 0.0])
    # Preserve the hidden relocation internally for the next metric output,
    # but attach this non-metric point to the last visible trail segment.
    assert state.trail_segment_id == 1
    assert state.trail_break_required is True
    assert state.last_output_trail_segment_id == 0
    assert state.as_public_fields()["trail_segment_id"] == 0
    assert state.as_public_fields()["trail_break_required"] is False


def test_phase4_normalizes_restored_overspeed_velocity_before_prior() -> None:
    """A direct reader cannot observe an over-speed CV prior after update."""

    cfg = HumanGroundConfig(max_speed_mps=4.0, max_jump_m=0.75)
    state = PersonGroundState(
        world_x=0.0,
        world_z=0.0,
        vel_world_x=12.0,
        vel_world_z=0.0,
        filtered_ts=10.0,
        motion_mode="walk",
    )

    dt = 0.066789
    output = update_human_cv_filter(
        state,
        measurement=np.array([10.0, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=10.0 + dt,
        quality="good",
        config=cfg,
    )

    assert math.hypot(float(state.vel_world_x), float(state.vel_world_z)) <= cfg.max_speed_mps
    assert math.hypot(float(output[0]) - 0.0, float(output[2]) - 0.0) <= cfg.max_speed_mps * dt + 1e-9
    assert state.measurement_accepted is False
    assert state.measurement_rejection_reason in {
        "physical_innovation_exceeded",
        "physical_output_speed_exceeded",
    }


def test_missing_metric_frame_advances_the_canonical_cv_state() -> None:
    cfg = HumanGroundConfig(max_speed_mps=4.0, max_jump_m=0.75)
    state = PersonGroundState(last_good_world=(0.0, 0.0, 0.0), last_good_ts=0.0)
    update_human_cv_filter(
        state,
        measurement=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=0.0,
        quality="good",
        config=cfg,
    )
    state.motion_mode = "walk"
    state.vel_world_x = 1.0

    first = advance_human_cv_prediction(
        state,
        floor_y=0.0,
        now_ts=0.1,
        config=cfg,
        reason="depth_measurement_not_current",
    )
    assert first is not None
    assert first[0] == pytest.approx(0.1)
    assert state.measurement_accepted is False
    assert state.world_x == pytest.approx(0.1)

    second = advance_human_cv_prediction(
        state,
        floor_y=0.0,
        now_ts=0.2,
        config=cfg,
        reason="depth_measurement_not_current",
    )
    assert second is not None
    assert second[0] == pytest.approx(0.2)
    assert state.filtered_ts == pytest.approx(0.2)


def test_projective_observation_updates_same_state_before_metric_return() -> None:
    cfg = HumanGroundConfig(max_speed_mps=4.0, max_jump_m=0.75)
    state = PersonGroundState(last_good_world=(0.0, 0.0, 0.0), last_good_ts=0.0)
    update_human_cv_filter(
        state,
        measurement=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=0.0,
        quality="good",
        config=cfg,
    )
    state.motion_mode = "walk"
    state.vel_world_x = 1.0

    projective = integrate_projective_ground_observation(
        state,
        measurement=np.array([0.30, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=0.1,
        config=cfg,
    )
    assert projective is not None
    assert state.measurement_accepted is False
    assert state.measurement_rejection_reason == "projective_weak_observation"
    assert state.world_x == pytest.approx(float(projective[0]))
    assert state.world_x > 0.0
    assert state.last_good_world == pytest.approx((0.0, 0.0, 0.0))

    # Returning metric evidence is filtered from the projective posterior,
    # rather than snapping from the old last-good point.
    metric = update_human_cv_filter(
        state,
        measurement=np.array([0.40, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=0.2,
        quality="good",
        config=cfg,
    )
    assert metric[0] > float(projective[0])
    assert metric[0] - float(projective[0]) <= cfg.max_speed_mps * 0.1 + 1e-9
    assert state.measurement_accepted is True


def test_projective_visible_recovery_reduces_gain_and_proves_display_slew() -> None:
    cfg = HumanGroundConfig(max_speed_mps=4.0, max_jump_m=0.75)
    state = PersonGroundState(
        world_x=0.0,
        world_z=0.0,
        filtered_ts=0.7,
        motion_mode="walk",
    )
    proof: dict[str, object] = {}

    recovered = integrate_projective_ground_observation(
        state,
        measurement=np.array([1.6, 0.0, 0.8], dtype=np.float64),
        floor_y=0.0,
        now_ts=1.1,
        config=cfg,
        # The kinematic origin has a 400 ms budget, while the latest visible
        # point has only a 100 ms display budget.
        prior_output=(0.0, 0.0, 700_000_000, 0.7, 0),
        display_output=(0.0, 0.0, 1_000_000_000, 1.0, 0),
        media_pts_ns=1_100_000_000,
        transition_proof=proof,
    )

    assert recovered is not None
    assert math.hypot(float(recovered[0]), float(recovered[2])) <= 0.4 + 1e-9
    assert proof["origin_world"] == pytest.approx([0.0, 0.0, 0.0])
    assert proof["origin_media_pts_ns"] == 700_000_000
    assert proof["gate_dt_s"] == pytest.approx(0.4)
    assert proof["visible_origin_world"] == pytest.approx([0.0, 0.0, 0.0])
    assert proof["visible_gate_dt_s"] == pytest.approx(0.1)
    assert proof["visible_max_step_m"] == pytest.approx(0.4)
    assert proof["visible_position_gain_reduced"] is True
    base = proof["position_base"]
    gain = float(proof["position_gain"])
    assert isinstance(base, list)
    assert 0.0 < gain < float(proof["visible_unconstrained_position_gain"])
    assert recovered[0] == pytest.approx(float(base[0]) + gain * (1.6 - float(base[0])))
    assert recovered[2] == pytest.approx(float(base[2]) + gain * (0.8 - float(base[2])))
    # The reduced gain is a real process posterior, including velocity; the
    # next process step is not left with the discarded larger-gain velocity.
    assert state.world_x == pytest.approx(float(recovered[0]))
    assert state.world_z == pytest.approx(float(recovered[2]))


def test_projective_prior_output_reduces_gain_inside_physical_disk() -> None:
    """A weak projective posterior must be rerun, never clipped, to the prior disk."""

    cfg = HumanGroundConfig(max_speed_mps=4.0, max_jump_m=0.75)
    state = PersonGroundState(
        world_x=0.0,
        world_z=0.0,
        filtered_ts=0.0,
        motion_mode="walk",
    )
    proof: dict[str, object] = {}

    recovered = integrate_projective_ground_observation(
        state,
        measurement=np.array([1.1, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=0.1,
        config=cfg,
        prior_output=(0.0, 0.0, 1_000_000_000, 0.0, 0),
        media_pts_ns=1_100_000_000,
        transition_proof=proof,
    )

    assert recovered is not None
    assert math.hypot(float(recovered[0]), float(recovered[2])) <= 0.4 + 1e-12
    assert proof["prior_output_position_gain_reduced"] is True
    assert float(proof["prior_output_unconstrained_position_gain"]) > float(
        proof["position_gain"]
    )
    assert float(proof["prior_output_max_step_m"]) == pytest.approx(0.4)
    assert float(proof["position_gain"]) == pytest.approx(0.4 / 1.1)
    base = proof["position_base"]
    gain = float(proof["position_gain"])
    assert isinstance(base, list)
    # The emitted point is exactly the filter transition with the solved gain;
    # no coordinate clamp or post-hoc slew is involved.
    assert recovered[0] == pytest.approx(float(base[0]) + gain * (1.1 - float(base[0])))
    assert recovered[2] == pytest.approx(float(base[2]) + gain * (0.0 - float(base[2])))
    assert math.hypot(float(state.vel_world_x), float(state.vel_world_z)) <= cfg.max_speed_mps
    assert state.measurement_rejection_reason == "projective_weak_observation"


def test_projective_robust_innovation_limit_closes_real_occlusion_gap() -> None:
    """A noisy body reconstruction is bounded twice instead of dropping the dot."""

    cfg = HumanGroundConfig(max_speed_mps=4.0, max_jump_m=0.75)
    prior_x = 18.39725089290527
    prior_z = 7.9374314776827415
    state = PersonGroundState(
        world_x=prior_x,
        world_z=prior_z,
        vel_world_x=0.64843343745674,
        vel_world_z=-0.120795758053115,
        filtered_ts=0.5,
        motion_mode="walk",
    )
    raw_target = np.array(
        [17.702123258639375, 0.0, 6.799183234455305],
        dtype=np.float64,
    )
    proof: dict[str, object] = {}

    recovered = integrate_projective_ground_observation(
        state,
        measurement=raw_target,
        floor_y=0.0,
        now_ts=0.6,
        config=cfg,
        prior_output=(prior_x, prior_z, 500_000_000, 0.5, 0),
        media_pts_ns=600_000_000,
        transition_proof=proof,
    )

    assert recovered is not None
    assert proof["innovation_limit_applied"] is True
    assert proof["raw_position_target"] == pytest.approx(raw_target.tolist())
    assert float(proof["raw_innovation_m"]) > float(
        proof["innovation_limit_m"]
    )
    assert float(proof["innovation_limit_m"]) == pytest.approx(1.15)
    filter_observation = proof["filter_observation"]
    base = proof["position_base"]
    assert isinstance(filter_observation, list)
    assert isinstance(base, list)
    assert math.hypot(
        float(filter_observation[0]) - float(base[0]),
        float(filter_observation[2]) - float(base[2]),
    ) == pytest.approx(1.15)
    gain = float(proof["position_gain"])
    assert recovered[0] == pytest.approx(
        float(base[0])
        + gain * (float(filter_observation[0]) - float(base[0]))
    )
    assert recovered[2] == pytest.approx(
        float(base[2])
        + gain * (float(filter_observation[2]) - float(base[2]))
    )
    assert math.hypot(
        float(recovered[0]) - prior_x,
        float(recovered[2]) - prior_z,
    ) == pytest.approx(0.4)
    assert proof["prior_output_position_gain_reduced"] is True
    assert math.hypot(state.vel_world_x, state.vel_world_z) <= 4.0 + 1e-12


def test_projective_prior_output_zero_gain_or_no_intersection_is_rejected() -> None:
    """A line starting at the prior disk edge and moving outward has no positive gain."""

    cfg = HumanGroundConfig(max_speed_mps=4.0, max_jump_m=0.75)
    state = PersonGroundState(
        world_x=0.0,
        world_z=0.0,
        vel_world_x=4.0,
        filtered_ts=0.0,
        motion_mode="walk",
    )
    proof: dict[str, object] = {}

    recovered = integrate_projective_ground_observation(
        state,
        measurement=np.array([1.4, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=0.1,
        config=cfg,
        prior_output=(0.0, 0.0, 1_000_000_000, 0.0, 0),
        media_pts_ns=1_100_000_000,
        transition_proof=proof,
    )

    assert recovered is None
    assert proof == {}
    # The trial was isolated; rejection cannot mutate the canonical process
    # state or manufacture a gain-zero projective row.
    assert state.world_x == pytest.approx(0.0)
    assert state.world_z == pytest.approx(0.0)
    assert state.filtered_ts == pytest.approx(0.0)
    assert state.vel_world_x == pytest.approx(4.0)
    assert state.measurement_rejection_reason is None


def test_projective_prior_and_display_disks_both_constrain_one_transition() -> None:
    """The tighter display disk composes with, and cannot exceed, the prior disk."""

    cfg = HumanGroundConfig(max_speed_mps=4.0, max_jump_m=0.75)
    state = PersonGroundState(
        world_x=0.0,
        world_z=0.0,
        filtered_ts=0.0,
        motion_mode="walk",
    )
    proof: dict[str, object] = {}

    recovered = integrate_projective_ground_observation(
        state,
        measurement=np.array([1.1, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=0.1,
        config=cfg,
        prior_output=(0.0, 0.0, 1_000_000_000, 0.0, 0),
        # 50 ms of visible media time gives a 0.20 m display disk.
        display_output=(0.0, 0.0, 1_050_000_000, 0.05, 0),
        media_pts_ns=1_100_000_000,
        transition_proof=proof,
    )

    assert recovered is not None
    distance = math.hypot(float(recovered[0]), float(recovered[2]))
    assert distance <= 0.2 + 1e-12
    assert distance <= 0.4 + 1e-12
    assert proof["prior_output_position_gain_reduced"] is True
    assert proof["visible_position_gain_reduced"] is True
    assert float(proof["position_gain"]) == pytest.approx(0.2 / 1.1)
    base = proof["position_base"]
    gain = float(proof["position_gain"])
    assert isinstance(base, list)
    assert recovered[0] == pytest.approx(float(base[0]) + gain * (1.1 - float(base[0])))
    assert recovered[2] == pytest.approx(float(base[2]) + gain * (0.0 - float(base[2])))
    assert math.hypot(float(state.world_x), float(state.world_z)) <= 0.2 + 1e-12


def test_metric_filter_does_not_use_projective_prior_overrun_escape_hatch() -> None:
    """The trial-only escape hatch cannot weaken ordinary metric admission."""

    cfg = HumanGroundConfig(max_speed_mps=4.0, max_jump_m=0.75)
    state = PersonGroundState(
        world_x=0.0,
        world_z=0.0,
        filtered_ts=0.0,
        motion_mode="walk",
    )

    output = update_human_cv_filter(
        state,
        measurement=np.array([1.1, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=0.1,
        quality="weak",
        config=cfg,
    )

    assert math.hypot(float(output[0]), float(output[2])) <= 0.4 + 1e-12
    assert state.measurement_accepted is False
    assert state.measurement_rejection_reason == "physical_output_speed_exceeded"


def test_projective_without_display_output_preserves_kinematic_gain() -> None:
    cfg = HumanGroundConfig(max_speed_mps=4.0, max_jump_m=0.75)
    state = PersonGroundState(
        world_x=0.0,
        world_z=0.0,
        filtered_ts=0.7,
        motion_mode="walk",
    )
    proof: dict[str, object] = {}

    recovered = integrate_projective_ground_observation(
        state,
        measurement=np.array([1.6, 0.0, 0.8], dtype=np.float64),
        floor_y=0.0,
        now_ts=1.1,
        config=cfg,
        prior_output=(0.0, 0.0, 700_000_000, 0.7, 0),
        media_pts_ns=1_100_000_000,
        transition_proof=proof,
    )

    assert recovered is not None
    assert math.hypot(float(recovered[0]), float(recovered[2])) > 0.4
    base = proof["position_base"]
    gain = float(proof["position_gain"])
    assert isinstance(base, list)
    assert recovered[0] == pytest.approx(float(base[0]) + gain * (1.6 - float(base[0])))
    assert recovered[2] == pytest.approx(float(base[2]) + gain * (0.8 - float(base[2])))
    assert "visible_position_gain_reduced" not in proof


def test_projective_visible_recovery_uses_explicit_hold_when_only_zero_gain_fits() -> None:
    cfg = HumanGroundConfig(max_speed_mps=4.0, max_jump_m=0.75)
    state = PersonGroundState(
        world_x=0.4,
        world_z=0.0,
        filtered_ts=0.7,
        motion_mode="walk",
    )

    recovered = integrate_projective_ground_observation(
        state,
        measurement=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=1.1,
        config=cfg,
        prior_output=(0.4, 0.0, 700_000_000, 0.7, 0),
        display_output=(0.0, 0.0, 1_000_000_000, 1.0, 0),
        media_pts_ns=1_100_000_000,
        transition_proof={},
    )

    assert recovered is None
    assert state.world_x == pytest.approx(0.4)
    assert state.world_z == pytest.approx(0.0)


def test_metric_filter_can_apply_same_visible_recovery_gain_cap() -> None:
    cfg = HumanGroundConfig(max_speed_mps=4.0, max_jump_m=0.75)
    state = PersonGroundState(
        world_x=0.0,
        world_z=0.0,
        filtered_ts=0.7,
        motion_mode="walk",
    )
    proof: dict[str, object] = {}

    recovered = update_human_cv_filter(
        state,
        measurement=np.array([1.6, 0.0, 0.8], dtype=np.float64),
        floor_y=0.0,
        now_ts=1.1,
        quality="good",
        config=cfg,
        media_pts_ns=1_100_000_000,
        display_output=(0.0, 0.0, 1_000_000_000, 1.0, 0),
        transition_proof=proof,
    )

    assert recovered is not None
    assert math.hypot(float(recovered[0]), float(recovered[2])) <= 0.4 + 1e-9
    assert proof["visible_position_gain_reduced"] is True
    base = proof["position_base"]
    gain = float(proof["position_gain"])
    assert isinstance(base, list)
    assert recovered[0] == pytest.approx(float(base[0]) + gain * (1.6 - float(base[0])))
    assert recovered[2] == pytest.approx(float(base[2]) + gain * (0.8 - float(base[2])))
    assert state.world_x == pytest.approx(float(recovered[0]))
    assert state.world_z == pytest.approx(float(recovered[2]))


def test_phase4_output_speed_rejection_reanchors_only_with_trail_break() -> None:
    cfg = HumanGroundConfig(
        max_speed_mps=4.0,
        max_jump_m=0.75,
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
    state.motion_mode = "walk"

    for index, x in enumerate((0.88, 0.87), start=1):
        out = update_human_cv_filter(
            state,
            measurement=np.array([x, 0.0, 0.0], dtype=np.float64),
            floor_y=0.0,
            now_ts=index / 30.0,
            quality="good",
            config=cfg,
            contact_basis="pose:ankle_pair",
            image_motion_supported=True,
        )
        assert state.measurement_accepted is False
        assert state.measurement_rejection_reason == "physical_output_speed_exceeded"
        assert state.reacquire_count == index
        assert state.trail_break_required is False
        assert math.hypot(float(out[0]), float(out[2])) <= cfg.max_speed_mps * (index / 30.0) + 1e-9

    out = update_human_cv_filter(
        state,
        measurement=np.array([0.86, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=3.0 / 30.0,
        quality="good",
        config=cfg,
        contact_basis="pose:ankle_pair",
        image_motion_supported=True,
    )
    assert out == pytest.approx([0.86, 0.0, 0.0])
    assert state.measurement_accepted is True
    assert state.reacquired is True
    assert state.trail_break_required is True
    assert state.trail_segment_id == 1


def test_post_occlusion_bbox_only_candidate_cannot_reanchor_world() -> None:
    cfg = HumanGroundConfig(
        max_speed_mps=1.0,
        max_jump_m=0.20,
        reacquire_samples=3,
        reacquire_max_gap_s=0.5,
    )
    state = PersonGroundState(post_occlusion_reacquire_support_required=True)
    update_human_cv_filter(
        state,
        measurement=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=0.0,
        quality="good",
        config=cfg,
    )

    # Local bbox continuity remains usable; only a new distant world origin is
    # forbidden while contact remains unverified.
    local = update_human_cv_filter(
        state,
        measurement=np.array([0.05, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=0.1,
        quality="estimated",
        config=cfg,
        contact_basis="bbox:bottom_center",
        image_motion_supported=True,
    )
    assert local[0] > 0.0
    assert state.measurement_accepted is True
    assert state.post_occlusion_reacquire_support_required is True

    for index, x in enumerate((2.00, 2.02, 2.01), start=2):
        output = update_human_cv_filter(
            state,
            measurement=np.array([x, 0.0, 0.0], dtype=np.float64),
            floor_y=0.0,
            now_ts=index * 0.1,
            quality="estimated",
            config=cfg,
            contact_basis="bbox:bottom_center",
            image_motion_supported=True,
        )
        assert output[0] != pytest.approx(x)
        assert state.measurement_accepted is False
        assert state.measurement_rejection_reason == (
            "post_occlusion_reacquire_requires_verified_support"
        )
        assert state.reacquire_count == 0
        assert state.reacquired is False


def test_verified_support_clears_post_occlusion_reanchor_guard() -> None:
    cfg = HumanGroundConfig(
        max_speed_mps=1.0,
        max_jump_m=0.20,
        reacquire_samples=3,
        reacquire_max_gap_s=0.5,
    )
    state = PersonGroundState(post_occlusion_reacquire_support_required=True)
    update_human_cv_filter(
        state,
        measurement=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=0.0,
        quality="good",
        config=cfg,
    )

    for index, x in enumerate((2.00, 2.02, 2.01), start=1):
        output = update_human_cv_filter(
            state,
            measurement=np.array([x, 0.0, 0.0], dtype=np.float64),
            floor_y=0.0,
            now_ts=index * 0.1,
            quality="good",
            config=cfg,
            contact_basis="registered_lower_body_depth",
            image_motion_supported=True,
            verified_reacquire_support=True,
        )

    assert output == pytest.approx([2.01, 0.0, 0.0])
    assert state.measurement_accepted is True
    assert state.reacquired is True
    assert state.trail_break_required is True
    assert state.post_occlusion_reacquire_support_required is False


def test_upright_body_reanchor_accumulates_before_verified_final_sample() -> None:
    cfg = HumanGroundConfig(
        max_speed_mps=1.0,
        max_jump_m=0.20,
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
    state.post_occlusion_reacquire_support_required = True

    for index, x in enumerate((2.00, 2.05), start=1):
        output = update_human_cv_filter(
            state,
            measurement=np.array([x, 0.0, 0.0], dtype=np.float64),
            floor_y=0.0,
            now_ts=index * 0.1,
            quality="weak",
            config=cfg,
            contact_basis="pose:upright_body_planes",
            verified_reacquire_support=False,
        )
        assert output[0] != pytest.approx(x)
        assert state.measurement_accepted is False
        assert state.reacquired is False
        assert state.reacquire_count == index

    output = update_human_cv_filter(
        state,
        measurement=np.array([2.10, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=0.3,
        quality="estimated",
        config=cfg,
        contact_basis="pose:upright_body_planes",
        verified_reacquire_support=True,
    )
    assert output == pytest.approx([2.10, 0.0, 0.0])
    assert state.measurement_accepted is True
    assert state.reacquired is True
    assert state.trail_break_required is True
    assert state.post_occlusion_reacquire_support_required is False


def test_pose_floor_reanchor_uses_full_consensus_without_waiting_for_motion() -> None:
    cfg = HumanGroundConfig(
        max_speed_mps=4.0,
        max_jump_m=0.75,
        reacquire_samples=3,
        reacquire_max_gap_s=0.5,
    )
    state = PersonGroundState()
    update_human_cv_filter(
        state,
        measurement=np.array([0.0, 0.0, 3.446], dtype=np.float64),
        floor_y=0.0,
        now_ts=0.0,
        quality="good",
        config=cfg,
    )

    candidates = (
        (0.10, 1.679, 2.882, "pose_single_ankle_floor", False),
        (0.20, 1.825, 2.949, "pose_ankle_floor", False),
        (0.27, 1.810, 3.010, "pose_ankle_floor", False),
    )
    for index, (ts, x, z, basis, motion_supported) in enumerate(candidates):
        output = update_human_cv_filter(
            state,
            measurement=np.array([x, 0.0, z], dtype=np.float64),
            floor_y=0.0,
            now_ts=ts,
            quality="good",
            config=cfg,
            contact_basis=basis,
            image_motion_supported=motion_supported,
        )
        if index < 2:
            assert state.measurement_accepted is False
            assert state.reacquired is False
            assert state.trail_break_required is False
            assert state.reacquire_candidate_basis == "pose_floor"
            assert state.reacquire_count == min(index + 1, cfg.reacquire_samples)
            assert output[0] != pytest.approx(x)

    assert output == pytest.approx([1.810, 0.0, 3.010])
    assert state.measurement_accepted is True
    assert state.reacquired is True
    assert state.trail_break_required is True
    assert state.trail_segment_id == 1
    assert state.world_x == pytest.approx(1.810)
    assert state.world_z == pytest.approx(3.010)


def test_established_pose_floor_reacquisition_spans_one_bounded_unavailable_row() -> None:
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

    for now_ts, x, basis, expected_count in (
        (0.1, 5.00, "pose_single_ankle_floor", 1),
        (0.3, 5.02, "pose_ankle_floor", 2),
        (0.5, 5.04, "pose_ankle_floor", 3),
    ):
        output = update_human_cv_filter(
            state,
            measurement=np.array([x, 0.0, 0.0], dtype=np.float64),
            floor_y=0.0,
            now_ts=now_ts,
            quality="good",
            config=cfg,
            contact_basis=basis,
            image_motion_supported=False,
        )
        if expected_count < cfg.reacquire_samples:
            assert state.reacquire_count == expected_count
            mark_world_measurement_unavailable(
                state,
                reason="pose_floor_temporarily_unavailable",
            )
            assert state.reacquire_count == expected_count
            assert state.reacquire_unavailable_rows == 1

    assert output == pytest.approx([5.04, 0.0, 0.0])
    assert state.measurement_accepted is True
    assert state.reacquired is True
    assert state.trail_break_required is True
    assert state.trail_segment_id == 1


def test_pose_floor_reacquisition_gap_remains_one_row_and_time_bounded() -> None:
    cfg = HumanGroundConfig(
        max_speed_mps=1.0,
        max_jump_m=0.25,
        reacquire_samples=3,
        reacquire_max_gap_s=0.5,
    )

    for second_candidate_ts, unavailable_rows in ((0.3, 2), (0.7, 1)):
        state = PersonGroundState()
        update_human_cv_filter(
            state,
            measurement=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            floor_y=0.0,
            now_ts=0.0,
            quality="good",
            config=cfg,
        )
        update_human_cv_filter(
            state,
            measurement=np.array([5.0, 0.0, 0.0], dtype=np.float64),
            floor_y=0.0,
            now_ts=0.1,
            quality="good",
            config=cfg,
            contact_basis="pose_ankle_floor",
            image_motion_supported=False,
        )
        for _ in range(unavailable_rows):
            mark_world_measurement_unavailable(
                state,
                reason="pose_floor_temporarily_unavailable",
            )
        update_human_cv_filter(
            state,
            measurement=np.array([5.02, 0.0, 0.0], dtype=np.float64),
            floor_y=0.0,
            now_ts=second_candidate_ts,
            quality="good",
            config=cfg,
            contact_basis="pose_ankle_floor",
            image_motion_supported=False,
        )
        assert state.reacquire_count == 1
        assert state.reacquired is False
        assert state.trail_segment_id == 0


@pytest.mark.parametrize(
    "contact_basis",
    ("bbox_bottom", "gravity_drop", "pose:leg_pair_extension"),
)
def test_non_observed_floor_basis_still_requires_motion_to_reanchor(
    contact_basis: str,
) -> None:
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

    for index in range(1, 5):
        output = update_human_cv_filter(
            state,
            measurement=np.array([5.0 + 0.01 * index, 0.0, 0.0], dtype=np.float64),
            floor_y=0.0,
            now_ts=0.1 * index,
            quality="good",
            config=cfg,
            contact_basis=contact_basis,
            image_motion_supported=False,
        )
        assert output[0] == pytest.approx(0.0)
        assert state.reacquire_count == 0
        assert state.reacquired is False
        assert state.trail_break_required is False

    assert state.trail_segment_id == 0


def test_cold_pose_floor_bootstrap_spans_only_one_unavailable_sample() -> None:
    cfg = HumanGroundConfig(reacquire_samples=3, reacquire_max_gap_s=0.5)
    state = PersonGroundState()
    update_human_cv_filter(
        state,
        measurement=np.array([5.0, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=0.1,
        quality="weak",
        config=cfg,
        contact_basis="pose_ankle_floor",
        image_motion_supported=False,
    )
    assert state.world_x is None
    assert state.reacquire_count == 1

    mark_world_measurement_unavailable(
        state,
        reason="pose_floor_temporarily_unavailable",
    )

    assert state.reacquire_count == 1
    assert state.reacquire_candidate_basis == "pose_floor"
    assert state.reacquire_unavailable_rows == 1

    mark_world_measurement_unavailable(
        state,
        reason="pose_floor_still_unavailable",
    )

    assert state.reacquire_count == 0
    assert state.reacquire_candidate_basis is None
    assert state.reacquire_unavailable_rows == 0


def test_phase4_rejected_measurement_is_pure_cv_time_update() -> None:
    cfg = HumanGroundConfig(max_speed_mps=2.0, max_jump_m=0.5)
    state = PersonGroundState()
    update_human_cv_filter(
        state,
        measurement=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=0.0,
        quality="good",
        config=cfg,
    )
    state.last_good_world = (0.0, 0.0, 0.0)
    state.last_good_ts = 0.0
    state.vel_world_x = 1.25
    state.vel_world_z = -0.50

    out = update_human_cv_filter(
        state,
        measurement=np.array([10.0, 0.0, 10.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=0.2,
        quality="good",
        config=cfg,
    )

    assert out == pytest.approx([0.25, 0.0, -0.10])
    assert state.world_x == pytest.approx(0.25)
    assert state.world_z == pytest.approx(-0.10)
    assert state.filtered_ts == pytest.approx(0.2)
    assert state.vel_world_x == pytest.approx(1.25)
    assert state.vel_world_z == pytest.approx(-0.50)
    assert state.measurement_accepted is False
    assert state.measurement_rejection_reason == "physical_innovation_exceeded"
    assert state.last_good_world == pytest.approx((0.0, 0.0, 0.0))
    assert state.last_good_ts == pytest.approx(0.0)


def test_bounded_process_velocity_change_cannot_jump_from_previous_posterior() -> None:
    cfg = HumanGroundConfig(max_speed_mps=4.0, rejected_prediction_horizon_s=1.0)
    state = PersonGroundState(
        world_x=0.4,
        world_z=0.0,
        vel_world_x=-4.0,
        vel_world_z=0.0,
        filtered_ts=0.1,
        last_good_world=(0.0, 0.0, 0.0),
        last_good_ts=0.0,
        rejection_anchor_x=0.0,
        rejection_anchor_z=0.0,
        rejection_anchor_ts=0.0,
    )

    out = advance_human_cv_prediction(
        state,
        floor_y=0.0,
        now_ts=0.2,
        config=cfg,
    )

    # The anchor-relative candidate would be -0.8 m, a 1.2 m visible jump in
    # 100 ms.  Quarantine it as a hold rather than publishing a synthetic slew.
    assert out == pytest.approx([0.4, 0.0, 0.0])
    assert state.measurement_accepted is False
    assert state.measurement_rejection_reason == "bounded_process_continuity_exceeded"
    assert state.trail_append_allowed is False


def test_rewound_projective_update_cannot_move_again_at_same_frame_timestamp() -> None:
    cfg = HumanGroundConfig(max_speed_mps=4.0, max_jump_m=0.75)
    state = PersonGroundState(
        world_x=0.2,
        world_z=0.0,
        vel_world_x=1.0,
        vel_world_z=0.0,
        filtered_ts=0.2,
        last_good_world=(0.0, 0.0, 0.0),
        last_good_ts=0.0,
        rejection_anchor_x=0.0,
        rejection_anchor_z=0.0,
        rejection_anchor_ts=0.0,
        rejection_previous_world_x=0.4,
        rejection_previous_world_z=0.0,
        rejection_previous_world_ts=0.1,
        measurement_accepted=False,
        measurement_rejection_reason="physical_innovation_exceeded",
    )

    out = integrate_projective_ground_observation(
        state,
        measurement=np.array([-0.5, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=0.2,
        config=cfg,
    )

    assert out is None
    assert state.world_x == pytest.approx(0.2)
    assert state.world_z == pytest.approx(0.0)
    assert state.filtered_ts == pytest.approx(0.2)
    assert state.measurement_rejection_reason == "physical_innovation_exceeded"


def test_projective_update_uses_queue_visible_media_cadence() -> None:
    cfg = HumanGroundConfig(max_speed_mps=4.0, max_jump_m=0.75)
    hidden_state = PersonGroundState(
        world_x=5.0,
        world_z=0.0,
        vel_world_x=0.0,
        vel_world_z=0.0,
        filtered_ts=1.09,
        last_good_world=(0.0, 0.0, 0.0),
        last_good_ts=1.0,
        measurement_accepted=False,
        measurement_rejection_reason="hidden_callback_rejection",
    )
    without_visible_reference = integrate_projective_ground_observation(
        hidden_state,
        measurement=np.array([0.5, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=1.1,
        config=cfg,
    )
    assert without_visible_reference is None
    assert hidden_state.world_x == pytest.approx(5.0)

    state = PersonGroundState(
        world_x=5.0,
        world_z=0.0,
        vel_world_x=0.0,
        vel_world_z=0.0,
        filtered_ts=1.09,
        last_good_world=(0.0, 0.0, 0.0),
        last_good_ts=1.0,
        measurement_accepted=False,
        measurement_rejection_reason="hidden_callback_rejection",
    )
    transition_proof: dict[str, object] = {}
    admitted = integrate_projective_ground_observation(
        state,
        measurement=np.array([0.5, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=1.1,
        config=cfg,
        prior_output=(0.0, 0.0, 1_000_000_000, 1.0, 0),
        media_pts_ns=1_100_000_000,
        transition_proof=transition_proof,
    )
    assert admitted is not None
    assert 0.0 < float(admitted[0]) <= 0.4
    assert float(admitted[2]) == pytest.approx(0.0)
    assert state.measurement_accepted is False
    assert state.measurement_rejection_reason == "projective_weak_observation"
    assert transition_proof["version"] == 1
    assert transition_proof["kind"] == "innovation_update"
    assert transition_proof["origin_kind"] == "queue_admitted_world_output"
    assert transition_proof["origin_world"] == pytest.approx([0.0, 0.0, 0.0])
    assert transition_proof["origin_media_pts_ns"] == 1_000_000_000
    assert transition_proof["current_media_pts_ns"] == 1_100_000_000
    assert transition_proof["origin_trail_segment_id"] == 0
    assert transition_proof["gate_dt_s"] == pytest.approx(0.1)
    base = transition_proof["position_base"]
    assert isinstance(base, list)
    gain = float(transition_proof["position_gain"])
    assert 0.0 < gain <= 1.0
    expected_x = float(base[0]) + gain * (0.5 - float(base[0]))
    assert admitted[0] == pytest.approx(expected_x)


def test_projective_transition_proof_requires_committed_media_origin() -> None:
    cfg = HumanGroundConfig(max_speed_mps=4.0, max_jump_m=0.75)
    state = PersonGroundState(
        world_x=0.0,
        world_z=0.0,
        filtered_ts=1.0,
        last_good_world=(0.0, 0.0, 0.0),
        last_good_ts=1.0,
    )
    proof: dict[str, object] = {}

    result = integrate_projective_ground_observation(
        state,
        measurement=np.array([0.2, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=1.1,
        config=cfg,
        prior_output=(0.0, 0.0, None, 1.0, 0),
        media_pts_ns=1_100_000_000,
        transition_proof=proof,
    )

    assert result is None
    assert proof == {}
    assert state.world_x == pytest.approx(0.0)
    assert state.filtered_ts == pytest.approx(1.0)


def test_phase4_rejected_prediction_is_bounded_and_reacquirable() -> None:
    cfg = HumanGroundConfig(
        max_speed_mps=2.0,
        max_jump_m=0.5,
        rejected_prediction_horizon_s=0.40,
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
    state.last_good_world = (0.0, 0.0, 0.0)
    state.last_good_ts = 0.0
    state.vel_world_x = 1.25
    state.vel_world_z = -0.50

    for now_ts, expected_x, expected_z in (
        (0.2, 0.25, -0.10),
        (0.8, 0.50, -0.20),
        (1.4, 0.50, -0.20),
    ):
        out = update_human_cv_filter(
            state,
            measurement=np.array([10.0, 0.0, 10.0], dtype=np.float64),
            floor_y=0.0,
            now_ts=now_ts,
            quality="good",
            config=cfg,
        )
        assert out[0] == pytest.approx(expected_x)
        assert out[2] == pytest.approx(expected_z)
        assert state.world_x == pytest.approx(expected_x)
        assert state.world_z == pytest.approx(expected_z)
        assert state.last_good_world == pytest.approx((0.0, 0.0, 0.0))

    # Once exact-frame image evidence returns, a coherent run of observations
    # can still relocate the process state instead of remaining pinned to the
    # bounded horizon forever.
    for now_ts, x, expected_count in (
        (1.5, 5.0, 1),
        (1.6, 5.1, 2),
    ):
        out = update_human_cv_filter(
            state,
            measurement=np.array([x, 0.0, 0.0], dtype=np.float64),
            floor_y=0.0,
            now_ts=now_ts,
            quality="good",
            config=cfg,
            contact_basis="pose:ankle_pair",
            image_motion_supported=True,
        )
        assert out[0] == pytest.approx(0.50)
        assert state.reacquire_count == expected_count
        assert state.measurement_accepted is False

    out = update_human_cv_filter(
        state,
        measurement=np.array([5.2, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=1.7,
        quality="good",
        config=cfg,
        contact_basis="pose:ankle_pair",
        image_motion_supported=True,
    )
    assert out[0] == pytest.approx(5.2)
    assert state.world_x == pytest.approx(5.2)
    assert state.reacquired is True
    assert state.trail_break_required is True
    assert state.rejection_anchor_x is None
    assert state.rejection_anchor_z is None


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
    assert state.reacquire_count == 0


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
            contact_basis="pose:ankle_pair",
            image_motion_supported=True,
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
        contact_basis="pose:ankle_pair",
        image_motion_supported=True,
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
        contact_basis="pose:ankle_pair",
        image_motion_supported=True,
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
            contact_basis="pose:ankle_pair",
            image_motion_supported=True,
        )
    assert state.measurement_accepted is False
    assert state.reacquire_count == 1
    assert state.world_x == pytest.approx(0.0)


def test_bbox_stationarity_uses_bounded_published_observation_cadence() -> None:
    cfg = HumanGroundConfig(static_px_threshold=3.0, static_exit_frames=3)
    state = PersonGroundState()

    assert observe_bbox_stationarity(
        state,
        frame_id=10,
        observation_ts=1.0,
        bbox=[100.0, 50.0, 80.0, 120.0],
        config=cfg,
    ) is False
    assert observe_bbox_stationarity(
        state,
        frame_id=13,
        observation_ts=1.1,
        bbox=[102.0, 50.0, 80.0, 120.0],
        config=cfg,
    ) is False
    assert observe_bbox_stationarity(
        state,
        frame_id=16,
        observation_ts=1.2,
        bbox=[103.0, 50.0, 80.0, 120.0],
        config=cfg,
    ) is True
    assert state.bbox_stationary_streak == 3

    # A stale publication gap or a real box displacement revokes the evidence.
    assert observe_bbox_stationarity(
        state,
        frame_id=19,
        observation_ts=2.0,
        bbox=[103.0, 50.0, 80.0, 120.0],
        config=cfg,
    ) is False
    assert observe_bbox_stationarity(
        state,
        frame_id=22,
        observation_ts=2.1,
        bbox=[140.0, 50.0, 80.0, 120.0],
        config=cfg,
    ) is False
    assert state.bbox_stationary_supported is False


def test_bbox_stationarity_cannot_accumulate_latest_only_detector_drift() -> None:
    cfg = HumanGroundConfig(static_px_threshold=3.0, static_exit_frames=3)
    state = PersonGroundState()

    # Every local step is below the six-pixel jitter budget, but the box has
    # moved well beyond the immutable first observation by the third row.
    # Latest-only comparison would incorrectly authorize a frozen world hold.
    assert not observe_bbox_stationarity(
        state,
        frame_id=10,
        observation_ts=1.0,
        bbox=[100.0, 50.0, 80.0, 120.0],
        config=cfg,
    )
    assert not observe_bbox_stationarity(
        state,
        frame_id=13,
        observation_ts=1.1,
        bbox=[105.0, 50.0, 80.0, 120.0],
        config=cfg,
    )
    assert not observe_bbox_stationarity(
        state,
        frame_id=16,
        observation_ts=1.2,
        bbox=[110.0, 50.0, 80.0, 120.0],
        config=cfg,
    )
    assert state.bbox_stationary_streak == 1
    assert state.bbox_stationary_supported is False


def test_idle_preunlock_uses_three_published_slow_motion_observations() -> None:
    cfg = HumanGroundConfig(
        static_px_threshold=3.0,
        static_exit_frames=3,
        idle_deadzone_m=0.28,
    )
    state = PersonGroundState(
        world_x=0.0,
        world_z=0.0,
        filtered_ts=0.0,
        motion_mode="idle",
        locked_world=(0.0, 0.0),
    )

    outputs = []
    support = []
    for frame_id, px, world_x in (
        (10, 100.0, 0.00),
        (11, 101.75, 0.12),
        (12, 103.50, 0.24),
    ):
        supported = observe_coherent_image_motion(
            state,
            frame_id=frame_id,
            observation_ts=float(frame_id - 9) * 0.1,
            image_foot_uv=(px, 200.0),
            bbox=(px - 25.0, 100.0, 50.0, 100.0),
            contact_basis="pose:ankle_pair",
            config=cfg,
        )
        support.append(supported)
        outputs.append(
            update_human_cv_filter(
                state,
                measurement=np.array([world_x, 0.0, 0.0], dtype=np.float64),
                floor_y=0.0,
                now_ts=float(frame_id - 9) * 0.1,
                quality="good",
                config=cfg,
                contact_basis="pose:ankle_pair",
                image_motion_supported=supported,
            )
        )
        update_motion_mode(
            state,
            now_ts=float(frame_id - 9) * 0.1,
            image_foot_uv=(px, 200.0),
            config=cfg,
        )

    assert support == [False, False, True]
    assert state.motion_mode == "walk"
    assert state.locked_world is None
    assert outputs[0][0] == pytest.approx(0.0)
    assert outputs[1][0] == pytest.approx(0.0)
    assert outputs[2][0] > 0.0
    assert state.reacquired is False
    assert state.trail_break_required is False
    assert state.trail_segment_id == 0


def test_idle_preunlock_accepts_inference_cadence_source_frame_gaps() -> None:
    cfg = HumanGroundConfig(
        static_px_threshold=3.0,
        static_exit_frames=3,
        reacquire_max_gap_s=0.5,
    )
    state = PersonGroundState(motion_mode="idle", locked_world=(0.0, 0.0))

    support = []
    for frame_id, observation_ts, px in (
        (100, 1.0, 100.0),
        (103, 1.1, 102.0),
        (106, 1.2, 104.0),
    ):
        support.append(
            observe_coherent_image_motion(
                state,
                frame_id=frame_id,
                observation_ts=observation_ts,
                image_foot_uv=(px, 200.0),
                bbox=(px - 25.0, 100.0, 50.0, 100.0),
                contact_basis="pose:ankle_pair",
                config=cfg,
            )
        )

    assert support == [False, False, True]
    assert state.motion_mode == "walk"
    assert state.locked_world is None


def test_detector_bottom_motion_still_requires_persistent_silhouette_movement() -> None:
    cfg = HumanGroundConfig(
        static_px_threshold=3.0,
        static_exit_frames=3,
        reacquire_max_gap_s=0.5,
    )
    state = PersonGroundState(motion_mode="idle", locked_world=(0.0, 0.0))

    support = []
    for frame_id, observation_ts, left in (
        (100, 1.0, 75.0),
        (103, 1.1, 77.0),
        (106, 1.2, 79.0),
    ):
        bbox = (left, 100.0, 50.0, 100.0)
        support.append(
            observe_coherent_image_motion(
                state,
                frame_id=frame_id,
                observation_ts=observation_ts,
                image_foot_uv=(left + 25.0, 200.0),
                bbox=bbox,
                contact_basis="bbox:bottom_center",
                config=cfg,
            )
        )

    assert support == [False, False, True]

    static_state = PersonGroundState(motion_mode="idle", locked_world=(0.0, 0.0))
    for frame_id, observation_ts in ((100, 1.0), (103, 1.1), (106, 1.2)):
        assert observe_coherent_image_motion(
            static_state,
            frame_id=frame_id,
            observation_ts=observation_ts,
            image_foot_uv=(100.0, 200.0),
            bbox=(75.0, 100.0, 50.0, 100.0),
            contact_basis="bbox:bottom_center",
            config=cfg,
        ) is False
    assert static_state.motion_mode == "idle"


def test_image_motion_consensus_resets_across_stale_published_gap() -> None:
    cfg = HumanGroundConfig(
        static_px_threshold=3.0,
        static_exit_frames=3,
        reacquire_max_gap_s=0.5,
    )
    state = PersonGroundState(motion_mode="idle", locked_world=(0.0, 0.0))

    support = []
    for frame_id, observation_ts, px in (
        (100, 1.0, 100.0),
        (103, 1.1, 102.0),
        (106, 2.0, 104.0),
        (109, 2.1, 106.0),
    ):
        support.append(
            observe_coherent_image_motion(
                state,
                frame_id=frame_id,
                observation_ts=observation_ts,
                image_foot_uv=(px, 200.0),
                bbox=(px - 25.0, 100.0, 50.0, 100.0),
                contact_basis="pose:ankle_pair",
                config=cfg,
            )
        )

    assert support == [False, False, False, False]
    assert state.motion_mode == "idle"
    assert state.locked_world == (0.0, 0.0)


def test_stationary_bbox_anchor_jump_cannot_unlock_or_reacquire() -> None:
    cfg = HumanGroundConfig(
        static_px_threshold=3.0,
        static_exit_frames=3,
        max_speed_mps=2.0,
        max_jump_m=0.5,
        reacquire_samples=3,
    )
    state = PersonGroundState(
        world_x=0.0,
        world_z=0.0,
        filtered_ts=0.0,
        motion_mode="idle",
        locked_world=(0.0, 0.0),
    )

    for frame_id, contact_u, measurement_x in (
        (20, 100.0, 8.0),
        (21, 180.0, 8.1),
        (22, 260.0, 8.2),
        (23, 340.0, 8.3),
    ):
        supported = observe_coherent_image_motion(
            state,
            frame_id=frame_id,
            observation_ts=float(frame_id - 19) * 0.1,
            image_foot_uv=(contact_u, 200.0),
            bbox=(75.0, 100.0, 50.0, 100.0),
            contact_basis="pose:left_ankle",
            config=cfg,
        )
        assert supported is False
        output = update_human_cv_filter(
            state,
            measurement=np.array([measurement_x, 0.0, 0.0], dtype=np.float64),
            floor_y=0.0,
            now_ts=float(frame_id - 19) * 0.1,
            quality="good",
            config=cfg,
            contact_basis="pose:left_ankle",
            image_motion_supported=supported,
        )
        assert np.allclose(output, [0.0, 0.0, 0.0])

    assert state.motion_mode == "idle"
    assert state.locked_world == (0.0, 0.0)
    assert state.reacquire_count == 0
    assert state.reacquired is False
    assert state.trail_segment_id == 0


def test_reacquire_confirmation_cannot_cross_contact_basis() -> None:
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

    for index, basis in enumerate(
        ("pose:ankle_pair", "gravity_drop", "pose:ankle_pair"),
        start=1,
    ):
        output = update_human_cv_filter(
            state,
            measurement=np.array([5.0 + index * 0.1, 0.0, 0.0], dtype=np.float64),
            floor_y=0.0,
            now_ts=float(index) * 0.1,
            quality="good",
            config=cfg,
            contact_basis=basis,
            image_motion_supported=True,
        )
        assert np.allclose(output, [0.0, 0.0, 0.0])
        assert state.reacquire_count == 1

    assert state.reacquired is False
    assert state.trail_break_required is False
    assert state.trail_segment_id == 0


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


def test_inferred_ground_alignment_uses_one_immutable_visible_origin() -> None:
    state = PersonGroundState()
    first = align_inferred_ground_observation(
        state,
        raw_measurement=np.array([10.0, 0.0, 20.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=1.0,
        observed_at_us=1_000_000,
        media_pts_ns=1_000_000_000,
        lifecycle_generation=2,
        height_ref_scene=1.8,
        world_frame_id="backend_world_m",
        world_frame_revision="world-r1",
        world_transform_sha256="a" * 64,
        trusted_output_reference=(1.0, 2.0, 900_000_000, 0.9, 3),
    )
    assert first is not None
    (
        first_point,
        first_anchor,
        first_delta,
        first_robust_raw,
        first_consensus_count,
        first_consensus_span_s,
        first_consensus_ts_s,
        first_consensus_observed_at_us,
        first_consensus_media_pts_ns,
        first_raw_evidence_gap_s,
    ) = first
    assert first_point == pytest.approx([1.0, 0.0, 2.0])
    assert first_delta == pytest.approx((0.0, 0.0, 0.0))
    assert first_robust_raw == pytest.approx((10.0, 0.0, 20.0))
    assert first_consensus_count == 1
    assert first_consensus_span_s == pytest.approx(0.0)
    assert first_consensus_ts_s == pytest.approx(1.0)
    assert first_consensus_observed_at_us == 1_000_000
    assert first_consensus_media_pts_ns == 1_000_000_000
    assert first_raw_evidence_gap_s == pytest.approx(0.0)

    # The caller's visible watermark advances, but inferred rows may not
    # renew the episode origin. Only the exact current raw geometry delta is
    # transported from the original queue-visible point.
    second = align_inferred_ground_observation(
        state,
        raw_measurement=np.array([10.2, 0.0, 19.7], dtype=np.float64),
        floor_y=0.0,
        now_ts=1.1,
        observed_at_us=1_100_000,
        media_pts_ns=1_100_000_000,
        lifecycle_generation=2,
        height_ref_scene=1.8,
        world_frame_id="backend_world_m",
        world_frame_revision="world-r1",
        world_transform_sha256="a" * 64,
        trusted_output_reference=(1.1, 1.95, 1_000_000_000, 1.0, 3),
    )
    assert second is not None
    (
        second_point,
        second_anchor,
        second_delta,
        second_robust_raw,
        second_consensus_count,
        second_consensus_span_s,
        second_consensus_ts_s,
        second_consensus_observed_at_us,
        second_consensus_media_pts_ns,
        second_raw_evidence_gap_s,
    ) = second
    assert second_anchor is first_anchor
    assert second_point == pytest.approx([1.2, 0.0, 1.7])
    assert second_delta == pytest.approx((0.2, 0.0, -0.3))
    assert second_robust_raw == pytest.approx((10.2, 0.0, 19.7))
    assert second_consensus_count == 2
    assert second_consensus_span_s == pytest.approx(0.1)
    assert second_consensus_ts_s == pytest.approx(1.1)
    assert second_consensus_observed_at_us == 1_100_000
    assert second_consensus_media_pts_ns == 1_100_000_000
    assert second_raw_evidence_gap_s == pytest.approx(0.1)
    assert second_anchor.trusted_world_origin == pytest.approx((1.0, 0.0, 2.0))


def test_inferred_ground_alignment_cancels_arbitrary_absolute_bias() -> None:
    def aligned(raw_origin: tuple[float, float], raw_current: tuple[float, float]):
        state = PersonGroundState()
        first = align_inferred_ground_observation(
            state,
            raw_measurement=np.array([raw_origin[0], 0.0, raw_origin[1]]),
            floor_y=0.0,
            now_ts=1.0,
            observed_at_us=1_000_000,
            media_pts_ns=1_000_000_000,
            lifecycle_generation=1,
            height_ref_scene=1.75,
            world_frame_id="backend_world_m",
            world_frame_revision="world-r1",
            world_transform_sha256="b" * 64,
            trusted_output_reference=(4.0, 6.0, 900_000_000, 0.9, 0),
        )
        assert first is not None
        current = align_inferred_ground_observation(
            state,
            raw_measurement=np.array([raw_current[0], 0.0, raw_current[1]]),
            floor_y=0.0,
            now_ts=1.1,
            observed_at_us=1_100_000,
            media_pts_ns=1_100_000_000,
            lifecycle_generation=1,
            height_ref_scene=1.75,
            world_frame_id="backend_world_m",
            world_frame_revision="world-r1",
            world_transform_sha256="b" * 64,
            trusted_output_reference=(4.0, 6.0, 1_000_000_000, 1.0, 0),
        )
        assert current is not None
        return current[0]

    ordinary = aligned((10.0, 20.0), (10.3, 19.6))
    biased = aligned((1_010.0, -1_980.0), (1_010.3, -1_980.4))
    assert ordinary == pytest.approx([4.3, 0.0, 5.6])
    assert biased == pytest.approx(ordinary)


def test_inferred_ground_alignment_uses_recent_medoid_for_one_frame_outlier() -> None:
    state = PersonGroundState()
    common = {
        "floor_y": 0.0,
        "lifecycle_generation": 1,
        "height_ref_scene": 1.75,
        "world_frame_id": "backend_world_m",
        "world_frame_revision": "world-r1",
        "world_transform_sha256": "e" * 64,
    }
    first = align_inferred_ground_observation(
        state,
        raw_measurement=np.array([10.0, 0.0, 20.0]),
        now_ts=1.0,
        observed_at_us=1_000_000,
        media_pts_ns=1_000_000_000,
        trusted_output_reference=(4.0, 6.0, 900_000_000, 0.9, 0),
        **common,
    )
    second = align_inferred_ground_observation(
        state,
        raw_measurement=np.array([10.2, 0.0, 19.8]),
        now_ts=1.05,
        observed_at_us=1_050_000,
        media_pts_ns=1_050_000_000,
        trusted_output_reference=(4.0, 6.0, 1_000_000_000, 1.0, 0),
        **common,
    )
    outlier = align_inferred_ground_observation(
        state,
        raw_measurement=np.array([100.0, 0.0, -100.0]),
        now_ts=1.1,
        observed_at_us=1_100_000,
        media_pts_ns=1_100_000_000,
        trusted_output_reference=(4.1, 5.9, 1_050_000_000, 1.05, 0),
        **common,
    )
    assert first is not None and second is not None and outlier is not None
    assert outlier[0] == pytest.approx([4.2, 0.0, 5.8])
    assert outlier[2] == pytest.approx((0.2, 0.0, -0.2))
    assert outlier[3] == pytest.approx((10.2, 0.0, 19.8))
    assert outlier[4] == 3
    assert outlier[5] == pytest.approx(0.1)
    # The medoid chose the prior in-window sample and preserves that sample's
    # evidence times instead of relabeling it as exact-current.
    assert outlier[6] == pytest.approx(1.05)
    assert outlier[7] == 1_050_000
    assert outlier[8] == 1_050_000_000
    assert outlier[1] is first[1]


def test_inferred_ground_alignment_fails_closed_on_time_reversal() -> None:
    state = PersonGroundState()
    armed = align_inferred_ground_observation(
        state,
        raw_measurement=np.array([10.0, 0.0, 20.0]),
        floor_y=0.0,
        now_ts=2.0,
        observed_at_us=2_000_000,
        media_pts_ns=2_000_000_000,
        lifecycle_generation=1,
        height_ref_scene=1.8,
        world_frame_id="backend_world_m",
        world_frame_revision="world-r1",
        world_transform_sha256="c" * 64,
        trusted_output_reference=(1.0, 2.0, 1_900_000_000, 1.9, 0),
    )
    assert armed is not None
    rejected = align_inferred_ground_observation(
        state,
        raw_measurement=np.array([10.1, 0.0, 19.9]),
        floor_y=0.0,
        now_ts=1.0,
        observed_at_us=1_000_000,
        media_pts_ns=1_000_000_000,
        lifecycle_generation=1,
        height_ref_scene=1.8,
        world_frame_id="backend_world_m",
        world_frame_revision="world-r1",
        world_transform_sha256="c" * 64,
        trusted_output_reference=(1.0, 2.0, 1_900_000_000, 1.9, 0),
    )
    assert rejected is None
    assert state.inferred_ground_continuity_anchor is None

    clear_inferred_ground_continuity_anchor(
        state,
        block_rearm=False,
    )
    assert state.inferred_ground_continuity_anchor is None


@pytest.mark.parametrize(
    ("media_pts_ns", "trusted_media_pts_ns"),
    (
        (None, 900_000_000),
        (1_000_000_000, None),
        (0, 900_000_000),
        (1_000_000_000, 0),
    ),
)
def test_inferred_ground_alignment_requires_complete_media_time(
    media_pts_ns: int | None,
    trusted_media_pts_ns: int | None,
) -> None:
    state = PersonGroundState()

    rejected = align_inferred_ground_observation(
        state,
        raw_measurement=np.array([10.0, 0.0, 20.0]),
        floor_y=0.0,
        now_ts=1.0,
        observed_at_us=1_000_000,
        media_pts_ns=media_pts_ns,
        lifecycle_generation=1,
        height_ref_scene=1.8,
        world_frame_id="backend_world_m",
        world_frame_revision="world-r1",
        world_transform_sha256="f" * 64,
        trusted_output_reference=(
            1.0,
            2.0,
            trusted_media_pts_ns,
            0.9,
            0,
        ),
    )

    assert rejected is None
    assert state.inferred_ground_continuity_anchor is None


def test_inferred_ground_alignment_long_gap_resets_only_short_history() -> None:
    state = PersonGroundState()
    config = HumanGroundConfig(
        inferred_process_consensus_horizon_s=0.40,
        inferred_process_consensus_samples=3,
    )
    common = {
        "floor_y": 0.0,
        "lifecycle_generation": 1,
        "height_ref_scene": 1.8,
        "world_frame_id": "backend_world_m",
        "world_frame_revision": "world-r1",
        "world_transform_sha256": "9" * 64,
        "config": config,
    }
    armed = align_inferred_ground_observation(
        state,
        raw_measurement=np.array([10.0, 0.0, 20.0]),
        now_ts=1.0,
        observed_at_us=1_000_000,
        media_pts_ns=1_000_000_000,
        trusted_output_reference=(1.0, 2.0, 900_000_000, 0.9, 0),
        **common,
    )
    assert armed is not None
    immutable_anchor = armed[1]

    recent = align_inferred_ground_observation(
        state,
        raw_measurement=np.array([10.2, 0.0, 19.8]),
        now_ts=1.1,
        observed_at_us=1_100_000,
        media_pts_ns=1_100_000_000,
        trusted_output_reference=(1.2, 1.8, 1_000_000_000, 1.0, 0),
        **common,
    )
    assert recent is not None
    assert recent[1] is immutable_anchor
    assert len(state.inferred_ground_raw_history) == 2

    after_gap = align_inferred_ground_observation(
        state,
        raw_measurement=np.array([10.6, 0.0, 19.4]),
        now_ts=1.6,
        observed_at_us=1_600_000,
        media_pts_ns=1_600_000_000,
        # The queue-visible watermark may have advanced through inferred
        # publications. It must not replace either immutable authority origin.
        trusted_output_reference=(99.0, -99.0, 1_100_000_000, 1.1, 0),
        **common,
    )
    assert after_gap is not None
    (
        aligned,
        retained_anchor,
        delta,
        robust_raw,
        sample_count,
        consensus_span_s,
        consensus_ts_s,
        consensus_observed_at_us,
        consensus_media_pts_ns,
        raw_evidence_gap_s,
    ) = after_gap
    assert retained_anchor is immutable_anchor
    assert state.inferred_ground_continuity_anchor is immutable_anchor
    assert state.inferred_ground_continuity_blocked is False
    assert retained_anchor.raw_origin == pytest.approx((10.0, 0.0, 20.0))
    assert retained_anchor.trusted_world_origin == pytest.approx((1.0, 0.0, 2.0))
    assert retained_anchor.raw_origin_ts == pytest.approx(1.0)
    assert retained_anchor.raw_origin_observed_at_us == 1_000_000
    assert retained_anchor.raw_origin_media_pts_ns == 1_000_000_000
    assert retained_anchor.trusted_origin_filter_ts == pytest.approx(0.9)
    assert retained_anchor.trusted_origin_media_pts_ns == 900_000_000
    assert aligned == pytest.approx([1.6, 0.0, 1.4])
    assert delta == pytest.approx((0.6, 0.0, -0.6))
    assert robust_raw == pytest.approx((10.6, 0.0, 19.4))
    assert sample_count == 1
    assert consensus_span_s == pytest.approx(0.0)
    assert consensus_ts_s == pytest.approx(1.6)
    assert consensus_observed_at_us == 1_600_000
    assert consensus_media_pts_ns == 1_600_000_000
    assert raw_evidence_gap_s == pytest.approx(0.5)
    assert list(state.inferred_ground_raw_history) == [
        (1.6, 1_600_000, 1_600_000_000, 10.6, 0.0, 19.4)
    ]

    following = align_inferred_ground_observation(
        state,
        raw_measurement=np.array([10.8, 0.0, 19.2]),
        now_ts=1.7,
        observed_at_us=1_700_000,
        media_pts_ns=1_700_000_000,
        trusted_output_reference=(100.0, -100.0, 1_600_000_000, 1.6, 0),
        **common,
    )
    assert following is not None
    assert following[1] is immutable_anchor
    assert following[0] == pytest.approx([1.8, 0.0, 1.2])
    assert following[2] == pytest.approx((0.8, 0.0, -0.8))
    assert following[4] == 2
    assert following[5] == pytest.approx(0.1)
    assert len(state.inferred_ground_raw_history) == 2
