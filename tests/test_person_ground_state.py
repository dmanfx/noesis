"""Unit tests for human-realistic person ground-state pathing (Phases 1–6)."""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from noesis.telemetry.person_ground_state import (
    HumanGroundConfig,
    PersonGroundState,
    apply_source_hysteresis,
    classify_posture,
    commit_path_point,
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


def test_phase3_classify_posture_sitting_and_standing() -> None:
    cfg = HumanGroundConfig()
    assert classify_posture(kpts_abs=_kpts_standing(), bbox=[0, 0, 60, 220], height_ref_scene=1.8, config=cfg) in (
        "standing",
        "unknown",
    )
    assert classify_posture(kpts_abs=_kpts_sitting(), bbox=[0, 0, 90, 110], height_ref_scene=1.8, config=cfg) == "sitting"
    assert classify_posture(kpts_abs=None, bbox=[0, 0, 200, 80], height_ref_scene=None, config=cfg) == "lying"


def test_phase2_bent_leg_blocks_pose_leg_floor() -> None:
    cfg = HumanGroundConfig()
    assert legs_are_bent(_kpts_sitting(), config=cfg) is True
    # Zero ankle conf so only leg-extension path would fire for standing logic
    k = _kpts_sitting().copy()
    k[15, 2] = 0.0
    k[16, 2] = 0.0
    cand = resolve_pose_floor_anchor(k, posture="sitting", config=cfg)
    assert cand is not None
    assert cand.source in ("pose_hip_floor", "pose_body_floor")
    assert cand.source != "pose_leg_floor"


def test_phase3_standing_uses_ankle_mid() -> None:
    cfg = HumanGroundConfig()
    cand = resolve_pose_floor_anchor(_kpts_standing(), posture="standing", config=cfg)
    assert cand is not None
    assert cand.source == "pose_ankle_floor"
    assert cand.u == pytest.approx(120.0)
    assert cand.v == pytest.approx(300.0)


def test_phase2_source_hysteresis_sticky() -> None:
    cfg = HumanGroundConfig(source_hold_frames=5, source_switch_min_score_gain=0.20)
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


def test_phase4_cv_filter_tracks_walk_and_clamps_speed() -> None:
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
    # Allowed move ~ 2*0.1 + 0.5 = 0.7m, then alpha blends — must remain << 10m
    assert abs(float(p1[0])) <= 0.75 + 1e-3
    assert abs(float(state.vel_world_x)) <= 2.0 + 1e-3


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


def test_source_score_prefers_hip_when_sitting() -> None:
    ankle = source_score("pose_ankle_floor", quality="good", posture="sitting")
    hip = source_score("pose_hip_floor", quality="good", posture="sitting")
    assert hip > ankle
