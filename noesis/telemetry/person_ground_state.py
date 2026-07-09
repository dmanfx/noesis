"""Person ground-state estimator for human-realistic BEV/OSD pathing.

Phases implemented here (shared by baseline and V3DT analytics hooks):

1. Stationary lock — freeze world when idle; trails do not append.
2. Source hysteresis — sticky world_source selection; reject bent-leg ankle fakes.
3. Posture-aware contact — standing ankles vs sitting pelvis vs lying body.
4. Human constant-velocity filter with adaptive process/measurement noise.
5. PersonGroundState — single producer state consumed by BEV + OSD trails.
6. Path simplification — RDP / collinear merge on committed history only.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


# COCO-17 indices used by the pose SGIE path.
POSE_KPT_INDEX: Dict[str, int] = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


Posture = str  # standing | sitting | lying | unknown
MotionMode = str  # walk | idle | sit | lie | unknown


@dataclass(frozen=True)
class HumanGroundConfig:
    """Tunable priors for human ground tracking."""

    # Phase 1 — stationary lock
    static_px_threshold: float = 3.0
    static_speed_mps: float = 0.15
    static_enter_s: float = 0.40
    static_exit_speed_mps: float = 0.30
    static_exit_frames: int = 3
    idle_deadzone_m: float = 0.28
    idle_deadzone_sit_m: float = 0.40
    residual_window: int = 12

    # Phase 2 — source hysteresis
    source_hold_frames: int = 10
    source_switch_min_score_gain: float = 0.20

    # Phase 3 — posture
    sit_bbox_height_ratio: float = 0.62
    lie_bbox_aspect: float = 1.35
    bent_leg_vertical_ratio: float = 0.58
    kpt_conf_threshold: float = 0.35

    # Phase 4 — human CV filter
    max_speed_mps: float = 4.0
    max_accel_mps2: float = 4.5
    max_jump_m: float = 0.75
    process_noise_walk: float = 0.55
    process_noise_idle: float = 0.02
    process_noise_sit: float = 0.015
    meas_noise_good: float = 0.08
    meas_noise_weak: float = 0.28
    meas_noise_idle: float = 0.45
    meas_noise_sit: float = 0.55
    reset_after_s: float = 1.25
    min_dt_s: float = 1e-3
    alpha_good: float = 0.45
    alpha_weak: float = 0.20
    beta_scale: float = 0.22

    # Phase 6 — path simplification
    path_min_step_m: float = 0.05
    path_simplify_epsilon_m: float = 0.06
    path_max_points: int = 257

    def __post_init__(self) -> None:
        object.__setattr__(self, "static_px_threshold", max(0.0, float(self.static_px_threshold)))
        object.__setattr__(self, "static_speed_mps", max(0.0, float(self.static_speed_mps)))
        object.__setattr__(self, "static_enter_s", max(0.0, float(self.static_enter_s)))
        object.__setattr__(self, "static_exit_speed_mps", max(0.0, float(self.static_exit_speed_mps)))
        object.__setattr__(self, "static_exit_frames", max(1, int(self.static_exit_frames)))
        object.__setattr__(self, "idle_deadzone_m", max(0.0, float(self.idle_deadzone_m)))
        object.__setattr__(self, "idle_deadzone_sit_m", max(0.0, float(self.idle_deadzone_sit_m)))
        object.__setattr__(self, "residual_window", max(3, int(self.residual_window)))
        object.__setattr__(self, "source_hold_frames", max(1, int(self.source_hold_frames)))
        object.__setattr__(self, "source_switch_min_score_gain", max(0.0, float(self.source_switch_min_score_gain)))
        object.__setattr__(self, "sit_bbox_height_ratio", float(min(1.0, max(0.05, float(self.sit_bbox_height_ratio)))))
        object.__setattr__(self, "lie_bbox_aspect", max(0.5, float(self.lie_bbox_aspect)))
        object.__setattr__(self, "bent_leg_vertical_ratio", float(min(1.0, max(0.05, float(self.bent_leg_vertical_ratio)))))
        object.__setattr__(self, "kpt_conf_threshold", float(min(1.0, max(0.0, float(self.kpt_conf_threshold)))))
        object.__setattr__(self, "max_speed_mps", max(0.0, float(self.max_speed_mps)))
        object.__setattr__(self, "max_accel_mps2", max(0.0, float(self.max_accel_mps2)))
        object.__setattr__(self, "max_jump_m", max(0.0, float(self.max_jump_m)))
        object.__setattr__(self, "process_noise_walk", max(1e-6, float(self.process_noise_walk)))
        object.__setattr__(self, "process_noise_idle", max(1e-6, float(self.process_noise_idle)))
        object.__setattr__(self, "process_noise_sit", max(1e-6, float(self.process_noise_sit)))
        object.__setattr__(self, "meas_noise_good", max(1e-6, float(self.meas_noise_good)))
        object.__setattr__(self, "meas_noise_weak", max(1e-6, float(self.meas_noise_weak)))
        object.__setattr__(self, "meas_noise_idle", max(1e-6, float(self.meas_noise_idle)))
        object.__setattr__(self, "meas_noise_sit", max(1e-6, float(self.meas_noise_sit)))
        object.__setattr__(self, "reset_after_s", max(0.0, float(self.reset_after_s)))
        object.__setattr__(self, "min_dt_s", max(0.0, float(self.min_dt_s)))
        object.__setattr__(self, "alpha_good", float(min(1.0, max(0.0, float(self.alpha_good)))))
        object.__setattr__(self, "alpha_weak", float(min(1.0, max(0.0, float(self.alpha_weak)))))
        object.__setattr__(self, "beta_scale", float(min(1.0, max(0.0, float(self.beta_scale)))))
        object.__setattr__(self, "path_min_step_m", max(0.0, float(self.path_min_step_m)))
        object.__setattr__(self, "path_simplify_epsilon_m", max(0.0, float(self.path_simplify_epsilon_m)))
        object.__setattr__(self, "path_max_points", max(2, int(self.path_max_points)))

    @classmethod
    def from_env_defaults(
        cls,
        *,
        static_px_threshold: float = 3.0,
        static_jump_m: float = 0.75,
        max_speed_mps: float = 4.0,
        alpha_good: float = 0.45,
        alpha_weak: float = 0.20,
    ) -> "HumanGroundConfig":
        return cls(
            static_px_threshold=static_px_threshold,
            idle_deadzone_m=max(0.15, min(0.75, float(static_jump_m) * 0.35)),
            max_speed_mps=max_speed_mps,
            max_jump_m=max(0.25, float(static_jump_m)),
            alpha_good=alpha_good,
            alpha_weak=alpha_weak,
        )


@dataclass(frozen=True)
class PoseAnchorCandidate:
    u: float
    v: float
    source: str
    quality: str = "good"
    quality_reason: Optional[str] = None
    height_lock_eligible: bool = False
    score: float = 1.0


@dataclass
class PersonGroundState:
    """Per-track ground state shared by world estimation, BEV, and OSD trails."""

    ts: float = 0.0
    height_ref_scene: Optional[float] = None
    last_good_world: Optional[Tuple[float, float, float]] = None
    last_good_ts: float = 0.0
    world_x: Optional[float] = None
    world_z: Optional[float] = None
    vel_world_x: float = 0.0
    vel_world_z: float = 0.0
    # < 0 means "never updated"; 0.0 is a valid timestamp.
    filtered_ts: float = -1.0

    # Phase 1 / 3
    motion_mode: MotionMode = "unknown"
    posture: Posture = "unknown"
    idle_since_ts: float = 0.0
    exit_motion_frames: int = 0
    locked_world: Optional[Tuple[float, float]] = None
    residual_m: Deque[float] = field(default_factory=lambda: deque(maxlen=12))
    image_foot_history: Deque[Tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=12))

    # Phase 2
    sticky_source: Optional[str] = None
    sticky_source_frames: int = 0
    sticky_source_score: float = 0.0
    source_switch_count: int = 0

    # Phase 5 public mirrors
    image_foot_u: Optional[float] = None
    image_foot_v: Optional[float] = None
    trail_append_allowed: bool = True
    idle_jitter_m: float = 0.0

    def as_public_fields(self) -> Dict[str, Any]:
        return {
            "motion_mode": str(self.motion_mode),
            "posture": str(self.posture),
            "trail_append_allowed": bool(self.trail_append_allowed),
            "idle_jitter_m": float(self.idle_jitter_m) if math.isfinite(float(self.idle_jitter_m)) else None,
            "source_switch_count": int(self.source_switch_count),
            "sticky_world_source": str(self.sticky_source) if self.sticky_source else None,
        }


def _finite(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return float(out)


def pose_point(
    kpts_abs: np.ndarray,
    name: str,
    *,
    conf_threshold: float,
) -> Optional[Tuple[float, float, float]]:
    idx = POSE_KPT_INDEX.get(name)
    if idx is None or idx < 0 or idx >= int(kpts_abs.shape[0]):
        return None
    try:
        x = float(kpts_abs[idx, 0])
        y = float(kpts_abs[idx, 1])
        conf = float(kpts_abs[idx, 2])
    except Exception:
        return None
    if conf < float(conf_threshold):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return float(x), float(y), float(conf)


def _mid_xy(
    a: Optional[Tuple[float, float, float]],
    b: Optional[Tuple[float, float, float]],
) -> Optional[Tuple[float, float]]:
    if a is None and b is None:
        return None
    if a is None:
        return float(b[0]), float(b[1])  # type: ignore[index]
    if b is None:
        return float(a[0]), float(a[1])
    return (float(a[0]) + float(b[0])) * 0.5, (float(a[1]) + float(b[1])) * 0.5


def classify_posture(
    *,
    kpts_abs: Optional[np.ndarray],
    bbox: Optional[Sequence[float]],
    height_ref_scene: Optional[float],
    config: HumanGroundConfig,
) -> Posture:
    """Heuristic posture from bbox aspect/height and optional pose geometry."""
    bbox_w = bbox_h = None
    if bbox is not None and len(bbox) >= 4:
        try:
            bbox_w = float(bbox[2])
            bbox_h = float(bbox[3])
        except Exception:
            bbox_w = bbox_h = None

    if bbox_w is not None and bbox_h is not None and bbox_h > 1.0:
        aspect = float(bbox_w) / float(bbox_h)
        if aspect >= float(config.lie_bbox_aspect):
            # A short/wide box can also be lower-body occlusion of a standing person.
            # Only commit to "lying" from bbox alone when we have no standing height lock.
            if height_ref_scene is None:
                return "lying"

    if (
        bbox_h is not None
        and height_ref_scene is not None
        and float(height_ref_scene) > 1e-3
        and bbox_h > 1.0
    ):
        # Relative pixel height vs standing reference is only available after
        # a height lock; still useful once established.
        # Without camera scale we use a softer threshold on aspect alone below.
        pass

    if kpts_abs is not None and isinstance(kpts_abs, np.ndarray) and kpts_abs.shape[0] >= 17:
        thr = float(config.kpt_conf_threshold)
        left_hip = pose_point(kpts_abs, "left_hip", conf_threshold=thr)
        right_hip = pose_point(kpts_abs, "right_hip", conf_threshold=thr)
        left_shoulder = pose_point(kpts_abs, "left_shoulder", conf_threshold=thr)
        right_shoulder = pose_point(kpts_abs, "right_shoulder", conf_threshold=thr)
        left_ankle = pose_point(kpts_abs, "left_ankle", conf_threshold=thr)
        right_ankle = pose_point(kpts_abs, "right_ankle", conf_threshold=thr)
        left_knee = pose_point(kpts_abs, "left_knee", conf_threshold=thr)
        right_knee = pose_point(kpts_abs, "right_knee", conf_threshold=thr)

        hip = _mid_xy(left_hip, right_hip)
        shoulder = _mid_xy(left_shoulder, right_shoulder)
        ankle = _mid_xy(left_ankle, right_ankle)
        knee = _mid_xy(left_knee, right_knee)

        if shoulder is not None and hip is not None and ankle is not None:
            torso_v = abs(float(hip[1]) - float(shoulder[1]))
            body_v = abs(float(ankle[1]) - float(shoulder[1]))
            body_h = abs(float(ankle[0]) - float(shoulder[0]))
            if body_v > 1e-3 and (body_h / body_v) >= float(config.lie_bbox_aspect):
                return "lying"
            if body_v > 1e-3 and torso_v > 1e-3:
                leg_v = abs(float(ankle[1]) - float(hip[1]))
                if leg_v < float(config.sit_bbox_height_ratio) * max(torso_v, 1.0):
                    return "sitting"
            if legs_are_bent(kpts_abs, config=config):
                return "sitting"

        if bbox_w is not None and bbox_h is not None and bbox_h > 1.0:
            if float(bbox_w) / float(bbox_h) >= float(config.lie_bbox_aspect) * 0.92:
                return "lying"

    if bbox_w is not None and bbox_h is not None and bbox_h > 1.0:
        aspect = float(bbox_w) / float(bbox_h)
        # Tall thin boxes are standing; short boxes without pose → sitting guess.
        if aspect < 0.55 and bbox_h >= 90.0:
            return "standing"
        if aspect >= 0.85 and aspect < float(config.lie_bbox_aspect):
            return "sitting"

    return "unknown"


def legs_are_bent(
    kpts_abs: Optional[np.ndarray],
    *,
    config: HumanGroundConfig,
) -> bool:
    """True when legs look bent (sitting) rather than roughly extended (standing).

    Standing has the knee near the hip→ankle segment. Sitting compresses hip–ankle
    span and/or pulls the knee off that segment.
    """
    if kpts_abs is None or not isinstance(kpts_abs, np.ndarray) or kpts_abs.shape[0] < 17:
        return False
    thr = float(config.kpt_conf_threshold)
    bent_votes = 0
    extended_votes = 0
    for side in ("left", "right"):
        hip = pose_point(kpts_abs, f"{side}_hip", conf_threshold=thr)
        knee = pose_point(kpts_abs, f"{side}_knee", conf_threshold=thr)
        ankle = pose_point(kpts_abs, f"{side}_ankle", conf_threshold=thr)
        if hip is None or knee is None:
            continue
        if ankle is None:
            hip_knee_v = abs(float(knee[1]) - float(hip[1]))
            if hip_knee_v < 25.0:
                bent_votes += 1
            continue
        hx, hy = float(hip[0]), float(hip[1])
        kx, ky = float(knee[0]), float(knee[1])
        ax, ay = float(ankle[0]), float(ankle[1])
        dx = ax - hx
        dy = ay - hy
        seg_len = math.hypot(dx, dy)
        if seg_len < 20.0:
            # Collapsed hip-ankle span → sitting / tucked legs.
            bent_votes += 1
            continue
        # Project knee onto hip→ankle segment.
        t = ((kx - hx) * dx + (ky - hy) * dy) / max(1e-9, seg_len * seg_len)
        proj_x = hx + t * dx
        proj_y = hy + t * dy
        residual = math.hypot(kx - proj_x, ky - proj_y)
        if 0.2 <= t <= 0.85 and residual <= 0.28 * seg_len:
            extended_votes += 1
            continue
        # Knee well off the extended leg line, or past the ankle/hip → bent.
        if residual > 0.28 * seg_len or t < 0.05 or t > 0.95:
            bent_votes += 1
            continue
        hip_ankle_v = abs(ay - hy)
        hip_ankle_h = abs(ax - hx)
        if hip_ankle_v > 1e-3 and hip_ankle_h > 0.85 * hip_ankle_v:
            bent_votes += 1
            continue
        extended_votes += 1
    if bent_votes <= 0:
        return False
    return bent_votes >= max(1, extended_votes)


def estimate_ankle_from_leg(
    kpts_abs: np.ndarray,
    side: str,
    *,
    conf_threshold: float,
) -> Optional[Tuple[float, float]]:
    hip = pose_point(kpts_abs, f"{side}_hip", conf_threshold=conf_threshold)
    knee = pose_point(kpts_abs, f"{side}_knee", conf_threshold=conf_threshold)
    if hip is None or knee is None:
        return None
    ankle_x = (2.0 * float(knee[0])) - float(hip[0])
    ankle_y = (2.0 * float(knee[1])) - float(hip[1])
    if not (math.isfinite(ankle_x) and math.isfinite(ankle_y)):
        return None
    return float(ankle_x), float(ankle_y)


def resolve_pose_floor_anchor(
    kpts_abs: Optional[np.ndarray],
    *,
    posture: Posture,
    config: HumanGroundConfig,
) -> Optional[PoseAnchorCandidate]:
    """Posture-aware image floor/contact anchor from pose keypoints."""
    if kpts_abs is None or not isinstance(kpts_abs, np.ndarray) or kpts_abs.shape[0] < 17:
        return None
    thr = float(config.kpt_conf_threshold)
    bent = legs_are_bent(kpts_abs, config=config)
    effective_posture = posture
    if effective_posture in ("unknown", "") and bent:
        effective_posture = "sitting"

    left_ankle = pose_point(kpts_abs, "left_ankle", conf_threshold=thr)
    right_ankle = pose_point(kpts_abs, "right_ankle", conf_threshold=thr)
    left_hip = pose_point(kpts_abs, "left_hip", conf_threshold=thr)
    right_hip = pose_point(kpts_abs, "right_hip", conf_threshold=thr)
    left_shoulder = pose_point(kpts_abs, "left_shoulder", conf_threshold=thr)
    right_shoulder = pose_point(kpts_abs, "right_shoulder", conf_threshold=thr)
    hip_mid = _mid_xy(left_hip, right_hip)
    shoulder_mid = _mid_xy(left_shoulder, right_shoulder)

    # Sitting / lying: prefer pelvis / torso contact, not noisy ankles.
    if effective_posture in ("sitting", "lying"):
        if hip_mid is not None:
            return PoseAnchorCandidate(
                u=float(hip_mid[0]),
                v=float(hip_mid[1]),
                source="pose_hip_floor" if effective_posture == "sitting" else "pose_body_floor",
                quality="good" if left_hip is not None and right_hip is not None else "estimated",
                quality_reason=f"posture={effective_posture}",
                height_lock_eligible=False,
                score=0.92 if effective_posture == "sitting" else 0.88,
            )
        if shoulder_mid is not None and effective_posture == "lying":
            return PoseAnchorCandidate(
                u=float(shoulder_mid[0]),
                v=float(shoulder_mid[1]),
                source="pose_body_floor",
                quality="estimated",
                quality_reason="posture=lying,hip_missing",
                height_lock_eligible=False,
                score=0.70,
            )

    # Standing (or unknown upright): ankles first.
    if left_ankle is not None and right_ankle is not None:
        return PoseAnchorCandidate(
            u=float(left_ankle[0] + right_ankle[0]) * 0.5,
            v=float(left_ankle[1] + right_ankle[1]) * 0.5,
            source="pose_ankle_floor",
            quality="good",
            height_lock_eligible=True,
            score=1.0,
        )
    if left_ankle is not None or right_ankle is not None:
        ankle = left_ankle if left_ankle is not None else right_ankle
        assert ankle is not None
        return PoseAnchorCandidate(
            u=float(ankle[0]),
            v=float(ankle[1]),
            source="pose_single_ankle_floor",
            quality="good",
            height_lock_eligible=True,
            score=0.90,
        )

    # Leg extension only when legs are not bent (Phase 2).
    if bent or effective_posture in ("sitting", "lying"):
        if hip_mid is not None:
            return PoseAnchorCandidate(
                u=float(hip_mid[0]),
                v=float(hip_mid[1]),
                source="pose_hip_floor",
                quality="estimated",
                quality_reason="bent_leg_blocks_pose_leg_floor",
                height_lock_eligible=False,
                score=0.72,
            )
        return None

    estimates: List[Tuple[float, float]] = []
    for side in ("left", "right"):
        ankle_est = estimate_ankle_from_leg(kpts_abs, side, conf_threshold=thr)
        if ankle_est is not None:
            estimates.append((float(ankle_est[0]), float(ankle_est[1])))
    if not estimates:
        return None
    if len(estimates) == 1:
        u, v = estimates[0]
    else:
        u = float(sum(p[0] for p in estimates) / len(estimates))
        v = float(sum(p[1] for p in estimates) / len(estimates))
    return PoseAnchorCandidate(
        u=float(u),
        v=float(v),
        source="pose_leg_floor",
        quality="estimated",
        quality_reason="pose_leg_extension",
        height_lock_eligible=False,
        score=0.55,
    )


def source_score(
    source: Optional[str],
    *,
    quality: str,
    depth_weight: float = 0.0,
    posture: Posture = "unknown",
) -> float:
    src = str(source or "").strip().lower()
    base = {
        "pose_ankle_floor": 1.00,
        "pose_single_ankle_floor": 0.90,
        "pose_hip_floor": 0.88,
        "pose_body_floor": 0.84,
        "pose_depth_fused": 1.05,
        "pose_floor_only": 0.95,
        "person_anchor_depth_fused": 0.92,
        "person_anchor_floor_only": 0.78,
        "person_mask_floor": 0.80,
        "pose_leg_floor": 0.55,
        "gravity_drop": 0.40,
        "anchor_hold": 0.30,
    }.get(src, 0.50)
    if quality == "good":
        base += 0.05
    elif quality == "estimated":
        base -= 0.05
    elif quality == "invalid":
        base -= 0.40
    base += 0.10 * max(0.0, min(1.0, float(depth_weight)))
    if posture in ("sitting", "lying"):
        if "ankle" in src or src == "pose_leg_floor":
            base -= 0.25
        if "hip" in src or "body" in src:
            base += 0.12
        if src == "gravity_drop":
            base -= 0.20
    return float(base)


def apply_source_hysteresis(
    state: PersonGroundState,
    *,
    candidate_source: str,
    candidate_score: float,
    config: HumanGroundConfig,
) -> Tuple[str, bool]:
    """Return (accepted_source, switched). Sticky unless score gain is large."""
    candidate_source = str(candidate_source)
    candidate_score = float(candidate_score)
    if state.sticky_source is None:
        state.sticky_source = candidate_source
        state.sticky_source_frames = 1
        state.sticky_source_score = candidate_score
        return candidate_source, False

    if candidate_source == state.sticky_source:
        state.sticky_source_frames = int(state.sticky_source_frames) + 1
        state.sticky_source_score = max(float(state.sticky_source_score), candidate_score)
        return candidate_source, False

    hold = int(config.source_hold_frames)
    gain = candidate_score - float(state.sticky_source_score)
    allow_switch = (
        int(state.sticky_source_frames) >= hold
        and gain >= float(config.source_switch_min_score_gain)
    ) or gain >= (float(config.source_switch_min_score_gain) + 0.35)

    # Always allow recovery from hold/gravity when a real observation appears.
    if state.sticky_source in ("anchor_hold", "gravity_drop") and candidate_score >= 0.70:
        allow_switch = True

    if not allow_switch:
        state.sticky_source_frames = int(state.sticky_source_frames) + 1
        return str(state.sticky_source), False

    state.sticky_source = candidate_source
    state.sticky_source_frames = 1
    state.sticky_source_score = candidate_score
    state.source_switch_count = int(state.source_switch_count) + 1
    return candidate_source, True


def _clamp_speed(vx: float, vz: float, max_speed: float) -> Tuple[float, float]:
    speed = math.hypot(float(vx), float(vz))
    if max_speed <= 0.0 or speed <= max_speed or speed <= 1e-9:
        return float(vx), float(vz)
    scale = max_speed / speed
    return float(vx) * scale, float(vz) * scale


def update_human_cv_filter(
    state: PersonGroundState,
    *,
    measurement: np.ndarray,
    floor_y: float,
    now_ts: float,
    quality: str,
    config: HumanGroundConfig,
    force_accept: bool = False,
) -> np.ndarray:
    """Constant-velocity XZ filter with adaptive noise and idle deadzone."""
    mx = float(measurement[0])
    mz = float(measurement[2])
    now_ts = float(now_ts)
    floor_y = float(floor_y)

    if state.world_x is None or state.world_z is None or float(state.filtered_ts) < 0.0:
        state.world_x = mx
        state.world_z = mz
        state.vel_world_x = 0.0
        state.vel_world_z = 0.0
        state.filtered_ts = now_ts
        return np.array([mx, floor_y, mz], dtype=np.float64)

    dt = now_ts - float(state.filtered_ts)
    if dt <= 0.0:
        return np.array([float(state.world_x), floor_y, float(state.world_z)], dtype=np.float64)

    if float(config.reset_after_s) > 0.0 and dt > float(config.reset_after_s):
        state.world_x = mx
        state.world_z = mz
        state.vel_world_x = 0.0
        state.vel_world_z = 0.0
        state.filtered_ts = now_ts
        state.motion_mode = "unknown"
        state.locked_world = None
        state.exit_motion_frames = 0
        return np.array([mx, floor_y, mz], dtype=np.float64)

    # Predict with constant velocity.
    pred_x = float(state.world_x) + float(state.vel_world_x) * dt
    pred_z = float(state.world_z) + float(state.vel_world_z) * dt

    innov_x = mx - pred_x
    innov_z = mz - pred_z
    innov_dist = math.hypot(innov_x, innov_z)

    # Physical gate (speed * dt + jump slack).
    allowed = float(config.max_jump_m) + float(config.max_speed_mps) * dt
    if (not force_accept) and allowed > 0.0 and innov_dist > allowed and innov_dist > 1e-9:
        scale = allowed / innov_dist
        innov_x *= scale
        innov_z *= scale
        innov_dist = allowed
        mx = pred_x + innov_x
        mz = pred_z + innov_z

    posture = str(state.posture or "unknown")
    mode = str(state.motion_mode or "unknown")

    # Adaptive gains from posture / mode / quality (Phase 4).
    if mode in ("idle", "sit", "lie") or posture in ("sitting", "lying"):
        q = float(config.process_noise_sit if posture in ("sitting", "lying") else config.process_noise_idle)
        r = float(config.meas_noise_sit if posture in ("sitting", "lying") else config.meas_noise_idle)
        alpha = min(float(config.alpha_weak), 0.18)
    else:
        q = float(config.process_noise_walk)
        r = float(config.meas_noise_good if quality == "good" else config.meas_noise_weak)
        alpha = float(config.alpha_good if quality == "good" else config.alpha_weak)

    # Scalar Kalman-like gain on position innovation.
    # P≈q, K = P/(P+R)
    kalman = q / max(1e-9, q + r)
    alpha = float(min(1.0, max(0.0, 0.55 * alpha + 0.45 * kalman)))
    beta = float(min(1.0, max(0.0, alpha * float(config.beta_scale))))

    # Idle deadzone: absorb small innovations instead of vibrating.
    deadzone = float(config.idle_deadzone_m)
    if posture in ("sitting", "lying"):
        deadzone = max(deadzone, float(config.idle_deadzone_sit_m))
    if mode in ("idle", "sit", "lie") and (not force_accept) and innov_dist <= deadzone:
        # Hold position; gently decay velocity.
        state.vel_world_x *= 0.50
        state.vel_world_z *= 0.50
        state.filtered_ts = now_ts
        if state.locked_world is not None:
            state.world_x = float(state.locked_world[0])
            state.world_z = float(state.locked_world[1])
        return np.array([float(state.world_x), floor_y, float(state.world_z)], dtype=np.float64)

    next_x = pred_x + alpha * innov_x
    next_z = pred_z + alpha * innov_z

    if dt >= float(config.min_dt_s):
        meas_vx = innov_x / dt
        meas_vz = innov_z / dt
        next_vx = float(state.vel_world_x) + beta * (meas_vx - float(state.vel_world_x))
        next_vz = float(state.vel_world_z) + beta * (meas_vz - float(state.vel_world_z))
    else:
        next_vx = float(state.vel_world_x)
        next_vz = float(state.vel_world_z)

    # Acceleration clamp via velocity delta.
    if dt >= float(config.min_dt_s) and float(config.max_accel_mps2) > 0.0:
        dvx = next_vx - float(state.vel_world_x)
        dvz = next_vz - float(state.vel_world_z)
        max_dv = float(config.max_accel_mps2) * dt
        dv = math.hypot(dvx, dvz)
        if dv > max_dv and dv > 1e-9:
            scale = max_dv / dv
            next_vx = float(state.vel_world_x) + dvx * scale
            next_vz = float(state.vel_world_z) + dvz * scale

    next_vx, next_vz = _clamp_speed(next_vx, next_vz, float(config.max_speed_mps))

    state.world_x = float(next_x)
    state.world_z = float(next_z)
    state.vel_world_x = float(next_vx)
    state.vel_world_z = float(next_vz)
    state.filtered_ts = now_ts
    return np.array([float(next_x), floor_y, float(next_z)], dtype=np.float64)


def update_motion_mode(
    state: PersonGroundState,
    *,
    now_ts: float,
    image_foot_uv: Optional[Tuple[float, float]],
    config: HumanGroundConfig,
) -> None:
    """Phase 1 stationary lock + trail-append gate."""
    now_ts = float(now_ts)
    speed = 0.0
    if state.world_x is not None and state.world_z is not None:
        speed = math.hypot(float(state.vel_world_x), float(state.vel_world_z))

    residual = 0.0
    if state.locked_world is not None and state.world_x is not None and state.world_z is not None:
        residual = math.hypot(
            float(state.world_x) - float(state.locked_world[0]),
            float(state.world_z) - float(state.locked_world[1]),
        )
    elif state.world_x is not None and state.last_good_world is not None:
        residual = math.hypot(
            float(state.world_x) - float(state.last_good_world[0]),
            float(state.world_z) - float(state.last_good_world[2]),
        )
    state.residual_m.append(float(residual))
    if state.residual_m:
        state.idle_jitter_m = float(sum(state.residual_m) / max(1, len(state.residual_m)))

    image_static = True
    if image_foot_uv is not None:
        u, v = float(image_foot_uv[0]), float(image_foot_uv[1])
        state.image_foot_u = u
        state.image_foot_v = v
        state.image_foot_history.append((now_ts, u, v))
        if len(state.image_foot_history) >= 3:
            recent = list(state.image_foot_history)[-min(8, len(state.image_foot_history)) :]
            us = [p[1] for p in recent]
            vs = [p[2] for p in recent]
            rms = math.sqrt(
                (sum((x - sum(us) / len(us)) ** 2 for x in us) + sum((y - sum(vs) / len(vs)) ** 2 for y in vs))
                / max(1, len(recent))
            )
            image_static = rms <= float(config.static_px_threshold)

    posture = str(state.posture or "unknown")
    low_speed = speed <= float(config.static_speed_mps)
    deadzone = float(config.idle_deadzone_sit_m if posture in ("sitting", "lying") else config.idle_deadzone_m)
    low_residual = float(state.idle_jitter_m) <= max(deadzone, float(config.static_speed_mps) * 2.0)

    if state.motion_mode not in ("idle", "sit", "lie"):
        if low_speed and image_static and low_residual:
            if float(state.idle_since_ts or 0.0) <= 0.0:
                state.idle_since_ts = now_ts
            elif (now_ts - float(state.idle_since_ts)) >= float(config.static_enter_s):
                if posture == "lying":
                    state.motion_mode = "lie"
                elif posture == "sitting":
                    state.motion_mode = "sit"
                else:
                    state.motion_mode = "idle"
                if state.world_x is not None and state.world_z is not None:
                    state.locked_world = (float(state.world_x), float(state.world_z))
                state.vel_world_x = 0.0
                state.vel_world_z = 0.0
                state.exit_motion_frames = 0
        else:
            state.idle_since_ts = 0.0
            if speed > float(config.static_exit_speed_mps):
                state.motion_mode = "walk"
    else:
        # Already idle-like: require sustained motion to exit.
        moving = (speed > float(config.static_exit_speed_mps)) or (not image_static and residual > deadzone)
        if moving:
            state.exit_motion_frames = int(state.exit_motion_frames) + 1
        else:
            state.exit_motion_frames = 0
            if state.locked_world is not None:
                state.world_x = float(state.locked_world[0])
                state.world_z = float(state.locked_world[1])
            state.vel_world_x *= 0.4
            state.vel_world_z *= 0.4

        if int(state.exit_motion_frames) >= int(config.static_exit_frames):
            state.motion_mode = "walk"
            state.locked_world = None
            state.idle_since_ts = 0.0
            state.exit_motion_frames = 0
        else:
            # Stay locked.
            if posture == "lying":
                state.motion_mode = "lie"
            elif posture == "sitting":
                state.motion_mode = "sit"
            else:
                state.motion_mode = "idle"

    state.trail_append_allowed = state.motion_mode not in ("idle", "sit", "lie")
    if state.motion_mode == "unknown" and speed > float(config.static_exit_speed_mps):
        state.motion_mode = "walk"
        state.trail_append_allowed = True


def rdp_simplify(
    points: Sequence[Tuple[float, float, float]],
    *,
    epsilon: float,
) -> List[Tuple[float, float, float]]:
    """Ramer–Douglas–Peucker on (t, x, z) points using XZ distance."""
    pts = list(points)
    if len(pts) < 3 or epsilon <= 0.0:
        return pts

    def _dist_point_to_segment(
        p: Tuple[float, float, float],
        a: Tuple[float, float, float],
        b: Tuple[float, float, float],
    ) -> float:
        ax, az = float(a[1]), float(a[2])
        bx, bz = float(b[1]), float(b[2])
        px, pz = float(p[1]), float(p[2])
        dx = bx - ax
        dz = bz - az
        denom = dx * dx + dz * dz
        if denom <= 1e-12:
            return math.hypot(px - ax, pz - az)
        t = ((px - ax) * dx + (pz - az) * dz) / denom
        t = max(0.0, min(1.0, t))
        qx = ax + t * dx
        qz = az + t * dz
        return math.hypot(px - qx, pz - qz)

    def _rdp(segment: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        if len(segment) < 3:
            return segment
        start = segment[0]
        end = segment[-1]
        idx = -1
        max_dist = -1.0
        for i in range(1, len(segment) - 1):
            d = _dist_point_to_segment(segment[i], start, end)
            if d > max_dist:
                max_dist = d
                idx = i
        if max_dist > float(epsilon) and idx > 0:
            left = _rdp(segment[: idx + 1])
            right = _rdp(segment[idx:])
            return left[:-1] + right
        return [start, end]

    return _rdp(pts)


def commit_path_point(
    points: Deque[Tuple[float, float, float]],
    *,
    ts: float,
    x: float,
    z: float,
    config: HumanGroundConfig,
    append_allowed: bool,
) -> bool:
    """Commit a path sample with min-step and optional simplification.

    Returns True if a new point was appended (or last point updated).
    """
    if not append_allowed:
        return False

    ts = float(ts)
    x = float(x)
    z = float(z)
    min_step = float(config.path_min_step_m)

    if points:
        prev_ts, prev_x, prev_z = points[-1]
        dist = math.hypot(x - float(prev_x), z - float(prev_z))
        if dist < min_step:
            # Replace last sample to keep head fresh without growing noise path.
            points[-1] = (float(prev_ts), x, z)
            return True

    points.append((ts, x, z))

    # Bound then simplify history (Phase 6) — never the single latest point alone.
    max_points = int(config.path_max_points)
    while len(points) > max_points:
        points.popleft()

    if len(points) >= 6 and float(config.path_simplify_epsilon_m) > 0.0:
        simplified = rdp_simplify(list(points), epsilon=float(config.path_simplify_epsilon_m))
        # Always keep the newest raw head.
        if simplified and points:
            if simplified[-1] != points[-1]:
                simplified[-1] = points[-1]
        if len(simplified) >= 2 and len(simplified) < len(points):
            points.clear()
            points.extend(simplified)
    return True


def commit_image_path_point(
    points: Deque,
    *,
    ts: float,
    x: float,
    y: float,
    min_step_px: float,
    simplify_epsilon_px: float,
    max_points: int,
    append_allowed: bool,
    point_factory,
) -> bool:
    """Image-space path commit for OSD trails (same semantics as metric path)."""
    if not append_allowed:
        return False
    ts = float(ts)
    x = float(x)
    y = float(y)
    if points:
        prev = points[-1]
        prev_x = float(getattr(prev, "x", prev[1] if isinstance(prev, tuple) else x))
        prev_y = float(getattr(prev, "y", prev[2] if isinstance(prev, tuple) else y))
        prev_ts = float(getattr(prev, "ts", prev[0] if isinstance(prev, tuple) else ts))
        dist = math.hypot(x - prev_x, y - prev_y)
        if dist < float(min_step_px):
            points[-1] = point_factory(ts=prev_ts, x=x, y=y)
            return True
    points.append(point_factory(ts=ts, x=x, y=y))
    while len(points) > max(2, int(max_points)):
        points.popleft()
    if len(points) >= 6 and float(simplify_epsilon_px) > 0.0:
        as_tuples = []
        for p in points:
            as_tuples.append(
                (
                    float(getattr(p, "ts", p[0] if isinstance(p, tuple) else ts)),
                    float(getattr(p, "x", p[1] if isinstance(p, tuple) else x)),
                    float(getattr(p, "y", p[2] if isinstance(p, tuple) else y)),
                )
            )
        simplified = rdp_simplify(as_tuples, epsilon=float(simplify_epsilon_px))
        if simplified and as_tuples and simplified[-1] != as_tuples[-1]:
            simplified[-1] = as_tuples[-1]
        if len(simplified) >= 2 and len(simplified) < len(points):
            points.clear()
            for t_i, x_i, y_i in simplified:
                points.append(point_factory(ts=t_i, x=x_i, y=y_i))
    return True


class PersonGroundStateStore:
    """Keyed store for PersonGroundState with TTL prune."""

    def __init__(self, *, ttl_s: float = 3.0, prune_interval_s: float = 1.0) -> None:
        self._states: Dict[Hashable, PersonGroundState] = {}
        self.ttl_s = max(0.0, float(ttl_s))
        self.prune_interval_s = max(0.05, float(prune_interval_s))
        self._last_prune_ts = 0.0

    def get(self, key: Hashable) -> Optional[PersonGroundState]:
        return self._states.get(key)

    def get_or_create(self, key: Hashable) -> PersonGroundState:
        state = self._states.get(key)
        if state is None:
            state = PersonGroundState()
            self._states[key] = state
        return state

    def prune(self, now_ts: float) -> None:
        now_ts = float(now_ts)
        if (now_ts - float(self._last_prune_ts)) < float(self.prune_interval_s):
            return
        self._last_prune_ts = now_ts
        if self.ttl_s <= 0.0:
            self._states.clear()
            return
        expired = [k for k, st in self._states.items() if (now_ts - float(st.ts or 0.0)) > float(self.ttl_s)]
        for key in expired:
            self._states.pop(key, None)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._states

    def __len__(self) -> int:
        return len(self._states)

    @property
    def states(self) -> Dict[Hashable, PersonGroundState]:
        return self._states


__all__ = [
    "POSE_KPT_INDEX",
    "HumanGroundConfig",
    "PoseAnchorCandidate",
    "PersonGroundState",
    "PersonGroundStateStore",
    "apply_source_hysteresis",
    "classify_posture",
    "commit_image_path_point",
    "commit_path_point",
    "estimate_ankle_from_leg",
    "legs_are_bent",
    "pose_point",
    "rdp_simplify",
    "resolve_pose_floor_anchor",
    "source_score",
    "update_human_cv_filter",
    "update_motion_mode",
]
