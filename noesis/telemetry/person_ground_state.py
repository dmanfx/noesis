"""Person ground-state estimator for human-realistic BEV/OSD pathing.

Phases implemented here (shared by baseline and V3DT analytics hooks):

1. Stationary lock — freeze world when idle; trails do not append.
2. Source hysteresis — sticky world_source selection; reject bent-leg ankle fakes.
3. Posture-aware contact — only observed ground-contact anatomy is projected.
4. Human constant-velocity filter with adaptive process/measurement noise.
5. PersonGroundState — single producer state consumed by BEV + OSD trails.
6. Path simplification — RDP / collinear merge on committed history only.
"""

from __future__ import annotations

import math
from copy import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

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
    source_replacement_frames: int = 3
    source_switch_min_score_gain: float = 0.20

    # Phase 3 — posture
    sit_bbox_height_ratio: float = 0.62
    lie_bbox_aspect: float = 1.35
    bent_leg_vertical_ratio: float = 0.58
    kpt_conf_threshold: float = 0.35
    occlusion_exit_frames: int = 3
    occlusion_upright_memory_s: float = 15.0
    occlusion_bbox_height_ratio: float = 0.76
    occlusion_bbox_shoulder_ratio: float = 0.78
    upright_reference_alpha: float = 0.18

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
    reacquire_samples: int = 3
    reacquire_max_gap_s: float = 0.75
    # A rejected metric observation may advance the process model only for
    # this short horizon.  After that, keep the process state at the bounded
    # edge of the last-good window instead of integrating velocity forever.
    rejected_prediction_horizon_s: float = 0.40

    # Projective image-motion continuation is intentionally conservative. It
    # may update the bounded process posterior for a missing/rejected metric
    # observation, but never replaces a fresh floor/depth measurement as the
    # accepted metric authority.
    projective_max_bbox_step_px: float = 64.0
    projective_max_bbox_speed_px_s: float = 1200.0
    projective_max_scale_ratio: float = 1.45
    projective_world_slack_m: float = 0.35

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
        object.__setattr__(self, "source_replacement_frames", max(1, int(self.source_replacement_frames)))
        object.__setattr__(self, "source_switch_min_score_gain", max(0.0, float(self.source_switch_min_score_gain)))
        object.__setattr__(self, "sit_bbox_height_ratio", float(min(1.0, max(0.05, float(self.sit_bbox_height_ratio)))))
        object.__setattr__(self, "lie_bbox_aspect", max(0.5, float(self.lie_bbox_aspect)))
        object.__setattr__(self, "bent_leg_vertical_ratio", float(min(1.0, max(0.05, float(self.bent_leg_vertical_ratio)))))
        object.__setattr__(self, "kpt_conf_threshold", float(min(1.0, max(0.0, float(self.kpt_conf_threshold)))))
        object.__setattr__(self, "occlusion_exit_frames", max(1, int(self.occlusion_exit_frames)))
        object.__setattr__(
            self,
            "occlusion_upright_memory_s",
            max(0.0, float(self.occlusion_upright_memory_s)),
        )
        object.__setattr__(
            self,
            "occlusion_bbox_height_ratio",
            float(min(1.0, max(0.05, float(self.occlusion_bbox_height_ratio)))),
        )
        object.__setattr__(
            self,
            "occlusion_bbox_shoulder_ratio",
            float(min(1.0, max(0.05, float(self.occlusion_bbox_shoulder_ratio)))),
        )
        object.__setattr__(
            self,
            "upright_reference_alpha",
            float(min(1.0, max(0.0, float(self.upright_reference_alpha)))),
        )
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
        object.__setattr__(self, "reacquire_samples", max(2, int(self.reacquire_samples)))
        object.__setattr__(self, "reacquire_max_gap_s", max(0.0, float(self.reacquire_max_gap_s)))
        object.__setattr__(
            self,
            "rejected_prediction_horizon_s",
            max(0.0, float(self.rejected_prediction_horizon_s)),
        )
        object.__setattr__(
            self,
            "projective_max_bbox_step_px",
            max(1.0, float(self.projective_max_bbox_step_px)),
        )
        object.__setattr__(
            self,
            "projective_max_bbox_speed_px_s",
            max(1.0, float(self.projective_max_bbox_speed_px_s)),
        )
        object.__setattr__(
            self,
            "projective_max_scale_ratio",
            max(1.01, float(self.projective_max_scale_ratio)),
        )
        object.__setattr__(
            self,
            "projective_world_slack_m",
            max(0.0, float(self.projective_world_slack_m)),
        )
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
    # Stable anatomical/evidence basis used to prove that consecutive image
    # observations describe the same contact, not an ankle/source swap.
    contact_basis: Optional[str] = None


@dataclass(frozen=True)
class LowerBodyOcclusionAssessment:
    """Current lower-body visibility state for one tracked person."""

    active: bool
    level: str = "none"
    confidence: float = 0.0
    reason: Optional[str] = None
    visible_ankles: int = 0
    visible_knees: int = 0
    visible_hips: int = 0
    visible_shoulders: int = 0
    bbox_height_ratio: Optional[float] = None
    bbox_shoulder_ratio: Optional[float] = None


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

    # Revisioned world-frame binding for the metric filter. A calibration
    # reload can change the camera-to-world transform while a numeric tracker
    # ID remains alive. Keeping this identity on the state makes it impossible
    # for a rejected-frame hold or CV prediction from the previous world to be
    # published under the new world's revision.
    world_frame_id: Optional[str] = None
    world_frame_revision: Optional[str] = None
    world_transform_sha256: Optional[str] = None

    # Phase 1 / 3
    motion_mode: MotionMode = "unknown"
    posture: Posture = "unknown"
    idle_since_ts: float = 0.0
    exit_motion_frames: int = 0
    locked_world: Optional[Tuple[float, float]] = None
    residual_m: Deque[float] = field(default_factory=lambda: deque(maxlen=12))
    image_foot_history: Deque[Tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=12))
    # Exact-frame image evidence is deliberately separate from the smoothed
    # history above.  It is used before the world filter to unlock a genuinely
    # walking idle track and to authorize outlier reacquisition.
    image_motion_frame_id: int = -1
    image_motion_contact_basis: Optional[str] = None
    image_motion_window: Deque[Tuple[int, str, float, float, float, float]] = field(
        default_factory=lambda: deque(maxlen=3)
    )
    image_motion_streak: int = 0
    image_motion_supported: bool = False
    # Exact-frame detector-box continuity used only to gate a short stationary
    # hold when a seated target has no current ground contact.  This is compact
    # scalar state; it does not retain or copy image data.
    bbox_motion_frame_id: int = -1
    bbox_center_u: Optional[float] = None
    bbox_bottom_v: Optional[float] = None
    bbox_geometry: Optional[Tuple[float, float, float, float]] = None
    bbox_stationary_streak: int = 0
    bbox_stationary_supported: bool = False
    # Last *physically accepted* image geometry.  These fields are deliberately
    # distinct from the current image evidence and from bbox_stationarity: a
    # rejected/missing frame must never become a new predictor origin.
    last_accepted_image_foot: Optional[Tuple[float, float]] = None
    last_accepted_bbox_geometry: Optional[Tuple[float, float, float, float]] = None
    last_accepted_image_ts: float = -1.0
    last_accepted_lifecycle_generation: Optional[int] = None
    last_full_body_ts: float = -1.0
    last_non_upright_ts: float = -1.0
    upright_bbox_height_px: Optional[float] = None
    upright_bbox_shoulder_ratio: Optional[float] = None
    last_bbox_height_px: Optional[float] = None
    last_bbox_ts: float = -1.0
    lower_body_occluded: bool = False
    lower_body_occlusion_level: str = "none"
    lower_body_occlusion_confidence: float = 0.0
    lower_body_occlusion_reason: Optional[str] = None
    lower_body_clear_frames: int = 0
    # Learned calibrated heights for visible upper-body reference points.
    # Values are fractions of the track's standing height and stay process-local.
    body_plane_height_fractions: Dict[str, float] = field(default_factory=dict)

    # Phase 2
    sticky_source: Optional[str] = None
    sticky_source_frames: int = 0
    sticky_source_score: float = 0.0
    source_switch_count: int = 0
    # A lower-scoring source must still be able to take over when the current
    # sticky source has disappeared.  Track a *consecutive* replacement streak
    # instead of treating rejected candidates as more observations of the old
    # source (which could otherwise pin a vanished pose source forever).
    source_candidate: Optional[str] = None
    source_candidate_frames: int = 0

    # Phase 5 public mirrors
    image_foot_u: Optional[float] = None
    image_foot_v: Optional[float] = None
    trail_append_allowed: bool = True
    idle_jitter_m: float = 0.0

    # Physical-admission diagnostics.  These describe the current measurement,
    # not the filtered output, and are intentionally shared by DS8 and DS9.
    measurement_accepted: bool = True
    measurement_rejection_reason: Optional[str] = None
    measurement_innovation_m: float = 0.0
    measurement_allowed_m: float = 0.0
    reacquire_candidate_x: Optional[float] = None
    reacquire_candidate_z: Optional[float] = None
    reacquire_candidate_ts: float = -1.0
    reacquire_candidate_basis: Optional[str] = None
    reacquire_count: int = 0
    reacquired: bool = False
    trail_break_required: bool = False
    trail_segment_id: int = 0

    # Final canonical-output continuity.  The filter normally owns the same
    # state, but this independent compact watermark protects the publication
    # boundary when an alternate continuation path or duplicate SDK object
    # mutates the process state between emitted observations.  Media PTS is
    # preferred so file replay and a delayed callback obey physical stream
    # time rather than processing speed.
    last_output_world_x: Optional[float] = None
    last_output_world_z: Optional[float] = None
    last_output_media_pts_ns: Optional[int] = None
    last_output_filter_ts: float = -1.0
    last_output_trail_segment_id: int = 0

    # A live-source change is provisional until the physical measurement gate
    # accepts the corresponding point.  These fields never leave the process.
    pending_source_previous: Optional[
        Tuple[Optional[str], int, float, int, Optional[str], int]
    ] = None

    # Internal process anchor for a run of rejected observations.  Runtime
    # tracks normally have ``last_good_world``; this fallback also keeps the
    # SDK-neutral filter bounded when it is used directly by a caller that has
    # not mirrored the last-good publication fields yet.
    rejection_anchor_x: Optional[float] = None
    rejection_anchor_z: Optional[float] = None
    rejection_anchor_ts: float = -1.0
    # Exact state immediately before the current frame's rejection update.
    # A second weak/projective hypothesis in the same frame uses this to prove
    # continuity against the last process posterior, not merely against the
    # older fixed rejection anchor.
    rejection_previous_world_x: Optional[float] = None
    rejection_previous_world_z: Optional[float] = None
    rejection_previous_world_ts: float = -1.0

    def as_public_fields(self) -> Dict[str, Any]:
        return {
            "motion_mode": str(self.motion_mode),
            "posture": str(self.posture),
            "trail_append_allowed": bool(self.trail_append_allowed),
            "idle_jitter_m": float(self.idle_jitter_m) if math.isfinite(float(self.idle_jitter_m)) else None,
            "source_switch_count": int(self.source_switch_count),
            "sticky_world_source": str(self.sticky_source) if self.sticky_source else None,
            "lower_body_occluded": bool(self.lower_body_occluded),
            "lower_body_occlusion_level": str(self.lower_body_occlusion_level),
            "lower_body_occlusion_confidence": (
                float(self.lower_body_occlusion_confidence)
                if math.isfinite(float(self.lower_body_occlusion_confidence))
                else None
            ),
            "lower_body_occlusion_reason": (
                str(self.lower_body_occlusion_reason)
                if self.lower_body_occlusion_reason
                else None
            ),
            "world_measurement_accepted": bool(self.measurement_accepted),
            "world_rejection_reason": (
                str(self.measurement_rejection_reason)
                if self.measurement_rejection_reason
                else None
            ),
            "world_innovation_m": (
                float(self.measurement_innovation_m)
                if math.isfinite(float(self.measurement_innovation_m))
                else None
            ),
            "world_innovation_limit_m": (
                float(self.measurement_allowed_m)
                if math.isfinite(float(self.measurement_allowed_m))
                else None
            ),
            "world_reacquire_count": int(self.reacquire_count),
            "world_reacquired": bool(self.reacquired),
            "world_contact_basis": (
                str(self.image_motion_contact_basis)
                if self.image_motion_contact_basis
                else None
            ),
            "world_image_motion_supported": bool(self.image_motion_supported),
            "world_image_motion_streak": int(self.image_motion_streak),
            "trail_break_required": bool(self.trail_break_required),
            "trail_segment_id": int(self.trail_segment_id),
        }


def _clear_reacquire_candidate(state: PersonGroundState) -> None:
    state.reacquire_candidate_x = None
    state.reacquire_candidate_z = None
    state.reacquire_candidate_ts = -1.0
    state.reacquire_candidate_basis = None
    state.reacquire_count = 0


def _clear_rejection_anchor(state: PersonGroundState) -> None:
    state.rejection_anchor_x = None
    state.rejection_anchor_z = None
    state.rejection_anchor_ts = -1.0
    state.rejection_previous_world_x = None
    state.rejection_previous_world_z = None
    state.rejection_previous_world_ts = -1.0


def world_frame_binding_from_calibration(
    calibration: Any,
    *,
    default_frame_id: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract the immutable world identity from either calibration snapshot.

    The native calibration manager exposes a ``WorldCalibrationSnapshot``;
    SDK-neutral tests and older callers use the telemetry snapshot.  Keep this
    adapter deliberately duck-typed so the filter does not import the runtime
    calibration manager or retain a mutable provider object.
    """

    def _text(value: Any) -> Optional[str]:
        try:
            value = str(value).strip()
        except Exception:
            return None
        return value or None

    if calibration is None:
        return _text(default_frame_id), None, None

    contract = getattr(calibration, "frame_contract", None)
    frame = getattr(calibration, "world_frame", None)
    frame_id = _text(getattr(calibration, "world_frame_id", None))
    revision = _text(getattr(calibration, "world_frame_revision", None))
    if frame is not None:
        if frame_id is None:
            frame_id = _text(getattr(frame, "frame_id", None))
        if revision is None:
            revision = _text(getattr(frame, "revision", None))
    if frame_id is None and contract is not None:
        target_frame = getattr(contract, "target_frame", None)
        if target_frame is not None:
            frame_id = _text(getattr(target_frame, "frame_id", None))
            if revision is None:
                revision = _text(getattr(target_frame, "revision", None))
    if frame_id is None:
        frame_id = _text(default_frame_id)

    transform = _text(getattr(calibration, "frame_transform_sha256", None))
    if transform is None and contract is not None:
        transform = _text(getattr(contract, "transform_sha256", None))
    # The alignment hash is a useful conservative fallback for snapshot types
    # that do not expose the transform digest. The explicit frame revision
    # remains the primary identity when both are present.
    if transform is None:
        transform = _text(getattr(calibration, "world_alignment_sha256", None))
    return frame_id, revision, transform


def bind_world_frame(
    state: PersonGroundState,
    *,
    world_frame_id: Optional[str],
    world_frame_revision: Optional[str],
    world_transform_sha256: Optional[str],
) -> bool:
    """Bind a filter state to one world revision, resetting on a change.

    Returns ``True`` when the prior state belonged to a different frame or
    transform. The reset is in-place so references held by the current
    tracking callback remain safe, while all metric history, CV velocity,
    seated lock, accepted image origin, and rejected-frame anchors are cleared.
    Lifecycle identity is retained because a calibration change is not a
    tracker-ID reuse; trails are explicitly broken through a new segment.
    """

    requested = (
        str(world_frame_id).strip() if world_frame_id else None,
        str(world_frame_revision).strip() if world_frame_revision else None,
        str(world_transform_sha256).strip() if world_transform_sha256 else None,
    )
    current = (
        str(state.world_frame_id).strip() if state.world_frame_id else None,
        str(state.world_frame_revision).strip() if state.world_frame_revision else None,
        str(state.world_transform_sha256).strip()
        if state.world_transform_sha256
        else None,
    )
    if current == (None, None, None):
        state.world_frame_id, state.world_frame_revision, state.world_transform_sha256 = requested
        return False
    if current == requested:
        return False

    lifecycle_generation = getattr(state, "tracker_lifecycle_generation", None)
    try:
        prior_segment = int(state.trail_segment_id)
    except Exception:
        prior_segment = 0
    state.__dict__.clear()
    fresh = PersonGroundState(
        world_frame_id=requested[0],
        world_frame_revision=requested[1],
        world_transform_sha256=requested[2],
    )
    state.__dict__.update(fresh.__dict__)
    state.trail_break_required = True
    state.trail_segment_id = max(0, prior_segment) + 1
    if lifecycle_generation is not None:
        setattr(state, "tracker_lifecycle_generation", lifecycle_generation)
    return True


def world_frame_matches_calibration(
    track: Mapping[str, Any],
    calibration: Any,
    *,
    default_frame_id: Optional[str] = None,
) -> bool:
    """Return whether a published world row belongs to the active snapshot.

    Missing one side of an explicit revision is treated as a mismatch. This
    is intentional at the OSD boundary: an unknown frame cannot safely be
    reprojected with a known active transform.
    """

    expected = world_frame_binding_from_calibration(
        calibration,
        default_frame_id=default_frame_id,
    )
    observed = (
        str(track.get("world_frame") or "").strip() or None,
        str(track.get("world_frame_revision") or "").strip() or None,
        str(track.get("world_transform_sha256") or "").strip() or None,
    )
    expected_frame, expected_revision, expected_transform = expected
    observed_frame, observed_revision, observed_transform = observed
    if expected_frame is not None and observed_frame != expected_frame:
        return False
    if expected_revision is not None and observed_revision != expected_revision:
        return False
    if observed_revision is not None and expected_revision is None:
        return False
    if expected_transform is not None and observed_transform is not None:
        if observed_transform != expected_transform:
            return False
    return True


def _begin_filter_measurement(state: PersonGroundState) -> None:
    state.measurement_accepted = True
    state.measurement_rejection_reason = None
    state.measurement_innovation_m = 0.0
    state.measurement_allowed_m = 0.0
    state.reacquired = False
    state.trail_break_required = False


def mark_world_measurement_unavailable(
    state: PersonGroundState,
    *,
    reason: str,
) -> None:
    """Record an honest no-measurement frame without advancing the filter."""

    _begin_filter_measurement(state)
    state.measurement_accepted = False
    state.measurement_rejection_reason = str(reason or "measurement_unavailable")
    state.measurement_innovation_m = math.nan
    state.measurement_allowed_m = math.nan
    _clear_reacquire_candidate(state)


def advance_human_cv_prediction(
    state: PersonGroundState,
    *,
    floor_y: float,
    now_ts: float,
    config: HumanGroundConfig,
    reason: str = "world_measurement_unavailable",
) -> Optional[np.ndarray]:
    """Advance the one canonical ground state when no metric sample exists.

    Missing/rejected observations still have a physical output: the bounded
    constant-velocity process model.  Keeping that prediction in
    ``PersonGroundState`` prevents the display path from switching to a
    second, display-only point and then snapping back to the filter state when
    metric evidence returns.  This function does not create a new metric
    anchor and never starts a trail segment.
    """

    if state.world_x is None or state.world_z is None or float(state.filtered_ts) < 0.0:
        return None
    now_ts = float(now_ts)
    previous_ts = float(state.filtered_ts)
    if not math.isfinite(now_ts) or now_ts < previous_ts:
        state.measurement_accepted = False
        state.measurement_rejection_reason = "non_monotonic_timestamp"
        state.trail_append_allowed = False
        return np.array(
            [float(state.world_x), float(floor_y), float(state.world_z)],
            dtype=np.float64,
        )
    if now_ts == previous_ts:
        state.measurement_accepted = False
        state.measurement_rejection_reason = str(reason or "world_measurement_unavailable")
        state.trail_append_allowed = False
        return np.array(
            [float(state.world_x), float(floor_y), float(state.world_z)],
            dtype=np.float64,
        )

    gate_dt = float(now_ts - previous_ts)
    if float(config.reset_after_s) > 0.0:
        gate_dt = min(gate_dt, float(config.reset_after_s))
    pred_x = float(state.world_x) + float(state.vel_world_x) * gate_dt
    pred_z = float(state.world_z) + float(state.vel_world_z) * gate_dt
    return _reject_with_cv_time_update(
        state,
        pred_x=pred_x,
        pred_z=pred_z,
        now_ts=now_ts,
        floor_y=float(floor_y),
        reason=str(reason or "world_measurement_unavailable"),
        config=config,
    )


def integrate_projective_ground_observation(
    state: PersonGroundState,
    *,
    measurement: np.ndarray,
    floor_y: float,
    now_ts: float,
    config: HumanGroundConfig,
) -> Optional[np.ndarray]:
    """Integrate an admissible image-motion floor point into filter state.

    The caller must apply the calibrated world-fusion policy before invoking
    this function.  The point is still weak evidence: it is passed through
    the exact physical CV admission and posterior-speed gate, does not update
    ``last_good_world``/the accepted image origin, and is marked estimated for
    telemetry.  A rejected projective point leaves the bounded process state
    as-is and returns ``None``; it is never clamped into a plausible-looking
    relocation.
    """

    try:
        candidate = np.asarray(measurement, dtype=np.float64)
        if candidate.shape[0] < 3 or not np.all(np.isfinite(candidate[:3])):
            return None
        if state.world_x is None or state.world_z is None or float(state.filtered_ts) < 0.0:
            return None
    except Exception:
        return None

    previous_world_x = float(
        state.rejection_previous_world_x
        if state.rejection_previous_world_x is not None
        else state.world_x
    )
    previous_world_z = float(
        state.rejection_previous_world_z
        if state.rejection_previous_world_z is not None
        else state.world_z
    )
    previous_filtered_ts = float(
        state.rejection_previous_world_ts
        if math.isfinite(float(state.rejection_previous_world_ts))
        and float(state.rejection_previous_world_ts) >= 0.0
        else (
            state.rejection_anchor_ts
            if math.isfinite(float(state.rejection_anchor_ts))
            and float(state.rejection_anchor_ts) >= 0.0
            else state.filtered_ts
        )
    )

    # A rejected metric candidate may already have advanced the process state
    # to this exact frame timestamp.  Evaluate the independent image-motion
    # candidate from the fixed rejection anchor in that case; otherwise the
    # normal filter sees a zero dt and incorrectly reports a timestamp reject.
    # The temporary state keeps the original bounded posterior untouched when
    # the projective candidate fails the same physical gate.
    evaluation_state = state
    rewound_from_rejection = bool(
        not bool(state.measurement_accepted)
        and state.rejection_anchor_x is not None
        and state.rejection_anchor_z is not None
        and math.isfinite(float(state.rejection_anchor_ts))
        and math.isfinite(float(state.filtered_ts))
        and float(state.filtered_ts) == float(now_ts)
        and float(state.rejection_anchor_ts) < float(now_ts)
    )
    if rewound_from_rejection:
        evaluation_state = copy(state)
        evaluation_state.world_x = float(evaluation_state.rejection_anchor_x)
        evaluation_state.world_z = float(evaluation_state.rejection_anchor_z)
        evaluation_state.filtered_ts = float(evaluation_state.rejection_anchor_ts)
        evaluation_state.measurement_accepted = True
        evaluation_state.measurement_rejection_reason = None
        evaluation_state.measurement_innovation_m = 0.0
        evaluation_state.measurement_allowed_m = 0.0
        evaluation_state.reacquired = False
        _clear_reacquire_candidate(evaluation_state)
        _clear_rejection_anchor(evaluation_state)

    result = update_human_cv_filter(
        evaluation_state,
        measurement=candidate,
        floor_y=float(floor_y),
        now_ts=float(now_ts),
        quality="weak",
        config=config,
        force_accept=False,
        contact_basis=None,
        image_motion_supported=False,
    )
    if not bool(evaluation_state.measurement_accepted):
        return None

    if rewound_from_rejection:
        visible_dt = max(0.0, float(now_ts) - previous_filtered_ts)
        visible_step = math.hypot(
            float(result[0]) - previous_world_x,
            float(result[2]) - previous_world_z,
        )
        if (
            float(config.max_speed_mps) > 0.0
            and visible_step
            > float(config.max_speed_mps) * visible_dt + 1e-9
        ):
            # Leave the already-bounded rejected posterior untouched.  The
            # caller will publish its hold/prediction instead of a projective
            # point that would create an unreachable same-segment step.
            return None
        state.__dict__.update(evaluation_state.__dict__)

    # This posterior is physically admissible but not a fresh metric/depth
    # measurement. Keep it out of the metric authority fields while making it
    # the canonical process origin for the next frame.
    state.measurement_accepted = False
    state.measurement_rejection_reason = "projective_weak_observation"
    state.trail_append_allowed = True
    state.rejection_anchor_x = float(state.world_x)
    state.rejection_anchor_z = float(state.world_z)
    state.rejection_anchor_ts = float(state.filtered_ts)
    return np.asarray(result, dtype=np.float64)


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


def _bbox_dimensions(
    bbox: Optional[Sequence[float]],
) -> Tuple[Optional[float], Optional[float]]:
    if bbox is None or len(bbox) < 4:
        return None, None
    try:
        width = float(bbox[2])
        height = float(bbox[3])
    except Exception:
        return None, None
    if not (math.isfinite(width) and math.isfinite(height)):
        return None, None
    if width <= 1.0 or height <= 1.0:
        return None, None
    return float(width), float(height)


def _visible_pose_count(
    kpts_abs: Optional[np.ndarray],
    names: Sequence[str],
    *,
    conf_threshold: float,
) -> int:
    if (
        kpts_abs is None
        or not isinstance(kpts_abs, np.ndarray)
        or kpts_abs.shape[0] < 17
    ):
        return 0
    return sum(
        pose_point(kpts_abs, name, conf_threshold=conf_threshold) is not None
        for name in names
    )


def _ema_reference(
    current: Optional[float],
    measurement: Optional[float],
    *,
    alpha: float,
) -> Optional[float]:
    if measurement is None or not math.isfinite(float(measurement)):
        return current
    value = float(measurement)
    if value <= 0.0:
        return current
    if current is None or not math.isfinite(float(current)) or float(current) <= 0.0:
        return value
    return float(current) + float(alpha) * (value - float(current))


def assess_lower_body_occlusion(
    state: PersonGroundState,
    *,
    kpts_abs: Optional[np.ndarray],
    bbox: Optional[Sequence[float]],
    posture: Posture,
    now_ts: float,
    config: HumanGroundConfig,
) -> LowerBodyOcclusionAssessment:
    """Classify standing-person occlusion from feet through waist/hips.

    The state is intentionally anchored to previously observed upright anatomy.
    A short detector box or missing lower keypoints can therefore demote a
    syntactically valid waist/counter-edge anchor before it reaches world fusion.
    Clear lower-body evidence is required for several frames before direct
    ankle/depth authority resumes, which prevents one-frame pose hallucinations
    from snapping a path back and forth.
    """

    now_ts = float(now_ts)
    threshold = float(config.kpt_conf_threshold)
    visible_shoulders = _visible_pose_count(
        kpts_abs,
        ("left_shoulder", "right_shoulder"),
        conf_threshold=threshold,
    )
    visible_hips = _visible_pose_count(
        kpts_abs,
        ("left_hip", "right_hip"),
        conf_threshold=threshold,
    )
    visible_knees = _visible_pose_count(
        kpts_abs,
        ("left_knee", "right_knee"),
        conf_threshold=threshold,
    )
    visible_ankles = _visible_pose_count(
        kpts_abs,
        ("left_ankle", "right_ankle"),
        conf_threshold=threshold,
    )

    _bbox_width, bbox_height = _bbox_dimensions(bbox)
    previous_bbox_height = state.last_bbox_height_px
    bbox_height_ratio: Optional[float] = None
    if (
        bbox_height is not None
        and state.upright_bbox_height_px is not None
        and float(state.upright_bbox_height_px) > 1e-6
    ):
        bbox_height_ratio = float(bbox_height) / float(state.upright_bbox_height_px)

    shoulder_width: Optional[float] = None
    if (
        kpts_abs is not None
        and isinstance(kpts_abs, np.ndarray)
        and kpts_abs.shape[0] >= 17
    ):
        left_shoulder = pose_point(
            kpts_abs,
            "left_shoulder",
            conf_threshold=threshold,
        )
        right_shoulder = pose_point(
            kpts_abs,
            "right_shoulder",
            conf_threshold=threshold,
        )
        if left_shoulder is not None and right_shoulder is not None:
            shoulder_width = math.hypot(
                float(right_shoulder[0]) - float(left_shoulder[0]),
                float(right_shoulder[1]) - float(left_shoulder[1]),
            )
            if not math.isfinite(shoulder_width) or shoulder_width <= 3.0:
                shoulder_width = None

    bbox_shoulder_ratio: Optional[float] = None
    current_bbox_shoulder_ratio: Optional[float] = None
    if bbox_height is not None and shoulder_width is not None:
        current_bbox_shoulder_ratio = float(bbox_height) / float(shoulder_width)
        if (
            state.upright_bbox_shoulder_ratio is not None
            and float(state.upright_bbox_shoulder_ratio) > 1e-6
        ):
            bbox_shoulder_ratio = (
                float(current_bbox_shoulder_ratio)
                / float(state.upright_bbox_shoulder_ratio)
            )

    explicit_non_upright = bool(
        str(posture or "unknown") in ("sitting", "lying")
        or legs_are_bent(kpts_abs, config=config)
    )
    if explicit_non_upright:
        state.last_non_upright_ts = now_ts

    full_body_visible = bool(
        visible_shoulders >= 1
        and visible_hips >= 1
        and visible_knees >= 1
        and visible_ankles >= 1
        and not explicit_non_upright
        and str(posture or "unknown") not in ("sitting", "lying")
    )
    if full_body_visible:
        state.last_full_body_ts = now_ts
        alpha = float(config.upright_reference_alpha)
        state.upright_bbox_height_px = _ema_reference(
            state.upright_bbox_height_px,
            bbox_height,
            alpha=alpha,
        )
        state.upright_bbox_shoulder_ratio = _ema_reference(
            state.upright_bbox_shoulder_ratio,
            current_bbox_shoulder_ratio,
            alpha=alpha,
        )

    recent_upright = bool(
        state.height_ref_scene is not None
        and (
            state.lower_body_occluded
            or (
                state.last_full_body_ts >= 0.0
                and (
                    float(config.occlusion_upright_memory_s) <= 0.0
                    or (now_ts - float(state.last_full_body_ts))
                    <= float(config.occlusion_upright_memory_s)
                )
            )
        )
        and float(state.last_full_body_ts) >= float(state.last_non_upright_ts)
    )

    collapsed_by_scale = bool(
        bbox_shoulder_ratio is not None
        and bbox_shoulder_ratio < float(config.occlusion_bbox_shoulder_ratio)
    )
    collapsed_by_height = bool(
        bbox_height_ratio is not None
        and bbox_height_ratio < float(config.occlusion_bbox_height_ratio)
    )
    sudden_height_collapse = bool(
        bbox_height is not None
        and previous_bbox_height is not None
        and float(previous_bbox_height) > 1e-6
        and (float(bbox_height) / float(previous_bbox_height))
        < float(config.occlusion_bbox_height_ratio)
    )
    collapsed_bbox = bool(
        collapsed_by_scale or collapsed_by_height or sudden_height_collapse
    )

    detected_level = "none"
    confidence = 0.0
    reasons: List[str] = []
    upper_body_visible = visible_shoulders >= 1
    if recent_upright and not explicit_non_upright:
        if upper_body_visible and visible_hips == 0 and visible_knees == 0:
            detected_level = "waist_hips"
            confidence = 0.96 if visible_ankles == 0 else 0.88
            reasons.append("hips_knees_missing")
        elif (
            upper_body_visible
            and visible_hips == 1
            and visible_knees == 0
            and visible_ankles == 0
        ):
            detected_level = "waist_hips"
            confidence = 0.92
            reasons.append("partial_hips_only")
        elif visible_hips >= 1 and visible_knees == 0 and visible_ankles == 0:
            detected_level = "knees"
            confidence = 0.89
            reasons.append("knees_ankles_missing")
        elif visible_hips >= 1 and visible_knees >= 1 and visible_ankles == 0:
            detected_level = "feet_ankles"
            confidence = 0.82
            reasons.append("ankles_missing")
        elif kpts_abs is None and collapsed_bbox:
            detected_level = "waist_hips"
            confidence = 0.84
            reasons.append("pose_missing_bbox_collapsed")
        elif collapsed_bbox:
            # Covers pose hallucinations on the counter edge: lower keypoints may
            # exist numerically even though the tracked silhouette has collapsed.
            detected_level = "waist_hips"
            confidence = 0.88
            reasons.append("upright_bbox_collapsed")

    if collapsed_by_scale:
        reasons.append("shoulder_scaled_bbox_collapse")
    if collapsed_by_height:
        reasons.append("upright_bbox_height_collapse")
    if sudden_height_collapse:
        reasons.append("sudden_bbox_height_collapse")

    detected = detected_level != "none"
    if detected:
        state.lower_body_occluded = True
        state.lower_body_occlusion_level = str(detected_level)
        state.lower_body_occlusion_confidence = float(confidence)
        state.lower_body_occlusion_reason = ",".join(dict.fromkeys(reasons))
        state.lower_body_clear_frames = 0
    elif state.lower_body_occluded and explicit_non_upright:
        state.lower_body_occluded = False
        state.lower_body_occlusion_level = "none"
        state.lower_body_occlusion_confidence = 0.0
        state.lower_body_occlusion_reason = None
        state.lower_body_clear_frames = 0
    elif state.lower_body_occluded:
        state.lower_body_clear_frames = int(state.lower_body_clear_frames) + 1
        if int(state.lower_body_clear_frames) >= int(config.occlusion_exit_frames):
            state.lower_body_occluded = False
            state.lower_body_occlusion_level = "none"
            state.lower_body_occlusion_confidence = 0.0
            state.lower_body_occlusion_reason = None
            state.lower_body_clear_frames = 0
    else:
        state.lower_body_clear_frames = 0

    state.last_bbox_height_px = bbox_height
    state.last_bbox_ts = now_ts

    return LowerBodyOcclusionAssessment(
        active=bool(state.lower_body_occluded),
        level=str(state.lower_body_occlusion_level),
        confidence=float(state.lower_body_occlusion_confidence),
        reason=state.lower_body_occlusion_reason,
        visible_ankles=int(visible_ankles),
        visible_knees=int(visible_knees),
        visible_hips=int(visible_hips),
        visible_shoulders=int(visible_shoulders),
        bbox_height_ratio=bbox_height_ratio,
        bbox_shoulder_ratio=bbox_shoulder_ratio,
    )


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
        hip = _mid_xy(left_hip, right_hip)
        shoulder = _mid_xy(left_shoulder, right_shoulder)
        ankle = _mid_xy(left_ankle, right_ankle)
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
        # The missing-ankle branch in legs_are_bent deliberately recognizes
        # compact hip/knee geometry. Keep that signal available when seated
        # feet are occluded so an old standing height lock cannot trigger a
        # fabricated gravity-drop floor measurement.
        if legs_are_bent(kpts_abs, config=config):
            return "sitting"

        if (
            height_ref_scene is None
            and bbox_w is not None
            and bbox_h is not None
            and bbox_h > 1.0
        ):
            if (
                float(bbox_w) / float(bbox_h)
                >= float(config.lie_bbox_aspect) * 0.92
            ):
                return "lying"

    if bbox_w is not None and bbox_h is not None and bbox_h > 1.0:
        aspect = float(bbox_w) / float(bbox_h)
        # Tall thin boxes are standing. Once an upright height lock exists, a
        # short detector box is ambiguous with lower-body occlusion and must not
        # become a bbox-only sitting decision.
        if aspect < 0.55 and bbox_h >= 90.0:
            return "standing"
        if (
            height_ref_scene is None
            and aspect >= 0.85
            and aspect < float(config.lie_bbox_aspect)
        ):
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
    # Ankles are observed floor-contact anatomy for standing and sitting people.
    # A pelvis/torso keypoint is elevated above the floor and must never be
    # ray-intersected with the floor as if it were a contact point.
    if left_ankle is not None and right_ankle is not None:
        return PoseAnchorCandidate(
            u=float(left_ankle[0] + right_ankle[0]) * 0.5,
            v=float(left_ankle[1] + right_ankle[1]) * 0.5,
            source="pose_ankle_floor",
            contact_basis="pose:ankle_pair",
            quality="good",
            quality_reason=(
                f"posture={effective_posture},observed_ankles"
                if effective_posture in ("sitting", "lying")
                else None
            ),
            height_lock_eligible=effective_posture not in ("sitting", "lying"),
            score=1.0,
        )
    if left_ankle is not None or right_ankle is not None:
        ankle = left_ankle if left_ankle is not None else right_ankle
        ankle_side = "left" if left_ankle is not None else "right"
        assert ankle is not None
        return PoseAnchorCandidate(
            u=float(ankle[0]),
            v=float(ankle[1]),
            source="pose_single_ankle_floor",
            contact_basis=f"pose:{ankle_side}_ankle",
            quality="good",
            quality_reason=(
                f"posture={effective_posture},observed_single_ankle"
                if effective_posture in ("sitting", "lying")
                else None
            ),
            height_lock_eligible=effective_posture not in ("sitting", "lying"),
            score=0.90,
        )

    # Bent/non-upright legs cannot be safely extended, and upper-body points do
    # not provide a ground contact. Fail closed until another contact-bearing
    # source (for example an instance mask) is available.
    if bent or effective_posture in ("sitting", "lying"):
        return None

    estimates: List[Tuple[str, float, float]] = []
    for side in ("left", "right"):
        ankle_est = estimate_ankle_from_leg(kpts_abs, side, conf_threshold=thr)
        if ankle_est is not None:
            estimates.append((str(side), float(ankle_est[0]), float(ankle_est[1])))
    if not estimates:
        return None
    if len(estimates) == 1:
        side, u, v = estimates[0]
        contact_basis = f"pose:{side}_leg_extension"
    else:
        u = float(sum(p[1] for p in estimates) / len(estimates))
        v = float(sum(p[2] for p in estimates) / len(estimates))
        contact_basis = "pose:leg_pair_extension"
    return PoseAnchorCandidate(
        u=float(u),
        v=float(v),
        source="pose_leg_floor",
        contact_basis=contact_basis,
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
        # Compatibility-only labels from older payloads. They are deliberately
        # non-authoritative: current code never emits an elevated hip/body
        # point as a floor contact.
        "pose_hip_floor": 0.05,
        "pose_body_floor": 0.05,
        "pose_depth_fused": 1.05,
        "pose_depth_only": 1.05,
        "pose_floor_only": 0.95,
        "person_anchor_depth_fused": 0.92,
        "person_anchor_depth_only": 0.92,
        "person_anchor_floor_only": 0.78,
        "person_mask_floor": 0.80,
        "pose_leg_floor": 0.55,
        "gravity_drop": 0.40,
        "cv_prediction": 0.35,
        "image_motion_prediction": 0.35,
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
        if src == "pose_leg_floor":
            base -= 0.25
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
    """Return ``(accepted_source, switched)`` with bounded source failover.

    A stronger source may replace the current source immediately.  A weaker
    source must be observed for ``source_replacement_frames`` consecutive
    updates.  Replacement confirmation is intentionally shorter than the
    normal sticky-source hold: at the 15 Hz publication cadence the default
    completes inside the producer's 0.40 second anchor-hold window, preventing
    a valid replacement from becoming a visible track dropout.
    This keeps single-frame pose/person-anchor churn out of the published path
    without allowing a source that is no longer present to suppress valid
    geometry indefinitely.
    """
    candidate_source = str(candidate_source)
    candidate_score = float(candidate_score)
    if state.sticky_source is None:
        state.sticky_source = candidate_source
        state.sticky_source_frames = 1
        state.sticky_source_score = candidate_score
        state.source_candidate = None
        state.source_candidate_frames = 0
        return candidate_source, False

    if candidate_source == state.sticky_source:
        state.sticky_source_frames = int(state.sticky_source_frames) + 1
        state.sticky_source_score = max(float(state.sticky_source_score), candidate_score)
        state.source_candidate = None
        state.source_candidate_frames = 0
        return candidate_source, False

    replacement_frames = int(config.source_replacement_frames)
    gain = candidate_score - float(state.sticky_source_score)
    if state.source_candidate == candidate_source:
        state.source_candidate_frames = int(state.source_candidate_frames) + 1
    else:
        state.source_candidate = candidate_source
        state.source_candidate_frames = 1

    sustained_replacement = int(state.source_candidate_frames) >= replacement_frames
    urgent_score_gain = gain >= (float(config.source_switch_min_score_gain) + 0.35)
    allow_switch = bool(sustained_replacement or urgent_score_gain)

    # Always allow recovery from hold/gravity when a real observation appears.
    if state.sticky_source in ("anchor_hold", "gravity_drop") and candidate_score >= 0.70:
        allow_switch = True

    if not allow_switch:
        return str(state.sticky_source), False

    state.sticky_source = candidate_source
    state.sticky_source_frames = 1
    state.sticky_source_score = candidate_score
    state.source_switch_count = int(state.source_switch_count) + 1
    state.source_candidate = None
    state.source_candidate_frames = 0
    return candidate_source, True


def begin_source_admission(
    state: PersonGroundState,
    *,
    candidate_source: str,
    candidate_score: float,
    config: HumanGroundConfig,
    authoritative: bool = False,
) -> bool:
    """Evaluate source hysteresis while deferring a source switch until physics accepts."""
    previous = (
        state.sticky_source,
        int(state.sticky_source_frames),
        float(state.sticky_source_score),
        int(state.source_switch_count),
        state.source_candidate,
        int(state.source_candidate_frames),
    )
    if authoritative:
        source_changed = str(state.sticky_source) != str(candidate_source)
        state.sticky_source = str(candidate_source)
        state.sticky_source_frames = 1
        state.sticky_source_score = float(candidate_score)
        state.source_candidate = None
        state.source_candidate_frames = 0
        if source_changed and previous[0] is not None:
            state.source_switch_count = int(state.source_switch_count) + 1
        accepted_source = str(candidate_source)
    else:
        accepted_source, _switched = apply_source_hysteresis(
            state,
            candidate_source=str(candidate_source),
            candidate_score=float(candidate_score),
            config=config,
        )
    if str(accepted_source) != str(candidate_source):
        state.pending_source_previous = None
        return False
    state.pending_source_previous = previous
    return True


def complete_source_admission(state: PersonGroundState, *, measurement_accepted: bool) -> None:
    """Commit an admitted source, or roll it back when the physical gate rejects."""
    previous = state.pending_source_previous
    state.pending_source_previous = None
    if measurement_accepted or previous is None:
        return
    (
        state.sticky_source,
        state.sticky_source_frames,
        state.sticky_source_score,
        state.source_switch_count,
        state.source_candidate,
        state.source_candidate_frames,
    ) = previous


def observe_coherent_image_motion(
    state: PersonGroundState,
    *,
    frame_id: int,
    image_foot_uv: Optional[Tuple[float, float]],
    bbox: Optional[Sequence[float]],
    contact_basis: Optional[str],
    config: HumanGroundConfig,
) -> bool:
    """Observe one exact-frame, independently corroborated contact movement.

    A moving ankle/contact alone is not evidence that the person moved: pose
    swaps and hallucinations commonly move that point while the tracker box is
    stationary.  This admission requires the current contact and the current
    bbox bottom-center to both move, in a coherent direction, across a bounded
    window of strictly consecutive frames and on one stable contact basis.

    The returned boolean describes the *current* coherent observation.  Three
    consecutive observations (``static_exit_frames``) pre-unlock an idle state
    before the world filter runs.  This does not admit a world measurement;
    the physical innovation gate remains authoritative.
    """

    try:
        current_frame_id = int(frame_id)
        basis = str(contact_basis or "").strip()
        if current_frame_id < 0 or not basis or image_foot_uv is None:
            raise ValueError
        contact_u = float(image_foot_uv[0])
        contact_v = float(image_foot_uv[1])
        if bbox is None or len(bbox) < 4:
            raise ValueError
        left, top, width, height = (float(value) for value in bbox[:4])
        bbox_u = left + width * 0.5
        bbox_v = top + height
        if width <= 0.0 or height <= 0.0:
            raise ValueError
        if not all(
            math.isfinite(value)
            for value in (contact_u, contact_v, bbox_u, bbox_v)
        ):
            raise ValueError
    except Exception:
        state.image_motion_frame_id = -1
        state.image_motion_contact_basis = None
        state.image_motion_window.clear()
        state.image_motion_streak = 0
        state.image_motion_supported = False
        return False

    required_observations = max(2, int(config.static_exit_frames))
    if state.image_motion_window.maxlen != required_observations:
        state.image_motion_window = deque(
            list(state.image_motion_window)[-required_observations:],
            maxlen=required_observations,
        )
    consecutive = current_frame_id == int(state.image_motion_frame_id) + 1
    same_basis = basis == str(state.image_motion_contact_basis or "")
    if not consecutive or not same_basis:
        state.image_motion_window.clear()
    state.image_motion_window.append(
        (
            int(current_frame_id),
            str(basis),
            float(contact_u),
            float(contact_v),
            float(bbox_u),
            float(bbox_v),
        )
    )

    coherent = False
    if len(state.image_motion_window) == required_observations:
        first = state.image_motion_window[0]
        last = state.image_motion_window[-1]
        contact_du = float(last[2]) - float(first[2])
        contact_dv = float(last[3]) - float(first[3])
        bbox_du = float(last[4]) - float(first[4])
        bbox_dv = float(last[5]) - float(first[5])
        contact_distance = math.hypot(contact_du, contact_dv)
        bbox_distance = math.hypot(bbox_du, bbox_dv)
        step_directions_coherent = True
        for index in range(1, len(state.image_motion_window)):
            previous = state.image_motion_window[index - 1]
            current = state.image_motion_window[index]
            step_dot = (
                (float(current[2]) - float(previous[2]))
                * (float(current[4]) - float(previous[4]))
                + (float(current[3]) - float(previous[3]))
                * (float(current[5]) - float(previous[5]))
            )
            if step_dot <= 0.0:
                step_directions_coherent = False
                break
        coherent = bool(
            contact_distance > float(config.static_px_threshold)
            and bbox_distance > float(config.static_px_threshold)
            and (contact_du * bbox_du + contact_dv * bbox_dv) > 0.0
            and step_directions_coherent
        )

    state.image_motion_frame_id = int(current_frame_id)
    state.image_motion_contact_basis = str(basis)
    state.image_motion_streak = len(state.image_motion_window) if coherent else 0
    state.image_motion_supported = bool(coherent)

    if (
        state.motion_mode == "idle"
        and int(state.image_motion_streak) >= int(config.static_exit_frames)
    ):
        state.motion_mode = "walk"
        state.locked_world = None
        state.idle_since_ts = 0.0
        state.exit_motion_frames = 0
        state.trail_append_allowed = True
    return bool(coherent)


def observe_bbox_stationarity(
    state: PersonGroundState,
    *,
    frame_id: int,
    bbox: Optional[Sequence[float]],
    config: HumanGroundConfig,
) -> bool:
    """Gate a short hold on exact-frame detector-box continuity.

    This is deliberately weaker than metric admission: it can support only a
    non-appending display hold for an already trusted point.  It never creates
    a world measurement or authorizes a trail update.  Missing/non-consecutive
    frame IDs and movement beyond a small detector-jitter tolerance reset the
    streak.
    """

    try:
        current_frame_id = int(frame_id)
        if current_frame_id < 0 or bbox is None or len(bbox) < 4:
            raise ValueError
        left, _top, width, height = (float(value) for value in bbox[:4])
        center_u = left + width * 0.5
        bottom_v = float(_top) + height
        if width <= 1.0 or height <= 1.0 or not all(
            math.isfinite(value) for value in (center_u, bottom_v)
        ):
            raise ValueError
    except Exception:
        state.bbox_motion_frame_id = -1
        state.bbox_center_u = None
        state.bbox_bottom_v = None
        state.bbox_geometry = None
        state.bbox_stationary_streak = 0
        state.bbox_stationary_supported = False
        return False

    consecutive = current_frame_id == int(state.bbox_motion_frame_id) + 1
    tolerance = max(4.0, float(config.static_px_threshold) * 2.0)
    if (
        not consecutive
        or state.bbox_center_u is None
        or state.bbox_bottom_v is None
        or abs(float(center_u) - float(state.bbox_center_u)) > tolerance
        or abs(float(bottom_v) - float(state.bbox_bottom_v)) > tolerance
    ):
        state.bbox_stationary_streak = 1
    else:
        state.bbox_stationary_streak = int(state.bbox_stationary_streak) + 1
    state.bbox_motion_frame_id = current_frame_id
    state.bbox_center_u = float(center_u)
    state.bbox_bottom_v = float(bottom_v)
    state.bbox_geometry = (
        float(left),
        float(_top),
        float(width),
        float(height),
    )
    required = max(3, int(config.static_exit_frames))
    state.bbox_stationary_supported = bool(
        int(state.bbox_stationary_streak) >= required
    )
    return bool(state.bbox_stationary_supported)


def record_accepted_image_geometry(
    state: PersonGroundState,
    *,
    image_foot_uv: Optional[Tuple[float, float]],
    bbox: Optional[Sequence[float]],
    now_ts: float,
    lifecycle_generation: Optional[int] = None,
) -> bool:
    """Record the image basis belonging to an accepted world observation.

    ``bbox`` and ``image_foot_uv`` must be in the same image coordinate space
    (the runtime uses the calibration raster).  Keeping this write behind the
    physical world admission gate prevents rejected detections from teaching
    the projective predictor a bad origin.
    """

    try:
        if image_foot_uv is None or bbox is None or len(bbox) < 4:
            raise ValueError
        u, v = (float(image_foot_uv[0]), float(image_foot_uv[1]))
        left, top, width, height = (float(value) for value in bbox[:4])
        values = (u, v, left, top, width, height, float(now_ts))
        if not all(math.isfinite(value) for value in values):
            raise ValueError
        if width <= 1.0 or height <= 1.0 or float(now_ts) < 0.0:
            raise ValueError
    except Exception:
        return False
    state.last_accepted_image_foot = (u, v)
    state.last_accepted_bbox_geometry = (left, top, width, height)
    state.last_accepted_image_ts = float(now_ts)
    state.last_accepted_lifecycle_generation = (
        int(lifecycle_generation) if lifecycle_generation is not None else None
    )
    return True


def transport_accepted_image_foot(
    state: PersonGroundState,
    *,
    bbox: Optional[Sequence[float]],
    now_ts: float,
    lifecycle_generation: Optional[int] = None,
    ttl_s: float = 0.40,
    config: Optional[HumanGroundConfig] = None,
) -> Optional[Tuple[float, float, float, float, float, str]]:
    """Transport the accepted foot through current bbox affine motion.

    The returned tuple is ``(u, v, age_s, center_step_px, scale_ratio,
    reason)``.  It is intentionally pure with respect to state: callers may
    evaluate it every frame, but only ``record_accepted_image_geometry`` may
    change the predictor origin.  The pixel transport is not yet metric; the
    caller must project it through the active floor plane and apply its ray and
    world-space gates.
    """

    cfg = config or HumanGroundConfig()
    try:
        prior_foot = state.last_accepted_image_foot
        prior_bbox = state.last_accepted_bbox_geometry
        accepted_ts = float(state.last_accepted_image_ts)
        current_ts = float(now_ts)
        if (
            prior_foot is None
            or prior_bbox is None
            or accepted_ts < 0.0
            or not math.isfinite(current_ts)
        ):
            return None
        if (
            lifecycle_generation is not None
            and state.last_accepted_lifecycle_generation is not None
            and int(lifecycle_generation)
            != int(state.last_accepted_lifecycle_generation)
        ):
            return None
        age_s = current_ts - accepted_ts
        if age_s < 0.0 or age_s > max(0.0, float(ttl_s)):
            return None
        if bbox is None or len(bbox) < 4:
            return None
        left, top, width, height = (float(value) for value in bbox[:4])
        prior_left, prior_top, prior_width, prior_height = prior_bbox
        values = (
            *prior_foot,
            left,
            top,
            width,
            height,
            prior_left,
            prior_top,
            prior_width,
            prior_height,
        )
        if not all(math.isfinite(float(value)) for value in values):
            return None
        if min(width, height, prior_width, prior_height) <= 1.0:
            return None
        prior_center = (
            float(prior_left) + float(prior_width) * 0.5,
            float(prior_top) + float(prior_height) * 0.5,
        )
        current_center = (left + width * 0.5, top + height * 0.5)
        center_step_px = math.hypot(
            current_center[0] - prior_center[0],
            current_center[1] - prior_center[1],
        )
        if center_step_px > float(cfg.projective_max_bbox_step_px):
            return None
        # Use at least a 30 FPS interval for the speed gate so a repeated
        # timestamp cannot turn an ordinary one-frame displacement into an
        # infinite speed.  The hard step gate above remains the fail-closed
        # protection for implausible jumps.
        speed_dt = max(age_s, 1.0 / 30.0)
        if center_step_px / speed_dt > float(cfg.projective_max_bbox_speed_px_s):
            return None
        width_ratio = width / float(prior_width)
        height_ratio = height / float(prior_height)
        scale_ratio = max(width_ratio, height_ratio, 1.0 / width_ratio, 1.0 / height_ratio)
        if scale_ratio > float(cfg.projective_max_scale_ratio):
            return None
        rel_u = (float(prior_foot[0]) - float(prior_left)) / float(prior_width)
        rel_v = (float(prior_foot[1]) - float(prior_top)) / float(prior_height)
        # A floor contact should stay in/near the tracked silhouette.  This
        # rejects old corrupted image anchors without clipping them into the
        # current box.
        if not (-0.25 <= rel_u <= 1.25 and -0.25 <= rel_v <= 1.25):
            return None
        transported = (left + rel_u * width, top + rel_v * height)
        if not all(math.isfinite(float(value)) for value in transported):
            return None
        return (
            float(transported[0]),
            float(transported[1]),
            float(age_s),
            float(center_step_px),
            float(scale_ratio),
            "bbox_affine",
        )
    except Exception:
        return None


def _clamp_speed(vx: float, vz: float, max_speed: float) -> Tuple[float, float]:
    speed = math.hypot(float(vx), float(vz))
    if max_speed <= 0.0 or speed <= max_speed or speed <= 1e-9:
        return float(vx), float(vz)
    scale = max_speed / speed
    return float(vx) * scale, float(vz) * scale


def admit_human_ground_output(
    state: PersonGroundState,
    *,
    candidate: np.ndarray,
    floor_y: float,
    now_ts: float,
    media_pts_ns: Any,
    config: HumanGroundConfig,
) -> Tuple[np.ndarray, bool]:
    """Enforce physical continuity at the canonical world-output boundary.

    This is a fail-closed admission check, not a display smoother or a clamp.
    A same-segment point that cannot be reached in exact source-media time is
    quarantined and the prior canonical point is held.  A confirmed
    reacquisition remains admissible because it carries an explicit trail
    break/segment change from the measurement filter.
    """

    try:
        point = np.asarray(candidate, dtype=np.float64).reshape(-1)
        x = float(point[0])
        z = float(point[2])
        floor = float(floor_y)
        filter_ts = float(now_ts)
    except (TypeError, ValueError, IndexError, OverflowError):
        return np.array([math.nan, math.nan, math.nan], dtype=np.float64), False
    if not all(math.isfinite(value) for value in (x, z, floor, filter_ts)):
        return np.array([math.nan, math.nan, math.nan], dtype=np.float64), False

    try:
        parsed_pts = int(media_pts_ns or 0)
    except (TypeError, ValueError, OverflowError):
        parsed_pts = 0
    current_pts = (
        parsed_pts
        if 0 < parsed_pts < (1 << 64) - 1
        else None
    )
    previous_x = state.last_output_world_x
    previous_z = state.last_output_world_z
    previous_pts = state.last_output_media_pts_ns
    previous_filter_ts = float(state.last_output_filter_ts)
    current_segment = int(state.trail_segment_id)
    previous_segment = int(state.last_output_trail_segment_id)

    if previous_x is None or previous_z is None:
        accepted = True
    elif bool(state.trail_break_required) or current_segment != previous_segment:
        accepted = True
    else:
        if (
            current_pts is not None
            and previous_pts is not None
            and int(current_pts) >= int(previous_pts)
        ):
            dt = (int(current_pts) - int(previous_pts)) / 1_000_000_000.0
        else:
            dt = max(0.0, filter_ts - previous_filter_ts)
        if float(config.reset_after_s) > 0.0:
            dt = min(float(dt), float(config.reset_after_s))
        step = math.hypot(x - float(previous_x), z - float(previous_z))
        allowed = float(config.max_speed_mps) * float(dt)
        accepted = bool(
            float(config.max_speed_mps) <= 0.0
            or step <= allowed + 1e-9
        )
        if not accepted:
            # Quarantine the divergent process result and make the prior
            # canonical output the next filter origin.  Do not manufacture an
            # intermediate point or update the accepted metric anchor.
            state.world_x = float(previous_x)
            state.world_z = float(previous_z)
            state.vel_world_x = 0.0
            state.vel_world_z = 0.0
            state.filtered_ts = filter_ts
            state.measurement_accepted = False
            state.measurement_rejection_reason = "physical_output_continuity_exceeded"
            state.measurement_innovation_m = float(step)
            state.measurement_allowed_m = float(allowed)
            state.trail_append_allowed = False
            state.rejection_anchor_x = float(previous_x)
            state.rejection_anchor_z = float(previous_z)
            state.rejection_anchor_ts = filter_ts
            state.rejection_previous_world_x = float(previous_x)
            state.rejection_previous_world_z = float(previous_z)
            state.rejection_previous_world_ts = filter_ts
            # Advance the output timestamp while holding position so repeated
            # rejected frames cannot accumulate an artificial speed budget.
            state.last_output_media_pts_ns = current_pts
            state.last_output_filter_ts = filter_ts
            return (
                np.array([float(previous_x), floor, float(previous_z)], dtype=np.float64),
                False,
            )

    state.last_output_world_x = x
    state.last_output_world_z = z
    state.last_output_media_pts_ns = current_pts
    state.last_output_filter_ts = filter_ts
    state.last_output_trail_segment_id = current_segment
    return np.array([x, floor, z], dtype=np.float64), True


def _reject_with_cv_time_update(
    state: PersonGroundState,
    *,
    pred_x: float,
    pred_z: float,
    now_ts: float,
    floor_y: float,
    reason: str,
    config: HumanGroundConfig,
) -> np.ndarray:
    """Advance only the process model after rejecting a metric observation.

    A rejected world measurement is not a request to discard the track.  Keep
    the already-bounded velocity, advance only within the configured
    last-good horizon, and mark the current observation as unavailable.  The
    ``last_good_*`` fields deliberately remain untouched; the caller decides
    whether this bounded prediction is displayable during the existing hold
    TTL.
    """

    previous_world_x = (
        float(state.world_x) if state.world_x is not None else None
    )
    previous_world_z = (
        float(state.world_z) if state.world_z is not None else None
    )
    previous_filtered_ts = float(state.filtered_ts)
    state.rejection_previous_world_x = previous_world_x
    state.rejection_previous_world_z = previous_world_z
    state.rejection_previous_world_ts = previous_filtered_ts

    # Keep one fixed origin for a run of rejected observations.  Reusing the
    # already-predicted state here would integrate velocity on every rejected
    # frame and eventually publish an unbounded point (even though the display
    # hold has already expired).  Prefer the runtime's independently accepted
    # point; the local anchor is a safe fallback for direct filter consumers
    # that have not mirrored ``last_good_world`` yet.
    if state.rejection_anchor_x is None or state.rejection_anchor_z is None:
        anchor_x: Optional[float] = None
        anchor_z: Optional[float] = None
        anchor_ts: Optional[float] = None
        if (
            state.last_good_world is not None
            and len(state.last_good_world) >= 3
        ):
            try:
                candidate_x = float(state.last_good_world[0])
                candidate_z = float(state.last_good_world[2])
                candidate_ts = float(state.last_good_ts)
            except Exception:
                candidate_x = candidate_z = candidate_ts = math.nan
            if (
                math.isfinite(candidate_x)
                and math.isfinite(candidate_z)
                and math.isfinite(candidate_ts)
            ):
                anchor_x, anchor_z, anchor_ts = candidate_x, candidate_z, candidate_ts
        if anchor_x is None or anchor_z is None or anchor_ts is None:
            try:
                candidate_x = float(state.world_x)
                candidate_z = float(state.world_z)
                candidate_ts = float(state.filtered_ts)
            except Exception:
                candidate_x = candidate_z = candidate_ts = math.nan
            if (
                math.isfinite(candidate_x)
                and math.isfinite(candidate_z)
                and math.isfinite(candidate_ts)
            ):
                anchor_x, anchor_z, anchor_ts = candidate_x, candidate_z, candidate_ts
        if anchor_x is not None and anchor_z is not None and anchor_ts is not None:
            state.rejection_anchor_x = float(anchor_x)
            state.rejection_anchor_z = float(anchor_z)
            state.rejection_anchor_ts = float(anchor_ts)

    state.vel_world_x, state.vel_world_z = _clamp_speed(
        float(state.vel_world_x),
        float(state.vel_world_z),
        float(config.max_speed_mps),
    )
    if state.rejection_anchor_x is not None and state.rejection_anchor_z is not None:
        anchor_ts = float(state.rejection_anchor_ts)
        age = max(0.0, float(now_ts) - anchor_ts)
        age = min(age, float(config.rejected_prediction_horizon_s))
        bounded_x = float(state.rejection_anchor_x) + float(state.vel_world_x) * age
        bounded_z = float(state.rejection_anchor_z) + float(state.vel_world_z) * age
    else:
        # This is only reachable for a malformed/uninitialized state.  Keep
        # the original prediction as the least surprising fallback, while all
        # normal runtime states use the bounded anchor above.
        bounded_x = float(pred_x)
        bounded_z = float(pred_z)
    # The fixed anchor prevents cumulative drift, but a velocity update from
    # an independent weak/projective observation can change the anchor-relative
    # prediction between frames.  Two predictions may each be within the total
    # horizon while the visible step between them is impossible.  Quarantine
    # that transition as a hold; do not invent an intermediate slew.
    continuity_rejected = False
    if (
        previous_world_x is not None
        and previous_world_z is not None
        and math.isfinite(previous_filtered_ts)
        and float(now_ts) >= previous_filtered_ts
        and float(config.max_speed_mps) > 0.0
    ):
        visible_dt = float(now_ts) - previous_filtered_ts
        visible_step = math.hypot(
            float(bounded_x) - previous_world_x,
            float(bounded_z) - previous_world_z,
        )
        if visible_step > float(config.max_speed_mps) * visible_dt + 1e-9:
            bounded_x = previous_world_x
            bounded_z = previous_world_z
            continuity_rejected = True
    state.world_x = float(bounded_x)
    state.world_z = float(bounded_z)
    state.filtered_ts = float(now_ts)
    state.measurement_accepted = False
    state.measurement_rejection_reason = (
        "bounded_process_continuity_exceeded"
        if continuity_rejected
        else str(reason or "physical_measurement_rejected")
    )
    state.trail_append_allowed = False
    return np.array([float(bounded_x), float(floor_y), float(bounded_z)], dtype=np.float64)


def _quarantine_or_reacquire_measurement(
    state: PersonGroundState,
    *,
    measurement_x: float,
    measurement_z: float,
    pred_x: float,
    pred_z: float,
    now_ts: float,
    floor_y: float,
    reason: str,
    config: HumanGroundConfig,
    contact_basis: Optional[str],
    image_motion_supported: bool,
) -> np.ndarray:
    """Quarantine one unreachable update or explicitly reanchor a track.

    A same-lifecycle relocation is never smuggled through as a high-speed
    continuous step.  Reanchoring requires a bounded run on one anatomical
    contact basis plus independent exact-frame image motion, and always starts
    a new trail segment.
    """

    mx = float(measurement_x)
    mz = float(measurement_z)
    admitted_basis = str(contact_basis or "").strip()
    if not bool(image_motion_supported) or not admitted_basis:
        _clear_reacquire_candidate(state)
        return _reject_with_cv_time_update(
            state,
            pred_x=float(pred_x),
            pred_z=float(pred_z),
            now_ts=float(now_ts),
            floor_y=float(floor_y),
            reason=str(reason),
            config=config,
        )

    candidate_consistent = False
    candidate_dt = float(now_ts) - float(state.reacquire_candidate_ts)
    if (
        state.reacquire_candidate_x is not None
        and state.reacquire_candidate_z is not None
        and admitted_basis == str(state.reacquire_candidate_basis or "")
        and candidate_dt > 0.0
        and candidate_dt <= float(config.reacquire_max_gap_s)
    ):
        candidate_step = math.hypot(
            mx - float(state.reacquire_candidate_x),
            mz - float(state.reacquire_candidate_z),
        )
        candidate_allowed = (
            float(config.max_jump_m)
            + float(config.max_speed_mps) * candidate_dt
        )
        candidate_consistent = candidate_step <= candidate_allowed
    state.reacquire_count = (
        int(state.reacquire_count) + 1 if candidate_consistent else 1
    )
    state.reacquire_candidate_x = mx
    state.reacquire_candidate_z = mz
    state.reacquire_candidate_ts = float(now_ts)
    state.reacquire_candidate_basis = admitted_basis

    if int(state.reacquire_count) >= int(config.reacquire_samples):
        state.world_x = mx
        state.world_z = mz
        state.vel_world_x = 0.0
        state.vel_world_z = 0.0
        state.filtered_ts = float(now_ts)
        state.motion_mode = "unknown"
        state.locked_world = None
        state.exit_motion_frames = 0
        state.reacquired = True
        state.trail_break_required = True
        state.trail_segment_id = int(state.trail_segment_id) + 1
        _clear_reacquire_candidate(state)
        _clear_rejection_anchor(state)
        return np.array([mx, float(floor_y), mz], dtype=np.float64)

    return _reject_with_cv_time_update(
        state,
        pred_x=float(pred_x),
        pred_z=float(pred_z),
        now_ts=float(now_ts),
        floor_y=float(floor_y),
        reason=str(reason),
        config=config,
    )


def update_human_cv_filter(
    state: PersonGroundState,
    *,
    measurement: np.ndarray,
    floor_y: float,
    now_ts: float,
    quality: str,
    config: HumanGroundConfig,
    force_accept: bool = False,
    contact_basis: Optional[str] = None,
    image_motion_supported: bool = False,
) -> np.ndarray:
    """Constant-velocity XZ filter with explicit physical admission.

    Impossible observations are quarantined instead of clipped into plausible-
    looking motion.  An existing tracker lifecycle may relocate only after a
    bounded run of mutually consistent observations on one contact basis with
    independent exact-frame image motion, at which point callers are told to
    start a new trail segment.
    """
    _begin_filter_measurement(state)

    # A filter cannot make a safe first state from malformed producer input.
    # Validate the complete measurement, floor, and timestamp before touching
    # any kinematic state. A bad frame is an honest unavailable observation,
    # never a state advance or trail point.
    try:
        candidate = np.asarray(measurement, dtype=np.float64).reshape(-1)
        mx = float(candidate[0])
        mz = float(candidate[2])
        now_ts = float(now_ts)
        floor_y = float(floor_y)
    except (TypeError, ValueError, IndexError, OverflowError):
        mx = mz = now_ts = floor_y = math.nan
    if not all(math.isfinite(value) for value in (mx, mz, now_ts, floor_y)):
        state.measurement_accepted = False
        state.measurement_rejection_reason = "nonfinite_measurement"
        state.measurement_innovation_m = math.nan
        state.measurement_allowed_m = math.nan
        state.trail_append_allowed = False
        _clear_reacquire_candidate(state)
        return np.array([math.nan, math.nan, math.nan], dtype=np.float64)

    # The state is shared with the runtime's prediction/readout path.  A
    # state restored from an older runtime, or populated by a direct caller,
    # can carry a velocity above the active contract even though this filter
    # clamps newly estimated velocities below.  Normalize it before forming
    # the prior so every subsequent reader sees the same physically bounded
    # process model; otherwise a hook that reads the prior immediately after
    # this update could emit an unreachable same-segment step.
    state.vel_world_x, state.vel_world_z = _clamp_speed(
        float(state.vel_world_x),
        float(state.vel_world_z),
        float(config.max_speed_mps),
    )

    if state.world_x is None or state.world_z is None or float(state.filtered_ts) < 0.0:
        state.world_x = mx
        state.world_z = mz
        state.vel_world_x = 0.0
        state.vel_world_z = 0.0
        state.filtered_ts = now_ts
        _clear_reacquire_candidate(state)
        _clear_rejection_anchor(state)
        return np.array([mx, floor_y, mz], dtype=np.float64)

    dt = now_ts - float(state.filtered_ts)
    if dt <= 0.0:
        state.measurement_accepted = False
        state.measurement_rejection_reason = "non_monotonic_timestamp"
        state.trail_append_allowed = False
        return np.array([float(state.world_x), floor_y, float(state.world_z)], dtype=np.float64)

    # Predict with constant velocity.
    gate_dt = float(dt)
    if float(config.reset_after_s) > 0.0:
        gate_dt = min(gate_dt, float(config.reset_after_s))
    pred_x = float(state.world_x) + float(state.vel_world_x) * gate_dt
    pred_z = float(state.world_z) + float(state.vel_world_z) * gate_dt

    innov_x = mx - pred_x
    innov_z = mz - pred_z
    innov_dist = math.hypot(innov_x, innov_z)

    # Physical gate (speed * dt + jump slack).
    allowed = float(config.max_jump_m) + float(config.max_speed_mps) * gate_dt
    state.measurement_innovation_m = float(innov_dist)
    state.measurement_allowed_m = float(allowed)
    if (not force_accept) and allowed > 0.0 and innov_dist > allowed and innov_dist > 1e-9:
        return _quarantine_or_reacquire_measurement(
            state,
            measurement_x=mx,
            measurement_z=mz,
            pred_x=float(pred_x),
            pred_z=float(pred_z),
            now_ts=float(now_ts),
            floor_y=float(floor_y),
            reason="physical_innovation_exceeded",
            config=config,
            contact_basis=contact_basis,
            image_motion_supported=bool(image_motion_supported),
        )

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
        _clear_reacquire_candidate(state)
        _clear_rejection_anchor(state)
        return np.array([float(state.world_x), floor_y, float(state.world_z)], dtype=np.float64)

    next_x = pred_x + alpha * innov_x
    next_z = pred_z + alpha * innov_z

    # The innovation gate includes a measurement-noise allowance.  That
    # allowance must not become an impossible same-segment output step.  A
    # proposed posterior beyond the configured human speed is quarantined;
    # only the explicit evidence-backed reanchor path above may relocate it.
    proposed_step = math.hypot(
        float(next_x) - float(state.world_x),
        float(next_z) - float(state.world_z),
    )
    proposed_step_limit = float(config.max_speed_mps) * float(gate_dt)
    if (
        not force_accept
        and float(config.max_speed_mps) > 0.0
        and proposed_step > proposed_step_limit + 1e-9
    ):
        return _quarantine_or_reacquire_measurement(
            state,
            measurement_x=mx,
            measurement_z=mz,
            pred_x=float(pred_x),
            pred_z=float(pred_z),
            now_ts=float(now_ts),
            floor_y=float(floor_y),
            reason="physical_output_speed_exceeded",
            config=config,
            contact_basis=contact_basis,
            image_motion_supported=bool(image_motion_supported),
        )

    _clear_reacquire_candidate(state)

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
    _clear_rejection_anchor(state)
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
    "LowerBodyOcclusionAssessment",
    "PoseAnchorCandidate",
    "PersonGroundState",
    "PersonGroundStateStore",
    "apply_source_hysteresis",
    "admit_human_ground_output",
    "advance_human_cv_prediction",
    "assess_lower_body_occlusion",
    "bind_world_frame",
    "begin_source_admission",
    "classify_posture",
    "commit_image_path_point",
    "commit_path_point",
    "complete_source_admission",
    "estimate_ankle_from_leg",
    "integrate_projective_ground_observation",
    "legs_are_bent",
    "record_accepted_image_geometry",
    "observe_bbox_stationarity",
    "pose_point",
    "rdp_simplify",
    "resolve_pose_floor_anchor",
    "source_score",
    "transport_accepted_image_foot",
    "update_human_cv_filter",
    "update_motion_mode",
    "world_frame_binding_from_calibration",
    "world_frame_matches_calibration",
]
