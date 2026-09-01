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
    occlusion_exit_s: float = 0.20
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
    projective_pose_motion_ttl_s: float = 2.50
    inferred_process_consensus_samples: int = 3
    inferred_process_consensus_horizon_s: float = 0.40

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
        object.__setattr__(self, "occlusion_exit_s", max(0.0, float(self.occlusion_exit_s)))
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
        object.__setattr__(
            self,
            "projective_pose_motion_ttl_s",
            max(0.0, float(self.projective_pose_motion_ttl_s)),
        )
        object.__setattr__(
            self,
            "inferred_process_consensus_samples",
            min(5, max(1, int(self.inferred_process_consensus_samples))),
        )
        object.__setattr__(
            self,
            "inferred_process_consensus_horizon_s",
            max(0.0, float(self.inferred_process_consensus_horizon_s)),
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


@dataclass(frozen=True)
class AcceptedPoseProjectiveOrigin:
    """One coherent accepted foot/pose bundle for projective transport.

    The ordinary accepted-image fields remain the freshest generic predictor
    origin for bbox transport.  This independent immutable bundle preserves
    the last accepted image geometry that also had an exact pose reference,
    so a later bbox-only metric row cannot mix its newer foot/bbox with an
    older torso anchor or erase the pose-compatible origin altogether.
    """

    image_foot: Tuple[float, float]
    world: Tuple[float, float, float]
    bbox_geometry: Tuple[float, float, float, float]
    motion_anchor: Tuple[float, float]
    motion_basis: str
    ts: float
    lifecycle_generation: Optional[int] = None


@dataclass(frozen=True)
class InferredGroundContinuityAnchor:
    """Fixed raw/world origin for one established occlusion episode.

    A learned-height reconstruction has useful *current* perspective motion,
    but its absolute floor point can have a stable bias relative to the last
    metric contact.  Pair the first raw reconstruction with the last
    queue-visible world output and transport only subsequent raw deltas.  The
    pair is immutable for the episode, so inferred rows cannot integrate their
    own error into a drifting origin.
    """

    raw_origin: Tuple[float, float, float]
    trusted_world_origin: Tuple[float, float, float]
    raw_origin_ts: float
    raw_origin_observed_at_us: int
    raw_origin_media_pts_ns: Optional[int]
    trusted_origin_filter_ts: float
    trusted_origin_media_pts_ns: Optional[int]
    lifecycle_generation: int
    height_ref_scene: float
    world_frame_id: Optional[str]
    world_frame_revision: Optional[str]
    world_transform_sha256: Optional[str]
    trail_segment_id: int


@dataclass(frozen=True)
class FirstOutputAnkleProof:
    """Pending exact ankle-pair consensus before the first public output.

    World estimation runs on callbacks that may be suppressed by the ordered
    tracking/BEV publisher. Those callbacks may advance the ordinary filter,
    so its mutable state cannot prove that a lifecycle already owns a public
    coordinate. Retain the tiny two-sample geometric proof independently
    until the queue commits the first metric output.
    """

    world_x: float
    world_z: float
    observation_ts: float
    consistent_samples: int
    binding: Hashable
    unavailable_rows: int = 0


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
    image_motion_ts: float = -1.0
    image_motion_contact_basis: Optional[str] = None
    image_motion_window: Deque[
        Tuple[int, str, float, float, float, float, float]
    ] = field(
        default_factory=lambda: deque(maxlen=3)
    )
    image_motion_streak: int = 0
    image_motion_supported: bool = False
    # Cold-start admission needs a stronger image-motion proof than ordinary
    # idle exit.  Keep the exact-window displacements and the resulting gate
    # separate so clearing the seated/stationary display hold cannot
    # accidentally authorize a two-sample world seed.
    image_motion_contact_distance_px: float = 0.0
    image_motion_bbox_distance_px: float = 0.0
    image_motion_bootstrap_supported: bool = False
    image_motion_bootstrap_ts: float = -1.0
    image_motion_bootstrap_basis: Optional[str] = None
    # Exact-cohort detector-box continuity used only to gate a short stationary
    # hold when a seated target has no current ground contact.  This is compact
    # scalar state; it does not retain or copy image data.
    bbox_motion_frame_id: int = -1
    bbox_motion_ts: float = -1.0
    bbox_center_u: Optional[float] = None
    bbox_bottom_v: Optional[float] = None
    bbox_geometry: Optional[Tuple[float, float, float, float]] = None
    bbox_stationary_root_geometry: Optional[
        Tuple[float, float, float, float]
    ] = None
    bbox_stationary_streak: int = 0
    bbox_stationary_supported: bool = False
    # Root of the most recent object-depth sample computed for this exact
    # tracker lifecycle.  Cached object-depth metadata may corroborate a
    # current pose floor ray only when it names this root; this prevents a
    # reused numeric tracker ID from inheriting another person's cached range.
    depth_corroboration_measurement_frame_id: int = -1
    depth_corroboration_measurement_ts_us: int = -1
    depth_corroboration_lifecycle_generation: Optional[int] = None
    # Last *physically accepted* image geometry.  These fields are deliberately
    # distinct from the current image evidence and from bbox_stationarity: a
    # rejected/missing frame must never become a new predictor origin.
    last_accepted_image_foot: Optional[Tuple[float, float]] = None
    last_accepted_image_world: Optional[Tuple[float, float, float]] = None
    last_accepted_bbox_geometry: Optional[Tuple[float, float, float, float]] = None
    last_accepted_motion_anchor: Optional[Tuple[float, float]] = None
    last_accepted_motion_basis: Optional[str] = None
    last_accepted_image_ts: float = -1.0
    last_accepted_lifecycle_generation: Optional[int] = None
    # The most recent accepted origin with a complete pose reference. This is
    # intentionally independent from the generic bbox origin above: bbox-only
    # accepted rows may refresh bbox transport without invalidating the fixed
    # torso-to-foot offset used by pose transport.
    last_accepted_pose_projective_origin: Optional[
        AcceptedPoseProjectiveOrigin
    ] = None
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
    lower_body_clear_since_ts: float = -1.0
    # A walking, image-motion-proven standing occlusion is not revoked by one
    # pose frame that hallucinates bent legs.  True non-upright evidence must
    # persist for the same bounded confirmation count before it clears the
    # occlusion episode.
    lower_body_non_upright_frames: int = 0
    lower_body_non_upright_since_ts: float = -1.0
    # Once a standing lifecycle has lost its lower-body contact, a distant
    # bbox-only floor ray may continue locally but cannot relocate the metric
    # state. A verified ankle, registered lower-body depth contact, or strong
    # current detector-confirmed pose associated with the floor hypothesis
    # clears this guard. This
    # preserves ordinary pose-dropout continuity while preventing a displaced
    # tracker box from earning a new world origin.
    post_occlusion_reacquire_support_required: bool = False
    # Learned calibrated heights for visible upper-body reference points.
    # Values are fractions of the track's standing height and stay process-local.
    body_plane_height_fractions: Dict[str, float] = field(default_factory=dict)
    # Exact-current learned-height geometry is useful during a standing
    # lower-body occlusion, but only after its absolute bias has been bound to
    # the last queue-visible metric/process output.  This immutable origin is
    # reset at the episode/frame/lifecycle boundary and is never renewed by an
    # inferred row.
    inferred_ground_continuity_anchor: Optional[
        InferredGroundContinuityAnchor
    ] = None
    # A discontinuity larger than the raw-consensus horizon ends the current
    # inferred episode.  Do not immediately rebase from the last inferred
    # output on the next callback: only an explicit metric recovery or the end
    # of lower-body occlusion may re-arm a new immutable origin.
    inferred_ground_continuity_blocked: bool = False
    inferred_ground_blocked_after_media_pts_ns: Optional[int] = None
    inferred_ground_raw_history: Deque[
        Tuple[float, int, Optional[int], float, float, float]
    ] = field(default_factory=lambda: deque(maxlen=5))

    # An exact lifecycle has no canonical position until the ordered
    # tracking/world/BEV cohort commits one. Keep observed ankle-pair
    # bootstrap evidence outside the mutable CV/reacquisition fields so
    # rate-suppressed callbacks cannot erase it or impersonate that commit.
    first_output_ankle_proof: Optional[FirstOutputAnkleProof] = None

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
    # not the filtered output, and are shared across DS9.1 tracking modes.
    measurement_accepted: bool = True
    measurement_rejection_reason: Optional[str] = None
    measurement_innovation_m: float = 0.0
    measurement_allowed_m: float = 0.0
    reacquire_candidate_x: Optional[float] = None
    reacquire_candidate_z: Optional[float] = None
    reacquire_candidate_ts: float = -1.0
    reacquire_candidate_basis: Optional[str] = None
    # Preserve the exact support subtype inside a collapsed trajectory family.
    # This distinguishes ankle-pair from single-ankle evidence and exact
    # four-plane body solves from generic partial-body rows.
    reacquire_candidate_exact_basis: Optional[str] = None
    reacquire_count: int = 0
    # One missing exact cohort may occur between otherwise consistent observed
    # ankle-floor samples. Upright-body cold evidence may cross multiple
    # non-measurement rows, but its original observation timestamp is never
    # renewed and the next body sample must still satisfy
    # ``reacquire_max_gap_s``. Other bases fail closed immediately.
    reacquire_unavailable_rows: int = 0
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

    # One explicit, non-renewing row may continue from the immediately prior
    # accepted projective process posterior. Keep that authorization separate
    # from the generic rejection anchor: a later metric attempt may legitimately
    # replace rejection diagnostics before fallback selection, but it must not
    # silently erase or renew this one-row budget.
    projective_bridge_origin_ts: float = -1.0
    projective_bridge_process_ts: float = -1.0
    projective_bridge_rows_remaining: int = 0

    # A bbox-compatible tracker return preserves useful filter priors, but row
    # presence alone is not current position evidence. Keep this latch private
    # until a new metric or independently proven projective observation lands;
    # generic CV, output-hold, and stationary-hold paths may not clear it.
    post_ghost_position_support_required: bool = False

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
        output_watermark_exists = bool(
            self.last_output_world_x is not None
            and self.last_output_world_z is not None
        )
        public_trail_segment_id = (
            int(self.last_output_trail_segment_id)
            if output_watermark_exists
            else int(self.trail_segment_id)
        )
        public_trail_break_required = bool(
            self.trail_break_required
            and public_trail_segment_id == int(self.trail_segment_id)
        )
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
            "world_post_occlusion_support_required": bool(
                self.post_occlusion_reacquire_support_required
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
            "world_image_motion_contact_distance_px": float(
                self.image_motion_contact_distance_px
            ),
            "world_bbox_stationary_supported": bool(
                self.bbox_stationary_supported
            ),
            "world_image_motion_bbox_distance_px": float(
                self.image_motion_bbox_distance_px
            ),
            "world_image_motion_bootstrap_supported": bool(
                self.image_motion_bootstrap_supported
            ),
            # The filter may have accepted a metric relocation on a callback
            # that never reached the ordered publication queue.  Public trail
            # fields belong to the exact admitted output watermark, while the
            # newer internal segment remains available for the next visible
            # metric row to publish as a real break.
            "trail_break_required": public_trail_break_required,
            "trail_segment_id": public_trail_segment_id,
        }


def _clear_reacquire_candidate(state: PersonGroundState) -> None:
    state.reacquire_candidate_x = None
    state.reacquire_candidate_z = None
    state.reacquire_candidate_ts = -1.0
    state.reacquire_candidate_basis = None
    state.reacquire_candidate_exact_basis = None
    state.reacquire_count = 0
    state.reacquire_unavailable_rows = 0


def clear_first_output_ankle_proof(state: PersonGroundState) -> None:
    """Clear pending first-public-output ankle evidence."""

    state.first_output_ankle_proof = None


def observe_first_output_ankle_proof(
    state: PersonGroundState,
    *,
    measurement: Optional[np.ndarray],
    now_ts: float,
    exact_ankle_pair: bool,
    output_committed: bool,
    proof_binding: Optional[Hashable],
    config: HumanGroundConfig,
) -> bool:
    """Return whether two bounded exact ankle-pair samples prove a cold seed.

    Lower-authority callbacks are tolerated only inside the same short time
    bound because pose and publication cadence differ. Counting callbacks is
    incorrect here: world estimation runs at media cadence while the ordered
    tracking/BEV cohort publishes at a lower cadence, so several callbacks can
    occur between the two exact observations. The proof is lifecycle-local
    through its owning ``PersonGroundState`` and is cleared on timeout,
    malformed geometry, world-frame reset, or the first queue commit.
    """

    if bool(output_committed):
        clear_first_output_ankle_proof(state)
        return False
    if proof_binding is None:
        clear_first_output_ankle_proof(state)
        return False

    proof = state.first_output_ankle_proof
    if proof is not None and proof.binding != proof_binding:
        # Tracker generation, world revision/transform, and camera
        # calibration are all part of the caller's immutable output key. Two
        # samples from different keys can never form one first-output proof.
        clear_first_output_ankle_proof(state)
        proof = None
    try:
        observation_ts = float(now_ts)
    except (TypeError, ValueError, OverflowError):
        clear_first_output_ankle_proof(state)
        return False
    if not math.isfinite(observation_ts):
        clear_first_output_ankle_proof(state)
        return False

    if not bool(exact_ankle_pair):
        if proof is None:
            return False
        age_s = observation_ts - float(proof.observation_ts)
        if (
            not math.isfinite(age_s)
            or age_s < 0.0
            or age_s > float(config.reacquire_max_gap_s)
        ):
            clear_first_output_ankle_proof(state)
            return False
        state.first_output_ankle_proof = FirstOutputAnkleProof(
            world_x=float(proof.world_x),
            world_z=float(proof.world_z),
            observation_ts=float(proof.observation_ts),
            consistent_samples=int(proof.consistent_samples),
            binding=proof.binding,
            unavailable_rows=min(1, int(proof.unavailable_rows) + 1),
        )
        return False

    try:
        point = np.asarray(measurement, dtype=np.float64).reshape(-1)
        world_x = float(point[0])
        world_z = float(point[2])
    except (TypeError, ValueError, IndexError, OverflowError):
        clear_first_output_ankle_proof(state)
        return False
    if not all(math.isfinite(value) for value in (world_x, world_z)):
        clear_first_output_ankle_proof(state)
        return False

    consistent = False
    if proof is not None:
        dt_s = observation_ts - float(proof.observation_ts)
        if (
            math.isfinite(dt_s)
            and 0.0 < dt_s <= float(config.reacquire_max_gap_s)
        ):
            step_m = math.hypot(
                world_x - float(proof.world_x),
                world_z - float(proof.world_z),
            )
            allowed_m = (
                float(config.max_jump_m)
                + float(config.max_speed_mps) * float(dt_s)
            )
            consistent = bool(step_m <= allowed_m + 1e-9)

    sample_count = (
        int(proof.consistent_samples) + 1
        if proof is not None and consistent
        else 1
    )
    state.first_output_ankle_proof = FirstOutputAnkleProof(
        world_x=world_x,
        world_z=world_z,
        observation_ts=observation_ts,
        consistent_samples=min(2, sample_count),
        binding=proof_binding,
        unavailable_rows=0,
    )
    return bool(consistent and sample_count >= 2)


def _reacquire_basis_family(contact_basis: Optional[str]) -> str:
    """Collapse exact support subtypes into their trajectory family."""

    basis = str(contact_basis or "").strip()
    if basis in {"pose_single_ankle_floor", "pose_ankle_floor"}:
        return "pose_floor"
    if basis == "pose:upright_body_planes:four_plane":
        # Retain the exact four-plane strength below while accumulating one
        # trajectory family with the generic/verified upright-body basis.
        return "pose:upright_body_planes"
    return basis


def _clear_rejection_anchor(state: PersonGroundState) -> None:
    state.rejection_anchor_x = None
    state.rejection_anchor_z = None
    state.rejection_anchor_ts = -1.0
    state.rejection_previous_world_x = None
    state.rejection_previous_world_z = None
    state.rejection_previous_world_ts = -1.0


def clear_inferred_ground_continuity_anchor(
    state: PersonGroundState,
    *,
    block_rearm: Optional[bool] = None,
    blocked_after_media_pts_ns: Optional[int] = None,
) -> None:
    """End the current learned-height occlusion episode, if any.

    Generic clearing preserves an existing re-arm block.  A caller that has
    accepted a genuine metric observation may pass ``False`` explicitly.
    """

    state.inferred_ground_continuity_anchor = None
    state.inferred_ground_raw_history.clear()
    if block_rearm is not None:
        state.inferred_ground_continuity_blocked = bool(block_rearm)
        if bool(block_rearm):
            try:
                parsed_pts = int(blocked_after_media_pts_ns)
            except (TypeError, ValueError, OverflowError):
                parsed_pts = None
            state.inferred_ground_blocked_after_media_pts_ns = (
                parsed_pts if parsed_pts is not None and parsed_pts > 0 else None
            )
        else:
            state.inferred_ground_blocked_after_media_pts_ns = None


def align_inferred_ground_observation(
    state: PersonGroundState,
    *,
    raw_measurement: np.ndarray,
    floor_y: float,
    now_ts: float,
    observed_at_us: int,
    media_pts_ns: Optional[int],
    lifecycle_generation: int,
    height_ref_scene: float,
    world_frame_id: Optional[str],
    world_frame_revision: Optional[str],
    world_transform_sha256: Optional[str],
    trusted_output_reference: Optional[
        Tuple[float, float, Optional[int], float, int]
    ],
    config: Optional[HumanGroundConfig] = None,
) -> Optional[
    Tuple[
        np.ndarray,
        InferredGroundContinuityAnchor,
        Tuple[float, float, float],
        Tuple[float, float, float],
        int,
        float,
        float,
        int,
        Optional[int],
        float,
    ]
]:
    """Bias-align one current-cohort learned-height floor observation.

    ``trusted_output_reference`` is the immutable queue-admitted output
    watermark captured at frame entry.  The first eligible raw reconstruction
    is paired with that visible point; subsequent rows apply only the raw
    perspective-aware delta to the fixed visible origin.  No inferred row can
    renew either origin.  A three-sample XZ medoid rejects one-frame body-plane
    jitter while remaining an actual recent observation rather than a clipped
    or invented point. The selected sample retains its own timestamp and media
    PTS; it is never relabeled as exact-current when the medoid selects an older
    in-window row. A missing or time-incompatible visible origin fails closed
    instead of anchoring the display to hidden filter state.
    """

    try:
        raw = np.asarray(raw_measurement, dtype=np.float64).reshape(-1)
        raw_x = float(raw[0])
        raw_z = float(raw[2])
        floor = float(floor_y)
        current_ts = float(now_ts)
        current_observed_at_us = int(observed_at_us)
        generation = int(lifecycle_generation)
        height_ref = float(height_ref_scene)
    except (TypeError, ValueError, IndexError, OverflowError):
        clear_inferred_ground_continuity_anchor(state)
        return None
    if not all(
        math.isfinite(value)
        for value in (raw_x, raw_z, floor, current_ts, height_ref)
    ):
        clear_inferred_ground_continuity_anchor(state)
        return None
    if current_observed_at_us <= 0 or generation <= 0 or height_ref <= 0.0:
        clear_inferred_ground_continuity_anchor(state)
        return None
    if bool(state.inferred_ground_continuity_blocked):
        return None
    try:
        current_pts = int(media_pts_ns) if media_pts_ns is not None else None
    except (TypeError, ValueError, OverflowError):
        current_pts = None
    if current_pts is not None and current_pts <= 0:
        current_pts = None
    # This process lane is consumed by tracking, BEV, and the strict canonical
    # world service.  Require one complete stream-time basis up front so the
    # producer cannot display a row that Menon must reject for incomplete
    # provenance.
    if current_pts is None:
        clear_inferred_ground_continuity_anchor(state)
        return None

    if trusted_output_reference is None:
        clear_inferred_ground_continuity_anchor(state)
        return None
    try:
        (
            trusted_x,
            trusted_z,
            trusted_pts,
            trusted_filter_ts,
            trusted_segment,
        ) = trusted_output_reference
        trusted_x = float(trusted_x)
        trusted_z = float(trusted_z)
        trusted_filter_ts = float(trusted_filter_ts)
        trusted_segment = int(trusted_segment)
        trusted_pts = int(trusted_pts) if trusted_pts is not None else None
    except (TypeError, ValueError, OverflowError):
        clear_inferred_ground_continuity_anchor(state)
        return None
    if not all(
        math.isfinite(value)
        for value in (trusted_x, trusted_z, trusted_filter_ts)
    ):
        clear_inferred_ground_continuity_anchor(state)
        return None
    if trusted_pts is None or trusted_pts <= 0:
        clear_inferred_ground_continuity_anchor(state)
        return None
    if current_ts < trusted_filter_ts:
        clear_inferred_ground_continuity_anchor(state)
        return None
    if (
        current_pts is not None
        and trusted_pts is not None
        and current_pts < trusted_pts
    ):
        clear_inferred_ground_continuity_anchor(state)
        return None

    normalized_frame_id = str(world_frame_id).strip() if world_frame_id else None
    normalized_revision = (
        str(world_frame_revision).strip() if world_frame_revision else None
    )
    normalized_transform = (
        str(world_transform_sha256).strip() if world_transform_sha256 else None
    )
    cfg = config or HumanGroundConfig()
    anchor = state.inferred_ground_continuity_anchor
    anchor_matches = bool(
        anchor is not None
        and int(anchor.lifecycle_generation) == generation
        and math.isclose(
            float(anchor.height_ref_scene),
            height_ref,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        and anchor.world_frame_id == normalized_frame_id
        and anchor.world_frame_revision == normalized_revision
        and anchor.world_transform_sha256 == normalized_transform
        and int(anchor.trail_segment_id) == trusted_segment
        and current_ts >= float(anchor.raw_origin_ts)
        and current_observed_at_us >= int(anchor.raw_origin_observed_at_us)
        and (
            current_pts is None
            or anchor.trusted_origin_media_pts_ns is None
            or current_pts >= int(anchor.trusted_origin_media_pts_ns)
        )
    )
    if not anchor_matches:
        state.inferred_ground_raw_history.clear()
        anchor = InferredGroundContinuityAnchor(
            raw_origin=(raw_x, floor, raw_z),
            trusted_world_origin=(trusted_x, floor, trusted_z),
            raw_origin_ts=current_ts,
            raw_origin_observed_at_us=current_observed_at_us,
            raw_origin_media_pts_ns=current_pts,
            trusted_origin_filter_ts=trusted_filter_ts,
            trusted_origin_media_pts_ns=trusted_pts,
            lifecycle_generation=generation,
            height_ref_scene=height_ref,
            world_frame_id=normalized_frame_id,
            world_frame_revision=normalized_revision,
            world_transform_sha256=normalized_transform,
            trail_segment_id=trusted_segment,
        )
        state.inferred_ground_continuity_anchor = anchor

    history = state.inferred_ground_raw_history
    raw_evidence_gap_s = 0.0
    if history:
        (
            last_ts,
            last_observed_at_us,
            last_pts,
            _last_x,
            _last_y,
            _last_z,
        ) = history[-1]
        if (
            current_ts < float(last_ts)
            or current_observed_at_us < int(last_observed_at_us)
            or (
                current_pts is not None
                and last_pts is not None
                and current_pts < int(last_pts)
            )
        ):
            clear_inferred_ground_continuity_anchor(state)
            return None
        raw_evidence_gap_s = max(0.0, current_ts - float(last_ts))
        if (
            float(cfg.inferred_process_consensus_horizon_s) > 0.0
            and current_ts - float(last_ts)
            > float(cfg.inferred_process_consensus_horizon_s)
        ):
            # A gap invalidates only the short medoid window.  Keep the fixed
            # raw/trusted origins for the whole lower-body-occlusion episode:
            # clearing or recreating the anchor here would either suppress a
            # still-visible person forever or let an inferred publication
            # renew its own authority.  The current raw row therefore starts
            # a fresh consensus window while its delta remains measured from
            # the original metric-bound origin.  Downstream physical gates
            # still bound every admitted public step.
            history.clear()
        elif (
            current_ts == float(last_ts)
            and current_observed_at_us == int(last_observed_at_us)
            and current_pts == last_pts
        ):
            history.pop()
    history.append(
        (
            current_ts,
            current_observed_at_us,
            current_pts,
            raw_x,
            floor,
            raw_z,
        )
    )
    horizon_s = float(cfg.inferred_process_consensus_horizon_s)
    if horizon_s > 0.0:
        while history and current_ts - float(history[0][0]) > horizon_s:
            history.popleft()
    sample_count = min(
        len(history),
        int(cfg.inferred_process_consensus_samples),
    )
    samples = list(history)[-sample_count:]
    if len(samples) <= 1:
        robust_sample = samples[-1]
    elif len(samples) == 2:
        # With no majority yet, prefer the exact current row. The immutable
        # origin and downstream physical filter still prevent a jump.
        robust_sample = samples[-1]
    else:
        costs: List[Tuple[float, int]] = []
        for index, sample in enumerate(samples):
            cost = sum(
                math.hypot(
                    float(sample[3]) - float(other[3]),
                    float(sample[5]) - float(other[5]),
                )
                for other in samples
            )
            # On an exact tie prefer the newest actual observation.
            costs.append((float(cost), -int(index)))
        selected_index = -min(costs)[1]
        robust_sample = samples[selected_index]
    robust_raw = (
        float(robust_sample[3]),
        floor,
        float(robust_sample[5]),
    )
    consensus_span_s = max(
        0.0,
        current_ts - float(samples[0][0]),
    )
    delta = (
        float(robust_raw[0]) - float(anchor.raw_origin[0]),
        0.0,
        float(robust_raw[2]) - float(anchor.raw_origin[2]),
    )
    aligned = np.asarray(
        [
            float(anchor.trusted_world_origin[0]) + float(delta[0]),
            floor,
            float(anchor.trusted_world_origin[2]) + float(delta[2]),
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(aligned)):
        clear_inferred_ground_continuity_anchor(state)
        return None
    return (
        aligned,
        anchor,
        delta,
        robust_raw,
        int(sample_count),
        float(consensus_span_s),
        float(robust_sample[0]),
        int(robust_sample[1]),
        int(robust_sample[2]) if robust_sample[2] is not None else None,
        float(raw_evidence_gap_s),
    )


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

    Missing one side of an explicit revision or transform digest is treated
    as a mismatch. This is intentional at the OSD boundary: an unknown frame
    cannot safely be reprojected with a known active transform.
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
    if expected_transform is not None and observed_transform != expected_transform:
        return False
    if observed_transform is not None and expected_transform is None:
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
    now_ts: Optional[float] = None,
    config: Optional[HumanGroundConfig] = None,
) -> None:
    """Record an honest no-measurement frame without advancing the filter."""

    _begin_filter_measurement(state)
    state.measurement_accepted = False
    state.measurement_rejection_reason = str(reason or "measurement_unavailable")
    state.measurement_innovation_m = math.nan
    state.measurement_allowed_m = math.nan
    preserve_one_observed_pose_floor_gap = bool(
        str(state.reacquire_candidate_basis or "") == "pose_floor"
        and int(state.reacquire_count) > 0
        and int(state.reacquire_unavailable_rows) == 0
    )
    preserve_bounded_upright_body_gap = False
    if (
        str(state.reacquire_candidate_basis or "")
        == "pose:upright_body_planes"
        and int(state.reacquire_count) > 0
        and now_ts is not None
        and config is not None
    ):
        try:
            candidate_age_s = float(now_ts) - float(
                state.reacquire_candidate_ts
            )
        except (TypeError, ValueError, OverflowError):
            candidate_age_s = math.nan
        preserve_bounded_upright_body_gap = bool(
            math.isfinite(candidate_age_s)
            and 0.0 <= candidate_age_s
            <= float(config.reacquire_max_gap_s)
        )
    if preserve_one_observed_pose_floor_gap:
        state.reacquire_unavailable_rows = 1
    elif preserve_bounded_upright_body_gap:
        # Pose cadence can be lower than detector cadence while furniture
        # hides the lower body. Keep the last actual body solve as the
        # non-renewing evidence root; elapsed media time, not callback count,
        # bounds this exception.
        state.reacquire_unavailable_rows = min(
            1_000_000,
            int(state.reacquire_unavailable_rows) + 1,
        )
    else:
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


def _max_line_gain_inside_circle(
    *,
    base_x: float,
    base_z: float,
    target_x: float,
    target_z: float,
    center_x: float,
    center_z: float,
    radius: float,
    max_gain: float,
) -> Optional[float]:
    """Return the greatest ``gain`` on a segment that is inside a disk.

    The projective posterior is ``base + gain * (target - base)``.  Solving
    the quadratic for that parameter keeps the transition algebra exact; a
    post-hoc coordinate clamp would make the emitted point unprovable from
    the filter transition.  ``None`` means the bounded segment never reaches
    the visible disk.
    """

    values = (
        base_x,
        base_z,
        target_x,
        target_z,
        center_x,
        center_z,
        radius,
        max_gain,
    )
    if not all(math.isfinite(float(value)) for value in values):
        return None
    radius = max(0.0, float(radius))
    max_gain = min(1.0, max(0.0, float(max_gain)))
    vx = float(target_x) - float(base_x)
    vz = float(target_z) - float(base_z)
    offset_x = float(base_x) - float(center_x)
    offset_z = float(base_z) - float(center_z)
    a = vx * vx + vz * vz
    radius_sq = radius * radius
    c = offset_x * offset_x + offset_z * offset_z - radius_sq
    scale = max(
        1.0,
        abs(offset_x),
        abs(offset_z),
        abs(vx) * max_gain,
        abs(vz) * max_gain,
        radius,
    )
    tolerance = 1e-12 * scale * scale
    if a <= 1e-24 * scale * scale:
        return max_gain if c <= tolerance else None

    def _inside(gain: float) -> bool:
        px = offset_x + float(gain) * vx
        pz = offset_z + float(gain) * vz
        return px * px + pz * pz <= radius_sq + tolerance

    if _inside(max_gain):
        return max_gain

    # Solve |base + gain * direction - center|^2 = radius^2.  The
    # discriminant is clamped only for round-off at a tangent; a materially
    # negative value proves that this segment misses the visible disk.
    b = 2.0 * (offset_x * vx + offset_z * vz)
    discriminant = b * b - 4.0 * a * c
    discriminant_tolerance = 1e-12 * max(1.0, b * b, abs(4.0 * a * c))
    if discriminant < -discriminant_tolerance:
        return None
    discriminant = max(0.0, discriminant)
    root = math.sqrt(discriminant)
    root_low = (-b - root) / (2.0 * a)
    root_high = (-b + root) / (2.0 * a)
    if root_low > root_high:
        root_low, root_high = root_high, root_low

    # The disk occupies the interval between the roots.  This also handles
    # an origin outside the disk: reducing the gain cannot teleport to the
    # disk before the first intersection, so report no feasible gain there.
    feasible_low = max(0.0, root_low)
    feasible_high = min(max_gain, root_high)
    if feasible_high + tolerance < feasible_low:
        return None
    if not _inside(feasible_high):
        # A root can be one ulp outside the disk after division.  Pull it
        # toward the feasible interval and verify once more before returning.
        feasible_high = min(max_gain, math.nextafter(feasible_high, -math.inf))
        if feasible_high + tolerance < feasible_low or not _inside(feasible_high):
            return None
    return float(max(0.0, min(max_gain, feasible_high)))


def _display_output_constraint(
    display_output: Optional[Tuple[float, float, Optional[int], float, int]],
    *,
    now_ts: float,
    media_pts_ns: Optional[int],
    config: HumanGroundConfig,
) -> Optional[Tuple[float, float, Optional[int], float, float]]:
    """Parse one display watermark and derive its physical slew radius.

    The returned tuple is ``(x, z, media_pts_ns, display_dt_s, max_step_m)``.
    A malformed, non-monotonic, or zero-duration watermark is ignored so the
    historical no-display behavior remains unchanged for callers that have no
    usable queue-visible timestamp.
    """

    if display_output is None:
        return None
    try:
        display_x = float(display_output[0])
        display_z = float(display_output[1])
        display_pts = (
            int(display_output[2])
            if display_output[2] is not None
            else None
        )
        display_filter_ts = float(display_output[3])
        current_pts = (
            int(media_pts_ns) if media_pts_ns is not None else None
        )
        now_ts = float(now_ts)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in (display_x, display_z, now_ts)):
        return None
    display_dt = math.nan
    if (
        current_pts is not None
        and display_pts is not None
        and current_pts >= display_pts
    ):
        display_dt = (current_pts - display_pts) / 1_000_000_000.0
    elif math.isfinite(display_filter_ts) and now_ts >= display_filter_ts:
        display_dt = now_ts - display_filter_ts
    if not math.isfinite(display_dt) or display_dt <= 0.0:
        return None
    gate_dt = float(display_dt)
    if float(config.reset_after_s) > 0.0:
        gate_dt = min(gate_dt, float(config.reset_after_s))
    max_step = float(config.max_speed_mps) * max(0.0, gate_dt)
    return (
        float(display_x),
        float(display_z),
        display_pts,
        float(display_dt),
        float(max_step),
    )


def integrate_projective_ground_observation(
    state: PersonGroundState,
    *,
    measurement: np.ndarray,
    floor_y: float,
    now_ts: float,
    config: HumanGroundConfig,
    prior_output: Optional[
        Tuple[float, float, Optional[int], float, int]
    ] = None,
    display_output: Optional[
        Tuple[float, float, Optional[int], float, int]
    ] = None,
    media_pts_ns: Optional[int] = None,
    transition_proof: Optional[Dict[str, Any]] = None,
) -> Optional[np.ndarray]:
    """Integrate an admissible image-motion floor point into filter state.

    The caller must apply the calibrated world-fusion policy before invoking
    this function.  The point is still weak evidence: it is passed through
    the exact physical CV admission and posterior-speed gate, does not update
    ``last_good_world``/the accepted image origin, and is marked estimated for
    telemetry.  ``prior_output`` is the kinematic/proof origin.  When supplied,
    ``display_output`` independently bounds the posterior against the latest
    displayed coordinate by reducing the transition gain along the same line;
    it never clips the resulting coordinate.  A rejected projective point
    leaves the bounded process state as-is and returns ``None``; it is never
    clamped into a plausible-looking relocation.
    """

    if transition_proof is not None:
        transition_proof.clear()
    try:
        candidate = np.asarray(measurement, dtype=np.float64)
        if candidate.shape[0] < 3 or not np.all(np.isfinite(candidate[:3])):
            return None
        if state.world_x is None or state.world_z is None or float(state.filtered_ts) < 0.0:
            return None
    except Exception:
        return None

    try:
        current_pts = int(media_pts_ns) if media_pts_ns is not None else None
    except (TypeError, ValueError, OverflowError):
        current_pts = None

    prior_reference: Optional[Tuple[float, float, float]] = None
    proof_prior_pts: Optional[int] = None
    proof_current_pts: Optional[int] = None
    proof_prior_segment: Optional[int] = None
    if prior_output is not None:
        try:
            prior_x = float(prior_output[0])
            prior_z = float(prior_output[1])
            prior_pts = (
                int(prior_output[2])
                if prior_output[2] is not None
                else None
            )
            prior_filter_ts = float(prior_output[3])
            prior_segment = int(prior_output[4])
        except (TypeError, ValueError, IndexError, OverflowError):
            prior_x = prior_z = prior_filter_ts = math.nan
            prior_pts = None
            prior_segment = -1
        prior_dt = math.nan
        if (
            current_pts is not None
            and prior_pts is not None
            and current_pts >= prior_pts
        ):
            prior_dt = (current_pts - prior_pts) / 1_000_000_000.0
        elif math.isfinite(prior_filter_ts) and float(now_ts) >= prior_filter_ts:
            prior_dt = float(now_ts) - prior_filter_ts
        if (
            all(math.isfinite(value) for value in (prior_x, prior_z, prior_dt))
            and prior_dt > 0.0
        ):
            prior_reference = (
                float(prior_x),
                float(prior_z),
                float(now_ts) - float(prior_dt),
            )
            proof_prior_pts = prior_pts
            proof_current_pts = current_pts
            proof_prior_segment = prior_segment

    display_constraint = _display_output_constraint(
        display_output,
        now_ts=float(now_ts),
        media_pts_ns=current_pts,
        config=config,
    )
    display_reference: Optional[Tuple[float, float, float]] = (
        (
            float(display_constraint[0]),
            float(display_constraint[1]),
            float(now_ts) - float(display_constraint[3]),
        )
        if display_constraint is not None
        else None
    )
    display_pts = (
        display_constraint[2] if display_constraint is not None else None
    )
    display_dt = (
        float(display_constraint[3]) if display_constraint is not None else math.nan
    )

    if transition_proof is not None and (
        prior_reference is None
        or proof_prior_pts is None
        or proof_current_pts is None
        or proof_prior_pts <= 0
        or proof_current_pts <= proof_prior_pts
        or proof_prior_segment is None
        or proof_prior_segment < 0
    ):
        # Canonical image-motion rows require one exact queue-visible origin
        # and one complete media-time transition.  Without it tracking/BEV
        # could display a row that the strict world/Menon boundary cannot
        # independently verify.
        return None

    previous_world_x = float(
        prior_reference[0]
        if prior_reference is not None
        else (
            state.rejection_previous_world_x
            if state.rejection_previous_world_x is not None
            else state.world_x
        )
    )
    previous_world_z = float(
        prior_reference[1]
        if prior_reference is not None
        else (
            state.rejection_previous_world_z
            if state.rejection_previous_world_z is not None
            else state.world_z
        )
    )
    previous_filtered_ts = float(
        prior_reference[2]
        if prior_reference is not None
        else (
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
    )
    # Projective evidence may bridge display/process continuity between exact
    # metric pose samples, but it is neither allowed to increment nor erase
    # their same-basis reacquisition consensus.
    metric_reacquire_consensus = (
        state.reacquire_candidate_x,
        state.reacquire_candidate_z,
        state.reacquire_candidate_ts,
        state.reacquire_candidate_basis,
        state.reacquire_candidate_exact_basis,
        state.reacquire_count,
        state.reacquire_unavailable_rows,
    )

    def _restore_metric_reacquire_consensus() -> None:
        (
            state.reacquire_candidate_x,
            state.reacquire_candidate_z,
            state.reacquire_candidate_ts,
            state.reacquire_candidate_basis,
            state.reacquire_candidate_exact_basis,
            state.reacquire_count,
            state.reacquire_unavailable_rows,
        ) = metric_reacquire_consensus

    # A rejected metric candidate may already have advanced the process state
    # to this exact frame timestamp.  Evaluate the independent image-motion
    # candidate from the fixed rejection anchor in that case; otherwise the
    # normal filter sees a zero dt and incorrectly reports a timestamp reject.
    # The temporary state keeps the original bounded posterior untouched when
    # the projective candidate fails the same physical gate.
    evaluation_state = state
    rewound_from_prior_output = prior_reference is not None
    rewound_from_rejection = bool(
        not rewound_from_prior_output
        and not bool(state.measurement_accepted)
        and state.rejection_anchor_x is not None
        and state.rejection_anchor_z is not None
        and math.isfinite(float(state.rejection_anchor_ts))
        and math.isfinite(float(state.filtered_ts))
        and float(state.filtered_ts) == float(now_ts)
        and float(state.rejection_anchor_ts) < float(now_ts)
    )
    if rewound_from_prior_output:
        evaluation_state = copy(state)
        evaluation_state.world_x = float(prior_reference[0])
        evaluation_state.world_z = float(prior_reference[1])
        evaluation_state.filtered_ts = float(prior_reference[2])
        evaluation_state.measurement_accepted = True
        evaluation_state.measurement_rejection_reason = None
        evaluation_state.measurement_innovation_m = 0.0
        evaluation_state.measurement_allowed_m = 0.0
        evaluation_state.reacquired = False
        _clear_reacquire_candidate(evaluation_state)
        _clear_rejection_anchor(evaluation_state)
    elif rewound_from_rejection:
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

    # A prior/display bound is an alternate constraint on this exact
    # transition.  Isolate the trial state so an infeasible line/disk
    # intersection cannot mutate the kinematic state; callers with neither
    # bound keep the historical in-place path byte-for-byte in behavior.
    isolated_for_display = bool(display_reference is not None and evaluation_state is state)
    isolated_for_constraint = bool(
        (prior_reference is not None or display_reference is not None)
        and evaluation_state is state
    )
    if isolated_for_constraint:
        evaluation_state = copy(state)
    initial_evaluation_state = (
        copy(evaluation_state)
        if prior_reference is not None or display_reference is not None
        else None
    )

    # Projective/body-geometry observations are displacement evidence, not a
    # second metric position authority.  A noisy reconstruction can therefore
    # land just outside the weak-measurement innovation disk even though the
    # exact queue-visible output and its constant-velocity prior are sound.  A
    # hard reject in that case creates a display hole before the stricter
    # output-speed gate can do its job.  Robustify only this proof-bearing
    # projective lane: move the filter target to the exact intersection of the
    # predicted-base -> raw-observation ray and the ordinary innovation disk.
    # The raw observation remains in provenance; the canonical service verifies
    # this algebra independently.  Ordinary metric observations retain the
    # historical reject-only gate.
    filter_candidate = np.asarray(candidate, dtype=np.float64).copy()
    innovation_limit_applied = False
    innovation_scale = 1.0
    raw_innovation_m = math.nan
    innovation_limit_m = math.nan
    if prior_reference is not None:
        candidate_dt = float(now_ts) - float(evaluation_state.filtered_ts)
        if candidate_dt > 0.0:
            candidate_gate_dt = float(candidate_dt)
            if float(config.reset_after_s) > 0.0:
                candidate_gate_dt = min(
                    candidate_gate_dt,
                    float(config.reset_after_s),
                )
            candidate_vel_x, candidate_vel_z = _clamp_speed(
                float(evaluation_state.vel_world_x),
                float(evaluation_state.vel_world_z),
                float(config.max_speed_mps),
            )
            candidate_base_x = float(evaluation_state.world_x) + (
                float(candidate_vel_x) * candidate_gate_dt
            )
            candidate_base_z = float(evaluation_state.world_z) + (
                float(candidate_vel_z) * candidate_gate_dt
            )
            raw_innovation_m = math.hypot(
                float(candidate[0]) - candidate_base_x,
                float(candidate[2]) - candidate_base_z,
            )
            innovation_limit_m = float(config.max_jump_m) + (
                float(config.max_speed_mps) * candidate_gate_dt
            )
            if (
                math.isfinite(raw_innovation_m)
                and math.isfinite(innovation_limit_m)
                and innovation_limit_m > 0.0
                and raw_innovation_m > innovation_limit_m + 1e-9
            ):
                innovation_scale = innovation_limit_m / raw_innovation_m
                filter_candidate[0] = candidate_base_x + innovation_scale * (
                    float(candidate[0]) - candidate_base_x
                )
                filter_candidate[2] = candidate_base_z + innovation_scale * (
                    float(candidate[2]) - candidate_base_z
                )
                innovation_limit_applied = True

    filter_update_proof: Optional[Dict[str, Any]] = (
        {}
        if (
            transition_proof is not None
            or prior_reference is not None
            or display_reference is not None
        )
        else None
    )
    result = update_human_cv_filter(
        evaluation_state,
        measurement=filter_candidate,
        floor_y=float(floor_y),
        now_ts=float(now_ts),
        quality="weak",
        config=config,
        force_accept=False,
        contact_basis=None,
        image_motion_supported=False,
        # The first projective trial is deliberately allowed to expose a
        # posterior beyond the queue-visible prior disk.  It is evaluated on
        # a copy solely to obtain the exact filter transition line; the final
        # emitted state is always rerun with a reduced gain below.  This
        # escape hatch is private to this function and does not alter metric
        # observations.
        allow_physical_output_overrun=bool(prior_reference is not None),
        transition_proof=filter_update_proof,
    )
    if not bool(evaluation_state.measurement_accepted):
        _restore_metric_reacquire_consensus()
        return None

    if rewound_from_prior_output or rewound_from_rejection:
        prior_step_dt = max(0.0, float(now_ts) - previous_filtered_ts)
        prior_step = math.hypot(
            float(result[0]) - previous_world_x,
            float(result[2]) - previous_world_z,
        )
        prior_gate_dt = float(prior_step_dt)
        if float(config.reset_after_s) > 0.0:
            prior_gate_dt = min(prior_gate_dt, float(config.reset_after_s))
        prior_step_limit = float(config.max_speed_mps) * max(0.0, prior_gate_dt)
        if (
            float(config.max_speed_mps) > 0.0
            and prior_step > prior_step_limit + 1e-9
        ):
            # The ordinary projective posterior may be physically admissible
            # from the hidden filter state while still exceeding the exact
            # queue-visible prior disk.  Solve for the greatest gain on the
            # *same* filter transition line inside that disk, then rerun from
            # the untouched evaluation state.  Never clip the already emitted
            # coordinate: the rerun updates position, velocity, and proof as
            # one coherent transition.
            try:
                prior_base_x = float(filter_update_proof["position_base"][0])
                prior_base_z = float(filter_update_proof["position_base"][2])
                prior_unconstrained_gain = float(
                    filter_update_proof["position_gain"]
                )
            except (KeyError, TypeError, ValueError, IndexError, OverflowError):
                prior_base_x = prior_base_z = prior_unconstrained_gain = math.nan
            prior_gain = _max_line_gain_inside_circle(
                base_x=prior_base_x,
                base_z=prior_base_z,
                target_x=float(filter_candidate[0]),
                target_z=float(filter_candidate[2]),
                center_x=float(previous_world_x),
                center_z=float(previous_world_z),
                radius=float(prior_step_limit),
                max_gain=float(prior_unconstrained_gain),
            )
            if (
                prior_gain is None
                or prior_gain <= 1e-12
                or initial_evaluation_state is None
            ):
                # A zero/no intersection is not a projective observation.
                # Leave the caller's existing process state untouched so its
                # explicit hold path remains authoritative.
                _restore_metric_reacquire_consensus()
                return None
            if prior_gain < float(prior_unconstrained_gain) - 1e-12:
                evaluation_state = copy(initial_evaluation_state)
                filter_update_proof = {}
                result = update_human_cv_filter(
                    evaluation_state,
                    measurement=filter_candidate,
                    floor_y=float(floor_y),
                    now_ts=float(now_ts),
                    quality="weak",
                    config=config,
                    force_accept=False,
                    contact_basis=None,
                    image_motion_supported=False,
                    position_gain_override=float(prior_gain),
                    transition_proof=filter_update_proof,
                )
                if not bool(evaluation_state.measurement_accepted):
                    _restore_metric_reacquire_consensus()
                    return None
                prior_gain_reduced = True
            else:
                prior_gain_reduced = False
            prior_unconstrained_gain_for_proof = float(prior_unconstrained_gain)
        else:
            prior_step_limit = float(prior_step_limit)
            prior_gain_reduced = False
            try:
                prior_unconstrained_gain_for_proof = float(
                    filter_update_proof["position_gain"]
                )
            except (KeyError, TypeError, ValueError, IndexError, OverflowError):
                prior_unconstrained_gain_for_proof = math.nan
    else:
        prior_step_limit = None
        prior_gain_reduced = False
        prior_unconstrained_gain_for_proof = None

    # The prior reduction above is the kinematic constraint.  Apply the
    # display constraint to the resulting gain on the same line.  If both
    # disks bind, this computes the greatest gain satisfying both and reruns
    # exactly once from the untouched origin with that final gain.
    display_gain_reduced = False
    display_unconstrained_gain: Optional[float] = None
    display_step_limit: Optional[float] = None
    if display_reference is not None:
        display_step_limit = float(display_constraint[4])
        try:
            display_step = math.hypot(
                float(result[0]) - float(display_reference[0]),
                float(result[2]) - float(display_reference[1]),
            )
            proof_base = (
                float(filter_update_proof["position_base"][0])
                if filter_update_proof is not None
                else math.nan
            )
            proof_base_z = (
                float(filter_update_proof["position_base"][2])
                if filter_update_proof is not None
                else math.nan
            )
            unconstrained_gain = (
                float(filter_update_proof["position_gain"])
                if filter_update_proof is not None
                else math.nan
            )
        except (KeyError, TypeError, ValueError, IndexError, OverflowError):
            display_step = proof_base = proof_base_z = unconstrained_gain = math.nan
        if (
            math.isfinite(display_step)
            and math.isfinite(display_step_limit)
            and display_step > display_step_limit + 1e-9
            and math.isfinite(proof_base)
            and math.isfinite(proof_base_z)
            and math.isfinite(unconstrained_gain)
        ):
            display_gain = _max_line_gain_inside_circle(
                base_x=proof_base,
                base_z=proof_base_z,
                target_x=float(filter_candidate[0]),
                target_z=float(filter_candidate[2]),
                center_x=float(display_reference[0]),
                center_z=float(display_reference[1]),
                radius=float(display_step_limit),
                max_gain=float(unconstrained_gain),
            )
            if display_gain is None or display_gain <= 1e-12:
                # A zero-gain innovation_update is not a projective
                # observation. Let the caller emit its explicit gain-zero
                # output hold instead of constructing a row the strict world
                # boundary would (correctly) reject as image motion.
                _restore_metric_reacquire_consensus()
                return None
            if initial_evaluation_state is None:
                _restore_metric_reacquire_consensus()
                return None
            # Re-run the same filter transition from the untouched kinematic
            # origin with the smaller gain.  This updates velocity and state
            # consistently; it does not clip the already computed coordinate.
            evaluation_state = copy(initial_evaluation_state)
            filter_update_proof = (
                {} if transition_proof is not None or display_reference is not None else None
            )
            result = update_human_cv_filter(
                evaluation_state,
                measurement=filter_candidate,
                floor_y=float(floor_y),
                now_ts=float(now_ts),
                quality="weak",
                config=config,
                force_accept=False,
                contact_basis=None,
                image_motion_supported=False,
                position_gain_override=float(display_gain),
                transition_proof=filter_update_proof,
            )
            if not bool(evaluation_state.measurement_accepted):
                _restore_metric_reacquire_consensus()
                return None
            display_gain_reduced = bool(display_gain < float(unconstrained_gain) - 1e-12)
            display_unconstrained_gain = float(unconstrained_gain)

    if (
        rewound_from_prior_output
        or rewound_from_rejection
        or isolated_for_display
        or display_gain_reduced
    ):
        state.__dict__.update(evaluation_state.__dict__)
    _restore_metric_reacquire_consensus()

    if transition_proof is not None:
        if not filter_update_proof:
            # Every accepted established-state update emits this proof at the
            # exact branch that chose its posterior.  Treat absence as an
            # internal contract failure instead of publishing an unprovable
            # coordinate.
            return None
        transition_proof.update(filter_update_proof)
        if innovation_limit_applied:
            transition_proof.update(
                {
                    "innovation_limit_applied": True,
                    "raw_position_target": [
                        float(candidate[0]),
                        float(floor_y),
                        float(candidate[2]),
                    ],
                    "filter_observation": [
                        float(filter_candidate[0]),
                        float(floor_y),
                        float(filter_candidate[2]),
                    ],
                    "raw_innovation_m": float(raw_innovation_m),
                    "innovation_limit_m": float(innovation_limit_m),
                    "innovation_scale": float(innovation_scale),
                }
            )
        transition_proof.update(
            {
                "origin_kind": "queue_admitted_world_output",
                "origin_world": [
                    float(previous_world_x),
                    float(floor_y),
                    float(previous_world_z),
                ],
                "origin_media_pts_ns": int(proof_prior_pts),
                "current_media_pts_ns": int(proof_current_pts),
                "origin_trail_segment_id": int(proof_prior_segment),
                "max_speed_mps": float(config.max_speed_mps),
                "max_jump_m": float(config.max_jump_m),
                "reset_after_s": float(config.reset_after_s),
            }
        )
        if (
            prior_reference is not None
            and prior_step_limit is not None
            and prior_unconstrained_gain_for_proof is not None
            and math.isfinite(float(prior_unconstrained_gain_for_proof))
        ):
            transition_proof.update(
                {
                    "prior_output_gate_dt_s": float(prior_gate_dt),
                    "prior_output_max_step_m": float(prior_step_limit),
                    "prior_output_position_gain_reduced": bool(
                        prior_gain_reduced
                    ),
                    "prior_output_unconstrained_position_gain": float(
                        prior_unconstrained_gain_for_proof
                    ),
                }
            )
        if display_reference is not None and display_step_limit is not None:
            transition_proof.update(
                {
                    "visible_origin_kind": "latest_displayed_world_output",
                    "visible_origin_world": [
                        float(display_reference[0]),
                        float(floor_y),
                        float(display_reference[1]),
                    ],
                    "visible_origin_media_pts_ns": (
                        int(display_pts) if display_pts is not None else None
                    ),
                    "visible_gate_dt_s": float(
                        min(
                            float(display_dt),
                            float(config.reset_after_s)
                            if float(config.reset_after_s) > 0.0
                            else float(display_dt),
                        )
                    ),
                    "visible_max_step_m": float(display_step_limit),
                    "visible_position_gain_reduced": bool(display_gain_reduced),
                    "visible_unconstrained_position_gain": (
                        float(display_unconstrained_gain)
                        if display_unconstrained_gain is not None
                        else float(filter_update_proof.get("position_gain", 0.0))
                    ),
                }
            )

    # This posterior is physically admissible but not a fresh metric/depth
    # measurement. Keep it out of the metric authority fields while making it
    # the canonical process origin for the next frame.
    state.measurement_accepted = False
    state.measurement_rejection_reason = "projective_weak_observation"
    state.trail_append_allowed = True
    state.rejection_anchor_x = float(state.world_x)
    state.rejection_anchor_z = float(state.world_z)
    state.rejection_anchor_ts = float(state.filtered_ts)
    state.projective_bridge_origin_ts = float(state.filtered_ts)
    state.projective_bridge_process_ts = float(state.filtered_ts)
    state.projective_bridge_rows_remaining = 1
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
    evidence_ts: Optional[float] = None,
    config: HumanGroundConfig,
) -> LowerBodyOcclusionAssessment:
    """Classify standing-person occlusion from feet through waist/hips.

    The state is intentionally anchored to previously observed upright anatomy.
    A short detector box corroborated by missing lower keypoints can therefore
    demote a syntactically valid waist/counter-edge anchor before it reaches
    world fusion. Missing pose keypoints alone are model visibility loss, not
    proof that the detector silhouette is physically truncated; when the
    established upright box remains intact its bottom edge remains eligible for
    the separately guarded floor-contact path.
    Clear lower-body evidence is required for several frames before direct
    ankle/depth authority resumes, which prevents one-frame pose hallucinations
    from snapping a path back and forth.
    """

    now_ts = float(now_ts)
    try:
        exit_evidence_ts = (
            float(evidence_ts) if evidence_ts is not None else float(now_ts)
        )
    except (TypeError, ValueError, OverflowError):
        exit_evidence_ts = float(now_ts)
    if not math.isfinite(exit_evidence_ts):
        exit_evidence_ts = float(now_ts)
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

    bbox_width, bbox_height = _bbox_dimensions(bbox)
    try:
        bbox_left = float(bbox[0]) if bbox is not None and len(bbox) >= 4 else None
        bbox_top = float(bbox[1]) if bbox is not None and len(bbox) >= 4 else None
    except (TypeError, ValueError, IndexError, OverflowError):
        bbox_left = bbox_top = None
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

    visible_ankle_points = (
        [
            point
            for point in (
                pose_point(
                    kpts_abs,
                    "left_ankle",
                    conf_threshold=threshold,
                ),
                pose_point(
                    kpts_abs,
                    "right_ankle",
                    conf_threshold=threshold,
                ),
            )
            if point is not None
        ]
        if isinstance(kpts_abs, np.ndarray) and kpts_abs.shape[0] >= 17
        else []
    )
    pose_contact_inside_silhouette = True
    if (
        visible_ankle_points
        and bbox_left is not None
        and bbox_top is not None
        and bbox_width is not None
        and bbox_height is not None
        and bbox_width > 1.0
        and bbox_height > 1.0
    ):
        margin_x = 0.15 * float(bbox_width)
        lower_margin = max(8.0, 0.12 * float(bbox_height))
        bbox_bottom = float(bbox_top) + float(bbox_height)
        pose_contact_inside_silhouette = any(
            float(bbox_left) - margin_x
            <= float(point[0])
            <= float(bbox_left) + float(bbox_width) + margin_x
            and float(bbox_top) + 0.45 * float(bbox_height)
            <= float(point[1])
            <= bbox_bottom + lower_margin
            for point in visible_ankle_points
        )
    pose_contact_conflict = bool(
        visible_ankle_points and not pose_contact_inside_silhouette
    )

    full_body_visible = bool(
        visible_shoulders >= 1
        and visible_hips >= 1
        and visible_knees >= 1
        and visible_ankles >= 1
        and pose_contact_inside_silhouette
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
        if pose_contact_conflict:
            detected_level = "feet_ankles"
            confidence = 0.97
            reasons.append("pose_contact_outside_detector_silhouette")
        elif (
            collapsed_bbox
            and upper_body_visible
            and visible_hips == 0
            and visible_knees == 0
        ):
            detected_level = "waist_hips"
            confidence = 0.96 if visible_ankles == 0 else 0.88
            reasons.append("hips_knees_missing")
        elif (
            collapsed_bbox
            and upper_body_visible
            and visible_hips == 1
            and visible_knees == 0
            and visible_ankles == 0
        ):
            detected_level = "waist_hips"
            confidence = 0.92
            reasons.append("partial_hips_only")
        elif (
            collapsed_bbox
            and visible_hips >= 1
            and visible_knees == 0
            and visible_ankles == 0
        ):
            detected_level = "knees"
            confidence = 0.89
            reasons.append("knees_ankles_missing")
        elif (
            collapsed_bbox
            and visible_hips >= 1
            and visible_knees >= 1
            and visible_ankles == 0
        ):
            detected_level = "feet_ankles"
            confidence = 0.82
            reasons.append("ankles_missing")
        elif kpts_abs is None and collapsed_bbox:
            detected_level = "waist_hips"
            confidence = 0.84
            reasons.append("pose_missing_bbox_collapsed")
        elif collapsed_bbox and not full_body_visible:
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
        state.post_occlusion_reacquire_support_required = True
        state.lower_body_occlusion_level = str(detected_level)
        state.lower_body_occlusion_confidence = float(confidence)
        state.lower_body_occlusion_reason = ",".join(dict.fromkeys(reasons))
        state.lower_body_clear_frames = 0
        state.lower_body_clear_since_ts = -1.0
        state.lower_body_non_upright_frames = 0
        state.lower_body_non_upright_since_ts = -1.0
    elif state.lower_body_occluded and explicit_non_upright:
        moving_occlusion_transition = bool(
            str(state.motion_mode) == "walk"
            and bool(state.image_motion_supported)
            and int(state.image_motion_streak) >= int(config.static_exit_frames)
        )
        pending_transition_confirmation = bool(
            int(state.lower_body_non_upright_frames) > 0
            and float(state.lower_body_non_upright_since_ts) >= 0.0
        )
        if moving_occlusion_transition or pending_transition_confirmation:
            if float(state.lower_body_non_upright_since_ts) < 0.0:
                state.lower_body_non_upright_since_ts = exit_evidence_ts
            state.lower_body_non_upright_frames = (
                int(state.lower_body_non_upright_frames) + 1
            )
            state.lower_body_clear_frames = 0
            state.lower_body_clear_since_ts = -1.0
            non_upright_elapsed_s = max(
                0.0,
                exit_evidence_ts
                - float(state.lower_body_non_upright_since_ts),
            )
            if (
                int(state.lower_body_non_upright_frames)
                >= int(config.occlusion_exit_frames)
                and non_upright_elapsed_s + 1e-9
                >= float(config.occlusion_exit_s)
            ):
                state.lower_body_occluded = False
                state.lower_body_occlusion_level = "none"
                state.lower_body_occlusion_confidence = 0.0
                state.lower_body_occlusion_reason = None
                state.lower_body_non_upright_frames = 0
                state.lower_body_non_upright_since_ts = -1.0
        else:
            state.lower_body_occluded = False
            state.lower_body_occlusion_level = "none"
            state.lower_body_occlusion_confidence = 0.0
            state.lower_body_occlusion_reason = None
            state.lower_body_clear_frames = 0
            state.lower_body_clear_since_ts = -1.0
            state.lower_body_non_upright_frames = 0
            state.lower_body_non_upright_since_ts = -1.0
    elif state.lower_body_occluded:
        state.lower_body_non_upright_frames = 0
        state.lower_body_non_upright_since_ts = -1.0
        if full_body_visible:
            if float(state.lower_body_clear_since_ts) < 0.0:
                state.lower_body_clear_since_ts = exit_evidence_ts
            state.lower_body_clear_frames = int(state.lower_body_clear_frames) + 1
            clear_elapsed_s = max(
                0.0,
                exit_evidence_ts - float(state.lower_body_clear_since_ts),
            )
            if (
                int(state.lower_body_clear_frames)
                >= int(config.occlusion_exit_frames)
                and clear_elapsed_s + 1e-9 >= float(config.occlusion_exit_s)
            ):
                state.lower_body_occluded = False
                state.lower_body_occlusion_level = "none"
                state.lower_body_occlusion_confidence = 0.0
                state.lower_body_occlusion_reason = None
                state.lower_body_clear_frames = 0
                state.lower_body_clear_since_ts = -1.0
        else:
            # Missing pose or an ambiguous detector silhouette is not evidence
            # that the hidden lower body returned.  Retain the established
            # occlusion episode so its bounded floor-contact process remains
            # available, and require a fresh uninterrupted full-body window.
            state.lower_body_clear_frames = 0
            state.lower_body_clear_since_ts = -1.0
    else:
        state.lower_body_clear_frames = 0
        state.lower_body_clear_since_ts = -1.0
        state.lower_body_non_upright_frames = 0
        state.lower_body_non_upright_since_ts = -1.0

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
    observation_ts: float,
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
    window of consecutive *published observations* and on one stable contact
    basis.  Published tracking cohorts may intentionally skip source frames at
    the configured inference cadence, so continuity is proved by monotonically
    increasing source frame IDs plus a bounded capture-time gap rather than by
    requiring adjacent raw frame numbers.

    The returned boolean describes the *current* coherent observation.  Three
    consecutive observations (``static_exit_frames``) pre-unlock an idle state
    before the world filter runs.  This does not admit a world measurement;
    the physical innovation gate remains authoritative.
    """

    try:
        current_frame_id = int(frame_id)
        current_ts = float(observation_ts)
        basis = str(contact_basis or "").strip()
        if (
            current_frame_id < 0
            or not math.isfinite(current_ts)
            or current_ts < 0.0
            or not basis
            or image_foot_uv is None
        ):
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
        state.image_motion_ts = -1.0
        state.image_motion_contact_basis = None
        state.image_motion_window.clear()
        state.image_motion_streak = 0
        state.image_motion_supported = False
        state.image_motion_contact_distance_px = 0.0
        state.image_motion_bbox_distance_px = 0.0
        state.image_motion_bootstrap_supported = False
        state.image_motion_bootstrap_ts = -1.0
        state.image_motion_bootstrap_basis = None
        return False

    required_observations = max(2, int(config.static_exit_frames))
    if state.image_motion_window.maxlen != required_observations:
        state.image_motion_window = deque(
            list(state.image_motion_window)[-required_observations:],
            maxlen=required_observations,
        )
    observation_gap_s = current_ts - float(state.image_motion_ts)
    consecutive = bool(
        current_frame_id > int(state.image_motion_frame_id)
        and observation_gap_s > 0.0
        and observation_gap_s <= float(config.reacquire_max_gap_s)
    )
    same_basis = basis == str(state.image_motion_contact_basis or "")
    if not consecutive or not same_basis:
        state.image_motion_window.clear()
        state.image_motion_bootstrap_ts = -1.0
        state.image_motion_bootstrap_basis = None
    state.image_motion_window.append(
        (
            int(current_frame_id),
            str(basis),
            float(current_ts),
            float(contact_u),
            float(contact_v),
            float(bbox_u),
            float(bbox_v),
        )
    )

    coherent = False
    bootstrap_supported = False
    contact_distance = 0.0
    bbox_distance = 0.0
    if len(state.image_motion_window) == required_observations:
        first = state.image_motion_window[0]
        last = state.image_motion_window[-1]
        contact_du = float(last[3]) - float(first[3])
        contact_dv = float(last[4]) - float(first[4])
        bbox_du = float(last[5]) - float(first[5])
        bbox_dv = float(last[6]) - float(first[6])
        contact_distance = math.hypot(contact_du, contact_dv)
        bbox_distance = math.hypot(bbox_du, bbox_dv)
        step_directions_coherent = True
        for index in range(1, len(state.image_motion_window)):
            previous = state.image_motion_window[index - 1]
            current = state.image_motion_window[index]
            step_dot = (
                (float(current[3]) - float(previous[3]))
                * (float(current[5]) - float(previous[5]))
                + (float(current[4]) - float(previous[4]))
                * (float(current[6]) - float(previous[6]))
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
        if coherent:
            displacement_dot = contact_du * bbox_du + contact_dv * bbox_dv
            displacement_norm = contact_distance * bbox_distance
            direction_cosine = (
                displacement_dot / displacement_norm
                if displacement_norm > 1e-9
                else -1.0
            )
            displacement_ratio = min(contact_distance, bbox_distance) / max(
                contact_distance,
                bbox_distance,
            )
            bootstrap_motion_px = max(
                8.0,
                float(config.static_px_threshold) * 2.0,
            )
            bootstrap_supported = bool(
                contact_distance > bootstrap_motion_px
                and bbox_distance > bootstrap_motion_px
                and direction_cosine >= 0.75
                and displacement_ratio >= 0.35
            )

    state.image_motion_frame_id = int(current_frame_id)
    state.image_motion_ts = float(current_ts)
    state.image_motion_contact_basis = str(basis)
    state.image_motion_streak = len(state.image_motion_window) if coherent else 0
    state.image_motion_supported = bool(coherent)
    state.image_motion_contact_distance_px = float(contact_distance)
    state.image_motion_bbox_distance_px = float(bbox_distance)
    if bootstrap_supported:
        state.image_motion_bootstrap_ts = float(current_ts)
        state.image_motion_bootstrap_basis = str(basis)
    bootstrap_age_s = current_ts - float(state.image_motion_bootstrap_ts)
    state.image_motion_bootstrap_supported = bool(
        state.image_motion_bootstrap_ts >= 0.0
        and str(state.image_motion_bootstrap_basis or "") == str(basis)
        and 0.0 <= bootstrap_age_s <= float(config.reacquire_max_gap_s)
    )

    if coherent:
        # Stationary hold uses a deliberately wider per-observation jitter
        # tolerance than the accumulated motion detector. Slow real movement
        # can therefore look stationary one step at a time while becoming
        # unambiguously directional across the complete window. The stronger
        # current motion proof wins; a moving track must not retain eligibility
        # for a frozen-anchor hold or block motion-backed cold bootstrap.
        state.bbox_stationary_streak = 0
        state.bbox_stationary_supported = False

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


def mark_image_motion_observation_unavailable(
    state: PersonGroundState,
    *,
    observation_ts: float,
    config: HumanGroundConfig,
) -> None:
    """Record a missing motion witness without inventing or erasing evidence.

    Pose inference can omit a torso/contact witness for one published cohort
    even though the surrounding exact observations remain usable.  A missing
    witness is not a zero-motion sample and must not advance the consensus,
    but it also must not erase prior *observed* samples inside the existing
    reacquisition horizon.  Expired or malformed time still resets everything
    so evidence cannot bridge an unbounded gap.
    """

    state.image_motion_streak = 0
    state.image_motion_supported = False
    state.image_motion_contact_distance_px = 0.0
    state.image_motion_bbox_distance_px = 0.0
    state.image_motion_bootstrap_supported = False

    try:
        current_ts = float(observation_ts)
        gap_s = current_ts - float(state.image_motion_ts)
        evidence_is_current = bool(
            math.isfinite(current_ts)
            and current_ts >= 0.0
            and float(state.image_motion_ts) >= 0.0
            and gap_s >= 0.0
            and gap_s <= float(config.reacquire_max_gap_s)
        )
    except Exception:
        evidence_is_current = False

    if evidence_is_current:
        return

    state.image_motion_frame_id = -1
    state.image_motion_ts = -1.0
    state.image_motion_contact_basis = None
    state.image_motion_window.clear()
    state.image_motion_bootstrap_ts = -1.0
    state.image_motion_bootstrap_basis = None


def observe_bbox_stationarity(
    state: PersonGroundState,
    *,
    frame_id: int,
    observation_ts: float,
    bbox: Optional[Sequence[float]],
    config: HumanGroundConfig,
) -> bool:
    """Gate a short hold on exact-cohort detector-box continuity.

    This is deliberately weaker than metric admission: it can support only a
    non-appending display hold for an already trusted point.  It never creates
    a world measurement or authorizes a trail update.  Tracking publication
    intentionally skips raw source frames, so consecutive observations mean a
    monotonically increasing frame ID within the bounded media-time gap.
    Missing/stale observations and movement beyond a small detector-jitter
    tolerance reset the streak.
    """

    try:
        current_frame_id = int(frame_id)
        current_ts = float(observation_ts)
        if current_frame_id < 0 or bbox is None or len(bbox) < 4:
            raise ValueError
        left, _top, width, height = (float(value) for value in bbox[:4])
        center_u = left + width * 0.5
        bottom_v = float(_top) + height
        if width <= 1.0 or height <= 1.0 or not all(
            math.isfinite(value) for value in (current_ts, center_u, bottom_v)
        ):
            raise ValueError
        if current_ts < 0.0:
            raise ValueError
    except Exception:
        state.bbox_motion_frame_id = -1
        state.bbox_motion_ts = -1.0
        state.bbox_center_u = None
        state.bbox_bottom_v = None
        state.bbox_geometry = None
        state.bbox_stationary_root_geometry = None
        state.bbox_stationary_streak = 0
        state.bbox_stationary_supported = False
        return False

    observation_gap_s = current_ts - float(state.bbox_motion_ts)
    consecutive = bool(
        current_frame_id > int(state.bbox_motion_frame_id)
        and observation_gap_s > 0.0
        and observation_gap_s <= float(config.reacquire_max_gap_s)
    )
    tolerance = max(4.0, float(config.static_px_threshold) * 2.0)
    current_geometry = (
        float(left),
        float(_top),
        float(width),
        float(height),
    )

    def _matches_stationary_reference(
        reference: Optional[Tuple[float, float, float, float]],
    ) -> bool:
        if reference is None:
            return False
        try:
            ref_left, ref_top, ref_width, ref_height = (
                float(value) for value in reference
            )
        except (TypeError, ValueError, OverflowError):
            return False
        if ref_width <= 1.0 or ref_height <= 1.0:
            return False
        ref_center_u = ref_left + ref_width * 0.5
        ref_bottom_v = ref_top + ref_height
        # Position uses the existing detector-jitter budget. Scale uses a
        # dimensionless ten-percent envelope so the rule behaves consistently
        # across raster sizes and near/far person boxes.
        return bool(
            abs(center_u - ref_center_u) <= tolerance
            and abs(bottom_v - ref_bottom_v) <= tolerance
            and abs(width / ref_width - 1.0) <= 0.10
            and abs(height / ref_height - 1.0) <= 0.10
        )

    latest_stationary = _matches_stationary_reference(state.bbox_geometry)
    root_stationary = _matches_stationary_reference(
        state.bbox_stationary_root_geometry
    )
    if not consecutive or not latest_stationary or not root_stationary:
        state.bbox_stationary_streak = 1
        state.bbox_stationary_root_geometry = current_geometry
    else:
        state.bbox_stationary_streak = int(state.bbox_stationary_streak) + 1
    state.bbox_motion_frame_id = current_frame_id
    state.bbox_motion_ts = float(current_ts)
    state.bbox_center_u = float(center_u)
    state.bbox_bottom_v = float(bottom_v)
    state.bbox_geometry = current_geometry
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
    motion_anchor_uv: Optional[Tuple[float, float]] = None,
    motion_basis: Optional[str] = None,
    world_point: Optional[Sequence[float]] = None,
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
        matched_world = (
            tuple(float(value) for value in world_point[:3])
            if world_point is not None and len(world_point) >= 3
            else state.last_good_world
        )
        if matched_world is None or len(matched_world) < 3:
            raise ValueError
        world_x, world_y, world_z = (
            float(matched_world[0]),
            float(matched_world[1]),
            float(matched_world[2]),
        )
        values = (
            u,
            v,
            left,
            top,
            width,
            height,
            float(now_ts),
            world_x,
            world_y,
            world_z,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError
        if width <= 1.0 or height <= 1.0 or float(now_ts) < 0.0:
            raise ValueError
        relative_u = (u - left) / width
        relative_v = (v - top) / height
        if not (-0.25 <= relative_u <= 1.25 and -0.25 <= relative_v <= 1.25):
            raise ValueError
    except Exception:
        return False

    try:
        motion_u = float(motion_anchor_uv[0]) if motion_anchor_uv is not None else math.nan
        motion_v = float(motion_anchor_uv[1]) if motion_anchor_uv is not None else math.nan
        parsed_motion_basis = str(motion_basis or "").strip()
    except Exception:
        motion_u = motion_v = math.nan
        parsed_motion_basis = ""
    motion_supplied = bool(motion_anchor_uv is not None or parsed_motion_basis)
    motion_valid = bool(
        math.isfinite(motion_u)
        and math.isfinite(motion_v)
        and parsed_motion_basis
        and left - 0.15 * width <= motion_u <= left + 1.15 * width
        and top - 0.15 * height <= motion_v <= top + 1.15 * height
        and v >= motion_v + max(4.0, 0.08 * height)
    )
    if motion_supplied and not motion_valid:
        return False

    # Commit the image point, matched world point, bbox, optional anatomy,
    # timestamp, and lifecycle as one generic predictor bundle. A malformed
    # new observation returns above without mixing it with the previous
    # bundle.
    state.last_accepted_image_foot = (u, v)
    state.last_accepted_image_world = (world_x, world_y, world_z)
    state.last_accepted_bbox_geometry = (left, top, width, height)
    if motion_valid:
        state.last_accepted_motion_anchor = (motion_u, motion_v)
        state.last_accepted_motion_basis = parsed_motion_basis
    else:
        state.last_accepted_motion_anchor = None
        state.last_accepted_motion_basis = None
    state.last_accepted_image_ts = float(now_ts)
    state.last_accepted_lifecycle_generation = (
        int(lifecycle_generation) if lifecycle_generation is not None else None
    )
    # Preserve a second, immutable pose-compatible origin. Bbox-only accepted
    # observations deliberately leave it untouched; otherwise a run of exact
    # bbox floor points would erase the torso reference just before lower-body
    # occlusion needs it. This bundle is image-process evidence only and does
    # not mutate last_good_world or any metric filter state.
    if motion_valid and parsed_motion_basis.startswith("pose:"):
        state.last_accepted_pose_projective_origin = (
            AcceptedPoseProjectiveOrigin(
                image_foot=(u, v),
                world=(world_x, world_y, world_z),
                bbox_geometry=(left, top, width, height),
                motion_anchor=(motion_u, motion_v),
                motion_basis=parsed_motion_basis,
                ts=float(now_ts),
                lifecycle_generation=(
                    int(lifecycle_generation)
                    if lifecycle_generation is not None
                    else None
                ),
            )
        )
    return True


def transport_accepted_image_foot_from_pose(
    state: PersonGroundState,
    *,
    motion_anchor_uv: Optional[Tuple[float, float]],
    motion_basis: Optional[str],
    bbox: Optional[Sequence[float]],
    now_ts: float,
    lifecycle_generation: Optional[int] = None,
    config: Optional[HumanGroundConfig] = None,
    rejection_diagnostics: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[float, float, float, float, float, str]]:
    """Transport one accepted foot from an exact current anatomy reference.

    The origin is always the last physically accepted foot, bbox, and torso
    point; predicted rows never update it. This permits a longer bridge across
    lower-body occlusion than bbox-only transport without integrating pixel
    drift. The caller still must floor-project the returned pixel and apply
    calibrated range and world-space physical admission.
    """

    cfg = config or HumanGroundConfig()
    try:
        pose_origin = state.last_accepted_pose_projective_origin
        if pose_origin is None:
            return None
        prior_foot = pose_origin.image_foot
        prior_bbox = pose_origin.bbox_geometry
        prior_anchor = pose_origin.motion_anchor
        prior_basis = str(pose_origin.motion_basis or "").strip()
        current_basis = str(motion_basis or "").strip()
        accepted_ts = float(pose_origin.ts)
        current_ts = float(now_ts)
        if (
            prior_foot is None
            or prior_bbox is None
            or prior_anchor is None
            or motion_anchor_uv is None
            or not prior_basis
            or current_basis != prior_basis
            or accepted_ts < 0.0
            or not math.isfinite(current_ts)
        ):
            return None
        if (
            lifecycle_generation is not None
            and pose_origin.lifecycle_generation is not None
            and int(lifecycle_generation)
            != int(pose_origin.lifecycle_generation)
        ):
            return None
        age_s = current_ts - accepted_ts
        if (
            age_s < 0.0
            or age_s > float(cfg.projective_pose_motion_ttl_s)
            or bbox is None
            or len(bbox) < 4
        ):
            return None

        current_u, current_v = (
            float(motion_anchor_uv[0]),
            float(motion_anchor_uv[1]),
        )
        prior_u, prior_v = (float(prior_anchor[0]), float(prior_anchor[1]))
        foot_u, foot_v = (float(prior_foot[0]), float(prior_foot[1]))
        left, top, width, height = (float(value) for value in bbox[:4])
        prior_left, prior_top, prior_width, prior_height = (
            float(value) for value in prior_bbox
        )
        values = (
            current_u,
            current_v,
            prior_u,
            prior_v,
            foot_u,
            foot_v,
            left,
            top,
            width,
            height,
            prior_left,
            prior_top,
            prior_width,
            prior_height,
        )
        if not all(math.isfinite(value) for value in values):
            return None
        if min(width, height, prior_width, prior_height) <= 1.0:
            return None

        def _inside_anchor(
            anchor_u: float,
            anchor_v: float,
            box_left: float,
            box_top: float,
            box_width: float,
            box_height: float,
        ) -> bool:
            return bool(
                box_left - 0.15 * box_width
                <= anchor_u
                <= box_left + 1.15 * box_width
                and box_top - 0.15 * box_height
                <= anchor_v
                <= box_top + 1.15 * box_height
            )

        if not _inside_anchor(current_u, current_v, left, top, width, height):
            return None
        if not _inside_anchor(
            prior_u,
            prior_v,
            prior_left,
            prior_top,
            prior_width,
            prior_height,
        ):
            return None

        def _inside_foot(
            image_u: float,
            image_v: float,
            box_left: float,
            box_top: float,
            box_width: float,
            box_height: float,
        ) -> bool:
            relative_u = (image_u - box_left) / box_width
            relative_v = (image_v - box_top) / box_height
            return bool(
                -0.25 <= relative_u <= 1.25
                and -0.25 <= relative_v <= 1.25
            )

        if not _inside_foot(
            foot_u,
            foot_v,
            prior_left,
            prior_top,
            prior_width,
            prior_height,
        ):
            return None
        # A stored ground contact must remain anatomically below the torso
        # reference. This rejects mislabeled torso pixels before any ray is
        # cast through them as a floor observation.
        if foot_v < prior_v + max(4.0, 0.08 * prior_height):
            return None

        prior_center_u = prior_left + 0.5 * prior_width
        prior_center_v = prior_top + 0.5 * prior_height
        current_center_u = left + 0.5 * width
        current_center_v = top + 0.5 * height
        center_du = current_center_u - prior_center_u
        center_dv = current_center_v - prior_center_v
        # A foot is attached to the silhouette's ground edge, not its centre.
        # During sit-to-stand the box top and torso rise together while the
        # bottom remains planted. Comparing torso motion with bbox centre in
        # that case mistakes articulation for a translating ground contact.
        prior_ground_u = prior_center_u
        prior_ground_v = prior_top + prior_height
        current_ground_u = current_center_u
        current_ground_v = top + height
        bbox_du = current_ground_u - prior_ground_u
        bbox_dv = current_ground_v - prior_ground_v
        pose_du = current_u - prior_u
        pose_dv = current_v - prior_v
        pose_displacement_px = math.hypot(pose_du, pose_dv)
        bbox_displacement_px = math.hypot(bbox_du, bbox_dv)
        center_displacement_px = math.hypot(center_du, center_dv)
        speed_dt = max(age_s, 1.0 / 30.0)
        if (
            pose_displacement_px / speed_dt
            > float(cfg.projective_max_bbox_speed_px_s)
            or bbox_displacement_px / speed_dt
            > float(cfg.projective_max_bbox_speed_px_s)
        ):
            return None
        material_px = float(cfg.static_px_threshold)
        displacement_residual_limit_px = max(
            12.0,
            0.35 * max(width, prior_width),
            0.25 * max(height, prior_height),
        )

        def _displacements_are_coherent(
            reference_du: float,
            reference_dv: float,
            reference_displacement_px: float,
        ) -> bool:
            if max(pose_displacement_px, reference_displacement_px) <= material_px:
                return True
            displacement_dot = pose_du * reference_du + pose_dv * reference_dv
            if displacement_dot <= 0.0:
                return False
            if (
                pose_displacement_px > material_px
                and reference_displacement_px > material_px
            ):
                direction_cosine = displacement_dot / (
                    pose_displacement_px * reference_displacement_px
                )
                if direction_cosine < 0.50:
                    return False
            displacement_residual_px = math.hypot(
                pose_du - reference_du,
                pose_dv - reference_dv,
            )
            return displacement_residual_px <= displacement_residual_limit_px

        bottom_coherent = _displacements_are_coherent(
            bbox_du,
            bbox_dv,
            bbox_displacement_px,
        )
        if not bottom_coherent:
            # If torso motion agrees with the box centre but not its ground
            # edge, the silhouette is articulating or changing truncation.
            # Mark this as a hard rejection so the caller cannot immediately
            # bypass it with bbox-affine transport of the same changing box.
            center_coherent = _displacements_are_coherent(
                center_du,
                center_dv,
                center_displacement_px,
            )
            if (
                rejection_diagnostics is not None
                and center_coherent
                and pose_displacement_px > material_px
            ):
                rejection_diagnostics["hard_rejection"] = (
                    "pose_bbox_ground_articulation_conflict"
                )
                rejection_diagnostics["pose_step_px"] = float(
                    pose_displacement_px
                )
                rejection_diagnostics["bbox_ground_step_px"] = float(
                    bbox_displacement_px
                )
            return None

        # Transport remains translation-only: bbox perspective scale is never
        # applied. Genuine whole-body translation moves the torso and ground
        # edge coherently; articulation or truncation changes fail above and
        # fall back to bounded process continuity rather than inventing a foot.
        transported_u = current_u + (foot_u - prior_u)
        transported_v = current_v + (foot_v - prior_v)
        if not all(math.isfinite(value) for value in (transported_u, transported_v)):
            return None
        if not _inside_foot(
            transported_u,
            transported_v,
            left,
            top,
            width,
            height,
        ):
            return None
        if transported_v < current_v + max(4.0, 0.08 * height):
            return None
        return (
            float(transported_u),
            float(transported_v),
            float(age_s),
            float(bbox_displacement_px),
            1.0,
            "pose_torso_translation",
        )
    except Exception:
        return None


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
    allow_segment_break: bool = True,
    prior_output: Optional[
        Tuple[float, float, Optional[int], float, int]
    ] = None,
    ignore_uncommitted_state_output: bool = False,
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
    current_segment = int(state.trail_segment_id)
    if bool(ignore_uncommitted_state_output):
        previous_x = None
        previous_z = None
        previous_pts = None
        previous_filter_ts = -1.0
        previous_segment = current_segment
    else:
        previous_x = state.last_output_world_x
        previous_z = state.last_output_world_z
        previous_pts = state.last_output_media_pts_ns
        previous_filter_ts = float(state.last_output_filter_ts)
        previous_segment = int(state.last_output_trail_segment_id)
    output_speed_rejected = bool(
        state.measurement_accepted is False
        and str(state.measurement_rejection_reason or "")
        == "physical_output_speed_exceeded"
    )
    if prior_output is not None:
        try:
            (
                prior_x,
                prior_z,
                prior_pts,
                prior_filter_ts,
                prior_segment,
            ) = prior_output
            parsed_prior_x = float(prior_x)
            parsed_prior_z = float(prior_z)
            parsed_prior_filter_ts = float(prior_filter_ts)
            parsed_prior_segment = int(prior_segment)
            parsed_prior_pts = int(prior_pts) if prior_pts is not None else None
        except (TypeError, ValueError, OverflowError):
            pass
        else:
            if all(
                math.isfinite(value)
                for value in (
                    parsed_prior_x,
                    parsed_prior_z,
                    parsed_prior_filter_ts,
                )
            ):
                previous_x = parsed_prior_x
                previous_z = parsed_prior_z
                previous_pts = parsed_prior_pts
                previous_filter_ts = parsed_prior_filter_ts
                previous_segment = parsed_prior_segment

    if previous_x is None or previous_z is None:
        if output_speed_rejected:
            # There is no queue-visible coordinate that can prove a hold.  The
            # caller must fail this row closed instead of restamping the
            # rejected process point as a first canonical output.
            state.trail_append_allowed = False
            return np.array([x, floor, z], dtype=np.float64), False
        accepted = True
    elif output_speed_rejected:
        # The metric filter already rejected this row's proposed posterior for
        # exceeding the human output-speed bound.  Even when a later CV step is
        # numerically close enough to pass this final gate, it may not turn the
        # rejected row into a fresh process publication.  Emit the exact prior
        # queue-visible coordinate; the producer attaches a gain-zero
        # bounded_output_hold proof to that restamp.
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
        accepted = False
    elif bool(allow_segment_break) and (
        bool(state.trail_break_required) or current_segment != previous_segment
    ):
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
        # Quarantine the divergent process result and make the prior canonical
        # output the next filter origin.  Do not manufacture an intermediate
        # point or update the accepted metric anchor.
        state.world_x = float(previous_x)
        state.world_z = float(previous_z)
        state.vel_world_x = 0.0
        state.vel_world_z = 0.0
        state.filtered_ts = filter_ts
        state.measurement_accepted = False
        state.measurement_rejection_reason = (
            "physical_output_speed_exceeded"
            if output_speed_rejected
            else "physical_output_continuity_exceeded"
        )
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
        # Restore the complete published watermark from the immutable
        # frame-entry reference: current-frame metric evaluation may have
        # overwritten its coordinates and segment before this alternate source
        # reached final admission.
        state.last_output_world_x = float(previous_x)
        state.last_output_world_z = float(previous_z)
        state.last_output_media_pts_ns = current_pts
        state.last_output_filter_ts = filter_ts
        state.last_output_trail_segment_id = int(previous_segment)
        return (
            np.array([float(previous_x), floor, float(previous_z)], dtype=np.float64),
            False,
        )

    state.last_output_world_x = x
    state.last_output_world_z = z
    state.last_output_media_pts_ns = current_pts
    state.last_output_filter_ts = filter_ts
    state.last_output_trail_segment_id = (
        current_segment if bool(allow_segment_break) else previous_segment
    )
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
    verified_reacquire_support: bool,
) -> np.ndarray:
    """Quarantine one unreachable update or explicitly reanchor a track.

    A same-lifecycle relocation is never smuggled through as a high-speed
    continuous step. Reanchoring requires a bounded run in one contact family
    and always starts a new trail segment. A full consensus of exact observed
    ankle-floor samples is sufficient for an established lifecycle. Moderate
    upright body-plane rows may accumulate same-basis trajectory evidence, but
    only a current verified five-plane row may finalize that reanchor. Bbox and
    other inferred bases still require independent exact-frame image motion.
    """

    mx = float(measurement_x)
    mz = float(measurement_z)
    admitted_basis = _reacquire_basis_family(contact_basis)
    pose_floor_candidate = admitted_basis == "pose_floor"
    upright_body_candidate = admitted_basis == "pose:upright_body_planes"
    verified_reacquire_support = bool(
        verified_reacquire_support or pose_floor_candidate
    )
    if (
        state.post_occlusion_reacquire_support_required
        and not verified_reacquire_support
        and not upright_body_candidate
    ):
        _clear_reacquire_candidate(state)
        return _reject_with_cv_time_update(
            state,
            pred_x=float(pred_x),
            pred_z=float(pred_z),
            now_ts=float(now_ts),
            floor_y=float(floor_y),
            reason="post_occlusion_reacquire_requires_verified_support",
            config=config,
        )
    if not admitted_basis or (
        not bool(image_motion_supported)
        and not pose_floor_candidate
        and not verified_reacquire_support
        and not upright_body_candidate
    ):
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
    state.reacquire_count = min(
        max(1, int(config.reacquire_samples)),
        int(state.reacquire_count) + 1 if candidate_consistent else 1,
    )
    state.reacquire_candidate_x = mx
    state.reacquire_candidate_z = mz
    state.reacquire_candidate_ts = float(now_ts)
    state.reacquire_candidate_basis = admitted_basis
    state.reacquire_candidate_exact_basis = str(
        contact_basis or admitted_basis
    )
    state.reacquire_unavailable_rows = 0

    reacquire_consensus_ready = bool(
        int(state.reacquire_count) >= int(config.reacquire_samples)
    )
    reacquire_support_ready = bool(
        verified_reacquire_support
        if upright_body_candidate
        else (
            bool(image_motion_supported)
            or pose_floor_candidate
            or verified_reacquire_support
        )
    )
    if reacquire_consensus_ready and reacquire_support_ready:
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
        state.post_occlusion_reacquire_support_required = False
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
    verified_reacquire_support: bool = False,
    verified_first_output_pose_floor_seed: bool = False,
    position_gain_override: Optional[float] = None,
    allow_physical_output_overrun: bool = False,
    media_pts_ns: Optional[int] = None,
    display_output: Optional[
        Tuple[float, float, Optional[int], float, int]
    ] = None,
    transition_proof: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Constant-velocity XZ filter with explicit physical admission.

    Impossible observations are quarantined instead of clipped into plausible-
    looking motion. An existing tracker lifecycle may relocate only after a
    bounded run of mutually consistent observations on one contact basis.
    Exact observed ankle-floor consensus may stand on its own; every other
    basis also needs independent exact-frame image motion. Either path starts
    a new trail segment.  An optional display_output watermark applies the
    same line-segment/circle gain reduction used by projective recovery, while
    keeping the kinematic prior and emitted posterior in one transition.
    """
    if transition_proof is not None:
        transition_proof.clear()
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

    if bool(verified_first_output_pose_floor_seed):
        # This narrow path is proof-bearing, not a generic force accept. The
        # caller has independently retained two compatible exact ankle-pair
        # observations while no queue-visible output existed. Discard only
        # mutable, never-published kinematics and seed the current sample as
        # the first canonical candidate; final output/service admission still
        # applies normally.
        if (
            _reacquire_basis_family(contact_basis) != "pose_floor"
            or str(contact_basis or "") != "pose_ankle_floor"
        ):
            state.measurement_accepted = False
            state.measurement_rejection_reason = (
                "first_output_pose_floor_proof_invalid"
            )
            state.trail_append_allowed = False
            return np.array([mx, floor_y, mz], dtype=np.float64)
        state.world_x = mx
        state.world_z = mz
        state.vel_world_x = 0.0
        state.vel_world_z = 0.0
        state.filtered_ts = now_ts
        state.last_good_world = None
        state.last_good_ts = 0.0
        state.locked_world = None
        state.idle_since_ts = 0.0
        state.exit_motion_frames = 0
        state.reacquired = False
        state.trail_break_required = False
        state.post_occlusion_reacquire_support_required = False
        _clear_reacquire_candidate(state)
        _clear_rejection_anchor(state)
        if transition_proof is not None:
            transition_proof.update(
                {
                    "version": 1,
                    "kind": "verified_first_output_pose_floor_seed",
                    "position_base": [mx, floor_y, mz],
                    "position_gain": 1.0,
                }
            )
        return np.array([mx, floor_y, mz], dtype=np.float64)

    if state.world_x is None or state.world_z is None or float(state.filtered_ts) < 0.0:
        # A weak resolver result is still a current metric hypothesis, but a
        # single low-support frame must not seed canonical world state. Use
        # the same bounded, contact-basis-specific consensus required for an
        # established track to reacquire after a divergent observation. This
        # gives persistent far/partially occluded people a generic bootstrap
        # path while keeping one-frame pose/photo/noise hits fail-closed.
        if str(quality or "").strip().lower() == "weak" and not force_accept:
            admitted_basis = _reacquire_basis_family(contact_basis)
            pending_pose_floor_age_s = (
                float(now_ts) - float(state.reacquire_candidate_ts)
            )
            preserve_pending_pose_floor = bool(
                str(state.reacquire_candidate_basis or "") == "pose_floor"
                and int(state.reacquire_count) > 0
                and admitted_basis != "pose_floor"
                and pending_pose_floor_age_s > 0.0
                and pending_pose_floor_age_s <= float(config.reacquire_max_gap_s)
                and int(state.reacquire_unavailable_rows) == 0
            )
            if preserve_pending_pose_floor:
                # One lower-authority bbox/non-floor row must not replace a
                # stronger observed-ankle cold consensus.  The run is still
                # time-bounded and a second intervening row clears it.
                state.reacquire_unavailable_rows = 1
                state.measurement_accepted = False
                state.measurement_rejection_reason = (
                    "weak_measurement_bootstrap_pending"
                )
                state.measurement_innovation_m = math.nan
                state.measurement_allowed_m = math.nan
                state.trail_append_allowed = False
                return np.array([mx, floor_y, mz], dtype=np.float64)
            if not admitted_basis:
                state.measurement_accepted = False
                state.measurement_rejection_reason = "weak_measurement_contact_basis_missing"
                state.measurement_innovation_m = math.nan
                state.measurement_allowed_m = math.nan
                state.trail_append_allowed = False
                _clear_reacquire_candidate(state)
                return np.array([mx, floor_y, mz], dtype=np.float64)

            previous_exact_basis = str(
                state.reacquire_candidate_exact_basis or ""
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
                state.measurement_innovation_m = float(candidate_step)
                state.measurement_allowed_m = float(candidate_allowed)
            state.reacquire_count = (
                int(state.reacquire_count) + 1 if candidate_consistent else 1
            )
            state.reacquire_candidate_x = mx
            state.reacquire_candidate_z = mz
            state.reacquire_candidate_ts = float(now_ts)
            state.reacquire_candidate_basis = admitted_basis
            state.reacquire_candidate_exact_basis = str(
                contact_basis or admitted_basis
            )
            state.reacquire_unavailable_rows = 0
            # A current observed-ankle floor ray may seed after two mutually
            # consistent samples only when three exact published observations
            # of the independent torso point and detector silhouette already
            # prove coherent motion. Other weak sources retain the full
            # three-sample default unless the separate exact four-plane body
            # pair below proves a cold start; a static photo or one-frame pose
            # hit cannot take either path.
            torso_motion_corroborated = bool(
                admitted_basis == "pose_floor"
                and image_motion_supported
                and str(state.image_motion_contact_basis or "")
                == "pose:torso_motion"
                and int(state.image_motion_streak)
                >= int(config.static_exit_frames)
                and bool(state.image_motion_bootstrap_supported)
            )
            ankle_pair_consensus = bool(
                admitted_basis == "pose_floor"
                and str(contact_basis or "") == "pose_ankle_floor"
                and previous_exact_basis == "pose_ankle_floor"
                and candidate_consistent
            )
            cold_upright_body_candidate = bool(
                admitted_basis == "pose:upright_body_planes"
            )
            upright_body_four_plane_consensus = bool(
                cold_upright_body_candidate
                and str(contact_basis or "")
                == "pose:upright_body_planes:four_plane"
                and previous_exact_basis
                == "pose:upright_body_planes:four_plane"
                and candidate_consistent
            )
            if upright_body_four_plane_consensus:
                # This is a cold-start-only proof. Each row is an
                # exact-current four-plane solve, so two compatible samples
                # are sufficient even though neither row can authorize an
                # established relocation on its own.
                required_samples = 2
            elif torso_motion_corroborated or ankle_pair_consensus:
                required_samples = min(2, int(config.reacquire_samples))
            else:
                required_samples = int(config.reacquire_samples)
            if (
                int(state.reacquire_count) < int(required_samples)
                or (
                    cold_upright_body_candidate
                    and not (
                        bool(verified_reacquire_support)
                        or upright_body_four_plane_consensus
                    )
                )
            ):
                state.measurement_accepted = False
                state.measurement_rejection_reason = "weak_measurement_bootstrap_pending"
                state.trail_append_allowed = False
                return np.array([mx, floor_y, mz], dtype=np.float64)

        state.world_x = mx
        state.world_z = mz
        state.vel_world_x = 0.0
        state.vel_world_z = 0.0
        state.filtered_ts = now_ts
        if verified_reacquire_support or _reacquire_basis_family(contact_basis) == "pose_floor":
            state.post_occlusion_reacquire_support_required = False
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
    if (
        not force_accept
        and allowed > 0.0
        and innov_dist > allowed + 1e-9
        and innov_dist > 1e-9
    ):
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
            verified_reacquire_support=bool(verified_reacquire_support),
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
    if position_gain_override is not None:
        try:
            requested_gain = float(position_gain_override)
        except (TypeError, ValueError, OverflowError):
            requested_gain = math.nan
        if math.isfinite(requested_gain):
            # A display-bound recovery may only reduce the normal posterior
            # gain.  Keeping this in the filter transition also updates the
            # velocity estimate from the emitted point instead of leaving a
            # larger-gain process state behind for the next frame.
            alpha = min(alpha, max(0.0, min(1.0, requested_gain)))

    display_constraint = _display_output_constraint(
        display_output,
        now_ts=float(now_ts),
        media_pts_ns=media_pts_ns,
        config=config,
    )
    display_gain_reduced = False
    display_unconstrained_gain = float(alpha) if display_constraint is not None else None
    if display_constraint is not None and not force_accept:
        display_x, display_z, _display_pts, _display_dt, display_limit = (
            display_constraint
        )
        unconstrained_x = pred_x + float(alpha) * innov_x
        unconstrained_z = pred_z + float(alpha) * innov_z
        display_step = math.hypot(
            unconstrained_x - float(display_x),
            unconstrained_z - float(display_z),
        )
        if display_step > float(display_limit) + 1e-9:
            display_gain = _max_line_gain_inside_circle(
                base_x=float(pred_x),
                base_z=float(pred_z),
                target_x=float(mx),
                target_z=float(mz),
                center_x=float(display_x),
                center_z=float(display_z),
                radius=float(display_limit),
                max_gain=float(alpha),
            )
            if display_gain is None:
                return _quarantine_or_reacquire_measurement(
                    state,
                    measurement_x=mx,
                    measurement_z=mz,
                    pred_x=float(pred_x),
                    pred_z=float(pred_z),
                    now_ts=float(now_ts),
                    floor_y=float(floor_y),
                    reason="physical_display_continuity_exceeded",
                    config=config,
                    contact_basis=contact_basis,
                    image_motion_supported=bool(image_motion_supported),
                    verified_reacquire_support=bool(verified_reacquire_support),
                )
            display_gain_reduced = bool(display_gain < float(alpha) - 1e-12)
            alpha = float(display_gain)
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
        if verified_reacquire_support or _reacquire_basis_family(contact_basis) == "pose_floor":
            state.post_occlusion_reacquire_support_required = False
        held = np.array(
            [float(state.world_x), floor_y, float(state.world_z)],
            dtype=np.float64,
        )
        if transition_proof is not None:
            transition_proof.update(
                {
                    "version": 1,
                    "kind": "deadzone_hold",
                    "gate_dt_s": float(gate_dt),
                    "position_base": [
                        float(held[0]),
                        float(floor_y),
                        float(held[2]),
                    ],
                    "position_gain": 0.0,
                }
            )
        return held

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
        and not allow_physical_output_overrun
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
            verified_reacquire_support=bool(verified_reacquire_support),
        )

    if transition_proof is not None:
        transition_proof.update(
            {
                "version": 1,
                "kind": "innovation_update",
                "gate_dt_s": float(gate_dt),
                "position_base": [
                    float(pred_x),
                    float(floor_y),
                    float(pred_z),
                ],
                "position_gain": float(alpha),
            }
        )
        if display_constraint is not None:
            display_x, display_z, display_pts, display_dt, _display_limit = (
                display_constraint
            )
            transition_proof.update(
                {
                    "visible_origin_kind": "latest_displayed_world_output",
                    "visible_origin_world": [
                        float(display_x),
                        float(floor_y),
                        float(display_z),
                    ],
                    "visible_origin_media_pts_ns": (
                        int(display_pts) if display_pts is not None else None
                    ),
                    "visible_gate_dt_s": float(
                        min(
                            float(display_dt),
                            float(config.reset_after_s)
                            if float(config.reset_after_s) > 0.0
                            else float(display_dt),
                        )
                    ),
                    "visible_max_step_m": float(_display_limit),
                    "visible_position_gain_reduced": bool(display_gain_reduced),
                    "visible_unconstrained_position_gain": float(
                        display_unconstrained_gain
                        if display_unconstrained_gain is not None
                        else alpha
                    ),
                }
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
    if verified_reacquire_support or _reacquire_basis_family(contact_basis) == "pose_floor":
        state.post_occlusion_reacquire_support_required = False
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
    "AcceptedPoseProjectiveOrigin",
    "HumanGroundConfig",
    "FirstOutputAnkleProof",
    "InferredGroundContinuityAnchor",
    "LowerBodyOcclusionAssessment",
    "PoseAnchorCandidate",
    "PersonGroundState",
    "PersonGroundStateStore",
    "apply_source_hysteresis",
    "admit_human_ground_output",
    "align_inferred_ground_observation",
    "advance_human_cv_prediction",
    "assess_lower_body_occlusion",
    "bind_world_frame",
    "begin_source_admission",
    "classify_posture",
    "clear_first_output_ankle_proof",
    "clear_inferred_ground_continuity_anchor",
    "commit_image_path_point",
    "commit_path_point",
    "complete_source_admission",
    "estimate_ankle_from_leg",
    "integrate_projective_ground_observation",
    "legs_are_bent",
    "mark_image_motion_observation_unavailable",
    "record_accepted_image_geometry",
    "observe_bbox_stationarity",
    "observe_first_output_ankle_proof",
    "pose_point",
    "rdp_simplify",
    "resolve_pose_floor_anchor",
    "source_score",
    "transport_accepted_image_foot",
    "transport_accepted_image_foot_from_pose",
    "update_human_cv_filter",
    "update_motion_mode",
    "world_frame_binding_from_calibration",
    "world_frame_matches_calibration",
]
