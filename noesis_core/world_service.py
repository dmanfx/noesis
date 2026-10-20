from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from threading import RLock, get_ident
from typing import Any, Callable, Mapping

from noesis_core.analytics_zones import is_exact_zone_label
from noesis_core.contracts.base import ArtifactFingerprint, Matrix3, ProducerRef, Vector3
from noesis_core.contracts.identity import IdentityKind, SubjectRef, TrackletRef
from noesis_core.contracts.observation import (
    ObservationEnvelope,
    PersonObservation,
    WorldObservationDiagnostics,
    WorldPositionObservation,
    ZoneSource,
)
from noesis_core.contracts.world import (
    CanonicalWorldIdentityError, EntityLifecycle, WorldEntity, WorldEvent, WorldSnapshot,
    WorldIdentityCollision, validate_world_identity_uniqueness,
    WorldAttributionClaim, WorldAttributionQuarantine, attribution_conflict_sets,
)
from noesis_core.contracts.world_measurement import WorldMeasurementCandidateDiagnostic
from noesis_core.contracts.support_surface import SupportSurfaceReference
from noesis_core.depth_contract import usable_registered_depth_m
from noesis_core.journal import (
    AsyncContractJournal,
    AsyncJournalAdmissionReceipt,
    ContractJournal,
)
from noesis_core.world import GlobalWorldFusion
from noesis_core.world.fusion import (
    select_source_local_observation, spatial_rejection_category, position_quality_sort_key,
    SPATIAL_NO_VOTE_REASONS, SPATIAL_INTEGRITY_REASONS,
    PreparedUprightFootprintSupport,
)
from noesis_core.world.support_surfaces import contains_xz
from noesis_core.tracking_continuity import parse_tracker_presence_evidence
from noesis_core.contracts.scene_prior import ScenePriorFrameBinding
from noesis.identity_association_service import (
    AssociationDTO, AssociationReservation, IdentityAssociationService,
    MeasurementCohort, ProducerAdmission,
    HouseholdClaimConflict, ServiceResult, ReservationResult, CommitResult,
    SpatialMembershipDTO, SpatialMeasurementReference, SpatialMemberReservation, SpatialMemberResult,
)
from reid.mv3dt_person_association import AcquisitionTime, MemberKey, PhysicalPersonKey, ObservationStamp, SourceObservation, SpatialClockContext

_LOGGER = logging.getLogger(__name__)


class WorldPublicationRefused(RuntimeError):
    """Public atomic refusal; retain owner-issued conflict object verbatim."""

    def __init__(self, reason: str, *, household_conflict: HouseholdClaimConflict | None = None,
                 identity_conflicts: tuple[WorldIdentityCollision, ...] = ()) -> None:
        self.reason = reason
        self.household_conflict = household_conflict
        self.identity_conflicts = identity_conflicts
        super().__init__(f"World publication denied: {reason}")

    @classmethod
    def from_association_result(cls, result: ServiceResult | ReservationResult | CommitResult | SpatialMemberResult):
        if type(result) not in {ServiceResult, ReservationResult, CommitResult, SpatialMemberResult} or result.disposition not in {
            "conflict", "capacity", "denied", "rejected", "unavailable", "unresolved",
            "association_unresolved",
        }:
            raise ValueError("actual refused public association result required")
        return cls(result.reason, household_conflict=result.household_conflict)


class WorldPreparedHoldExpired(WorldPublicationRefused):
    """An exact uncommitted hold crossed its ORIGINAL wire deadline."""

    def __init__(self, prepared: PreparedWorldPublication, memberships, context, *,
                 reason: str = "split_custody_original_changed") -> None:
        if reason not in {"split_custody_original_changed","committed_spatial_hold_custody_changed",
                          "ordinary_join_original_changed"}:
            raise ValueError("exact finite-hold expiry reason required")
        super().__init__(reason)
        self.prepared = prepared
        self.memberships = memberships
        self.context = context


def _guard_world_snapshot(snapshot: WorldSnapshot) -> None:
    try:
        validate_world_identity_uniqueness(snapshot.entities)
    except CanonicalWorldIdentityError as error:
        raise WorldPublicationRefused(error.reason, identity_conflicts=error.collisions) from error


def _physical_snapshot(fusion: GlobalWorldFusion, now_us: int) -> WorldSnapshot:
    try:
        return fusion.snapshot(published_at_us=now_us)
    except CanonicalWorldIdentityError as error:
        raise WorldPublicationRefused(error.reason, identity_conflicts=error.collisions) from error


_WORLD_DIAGNOSTIC_TRACK_FIELDS = frozenset(
    {
        "world_estimator_evaluated",
        "world_floor_candidate",
        "world_floor_range_m",
        "world_floor_range_limit_m",
        "world_floor_incidence_sin",
        "world_floor_admitted",
        "world_floor_rejection_reason",
        "world_depth_candidate",
        "world_prefilter_measurement",
        "world_filter_prediction",
        "world_measurement_accepted",
        "world_rejection_reason",
        "world_innovation_m",
        "world_innovation_limit_m",
        "world_fusion_policy_id",
        "world_floor_weight_scale",
        "world_depth_weight_scale",
        "world_floor_weight_effective",
        "world_depth_weight_effective",
        "depth_status",
        "depth_anchor_m",
        "depth_registered_m",
        "depth_used_m",
        "depth_registration_status",
        "depth_registration_id",
        "world_resolver",
    }
)


@dataclass(frozen=True)
class WorldArtifacts:
    calibration: ArtifactFingerprint
    model: ArtifactFingerprint
    config: ArtifactFingerprint
    # Private producer/consumer authority binding. ``calibration`` fingerprints
    # the complete resolved snapshot for the public observation contract, while
    # this is the exact raw camera-calibration file digest stamped on the world
    # coordinate by the DS9 estimator. They are intentionally distinct: a
    # queued coordinate must never be relabeled with a newer provider snapshot.
    camera_calibration_sha256: str | None = None


@dataclass(frozen=True)
class WorldPublication:
    observations: tuple[ObservationEnvelope, ...]
    snapshot: WorldSnapshot
    events: tuple[WorldEvent, ...]


@dataclass(frozen=True)
class PreparedWorldPublication:
    """Opaque owner-bound receipt for one non-authoritative world candidate."""

    base_revision: int
    publication: WorldPublication
    _service_token: object = field(repr=False, compare=False)
    _candidate_id: int = field(repr=False, compare=False)
    # The world candidate is model-dumped once while preparing the durable
    # journal cohort.  Keep that exact JSON-native tuple on the owner-bound
    # receipt so the canonical tracking publisher can reuse it instead of
    # serializing the immutable models a second time before admission.
    _serialized_payloads: tuple[Mapping[str, Any], ...] = field(
        default_factory=tuple,
        repr=False,
        compare=False,
    )

    @property
    def serialized_payloads(self) -> tuple[Mapping[str, Any], ...]:
        """Return the prepared JSON payloads in journal cohort order."""

        return self._serialized_payloads


class _SpatialDecisionObserver:
    """Private scalar-only final observer, never an authority input."""

    def __init__(self):
        self.records = []
        self.omitted = 0
        self.errors = 0
        self.targets = ()
        try:
            value=os.environ.get('NOESIS_XC3D_BODY_DIAGNOSTIC_TARGETS_S','')
            targets=tuple(float(v.strip()) for v in value.split(',')) if value else ()
            if len(value)<=128 and len(targets)<=4 and all(math.isfinite(v) and 0<=v<=86400 for v in targets):
                self.targets=targets
        except (ValueError,TypeError):pass  # Optional diagnostics never affect admission.

    def record(self, row):
        if (type(row) is not dict or len(row) > 32 or any(
                type(k) is not str or len(k) > 64 or
                not (v is None or type(v) in (bool,int) or type(v) is float and math.isfinite(v)
                     or type(v) is str and len(v) <= (4096 if k=='target_members' else
                         2048 if k in {'prior_owner_guard','target_losses','target_barriers'} else 240)) for k,v in row.items())):
            self.omitted += 1
            return
        row = dict(row)
        if row.get('transaction')=='committed':
            for old in self.records:
                if (old.get('kind'),old.get('candidate'),old.get('original_id')) == (
                        row.get('kind'),row.get('candidate'),row.get('original_id')) and (
                        row.get('kind')!='target_cohort_admission' or
                        (old.get('source'),old.get('target_index'))==(row.get('source'),row.get('target_index'))):
                    old['transaction']='committed'
                    return  # Commit the retained observation, not a replacement frame.
        pin = row.get('kind') == 'unexpired_held_loss'
        if pin and any(r['kind'] == 'unexpired_held_loss' and (
                r.get('transaction')=='committed' or r.get('transaction')=='prepared'
                and r.get('candidate')==row.get('candidate')) for r in self.records):
            return
        key = (row['kind'],row.get('source'),row.get('target_index') if row['kind']=='target_cohort_admission' else row.get('method'))
        for index, old in enumerate(self.records):
            if (old['kind'],old.get('source'),old.get('target_index') if old['kind']=='target_cohort_admission' else old.get('method')) == key:
                if row['kind']=='target_cohort_admission' and row['distance_s']>=old['distance_s']:return
                lineage = ('source','raw','generation','birth','epoch')
                denied = (old.get('raw_entry_eligible') is False or old.get('owner_eligible') is False
                    or old.get('selector_reason') not in {None,'source_local_alternative','unobserved_stage'}
                    or old.get('reservation_outcome')=='reserved' and old.get('final_source_accepted') is False)
                if (row['kind']=='current_original_nonvote'
                        and denied
                        and all(type(row.get(k)) is int and row[k]>=0 for k in lineage)
                        and all(type(old.get(k)) is int and old[k]==row[k] for k in lineage)
                        and old.get('transaction') in {'prepared','committed'}):
                    return  # First denial for this native lineage, not a later-frame substitute.
                self.records.pop(index);break
        if len(self.records) >= 8:
            index = next((i for i,r in enumerate(self.records) if r['kind'] not in
                {'unexpired_held_loss','target_cohort_admission'}),None)
            if index is None:
                self.omitted += 1;return
            self.records.pop(index);self.omitted += 1
        self.records.append(row)

    def outcome(self, candidate, state):
        for row in self.records:
            if row.get('candidate') == candidate:row['transaction'] = state

    def encoded(self):
        rows = [dict(r) for r in self.records]
        omitted = self.omitted
        while True:
            result = json.dumps({'contract':'noesis.world.private_decisions.v1',
                'records':rows,'omitted_records':omitted,'observer_errors':self.errors},
                allow_nan=False,sort_keys=True,separators=(',',':'))
            if len(result.encode('utf-8')) <= 8192:return result
            index=next((i for i,r in enumerate(rows) if r['kind'] not in
                {'unexpired_held_loss','target_cohort_admission'}),0)
            rows.pop(index);omitted += 1


@dataclass
class _PreparedWorldState:
    """Service-private mutable state that must never cross the boundary."""

    publication: WorldPublication
    fusion: GlobalWorldFusion
    sequence_by_source: dict[int, int]
    event_sequence: int
    previous_entities: dict[str, WorldEntity]
    canonical_track_outputs: dict[
        tuple[int, str, int, int, str, str, str, int],
        "_CanonicalTrackOutput",
    ]
    journal_payloads: tuple[Mapping[str, Any], ...]
    recorded_at_us: int
    source_id: int
    artifact_metadata: Mapping[str, Any]
    calibration_artifact_sha256: str
    camera_calibration_sha256: str | None
    camera_calibration_binding_required: bool
    # Original normalized World values and detached native lineage, not another
    # association DTO or a last-seen lookup. Normalization receipts own these.
    physical_members: dict[str, tuple[MemberKey, int, int | None, bool]] = field(default_factory=dict)
    association_candidate: bool = False
    association_reservations: tuple[AssociationReservation, ...] = ()
    association_bindings: tuple[tuple[int, ScenePriorFrameBinding, tuple[int, int]], ...] = ()
    association_artifacts: tuple[tuple[int, Mapping[str, Any], WorldArtifacts], ...] = ()
    resolved_artifacts: WorldArtifacts | None = None
    physical_stamps: dict[str, ObservationStamp] = field(default_factory=dict)
    physical_observations: tuple[ObservationEnvelope, ...] = ()
    # Only raw producer quality presence, never person/position authority.
    # Keep legacy normalization fallbacks out of the future spatial selector.
    spatial_quality: dict[str, tuple[float | None, str | None]] = field(default_factory=dict)
    spatial_upright_support: dict[str, PreparedUprightFootprintSupport] = field(default_factory=dict)
    spatial_memberships: tuple[SpatialMembershipDTO, ...] = ()
    spatial_reservations: tuple[SpatialMemberReservation, ...] = ()
    spatial_bindings: tuple[tuple[int, ScenePriorFrameBinding], ...] = ()
    spatial_context: SpatialClockContext | None = None
    spatial_input_diagnostics: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    attribution_mode: str = "strict"
    spatial_holds: tuple[tuple[PhysicalPersonKey, ObservationStamp], ...] = ()
    spatial_reindexes: tuple = ()
    spatial_split_custody: tuple = ()
    spatial_split_episode_views: tuple = ()
    spatial_observer_records: tuple = ()


@dataclass(frozen=True)
class _InferredGroundEpisodeRoot:
    """Immutable producer proof root accepted and then owned by the service."""

    raw_origin: tuple[float, float, float]
    raw_origin_media_pts_ns: int
    raw_origin_observed_at_us: int
    raw_origin_ts_s: float
    trusted_world_origin: tuple[float, float, float]
    trusted_origin_media_pts_ns: int
    trusted_origin_filter_ts_s: float
    lifecycle_generation: int
    trail_segment_id: int
    height_ref_scene: float
    world_frame: str
    world_frame_revision: str
    world_transform_sha256: str


@dataclass(frozen=True)
class _CanonicalOutputOrigin:
    """One proven earlier queue output retained for asynchronous validation."""

    position: tuple[float, float, float]
    media_pts_ns: int
    trail_segment_id: int
    committed_world_source: str
    support_surface: SupportSurfaceReference | None = None
    committed_provenance_type: str | None = None
    metric_position: tuple[float, float, float] | None = None
    metric_media_pts_ns: int | None = None
    metric_trail_segment_id: int | None = None
    projective_bridge_root_media_pts_ns: int | None = None
    kinematic_position: tuple[float, float, float] | None = None
    kinematic_media_pts_ns: int | None = None
    kinematic_trail_segment_id: int | None = None
    bbox_geometry: tuple[float, float, float, float] | None = None
    kinematic_bbox_geometry: tuple[float, float, float, float] | None = None
    inferred_ground_episode_root: _InferredGroundEpisodeRoot | None = None


@dataclass(frozen=True)
class _TrackerPresenceOrigin:
    """One exact accepted source envelope; native presence cannot renew it."""

    envelope: ObservationEnvelope
    subject: SubjectRef
    trail_segment_id: int | None
    trail_break_required: bool
    tracker_lifecycle_generation: int
    source_epoch: int


@dataclass(frozen=True)
class _CanonicalTrackOutput:
    """Last committed source-local point used by projective continuity."""

    position: tuple[float, float, float]
    media_pts_ns: int
    trail_segment_id: int
    committed_world_source: str
    support_surface: SupportSurfaceReference | None = None
    committed_provenance_type: str | None = None
    metric_position: tuple[float, float, float] | None = None
    metric_media_pts_ns: int | None = None
    metric_trail_segment_id: int | None = None
    projective_bridge_root_media_pts_ns: int | None = None
    # A gain-zero output hold is a current publication of an older uncertain
    # coordinate, not evidence that the person was physically stationary at
    # the hold's media PTS. Preserve the last motion-bearing output separately
    # so a later accurate recovery receives the real elapsed motion budget.
    kinematic_position: tuple[float, float, float] | None = None
    kinematic_media_pts_ns: int | None = None
    kinematic_trail_segment_id: int | None = None
    bbox_geometry: tuple[float, float, float, float] | None = None
    kinematic_bbox_geometry: tuple[float, float, float, float] | None = None
    inferred_ground_episode_root: _InferredGroundEpisodeRoot | None = None
    prior_outputs: tuple[_CanonicalOutputOrigin, ...] = ()
    presence_origin: _TrackerPresenceOrigin | None = None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _jsonable(tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda row: str(row[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def fingerprint_payload(role: str, payload: Any, *, version: str | None = None) -> ArtifactFingerprint:
    encoded = json.dumps(
        _jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return ArtifactFingerprint(role=role, sha256=hashlib.sha256(encoded).hexdigest(), version=version)


class CanonicalWorldService:
    """Normalize primitive tracking rows and own canonical global world state."""

    MAX_PREPARED_CANDIDATES = 16
    MAX_CANONICAL_TRACK_OUTPUTS = 4096
    # The estimator runs in the media callback while strict world publication
    # drains through an ordered non-blocking worker. A current proof can
    # therefore bind a slightly older output by the time the service consumes
    # it. Retain only exact outputs this service actually committed, with a
    # hard per-lifecycle cap; the current candidate is still gated against the
    # latest output below.
    MAX_CANONICAL_TRACK_OUTPUT_HISTORY = 128
    IMAGE_MOTION_COMMITTED_ORIGIN_HISTORY_HORIZON_S = 1.25
    # Image-motion filter-transition proof contract v1. These are the
    # canonical human filter's physical bounds, not producer-declared tuning.
    IMAGE_MOTION_TRANSITION_VERSION = 1
    IMAGE_MOTION_MAX_SPEED_MPS = 4.0
    IMAGE_MOTION_MAX_JUMP_M = 0.75
    IMAGE_MOTION_RESET_AFTER_S = 1.25
    INFERRED_PROCESS_RESTART_MAX_AGE_S = IMAGE_MOTION_RESET_AFTER_S
    IMAGE_MOTION_PROJECTIVE_SLACK_M = 0.35
    # DS9's filter clock is an epoch-shaped float anchored to integer media
    # PTS.  Subtracting adjacent values near 1.8e9 seconds can differ from the
    # exact integer-PTS delta by a few float ULPs.  One microsecond still binds
    # the physical gate to the same media interval (4 micrometres at 4 m/s)
    # while rejecting any operationally meaningful producer-declared slack.
    IMAGE_MOTION_GATE_DT_ABS_TOL_S = 1e-6
    PROJECTIVE_BRIDGE_IMAGE_PROVENANCE_TYPES = frozenset(
        {
            "bbox_affine_floor_projection",
            "inferred_ground_process_observation",
        }
    )
    INFERRED_PROCESS_CONSENSUS_HORIZON_S = 0.40
    # A learned-height continuation outside an explicit occlusion episode is
    # admitted only when the current detector row and three exact-cohort torso
    # observations independently agree that the established person is moving.
    # These are service-owned proof thresholds, not producer-declared tuning.
    INFERRED_PROCESS_MIN_IMAGE_MOTION_STREAK = 3
    INFERRED_PROCESS_MIN_DETECTOR_CONFIDENCE = 0.65
    # Four 10 Hz publication intervals bridge an isolated metric dropout. A
    # projective episode receives one additional, non-renewing tail after the
    # image-origin history window: this covers detector/reacquisition cadence
    # without letting CV descendants create a new root. Every descendant is
    # still tied to one service-committed image root and bounded to 4 m/s.
    BOUNDED_PROCESS_HORIZON_S = 0.40
    PROJECTIVE_PROCESS_BRIDGE_HORIZON_S = (
        IMAGE_MOTION_COMMITTED_ORIGIN_HISTORY_HORIZON_S
        + BOUNDED_PROCESS_HORIZON_S
    )
    STATIONARY_ANCHOR_HOLD_HORIZON_S = 2.0
    # A standing person can remain visually present while the current floor
    # candidate is rejected by the kinematic output gate.  Extend only an
    # exact latest-output hold backed by a current pose, a non-seated upright
    # silhouette, torso motion, and a plausible floor candidate close to that
    # public output. These fixed service-owned thresholds keep the lane camera
    # agnostic and prevent producer-declared evidence from widening it.
    UPRIGHT_PRESENCE_HOLD_HORIZON_S = 2.0
    UPRIGHT_PRESENCE_MIN_DETECTOR_CONFIDENCE = 0.70
    UPRIGHT_PRESENCE_MIN_TRACKER_CONFIDENCE = 0.50
    UPRIGHT_PRESENCE_MAX_OUTPUT_DISTANCE_M = 1.25
    UPRIGHT_PRESENCE_MIN_BBOX_HEIGHT_FRACTION = 48.0 / 1080.0
    UPRIGHT_PRESENCE_MAX_BBOX_ASPECT = 0.85
    # A current detector/tracker row may retain the last accurate coordinate
    # through one short localization hole. This lane is rooted at the last
    # motion-bearing output (not the previous hold), so repeated rows cannot
    # renew it. Bbox limits match the source-local lifecycle continuity rule.
    CURRENT_PRESENCE_HOLD_HORIZON_S = 1.0
    CURRENT_PRESENCE_MIN_DETECTOR_CONFIDENCE = 0.25
    CURRENT_PRESENCE_MIN_TRACKER_CONFIDENCE = 0.45
    CURRENT_PRESENCE_STRONG_DETECTOR_CONFIDENCE = 0.50
    CURRENT_PRESENCE_MAX_BBOX_SIZE_RATIO = 1.8
    CURRENT_PRESENCE_MAX_CENTER_DISPLACEMENT_NORM = 0.65
    # Match the producer's fixed media-clock tolerance at the hold boundary.
    # This is numeric/cadence slack only; it is never accumulated per row.
    BOUNDED_PROCESS_HORIZON_EPSILON_S = 0.005

    def __init__(
        self,
        *,
        producer: ProducerRef,
        artifacts: WorldArtifacts | Callable[[int, Mapping[str, Any]], WorldArtifacts],
        fusion: GlobalWorldFusion | None = None,
        clock_us: Callable[[], int] | None = None,
        journal: ContractJournal | AsyncContractJournal | None = None,
        association_service: IdentityAssociationService | None = None,
        physical_admission: Callable[[], ProducerAdmission] | None = None,
        physical_bindings: Callable[[], Mapping[int, ScenePriorFrameBinding]] | None = None,
        physical_rasters: Callable[[], Mapping[int, tuple[int, int]]] | None = None,
        spatial_bindings: Callable[[], Mapping[int, ScenePriorFrameBinding]] | None = None,
    ) -> None:
        self.producer = producer
        self._artifacts = artifacts
        self._fusion = (
            fusion.fork()
            if fusion is not None
            else GlobalWorldFusion(producer)
        )
        self._clock_us = clock_us or (lambda: int(time.time() * 1_000_000))
        self._journal = journal
        self._sequence_by_source: dict[int, int] = {}
        self._event_sequence = 0
        self._previous_entities: dict[str, WorldEntity] = {}
        self._canonical_track_outputs: dict[
            tuple[int, str, int, int, str, str, str, int],
            _CanonicalTrackOutput,
        ] = {}
        self._lock = RLock()
        self._closed = False
        self._revision = 0
        self._current_snapshot: WorldSnapshot | None = None
        self._service_token = object()
        self._next_candidate_id = 1
        self._prepared_candidates: dict[
            int,
            tuple[PreparedWorldPublication, _PreparedWorldState],
        ] = {}
        self._persistence_rejected_cohorts = 0
        self._persistence_last_error: str | None = None
        # All defaults remain subject-keyed. This optional candidate is entered
        # only through prepare_physical; it grants no native/live activation.
        if association_service is not None and type(association_service) is not IdentityAssociationService:
            raise TypeError("actual IdentityAssociationService required")
        self._association_service = association_service
        self._physical_admission = physical_admission
        self._physical_bindings = physical_bindings
        self._physical_rasters = physical_rasters
        self._spatial_bindings = spatial_bindings
        self._association_actor_id: int | None = None
        self._spatial_reason_counts: dict[str, int] = {}
        self._spatial_diagnostic_examples = 0
        self._spatial_fatal_logged = False
        self._spatial_observer = _SpatialDecisionObserver()

    def _observe_spatial(self, operation, *args):
        try:
            getattr(self._spatial_observer, operation)(*args)
        except Exception:
            try:self._spatial_observer.errors = min(2**63-1,self._spatial_observer.errors+1)
            except Exception:pass

    def _private_spatial_diagnostics(self):
        return json.loads(self._spatial_observer.encoded())

    def _log_spatial_hold_refusal(self, prepared, state, key, stamp, prior, custody, *,
                                  commit_us, expected=None, historical=None,
                                  deadline=None, admitted=None, branch="custody_or_stamp"):
        """One reserved fatal example; detached scalars, never a custody input."""
        try:
            if self._spatial_fatal_logged:
                return
            self._spatial_fatal_logged = True
            self._spatial_diagnostic_examples = min(8, self._spatial_diagnostic_examples+1)
            omitted_text = 0
            def text(value):
                nonlocal omitted_text
                if value is None:return None
                value = str(value)
                if len(value)>240:
                    omitted_text += 1
                    return value[:240]
                return value
            entity = prior.physical_entity_id(key)
            selected = [r for r in prior._by_entity.get(entity, {}).values()
                        if r.spatial_selected and r.spatial_stamp is not None]
            row = selected[0] if len(selected)==1 else None
            interval = next((i for i in prior._spatial_context.intervals
                if row is not None and i.binding.source_id==row.spatial_stamp.member.source_id),None)
            dto = next((d for d in state.spatial_memberships if d.physical_key==key),None)
            eligibility = next((e for e in dto.eligibility
                if e.observation.member==stamp.member),None) if dto else None
            candidate_entity=next((e for e in state.publication.snapshot.entities if e.entity_id==entity),None)
            candidate_selected=next((s for s in candidate_entity.sources if s.accepted),None) if candidate_entity else None
            record = dict(contract="noesis.world.private_hold_refusal.v1", branch=branch,
                candidate=prepared._candidate_id, source=state.source_id, key=key.serial,
                requested_source=stamp.member.source_id, raw=stamp.member.raw_id,
                generation=stamp.member.lifecycle_generation, epoch=stamp.member.source_epoch,
                birth=stamp.lifecycle_started_frame, frame=stamp.frame_id,
                requested_id=text(stamp.observation_id), custody_present=custody is not None,
                custody_stamp_equal=custody[1]==stamp if custody else None,
                custody_id=text(custody[0].observation_id) if custody else None,
                expiry_check_attempted=custody is None,
                expected_present=expected is not None, historical_present=historical is not None,
                expected_id=text(expected[0].observation_id) if expected else None,
                historical_id=text(historical[0].observation_id) if historical else None,
                historical_stamp_equal=historical[1]==stamp if historical else None,
                envelope_identical=historical[0] is expected[0] if historical and expected else None,
                stamp_equal=historical[1]==expected[1] if historical and expected else None,
                confidence_equal=historical[2]==expected[2] if historical and expected else None,
                subject_equal=historical[3]==expected[3] if historical and expected else None,
                prepared_us=state.publication.snapshot.published_at_us, commit_us=commit_us,
                classifier_deadline_us=deadline, classifier_admitted=admitted is not None if deadline is not None else None,
                candidate_present=candidate_entity is not None,
                candidate_state=candidate_entity.position_state if candidate_entity else None,
                candidate_deadline_us=candidate_entity.stale_after_us if candidate_entity else None,
                candidate_selected_id=text(candidate_selected.observation_id) if candidate_selected else None,
                candidate_selected_equal=candidate_selected.observation_id==stamp.observation_id if candidate_selected else None,
                selected_count=len(selected), selected_id=text(row.envelope.observation_id) if row else None,
                selected_stamp_equal=row.spatial_stamp==stamp if row else None,
                selected_deadline_us=row.envelope.observed_at_us+prior.config.lost_ttl_us if row else None,
                # Raw clock difference only, not another authority evaluation.
                source_elapsed_us=round((interval.evaluation_max_s-row.spatial_stamp.acquired_at_s)*1e6)
                    if interval is not None else None,
                owner_basis=text(dto.association_basis) if dto else None,
                owner_revision=dto.association_revision if dto else None,
                member_eligible=eligibility.eligible if eligibility else None,
                member_reason=text(eligibility.reason) if eligibility else None,
                member_stamp_equal=eligibility.observation==stamp if eligibility else None,
                native_state=text(eligibility.lifecycle_receipt.native_state)
                    if eligibility and eligibility.lifecycle_receipt else None)
            record['omitted_text_values']=omitted_text
            encoded=json.dumps(record,allow_nan=False,sort_keys=True,separators=(',',':'))
            if len(encoded.encode('utf-8'))>8192:
                encoded=json.dumps(dict(contract=record['contract'],candidate=prepared._candidate_id,
                    branch=branch,detail_omitted=True),separators=(',',':'))
            _LOGGER.error("Committed spatial hold refusal: %s",encoded)
        except Exception:
            pass  # Diagnostics cannot replace the exact authoritative refusal.

    def _capture_target_admissions(self, receipt, rows, stamps, decisions, snapshot, working,
                                   memberships, installed, selections, withdrawals, retirements):
        targets=self._spatial_observer.targets
        if not targets:return
        def encoded(items,limit):
            omitted=0
            while True:
                value=json.dumps({'items':items,'omitted':omitted},allow_nan=False,separators=(',',':'))
                if len(value.encode())<=limit:return value
                items.pop();omitted+=1
        final={s.observation_id:s for e in snapshot.entities for s in e.sources}
        present={e.entity_id for e in snapshot.entities}
        state=self._prepared_candidates[receipt._candidate_id][1]
        for source in sorted({state.source_id}|{stamps[o.observation_id].member.source_id for o,_,_ in rows}):
            current=[stamps[o.observation_id] for o,_,_ in rows if stamps[o.observation_id].member.source_id==source]
            media=max((s.media_time_s for s in current if s.media_time_s is not None),default=None)
            if media is None:
                media=next((i.evaluation_max_s for i in state.spatial_context.intervals
                    if i.binding.source_id==source and i.binding.clock_basis=='source-media'),None)
            if media is None:continue
            frame=max((s.frame_id for s in current),default=None)
            matches=[(i,abs(media-t)) for i,t in enumerate(targets) if abs(media-t)<=.100000001]
            if not matches:continue
            members=[]
            for dto in memberships:
                eligible={e.observation.member:e for e in dto.eligibility}
                for member in dto.members:
                    stamp=member.observation
                    if stamp.member.source_id!=source:continue
                    e=eligible.get(stamp.member)
                    decision=decisions.get(stamp.observation_id,(None,None,'unobserved_stage',None))
                    accepted=final.get(stamp.observation_id)
                    members.append(dict(raw=stamp.member.raw_id,generation=stamp.member.lifecycle_generation,
                        birth=stamp.lifecycle_started_frame,frame=stamp.frame_id,key=dto.physical_key.serial,
                        basis=dto.association_basis,eligible=e.eligible if e else None,reason=e.reason if e else None,
                        native=e.lifecycle_receipt.native_state if e and e.lifecycle_receipt else None,
                        selector_reason=decision[2],preliminary_selected=decision[3],
                        final_accepted=accepted.accepted if accepted else None))
                    if e is not None and e.reason=='image_region_ambiguous_member':
                        try:members[-1]['ambiguity']=self._association_service._private_image_region_ambiguity(stamp)
                        except Exception:
                            members[-1]['ambiguity']=dict(available=False,stamp_match=None,cause='diagnostic_unavailable')
            losses=[]
            for old in self._previous_entities.values():
                if old.entity_id in present or old.stale_after_us<=snapshot.published_at_us:continue
                accepted=next((s for s in old.sources if s.accepted),None)
                if accepted is None or accepted.observation_id in final and final[accepted.observation_id].accepted:continue
                previous=next((r for r in self._fusion._by_entity.get(old.entity_id,{}).values()
                    if r.envelope.observation_id==accepted.observation_id),None)
                if previous is None:
                    previous=next((c.row for c in self._fusion._split_custody.values()
                        if c.row.envelope.observation_id==accepted.observation_id),None)
                stamp=previous.spatial_stamp if previous is not None else None
                if stamp is None or stamp.member.source_id!=source:continue
                incoming=next((d for d in memberships if any(m.observation.member==stamp.member for m in d.members)),None)
                view=incoming.estimated_split_custody if incoming else None
                note,age=working._omission_diagnostics.get(old.entity_id,('unobserved_stage',None))
                losses.append(dict(original_id=accepted.observation_id,key=int(old.entity_id.rsplit(':',1)[-1]),
                    raw=stamp.member.raw_id,generation=stamp.member.lifecycle_generation,birth=stamp.lifecycle_started_frame,
                    frame=stamp.frame_id,observed_us=old.observed_at_us,deadline_us=old.stale_after_us,
                    remaining_us=old.stale_after_us-snapshot.published_at_us,prior_state=getattr(old.position_state,'value',old.position_state),
                    last_executed_omission_branch=note,source_age_us=age,withdrawn=stamp.member in withdrawals,
                    retired=any(working.physical_entity_id(k)==old.entity_id for k in retirements),
                    incoming_key=incoming.physical_key.serial if incoming else None,
                    incoming_view=view.episode_id if view else None,
                    fixed_owner_deadline=view.owner_deadline_monotonic_s if view else None,
                    installed=any(c.row.envelope.observation_id==accepted.observation_id for c in installed),
                    successor_selected=bool(view and any(d.physical_key in view.successor_keys for d,_ in selections))))
            barriers={'withdrawn':[(m.source_id,m.source_epoch,m.raw_id,m.lifecycle_generation) for m in sorted(withdrawals)],
                      'retired':[k.serial for k in sorted(retirements)]}
            barrier_text=json.dumps(barriers,separators=(',',':'))
            if len(barrier_text.encode())>2048:barrier_text=json.dumps({'unavailable':'byte_bound','withdrawn_count':len(withdrawals),'retired_count':len(retirements)})
            for index,distance in matches:
                self._observe_spatial('record',dict(kind='target_cohort_admission',candidate=receipt._candidate_id,
                    transaction='prepared',source=source,method=None,frame=frame,media_s=media,
                    target_index=index,distance_s=distance,target_members=encoded(list(members),4096),
                    target_losses=encoded(list(losses),2048),target_barriers=barrier_text))

    def _capture_spatial_decisions(self, receipt, rows, summaries, stamps, decisions, snapshot, working,
                                   memberships=(), split_checks=None, installed=(), selections=(), withdrawals=(), retirements=()):
        try:
            self._capture_target_admissions(receipt,rows,stamps,decisions,snapshot,working,
                memberships,installed,selections,withdrawals,retirements)
            final = {s.observation_id:s for e in snapshot.entities for s in e.sources}
            for original, confidence, raw_reason in rows:
                summary=summaries[original.observation_id]
                raw_entry = summary.get('accepted') is True and summary.get('world_valid') is True
                if not raw_entry and summary.get('method') not in {'registered_depth','registered_body_column'}:
                    continue
                dto,eligibility,selector,selected_id=decisions.get(original.observation_id,(None,None,'unobserved_stage',None))
                source=final.get(original.observation_id)
                if source is not None and source.accepted:continue
                stamp=stamps[original.observation_id]
                self._observe_spatial('record',{'kind':'current_original_nonvote',
                    'candidate':receipt._candidate_id,'transaction':'prepared',
                    'original_id':original.observation_id,'source':stamp.member.source_id,
                    'frame':stamp.frame_id,'raw':stamp.member.raw_id,'generation':stamp.member.lifecycle_generation,
                    'birth':stamp.lifecycle_started_frame,'epoch':stamp.member.source_epoch,
                    'physical_serial':dto.physical_key.serial if dto else None,
                    'owner_revision':dto.association_revision if dto else None,
                    'basis':dto.association_basis if dto else None,'method':summary.get('method'),
                    'raw_accepted':summary.get('accepted'),'raw_world_valid':summary.get('world_valid'),
                    'raw_entry_eligible':raw_entry,'selector_evaluated':original.observation_id in decisions,
                    'raw_reason':raw_reason,'resolver_confidence':confidence,
                    'covariance_check':summary.get('covariance',{}).get('check'),
                    'owner_eligible':eligibility.eligible if eligibility else None,
                    'owner_stamp_matches':eligibility.observation==stamp if eligibility else None,
                    'owner_reason':eligibility.reason if eligibility else None,
                    'native_state':eligibility.lifecycle_receipt.native_state if eligibility and eligibility.lifecycle_receipt else None,
                    'selector_reason':selector,'selected_original_id':selected_id,
                    'reservation_outcome':'reserved' if any(r.reference.stamp.observation_id==original.observation_id
                        for r in self._prepared_candidates[receipt._candidate_id][1].spatial_reservations) else 'not_requested',
                    'final_reason':source.rejection_reason if source else 'original_absent_from_snapshot',
                    'final_source_accepted':source.accepted if source else None})
            present={e.entity_id for e in snapshot.entities}
            for old in self._previous_entities.values():
                if old.entity_id in present or old.stale_after_us<=snapshot.published_at_us:
                    continue
                reason,age=working._omission_diagnostics.get(old.entity_id,('unobserved_stage',None))
                accepted=next((s for s in old.sources if s.accepted),None)
                represented=final.get(accepted.observation_id) if accepted else None
                if represented is not None and represented.accepted:
                    continue  # Same original moved keys: its point was not lost.
                # Read the committed original, not the transient clear/reinstall
                # omission note. These are diagnostic views of this candidate only.
                prior=next((c for c in self._fusion.split_custody_originals()
                            if GlobalWorldFusion.physical_entity_id(c.destination_key or c.origin_key)==old.entity_id),None)
                incoming=next((d.estimated_split_custody for d in memberships
                    if prior is not None and d.estimated_split_custody is not None
                    and d.estimated_split_custody.episode_id==prior.episode_id
                    and d.estimated_split_custody.origin_key==prior.origin_key),None)
                check=(split_checks or {}).get(prior.episode_id) if prior else None
                eligible=[e for d in memberships if prior and d.physical_key in prior.successor_keys
                          for e in d.eligibility]
                own=next((e for e in eligible if e.observation.member==prior.row.spatial_stamp.member),None) if prior else None
                peer=next((e for e in eligible if e is not own),None)
                prior_guard=None
                if prior is not None and prior.owner_diagnostic_view is not None:
                    try:
                        prior_guard=self._association_service.estimated_split_custody_diagnostic(
                            prior.owner_diagnostic_view,
                            at_interval=self._prepared_candidates[receipt._candidate_id][1].spatial_context,
                            observed_monotonic_s=time.monotonic())
                    except Exception:prior_guard={'available':False}
                self._observe_spatial('record',{'kind':'unexpired_held_loss','candidate':receipt._candidate_id,
                    'transaction':'prepared','original_id':accepted.observation_id if accepted else None,
                    'prior_position_state':getattr(old.position_state,'value',old.position_state),
                    'source':accepted.camera_id if accepted else None,'method':None,
                    'wire_remaining_us':old.stale_after_us-snapshot.published_at_us,
                    'original_observed_at_us':old.observed_at_us,'original_wire_deadline_us':old.stale_after_us,
                    'prepared_published_at_us':snapshot.published_at_us,
                    'omission_reason':reason,'source_age_us':age,
                    'prior_episode':prior.episode_id if prior else None,
                    'prior_origin_serial':prior.origin_key.serial if prior else None,
                    'prior_successor_a':prior.successor_keys[0].serial if prior else None,
                    'prior_successor_b':prior.successor_keys[1].serial if prior else None,
                    'incoming_view_present':incoming is not None,
                    'owner_deadline_mono_s':incoming.owner_deadline_monotonic_s if incoming else None,
                    'owner_checked_mono_s':check[0] if check else None,
                    'owner_lease_expired':check[1] if check else None,
                    'custody_installed':any(c.episode_id==prior.episode_id for c in installed) if prior else None,
                    'successor_selected':any(d.physical_key in prior.successor_keys for d,r in selections) if prior else None,
                    'successor_published':any(working.physical_entity_id(k) in present for k in prior.successor_keys) if prior else None,
                    'original_member_eligible':own.eligible if own else None,
                    'original_member_reason':own.reason if own else None,
                    'peer_member_eligible':peer.eligible if peer else None,
                    'peer_member_reason':peer.reason if peer else None,
                    'prior_owner_guard':json.dumps(prior_guard,allow_nan=False,separators=(',',':')) if prior_guard is not None else None})
            self._prepared_candidates[receipt._candidate_id][1].spatial_observer_records=tuple(
                dict(r) for r in self._spatial_observer.records if r.get('candidate')==receipt._candidate_id)
        except Exception:
            try:self._spatial_observer.errors = min(2**63-1,self._spatial_observer.errors+1)
            except Exception:pass

    def persistence_health(self) -> dict[str, Any]:
        """Return bounded optional-persistence health without touching media."""

        with self._lock:
            journal = self._journal
            rejected = int(self._persistence_rejected_cohorts)
            last_error = self._persistence_last_error
        if journal is None:
            return {
                "status": "disabled",
                "rejected_cohorts": rejected,
                "last_error": last_error,
            }
        snapshot_getter = getattr(journal, "health_snapshot", None)
        snapshot = (
            dict(snapshot_getter())
            if callable(snapshot_getter)
            else {"status": "synchronous"}
        )
        if rejected or last_error:
            snapshot["status"] = "degraded"
        snapshot["rejected_cohorts"] = rejected
        snapshot["last_error"] = last_error or snapshot.get("last_error")
        return snapshot

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def current_snapshot(self) -> WorldSnapshot | None:
        """Return the last committed immutable snapshot without mutation."""

        with self._lock:
            return self._current_snapshot

    def publish(
        self,
        source_id: int,
        tracks: list[Mapping[str, Any]],
        *,
        metadata: Mapping[str, Any],
    ) -> WorldPublication:
        """Compatibility transaction that prepares and commits under one lock."""

        with self._lock:
            prepared = self._prepare_locked(
                source_id,
                tracks,
                metadata=metadata,
            )
            try:
                return self._commit_locked(prepared)
            except Exception:
                self._prepared_candidates.pop(prepared._candidate_id, None)
                raise

    def prepare(
        self,
        source_id: int,
        tracks: list[Mapping[str, Any]],
        *,
        metadata: Mapping[str, Any],
        physical_observations: tuple[SourceObservation, ...] = (),
    ) -> PreparedWorldPublication:
        """Build an exact candidate without mutating fusion, counters, or journal."""

        with self._lock:
            return self._prepare_locked(
                source_id,
                tracks,
                metadata=metadata,
                physical_observations=physical_observations,
            )

    def commit(
        self,
        prepared: PreparedWorldPublication,
        *,
        association_at: AcquisitionTime | None = None,
        association_at_interval: SpatialClockContext | None = None,
    ) -> WorldPublication:
        """Commit one still-current candidate after outbound admission."""

        with self._lock:
            return self._commit_locked(prepared, association_at=association_at, association_at_interval=association_at_interval)

    def discard(self, prepared: PreparedWorldPublication) -> None:
        """Forget an unadmitted candidate without changing authority state."""

        with self._lock:
            self._validate_prepared_owner(prepared)
            entry = self._prepared_candidates.get(prepared._candidate_id)
            if entry is None:
                return
            receipt, state = entry
            if receipt is not prepared:
                raise RuntimeError("canonical world preparation identity mismatch")
            if self._association_service is not None:
                self._association_actor()
            self._prepared_candidates.pop(prepared._candidate_id, None)
            self._observe_spatial('outcome',prepared._candidate_id,'discarded')
            self._discard_association_reservations(state.association_reservations)
            self._discard_spatial_reservations(state.spatial_reservations)

    def _association_actor(self) -> None:
        if self._association_service is None:
            raise RuntimeError("physical association consumer is not configured")
        thread_id = get_ident()
        if self._association_actor_id is None:
            self._association_actor_id = thread_id
        if self._association_actor_id != thread_id:
            raise RuntimeError("physical association requires one ordered publication actor")

    def _discard_association_reservations(self, reservations: tuple[AssociationReservation, ...]) -> None:
        if self._association_service is not None:
            for reservation in reservations:
                self._association_service.discard(reservation)

    def _discard_spatial_reservations(self, reservations: tuple[SpatialMemberReservation, ...]) -> None:
        if self._association_service is not None:
            for reservation in reservations:
                self._association_service.discard_spatial_member(reservation)

    def _spatial_input_quality(self, track: Mapping[str, Any], original: ObservationEnvelope):
        """Preserve raw absence/invalidity before legacy normalized fallbacks."""
        raw_conf, raw_cov, raw_point = (track.get(k) for k in ("world_resolver_confidence", "world_covariance", "world"))
        confidence = self._confidence(raw_conf) if type(raw_conf) in {int, float} else None
        covariance = self._world_covariance(raw_cov)
        conf_status = "absent" if raw_conf is None else "valid" if confidence is not None else "invalid"
        cov_status = "absent" if raw_cov is None else "valid" if covariance is not None else "invalid"
        conf_finite = None
        if type(raw_conf) in {int, float}:
            try:
                conf_finite = math.isfinite(raw_conf)
            except OverflowError:
                conf_finite = False
        point_valid = False
        if isinstance(raw_point, (list, tuple)) and len(raw_point) >= 3:
            try:
                point_valid = all(math.isfinite(float(v)) for v in raw_point[:3])
            except (TypeError, ValueError, OverflowError):
                pass
        accepted, valid = track.get("world_measurement_accepted"), track.get("world_valid")
        registration = (
            track.get("world_frame") == "backend_world_m"
            and self._required_revision(track.get("world_frame_revision")) is not None
            and self._sha256_text(track.get("world_transform_sha256")) is not None
            and self._required_revision(track.get("calibration_revision")) is not None
        )
        reason = None
        if raw_point is not None and not point_valid:
            reason = "world_position_invalid"
        elif cov_status == "invalid":
            reason = "resolver_covariance_invalid"
        elif conf_status == "invalid":
            reason = "resolver_confidence_invalid"
        elif any(v is not None and type(v) is not bool for v in (accepted, valid)):
            reason = "producer_state_invalid"
        elif accepted is True and valid is False:
            reason = "producer_state_conflict"
        elif raw_point is not None and not registration:
            reason = "world_registration_unavailable"
        elif track.get("world_quantity") is not None and track.get("world_quantity") != "ground_footprint":
            reason = "world_quantity_invalid"
        elif track.get("world_support_state") is not None and (
                type(track["world_support_state"]) is not str
                or track["world_support_state"].strip().lower() not in {"floor", "seat", "couch", "unknown"}):
            reason = "world_support_invalid"
        elif accepted is False:
            reason = "producer_no_position" if raw_point is None or valid is False else "non_current_measurement"
        elif valid is False and raw_point is None:
            reason = "producer_no_position"
        elif accepted is not True or valid is not True:
            reason = "measurement_state_unavailable"
        elif raw_point is None:
            reason = "accepted_world_position_unavailable"
        elif str(track.get("world_source") or "").lower() in {"cv_prediction", "image_motion_prediction", "anchor_hold"} or track.get("world_quality") == "held":
            reason = "producer_state_conflict"
        elif cov_status == "absent":
            reason = "resolver_covariance_absent"
        elif conf_status == "absent":
            reason = "resolver_confidence_absent"
        if original.payload.world is None and accepted is True and valid is True and spatial_rejection_category(reason) != "integrity":
            # A missing normalized position is benign ONLY if explicit raw
            # producer state explained it. Missing quality cannot hide a
            # calibration, support, or other normalization contradiction.
            divergence = original.payload.world_diagnostics
            exact = divergence.first_divergence_reason if divergence else None
            reason = exact if exact in SPATIAL_INTEGRITY_REASONS else "accepted_world_position_unavailable"
        def kind(value):
            return ("null" if value is None else "bool" if type(value) is bool else
                    "number" if type(value) in {int, float} else "array" if type(value) in {list, tuple}
                    else "string" if type(value) is str else "object" if type(value) is dict else "other")
        finite_cov = None
        if isinstance(raw_cov, (list, tuple)) and len(raw_cov) <= 9:
            try:
                values = tuple(float(v) for v in raw_cov)
                if all(math.isfinite(v) for v in values):
                    finite_cov = values
            except (TypeError, ValueError, OverflowError):
                pass
        summary = {
            "source": original.payload.tracklet.source_id if 0 <= original.payload.tracklet.source_id < 2**64 else None,
            "frame": original.payload.tracklet.frame_id if 0 <= original.payload.tracklet.frame_id < 2**64 else None,
            "accepted": accepted if type(accepted) is bool else None, "world_valid": valid if type(valid) is bool else None,
            "owner_current": None,
            "method": track.get('world_resolver_selected_id') if track.get('world_resolver_selected_id') in
                {'registered_body_column','registered_depth','floor_ray','pose_scale','upright_pose_projection'} else 'other',
            "point_present": raw_point is not None, "point_valid": point_valid,
            "normalized_point": original.payload.world is not None,
            "support_present": track.get("world_support_state") is not None, "binding_present": bool(registration),
            "confidence": {"present": raw_conf is not None, "type": kind(raw_conf), "check": conf_status,
                "finite": conf_finite, "range": 0.0 <= raw_conf <= 1.0 if conf_finite else None},
            "covariance": {"present": raw_cov is not None, "type": kind(raw_cov), "check": cov_status,
                "shape": min(len(raw_cov), 1000) if isinstance(raw_cov, (list, tuple)) else None},
        }
        if finite_cov is not None:
            summary["covariance"]["values"] = finite_cov
        return confidence, reason, summary

    def _log_spatial_candidates(self, rows, summaries) -> None:
        """<=8 bounded examples/run; reserve space for the first fatal case."""
        rejected = [row for row in rows if row[2] is not None]
        if not rejected:
            return
        fatal = any(spatial_rejection_category(row[2]) == "integrity" for row in rejected)
        examples = []
        for original, _, reason in rows[:2]:
            safe_reason = reason if reason in SPATIAL_NO_VOTE_REASONS | SPATIAL_INTEGRITY_REASONS or reason is None else "unexpected_spatial_rejection_reason"
            if safe_reason is not None:
                self._spatial_reason_counts[safe_reason] = min(2**63 - 1, self._spatial_reason_counts.get(safe_reason, 0) + 1)
            examples.append({**summaries.get(original.observation_id, {
                "source": original.payload.tracklet.source_id, "frame": original.payload.tracklet.frame_id}),
                "reason": safe_reason, "category": spatial_rejection_category(reason),
                "count": self._spatial_reason_counts.get(safe_reason, 0)})
            if "covariance" in examples[-1]:
                examples[-1]["covariance"] = dict(examples[-1]["covariance"])
            examples[-1].pop('method',None)  # Preserve the existing classifier log contract.
        limit = 8 if self._spatial_fatal_logged or fatal else 7
        if self._spatial_diagnostic_examples >= limit:
            return
        encoded = json.dumps({"candidates": examples}, allow_nan=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > 1950:
            for example in examples:
                example.get("covariance", {}).pop("values", None)
            encoded = json.dumps({"candidates": examples}, allow_nan=False, separators=(",", ":"))
        self._spatial_diagnostic_examples += 1
        self._spatial_fatal_logged |= fatal
        _LOGGER.warning("Spatial candidate classification %s", encoded)

    def prepare_spatial(
        self,
        source_id: int,
        measurements: tuple[PreparedWorldPublication, ...],
        memberships: tuple[SpatialMembershipDTO, ...],
        *,
        at_interval: SpatialClockContext,
        withdrawn_members: tuple[MemberKey, ...] = (),
        retired_keys: tuple[PhysicalPersonKey, ...] = (),
        attribution_mode: str = "strict",
    ) -> PreparedWorldPublication:
        """ONE source-local transaction over ALL originals and FINAL owner DTOs.

        Finish native observations/joins/barriers before issuing memberships.
        World selects quality. A new current original may be reserved as an
        alternative while a separately custodied committed point stays HELD.
        Alternatives never grant CI, even with identical frame labels.
        """
        with self._lock:
            self._association_actor()
            if attribution_mode not in {"strict", "physical_only_quarantine"}:
                raise WorldPublicationRefused("unsupported_attribution_mode")
            if (type(memberships) is not tuple or len(memberships) > min(64, self._fusion.config.max_entities)
                    or any(type(dto) is not SpatialMembershipDTO for dto in memberships)
                    or type(at_interval) is not SpatialClockContext
                    or type(withdrawn_members) is not tuple or len(withdrawn_members) > 128
                    or any(type(m) is not MemberKey for m in withdrawn_members)
                    or type(retired_keys) is not tuple or len(retired_keys) > 64
                    or any(type(k) is not PhysicalPersonKey for k in retired_keys)):
                raise ValueError("bounded spatial cohort required")
            rows = self._spatial_measurements_locked(source_id, measurements)
            resolver_confidences = {row[0].observation_id:row[1] for row in rows}
            originals = tuple(self._prepared_candidates[r._candidate_id][1] for r in measurements)
            observations = tuple(row[0] for row in rows)
            contexts = {k: v for state in originals for k, v in state.physical_members.items()}
            stamps = {k: v for state in originals for k, v in state.physical_stamps.items()}
            upright_support = {k: v for state in originals for k, v in state.spatial_upright_support.items()}
            input_diagnostics = {k: v for state in originals for k, v in state.spatial_input_diagnostics.items()}
            reservations: list[SpatialMemberReservation] = []
            try:
                self._validate_spatial_memberships(memberships, at_interval)
                for dto in memberships:
                    if dto.household_conflict is not None and attribution_mode == "strict":
                        raise WorldPublicationRefused(dto.household_conflict.reason, household_conflict=dto.household_conflict)
                for row in rows:
                    if spatial_rejection_category(row[2]) == "integrity":
                        self._log_spatial_candidates((row,), input_diagnostics)
                        raise WorldPublicationRefused(row[2] if row[2] in SPATIAL_INTEGRITY_REASONS else "unexpected_spatial_rejection_reason")
                bindings = self._validated_spatial_bindings(observations, memberships,
                    {state.source_id: self._prepared_artifacts(state) for state in originals})
                working = self._fusion.fork()
                working.set_spatial_context(at_interval)
                split_checks = {}
                split_views = self._split_custody_views(memberships, attribution_mode,
                    observed_checks=split_checks)
                working.synchronize_split_episodes(split_views)
                split_originals = []
                for view in split_views:
                    custody = self._capture_attributed_split_original(working, view.episode_id, view.origin_key,
                        view.successor_keys, published_at_us=max(1, int(self._clock_us())))
                    if custody is not None:
                        custody=replace(custody,destination_key=self._split_original_destination(
                            view,custody.row.spatial_stamp,at_interval),owner_diagnostic_view=view)
                        self._validate_split_original(view, custody.row.spatial_stamp)
                        stamp = custody.row.spatial_stamp
                        active = self._spatial_bindings().get(stamp.member.source_id)
                        world = custody.row.envelope.payload.world
                        if (type(active) is not ScenePriorFrameBinding or world is None
                                or (active.source_frame.revision, active.target_frame.frame_id,
                                    active.target_frame.revision, active.target_from_source_sha256) !=
                                   (world.calibration_revision, world.world_frame,
                                    world.world_frame_revision, world.world_transform_sha256)):
                            raise WorldPublicationRefused("spatial_source_binding_changed")
                        bindings = tuple(dict((*bindings, (stamp.member.source_id, active))).items())
                        split_originals.append((view, custody))
                working.replace_split_custody(())
                removals = set(withdrawn_members) | {m for dto in memberships for m in dto.withdrawn_members}
                retired = set(retired_keys) | {k for dto in memberships for k in dto.retired_keys}
                if len(removals) > 128 or len(retired) > 64:
                    raise WorldPublicationRefused("spatial_barrier_capacity")
                reindexes = self._ordinary_observation_reindexes(memberships, retired, at_interval,
                    prior=working)
                for member in sorted(removals):
                    working.withdraw_physical_member(member)
                for key in sorted(retired):
                    working.retire_physical_key(key)
                for origin, dto, original in reindexes:
                    working.reindex_committed_spatial(self._fusion, origin, dto.physical_key,
                        original[1], published_at_us=max(1, int(self._clock_us())))
                    active = self._spatial_bindings()[original[1].member.source_id]
                    bindings = tuple(sorted({**dict(bindings), original[1].member.source_id: active}.items()))
                used: set[str] = set()
                ingestions = []
                selected_ids: set[str] = set()
                selections = []
                holds = []
                observer_decisions = {}
                for dto in memberships:
                    eligible = {e.observation.member: e for e in dto.eligibility}
                    members = {m.observation.member: m for m in dto.members}
                    candidates = []
                    for original, confidence, reason in rows:
                        stamp = stamps[original.observation_id]
                        member = members.get(stamp.member)
                        if member is None:
                            continue
                        if member.observation != stamp or contexts[original.observation_id][:2] != (stamp.member, stamp.lifecycle_started_frame):
                            raise WorldPublicationRefused("spatial_original_member_context")
                        used.add(original.observation_id)
                        eligibility = eligible.get(stamp.member)
                        input_diagnostics[original.observation_id] = {
                            **input_diagnostics[original.observation_id],
                            "owner_current": bool(eligibility and eligibility.observation == stamp and eligibility.eligible),
                        }
                        if eligibility is None or eligibility.observation != stamp or not eligibility.eligible:
                            reason = eligibility.reason if eligibility else "spatial_member_ineligible"
                        if (stamp.media_time_s is None or original.media_pts_ns is None
                                or not math.isclose(stamp.media_time_s, original.media_pts_ns / 1_000_000_000, abs_tol=1e-9, rel_tol=0.0)):
                            reason = "spatial_original_media_binding"
                        candidates.append((original, confidence, reason))
                    if not candidates:
                        continue  # Legitimately retained other person; no renewal.
                    selected, diagnostics = select_source_local_observation(tuple(candidates),
                        upright_support=tuple(upright_support[o.observation_id] for o, _, _ in candidates
                            if o.observation_id in upright_support))
                    for original,reason in diagnostics:
                        observer_decisions[original.observation_id]=(dto,eligible.get(stamps[original.observation_id].member),
                            reason,selected.observation_id if selected else None)
                    classified = tuple((original, None, reason) for original, reason in diagnostics
                                       if reason != "source_local_alternative")
                    self._log_spatial_candidates(classified, input_diagnostics)
                    fatal = next((reason for _, reason in diagnostics
                                  if reason != "source_local_alternative" and spatial_rejection_category(reason) == "integrity"), None)
                    if fatal is not None:
                        raise WorldPublicationRefused(fatal if fatal in SPATIAL_INTEGRITY_REASONS else "unexpected_spatial_rejection_reason")
                    if selected is None:
                        if dto.association_basis == "estimated_image_region_conflict_custody":
                            custody = self._conflict_custody_original(working, dto)
                            if custody is not None:
                                old, old_stamp, _, _ = custody
                                holds.append((dto.physical_key, old_stamp))
                                active = self._spatial_bindings()[old_stamp.member.source_id]
                                bindings = tuple(sorted({**dict(bindings),old_stamp.member.source_id:active}.items()))
                        continue  # Keep only previously granted bounded originals.
                    stamp = stamps[selected.observation_id]
                    interval = next(i for i in at_interval.intervals if i.binding.source_id == stamp.member.source_id)
                    world = selected.payload.world
                    house = dto.household_claim or members[stamp.member].household
                    subject = self._subject(selected, {"identity_kind": house.state,
                        "resident_uuid": house.identity_key if house.state == "resident" else None,
                        "stable_id": house.stable_id, "visitor_generation": house.visitor_generation})
                    retain_previous = False
                    image_basis = dto.association_basis in {
                        "estimated_current_image_regions", "estimated_image_region_continuation",
                        "estimated_live_track_continuity", "estimated_semantic_joint_correspondence"}
                    previous = (working.committed_spatial_original(dto.physical_key,
                                    published_at_us=max(1,int(self._clock_us()))) if image_basis else
                                working.selected_spatial_original(dto.physical_key)
                        if dto.association_basis == "prior_projective_continuity" or image_basis else None)
                    if previous is not None:
                        old,old_stamp,old_confidence,old_subject = previous
                        old_eligibility = eligible.get(old_stamp.member)
                        # A fresh same-source original always advances. Current
                        # representative reuse and committed HELD custody are
                        # separate; neither regrants/restamps the old original.
                        same_subject = old_subject == subject or (
                            old_subject.kind in {IdentityKind.UNKNOWN,IdentityKind.PROVISIONAL}
                            and subject.kind in {IdentityKind.UNKNOWN,IdentityKind.PROVISIONAL}
                            and old_subject.stable_id is None and subject.stable_id is None)
                        if image_basis:
                            old_member = members.get(old_stamp.member)
                            old_house = old_member.household if old_member is not None else None
                            original_subject = (self._subject(old, {"identity_kind": old_house.state,
                                "resident_uuid": old_house.identity_key if old_house.state == "resident" else None,
                                "stable_id": old_house.stable_id,
                                "visitor_generation": old_house.visitor_generation}) if old_house is not None else None)
                            same_subject = (old_house == house and original_subject == old_subject
                                and (old_subject == subject or old_house.state in {"unknown", "provisional"}
                                     and all(v is None for v in (old_subject.stable_id, subject.stable_id,
                                                                old_subject.resident_uuid, subject.resident_uuid))))
                            conflict = dto.household_conflict
                            covered = (attribution_mode == "physical_only_quarantine"
                                and type(conflict) is HouseholdClaimConflict
                                and conflict.reason == "settled_household_claim_conflict"
                                and conflict.run_id == dto.run_id
                                and conflict.association_revision == dto.association_revision
                                and old_house is not None and original_subject == old_subject
                                and (dto.physical_key, SourceObservation(old_member.observation, old_house)) in conflict.owners
                                and (dto.physical_key, SourceObservation(stamp, members[stamp.member].household)) in conflict.owners)
                            same_subject = same_subject or covered
                        old_quality = position_quality_sort_key(old, resolver_confidence=old_confidence, comparable_clock=False)
                        new_quality = position_quality_sort_key(selected, resolver_confidence=resolver_confidences[selected.observation_id], comparable_clock=False)
                        better_quality = (old_quality[:3] < new_quality[:3] if image_basis else old_quality < new_quality)
                        if (old_stamp.member.source_id != stamp.member.source_id and same_subject
                                and old_eligibility is not None and old_eligibility.eligible
                                and (old_eligibility.observation == old_stamp or image_basis
                                     and old_member is not None
                                     and old_eligibility.observation == old_member.observation
                                     and old_member.observation.lifecycle_started_frame == old_stamp.lifecycle_started_frame
                                     and old_member.observation.frame_id >= old_stamp.frame_id
                                     and old_eligibility.lifecycle_receipt is not None
                                     and old_eligibility.lifecycle_receipt.native_state == "ACTIVE")
                                and better_quality):
                            active_frames = self._spatial_bindings()
                            active = active_frames.get(old_stamp.member.source_id)
                            old_world = old.payload.world
                            if (type(active) is ScenePriorFrameBinding
                                    and (active.source_frame.revision,active.target_frame.frame_id,
                                         active.target_frame.revision,active.target_from_source_sha256) ==
                                        (old_world.calibration_revision,old_world.world_frame,
                                         old_world.world_frame_revision,old_world.world_transform_sha256)):
                                retain_previous = True
                                # Commit checks cover the retained original's
                                # active binding too; no old artifact is renewed.
                                bindings = tuple(sorted({**dict(bindings),old_stamp.member.source_id:active}.items()))
                                if image_basis and old_eligibility.observation != old_stamp:
                                    holds.append((dto.physical_key,old_stamp))
                    for original, reason in diagnostics:
                        if original.payload.world is None:
                            continue
                        member, _, trail, broken = contexts[original.observation_id]
                        ingestions.append((original, subject, dto.physical_key, member, trail, broken,
                                           stamps[original.observation_id], "source_local_alternative" if retain_previous else reason))
                    if retain_previous and (dto.physical_key, old_stamp) not in holds:
                        continue  # No new reference/reservation/vote/time/trail.
                    reference = SpatialMeasurementReference(
                        stamp, stamp.member, selected.observation_id, world.world_frame,
                        world.world_frame_revision, world.calibration_revision, world.world_transform_sha256,
                        "measured", "m", selected.observation_id, interval,
                    )
                    selections.append((dto, reference))
                    if not retain_previous:
                        selected_ids.add(selected.observation_id)
                if len(used) != len(observations):
                    raise WorldPublicationRefused("spatial_original_membership_missing")
                claims_by_key = {dto.physical_key: tuple(self._attribution_claim(dto.physical_key,
                    SourceObservation(m.observation, m.household)) for m in dto.members) for dto in memberships}
                for original, subject, key, member, trail, broken, stamp, reason in sorted(
                    ingestions, key=lambda row: (row[0].payload.tracklet.source_id, row[0].sequence),
                ):
                    working.ingest(original, subject, physical_key=key, physical_member=member,
                                   source_epoch=member.source_epoch, tracker_lifecycle_generation=member.lifecycle_generation,
                                   trail_segment_id=trail, trail_break_required=broken, spatial_stamp=stamp,
                                   spatial_selected=original.observation_id in selected_ids, spatial_rejection=reason,
                                   spatial_resolver_confidence=resolver_confidences[original.observation_id],
                                   attribution_claims=claims_by_key[key])
                for dto, reference in selections:
                    if not any(key == dto.physical_key for key, _ in holds):
                        working.select_spatial(dto.physical_key, reference.stamp.observation_id)
                for key, stamp in holds:
                    working.hold_committed_spatial(key,stamp)
                installed = []
                for view in split_views:
                    custody = next((c for v, c in split_originals if v is view), None)
                    successors = [(dto.physical_key, ref.stamp.observation_id) for dto, ref in selections
                                  if dto.physical_key in view.successor_keys]
                    if len(successors) > 1:
                        raise WorldPublicationRefused("split_custody_multiple_successors")
                    if successors or custody is not None:
                        working.enforce_split_exclusivity(view.successor_keys,
                            successors[0] if successors else None)
                    if successors:
                        working.consume_split_episode(view)
                    elif custody is not None:
                        installed.append(custody)
                working.replace_split_custody(tuple(installed))
                try:
                    self._prepare_attribution_quarantines(working, memberships, attribution_mode,
                        physical_selections=tuple(selections), committed_holds=holds,
                        observation_reindexes=reindexes)
                    snapshot = _physical_snapshot(working, max(1, int(self._clock_us())))
                except ValueError as error:
                    raise WorldPublicationRefused(str(error)) from error
                # COMPLETE fresh+retained uniqueness BEFORE every reservation.
                for dto, reference in selections:
                    _guard_world_snapshot(snapshot)
                    prepare_member = (self._association_service.prepare_spatial_physical_member
                        if dto.household_conflict is not None and attribution_mode == "physical_only_quarantine"
                        else self._association_service.prepare_spatial_member)
                    result = prepare_member(dto, reference, at_interval=at_interval)
                    if result.disposition != "reserved" or result.reservation is None:
                        raise WorldPublicationRefused.from_association_result(result)
                    expected_purpose = "physical_only_quarantine" if dto.household_conflict is not None else "strict"
                    if (result.reservation.purpose != expected_purpose
                            or result.reservation.household_conflict is not dto.household_conflict):
                        self._association_service.discard_spatial_member(result.reservation)
                        raise WorldPublicationRefused("attribution_reservation_changed")
                    reservations.append(result.reservation)
                sequences = dict(self._sequence_by_source)
                outputs = dict(self._canonical_track_outputs)
                for member in removals:
                    for cache_key in tuple(outputs):
                        if cache_key[:4] == (member.source_id, member.camera_id, member.raw_id, member.lifecycle_generation) and cache_key[7] == member.source_epoch:
                            outputs.pop(cache_key, None)
                for state in originals:
                    for source, sequence in state.sequence_by_source.items():
                        sequences[source] = max(sequence, sequences.get(source, 0))
                    for key, output in state.canonical_track_outputs.items():
                        if output.presence_origin and output.presence_origin.envelope.observation_id in selected_ids:
                            outputs[key] = output
                self._bound_track_outputs(outputs)
                trigger = next(state for state in originals if state.source_id == source_id)
                receipt = self._stage_candidate_locked(
                    trigger.publication.observations, snapshot, working, sequences, outputs,
                    source_id, trigger.artifact_metadata, self._prepared_artifacts(trigger),
                    association_candidate=True, physical_observations=observations,
                    association_artifacts=tuple((s.source_id, s.artifact_metadata, self._prepared_artifacts(s)) for s in originals),
                    spatial_memberships=memberships, spatial_reservations=tuple(reservations),
                    spatial_bindings=bindings, spatial_context=at_interval,
                    spatial_input_diagnostics=input_diagnostics,
                    attribution_mode=attribution_mode,
                    spatial_holds=tuple((key, stamp) for key, stamp in holds
                        if not any(dto.physical_key == key and original[1] == stamp
                                   for _, dto, original in reindexes)),
                    spatial_reindexes=tuple((origin, dto, original) for origin, dto, original in reindexes
                        if any(e.entity_id == working.physical_entity_id(dto.physical_key)
                            and any(s.accepted and s.observation_id == original[0].observation_id for s in e.sources)
                            for e in snapshot.entities)),
                    spatial_split_custody=tuple((v,c) for v,c in split_originals if c in working.split_custody_originals()),
                    spatial_split_episode_views=split_views,
                )
                self._capture_spatial_decisions(receipt,rows,input_diagnostics,stamps,observer_decisions,snapshot,working,
                    memberships,split_checks,installed,selections,removals,retired)
            except Exception:
                self._discard_spatial_reservations(tuple(reservations))
                for original in measurements:
                    self._prepared_candidates.pop(original._candidate_id, None)
                raise
            for original in measurements:
                self._prepared_candidates.pop(original._candidate_id, None)
            return receipt

    @staticmethod
    def _attribution_claim(key: PhysicalPersonKey, original: SourceObservation) -> WorldAttributionClaim:
        stamp, house = original.stamp, original.household
        if key.run_id != stamp.member.run_id:
            raise WorldPublicationRefused("attribution_original_run_mismatch")
        return WorldAttributionClaim(run_id=key.run_id, physical_serial=key.serial,
            camera_id=stamp.member.camera_id, source_id=stamp.member.source_id,
            source_epoch=stamp.member.source_epoch, raw_id=stamp.member.raw_id,
            lifecycle_generation=stamp.member.lifecycle_generation, lifecycle_started_frame=stamp.lifecycle_started_frame,
            frame_id=stamp.frame_id, observation_id=stamp.observation_id, acquired_at_s=stamp.acquired_at_s,
            clock_domain=stamp.clock_domain, media_time_s=stamp.media_time_s, clock_basis=stamp.clock_basis,
            household_state=house.state, household_identity_key=house.identity_key,
            household_stable_id=house.stable_id, household_visitor_generation=house.visitor_generation,
            household_constraint=house.constraint)

    def _prepare_attribution_quarantines(self, working, memberships, mode, *, physical_selections=(),
                                        committed_holds=None, observation_reindexes=()) -> None:
        """One bounded attribution projection over fresh AND retained original claims."""
        retained = working.spatial_attribution_claims()
        retained_originals = working.spatial_original_attribution_claims()
        current = {GlobalWorldFusion.physical_entity_id(d.physical_key): d for d in memberships}
        # A current owner decision supersedes its SAME original member, not another
        # contributor's claim or measurement lease.
        claims = {}
        def add(claim):
            member = (claim.entity_id, claim.camera_id, claim.source_id, claim.source_epoch,
                      claim.raw_id, claim.lifecycle_generation)
            previous = claims.get(member)
            if previous is None or claim.frame_id >= previous.frame_id:
                claims[member] = claim
        for rows in retained.values():
            for claim in rows:
                add(claim)
        for dto in memberships:
            for member in dto.members:
                add(self._attribution_claim(dto.physical_key, SourceObservation(member.observation, member.household)))
        conflicts = []
        for dto in memberships:
            conflict = dto.household_conflict
            if conflict is None:
                continue
            if (mode != "physical_only_quarantine" or type(conflict) is not HouseholdClaimConflict
                    or conflict.reason != "settled_household_claim_conflict"
                    or conflict.run_id != dto.run_id or conflict.association_revision != dto.association_revision
                    or not 2 <= len(conflict.owners) <= 128):
                raise WorldPublicationRefused("attribution_conflict_incomplete")
            if conflict not in conflicts:
                conflicts.append(conflict)
        original_owners = tuple(dict.fromkeys(self._attribution_claim(key, original)
            for conflict in conflicts for key, original in conflict.owners))
        all_claims = tuple(dict.fromkeys((*claims.values(), *original_owners,
            *(claim for rows in retained_originals.values() for claim in rows))))
        if len(all_claims) > 128:
            raise WorldPublicationRefused("attribution_claim_capacity")
        nodes = {c.entity_id for c in all_claims}
        if len(nodes) > 64:
            raise WorldPublicationRefused("attribution_component_capacity")
        _, _, internally_divergent = attribution_conflict_sets(all_claims)
        components = []
        while nodes:
            component = {nodes.pop()}
            while True:
                subject_ids = {c.subject_id for c in all_claims if c.entity_id in component and c.subject_id is not None}
                stable_ids = {c.household_stable_id for c in all_claims if c.entity_id in component and c.household_stable_id is not None}
                added = {c.entity_id for c in all_claims if c.entity_id in nodes and (
                    c.subject_id is not None and c.subject_id in subject_ids
                    or c.household_stable_id is not None and c.household_stable_id in stable_ids)}
                if not added:
                    break
                component.update(added)
                nodes.difference_update(added)
            if len(component) > 1 or component & internally_divergent:
                components.append(component)
        overlays = {}
        custody_at_us = max(1, int(self._clock_us()))
        def committed_original(fusion, key):
            # Attribution custody is bounded by the ORIGINAL source/wire TTL,
            # not the .35s accessor for a current position vote.
            return fusion.committed_spatial_original(key, published_at_us=custody_at_us)
        def retain_for_commit(key, stamp):
            if committed_holds is not None and (key, stamp) not in committed_holds:
                committed_holds.append((key, stamp))
        for component in components:
            matching = [conflict for conflict in conflicts if component & {
                GlobalWorldFusion.physical_entity_id(key) for key, _ in conflict.owners}]
            if not matching and mode == "physical_only_quarantine":
                # A geometry split may revoke CURRENT joint attribution while
                # a peer still legitimately holds its previously granted point.
                # Its committed typed quarantine remains proof of those exact
                # ORIGINAL claims, not a grant for any fresh/rekeyed evidence.
                prior = self._fusion.attribution_quarantines()
                visible = component & retained.keys()
                records = [prior.get(entity) for entity in visible]
                previous = records[0] if records else None
                if previous is not None and all(record == previous for record in records):
                    survivors = set(previous.component_entity_ids) & retained.keys()
                    covered = set((*previous.owners, *previous.retained_owners))
                    unchanged = visible <= set(previous.component_entity_ids)
                    survivor_holds = []
                    for entity in survivors:
                        key = PhysicalPersonKey(previous.run_id, int(entity.rsplit(":", 1)[-1]))
                        before = committed_original(self._fusion, key)
                        after = committed_original(working, key)
                        unchanged &= (before is not None and after is not None
                            and before[0] is after[0] and before[1:] == after[1:]
                            and set(retained[entity]) <= covered)
                        if before is not None:
                            survivor_holds.append((key, before[1]))
                    if unchanged and survivors:
                        for key, stamp in survivor_holds:
                            retain_for_commit(key, stamp)
                        overlays.update((entity, previous) for entity in survivors)
                        continue  # No new vote/reservation/claim/clock/digest.
                # LABEL-ONLY attenuation may extend the old retained conflict
                # to a genuinely current rekeyed physical sample. Identity does
                # not know World-retained custody. Never forge its conflict or
                # grant: the real selected reference is reserved/revalidated by
                # the normal owner API after this whole-snapshot projection.
                custody = self._fusion.spatial_original_attribution_claims()
                # Exact owner-certified OLD-original transfer may move its
                # label attenuation anchor, never fabricate a new claim or
                # measurement. Keep historical claim entity IDs untouched.
                clean_transfers = set()
                for origin_key, dto, original in observation_reindexes:
                    entity = GlobalWorldFusion.physical_entity_id(dto.physical_key)
                    origin = GlobalWorldFusion.physical_entity_id(origin_key)
                    after = committed_original(working, dto.physical_key)
                    previous_entity = self._previous_entities.get(origin)
                    if (entity in component and after is not None and after[0] is original[0]
                            and after[1:] == original[1:] and origin in custody
                            and set(retained.get(entity, ())) <= set(custody[origin])):
                        custody[entity] = custody[origin]
                        if origin in prior:
                            prior[entity] = prior[origin]
                        elif previous_entity is not None and previous_entity.attribution_quarantine is None:
                            clean_transfers.add(entity)
                for transfer in working.split_custody_originals():
                    destination=transfer.destination_key
                    if destination is None or destination==transfer.origin_key:continue
                    entity=GlobalWorldFusion.physical_entity_id(destination)
                    origin=GlobalWorldFusion.physical_entity_id(transfer.origin_key)
                    before=self._fusion.committed_spatial_original(transfer.origin_key,
                        published_at_us=max(1,int(self._clock_us())))
                    if (entity in component and before is not None
                            and before[0] is transfer.row.envelope and before[1]==transfer.row.spatial_stamp
                            and origin in custody and set(retained.get(entity,()))<=set(custody[origin])):
                        custody[entity]=custody[origin]
                        if origin in prior:prior[entity]=prior[origin]
                        previous_entity = self._previous_entities.get(origin)
                        if (origin not in prior and entity not in prior
                                and previous_entity is not None
                                and previous_entity.attribution_quarantine is None
                                and previous_entity == transfer.entity):
                            # The certified destination has no ordinary selected
                            # row: it owns the exact private OLD-original hold.
                            # A clean committed point needs no invented prior
                            # conflict to attenuate its historical origin label.
                            # Previously disputed points still need their full
                            # committed quarantine; absence is not cleanliness.
                            clean_transfers.add(entity)
                prior_claims = {c for entity, rows in custody.items() if entity in component for c in rows}
                transferred_history = []
                for transfer in working.split_custody_originals():
                    # The public claim serial remains its HISTORICAL owner.
                    # Its current native destination can be anonymous and thus
                    # outside this LABEL component. Only the exact committed
                    # transfer may supply its saved original quarantine here;
                    # an unrelated prior quarantine is never an authority.
                    record = transfer.entity.attribution_quarantine
                    if (record is None or transfer.destination_key is None
                            or transfer.destination_key == transfer.origin_key
                            or GlobalWorldFusion.physical_entity_id(transfer.destination_key) in component
                            or not set(record.component_entity_ids) <= component):
                        continue
                    before = self._capture_attributed_split_original(self._fusion, transfer.episode_id,
                        transfer.origin_key, transfer.successor_keys, published_at_us=custody_at_us)
                    after = committed_original(working, transfer.destination_key)
                    original_claims = set(transfer.row.attribution_claims)
                    if (before is not None and before.entity is transfer.entity
                            and before.row.envelope is transfer.row.envelope
                            and before.row.spatial_stamp == transfer.row.spatial_stamp
                            and before.row.attribution_claims == transfer.row.attribution_claims
                            and after is not None and after[0] is transfer.row.envelope
                            and after[1] == transfer.row.spatial_stamp
                            and original_claims
                            and original_claims <= set((*record.owners, *record.retained_owners))
                            and {c.entity_id for c in original_claims} <= component):
                        prior_claims.update(original_claims)
                        transferred_history.append(record)
                current_claims = {self._attribution_claim(dto.physical_key, SourceObservation(m.observation, m.household))
                    for entity, dto in current.items() if entity in component for m in dto.members}
                active_native_claims = {self._attribution_claim(dto.physical_key,
                    SourceObservation(m.observation, m.household))
                    for entity, dto in current.items() if entity in component for m in dto.members
                    if any(e.observation == m.observation and e.lifecycle_receipt is not None
                        and e.lifecycle_receipt.stamp == m.observation
                        and e.lifecycle_receipt.native_state == "ACTIVE" for e in dto.eligibility)}
                selections = {GlobalWorldFusion.physical_entity_id(dto.physical_key): (dto, ref)
                              for dto, ref in physical_selections}
                position_holds = dict(committed_holds or ())
                anchored = bool(transferred_history)
                rekeyed_history = []
                # A certified transfer can move EVERY retained original outside
                # its historical label component. Its exact committed quarantine
                # already proves that history; no in-component head is required.
                complete = bool(visible or transferred_history)
                for entity in visible:
                    # Custody lasts for the ORIGINAL World position TTL, not
                    # the shorter .35s current-vote eligibility accessor. No
                    # selected reference for this key means no new position vote;
                    # every still-live claim must remain in committed custody.
                    held = (entity not in selections and entity in custody
                            and set(retained[entity]) <= set(custody[entity]))
                    record = prior.get(entity)
                    if (entity in selections and record is not None and custody.get(entity)
                            and set(custody[entity]) <= set((*record.owners, *record.retained_owners))
                            and any(e.attribution_quarantine == record for e in self._previous_entities.values())):
                        # Prior quarantine proves only the ORIGINAL claims. It
                        # does not require freezing the position: below, the
                        # exact current reference must independently qualify,
                        # then reserve/commit through the ordinary owner gates.
                        anchored = True
                    if held and entity in clean_transfers:
                        anchored = True
                    key = PhysicalPersonKey(self.producer.run_id, int(entity.rsplit(":", 1)[-1]))
                    retained_position = False
                    if key in position_holds:
                        before = committed_original(self._fusion, key)
                        after = committed_original(working, key)
                        previous_entity = self._previous_entities.get(entity)
                        retained_position = (before is not None and after is not None
                            and before[0] is after[0] and before[1:] == after[1:]
                            and before[1] == position_holds[key]
                            and previous_entity is not None
                            and previous_entity.observed_at_us == before[0].observed_at_us
                            and previous_entity.stale_after_us == before[0].observed_at_us + self._fusion.config.lost_ttl_us
                            and any(s.accepted and s.observation_id == before[1].observation_id
                                    for s in previous_entity.sources))
                    if record is None and (held or retained_position or entity in selections):
                        before = committed_original(self._fusion, key)
                        previous_entity = self._previous_entities.get(entity)
                        clean_committed = (before is not None and previous_entity is not None
                            and previous_entity.attribution_quarantine is None
                            and previous_entity.observed_at_us == before[0].observed_at_us
                            and any(s.accepted and s.observation_id == before[1].observation_id
                                    for s in previous_entity.sources))
                        # A CURRENT owner may rekey the exact original or advance
                        # that SAME native life with an unchanged household.
                        # Its old statement remains historical (serial included),
                        # while the exact current DTO proves only label provenance.
                        # Geometry still needs the normal current ref or unchanged
                        # finite committed hold checked below and at commit.
                        after = committed_original(working, key)
                        unchanged_original = (before is not None and after is not None
                            and before[0] is after[0] and before[1:] == after[1:]
                            and previous_entity is not None
                            and previous_entity.stale_after_us == before[0].observed_at_us + self._fusion.config.lost_ttl_us)
                        if clean_committed and any(old.entity_id == entity
                                and old.household_state in {"resident", "visitor", "handoff"}
                                and old.household_constraint == "eligible"
                                and old in custody.get(entity, ())
                                and sum(self._same_rekeyed_original_claim(old, new)
                                    or (unchanged_original and (held or retained_position)
                                        and new in active_native_claims
                                        and self._same_forward_rekeyed_native_claim(old, new))
                                    for new in current_claims) == 1 for old in retained[entity]):
                            anchored = True
                            rekeyed_history.extend(custody[entity])
                            if held or retained_position:
                                retain_for_commit(key, before[1])
                        original_current = (before is not None and entity in current
                            and any(e.eligible and e.observation == before[1]
                                and e.lifecycle_receipt is not None
                                and e.lifecycle_receipt.stamp == before[1]
                                and e.lifecycle_receipt.native_state == "ACTIVE"
                                for e in current[entity].eligibility))
                        if (clean_committed and unchanged_original
                                and (retained_position or held and original_current)):
                            # A clean selected point can already own a settled
                            # UNSELECTED peer claim. Its exact current native
                            # owner may attenuate that label while a new person
                            # claims it. The historical statement stays intact;
                            # only labels are quarantined over this proven hold.
                            member_fields = ("run_id", "physical_serial", "camera_id", "source_id",
                                "source_epoch", "raw_id", "lifecycle_generation", "lifecycle_started_frame",
                                "clock_domain", "clock_basis")
                            owned = set(custody.get(entity, ()))
                            attenuated = any(old.entity_id == entity
                                and old.household_state in {"resident", "visitor", "handoff"}
                                and old.household_constraint == "eligible"
                                and sum(new in active_native_claims
                                    and new.household_state in {"unknown", "provisional"}
                                    and new.household_constraint == "eligible"
                                    and all(getattr(new, f) == getattr(old, f) for f in member_fields)
                                    and new.frame_id > old.frame_id and new.observation_id != old.observation_id
                                    and new.acquired_at_s > old.acquired_at_s
                                    and old.media_time_s is not None and new.media_time_s is not None
                                    and new.media_time_s > old.media_time_s
                                    and any(e.eligible and e.observation.observation_id == new.observation_id
                                        for e in current[entity].eligibility)
                                    for new in current_claims) == 1 for old in owned)
                            if attenuated and set(retained[entity]) <= owned | current_claims:
                                held = True
                                anchored = True
                                retain_for_commit(key, before[1])
                    if (not held and (entity not in selections or retained_position)
                            and record is not None and entity in custody):
                        # A newer UNSELECTED alternative may attenuate its own
                        # household claim without replacing the committed point.
                        # Do not confuse that exact current decision with lost
                        # custody of the selected ORIGINAL or discard its claims.
                        key = PhysicalPersonKey(self.producer.run_id, int(entity.rsplit(":", 1)[-1]))
                        before = committed_original(self._fusion, key)
                        after = committed_original(working, key)
                        owned = set(custody[entity])
                        added = set(retained[entity]) - owned
                        covered = set((*record.owners, *record.retained_owners))
                        member_fields = ("run_id", "physical_serial", "camera_id", "source_id",
                            "source_epoch", "raw_id", "lifecycle_generation", "lifecycle_started_frame",
                            "clock_domain", "clock_basis")
                        household_fields = ("household_state", "household_identity_key", "household_stable_id",
                            "household_visitor_generation", "household_constraint")
                        attenuation = bool(added) and all(new in current_claims
                            and new.household_constraint == "eligible"
                            and any(old.household_constraint == "eligible"
                                and (all(getattr(new, f) == getattr(old, f) for f in household_fields)
                                     or old.household_state in {"resident", "visitor", "handoff"}
                                        and new.household_state in {"unknown", "provisional"})
                                and all(getattr(new, f) == getattr(old, f) for f in member_fields)
                                and new.frame_id > old.frame_id and new.acquired_at_s >= old.acquired_at_s
                                and (old.media_time_s is None and new.media_time_s is None
                                     or old.media_time_s is not None and new.media_time_s is not None
                                     and new.media_time_s >= old.media_time_s)
                                for old in owned) for new in added)
                        # A uniquely accepted current pair can replace an
                        # UNSELECTED member while the selected native owner
                        # still owns this exact committed point. Its previous
                        # quarantine proves the displaced historical claims;
                        # current DTOs prove only their forward label custody.
                        dto = current.get(entity)
                        selected_claim = next((c for c in owned if before is not None
                            and c.observation_id == before[1].observation_id), None)
                        current_owner = next((c for c in current_claims if selected_claim is not None
                            and all(getattr(c, f) == getattr(selected_claim, f)
                                for f in (*member_fields, *household_fields))
                            and c.frame_id > selected_claim.frame_id
                            and c.acquired_at_s > selected_claim.acquired_at_s
                            and c.media_time_s is not None and selected_claim.media_time_s is not None
                            and c.media_time_s > selected_claim.media_time_s
                            and c in active_native_claims
                            and dto is not None and any(e.eligible
                                and e.observation.observation_id == c.observation_id for e in dto.eligibility)), None)
                        replaced = bool(added) and dto is not None and len(dto.members) == 2 and (
                            dto.association_basis == "estimated_semantic_joint_correspondence"
                            and current_owner is not None
                            and all(new in current_claims and (new == current_owner or (
                                new.entity_id == entity and new in active_native_claims
                                and new.household_state in {"unknown", "provisional"}
                                and new.household_constraint == "eligible"
                                and new.household_identity_key is None and new.household_stable_id is None
                                and new.household_visitor_generation is None
                                and any(e.eligible and e.observation.observation_id == new.observation_id
                                    for e in dto.eligibility))) for new in added)
                            and all(old == selected_claim or sum(new in active_native_claims and (
                                self._same_rekeyed_original_claim(old, new)
                                or self._same_forward_rekeyed_native_claim(old, new))
                                for new in current_claims) == 1 for old in owned))
                        if (before is not None and after is not None
                                and before[0] is after[0] and before[1:] == after[1:]
                                and owned <= covered and (attenuation or replaced)
                                and entity in record.component_entity_ids
                                and (published := self._previous_entities.get(entity)) is not None
                                and published.attribution_quarantine == record):
                            held = True
                            anchored = True
                            retain_for_commit(key, before[1])
                    if held and record is not None and set(retained[entity]) <= set((*record.owners, *record.retained_owners)):
                        anchored = True
                    if held and record is None:
                        # A clean committed point may need label attenuation
                        # when a CURRENT same-member claim changes. This cannot
                        # renew its position or authorize foreign retained claims.
                        key = PhysicalPersonKey(self.producer.run_id, int(entity.rsplit(":", 1)[-1]))
                        before = committed_original(self._fusion, key)
                        after = committed_original(working, key)
                        same_point = (before is not None and after is not None
                            and before[0] is after[0] and before[1:] == after[1:])
                        if same_point:
                            previous_entity = self._previous_entities.get(entity)
                            clean_committed = (previous_entity is not None
                                and previous_entity.attribution_quarantine is None
                                and previous_entity.observed_at_us == before[0].observed_at_us
                                and any(s.accepted and s.observation_id == before[1].observation_id
                                        for s in previous_entity.sources))
                            member_fields = ("run_id", "physical_serial", "camera_id", "source_id",
                                "source_epoch", "raw_id", "lifecycle_generation", "lifecycle_started_frame",
                                "clock_domain", "clock_basis")
                            for old in retained[entity]:
                                if (old.observation_id != before[1].observation_id
                                        or old.frame_id != before[1].frame_id
                                        or old.household_constraint != "eligible"):
                                    continue
                                # Clean committed custody may predate a new
                                # SAME-member settled label conflicting with a
                                # historical peer claim. Quarantine labels only;
                                # the unchanged original gets no fresh vote.
                                if clean_committed and any(new.entity_id == entity
                                        and all(getattr(new, f) == getattr(old, f) for f in member_fields)
                                        and new.frame_id > old.frame_id and new.acquired_at_s >= old.acquired_at_s
                                        and (old.media_time_s is None and new.media_time_s is None
                                             or old.media_time_s is not None and new.media_time_s is not None
                                             and new.media_time_s >= old.media_time_s)
                                        and new.household_constraint == "eligible" for new in current_claims):
                                    anchored = True
                                    retain_for_commit(key, before[1])
                                if old.household_state not in {"resident", "visitor", "handoff"}:
                                    continue
                                attenuated = any(new.entity_id == entity
                                    and all(getattr(new, f) == getattr(old, f) for f in member_fields)
                                    and new.frame_id > old.frame_id and new.acquired_at_s >= old.acquired_at_s
                                    and (old.media_time_s is None and new.media_time_s is None
                                         or old.media_time_s is not None and new.media_time_s is not None
                                         and new.media_time_s >= old.media_time_s)
                                    and new.household_state in {"unknown", "provisional"}
                                    and new.household_constraint == "eligible"
                                    for new in current_claims)
                                if attenuated:
                                    anchored = True
                                    retain_for_commit(key, before[1])
                    if not held:
                        key = PhysicalPersonKey(self.producer.run_id, int(entity.rsplit(":", 1)[-1]))
                        after = working.selected_spatial_original(key)
                        selection = selections.get(entity)
                        if selection is None or after is None:
                            complete = False
                        else:
                            dto, ref = selection
                            complete &= (current.get(entity) is dto and ref.stamp == after[1]
                                and ref.evidence_id == after[0].observation_id
                                and any(e.eligible and e.observation == ref.stamp for e in dto.eligibility))
                    complete &= set(retained_originals[entity]) <= prior_claims | current_claims
                revisions = {dto.association_revision for entity, dto in current.items() if entity in component}
                component_claims = tuple(c for c in all_claims if c.entity_id in component)
                complete &= set(component_claims) <= prior_claims | current_claims
                if anchored and complete and len(revisions) == 1:
                    subjects, sids, _ = attribution_conflict_sets(component_claims)
                    preserved = tuple(dict.fromkeys(c for c in (
                        *(c for entity, rows in retained_originals.items() if entity in component for c in rows),
                        *(c for record in transferred_history for c in (*record.owners, *record.retained_owners)),
                        *rekeyed_history)
                        if c not in component_claims))
                    payload = dict(status="quarantined", purpose="physical_only_quarantine",
                        reason="settled_household_claim_conflict", run_id=self.producer.run_id,
                        association_revision=next(iter(revisions)), subject_ids=subjects, stable_ids=sids,
                        owners=[c.model_dump(mode="json") for c in component_claims],
                        retained_owners=[c.model_dump(mode="json") for c in preserved],
                        component_entity_ids=sorted(component))
                    projected = WorldAttributionQuarantine(conflict_id=WorldAttributionQuarantine.digest(payload), **payload)
                    overlays.update((entity, projected) for entity in visible)
                    continue
            if not matching or any(conflict != matching[0] for conflict in matching):
                if not self._spatial_fatal_logged:
                    # One bounded scalar diagnostic/run; no image/vector/cache
                    # dump and no attempt to infer the unavailable caller tuple.
                    self._spatial_fatal_logged = True
                    _LOGGER.error("Attribution closure refusal %s", json.dumps({
                        "component": sorted(component),
                        "current": [{"entity": entity, "revision": dto.association_revision,
                            "conflict": dto.household_conflict is not None,
                            "owners": [self._attribution_claim(dto.physical_key,
                                SourceObservation(m.observation, m.household)).model_dump(mode="json")
                                for m in dto.members]} for entity, dto in current.items()],
                        "retained": {entity: [c.model_dump(mode="json") for c in rows]
                                     for entity, rows in retained.items()},
                        "authorized": {"revisions": sorted({c.association_revision for c in conflicts}),
                            "owners": [c.model_dump(mode="json") for c in original_owners]},
                    }, allow_nan=False, separators=(",", ":")))
                raise WorldPublicationRefused("retained_attribution_conflict_not_authorized")
            conflict = matching[0]
            # One legitimate owner conflict can describe several disconnected
            # divergent people. The World overlay is component-scoped: keep
            # every ORIGINAL owner in this component, not another person's
            # unrelated claims/contested labels. The owner conflict itself is
            # unchanged on the physical reservation and revalidated at commit.
            owners = tuple(claim for key, original in conflict.owners
                if (claim := self._attribution_claim(key, original)).entity_id in component)
            # A newer owner decision may supersede a claim for arbitration, but
            # must not erase the ORIGINAL claim attached to a held position.
            retained_owners = tuple(dict.fromkeys(c for c in (
                *all_claims, *(claim for rows in retained_originals.values() for claim in rows))
                if c.entity_id in component and c not in owners))
            subjects, stable_ids, _ = attribution_conflict_sets(owners)
            payload = dict(status="quarantined", purpose="physical_only_quarantine", reason=conflict.reason,
                run_id=conflict.run_id, association_revision=conflict.association_revision,
                subject_ids=subjects, stable_ids=stable_ids,
                owners=[c.model_dump(mode="json") for c in owners],
                retained_owners=[c.model_dump(mode="json") for c in retained_owners],
                component_entity_ids=sorted(component))
            quarantine = WorldAttributionQuarantine(conflict_id=WorldAttributionQuarantine.digest(payload), **payload)
            overlays.update((entity, quarantine) for entity in component if entity in retained)
        # Never silently clear a former quarantine because an incoming subset omitted
        # an owner. Every surviving component head requires a genuine newer decision.
        for entity, previous in working.attribution_quarantines().items():
            if entity not in retained or entity in overlays:
                continue
            live = set(previous.component_entity_ids) & retained.keys()
            if any(e not in current or current[e].association_revision <= previous.association_revision for e in live):
                overlays[entity] = previous
        working.set_attribution_quarantines(overlays)

    def _validate_spatial_memberships(self, memberships: tuple[SpatialMembershipDTO, ...], context: SpatialClockContext) -> None:
        try:
            current = self._association_service.spatial_memberships(at_interval=context)
        except ValueError as error:
            raise WorldPublicationRefused(str(error)) from error
        originals = {dto.physical_key: dto for dto in memberships}
        if len(originals) != len(memberships) or set(originals) != {dto.physical_key for dto in current}:
            raise WorldPublicationRefused("spatial_final_membership_cohort")
        if any(originals[dto.physical_key] is not dto for dto in current):
            raise WorldPublicationRefused("spatial_membership_revision_changed")

    def _ordinary_observation_reindexes(self, memberships, retired, context, *, prior=None,
                                       published_at_us=None):
        """Exact CURRENT observation identity, not a historical merge permission."""
        if not retired:
            return ()
        if prior is None:
            prior = self._fusion.fork()
            prior.set_spatial_context(context)
        publication_us = max(1, int(self._clock_us())) if published_at_us is None else published_at_us
        result = []
        for origin in sorted(retired):
            original = prior.committed_spatial_original(origin, published_at_us=publication_us)
            if original is None:
                continue
            envelope, stamp, _, _ = original
            matches = [d for d in memberships if d.physical_key != origin
                and origin in d.retired_keys and d.association_basis == "estimated_semantic_joint_correspondence"
                and d.estimated_split_custody is None and len(d.members) == 2 and len(d.eligibility) == 2
                and sum(m.observation == stamp for m in d.members) == 1
                and all(e.eligible and e.lifecycle_receipt is not None
                    and e.lifecycle_receipt.stamp == e.observation
                    and e.lifecycle_receipt.native_state == "ACTIVE"
                    and any(m.observation == e.observation for m in d.members) for e in d.eligibility)]
            if len(matches) != 1:
                continue
            dto, = matches
            prior.prune_expired_spatial_original(dto.physical_key,
                published_at_us=publication_us)
            if prior.has_spatial_original_slot(dto.physical_key):
                continue  # A competing committed point is not an unmeasured destination.
            active = self._spatial_bindings().get(stamp.member.source_id)
            world = envelope.payload.world
            if (type(active) is not ScenePriorFrameBinding or world is None
                    or (active.source_frame.revision,active.target_frame.frame_id,
                        active.target_frame.revision,active.target_from_source_sha256) !=
                       (world.calibration_revision,world.world_frame,
                        world.world_frame_revision,world.world_transform_sha256)):
                raise WorldPublicationRefused("spatial_source_binding_changed")
            result.append((origin, dto, original))
        # Do not choose arbitrarily between two retired originals.
        return tuple(r for r in result if sum(x[1].physical_key == r[1].physical_key for x in result) == 1)

    @staticmethod
    def _same_rekeyed_original_claim(old, new):
        return (type(old) is WorldAttributionClaim and type(new) is WorldAttributionClaim
            and old.physical_serial != new.physical_serial
            and all(getattr(old, field) == getattr(new, field)
                for field in WorldAttributionClaim.model_fields if field != "physical_serial"))

    @staticmethod
    def _same_forward_rekeyed_native_claim(old, new):
        """Label provenance only: one native life advances under a new key.

        The old claim stays historical. This cannot grant either position or
        household identity, and is used only with unchanged committed custody.
        """
        changing = {"physical_serial", "frame_id", "observation_id", "acquired_at_s", "media_time_s"}
        return (type(old) is WorldAttributionClaim and type(new) is WorldAttributionClaim
            and old.physical_serial != new.physical_serial
            and all(getattr(old, field) == getattr(new, field)
                for field in WorldAttributionClaim.model_fields if field not in changing)
            and new.frame_id > old.frame_id and new.observation_id != old.observation_id
            and new.acquired_at_s > old.acquired_at_s
            and old.media_time_s is not None and new.media_time_s is not None
            and new.media_time_s > old.media_time_s)

    def _capture_attributed_split_original(self, fusion, episode, origin, successors, *, published_at_us):
        custody = fusion.capture_split_custody(episode, origin, successors,
            published_at_us=published_at_us)
        if custody is not None and custody.entity.attribution_quarantine is None:
            # Fusion's cached geometric entity precedes its public label overlay.
            # Preserve ONLY the actual committed overlay of this SAME original.
            previous = self._previous_entities.get(GlobalWorldFusion.physical_entity_id(origin))
            if (previous is not None and previous.attribution_quarantine is not None
                    and fusion.attribution_quarantines().get(previous.entity_id) == previous.attribution_quarantine
                    and previous.model_copy(update={'subject': custody.entity.subject,
                        'attribution_quarantine': None}) == custody.entity):
                custody = replace(custody, entity=previous)
        return custody

    def _split_custody_views(self, memberships, mode, *, committing=False, observed_checks=None):
        """Exact owner-issued hold-only scopes, never measurement authority."""
        values = [getattr(d, "estimated_split_custody", None) for d in memberships]
        if not any(v is not None for v in values):
            return ()
        from noesis.identity_association_service import EstimatedSplitCustodyView
        current = {d.physical_key:d for d in memberships}
        views = {}
        occupied = set()
        now = time.monotonic()
        for dto, view in zip(memberships, values):
            if view is None:
                continue
            if (type(view) is not EstimatedSplitCustodyView or mode != "physical_only_quarantine"
                    or type(view.episode_id) is not str or not 1 <= len(view.episode_id) <= 256
                    or type(view.owner_deadline_monotonic_s) not in (int,float)
                    or not math.isfinite(view.owner_deadline_monotonic_s)
                    or len(view.successor_keys) != 2 or len(set(view.successor_keys)) != 2
                    or view.origin_key not in view.successor_keys
                    or dto.physical_key not in view.successor_keys
                    or view.origin_key.run_id != dto.run_id
                    or type(view.split_revision) is not int or view.split_revision < 0):
                raise WorldPublicationRefused("split_custody_view_invalid")
            if observed_checks is not None:
                try:observed_checks[view.episode_id]=(now,now >= view.owner_deadline_monotonic_s)
                except Exception:pass  # Optional observer cannot change authority.
            if now >= view.owner_deadline_monotonic_s:
                if committing:
                    raise WorldPublicationRefused("split_custody_owner_expired")
                continue  # An expired optional lease cannot kill other people's publication.
            if view.episode_id in views:
                if views[view.episode_id] is not view:
                    raise WorldPublicationRefused("split_custody_episode_changed")
                continue
            if occupied.intersection(view.successor_keys):
                raise WorldPublicationRefused("split_custody_overlap")
            occupied.update(view.successor_keys)
            rejoined=view.rejoined_key is not None
            keys=(view.rejoined_key,) if rejoined else view.successor_keys
            if rejoined and (view.rejoined_key!=view.origin_key or type(view.rejoin_revision) is not int
                    # Ledger event revision and service DTO revision are
                    # distinct counters; current DTO identity is checked above.
                    or not view.split_revision<=view.rejoin_revision
                    or len(view.rejoined_stamps)!=2
                    or {s.member for s in view.rejoined_stamps}!={s.member for s in view.prior_seed}):
                raise WorldPublicationRefused('split_custody_rejoin_binding')
            if any(k not in current or current[k].estimated_split_custody is not view for k in keys):
                raise WorldPublicationRefused("split_custody_complete_cohort_required")
            if view.native_destinations:
                if (type(view.native_destinations) is not tuple or len(view.native_destinations)!=2
                        or any(type(row) is not tuple or len(row)!=2 or type(row[0]) is not ObservationStamp
                            or type(row[1]) is not PhysicalPersonKey for row in view.native_destinations)
                        or tuple(s for s,k in view.native_destinations)!=view.split_stamps
                        or {k for s,k in view.native_destinations}!=set(view.successor_keys)
                        or not rejoined and any(k not in current or not any(
                            m.observation.member==s.member and m.observation.lifecycle_started_frame==s.lifecycle_started_frame
                            for m in current[k].members) for s,k in view.native_destinations)):
                    raise WorldPublicationRefused('split_custody_original_binding')
            members = {m.observation.member:m.observation
                       for k in keys for m in current[k].members}
            if (len(members) != 2 or len(view.prior_seed) != 2 or len(view.split_stamps) != 2
                    or {s.member for s in view.prior_seed} != set(members)
                    or {s.member for s in view.split_stamps} != set(members)):
                raise WorldPublicationRefused("split_custody_original_members_changed")
            for key in keys:
                for eligibility in current[key].eligibility:
                    stamp = eligibility.observation
                    receipt = eligibility.lifecycle_receipt
                    if (members.get(stamp.member) != stamp or receipt is None
                            or receipt.stamp != stamp or receipt.native_state != "ACTIVE"):
                        raise WorldPublicationRefused("split_custody_native_required")
                    if rejoined:
                        joined=next(s for s in view.rejoined_stamps if s.member==stamp.member)
                        if (joined.lifecycle_started_frame!=stamp.lifecycle_started_frame
                                or joined.frame_id>stamp.frame_id or joined.acquired_at_s>stamp.acquired_at_s
                                or joined.clock_domain!=stamp.clock_domain or joined.clock_basis!=stamp.clock_basis):
                            raise WorldPublicationRefused('split_custody_rejoin_binding')
            views[view.episode_id] = view
        return tuple(views.values())

    def _split_original_destination(self, view, stamp, context):
        try:
            key=self._association_service.estimated_split_original_destination(view,stamp,at_interval=context)
        except (ValueError,TypeError,StopIteration) as exc:
            raise WorldPublicationRefused('split_custody_original_binding') from exc
        if key not in view.successor_keys:
            raise WorldPublicationRefused('split_custody_original_binding')
        return key

    @staticmethod
    def _validate_split_original(view, stamp):
        split = next((s for s in view.split_stamps if s.member == stamp.member), None)
        seed = next((s for s in view.prior_seed if s.member == stamp.member), None)
        if (split is None or seed is None or stamp.member.run_id != view.origin_key.run_id
                or (stamp.lifecycle_started_frame, stamp.clock_domain, stamp.clock_basis) !=
                   (split.lifecycle_started_frame, split.clock_domain, split.clock_basis)
                or seed.lifecycle_started_frame != stamp.lifecycle_started_frame
                or stamp.frame_id > split.frame_id or stamp.acquired_at_s > split.acquired_at_s
                or stamp.media_time_s is None or split.media_time_s is None
                or stamp.media_time_s > split.media_time_s):
            raise WorldPublicationRefused("split_custody_original_binding")

    def _conflict_custody_original(self, fusion, dto):
        """Attenuate an exact committed point; never grant conflicted originals.

        Caller first verifies the actual final owner-issued DTO by identity.
        The owner alone qualifies the task-opt-in seed/native mixed-cue policy.
        World independently verifies its original point, lifecycle and binding.
        """
        if (dto.association_basis != "estimated_image_region_conflict_custody"
                or len(dto.members) != 2 or len(dto.eligibility) != 2):
            raise WorldPublicationRefused("conflict_custody_membership_invalid")
        members = {m.observation.member:m for m in dto.members}
        if (len(members) != 2 or {e.observation.member for e in dto.eligibility} != set(members)
                or any(e.eligible or e.reason != "image_region_contradictory_cue_member"
                    or e.observation != members[e.observation.member].observation
                    or e.lifecycle_receipt is None
                    or e.lifecycle_receipt.stamp != e.observation
                    or e.lifecycle_receipt.native_state != "ACTIVE" for e in dto.eligibility)):
            raise WorldPublicationRefused("conflict_custody_membership_invalid")
        custody = fusion.committed_spatial_original(dto.physical_key,
            published_at_us=max(1,int(self._clock_us())))
        if custody is None:
            return None  # No committed point or original source/wire TTL expired.
        original, stamp, _, _ = custody
        current = members.get(stamp.member)
        if (current is None or current.observation.lifecycle_started_frame != stamp.lifecycle_started_frame
                or current.observation.frame_id < stamp.frame_id
                or current.observation.acquired_at_s < stamp.acquired_at_s
                or current.observation.clock_domain != stamp.clock_domain
                or current.observation.clock_basis != stamp.clock_basis
                or (stamp.media_time_s is not None and (current.observation.media_time_s is None
                    or current.observation.media_time_s < stamp.media_time_s))):
            raise WorldPublicationRefused("committed_spatial_hold_custody_changed")
        frames = self._spatial_bindings() if callable(self._spatial_bindings) else {}
        active = frames.get(stamp.member.source_id)
        world = original.payload.world
        if (type(active) is not ScenePriorFrameBinding or world is None
                or (active.source_frame.revision,active.target_frame.frame_id,
                    active.target_frame.revision,active.target_from_source_sha256) !=
                   (world.calibration_revision,world.world_frame,
                    world.world_frame_revision,world.world_transform_sha256)):
            raise WorldPublicationRefused("spatial_source_binding_changed")
        return custody

    def _validated_spatial_bindings(self, observations, memberships, artifacts):
        if not observations:
            return ()  # Empty removal/expiry barrier needs no new position binding.
        if not callable(self._spatial_bindings):
            raise WorldPublicationRefused("spatial_source_bindings_unavailable")
        frames = self._spatial_bindings()
        if not isinstance(frames, Mapping):
            raise WorldPublicationRefused("spatial_source_bindings_unavailable")
        result = {}
        sources = {}
        for dto in memberships:
            for binding in dto.source_bindings:
                sources.setdefault(binding.source_id, set()).add(binding)
        for original in observations:
            world = original.payload.world
            if world is None:
                continue
            source = original.payload.tracklet.source_id
            binding = frames.get(source)
            source_bindings = sources.get(source, set())
            if (type(binding) is not ScenePriorFrameBinding or len(source_bindings) != 1
                    or binding.source_frame.frame_id != world.world_frame
                    or binding.target_frame.frame_id != world.world_frame
                    or (binding.target_frame.revision, binding.target_from_source_sha256, binding.source_frame.revision)
                    != (world.world_frame_revision, world.world_transform_sha256, world.calibration_revision)
                    or binding.source_camera_calibration_sha256 != artifacts[source].camera_calibration_sha256):
                raise WorldPublicationRefused("spatial_original_frame_binding")
            active, = source_bindings
            if (active.camera_id, active.calibration_revision, active.transform_id) != (
                original.payload.tracklet.camera_id, world.calibration_revision, world.world_transform_sha256,
            ):
                raise WorldPublicationRefused("spatial_original_source_binding")
            result[source] = binding
        return tuple(sorted(result.items()))

    def _spatial_measurements_locked(
        self, source_id: int, measurements: tuple[PreparedWorldPublication, ...],
    ) -> tuple[tuple[ObservationEnvelope, float | None, str | None], ...]:
        """Bounded non-authoritative staging of ALL original ordered receipts."""
        if self._closed:
            raise RuntimeError("canonical world service is closed")
        if any(state.association_candidate for _, state in self._prepared_candidates.values()):
            raise RuntimeError("an ordered physical cohort is already pending")
        if type(source_id) is not int or type(measurements) is not tuple or not 1 <= len(measurements) <= 3:
            raise ValueError("one to three exact spatial source preparations required")
        originals: list[_PreparedWorldState] = []
        for receipt in measurements:
            self._validate_prepared_owner(receipt)
            entry = self._prepared_candidates.get(receipt._candidate_id)
            if (entry is None or entry[0] is not receipt or entry[1].association_candidate
                    or receipt.base_revision != self._revision):
                raise RuntimeError("original World measurement preparation is stale or consumed")
            originals.append(entry[1])
        if len({state.source_id for state in originals}) != len(originals):
            raise ValueError("one exact source frame per World spatial cohort required")
        if not any(state.source_id == source_id for state in originals):
            raise ValueError("ordered trigger source must own an original preparation")
        if sum(len(state.publication.observations) for state in originals) > 2 * min(64, self._fusion.config.max_entities):
            raise ValueError("spatial observation cohort exceeds the existing entity/member bound")
        rows = tuple(
            (original, *state.spatial_quality[original.observation_id])
            for state in originals for original in state.publication.observations
        )
        if len({row[0].observation_id for row in rows}) != len(rows):
            raise ValueError("duplicate World measurement identity")
        for state in originals:
            if any(original.observation_id not in state.physical_members
                   or original.observation_id not in state.physical_stamps
                   for original in state.publication.observations):
                raise ValueError("original native epoch/lifecycle observation required")
        return rows

    def prepare_physical(
        self,
        source_id: int,
        measurements: tuple[PreparedWorldPublication, ...],
        associations: tuple[tuple[AssociationDTO, MeasurementCohort], ...],
        *,
        at: AcquisitionTime,
        withdrawn_members: tuple[MemberKey, ...] = (),
        retired_keys: tuple[PhysicalPersonKey, ...] = (),
    ) -> PreparedWorldPublication:
        """Stage ONE gated World transaction from exact public service grants.

        measurements are this service's original prepare() receipts, never
        reconstructed envelopes/positions. The same actor first finishes all
        Identity mutations, then reissues DTOs by each ORIGINAL operation (link
        or native transition), accumulating every exact withdrawal. No later
        media mutation may interleave. All grants are revalidated at commit.
        """
        with self._lock:
            self._association_actor()
            if self._closed:
                raise RuntimeError("canonical world service is closed")
            if any(state.association_candidate for _, state in self._prepared_candidates.values()):
                raise RuntimeError("an ordered physical cohort is already pending")
            if (type(measurements) is not tuple or not 1 <= len(measurements) <= 3
                    or type(associations) is not tuple
                    or len(associations) > min(64, self._fusion.config.max_entities)
                    or type(at) is not AcquisitionTime
                    or type(withdrawn_members) is not tuple or len(withdrawn_members) > 128
                    or any(type(m) is not MemberKey for m in withdrawn_members)
                    or type(retired_keys) is not tuple or len(retired_keys) > 64
                    or any(type(k) is not PhysicalPersonKey for k in retired_keys)):
                raise ValueError("bounded exact physical cohort required")
            originals: list[_PreparedWorldState] = []
            for receipt in measurements:
                self._validate_prepared_owner(receipt)
                entry = self._prepared_candidates.get(receipt._candidate_id)
                if (entry is None or entry[0] is not receipt or entry[1].association_candidate
                        or receipt.base_revision != self._revision):
                    raise RuntimeError("original World measurement preparation is stale or consumed")
                originals.append(entry[1])
            if len({state.source_id for state in originals}) != len(originals):
                raise ValueError("one exact source frame per World physical cohort required")
            trigger = next((state for state in originals if state.source_id == source_id), None)
            if trigger is None:
                raise ValueError("ordered trigger source must own an original preparation")
            observations = tuple(o for state in originals for o in state.publication.observations)
            if len(observations) > 2 * min(64, self._fusion.config.max_entities):
                raise ValueError("physical observation cohort exceeds the existing entity/member bound")
            by_id = {o.observation_id: o for o in observations}
            if len(by_id) != len(observations):
                raise ValueError("duplicate World measurement identity")
            contexts = {key: value for state in originals for key, value in state.physical_members.items()}
            stamps = {key: value for state in originals for key, value in state.physical_stamps.items()}
            artifacts = tuple((state.source_id, state.artifact_metadata,
                               self._prepared_artifacts(state)) for state in originals)
            bindings = self._validated_physical_bindings(associations, observations)
            for source, _, original_artifacts in artifacts:
                active = next((b for s, b, _ in bindings if s == source), None)
                if active is not None and original_artifacts.camera_calibration_sha256 != active.source_camera_calibration_sha256:
                    raise ValueError("original active camera calibration fingerprint mismatch")
            working = self._fusion.fork()
            # Exact old contributors go first, including original-operation
            # withdrawals that a final replay DTO no longer repeats.
            removals = set(withdrawn_members)
            retired = set(retired_keys)
            seen_keys: set[PhysicalPersonKey] = set()
            seen_members: set[MemberKey] = set()
            used: set[str] = set()
            reservations: list[AssociationReservation] = []
            try:
                for dto, cohort in associations:
                    if (type(dto) is not AssociationDTO or type(cohort) is not MeasurementCohort
                            or dto.physical_key in seen_keys or dto.run_id != self.producer.run_id):
                        raise ValueError("distinct current run-owned physical DTOs required")
                    seen_keys.add(dto.physical_key)
                    removals.update(dto.withdrawn_members)
                    retired.update(dto.retired_keys)
                    positions = []
                    for reference in cohort.measurements:
                        original = by_id.get(reference.stamp.observation_id)
                        if original is None or original.observation_id in used:
                            raise ValueError("exact original World observation missing or reused")
                        context = contexts.get(original.observation_id)
                        if (context is None or context[:2] != (reference.stamp.member, reference.stamp.lifecycle_started_frame)
                                or (original.observation_id in stamps and stamps[original.observation_id] != reference.stamp)
                                or reference.stamp.member in seen_members):
                            raise ValueError("original epoch/lifecycle member context mismatch")
                        self._validate_physical_measurement(dto, reference, original, bindings)
                        positions.append(original.payload.world.position)
                        used.add(original.observation_id)
                        seen_members.add(reference.stamp.member)
                    if len(positions) == 2 and math.dist(
                        (positions[0].x, positions[0].y, positions[0].z),
                        (positions[1].x, positions[1].y, positions[1].z),
                    ) > min(dto.policy.max_distance_m, self._fusion.config.conflict_distance_m):
                        raise ValueError("actual original World measurements exceed the admitted separation gate")
                    result = self._association_service.prepare(dto, cohort, at)
                    if result.disposition != "reserved" or result.reservation is None:
                        raise WorldPublicationRefused.from_association_result(result)
                    reservations.append(result.reservation)
                if any(o.observation_id not in used and o.payload.world is not None
                       and o.payload.world.quality != "held"
                       and o.payload.world.source not in {"anchor_hold", "cv_prediction", "image_motion_prediction"}
                       for o in observations):
                    raise ValueError("every new measured World observation requires its exact association grant")
                for member in sorted(removals):
                    working.withdraw_physical_member(member)
                for key in sorted(retired):
                    working.retire_physical_key(key)
                ingestions = []
                for dto, cohort in associations:
                    households = {m.observation.member: m.household for m in dto.members}
                    for reference in cohort.measurements:
                        original = by_id[reference.stamp.observation_id]
                        member, _, trail, broken = contexts[original.observation_id]
                        house = households[member]
                        subject = self._subject(original, {
                            "identity_kind": house.state,
                            "resident_uuid": house.identity_key if house.state == "resident" else None,
                            "stable_id": house.stable_id,
                            "visitor_generation": house.visitor_generation,
                        })
                        ingestions.append((original, subject, dto.physical_key, member, trail, broken))
                # DTO/person ordering is not measurement ordering. Preserve
                # each source's ORIGINAL World sequence even for two persons.
                for original, subject, key, member, trail, broken in sorted(
                    ingestions, key=lambda row: (row[0].payload.tracklet.source_id, row[0].sequence)
                ):
                    working.ingest(original, subject, physical_key=key,
                                   physical_member=member, source_epoch=member.source_epoch,
                                   tracker_lifecycle_generation=member.lifecycle_generation,
                                   trail_segment_id=trail, trail_break_required=broken)
                sequences = dict(self._sequence_by_source)
                for state in originals:
                    for source, sequence in state.sequence_by_source.items():
                        sequences[source] = max(sequence, sequences.get(source, 0))
                now_us = max(1, int(self._clock_us()))
                snapshot = _physical_snapshot(working, now_us)
                outputs = dict(self._canonical_track_outputs)
                for member in removals:
                    for cache_key in tuple(outputs):
                        if (cache_key[:4] == (member.source_id, member.camera_id, member.raw_id,
                                              member.lifecycle_generation)
                                and cache_key[7] == member.source_epoch):
                            outputs.pop(cache_key, None)
                admitted_ids = {s.observation_id for e in snapshot.entities for s in e.sources
                                if s.accepted or s.rejection_reason == "non_authoritative_held_continuation"}
                for state in originals:
                    for key, output in state.canonical_track_outputs.items():
                        origin = output.presence_origin
                        if origin is not None and origin.envelope.observation_id in used & admitted_ids:
                            outputs[key] = output
                self._bound_track_outputs(outputs)
                receipt = self._stage_candidate_locked(
                    trigger.publication.observations, snapshot, working, sequences, outputs,
                    source_id, trigger.artifact_metadata, self._prepared_artifacts(trigger),
                    association_candidate=True, association_reservations=tuple(reservations),
                    association_bindings=bindings, association_artifacts=artifacts,
                    physical_observations=observations,
                )
            except Exception:
                self._discard_association_reservations(tuple(reservations))
                for original in measurements:
                    self._prepared_candidates.pop(original._candidate_id, None)
                raise
            for original in measurements:
                self._prepared_candidates.pop(original._candidate_id, None)
            return receipt

    @staticmethod
    def _prepared_artifacts(state: _PreparedWorldState) -> WorldArtifacts:
        if state.resolved_artifacts is None:
            raise RuntimeError("original resolved World artifacts unavailable")
        return state.resolved_artifacts

    def _validated_physical_bindings(
        self, associations: tuple[tuple[AssociationDTO, MeasurementCohort], ...],
        observations: tuple[ObservationEnvelope, ...],
    ) -> tuple[tuple[int, ScenePriorFrameBinding, tuple[int, int]], ...]:
        if not associations:
            if any(o.payload.world is not None and o.payload.world.quality != "held"
                   and o.payload.world.source not in {"anchor_hold", "cv_prediction", "image_motion_prediction"}
                   for o in observations):
                raise ValueError("new physical observations require current service grants")
            return ()  # Exact removal/expiry barriers do not bootstrap geometry.
        if not all(callable(p) for p in (self._physical_admission, self._physical_bindings, self._physical_rasters)):
            raise RuntimeError("physical metric/time/independent admission unavailable")
        admission = self._physical_admission()
        if (type(admission) is not ProducerAdmission or admission.status != "admitted"
                or admission.run_id != self.producer.run_id
                or not all(self._bounded_text(v, maximum=256) for v in (
                    admission.metric_receipt_id, admission.acquisition_receipt_id,
                    admission.independent_source_receipt_id))):
            raise RuntimeError("physical metric/time/independent admission unavailable")
        frames = self._physical_bindings()
        rasters = self._physical_rasters()
        if not isinstance(frames, Mapping) or not isinstance(rasters, Mapping) or not 1 <= len(frames) <= 3:
            raise RuntimeError("physically admitted common metric bindings unavailable")
        common = None
        accepted: dict[int, tuple[int, ScenePriorFrameBinding, tuple[int, int]]] = {}
        revisions = set()
        for dto, cohort in associations:
            if (type(dto) is not AssociationDTO or type(cohort) is not MeasurementCohort
                    or dto.producer_admission is not admission
                    or dto.policy_identity != admission.policy_identity):
                raise ValueError("exact current producer admission required")
            revisions.add(dto.association_revision)
            for reference in cohort.measurements:
                source = reference.stamp.member.source_id
                binding = frames.get(source)
                raster = rasters.get(source)
                if (type(binding) is not ScenePriorFrameBinding or binding.contract_version != 2
                        or type(raster) is not tuple or len(raster) != 2
                        or any(type(n) is not int or n <= 0 for n in raster)):
                    raise RuntimeError("physically admitted common metric/raster binding unavailable")
                # Existing contracts verify revision hash, proper rigid edge and
                # exact source/target floor (unchanged 0.01m plane tolerance).
                ScenePriorFrameBinding.model_validate(binding.model_dump(mode="python"))
                if binding.metric_frame is None or (common is not None and binding.metric_frame != common):
                    raise ValueError("different physically admitted common metric frame/floor")
                common = binding.metric_frame
                active = next((s for s in dto.policy.sources if s.source_id == source), None)
                if (active is None or active.camera_id != reference.stamp.member.camera_id
                        or active.source_epoch != reference.stamp.member.source_epoch
                        or active.calibration_revision != binding.source_frame.revision
                        or active.transform_id != binding.target_from_source_sha256
                        or dto.policy.target_frame != binding.target_frame.frame_id
                        or dto.policy.world_revision != binding.target_frame.revision):
                    raise ValueError("active metric/source/calibration/epoch binding mismatch")
                accepted[source] = (source, binding, raster)
        if len(revisions) != 1:
            raise ValueError("all physical persons require final same-revision DTO issuance")
        return tuple(accepted[s] for s in sorted(accepted))

    def _validate_physical_measurement(self, dto, reference, original, bindings) -> None:
        member = reference.stamp.member
        _, binding, raster = next(row for row in bindings if row[0] == member.source_id)
        t = original.payload.tracklet
        w = original.payload.world
        source = next(s for s in dto.policy.sources if s.source_id == member.source_id)
        measured_sources = {
            "bbox3d", "pose_floor_only", "person_anchor_floor_only",
            "pose_depth_only", "person_anchor_depth_only",
            "pose_depth_fused", "person_anchor_depth_fused",
        }
        d = original.payload.world_diagnostics
        if (original.producer != self.producer or member.run_id != self.producer.run_id
                or (t.camera_id, t.source_id, t.tracker_id, t.frame_id)
                != (member.camera_id, member.source_id, member.raw_id, reference.stamp.frame_id)
                or original.payload.image_size != raster
                or original.capture_time_status != "synced"
                or original.captured_at_us != round(reference.stamp.acquired_at_s * 1_000_000)
                or reference.stamp.clock_domain != source.clock_domain
                or reference.stamp.clock_basis != source.clock_basis
                or reference.stamp.media_time_s is None or original.media_pts_ns is None
                or abs(original.media_pts_ns - round(reference.stamp.media_time_s * 1_000_000_000)) > 1
                or w is None or w.quality == "held" or w.source not in measured_sources
                or d is None or d.measurement_accepted is not True
                or reference.origin != "measured" or reference.units != "m"
                or not self._bounded_text(reference.independence_id, maximum=256)
                or w.world_frame != "backend_world_m" or w.units != "meters"
                or w.world_frame_revision != binding.target_frame.revision
                or w.world_transform_sha256 != binding.target_from_source_sha256
                or w.calibration_revision != source.calibration_revision):
            raise ValueError("original independent measured World/time/raster/source evidence mismatch")

    def _prepare_locked(
        self,
        source_id: int,
        tracks: list[Mapping[str, Any]],
        *,
        metadata: Mapping[str, Any],
        physical_observations: tuple[SourceObservation, ...] = (),
    ) -> PreparedWorldPublication:
        if self._association_service is not None:
            self._association_actor()
            if any(state.association_candidate for _, state in self._prepared_candidates.values()):
                raise RuntimeError("later work must queue behind the pending ordered physical cohort")
        if self._closed:
            raise RuntimeError("canonical world service is closed")
        if len(self._prepared_candidates) >= int(
            self.MAX_PREPARED_CANDIDATES
        ):
            raise RuntimeError(
                "canonical world preparation capacity is exhausted"
            )
        source = int(source_id)
        if (type(physical_observations) is not tuple or len(physical_observations) > 128
                or any(type(o) is not SourceObservation or type(o.stamp) is not ObservationStamp
                       or type(o.stamp.member) is not MemberKey for o in physical_observations)
                or (physical_observations and self._association_service is None)):
            raise ValueError("original typed source observations require the gated consumer")
        native_records = {(o.stamp.member.raw_id, o.stamp.frame_id): o for o in physical_observations}
        if (len(native_records) != len(physical_observations)
                or len({o.stamp.observation_id for o in physical_observations}) != len(physical_observations)):
            raise ValueError("duplicate original native observation")
        used_native_records: set[tuple[int, int]] = set()
        artifacts = (
            self._artifacts(source, metadata)
            if callable(self._artifacts)
            else self._artifacts
        )
        now_us = max(1, int(self._clock_us()))
        working_fusion = self._fusion.fork()
        working_sequences = dict(self._sequence_by_source)
        working_track_outputs = dict(self._canonical_track_outputs)
        source_media_pts: list[int] = []
        for value in (
            metadata.get("media_pts_ns"),
            *(track.get("media_pts_ns") for track in tracks),
        ):
            try:
                parsed_pts = int(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if parsed_pts > 0:
                source_media_pts.append(parsed_pts)
        current_source_media_pts_ns = (
            max(source_media_pts) if source_media_pts else None
        )
        # Expire source-local authority before it can validate this cohort.
        # Pruning only after observation construction admitted one extra stale
        # continuation at the end of a long gap.
        self._prune_expired_track_outputs(
            working_track_outputs,
            source_id=source,
            current_media_pts_ns=current_source_media_pts_ns,
        )
        shadow_keys, terminated_ids = parse_tracker_presence_evidence(
            metadata.get("tracker_lifecycle_evidence"),
            source_id=source,
            metadata=metadata,
        )
        camera_id = str(metadata.get("camera_id") or "").strip()
        calibration_sha256 = self._sha256_text(artifacts.camera_calibration_sha256) or artifacts.calibration.sha256
        source_epoch = self._nonnegative_int(metadata.get("source_epoch")) or 0
        retained_keys: set[tuple[int, int, int]] = set()
        if shadow_keys:
            for key, output in working_track_outputs.items():
                origin = output.presence_origin
                if (
                    key[0] == source and key[1] == camera_id
                    and (key[2], key[3]) in shadow_keys
                    and key[6] == calibration_sha256 and key[7] == source_epoch
                    and key[2] not in terminated_ids and origin is not None
                    and working_fusion._retain_admitted_tracker_origin(
                        origin.envelope, origin.subject,
                        tracker_lifecycle_generation=origin.tracker_lifecycle_generation,
                        source_epoch=origin.source_epoch,
                        trail_segment_id=origin.trail_segment_id,
                        trail_break_required=origin.trail_break_required,
                    )
                ):
                    retained_keys.add((key[2], key[3], key[7]))
        retained_lifecycles = frozenset(retained_keys)
        # Termination invalidates every process origin for that physical key.
        # Presence receipts never update these origins or observation clocks.
        if terminated_ids:
            for key in tuple(working_track_outputs):
                if key[0] == source and key[1] == camera_id and key[2] in terminated_ids:
                    working_track_outputs.pop(key, None)
        observations: list[ObservationEnvelope] = []
        physical_members: dict[str, tuple[MemberKey, int, int | None, bool]] = {}
        physical_stamps: dict[str, ObservationStamp] = {}
        spatial_quality: dict[str, tuple[float | None, str | None]] = {}
        spatial_upright_support: dict[str, PreparedUprightFootprintSupport] = {}
        spatial_input_diagnostics: dict[str, Mapping[str, Any]] = {}
        pending_presence_origins: dict[str, _TrackerPresenceOrigin] = {}
        pending_track_outputs: dict[
            str,
            tuple[
                tuple[int, str, int, int, str, str, str, int],
                _CanonicalTrackOutput,
            ],
        ] = {}
        for track in tracks:
            continuity_key = self._canonical_track_output_key(
                source,
                track,
                metadata=metadata,
                calibration_sha256=(
                    self._sha256_text(
                        artifacts.camera_calibration_sha256
                    )
                    or artifacts.calibration.sha256
                ),
            )
            expected_continuity_origin = None
            if continuity_key is not None:
                self._prune_superseded_track_outputs(
                    working_track_outputs,
                    continuity_key,
                )
                # Refresh an exact-current origin in insertion order so the
                # hard cap evicts inactive history before a live lifecycle.
                expected_continuity_origin = working_track_outputs.pop(
                    continuity_key,
                    None,
                )
                if expected_continuity_origin is not None:
                    working_track_outputs[continuity_key] = (
                        expected_continuity_origin
                    )
            observation = self._observation(
                source,
                track,
                metadata=metadata,
                artifacts=artifacts,
                published_at_us=now_us,
                sequence_by_source=working_sequences,
                expected_image_motion_origin=expected_continuity_origin,
            )
            if observation is None:
                continue
            native = native_records.get((observation.payload.tracklet.tracker_id, observation.payload.tracklet.frame_id))
            if native is not None:
                t, stamp = observation.payload.tracklet, native.stamp
                if (stamp.member.run_id != self.producer.run_id
                        or (stamp.member.camera_id, stamp.member.source_id, stamp.member.raw_id)
                        != (t.camera_id, t.source_id, t.tracker_id)):
                    raise ValueError("original native source observation does not match normalized row")
                # Retain the producer's ORIGINAL token; never rename its proof
                # or extraction to the World's implementation-local counter.
                observation = ObservationEnvelope.model_validate({
                    **observation.model_dump(mode="python"), "observation_id": stamp.observation_id,
                })
                physical_stamps[observation.observation_id] = stamp
                used_native_records.add((t.tracker_id, t.frame_id))
            observations.append(observation)
            if self._association_service is not None:
                resolver_confidence, quality_rejection, raw_summary = self._spatial_input_quality(track, observation)
                spatial_quality[observation.observation_id] = (resolver_confidence, quality_rejection)
                # This compact descriptor distinguishes an admitted typed
                # body-plane footprint from untyped learned-height gravity.
                # It does not change the UNKNOWN support or re-run estimation.
                if (quality_rejection is None and native is not None
                        and ((track.get("world_resolver_selected_id") == "upright_pose_projection"
                              and track.get("world_resolver_contact_basis") == "pose:upright_body_planes"
                              and track.get("world_posture") == "standing")
                             or (track.get("world_resolver_selected_id") == "registered_body_column"
                                 and track.get("world_resolver_contact_basis") == "depth:registered_body_column"
                                 and track.get("world_source") == "registered_body_column"
                                 and track.get("world_quality") == "estimated"
                                 and track.get("world_posture") in {"standing", "sitting", "lying", "unknown"}))
                        and track.get("world_support_state") == "unknown"
                        and track.get("frame_id") == observation.payload.tracklet.frame_id
                        and track.get("observed_at_us") == observation.observed_at_us):
                    support = PreparedUprightFootprintSupport(observation, native.stamp,
                        track["world_resolver_selected_id"], track["world_resolver_contact_basis"])
                    if support.matches(observation):
                        spatial_upright_support[observation.observation_id] = support
                spatial_input_diagnostics[observation.observation_id] = raw_summary
                epoch = track.get("source_epoch", metadata.get("source_epoch"))
                generation = track.get("tracker_lifecycle_generation")
                started = track.get("lifecycle_started_frame")
                if (all(type(v) is int and v >= 0 for v in (epoch, generation, started))
                        and started <= observation.payload.tracklet.frame_id
                        and (metadata.get("source_epoch") is None or metadata["source_epoch"] == epoch)):
                    physical_members[observation.observation_id] = (
                        MemberKey(self.producer.run_id, observation.payload.tracklet.camera_id,
                                  source, epoch, observation.payload.tracklet.tracker_id, generation),
                        started, self._nonnegative_int(track.get("trail_segment_id")),
                        track.get("trail_break_required") is True,
                    )
            if continuity_key is not None and observation.payload.world is not None:
                position = observation.payload.world.position
                try:
                    media_pts_ns = int(observation.media_pts_ns)
                    trail_segment_id = int(track.get("trail_segment_id", 0))
                except (TypeError, ValueError, OverflowError):
                    media_pts_ns = -1
                    trail_segment_id = -1
                if media_pts_ns >= 0 and trail_segment_id >= 0:
                    previous_output = working_track_outputs.get(
                        continuity_key
                    )
                    current_position = (
                        float(position.x),
                        float(position.y),
                        float(position.z),
                    )
                    current_bbox_geometry = self._bbox_geometry(
                        track.get("bbox")
                    )
                    world_source = str(
                        track.get("world_source") or ""
                    ).strip().lower()
                    provenance = track.get("world_prediction_provenance")
                    committed_provenance_type = (
                        str(provenance.get("type") or "").strip()
                        if isinstance(provenance, Mapping)
                        else ""
                    )
                    gain_zero_output_hold = bool(
                        world_source == "anchor_hold"
                        and committed_provenance_type
                        == "bounded_output_hold"
                    )
                    projective_bridge_root_media_pts_ns = (
                        self._projective_bridge_root_for_admitted_output(
                            track,
                            expected_origin=previous_output,
                            current_media_pts_ns=media_pts_ns,
                        )
                    )
                    inferred_ground_episode_root = (
                        self._inferred_ground_episode_root_for_admitted_output(
                            track,
                            expected_origin=previous_output,
                            current_media_pts_ns=media_pts_ns,
                        )
                    )
                    metric_authoritative = bool(
                        world_source
                        not in {
                            "anchor_hold",
                            "cv_prediction",
                            "image_motion_prediction",
                        }
                        and str(
                            track.get("world_quality") or "estimated"
                        ).strip().lower()
                        != "held"
                        and track.get("world_measurement_accepted") is not False
                    )
                    prior_outputs: tuple[_CanonicalOutputOrigin, ...] = ()
                    if (
                        previous_output is not None
                        and int(previous_output.trail_segment_id)
                        == int(trail_segment_id)
                    ):
                        # Queue delay can make a valid descendant refer to an
                        # older committed image root after a newer descendant
                        # has already committed. Retain history for the full
                        # nonrenewing projective episode; individual validators
                        # still apply their narrower lane-specific horizons.
                        history_horizon_ns = int(
                            round(
                                (
                                    self.PROJECTIVE_PROCESS_BRIDGE_HORIZON_S
                                    + self.BOUNDED_PROCESS_HORIZON_EPSILON_S
                                )
                                * 1_000_000_000.0
                            )
                        )
                        prior_outputs = (
                            _CanonicalOutputOrigin(
                                position=tuple(previous_output.position),
                                media_pts_ns=int(previous_output.media_pts_ns),
                                trail_segment_id=int(
                                    previous_output.trail_segment_id
                                ),
                                committed_world_source=(
                                    previous_output.committed_world_source
                                ),
                                support_surface=previous_output.support_surface,
                                committed_provenance_type=(
                                    previous_output.committed_provenance_type
                                ),
                                metric_position=previous_output.metric_position,
                                metric_media_pts_ns=(
                                    previous_output.metric_media_pts_ns
                                ),
                                metric_trail_segment_id=(
                                    previous_output.metric_trail_segment_id
                                ),
                                projective_bridge_root_media_pts_ns=(
                                    previous_output.projective_bridge_root_media_pts_ns
                                ),
                                kinematic_position=(
                                    previous_output.kinematic_position
                                ),
                                kinematic_media_pts_ns=(
                                    previous_output.kinematic_media_pts_ns
                                ),
                                kinematic_trail_segment_id=(
                                    previous_output.kinematic_trail_segment_id
                                ),
                                bbox_geometry=(
                                    previous_output.bbox_geometry
                                ),
                                kinematic_bbox_geometry=(
                                    previous_output.kinematic_bbox_geometry
                                ),
                                inferred_ground_episode_root=(
                                    previous_output.inferred_ground_episode_root
                                ),
                            ),
                            *tuple(previous_output.prior_outputs),
                        )
                        prior_outputs = tuple(
                            origin
                            for origin in prior_outputs
                            if 0
                            <= media_pts_ns - int(origin.media_pts_ns)
                            <= history_horizon_ns
                        )[: int(self.MAX_CANONICAL_TRACK_OUTPUT_HISTORY)]
                    pending_track_outputs[observation.observation_id] = (
                        continuity_key,
                        _CanonicalTrackOutput(
                            position=current_position,
                            media_pts_ns=media_pts_ns,
                            trail_segment_id=trail_segment_id,
                            committed_world_source=world_source,
                            support_surface=observation.payload.world.support_surface,
                            committed_provenance_type=(
                                committed_provenance_type or None
                            ),
                            metric_position=(
                                current_position
                                if metric_authoritative
                                else (
                                    previous_output.metric_position
                                    if previous_output is not None
                                    else None
                                )
                            ),
                            metric_media_pts_ns=(
                                media_pts_ns
                                if metric_authoritative
                                else (
                                    previous_output.metric_media_pts_ns
                                    if previous_output is not None
                                    else None
                                )
                            ),
                            metric_trail_segment_id=(
                                trail_segment_id
                                if metric_authoritative
                                else (
                                    previous_output.metric_trail_segment_id
                                    if previous_output is not None
                                    else None
                                )
                            ),
                            projective_bridge_root_media_pts_ns=(
                                projective_bridge_root_media_pts_ns
                            ),
                            kinematic_position=(
                                (
                                    previous_output.kinematic_position
                                    or previous_output.position
                                )
                                if gain_zero_output_hold
                                and previous_output is not None
                                else current_position
                            ),
                            kinematic_media_pts_ns=(
                                (
                                    previous_output.kinematic_media_pts_ns
                                    if previous_output.kinematic_media_pts_ns
                                    is not None
                                    else previous_output.media_pts_ns
                                )
                                if gain_zero_output_hold
                                and previous_output is not None
                                else media_pts_ns
                            ),
                            kinematic_trail_segment_id=(
                                (
                                    previous_output.kinematic_trail_segment_id
                                    if previous_output.kinematic_trail_segment_id
                                    is not None
                                    else previous_output.trail_segment_id
                                )
                                if gain_zero_output_hold
                                and previous_output is not None
                                else trail_segment_id
                            ),
                            bbox_geometry=current_bbox_geometry,
                            kinematic_bbox_geometry=(
                                (
                                    previous_output.kinematic_bbox_geometry
                                    or previous_output.bbox_geometry
                                )
                                if gain_zero_output_hold
                                and previous_output is not None
                                else current_bbox_geometry
                            ),
                            inferred_ground_episode_root=(
                                inferred_ground_episode_root
                            ),
                            prior_outputs=prior_outputs,
                        ),
                    )
            subject = self._subject(observation, track)
            if observation.observation_id in pending_track_outputs:
                pending_presence_origins[observation.observation_id] = _TrackerPresenceOrigin(
                    envelope=observation, subject=subject,
                    trail_segment_id=self._nonnegative_int(track.get("trail_segment_id")),
                    trail_break_required=track.get("trail_break_required") is True,
                    tracker_lifecycle_generation=continuity_key[3],
                    source_epoch=continuity_key[7],
                )
            if self._association_service is not None:
                # This is an ORIGINAL measurement normalization receipt, not
                # a physical candidate. Do not insert a temporary subject-key
                # head next to the retained physical head for the same member.
                # Only prepare_physical may ingest it after actual DTO checks.
                continue
            # Trail episodes are producer-internal continuity authority, not
            # part of the public observation schema. Pass the exact accepted
            # row's break/segment context privately so global fusion does not
            # mistake a deliberate reacquisition reset for a same-episode
            # teleport. Fusion consumes each source/lifecycle/segment break
            # once and continues gating ordinary camera handoffs.
            working_fusion.ingest(
                observation,
                subject,
                trail_segment_id=self._nonnegative_int(
                    track.get("trail_segment_id")
                ),
                trail_break_required=(
                    track.get("trail_break_required") is True
                ),
                tracker_lifecycle_generation=self._nonnegative_int(
                    track.get("tracker_lifecycle_generation")
                ),
                source_epoch=self._nonnegative_int(
                    track.get("source_epoch")
                    if track.get("source_epoch") is not None
                    else metadata.get("source_epoch")
                ),
            )
        if len(used_native_records) != len(native_records):
            raise ValueError("original native observation lacks its exact normalized row")
        if not tracks or terminated_ids:
            if not camera_id:
                raise ValueError(
                    "empty source publication requires metadata.camera_id"
                )
            if self._association_service is None:
                working_fusion.clear_source(
                    camera_id, source, self.producer.run_id,
                    retained_lifecycles=retained_lifecycles,
                    terminated_tracker_ids=terminated_ids, terminated_only=bool(tracks),
                )
        self._prune_expired_track_outputs(
            working_track_outputs,
            source_id=source,
            current_media_pts_ns=current_source_media_pts_ns,
        )
        snapshot = _physical_snapshot(working_fusion, now_us)
        source_evidence_by_observation_id = {
            source.observation_id: source
            for entity in snapshot.entities
            for source in entity.sources
        }
        # Global fusion may intentionally ignore a held continuation when a
        # fresh metric camera owns the entity. That source-local row remains a
        # valid process origin, but a positional/registration/velocity
        # rejection must never become a committed origin for a later proof.
        cache_admitted_observation_ids = {
            observation_id
            for observation_id in pending_track_outputs
            if (
                (
                    evidence := source_evidence_by_observation_id.get(observation_id)
                )
                is not None
                and (
                    evidence.accepted is True
                    or evidence.rejection_reason
                    == "non_authoritative_held_continuation"
                )
            )
        }
        if self._association_service is not None:
            # Provisional origins remain private. The final physical staging
            # selects ONLY used observations actually admitted by its fusion.
            cache_admitted_observation_ids = set(pending_track_outputs)
        for (
            observation_id,
            (continuity_key, output),
        ) in pending_track_outputs.items():
            if observation_id in cache_admitted_observation_ids:
                evidence = source_evidence_by_observation_id.get(observation_id)
                prior = working_track_outputs.get(continuity_key)
                working_track_outputs[continuity_key] = replace(
                    output,
                    presence_origin=(
                        pending_presence_origins[observation_id]
                        if self._association_service is not None or evidence.accepted is True
                        else prior.presence_origin if prior is not None else None
                    ),
                )
        self._bound_track_outputs(working_track_outputs)
        return self._stage_candidate_locked(
            tuple(observations), snapshot, working_fusion, working_sequences,
            working_track_outputs, source, metadata, artifacts, physical_members=physical_members,
            physical_stamps=physical_stamps,
            spatial_quality=spatial_quality,
            spatial_upright_support=spatial_upright_support,
            spatial_input_diagnostics=spatial_input_diagnostics,
        )

    def _stage_candidate_locked(
        self, observations: tuple[ObservationEnvelope, ...], snapshot: WorldSnapshot,
        working_fusion: GlobalWorldFusion, working_sequences: dict[int, int],
        working_track_outputs: dict, source: int, metadata: Mapping[str, Any],
        artifacts: WorldArtifacts, *, physical_members: dict | None = None,
        association_candidate: bool = False,
        association_reservations: tuple[AssociationReservation, ...] = (),
        association_bindings: tuple[tuple[int, ScenePriorFrameBinding, tuple[int, int]], ...] = (),
        association_artifacts: tuple[tuple[int, Mapping[str, Any], WorldArtifacts], ...] = (),
        physical_stamps: dict[str, ObservationStamp] | None = None,
        physical_observations: tuple[ObservationEnvelope, ...] = (),
        spatial_quality: dict[str, tuple[float | None, str | None]] | None = None,
        spatial_upright_support: dict[str, PreparedUprightFootprintSupport] | None = None,
        spatial_memberships: tuple[SpatialMembershipDTO, ...] = (),
        spatial_reservations: tuple[SpatialMemberReservation, ...] = (),
        spatial_bindings: tuple[tuple[int, ScenePriorFrameBinding], ...] = (),
        spatial_context: SpatialClockContext | None = None,
        spatial_input_diagnostics: dict[str, Mapping[str, Any]] | None = None,
        attribution_mode: str = "strict",
        spatial_holds: tuple[tuple[PhysicalPersonKey, ObservationStamp], ...] = (),
        spatial_reindexes: tuple = (),
        spatial_split_custody: tuple = (),
        spatial_split_episode_views: tuple = (),
    ) -> PreparedWorldPublication:
        _guard_world_snapshot(snapshot)
        if len(self._prepared_candidates) >= self.MAX_PREPARED_CANDIDATES:
            raise RuntimeError("canonical world preparation capacity is exhausted")
        events, next_event_sequence, next_previous_entities = (
            self._events_for_state(
                snapshot,
                previous_entities=self._previous_entities,
                event_sequence=self._event_sequence,
            )
        )
        publication = WorldPublication(
            observations=tuple(observations),
            snapshot=snapshot,
            events=events,
        )
        journal_observations = physical_observations or observations
        dumped = {item.observation_id: item.model_dump(mode="json") for item in journal_observations}
        snapshot_json = snapshot.model_dump(mode="json")
        event_json = tuple(item.model_dump(mode="json") for item in events)
        journal_payloads = (*(dumped[item.observation_id] for item in journal_observations), snapshot_json, *event_json)
        wire_payloads = (*(dumped[item.observation_id] for item in observations), snapshot_json, *event_json)
        candidate_id = int(self._next_candidate_id)
        self._next_candidate_id += 1
        receipt = PreparedWorldPublication(
            base_revision=int(self._revision),
            # Public contracts and the containing dataclass are immutable;
            # mutable fusion/counter state remains service-private.
            publication=publication,
            _service_token=self._service_token,
            _candidate_id=candidate_id,
            _serialized_payloads=wire_payloads,
        )
        state = _PreparedWorldState(
            publication=publication,
            fusion=working_fusion,
            sequence_by_source=working_sequences,
            event_sequence=next_event_sequence,
            previous_entities=next_previous_entities,
            canonical_track_outputs=working_track_outputs,
            journal_payloads=journal_payloads,
            recorded_at_us=int(snapshot.published_at_us),
            source_id=source,
            artifact_metadata=dict(metadata),
            calibration_artifact_sha256=str(
                artifacts.calibration.sha256
            ),
            camera_calibration_sha256=self._sha256_text(
                artifacts.camera_calibration_sha256
            ),
            camera_calibration_binding_required=(
                artifacts.camera_calibration_sha256 is not None
            ),
            physical_members=physical_members or {},
            association_candidate=association_candidate,
            association_reservations=association_reservations,
            association_bindings=association_bindings,
            association_artifacts=association_artifacts,
            resolved_artifacts=artifacts,
            physical_stamps=physical_stamps or {},
            physical_observations=physical_observations,
            spatial_quality=spatial_quality or {},
            spatial_upright_support=spatial_upright_support or {},
            spatial_memberships=spatial_memberships, spatial_reservations=spatial_reservations,
            spatial_bindings=spatial_bindings, spatial_context=spatial_context,
            spatial_input_diagnostics=spatial_input_diagnostics or {},
            attribution_mode=attribution_mode,
            spatial_holds=spatial_holds,
            spatial_reindexes=spatial_reindexes,
            spatial_split_custody=spatial_split_custody,
            spatial_split_episode_views=spatial_split_episode_views,
        )
        self._prepared_candidates[candidate_id] = (receipt, state)
        return receipt

    def _commit_locked(
        self,
        prepared: PreparedWorldPublication,
        *,
        association_at: AcquisitionTime | None = None,
        association_at_interval: SpatialClockContext | None = None,
    ) -> WorldPublication:
        if self._closed:
            raise RuntimeError("canonical world service is closed")
        self._validate_prepared_owner(prepared)
        entry = self._prepared_candidates.get(prepared._candidate_id)
        if entry is None or entry[0] is not prepared:
            raise RuntimeError("canonical world preparation is unknown or consumed")
        _receipt, state = entry
        if state.association_candidate:
            self._association_actor()
            try:
                return self._commit_candidate_locked(prepared, state, association_at=association_at,
                                                     association_at_interval=association_at_interval)
            except Exception:
                self._prepared_candidates.pop(prepared._candidate_id, None)
                self._observe_spatial('outcome',prepared._candidate_id,'commit_failed')
                self._discard_association_reservations(state.association_reservations)
                self._discard_spatial_reservations(state.spatial_reservations)
                raise
        return self._commit_candidate_locked(prepared, state, association_at=association_at,
                                             association_at_interval=association_at_interval)

    def _commit_candidate_locked(
        self, prepared: PreparedWorldPublication, state: _PreparedWorldState,
        *, association_at: AcquisitionTime | None, association_at_interval: SpatialClockContext | None = None,
    ) -> WorldPublication:
        # Recheck the exact immutable candidate, not its cached JSON or only
        # incoming DTOs. This precedes every service grant and journal effect.
        _guard_world_snapshot(state.publication.snapshot)
        if self._association_service is not None:
            self._association_actor()
            if not state.association_candidate:
                raise RuntimeError("configured physical consumer requires prepare_physical before public commit")
        if int(prepared.base_revision) != int(self._revision):
            raise RuntimeError(
                "canonical world preparation is stale relative to service state"
            )
        if state.publication.snapshot.producer != self.producer:
            raise RuntimeError("canonical world preparation producer mismatch")
        if callable(self._artifacts):
            current_artifacts = self._artifacts(
                int(state.source_id),
                state.artifact_metadata,
            )
            current_camera_calibration_sha256 = self._sha256_text(
                current_artifacts.camera_calibration_sha256
            )
            if state.camera_calibration_binding_required and (
                current_camera_calibration_sha256 is None
                or current_camera_calibration_sha256
                != state.camera_calibration_sha256
            ):
                raise RuntimeError(
                    "canonical world preparation camera calibration authority "
                    "changed before commit"
                )
            if (
                str(current_artifacts.calibration.sha256)
                != str(state.calibration_artifact_sha256)
            ):
                raise RuntimeError(
                    "canonical world preparation calibration artifact changed "
                    "before commit"
                )
        if state.spatial_context is not None:
            if type(association_at_interval) is not SpatialClockContext or association_at is not None:
                raise ValueError("truthful original spatial commit context required")
            current_spatial_bindings = self._spatial_bindings() if callable(self._spatial_bindings) else {}
            if any(current_spatial_bindings.get(source) != binding for source, binding in state.spatial_bindings):
                raise WorldPublicationRefused("spatial_source_binding_changed")
            for source, metadata, original_artifacts in state.association_artifacts:
                current = self._artifacts(source, metadata) if callable(self._artifacts) else self._artifacts
                if current.calibration != original_artifacts.calibration or current.camera_calibration_sha256 != original_artifacts.camera_calibration_sha256:
                    raise WorldPublicationRefused("spatial_source_calibration_changed")
            if state.spatial_context is not None:
                # No grant can check the revision on an empty removal/hold
                # transaction. Use the actual public final owner view instead.
                self._validate_spatial_memberships(state.spatial_memberships, association_at_interval)
                reindex_prior = None
                if state.spatial_reindexes:
                    reindex_prior = self._fusion.fork()
                    reindex_prior.set_spatial_context(association_at_interval)
                for origin, dto, original in state.spatial_reindexes:
                    current = self._ordinary_observation_reindexes(state.spatial_memberships, {origin},
                        association_at_interval, prior=reindex_prior)
                    if (len(current) != 1 or current[0][0] != origin or current[0][1] is not dto
                            or current[0][2][0] is not original[0] or current[0][2][1:] != original[1:]):
                        prepared_us = state.publication.snapshot.published_at_us
                        commit_us = max(1, int(self._clock_us()))
                        deadline = original[0].observed_at_us + reindex_prior.config.lost_ttl_us
                        if not current and prepared_us < deadline <= commit_us:
                            # Recheck every current reindex predicate at the
                            # prepared wire time; this classifies, never grants,
                            # an otherwise identical now-expired original.
                            historical = self._ordinary_observation_reindexes(state.spatial_memberships,
                                {origin}, association_at_interval, prior=reindex_prior,
                                published_at_us=prepared_us)
                            if (len(historical) == 1 and historical[0][0] == origin
                                    and historical[0][1] is dto and historical[0][2][0] is original[0]
                                    and historical[0][2][1:] == original[1:]):
                                raise WorldPreparedHoldExpired(prepared, state.spatial_memberships,
                                    state.spatial_context, reason="ordinary_join_original_changed")
                        raise WorldPublicationRefused("ordinary_join_original_changed")
                if state.spatial_split_episode_views:
                    views = self._split_custody_views(state.spatial_memberships,
                        state.attribution_mode, committing=True)
                    if (len(views) != len(state.spatial_split_episode_views)
                            or any(not any(v is old for v in views) for old in state.spatial_split_episode_views)):
                        raise WorldPublicationRefused("split_custody_owner_changed")
                if state.spatial_split_custody:
                    views = self._split_custody_views(state.spatial_memberships,
                        state.attribution_mode, committing=True)
                    for view, custody in state.spatial_split_custody:
                        if not any(v is view for v in views):
                            raise WorldPublicationRefused("split_custody_owner_changed")
                        prior = self._fusion.fork()
                        prior.set_spatial_context(association_at_interval)
                        commit_us = max(1,int(self._clock_us()))
                        current = self._capture_attributed_split_original(prior, view.episode_id, view.origin_key,
                            view.successor_keys, published_at_us=commit_us)
                        if current is not None:
                            current=replace(current,destination_key=self._split_original_destination(
                                view,current.row.spatial_stamp,association_at_interval))
                        if current != custody:
                            if current is None and custody.entity.stale_after_us <= commit_us:
                                original = self._capture_attributed_split_original(prior, view.episode_id, view.origin_key,
                                    view.successor_keys, published_at_us=state.publication.snapshot.published_at_us)
                                if original is not None:
                                    original=replace(original,destination_key=self._split_original_destination(
                                        view,original.row.spatial_stamp,association_at_interval))
                                if original == custody:
                                    # Historical equality classifies only the
                                    # refusal; it never grants the expired row.
                                    raise WorldPreparedHoldExpired(prepared,state.spatial_memberships,state.spatial_context)
                            raise WorldPublicationRefused("split_custody_original_changed")
                        self._validate_split_original(view, custody.row.spatial_stamp)
                for key, stamp in state.spatial_holds:
                    prior = self._fusion.fork()
                    prior.set_spatial_context(association_at_interval)
                    commit_us=max(1,int(self._clock_us()))
                    custody = prior.committed_spatial_original(key,published_at_us=commit_us)
                    if custody is None or custody[1] != stamp:
                        expected=historical=deadline=admitted=None
                        if custody is None:
                            prepared_us=state.publication.snapshot.published_at_us
                            expected=state.fusion.committed_spatial_original(key,published_at_us=prepared_us)
                            historical=prior.committed_spatial_original(key,published_at_us=prepared_us)
                            if (expected is not None and historical is not None
                                    and historical[0] is expected[0] and historical[1:]==expected[1:]
                                    and historical[1]==stamp):
                                deadline=historical[0].observed_at_us+prior.config.lost_ttl_us
                                # Label-only custody can leave the original
                                # source-current/MEASURED. Wire expiry is still
                                # exact and independent of that presentation.
                                admitted=next((e for e in state.publication.snapshot.entities
                                    if e.observed_at_us==historical[0].observed_at_us
                                    and e.stale_after_us==deadline and any(s.accepted
                                        and s.observation_id==historical[0].observation_id for s in e.sources)),None)
                                if admitted is not None and prepared_us<deadline<=commit_us:
                                    # Historical equality classifies ONLY the
                                    # refusal. The expired bytes cannot release.
                                    raise WorldPreparedHoldExpired(prepared,state.spatial_memberships,
                                        state.spatial_context,reason="committed_spatial_hold_custody_changed")
                        self._log_spatial_hold_refusal(prepared,state,key,stamp,prior,custody,
                            commit_us=commit_us,expected=expected,historical=historical,
                            deadline=deadline,admitted=admitted,
                            branch="custody_none" if custody is None else "stamp_mismatch")
                        raise WorldPublicationRefused("committed_spatial_hold_custody_changed")
                    dto = next(d for d in state.spatial_memberships if d.physical_key == key)
                    if dto.association_basis == "estimated_image_region_conflict_custody":
                        check = self._conflict_custody_original(prior, dto)
                        if check is None or check[1] != stamp:
                            raise WorldPublicationRefused("committed_spatial_hold_custody_changed")
            # Preflight ALL public grants at the latest source context BEFORE
            # consuming any selected member frontier. Never renew a stale peer.
            original_reservations = state.spatial_reservations
            self._discard_spatial_reservations(original_reservations)
            state.spatial_reservations = ()
            for reservation in original_reservations:
                _guard_world_snapshot(state.publication.snapshot)
                prepare_member = (self._association_service.prepare_spatial_physical_member
                    if reservation.purpose == "physical_only_quarantine" else self._association_service.prepare_spatial_member)
                check = prepare_member(
                    reservation.dto, reservation.reference, at_interval=association_at_interval,
                )
                if check.disposition != "reserved" or check.reservation is None:
                    raise WorldPublicationRefused.from_association_result(check)
                if (check.reservation.purpose != reservation.purpose
                        or check.reservation.household_conflict is not reservation.household_conflict):
                    raise WorldPublicationRefused("attribution_reservation_changed")
                state.spatial_reservations += (check.reservation,)
            self._prepared_candidates.pop(prepared._candidate_id, None)
            for reservation in state.spatial_reservations:
                _guard_world_snapshot(state.publication.snapshot)
                def stage_spatial_grant(dto, reference, expected=reservation):
                    if dto is not expected.dto or reference is not expected.reference:
                        raise RuntimeError("spatial grant did not retain exact owner objects")
                    return None
                commit_member = (self._association_service.commit_spatial_physical_member
                    if reservation.purpose == "physical_only_quarantine" else self._association_service.commit_spatial_member)
                result = commit_member(
                    reservation, at_interval=association_at_interval, consumer=stage_spatial_grant,
                )
                if result.disposition != "committed":
                    raise WorldPublicationRefused.from_association_result(result)
        elif state.association_candidate:
            if association_at_interval is not None:
                raise ValueError("strict physical route requires association_at, not a spatial interval")
            # Grants consume intents, not public World authority. A later
            # denial must abort the WHOLE private candidate, including earlier
            # successful grants; it is never safe to retry consumed intents.
            self._prepared_candidates.pop(prepared._candidate_id, None)
            if type(association_at) is not AcquisitionTime:
                raise ValueError("truthful admitted association commit clock required")
            pairs = tuple((r.dto, r.cohort) for r in state.association_reservations)
            current_bindings = self._validated_physical_bindings(pairs, state.physical_observations or state.publication.observations)
            if current_bindings != state.association_bindings:
                raise RuntimeError("physical metric/source/raster binding changed before commit")
            for source, metadata, original_artifacts in state.association_artifacts:
                current = self._artifacts(source, metadata) if callable(self._artifacts) else self._artifacts
                if (current.calibration != original_artifacts.calibration
                        or current.camera_calibration_sha256 != original_artifacts.camera_calibration_sha256):
                    raise RuntimeError("physical source calibration authority changed before commit")
            for reservation in state.association_reservations:
                _guard_world_snapshot(state.publication.snapshot)
                def stage_grant(dto, cohort, expected=reservation):
                    if dto is not expected.dto or cohort is not expected.cohort:
                        raise RuntimeError("physical grant did not retain exact owner objects")
                    return None  # No public/per-person World mutation.
                result = self._association_service.commit(reservation, association_at, stage_grant)
                if result.disposition != "committed":
                    raise WorldPublicationRefused.from_association_result(result)
        _guard_world_snapshot(state.publication.snapshot)
        if self._journal is not None:
            try:
                expected_records = len(state.journal_payloads)
                journal_capacity = getattr(
                    self._journal,
                    "max_records",
                    None,
                )
                if (
                    isinstance(journal_capacity, int)
                    and not isinstance(journal_capacity, bool)
                    and journal_capacity < expected_records
                ):
                    raise RuntimeError(
                        "canonical world journal retention cannot hold the "
                        f"complete cohort: required={expected_records} "
                        f"max_records={journal_capacity}"
                    )
                appended = self._journal.append_many(
                    state.journal_payloads,
                    recorded_at_us=state.recorded_at_us,
                )
                if isinstance(self._journal, AsyncContractJournal):
                    if (
                        not isinstance(appended, AsyncJournalAdmissionReceipt)
                        or int(appended.payload_count) != expected_records
                    ):
                        raise RuntimeError(
                            "canonical world persistence admission did not confirm "
                            f"the exact payload count: expected={expected_records}"
                        )
                elif len(appended) != expected_records:
                    raise RuntimeError(
                        "canonical world journal did not confirm the exact "
                        f"append count: expected={expected_records} "
                        f"actual={len(appended)}"
                    )
            except Exception as exc:
                if isinstance(self._journal, AsyncContractJournal):
                    # The journal is reconstructable persistence, not live
                    # spatial authority.  Record its bounded failure and keep
                    # the exact in-memory tracking/world/BEV cohort moving.
                    self._persistence_rejected_cohorts += 1
                    self._persistence_last_error = (
                        f"{type(exc).__name__}: {exc}"
                    )
                else:
                    self._prepared_candidates.pop(
                        prepared._candidate_id,
                        None,
                    )
                    raise
        self._fusion = state.fusion
        self._sequence_by_source = state.sequence_by_source
        self._event_sequence = int(state.event_sequence)
        self._previous_entities = state.previous_entities
        self._canonical_track_outputs = state.canonical_track_outputs
        self._current_snapshot = state.publication.snapshot
        self._revision += 1
        for row in state.spatial_observer_records:
            self._observe_spatial('record',{**row,'transaction':'committed'})
        for candidate in self._prepared_candidates:
            self._observe_spatial('outcome',candidate,'committed' if candidate==prepared._candidate_id else 'superseded')
        self._prepared_candidates.clear()
        return state.publication

    def _validate_prepared_owner(
        self,
        prepared: PreparedWorldPublication,
    ) -> None:
        if not isinstance(prepared, PreparedWorldPublication):
            raise TypeError("canonical world operation requires a prepared publication")
        if prepared._service_token is not self._service_token:
            raise RuntimeError("canonical world preparation belongs to another service")

    def close(self) -> None:
        # Serialize close with publication so an in-flight frame either reaches
        # the journal completely or finishes before the journal is closed.
        with self._lock:
            if self._closed:
                return
            if self._association_service is not None:
                self._association_actor()
            for _, state in self._prepared_candidates.values():
                self._discard_association_reservations(state.association_reservations)
                self._discard_spatial_reservations(state.spatial_reservations)
            for candidate in self._prepared_candidates:
                self._observe_spatial('outcome',candidate,'abandoned_close')
            self._closed = True
            self._prepared_candidates.clear()
            try:
                _LOGGER.info('Private World decision summary: %s',self._spatial_observer.encoded())
            except Exception:
                pass  # Optional observer cannot change real close/refusal behavior.
            close = getattr(self._journal, "close", None)
            close_error: BaseException | None = None
            if callable(close):
                try:
                    close()
                except BaseException as exc:
                    close_error = exc
            if isinstance(self._journal, AsyncContractJournal):
                durable_close = getattr(self._journal.journal, "close", None)
                if callable(durable_close):
                    try:
                        durable_close()
                    except BaseException as exc:
                        if close_error is None:
                            close_error = exc
            if close_error is not None:
                raise close_error

    def _events_for_state(
        self,
        snapshot: WorldSnapshot,
        *,
        previous_entities: Mapping[str, WorldEntity],
        event_sequence: int,
    ) -> tuple[tuple[WorldEvent, ...], int, dict[str, WorldEntity]]:
        current = {entity.entity_id: entity for entity in snapshot.entities}
        pending: list[tuple[str, WorldEntity, str | None]] = []
        for entity_id, entity in sorted(current.items()):
            previous = previous_entities.get(entity_id)
            if previous is None:
                pending.append(("appeared", entity, None))
            else:
                if (
                    previous.lifecycle != EntityLifecycle.HELD
                    and entity.lifecycle == EntityLifecycle.HELD
                ):
                    pending.append(("held", entity, "producer_evidence_stale"))
                elif (
                    previous.lifecycle == EntityLifecycle.HELD
                    and entity.lifecycle == EntityLifecycle.PRESENT
                ):
                    pending.append(("resumed", entity, "fresh_producer_evidence"))
                if not previous.conflict and entity.conflict:
                    pending.append(("conflict_started", entity, entity.conflict_reason))
                elif previous.conflict and not entity.conflict:
                    pending.append(("conflict_cleared", entity, "camera_evidence_reconciled"))
        for entity_id, previous in sorted(previous_entities.items()):
            if entity_id not in current:
                pending.append(("lost", previous, "producer_evidence_expired"))

        events: list[WorldEvent] = []
        for event_type, entity, reason in pending:
            sequence = int(event_sequence)
            event_sequence += 1
            events.append(
                WorldEvent(
                    contract="noesis.world.event",
                    contract_version=1,
                    event_id=f"{self.producer.run_id}:event:{sequence}",
                    producer=self.producer,
                    sequence=sequence,
                    event_type=event_type,  # type: ignore[arg-type]
                    entity_id=entity.entity_id,
                    subject=entity.subject,
                    attribution_quarantine=entity.attribution_quarantine,
                    observed_at_us=min(entity.observed_at_us, snapshot.published_at_us),
                    published_at_us=snapshot.published_at_us,
                    frame="backend_world_m",
                    units="meters",
                    world_frame="backend_world_m",
                    world_frame_revision=entity.world_frame_revision,
                    world_transform_sha256=entity.world_transform_sha256,
                    calibration_revision=entity.calibration_revision,
                    position=entity.position,
                    reason=reason,
                )
            )
        return tuple(events), int(event_sequence), current

    @staticmethod
    def _next_sequence(
        source_id: int,
        sequence_by_source: dict[int, int],
    ) -> int:
        value = sequence_by_source.get(source_id, 0)
        sequence_by_source[source_id] = value + 1
        return value

    def _canonical_track_output_key(
        self,
        source_id: int,
        track: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any],
        calibration_sha256: str,
    ) -> tuple[int, str, int, int, str, str, str, int] | None:
        """Resolve one epoch-, revision-, and lifecycle-bound output key."""

        try:
            tracker_id = int(track.get("tracker_id", track.get("track_id")))
            lifecycle_generation = int(
                track.get("tracker_lifecycle_generation")
            )
        except (TypeError, ValueError, OverflowError):
            return None
        camera_id = str(
            track.get("camera_id")
            or metadata.get("camera_id")
            or f"camera_{int(source_id)}"
        ).strip()
        world_frame_revision = self._required_revision(
            track.get("world_frame_revision")
        )
        world_transform_sha256 = self._sha256_text(
            track.get("world_transform_sha256")
        )
        calibration_artifact_sha256 = self._sha256_text(
            calibration_sha256
        )
        source_epoch = self._nonnegative_int(
            track.get("source_epoch")
            if track.get("source_epoch") is not None
            else metadata.get("source_epoch")
        )
        if source_epoch is None:
            source_epoch = 0
        if (
            tracker_id < 0
            or lifecycle_generation <= 0
            or not camera_id
            or world_frame_revision is None
            or world_transform_sha256 is None
            or calibration_artifact_sha256 is None
        ):
            return None
        return (
            int(source_id),
            camera_id,
            tracker_id,
            lifecycle_generation,
            world_frame_revision,
            world_transform_sha256,
            calibration_artifact_sha256,
            int(source_epoch),
        )

    @staticmethod
    def _prune_superseded_track_outputs(
        outputs: dict[
            tuple[int, str, int, int, str, str, str, int],
            _CanonicalTrackOutput,
        ],
        current_key: tuple[int, str, int, int, str, str, str, int],
    ) -> None:
        """Drop older lifecycle/revision bindings for one source-local ID."""

        source_id, camera_id, tracker_id = current_key[:3]
        for key in tuple(outputs):
            if (
                key != current_key
                and key[0] == source_id
                and key[1] == camera_id
                and key[2] == tracker_id
            ):
                outputs.pop(key, None)

    def _bound_track_outputs(
        self,
        outputs: dict[
            tuple[int, str, int, int, str, str, str, int],
            _CanonicalTrackOutput,
        ],
    ) -> None:
        """Apply a deterministic global cap to compact continuity origins."""

        capacity = max(1, int(self.MAX_CANONICAL_TRACK_OUTPUTS))
        while len(outputs) > capacity:
            oldest_key = next(iter(outputs))
            outputs.pop(oldest_key, None)

    @classmethod
    def _prune_expired_track_outputs(
        cls,
        outputs: dict[
            tuple[int, str, int, int, str, str, str, int],
            _CanonicalTrackOutput,
        ],
        *,
        source_id: int,
        current_media_pts_ns: int | None,
    ) -> None:
        """Drop origins too old for every canonical continuation lane."""

        if current_media_pts_ns is None or current_media_pts_ns <= 0:
            return
        short_horizon_ns = int(
            round(
                (
                    cls.BOUNDED_PROCESS_HORIZON_S
                    + cls.BOUNDED_PROCESS_HORIZON_EPSILON_S
                )
                * 1_000_000_000.0
            )
        )
        stationary_horizon_ns = int(
            round(
                (
                    cls.STATIONARY_ANCHOR_HOLD_HORIZON_S
                    + cls.BOUNDED_PROCESS_HORIZON_EPSILON_S
                )
                * 1_000_000_000.0
            )
        )
        upright_presence_horizon_ns = int(
            round(
                (
                    cls.UPRIGHT_PRESENCE_HOLD_HORIZON_S
                    + cls.BOUNDED_PROCESS_HORIZON_EPSILON_S
                )
                * 1_000_000_000.0
            )
        )
        committed_origin_history_horizon_ns = int(
            round(
                cls.IMAGE_MOTION_COMMITTED_ORIGIN_HISTORY_HORIZON_S
                * 1_000_000_000.0
            )
        )
        for key, output in tuple(outputs.items()):
            if key[0] != int(source_id):
                continue
            output_age_ns = current_media_pts_ns - int(output.media_pts_ns)
            metric_age_ns = (
                current_media_pts_ns - int(output.metric_media_pts_ns)
                if output.metric_media_pts_ns is not None
                else None
            )
            if output_age_ns < 0 or (
                metric_age_ns is not None and metric_age_ns < 0
            ):
                # A reordered source cohort is handled by the producer's epoch
                # contract. Do not let this cache invent a reset independently.
                continue
            recent_projective_origin = bool(
                output.committed_world_source == "image_motion_prediction"
                and output_age_ns <= short_horizon_ns
            )
            image_motion_validation_origin = bool(
                output_age_ns <= committed_origin_history_horizon_ns
            )
            upright_presence_validation_origin = bool(
                output_age_ns <= upright_presence_horizon_ns
            )
            stationary_metric_origin = bool(
                metric_age_ns is not None
                and metric_age_ns <= stationary_horizon_ns
            )
            if not (
                recent_projective_origin
                or image_motion_validation_origin
                or upright_presence_validation_origin
                or stationary_metric_origin
            ):
                outputs.pop(key, None)

    def _observation(
        self,
        source_id: int,
        track: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any],
        artifacts: WorldArtifacts,
        published_at_us: int,
        sequence_by_source: dict[int, int],
        expected_image_motion_origin: _CanonicalTrackOutput | None,
    ) -> ObservationEnvelope | None:
        try:
            tracker_id = int(track.get("tracker_id", track.get("track_id")))
            frame_id = int(track.get("frame_id", 0))
        except Exception:
            return None
        if tracker_id < 0 or frame_id < 0:
            return None
        camera_id = str(track.get("camera_id") or metadata.get("camera_id") or f"camera_{source_id}").strip()
        if not camera_id:
            return None
        observed_at_us = self._positive_int(track.get("observed_at_us")) or published_at_us
        published_at_us = max(published_at_us, observed_at_us)
        capture_status = str(track.get("capture_time_status") or "estimated")
        if capture_status not in {"synced", "estimated", "unavailable"}:
            capture_status = "unavailable"
        captured_at_us = None
        if capture_status != "unavailable":
            captured_at_us = (
                self._positive_int(track.get("captured_at_us"))
                or self._positive_int(metadata.get("captured_at_us"))
                or observed_at_us
            )
        media_pts_ns = self._nonnegative_int(track.get("media_pts_ns"))

        bbox = track.get("bbox")
        image_size = track.get("image_size") or metadata.get("image_size") or metadata.get("frame_size")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return None
        if not isinstance(image_size, (list, tuple)) or len(image_size) < 2:
            return None
        try:
            bbox_xywh = tuple(float(value) for value in bbox[:4])
            size = (int(image_size[0]), int(image_size[1]))
        except Exception:
            return None
        if not all(math.isfinite(value) for value in bbox_xywh) or size[0] <= 0 or size[1] <= 0:
            return None

        calibration_revision = track.get("calibration_revision")
        if calibration_revision is None:
            calibration_revision = artifacts.calibration.version
        raw_camera_calibration_sha256 = (
            artifacts.camera_calibration_sha256
        )
        camera_calibration_binding_required = bool(
            raw_camera_calibration_sha256 is not None
        )
        camera_calibration_rejection_reason = None
        if camera_calibration_binding_required:
            expected_camera_calibration_sha256 = self._sha256_text(
                raw_camera_calibration_sha256
            )
            raw_measurement_calibration_sha256 = track.get(
                "world_calibration_sha256"
            )
            measurement_calibration_sha256 = self._sha256_text(
                raw_measurement_calibration_sha256
            )
            camera_calibration_matches = bool(
                expected_camera_calibration_sha256 is not None
                and measurement_calibration_sha256
                == expected_camera_calibration_sha256
            )
            if not camera_calibration_matches:
                if expected_camera_calibration_sha256 is None:
                    camera_calibration_rejection_reason = (
                        "camera_calibration_authority_invalid"
                    )
                elif raw_measurement_calibration_sha256 is None or (
                    isinstance(raw_measurement_calibration_sha256, str)
                    and not raw_measurement_calibration_sha256.strip()
                ):
                    # Unavailable geometry often has no measurement binding.
                    # Preserve its producer rejection without presenting that
                    # absence as a conflicting calibration. Unbound numeric
                    # candidates remain quarantined below.
                    camera_calibration_rejection_reason = (
                        self._world_first_divergence_reason(
                            track,
                            world_observation=None,
                            expected_continuity_origin=expected_image_motion_origin,
                        )
                        if track.get("world_valid") is not True
                        else "camera_calibration_sha256_missing"
                    )
                elif measurement_calibration_sha256 is None:
                    camera_calibration_rejection_reason = (
                        "camera_calibration_sha256_invalid"
                    )
                else:
                    camera_calibration_rejection_reason = (
                        "camera_calibration_sha256_mismatch"
                    )
        else:
            camera_calibration_matches = True
        world_observation = (
            self._world_observation(
                track,
                calibration_revision=calibration_revision,
                expected_image_motion_origin=expected_image_motion_origin,
            )
            if camera_calibration_matches
            else None
        )
        world_diagnostics = (
            self._world_diagnostics(
                track,
                world_observation=world_observation,
                expected_continuity_origin=expected_image_motion_origin,
            )
            if camera_calibration_matches
            else WorldObservationDiagnostics(
                first_divergence_reason=camera_calibration_rejection_reason
            )
        )
        coordinate_frame = "backend_world_m" if world_observation is not None else "image_px"
        units = "meters" if world_observation is not None else "pixels"
        tracklet = TrackletRef(
            run_id=self.producer.run_id,
            camera_id=camera_id,
            source_id=source_id,
            tracker_id=tracker_id,
            frame_id=frame_id,
            observed_at_us=observed_at_us,
        )
        embedding_provenance = self._embedding_provenance(track)
        payload = PersonObservation(
            tracklet=tracklet,
            detection_confidence=self._confidence(track.get("confidence")),
            tracker_confidence=self._confidence(track.get("tracker_confidence")),
            bbox_xywh=bbox_xywh,  # type: ignore[arg-type]
            image_size=size,
            zone=self._exact_zone(track.get("zone")),
            zone_source=self._zone_source(track.get("zone_source")),
            zone_authoritative=track.get("zone_authoritative") is True,
            world=world_observation,
            world_diagnostics=world_diagnostics,
            **embedding_provenance,
            pose_present=bool(track.get("pose_present", False)),
            depth_present=self._depth_present(track),
            occluded=bool(track.get("occluded", False)),
        )
        sequence = self._next_sequence(source_id, sequence_by_source)
        observation_id = f"{self.producer.run_id}:{source_id}:{sequence}:{tracker_id}"
        return ObservationEnvelope(
            contract="noesis.observation.person",
            contract_version=1,
            observation_id=observation_id,
            producer=self.producer,
            sequence=sequence,
            captured_at_us=captured_at_us,
            observed_at_us=observed_at_us,
            published_at_us=published_at_us,
            capture_time_status=capture_status,  # type: ignore[arg-type]
            media_pts_ns=media_pts_ns,
            coordinate_frame=coordinate_frame,  # type: ignore[arg-type]
            units=units,  # type: ignore[arg-type]
            calibration=artifacts.calibration,
            model=artifacts.model,
            config=artifacts.config,
            payload=payload,
        )

    @staticmethod
    def _embedding_provenance(track: Mapping[str, Any]) -> dict[str, Any]:
        fields = (
            "embedding_sequence",
            "embedding_model_sha256",
            "embedding_dimension",
        )
        present = tuple(track.get(field) is not None for field in fields)
        if any(present) and not all(present):
            populated = ", ".join(
                field for field, is_present in zip(fields, present) if is_present
            )
            raise ValueError(
                "person track contains partial embedding provenance; "
                f"populated fields: {populated}"
            )
        if not any(present):
            return {}
        return {field: track[field] for field in fields}

    @staticmethod
    def _depth_present(track: Mapping[str, Any]) -> bool:
        return usable_registered_depth_m(track) is not None

    @staticmethod
    def _inferred_ground_episode_root_from_provenance(
        provenance: Mapping[str, Any],
    ) -> _InferredGroundEpisodeRoot | None:
        """Parse the immutable root fields without accepting producer authority."""

        raw_origin = provenance.get("raw_origin")
        trusted_world_origin = provenance.get("trusted_world_origin")
        if (
            not isinstance(raw_origin, (list, tuple))
            or len(raw_origin) < 3
            or not isinstance(trusted_world_origin, (list, tuple))
            or len(trusted_world_origin) < 3
        ):
            return None
        world_frame = provenance.get("world_frame")
        world_frame_revision = provenance.get("world_frame_revision")
        world_transform_sha256 = provenance.get("world_transform_sha256")
        if not all(
            isinstance(value, str) and bool(value)
            for value in (
                world_frame,
                world_frame_revision,
                world_transform_sha256,
            )
        ):
            return None
        try:
            root = _InferredGroundEpisodeRoot(
                raw_origin=tuple(float(value) for value in raw_origin[:3]),
                raw_origin_media_pts_ns=int(
                    provenance.get("raw_origin_media_pts_ns")
                ),
                raw_origin_observed_at_us=int(
                    provenance.get("origin_observed_at_us")
                ),
                raw_origin_ts_s=float(provenance.get("raw_origin_ts_s")),
                trusted_world_origin=tuple(
                    float(value) for value in trusted_world_origin[:3]
                ),
                trusted_origin_media_pts_ns=int(
                    provenance.get("trusted_origin_media_pts_ns")
                ),
                trusted_origin_filter_ts_s=float(
                    provenance.get("trusted_origin_filter_ts_s")
                ),
                lifecycle_generation=int(
                    provenance.get("tracker_lifecycle_generation")
                ),
                trail_segment_id=int(
                    provenance.get("origin_trail_segment_id")
                ),
                height_ref_scene=float(provenance.get("height_ref_scene")),
                world_frame=world_frame,
                world_frame_revision=world_frame_revision,
                world_transform_sha256=world_transform_sha256,
            )
        except (TypeError, ValueError, OverflowError):
            return None
        if not all(
            math.isfinite(value)
            for value in (
                *root.raw_origin,
                root.raw_origin_ts_s,
                *root.trusted_world_origin,
                root.trusted_origin_filter_ts_s,
                root.height_ref_scene,
            )
        ):
            return None
        return root

    @classmethod
    def _inferred_ground_episode_root_for_admitted_output(
        cls,
        track: Mapping[str, Any],
        *,
        expected_origin: _CanonicalTrackOutput | None,
        current_media_pts_ns: int,
    ) -> _InferredGroundEpisodeRoot | None:
        """Establish one inferred root or retain it through bounded descendants."""

        provenance = track.get("world_prediction_provenance")
        if not isinstance(provenance, Mapping):
            return None
        provenance_type = str(provenance.get("type") or "")
        source = str(track.get("world_source") or "").strip().lower()
        established_root = (
            expected_origin.inferred_ground_episode_root
            if expected_origin is not None
            else None
        )
        if (
            source == "image_motion_prediction"
            and provenance_type == "inferred_ground_process_observation"
        ):
            current_root = cls._inferred_ground_episode_root_from_provenance(
                provenance
            )
            if current_root is None:
                return None
            if established_root is not None and current_root != established_root:
                return None
            return established_root or current_root
        if (
            expected_origin is None
            or established_root is None
            or source not in {"cv_prediction", "anchor_hold"}
            or provenance_type not in {
                "bounded_cv_process",
                "bounded_output_hold",
            }
        ):
            return None
        # The immutable root belongs to one *active lower-body occlusion*
        # episode, not to the tracker lifecycle indefinitely.  A bounded CV
        # row or exact hold can bridge a brief estimator miss while that same
        # standing occlusion is still visible.  Once the public row says the
        # lower body is no longer occluded (or posture/motion leaves the
        # upright lane), terminate the service-owned root.  This lets a later
        # independently observed occlusion establish a fresh immutable root
        # instead of being rejected as an attempted renewal of an episode
        # that has already ended.
        occlusion_level = str(
            track.get("lower_body_occlusion_level") or ""
        ).strip()
        if not (
            track.get("lower_body_occluded") is True
            and occlusion_level
            and occlusion_level != "none"
            and str(track.get("posture") or "").strip().lower()
            == "standing"
            and str(track.get("motion_mode") or "").strip().lower()
            not in {"sit", "lie"}
        ):
            return None
        try:
            lifecycle_generation = int(
                track.get("tracker_lifecycle_generation")
            )
            trail_segment_id = int(track.get("trail_segment_id"))
        except (TypeError, ValueError, OverflowError):
            return None
        maximum_age_ns = int(
            round(cls.INFERRED_PROCESS_RESTART_MAX_AGE_S * 1_000_000_000.0)
        )
        if not (
            0 < current_media_pts_ns - int(expected_origin.media_pts_ns)
            <= maximum_age_ns
            and lifecycle_generation == established_root.lifecycle_generation
            and trail_segment_id == established_root.trail_segment_id
            and trail_segment_id == int(expected_origin.trail_segment_id)
            and track.get("world_frame") == established_root.world_frame
            and track.get("world_frame_revision")
            == established_root.world_frame_revision
            and track.get("world_transform_sha256")
            == established_root.world_transform_sha256
        ):
            return None
        return established_root

    @classmethod
    def _matched_committed_output_origin(
        cls,
        expected_origin: _CanonicalTrackOutput,
        *,
        origin_point: tuple[float, float, float],
        origin_media_pts_ns: int,
        origin_trail_segment_id: int,
        current_media_pts_ns: int,
        maximum_age_s: float,
    ) -> _CanonicalOutputOrigin | None:
        """Bind a proof to one exact output this service committed.

        The media estimator and ordered publication worker advance on separate
        threads, so a proof created from a committed output can arrive after a
        few newer outputs have also committed.  The bounded history contains
        service-owned facts only; producer-supplied coordinates cannot create
        an origin.
        """

        candidates = (
            _CanonicalOutputOrigin(
                position=expected_origin.position,
                media_pts_ns=int(expected_origin.media_pts_ns),
                trail_segment_id=int(expected_origin.trail_segment_id),
                committed_world_source=expected_origin.committed_world_source,
                support_surface=expected_origin.support_surface,
                committed_provenance_type=(
                    expected_origin.committed_provenance_type
                ),
                metric_position=expected_origin.metric_position,
                metric_media_pts_ns=expected_origin.metric_media_pts_ns,
                metric_trail_segment_id=(
                    expected_origin.metric_trail_segment_id
                ),
                projective_bridge_root_media_pts_ns=(
                    expected_origin.projective_bridge_root_media_pts_ns
                ),
                kinematic_position=expected_origin.kinematic_position,
                kinematic_media_pts_ns=(
                    expected_origin.kinematic_media_pts_ns
                ),
                kinematic_trail_segment_id=(
                    expected_origin.kinematic_trail_segment_id
                ),
                bbox_geometry=expected_origin.bbox_geometry,
                kinematic_bbox_geometry=(
                    expected_origin.kinematic_bbox_geometry
                ),
                inferred_ground_episode_root=(
                    expected_origin.inferred_ground_episode_root
                ),
            ),
            *expected_origin.prior_outputs,
        )
        history_horizon_ns = int(
            round(max(0.0, float(maximum_age_s)) * 1_000_000_000.0)
        )
        for candidate in candidates:
            if (
                0 < current_media_pts_ns - int(candidate.media_pts_ns)
                <= history_horizon_ns
                and origin_media_pts_ns == int(candidate.media_pts_ns)
                and origin_trail_segment_id == int(candidate.trail_segment_id)
                and all(
                    math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
                    for left, right in zip(origin_point, candidate.position)
                )
            ):
                return candidate
        return None

    @classmethod
    def _matches_committed_output_origin(
        cls,
        expected_origin: _CanonicalTrackOutput,
        *,
        origin_point: tuple[float, float, float],
        origin_media_pts_ns: int,
        origin_trail_segment_id: int,
        current_media_pts_ns: int,
    ) -> bool:
        return (
            cls._matched_committed_output_origin(
                expected_origin,
                origin_point=origin_point,
                origin_media_pts_ns=origin_media_pts_ns,
                origin_trail_segment_id=origin_trail_segment_id,
                current_media_pts_ns=current_media_pts_ns,
                maximum_age_s=(
                    cls.IMAGE_MOTION_COMMITTED_ORIGIN_HISTORY_HORIZON_S
                ),
            )
            is not None
        )

    @staticmethod
    def _support_plane_elevation(
        surface: SupportSurfaceReference, x: float, z: float,
    ) -> float:
        nx, ny, nz = surface.normal_world
        return -(nx * x + nz * z + surface.offset_m) / ny

    @staticmethod
    def _point_matches_support_plane(
        surface: SupportSurfaceReference, point: tuple[float, float, float],
    ) -> bool:
        nx, ny, nz = surface.normal_world
        residual = nx * point[0] + ny * point[1] + nz * point[2] + surface.offset_m
        return (
            math.isfinite(residual)
            and abs(residual) <= 1e-8
            and contains_xz(surface, point[0], point[2])
        )

    @classmethod
    def _continuation_support_surface(
        cls,
        track: Mapping[str, Any],
        expected_origin: _CanonicalTrackOutput | None,
    ) -> tuple[bool, SupportSurfaceReference | None]:
        """A continuation can only retain the last accepted plane identity."""

        if expected_origin is None:
            return False, None
        payload = track.get("world_support_surface")
        if payload is None:
            surface = None
        else:
            try:
                surface = SupportSurfaceReference.model_validate(payload)
            except (TypeError, ValueError):
                return False, None
        if surface != expected_origin.support_surface:
            return False, None
        if surface is not None and not cls._point_matches_support_plane(
            surface, expected_origin.position,
        ):
            return False, None
        return True, surface

    @classmethod
    def _matched_inferred_ground_restart_origin(
        cls,
        expected_origin: _CanonicalTrackOutput,
        provenance: Mapping[str, Any],
        *,
        current_media_pts_ns: int,
    ) -> _CanonicalOutputOrigin | None:
        """Resolve the exact recent service output named by a restart proof."""

        transition = provenance.get("filter_transition")
        if not isinstance(transition, Mapping):
            return None
        origin_world = transition.get("origin_world")
        if not isinstance(origin_world, (list, tuple)) or len(origin_world) < 3:
            return None
        try:
            origin_point = tuple(float(value) for value in origin_world[:3])
            origin_media_pts_ns = int(
                transition.get("origin_media_pts_ns")
            )
            origin_trail_segment_id = int(
                transition.get("origin_trail_segment_id")
            )
        except (TypeError, ValueError, OverflowError):
            return None
        return cls._matched_committed_output_origin(
            expected_origin,
            origin_point=origin_point,
            origin_media_pts_ns=origin_media_pts_ns,
            origin_trail_segment_id=origin_trail_segment_id,
            current_media_pts_ns=current_media_pts_ns,
            maximum_age_s=cls.INFERRED_PROCESS_RESTART_MAX_AGE_S,
        )

    @classmethod
    def _is_projective_image_output(
        cls,
        origin: _CanonicalOutputOrigin | _CanonicalTrackOutput,
    ) -> bool:
        return bool(
            origin.committed_world_source == "image_motion_prediction"
            and origin.committed_provenance_type
            in cls.PROJECTIVE_BRIDGE_IMAGE_PROVENANCE_TYPES
        )

    @classmethod
    def _active_projective_bridge_root(
        cls,
        latest_origin: _CanonicalTrackOutput,
        *,
        current_media_pts_ns: int,
    ) -> int | None:
        """Return the latest service-owned immutable projective episode root.

        A newly admitted image-motion output establishes a root. Bounded CV
        descendants and exact gain-zero holds may inherit that same root, but
        never replace it with their own publication PTS or resurrect a
        historical image after a metric/ordinary output terminated the lane.
        """

        if cls._is_projective_image_output(latest_origin):
            root_media_pts_ns = int(latest_origin.media_pts_ns)
        elif latest_origin.projective_bridge_root_media_pts_ns is not None:
            root_media_pts_ns = int(
                latest_origin.projective_bridge_root_media_pts_ns
            )
        else:
            return None
        horizon_ns = int(
            round(
                (
                    cls.PROJECTIVE_PROCESS_BRIDGE_HORIZON_S
                    + cls.BOUNDED_PROCESS_HORIZON_EPSILON_S
                )
                * 1_000_000_000.0
            )
        )
        if not (
            0 < root_media_pts_ns < current_media_pts_ns
            and current_media_pts_ns - root_media_pts_ns <= horizon_ns
        ):
            return None
        return root_media_pts_ns

    @classmethod
    def _projective_bridge_root_for_admitted_output(
        cls,
        track: Mapping[str, Any],
        *,
        expected_origin: _CanonicalTrackOutput | None,
        current_media_pts_ns: int,
    ) -> int | None:
        """Retain service-owned nonrenewing lineage for an admitted output."""

        source = str(track.get("world_source") or "").strip().lower()
        provenance = track.get("world_prediction_provenance")
        provenance_type = (
            str(provenance.get("type") or "").strip()
            if isinstance(provenance, Mapping)
            else ""
        )
        if (
            source == "image_motion_prediction"
            and provenance_type in cls.PROJECTIVE_BRIDGE_IMAGE_PROVENANCE_TYPES
        ):
            return int(current_media_pts_ns)
        if (
            expected_origin is None
            or source not in {"cv_prediction", "anchor_hold"}
            or provenance_type
            not in {"bounded_cv_process", "bounded_output_hold"}
        ):
            return None
        return cls._active_projective_bridge_root(
            expected_origin,
            current_media_pts_ns=current_media_pts_ns,
        )

    @classmethod
    def _latest_output_step_is_physical(
        cls,
        expected_origin: _CanonicalTrackOutput,
        *,
        world_point: tuple[float, float, float],
        current_media_pts_ns: int,
        max_speed_mps: float | None = None,
    ) -> bool:
        """Prevent a valid older-origin proof from jumping past latest state."""

        kinematic_position = (
            expected_origin.kinematic_position
            or expected_origin.position
        )
        latest_media_pts_ns = int(
            expected_origin.kinematic_media_pts_ns
            if expected_origin.kinematic_media_pts_ns is not None
            else expected_origin.media_pts_ns
        )
        if current_media_pts_ns <= latest_media_pts_ns:
            return False
        latest_dt_s = (
            current_media_pts_ns - latest_media_pts_ns
        ) / 1_000_000_000.0
        latest_gate_dt_s = min(
            latest_dt_s,
            cls.IMAGE_MOTION_RESET_AFTER_S,
        )
        latest_step_m = math.hypot(
            world_point[0] - kinematic_position[0],
            world_point[2] - kinematic_position[2],
        )
        speed_limit_mps = (
            cls.IMAGE_MOTION_MAX_SPEED_MPS
            if max_speed_mps is None
            else float(max_speed_mps)
        )
        return (
            math.isfinite(speed_limit_mps)
            and 0.0 < speed_limit_mps <= cls.IMAGE_MOTION_MAX_SPEED_MPS
            and math.isfinite(latest_step_m)
            and latest_step_m
            <= speed_limit_mps * latest_gate_dt_s + 1e-9
        )

    @classmethod
    def _latest_visible_output_step_is_physical(
        cls,
        expected_origin: _CanonicalTrackOutput,
        *,
        world_point: tuple[float, float, float],
        current_media_pts_ns: int,
        max_speed_mps: float | None = None,
    ) -> bool:
        """Bound motion against the most recent actually displayed point.

        Gain-zero holds deliberately retain an older kinematic clock so a
        later observation can still be evaluated against the real motion
        interval. That proof budget must not become a one-frame display
        teleport. The producer reduces its canonical filter gain to satisfy
        this second bound; the service independently verifies the emitted
        coordinate from its own latest committed output.
        """

        latest_media_pts_ns = int(expected_origin.media_pts_ns)
        if current_media_pts_ns <= latest_media_pts_ns:
            return False
        visible_dt_s = (
            current_media_pts_ns - latest_media_pts_ns
        ) / 1_000_000_000.0
        visible_gate_dt_s = min(
            visible_dt_s,
            cls.IMAGE_MOTION_RESET_AFTER_S,
        )
        visible_step_m = math.hypot(
            world_point[0] - expected_origin.position[0],
            world_point[2] - expected_origin.position[2],
        )
        speed_limit_mps = (
            cls.IMAGE_MOTION_MAX_SPEED_MPS
            if max_speed_mps is None
            else float(max_speed_mps)
        )
        return (
            math.isfinite(speed_limit_mps)
            and 0.0 < speed_limit_mps <= cls.IMAGE_MOTION_MAX_SPEED_MPS
            and math.isfinite(visible_step_m)
            and visible_step_m
            <= speed_limit_mps * visible_gate_dt_s + 1e-9
        )

    @classmethod
    def _image_motion_filter_transition_is_canonical(
        cls,
        track: Mapping[str, Any],
        provenance: Mapping[str, Any],
        *,
        world_point: tuple[float, float, float],
        expected_origin: _CanonicalTrackOutput | None,
    ) -> bool:
        """Verify the exact queue-origin -> weak-filter posterior algebra."""

        if expected_origin is None:
            return False
        support_valid, support_surface = cls._continuation_support_surface(
            track, expected_origin,
        )
        if not support_valid:
            return False
        process_observation = provenance.get("process_observation")
        transition = provenance.get("filter_transition")
        if (
            not isinstance(process_observation, (list, tuple))
            or len(process_observation) < 3
            or not isinstance(transition, Mapping)
        ):
            return False
        origin_world = transition.get("origin_world")
        position_base = transition.get("position_base")
        if (
            not isinstance(origin_world, (list, tuple))
            or len(origin_world) < 3
            or not isinstance(position_base, (list, tuple))
            or len(position_base) < 3
        ):
            return False
        try:
            process_point = tuple(
                float(value) for value in process_observation[:3]
            )
            origin_point = tuple(float(value) for value in origin_world[:3])
            base_point = tuple(float(value) for value in position_base[:3])
            version = int(transition.get("version"))
            origin_media_pts_ns = int(
                transition.get("origin_media_pts_ns")
            )
            current_media_pts_ns = int(
                transition.get("current_media_pts_ns")
            )
            track_media_pts_ns = int(track.get("media_pts_ns"))
            origin_trail_segment_id = int(
                transition.get("origin_trail_segment_id")
            )
            track_trail_segment_id = int(track.get("trail_segment_id"))
            gate_dt_s = float(transition.get("gate_dt_s"))
            position_gain = float(transition.get("position_gain"))
            max_speed_mps = float(transition.get("max_speed_mps"))
            max_jump_m = float(transition.get("max_jump_m"))
            reset_after_s = float(transition.get("reset_after_s"))
        except (TypeError, ValueError, OverflowError):
            return False
        filter_point = process_point
        innovation_limit_applied = bool(
            transition.get("innovation_limit_applied") is True
        )
        raw_position_target = transition.get("raw_position_target")
        filter_observation = transition.get("filter_observation")
        innovation_scale = math.nan
        proof_raw_innovation_m = math.nan
        proof_innovation_limit_m = math.nan
        if innovation_limit_applied:
            if (
                not isinstance(raw_position_target, (list, tuple))
                or len(raw_position_target) < 3
                or not isinstance(filter_observation, (list, tuple))
                or len(filter_observation) < 3
            ):
                return False
            try:
                proof_raw_target = tuple(
                    float(value) for value in raw_position_target[:3]
                )
                filter_point = tuple(
                    float(value) for value in filter_observation[:3]
                )
                innovation_scale = float(transition.get("innovation_scale"))
                proof_raw_innovation_m = float(
                    transition.get("raw_innovation_m")
                )
                proof_innovation_limit_m = float(
                    transition.get("innovation_limit_m")
                )
            except (TypeError, ValueError, OverflowError):
                return False
            if not all(
                math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
                for left, right in zip(proof_raw_target, process_point)
            ):
                return False
        elif (
            raw_position_target is not None
            or filter_observation is not None
            or transition.get("innovation_scale") is not None
            or transition.get("raw_innovation_m") is not None
            or transition.get("innovation_limit_m") is not None
        ):
            # The alternate target is meaningful only as one complete robust
            # innovation proof.  Partial fields must not change the posterior
            # algebra selected by the producer.
            return False

        vectors = (
            world_point,
            process_point,
            filter_point,
            origin_point,
            base_point,
        )
        if not all(
            math.isfinite(value)
            for point in vectors
            for value in point
        ):
            return False
        if (
            version != cls.IMAGE_MOTION_TRANSITION_VERSION
            or transition.get("origin_kind")
            != "queue_admitted_world_output"
            or not math.isclose(
                max_speed_mps,
                cls.IMAGE_MOTION_MAX_SPEED_MPS,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                max_jump_m,
                cls.IMAGE_MOTION_MAX_JUMP_M,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                reset_after_s,
                cls.IMAGE_MOTION_RESET_AFTER_S,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            return False
        if (
            origin_media_pts_ns <= 0
            or current_media_pts_ns <= origin_media_pts_ns
            or current_media_pts_ns != track_media_pts_ns
            or origin_trail_segment_id != track_trail_segment_id
        ):
            return False
        matched_origin = cls._matched_committed_output_origin(
            expected_origin,
            origin_point=origin_point,
            origin_media_pts_ns=origin_media_pts_ns,
            origin_trail_segment_id=origin_trail_segment_id,
            current_media_pts_ns=current_media_pts_ns,
            maximum_age_s=cls.IMAGE_MOTION_COMMITTED_ORIGIN_HISTORY_HORIZON_S,
        )
        if matched_origin is None or matched_origin.support_surface != support_surface:
            return False
        if support_surface is not None:
            if not all(
                cls._point_matches_support_plane(support_surface, point)
                for point in (
                    world_point, process_point, filter_point, origin_point,
                    base_point, expected_origin.position,
                )
            ):
                return False
        elif not all(
            math.isclose(point[1], world_point[1], rel_tol=0.0, abs_tol=1e-9)
            for point in (process_point, origin_point, base_point)
        ):
            return False

        media_dt_s = (
            current_media_pts_ns - origin_media_pts_ns
        ) / 1_000_000_000.0
        expected_gate_dt_s = min(
            media_dt_s,
            cls.IMAGE_MOTION_RESET_AFTER_S,
        )
        if (
            not math.isfinite(gate_dt_s)
            or gate_dt_s <= 0.0
            or not math.isclose(
                gate_dt_s,
                expected_gate_dt_s,
                rel_tol=0.0,
                abs_tol=cls.IMAGE_MOTION_GATE_DT_ABS_TOL_S,
            )
        ):
            return False

        base_step_m = math.hypot(
            base_point[0] - origin_point[0],
            base_point[2] - origin_point[2],
        )
        raw_process_innovation_m = math.hypot(
            process_point[0] - base_point[0],
            process_point[2] - base_point[2],
        )
        filter_innovation_m = math.hypot(
            filter_point[0] - base_point[0],
            filter_point[2] - base_point[2],
        )
        innovation_limit_m = (
            cls.IMAGE_MOTION_MAX_JUMP_M
            + cls.IMAGE_MOTION_MAX_SPEED_MPS * gate_dt_s
        )
        posterior_step_m = math.hypot(
            world_point[0] - origin_point[0],
            world_point[2] - origin_point[2],
        )
        if innovation_limit_applied:
            expected_scale = innovation_limit_m / raw_process_innovation_m
            expected_filter_point = (
                base_point[0]
                + expected_scale * (process_point[0] - base_point[0]),
                (
                    cls._support_plane_elevation(
                        support_surface,
                        base_point[0] + expected_scale * (process_point[0] - base_point[0]),
                        base_point[2] + expected_scale * (process_point[2] - base_point[2]),
                    )
                    if support_surface is not None else base_point[1]
                ),
                base_point[2]
                + expected_scale * (process_point[2] - base_point[2]),
            )
            if (
                raw_process_innovation_m <= innovation_limit_m + 1e-9
                or not (0.0 < innovation_scale < 1.0)
                or not math.isclose(
                    proof_raw_innovation_m,
                    raw_process_innovation_m,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    proof_innovation_limit_m,
                    innovation_limit_m,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    innovation_scale,
                    expected_scale,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or not all(
                    math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
                    for left, right in zip(
                        filter_point,
                        expected_filter_point,
                    )
                )
            ):
                return False
        if (
            base_step_m
            > cls.IMAGE_MOTION_MAX_SPEED_MPS * gate_dt_s + 1e-9
            or filter_innovation_m > innovation_limit_m + 1e-9
            or posterior_step_m
            > cls.IMAGE_MOTION_MAX_SPEED_MPS * gate_dt_s + 1e-9
        ):
            return False
        if not cls._latest_output_step_is_physical(
            expected_origin,
            world_point=world_point,
            current_media_pts_ns=current_media_pts_ns,
        ):
            return False
        if not cls._latest_visible_output_step_is_physical(
            expected_origin,
            world_point=world_point,
            current_media_pts_ns=current_media_pts_ns,
        ):
            return False

        transition_kind = str(transition.get("kind") or "")
        if transition_kind == "innovation_update":
            if not (0.0 < position_gain <= 1.0):
                return False
        elif transition_kind == "deadzone_hold":
            if not math.isclose(
                position_gain,
                0.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                return False
        else:
            return False
        expected_posterior = (
            base_point[0]
            + position_gain * (filter_point[0] - base_point[0]),
            (
                cls._support_plane_elevation(
                    support_surface,
                    base_point[0] + position_gain * (filter_point[0] - base_point[0]),
                    base_point[2] + position_gain * (filter_point[2] - base_point[2]),
                )
                if support_surface is not None else base_point[1]
            ),
            base_point[2]
            + position_gain * (filter_point[2] - base_point[2]),
        )
        return all(
            math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
            for left, right in zip(world_point, expected_posterior)
        )

    @classmethod
    def _image_motion_observation_is_canonical(
        cls,
        track: Mapping[str, Any],
        *,
        expected_origin: _CanonicalTrackOutput | None = None,
    ) -> bool:
        """Validate one current calibrated image-foot continuation.

        This is not fresh metric evidence, but it is a current Noesis-owned
        floor coordinate. Admit it only with the complete fixed-origin
        projective proof emitted by the DS9 estimator so the canonical held
        snapshot consumed by Menon is exactly the same point as tracking/BEV.
        """

        if str(track.get("world_quality") or "").strip().lower() != "held":
            return False
        if track.get("world_measurement_accepted") is not False:
            return False
        provenance = track.get("world_prediction_provenance")
        if not isinstance(provenance, Mapping):
            return False
        if provenance.get("non_authoritative") is not True:
            return False
        if provenance.get("state_integrated") is not True:
            return False
        world = track.get("world")
        posterior = track.get("world_filter_prediction")
        if (
            not isinstance(world, (list, tuple))
            or len(world) < 3
            or not isinstance(posterior, (list, tuple))
            or len(posterior) < 3
        ):
            return False
        try:
            world_point = tuple(float(value) for value in world[:3])
            posterior_point = tuple(float(value) for value in posterior[:3])
        except (TypeError, ValueError, OverflowError):
            return False
        if not (
            all(math.isfinite(value) for value in world_point + posterior_point)
            and all(
                math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
                for left, right in zip(world_point, posterior_point)
            )
        ):
            return False
        if not cls._image_motion_filter_transition_is_canonical(
            track,
            provenance,
            world_point=world_point,
            expected_origin=expected_origin,
        ):
            return False
        provenance_type = str(provenance.get("type") or "")
        if provenance_type == "bbox_affine_floor_projection":
            if provenance.get("origin") not in {
                "last_accepted_image_foot",
                "last_accepted_pose_projective_origin",
            }:
                return False
            if provenance.get("transport") not in {
                "bbox_affine",
                "pose_torso_translation",
            }:
                return False
            image_foot = provenance.get("image_foot")
            projective_origin_world = provenance.get(
                "projective_origin_world"
            )
            process_observation = provenance.get("process_observation")
            if not isinstance(image_foot, (list, tuple)) or len(image_foot) != 2:
                return False
            if (
                not isinstance(projective_origin_world, (list, tuple))
                or len(projective_origin_world) < 3
                or not isinstance(process_observation, (list, tuple))
                or len(process_observation) < 3
            ):
                return False
            try:
                values = tuple(float(value) for value in image_foot)
                projective_origin_point = tuple(
                    float(value) for value in projective_origin_world[:3]
                )
                process_point = tuple(
                    float(value) for value in process_observation[:3]
                )
                age_s = float(provenance.get("age_s"))
                world_delta_m = float(provenance.get("world_delta_m"))
                world_delta_limit_m = float(
                    provenance.get("world_delta_limit_m")
                )
            except (TypeError, ValueError, OverflowError):
                return False
            expected_world_delta_m = math.hypot(
                process_point[0] - projective_origin_point[0],
                process_point[2] - projective_origin_point[2],
            )
            expected_world_delta_limit_m = (
                cls.IMAGE_MOTION_MAX_SPEED_MPS * age_s
                + cls.IMAGE_MOTION_PROJECTIVE_SLACK_M
            )
            _, support_surface = cls._continuation_support_surface(
                track, expected_origin,
            )
            return bool(
                all(
                    math.isfinite(value)
                    for value in (
                        *values,
                        *projective_origin_point,
                        *process_point,
                    )
                )
                and math.isfinite(age_s)
                and 0.0 <= age_s <= 5.0
                and math.isfinite(world_delta_m)
                and math.isfinite(world_delta_limit_m)
                and math.isclose(
                    world_delta_m,
                    expected_world_delta_m,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                and math.isclose(
                    world_delta_limit_m,
                    expected_world_delta_limit_m,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                and 0.0 <= world_delta_m <= world_delta_limit_m + 1e-9
                and (
                    support_surface is None
                    or cls._point_matches_support_plane(
                        support_surface, projective_origin_point,
                    )
                )
            )

        if provenance_type != "inferred_ground_process_observation":
            return False
        # Learned body height is useful only as current-cohort process evidence
        # for an already-established standing lifecycle. It is authorized by
        # either explicit lower-body occlusion or three coherent exact-current
        # torso observations on a confident detector/tracker row. It never
        # becomes a metric measurement. Bind the proof to the current cohort
        # and require the DS9 filter posterior to equal the published
        # tracking/BEV point exactly before Menon sees it.
        if provenance.get("origin") != "learned_body_height":
            return False
        support_kind = str(provenance.get("support_kind") or "")
        support_is_explicit_occlusion = support_kind == "lower_body_occlusion"
        support_is_coherent_torso = support_kind == "coherent_torso_motion"
        if support_is_explicit_occlusion:
            if (
                provenance.get("transport")
                != "fixed_occlusion_origin_raw_world_delta"
            ):
                return False
        elif support_is_coherent_torso:
            if (
                provenance.get("transport")
                != "fixed_torso_origin_raw_world_delta"
            ):
                return False
        else:
            return False
        if (
            provenance.get("trusted_origin_kind")
            != "queue_admitted_metric_world_output"
        ):
            return False
        if provenance.get("resolver_candidate_id") != "gravity_reconstruction":
            return False
        if provenance.get("raw_consensus_method") != "xz_medoid":
            return False
        occlusion_level = str(provenance.get("occlusion_level") or "")
        if support_is_explicit_occlusion:
            if track.get("lower_body_occluded") is not True:
                return False
            if (
                not occlusion_level
                or occlusion_level == "none"
                or occlusion_level
                != str(track.get("lower_body_occlusion_level") or "")
            ):
                return False
        else:
            try:
                image_motion_streak = int(
                    provenance.get("image_motion_streak")
                )
                track_image_motion_streak = int(
                    track.get("world_image_motion_streak")
                )
                detector_confidence = float(
                    provenance.get("detector_confidence")
                )
                track_detector_confidence = max(
                    float(track.get("confidence", 0.0) or 0.0),
                    float(track.get("tracker_confidence", 0.0) or 0.0),
                )
            except (TypeError, ValueError, OverflowError):
                return False
            track_detector_confidence = min(1.0, track_detector_confidence)
            if not (
                track.get("lower_body_occluded") is False
                and occlusion_level == "none"
                and str(track.get("lower_body_occlusion_level") or "")
                == "none"
                and provenance.get("image_motion_supported") is True
                and track.get("world_image_motion_supported") is True
                and provenance.get("image_motion_contact_basis")
                == "pose:torso_motion"
                and track.get("world_contact_basis") == "pose:torso_motion"
                and image_motion_streak
                >= cls.INFERRED_PROCESS_MIN_IMAGE_MOTION_STREAK
                and image_motion_streak == track_image_motion_streak
                and math.isfinite(detector_confidence)
                and math.isfinite(track_detector_confidence)
                and detector_confidence
                >= cls.INFERRED_PROCESS_MIN_DETECTOR_CONFIDENCE
                and math.isclose(
                    detector_confidence,
                    track_detector_confidence,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                return False
        if str(track.get("posture") or "") != "standing":
            return False
        if str(track.get("motion_mode") or "") in {"sit", "lie"}:
            return False
        if track.get("world_observation_range_admitted") is not True:
            return False
        raw_sample = provenance.get("raw_sample")
        raw_consensus = provenance.get("raw_consensus")
        raw_origin = provenance.get("raw_origin")
        raw_delta = provenance.get("raw_delta")
        trusted_world_origin = provenance.get("trusted_world_origin")
        process_observation = provenance.get("process_observation")
        track_raw = track.get("world_inferred_raw_observation")
        track_process = track.get("world_inferred_process_observation")
        if (
            any(
                not isinstance(value, (list, tuple)) or len(value) < 3
                for value in (
                    raw_sample,
                    raw_consensus,
                    raw_origin,
                    raw_delta,
                    trusted_world_origin,
                    process_observation,
                    track_raw,
                    track_process,
                )
            )
        ):
            return False
        episode_root = cls._inferred_ground_episode_root_from_provenance(
            provenance
        )
        if episode_root is None:
            return False
        established_episode_root = (
            expected_origin.inferred_ground_episode_root
            if expected_origin is not None
            else None
        )
        if (
            established_episode_root is not None
            and episode_root != established_episode_root
        ):
            return False
        try:
            raw_sample_point = tuple(float(value) for value in raw_sample[:3])
            raw_consensus_point = tuple(
                float(value) for value in raw_consensus[:3]
            )
            raw_origin_point = tuple(float(value) for value in raw_origin[:3])
            raw_delta_point = tuple(float(value) for value in raw_delta[:3])
            trusted_origin_point = tuple(
                float(value) for value in trusted_world_origin[:3]
            )
            process_point = tuple(float(value) for value in process_observation[:3])
            track_raw_point = tuple(float(value) for value in track_raw[:3])
            track_process_point = tuple(float(value) for value in track_process[:3])
            age_s = float(provenance.get("age_s"))
            raw_origin_ts_s = float(provenance.get("raw_origin_ts_s"))
            raw_consensus_ts_s = float(
                provenance.get("raw_consensus_ts_s")
            )
            current_ts_s = float(provenance.get("current_ts_s"))
            trusted_origin_filter_ts_s = float(
                provenance.get("trusted_origin_filter_ts_s")
            )
            height_ref_scene = float(provenance.get("height_ref_scene"))
            occlusion_confidence = float(
                provenance.get("occlusion_confidence")
            )
            track_occlusion_confidence = float(
                track.get("lower_body_occlusion_confidence")
            )
            observed_at_us = int(provenance.get("observed_at_us"))
            origin_observed_at_us = int(
                provenance.get("origin_observed_at_us")
            )
            raw_consensus_observed_at_us = int(
                provenance.get("raw_consensus_observed_at_us")
            )
            raw_origin_media_pts_ns = int(
                provenance.get("raw_origin_media_pts_ns")
            )
            raw_consensus_media_pts_ns = int(
                provenance.get("raw_consensus_media_pts_ns")
            )
            trusted_origin_media_pts_ns = int(
                provenance.get("trusted_origin_media_pts_ns")
            )
            current_media_pts_ns = int(
                provenance.get("current_media_pts_ns")
            )
            raw_consensus_count = int(
                provenance.get("raw_consensus_count")
            )
            raw_consensus_span_s = float(
                provenance.get("raw_consensus_span_s")
            )
            raw_evidence_gap_s = float(
                provenance.get("raw_evidence_gap_s")
            )
            lifecycle_generation = int(
                provenance.get("tracker_lifecycle_generation")
            )
            origin_trail_segment_id = int(
                provenance.get("origin_trail_segment_id")
            )
            track_observed_at_us = int(track.get("observed_at_us"))
            track_media_pts_ns = int(track.get("media_pts_ns"))
            track_lifecycle_generation = int(
                track.get("tracker_lifecycle_generation")
            )
            track_trail_segment_id = int(track.get("trail_segment_id"))
        except (TypeError, ValueError, OverflowError):
            return False
        proof_vectors = (
            raw_sample_point,
            raw_consensus_point,
            raw_origin_point,
            raw_delta_point,
            trusted_origin_point,
            process_point,
            track_raw_point,
            track_process_point,
            world_point,
            posterior_point,
        )
        raw_delta_matches = all(
            math.isclose(
                raw_delta_point[index],
                raw_consensus_point[index] - raw_origin_point[index],
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for index in range(3)
        )
        process_algebra_matches = all(
            math.isclose(
                process_point[index],
                trusted_origin_point[index] + raw_delta_point[index],
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for index in range(3)
        )
        expected_metric_origin_matches = bool(
            expected_origin is not None
            and expected_origin.metric_position is not None
            and expected_origin.metric_media_pts_ns is not None
            and expected_origin.metric_trail_segment_id is not None
            and all(
                math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
                for left, right in zip(
                    trusted_origin_point,
                    expected_origin.metric_position,
                )
            )
            and trusted_origin_media_pts_ns
            == int(expected_origin.metric_media_pts_ns)
            and origin_trail_segment_id
            == int(expected_origin.metric_trail_segment_id)
        )
        recent_consensus_evidence = bool(
            math.isfinite(raw_evidence_gap_s)
            and 0.0
            <= raw_evidence_gap_s
            <= cls.INFERRED_PROCESS_CONSENSUS_HORIZON_S + 1e-9
        )
        restart_origin = (
            cls._matched_inferred_ground_restart_origin(
                expected_origin,
                provenance,
                current_media_pts_ns=current_media_pts_ns,
            )
            if (
                expected_origin is not None
                and raw_evidence_gap_s
                > cls.INFERRED_PROCESS_CONSENSUS_HORIZON_S + 1e-9
            )
            else None
        )
        restart_age_ns = int(
            round(
                cls.INFERRED_PROCESS_RESTART_MAX_AGE_S
                * 1_000_000_000.0
            )
        )
        exact_root_restart = bool(
            expected_origin is not None
            and established_episode_root is not None
            and episode_root == established_episode_root
            and restart_origin is not None
            and restart_origin.inferred_ground_episode_root == episode_root
            and math.isfinite(raw_evidence_gap_s)
            and cls.INFERRED_PROCESS_CONSENSUS_HORIZON_S
            < raw_evidence_gap_s
            <= cls.INFERRED_PROCESS_RESTART_MAX_AGE_S + 1e-9
            and 0
            < current_media_pts_ns - int(expected_origin.media_pts_ns)
            <= restart_age_ns
        )
        return bool(
            all(
                math.isfinite(value)
                for point in proof_vectors
                for value in point
            )
            and raw_delta_matches
            and process_algebra_matches
            and expected_metric_origin_matches
            and all(
                math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
                for left, right in zip(raw_consensus_point, track_raw_point)
            )
            and all(
                math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
                for left, right in zip(process_point, track_process_point)
            )
            and math.isfinite(age_s)
            and age_s >= 0.0
            and math.isfinite(raw_origin_ts_s)
            and math.isfinite(raw_consensus_ts_s)
            and math.isfinite(current_ts_s)
            and math.isfinite(trusted_origin_filter_ts_s)
            and raw_origin_ts_s >= trusted_origin_filter_ts_s
            and raw_origin_ts_s <= raw_consensus_ts_s
            and raw_consensus_ts_s <= current_ts_s
            and current_ts_s >= raw_origin_ts_s
            and math.isclose(
                age_s,
                current_ts_s - raw_origin_ts_s,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and raw_evidence_gap_s <= age_s + 1e-9
            and math.isfinite(height_ref_scene)
            and height_ref_scene > 0.0
            and (
                (
                    support_is_explicit_occlusion
                    and math.isfinite(occlusion_confidence)
                    and 0.75 <= occlusion_confidence <= 1.0
                    and math.isfinite(track_occlusion_confidence)
                    and math.isclose(
                        occlusion_confidence,
                        track_occlusion_confidence,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                )
                or (
                    support_is_coherent_torso
                    and math.isfinite(occlusion_confidence)
                    and math.isclose(
                        occlusion_confidence,
                        0.0,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                    and math.isfinite(track_occlusion_confidence)
                    and math.isclose(
                        track_occlusion_confidence,
                        0.0,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                )
            )
            and origin_observed_at_us > 0
            and raw_consensus_observed_at_us > 0
            and observed_at_us > 0
            and origin_observed_at_us <= raw_consensus_observed_at_us
            and raw_consensus_observed_at_us <= observed_at_us
            and observed_at_us == track_observed_at_us
            and raw_origin_media_pts_ns > 0
            and raw_consensus_media_pts_ns > 0
            and trusted_origin_media_pts_ns > 0
            and current_media_pts_ns > 0
            and trusted_origin_media_pts_ns <= raw_origin_media_pts_ns
            and raw_origin_media_pts_ns <= raw_consensus_media_pts_ns
            and raw_consensus_media_pts_ns <= current_media_pts_ns
            and current_media_pts_ns == track_media_pts_ns
            and 1 <= raw_consensus_count <= 5
            and math.isfinite(raw_consensus_span_s)
            and 0.0
            <= raw_consensus_span_s
            <= cls.INFERRED_PROCESS_CONSENSUS_HORIZON_S + 1e-9
            and (recent_consensus_evidence or exact_root_restart)
            and current_ts_s - raw_consensus_ts_s
            <= cls.INFERRED_PROCESS_CONSENSUS_HORIZON_S + 1e-9
            and lifecycle_generation > 0
            and lifecycle_generation == track_lifecycle_generation
            and origin_trail_segment_id >= 0
            and origin_trail_segment_id == track_trail_segment_id
            and provenance.get("world_frame") == track.get("world_frame")
            and provenance.get("world_frame_revision")
            == track.get("world_frame_revision")
            and provenance.get("world_transform_sha256")
            == track.get("world_transform_sha256")
            and all(
                math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
                for left, right in zip(world_point, posterior_point)
            )
        )

    @classmethod
    def _current_presence_evidence_is_canonical(
        cls,
        track: Mapping[str, Any],
        evidence: Any,
        *,
        expected_origin: _CanonicalTrackOutput,
        current_media_pts_ns: int,
    ) -> bool:
        """Validate one fixed-root current-person bbox continuity proof."""

        if not isinstance(evidence, Mapping):
            return False
        root_bbox = expected_origin.kinematic_bbox_geometry
        root_media_pts_ns = expected_origin.kinematic_media_pts_ns
        root_segment = expected_origin.kinematic_trail_segment_id
        root_position = expected_origin.kinematic_position
        track_bbox = cls._bbox_geometry(track.get("bbox"))
        evidence_root_bbox = cls._bbox_geometry(
            evidence.get("kinematic_origin_bbox")
        )
        evidence_current_bbox = cls._bbox_geometry(
            evidence.get("current_bbox")
        )
        if (
            root_bbox is None
            or root_media_pts_ns is None
            or root_segment is None
            or root_position is None
            or track_bbox is None
            or evidence_root_bbox is None
            or evidence_current_bbox is None
        ):
            return False
        try:
            class_id = int(track.get("class_id"))
            frame_id = int(track.get("frame_id"))
            evidence_frame_id = int(evidence.get("frame_id"))
            evidence_media_pts_ns = int(evidence.get("media_pts_ns"))
            evidence_root_media_pts_ns = int(
                evidence.get("kinematic_origin_media_pts_ns")
            )
            evidence_root_segment = int(
                evidence.get("kinematic_origin_trail_segment_id")
            )
            detector_confidence = float(track.get("confidence"))
            tracker_confidence = float(track.get("tracker_confidence"))
            evidence_detector_confidence = float(
                evidence.get("detector_confidence")
            )
            evidence_tracker_confidence = float(
                evidence.get("tracker_confidence")
            )
            evidence_size_ratio = float(evidence.get("bbox_size_ratio"))
            evidence_displacement_norm = float(
                evidence.get("bbox_center_displacement_norm")
            )
            root_left, root_top, root_width, root_height = (
                float(value) for value in root_bbox
            )
            current_left, current_top, current_width, current_height = (
                float(value) for value in track_bbox
            )
            size_ratio = max(
                root_width / current_width,
                current_width / root_width,
                root_height / current_height,
                current_height / root_height,
            )
            root_center = (
                root_left + 0.5 * root_width,
                root_top + 0.5 * root_height,
            )
            current_center = (
                current_left + 0.5 * current_width,
                current_top + 0.5 * current_height,
            )
            displacement_px = math.hypot(
                current_center[0] - root_center[0],
                current_center[1] - root_center[1],
            )
            bbox_scale_px = 0.5 * (
                math.hypot(root_width, root_height)
                + math.hypot(current_width, current_height)
            )
            displacement_norm = displacement_px / bbox_scale_px
        except (
            TypeError,
            ValueError,
            ZeroDivisionError,
            OverflowError,
        ):
            return False
        horizon_ns = int(
            round(
                (
                    cls.CURRENT_PRESENCE_HOLD_HORIZON_S
                    + cls.BOUNDED_PROCESS_HORIZON_EPSILON_S
                )
                * 1_000_000_000.0
            )
        )
        pose_present = track.get("pose_present") is True
        return bool(
            evidence.get("version") == 1
            and evidence.get("kind")
            == "same_lifecycle_bbox_from_kinematic_output"
            and class_id == 0
            and frame_id >= 0
            and evidence_frame_id == frame_id
            and evidence_media_pts_ns == current_media_pts_ns
            and evidence_root_media_pts_ns == int(root_media_pts_ns)
            and evidence_root_segment == int(root_segment)
            and 0
            < current_media_pts_ns - int(root_media_pts_ns)
            <= horizon_ns
            and evidence.get("pose_present") is pose_present
            and all(
                math.isfinite(value)
                for value in (
                    detector_confidence,
                    tracker_confidence,
                    evidence_detector_confidence,
                    evidence_tracker_confidence,
                    size_ratio,
                    displacement_norm,
                    evidence_size_ratio,
                    evidence_displacement_norm,
                    bbox_scale_px,
                )
            )
            and 0.0 <= detector_confidence <= 1.0
            and detector_confidence
            >= cls.CURRENT_PRESENCE_MIN_DETECTOR_CONFIDENCE
            and tracker_confidence
            >= cls.CURRENT_PRESENCE_MIN_TRACKER_CONFIDENCE
            and (
                pose_present
                or detector_confidence
                >= cls.CURRENT_PRESENCE_STRONG_DETECTOR_CONFIDENCE
            )
            and math.isclose(
                evidence_detector_confidence,
                detector_confidence,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and math.isclose(
                evidence_tracker_confidence,
                tracker_confidence,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and all(
                math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
                for left, right in zip(track_bbox, evidence_current_bbox)
            )
            and all(
                math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
                for left, right in zip(root_bbox, evidence_root_bbox)
            )
            and all(
                math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
                for left, right in zip(
                    root_position,
                    expected_origin.position,
                )
            )
            and size_ratio <= cls.CURRENT_PRESENCE_MAX_BBOX_SIZE_RATIO
            and displacement_norm
            <= cls.CURRENT_PRESENCE_MAX_CENTER_DISPLACEMENT_NORM
            and math.isclose(
                evidence_size_ratio,
                size_ratio,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and math.isclose(
                evidence_displacement_norm,
                displacement_norm,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )

    @classmethod
    def _bounded_process_observation_is_canonical(
        cls,
        track: Mapping[str, Any],
        *,
        expected_origin: _CanonicalTrackOutput | None = None,
    ) -> bool:
        """Validate one exact bounded process continuation.

        CV predictions and stationary anchor holds are not new metric camera
        measurements.  They are nevertheless the current Noesis-owned world
        point after the DS9 physical filter has integrated and bounded the
        process state.  Admit only the complete proof emitted by that filter,
        including exact agreement between its posterior diagnostic and the
        coordinate carried by the tracking row.  This gives Menon the same
        held point as tracking/BEV without allowing an arbitrary display-only
        payload to masquerade as canonical world state.
        """

        if expected_origin is None:
            return False
        support_valid, support_surface = cls._continuation_support_surface(
            track, expected_origin,
        )
        if not support_valid:
            return False
        source = str(track.get("world_source") or "").strip().lower()
        if source not in {"cv_prediction", "anchor_hold"}:
            return False
        if str(track.get("world_quality") or "").strip().lower() != "held":
            return False
        if track.get("world_measurement_accepted") is not False:
            return False
        provenance = track.get("world_prediction_provenance")
        if not isinstance(provenance, Mapping):
            return False
        provenance_type = provenance.get("type")
        provenance_origin = provenance.get("origin")
        if provenance_type == "bounded_cv_process":
            if provenance_origin not in {
                "recent_projective_process",
                "last_metric_process",
            }:
                return False
        elif provenance_type == "bounded_output_hold":
            if source != "anchor_hold" or provenance_origin != "last_published_output":
                return False
        else:
            return False
        if provenance.get("non_authoritative") is not True:
            return False
        if provenance.get("state_integrated") is not True:
            return False
        reason = provenance.get("reason")
        quality_reason = track.get("world_quality_reason")
        if (
            not isinstance(reason, str)
            or not reason.strip()
            or len(reason) > 240
            or not isinstance(quality_reason, str)
            or not quality_reason.strip()
            or len(quality_reason) > 240
            or reason != quality_reason
        ):
            return False
        world = track.get("world")
        posterior = track.get("world_filter_prediction")
        process_observation = provenance.get("process_observation")
        transition = provenance.get("filter_transition")
        if (
            not isinstance(world, (list, tuple))
            or len(world) < 3
            or not isinstance(posterior, (list, tuple))
            or len(posterior) < 3
            or not isinstance(process_observation, (list, tuple))
            or len(process_observation) < 3
            or not isinstance(transition, Mapping)
        ):
            return False
        origin_world = transition.get("origin_world")
        metric_origin_world = transition.get("metric_origin_world")
        position_base = transition.get("position_base")
        if (
            not isinstance(origin_world, (list, tuple))
            or len(origin_world) < 3
            or not isinstance(metric_origin_world, (list, tuple))
            or len(metric_origin_world) < 3
            or not isinstance(position_base, (list, tuple))
            or len(position_base) < 3
        ):
            return False
        try:
            world_point = tuple(float(value) for value in world[:3])
            posterior_point = tuple(float(value) for value in posterior[:3])
            process_point = tuple(
                float(value) for value in process_observation[:3]
            )
            origin_point = tuple(float(value) for value in origin_world[:3])
            metric_origin_point = tuple(
                float(value) for value in metric_origin_world[:3]
            )
            base_point = tuple(float(value) for value in position_base[:3])
            version = int(transition.get("version"))
            origin_media_pts_ns = int(transition.get("origin_media_pts_ns"))
            current_media_pts_ns = int(transition.get("current_media_pts_ns"))
            track_media_pts_ns = int(track.get("media_pts_ns"))
            origin_trail_segment_id = int(
                transition.get("origin_trail_segment_id")
            )
            metric_origin_media_pts_ns = int(
                transition.get("metric_origin_media_pts_ns")
            )
            metric_origin_trail_segment_id = int(
                transition.get("metric_origin_trail_segment_id")
            )
            track_trail_segment_id = int(track.get("trail_segment_id"))
            lifecycle_generation = int(
                transition.get("tracker_lifecycle_generation")
            )
            track_lifecycle_generation = int(
                track.get("tracker_lifecycle_generation")
            )
            gate_dt_s = float(transition.get("gate_dt_s"))
            position_gain = float(transition.get("position_gain"))
            max_speed_mps = float(transition.get("max_speed_mps"))
            reset_after_s = float(transition.get("reset_after_s"))
        except (TypeError, ValueError, OverflowError):
            return False
        vectors = (
            world_point,
            posterior_point,
            process_point,
            origin_point,
            metric_origin_point,
            base_point,
        )
        if not all(math.isfinite(value) for point in vectors for value in point):
            return False
        if (
            version != 1
            or transition.get("origin_kind")
            != "queue_admitted_world_output"
            or transition.get("metric_origin_kind")
            != "queue_admitted_metric_world_output"
            or not math.isfinite(max_speed_mps)
            or not (0.0 < max_speed_mps <= cls.IMAGE_MOTION_MAX_SPEED_MPS)
            or not math.isclose(
                reset_after_s,
                cls.IMAGE_MOTION_RESET_AFTER_S,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            return False
        expected_metric_position = expected_origin.metric_position
        expected_metric_pts = expected_origin.metric_media_pts_ns
        expected_metric_segment = expected_origin.metric_trail_segment_id
        if (
            expected_metric_position is None
            or expected_metric_pts is None
            or expected_metric_segment is None
        ):
            return False
        stationary_evidence = provenance.get("stationary_evidence")
        # Stationary evidence is emitted from the tracking posture state and
        # binds that exact public field. ``world_posture`` is a resolver
        # diagnostic and may legitimately remain ``unknown`` while tracking
        # has classified the same row as sitting/lying; it must neither
        # suppress nor grant the longer exact-hold authority.
        posture = str(
            track.get("posture")
            or track.get("world_posture")
            or ""
        ).strip().lower()
        stationary_anchor_hold = bool(
            source == "anchor_hold"
            and provenance_type == "bounded_output_hold"
            and provenance_origin == "last_published_output"
            and isinstance(stationary_evidence, Mapping)
            and stationary_evidence.get("version") == 1
            and stationary_evidence.get("kind") == "bbox_stationary"
            and stationary_evidence.get("posture") == posture
            and posture in {"sitting", "lying"}
            and stationary_evidence.get("frame_id") == track.get("frame_id")
            and stationary_evidence.get("media_pts_ns")
            == current_media_pts_ns
            and track.get("world_bbox_stationary_supported") is True
        )
        upright_presence_evidence = provenance.get(
            "upright_presence_evidence"
        )
        upright_presence_hold = False
        if isinstance(upright_presence_evidence, Mapping):
            floor_candidate = track.get("world_floor_candidate")
            evidence_floor_candidate = upright_presence_evidence.get(
                "floor_candidate"
            )
            bbox = track.get("bbox")
            image_size = track.get("image_size")
            try:
                if (
                    not isinstance(floor_candidate, (list, tuple))
                    or len(floor_candidate) != 3
                    or not isinstance(evidence_floor_candidate, (list, tuple))
                    or len(evidence_floor_candidate) != 3
                ):
                    raise ValueError(
                        "upright presence floor candidates must be exact vectors"
                    )
                evidence_detector_confidence = float(
                    upright_presence_evidence.get("detector_confidence")
                )
                evidence_tracker_confidence = float(
                    upright_presence_evidence.get("tracker_confidence")
                )
                detector_confidence = float(track.get("confidence"))
                tracker_confidence = float(track.get("tracker_confidence"))
                candidate_point = tuple(
                    float(value) for value in floor_candidate[:3]
                )
                evidence_candidate_point = tuple(
                    float(value) for value in evidence_floor_candidate[:3]
                )
                output_distance_m = math.hypot(
                    candidate_point[0] - world_point[0],
                    candidate_point[2] - world_point[2],
                )
                evidence_output_distance_m = float(
                    upright_presence_evidence.get("output_distance_m")
                )
                evidence_output_distance_limit_m = float(
                    upright_presence_evidence.get(
                        "output_distance_limit_m"
                    )
                )
                bbox_width = float(bbox[2])
                bbox_height = float(bbox[3])
                image_height = float(image_size[1])
            except (
                TypeError,
                ValueError,
                IndexError,
                OverflowError,
            ):
                upright_presence_hold = False
            else:
                motion_mode = str(track.get("motion_mode") or "").strip().lower()
                upright_presence_hold = bool(
                    source == "anchor_hold"
                    and provenance_type == "bounded_output_hold"
                    and provenance_origin == "last_published_output"
                    and upright_presence_evidence.get("version") == 1
                    and upright_presence_evidence.get("kind")
                    == "pose_confirmed_nonseated_floor_near_output"
                    and upright_presence_evidence.get("frame_id")
                    == track.get("frame_id")
                    and upright_presence_evidence.get("media_pts_ns")
                    == current_media_pts_ns
                    and upright_presence_evidence.get("posture") == posture
                    and posture in {"standing", "unknown"}
                    and upright_presence_evidence.get("motion_mode")
                    == motion_mode
                    and motion_mode not in {"sit", "lie"}
                    and upright_presence_evidence.get("contact_basis")
                    == "pose:torso_motion"
                    and track.get("world_contact_basis")
                    == "pose:torso_motion"
                    and track.get("pose_present") is True
                    and track.get("lower_body_occluded") is False
                    and track.get("world_floor_admitted") is True
                    and track.get("world_floor_contact_plausible") is True
                    and track.get("world_observation_range_admitted") is True
                    and track.get("world_support_state") == "floor"
                    and all(
                        math.isfinite(value)
                        for value in (
                            detector_confidence,
                            tracker_confidence,
                            evidence_detector_confidence,
                            evidence_tracker_confidence,
                            *candidate_point,
                            *evidence_candidate_point,
                            output_distance_m,
                            evidence_output_distance_m,
                            evidence_output_distance_limit_m,
                            bbox_width,
                            bbox_height,
                            image_height,
                        )
                    )
                    and detector_confidence
                    >= cls.UPRIGHT_PRESENCE_MIN_DETECTOR_CONFIDENCE
                    and detector_confidence <= 1.0
                    and tracker_confidence
                    >= cls.UPRIGHT_PRESENCE_MIN_TRACKER_CONFIDENCE
                    and tracker_confidence <= 1.0
                    and 0.0 <= evidence_detector_confidence <= 1.0
                    and 0.0 <= evidence_tracker_confidence <= 1.0
                    and math.isclose(
                        evidence_detector_confidence,
                        detector_confidence,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                    and math.isclose(
                        evidence_tracker_confidence,
                        tracker_confidence,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                    and bbox_width > 0.0
                    and bbox_height > 0.0
                    and image_height > 0.0
                    and bbox_height / image_height
                    >= cls.UPRIGHT_PRESENCE_MIN_BBOX_HEIGHT_FRACTION
                    and bbox_width / bbox_height
                    <= cls.UPRIGHT_PRESENCE_MAX_BBOX_ASPECT
                    and all(
                        math.isclose(
                            left,
                            right,
                            rel_tol=0.0,
                            abs_tol=1e-9,
                        )
                        for left, right in zip(
                            candidate_point,
                            evidence_candidate_point,
                        )
                    )
                    and (
                        cls._point_matches_support_plane(support_surface, candidate_point)
                        if support_surface is not None
                        else math.isclose(
                            candidate_point[1], world_point[1],
                            rel_tol=0.0, abs_tol=1e-6,
                        )
                    )
                    and output_distance_m
                    <= cls.UPRIGHT_PRESENCE_MAX_OUTPUT_DISTANCE_M
                    and math.isclose(
                        evidence_output_distance_m,
                        output_distance_m,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                    and math.isclose(
                        evidence_output_distance_limit_m,
                        cls.UPRIGHT_PRESENCE_MAX_OUTPUT_DISTANCE_M,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                )
        current_presence_evidence = provenance.get(
            "current_presence_evidence"
        )
        current_presence_hold = bool(
            source == "anchor_hold"
            and provenance_type == "bounded_output_hold"
            and provenance_origin == "last_published_output"
            and cls._current_presence_evidence_is_canonical(
                track,
                current_presence_evidence,
                expected_origin=expected_origin,
                current_media_pts_ns=current_media_pts_ns,
            )
        )
        recent_projective_bridge = bool(
            source == "cv_prediction"
            and provenance_type == "bounded_cv_process"
            and provenance_origin == "recent_projective_process"
        )
        projective_bridge_root_media_pts_ns = (
            cls._active_projective_bridge_root(
                expected_origin,
                current_media_pts_ns=current_media_pts_ns,
            )
            if provenance_type
            in {"bounded_cv_process", "bounded_output_hold"}
            else None
        )
        projective_episode_continuation = bool(
            projective_bridge_root_media_pts_ns is not None
        )
        if recent_projective_bridge and not projective_episode_continuation:
            # The producer label is descriptive only. It can request this lane,
            # but only a live service-owned image root can authorize it.
            return False
        metric_horizon_s = (
            cls.STATIONARY_ANCHOR_HOLD_HORIZON_S
            if stationary_anchor_hold
            else (
                cls.UPRIGHT_PRESENCE_HOLD_HORIZON_S
                if upright_presence_hold
                else (
                    cls.CURRENT_PRESENCE_HOLD_HORIZON_S
                    if current_presence_hold
                    else cls.BOUNDED_PROCESS_HORIZON_S
                )
            )
        )
        max_metric_age_ns = int(
            round(
                (
                    metric_horizon_s
                    + cls.BOUNDED_PROCESS_HORIZON_EPSILON_S
                )
                * 1_000_000_000.0
            )
        )
        origin_horizon_s = (
            (
                cls.PROJECTIVE_PROCESS_BRIDGE_HORIZON_S
                + cls.BOUNDED_PROCESS_HORIZON_EPSILON_S
            )
            if projective_episode_continuation
            else (
                (
                    cls.UPRIGHT_PRESENCE_HOLD_HORIZON_S
                    + cls.BOUNDED_PROCESS_HORIZON_EPSILON_S
                )
                if upright_presence_hold
                else (
                    (
                        cls.CURRENT_PRESENCE_HOLD_HORIZON_S
                        + cls.BOUNDED_PROCESS_HORIZON_EPSILON_S
                    )
                    if current_presence_hold
                    else (
                        metric_horizon_s
                        + cls.BOUNDED_PROCESS_HORIZON_EPSILON_S
                    )
                )
            )
        )
        matched_origin = cls._matched_committed_output_origin(
            expected_origin,
            origin_point=origin_point,
            origin_media_pts_ns=origin_media_pts_ns,
            origin_trail_segment_id=origin_trail_segment_id,
            current_media_pts_ns=current_media_pts_ns,
            maximum_age_s=origin_horizon_s,
        )
        if matched_origin is None or matched_origin.support_surface != support_surface:
            return False
        if provenance_type == "bounded_output_hold" and not (
            origin_trail_segment_id
            == int(expected_origin.trail_segment_id)
            and all(
                math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
                for left, right in zip(
                    origin_point,
                    expected_origin.position,
                )
            )
        ):
            # A queue-delayed proof may name an older service-committed hold,
            # but a gain-zero hold can only retain the latest public
            # coordinate in the same visible segment. ``matched_origin``
            # above proves the historical point was genuinely committed;
            # this equality prevents historical motion from being replayed.
            return False
        if not (
            matched_origin.metric_position is not None
            and matched_origin.metric_media_pts_ns is not None
            and matched_origin.metric_trail_segment_id is not None
            and matched_origin.metric_media_pts_ns == int(expected_metric_pts)
            and matched_origin.metric_trail_segment_id
            == int(expected_metric_segment)
            and all(
                math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
                for left, right in zip(
                    matched_origin.metric_position,
                    expected_metric_position,
                )
            )
        ):
            # A metric reanchor invalidates queued proofs formed before it.
            return False
        if (
            origin_media_pts_ns <= 0
            or current_media_pts_ns <= origin_media_pts_ns
            or current_media_pts_ns != track_media_pts_ns
            or metric_origin_media_pts_ns <= 0
            or metric_origin_media_pts_ns != int(expected_metric_pts)
            or metric_origin_media_pts_ns > origin_media_pts_ns
            or (
                projective_episode_continuation
                and current_media_pts_ns
                - int(projective_bridge_root_media_pts_ns or 0)
                > int(
                    round(
                        (
                            cls.PROJECTIVE_PROCESS_BRIDGE_HORIZON_S
                            + cls.BOUNDED_PROCESS_HORIZON_EPSILON_S
                        )
                        * 1_000_000_000.0
                    )
                )
            )
            or (
                not projective_episode_continuation
                and not upright_presence_hold
                and not current_presence_hold
                and current_media_pts_ns - metric_origin_media_pts_ns
                > max_metric_age_ns
            )
            or metric_origin_trail_segment_id != int(expected_metric_segment)
            or origin_trail_segment_id != track_trail_segment_id
            or lifecycle_generation <= 0
            or lifecycle_generation != track_lifecycle_generation
            or transition.get("world_frame") != track.get("world_frame")
            or transition.get("world_frame_revision")
            != track.get("world_frame_revision")
            or transition.get("world_transform_sha256")
            != track.get("world_transform_sha256")
        ):
            return False
        if not all(
            math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
            for left, right in zip(
                metric_origin_point,
                expected_metric_position,
            )
        ):
            return False
        if not all(
            math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
            for left, right in zip(base_point, origin_point)
        ):
            return False
        if support_surface is not None:
            if not all(
                cls._point_matches_support_plane(support_surface, point)
                for point in (
                    world_point, posterior_point, process_point, origin_point,
                    metric_origin_point, base_point, expected_origin.position,
                    expected_metric_position,
                )
            ):
                return False
        elif not all(
            math.isclose(point[1], world_point[1], rel_tol=0.0, abs_tol=1e-9)
            for point in (
                posterior_point, process_point, origin_point,
                metric_origin_point, base_point,
            )
        ):
            return False

        media_dt_s = (
            current_media_pts_ns - origin_media_pts_ns
        ) / 1_000_000_000.0
        expected_gate_dt_s = min(media_dt_s, cls.IMAGE_MOTION_RESET_AFTER_S)
        if (
            not math.isfinite(gate_dt_s)
            or gate_dt_s <= 0.0
            or not math.isclose(
                gate_dt_s,
                expected_gate_dt_s,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            return False

        transition_kind = str(transition.get("kind") or "")
        if provenance_type == "bounded_cv_process":
            if transition_kind != "bounded_process_step" or not math.isclose(
                position_gain,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                return False
        else:
            if transition_kind != "output_hold" or not math.isclose(
                position_gain,
                0.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                return False
        expected_posterior = (
            base_point[0] + position_gain * (process_point[0] - base_point[0]),
            (
                cls._support_plane_elevation(
                    support_surface,
                    base_point[0] + position_gain * (process_point[0] - base_point[0]),
                    base_point[2] + position_gain * (process_point[2] - base_point[2]),
                )
                if support_surface is not None else base_point[1]
            ),
            base_point[2] + position_gain * (process_point[2] - base_point[2]),
        )
        if not all(
            math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
            for left, right in zip(world_point, posterior_point)
        ):
            return False
        if not all(
            math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
            for left, right in zip(world_point, expected_posterior)
        ):
            return False
        if (
            math.hypot(
                world_point[0] - origin_point[0],
                world_point[2] - origin_point[2],
            )
            > max_speed_mps * gate_dt_s + 1e-9
        ):
            return False
        return bool(
            cls._latest_output_step_is_physical(
                expected_origin,
                world_point=world_point,
                current_media_pts_ns=current_media_pts_ns,
                max_speed_mps=max_speed_mps,
            )
            and cls._latest_visible_output_step_is_physical(
                expected_origin,
                world_point=world_point,
                current_media_pts_ns=current_media_pts_ns,
                max_speed_mps=max_speed_mps,
            )
        )

    def _world_observation(
        self,
        track: Mapping[str, Any],
        *,
        calibration_revision: Any = None,
        expected_image_motion_origin: _CanonicalTrackOutput | None = None,
    ) -> WorldPositionObservation | None:
        if track.get("world_valid") is not True:
            return None
        world_source = str(track.get("world_source") or "").strip().lower()
        proof_bearing_continuation_sources = {
            "anchor_hold",
            "cv_prediction",
            "image_motion_prediction",
        }
        if (
            track.get("world_measurement_accepted") is False
            and world_source not in proof_bearing_continuation_sources
        ):
            # An ordinary metric source cannot become canonical after its own
            # producer explicitly rejected the measurement.  The three
            # continuation lanes below are the sole exception because this
            # service independently verifies their exact origin, transition,
            # and physical bounds before admitting them.
            return None
        if (
            str(track.get("world_quality") or "").strip().lower()
            == "held"
            and world_source not in proof_bearing_continuation_sources
        ):
            # ``held`` is process state, not a metric observation quality.
            # Fail closed if an ordinary source is relabelled held without a
            # proof-bearing continuation contract.
            return None
        if world_source == "relative_motion_prediction":
            # Retired producer lane: quarantine stale payloads rather than
            # allowing an old relative-body estimate to move canonical world.
            return None
        if world_source in {"cv_prediction", "anchor_hold"}:
            bounded_process_is_canonical = (
                self._bounded_process_observation_is_canonical(
                    track,
                    expected_origin=expected_image_motion_origin,
                )
            )
            if not bounded_process_is_canonical:
                return None
        if world_source == "image_motion_prediction":
            image_motion_is_canonical = (
                self._image_motion_observation_is_canonical(
                    track,
                    expected_origin=expected_image_motion_origin,
                )
            )
            if not image_motion_is_canonical:
                return None
        frame = str(track.get("world_frame") or "")
        if frame != "backend_world_m":
            return None
        world_frame_revision = self._required_revision(
            track.get("world_frame_revision")
        )
        world_transform_sha256 = self._sha256_text(
            track.get("world_transform_sha256")
        )
        if world_frame_revision is None or world_transform_sha256 is None:
            # A world-valid row without a complete registration identity is
            # display/debug data only.  The canonical backend must fail closed
            # instead of joining it to another registration.
            return None
        if track.get("calibration_revision") is not None:
            calibration_revision = self._required_revision(calibration_revision)
            if calibration_revision is None:
                return None
        elif calibration_revision is not None:
            calibration_revision = self._required_revision(calibration_revision)
        raw = track.get("world")
        if not isinstance(raw, (list, tuple)) or len(raw) < 3:
            return None
        try:
            position = tuple(float(value) for value in raw[:3])
        except Exception:
            return None
        if not all(math.isfinite(value) for value in position):
            return None
        quantity_raw = track.get("world_quantity")
        if quantity_raw is not None and str(quantity_raw) != "ground_footprint":
            # The current canonical contract localizes the person's ground
            # footprint.  A body root or other quantity must use a future
            # explicit field/contract rather than masquerading as this point.
            return None
        quality_raw = str(track.get("world_quality") or "estimated").strip().lower()
        quality = quality_raw if quality_raw in {"good", "estimated", "held"} else "estimated"
        covariance = self._world_covariance(track.get("world_covariance"))
        if track.get("world_covariance") is not None and covariance is None:
            # A resolver-owned observation must never be relabelled with a
            # convenient legacy variance when its uncertainty contract is
            # malformed.  Older producers that do not publish covariance keep
            # the quality-bucket compatibility path below.
            return None
        if covariance is None:
            variance = {"good": 0.04, "estimated": 0.25, "held": 0.64}[quality]
            covariance = (
                variance,
                0.0,
                0.0,
                0.0,
                variance,
                0.0,
                0.0,
                0.0,
                variance,
            )
        confidence = {"good": 0.90, "estimated": 0.60, "held": 0.35}[quality]
        resolver_confidence = self._confidence(
            track.get("world_resolver_confidence")
        )
        if resolver_confidence is not None:
            confidence = resolver_confidence
        measured_confidence = self._confidence(track.get("tracker_confidence"))
        if measured_confidence is not None:
            confidence = min(confidence, measured_confidence)
        posture_raw = str(
            track.get("world_posture") or track.get("posture") or "unknown"
        ).strip().lower()
        posture = (
            posture_raw
            if posture_raw in {"standing", "sitting", "lying", "unknown"}
            else "unknown"
        )
        support_raw = str(
            track.get("world_support_state") or "unknown"
        ).strip().lower()
        support_state = (
            support_raw
            if support_raw in {"floor", "seat", "couch", "unknown"}
            else "unknown"
        )
        support_surface = None
        if track.get("world_support_surface") is not None:
            try:
                support_surface = SupportSurfaceReference.model_validate(track["world_support_surface"])
            except (TypeError, ValueError):
                return None
            if not self._point_matches_support_plane(support_surface, position):
                return None
        return WorldPositionObservation(
            position=Vector3(x=position[0], y=position[1], z=position[2]),
            covariance=Matrix3(values=covariance),
            frame="backend_world_m",
            world_frame="backend_world_m",
            units="meters",
            world_frame_revision=world_frame_revision,
            world_transform_sha256=world_transform_sha256,
            calibration_revision=calibration_revision,
            quantity="ground_footprint",
            support_state=support_state,  # type: ignore[arg-type]
            support_surface=support_surface,
            posture=posture,  # type: ignore[arg-type]
            source=str(track.get("world_source") or "unspecified"),
            quality=quality,  # type: ignore[arg-type]
            confidence=confidence,
            reason=(
                self._bounded_text(
                    track.get("world_quality_reason"),
                    maximum=200,
                )
            ),
        )

    @staticmethod
    def _world_covariance(value: Any) -> tuple[float, ...] | None:
        """Validate one row-major 3x3 covariance without importing NumPy.

        The canonical boundary accepts the resolver's anisotropic covariance
        only when it is finite, symmetric, and positive semidefinite.  The
        positive diagonal requirement keeps downstream precision weighting
        well defined while still allowing zero off-axis correlation.
        """

        if not isinstance(value, (list, tuple)) or len(value) != 9:
            return None
        try:
            matrix = tuple(float(item) for item in value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not all(math.isfinite(item) for item in matrix):
            return None

        scale = max(1.0, *(abs(item) for item in matrix))
        tolerance = 1e-8 * scale
        if any(matrix[index] <= 0.0 for index in (0, 4, 8)):
            return None
        if (
            abs(matrix[1] - matrix[3]) > tolerance
            or abs(matrix[2] - matrix[6]) > tolerance
            or abs(matrix[5] - matrix[7]) > tolerance
        ):
            return None

        # Every principal minor of a symmetric 3x3 PSD matrix is
        # non-negative.  Evaluate them explicitly to keep this boundary small
        # and dependency-free.
        minors = (
            matrix[0] * matrix[4] - matrix[1] * matrix[3],
            matrix[0] * matrix[8] - matrix[2] * matrix[6],
            matrix[4] * matrix[8] - matrix[5] * matrix[7],
        )
        if any(minor < -tolerance for minor in minors):
            return None
        determinant = (
            matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
            - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
            + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6])
        )
        if determinant < -tolerance:
            return None
        return matrix

    def _world_diagnostics(
        self,
        track: Mapping[str, Any],
        *,
        world_observation: WorldPositionObservation | None,
        expected_continuity_origin: _CanonicalTrackOutput | None = None,
    ) -> WorldObservationDiagnostics | None:
        if (
            world_observation is not None
            and _WORLD_DIAGNOSTIC_TRACK_FIELDS.isdisjoint(track)
        ):
            return None
        resolver_fields = self._resolver_diagnostics_fields(track)
        return WorldObservationDiagnostics(
            estimator_evaluated=self._optional_bool(
                track.get("world_estimator_evaluated")
            ),
            first_divergence_reason=self._world_first_divergence_reason(
                track,
                world_observation=world_observation,
                expected_continuity_origin=expected_continuity_origin,
            ),
            floor_candidate_m=self._diagnostic_vector3(
                track.get("world_floor_candidate")
            ),
            depth_candidate_m=self._diagnostic_vector3(
                track.get("world_depth_candidate")
            ),
            prefilter_measurement_m=self._diagnostic_vector3(
                track.get("world_prefilter_measurement")
            ),
            filter_prediction_m=self._diagnostic_vector3(
                track.get("world_filter_prediction")
            ),
            floor_range_m=self._finite_float(track.get("world_floor_range_m")),
            floor_range_limit_m=self._finite_float(
                track.get("world_floor_range_limit_m")
            ),
            floor_incidence_sin=self._finite_float(
                track.get("world_floor_incidence_sin")
            ),
            floor_admitted=self._optional_bool(
                track.get("world_floor_admitted")
            ),
            floor_rejection_reason=self._bounded_text(
                track.get("world_floor_rejection_reason"),
                maximum=200,
            ),
            measurement_accepted=self._optional_bool(
                track.get("world_measurement_accepted")
            ),
            measurement_rejection_reason=self._bounded_text(
                track.get("world_rejection_reason"),
                maximum=200,
            ),
            innovation_m=self._finite_float(track.get("world_innovation_m")),
            innovation_limit_m=self._finite_float(
                track.get("world_innovation_limit_m")
            ),
            fusion_policy_id=self._bounded_text(
                track.get("world_fusion_policy_id"),
                maximum=200,
            ),
            floor_weight_scale=self._finite_float(
                track.get("world_floor_weight_scale")
            ),
            depth_weight_scale=self._finite_float(
                track.get("world_depth_weight_scale")
            ),
            floor_weight_effective=self._finite_float(
                track.get("world_floor_weight_effective")
            ),
            depth_weight_effective=self._finite_float(
                track.get("world_depth_weight_effective")
            ),
            depth_status=self._bounded_text(
                track.get("depth_status"),
                maximum=120,
            ),
            depth_anchor_m=self._finite_float(track.get("depth_anchor_m")),
            depth_registered_m=self._finite_float(
                track.get("depth_registered_m")
            ),
            depth_used_m=self._finite_float(track.get("depth_used_m")),
            depth_registration_status=self._bounded_text(
                track.get("depth_registration_status"),
                maximum=120,
            ),
            depth_registration_id=self._bounded_text(
                track.get("depth_registration_id"),
                maximum=200,
            ),
            **resolver_fields,
        )

    def _resolver_diagnostics_fields(
        self,
        track: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Normalize the bounded exact-cohort resolver summary.

        The richer dashboard payload stays track-local.  The canonical world
        contract retains only the decision and compact candidate diagnostics,
        and only when the record identifies this exact track observation.
        Malformed or stale optional diagnostics are omitted; they never alter
        admission of the already-validated canonical world point.
        """

        source = track.get("world_resolver")
        if not isinstance(source, Mapping):
            return {}
        if (
            source.get("contract") != "noesis.world_resolver_diagnostics"
            or source.get("version") != 1
        ):
            return {}

        exact_integer_fields = (
            ("source_id", "source_id"),
            ("tracker_id", "tracker_id"),
            ("frame_id", "frame_id"),
            ("observed_at_us", "observed_at_us"),
        )
        for diagnostic_key, track_key in exact_integer_fields:
            diagnostic_value = source.get(diagnostic_key)
            track_value = track.get(track_key)
            if (
                isinstance(diagnostic_value, bool)
                or isinstance(track_value, bool)
                or not isinstance(diagnostic_value, int)
            ):
                return {}
            try:
                if int(diagnostic_value) != int(track_value):
                    return {}
            except (TypeError, ValueError, OverflowError):
                return {}
        if str(source.get("camera_id") or "") != str(track.get("camera_id") or ""):
            return {}
        if str(source.get("world_frame") or "") != str(track.get("world_frame") or ""):
            return {}
        if str(source.get("world_frame_revision") or "") != str(
            track.get("world_frame_revision") or ""
        ):
            return {}

        supported_kinds = {
            "floor_ray",
            "registered_depth",
            "pose_scale",
            "gravity_reconstruction",
        }
        selected_kind_raw = source.get("selected_kind")
        selected_kind = (
            str(selected_kind_raw)
            if selected_kind_raw is not None
            and str(selected_kind_raw) in supported_kinds
            else None
        )
        selected_id = self._bounded_text(source.get("selected_id"), maximum=120)
        contributor_source = source.get("contributor_ids")
        if not isinstance(contributor_source, (list, tuple)) or len(contributor_source) > 4:
            return {}
        contributor_ids = tuple(
            item
            for item in (
                self._bounded_text(value, maximum=120)
                for value in contributor_source
            )
            if item is not None
        )
        if len(contributor_ids) != len(contributor_source):
            return {}

        raw_candidates = source.get("candidates")
        if not isinstance(raw_candidates, (list, tuple)) or len(raw_candidates) > 4:
            return {}
        candidates: list[WorldMeasurementCandidateDiagnostic] = []
        candidate_ids: set[str] = set()
        for raw in raw_candidates:
            if not isinstance(raw, Mapping):
                return {}
            candidate_id = self._bounded_text(raw.get("id"), maximum=120)
            kind = str(raw.get("kind") or "")
            score = self._finite_float(raw.get("score"))
            pcf_score = self._finite_float(raw.get("pcf_score"))
            if (
                candidate_id is None
                or candidate_id in candidate_ids
                or kind not in supported_kinds
                or score is None
                or not 0.0 <= score <= 1.0
                or pcf_score is None
                or not 0.0 <= pcf_score <= 1.0
            ):
                return {}
            selected = raw.get("selected")
            compatible = raw.get("compatible_with_selected")
            alternate = raw.get("retained_as_alternate")
            if not all(isinstance(value, bool) for value in (selected, compatible, alternate)):
                return {}
            innovation = self._finite_float(raw.get("innovation_m"))
            agreement = self._finite_float(raw.get("agreement_mahalanobis_sq"))
            if innovation is not None and innovation < 0.0:
                return {}
            if agreement is not None and agreement < 0.0:
                return {}
            try:
                candidates.append(
                    WorldMeasurementCandidateDiagnostic(
                        candidate_id=candidate_id,
                        kind=kind,  # type: ignore[arg-type]
                        score=score,
                        pcf_score=pcf_score,
                        innovation_m=innovation,
                        agreement_mahalanobis_sq=agreement,
                        compatible_with_selected=compatible,
                        selected=selected,
                        retained_as_alternate=alternate,
                        rejection_reason=self._bounded_text(
                            raw.get("rejection_reason"), maximum=200
                        ),
                    )
                )
            except ValueError:
                return {}
            candidate_ids.add(candidate_id)

        if selected_id is not None and selected_id not in candidate_ids:
            return {}
        if any(item not in candidate_ids for item in contributor_ids):
            return {}
        alternate_id = self._bounded_text(source.get("alternate_id"), maximum=120)
        if alternate_id is not None and alternate_id not in candidate_ids:
            return {}
        fused = source.get("fused")
        if not isinstance(fused, bool):
            return {}
        confidence = self._finite_float(source.get("confidence"))
        disagreement = source.get("disagreement")
        disagreement_m = None
        if disagreement is not None:
            if not isinstance(disagreement, Mapping):
                return {}
            disagreement_m = self._finite_float(disagreement.get("distance_m"))
            if disagreement_m is not None and disagreement_m < 0.0:
                return {}
        agreement = self._finite_float(source.get("agreement_mahalanobis_sq"))
        pcf_score = self._finite_float(source.get("pcf_score"))
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            return {}
        if agreement is not None and agreement < 0.0:
            return {}
        if pcf_score is not None and not 0.0 <= pcf_score <= 1.0:
            return {}
        return {
            "resolver_selected_kind": selected_kind,
            "resolver_selected_candidate_id": selected_id,
            "resolver_contributor_ids": contributor_ids,
            "resolver_alternate_candidate_id": alternate_id,
            "resolver_fused": fused,
            "resolver_confidence": confidence,
            "resolver_disagreement_m": disagreement_m,
            "resolver_agreement_mahalanobis_sq": agreement,
            "resolver_pcf_score": pcf_score,
            "resolver_reason": self._bounded_text(source.get("reason"), maximum=240),
            "resolver_candidate_diagnostics": tuple(candidates),
        }

    def _world_first_divergence_reason(
        self,
        track: Mapping[str, Any],
        *,
        world_observation: WorldPositionObservation | None,
        expected_continuity_origin: _CanonicalTrackOutput | None = None,
    ) -> str | None:
        if world_observation is not None:
            return None
        if track.get("world_valid") is True:
            if str(track.get("world_frame") or "") != "backend_world_m":
                return "world_frame_invalid"
            source = str(track.get("world_source") or "").strip().lower()
            if source == "relative_motion_prediction":
                return "display_continuity_not_authoritative"
            if (
                source in {"cv_prediction", "anchor_hold"}
                and not self._bounded_process_observation_is_canonical(
                    track,
                    expected_origin=expected_continuity_origin,
                )
            ):
                return "bounded_process_continuity_provenance_invalid"
            if (
                source == "image_motion_prediction"
                and not self._image_motion_observation_is_canonical(
                    track,
                    expected_origin=expected_continuity_origin,
                )
            ):
                return "image_motion_continuity_provenance_invalid"
            if self._required_revision(track.get("world_frame_revision")) is None:
                return "world_frame_revision_missing"
            if self._sha256_text(track.get("world_transform_sha256")) is None:
                return "world_transform_sha256_missing"
            raw = track.get("world")
            if not isinstance(raw, (list, tuple)) or len(raw) < 3:
                return "world_position_missing"
            if self._diagnostic_vector3(raw) is None:
                return "world_position_invalid"
        for key in (
            "world_rejection_reason",
            "world_floor_rejection_reason",
            "world_quality_reason",
        ):
            reason = self._bounded_text(track.get(key), maximum=240)
            if reason is not None:
                return reason
        registration_status = self._bounded_text(
            track.get("depth_registration_status"),
            maximum=120,
        )
        if registration_status not in {None, "ok"}:
            return self._bounded_text(
                f"depth_registration_{registration_status}",
                maximum=240,
            )
        depth_status = self._bounded_text(
            track.get("depth_status"),
            maximum=120,
        )
        if depth_status not in {None, "ok"}:
            return self._bounded_text(
                f"depth_{depth_status}",
                maximum=240,
            )
        return "world_observation_unavailable"

    @staticmethod
    def _diagnostic_vector3(value: Any) -> Vector3 | None:
        if not isinstance(value, (list, tuple)) or len(value) < 3:
            return None
        try:
            result = tuple(float(item) for item in value[:3])
        except Exception:
            return None
        if not all(math.isfinite(item) for item in result):
            return None
        return Vector3(x=result[0], y=result[1], z=result[2])

    @staticmethod
    def _finite_float(value: Any) -> float | None:
        try:
            result = float(value)
        except Exception:
            return None
        return result if math.isfinite(result) else None

    @staticmethod
    def _bbox_geometry(
        value: Any,
    ) -> tuple[float, float, float, float] | None:
        if not isinstance(value, (list, tuple)) or len(value) < 4:
            return None
        try:
            result = tuple(float(item) for item in value[:4])
        except (TypeError, ValueError, OverflowError):
            return None
        if (
            len(result) != 4
            or not all(math.isfinite(item) for item in result)
            or result[2] <= 0.0
            or result[3] <= 0.0
        ):
            return None
        return result

    @staticmethod
    def _optional_bool(value: Any) -> bool | None:
        return value if isinstance(value, bool) else None

    @staticmethod
    def _zone_source(value: Any) -> ZoneSource | None:
        if value is None:
            return None
        if isinstance(value, str) and value in {
            "nvdsanalytics_roi",
            "camera_default",
        }:
            return value  # type: ignore[return-value]
        raise ValueError("zone_source must be an exact supported provenance value")

    @staticmethod
    def _exact_zone(value: Any) -> str | None:
        if value is None:
            return None
        if not is_exact_zone_label(value):
            raise ValueError(
                "zone labels must be nonempty, unpadded, and at most 160 characters"
            )
        return value

    @staticmethod
    def _bounded_text(value: Any, *, maximum: int) -> str | None:
        if value is None:
            return None
        result = str(value).strip()
        return result[:maximum] if result else None

    @staticmethod
    def _required_revision(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        result = value.strip()
        return result if 1 <= len(result) <= 200 else None

    @staticmethod
    def _sha256_text(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        result = value.strip().lower()
        if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
            return None
        return result

    def _subject(self, observation: ObservationEnvelope, track: Mapping[str, Any]) -> SubjectRef:
        stable_id = self._positive_int(track.get("stable_id"))
        kind_raw = str(track.get("identity_kind") or track.get("identity_state") or "provisional").lower()
        resident_uuid = str(track.get("resident_uuid") or "").strip()
        display_name = str(track.get("display_name") or "").strip() or None
        if kind_raw == "resident" and resident_uuid:
            return SubjectRef(
                subject_id=f"resident:{resident_uuid}",
                kind=IdentityKind.RESIDENT,
                resident_uuid=resident_uuid,
                display_name=display_name,
                stable_id=stable_id,
            )
        if kind_raw == "visitor" and stable_id is not None:
            generation = self._nonnegative_int(track.get("visitor_generation")) or 0
            return SubjectRef(
                subject_id=f"visitor:{self.producer.run_id}:{stable_id}:g{generation}",
                kind=IdentityKind.VISITOR,
                generation=generation,
                stable_id=stable_id,
            )
        tracklet = observation.payload.tracklet
        if kind_raw == "unknown":
            return SubjectRef(
                subject_id=(
                    f"unknown:{tracklet.run_id}:{tracklet.camera_id}:"
                    f"{tracklet.tracker_id}"
                ),
                kind=IdentityKind.UNKNOWN,
                stable_id=stable_id,
            )
        return SubjectRef(
            subject_id=(
                f"tracklet:{tracklet.run_id}:{tracklet.camera_id}:"
                f"{tracklet.tracker_id}"
            ),
            kind=IdentityKind.PROVISIONAL,
            stable_id=stable_id,
        )

    @staticmethod
    def _confidence(value: Any) -> float | None:
        try:
            result = float(value)
        except Exception:
            return None
        return result if math.isfinite(result) and 0.0 <= result <= 1.0 else None

    @staticmethod
    def _positive_int(value: Any) -> int | None:
        try:
            result = int(value)
        except Exception:
            return None
        return result if result > 0 else None

    @staticmethod
    def _nonnegative_int(value: Any) -> int | None:
        try:
            result = int(value)
        except Exception:
            return None
        return result if result >= 0 else None
