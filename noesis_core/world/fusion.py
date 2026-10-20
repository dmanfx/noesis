from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Iterable

from reid.mv3dt_person_association import MemberKey, PhysicalPersonKey, ObservationStamp, SpatialClockContext

from noesis_core.contracts.base import Matrix3, ProducerRef, Vector3
from noesis_core.contracts.identity import SubjectRef
from noesis_core.contracts.observation import ObservationEnvelope
from noesis_core.contracts.support_surface import SupportSurfaceReference
from noesis_core.contracts.world import (
    EntityLifecycle,
    PositionState,
    WorldEntity,
    WorldSnapshot,
    WorldSourceEvidence,
    WorldAttributionClaim,
    WorldAttributionQuarantine,
    validate_world_identity_uniqueness,
)
from noesis_core.world.support_surfaces import contains_xz, constrained_plane_covariance


class ObservationOrderError(ValueError):
    pass


def position_quality_sort_key(
    envelope: ObservationEnvelope,
    *,
    resolver_confidence: float | None = None,
    comparable_clock: bool = True,
) -> tuple[int, float, float, int, str]:
    """Existing quality order; time breaks ties only for comparable clocks.

    This is selection math, not frame/time or current-person admission. The
    spatial caller must supply the ORIGINAL resolver confidence, not a legacy
    quality bucket or a tracker-confidence replacement.
    """
    world = envelope.payload.world
    if world is None:
        raise ValueError("position quality requires an original world observation")
    return (
        0 if GlobalWorldFusion._has_canonical_identity(world) else 1,
        GlobalWorldFusion._variance_sum(world.covariance),
        -float(world.confidence if resolver_confidence is None else resolver_confidence),
        -int(envelope.observed_at_us) if comparable_clock else 0,
        envelope.payload.tracklet.camera_id,
    )


SPATIAL_NO_VOTE_REASONS = frozenset({
    "producer_no_position", "non_current_measurement", "native_active_required",
    "stale_spatial_member", "world_support_unavailable",
    "resolver_covariance_absent", "resolver_confidence_absent",
    "projective_current_support_required", "current_visual_stale_member_delivery",
    "anatomical_current_support_required",
    "image_region_current_support_required",
    "image_region_rejoin_proof_required",
    "image_region_ambiguous_member",
    "image_region_contradictory_cue_member",
})
SPATIAL_INTEGRITY_REASONS = frozenset({
    "resolver_covariance_invalid", "resolver_confidence_invalid", "world_position_invalid",
    "world_registration_unavailable", "world_covariance_or_position_invalid",
    "world_position_unavailable", "accepted_world_position_unavailable", "producer_state_invalid",
    "producer_state_conflict", "measurement_state_unavailable", "world_quantity_invalid",
    "world_support_invalid", "spatial_original_media_binding", "spatial_original_member_context",
    "spatial_member_ineligible", "source_clock_unavailable", "original_acquisition_interval",
    "current_rejoin_proof_required", "noncurrent_spatial_member",
    "spatial_original_frame_binding", "spatial_original_source_binding",
    "camera_calibration_sha256_missing", "camera_calibration_sha256_invalid",
    "camera_calibration_sha256_mismatch", "unexpected_spatial_rejection_reason",
})


def spatial_rejection_category(reason: str | None) -> str:
    """Closed no-vote vocabulary; unknown/unexpected states fail closed."""
    if reason is None:
        return "vote"
    return "no_vote" if reason in SPATIAL_NO_VOTE_REASONS else "integrity"


@dataclass(frozen=True)
class PreparedUprightFootprintSupport:
    """Private same-preparation support, not a floor/depth or person grant.

    The historical name also covers the explicit registered-body-column
    method (truthful posture, UNKNOWN support, no observed foot or seat claim).
    Only World normalization creates this from the compact typed resolver
    descriptor of a producer-accepted current original. The exact immutable
    envelope and native stamp prevent borrowing support from another row.
    """

    original: ObservationEnvelope
    stamp: ObservationStamp
    selected_id: str
    contact_basis: str

    def matches(self, original: ObservationEnvelope) -> bool:
        world, track = original.payload.world, original.payload.tracklet
        return (self.original is original and type(self.stamp) is ObservationStamp
            and self.stamp.observation_id == original.observation_id
            and self.stamp.member.run_id == original.producer.run_id
            and original.media_pts_ns is not None and self.stamp.media_time_s is not None
            and math.isclose(self.stamp.media_time_s, original.media_pts_ns / 1e9,
                             abs_tol=1e-9, rel_tol=0.)
            and (self.stamp.member.camera_id, self.stamp.member.source_id,
                 self.stamp.member.raw_id, self.stamp.frame_id) ==
                (track.camera_id, track.source_id, track.tracker_id, track.frame_id)
            and world is not None and world.quantity == "ground_footprint"
            and world.support_state == "unknown" and (
                self.selected_id == "upright_pose_projection"
                and self.contact_basis == "pose:upright_body_planes" and world.posture == "standing"
                or self.selected_id == "registered_body_column"
                and self.contact_basis == "depth:registered_body_column"
                and world.source == "registered_body_column"
                and world.posture in {"standing", "sitting", "lying", "unknown"}
                and world.quality == "estimated"))


def select_source_local_observation(
    candidates: tuple[tuple[ObservationEnvelope, float | None, str | None], ...],
    *,
    comparable_clock: bool = False,
    upright_support: tuple[PreparedUprightFootprintSupport, ...] = (),
) -> tuple[ObservationEnvelope | None, tuple[tuple[ObservationEnvelope, str | None], ...]]:
    """Choose one exact measured original; retain nonvoting alternatives.

    At most two ALREADY owner-associated members are supplied by the future
    spatial transaction. Each tuple carries original resolver confidence and
    the private raw-quality rejection recorded during World.prepare(). This
    helper neither associates people nor issues grants, checks freshness, or
    authorizes CI; coincident frame labels never cause averaging here.
    """
    if (type(candidates) is not tuple or not 1 <= len(candidates) <= 2
            or type(comparable_clock) is not bool
            or any(type(row) is not tuple or len(row) != 3 for row in candidates)):
        raise ValueError("one or two exact source-local candidates required")
    if any(type(row[0]) is not ObservationEnvelope for row in candidates):
        raise TypeError("original World observations required")
    if len({row[0].observation_id for row in candidates}) != len(candidates):
        raise ValueError("duplicate source-local observation identity")
    if (type(upright_support) is not tuple or len(upright_support) > 2
            or any(type(proof) is not PreparedUprightFootprintSupport
                or not any(proof.matches(row[0]) for row in candidates) for proof in upright_support)):
        raise ValueError("exact prepared upright support required")
    eligible: list[tuple[ObservationEnvelope, float]] = []
    diagnostics: list[tuple[ObservationEnvelope, str | None]] = []
    for original, confidence, reason in candidates:
        world = original.payload.world
        if reason is not None and (type(reason) is not str or not 1 <= len(reason) <= 200):
            raise ValueError("bounded original quality rejection required")
        # Raw integrity cannot be erased by an inactive member or a usable
        # peer. Conversely a missing quality field isn't a malformed value.
        if spatial_rejection_category(reason) != "integrity":
            if world is None:
                reason = reason or "world_position_unavailable"
            elif not (GlobalWorldFusion._has_canonical_identity(world) and world.calibration_revision):
                reason = "world_registration_unavailable"
            else:
                # Revalidate without replacing the original. model_copy() can
                # bypass public validators; no malformed covariance may win.
                try:
                    type(world).model_validate(world.model_dump(mode="python"))
                    if (not all(math.isfinite(v) for v in (world.position.x, world.position.y, world.position.z))
                            or any(world.covariance.values[i] <= 0.0 for i in (0, 4, 8))):
                        raise ValueError("finite position and positive covariance diagonal required")
                except (TypeError, ValueError):
                    reason = "world_covariance_or_position_invalid"
                if spatial_rejection_category(reason) != "integrity":
                    if world.quantity != "ground_footprint":
                        reason = "world_quantity_invalid"
                    elif confidence is not None and (type(confidence) not in {int, float}
                            or not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0):
                        reason = "resolver_confidence_invalid"
                    elif reason is None:
                        if world.source in {"cv_prediction", "image_motion_prediction", "anchor_hold"} or world.quality == "held":
                            reason = "non_current_measurement"
                        elif world.support_state == "unknown" and not any(
                                proof.matches(original) for proof in upright_support):
                            reason = "world_support_unavailable"
                        elif confidence is None:
                            reason = "resolver_confidence_absent"
        diagnostics.append((original, reason))
        if reason is None:
            eligible.append((original, float(confidence)))
    selected = min(eligible, key=lambda row: position_quality_sort_key(
        row[0], resolver_confidence=row[1], comparable_clock=comparable_clock,
    ))[0] if eligible else None
    return selected, tuple(
        (original, reason if reason is not None or original is selected else "source_local_alternative")
        for original, reason in diagnostics
    )


@dataclass(frozen=True)
class WorldFusionConfig:
    simultaneous_window_us: int = 350_000
    present_ttl_us: int = 750_000
    lost_ttl_us: int = 2_000_000
    conflict_distance_m: float = 1.25
    # Match the canonical human-ground/output contract. A fused entity must
    # never reintroduce a faster velocity than the producer is allowed to
    # publish, even when identity continuity spans a tracker transition.
    max_velocity_mps: float = 4.0
    min_variance_m2: float = 0.0025
    max_entities: int = 64

    def __post_init__(self) -> None:
        if self.simultaneous_window_us <= 0:
            raise ValueError("simultaneous_window_us must be positive")
        if not 0 < self.present_ttl_us <= self.lost_ttl_us:
            raise ValueError("expected 0 < present_ttl_us <= lost_ttl_us")
        if self.conflict_distance_m <= 0.0 or self.max_velocity_mps <= 0.0:
            raise ValueError("fusion distance and velocity limits must be positive")
        if self.min_variance_m2 <= 0.0 or self.max_entities <= 0:
            raise ValueError("variance and entity capacity must be positive")


@dataclass(frozen=True)
class _ResolvedObservation:
    envelope: ObservationEnvelope
    subject: SubjectRef
    trail_segment_id: int | None = None
    trail_break_required: bool = False
    tracker_lifecycle_generation: int | None = None
    source_epoch: int | None = None
    physical_member: MemberKey | None = None
    spatial_stamp: ObservationStamp | None = None
    spatial_selected: bool = False
    spatial_rejection: str | None = None
    attribution_claims: tuple[WorldAttributionClaim, ...] = ()
    spatial_resolver_confidence: float | None = None
    spatial_committed_held: bool = False


@dataclass(frozen=True)
class _FusedState:
    observed_at_us: int
    media_pts_ns: int | None
    position_source: tuple[str, int, str, int | None] | None
    position: Vector3
    continuity_identity: tuple[object, ...] | None
    last_trail_reset_tokens: tuple[
        tuple[str, int, str, int, int | None, int], ...
    ]
    last_trail_segment_tokens: tuple[
        tuple[str, int, str, int, int | None, int], ...
    ]
    entity: WorldEntity


@dataclass(frozen=True)
class _SplitCustodyOriginal:
    """One previously committed ORIGINAL, never a current tracklet or vote."""

    episode_id: str
    origin_key: PhysicalPersonKey
    successor_keys: tuple[PhysicalPersonKey, ...]
    row: _ResolvedObservation
    entity: WorldEntity
    continuity: _ContinuityState | None
    destination_key: PhysicalPersonKey | None = None
    # Diagnostic provenance only: excluded from custody equality/authority.
    owner_diagnostic_view: object | None = field(default=None,compare=False,repr=False)


@dataclass(frozen=True)
class _ContinuityState:
    segment_id: int
    frame_identity: tuple[str, str | None]
    support_state: str
    support_surface: SupportSurfaceReference | None
    # One token per accepted source. A changed lifecycle or trail segment for
    # the same camera/source is a visible trail break, even across held heads.
    source_tokens: tuple[
        tuple[str, int, str, int, int | None, int | None, int | None], ...
    ]
    spatial_identity: tuple[object, ...] | None = None


@dataclass(frozen=True)
class _PositionStep:
    dt_s: float
    distance_m: float
    speed_mps: float
    velocity: Vector3


class GlobalWorldFusion:
    """Deterministic backend owner for global entity snapshots.

    The service keeps raw per-camera observations, fuses only contemporaneous
    compatible positions, and exposes conflicts rather than averaging them.
    """

    # Retain only a bounded number of camera-local trail reset/segment tokens
    # per fused identity. Normal deployments have far fewer contributing
    # cameras, while the hard cap prevents identity churn from growing private
    # continuity state without bound.
    MAX_TRAIL_SOURCE_TOKENS = 64

    def __init__(self, producer: ProducerRef, *, config: WorldFusionConfig | None = None) -> None:
        self.producer = producer
        self.config = config or WorldFusionConfig()
        self._source_runs: dict[tuple[str, int], str] = {}
        self._last_sequence: dict[tuple[str, int, str], int] = {}
        self._by_entity: OrderedDict[str, dict[str, _ResolvedObservation]] = OrderedDict()
        self._entity_by_tracklet: dict[tuple[object, ...], str] = {}
        self._last_fused: dict[str, _FusedState] = {}
        self._continuity_by_entity: dict[str, _ContinuityState] = {}
        self._snapshot_sequence = 0
        self._spatial_context: SpatialClockContext | None = None
        self._attribution_quarantines: dict[str, WorldAttributionQuarantine] = {}
        self._split_custody: dict[str, _SplitCustodyOriginal] = {}
        self._consumed_split_episodes: dict[str, object] = {}
        self._omission_diagnostics = {}

    def _note_omission(self, entity_id, reason, age=None):
        try:
            if entity_id in self._omission_diagnostics or len(self._omission_diagnostics)<64:
                self._omission_diagnostics[entity_id]=(reason,age)
        except Exception:
            pass  # Private observation cannot change fusion.

    def fork(self) -> GlobalWorldFusion:
        """Return an isolated copy-on-write candidate of the fusion state.

        Contract values stored inside the maps are immutable.  Candidate
        ingestion replaces those values and mutates only the copied container
        layers, avoiding a full recursive copy on every publication frame.
        """

        candidate = object.__new__(type(self))
        candidate.producer = self.producer
        candidate.config = self.config
        candidate._source_runs = dict(self._source_runs)
        candidate._last_sequence = dict(self._last_sequence)
        candidate._by_entity = OrderedDict(
            (entity_id, dict(observations))
            for entity_id, observations in self._by_entity.items()
        )
        candidate._entity_by_tracklet = dict(self._entity_by_tracklet)
        candidate._last_fused = dict(self._last_fused)
        candidate._continuity_by_entity = dict(self._continuity_by_entity)
        candidate._snapshot_sequence = int(self._snapshot_sequence)
        candidate._spatial_context = self._spatial_context
        candidate._attribution_quarantines = dict(self._attribution_quarantines)
        candidate._split_custody = dict(self._split_custody)
        candidate._consumed_split_episodes = dict(self._consumed_split_episodes)
        candidate._omission_diagnostics = {}
        return candidate

    def ingest(
        self,
        envelope: ObservationEnvelope,
        subject: SubjectRef,
        *,
        trail_segment_id: int | None = None,
        trail_break_required: bool = False,
        tracker_lifecycle_generation: int | None = None,
        source_epoch: int | None = None,
        physical_key: PhysicalPersonKey | None = None,
        physical_member: MemberKey | None = None,
        spatial_stamp: ObservationStamp | None = None,
        spatial_selected: bool = False,
        spatial_rejection: str | None = None,
        attribution_claims: tuple[WorldAttributionClaim, ...] = (),
        spatial_resolver_confidence: float | None = None,
    ) -> None:
        if envelope.payload.world is None:
            return
        if spatial_resolver_confidence is not None and (
                type(spatial_resolver_confidence) not in (int,float)
                or not math.isfinite(spatial_resolver_confidence) or not 0 <= spatial_resolver_confidence <= 1):
            raise ValueError("original spatial resolver confidence required")
        if spatial_stamp is not None and (
            type(spatial_stamp) is not ObservationStamp or spatial_stamp.member != physical_member
            or spatial_stamp.observation_id != envelope.observation_id
        ):
            raise ValueError("exact original spatial stamp/member required")
        try:
            normalized_trail_segment_id = (
                int(trail_segment_id)
                if trail_segment_id is not None and int(trail_segment_id) >= 0
                else None
            )
        except (TypeError, ValueError, OverflowError):
            normalized_trail_segment_id = None
        try:
            normalized_lifecycle_generation = (
                int(tracker_lifecycle_generation)
                if tracker_lifecycle_generation is not None
                and int(tracker_lifecycle_generation) >= 0
                else None
            )
        except (TypeError, ValueError, OverflowError):
            normalized_lifecycle_generation = None
        try:
            normalized_source_epoch = (
                int(source_epoch)
                if source_epoch is not None and int(source_epoch) >= 0
                else None
            )
        except (TypeError, ValueError, OverflowError):
            normalized_source_epoch = None
        entity_id = subject.subject_id
        if physical_key is not None or physical_member is not None:
            if (type(physical_key) is not PhysicalPersonKey
                    or type(physical_member) is not MemberKey
                    or physical_key.run_id != envelope.producer.run_id
                    or physical_member.run_id != physical_key.run_id
                    or (physical_member.camera_id, physical_member.source_id, physical_member.raw_id)
                    != (envelope.payload.tracklet.camera_id, envelope.payload.tracklet.source_id,
                        envelope.payload.tracklet.tracker_id)
                    or physical_member.source_epoch != normalized_source_epoch
                    or physical_member.lifecycle_generation != normalized_lifecycle_generation):
                raise ValueError("exact physical member and run-owned key required")
            entity_id = self.physical_entity_id(physical_key)
        source_base = (envelope.payload.tracklet.camera_id, envelope.payload.tracklet.source_id)
        previous_run = self._source_runs.get(source_base)
        if previous_run is not None and previous_run != envelope.producer.run_id:
            self._drop_source_run(source_base, previous_run)
        self._source_runs[source_base] = envelope.producer.run_id

        sequence_key = (*source_base, envelope.producer.run_id)
        last_sequence = self._last_sequence.get(sequence_key)
        if last_sequence is not None and envelope.sequence <= last_sequence:
            raise ObservationOrderError(
                f"non-monotonic observation sequence for {source_base}: "
                f"received={envelope.sequence} last={last_sequence}"
            )
        self._last_sequence[sequence_key] = envelope.sequence

        camera_id = envelope.payload.tracklet.camera_id
        tracklet_key: tuple[object, ...] = (
            camera_id,
            envelope.payload.tracklet.source_id,
            envelope.producer.run_id,
            envelope.payload.tracklet.tracker_id,
        )
        if physical_member is not None:
            tracklet_key += (physical_member.source_epoch, physical_member.lifecycle_generation)
        prior_entity_id = self._entity_by_tracklet.get(tracklet_key)
        if prior_entity_id is not None and prior_entity_id != entity_id:
            self._remove_tracklet_observation(prior_entity_id, tracklet_key)

        entity_observations = self._by_entity.setdefault(entity_id, {})
        replaced = entity_observations.get(camera_id)
        if replaced is not None:
            replaced_world = replaced.envelope.payload.world
            incoming_world = envelope.payload.world
            if (
                replaced_world is not None
                and incoming_world is not None
                and self._has_canonical_identity(replaced_world)
                and self._has_canonical_identity(incoming_world)
                and self._registration_identity(replaced_world)
                != self._registration_identity(incoming_world)
            ):
                # The per-camera slot is the replacement/reset boundary.  Do
                # not carry velocity across a registration change, even when
                # other cameras continue contributing to the same subject.
                self._last_fused.pop(entity_id, None)
            replaced_key = self._tracklet_key(replaced)
            if replaced_key != tracklet_key:
                self._entity_by_tracklet.pop(replaced_key, None)
        entity_observations[camera_id] = _ResolvedObservation(
            envelope=envelope,
            subject=subject,
            trail_segment_id=normalized_trail_segment_id,
            trail_break_required=bool(
                trail_break_required is True
                and normalized_trail_segment_id is not None
            ),
            tracker_lifecycle_generation=normalized_lifecycle_generation,
            source_epoch=normalized_source_epoch,
            physical_member=physical_member,
            spatial_stamp=spatial_stamp,
            spatial_selected=spatial_selected,
            spatial_rejection=spatial_rejection,
            attribution_claims=attribution_claims,
            spatial_resolver_confidence=spatial_resolver_confidence,
        )
        self._entity_by_tracklet[tracklet_key] = entity_id
        self._by_entity.move_to_end(entity_id)
        while len(self._by_entity) > self.config.max_entities:
            entity_id, _ = self._by_entity.popitem(last=False)
            self._last_fused.pop(entity_id, None)
            self._continuity_by_entity.pop(entity_id, None)

            self._drop_entity_tracklets(entity_id)
        if len(self._continuity_by_entity) > self.config.max_entities:
            self._prune_absent_continuity()

    def selected_spatial_original(self, key: PhysicalPersonKey):
        """One already granted immutable original; NOT an alternative/new vote.

        The service must additionally validate current owner eligibility and
        literal active frame bindings. No normalized confidence reconstruction.
        """
        rows = self._by_entity.get(self.physical_entity_id(key),{})
        selected = [r for r in rows.values() if r.spatial_selected and r.spatial_stamp is not None]
        if len(selected) != 1:
            return None
        row, = selected
        if row.spatial_resolver_confidence is None or self._spatial_age_us(row) > 350_000:
            return None
        return row.envelope,row.spatial_stamp,row.spatial_resolver_confidence,row.subject

    def select_spatial(self, key: PhysicalPersonKey, observation_id: str) -> None:
        """Mark ONE publicly granted original; older peers stay nonvoting."""
        rows = self._by_entity.get(f"physical:{key.run_id}:{key.serial}", {})
        for camera, row in tuple(rows.items()):
            if row.spatial_stamp is not None:
                chosen = row.envelope.observation_id == observation_id
                rows[camera] = replace(row, spatial_selected=chosen, spatial_committed_held=False,
                                       spatial_rejection=None if chosen else row.spatial_rejection or "source_local_alternative")

    def committed_spatial_original(self, key: PhysicalPersonKey, *, published_at_us: int):
        """Only the selected committed custody, not a current measurement grant."""
        rows = self._by_entity.get(self.physical_entity_id(key), {})
        selected = [r for r in rows.values() if r.spatial_selected and r.spatial_stamp is not None]
        if len(selected) == 1:
            row, = selected
        else:
            custody = next((c for c in self._split_custody.values()
                if c.origin_key == key or c.destination_key == key), None)
            if custody is None:
                return None
            row = custody.row
        if (row.spatial_resolver_confidence is None
                or self._spatial_age_us(row) > self.config.lost_ttl_us
                or row.envelope.observed_at_us + self.config.lost_ttl_us <= published_at_us):
            return None
        return row.envelope, row.spatial_stamp, row.spatial_resolver_confidence, row.subject

    def capture_split_custody(self, episode_id, origin_key, successor_keys, *, published_at_us):
        """Read ONLY selected committed custody, before ordinary withdrawal."""
        if episode_id in self._consumed_split_episodes:
            return None  # Its authorized successor is NEVER a pre-split original.
        original = self.committed_spatial_original(origin_key, published_at_us=published_at_us)
        if original is None:
            return None
        prior = self._split_custody.get(episode_id)
        if prior is not None:
            if (prior.origin_key, prior.successor_keys) != (origin_key, successor_keys):
                raise ValueError("split_custody_episode_changed")
            return prior
        entity_id = self.physical_entity_id(origin_key)
        rows = self._by_entity.get(entity_id, {})
        selected = [r for r in rows.values() if r.spatial_selected and r.spatial_stamp == original[1]]
        previous = self._last_fused.get(entity_id)
        if len(selected) != 1 or previous is None:
            return None
        entity = previous.entity
        # A rejected/filter-retained newer sample is not the committed point.
        if entity.observed_at_us != selected[0].envelope.observed_at_us:
            return None
        return _SplitCustodyOriginal(episode_id, origin_key, successor_keys, selected[0], entity,
                                     self._continuity_by_entity.get(entity_id))

    def replace_split_custody(self, originals):
        """Complete owner-certified view; omission revokes, never refreshes."""
        if len(originals) > self.config.max_entities:
            raise ValueError("split_custody_capacity")
        keys = [k for c in originals for k in c.successor_keys]
        if len(keys) != len(set(keys)):
            raise ValueError("split_custody_overlap")
        incoming={c.episode_id: c for c in originals}
        for episode,custody in self._split_custody.items():
            if episode not in incoming:
                self._note_omission(self.physical_entity_id(custody.destination_key or custody.origin_key),'split_custody_omitted')
        self._split_custody = incoming
        for custody in originals:
            if custody.continuity is not None:
                self._continuity_by_entity[self.physical_entity_id(custody.origin_key)] = custody.continuity

    def enforce_split_exclusivity(self, keys, selected=None):
        """Owner-certified scope: diagnostics never become a second head."""
        for key in keys:
            entity_id = self.physical_entity_id(key)
            for camera, row in tuple(self._by_entity.get(entity_id, {}).items()):
                if row.spatial_stamp is not None and (selected is None or
                        (key, row.envelope.observation_id) != selected):
                    self._by_entity[entity_id][camera] = replace(row, spatial_selected=False)

    def split_custody_originals(self):
        return tuple(self._split_custody.values())

    def synchronize_split_episodes(self, views):
        """Bound consumption to COMPLETE live owner views, not tombstone history."""
        current = {view.episode_id: view for view in views}
        if len(current) > self.config.max_entities:
            raise ValueError("split_custody_capacity")
        for episode, view in current.items():
            if episode in self._consumed_split_episodes and self._consumed_split_episodes[episode] is not view:
                prior=self._consumed_split_episodes[episode]
                # A certified reunion must never make an already replaced
                # original eligible again. Carry its consumed marker forward.
                if (view.rejoined_key!=prior.origin_key or replace(view,rejoined_key=None,
                        rejoined_stamps=(),rejoin_revision=None)!=replace(prior,rejoined_key=None,
                        rejoined_stamps=(),rejoin_revision=None)):
                    raise ValueError("split_custody_episode_changed")
                self._consumed_split_episodes[episode]=view
        self._consumed_split_episodes = {episode: view for episode, view
            in self._consumed_split_episodes.items() if episode in current}

    def consume_split_episode(self, view):
        """Candidate-only bookkeeping; only the atomic World commit installs it."""
        if len(self._consumed_split_episodes) >= self.config.max_entities and view.episode_id not in self._consumed_split_episodes:
            raise ValueError("split_custody_capacity")
        self._consumed_split_episodes[view.episode_id] = view
        self._split_custody.pop(view.episode_id, None)

    def hold_committed_spatial(self, key: PhysicalPersonKey, stamp: ObservationStamp) -> None:
        rows = self._by_entity.get(self.physical_entity_id(key), {})
        selected = [r for r in rows.values() if r.spatial_selected]
        if len(selected) != 1 or selected[0].spatial_stamp != stamp:
            raise ValueError("committed_spatial_hold_custody_changed")
        row, = selected
        rows[row.envelope.payload.tracklet.camera_id] = replace(row, spatial_committed_held=True)

    def reindex_committed_spatial(self, prior, origin, destination, stamp, *, published_at_us):
        """Move ONE exact committed observation; caller owns current DTO proof.

        No new original, claim, position vote, clock or split episode is made.
        This runs after ordinary removals on a transaction-local fusion fork.
        """
        original = prior.committed_spatial_original(origin, published_at_us=published_at_us)
        old_id, new_id = self.physical_entity_id(origin), self.physical_entity_id(destination)
        rows = prior._by_entity.get(old_id, {})
        selected = [r for r in rows.values() if r.spatial_selected and r.spatial_stamp == stamp]
        previous = prior._last_fused.get(old_id)
        continuity = prior._continuity_by_entity.get(old_id)
        if (origin == destination or origin.run_id != destination.run_id or original is None
                or original[1] != stamp or len(selected) != 1 or previous is None or continuity is None
                or previous.entity.observed_at_us != original[0].observed_at_us
                or not any(s.accepted and s.observation_id == original[0].observation_id
                           for s in previous.entity.sources)
                or any(r.spatial_selected for r in self._by_entity.get(new_id, {}).values())):
            raise ValueError("ordinary_join_original_changed")
        row, = selected
        self._by_entity.setdefault(new_id, {})[row.envelope.payload.tracklet.camera_id] = replace(
            row, spatial_committed_held=True)
        self._entity_by_tracklet[self._tracklet_key(row)] = new_id
        self._last_fused[new_id] = replace(previous,
            entity=previous.entity.model_copy(update={"entity_id": new_id}))
        self._continuity_by_entity[new_id] = continuity

    def has_spatial_original_slot(self, key):
        """An expired selected slot still competes until normal snapshot pruning."""
        return any(row.spatial_selected for row in self._by_entity.get(self.physical_entity_id(key), {}).values())

    def prune_expired_spatial_original(self, key, *, published_at_us):
        """Only positive original source/wire expiry, on a transaction fork."""
        entity = self.physical_entity_id(key)
        selected = [row for row in self._by_entity.get(entity, {}).values() if row.spatial_selected]
        if len(selected) != 1 or selected[0].spatial_stamp is None:
            return
        row, = selected
        age = self._spatial_age_us(row)  # Unknown/foreign binding is NOT expiry.
        if (age <= self.config.lost_ttl_us
                and row.envelope.observed_at_us + self.config.lost_ttl_us > published_at_us):
            return
        self._note_omission(entity, 'source_age_expired' if age > self.config.lost_ttl_us
                            else 'original_wire_expired', age)
        self._by_entity.pop(entity, None)
        self._last_fused.pop(entity, None)
        self._continuity_by_entity.pop(entity, None)
        self._drop_entity_tracklets(entity)

    def set_spatial_context(self, context: SpatialClockContext) -> None:
        """Use the public owner-validated context, never publication arrival."""
        if type(context) is not SpatialClockContext:
            raise ValueError("actual spatial source-clock context required")
        self._spatial_context = context

    def spatial_attribution_claims(self) -> dict[str, tuple[WorldAttributionClaim, ...]]:
        """Original claim evidence owned by still-live position samples, not new leases."""
        result = {}
        for entity_id, rows in self._by_entity.items():
            selected = next((row for row in rows.values() if row.spatial_selected), None)
            if selected is None or self._spatial_age_us(selected) > self.config.lost_ttl_us:
                continue
            claims = {}
            for row in rows.values():
                for claim in row.attribution_claims:
                    key = (claim.camera_id, claim.source_id, claim.source_epoch, claim.raw_id,
                           claim.lifecycle_generation)
                    if key not in claims or claim.frame_id > claims[key].frame_id:
                        claims[key] = claim
            if not claims:
                raise ValueError("retained_attribution_provenance_unavailable")
            result[entity_id] = tuple(claims.values())
        for custody in self._split_custody.values():
            if self._spatial_age_us(custody.row) <= self.config.lost_ttl_us:
                entity = self.physical_entity_id(custody.destination_key or custody.origin_key)
                result[entity] = tuple(dict.fromkeys((*result.get(entity, ()), *custody.row.attribution_claims)))
        if sum(map(len, result.values())) > 128:
            raise ValueError("attribution_claim_capacity")
        return result

    def spatial_original_attribution_claims(self) -> dict[str, tuple[WorldAttributionClaim, ...]]:
        """ALL originals attached to live samples, separate from latest arbitration.

        A newer peer slot can hide an older claim until a rekey removes that
        slot. Preserve that genuine custody without growing a history ledger.
        Original selected-position TTL and the existing bounded rows apply.
        """
        result = {}
        total = 0
        for entity_id, rows in self._by_entity.items():
            selected = next((row for row in rows.values() if row.spatial_selected), None)
            if selected is None or self._spatial_age_us(selected) > self.config.lost_ttl_us:
                continue
            claims = tuple(dict.fromkeys(c for row in rows.values() for c in row.attribution_claims))
            if not claims:
                raise ValueError("retained_attribution_provenance_unavailable")
            total += len(claims)
            if total > 128:
                raise ValueError("attribution_claim_capacity")
            result[entity_id] = claims
        for custody in self._split_custody.values():
            if self._spatial_age_us(custody.row) <= self.config.lost_ttl_us:
                entity = self.physical_entity_id(custody.destination_key or custody.origin_key)
                result[entity] = tuple(dict.fromkeys((*result.get(entity, ()), *custody.row.attribution_claims)))
        if sum(map(len, result.values())) > 128:
            raise ValueError("attribution_claim_capacity")
        return result

    def attribution_quarantines(self) -> dict[str, WorldAttributionQuarantine]:
        return dict(self._attribution_quarantines)

    def set_attribution_quarantines(self, records: dict[str, WorldAttributionQuarantine]) -> None:
        if len(records) > self.config.max_entities or any(type(v) is not WorldAttributionQuarantine for v in records.values()):
            raise ValueError("attribution_quarantine_capacity")
        self._attribution_quarantines = dict(records)

    def _spatial_age_us(self, row: _ResolvedObservation) -> int:
        stamp = row.spatial_stamp
        context = self._spatial_context
        interval = next((i for i in context.intervals if i.binding.source_id == stamp.member.source_id), None) if context and stamp else None
        world = row.envelope.payload.world
        if (interval is None or world is None
                or (interval.binding.camera_id, interval.binding.source_epoch,
                    interval.binding.clock_domain, interval.binding.clock_basis)
                != (stamp.member.camera_id, stamp.member.source_epoch, stamp.clock_domain, stamp.clock_basis)
                or interval.binding.calibration_revision != world.calibration_revision
                or interval.binding.transform_id != world.world_transform_sha256):
            raise ValueError("retained_spatial_clock_or_binding_unavailable")
        return max(0, round((interval.evaluation_max_s - stamp.acquired_at_s) * 1_000_000))

    @staticmethod
    def physical_entity_id(key: PhysicalPersonKey) -> str:
        if (type(key) is not PhysicalPersonKey or not key.run_id
                or type(key.serial) is not int or key.serial < 1):
            raise ValueError("run-owned physical person key required")
        return f"physical:{key.run_id}:{key.serial}"

    @staticmethod
    def _tracklet_key(item: _ResolvedObservation) -> tuple[object, ...]:
        t = item.envelope.payload.tracklet
        key: tuple[object, ...] = (t.camera_id, t.source_id, item.envelope.producer.run_id, t.tracker_id)
        if item.physical_member is not None:
            key += (item.physical_member.source_epoch, item.physical_member.lifecycle_generation)
        return key

    def withdraw_physical_member(self, member: MemberKey) -> None:
        """Remove exactly an old contributor, never a reused numeric raw ID."""
        if type(member) is not MemberKey:
            raise TypeError("exact association MemberKey required")
        key = (member.camera_id, member.source_id, member.run_id, member.raw_id,
               member.source_epoch, member.lifecycle_generation)
        entity_id = self._entity_by_tracklet.get(key)
        if entity_id is not None:
            self._note_omission(entity_id,'withdrawn_physical_member')
            self._remove_tracklet_observation(entity_id, key)
            self._prune_absent_continuity()

    def retire_physical_key(self, key: PhysicalPersonKey) -> None:
        entity_id = self.physical_entity_id(key)
        self._note_omission(entity_id,'retired_physical_key')
        self._by_entity.pop(entity_id, None)
        self._last_fused.pop(entity_id, None)
        self._continuity_by_entity.pop(entity_id, None)
        self._drop_entity_tracklets(entity_id)

    def _retain_admitted_tracker_origin(
        self, envelope: ObservationEnvelope, subject: SubjectRef, *,
        tracker_lifecycle_generation: int, source_epoch: int,
        trail_segment_id: int | None, trail_break_required: bool,
    ) -> bool:
        """Restore an already admitted source sample without replaying ingest.

        A later rejected/image-only row may occupy this camera's slot. Native
        presence must retain the admitted sample, not make that row authority
        when another camera disappears. Ordering watermarks and all original
        observation clocks are deliberately unchanged.
        """
        tracklet = envelope.payload.tracklet
        world = envelope.payload.world
        if world is None or envelope.producer.run_id != self.producer.run_id:
            return False
        observations = self._by_entity.get(subject.subject_id)
        current = observations.get(tracklet.camera_id) if observations else None
        if current is None:
            return False
        current_tracklet = current.envelope.payload.tracklet
        if (
            (current_tracklet.run_id, current_tracklet.camera_id,
             current_tracklet.source_id, current_tracklet.tracker_id)
            != (tracklet.run_id, tracklet.camera_id, tracklet.source_id, tracklet.tracker_id)
            or current.envelope.producer.run_id != envelope.producer.run_id
            or current.subject != subject
            or current.tracker_lifecycle_generation != tracker_lifecycle_generation
            or current.source_epoch != source_epoch
            or current.envelope.observed_at_us < envelope.observed_at_us
        ):
            return False
        current_world = current.envelope.payload.world
        if current_world is not None and (
            self._registration_identity(current_world) != self._registration_identity(world)
        ):
            return False
        observations[tracklet.camera_id] = _ResolvedObservation(
            envelope=envelope, subject=subject,
            trail_segment_id=trail_segment_id,
            trail_break_required=trail_break_required,
            tracker_lifecycle_generation=tracker_lifecycle_generation,
            source_epoch=source_epoch,
        )
        return True

    def clear_source(
        self, camera_id: str, source_id: int, run_id: str, *,
        retained_lifecycles: frozenset[tuple[int, int, int]] = frozenset(),
        terminated_tracker_ids: frozenset[int] = frozenset(),
        terminated_only: bool = False,
    ) -> int:
        """Remove current evidence for one empty tracker source frame.

        Observation ordering state is deliberately retained. A delayed
        observation from before the clear must not become current again.
        """

        camera_key = str(camera_id).strip()
        run_key = str(run_id).strip()
        source_key = int(source_id)
        if not camera_key or not run_key or source_key < 0:
            raise ValueError("camera_id, source_id, and run_id must identify a source")

        source_base = (camera_key, source_key)
        previous_run = self._source_runs.get(source_base)
        if previous_run is not None and previous_run != run_key:
            self._drop_source_run(source_base, previous_run)
        self._source_runs[source_base] = run_key

        removed = 0
        for entity_id, observations in list(self._by_entity.items()):
            current = observations.get(camera_key)
            if current is None:
                continue
            tracklet = current.envelope.payload.tracklet
            if (
                int(tracklet.source_id) != source_key
                or current.envelope.producer.run_id != run_key
            ):
                continue
            tracker = int(tracklet.tracker_id)
            if tracker not in terminated_tracker_ids:
                if terminated_only or (
                    tracker, current.tracker_lifecycle_generation, current.source_epoch
                ) in retained_lifecycles:
                    continue
            tracklet_key = self._tracklet_key(current)
            observations.pop(camera_key, None)
            self._entity_by_tracklet.pop(tracklet_key, None)
            removed += 1
            if not observations:
                self._by_entity.pop(entity_id, None)
                self._last_fused.pop(entity_id, None)
        self._prune_absent_continuity()
        return removed

    def _prune_absent_continuity(self) -> None:
        # Saved continuity belongs to the origin; a certified transfer's
        # visible attribution belongs to its destination. Both are already
        # bounded members of this SAME finite two-key custody scope.
        held = {self.physical_entity_id(key) for c in self._split_custody.values()
                for key in (c.origin_key, c.destination_key or c.origin_key)}
        for entity_id in tuple(self._continuity_by_entity):
            if entity_id not in self._by_entity and entity_id not in held:
                self._continuity_by_entity.pop(entity_id, None)
        for entity_id in tuple(self._attribution_quarantines):
            if entity_id not in self._by_entity and entity_id not in held:
                self._attribution_quarantines.pop(entity_id, None)

    def snapshot(self, *, published_at_us: int) -> WorldSnapshot:
        # The public fusion boundary must not consume a sequence, expire an
        # index or update velocity/continuity on a refused whole snapshot.
        candidate = self.fork()
        candidate._omission_diagnostics = dict(self._omission_diagnostics)
        snapshot = candidate._snapshot_candidate(published_at_us=published_at_us)
        self._by_entity = candidate._by_entity
        self._entity_by_tracklet = candidate._entity_by_tracklet
        self._last_fused = candidate._last_fused
        self._continuity_by_entity = candidate._continuity_by_entity
        self._snapshot_sequence = candidate._snapshot_sequence
        self._attribution_quarantines = candidate._attribution_quarantines
        self._split_custody = candidate._split_custody
        self._consumed_split_episodes = candidate._consumed_split_episodes
        self._omission_diagnostics = candidate._omission_diagnostics
        return snapshot

    def _snapshot_candidate(self, *, published_at_us: int) -> WorldSnapshot:
        if published_at_us <= 0:
            raise ValueError("published_at_us must be positive")
        self._snapshot_sequence += 1
        entities: list[WorldEntity] = []
        observation_times: list[int] = []
        expired: list[str] = []
        for entity_id, observations in list(self._by_entity.items()):
            entity = self._fuse_entity(entity_id, observations.values(), published_at_us)
            # Source-clock eligibility and the immutable wire deadline are
            # independent bounds (e.g. a paced/queued replay). Intersect them;
            # never publish a head that the canonical consumer must expire at
            # this very snapshot's published_at_us. No original is restamped.
            if entity is None or entity.stale_after_us <= published_at_us:
                if entity is not None:
                    self._note_omission(entity_id,'original_wire_expired')
                expired.append(entity_id)
                continue
            quarantine = self._attribution_quarantines.get(entity_id)
            if quarantine is not None:
                serial = entity_id.rsplit(":", 1)[-1]
                entity = WorldEntity.model_validate({**entity.model_dump(mode="python"),
                    "subject": SubjectRef(subject_id=f"quarantine:{quarantine.run_id}:{serial}", kind="provisional"),
                    "attribution_quarantine": quarantine})
            entities.append(entity)
            observation_times.append(entity.observed_at_us)
        for entity_id in expired:
            self._by_entity.pop(entity_id, None)
            self._last_fused.pop(entity_id, None)
            self._continuity_by_entity.pop(entity_id, None)
            self._drop_entity_tracklets(entity_id)
        for episode, custody in tuple(self._split_custody.items()):
            scope = {self.physical_entity_id(k) for k in custody.successor_keys}
            # A genuinely admitted current successor consumes the OLD hold.
            if any(e.entity_id in scope for e in entities):
                del self._split_custody[episode]
                continue
            old = custody.entity
            custody_age = self._spatial_age_us(custody.row)
            if (custody_age > self.config.lost_ttl_us
                    or old.stale_after_us <= published_at_us):
                self._note_omission(old.entity_id,'split_source_age_expired' if custody_age>self.config.lost_ttl_us
                    else 'split_original_wire_expired',custody_age)
                del self._split_custody[episode]
                continue
            destination=custody.destination_key or custody.origin_key
            destination_id=self.physical_entity_id(destination)
            quarantine = self._attribution_quarantines.get(destination_id)
            subject_id = (f"quarantine:{quarantine.run_id}:{destination.serial}"
                          if quarantine is not None else f"split-custody:{episode}")
            entity = old.model_copy(update={
                "entity_id": destination_id,
                "subject": SubjectRef(subject_id=subject_id, kind="provisional"),
                "lifecycle": EntityLifecycle.HELD, "position_state": PositionState.HELD,
                "velocity_mps": None,
                "attribution_quarantine": quarantine,
            })
            entities.append(entity)
            observation_times.append(entity.observed_at_us)
        self._prune_absent_continuity()

        observed_end = max(observation_times, default=published_at_us)
        observed_start = min(observation_times, default=published_at_us)
        validate_world_identity_uniqueness(entities)
        return WorldSnapshot(
            contract="noesis.world.snapshot",
            contract_version=1,
            snapshot_id=f"{self.producer.run_id}:{self._snapshot_sequence}",
            producer=self.producer,
            sequence=self._snapshot_sequence,
            observed_start_us=observed_start,
            observed_end_us=observed_end,
            published_at_us=max(published_at_us, observed_end),
            frame="backend_world_m",
            units="meters",
            entities=tuple(sorted(entities, key=lambda item: item.entity_id)),
        )

    def _drop_source_run(self, source_base: tuple[str, int], run_id: str) -> None:
        camera_id, source_id = source_base
        self._last_sequence.pop((camera_id, source_id, run_id), None)
        for key in tuple(self._entity_by_tracklet):
            if key[:3] == (camera_id, source_id, run_id):
                self._entity_by_tracklet.pop(key, None)
        for entity_id, observations in list(self._by_entity.items()):
            current = observations.get(camera_id)
            if current is not None and current.envelope.producer.run_id == run_id:
                observations.pop(camera_id, None)
            if not observations:
                self._by_entity.pop(entity_id, None)
                self._last_fused.pop(entity_id, None)

    def _remove_tracklet_observation(
        self,
        entity_id: str,
        tracklet_key: tuple[object, ...],
    ) -> None:
        observations = self._by_entity.get(entity_id)
        if observations is None:
            self._entity_by_tracklet.pop(tracklet_key, None)
            return
        camera_id = tracklet_key[0]
        current = observations.get(camera_id)
        if current is not None:
            current_key = self._tracklet_key(current)
            if current_key == tracklet_key:
                observations.pop(camera_id, None)
        self._entity_by_tracklet.pop(tracklet_key, None)
        if not observations:
            self._by_entity.pop(entity_id, None)
            self._last_fused.pop(entity_id, None)

    def _drop_entity_tracklets(self, entity_id: str) -> None:
        for key, mapped_entity_id in tuple(self._entity_by_tracklet.items()):
            if mapped_entity_id == entity_id:
                self._entity_by_tracklet.pop(key, None)

    def _fuse_entity(
        self,
        entity_id: str,
        observations: Iterable[_ResolvedObservation],
        now_us: int,
    ) -> WorldEntity | None:
        usable = [item for item in observations if item.envelope.payload.world is not None]
        if not usable:
            self._note_omission(entity_id,'no_usable_world_original')
            return None
        usable.sort(
            key=lambda item: (
                item.envelope.observed_at_us,
                item.envelope.payload.world.confidence if item.envelope.payload.world else 0.0,
                item.envelope.payload.tracklet.camera_id,
            ),
            reverse=True,
        )
        bounded_usable = [
            item for item in usable
            if (world := item.envelope.payload.world) is not None
            and (world.support_surface is None or contains_xz(
                world.support_surface, world.position.x, world.position.z
            ))
        ]
        if not bounded_usable:
            self._note_omission(entity_id,'position_outside_support_extent')
            return None
        # An out-of-extent named observation cannot refresh entity time or
        # shorten the cohort window for an older valid measurement.
        spatial = any(item.spatial_stamp is not None for item in usable)
        selected = next((item for item in bounded_usable if item.spatial_selected), None) if spatial else None
        if spatial and selected is None:
            self._note_omission(entity_id,'no_selected_spatial_original')
            return None  # A diagnostic peer is never an implicit position grant.
        latest_at = (selected or bounded_usable[0]).envelope.observed_at_us
        age_us = self._spatial_age_us(selected) if spatial else max(0, now_us - latest_at)
        if age_us > self.config.lost_ttl_us:
            self._note_omission(entity_id,'source_age_expired',age_us)
            return None
        lifecycle = EntityLifecycle.PRESENT if age_us <= self.config.present_ttl_us else EntityLifecycle.HELD
        contemporaneous = [
            item
            for item in usable
            if spatial or abs(latest_at - item.envelope.observed_at_us) <= self.config.simultaneous_window_us
        ]
        bounded_ids = {id(item) for item in bounded_usable}
        bounded_contemporaneous = [
            item for item in contemporaneous if id(item) in bounded_ids
        ]
        out_of_extent = [
            (item, "position_outside_support_extent")
            for item in contemporaneous if id(item) not in bounded_ids
        ]
        authoritative = [
            item
            for item in bounded_contemporaneous
            if item.envelope.payload.world is not None
            and self._position_state(item) == PositionState.MEASURED
            and (not spatial or item is selected)
        ]
        ignored_held: list[tuple[_ResolvedObservation, str]] = []
        if authoritative:
            # A state-integrated held point keeps one camera current, but it is
            # not an independent metric hypothesis and must not pull a fresh
            # camera observation through covariance intersection.
            fusion_candidates = authoritative
            ignored_held = [
                (item, "non_authoritative_held_continuation")
                for item in bounded_contemporaneous
                if item not in authoritative
            ]
        else:
            # With no fresh metric camera, use the newest held continuation
            # exactly.  Never average two process-model continuations: their
            # errors can be correlated and the resulting point would exist in
            # neither tracking cohort.
            selected_held = selected if spatial else min(
                bounded_contemporaneous,
                key=self._held_continuation_sort_key,
            )
            fusion_candidates = [selected_held]
            ignored_held = [
                (item, "non_authoritative_held_continuation")
                for item in bounded_contemporaneous
                if item is not selected_held
            ]

        anchor = selected if spatial else min(fusion_candidates, key=self._anchor_sort_key)
        anchor_world = anchor.envelope.payload.world
        assert anchor_world is not None
        anchor_target_identity = self._target_frame_identity(anchor_world)
        anchor_has_target_identity = self._has_canonical_identity(anchor_world)

        accepted: list[_ResolvedObservation] = []
        rejected: list[tuple[_ResolvedObservation, str]] = out_of_extent
        for item in fusion_candidates:
            world = item.envelope.payload.world
            assert world is not None
            item_has_any_identity = self._has_any_registration_identity(world)
            if anchor_has_target_identity:
                if not self._has_canonical_identity(world):
                    rejected.append((item, "registration_identity_missing"))
                    continue
                if self._target_frame_identity(world) != anchor_target_identity:
                    rejected.append((item, "registration_identity_conflict"))
                    continue
            elif item_has_any_identity:
                # A legacy direct-contract observation may still be fused with
                # another legacy observation.  Once an observation carries a
                # registration field, however, a partial identity cannot be
                # silently averaged into any canonical camera cohort.
                rejected.append((item, "registration_identity_missing"))
                continue
            if world.support_surface != anchor_world.support_surface:
                rejected.append((item, "support_surface_conflict"))
                continue
            distance = self._distance(anchor_world.position, world.position)
            if distance > self.config.conflict_distance_m:
                rejected.append((item, f"position_conflict:{distance:.3f}m"))
            else:
                accepted.append(item)
        if not accepted:
            accepted = [anchor]

        if anchor_world.support_surface is not None:
            # On named finite geometry the best-supported measurement is the
            # fallback if matrix CI leaves the observed support extent.
            accepted = [anchor] + [item for item in accepted if item is not anchor]
        if spatial:
            # Source-local capability grants exactly one original, NOT CI.
            position, covariance, accepted, extent_rejected = anchor_world.position, anchor_world.covariance, [anchor], []
            for item in contemporaneous:
                world = item.envelope.payload.world
                if item is anchor or item.spatial_rejection not in {None, "source_local_alternative"}:
                    continue
                reason = (
                    "registration_identity_conflict" if self._target_frame_identity(world) != anchor_target_identity
                    else "support_surface_conflict" if world.support_surface != anchor_world.support_surface
                    else "position_conflict" if self._distance(world.position, position) > self.config.conflict_distance_m
                    else None
                )
                if reason:
                    rejected.append((item, reason))
        else:
            position, covariance, accepted, extent_rejected = self._weighted_position(accepted)
        rejected.extend(extent_rejected)
        source_evidence: list[WorldSourceEvidence] = []
        rejected_ids = {id(item): reason for item, reason in rejected}
        ignored_held_ids = {id(item): reason for item, reason in ignored_held}
        for item in contemporaneous:
            world = item.envelope.payload.world
            assert world is not None
            reason = rejected_ids.get(id(item)) or ignored_held_ids.get(id(item))
            if spatial and item is not anchor:
                reason = rejected_ids.get(id(item)) or item.spatial_rejection or "source_local_alternative"
            source_evidence.append(
                WorldSourceEvidence(
                    observation_id=item.envelope.observation_id,
                    camera_id=item.envelope.payload.tracklet.camera_id,
                    zone=item.envelope.payload.zone,
                    zone_source=item.envelope.payload.zone_source,
                    zone_authoritative=item.envelope.payload.zone_authoritative,
                    observed_at_us=item.envelope.observed_at_us,
                    position=world.position,
                    covariance=world.covariance,
                    world_frame=world.world_frame,
                    world_frame_revision=world.world_frame_revision,
                    world_transform_sha256=world.world_transform_sha256,
                    calibration_revision=world.calibration_revision,
                    support_surface=world.support_surface,
                    position_source=world.source,
                    position_quality=world.quality,
                    position_state=self._position_state(item),
                    accepted=reason is None,
                    rejection_reason=reason,
                )
            )

        accepted_zones = {
            zone
            for item in accepted
            if item.envelope.payload.zone_authoritative
            and item.envelope.payload.zone_source == "nvdsanalytics_roi"
            and (zone := item.envelope.payload.zone)
        }
        room_id = next(iter(accepted_zones)) if len(accepted_zones) == 1 else None
        room_conflict = len(accepted_zones) > 1
        position_observed_at: int | None = None
        continuity_identity: tuple[object, ...] | None = None
        accepted_trail_reset_tokens: tuple[
            tuple[str, int, str, int, int | None, int], ...
        ] = ()
        accepted_trail_segment_tokens: tuple[
            tuple[str, int, str, int, int | None, int], ...
        ] = ()
        consumed_trail_reset_tokens: tuple[
            tuple[str, int, str, int, int | None, int], ...
        ] = ()
        position_media_pts_ns: int | None = None
        position_source: tuple[str, int, str, int | None] | None = None
        if authoritative:
            position_observed_at = max(
                item.envelope.observed_at_us for item in accepted
            )
            position_media_pts_ns, position_source = self._position_clock(accepted)
            continuity_identity = self._fusion_continuity_identity(accepted)
            if spatial:
                continuity_identity = (*continuity_identity, anchor.physical_member,
                                       anchor_world.world_transform_sha256, anchor_world.calibration_revision)
            accepted_trail_reset_tokens = (
                self._accepted_trail_tokens(
                    accepted,
                    explicit_breaks_only=True,
                )
            )
            accepted_trail_segment_tokens = (
                self._accepted_trail_tokens(
                    accepted,
                    explicit_breaks_only=False,
                )
            )
            previous = self._last_fused.get(entity_id)
            step = self._position_step(
                previous,
                observed_at_us=position_observed_at,
                media_pts_ns=position_media_pts_ns,
                position_source=position_source,
                position=position,
            )
            continuity_matches = bool(
                previous is not None
                and continuity_identity is not None
                and continuity_identity == previous.continuity_identity
            )
            previous_trail_reset_tokens = (
                previous.last_trail_reset_tokens
                if previous is not None
                else ()
            )
            previous_trail_segment_tokens = (
                previous.last_trail_segment_tokens
                if previous is not None
                else ()
            )
            trail_reset_applies = bool(
                continuity_matches
                and self._contains_new_trail_token(
                    previous_trail_reset_tokens,
                    accepted_trail_reset_tokens,
                )
            )
            changed_trail_segment_tokens = self._changed_trail_segment_tokens(
                previous_trail_segment_tokens,
                accepted_trail_segment_tokens,
            )
            trail_segment_reset_applies = bool(
                continuity_matches and changed_trail_segment_tokens
            )
            consumed_trail_reset_tokens = self._merge_trail_tokens(
                accepted_trail_reset_tokens,
                changed_trail_segment_tokens,
            )
            trail_reset_applies = bool(
                trail_reset_applies or trail_segment_reset_applies
            )
            if (
                continuity_matches
                and not trail_reset_applies
                and step is not None
                and step.speed_mps > self.config.max_velocity_mps
            ):
                return self._retain_after_velocity_rejection(
                    previous=previous,
                    now_us=now_us,
                    step=step,
                    accepted=accepted,
                    source_evidence=source_evidence,
                    simultaneous_conflict=bool(rejected),
                    room_conflict=room_conflict,
                )
            velocity = (
                step.velocity
                if continuity_matches
                and not trail_reset_applies
                and step is not None
                else None
            )
        else:
            # Held continuation is already the output of the per-track process
            # model.  Do not feed it back into the independent global velocity
            # baseline or report a second derived velocity as fresh evidence.
            velocity = None
        subject = anchor.subject
        conflict_reasons: list[str] = []
        if rejected:
            conflict_reasons.append(
                "incompatible simultaneous camera observations; "
                "best-supported observation retained"
            )
        if room_conflict:
            conflict_reasons.append(
                "incompatible simultaneous accepted source room observations"
            )
        accepted_target_identities = {
            self._target_frame_identity(item.envelope.payload.world)
            for item in accepted
            if item.envelope.payload.world is not None
        }
        common_target_identity = (
            next(iter(accepted_target_identities))
            if len(accepted_target_identities) == 1
            else None
        )
        if spatial and anchor.spatial_committed_held:
            continuity = self._continuity_by_entity.get(entity_id)
            if continuity is None:
                raise ValueError("committed_spatial_hold_continuity_missing")
        else:
            continuity = self._next_continuity_state(
                entity_id,
                accepted,
                frame_identity=common_target_identity or ("backend_world_m", None),
                support_state=anchor_world.support_state,
                support_surface=anchor_world.support_surface,
                spatial_identity=continuity_identity if spatial else None,
            )
        position_state = (
            PositionState.HELD
            if age_us > self.config.present_ttl_us
            else (
                PositionState.MEASURED
                if authoritative
                else self._position_state(anchor)
            )
        )
        accepted_calibrations = {
            item.envelope.payload.world.calibration_revision
            for item in accepted
            if item.envelope.payload.world is not None
        }
        accepted_transform_sha256s = {
            item.envelope.payload.world.world_transform_sha256
            for item in accepted
            if item.envelope.payload.world is not None
        }
        entity = WorldEntity(
            entity_id=entity_id,
            subject=subject,
            lifecycle=lifecycle,
            position=position,
            covariance=covariance,
            position_state=position_state,
            continuity_segment_id=continuity.segment_id,
            world_frame="backend_world_m",
            world_frame_revision=(
                common_target_identity[1]
                if common_target_identity is not None
                else None
            ),
            world_transform_sha256=(
                next(iter(accepted_transform_sha256s))
                if len(accepted_transform_sha256s) == 1
                else None
            ),
            calibration_revision=(
                next(iter(accepted_calibrations))
                if len(accepted_calibrations) == 1
                else None
            ),
            position_quantity="ground_footprint",
            support_state=anchor_world.support_state,
            support_surface=anchor_world.support_surface,
            posture=anchor_world.posture,
            velocity_mps=velocity,
            room_id=room_id,
            observed_at_us=latest_at,
            stale_after_us=latest_at + self.config.lost_ttl_us,
            sources=tuple(sorted(source_evidence, key=lambda item: item.camera_id)),
            conflict=bool(rejected) or room_conflict,
            conflict_reason="; ".join(conflict_reasons) or None,
        )
        self._continuity_by_entity[entity_id] = continuity
        if authoritative and position_observed_at is not None:
            previous = self._last_fused.get(entity_id)
            if (
                previous is None
                or continuity_identity != previous.continuity_identity
                or position_observed_at >= previous.observed_at_us
            ):
                carried_trail_reset_tokens = (
                    previous.last_trail_reset_tokens
                    if previous is not None
                    and continuity_identity == previous.continuity_identity
                    else ()
                )
                carried_trail_segment_tokens = (
                    previous.last_trail_segment_tokens
                    if previous is not None
                    and continuity_identity == previous.continuity_identity
                    else ()
                )
                self._last_fused[entity_id] = _FusedState(
                    observed_at_us=position_observed_at,
                    media_pts_ns=position_media_pts_ns,
                    position_source=position_source,
                    position=position,
                    continuity_identity=continuity_identity,
                    last_trail_reset_tokens=self._merge_trail_tokens(
                        carried_trail_reset_tokens,
                        consumed_trail_reset_tokens,
                    ),
                    last_trail_segment_tokens=self._merge_trail_tokens(
                        carried_trail_segment_tokens,
                        accepted_trail_segment_tokens,
                    ),
                    entity=entity,
                )
        return entity

    def _retain_after_velocity_rejection(
        self,
        *,
        previous: _FusedState,
        now_us: int,
        step: _PositionStep,
        accepted: list[_ResolvedObservation],
        source_evidence: list[WorldSourceEvidence],
        simultaneous_conflict: bool,
        room_conflict: bool,
    ) -> WorldEntity | None:
        age_us = max(0, now_us - previous.observed_at_us)
        if age_us > self.config.lost_ttl_us:
            return None

        rejection_reason = (
            f"velocity_gate:{step.distance_m:.3f}m/{step.dt_s:.3f}s="
            f"{step.speed_mps:.3f}mps>{self.config.max_velocity_mps:.3f}mps"
        )[:200]
        rejected_observation_ids = {
            item.envelope.observation_id for item in accepted
        }
        rejected_current = tuple(
            evidence.model_copy(
                update={
                    "accepted": False,
                    "rejection_reason": rejection_reason,
                }
            )
            if evidence.observation_id in rejected_observation_ids
            else evidence
            for evidence in source_evidence
        )
        prior_sources = previous.entity.sources
        # A prior source may also be present in the current simultaneous
        # cohort and have been rejected only because the implausible candidate
        # temporarily became that cohort's anchor.  Retaining the prior fused
        # position must retain the exact evidence that authorized it as well;
        # only genuinely new current observations are appended as rejected.
        sources_by_observation = {
            source.observation_id: source
            for source in (*rejected_current, *prior_sources)
        }

        conflict_reasons = [
            "implausible same-identity position step rejected; prior fused position retained"
        ]
        if simultaneous_conflict:
            conflict_reasons.append("simultaneous source position conflict")
        if room_conflict:
            conflict_reasons.append("simultaneous source room conflict")
        if previous.entity.conflict_reason:
            conflict_reasons.append(previous.entity.conflict_reason)
        conflict_reason = "; ".join(dict.fromkeys(conflict_reasons))[:240]
        lifecycle = (
            EntityLifecycle.PRESENT
            if age_us <= self.config.present_ttl_us
            else EntityLifecycle.HELD
        )
        return previous.entity.model_copy(
            update={
                "lifecycle": lifecycle,
                "position_state": PositionState.HELD,
                "velocity_mps": None,
                "observed_at_us": previous.observed_at_us,
                "stale_after_us": previous.observed_at_us + self.config.lost_ttl_us,
                "sources": tuple(
                    sorted(
                        sources_by_observation.values(),
                        key=lambda item: (item.camera_id, item.observed_at_us),
                    )
                ),
                "conflict": True,
                "conflict_reason": conflict_reason,
            }
        )

    @staticmethod
    def _position_state(item: _ResolvedObservation) -> PositionState:
        world = item.envelope.payload.world
        assert world is not None
        if item.spatial_committed_held:
            return PositionState.HELD
        # The strict service already verified process provenance before
        # ingestion. Source and quality, rather than entity age or confidence,
        # distinguish a current metric point from its continuation.
        if world.source in {"cv_prediction", "image_motion_prediction"}:
            return PositionState.PREDICTED
        if world.source == "anchor_hold" or world.quality == "held":
            return PositionState.HELD
        return PositionState.MEASURED

    def _next_continuity_state(
        self,
        entity_id: str,
        accepted: list[_ResolvedObservation],
        *,
        frame_identity: tuple[str, str | None],
        support_state: str,
        support_surface: SupportSurfaceReference | None,
        spatial_identity: tuple[object, ...] | None = None,
    ) -> _ContinuityState:
        current_tokens = tuple(sorted((
            (
                item.envelope.payload.tracklet.camera_id,
                int(item.envelope.payload.tracklet.source_id),
                item.envelope.producer.run_id,
                int(item.envelope.payload.tracklet.tracker_id),
                item.tracker_lifecycle_generation,
                item.source_epoch,
                item.trail_segment_id,
            )
            for item in accepted
        ), key=lambda token: (token[0], token[1], token[2], token[3])))
        current_tokens = current_tokens[:self.MAX_TRAIL_SOURCE_TOKENS]
        previous = self._continuity_by_entity.get(entity_id)
        if previous is None:
            segment_id = 1
        else:
            prior_by_source = {token[:2]: token for token in previous.source_tokens}
            source_reset = any(
                (prior := prior_by_source.get(token[:2])) is not None
                and (
                    prior[2:4] != token[2:4]
                    or (prior[4] is not None and token[4] is not None and prior[4] != token[4])
                    or (prior[5] is not None and token[5] is not None and prior[5] != token[5])
                    or (prior[6] is not None and token[6] is not None and prior[6] != token[6])
                )
                for token in current_tokens
            )
            reset = (
                previous.frame_identity != frame_identity
                or previous.support_state != support_state
                or previous.support_surface != support_surface
                or source_reset
                or previous.spatial_identity != spatial_identity
            )
            segment_id = previous.segment_id + int(reset)
        # Remember a bounded token for a temporarily absent accepted source.
        # Otherwise A -> B -> A would hide a reset that happened while B was
        # the selected head of the same canonical entity.
        current_source_keys = {token[:2] for token in current_tokens}
        tokens = current_tokens + tuple(
            token for token in (previous.source_tokens if previous is not None else ())
            if token[:2] not in current_source_keys
        )
        return _ContinuityState(
            segment_id=segment_id,
            frame_identity=frame_identity,
            support_state=support_state,
            support_surface=support_surface,
            source_tokens=tokens[:self.MAX_TRAIL_SOURCE_TOKENS],
            spatial_identity=spatial_identity,
        )

    def _anchor_sort_key(self, item: _ResolvedObservation) -> tuple[int, float, float, int, str]:
        return position_quality_sort_key(item.envelope)

    def _held_continuation_sort_key(
        self,
        item: _ResolvedObservation,
    ) -> tuple[int, int, float, float, str]:
        world = item.envelope.payload.world
        assert world is not None
        return (
            -int(item.envelope.observed_at_us),
            0 if self._has_canonical_identity(world) else 1,
            self._variance_sum(world.covariance),
            -float(world.confidence),
            item.envelope.payload.tracklet.camera_id,
        )

    def _weighted_position(
        self, observations: list[_ResolvedObservation]
    ) -> tuple[Vector3, Matrix3, list[_ResolvedObservation], list[tuple[_ResolvedObservation, str]]]:
        first = observations[0]
        first_world = first.envelope.payload.world
        assert first_world is not None
        position = first_world.position
        covariance = Matrix3(values=self._regularized_covariance(first_world.covariance))
        used = [first]
        rejected: list[tuple[_ResolvedObservation, str]] = []
        # Camera correlations are not proven by this service.  Covariance
        # intersection retains the complete anisotropic matrix while avoiding
        # the unjustified overconfidence of summing independent precisions.
        for item in observations[1:]:
            world = item.envelope.payload.world
            assert world is not None
            candidate_position, candidate_covariance = self._covariance_intersection(
                position,
                covariance,
                world.position,
                world.covariance,
            )
            if first_world.support_surface is not None and not contains_xz(
                first_world.support_surface, candidate_position.x, candidate_position.z
            ):
                rejected.append((item, "ci_position_outside_support_extent"))
                continue
            position, covariance = candidate_position, candidate_covariance
            used.append(item)
        if first_world.support_surface is not None:
            surface = first_world.support_surface
            nx, ny, nz = surface.normal_world
            y = -(nx * position.x + nz * position.z + surface.offset_m) / ny
            covariance = constrained_plane_covariance(
                surface, covariance, vertical_displacement_m=y - position.y
            )
            position = Vector3(x=position.x, y=y, z=position.z)
        return (
            position,
            covariance,
            used,
            rejected,
        )

    def _regularized_covariance(self, matrix: Matrix3) -> tuple[float, ...]:
        values = tuple(float(value) for value in matrix.values)
        minimum_diagonal = min(values[0], values[4], values[8])
        diagonal_floor = max(0.0, self.config.min_variance_m2 - minimum_diagonal)
        return tuple(
            value + (diagonal_floor if index in (0, 4, 8) else 0.0)
            for index, value in enumerate(values)
        )

    @staticmethod
    def _zero_matrix() -> tuple[float, ...]:
        return (0.0,) * 9

    @staticmethod
    def _matrix_add(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(a + b for a, b in zip(left, right))

    @staticmethod
    def _matrix_vector(matrix: tuple[float, ...], vector: Vector3 | tuple[float, float, float]) -> tuple[float, float, float]:
        values = (float(vector.x), float(vector.y), float(vector.z)) if isinstance(vector, Vector3) else tuple(float(value) for value in vector)
        return tuple(
            sum(matrix[row * 3 + column] * values[column] for column in range(3))
            for row in range(3)
        )  # type: ignore[return-value]

    @staticmethod
    def _matrix_vector_add(
        left: tuple[float, float, float], right: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        return tuple(a + b for a, b in zip(left, right))  # type: ignore[return-value]

    def _covariance_intersection(
        self,
        left_position: Vector3,
        left_covariance: Matrix3,
        right_position: Vector3,
        right_covariance: Matrix3,
    ) -> tuple[Vector3, Matrix3]:
        left = self._regularized_covariance(left_covariance)
        right = self._regularized_covariance(right_covariance)
        left_inverse = self._invert_matrix(left)
        right_inverse = self._invert_matrix(right)
        if left_inverse is None or right_inverse is None:
            return left_position, left_covariance

        left_vector = self._matrix_vector((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), left_position)
        right_vector = self._matrix_vector((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), right_position)
        # A fixed midpoint is bounded and deterministic.  It is a valid
        # covariance-intersection update and keeps this global hot path cheap
        # when many entities have two active camera sources.
        weight = 0.5
        information = self._matrix_add(
            self._matrix_scale(left_inverse, weight),
            self._matrix_scale(right_inverse, 1.0 - weight),
        )
        covariance = self._invert_matrix(information)
        if covariance is None:
            return left_position, left_covariance
        information_vector = self._matrix_vector_add(
            tuple(
                weight * value
                for value in self._matrix_vector(left_inverse, left_vector)
            ),
            tuple(
                (1.0 - weight) * value
                for value in self._matrix_vector(right_inverse, right_vector)
            ),
        )
        candidate_position = self._matrix_vector(covariance, information_vector)
        return (
            Vector3(
                x=candidate_position[0],
                y=candidate_position[1],
                z=candidate_position[2],
            ),
            Matrix3(values=covariance),
        )

    @staticmethod
    def _matrix_scale(matrix: tuple[float, ...], scale: float) -> tuple[float, ...]:
        return tuple(float(scale) * value for value in matrix)

    @staticmethod
    def _invert_matrix(values: tuple[float, ...]) -> tuple[float, ...] | None:
        if len(values) != 9 or not all(math.isfinite(value) for value in values):
            return None
        augmented = [
            [float(values[row * 3 + column]) for column in range(3)]
            + [1.0 if row == column else 0.0 for column in range(3)]
            for row in range(3)
        ]
        scale = max(1.0, *(abs(value) for value in values))
        tolerance = 1e-12 * scale
        for column in range(3):
            pivot = max(range(column, 3), key=lambda row: abs(augmented[row][column]))
            if abs(augmented[pivot][column]) <= tolerance:
                return None
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
            divisor = augmented[column][column]
            augmented[column] = [value / divisor for value in augmented[column]]
            for row in range(3):
                if row == column:
                    continue
                factor = augmented[row][column]
                if factor == 0.0:
                    continue
                augmented[row] = [
                    left - factor * right
                    for left, right in zip(augmented[row], augmented[column])
                ]
        result = tuple(augmented[row][column] for row in range(3) for column in range(3, 6))
        return result if all(math.isfinite(value) for value in result) else None

    @staticmethod
    def _has_any_registration_identity(world: object) -> bool:
        return any(
            getattr(world, field, None) is not None
            for field in (
                "world_frame_revision",
                "world_transform_sha256",
                "calibration_revision",
            )
        )

    @staticmethod
    def _has_canonical_identity(world: object) -> bool:
        return bool(
            getattr(world, "world_frame", None) == "backend_world_m"
            and getattr(world, "world_frame_revision", None)
            and getattr(world, "world_transform_sha256", None)
        )

    @staticmethod
    def _target_frame_identity(world: object) -> tuple[object, ...]:
        # A transform digest identifies one source-revision -> target-revision
        # edge.  Different cameras can legitimately use different edges to
        # reach the same target frame, so it is provenance rather than part of
        # the target-frame identity used for cross-camera fusion.
        return (
            getattr(world, "world_frame", None),
            getattr(world, "world_frame_revision", None),
        )

    @staticmethod
    def _registration_identity(world: object) -> tuple[object, ...]:
        return (
            getattr(world, "world_frame", None),
            getattr(world, "world_frame_revision", None),
            getattr(world, "world_transform_sha256", None),
            getattr(world, "calibration_revision", None),
        )

    @classmethod
    def _fusion_continuity_identity(
        cls,
        accepted: list[_ResolvedObservation],
    ) -> tuple[str, str, str | None] | None:
        run_ids = {item.envelope.producer.run_id for item in accepted}
        target_identities = {
            cls._target_frame_identity(item.envelope.payload.world)
            for item in accepted
            if item.envelope.payload.world is not None
        }
        if len(run_ids) != 1 or len(target_identities) != 1:
            return None
        world_frame, world_frame_revision = next(iter(target_identities))
        if world_frame != "backend_world_m":
            return None
        return (
            next(iter(run_ids)),
            world_frame,
            world_frame_revision,
        )

    @staticmethod
    def _accepted_trail_tokens(
        accepted: list[_ResolvedObservation],
        *,
        explicit_breaks_only: bool,
    ) -> tuple[tuple[str, int, str, int, int | None, int], ...]:
        tokens: list[tuple[str, int, str, int, int | None, int]] = []
        for item in accepted:
            if item.trail_segment_id is None:
                continue
            if explicit_breaks_only and not item.trail_break_required:
                continue
            tracklet = item.envelope.payload.tracklet
            tokens.append(
                (
                    tracklet.camera_id,
                    int(tracklet.source_id),
                    item.envelope.producer.run_id,
                    int(tracklet.tracker_id),
                    item.tracker_lifecycle_generation,
                    int(item.trail_segment_id),
                )
            )
        return tuple(
            sorted(
                tokens,
                key=lambda token: (
                    token[0],
                    token[1],
                    token[2],
                    token[3],
                    token[4] if token[4] is not None else -1,
                ),
            )
        )

    @staticmethod
    def _contains_new_trail_token(
        previous: tuple[tuple[str, int, str, int, int | None, int], ...],
        current: tuple[tuple[str, int, str, int, int | None, int], ...],
    ) -> bool:
        previous_by_source = {token[:5]: token for token in previous}
        return any(previous_by_source.get(token[:5]) != token for token in current)

    @staticmethod
    def _changed_trail_segment_tokens(
        previous: tuple[tuple[str, int, str, int, int | None, int], ...],
        current: tuple[tuple[str, int, str, int, int | None, int], ...],
    ) -> tuple[tuple[str, int, str, int, int | None, int], ...]:
        previous_by_source = {token[:5]: token for token in previous}
        return tuple(
            token
            for token in current
            if (prior := previous_by_source.get(token[:5])) is not None
            and prior[5] != token[5]
        )

    @classmethod
    def _merge_trail_tokens(
        cls,
        previous: tuple[tuple[str, int, str, int, int | None, int], ...],
        current: tuple[tuple[str, int, str, int, int | None, int], ...],
    ) -> tuple[tuple[str, int, str, int, int | None, int], ...]:
        merged: OrderedDict[
            tuple[str, int, str, int, int | None],
            tuple[str, int, str, int, int | None, int],
        ] = OrderedDict()
        for token in (*previous, *current):
            source_key = token[:5]
            merged.pop(source_key, None)
            merged[source_key] = token
        while len(merged) > cls.MAX_TRAIL_SOURCE_TOKENS:
            merged.popitem(last=False)
        return tuple(merged.values())

    @staticmethod
    def _position_clock(
        accepted: list[_ResolvedObservation],
    ) -> tuple[int | None, tuple[str, int, str, int | None] | None]:
        # A fused position spanning cameras has no single media clock. Use the
        # arrival clock for that cohort and across the subsequent handoff. A
        # single-source position can retain its private media clock identity.
        if len(accepted) != 1:
            return None, None
        item = accepted[0]
        tracklet = item.envelope.payload.tracklet
        return (
            item.envelope.media_pts_ns,
            (
                tracklet.camera_id,
                int(tracklet.source_id),
                item.envelope.producer.run_id,
                item.source_epoch,
            ),
        )

    @staticmethod
    def _position_step(
        previous: _FusedState | None,
        *,
        observed_at_us: int,
        media_pts_ns: int | None,
        position_source: tuple[str, int, str, int | None] | None,
        position: Vector3,
    ) -> _PositionStep | None:
        if previous is None:
            return None
        if (
            position_source is not None
            and previous.position_source is not None
            and position_source[:3] == previous.position_source[:3]
            and position_source[3] is not None
            and previous.position_source[3] is not None
            and position_source[3] != previous.position_source[3]
        ):
            # A source reconnect can rewind PTS while retaining the process run.
            # The private epoch is the clock identity boundary: seed this exact
            # position without deriving velocity across unrelated timelines.
            return None
        if (
            position_source is not None
            and position_source == previous.position_source
            and media_pts_ns is not None
            and previous.media_pts_ns is not None
            and media_pts_ns > previous.media_pts_ns
        ):
            dt_s = (media_pts_ns - previous.media_pts_ns) / 1_000_000_000.0
        elif observed_at_us > previous.observed_at_us:
            # Cross-source steps and missing, repeated, or regressed media PTS
            # have no common valid media clock, so retain the service-arrival
            # fallback used by the public snapshot lifecycle.
            dt_s = (observed_at_us - previous.observed_at_us) / 1_000_000.0
        else:
            return None
        if dt_s <= 0.0:
            return None
        velocity = Vector3(
            x=(position.x - previous.position.x) / dt_s,
            y=(position.y - previous.position.y) / dt_s,
            z=(position.z - previous.position.z) / dt_s,
        )
        speed = math.sqrt((velocity.x**2) + (velocity.y**2) + (velocity.z**2))
        return _PositionStep(
            dt_s=dt_s,
            distance_m=GlobalWorldFusion._distance(previous.position, position),
            speed_mps=speed,
            velocity=velocity,
        )

    @staticmethod
    def _diagonal(matrix: Matrix3) -> tuple[float, float, float]:
        values = matrix.values
        return float(values[0]), float(values[4]), float(values[8])

    @classmethod
    def _variance_sum(cls, matrix: Matrix3) -> float:
        return sum(cls._diagonal(matrix))

    @staticmethod
    def _distance(left: Vector3, right: Vector3) -> float:
        return math.sqrt(((left.x - right.x) ** 2) + ((left.y - right.y) ** 2) + ((left.z - right.z) ** 2))
