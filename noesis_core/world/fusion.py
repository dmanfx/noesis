from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

from noesis_core.contracts.base import Matrix3, ProducerRef, Vector3
from noesis_core.contracts.identity import SubjectRef
from noesis_core.contracts.observation import ObservationEnvelope
from noesis_core.contracts.world import (
    EntityLifecycle,
    WorldEntity,
    WorldSnapshot,
    WorldSourceEvidence,
)


class ObservationOrderError(ValueError):
    pass


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


@dataclass(frozen=True)
class _FusedState:
    observed_at_us: int
    media_pts_ns: int | None
    position_source: tuple[str, int, str, int | None] | None
    position: Vector3
    continuity_identity: tuple[str, str, str | None] | None
    last_trail_reset_tokens: tuple[
        tuple[str, int, str, int, int | None, int], ...
    ]
    last_trail_segment_tokens: tuple[
        tuple[str, int, str, int, int | None, int], ...
    ]
    entity: WorldEntity


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
        self._entity_by_tracklet: dict[tuple[str, int, str, int], str] = {}
        self._last_fused: dict[str, _FusedState] = {}
        self._snapshot_sequence = 0

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
        candidate._snapshot_sequence = int(self._snapshot_sequence)
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
    ) -> None:
        if envelope.payload.world is None:
            return
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
        tracklet_key = (
            camera_id,
            envelope.payload.tracklet.source_id,
            envelope.producer.run_id,
            envelope.payload.tracklet.tracker_id,
        )
        prior_entity_id = self._entity_by_tracklet.get(tracklet_key)
        if prior_entity_id is not None and prior_entity_id != subject.subject_id:
            self._remove_tracklet_observation(prior_entity_id, tracklet_key)

        entity_observations = self._by_entity.setdefault(subject.subject_id, {})
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
                self._last_fused.pop(subject.subject_id, None)
            replaced_key = (
                camera_id,
                replaced.envelope.payload.tracklet.source_id,
                replaced.envelope.producer.run_id,
                replaced.envelope.payload.tracklet.tracker_id,
            )
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
        )
        self._entity_by_tracklet[tracklet_key] = subject.subject_id
        self._by_entity.move_to_end(subject.subject_id)
        while len(self._by_entity) > self.config.max_entities:
            entity_id, _ = self._by_entity.popitem(last=False)
            self._last_fused.pop(entity_id, None)
            self._drop_entity_tracklets(entity_id)

    def clear_source(self, camera_id: str, source_id: int, run_id: str) -> int:
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
            tracklet_key = (
                camera_key,
                source_key,
                run_key,
                int(tracklet.tracker_id),
            )
            observations.pop(camera_key, None)
            self._entity_by_tracklet.pop(tracklet_key, None)
            removed += 1
            if not observations:
                self._by_entity.pop(entity_id, None)
                self._last_fused.pop(entity_id, None)
        return removed

    def snapshot(self, *, published_at_us: int) -> WorldSnapshot:
        if published_at_us <= 0:
            raise ValueError("published_at_us must be positive")
        self._snapshot_sequence += 1
        entities: list[WorldEntity] = []
        observation_times: list[int] = []
        expired: list[str] = []
        for entity_id, observations in list(self._by_entity.items()):
            entity = self._fuse_entity(entity_id, observations.values(), published_at_us)
            if entity is None:
                expired.append(entity_id)
                continue
            entities.append(entity)
            observation_times.append(entity.observed_at_us)
        for entity_id in expired:
            self._by_entity.pop(entity_id, None)
            self._last_fused.pop(entity_id, None)
            self._drop_entity_tracklets(entity_id)

        observed_end = max(observation_times, default=published_at_us)
        observed_start = min(observation_times, default=published_at_us)
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
        tracklet_key: tuple[str, int, str, int],
    ) -> None:
        observations = self._by_entity.get(entity_id)
        if observations is None:
            self._entity_by_tracklet.pop(tracklet_key, None)
            return
        camera_id = tracklet_key[0]
        current = observations.get(camera_id)
        if current is not None:
            current_key = (
                camera_id,
                current.envelope.payload.tracklet.source_id,
                current.envelope.producer.run_id,
                current.envelope.payload.tracklet.tracker_id,
            )
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
            return None
        usable.sort(
            key=lambda item: (
                item.envelope.observed_at_us,
                item.envelope.payload.world.confidence if item.envelope.payload.world else 0.0,
                item.envelope.payload.tracklet.camera_id,
            ),
            reverse=True,
        )
        latest_at = usable[0].envelope.observed_at_us
        age_us = max(0, now_us - latest_at)
        if age_us > self.config.lost_ttl_us:
            return None
        lifecycle = EntityLifecycle.PRESENT if age_us <= self.config.present_ttl_us else EntityLifecycle.HELD
        contemporaneous = [
            item
            for item in usable
            if latest_at - item.envelope.observed_at_us <= self.config.simultaneous_window_us
        ]
        authoritative = [
            item
            for item in contemporaneous
            if item.envelope.payload.world is not None
            and item.envelope.payload.world.quality != "held"
        ]
        ignored_held: list[tuple[_ResolvedObservation, str]] = []
        if authoritative:
            # A state-integrated held point keeps one camera current, but it is
            # not an independent metric hypothesis and must not pull a fresh
            # camera observation through covariance intersection.
            fusion_candidates = authoritative
            ignored_held = [
                (item, "non_authoritative_held_continuation")
                for item in contemporaneous
                if item not in authoritative
            ]
        else:
            # With no fresh metric camera, use the newest held continuation
            # exactly.  Never average two process-model continuations: their
            # errors can be correlated and the resulting point would exist in
            # neither tracking cohort.
            selected_held = min(
                contemporaneous,
                key=self._held_continuation_sort_key,
            )
            fusion_candidates = [selected_held]
            ignored_held = [
                (item, "non_authoritative_held_continuation")
                for item in contemporaneous
                if item is not selected_held
            ]

        anchor = min(fusion_candidates, key=self._anchor_sort_key)
        anchor_world = anchor.envelope.payload.world
        assert anchor_world is not None
        anchor_target_identity = self._target_frame_identity(anchor_world)
        anchor_has_target_identity = self._has_canonical_identity(anchor_world)

        accepted: list[_ResolvedObservation] = []
        rejected: list[tuple[_ResolvedObservation, str]] = []
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
            distance = self._distance(anchor_world.position, world.position)
            if distance > self.config.conflict_distance_m:
                rejected.append((item, f"position_conflict:{distance:.3f}m"))
            else:
                accepted.append(item)
        if not accepted:
            accepted = [anchor]

        position, covariance = self._weighted_position(accepted)
        source_evidence: list[WorldSourceEvidence] = []
        rejected_ids = {id(item): reason for item, reason in rejected}
        ignored_held_ids = {id(item): reason for item, reason in ignored_held}
        for item in contemporaneous:
            world = item.envelope.payload.world
            assert world is not None
            reason = rejected_ids.get(id(item)) or ignored_held_ids.get(id(item))
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
        continuity_identity: tuple[str, str, str | None] | None = None
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
            posture=anchor_world.posture,
            velocity_mps=velocity,
            room_id=room_id,
            observed_at_us=latest_at,
            stale_after_us=latest_at + self.config.lost_ttl_us,
            sources=tuple(sorted(source_evidence, key=lambda item: item.camera_id)),
            conflict=bool(rejected) or room_conflict,
            conflict_reason="; ".join(conflict_reasons) or None,
        )
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

    def _anchor_sort_key(self, item: _ResolvedObservation) -> tuple[int, float, float, int, str]:
        world = item.envelope.payload.world
        assert world is not None
        variance = self._variance_sum(world.covariance)
        return (
            0 if self._has_canonical_identity(world) else 1,
            variance,
            -float(world.confidence),
            -int(item.envelope.observed_at_us),
            item.envelope.payload.tracklet.camera_id,
        )

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

    def _weighted_position(self, observations: list[_ResolvedObservation]) -> tuple[Vector3, Matrix3]:
        first = observations[0]
        first_world = first.envelope.payload.world
        assert first_world is not None
        position = first_world.position
        covariance = Matrix3(values=self._regularized_covariance(first_world.covariance))
        # Camera correlations are not proven by this service.  Covariance
        # intersection retains the complete anisotropic matrix while avoiding
        # the unjustified overconfidence of summing independent precisions.
        for item in observations[1:]:
            world = item.envelope.payload.world
            assert world is not None
            position, covariance = self._covariance_intersection(
                position,
                covariance,
                world.position,
                world.covariance,
            )
        return (
            position,
            covariance,
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
