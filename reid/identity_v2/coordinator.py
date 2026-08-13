"""Order-invariant frame coordination above the identity-v2 runtime kernel."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence, Tuple

from .models import (
    HardConstraint,
    IdentityDecision,
    IdentityKind,
    OverlapSharePermit,
    TrackletObservation,
)
from .runtime import (
    IdentityV2Runtime,
    RuntimeIdentityDecision,
    RuntimeObservation,
    SubjectDescriptor,
)
from .store import EnrollmentObservationKey, PurgeResult, VisitorSession


class FrameCoordinatorError(RuntimeError):
    pass


class FrameReplayError(FrameCoordinatorError):
    pass


@dataclass(frozen=True)
class CameraOverlapEdge:
    """A topology edge on which cross-camera duplicate views are possible."""

    camera_a: str
    camera_b: str
    max_batch_gap_s: float = 0.5

    def __post_init__(self) -> None:
        camera_a = str(self.camera_a).strip()
        camera_b = str(self.camera_b).strip()
        if not camera_a or not camera_b:
            raise ValueError("camera overlap edge requires two camera IDs")
        if camera_a == camera_b:
            raise ValueError("camera overlap edge requires distinct cameras")
        if camera_b < camera_a:
            camera_a, camera_b = camera_b, camera_a
        max_gap = float(self.max_batch_gap_s)
        if not math.isfinite(max_gap) or max_gap <= 0.0:
            raise ValueError("camera overlap max_batch_gap_s must be positive")
        object.__setattr__(self, "camera_a", camera_a)
        object.__setattr__(self, "camera_b", camera_b)
        object.__setattr__(self, "max_batch_gap_s", max_gap)

    @property
    def pair(self) -> Tuple[str, str]:
        return self.camera_a, self.camera_b


@dataclass(frozen=True)
class FrameCoordinatorConfig:
    resident_confirmation_frames: int = 2
    visitor_match_confirmation_frames: int = 1
    provisional_frames_for_visitor: int = 3
    provisional_quality_floor: float = 0.60
    accepted_unknown_reset_frames: int = 6
    tracker_state_ttl_s: float = 2.0
    active_claim_ttl_s: float = 1.5
    visitor_release_after_s: float = 10.0
    visitor_session_ttl_s: float = 30.0
    visitor_slot_min: int = 1000
    visitor_slot_max: int = 1031
    visitor_record_interval_frames: int = 5
    retention_interval_s: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "resident_confirmation_frames",
            "visitor_match_confirmation_frames",
            "provisional_frames_for_visitor",
            "accepted_unknown_reset_frames",
            "visitor_record_interval_frames",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, int(getattr(self, name)))
        for name in (
            "tracker_state_ttl_s",
            "active_claim_ttl_s",
            "visitor_release_after_s",
            "visitor_session_ttl_s",
            "retention_interval_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        quality = float(self.provisional_quality_floor)
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("provisional_quality_floor must be within [0, 1]")
        object.__setattr__(self, "provisional_quality_floor", quality)
        slot_min = int(self.visitor_slot_min)
        slot_max = int(self.visitor_slot_max)
        if slot_min <= 0 or slot_max < slot_min:
            raise ValueError("visitor slot range is invalid")
        object.__setattr__(self, "visitor_slot_min", slot_min)
        object.__setattr__(self, "visitor_slot_max", slot_max)


@dataclass(frozen=True)
class PrimitiveFrameObservation:
    run_id: str
    camera_id: str
    tracker_id: str
    frame_id: int
    observation_id: str
    quality: float
    embedding: Tuple[float, ...]
    evidence: Tuple[str, ...] = field(default_factory=tuple)

    def to_runtime(self) -> RuntimeObservation:
        return RuntimeObservation(
            key=EnrollmentObservationKey(
                run_id=self.run_id,
                camera_id=self.camera_id,
                tracker_id=self.tracker_id,
                frame_id=self.frame_id,
                observation_id=self.observation_id,
            ),
            quality=self.quality,
            embedding=tuple(self.embedding),
            evidence=tuple(self.evidence),
        )


@dataclass(frozen=True)
class PublicIdentityOverlay:
    key: EnrollmentObservationKey
    identity_state: str
    subject_id: Optional[str]
    compatibility_sid: Optional[int]
    display_name: Optional[str]
    visitor_generation: Optional[int]
    calibrated_confidence: float
    reason: str
    provisional_evidence_count: int
    resolver_decision: IdentityDecision


@dataclass(frozen=True)
class ActiveIdentityClaim:
    subject_id: str
    tracklet_id: str
    camera_id: str
    last_seen_at: float
    expires_at: float


@dataclass(frozen=True)
class FrameBatchResult:
    overlays: Tuple[PublicIdentityOverlay, ...]
    minted_visitor_sessions: Tuple[VisitorSession, ...]
    released_visitor_session_uuids: Tuple[str, ...]
    retention: PurgeResult
    active_claims: Tuple[ActiveIdentityClaim, ...]
    candidate_rows: Tuple[TrackletObservation, ...] = field(default_factory=tuple)


@dataclass
class _TrackerState:
    last_seen_at: float
    last_frame_id: int
    last_observation_id: str
    continuity_subject_id: Optional[str] = None
    accepted_subject_id: Optional[str] = None
    pending_subject_id: Optional[str] = None
    pending_count: int = 0
    unknown_streak: int = 0
    provisional_observations: list[RuntimeObservation] = field(default_factory=list)
    last_visitor_record_frame: Optional[int] = None


@dataclass
class _ActiveClaimState:
    subject_id: str
    tracklet_id: str
    camera_id: str
    last_seen_at: float


class IdentityFrameCoordinator:
    """Convert whole-frame evidence into conservative public identity overlays."""

    def __init__(
        self,
        runtime: IdentityV2Runtime,
        *,
        config: Optional[FrameCoordinatorConfig] = None,
        camera_overlap_edges: Sequence[CameraOverlapEdge] = (),
    ) -> None:
        self.runtime = runtime
        self.config = config or FrameCoordinatorConfig()
        edges = tuple(camera_overlap_edges)
        overlap_by_pair: dict[Tuple[str, str], CameraOverlapEdge] = {}
        for edge in edges:
            if not isinstance(edge, CameraOverlapEdge):
                raise TypeError("camera_overlap_edges must contain CameraOverlapEdge")
            if edge.pair in overlap_by_pair:
                raise ValueError(f"duplicate camera overlap edge: {edge.pair!r}")
            if edge.max_batch_gap_s > self.config.active_claim_ttl_s:
                raise ValueError(
                    "camera overlap max gap cannot exceed active_claim_ttl_s"
                )
            overlap_by_pair[edge.pair] = edge
        self._lock = threading.RLock()
        self._trackers: dict[str, _TrackerState] = {}
        self._visitor_last_seen: dict[str, float] = {}
        self._active_claims: dict[str, dict[str, _ActiveClaimState]] = {}
        self._camera_overlap_by_pair = overlap_by_pair
        self._last_batch_timestamp: Optional[float] = None
        self._last_retention_at: Optional[float] = None

    def process_frame(
        self,
        observations: Sequence[PrimitiveFrameObservation],
        *,
        timestamp: Optional[float] = None,
        constraints: Sequence[HardConstraint] = (),
        overlap_permits: Sequence[OverlapSharePermit] = (),
    ) -> FrameBatchResult:
        explicit_timestamp = None if timestamp is None else float(timestamp)
        if explicit_timestamp is not None and not math.isfinite(explicit_timestamp):
            raise ValueError("timestamp must be finite")
        runtime_observations = tuple(
            sorted(
                (row.to_runtime() for row in observations),
                key=lambda row: (
                    row.key.tracklet_id,
                    row.key.frame_id,
                    row.key.observation_id,
                ),
            )
        )
        with self._lock:
            # The integration path should omit timestamp so the processing clock
            # is sampled after serialization. Explicit timestamps exist for
            # deterministic replay/tests and must use the same epoch clock domain.
            now = (
                self.runtime.current_time()
                if explicit_timestamp is None
                else explicit_timestamp
            )
            if (
                self._last_batch_timestamp is not None
                and now < self._last_batch_timestamp
            ):
                raise FrameReplayError("frame batch timestamp moved backwards")
            self._validate_monotonic(runtime_observations)
            retention = self._run_retention_if_due(now)
            self._expire_tracker_state(now)
            self._expire_active_claims(now)
            released = self._release_inactive_visitors(now)
            hot_subject_ids = {
                row.subject_id for row in self.runtime.hot_subjects(now=now)
            }
            self._prune_inactive_claims(hot_subject_ids)
            current_permits, cross_batch_permits = self._partition_overlap_permits(
                runtime_observations,
                overlap_permits,
                now,
            )
            effective_constraints = self._active_claim_constraints(
                runtime_observations,
                constraints,
                cross_batch_permits,
                hot_subject_ids,
            )
            decisions = self.runtime.resolve_batch(
                runtime_observations,
                constraints=effective_constraints,
                overlap_permits=current_permits,
                now=now,
            )
            self.runtime.ingest_observations(runtime_observations, now=now)
            decision_by_tracklet = {
                row.observation_key.tracklet_id: row for row in decisions
            }
            minted: list[VisitorSession] = []
            overlays: list[PublicIdentityOverlay] = []
            current_tracklets = {
                observation.key.tracklet_id for observation in runtime_observations
            }
            for observation in runtime_observations:
                resolved = decision_by_tracklet[observation.key.tracklet_id]
                resolved = self._annotate_cross_batch_overlap(
                    resolved,
                    cross_batch_permits,
                    current_tracklets,
                )
                overlay, new_session = self._apply_decision(observation, resolved, now)
                overlays.append(overlay)
                if overlay.subject_id is not None:
                    self._record_active_claim(
                        overlay.subject_id,
                        observation.key.tracklet_id,
                        observation.key.camera_id,
                        now,
                    )
                if new_session is not None:
                    minted.append(new_session)
            result = FrameBatchResult(
                overlays=tuple(overlays),
                minted_visitor_sessions=tuple(minted),
                released_visitor_session_uuids=tuple(released),
                retention=retention,
                active_claims=self._active_claim_snapshot(),
                candidate_rows=tuple(row.candidate_observation for row in decisions),
            )
            self._last_batch_timestamp = now
            return result

    @staticmethod
    def _annotate_cross_batch_overlap(
        resolved: RuntimeIdentityDecision,
        valid_claim_permits: set[Tuple[str, str, str]],
        current_tracklets: set[str],
    ) -> RuntimeIdentityDecision:
        subject = resolved.subject
        if subject is None or resolved.is_unknown:
            return resolved
        tracklet_id = resolved.observation_key.tracklet_id
        shared = any(
            identity_id == subject.subject_id
            and tracklet_id in (tracklet_a, tracklet_b)
            and (
                tracklet_a not in current_tracklets
                or tracklet_b not in current_tracklets
            )
            for identity_id, tracklet_a, tracklet_b in valid_claim_permits
        )
        if not shared:
            return resolved
        decision = replace(
            resolved.decision,
            evidence=resolved.decision.evidence
            + ("cross_batch_overlap_share_permitted",),
        )
        return replace(resolved, decision=decision)

    def tracker_state_count(self) -> int:
        with self._lock:
            return len(self._trackers)

    def active_claims(
        self, *, now: Optional[float] = None
    ) -> Tuple[ActiveIdentityClaim, ...]:
        with self._lock:
            timestamp = self.runtime.current_time() if now is None else float(now)
            self._expire_active_claims(timestamp)
            return self._active_claim_snapshot()

    def _validate_monotonic(
        self,
        observations: Sequence[RuntimeObservation],
    ) -> None:
        tracklets = [row.key.tracklet_id for row in observations]
        if len(tracklets) != len(set(tracklets)):
            raise FrameReplayError(
                "a frame contains duplicate tracker-local observations"
            )
        for observation in observations:
            state = self._trackers.get(observation.key.tracklet_id)
            if state is None:
                continue
            if observation.key.frame_id <= state.last_frame_id:
                raise FrameReplayError(
                    "tracker frame replay or non-monotonic frame ID was rejected"
                )

    def _partition_overlap_permits(
        self,
        observations: Sequence[RuntimeObservation],
        permits: Sequence[OverlapSharePermit],
        now: float,
    ) -> Tuple[Tuple[OverlapSharePermit, ...], set[Tuple[str, str, str]]]:
        current = {
            observation.key.tracklet_id: observation.key.camera_id
            for observation in observations
        }
        active_by_tracklet = {
            claim.tracklet_id: claim
            for claims in self._active_claims.values()
            for claim in claims.values()
        }
        current_permits: list[OverlapSharePermit] = []
        valid_claim_permits: set[Tuple[str, str, str]] = set()
        seen: set[Tuple[str, str, str]] = set()
        for permit in permits:
            token = (permit.identity_id, permit.tracklet_a, permit.tracklet_b)
            if token in seen:
                raise ValueError(f"duplicate overlap permit: {token!r}")
            seen.add(token)
            a_current = permit.tracklet_a in current
            b_current = permit.tracklet_b in current
            if a_current and b_current:
                self._validate_overlap_topology(
                    current[permit.tracklet_a],
                    current[permit.tracklet_b],
                    batch_gap_s=0.0,
                )
                current_permits.append(permit)
                valid_claim_permits.add(token)
                continue
            if a_current == b_current:
                raise ValueError(
                    "overlap permit must join two current tracklets or one current "
                    "tracklet to one active claim"
                )
            current_tracklet = permit.tracklet_a if a_current else permit.tracklet_b
            active_tracklet = permit.tracklet_b if a_current else permit.tracklet_a
            active_claim = active_by_tracklet.get(active_tracklet)
            if active_claim is None:
                raise ValueError(
                    f"overlap permit references inactive tracklet {active_tracklet!r}"
                )
            if active_claim.subject_id != permit.identity_id:
                raise ValueError(
                    "cross-batch overlap permit identity does not match the active claim"
                )
            self._validate_overlap_topology(
                current[current_tracklet],
                active_claim.camera_id,
                batch_gap_s=now - active_claim.last_seen_at,
            )
            valid_claim_permits.add(token)
        return tuple(current_permits), valid_claim_permits

    def _validate_overlap_topology(
        self,
        camera_a: str,
        camera_b: str,
        *,
        batch_gap_s: float,
    ) -> None:
        if camera_a == camera_b:
            raise ValueError("identity sharing is not allowed within one camera")
        pair = tuple(sorted((camera_a, camera_b)))
        edge = self._camera_overlap_by_pair.get(pair)
        if edge is None:
            raise ValueError(
                f"identity sharing is not configured for camera pair {pair!r}"
            )
        if batch_gap_s < 0.0 or batch_gap_s > edge.max_batch_gap_s:
            raise ValueError(
                "overlap permit is outside the configured cross-camera batch window"
            )

    def _active_claim_constraints(
        self,
        observations: Sequence[RuntimeObservation],
        constraints: Sequence[HardConstraint],
        valid_claim_permits: set[Tuple[str, str, str]],
        hot_subject_ids: set[str],
    ) -> Tuple[HardConstraint, ...]:
        current_tracklets = {
            observation.key.tracklet_id for observation in observations
        }
        merged: dict[Tuple[str, str], HardConstraint] = {}
        for constraint in constraints:
            key = (constraint.tracklet_id, constraint.identity_id)
            if key in merged:
                raise ValueError(f"duplicate hard constraint for {key!r}")
            if constraint.tracklet_id not in current_tracklets:
                raise ValueError(
                    "hard constraint references tracklet outside the current batch"
                )
            merged[key] = constraint

        # Once a subject has been accepted for a camera-local tracker, that
        # subject is immutable for the lifetime of the tracker state.  Keep the
        # accepted subject eligible only when it independently passes every
        # open-set gate; hard-mask every alternative instead of using a score
        # bonus that could force a weak match.  A tracker may acquire a new
        # subject only after its evidence gap expires the complete tracker
        # state in ``_expire_tracker_state``.
        for observation in observations:
            tracklet_id = observation.key.tracklet_id
            state = self._trackers.get(tracklet_id)
            locked_subject_id = None if state is None else state.continuity_subject_id
            if locked_subject_id is None:
                continue
            for subject_id in sorted(hot_subject_ids):
                if subject_id == locked_subject_id:
                    continue
                key = (tracklet_id, subject_id)
                existing = merged.get(key)
                if existing is not None and not existing.allowed:
                    continue
                merged[key] = HardConstraint(
                    tracklet_id,
                    subject_id,
                    False,
                    "tracker_continuity_lock",
                )

        for observation in observations:
            tracklet_id = observation.key.tracklet_id
            for subject_id, claims in sorted(self._active_claims.items()):
                if subject_id not in hot_subject_ids:
                    continue
                conflict = False
                for active_tracklet in sorted(claims):
                    if active_tracklet == tracklet_id:
                        continue
                    a, b = sorted((tracklet_id, active_tracklet))
                    if (subject_id, a, b) not in valid_claim_permits:
                        conflict = True
                        break
                if not conflict:
                    continue
                key = (tracklet_id, subject_id)
                existing = merged.get(key)
                if existing is not None and not existing.allowed:
                    continue
                merged[key] = HardConstraint(
                    tracklet_id,
                    subject_id,
                    False,
                    "active_copresence_claim",
                )
        return tuple(merged[key] for key in sorted(merged))

    def _expire_tracker_state(self, now: float) -> None:
        expired = sorted(
            tracklet_id
            for tracklet_id, state in self._trackers.items()
            if now - state.last_seen_at > self.config.tracker_state_ttl_s
        )
        for tracklet_id in expired:
            self._trackers.pop(tracklet_id, None)

    def _run_retention_if_due(self, now: float) -> PurgeResult:
        """Keep durable cleanup off the per-frame identity hot path."""

        previous = self._last_retention_at
        if (
            previous is not None
            and now - previous < self.config.retention_interval_s
        ):
            return PurgeResult(0, 0, 0, 0)
        result = self.runtime.run_retention(now=now)
        self._last_retention_at = now
        return result

    def _expire_active_claims(self, now: float) -> None:
        for subject_id in sorted(tuple(self._active_claims)):
            claims = self._active_claims[subject_id]
            expired = [
                tracklet_id
                for tracklet_id, claim in claims.items()
                if now - claim.last_seen_at >= self.config.active_claim_ttl_s
            ]
            for tracklet_id in expired:
                claims.pop(tracklet_id, None)
            if not claims:
                self._active_claims.pop(subject_id, None)

    def _prune_inactive_claims(self, hot_subject_ids: set[str]) -> None:
        for subject_id in tuple(self._active_claims):
            if subject_id not in hot_subject_ids:
                self._active_claims.pop(subject_id, None)

    def _record_active_claim(
        self,
        subject_id: str,
        tracklet_id: str,
        camera_id: str,
        now: float,
    ) -> None:
        for claimed_subject_id in tuple(self._active_claims):
            claims = self._active_claims[claimed_subject_id]
            claims.pop(tracklet_id, None)
            if not claims:
                self._active_claims.pop(claimed_subject_id, None)
        self._active_claims.setdefault(subject_id, {})[tracklet_id] = _ActiveClaimState(
            subject_id=subject_id,
            tracklet_id=tracklet_id,
            camera_id=camera_id,
            last_seen_at=now,
        )

    def _active_claim_snapshot(self) -> Tuple[ActiveIdentityClaim, ...]:
        return tuple(
            ActiveIdentityClaim(
                subject_id=claim.subject_id,
                tracklet_id=claim.tracklet_id,
                camera_id=claim.camera_id,
                last_seen_at=claim.last_seen_at,
                expires_at=claim.last_seen_at + self.config.active_claim_ttl_s,
            )
            for subject_id in sorted(self._active_claims)
            for claim in sorted(
                self._active_claims[subject_id].values(),
                key=lambda row: row.tracklet_id,
            )
        )

    def _release_inactive_visitors(self, now: float) -> list[str]:
        active = {
            row.session_uuid: row
            for row in self.runtime.active_visitor_sessions(now=now)
        }
        released = []
        for session_uuid, last_seen in sorted(self._visitor_last_seen.items()):
            if session_uuid not in active:
                self._visitor_last_seen.pop(session_uuid, None)
                continue
            if now - last_seen < self.config.visitor_release_after_s:
                continue
            self.runtime.release_visitor_session(session_uuid, now=now)
            self._visitor_last_seen.pop(session_uuid, None)
            released.append(session_uuid)
        return released

    def _apply_decision(
        self,
        observation: RuntimeObservation,
        resolved: RuntimeIdentityDecision,
        now: float,
    ) -> Tuple[PublicIdentityOverlay, Optional[VisitorSession]]:
        tracklet_id = observation.key.tracklet_id
        state = self._trackers.get(tracklet_id)
        if state is None:
            state = _TrackerState(
                last_seen_at=now,
                last_frame_id=observation.key.frame_id,
                last_observation_id=observation.key.observation_id,
            )
            self._trackers[tracklet_id] = state

        minted: Optional[VisitorSession] = None
        if resolved.subject is not None and not resolved.is_unknown:
            overlay = self._known_decision(observation, resolved, state, now)
        else:
            overlay, minted = self._unknown_decision(observation, resolved, state, now)
        state.last_seen_at = now
        state.last_frame_id = observation.key.frame_id
        state.last_observation_id = observation.key.observation_id
        return overlay, minted

    def _known_decision(
        self,
        observation: RuntimeObservation,
        resolved: RuntimeIdentityDecision,
        state: _TrackerState,
        now: float,
    ) -> PublicIdentityOverlay:
        subject = resolved.subject
        assert subject is not None
        if (
            state.continuity_subject_id is not None
            and state.continuity_subject_id != subject.subject_id
        ):
            # The pre-resolver hard mask above covers the stable gallery
            # snapshot.  This postcondition also closes a concurrent gallery
            # refresh race: changed evidence becomes open-set unknown and can
            # never publish a direct subject flip.
            return self._continuity_conflict_overlay(
                observation,
                resolved,
                state,
            )
        if state.pending_subject_id == subject.subject_id:
            state.pending_count += 1
        else:
            state.pending_subject_id = subject.subject_id
            state.pending_count = 1
        threshold = (
            self.config.resident_confirmation_frames
            if subject.identity_kind is IdentityKind.RESIDENT
            else self.config.visitor_match_confirmation_frames
        )
        state.provisional_observations.clear()
        state.unknown_streak = 0
        if state.pending_count < threshold:
            return self._unknown_overlay(
                observation,
                resolved.decision,
                reason="identity_hysteresis_pending",
                provisional_count=0,
            )

        state.accepted_subject_id = subject.subject_id
        if state.continuity_subject_id is None:
            state.continuity_subject_id = subject.subject_id
        if subject.identity_kind is IdentityKind.VISITOR:
            assert subject.visitor_session_uuid is not None
            self._visitor_last_seen[subject.visitor_session_uuid] = now
            if (
                state.last_visitor_record_frame is None
                or observation.key.frame_id - state.last_visitor_record_frame
                >= self.config.visitor_record_interval_frames
            ):
                self.runtime.record_visitor_observation(
                    subject.visitor_session_uuid,
                    observation,
                    now=now,
                )
                state.last_visitor_record_frame = observation.key.frame_id
        return self._known_overlay(observation, resolved.decision, subject)

    def _unknown_decision(
        self,
        observation: RuntimeObservation,
        resolved: RuntimeIdentityDecision,
        state: _TrackerState,
        now: float,
    ) -> Tuple[PublicIdentityOverlay, Optional[VisitorSession]]:
        state.pending_subject_id = None
        state.pending_count = 0
        state.unknown_streak += 1
        if state.continuity_subject_id is not None:
            if state.unknown_streak >= self.config.accepted_unknown_reset_frames:
                state.accepted_subject_id = None
            state.provisional_observations.clear()
            return (
                self._unknown_overlay(
                    observation,
                    resolved.decision,
                    reason=resolved.decision.reason,
                    provisional_count=0,
                ),
                None,
            )

        if observation.quality < self.config.provisional_quality_floor:
            state.provisional_observations.clear()
            return (
                self._unknown_overlay(
                    observation,
                    resolved.decision,
                    reason="provisional_quality_below_floor",
                    provisional_count=0,
                ),
                None,
            )
        state.provisional_observations.append(observation)
        state.provisional_observations = state.provisional_observations[
            -self.config.provisional_frames_for_visitor :
        ]
        count = len(state.provisional_observations)
        if count < self.config.provisional_frames_for_visitor:
            return (
                self._unknown_overlay(
                    observation,
                    resolved.decision,
                    reason="provisional_evidence_pending",
                    provisional_count=count,
                ),
                None,
            )

        slot = self._next_visitor_slot(now)
        if slot is None:
            return (
                self._unknown_overlay(
                    observation,
                    resolved.decision,
                    reason="visitor_capacity_exhausted",
                    provisional_count=count,
                ),
                None,
            )
        session = self.runtime.open_visitor_session(
            slot=slot,
            ttl_s=self.config.visitor_session_ttl_s,
            now=now,
        )
        for provisional in state.provisional_observations:
            self.runtime.record_visitor_observation(
                session.session_uuid,
                provisional,
                now=now,
            )
        descriptor = self._visitor_descriptor(session, now)
        state.accepted_subject_id = descriptor.subject_id
        state.continuity_subject_id = descriptor.subject_id
        state.pending_subject_id = descriptor.subject_id
        state.pending_count = self.config.visitor_match_confirmation_frames
        state.unknown_streak = 0
        state.provisional_observations.clear()
        state.last_visitor_record_frame = observation.key.frame_id
        self._visitor_last_seen[session.session_uuid] = now
        return (
            self._known_overlay(
                observation,
                resolved.decision,
                descriptor,
                reason="visitor_provisional_confirmed",
                confidence=observation.quality,
            ),
            session,
        )

    def _continuity_conflict_overlay(
        self,
        observation: RuntimeObservation,
        resolved: RuntimeIdentityDecision,
        state: _TrackerState,
    ) -> PublicIdentityOverlay:
        """Fail closed if a refreshed gallery escaped the continuity mask."""

        state.pending_subject_id = None
        state.pending_count = 0
        state.unknown_streak += 1
        if state.unknown_streak >= self.config.accepted_unknown_reset_frames:
            state.accepted_subject_id = None
        state.provisional_observations.clear()
        selected = resolved.decision
        decision = replace(
            selected,
            identity_id=None,
            identity_kind=None,
            is_unknown=True,
            calibrated_confidence=max(
                0.0,
                min(1.0, 1.0 - float(selected.calibrated_confidence)),
            ),
            prior_contribution=0.0,
            assignment_utility=self.runtime.resolver.scorer.policy.unknown_utility,
            reason="tracker_continuity_subject_conflict",
            evidence=selected.evidence
            + ("tracker_continuity_lock_rejected_subject_change",),
        )
        return self._unknown_overlay(
            observation,
            decision,
            reason="tracker_continuity_subject_conflict",
            provisional_count=0,
        )

    def _next_visitor_slot(self, now: float) -> Optional[int]:
        used = {
            session.slot for session in self.runtime.active_visitor_sessions(now=now)
        }
        for slot in range(
            self.config.visitor_slot_min, self.config.visitor_slot_max + 1
        ):
            if slot not in used:
                return slot
        return None

    def _visitor_descriptor(
        self,
        session: VisitorSession,
        now: float,
    ) -> SubjectDescriptor:
        for descriptor in self.runtime.hot_subjects(now=now):
            if descriptor.visitor_session_uuid == session.session_uuid:
                return descriptor
        raise FrameCoordinatorError(
            "new visitor session did not become a hot subject after exemplar recording"
        )

    @staticmethod
    def _known_overlay(
        observation: RuntimeObservation,
        decision: IdentityDecision,
        subject: SubjectDescriptor,
        *,
        reason: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> PublicIdentityOverlay:
        return PublicIdentityOverlay(
            key=observation.key,
            identity_state=subject.identity_kind.value,
            subject_id=subject.subject_id,
            compatibility_sid=subject.compatibility_sid,
            display_name=(
                subject.display_name
                if subject.identity_kind is IdentityKind.RESIDENT
                else None
            ),
            visitor_generation=subject.visitor_generation,
            calibrated_confidence=(
                decision.calibrated_confidence if confidence is None else confidence
            ),
            reason=decision.reason if reason is None else reason,
            provisional_evidence_count=0,
            resolver_decision=decision,
        )

    @staticmethod
    def _unknown_overlay(
        observation: RuntimeObservation,
        decision: IdentityDecision,
        *,
        reason: str,
        provisional_count: int,
    ) -> PublicIdentityOverlay:
        return PublicIdentityOverlay(
            key=observation.key,
            identity_state="unknown",
            subject_id=None,
            compatibility_sid=None,
            display_name=None,
            visitor_generation=None,
            calibrated_confidence=decision.calibrated_confidence,
            reason=reason,
            provisional_evidence_count=provisional_count,
            resolver_decision=decision,
        )


__all__ = [
    "ActiveIdentityClaim",
    "CameraOverlapEdge",
    "FrameBatchResult",
    "FrameCoordinatorConfig",
    "FrameCoordinatorError",
    "FrameReplayError",
    "IdentityFrameCoordinator",
    "PrimitiveFrameObservation",
    "PublicIdentityOverlay",
]
