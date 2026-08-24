"""Store-backed identity-v2 runtime facade without legacy runtime coupling."""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from .models import (
    CandidateEvidence,
    HardConstraint,
    IdentityDecision,
    IdentityKind,
    OverlapSharePermit,
    TrackletObservation,
)
from .resolver import BatchIdentityResolver
from .scoring import OpenSetPolicy
from .store import (
    DeletionResult,
    EnrollmentConfirmationResult,
    EnrollmentObservationKey,
    EnrollmentProposalRecord,
    EnrollmentProposalResult,
    IdentityHealth,
    IdentityStore,
    ModelProfileMismatch,
    PurgeResult,
    ResidentRecord,
    VisitorExemplarRecord,
    VisitorSession,
)


@dataclass(frozen=True)
class RuntimeObservation:
    key: EnrollmentObservationKey
    quality: float
    embedding: Tuple[float, ...]
    evidence: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.key, EnrollmentObservationKey):
            raise TypeError("key must be an EnrollmentObservationKey")
        quality = float(self.quality)
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("quality must be finite and within [0, 1]")
        vector = tuple(float(value) for value in self.embedding)
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("embedding must contain finite values")
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "embedding", vector)
        object.__setattr__(self, "evidence", tuple(str(item) for item in self.evidence))


@dataclass(frozen=True)
class SubjectDescriptor:
    subject_id: str
    identity_kind: IdentityKind
    compatibility_sid: int
    display_name: Optional[str]
    resident_uuid: Optional[str]
    visitor_session_uuid: Optional[str]
    visitor_slot: Optional[int]
    visitor_generation: Optional[int]
    expires_at: Optional[float]


@dataclass(frozen=True)
class RuntimeIdentityDecision:
    observation_key: EnrollmentObservationKey
    candidate_observation: TrackletObservation
    decision: IdentityDecision
    subject: Optional[SubjectDescriptor]

    @property
    def is_unknown(self) -> bool:
        return self.decision.is_unknown


class ObservationEvidenceError(RuntimeError):
    pass


class ObservationEvidenceUnavailable(ObservationEvidenceError):
    pass


class ObservationEvidenceExpired(ObservationEvidenceError):
    pass


class ObservationEvidenceConsumed(ObservationEvidenceError):
    def __init__(self, proposal_uuid: str) -> None:
        self.proposal_uuid = str(proposal_uuid)
        super().__init__(
            f"observation evidence was already consumed by proposal {self.proposal_uuid}"
        )


class ObservationEvidenceConflict(ObservationEvidenceError):
    pass


@dataclass(frozen=True)
class ObservationCacheHealth:
    entry_count: int
    max_entries: int
    ttl_s: float
    expired_evictions: int
    capacity_evictions: int


@dataclass(frozen=True)
class _CachedObservation:
    observation: RuntimeObservation
    model_fingerprint: str
    embedding_dim: int
    cached_at: float
    expires_at: float


@dataclass(frozen=True)
class _HotSubject:
    descriptor: SubjectDescriptor
    normalized_vectors: Tuple[Tuple[float, ...], ...]


def resident_subject_id(resident_uuid: str) -> str:
    return f"resident:{resident_uuid}"


def visitor_subject_id(session_uuid: str, generation: int) -> str:
    return f"visitor:{session_uuid}:generation:{int(generation)}"


class IdentityV2Runtime:
    """Single-home identity kernel backed only by :class:`IdentityStore`.

    Resolution reads immutable in-memory gallery snapshots. Mutations are
    serialized, committed to SQLite first, and then refresh the snapshot. An
    in-flight resolution is therefore linearizable at the snapshot it acquired.
    """

    def __init__(
        self,
        store: IdentityStore,
        *,
        model_fingerprint: str,
        embedding_dim: int,
        policy: Optional[OpenSetPolicy] = None,
        clock: Callable[[], float] = time.time,
        observation_cache_ttl_s: float = 10.0,
        observation_cache_max_entries: int = 512,
        visitor_gallery_max_exemplars: int = 32,
    ) -> None:
        fingerprint = str(model_fingerprint or "").strip()
        dimension = int(embedding_dim)
        if not fingerprint or dimension <= 0:
            raise ValueError("model_fingerprint and embedding_dim are required")
        cache_ttl = float(observation_cache_ttl_s)
        cache_max = int(observation_cache_max_entries)
        visitor_gallery_max = int(visitor_gallery_max_exemplars)
        if cache_ttl <= 0.0 or cache_max < 1 or visitor_gallery_max < 1:
            raise ValueError("observation cache TTL and maximum size must be positive")
        self.store = store
        self.model_fingerprint = fingerprint
        self.embedding_dim = dimension
        self.resolver = BatchIdentityResolver(policy=policy)
        self._clock = clock
        self._lock = threading.RLock()
        self._hot_subjects: Tuple[_HotSubject, ...] = ()
        # ReID embeddings have already crossed the explicit metadata boundary
        # before they reach this runtime.  Keep one immutable, contiguous
        # gallery matrix so a frame is scored with a single native BLAS call
        # instead of millions of Python float multiplications on the media
        # callback.  The spans remain aligned with ``_hot_subjects``.
        self._hot_reference_matrix = np.empty(
            (0, self.embedding_dim), dtype=np.float64
        )
        self._hot_reference_matrix.setflags(write=False)
        self._hot_reference_spans: Tuple[Tuple[int, int], ...] = ()
        self._subject_index: dict[str, SubjectDescriptor] = {}
        self._refreshed_at = 0.0
        self._observation_cache_ttl_s = cache_ttl
        self._observation_cache_max_entries = cache_max
        calibrated_exemplar_maximum = (
            self.resolver.scorer.policy.maximum_exemplars_per_candidate
        )
        self._visitor_gallery_max_exemplars = (
            visitor_gallery_max
            if calibrated_exemplar_maximum is None
            else min(visitor_gallery_max, calibrated_exemplar_maximum)
        )
        self._observation_cache: dict[EnrollmentObservationKey, _CachedObservation] = {}
        self._cache_expired_evictions = 0
        self._cache_capacity_evictions = 0
        self.refresh()

    @property
    def refreshed_at(self) -> float:
        with self._lock:
            return self._refreshed_at

    @property
    def scoring_calibration_status(self) -> str:
        return self.resolver.scorer.policy.calibration_status

    @property
    def scoring_calibration_artifact_id(self) -> Optional[str]:
        return self.resolver.scorer.policy.calibration_artifact_id

    def current_time(self) -> float:
        return float(self._clock())

    def _gallery_inventory(
        self,
        *,
        now: float,
    ) -> tuple[dict[str, int], dict[str, int]]:
        residents: dict[str, int] = {}
        for resident in self.store.list_residents():
            gallery = self.store.load_resident_gallery(
                resident.resident_uuid,
                model_fingerprint=self.model_fingerprint,
                embedding_dim=self.embedding_dim,
            )
            if gallery:
                residents[resident.resident_uuid] = len(gallery)
        visitors = {
            session.session_uuid: len(exemplars)
            for session, exemplars in self.store.load_active_visitor_galleries(now=now)
            if exemplars
        }
        return residents, visitors

    def _hot_gallery_inventory_locked(
        self,
        *,
        now: float,
    ) -> tuple[dict[str, int], dict[str, int]]:
        """Return gallery counts from the immutable in-memory snapshot.

        Runtime mutations hold ``_lock`` and refresh the snapshot after the
        durable store commit.  Authority checks on that path therefore do not
        need to reopen and decode every gallery from SQLite.
        """

        residents: dict[str, int] = {}
        visitors: dict[str, int] = {}
        for subject in self._hot_subjects:
            descriptor = subject.descriptor
            if descriptor.expires_at is not None and descriptor.expires_at <= now:
                continue
            exemplar_count = len(subject.normalized_vectors)
            if exemplar_count == 0:
                continue
            if descriptor.identity_kind is IdentityKind.RESIDENT:
                if descriptor.resident_uuid is not None:
                    residents[descriptor.resident_uuid] = exemplar_count
            elif descriptor.visitor_session_uuid is not None:
                visitors[descriptor.visitor_session_uuid] = exemplar_count
        return residents, visitors

    def _assert_prospective_gallery_authority_locked(
        self,
        *,
        now: float,
        add_resident_candidates: int = 0,
        add_visitor_candidates: int = 0,
        prospective_exemplar_count: int = 0,
        inventory: Optional[tuple[dict[str, int], dict[str, int]]] = None,
    ) -> None:
        policy = self.resolver.scorer.policy
        if all(
            value is None
            for value in (
                policy.maximum_resident_candidates,
                policy.maximum_visitor_candidates,
                policy.maximum_total_candidates,
                policy.maximum_exemplars_per_candidate,
            )
        ):
            return
        if inventory is None:
            residents, visitors = self._gallery_inventory(now=now)
        else:
            residents, visitors = inventory
        resident_count = len(residents) + int(add_resident_candidates)
        visitor_count = len(visitors) + int(add_visitor_candidates)
        maximum_exemplars = max(
            (*residents.values(), *visitors.values(), int(prospective_exemplar_count)),
            default=0,
        )
        limits = (
            (
                "resident candidates",
                resident_count,
                policy.maximum_resident_candidates,
            ),
            (
                "visitor candidates",
                visitor_count,
                policy.maximum_visitor_candidates,
            ),
            (
                "total candidates",
                resident_count + visitor_count,
                policy.maximum_total_candidates,
            ),
            (
                "exemplars per candidate",
                maximum_exemplars,
                policy.maximum_exemplars_per_candidate,
            ),
        )
        for label, actual, maximum in limits:
            if maximum is not None and actual > maximum:
                raise RuntimeError(
                    "identity calibration gallery authority exceeded before mutation: "
                    f"{label} {actual} > {maximum}"
                )

    def ingest_observations(
        self,
        observations: Sequence[RuntimeObservation],
        *,
        model_fingerprint: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        ttl_s: Optional[float] = None,
        now: Optional[float] = None,
    ) -> ObservationCacheHealth:
        """Cache immutable server-produced evidence for later enrollment intent."""

        timestamp = float(self._clock() if now is None else now)
        fingerprint = (
            self.model_fingerprint
            if model_fingerprint is None
            else str(model_fingerprint or "").strip()
        )
        dimension = self.embedding_dim if embedding_dim is None else int(embedding_dim)
        self._assert_profile(fingerprint, dimension, subject="observation ingestion")
        ttl = self._observation_cache_ttl_s if ttl_s is None else float(ttl_s)
        if ttl <= 0.0 or ttl > self._observation_cache_ttl_s:
            raise ValueError(
                "observation cache entry TTL must be positive and no greater than runtime TTL"
            )
        rows = tuple(observations)
        keys = [row.key for row in rows]
        if len(keys) != len(set(keys)):
            raise ObservationEvidenceConflict(
                "observation ingestion batch contains duplicate exact keys"
            )
        for row in rows:
            if not isinstance(row, RuntimeObservation):
                raise TypeError(
                    "observation cache accepts RuntimeObservation values only"
                )
            self._normalize_vector(
                row.embedding,
                subject=f"observation {row.key.observation_id}",
            )
        with self._lock:
            self._evict_expired_observations_locked(timestamp)
            for row in rows:
                existing = self._observation_cache.get(row.key)
                if existing is not None:
                    if (
                        existing.observation != row
                        or existing.model_fingerprint != fingerprint
                        or existing.embedding_dim != dimension
                    ):
                        raise ObservationEvidenceConflict(
                            "exact observation key was replayed with different evidence"
                        )
                    continue
                self._observation_cache[row.key] = _CachedObservation(
                    observation=row,
                    model_fingerprint=fingerprint,
                    embedding_dim=dimension,
                    cached_at=timestamp,
                    expires_at=timestamp + ttl,
                )
            self._enforce_observation_capacity_locked()
            return self._observation_cache_health_locked()

    def observation_cache_health(
        self,
        *,
        now: Optional[float] = None,
    ) -> ObservationCacheHealth:
        timestamp = float(self._clock() if now is None else now)
        with self._lock:
            self._evict_expired_observations_locked(timestamp)
            return self._observation_cache_health_locked()

    def _evict_expired_observations_locked(self, now: float) -> None:
        expired = sorted(
            (
                key
                for key, row in self._observation_cache.items()
                if row.expires_at <= now
            ),
            key=lambda key: (
                key.tracklet_id,
                key.frame_id,
                key.observation_id,
            ),
        )
        for key in expired:
            self._observation_cache.pop(key, None)
        self._cache_expired_evictions += len(expired)

    def _enforce_observation_capacity_locked(self) -> None:
        excess = len(self._observation_cache) - self._observation_cache_max_entries
        if excess <= 0:
            return
        ordered = sorted(
            self._observation_cache,
            key=lambda key: (
                self._observation_cache[key].cached_at,
                self._observation_cache[key].expires_at,
                key.tracklet_id,
                key.frame_id,
                key.observation_id,
            ),
        )
        for key in ordered[:excess]:
            self._observation_cache.pop(key, None)
        self._cache_capacity_evictions += excess

    def _observation_cache_health_locked(self) -> ObservationCacheHealth:
        return ObservationCacheHealth(
            entry_count=len(self._observation_cache),
            max_entries=self._observation_cache_max_entries,
            ttl_s=self._observation_cache_ttl_s,
            expired_evictions=self._cache_expired_evictions,
            capacity_evictions=self._cache_capacity_evictions,
        )

    def _assert_hot_gallery_authority_locked(
        self,
        hot: Sequence[_HotSubject],
    ) -> None:
        policy = self.resolver.scorer.policy
        resident_candidates = sum(
            item.descriptor.identity_kind is IdentityKind.RESIDENT for item in hot
        )
        visitor_candidates = sum(
            item.descriptor.identity_kind is IdentityKind.VISITOR for item in hot
        )
        total_candidates = len(hot)
        maximum_exemplars = max(
            (len(item.normalized_vectors) for item in hot), default=0
        )
        authority_limits = (
            (
                "resident candidates",
                resident_candidates,
                policy.maximum_resident_candidates,
            ),
            (
                "visitor candidates",
                visitor_candidates,
                policy.maximum_visitor_candidates,
            ),
            ("total candidates", total_candidates, policy.maximum_total_candidates),
            (
                "exemplars per candidate",
                maximum_exemplars,
                policy.maximum_exemplars_per_candidate,
            ),
        )
        for label, actual, maximum in authority_limits:
            if maximum is not None and actual > maximum:
                raise RuntimeError(
                    "identity calibration gallery authority exceeded: "
                    f"{label} {actual} > {maximum}"
                )

    def refresh(self, *, now: Optional[float] = None) -> Tuple[SubjectDescriptor, ...]:
        timestamp = float(self._clock() if now is None else now)
        with self._lock:
            residents = self.store.list_residents()
            hot: list[_HotSubject] = []
            for resident in residents:
                self._assert_profile(
                    resident.model_fingerprint,
                    resident.embedding_dim,
                    subject=f"resident {resident.resident_uuid}",
                )
                gallery = self.store.load_resident_gallery(
                    resident.resident_uuid,
                    model_fingerprint=self.model_fingerprint,
                    embedding_dim=self.embedding_dim,
                )
                if gallery:
                    hot.append(
                        _HotSubject(
                            descriptor=self._resident_descriptor(resident),
                            normalized_vectors=tuple(
                                self._normalize_vector(
                                    row.vector, subject=row.exemplar_uuid
                                )
                                for row in gallery
                            ),
                        )
                    )

            for session, exemplars in self.store.load_active_visitor_galleries(
                now=timestamp
            ):
                self._assert_profile(
                    session.model_fingerprint,
                    session.embedding_dim,
                    subject=f"visitor session {session.session_uuid}",
                )
                for exemplar in exemplars:
                    self._assert_profile(
                        exemplar.model_fingerprint,
                        exemplar.embedding_dim,
                        subject=f"visitor exemplar {exemplar.exemplar_uuid}",
                    )
                if exemplars:
                    hot.append(
                        _HotSubject(
                            descriptor=self._visitor_descriptor(session),
                            normalized_vectors=tuple(
                                self._normalize_vector(
                                    row.vector, subject=row.exemplar_uuid
                                )
                                for row in exemplars
                            ),
                        )
                    )
            self._assert_hot_gallery_authority_locked(hot)
            self._install_hot_subjects_locked(hot)
            self._refreshed_at = timestamp
            return tuple(item.descriptor for item in self._hot_subjects)

    def _install_hot_subjects_locked(
        self,
        subjects: Sequence[_HotSubject],
    ) -> None:
        """Atomically install tuple and vectorized views of one gallery."""

        ordered = tuple(sorted(subjects, key=lambda item: item.descriptor.subject_id))
        spans: list[Tuple[int, int]] = []
        matrices: list[np.ndarray] = []
        offset = 0
        for subject in ordered:
            matrix = np.asarray(subject.normalized_vectors, dtype=np.float64)
            if matrix.ndim != 2 or matrix.shape[1:] != (self.embedding_dim,):
                raise ModelProfileMismatch(
                    "hot identity gallery contains an invalid normalized matrix"
                )
            end = offset + int(matrix.shape[0])
            spans.append((offset, end))
            matrices.append(matrix)
            offset = end
        if matrices:
            references = np.ascontiguousarray(
                np.concatenate(matrices, axis=0), dtype=np.float64
            )
        else:
            references = np.empty((0, self.embedding_dim), dtype=np.float64)
        references.setflags(write=False)
        self._hot_subjects = ordered
        self._hot_reference_matrix = references
        self._hot_reference_spans = tuple(spans)
        self._subject_index = {
            item.descriptor.subject_id: item.descriptor for item in ordered
        }

    def _refresh_visitor_subject_locked(
        self,
        session_uuid: str,
        *,
        now: float,
    ) -> None:
        """Replace one visitor's hot gallery after a durable mutation.

        ``add_visitor_exemplar`` bounds and orders the persisted gallery.  A
        single-session read gives the hot snapshot the exact post-prune state
        without reloading residents or unrelated visitor sessions.
        """

        session, exemplars = self.store.load_visitor_gallery(session_uuid)
        self._assert_profile(
            session.model_fingerprint,
            session.embedding_dim,
            subject=f"visitor session {session.session_uuid}",
        )
        for exemplar in exemplars:
            self._assert_profile(
                exemplar.model_fingerprint,
                exemplar.embedding_dim,
                subject=f"visitor exemplar {exemplar.exemplar_uuid}",
            )

        replacement: Optional[_HotSubject] = None
        if (
            session.state == "active"
            and session.expires_at > now
            and exemplars
        ):
            replacement = _HotSubject(
                descriptor=self._visitor_descriptor(session),
                normalized_vectors=tuple(
                    self._normalize_vector(row.vector, subject=row.exemplar_uuid)
                    for row in exemplars
                ),
            )

        updated = [
            item
            for item in self._hot_subjects
            if item.descriptor.visitor_session_uuid != session.session_uuid
            and (
                item.descriptor.expires_at is None
                or item.descriptor.expires_at > now
            )
        ]
        if replacement is not None:
            updated.append(replacement)
        self._assert_hot_gallery_authority_locked(updated)
        self._install_hot_subjects_locked(updated)
        self._refreshed_at = now

    def hot_subjects(
        self, *, now: Optional[float] = None
    ) -> Tuple[SubjectDescriptor, ...]:
        timestamp = float(self._clock() if now is None else now)
        with self._lock:
            return tuple(
                item.descriptor
                for item in self._hot_subjects
                if item.descriptor.expires_at is None
                or item.descriptor.expires_at > timestamp
            )

    def _active_hot_gallery_snapshot_locked(
        self,
        timestamp: float,
    ) -> Tuple[
        Tuple[Tuple[int, _HotSubject], ...],
        np.ndarray,
        Tuple[Tuple[int, int], ...],
    ]:
        active = tuple(
            (index, item)
            for index, item in enumerate(self._hot_subjects)
            if item.descriptor.expires_at is None
            or item.descriptor.expires_at > timestamp
        )
        return active, self._hot_reference_matrix, self._hot_reference_spans

    def _batched_gallery_similarities(
        self,
        observations: Sequence[RuntimeObservation],
        active_subjects: Sequence[Tuple[int, _HotSubject]],
        references: np.ndarray,
        spans: Sequence[Tuple[int, int]],
    ) -> Tuple[Tuple[float, ...], ...]:
        """Return one max cosine score per observation and active subject."""

        rows = tuple(observations)
        normalized = tuple(
            self._normalize_vector(
                observation.embedding,
                subject=f"observation {observation.key.observation_id}",
            )
            for observation in rows
        )
        if not rows or not active_subjects:
            return tuple(() for _ in rows)
        query_matrix = np.ascontiguousarray(normalized, dtype=np.float64)
        if references.ndim != 2 or references.shape[1:] != (self.embedding_dim,):
            raise RuntimeError("hot identity gallery matrix is inconsistent")
        similarities = np.matmul(query_matrix, references.T)
        np.clip(similarities, -1.0, 1.0, out=similarities)
        return tuple(
            tuple(
                float(
                    np.max(
                        similarities[
                            row_index,
                            spans[subject_index][0] : spans[subject_index][1],
                        ]
                    )
                )
                for subject_index, _subject in active_subjects
            )
            for row_index in range(len(rows))
        )

    def active_visitor_sessions(
        self,
        *,
        now: Optional[float] = None,
    ) -> Tuple[VisitorSession, ...]:
        timestamp = float(self._clock() if now is None else now)
        return self.store.list_active_visitor_sessions(now=timestamp)

    def calibration_candidate_observations(
        self,
        observations: Sequence[RuntimeObservation],
        *,
        now: Optional[float] = None,
    ) -> Tuple[TrackletObservation, ...]:
        """Return a score-only gallery snapshot for offline shadow evidence.

        This intentionally returns candidate similarities and metadata only.
        Query embeddings and gallery vectors never leave the process through
        this interface.
        """

        timestamp = float(self._clock() if now is None else now)
        rows = tuple(observations)
        track_ids = [row.key.tracklet_id for row in rows]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError(
                "a calibration batch may contain only one observation per exact tracklet"
            )
        with self._lock:
            active_subjects, references, spans = (
                self._active_hot_gallery_snapshot_locked(timestamp)
            )
        similarities = self._batched_gallery_similarities(
            rows,
            active_subjects,
            references,
            spans,
        )
        out = []
        for observation_index, observation in enumerate(rows):
            candidates = []
            for subject_offset, (_subject_index, subject) in enumerate(
                active_subjects
            ):
                descriptor = subject.descriptor
                candidates.append(
                    CandidateEvidence(
                        identity_id=descriptor.subject_id,
                        identity_kind=descriptor.identity_kind,
                        raw_similarity=similarities[observation_index][subject_offset],
                        evidence=(
                            f"gallery_exemplars={len(subject.normalized_vectors)}",
                            f"compatibility_sid={descriptor.compatibility_sid}",
                        ),
                    )
                )
            out.append(
                TrackletObservation(
                    tracklet_id=observation.key.tracklet_id,
                    quality=observation.quality,
                    candidates=tuple(candidates),
                    evidence=observation.evidence,
                )
            )
        return tuple(out)

    def resolve_batch(
        self,
        observations: Sequence[RuntimeObservation],
        *,
        constraints: Sequence[HardConstraint] = (),
        overlap_permits: Sequence[OverlapSharePermit] = (),
        now: Optional[float] = None,
    ) -> Tuple[RuntimeIdentityDecision, ...]:
        timestamp = float(self._clock() if now is None else now)
        observation_rows = tuple(observations)
        track_ids = [row.key.tracklet_id for row in observation_rows]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError(
                "a batch may contain only one observation per exact tracklet"
            )
        with self._lock:
            active_subjects, references, spans = (
                self._active_hot_gallery_snapshot_locked(timestamp)
            )
            descriptors = {
                item.descriptor.subject_id: item.descriptor
                for _index, item in active_subjects
            }

        similarities = self._batched_gallery_similarities(
            observation_rows,
            active_subjects,
            references,
            spans,
        )

        resolver_observations = []
        key_by_tracklet = {}
        for observation_index, observation in enumerate(observation_rows):
            candidates = []
            for subject_offset, (_subject_index, subject) in enumerate(
                active_subjects
            ):
                similarity = similarities[observation_index][subject_offset]
                descriptor = subject.descriptor
                candidates.append(
                    CandidateEvidence(
                        identity_id=descriptor.subject_id,
                        identity_kind=descriptor.identity_kind,
                        raw_similarity=similarity,
                        evidence=(
                            f"gallery_exemplars={len(subject.normalized_vectors)}",
                            f"compatibility_sid={descriptor.compatibility_sid}",
                        ),
                    )
                )
            tracklet_id = observation.key.tracklet_id
            key_by_tracklet[tracklet_id] = observation.key
            resolver_observations.append(
                TrackletObservation(
                    tracklet_id=tracklet_id,
                    quality=observation.quality,
                    candidates=tuple(candidates),
                    evidence=observation.evidence
                    + (
                        f"run_id={observation.key.run_id}",
                        f"camera_id={observation.key.camera_id}",
                        f"tracker_id={observation.key.tracker_id}",
                        f"frame_id={observation.key.frame_id}",
                        f"observation_id={observation.key.observation_id}",
                    ),
                )
            )
        constrained_observations = self.resolver.apply_constraints(
            tuple(resolver_observations),
            constraints,
        )
        candidate_by_tracklet = {
            row.tracklet_id: row for row in constrained_observations
        }
        decisions = self.resolver.resolve(
            constrained_observations,
            overlap_permits=overlap_permits,
        )
        return tuple(
            RuntimeIdentityDecision(
                observation_key=key_by_tracklet[decision.tracklet_id],
                candidate_observation=candidate_by_tracklet[decision.tracklet_id],
                decision=decision,
                subject=(
                    descriptors.get(decision.identity_id)
                    if decision.identity_id is not None
                    else None
                ),
            )
            for decision in decisions
        )

    def open_visitor_session(
        self,
        *,
        slot: int,
        ttl_s: float,
        now: Optional[float] = None,
    ) -> VisitorSession:
        timestamp = float(self._clock() if now is None else now)
        with self._lock:
            session = self.store.open_visitor_session(
                slot=slot,
                model_fingerprint=self.model_fingerprint,
                embedding_dim=self.embedding_dim,
                ttl_s=ttl_s,
                now=timestamp,
            )
            self.refresh(now=timestamp)
            return session

    def touch_visitor_session(
        self,
        session_uuid: str,
        *,
        now: Optional[float] = None,
    ) -> VisitorSession:
        timestamp = float(self._clock() if now is None else now)
        with self._lock:
            session = self.store.touch_visitor_session(session_uuid, now=timestamp)
            self.refresh(now=timestamp)
            return session

    def record_visitor_observation(
        self,
        session_uuid: str,
        observation: RuntimeObservation,
        *,
        now: Optional[float] = None,
    ) -> VisitorExemplarRecord:
        timestamp = float(self._clock() if now is None else now)
        vector = self._validated_raw_vector(observation.embedding)
        observation_id = json.dumps(
            [
                observation.key.run_id,
                observation.key.camera_id,
                observation.key.tracker_id,
                observation.key.frame_id,
                observation.key.observation_id,
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        with self._lock:
            session = self.store.get_visitor_session(session_uuid)
            inventory = self._hot_gallery_inventory_locked(now=timestamp)
            _, visitors = inventory
            current_exemplars = visitors.get(session.session_uuid, 0)
            self._assert_prospective_gallery_authority_locked(
                now=timestamp,
                add_visitor_candidates=1 if current_exemplars == 0 else 0,
                prospective_exemplar_count=min(
                    current_exemplars + 1,
                    self._visitor_gallery_max_exemplars,
                ),
                inventory=inventory,
            )
            exemplar = self.store.add_visitor_exemplar(
                session_uuid=session_uuid,
                vector=vector,
                model_fingerprint=self.model_fingerprint,
                embedding_dim=self.embedding_dim,
                observation_id=observation_id,
                max_exemplars=self._visitor_gallery_max_exemplars,
                now=timestamp,
            )
            self._refresh_visitor_subject_locked(session.session_uuid, now=timestamp)
            return exemplar

    def release_visitor_session(
        self,
        session_uuid: str,
        *,
        now: Optional[float] = None,
    ) -> None:
        timestamp = float(self._clock() if now is None else now)
        with self._lock:
            self.store.release_visitor_session(session_uuid, now=timestamp)
            self.refresh(now=timestamp)

    def propose_enrollment_from_cache(
        self,
        key: EnrollmentObservationKey,
        *,
        display_name: str,
        compatibility_sid: Optional[int] = None,
        update_resident_uuid: Optional[str] = None,
        ttl_s: float = 300.0,
        now: Optional[float] = None,
    ) -> EnrollmentProposalResult:
        if not isinstance(key, EnrollmentObservationKey):
            raise TypeError("key must be an EnrollmentObservationKey")
        timestamp = float(self._clock() if now is None else now)
        with self._lock:
            prior = self.store.find_enrollment_proposal_by_key(key)
            if prior is not None:
                raise ObservationEvidenceConsumed(prior.proposal_uuid)
            cached = self._observation_cache.get(key)
            if cached is None:
                raise ObservationEvidenceUnavailable(
                    "no server-produced observation evidence is cached for this exact key"
                )
            if cached.expires_at <= timestamp:
                self._observation_cache.pop(key, None)
                self._cache_expired_evictions += 1
                raise ObservationEvidenceExpired(
                    "server-produced observation evidence has expired"
                )
            self._assert_profile(
                cached.model_fingerprint,
                cached.embedding_dim,
                subject="cached enrollment evidence",
            )
            observation = cached.observation
            vector = self._validated_raw_vector(observation.embedding)
            if observation.quality < self.resolver.scorer.policy.quality_floor:
                raise ValueError(
                    "enrollment observation quality is below the runtime quality floor"
                )
            result = self.store.create_enrollment_proposal(
                key=key,
                observation_quality=observation.quality,
                observation_evidence=observation.evidence,
                display_name=display_name,
                vector=vector,
                model_fingerprint=self.model_fingerprint,
                embedding_dim=self.embedding_dim,
                ttl_s=ttl_s,
                compatibility_sid=compatibility_sid,
                update_resident_uuid=update_resident_uuid,
                now=timestamp,
            )
            self._observation_cache.pop(key, None)
            if result.idempotent:
                raise ObservationEvidenceConsumed(result.proposal.proposal_uuid)
            return result

    def confirm_enrollment(
        self,
        proposal_uuid: str,
        *,
        expected_key: EnrollmentObservationKey,
        evidence_digest: str,
        now: Optional[float] = None,
    ) -> EnrollmentConfirmationResult:
        timestamp = float(self._clock() if now is None else now)
        with self._lock:
            proposal = self.store.get_enrollment_proposal(proposal_uuid)
            if proposal.state != "confirmed":
                residents, _ = self._gallery_inventory(now=timestamp)
                if proposal.action == "create_resident":
                    self._assert_prospective_gallery_authority_locked(
                        now=timestamp,
                        add_resident_candidates=1,
                        prospective_exemplar_count=1,
                    )
                elif proposal.action == "add_anchor":
                    resident_uuid = str(proposal.target_resident_uuid or "")
                    current_exemplars = residents.get(resident_uuid, 0)
                    self._assert_prospective_gallery_authority_locked(
                        now=timestamp,
                        add_resident_candidates=1 if current_exemplars == 0 else 0,
                        prospective_exemplar_count=current_exemplars + 1,
                    )
            result = self.store.confirm_enrollment_proposal(
                proposal_uuid,
                expected_key=expected_key,
                evidence_digest=evidence_digest,
                now=timestamp,
            )
            self.refresh(now=timestamp)
            return result

    def get_enrollment_proposal(self, proposal_uuid: str) -> EnrollmentProposalRecord:
        return self.store.get_enrollment_proposal(proposal_uuid)

    def list_enrollment_proposals(
        self,
        *,
        limit: int = 100,
    ) -> Tuple[EnrollmentProposalRecord, ...]:
        return self.store.list_enrollment_proposals(limit=limit)

    def list_residents(self) -> Tuple[ResidentRecord, ...]:
        return self.store.list_residents()

    def update_resident_display_name(
        self,
        resident_uuid: str,
        *,
        display_name: str,
        now: Optional[float] = None,
    ) -> ResidentRecord:
        timestamp = float(self._clock() if now is None else now)
        with self._lock:
            resident = self.store.update_resident_display_name(
                resident_uuid,
                display_name=display_name,
                now=timestamp,
            )
            self.refresh(now=timestamp)
            return resident

    def delete_resident(self, resident_uuid: str) -> DeletionResult:
        with self._lock:
            result = self.store.delete_resident(resident_uuid)
            self.refresh()
            return result

    def health(self, *, now: Optional[float] = None) -> IdentityHealth:
        timestamp = float(self._clock() if now is None else now)
        return self.store.health(now=timestamp)

    def run_retention(self, *, now: Optional[float] = None) -> PurgeResult:
        timestamp = float(self._clock() if now is None else now)
        with self._lock:
            self._evict_expired_observations_locked(timestamp)
            result = self.store.purge_expired(now=timestamp)
            # Expired visitor sessions cascade-delete their visitor galleries,
            # which changes the hot identity snapshot.  Provisional sessions,
            # pending proposals, and quarantine exemplars are not part of that
            # snapshot, so a no-op (or those unrelated retention changes) does
            # not justify reopening and decoding every gallery.
            if result.visitor_sessions:
                self.refresh(now=timestamp)
            return result

    def _validated_raw_vector(self, vector: Sequence[float]) -> Tuple[float, ...]:
        values = tuple(float(value) for value in vector)
        self._normalize_vector(values, subject="runtime input")
        return values

    def _normalize_vector(
        self,
        vector: Sequence[float],
        *,
        subject: str,
    ) -> Tuple[float, ...]:
        values = tuple(float(value) for value in vector)
        if len(values) != self.embedding_dim:
            raise ModelProfileMismatch(
                f"{subject} dimension {len(values)} does not match runtime {self.embedding_dim}"
            )
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{subject} embedding contains non-finite values")
        norm = math.sqrt(sum(value * value for value in values))
        if norm <= 1e-12:
            raise ValueError(f"{subject} embedding has zero norm")
        return tuple(value / norm for value in values)

    def _assert_profile(
        self, fingerprint: str, dimension: int, *, subject: str
    ) -> None:
        if str(fingerprint) != self.model_fingerprint:
            raise ModelProfileMismatch(
                f"{subject} fingerprint {fingerprint!r} does not match runtime {self.model_fingerprint!r}"
            )
        if int(dimension) != self.embedding_dim:
            raise ModelProfileMismatch(
                f"{subject} dimension {dimension} does not match runtime {self.embedding_dim}"
            )

    @staticmethod
    def _dot(left: Sequence[float], right: Sequence[float]) -> float:
        return max(-1.0, min(1.0, float(sum(a * b for a, b in zip(left, right)))))

    @staticmethod
    def _resident_descriptor(resident: ResidentRecord) -> SubjectDescriptor:
        return SubjectDescriptor(
            subject_id=resident_subject_id(resident.resident_uuid),
            identity_kind=IdentityKind.RESIDENT,
            compatibility_sid=resident.compatibility_sid,
            display_name=resident.display_name,
            resident_uuid=resident.resident_uuid,
            visitor_session_uuid=None,
            visitor_slot=None,
            visitor_generation=None,
            expires_at=None,
        )

    @staticmethod
    def _visitor_descriptor(session: VisitorSession) -> SubjectDescriptor:
        return SubjectDescriptor(
            subject_id=visitor_subject_id(session.session_uuid, session.generation),
            identity_kind=IdentityKind.VISITOR,
            compatibility_sid=session.slot,
            display_name=None,
            resident_uuid=None,
            visitor_session_uuid=session.session_uuid,
            visitor_slot=session.slot,
            visitor_generation=session.generation,
            expires_at=session.expires_at,
        )


__all__ = [
    "IdentityV2Runtime",
    "ObservationCacheHealth",
    "ObservationEvidenceConflict",
    "ObservationEvidenceConsumed",
    "ObservationEvidenceError",
    "ObservationEvidenceExpired",
    "ObservationEvidenceUnavailable",
    "RuntimeIdentityDecision",
    "RuntimeObservation",
    "SubjectDescriptor",
    "resident_subject_id",
    "visitor_subject_id",
]
