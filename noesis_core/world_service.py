from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from threading import RLock
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
from noesis_core.contracts.world import EntityLifecycle, WorldEntity, WorldEvent, WorldSnapshot
from noesis_core.contracts.world_measurement import WorldMeasurementCandidateDiagnostic
from noesis_core.depth_contract import usable_registered_depth_m
from noesis_core.journal import (
    AsyncContractJournal,
    AsyncJournalAdmissionReceipt,
    ContractJournal,
)
from noesis_core.world import GlobalWorldFusion


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


@dataclass
class _PreparedWorldState:
    """Service-private mutable state that must never cross the boundary."""

    publication: WorldPublication
    fusion: GlobalWorldFusion
    sequence_by_source: dict[int, int]
    event_sequence: int
    previous_entities: dict[str, WorldEntity]
    journal_payloads: tuple[Mapping[str, Any], ...]
    recorded_at_us: int


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

    def __init__(
        self,
        *,
        producer: ProducerRef,
        artifacts: WorldArtifacts | Callable[[int, Mapping[str, Any]], WorldArtifacts],
        fusion: GlobalWorldFusion | None = None,
        clock_us: Callable[[], int] | None = None,
        journal: ContractJournal | AsyncContractJournal | None = None,
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
    ) -> PreparedWorldPublication:
        """Build an exact candidate without mutating fusion, counters, or journal."""

        with self._lock:
            return self._prepare_locked(
                source_id,
                tracks,
                metadata=metadata,
            )

    def commit(
        self,
        prepared: PreparedWorldPublication,
    ) -> WorldPublication:
        """Commit one still-current candidate after outbound admission."""

        with self._lock:
            return self._commit_locked(prepared)

    def discard(self, prepared: PreparedWorldPublication) -> None:
        """Forget an unadmitted candidate without changing authority state."""

        with self._lock:
            self._validate_prepared_owner(prepared)
            entry = self._prepared_candidates.get(prepared._candidate_id)
            if entry is None:
                return
            receipt, _state = entry
            if receipt is not prepared:
                raise RuntimeError("canonical world preparation identity mismatch")
            self._prepared_candidates.pop(prepared._candidate_id, None)

    def _prepare_locked(
        self,
        source_id: int,
        tracks: list[Mapping[str, Any]],
        *,
        metadata: Mapping[str, Any],
    ) -> PreparedWorldPublication:
        if self._closed:
            raise RuntimeError("canonical world service is closed")
        if len(self._prepared_candidates) >= int(
            self.MAX_PREPARED_CANDIDATES
        ):
            raise RuntimeError(
                "canonical world preparation capacity is exhausted"
            )
        source = int(source_id)
        artifacts = (
            self._artifacts(source, metadata)
            if callable(self._artifacts)
            else self._artifacts
        )
        now_us = max(1, int(self._clock_us()))
        working_fusion = self._fusion.fork()
        working_sequences = dict(self._sequence_by_source)
        observations: list[ObservationEnvelope] = []
        for track in tracks:
            observation = self._observation(
                source,
                track,
                metadata=metadata,
                artifacts=artifacts,
                published_at_us=now_us,
                sequence_by_source=working_sequences,
            )
            if observation is None:
                continue
            observations.append(observation)
            subject = self._subject(observation, track)
            working_fusion.ingest(observation, subject)
        if not tracks:
            camera_id = str(metadata.get("camera_id") or "").strip()
            if not camera_id:
                raise ValueError(
                    "empty source publication requires metadata.camera_id"
                )
            working_fusion.clear_source(
                camera_id,
                source,
                self.producer.run_id,
            )
        snapshot = working_fusion.snapshot(published_at_us=now_us)
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
        journal_payloads = (
            *(item.model_dump(mode="json") for item in observations),
            snapshot.model_dump(mode="json"),
            *(item.model_dump(mode="json") for item in events),
        )
        candidate_id = int(self._next_candidate_id)
        self._next_candidate_id += 1
        receipt = PreparedWorldPublication(
            base_revision=int(self._revision),
            # Public contracts and the containing dataclass are immutable;
            # mutable fusion/counter state remains service-private.
            publication=publication,
            _service_token=self._service_token,
            _candidate_id=candidate_id,
            _serialized_payloads=journal_payloads,
        )
        state = _PreparedWorldState(
            publication=publication,
            fusion=working_fusion,
            sequence_by_source=working_sequences,
            event_sequence=next_event_sequence,
            previous_entities=next_previous_entities,
            journal_payloads=journal_payloads,
            recorded_at_us=int(snapshot.published_at_us),
        )
        self._prepared_candidates[candidate_id] = (receipt, state)
        return receipt

    def _commit_locked(
        self,
        prepared: PreparedWorldPublication,
    ) -> WorldPublication:
        if self._closed:
            raise RuntimeError("canonical world service is closed")
        self._validate_prepared_owner(prepared)
        entry = self._prepared_candidates.get(prepared._candidate_id)
        if entry is None or entry[0] is not prepared:
            raise RuntimeError("canonical world preparation is unknown or consumed")
        _receipt, state = entry
        if int(prepared.base_revision) != int(self._revision):
            raise RuntimeError(
                "canonical world preparation is stale relative to service state"
            )
        if state.publication.snapshot.producer != self.producer:
            raise RuntimeError("canonical world preparation producer mismatch")
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
        self._current_snapshot = state.publication.snapshot
        self._revision += 1
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
            self._closed = True
            self._prepared_candidates.clear()
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

    def _observation(
        self,
        source_id: int,
        track: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any],
        artifacts: WorldArtifacts,
        published_at_us: int,
        sequence_by_source: dict[int, int],
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
        world_observation = self._world_observation(
            track,
            calibration_revision=calibration_revision,
        )
        world_diagnostics = self._world_diagnostics(
            track,
            world_observation=world_observation,
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

    def _world_observation(
        self,
        track: Mapping[str, Any],
        *,
        calibration_revision: Any = None,
    ) -> WorldPositionObservation | None:
        if track.get("world_valid") is not True:
            return None
        world_source = str(track.get("world_source") or "").strip().lower()
        if world_source in {
            "cv_prediction",
            "anchor_hold",
            "image_motion_prediction",
        }:
            # These values preserve display continuity in the tracking/BEV
            # lane.  They are not fresh metric evidence for canonical world
            # authority and must never be admitted as a new camera sample.
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
            posture=posture,  # type: ignore[arg-type]
            source=str(track.get("world_source") or "unspecified"),
            quality=quality,  # type: ignore[arg-type]
            confidence=confidence,
            reason=(
                str(track.get("world_quality_reason"))
                if track.get("world_quality_reason") not in {None, ""}
                else None
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
    ) -> str | None:
        if world_observation is not None:
            return None
        if track.get("world_valid") is True:
            if str(track.get("world_frame") or "") != "backend_world_m":
                return "world_frame_invalid"
            source = str(track.get("world_source") or "").strip().lower()
            if source in {
                "cv_prediction",
                "anchor_hold",
                "image_motion_prediction",
            }:
                return "display_continuity_not_authoritative"
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
