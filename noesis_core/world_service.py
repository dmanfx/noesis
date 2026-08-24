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
from noesis_core.depth_contract import usable_registered_depth_m
from noesis_core.journal import AsyncContractJournal, ContractJournal
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
        if isinstance(journal, AsyncContractJournal):
            raise TypeError(
                "canonical world authority requires a completion-proven "
                "synchronous ContractJournal"
            )
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
                if len(appended) != expected_records:
                    raise RuntimeError(
                        "canonical world journal did not confirm the exact "
                        f"append count: expected={expected_records} "
                        f"actual={len(appended)}"
                    )
            except Exception:
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
            if callable(close):
                close()

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

        world_observation = self._world_observation(track)
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

    def _world_observation(self, track: Mapping[str, Any]) -> WorldPositionObservation | None:
        if track.get("world_valid") is not True:
            return None
        frame = str(track.get("world_frame") or "")
        if frame != "backend_world_m":
            return None
        raw = track.get("world")
        if not isinstance(raw, (list, tuple)) or len(raw) < 3:
            return None
        try:
            position = tuple(float(value) for value in raw[:3])
        except Exception:
            return None
        if not all(math.isfinite(value) for value in position):
            return None
        quality_raw = str(track.get("world_quality") or "estimated").strip().lower()
        quality = quality_raw if quality_raw in {"good", "estimated", "held"} else "estimated"
        variance = {"good": 0.04, "estimated": 0.25, "held": 0.64}[quality]
        confidence = {"good": 0.90, "estimated": 0.60, "held": 0.35}[quality]
        measured_confidence = self._confidence(track.get("tracker_confidence"))
        if measured_confidence is not None:
            confidence = min(confidence, measured_confidence)
        return WorldPositionObservation(
            position=Vector3(x=position[0], y=position[1], z=position[2]),
            covariance=Matrix3(
                values=(variance, 0.0, 0.0, 0.0, variance, 0.0, 0.0, 0.0, variance)
            ),
            frame="backend_world_m",
            units="meters",
            source=str(track.get("world_source") or "unspecified"),
            quality=quality,  # type: ignore[arg-type]
            confidence=confidence,
            reason=(
                str(track.get("world_quality_reason"))
                if track.get("world_quality_reason") not in {None, ""}
                else None
            ),
        )

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
        )

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
