"""Observed assignment spans from persisted point reports, never physical dwell."""
from __future__ import annotations

from typing import Literal
from pydantic import Field, model_validator
from .base import ProducerRef, Sha256
from .identity import SubjectRef
from .spatial_relationships import RelationshipModel, NativeUs, NativeSequence
from .world import EntityLifecycle, PositionState, WorldSourceEvidence
from .world_history import WorldHistoryPersistence

MAX_TEMPORAL_ENTITIES = 12
MAX_TEMPORAL_BYTES = 128 * 1024
MAX_TEMPORAL_SNAPSHOTS = 32


class TemporalSnapshotEvidence(RelationshipModel):
    journal_sequence: NativeSequence
    recorded_at_us: NativeUs
    previous_sha256: Sha256
    record_sha256: Sha256
    snapshot_id: str = Field(min_length=1, max_length=200)
    snapshot_sequence: NativeSequence
    producer: ProducerRef
    published_at_us: NativeUs
    observed_start_us: NativeUs
    observed_end_us: NativeUs
    frame: Literal["backend_world_m"]
    units: Literal["meters"]
    entity_id: str = Field(min_length=1, max_length=200)
    subject: SubjectRef
    lifecycle: EntityLifecycle
    position_state: PositionState | None
    conflict: bool
    quarantined: bool
    room_id: str | None = Field(max_length=160)
    entity_observed_at_us: NativeUs
    entity_stale_after_us: NativeUs
    world_frame_revision: str | None = Field(max_length=200)
    world_transform_sha256: Sha256 | None
    calibration_revision: str | None = Field(max_length=200)
    sources: tuple[WorldSourceEvidence, ...] = Field(min_length=1, max_length=4096)


class ObservedAssignmentSpan(RelationshipModel):
    state: Literal["known", "unknown"]
    reason: str = Field(min_length=1, max_length=96)
    room_id: str | None = Field(max_length=160)
    first_observed_at_us: NativeUs | None
    last_observed_at_us: NativeUs | None
    span_us: NativeSequence | None
    observation_count: NativeSequence
    snapshot_report_count: NativeSequence
    boundary_reason: str = Field(min_length=1, max_length=96)
    evidence: tuple[TemporalSnapshotEvidence, ...] = Field(max_length=MAX_TEMPORAL_SNAPSHOTS)

    @model_validator(mode="after")
    def coherent(self):
        if self.snapshot_report_count != len(self.evidence):
            raise ValueError("snapshot evidence count mismatch")
        if self.state == "known":
            if (self.room_id is None or self.first_observed_at_us is None or self.last_observed_at_us is None
                    or self.observation_count < 2 or self.span_us != self.last_observed_at_us - self.first_observed_at_us
                    or self.span_us <= 0 or self.observation_count > self.snapshot_report_count):
                raise ValueError("known observed span requires distinct original observations")
        elif any(value is not None for value in (self.room_id, self.first_observed_at_us, self.last_observed_at_us, self.span_us)):
            raise ValueError("unknown span cannot expose a settled span")
        return self


class TemporalUnknownDwell(RelationshipModel):
    state: Literal["unknown"] = "unknown"
    reason: Literal["point_observations_do_not_prove_continuous_occupancy"] = "point_observations_do_not_prove_continuous_occupancy"
    duration_us: None = None


class TemporalEntityRelationships(RelationshipModel):
    entity_id: str = Field(min_length=1, max_length=200)
    display_label: str = Field(min_length=1, max_length=200)
    observed_assignment_span: ObservedAssignmentSpan
    dwell: TemporalUnknownDwell = Field(default_factory=TemporalUnknownDwell)


class TemporalRelationshipSource(RelationshipModel):
    scope: Literal["persisted_as_of"] = "persisted_as_of"
    producer: ProducerRef
    head_snapshot_id: str = Field(min_length=1, max_length=200)
    head_snapshot_sequence: NativeSequence
    head_published_at_us: NativeUs
    head_journal_sequence: NativeSequence
    head_record_sha256: Sha256
    current_snapshot_id_at_read: str = Field(min_length=1, max_length=200)
    head_matches_current_at_read: bool
    inspected_snapshot_count: int = Field(ge=1, le=MAX_TEMPORAL_SNAPSHOTS)
    retained_first_journal_sequence: NativeSequence
    retained_first_previous_sha256: Sha256
    earlier_snapshots_omitted: bool
    retention_pruned: bool


class TemporalRelationshipsRead(RelationshipModel):
    contract: Literal["noesis.world.temporal-relationships"] = "noesis.world.temporal-relationships"
    version: Literal[1] = 1
    status: Literal["available", "partial", "unknown"]
    reason: str = Field(min_length=1, max_length=96)
    generated_at_us: NativeUs | None = None
    valid_until_us: NativeUs | None = None
    source: TemporalRelationshipSource | None = None
    persistence: WorldHistoryPersistence | None = None
    entities: tuple[TemporalEntityRelationships, ...] = Field(default=(), max_length=MAX_TEMPORAL_ENTITIES)

    @model_validator(mode="after")
    def coherent(self):
        if self.status == "unknown":
            if self.entities or self.source is not None or self.valid_until_us is not None:
                raise ValueError("unknown temporal read cannot expose a bound result")
        elif self.source is None or self.persistence is None or self.generated_at_us is None or self.valid_until_us is None:
            raise ValueError("temporal read requires exact source and freshness evidence")
        elif not self.source.head_published_at_us <= self.generated_at_us < self.valid_until_us <= self.source.head_published_at_us + 2_000_000:
            raise ValueError("temporal read cannot renew original head freshness")
        return self
