"""A bounded as-of projection, not new spatial or temporal authority."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import ConfigDict, Field, StrictInt, model_validator

from .base import ContractModel, ProducerRef, Sha256
from .support_surface import SupportSurfaceReference
from .world import EntityLifecycle, PositionState, WorldAttributionQuarantine, WorldSourceEvidence

MAX_ENTITIES = 256
MAX_RESPONSE_BYTES = 128 * 1024
MAX_SNAPSHOT_AGE_US = 2_000_000
NativeUs = Annotated[StrictInt, Field(ge=1, le=2**53 - 1)]
NativeSequence = Annotated[StrictInt, Field(ge=0, le=2**53 - 1)]
RelationshipReason = Literal[
    "recorded_authoritative_room_assignment", "recorded_support_surface", "attribution_quarantined",
    "entity_lost", "entity_held", "entity_conflict", "entity_not_measured", "entity_expired",
    "entity_observation_in_future", "binding_unavailable", "source_evidence_unavailable",
    "source_evidence_not_current", "room_unassigned", "room_source_unavailable",
    "support_unknown", "support_surface_unavailable", "support_source_unavailable",
]


class RelationshipModel(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=False, validate_default=True)


class AssignedToRoom(RelationshipModel):
    state: Literal["known", "unknown"]
    reason: RelationshipReason
    room_id: str | None = Field(max_length=160)

    @model_validator(mode="after")
    def coherent(self):
        if (self.state == "known") != (self.room_id is not None):
            raise ValueError("only a known assignment can expose a relation target")
        if (self.state == "known") != (self.reason == "recorded_authoritative_room_assignment"):
            raise ValueError("room assignment state and reason disagree")
        return self


class SupportedBy(RelationshipModel):
    state: Literal["known", "unknown"]
    reason: RelationshipReason
    support_state: Literal["floor", "seat", "couch"] | None
    support_surface: SupportSurfaceReference | None

    @model_validator(mode="after")
    def coherent(self):
        if (self.state == "known") != (self.reason == "recorded_support_surface"):
            raise ValueError("support state and reason disagree")
        if self.state == "known":
            if self.support_state is None or self.support_surface is None:
                raise ValueError("known support requires the original selected surface")
        elif self.support_state is not None or self.support_surface is not None:
            raise ValueError("unknown support cannot expose a settled target")
        return self


class UnknownDwell(RelationshipModel):
    state: Literal["unknown"] = "unknown"
    reason: Literal["single_snapshot_no_duration"] = "single_snapshot_no_duration"
    duration_us: None = None


class EntitySpatialRelationships(RelationshipModel):
    entity_id: str = Field(min_length=1, max_length=200)
    display_label: str = Field(min_length=1, max_length=200)
    lifecycle: EntityLifecycle
    position_state: PositionState | None
    conflict: bool
    conflict_reason: str | None = Field(max_length=240)
    attribution_quarantine: WorldAttributionQuarantine | None
    observed_at_us: NativeUs
    stale_after_us: NativeUs
    world_frame: Literal["backend_world_m"]
    world_frame_revision: str | None = Field(max_length=200)
    world_transform_sha256: Sha256 | None
    calibration_revision: str | None = Field(max_length=200)
    # These are original recorded fields, not independently settled claims.
    room_id: str | None = Field(max_length=160)
    support_state: Literal["floor", "seat", "couch", "unknown"]
    support_surface: SupportSurfaceReference | None
    sources: Annotated[tuple[WorldSourceEvidence, ...], Field(min_length=1, max_length=4096)]
    assigned_to_room: AssignedToRoom
    supported_by: SupportedBy
    dwell: UnknownDwell

    @model_validator(mode="after")
    def coherent(self):
        known = self.assigned_to_room.state == "known" or self.supported_by.state == "known"
        if known and (self.lifecycle != "present" or self.position_state != "measured"
                      or self.conflict or self.attribution_quarantine is not None):
            raise ValueError("unsettled original evidence cannot expose settled relations")
        if self.assigned_to_room.state == "known" and self.assigned_to_room.room_id != self.room_id:
            raise ValueError("relationship must retain the exact original room assignment")
        if self.supported_by.state == "known" and (self.supported_by.support_state != self.support_state
                                                   or self.supported_by.support_surface != self.support_surface):
            raise ValueError("relationship must retain exact original support evidence")
        return self


class SpatialRelationshipsRead(RelationshipModel):
    contract: Literal["noesis.world.relationships"] = "noesis.world.relationships"
    version: Literal[1] = 1
    status: Literal["available", "unknown"]
    reason: str = Field(min_length=1, max_length=80)
    generated_at_us: NativeUs | None
    valid_until_us: NativeUs | None
    max_snapshot_age_us: Literal[2_000_000] = MAX_SNAPSHOT_AGE_US
    producer: ProducerRef | None
    snapshot_id: str | None = Field(max_length=200)
    sequence: NativeSequence | None
    published_at_us: NativeUs | None
    observed_start_us: NativeUs | None
    observed_end_us: NativeUs | None
    frame: Literal["backend_world_m"] | None
    units: Literal["meters"] | None
    entities: Annotated[tuple[EntitySpatialRelationships, ...], Field(max_length=MAX_ENTITIES)]

    @model_validator(mode="after")
    def coherent(self):
        binding = (self.producer, self.snapshot_id, self.sequence, self.published_at_us,
                   self.observed_start_us, self.observed_end_us, self.frame, self.units, self.valid_until_us)
        if self.status == "unknown":
            if any(value is not None for value in binding) or self.entities:
                raise ValueError("unknown relationships require an empty unbound result")
        elif any(value is None for value in (*binding, self.generated_at_us)):
            raise ValueError("available relationships require exact snapshot provenance")
        elif not (self.observed_start_us <= self.observed_end_us <= self.published_at_us
                  <= self.generated_at_us < self.valid_until_us == self.published_at_us + MAX_SNAPSHOT_AGE_US):
            raise ValueError("relationship evidence cannot renew the original snapshot deadline")
        if self.status == "available" and any(
            (entity.assigned_to_room.state == "known" or entity.supported_by.state == "known")
            and not entity.observed_at_us <= self.generated_at_us < entity.stale_after_us
            for entity in self.entities
        ):
            raise ValueError("settled relationships must be within original entity expiry")
        return self
