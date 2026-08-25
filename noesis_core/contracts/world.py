from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from noesis_core.analytics_zones import is_exact_zone_label

from .base import (
    ContractModel,
    Matrix3,
    ProducerRef,
    SequenceNumber,
    Sha256,
    TimestampUs,
    Vector3,
)
from .identity import SubjectRef
from .observation import ZoneSource


class EntityLifecycle(StrEnum):
    PRESENT = "present"
    HELD = "held"
    LOST = "lost"


class WorldSourceEvidence(ContractModel):
    observation_id: str = Field(min_length=1, max_length=200)
    camera_id: str = Field(min_length=1, max_length=160)
    zone: str | None = Field(default=None, min_length=1, max_length=160)
    zone_source: ZoneSource | None = None
    zone_authoritative: bool = False
    observed_at_us: TimestampUs
    position: Vector3
    covariance: Matrix3
    world_frame: Literal["backend_world_m"] = "backend_world_m"
    world_frame_revision: str | None = Field(default=None, min_length=1, max_length=200)
    world_transform_sha256: Sha256 | None = None
    calibration_revision: str | None = Field(default=None, min_length=1, max_length=200)
    accepted: bool
    rejection_reason: str | None = Field(default=None, max_length=200)

    @field_validator("zone", mode="before")
    @classmethod
    def _zone_is_not_normalized(cls, value: object) -> object:
        if value is not None and not is_exact_zone_label(value):
            raise ValueError(
                "zone labels must be nonempty, unpadded, and at most 160 characters"
            )
        return value

    @model_validator(mode="after")
    def _zone_provenance_is_coherent(self) -> "WorldSourceEvidence":
        if (self.world_frame_revision is None) != (
            self.world_transform_sha256 is None
        ):
            raise ValueError(
                "world frame revision and transform fingerprint must be provided together"
            )
        if self.zone is None:
            if self.zone_source is not None or self.zone_authoritative:
                raise ValueError("zone provenance requires a zone label")
        elif not is_exact_zone_label(self.zone):
            raise ValueError(
                "zone labels must be nonempty, unpadded, and at most 160 characters"
            )
        elif self.zone_source == "nvdsanalytics_roi":
            if not self.zone_authoritative:
                raise ValueError(
                    "nvdsanalytics ROI zones must be spatially authoritative"
                )
        elif self.zone_source == "camera_default":
            if self.zone_authoritative:
                raise ValueError(
                    "camera-default zones cannot be spatially authoritative"
                )
        elif self.zone_authoritative:
            raise ValueError(
                "authoritative zones require nvdsanalytics ROI provenance"
            )
        return self


class WorldEntity(ContractModel):
    entity_id: str = Field(min_length=1, max_length=200)
    subject: SubjectRef
    lifecycle: EntityLifecycle
    position: Vector3 | None = None
    covariance: Matrix3 | None = None
    world_frame: Literal["backend_world_m"] = "backend_world_m"
    world_frame_revision: str | None = Field(default=None, min_length=1, max_length=200)
    world_transform_sha256: Sha256 | None = None
    calibration_revision: str | None = Field(default=None, min_length=1, max_length=200)
    position_quantity: Literal["ground_footprint"] = "ground_footprint"
    support_state: Literal["floor", "seat", "couch", "unknown"] = "unknown"
    posture: Literal["standing", "sitting", "lying", "unknown"] = "unknown"
    velocity_mps: Vector3 | None = None
    room_id: str | None = Field(default=None, min_length=1, max_length=160)
    observed_at_us: TimestampUs
    stale_after_us: TimestampUs
    sources: tuple[WorldSourceEvidence, ...] = Field(min_length=1)
    conflict: bool = False
    conflict_reason: str | None = Field(default=None, max_length=240)

    @field_validator("room_id", mode="before")
    @classmethod
    def _room_id_is_not_normalized(cls, value: object) -> object:
        if value is not None and not is_exact_zone_label(value):
            raise ValueError(
                "room_id must be nonempty, unpadded, and at most 160 characters"
            )
        return value

    @model_validator(mode="after")
    def _state_is_coherent(self) -> "WorldEntity":
        if (self.position is None) != (self.covariance is None):
            raise ValueError("position and covariance must be present together")
        if self.lifecycle != EntityLifecycle.LOST and self.position is None:
            raise ValueError("present/held entities require a position")
        if self.stale_after_us < self.observed_at_us:
            raise ValueError("stale_after_us cannot precede observed_at_us")
        if self.conflict and not self.conflict_reason:
            raise ValueError("conflicting entities require conflict_reason")
        authoritative_rooms = {
            source.zone
            for source in self.sources
            if source.accepted
            and source.zone_source == "nvdsanalytics_roi"
            and source.zone_authoritative
            and source.zone is not None
        }
        expected_room = (
            next(iter(authoritative_rooms))
            if len(authoritative_rooms) == 1
            else None
        )
        if self.room_id != expected_room:
            raise ValueError(
                "room_id must equal the sole accepted authoritative source zone"
            )
        if len(authoritative_rooms) > 1 and not self.conflict:
            raise ValueError(
                "conflicting authoritative source zones require conflict=true"
            )
        if self.world_frame != "backend_world_m":
            raise ValueError("world entities must use backend_world_m")
        if self.world_transform_sha256 is not None and self.world_frame_revision is None:
            raise ValueError(
                "world entity transform fingerprint requires a frame revision"
            )
        accepted_sources = tuple(source for source in self.sources if source.accepted)
        # Every accepted source must reach the same target revision.  Source
        # transform digests and calibration revisions can legitimately differ
        # by camera and remain preserved on WorldSourceEvidence.  Entity-level
        # transform/calibration fields are populated only when all accepted
        # sources share that exact provenance value.
        if self.world_frame_revision is not None:
            for source in accepted_sources:
                if (
                    source.world_frame != self.world_frame
                    or source.world_frame_revision != self.world_frame_revision
                ):
                    raise ValueError(
                        "entity target-frame identity disagrees with an accepted source"
                    )
                if (
                    self.world_transform_sha256 is not None
                    and source.world_transform_sha256 != self.world_transform_sha256
                ):
                    raise ValueError(
                        "entity common transform fingerprint disagrees with an accepted source"
                    )
                if (
                    self.calibration_revision is not None
                    and source.calibration_revision != self.calibration_revision
                ):
                    raise ValueError(
                        "entity common calibration revision disagrees with an accepted source"
                    )
        return self


class WorldSnapshot(ContractModel):
    contract: Literal["noesis.world.snapshot"]
    contract_version: Literal[1]
    snapshot_id: str = Field(min_length=1, max_length=200)
    producer: ProducerRef
    sequence: SequenceNumber
    observed_start_us: TimestampUs
    observed_end_us: TimestampUs
    published_at_us: TimestampUs
    frame: Literal["backend_world_m"]
    units: Literal["meters"]
    entities: tuple[WorldEntity, ...]

    @model_validator(mode="after")
    def _window_is_valid(self) -> "WorldSnapshot":
        if not (self.observed_start_us <= self.observed_end_us <= self.published_at_us):
            raise ValueError("invalid world snapshot observation/publish window")
        entity_ids = [entity.entity_id for entity in self.entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("world snapshot entity_id values must be unique")
        return self


class WorldEvent(ContractModel):
    contract: Literal["noesis.world.event"]
    contract_version: Literal[1]
    event_id: str = Field(min_length=1, max_length=200)
    producer: ProducerRef
    sequence: SequenceNumber
    event_type: Literal[
        "appeared",
        "held",
        "resumed",
        "lost",
        "conflict_started",
        "conflict_cleared",
    ]
    entity_id: str = Field(min_length=1, max_length=200)
    subject: SubjectRef
    observed_at_us: TimestampUs
    published_at_us: TimestampUs
    frame: Literal["backend_world_m"]
    units: Literal["meters"]
    world_frame: Literal["backend_world_m"] = "backend_world_m"
    world_frame_revision: str | None = Field(default=None, min_length=1, max_length=200)
    world_transform_sha256: Sha256 | None = None
    calibration_revision: str | None = Field(default=None, min_length=1, max_length=200)
    position: Vector3 | None = None
    reason: str | None = Field(default=None, max_length=240)

    @model_validator(mode="after")
    def _time_is_coherent(self) -> "WorldEvent":
        if self.world_transform_sha256 is not None and self.world_frame_revision is None:
            raise ValueError(
                "world event transform fingerprint requires a frame revision"
            )
        if self.observed_at_us > self.published_at_us:
            raise ValueError("world event observed_at_us cannot follow published_at_us")
        if self.event_type == "conflict_started" and not self.reason:
            raise ValueError("conflict_started requires a reason")
        return self
