from __future__ import annotations

from enum import StrEnum
from dataclasses import dataclass
import hashlib
import json
import math
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
from .support_surface import SupportSurfaceReference


class EntityLifecycle(StrEnum):
    PRESENT = "present"
    HELD = "held"
    LOST = "lost"


class PositionState(StrEnum):
    MEASURED = "measured"
    PREDICTED = "predicted"
    HELD = "held"


class WorldAttributionClaim(ContractModel):
    """Lossless bounded scalar copy of one ORIGINAL household claim owner."""

    run_id: str = Field(min_length=1, max_length=160)
    physical_serial: int = Field(ge=1, lt=2**64)
    camera_id: str = Field(min_length=1, max_length=160)
    source_id: int = Field(ge=0)
    source_epoch: int = Field(ge=0)
    raw_id: int = Field(ge=0)
    lifecycle_generation: int = Field(ge=0)
    lifecycle_started_frame: int = Field(ge=0)
    frame_id: int = Field(ge=0)
    observation_id: str = Field(min_length=1, max_length=200)
    acquired_at_s: float
    clock_domain: str = Field(min_length=1, max_length=160)
    media_time_s: float | None
    clock_basis: str = Field(min_length=1, max_length=160)
    household_state: Literal["unknown", "provisional", "resident", "visitor", "handoff"]
    household_identity_key: str | None = Field(max_length=160)
    household_stable_id: int | None = Field(ge=1)
    household_visitor_generation: int | None = Field(ge=0)
    household_constraint: Literal["eligible", "unresolved", "rejected"]

    @model_validator(mode="after")
    def _original_is_finite(self) -> "WorldAttributionClaim":
        if (not math.isfinite(self.acquired_at_s)
                or self.media_time_s is not None and not math.isfinite(self.media_time_s)
                or self.lifecycle_started_frame > self.frame_id):
            raise ValueError("invalid original attribution stamp")
        if ((self.household_state in {"unknown", "provisional"} and any(v is not None for v in (
                self.household_identity_key, self.household_stable_id, self.household_visitor_generation)))
                or self.household_state in {"resident", "visitor", "handoff"} and self.household_identity_key is None
                or (self.household_state == "visitor") != (self.household_visitor_generation is not None)):
            raise ValueError("invalid original household descriptor")
        return self

    @property
    def entity_id(self) -> str:
        return f"physical:{self.run_id}:{self.physical_serial}"

    @property
    def subject_id(self) -> str | None:
        if self.household_state == "resident" and self.household_identity_key is not None:
            return f"resident:{self.household_identity_key}"
        if self.household_state == "visitor" and self.household_stable_id is not None:
            return f"visitor:{self.run_id}:{self.household_stable_id}:g{self.household_visitor_generation}"
        return None


def attribution_conflict_sets(owners: tuple[WorldAttributionClaim, ...]):
    """Cross-key collisions plus divergent eligible originals within one key.

    Historical versions of the same member do not manufacture an internal
    contradiction; only that member's latest owner decision participates.
    """
    subjects, stable, latest = {}, {}, {}
    for claim in owners:
        if claim.subject_id is not None:
            subjects.setdefault(claim.subject_id, set()).add(claim.entity_id)
        if claim.household_stable_id is not None:
            stable.setdefault(claim.household_stable_id, set()).add(claim.entity_id)
        member = (claim.entity_id, claim.camera_id, claim.source_id, claim.source_epoch,
                  claim.raw_id, claim.lifecycle_generation)
        if member not in latest or claim.frame_id >= latest[member].frame_id:
            latest[member] = claim
    contested_subjects = {k for k, v in subjects.items() if len(v) > 1}
    contested_stable = {k for k, v in stable.items() if len(v) > 1}
    groups = {}
    for claim in latest.values():
        if (claim.household_state in {"resident", "visitor", "handoff"}
                and claim.household_constraint == "eligible"):
            groups.setdefault(claim.entity_id, []).append(claim)
    divergent = set()
    for entity, claims in groups.items():
        signatures = {(c.household_state, c.household_identity_key,
                       c.household_stable_id, c.household_visitor_generation) for c in claims}
        if len(signatures) > 1:
            divergent.add(entity)
            contested_subjects.update(c.subject_id for c in claims if c.subject_id is not None)
            contested_stable.update(c.household_stable_id for c in claims if c.household_stable_id is not None)
    return tuple(sorted(contested_subjects)), tuple(sorted(contested_stable)), frozenset(divergent)


class WorldAttributionQuarantine(ContractModel):
    status: Literal["quarantined"] = "quarantined"
    purpose: Literal["physical_only_quarantine"] = "physical_only_quarantine"
    reason: Literal["settled_household_claim_conflict"] = "settled_household_claim_conflict"
    conflict_id: Sha256
    run_id: str = Field(min_length=1, max_length=160)
    association_revision: int = Field(ge=0)
    subject_ids: tuple[str, ...] = Field(max_length=128)
    stable_ids: tuple[int, ...] = Field(max_length=128)
    owners: tuple[WorldAttributionClaim, ...] = Field(min_length=2, max_length=128)
    retained_owners: tuple[WorldAttributionClaim, ...] = Field(default=(), max_length=128)
    component_entity_ids: tuple[str, ...] = Field(min_length=1, max_length=64)

    @staticmethod
    def digest(payload: dict) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode()).hexdigest()

    @model_validator(mode="after")
    def _component_is_complete(self) -> "WorldAttributionQuarantine":
        if (any(c.run_id != self.run_id for c in (*self.owners, *self.retained_owners))
                or len(set(self.component_entity_ids)) != len(self.component_entity_ids)
                or any(not isinstance(v, str) or not 0 < len(v) <= 200 for v in self.subject_ids)
                or any(type(v) is not int or v < 1 for v in self.stable_ids)
                or not self.subject_ids and not self.stable_ids):
            raise ValueError("incomplete attribution conflict")
        claims = (*self.owners, *self.retained_owners)
        subjects, stable, divergent = attribution_conflict_sets(self.owners)
        if (self.subject_ids != subjects or self.stable_ids != stable
                or len(set(self.owners)) != len(self.owners)
                or len(set(self.retained_owners)) != len(self.retained_owners)):
            raise ValueError("incomplete original conflict claim set")
        known = {c.entity_id for c in claims}
        if not set(self.component_entity_ids) <= known:
            raise ValueError("missing component claim provenance")
        if len(self.component_entity_ids) == 1 and (
                self.component_entity_ids[0] not in divergent or known != set(self.component_entity_ids)):
            raise ValueError("singleton requires divergent original member claims")
        remaining = set(self.component_entity_ids)
        reached = {remaining.pop()}
        while remaining:
            subjects = {c.subject_id for c in claims if c.entity_id in reached and c.subject_id is not None}
            sids = {c.household_stable_id for c in claims if c.entity_id in reached and c.household_stable_id is not None}
            added = {c.entity_id for c in claims if c.entity_id in remaining
                     and (c.subject_id is not None and c.subject_id in subjects
                          or c.household_stable_id is not None and c.household_stable_id in sids)}
            if not added:
                raise ValueError("disconnected attribution component")
            reached.update(added)
            remaining.difference_update(added)
        if self.conflict_id != self.digest(self.model_dump(mode="json", exclude={"conflict_id"})):
            raise ValueError("attribution conflict digest mismatch")
        return self


def validate_attribution_quarantine(entity_id: str, subject: SubjectRef,
                                    record: WorldAttributionQuarantine | None) -> None:
    if record is None:
        if subject.subject_id.startswith("quarantine:"):
            raise ValueError("quarantine subject requires explicit attribution record")
        return
    prefix = f"physical:{record.run_id}:"
    serial = entity_id.removeprefix(prefix)
    if (not entity_id.startswith(prefix) or not serial.isdecimal() or str(int(serial)) != serial
            or not 0 < int(serial) < 2**64 or entity_id not in record.component_entity_ids
            or subject.subject_id != f"quarantine:{record.run_id}:{serial}"
            or subject.kind != "provisional" or subject.generation != 0
            or any(v is not None for v in (subject.stable_id, subject.resident_uuid, subject.display_name))):
        raise ValueError("quarantined attribution cannot expose a settled subject")


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
    support_surface: SupportSurfaceReference | None = None
    position_source: str | None = Field(default=None, min_length=1, max_length=120)
    position_quality: Literal["good", "estimated", "held"] | None = None
    position_state: PositionState | None = None
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
    attribution_quarantine: WorldAttributionQuarantine | None = None
    lifecycle: EntityLifecycle
    position: Vector3 | None = None
    covariance: Matrix3 | None = None
    position_state: PositionState | None = None
    continuity_segment_id: int | None = Field(default=None, ge=0)
    world_frame: Literal["backend_world_m"] = "backend_world_m"
    world_frame_revision: str | None = Field(default=None, min_length=1, max_length=200)
    world_transform_sha256: Sha256 | None = None
    calibration_revision: str | None = Field(default=None, min_length=1, max_length=200)
    position_quantity: Literal["ground_footprint"] = "ground_footprint"
    support_state: Literal["floor", "seat", "couch", "unknown"] = "unknown"
    support_surface: SupportSurfaceReference | None = None
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
        validate_attribution_quarantine(self.entity_id, self.subject, self.attribution_quarantine)
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


@dataclass(frozen=True, slots=True)
class WorldIdentityCollision:
    field: Literal["entity_id", "subject.subject_id", "subject.stable_id"]
    value: str | int
    entity_ids: tuple[str, ...]


class CanonicalWorldIdentityError(ValueError):
    """Deterministic bounded duplicate evidence, independent of live Identity."""

    def __init__(self, collisions: tuple[WorldIdentityCollision, ...]) -> None:
        self.reason = "canonical_identity_duplicate"
        self.collisions = collisions
        super().__init__(self.reason)


def validate_world_identity_uniqueness(entities: tuple[WorldEntity, ...] | list[WorldEntity]) -> None:
    # Same existing maximum as the direct canonical client; native fusion
    # remains at its unchanged smaller bound. No persistent household ledger.
    if len(entities) > 256:
        raise ValueError("canonical World identity validation capacity exceeded")
    claims: dict[str, dict[str | int, list[str]]] = {
        "entity_id": {}, "subject.subject_id": {}, "subject.stable_id": {},
    }
    for entity in entities:
        values = (entity.entity_id, entity.subject.subject_id, entity.subject.stable_id)
        for (field, owners), value in zip(claims.items(), values, strict=True):
            if value is not None:
                owners.setdefault(value, []).append(entity.entity_id)
    collisions = tuple(WorldIdentityCollision(field, value, tuple(sorted(owners)))
                       for field, values in claims.items()
                       for value, owners in sorted(values.items(), key=lambda item: str(item[0]))
                       if len(owners) > 1)
    if collisions:
        raise CanonicalWorldIdentityError(collisions)


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
        validate_world_identity_uniqueness(self.entities)
        by_id = {e.entity_id: e for e in self.entities}
        for entity in self.entities:
            record = entity.attribution_quarantine
            if record is not None and any(by_id[key].attribution_quarantine != record
                    for key in record.component_entity_ids if key in by_id):
                raise ValueError("partial attribution quarantine component")
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
    attribution_quarantine: WorldAttributionQuarantine | None = None
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
        validate_attribution_quarantine(self.entity_id, self.subject, self.attribution_quarantine)
        if self.world_transform_sha256 is not None and self.world_frame_revision is None:
            raise ValueError(
                "world event transform fingerprint requires a frame revision"
            )
        if self.observed_at_us > self.published_at_us:
            raise ValueError("world event observed_at_us cannot follow published_at_us")
        if self.event_type == "conflict_started" and not self.reason:
            raise ValueError("conflict_started requires a reason")
        return self
