"""Owner-declared names attached to one original image observation capture.

These declarations never establish physical identity, current presence, a room,
world coordinates or an automatically matched detector/tracker lifecycle.
"""
from __future__ import annotations

from typing import Annotated, Literal
from pydantic import ConfigDict, Field, StrictBool, StrictInt, field_validator, model_validator
from .base import ContractModel
from .object_observations import ObjectImageObservation
from .semantic_space import StableId

MAX_BYTES = 256 * 1024
MAX_ASSOCIATIONS = 128
REVIEW_US = 120_000_000
MAX_REVIEWS = 64
MAX_INSPECTIONS = 16
MAX_INSPECTION_ROWS = 50
MAX_OPERATIONS = 128
MAX_STORED_BYTES = 32 * 1024 * 1024
REPLAY_US = 24 * 60 * 60 * 1_000_000
Integer = Annotated[StrictInt, Field(ge=0, le=2**53 - 1)]
Timestamp = Annotated[StrictInt, Field(ge=1, le=2**53 - 1)]
Token = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{43}$")]


def clean_text(value):
    value.encode("utf-8")
    if not value or value != value.strip() or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError("text must be trimmed and contain no control characters")
    return value


def source_text(value):
    value.encode("utf-8")
    if not value or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError("source text contains invalid characters")
    return value


class AssociationModel(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=False, allow_inf_nan=False, validate_default=True)


class ObservationReference(AssociationModel):
    run_id: str = Field(min_length=1, max_length=160)
    camera_id: str = Field(min_length=1, max_length=160)
    source_id: Integer
    source_epoch: Integer
    frame_id: Integer
    observation_id: str = Field(min_length=1, max_length=1400)

    _text = field_validator("run_id", "camera_id", "observation_id")(source_text)


class ObservationCapture(AssociationModel):
    run_id: str = Field(min_length=1, max_length=160)
    camera_id: str = Field(min_length=1, max_length=160)
    source_id: Integer
    source_epoch: Integer
    frame_id: Integer
    observed_at_us: Timestamp
    captured_at_us: Timestamp
    capture_time_status: Literal["estimated"]
    media_pts_ns: Integer
    observation_clock: Literal["processing_unix_epoch_us"]
    media_pts_domain: Literal["stream_relative_ns"]
    coordinate_frame: Literal["pipeline_image_px"]
    image_width: Annotated[StrictInt, Field(ge=1, le=32768)]
    image_height: Annotated[StrictInt, Field(ge=1, le=32768)]
    source_image_width: Annotated[StrictInt, Field(ge=1, le=32768)]
    source_image_height: Annotated[StrictInt, Field(ge=1, le=32768)]
    valid_until_us: Timestamp
    omitted_count: Integer
    invalid_count: Integer
    observation: ObjectImageObservation

    _text = field_validator("run_id", "camera_id")(source_text)

    @model_validator(mode="after")
    def original_image(self):
        box = self.observation.bbox
        if self.valid_until_us != self.observed_at_us + 5_000_000 or box.left + box.width > self.image_width or box.top + box.height > self.image_height:
            raise ValueError("capture requires the original five-second image bounds")
        source_text(self.observation.label)
        source_text(self.observation.observation_id)
        return self


class ObjectAssociation(AssociationModel):
    id: StableId
    label: str = Field(min_length=1, max_length=120)
    declaration: Literal["owner_declared"]
    retired: StrictBool
    capture: ObservationCapture

    _label = field_validator("label")(clean_text)


class AssociationPrepare(AssociationModel):
    actor_id: str = Field(min_length=1, max_length=160)
    operation_id: StableId
    expected_revision: StableId | None
    association_id: StableId
    label: str = Field(min_length=1, max_length=120)
    retired: StrictBool
    observation: ObservationReference | None
    inspection_token: Token

    _text = field_validator("actor_id", "label")(clean_text)


class AssociationInspect(AssociationModel):
    actor_id: str = Field(min_length=1, max_length=160)
    operation_id: StableId

    _actor = field_validator("actor_id")(clean_text)


class AssociationInspection(AssociationModel):
    contract: Literal["noesis.object-associations.inspection"] = "noesis.object-associations.inspection"
    version: Literal[1] = 1
    inspection_token: Token
    expires_at_us: Timestamp
    remaining_ms: Annotated[StrictInt, Field(ge=0, le=120000)]
    run_id: str = Field(min_length=1, max_length=160)
    registry_revision: StableId | None
    captures: Annotated[tuple[ObservationCapture, ...], Field(max_length=MAX_INSPECTION_ROWS)]
    omitted_count: Integer

    _run = field_validator("run_id")(source_text)


class AssociationCommand(AssociationModel):
    actor_id: str = Field(min_length=1, max_length=160)
    operation_id: StableId
    expected_revision: StableId | None
    association: ObjectAssociation

    _actor = field_validator("actor_id")(clean_text)


class AssociationReview(AssociationModel):
    contract: Literal["noesis.object-associations.review"] = "noesis.object-associations.review"
    version: Literal[1] = 1
    review_token: Token
    expires_at_us: Timestamp
    remaining_ms: Annotated[StrictInt, Field(ge=0, le=120000)]
    command: AssociationCommand


class AssociationConfirm(AssociationModel):
    review_token: Token
    command: AssociationCommand


class AssociationRecord(AssociationModel):
    revision: StableId | None
    updated_at_us: Timestamp | None
    actor_id: str | None = Field(min_length=1, max_length=160)
    associations: Annotated[tuple[ObjectAssociation, ...], Field(max_length=MAX_ASSOCIATIONS)]

    @model_validator(mode="after")
    def coherent(self):
        if self.actor_id is not None:
            clean_text(self.actor_id)
        if len({row.id for row in self.associations}) != len(self.associations):
            raise ValueError("duplicate stable declared association ID")
        if self.revision is None:
            if self.updated_at_us is not None or self.actor_id is not None or self.associations:
                raise ValueError("unknown registry must be empty")
        elif self.updated_at_us is None or self.actor_id is None:
            raise ValueError("saved registry requires actor and time")
        return self


class AssociationSnapshot(AssociationModel):
    contract: Literal["noesis.object-associations"] = "noesis.object-associations"
    version: Literal[1] = 1
    status: Literal["unknown", "available"]
    record: AssociationRecord


class AssociationResult(AssociationModel):
    contract: Literal["noesis.object-associations.result"] = "noesis.object-associations.result"
    version: Literal[1] = 1
    operation_id: StableId
    association_id: StableId
    record: AssociationRecord
    replayed: StrictBool
