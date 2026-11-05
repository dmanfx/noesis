"""Explicit owner declarations in exact authored-artifact geometry coordinates.

Artifact units are deliberately not meters or a canonical world frame. These
records do not measure, move geometry, or establish occupancy/room membership.
"""
from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import Field, StrictBool, StrictInt, field_validator, model_validator

from .base import Sha256
from .semantic_space import SpaceBinding, SpaceModel, SpaceWrite, StableId
from noesis_core.scene_files import MAX_AUTHORED_SCENE_BYTES, normalized_scene_relative_path

Number = Annotated[float, Field(allow_inf_nan=False, ge=-1e9, le=1e9)]


class AnchorArtifact(SpaceModel):
    role: str = Field(min_length=1, max_length=96)
    relative_path: str = Field(min_length=1, max_length=500)
    sha256: Sha256
    size_bytes: Annotated[StrictInt, Field(ge=1, le=MAX_AUTHORED_SCENE_BYTES)]

    @model_validator(mode="after")
    def artifact(self):
        if self.role != self.role.strip() or any(ord(c) < 32 or ord(c) == 127 for c in self.role):
            raise ValueError("artifact role must be trimmed text")
        normalized_scene_relative_path(self.relative_path, label="anchor artifact")
        return self


class AnchorGeometry(SpaceModel):
    space: Literal["authored_scene_geometry_local"]
    units: Literal["artifact_units"]
    point: tuple[Number, Number, Number]
    normal: tuple[Number, Number, Number] | None

    @field_validator("point", "normal", mode="before")
    @classmethod
    def numbers(cls, value):
        if value is not None and (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(type(item) not in (int, float) for item in value)):
            raise ValueError("coordinates must contain exactly three finite numbers")
        return value

    @model_validator(mode="after")
    def unit_normal(self):
        if self.normal is not None and abs(math.sqrt(sum(value * value for value in self.normal)) - 1) > 1e-6:
            raise ValueError("declared normal must be a unit vector")
        return self


class SemanticAnchor(SpaceModel):
    id: StableId
    semantic_id: StableId
    semantic_revision: StableId
    binding: SpaceBinding
    artifact: AnchorArtifact
    declaration: Literal["owner_declared"]
    retired: StrictBool
    geometry: AnchorGeometry


class _Anchors(SpaceModel):
    anchors: Annotated[tuple[SemanticAnchor, ...], Field(max_length=128)]

    @model_validator(mode="after")
    def unique(self):
        if len({anchor.id for anchor in self.anchors}) != len(self.anchors):
            raise ValueError("duplicate stable anchor ID")
        return self


class AnchorWrite(_Anchors):
    actor_id: str = Field(min_length=1, max_length=160)
    operation_id: StableId
    expected_revision: StableId | None
    expected_binding: SpaceBinding
    expected_artifact: AnchorArtifact
    expected_semantic_revision: StableId

    @field_validator("actor_id")
    @classmethod
    def actor(cls, value):
        return SpaceWrite.actor(value)


class AnchorRecord(_Anchors):
    revision: StableId | None
    updated_at_us: Annotated[StrictInt, Field(ge=1)] | None
    actor_id: str | None = Field(min_length=1, max_length=160)

    @model_validator(mode="after")
    def coherent(self):
        if self.actor_id is not None:
            SpaceWrite.actor(self.actor_id)
        if self.revision is None:
            if self.updated_at_us is not None or self.actor_id is not None or self.anchors:
                raise ValueError("unknown anchor registry must be empty")
        elif self.updated_at_us is None or self.actor_id is None:
            raise ValueError("saved anchor registry requires actor and timestamp")
        return self


class AnchorReview(SpaceModel):
    id: StableId
    status: Literal["reviewed", "retired", "stale_scene", "stale_semantic", "semantic_missing", "semantic_retired", "semantic_unreviewed"]


class AnchorEligibility(SpaceModel):
    status: Literal["supported", "unsupported"]
    reason: Literal["authored_obj_contract", "unsupported_authored_scene_format"]


class AnchorSnapshot(SpaceModel):
    contract: Literal["noesis.semantic.anchors"] = "noesis.semantic.anchors"
    version: Literal[1] = 1
    binding: SpaceBinding
    artifact: AnchorArtifact
    eligibility: AnchorEligibility
    semantic_revision: StableId | None
    record: AnchorRecord
    status: Literal["unknown", "available"]
    review_states: Annotated[tuple[AnchorReview, ...], Field(max_length=128)]


class AnchorResult(SpaceModel):
    contract: Literal["noesis.semantic.anchors.result"] = "noesis.semantic.anchors.result"
    version: Literal[1] = 1
    operation_id: StableId
    binding: SpaceBinding
    artifact: AnchorArtifact
    semantic_revision: StableId
    record: AnchorRecord
    replayed: StrictBool
    review_states: Annotated[tuple[AnchorReview, ...], Field(max_length=128)]
