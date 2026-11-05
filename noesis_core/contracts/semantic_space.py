"""Owner-taught semantic names, not geometry or canonical membership evidence."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import ConfigDict, Field, StrictBool, StrictInt, field_validator, model_validator

from .base import ContractModel, Sha256

StableId = Annotated[str, Field(pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")]


class SpaceModel(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=False, validate_default=True)


class SpaceBinding(SpaceModel):
    scope: Literal["scene_release"]
    release_id: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$")
    release_sha256: Sha256
    authored_scene_sha256: Sha256
    calibration_sha256: Sha256
    model_sha256: Sha256
    config_sha256: Sha256


class SpaceRelation(SpaceModel):
    kind: Literal["part_of", "adjacent_to", "supports", "covers"]
    target_id: StableId


class SpaceObject(SpaceModel):
    id: StableId
    kind: Literal["area", "seat", "surface", "door", "coverage"]
    label: str = Field(min_length=1, max_length=80)
    description: str = Field(max_length=500)
    retired: StrictBool
    reviewed_binding: SpaceBinding | None
    relations: Annotated[tuple[SpaceRelation, ...], Field(max_length=8)]

    @model_validator(mode="after")
    def clean_text(self):
        self.label.encode("utf-8")
        self.description.encode("utf-8")
        if not self.label.strip() or self.label != self.label.strip():
            raise ValueError("label must be nonblank and trimmed")
        if any(ord(c) < 32 or ord(c) == 127 for c in self.label):
            raise ValueError("label control characters are not allowed")
        if any((ord(c) < 32 and c not in "\n\t") or ord(c) == 127 for c in self.description):
            raise ValueError("control characters are not allowed")
        return self


class _SpaceGraph(SpaceModel):
    objects: Annotated[tuple[SpaceObject, ...], Field(max_length=128)]

    @model_validator(mode="after")
    def graph(self):
        by_id = {item.id: item for item in self.objects}
        if len(by_id) != len(self.objects):
            raise ValueError("duplicate semantic ID")
        for item in self.objects:
            keys = [(r.kind, r.target_id) for r in item.relations]
            if len(set(keys)) != len(keys):
                raise ValueError("duplicate relationship")
            for relation in item.relations:
                target = by_id.get(relation.target_id)
                if not target or target.retired or target.id == item.id or item.retired:
                    raise ValueError("relationships require distinct active semantic IDs")
        visiting, visited = set(), set()

        def visit(key):
            if key in visiting:
                raise ValueError("part_of relationships must be acyclic")
            if key in visited:
                return
            visiting.add(key)
            for relation in by_id[key].relations:
                if relation.kind == "part_of":
                    visit(relation.target_id)
            visiting.remove(key)
            visited.add(key)

        for key in by_id:
            visit(key)
        return self


class SpaceWrite(_SpaceGraph):
    actor_id: str = Field(min_length=1, max_length=160)
    operation_id: StableId
    expected_revision: StableId | None
    expected_binding: SpaceBinding

    @field_validator("actor_id")
    @classmethod
    def actor(cls, value):
        value.encode("utf-8")
        if not value.strip() or value != value.strip() or any(ord(c) < 32 or ord(c) == 127 for c in value):
            raise ValueError("actor_id must be nonblank, trimmed text")
        return value


class SpaceRecord(_SpaceGraph):
    revision: StableId | None
    updated_at_us: Annotated[StrictInt, Field(ge=1)] | None
    actor_id: str | None = Field(min_length=1, max_length=160)

    @model_validator(mode="after")
    def coherent(self):
        if self.actor_id is not None:
            SpaceWrite.actor(self.actor_id)
        if self.revision is None:
            if self.updated_at_us is not None or self.actor_id is not None or self.objects:
                raise ValueError("unknown registry must be empty")
        elif self.updated_at_us is None or self.actor_id is None:
            raise ValueError("saved registry requires actor and timestamp")
        return self


class SpaceReview(SpaceModel):
    id: StableId
    status: Literal["reviewed", "unreviewed", "stale", "retired"]


class SpaceSnapshot(SpaceModel):
    contract: Literal["noesis.semantic.space"] = "noesis.semantic.space"
    version: Literal[1] = 1
    binding: SpaceBinding
    record: SpaceRecord
    status: Literal["unknown", "available"]
    review_states: Annotated[tuple[SpaceReview, ...], Field(max_length=128)]


class SpaceResult(SpaceModel):
    contract: Literal["noesis.semantic.space.result"] = "noesis.semantic.space.result"
    version: Literal[1] = 1
    operation_id: StableId
    record: SpaceRecord
    binding: SpaceBinding
    replayed: StrictBool
    review_states: Annotated[tuple[SpaceReview, ...], Field(max_length=128)]
