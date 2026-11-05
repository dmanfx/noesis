"""Owner-declared artifact-local shapes from exact saved declaration points."""
from __future__ import annotations
from typing import Annotated, Literal
from pydantic import Field, StrictBool, StrictInt, field_validator, model_validator
from .semantic_space import SpaceModel, SpaceBinding, SpaceWrite, StableId
from .semantic_anchor import AnchorArtifact, Number, SemanticAnchor

ShapeKind = Literal['area', 'seat', 'surface', 'door', 'coverage']

class ShapeInput(SpaceModel):
    id: StableId
    semantic_id: StableId
    kind: ShapeKind
    retired: StrictBool
    anchor_ids: Annotated[tuple[StableId, ...], Field(min_length=2, max_length=16)]

    @model_validator(mode='after')
    def validate_vertices(self):
        if len(set(self.anchor_ids)) != len(self.anchor_ids):
            raise ValueError('shape requires distinct ordered saved anchor IDs')
        if (self.kind == 'door' and len(self.anchor_ids) != 2) or (self.kind != 'door' and len(self.anchor_ids) < 3):
            raise ValueError('door requires two points; polygon requires three to sixteen points')
        return self

class ShapeVertex(SpaceModel):
    anchor_id: StableId
    point: tuple[Number, Number, Number]

class SemanticShape(ShapeInput):
    declaration: Literal['owner_declared']
    space: Literal['authored_scene_geometry_local']
    units: Literal['artifact_units']
    binding: SpaceBinding
    artifact: AnchorArtifact
    semantic_revision: StableId
    anchor_revision: StableId
    vertices: Annotated[tuple[ShapeVertex, ...], Field(min_length=2, max_length=16)]

    @model_validator(mode='after')
    def exact_order(self):
        if tuple(vertex.anchor_id for vertex in self.vertices) != self.anchor_ids:
            raise ValueError('saved vertices must match exact ordered anchor IDs')
        return self

class ShapeWrite(SpaceModel):
    actor_id: str = Field(min_length=1,max_length=160)
    operation_id: StableId
    expected_revision: StableId | None
    expected_binding: SpaceBinding
    expected_artifact: AnchorArtifact
    expected_semantic_revision: StableId | None
    expected_anchor_revision: StableId | None
    shapes: Annotated[tuple[ShapeInput,...], Field(max_length=64)]

    @field_validator('actor_id')
    @classmethod
    def actor(cls,value): return SpaceWrite.actor(value)

    @model_validator(mode='after')
    def unique(self):
        if len({shape.id for shape in self.shapes}) != len(self.shapes): raise ValueError('duplicate stable shape ID')
        return self

class ShapeRecord(SpaceModel):
    revision: StableId | None
    updated_at_us: Annotated[StrictInt,Field(ge=1)] | None
    actor_id: str | None = Field(min_length=1,max_length=160)
    shapes: Annotated[tuple[SemanticShape,...],Field(max_length=64)]

    @model_validator(mode='after')
    def coherent(self):
        if self.actor_id is not None: SpaceWrite.actor(self.actor_id)
        if self.revision is None:
            if self.updated_at_us is not None or self.actor_id is not None or self.shapes: raise ValueError('unknown shape registry must be empty')
        elif self.updated_at_us is None or self.actor_id is None: raise ValueError('saved registry requires actor and timestamp')
        if len({shape.id for shape in self.shapes}) != len(self.shapes): raise ValueError('duplicate saved shape ID')
        return self

class ShapeReview(SpaceModel):
    id: StableId
    status: Literal['reviewed','stale','retired','missing']

class ShapeSnapshot(SpaceModel):
    contract: Literal['noesis.semantic.shapes'] = 'noesis.semantic.shapes'
    version: Literal[1] = 1
    binding: SpaceBinding
    artifact: AnchorArtifact
    semantic_revision: StableId | None
    anchor_revision: StableId | None
    point_choices: Annotated[tuple[SemanticAnchor,...],Field(max_length=128)]
    record: ShapeRecord
    status: Literal['unknown','available']
    review_states: Annotated[tuple[ShapeReview,...],Field(max_length=64)]

class ShapeResult(SpaceModel):
    contract: Literal['noesis.semantic.shapes.result'] = 'noesis.semantic.shapes.result'
    version: Literal[1] = 1
    operation_id: StableId
    binding: SpaceBinding
    artifact: AnchorArtifact
    semantic_revision: StableId | None
    anchor_revision: StableId | None
    record: ShapeRecord
    replayed: StrictBool
    review_states: Annotated[tuple[ShapeReview,...],Field(max_length=64)]
