"""Offline reviewed static semantics; never a person-localization authority."""
from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import Field, model_validator

from .base import ContractModel, Sha256

SemanticVector = tuple[float, float, float]
SemanticRotation = tuple[float, float, float, float, float, float, float, float, float]


class SemanticArtifact(ContractModel):
    role: str = Field(min_length=1, max_length=96)
    path: str = Field(min_length=1, max_length=2048)
    sha256: Sha256


class SemanticBox(ContractModel):
    """Basis columns are local length, width, height axes in backend Y-up."""
    center_m: SemanticVector
    size_m: SemanticVector
    basis_row_major: SemanticRotation

    @model_validator(mode="after")
    def valid_geometry(self):
        if not all(math.isfinite(v) for v in (*self.center_m, *self.size_m, *self.basis_row_major)):
            raise ValueError("geometry must be finite")
        if any(abs(v) > 1000 for v in self.center_m):
            raise ValueError("geometry exceeds the bounded local scene range")
        if self.size_m[0] <= 0 or self.size_m[2] <= 0 or self.size_m[1] < 0 or max(self.size_m) > 32:
            raise ValueError("invalid metric dimensions; zero width is allowed only for a plane")
        r = self.basis_row_major
        for a in range(3):
            for b in range(3):
                dot = sum(r[i * 3 + a] * r[i * 3 + b] for i in range(3))
                if abs(dot - (1 if a == b else 0)) > 1e-6:
                    raise ValueError("box basis must be orthonormal")
        det = r[0] * (r[4]*r[8]-r[5]*r[7]) - r[1] * (r[3]*r[8]-r[5]*r[6]) + r[2] * (r[3]*r[7]-r[4]*r[6])
        if abs(det - 1) > 1e-6:
            raise ValueError("box basis must be right handed")
        # This contract deliberately describes upright static room elements.
        if max(abs(r[2]), abs(r[8]), abs(r[5] - 1)) > 1e-6:
            raise ValueError("local height axis must equal backend +Y")
        return self


class SemanticElement(ContractModel):
    element_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,95}$")
    kind: Literal["wall", "opening", "object"]
    category: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=160)
    geometry: SemanticBox
    wall_id: str | None = Field(default=None, max_length=96)
    review_state: Literal["proposed", "accepted", "rejected"] = "proposed"
    review_actor: str = Field(default="", max_length=96)
    review_reason: str = Field(default="", max_length=2000)
    physical_qualification: Literal["not_independently_qualified"] = "not_independently_qualified"
    source_kind: Literal["model_proposal", "manual_annotation"] = "model_proposal"

    @model_validator(mode="after")
    def valid_semantics(self):
        if self.kind == "object" and self.geometry.size_m[1] <= 0:
            raise ValueError("object width must be positive")
        if (self.kind == "opening") != bool(self.wall_id):
            raise ValueError("only openings require a wall reference")
        if self.kind == "opening" and self.category not in ("door", "window"):
            raise ValueError("opening must be a door or window")
        if self.kind == "wall" and self.category != "wall":
            raise ValueError("wall category must be wall")
        if self.review_state != "proposed" and (not self.review_actor or not self.review_reason):
            raise ValueError("review decisions require an actor and reason")
        if self.source_kind == "manual_annotation" and not self.review_reason:
            raise ValueError("manual geometry requires evidence/reason")
        return self


class SceneSemanticMap(ContractModel):
    contract: Literal["noesis.scene_semantics"] = "noesis.scene_semantics"
    contract_version: Literal[1] = 1
    map_id: str = Field(min_length=1, max_length=128)
    revision_sha256: Sha256
    proposal_revision_sha256: Sha256
    coordinate_frame: Literal["backend_world_m"] = "backend_world_m"
    units: Literal["meters"] = "meters"
    world_frame_revision: str = Field(min_length=1, max_length=256)
    world_transform_sha256: Sha256
    authority: Literal["offline_review_and_advisory_context_only"] = "offline_review_and_advisory_context_only"
    room_membership_authority: Literal["external_authored_labels_only"] = "external_authored_labels_only"
    source_artifacts: Annotated[tuple[SemanticArtifact, ...], Field(min_length=1, max_length=32)]
    prepared_to_backend_rotation: SemanticRotation
    conversion_note: str = Field(min_length=1, max_length=2000)
    elements: Annotated[tuple[SemanticElement, ...], Field(max_length=256)]

    @model_validator(mode="after")
    def references_are_valid(self):
        if not all(math.isfinite(v) for v in self.prepared_to_backend_rotation):
            raise ValueError("conversion matrix must be finite")
        # Validate the rotation using a composed upright basis in the service;
        # this matrix itself changes Z-up to Y-up and is not an upright box.
        r = self.prepared_to_backend_rotation
        for a in range(3):
            for b in range(3):
                if abs(sum(r[i*3+a]*r[i*3+b] for i in range(3)) - (a == b)) > 1e-6:
                    raise ValueError("conversion matrix must be orthonormal")
        det = r[0]*(r[4]*r[8]-r[5]*r[7])-r[1]*(r[3]*r[8]-r[5]*r[6])+r[2]*(r[3]*r[7]-r[4]*r[6])
        if abs(det - 1) > 1e-6:
            raise ValueError("conversion matrix must be right handed")
        by_id = {e.element_id: e for e in self.elements}
        if len(by_id) != len(self.elements):
            raise ValueError("duplicate element_id")
        for e in self.elements:
            if e.wall_id and (e.wall_id not in by_id or by_id[e.wall_id].kind != "wall"):
                raise ValueError("opening references a missing/non-wall element")
        return self
