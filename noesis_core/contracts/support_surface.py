"""Bounded, revision-bound measured support planes.

An offline reconstruction is review evidence. Only a separate calibration
artifact with ``authority='measured_calibration'`` may be supplied to live
localization. A Scene Prior is never converted to this contract implicitly.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, model_validator

from .base import ContractModel, Sha256


MAX_SURFACES = 16
MAX_EXTENT_VERTICES = 64
MAX_EXTENT_AREA_M2 = 10_000.0
MAX_WORLD_COORD_M = 10_000.0
MAX_RESIDUAL_SIGMA_M = 10.0


def _validate_extent_xz(polygon: tuple[tuple[float, float], ...]) -> None:
    if not all(math.isfinite(v) and abs(v) <= MAX_WORLD_COORD_M
               for point in polygon for v in point):
        raise ValueError("support surface extent coordinates must be finite and physically bounded")
    area_twice = sum(
        polygon[i][0] * polygon[(i + 1) % len(polygon)][1]
        - polygon[(i + 1) % len(polygon)][0] * polygon[i][1]
        for i in range(len(polygon))
    )
    if not 2e-4 <= abs(area_twice) <= 2 * MAX_EXTENT_AREA_M2:
        raise ValueError("support surface extent area is outside bounds")
    sign = 1.0 if area_twice > 0 else -1.0
    for i in range(len(polygon)):
        ax, az = polygon[i]
        bx, bz = polygon[(i + 1) % len(polygon)]
        if math.hypot(bx - ax, bz - az) < 1e-3:
            raise ValueError("support surface extent has a degenerate edge")
        for j, (cx, cz) in enumerate(polygon):
            if j == i or j == (i + 1) % len(polygon):
                continue
            cross = (bx - ax) * (cz - az) - (bz - az) * (cx - ax)
            if sign * cross < -1e-8:
                raise ValueError("support surface extent must be convex")
    if len(set(polygon)) != len(polygon):
        raise ValueError("support surface extent has repeated vertices")


class SupportSurface(ContractModel):
    """One observed finite plane in the bound world frame.

    The plane is ``normal_world dot (x, y, z) + offset_m = 0``. Its residual
    sigma measures local fit scatter; it does not bound global calibration,
    registration, or depth scale error.
    """

    surface_id: str = Field(min_length=1, max_length=160)
    revision: str = Field(min_length=1, max_length=200)
    normal_world: tuple[float, float, float]
    offset_m: float
    extent_xz: tuple[tuple[float, float], ...] = Field(min_length=3, max_length=MAX_EXTENT_VERTICES)
    residual_sigma_m: float = Field(gt=0.0)
    source_observation_count: int = Field(ge=3, le=1_000_000_000)
    source_kind: str = Field(min_length=1, max_length=120)
    source_artifact_sha256: Sha256

    @model_validator(mode="after")
    def _finite_convex_plane(self) -> "SupportSurface":
        values = (*self.normal_world, self.offset_m, self.residual_sigma_m,
                  *(v for point in self.extent_xz for v in point))
        if not all(math.isfinite(v) for v in values):
            raise ValueError("support surface coordinates and uncertainty must be finite")
        if abs(self.offset_m) > MAX_WORLD_COORD_M or any(
            abs(v) > MAX_WORLD_COORD_M for point in self.extent_xz for v in point
        ):
            raise ValueError("support surface coordinates exceed physical bounds")
        if self.residual_sigma_m > MAX_RESIDUAL_SIGMA_M:
            raise ValueError("support surface residual uncertainty exceeds physical bounds")
        nx, ny, nz = self.normal_world
        if abs(math.sqrt(nx * nx + ny * ny + nz * nz) - 1.0) > 1e-6 or ny <= 1e-3:
            raise ValueError("support surface normal must be unit length and point upward")

        _validate_extent_xz(self.extent_xz)
        return self


class SupportSurfaceSet(ContractModel):
    """One camera and world-revision binding for a small surface collection."""

    contract: Literal["noesis.support_surfaces"] = "noesis.support_surfaces"
    contract_version: Literal[1] = 1
    camera_id: str = Field(min_length=1, max_length=160)
    world_frame: Literal["backend_world_m"] = "backend_world_m"
    world_frame_revision: str = Field(min_length=1, max_length=200)
    calibration_revision: str = Field(min_length=1, max_length=200)
    world_transform_sha256: Sha256
    evidence_sha256: Sha256
    authority: Literal["review_only", "measured_calibration"] = "review_only"
    surfaces: tuple[SupportSurface, ...] = Field(default=(), max_length=MAX_SURFACES)

    @model_validator(mode="after")
    def _unique_surfaces(self) -> "SupportSurfaceSet":
        ids = [surface.surface_id for surface in self.surfaces]
        if len(set(ids)) != len(ids):
            raise ValueError("support surface IDs must be unique within one binding")
        return self


class SupportSurfaceReference(ContractModel):
    """Compact selected-plane identity for a canonical world measurement."""

    surface_id: str = Field(min_length=1, max_length=160)
    revision: str = Field(min_length=1, max_length=200)
    normal_world: tuple[float, float, float]
    offset_m: float
    extent_xz: tuple[tuple[float, float], ...] = Field(min_length=3, max_length=MAX_EXTENT_VERTICES)

    @classmethod
    def from_surface(cls, surface: SupportSurface) -> "SupportSurfaceReference":
        return cls(surface_id=surface.surface_id, revision=surface.revision,
                   normal_world=surface.normal_world, offset_m=surface.offset_m,
                   extent_xz=surface.extent_xz)

    @model_validator(mode="after")
    def _valid_plane(self) -> "SupportSurfaceReference":
        if not all(math.isfinite(v) for v in (*self.normal_world, self.offset_m)):
            raise ValueError("support surface reference plane must be finite")
        if abs(self.offset_m) > MAX_WORLD_COORD_M:
            raise ValueError("support surface reference offset exceeds physical bounds")
        norm = math.sqrt(sum(v * v for v in self.normal_world))
        if abs(norm - 1.0) > 1e-6 or self.normal_world[1] <= 1e-3:
            raise ValueError("support surface reference normal must be unit upward")
        _validate_extent_xz(self.extent_xz)
        return self
