from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


TimestampUs = Annotated[int, Field(ge=1)]
SequenceNumber = Annotated[int, Field(ge=0)]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class ContractModel(BaseModel):
    """Strict immutable base for all public product contracts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class ArtifactFingerprint(ContractModel):
    """Content identity for an input that can change product semantics."""

    role: str = Field(min_length=1, max_length=96)
    sha256: str
    version: str | None = Field(default=None, max_length=160)
    producer: str | None = Field(default=None, max_length=160)

    @field_validator("sha256")
    @classmethod
    def _valid_sha256(cls, value: str) -> str:
        normalized = value.strip().lower()
        if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
            raise ValueError("sha256 must contain exactly 64 lowercase hex characters")
        return normalized


class ContractHeader(ContractModel):
    contract: str = Field(min_length=1, max_length=128)
    contract_version: int = Field(ge=1)


class ProducerRef(ContractModel):
    runtime: Literal["ds8", "ds9", "replay", "test"]
    instance_id: str = Field(min_length=1, max_length=160)
    run_id: str = Field(min_length=1, max_length=160)
    software_revision: str = Field(min_length=1, max_length=160)


class Vector3(ContractModel):
    x: float
    y: float
    z: float


class Matrix3(ContractModel):
    """Row-major 3x3 matrix used for covariance and transform evidence."""

    values: tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ]


CoordinateFrame = Literal["backend_world_m", "camera_local_ground_m", "image_px"]
CoordinateUnits = Literal["meters", "pixels"]
