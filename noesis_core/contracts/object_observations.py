"""Camera-image detector observations, never physical object identities."""
from __future__ import annotations

from typing import Literal
from pydantic import ConfigDict, Field
from .base import ContractModel

TTL_MS = 5000
MAX_ROWS = 512
MAX_CAMERAS = 16
MAX_FRAME_ROWS = 32
MAX_BYTES = 256 * 1024


class ObjectImageBox(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False, strict=True)
    left: float = Field(ge=0)
    top: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class ObjectImageObservation(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False, strict=True, str_strip_whitespace=False)
    observation_id: str = Field(min_length=1, max_length=1400)
    ordinal: int = Field(ge=0)
    class_id: int = Field(ge=1, le=79)
    label: str = Field(min_length=1, max_length=160)
    confidence: float = Field(gt=0, le=1)
    bbox: ObjectImageBox


class ObjectObservationFrame(ContractModel):
    camera_id: str = Field(min_length=1, max_length=160)
    source_id: int = Field(ge=0)
    source_epoch: int = Field(ge=0)
    frame_id: int = Field(ge=0)
    observed_at_us: int = Field(gt=0, le=2**53 - 1)
    captured_at_us: int = Field(gt=0, le=2**53 - 1)
    capture_time_status: Literal["estimated"]
    media_pts_ns: int = Field(ge=0, le=2**53 - 1)
    observation_clock: Literal["processing_unix_epoch_us"] = "processing_unix_epoch_us"
    media_pts_domain: Literal["stream_relative_ns"] = "stream_relative_ns"
    coordinate_frame: Literal["pipeline_image_px"] = "pipeline_image_px"
    image_width: int = Field(gt=0, le=32768)
    image_height: int = Field(gt=0, le=32768)
    source_image_width: int = Field(gt=0, le=32768)
    source_image_height: int = Field(gt=0, le=32768)
    valid_until_us: int = Field(gt=0, le=2**53 - 1)
    omitted_count: int = Field(ge=0)
    invalid_count: int = Field(ge=0)
    observations: tuple[ObjectImageObservation, ...] = Field(max_length=MAX_FRAME_ROWS)
