from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, field_validator, model_validator

from noesis_core.analytics_zones import is_exact_zone_label

from .base import (
    ArtifactFingerprint,
    Confidence,
    ContractModel,
    CoordinateFrame,
    CoordinateUnits,
    Matrix3,
    ProducerRef,
    SequenceNumber,
    Sha256,
    TimestampUs,
    Vector3,
)
from .identity import TrackletRef


ZoneSource = Literal["nvdsanalytics_roi", "camera_default"]


class WorldPositionObservation(ContractModel):
    position: Vector3
    covariance: Matrix3
    frame: Literal["backend_world_m"]
    units: Literal["meters"]
    source: str = Field(min_length=1, max_length=120)
    quality: Literal["good", "estimated", "held"]
    confidence: Confidence
    reason: str | None = Field(default=None, max_length=200)


class WorldObservationDiagnostics(ContractModel):
    """Bounded non-biometric evidence for one world-placement decision."""

    estimator_evaluated: bool | None = None
    first_divergence_reason: str | None = Field(default=None, max_length=240)
    floor_candidate_m: Vector3 | None = None
    depth_candidate_m: Vector3 | None = None
    prefilter_measurement_m: Vector3 | None = None
    filter_prediction_m: Vector3 | None = None
    floor_range_m: float | None = None
    floor_range_limit_m: float | None = None
    floor_incidence_sin: float | None = None
    floor_admitted: bool | None = None
    floor_rejection_reason: str | None = Field(default=None, max_length=200)
    measurement_accepted: bool | None = None
    measurement_rejection_reason: str | None = Field(default=None, max_length=200)
    innovation_m: float | None = None
    innovation_limit_m: float | None = None
    fusion_policy_id: str | None = Field(default=None, max_length=200)
    floor_weight_scale: float | None = None
    depth_weight_scale: float | None = None
    floor_weight_effective: float | None = None
    depth_weight_effective: float | None = None
    depth_status: str | None = Field(default=None, max_length=120)
    depth_anchor_m: float | None = None
    depth_registered_m: float | None = None
    depth_used_m: float | None = None
    depth_registration_status: str | None = Field(default=None, max_length=120)
    depth_registration_id: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def _numeric_evidence_is_finite(self) -> "WorldObservationDiagnostics":
        for point in (
            self.floor_candidate_m,
            self.depth_candidate_m,
            self.prefilter_measurement_m,
            self.filter_prediction_m,
        ):
            if point is not None and not all(
                math.isfinite(value) for value in (point.x, point.y, point.z)
            ):
                raise ValueError("world diagnostic candidate points must be finite")
        for value in (
            self.floor_range_m,
            self.floor_range_limit_m,
            self.floor_incidence_sin,
            self.innovation_m,
            self.innovation_limit_m,
            self.floor_weight_scale,
            self.depth_weight_scale,
            self.floor_weight_effective,
            self.depth_weight_effective,
            self.depth_anchor_m,
            self.depth_registered_m,
            self.depth_used_m,
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError("world diagnostic scalar values must be finite")
        return self


class PersonObservation(ContractModel):
    tracklet: TrackletRef
    class_id: Literal[0] = 0
    detection_confidence: Confidence | None = None
    tracker_confidence: Confidence | None = None
    bbox_xywh: tuple[float, float, float, float]
    image_size: tuple[int, int]
    zone: str | None = Field(default=None, min_length=1, max_length=160)
    zone_source: ZoneSource | None = None
    zone_authoritative: bool = False
    world: WorldPositionObservation | None = None
    world_diagnostics: WorldObservationDiagnostics | None = None
    embedding_sequence: int | None = Field(default=None, ge=0)
    embedding_model_sha256: Sha256 | None = None
    embedding_dimension: int | None = Field(default=None, ge=1)
    pose_present: bool = False
    depth_present: bool = False
    occluded: bool = False

    @field_validator("zone", mode="before")
    @classmethod
    def _zone_is_not_normalized(cls, value: object) -> object:
        if value is not None and not is_exact_zone_label(value):
            raise ValueError(
                "zone labels must be nonempty, unpadded, and at most 160 characters"
            )
        return value

    @model_validator(mode="after")
    def _image_size_is_valid(self) -> "PersonObservation":
        if self.image_size[0] <= 0 or self.image_size[1] <= 0:
            raise ValueError("image_size must be positive")
        provenance_present = (
            self.embedding_sequence is not None,
            self.embedding_model_sha256 is not None,
            self.embedding_dimension is not None,
        )
        if any(provenance_present) and not all(provenance_present):
            raise ValueError(
                "embedding provenance requires sequence, model fingerprint, and dimension"
            )
        if (
            self.world is None
            and self.world_diagnostics is not None
            and self.world_diagnostics.first_divergence_reason is None
        ):
            raise ValueError(
                "invalid world diagnostics require first_divergence_reason"
            )
        if (
            self.world is not None
            and self.world_diagnostics is not None
            and self.world_diagnostics.first_divergence_reason is not None
        ):
            raise ValueError(
                "admitted world observations cannot carry a divergence reason"
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


class ObservationEnvelope(ContractModel):
    contract: Literal["noesis.observation.person"]
    contract_version: Literal[1]
    observation_id: str = Field(min_length=1, max_length=200)
    producer: ProducerRef
    sequence: SequenceNumber
    captured_at_us: TimestampUs | None = None
    observed_at_us: TimestampUs
    published_at_us: TimestampUs
    capture_time_status: Literal["synced", "estimated", "unavailable"]
    media_pts_ns: int | None = Field(default=None, ge=0)
    coordinate_frame: CoordinateFrame
    units: CoordinateUnits
    calibration: ArtifactFingerprint
    model: ArtifactFingerprint
    config: ArtifactFingerprint
    payload: PersonObservation

    @model_validator(mode="after")
    def _timestamps_and_frames_are_coherent(self) -> "ObservationEnvelope":
        if self.observed_at_us > self.published_at_us:
            raise ValueError("observed_at_us cannot follow published_at_us")
        if self.captured_at_us is not None and self.captured_at_us > self.observed_at_us:
            raise ValueError("captured_at_us cannot follow observed_at_us")
        if self.capture_time_status == "unavailable" and self.captured_at_us is not None:
            raise ValueError("unavailable capture time must not carry captured_at_us")
        if self.capture_time_status != "unavailable" and self.captured_at_us is None:
            raise ValueError("synced/estimated capture time requires captured_at_us")
        if self.payload.tracklet.run_id != self.producer.run_id:
            raise ValueError("tracklet run_id must match producer run_id")
        if self.payload.tracklet.observed_at_us != self.observed_at_us:
            raise ValueError("tracklet and envelope observed_at_us must match")
        if self.payload.world is not None:
            if self.coordinate_frame != self.payload.world.frame or self.units != self.payload.world.units:
                raise ValueError("envelope frame/units must match world observation")
        return self
