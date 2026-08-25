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
from .world_measurement import WorldMeasurementCandidateDiagnostic


ZoneSource = Literal["nvdsanalytics_roi", "camera_default"]


class WorldPositionObservation(ContractModel):
    position: Vector3
    covariance: Matrix3
    # ``frame``/``units`` are retained as the v1 envelope-facing names.
    # ``world_frame`` and the revisioned identity below are the canonical
    # registration identity and must travel with a world-valid observation.
    frame: Literal["backend_world_m"] = "backend_world_m"
    world_frame: Literal["backend_world_m"] = "backend_world_m"
    units: Literal["meters"] = "meters"
    world_frame_revision: str | None = Field(default=None, min_length=1, max_length=200)
    world_transform_sha256: Sha256 | None = None
    calibration_revision: str | None = Field(default=None, min_length=1, max_length=200)
    # The world position published by the tracking path is a ground
    # footprint.  Body-root geometry is intentionally not implied by this
    # contract, especially for seated or occluded people.
    quantity: Literal["ground_footprint"] = "ground_footprint"
    support_state: Literal["floor", "seat", "couch", "unknown"] = "unknown"
    posture: Literal["standing", "sitting", "lying", "unknown"] = "unknown"
    source: str = Field(min_length=1, max_length=120)
    quality: Literal["good", "estimated", "held"]
    confidence: Confidence
    reason: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def _registration_identity_is_coherent(self) -> "WorldPositionObservation":
        if self.frame != self.world_frame:
            raise ValueError("world frame aliases must agree")
        if (self.world_frame_revision is None) != (
            self.world_transform_sha256 is None
        ):
            raise ValueError(
                "world frame revision and transform fingerprint must be provided together"
            )
        values = tuple(float(value) for value in self.covariance.values)
        if len(values) != 9 or not all(math.isfinite(value) for value in values):
            raise ValueError("world covariance must be finite and 3x3")
        scale = max(1.0, *(abs(value) for value in values))
        tolerance = 1e-8 * scale
        if any(
            abs(values[left] - values[right]) > tolerance
            for left, right in ((1, 3), (2, 6), (5, 7))
        ):
            raise ValueError("world covariance must be symmetric")
        if any(values[index] < -tolerance for index in (0, 4, 8)):
            raise ValueError("world covariance must be positive semidefinite")
        minors = (
            values[0] * values[4] - values[1] * values[3],
            values[0] * values[8] - values[2] * values[6],
            values[4] * values[8] - values[5] * values[7],
        )
        if any(minor < -tolerance for minor in minors):
            raise ValueError("world covariance must be positive semidefinite")
        determinant = (
            values[0] * (values[4] * values[8] - values[5] * values[7])
            - values[1] * (values[3] * values[8] - values[5] * values[6])
            + values[2] * (values[3] * values[7] - values[4] * values[6])
        )
        if determinant < -tolerance:
            raise ValueError("world covariance must be positive semidefinite")
        return self


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
    # Universal resolver summary.  Candidate diagnostics are deliberately
    # bounded and carry no raw image/depth data, so this can travel with the
    # canonical observation/world cohort and later feed BEV debug rendering.
    resolver_selected_kind: Literal[
        "floor_ray",
        "registered_depth",
        "pose_scale",
        "gravity_reconstruction",
    ] | None = None
    resolver_selected_candidate_id: str | None = Field(default=None, max_length=120)
    resolver_contributor_ids: tuple[str, ...] = Field(default=(), max_length=4)
    resolver_alternate_candidate_id: str | None = Field(default=None, max_length=120)
    resolver_fused: bool | None = None
    resolver_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    resolver_disagreement_m: float | None = Field(default=None, ge=0.0)
    resolver_agreement_mahalanobis_sq: float | None = Field(default=None, ge=0.0)
    resolver_pcf_score: float | None = Field(default=None, ge=0.0, le=1.0)
    resolver_reason: str | None = Field(default=None, max_length=240)
    resolver_candidate_diagnostics: tuple[WorldMeasurementCandidateDiagnostic, ...] = Field(
        default=(), max_length=4
    )

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
            self.resolver_disagreement_m,
            self.resolver_agreement_mahalanobis_sq,
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
