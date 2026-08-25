"""Contracts for camera-local world measurement hypotheses.

The live tracking path can produce more than one defensible geometric
observation for a person (for example a floor ray and registered depth).  This
module keeps those observations together until one shared resolver adjudicates
them.  The contracts are intentionally small and immutable: they contain
compact scalar evidence and never contain images, masks, or histories.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .base import ContractModel, Matrix3, Sha256, TimestampUs, Vector3


MeasurementKind = Literal[
    "floor_ray",
    "registered_depth",
    "pose_scale",
    "gravity_reconstruction",
]
PostureKind = Literal["standing", "sitting", "lying", "unknown"]
SupportState = Literal["floor", "seat", "couch", "unknown"]
MeasurementStatus = Literal["measured", "prediction", "hold", "rejected"]
MeasurementQuality = Literal[
    "good",
    "estimated",
    "weak",
    "prediction",
    "held",
    "rejected",
]


class MeasurementKindName(StrEnum):
    FLOOR_RAY = "floor_ray"
    REGISTERED_DEPTH = "registered_depth"
    POSE_SCALE = "pose_scale"
    GRAVITY_RECONSTRUCTION = "gravity_reconstruction"


class WorldMeasurementCohort(ContractModel):
    """Exact identity shared by every candidate for one track observation."""

    track_key: str = Field(min_length=1, max_length=200)
    # Keep the generation explicit instead of requiring consumers to parse it
    # out of ``track_key``.  Exact lifecycle identity is mandatory because a
    # tracker ID can be reused after an absence/restart.
    tracker_lifecycle_generation: int = Field(ge=0)
    camera_id: str = Field(min_length=1, max_length=160)
    source_id: int = Field(ge=0)
    tracker_id: int = Field(ge=0)
    frame_id: int = Field(ge=0)
    observed_at_us: TimestampUs
    world_frame: Literal["backend_world_m"] = "backend_world_m"
    world_revision: str = Field(min_length=1, max_length=200)
    calibration_revision: str = Field(min_length=1, max_length=200)
    # This is the immutable world-to-camera transform identity used by BEV
    # admission.  It is optional only for old offline fixtures; live canonical
    # cohorts are rejected by the DS9 producer when the hash is absent.
    world_transform_sha256: Sha256 | None = None
    pcf_revision: str | None = Field(default=None, min_length=1, max_length=200)

    @property
    def identity_key(self) -> tuple[object, ...]:
        """Stable key used to reject stale or mixed-coordinate candidates."""

        return (
            self.track_key,
            int(self.tracker_lifecycle_generation),
            self.camera_id,
            int(self.source_id),
            int(self.tracker_id),
            int(self.frame_id),
            int(self.observed_at_us),
            self.world_frame,
            self.world_revision,
            self.calibration_revision,
            self.world_transform_sha256,
            self.pcf_revision,
        )


class WorldPriorEvidence(ContractModel):
    """Compact, revision-bound PCF/Scene Prior evidence for one candidate.

    ``obstacle_clearance_m`` is retained for diagnostics only.  It is
    deliberately not used as a localization penalty because the current PCF
    obstacle layer does not establish semantic furniture support surfaces.
    """

    prior_id: str = Field(min_length=1, max_length=200)
    revision_id: str = Field(min_length=1, max_length=200)
    status: str = Field(min_length=1, max_length=64)
    inside_extent: bool | None = None
    inside_authored_space: bool | None = None
    extent_outside_distance_m: float | None = Field(default=None, ge=0.0)
    evidence_observed: bool = False
    observed_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    boundary_signed_distance_m: float | None = None
    floor_height_m: float | None = None
    obstacle_clearance_m: float | None = None
    reasons: tuple[str, ...] = Field(default=(), max_length=8)

    @field_validator(
        "boundary_signed_distance_m",
        "extent_outside_distance_m",
        "floor_height_m",
        "obstacle_clearance_m",
        mode="before",
    )
    @classmethod
    def _finite_optional_scalar(cls, value: object) -> object:
        if value is not None and not math.isfinite(float(value)):
            raise ValueError("world prior evidence scalars must be finite")
        return value

    @field_validator("reasons")
    @classmethod
    def _bounded_reasons(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not reason or len(reason) > 160 for reason in value):
            raise ValueError("world prior evidence reasons must be bounded strings")
        return value


def _validate_finite_vector(value: Vector3) -> Vector3:
    if not all(math.isfinite(float(component)) for component in (value.x, value.y, value.z)):
        raise ValueError("world measurement position must be finite")
    return value


def _validate_psd(values: tuple[float, ...]) -> tuple[float, ...]:
    if len(values) != 9:
        raise ValueError("world measurement covariance must contain nine values")
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("world measurement covariance must be finite")

    v = tuple(float(value) for value in values)
    # Covariance matrices are row-major, symmetric, and positive
    # semidefinite.  A small tolerance permits normal floating-point roundoff
    # without silently accepting a materially negative variance.
    scale = max(1.0, max(abs(value) for value in v))
    tolerance = 1e-9 * scale
    for left, right in ((1, 3), (2, 6), (5, 7)):
        if abs(v[left] - v[right]) > tolerance:
            raise ValueError("world measurement covariance must be symmetric")
    d0, d1, d2 = v[0], v[4], v[8]
    if min(d0, d1, d2) < -tolerance:
        raise ValueError("world measurement covariance must be positive semidefinite")
    if d0 * d1 - v[1] * v[1] < -tolerance:
        raise ValueError("world measurement covariance must be positive semidefinite")
    if d0 * d2 - v[2] * v[2] < -tolerance:
        raise ValueError("world measurement covariance must be positive semidefinite")
    if d1 * d2 - v[5] * v[5] < -tolerance:
        raise ValueError("world measurement covariance must be positive semidefinite")
    determinant = (
        d0 * (d1 * d2 - v[5] * v[5])
        - v[1] * (v[3] * d2 - v[5] * v[6])
        + v[2] * (v[3] * v[7] - d1 * v[6])
    )
    if determinant < -tolerance:
        raise ValueError("world measurement covariance must be positive semidefinite")
    return v


class WorldProcessContinuation(ContractModel):
    """A prediction or bounded hold, explicitly separate from measurements."""

    kind: Literal["prediction", "hold"]
    cohort: WorldMeasurementCohort
    position: Vector3
    covariance: Matrix3
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str | None = Field(default=None, max_length=200)

    @field_validator("position")
    @classmethod
    def _position_is_finite(cls, value: Vector3) -> Vector3:
        return _validate_finite_vector(value)

    @field_validator("covariance")
    @classmethod
    def _covariance_is_psd(cls, value: Matrix3) -> Matrix3:
        _validate_psd(value.values)
        return value


class WorldMeasurementHypothesis(ContractModel):
    """One bounded geometric hypothesis for a single exact observation."""

    candidate_id: str = Field(min_length=1, max_length=120)
    cohort: WorldMeasurementCohort
    kind: MeasurementKind
    position: Vector3
    covariance: Matrix3
    anchor: str = Field(min_length=1, max_length=80)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    posture: PostureKind = "unknown"
    support_state: SupportState = "unknown"
    occlusion: float = Field(default=0.0, ge=0.0, le=1.0)
    motion_consistency: float = Field(default=1.0, ge=0.0, le=1.0)
    support_score: float = Field(default=1.0, ge=0.0, le=1.0)
    posture_compatibility: float = Field(default=1.0, ge=0.0, le=1.0)
    source_reliability: float = Field(default=1.0, ge=0.0, le=1.0)
    ray_incidence_sin: float | None = Field(default=None, ge=0.0, le=1.0)
    depth_support_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    depth_registration_sigma_m: float | None = Field(default=None, ge=0.0)
    floor_depth_disagreement_m: float | None = Field(default=None, ge=0.0)
    pixel_uncertainty_px: float | None = Field(default=None, ge=0.0)
    correlation_group: str | None = Field(default=None, min_length=1, max_length=120)
    pcf: WorldPriorEvidence | None = None
    valid: bool = True
    rejection_reason: str | None = Field(default=None, max_length=200)

    @field_validator("position")
    @classmethod
    def _position_is_finite(cls, value: Vector3) -> Vector3:
        return _validate_finite_vector(value)

    @field_validator("covariance")
    @classmethod
    def _covariance_is_psd(cls, value: Matrix3) -> Matrix3:
        _validate_psd(value.values)
        return value

    @field_validator(
        "depth_registration_sigma_m",
        "floor_depth_disagreement_m",
        "pixel_uncertainty_px",
        mode="before",
    )
    @classmethod
    def _optional_evidence_is_finite(cls, value: object) -> object:
        if value is not None and not math.isfinite(float(value)):
            raise ValueError("world measurement evidence must be finite")
        return value

    @model_validator(mode="after")
    def _rejection_has_reason(self) -> "WorldMeasurementHypothesis":
        if not self.valid and not self.rejection_reason:
            raise ValueError("invalid world measurement hypotheses require a reason")
        if self.pcf is not None and self.cohort.pcf_revision != self.pcf.revision_id:
            raise ValueError("PCF evidence revision must match the measurement cohort")
        return self


class WorldMeasurementSet(ContractModel):
    """At most four exact-cohort measurements plus one process continuation."""

    cohort: WorldMeasurementCohort
    hypotheses: tuple[WorldMeasurementHypothesis, ...] = Field(default=(), max_length=4)
    continuation: WorldProcessContinuation | None = None

    @model_validator(mode="after")
    def _cohort_is_exact_and_bounded(self) -> "WorldMeasurementSet":
        candidate_ids = [candidate.candidate_id for candidate in self.hypotheses]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("world measurement candidate_id values must be unique")
        expected = self.cohort.identity_key
        for candidate in self.hypotheses:
            if candidate.cohort.identity_key != expected:
                raise ValueError("world measurement hypotheses must share exact cohort identity")
        if self.continuation is not None and self.continuation.cohort.identity_key != expected:
            raise ValueError("world process continuation must share exact cohort identity")
        return self

    @property
    def prediction(self) -> WorldProcessContinuation | None:
        return self.continuation if self.continuation and self.continuation.kind == "prediction" else None

    @property
    def hold(self) -> WorldProcessContinuation | None:
        return self.continuation if self.continuation and self.continuation.kind == "hold" else None


class WorldMeasurementCandidateDiagnostic(ContractModel):
    """Bounded per-candidate explanation for telemetry and dashboard tooling."""

    candidate_id: str = Field(min_length=1, max_length=120)
    kind: MeasurementKind
    score: float = Field(ge=0.0, le=1.0)
    pcf_score: float = Field(ge=0.0, le=1.0)
    innovation_m: float | None = Field(default=None, ge=0.0)
    agreement_mahalanobis_sq: float | None = Field(default=None, ge=0.0)
    compatible_with_selected: bool = False
    selected: bool = False
    retained_as_alternate: bool = False
    rejection_reason: str | None = Field(default=None, max_length=200)


class ResolvedGroundMeasurement(ContractModel):
    """Canonical result consumed by PersonGroundState and world publishers."""

    cohort: WorldMeasurementCohort
    status: MeasurementStatus
    position: Vector3 | None = None
    covariance: Matrix3 | None = None
    selected_candidate_id: str | None = Field(default=None, max_length=120)
    selected_kind: MeasurementKind | None = None
    quantity: Literal["ground_footprint"] = "ground_footprint"
    posture: PostureKind = "unknown"
    support_state: SupportState = "unknown"
    contributor_ids: tuple[str, ...] = Field(default=(), max_length=4)
    alternate_candidate_id: str | None = Field(default=None, max_length=120)
    fused: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    quality: MeasurementQuality
    pcf_score: float | None = Field(default=None, ge=0.0, le=1.0)
    disagreement_m: float | None = Field(default=None, ge=0.0)
    agreement_mahalanobis_sq: float | None = Field(default=None, ge=0.0)
    reason: str = Field(min_length=1, max_length=240)
    diagnostics: tuple[WorldMeasurementCandidateDiagnostic, ...] = Field(
        default=(), max_length=4
    )

    @field_validator("position")
    @classmethod
    def _position_is_finite(cls, value: Vector3 | None) -> Vector3 | None:
        return None if value is None else _validate_finite_vector(value)

    @field_validator("covariance")
    @classmethod
    def _covariance_is_psd(cls, value: Matrix3 | None) -> Matrix3 | None:
        if value is not None:
            _validate_psd(value.values)
        return value

    @model_validator(mode="after")
    def _result_is_coherent(self) -> "ResolvedGroundMeasurement":
        if (self.position is None) != (self.covariance is None):
            raise ValueError("resolved position and covariance must be present together")
        if self.status == "measured":
            if self.position is None or self.selected_candidate_id is None:
                raise ValueError("measured result requires a selected candidate and position")
            if self.quality in {"prediction", "held", "rejected"}:
                raise ValueError("measured result cannot have continuation/rejected quality")
        elif self.status in {"prediction", "hold"}:
            if self.position is None or self.selected_candidate_id is not None:
                raise ValueError("continuation result must have position and no selected candidate")
            if self.quality not in {"prediction", "held"}:
                raise ValueError("continuation result must have prediction/held quality")
        elif self.position is not None:
            raise ValueError("rejected result must not expose a position")
        if self.fused and len(self.contributor_ids) < 2:
            raise ValueError("fused result requires at least two contributors")
        if (
            self.status == "measured"
            and self.selected_candidate_id not in self.contributor_ids
        ):
            raise ValueError("selected measurement must be a contributor")
        if (
            self.alternate_candidate_id is not None
            and self.alternate_candidate_id in self.contributor_ids
        ):
            raise ValueError("alternate measurement cannot also be a contributor")
        return self

    @property
    def legacy_source_label(self) -> str:
        """Map the universal decision to an existing PGS source semantic.

        PersonGroundState's hysteresis intentionally recognizes the historical
        source labels.  This compatibility view avoids introducing a new
        unknown source string into that state machine while the richer kind,
        contributor, and covariance fields remain authoritative.
        """

        if self.status == "prediction":
            return "cv_prediction"
        if self.status == "hold":
            return "anchor_hold"
        if self.status == "rejected":
            return "invalid"
        if self.fused:
            return "pose_depth_fused"
        return {
            "floor_ray": "pose_floor_only",
            "registered_depth": "pose_depth_only",
            "pose_scale": "pose_leg_floor",
            "gravity_reconstruction": "gravity_drop",
        }.get(self.selected_kind or "", "pose_floor_only")

    @property
    def legacy_quality_label(self) -> str:
        """Map richer resolver quality to the existing world contract."""

        if self.quality == "good":
            return "good"
        if self.quality == "held":
            return "held"
        # ``weak`` and process predictions remain admissible only through the
        # existing estimated/held compatibility vocabulary.  The richer
        # resolver quality remains available on this object for diagnostics.
        if self.quality in {"prediction", "rejected"}:
            return "held"
        return "estimated"


__all__ = [
    "MeasurementKindName",
    "MeasurementKind",
    "MeasurementQuality",
    "MeasurementStatus",
    "PostureKind",
    "SupportState",
    "ResolvedGroundMeasurement",
    "WorldMeasurementCandidateDiagnostic",
    "WorldMeasurementCohort",
    "WorldMeasurementHypothesis",
    "WorldMeasurementSet",
    "WorldPriorEvidence",
    "WorldProcessContinuation",
]
