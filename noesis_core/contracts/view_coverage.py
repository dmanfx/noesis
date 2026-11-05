"""Request-only mathematical projection and incomplete static-prior evidence."""
from __future__ import annotations

from typing import Annotated, Literal
from pydantic import ConfigDict, Field, StrictBool, StrictInt, model_validator
from .base import ContractModel, Sha256
from .scene_prior import ScenePriorQuality

MAX_REQUEST_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 256 * 1024
MAX_POINTS = 64
MAX_RAY_STEPS = 512
MAX_TOTAL_STEPS = 32768
Number = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=-1e6, le=1e6)]
Vector = Annotated[tuple[Number, Number, Number], Field(min_length=3, max_length=3)]
Token = Annotated[str, Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$")]
Count = Annotated[StrictInt, Field(ge=0, le=MAX_TOTAL_STEPS)]


class CoverageModel(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=False, allow_inf_nan=False, validate_default=True)


class ViewPoint(CoverageModel):
    id: Token
    position_m: Vector


class ViewAnalysisBinding(CoverageModel):
    prior_id: Token
    world_frame: Literal["backend_world_m"]
    world_frame_revision: Token
    camera_calibration_sha256: Sha256
    world_alignment_sha256: Sha256
    frame_transform_sha256: Sha256


class ViewAnalysisRequest(ViewAnalysisBinding):
    points: Annotated[tuple[ViewPoint, ...], Field(min_length=1, max_length=MAX_POINTS)]

    @model_validator(mode="after")
    def unique_ids(self):
        if len({point.id for point in self.points}) != len(self.points):
            raise ValueError("duplicate requested point ID")
        return self


class ViewProjection(CoverageModel):
    basis: Literal["mathematical_linear_k_only", "declared_opencv_fisheye", "unknown"]
    image_domain: Literal["linear_k_image_plane", "declared_distorted_calibration_image", "unknown"]
    distortion_applied: StrictBool
    state: Literal["in", "out", "unknown"]
    reason: Literal["inside_image", "outside_image", "behind_camera", "nonfinite_projection",
        "invalid_projection_metadata", "unsupported_projection_model", "unsupported_coefficient_order",
        "unsupported_projection_domain"]
    pixel: tuple[float, float] | None
    camera_depth_m: float | None
    optical_visibility: Literal["unknown"] = "unknown"


class StaticPriorAnalysis(CoverageModel):
    state: Literal["potential_static_obstruction", "no_sampled_static_obstruction", "unknown"]
    reason: Literal["sampled_height_envelope_intersects_ray", "sampled_prior_only", "unobserved_cell", "sparse_cell",
        "outside_grid", "invalid_height", "step_bound", "vertical_ray_not_resolved_by_2_5d", "prior_quality_not_passed", "ray_below_sampled_floor"]
    sample_count: Annotated[StrictInt, Field(ge=0, le=MAX_RAY_STEPS)]


class PointViewAnalysis(CoverageModel):
    id: Token
    position_m: Vector
    projection: ViewProjection
    static_prior: StaticPriorAnalysis


class ViewAnalysisProvenance(CoverageModel):
    prior_id: Token
    authored_scene_sha256: Sha256
    prior_manifest_sha256: Sha256
    grid_sha256: Sha256
    camera_calibration_sha256: Sha256
    world_alignment_sha256: Sha256
    frame_transform_sha256: Sha256
    source_frame_revision: Token
    world_frame: Literal["backend_world_m"]
    world_frame_revision: Token
    target_revision_id: Token
    image_size: tuple[Annotated[StrictInt, Field(ge=1, le=32768)], Annotated[StrictInt, Field(ge=1, le=32768)]]
    unit_scale: Literal[1.0]
    floor_y_m: float
    grid_resolution_m: float
    mode: Literal["shadow"]
    quality: ScenePriorQuality
    projection_metadata_status: Literal["undeclared", "declared", "invalid"] = "undeclared"
    projection_model: Annotated[str, Field(min_length=1, max_length=64)] | None = None
    distortion_model: Annotated[str, Field(min_length=1, max_length=64)] | None = None
    distortion_coefficients: Annotated[tuple[float, ...], Field(min_length=1, max_length=16)] | None = None
    distortion_coefficient_names: Annotated[tuple[Annotated[str, Field(min_length=1, max_length=32)], ...], Field(min_length=1, max_length=16)] | None = None
    distortion_applied: Literal[False] = False


class ViewAnalysisContext(CoverageModel):
    contract: Literal["noesis.scene-prior.view-analysis.context"] = "noesis.scene-prior.view-analysis.context"
    version: Literal[1] = 1
    camera_id: Annotated[str, Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$")]
    binding: ViewAnalysisBinding
    provenance: ViewAnalysisProvenance
    camera_position_m: Vector


class ViewAnalysisResultProvenance(ViewAnalysisProvenance):
    distortion_applied: StrictBool


class ViewAnalysis(CoverageModel):
    contract: Literal["noesis.scene-prior.view-analysis"] = "noesis.scene-prior.view-analysis"
    version: Literal[2] = 2
    camera_id: Annotated[str, Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$")]
    provenance: ViewAnalysisResultProvenance
    camera_position_m: Vector
    results: Annotated[tuple[PointViewAnalysis, ...], Field(min_length=1, max_length=MAX_POINTS)]
    total_samples: Count
    qualification: Literal["projection_and_incomplete_static_height_samples_do_not_establish_physical_visibility_or_clear_line_of_sight"] = "projection_and_incomplete_static_height_samples_do_not_establish_physical_visibility_or_clear_line_of_sight"
