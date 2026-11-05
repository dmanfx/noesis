"""Manual, source-bound sampling of a complete verified Scene Prior domain."""
from typing import Annotated, Literal
from pydantic import Field, StrictInt, model_validator
from .view_coverage import CoverageModel, ViewAnalysisBinding, ViewAnalysisResultProvenance, Vector

MAX_CELLS = 1024
MAX_SAMPLES = 262144

class RoomCoverageRequest(ViewAnalysisBinding):
    height_above_floor_m: Annotated[float, Field(strict=True, ge=0, le=5, allow_inf_nan=False)]

class CoverageGrid(CoverageModel):
    min_x: float
    min_z: float
    max_x: float
    max_z: float
    rows: Annotated[StrictInt, Field(ge=1, le=32)]
    columns: Annotated[StrictInt, Field(ge=1, le=32)]
    cell_width_m: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    cell_depth_m: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    plane_y_m: float
    orientation: Literal['row_increases_positive_z_column_increases_positive_x'] = 'row_increases_positive_z_column_increases_positive_x'

class CoverageCell(CoverageModel):
    state: Literal['outside_camera', 'potential_obstruction', 'no_sampled_obstruction', 'unknown']
    reason: Annotated[str, Field(min_length=1, max_length=80)]
    sample_count: Annotated[StrictInt, Field(ge=0, le=512)]

class RoomCoverageResult(CoverageModel):
    contract: Literal['noesis.scene-prior.domain-coverage'] = 'noesis.scene-prior.domain-coverage'
    version: Literal[1] = 1
    camera_id: Annotated[str, Field(min_length=1, max_length=160)]
    scope: Literal['verified_scene_prior_domain'] = 'verified_scene_prior_domain'
    provenance: ViewAnalysisResultProvenance
    camera_position_m: Vector
    height_above_floor_m: Annotated[float, Field(strict=True, ge=0, le=5, allow_inf_nan=False)]
    grid: CoverageGrid
    cells: Annotated[tuple[CoverageCell, ...], Field(min_length=1, max_length=MAX_CELLS)]
    total_samples: Annotated[StrictInt, Field(ge=0, le=MAX_SAMPLES)]
    remaining_ms: Annotated[StrictInt, Field(ge=1, le=120000)]
    qualification: Literal['cell_center_samples_of_static_geometry_not_optical_visibility_or_full_cell_clearance'] = 'cell_center_samples_of_static_geometry_not_optical_visibility_or_full_cell_clearance'

    @model_validator(mode='after')
    def shape(self):
        if len(self.cells) != self.grid.rows * self.grid.columns:
            raise ValueError('coverage grid shape mismatch')
        return self
