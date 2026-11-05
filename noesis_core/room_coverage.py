"""Whole-domain geometric sample map; never an optical or safety claim."""
import math
import time
import numpy as np
from .contracts.room_coverage import RoomCoverageRequest, RoomCoverageResult, CoverageGrid, CoverageCell, MAX_SAMPLES
from .contracts.view_coverage import ViewAnalysisResultProvenance
from .view_coverage import ViewCoverageService, ViewCoverageRetired
from .view_projection import project_camera_point

class RoomCoverageService(ViewCoverageService):
    def analyze(self, request):
        request = RoomCoverageRequest.model_validate(request)
        started = time.monotonic()
        E, K, origin, revision, provenance = self._inputs(request)
        domain = revision.manifest.grid
        # Preserve the entire extent. Each display cell represents its center,
        # not a proof that all points or volumes within that cell are clear.
        columns, rows = min(32, domain.columns), min(32, domain.rows)
        bounds = domain.bounds
        width = (bounds.max_x - bounds.min_x) / columns
        depth = (bounds.max_z - bounds.min_z) / rows
        plane = provenance.floor_y_m + request.height_above_floor_m
        cells, total, distorted = [], 0, False
        for row in range(rows):
            for column in range(columns):
                self._check()
                target = np.array([bounds.min_x + (column + .5)*width, plane, bounds.min_z + (row + .5)*depth])
                projection = project_camera_point((E @ np.r_[target, 1.])[:3], K, self.calibration.image_size, self.calibration)
                distorted |= projection.distortion_applied
                if projection.state != 'in':
                    cells.append(CoverageCell(state='outside_camera' if projection.state == 'out' else 'unknown', reason=projection.reason, sample_count=0))
                    continue
                target_sample = revision.sample(np.array([target[0]]), np.array([target[2]]))
                if not bool(target_sample['observed'][0]):
                    cells.append(CoverageCell(state='unknown', reason='unobserved_cell', sample_count=0))
                    continue
                static = self._ray(revision, origin, target, max(0, MAX_SAMPLES-total))
                total += static.sample_count
                state = {'potential_static_obstruction':'potential_obstruction', 'no_sampled_static_obstruction':'no_sampled_obstruction', 'unknown':'unknown'}[static.state]
                cells.append(CoverageCell(state=state, reason=static.reason, sample_count=static.sample_count))
        self._check()
        remaining = 120000 - math.ceil((time.monotonic()-started)*1000)
        if remaining <= 0:
            raise ViewCoverageRetired('domain_coverage_expired')
        return RoomCoverageResult(camera_id=self.camera_id,
            provenance=ViewAnalysisResultProvenance(**{**provenance.model_dump(), 'distortion_applied':distorted}),
            camera_position_m=tuple(float(v) for v in origin), height_above_floor_m=request.height_above_floor_m,
            grid=CoverageGrid(min_x=bounds.min_x, max_x=bounds.max_x, min_z=bounds.min_z, max_z=bounds.max_z,
                rows=rows, columns=columns, cell_width_m=width, cell_depth_m=depth, plane_y_m=plane),
            cells=tuple(cells), total_samples=total, remaining_ms=remaining)
