"""Whole-domain samples over real verified synthetic prior/calibration artifacts."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import pytest
from noesis_core.contracts.room_coverage import RoomCoverageRequest, MAX_SAMPLES
from noesis_core.room_coverage import RoomCoverageService
from noesis_core.view_coverage import ViewCoverageError, ViewCoverageRetired

spec = importlib.util.spec_from_file_location('coverage_fixture', Path(__file__).with_name('test_view_coverage.py'))
fixture_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture_module)

def test_complete_domain_original_coordinates_and_unknown_cells(tmp_path):
    f = fixture_module.fixture(tmp_path)
    request = RoomCoverageRequest(**f.context.binding.model_dump(), height_above_floor_m=1.)
    result = RoomCoverageService(f.priors, f.snapshot, 'camera-a').analyze(request)
    assert (result.grid.rows, result.grid.columns) == (2, 2)
    assert (result.grid.min_x, result.grid.max_x, result.grid.min_z, result.grid.max_z) == (0, 2, 0, 2)
    assert result.grid.plane_y_m == 1 and len(result.cells) == 4
    assert result.camera_position_m == (.5, 2., .5)
    assert result.total_samples == sum(c.sample_count for c in result.cells) <= MAX_SAMPLES
    assert any(c.state == 'no_sampled_obstruction' for c in result.cells)
    assert 0 < result.remaining_ms <= 120000
    hole = fixture_module.fixture(tmp_path/'holes', observed=0)
    value = RoomCoverageService(hole.priors, hole.snapshot, 'camera-a').analyze(request)
    assert all(c.state in ('unknown', 'outside_camera') for c in value.cells)
    assert all(c.reason == 'unobserved_cell' for c in value.cells if c.state == 'unknown')

def test_source_binding_projection_and_retirement(tmp_path):
    f = fixture_module.fixture(tmp_path)
    request = RoomCoverageRequest(**f.context.binding.model_dump(), height_above_floor_m=1.)
    with pytest.raises(ViewCoverageError):
        RoomCoverageService(f.priors, f.snapshot, 'camera-a').analyze(request.model_copy(update={'frame_transform_sha256':'f'*64}))
    with pytest.raises(ViewCoverageRetired):
        RoomCoverageService(f.priors, f.snapshot, 'camera-a', is_current=lambda:False).analyze(request)
    bad = replace(f.snapshot, projection_metadata_status='declared', projection_model='unsupported')
    result = RoomCoverageService(f.priors, bad, 'camera-a').analyze(request)
    assert all(c.state == 'unknown' and c.reason == 'unsupported_projection_model' for c in result.cells)
