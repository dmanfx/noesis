"""Finite larger-grid work allowance without changing surfel acceptance."""
from argparse import Namespace

import numpy as np
import pytest

from tools.mapanything_phone_scan.build_dense_review_surfels import build
from tools.mapanything_phone_scan.test_dense_review_surfels import _fixture
from tools.mapanything_phone_scan import clean_closed_surface_topology as cleanup
from tools.mapanything_phone_scan.texture_single_room_review_surface import mesh_work_bounds, TextureSurfaceError
from tools.mapanything_phone_scan.test_clean_closed_surface_topology import _pinched_tetrahedra


@pytest.mark.parametrize("budget", [0, -1, 64_000_001])
def test_dense_pixel_budget_remains_explicitly_bounded(tmp_path, budget):
    with pytest.raises(ValueError, match="total-pixel memory budget"):
        build(tmp_path / "consensus", tmp_path / "alignment", tmp_path / "output",
              camera_id="living-room", max_total_pixels=budget)
    assert not (tmp_path / "output").exists()


def test_explicit_larger_work_budget_preserves_exact_surfels_and_quality_gates(tmp_path):
    consensus, alignment = _fixture(tmp_path)
    normal = build(consensus, alignment, tmp_path / "default", camera_id="living-room")
    larger = build(consensus, alignment, tmp_path / "larger", camera_id="living-room",
                   max_total_pixels=48_000_000)
    assert normal["metrics"] == larger["metrics"]
    assert normal["parameters"]["max_total_pixels"] == 40_000_000
    assert larger["parameters"]["max_total_pixels"] == 48_000_000
    with np.load(tmp_path / "default/surfel_points.npz") as before, np.load(tmp_path / "larger/surfel_points.npz") as after:
        assert before.files == after.files
        for name in before.files:
            np.testing.assert_array_equal(before[name], after[name])


def test_larger_mesh_work_limit_keeps_exact_cleanup_and_classification(monkeypatch):
    inputs = _pinched_tetrahedra()
    baseline = cleanup.clean_closed_surface_topology(*inputs, minimum_component_faces=4)
    monkeypatch.setattr(cleanup, "MAX_VERTICES", 7)
    with pytest.raises(ValueError, match="output vertex bound"):
        cleanup.clean_closed_surface_topology(*inputs, minimum_component_faces=4)
    explicit = cleanup.clean_closed_surface_topology(*inputs, minimum_component_faces=4,
                                                     max_faces=5_000_000, max_vertices=3_000_000)
    for name in ("vertices", "faces", "colors", "classification", "retained_face_indices", "vertex_source_indices"):
        np.testing.assert_array_equal(getattr(baseline, name), getattr(explicit, name))
    assert explicit.report["new_geometry_created"] is False
    assert explicit.report["provider_confidence_or_geometry_gates_changed"] is False


@pytest.mark.parametrize("faces,vertices", [(0, 1), (True, 1), (5_000_001, 1), (1, 3_000_001), (1, False)])
def test_explicit_mesh_limits_fail_closed(faces, vertices):
    with pytest.raises(TextureSurfaceError, match="work bounds"):
        mesh_work_bounds(Namespace(max_faces=faces, max_vertices=vertices))
    with pytest.raises(ValueError, match="work bounds"):
        cleanup.clean_closed_surface_topology(*_pinched_tetrahedra(), max_faces=faces, max_vertices=vertices)


def test_mesh_work_defaults_are_unchanged():
    assert mesh_work_bounds(Namespace()) == (3_000_000, 2_000_000)
    assert mesh_work_bounds(Namespace(max_faces=5_000_000, max_vertices=3_000_000)) == (5_000_000, 3_000_000)
