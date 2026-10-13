from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from tools.mapanything_phone_scan.build_closed_room_review_surface import (
    close_boundary, edge_topology, orientable_envelope, _oriented_boundary_caps,
)


def _open_cube():
    mesh = o3d.geometry.TriangleMesh.create_box()
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile([.1, .2, .3], (len(mesh.vertices), 1)))
    mesh.remove_triangles_by_index([0, 1])
    return mesh


def test_boundary_completion_preserves_source_and_classifies_only_added_caps():
    mesh = _open_cube()
    vertices = np.asarray(mesh.vertices).copy()
    faces = np.asarray(mesh.triangles).copy()
    completed, cap, metrics = close_boundary(mesh, 2.)
    assert np.array_equal(vertices, np.asarray(completed.vertices))
    assert np.array_equal(np.sort(faces, axis=1), np.sort(np.asarray(completed.triangles)[:len(faces)], axis=1))
    assert cap.tolist() == [False] * 10 + [True] * 2
    assert np.allclose(np.asarray(completed.vertex_colors), [.1, .2, .3])
    assert metrics["before"]["boundary_edge_count"] == 4
    assert metrics["after"]["boundary_edge_count"] == 0
    assert metrics["after"]["nonmanifold_edge_count"] == 0
    assert metrics["after"]["inconsistent_winding_edge_count"] == 0


def test_holes_outside_explicit_radius_fail_instead_of_claiming_closed():
    with pytest.raises(ValueError, match="cap radius"):
        close_boundary(_open_cube(), .01)


@pytest.mark.parametrize("radius", [0, -1, float("nan"), 33])
def test_unbounded_hole_radius_rejected(radius):
    with pytest.raises(ValueError, match="hole radius"):
        close_boundary(_open_cube(), radius)


def test_closed_mesh_needs_no_caps_and_reports_intersection_limit():
    mesh = o3d.geometry.TriangleMesh.create_box()
    _, cap, metrics = close_boundary(mesh, 2.)
    assert not cap.any()
    assert metrics["after"]["closed_two_face_edge_incidence"]
    assert metrics["after"]["self_intersections"].startswith("not_assessed")


def test_topology_separates_boundary_nonmanifold_and_winding():
    mesh = o3d.geometry.TriangleMesh.create_box()
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles).copy()
    faces[0] = faces[0, ::-1]
    metrics = edge_topology(vertices, faces)
    assert metrics["boundary_edge_count"] == 0
    assert metrics["inconsistent_winding_edge_count"] == 3
    metrics = edge_topology(vertices, np.concatenate([faces, faces[:1]]))
    assert metrics["nonmanifold_edge_count"] == 3


def test_nonmanifold_envelope_cannot_be_silently_capped():
    mesh = o3d.geometry.TriangleMesh.create_box()
    mesh.triangles = o3d.utility.Vector3iVector(np.concatenate([np.asarray(mesh.triangles), np.asarray(mesh.triangles)[:1]]))
    with pytest.raises(ValueError, match="nonmanifold"):
        close_boundary(mesh, 2.)


def test_orientable_envelope_repairs_winding_without_losing_surface():
    mesh = o3d.geometry.TriangleMesh.create_box()
    faces = np.asarray(mesh.triangles).copy()
    faces[0] = faces[0, ::-1]
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    repaired, metrics = orientable_envelope(mesh)
    assert metrics["removed_extraction_defect_triangle_count"] == 0
    assert len(repaired.triangles) == len(faces)
    assert edge_topology(np.asarray(repaired.vertices), np.asarray(repaired.triangles))["inconsistent_winding_edge_count"] == 0


def test_oriented_cap_fan_uses_all_boundary_edges_and_preserves_source_faces():
    source = _open_cube()
    completed, loops = _oriented_boundary_caps(source, 2.)
    assert loops == 1
    assert len(completed.vertices) == len(source.vertices) + 1
    assert len(completed.triangles) == len(source.triangles) + 4
    assert np.array_equal(np.asarray(source.triangles), np.asarray(completed.triangles)[:10])
    assert np.array_equal(np.asarray(source.vertices), np.asarray(completed.vertices)[:8])
    metrics = edge_topology(np.asarray(completed.vertices), np.asarray(completed.triangles))
    assert metrics["closed_two_face_edge_incidence"]
    assert metrics["inconsistent_winding_edge_count"] == 0
    assert metrics["degenerate_triangle_count"] == 0


def test_oriented_cap_radius_cannot_hide_an_unbounded_loop():
    with pytest.raises(ValueError, match="cap radius"):
        _oriented_boundary_caps(_open_cube(), .01)
