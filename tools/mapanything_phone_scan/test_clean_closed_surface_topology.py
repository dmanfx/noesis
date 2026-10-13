"""Independent small-graph checks for closed component/fan topology cleanup."""
from __future__ import annotations

from collections import defaultdict
import json

import numpy as np
import pytest

from tools.mapanything_phone_scan import clean_closed_surface_topology as cleanup


TETRA = np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int32)


def _arrays(vertices, faces, classification=None):
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    colors = np.arange(len(vertices) * 4, dtype=np.uint8).reshape(-1, 4)
    if classification is None:
        classification = (np.arange(len(faces)) % 3).astype(np.uint8)
    return vertices, faces, colors, np.asarray(classification, dtype=np.uint8)


def _pinched_tetrahedra():
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                [-2, 0, 0], [0, -2, 0], [0, 0, -2]]
    other = np.asarray([0, 4, 5, 6])[TETRA]
    return _arrays(vertices, np.vstack((TETRA, other)))


def _same_component_pinched_prism():
    # One shared centre has two disconnected cap fans, but the side wall joins
    # their faces into ONE global edge-connected component. Global component
    # IDs alone therefore cannot repair this pinched indexed vertex.
    theta = np.arange(3) * (2 * np.pi / 3)
    ring = np.column_stack((np.cos(theta), np.sin(theta)))
    vertices = np.vstack(([0, 0, 0], np.column_stack((ring, np.ones(3))),
                          np.column_stack((ring, -np.ones(3)))))
    faces = []
    for index in range(3):
        top, next_top = 1 + index, 1 + (index + 1) % 3
        bottom, next_bottom = 4 + index, 4 + (index + 1) % 3
        faces.extend([[top, next_top, 0], [bottom, 0, next_bottom],
                      [next_top, top, bottom], [next_top, bottom, next_bottom]])
    return _arrays(vertices, faces)


def _tetra_and_opposed_bubble():
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                [.2, .3, -1], [-.2, .4, -1]]
    return _arrays(vertices, np.vstack((TETRA, [[0, 4, 5], [5, 4, 0]])), [0, 0, 0, 0, 2, 1])


def _independent_topology(faces):
    edge_faces = defaultdict(list)
    incident = defaultdict(set)
    for index, face in enumerate(faces):
        for vertex in face:
            incident[int(vertex)].add(index)
        for a, b in zip(face, np.roll(face, -1)):
            edge_faces[tuple(sorted((int(a), int(b))))].append((index, int(a), int(b)))
    assert all(len(owners) == 2 for owners in edge_faces.values())
    assert all(owners[0][1:] == owners[1][1:][::-1] for owners in edge_faces.values())
    peers = defaultdict(set)
    vertex_peers = defaultdict(lambda: defaultdict(set))
    for (a, b), owners in edge_faces.items():
        left, right = owners[0][0], owners[1][0]
        peers[left].add(right)
        peers[right].add(left)
        for vertex in (a, b):
            vertex_peers[vertex][left].add(right)
            vertex_peers[vertex][right].add(left)
    def count_components(nodes, adjacency):
        unseen = set(nodes)
        count = 0
        while unseen:
            count += 1
            pending = [unseen.pop()]
            while pending:
                for other in adjacency[pending.pop()]:
                    if other in unseen:
                        unseen.remove(other)
                        pending.append(other)
        return count
    global_count = count_components(range(len(faces)), peers)
    fans = {vertex: count_components(nodes, vertex_peers[vertex]) for vertex, nodes in incident.items()}
    return global_count, fans


def _assert_exact_retained_arrays(inputs, result):
    vertices, faces, colors, classification = inputs
    assert np.array_equal(result.vertices[result.faces], vertices[faces[result.retained_face_indices]])
    assert np.array_equal(result.colors[result.faces], colors[faces[result.retained_face_indices]])
    assert np.array_equal(result.classification, classification[result.retained_face_indices])
    assert np.array_equal(result.vertex_source_indices[result.faces], faces[result.retained_face_indices])
    _, fans = _independent_topology(result.faces)
    assert all(count == 1 for count in fans.values())
    assert result.vertices.dtype == vertices.dtype and result.colors.dtype == colors.dtype
    assert result.classification.dtype == classification.dtype
    json.dumps(result.report, allow_nan=False)


def test_pinched_closed_tetrahedra_split_local_vertex_fans_without_motion():
    inputs = _pinched_tetrahedra()
    components, fans = _independent_topology(inputs[1])
    assert components == 2 and fans[0] == 2
    result = cleanup.clean_closed_surface_topology(*inputs, minimum_component_faces=4)
    _assert_exact_retained_arrays(inputs, result)
    assert len(result.vertices) == 8 and len(result.faces) == 8
    assert result.report["fan_split"]["retained_source_vertices_with_disconnected_fans"] == 1
    assert result.report["fan_split"]["added_fan_vertex_copies"] == 1
    assert result.report["discarded"]["triangle_count"] == 0
    assert _independent_topology(result.faces)[0] == 2


def test_same_global_component_pinches_split_by_local_fan_not_component_id():
    inputs = _same_component_pinched_prism()
    components, fans = _independent_topology(inputs[1])
    assert components == 1 and fans[0] == 2
    result = cleanup.clean_closed_surface_topology(*inputs, minimum_component_faces=4)
    _assert_exact_retained_arrays(inputs, result)
    assert _independent_topology(result.faces)[0] == 1
    assert len(result.vertices) == len(inputs[0]) + 1
    assert result.report["fan_split"]["retained_source_vertices_with_disconnected_fans"] == 1


def test_remove_only_complete_closed_two_face_bubble_and_report_area_classes():
    inputs = _tetra_and_opposed_bubble()
    result = cleanup.clean_closed_surface_topology(*inputs, minimum_component_faces=4)
    _assert_exact_retained_arrays(inputs, result)
    np.testing.assert_array_equal(result.retained_face_indices, np.arange(4))
    assert len(result.faces) == 4 and len(result.vertices) == 4
    xyz = inputs[0][inputs[1][4:]].astype(np.float64)
    area = .5 * np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
    discarded = result.report["discarded"]
    assert discarded["closed_component_count"] == 1 and discarded["triangle_count"] == 2
    assert discarded["component_face_count_histogram"] == {"2": 1}
    assert discarded["class_counts"] == {"1": 1, "2": 1}
    np.testing.assert_allclose(discarded["surface_area_m2"], area.sum())
    np.testing.assert_allclose(discarded["class_area_m2"]["1"], area[1])
    assert result.report["output"]["class_counts"] == {"0": 4}


def test_threshold_is_exclusive_and_zero_disables_removal_not_fan_split():
    inputs = _tetra_and_opposed_bubble()
    for threshold in (0, 2):
        result = cleanup.clean_closed_surface_topology(*inputs, minimum_component_faces=threshold)
        _assert_exact_retained_arrays(inputs, result)
        assert result.report["discarded"]["triangle_count"] == 0
        assert len(result.faces) == 6 and len(result.vertices) == 7
    with pytest.raises(ValueError, match="no usable surface"):
        cleanup.clean_closed_surface_topology(*inputs, minimum_component_faces=5)


def test_closed_single_component_keeps_triangle_order_colors_classes_and_inputs():
    inputs = _arrays([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], TETRA)
    originals = [value.copy() for value in inputs]
    first = cleanup.clean_closed_surface_topology(*inputs, minimum_component_faces=4)
    second = cleanup.clean_closed_surface_topology(*inputs, minimum_component_faces=4)
    _assert_exact_retained_arrays(inputs, first)
    for name in ("vertices", "faces", "colors", "classification", "retained_face_indices", "vertex_source_indices"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    assert first.report == second.report
    for actual, original in zip(inputs, originals):
        np.testing.assert_array_equal(actual, original)


def test_unsigned_wide_classification_is_preserved_without_aggregation_cast_failure():
    inputs = list(_pinched_tetrahedra())
    inputs[3] = inputs[3].astype(np.uint64)
    result = cleanup.clean_closed_surface_topology(*inputs, minimum_component_faces=4)
    _assert_exact_retained_arrays(inputs, result)


def test_open_small_component_is_not_silently_discarded():
    inputs = _arrays([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                      [2, 0, 0], [2, 1, 0], [2, 0, 1]], np.vstack((TETRA, [[4, 5, 6]])))
    with pytest.raises(ValueError, match="closed edge"):
        cleanup.clean_closed_surface_topology(*inputs, minimum_component_faces=4)


def test_wrong_winding_and_nonmanifold_edges_fail_closed():
    inputs = list(_arrays([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], TETRA.copy()))
    inputs[1][0] = inputs[1][0, ::-1]
    with pytest.raises(ValueError, match="oppositely wound"):
        cleanup.clean_closed_surface_topology(*inputs, minimum_component_faces=4)
    bad = _arrays(inputs[0], np.vstack((TETRA, TETRA)))
    with pytest.raises(ValueError, match="two faces"):
        cleanup.clean_closed_surface_topology(*bad, minimum_component_faces=4)


@pytest.mark.parametrize("threshold", [-1, 3_000_001, 1.5])
def test_component_threshold_bound_fails_closed(threshold):
    with pytest.raises(ValueError, match="threshold"):
        cleanup.clean_closed_surface_topology(*_pinched_tetrahedra(), minimum_component_faces=threshold)


@pytest.mark.parametrize("malformation", ["nonfinite_vertex", "wrong_color_count", "wrong_class_count", "invalid_class", "degenerate"])
def test_provenance_and_geometry_arrays_fail_closed(malformation):
    inputs = list(_pinched_tetrahedra())
    if malformation == "nonfinite_vertex":
        inputs[0][0, 0] = np.nan
    elif malformation == "wrong_color_count":
        inputs[2] = inputs[2][:-1]
    elif malformation == "wrong_class_count":
        inputs[3] = inputs[3][:-1]
    elif malformation == "invalid_class":
        inputs[3] = inputs[3].astype(np.int32)
        inputs[3][0] = -1
    else:
        inputs[0][1] = inputs[0][0]
    with pytest.raises(ValueError):
        cleanup.clean_closed_surface_topology(*inputs, minimum_component_faces=4)


def test_input_and_post_split_vertex_work_caps_fail_closed(monkeypatch):
    inputs = _pinched_tetrahedra()
    monkeypatch.setattr(cleanup, "MAX_VERTICES", 7)
    with pytest.raises(ValueError, match="output vertex bound"):
        cleanup.clean_closed_surface_topology(*inputs, minimum_component_faces=4)
    monkeypatch.setattr(cleanup, "MAX_FACES", 1)
    with pytest.raises(ValueError, match="work bound"):
        cleanup.clean_closed_surface_topology(*inputs, minimum_component_faces=0)
