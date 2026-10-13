"""Bounded CPU topology cleanup of an already closed review presentation mesh.

Only complete small CLOSED edge-connected components are discarded. Retained
triangle corner positions, winding, colors, and classifications remain exact.
Disconnected local one-ring fans get separate vertex identities, including
fans belonging to the same global surface component. No geometric correction,
triangulation, caps, inference, color averaging, or self-intersection test occurs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


MAX_FACES = 3_000_000
MAX_VERTICES = 2_000_000
CHUNK_FACES = 8192
EDGE_SLOTS = np.asarray([[0, 1], [1, 2], [2, 0]])


@dataclass(frozen=True)
class CleanedSurface:
    vertices: np.ndarray
    faces: np.ndarray
    colors: np.ndarray
    classification: np.ndarray
    retained_face_indices: np.ndarray
    vertex_source_indices: np.ndarray
    report: dict[str, Any]


def _closed_edge_pairs(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return paired half-edge offsets; parallel face-pair edges remain separate."""
    directed = faces[:, EDGE_SLOTS].reshape(-1, 2)
    undirected = np.sort(directed, axis=1)
    order = np.lexsort((undirected[:, 1], undirected[:, 0]))
    ordered = undirected[order]
    if (len(order) % 2 or np.any(ordered[::2] != ordered[1::2])
            or np.any(np.all(ordered[1:-1:2] == ordered[2::2], axis=1))):
        raise ValueError("input must have exactly two faces per closed edge")
    first, second = order[::2].astype(np.int32), order[1::2].astype(np.int32)
    if (np.any(first // 3 == second // 3)
            or not np.array_equal(directed[first], directed[second][:, ::-1])):
        raise ValueError("closed edges must have distinct, oppositely wound incident faces")
    return first, second


def _component_labels(face_count: int, first: np.ndarray, second: np.ndarray):
    graph = sparse.csr_matrix((np.ones(len(first), dtype=np.uint8),
                               (first // 3, second // 3)), shape=(face_count, face_count))
    return connected_components(graph, directed=False)


def _local_corner_fans(faces: np.ndarray, first: np.ndarray, second: np.ndarray, max_vertices=MAX_VERTICES):
    """Each graph node is one triangle corner, NOT a global surface component."""
    first_end = first // 3 * 3 + (first % 3 + 1) % 3
    second_end = second // 3 * 3 + (second % 3 + 1) % 3
    left = np.concatenate((first, first_end))
    right = np.concatenate((second_end, second))
    graph = sparse.csr_matrix((np.ones(len(left), dtype=np.uint8), (left, right)),
                              shape=(faces.size, faces.size))
    count, corner_fan = connected_components(graph, directed=False)
    if count > max_vertices:
        raise ValueError("local fan splitting exceeds the output vertex bound")
    source_vertex = np.empty(count, dtype=np.int32)
    source_vertex[corner_fan] = faces.ravel()
    # Every graph link must connect the SAME original indexed vertex. This also
    # proves remapped corners preserve coordinates without large RGB/XYZ copies.
    for start in range(0, faces.size, 3 * CHUNK_FACES):
        stop = min(faces.size, start + 3 * CHUNK_FACES)
        if not np.array_equal(source_vertex[corner_fan[start:stop]], faces.ravel()[start:stop]):
            raise ValueError("a local fan joined different original vertex identities")
    return corner_fan.reshape(-1, 3).astype(np.int32), source_vertex


def _class_report(classes: np.ndarray, area: np.ndarray) -> dict[str, Any]:
    # Validation already bounds IDs to 0..255; uint64 inputs otherwise make
    # numpy.bincount reject its safe cast even though these values fit exactly.
    classes = classes.astype(np.int32, copy=False)
    counts = np.bincount(classes, minlength=256)
    summed_area = np.bincount(classes, weights=area, minlength=256)
    unique = np.flatnonzero(counts)
    return {
        "class_counts": {str(int(value)): int(counts[value]) for value in unique},
        "class_area_m2": {str(int(value)): float(summed_area[value]) for value in unique},
    }


def clean_closed_surface_topology(
    vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray, classification: np.ndarray,
    *, minimum_component_faces: int = 64, max_faces: int | None = None,
    max_vertices: int | None = None,
) -> CleanedSurface:
    """Remove complete small closed components, then split retained local fans.

    Threshold is exclusive: a component with exactly the requested number of
    faces is kept. Zero disables component removal, not fan splitting. An empty
    result, malformed provenance arrays, open/nonmanifold edges, inconsistent
    winding, or finite work-limit violation fails closed. Class IDs are retained
    integer values; the caller owns their names and publication disposition.
    """
    max_faces = MAX_FACES if max_faces is None else max_faces
    max_vertices = MAX_VERTICES if max_vertices is None else max_vertices
    if (isinstance(max_faces, bool) or not isinstance(max_faces, (int, np.integer))
            or not 1 <= max_faces <= 5_000_000
            or isinstance(max_vertices, bool) or not isinstance(max_vertices, (int, np.integer))
            or not 1 <= max_vertices <= 3_000_000):
        raise ValueError("explicit mesh work bounds exceed supported limits")
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    colors, classification = np.asarray(colors), np.asarray(classification)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not 1 <= len(vertices) <= max_vertices
            or faces.ndim != 2 or faces.shape[1] != 3 or not 1 <= len(faces) <= max_faces
            or not np.issubdtype(faces.dtype, np.integer) or not np.isfinite(vertices).all()
            or faces.min() < 0 or faces.max() >= len(vertices)
            or colors.ndim != 2 or colors.shape[0] != len(vertices) or colors.shape[1] not in (3, 4)
            or not np.isfinite(colors).all() or classification.shape != (len(faces),)
            or not np.issubdtype(classification.dtype, np.integer)
            or np.any((classification < 0) | (classification > 255))):
        raise ValueError("malformed source arrays or finite mesh work bound exceeded")
    if (not isinstance(minimum_component_faces, (int, np.integer))
            or not 0 <= minimum_component_faces <= max_faces):
        raise ValueError("component threshold exceeds the finite face bound")
    indexed_faces = faces.astype(np.int32, copy=False)
    area = np.empty(len(faces), dtype=np.float64)
    for start in range(0, len(faces), CHUNK_FACES):
        stop = min(len(faces), start + CHUNK_FACES)
        corners = np.asarray(vertices[indexed_faces[start:stop]], dtype=np.float64)
        twice = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
        if np.any(twice <= 1e-12):
            raise ValueError("degenerate source triangle is not a component-removal repair")
        area[start:stop] = twice * .5
    first, second = _closed_edge_pairs(indexed_faces)
    component_count, component = _component_labels(len(faces), first, second)
    sizes = np.bincount(component, minlength=component_count)
    discard_components = sizes < minimum_component_faces
    keep = ~discard_components[component]
    retained_indices = np.flatnonzero(keep).astype(np.uint32)
    if not len(retained_indices):
        raise ValueError("small closed component removal would leave no usable surface")
    # Whole edge components imply every retained half edge keeps its peer.
    edge_keep = keep[first // 3]
    if not np.array_equal(edge_keep, keep[second // 3]):
        raise ValueError("component selection removed only one side of an edge")
    old_face_to_new = np.full(len(faces), -1, dtype=np.int32)
    old_face_to_new[retained_indices] = np.arange(len(retained_indices), dtype=np.int32)
    retained_faces = indexed_faces[retained_indices]
    first = first[edge_keep]
    second = second[edge_keep]
    first = 3 * old_face_to_new[first // 3] + first % 3
    second = 3 * old_face_to_new[second // 3] + second % 3
    output_faces, vertex_sources = _local_corner_fans(retained_faces, first, second, max_vertices)
    fan_counts = np.bincount(vertex_sources, minlength=len(vertices))
    # Independent serialized-index-compatible edge check after splitting. Local
    # fan connectivity is guaranteed by its component-to-vertex construction.
    output_first, output_second = _closed_edge_pairs(output_faces)
    output_component_count, _ = _component_labels(len(output_faces), output_first, output_second)
    expected_components = int(np.count_nonzero(~discard_components))
    if output_component_count != expected_components:
        raise ValueError("local fan splitting changed retained edge-component connectivity")
    used_source_vertices = int(np.count_nonzero(fan_counts))
    dropped_sizes = sizes[discard_components]
    unique_size, frequency = np.unique(dropped_sizes, return_counts=True)
    report = {
        "method": "complete_closed_small_component_removal_then_local_corner_fan_vertex_split",
        "minimum_component_faces": int(minimum_component_faces),
        "threshold_policy": "remove_whole_closed_components_strictly_below_threshold",
        "input": {"vertex_count": len(vertices), "triangle_count": len(faces),
                  "edge_connected_component_count": int(component_count),
                  "closed_two_face_edge_incidence": True, "consistent_edge_winding": True,
                  "surface_area_m2": float(area.sum()), **_class_report(classification, area)},
        "discarded": {"closed_component_count": int(np.count_nonzero(discard_components)),
                      "triangle_count": int(np.count_nonzero(~keep)),
                      "surface_area_m2": float(area[~keep].sum()),
                      "component_face_count_histogram": {str(int(size)): int(count) for size, count in zip(unique_size, frequency)},
                      **_class_report(classification[~keep], area[~keep])},
        "fan_split": {"policy": "per_local_triangle_corner_fan_not_global_component_id",
                      "retained_source_vertices_with_disconnected_fans": int(np.count_nonzero(fan_counts > 1)),
                      "added_fan_vertex_copies": len(vertex_sources) - used_source_vertices,
                      "source_vertices_unreferenced_after_component_removal": len(vertices) - used_source_vertices,
                      "max_retained_fans_per_source_vertex": int(fan_counts.max())},
        "output": {"vertex_count": len(vertex_sources), "triangle_count": len(retained_indices),
                   "edge_connected_component_count": int(output_component_count),
                   "closed_two_face_edge_incidence": True, "consistent_edge_winding": True,
                   "vertices_with_disconnected_one_ring_fans": 0,
                   "vertex_fan_verification": "one_output_vertex_per_verified_local_corner_connected_component",
                   "surface_area_m2": float(area[keep].sum()),
                   **_class_report(classification[keep], area[keep])},
        "retained_corner_coordinates_unchanged": True, "retained_winding_unchanged": True,
        "retained_colors_unchanged": True, "retained_classification_unchanged": True,
        "provider_confidence_or_geometry_gates_changed": False,
        "new_geometry_created": False, "self_intersections": "not_assessed",
        "physical_completeness_or_accuracy_proven": False,
        "work_bounds": {"max_faces": int(max_faces), "max_vertices": int(max_vertices), "chunk_faces": CHUNK_FACES},
    }
    return CleanedSurface(vertices[vertex_sources].copy(), output_faces, colors[vertex_sources].copy(),
                          classification[retained_indices].copy(), retained_indices, vertex_sources, report)
