"""CPU-only, physically weighted source selection for a measured surface.

This module never projects, blends, invents candidates, or changes geometry.
All face labels remain in the caller's original valid photograph candidates.
Region updates are bounded, deterministic, and strictly decrease the physical
area/length objective. Neutral faces are immutable barriers, not filled gaps.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


MAX_FACES = 3_000_000
MAX_VERTICES = 2_000_000
MAX_SOURCES = 512
CHUNK_FACES = 8192


def _geometry(vertices, faces, neighbors, dihedral_weights, *, max_faces=None, max_vertices=None):
    max_faces = MAX_FACES if max_faces is None else max_faces
    max_vertices = MAX_VERTICES if max_vertices is None else max_vertices
    if (isinstance(max_faces, bool) or not isinstance(max_faces, (int, np.integer)) or not 1 <= max_faces <= 5_000_000
            or isinstance(max_vertices, bool) or not isinstance(max_vertices, (int, np.integer)) or not 1 <= max_vertices <= 3_000_000):
        raise ValueError("physical surface graph work bounds exceed supported limits")
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    neighbors, weights = np.asarray(neighbors), np.asarray(dihedral_weights)
    count = len(faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) > max_vertices
            or faces.ndim != 2 or faces.shape[1] != 3 or count > max_faces
            or not np.issubdtype(faces.dtype, np.integer) or not np.isfinite(vertices).all()
            or (faces.size and (faces.min() < 0 or faces.max() >= len(vertices)))
            or neighbors.shape != (count, 3) or weights.shape != (count, 3)
            or not np.issubdtype(neighbors.dtype, np.integer)
            or not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1))
            or np.any((neighbors < -1) | (neighbors >= count))):
        raise ValueError("malformed or unbounded physical surface graph")
    area = np.empty(count, dtype=np.float64)
    boundary = np.empty((count, 3), dtype=np.float64)
    # Slots must correspond to the caller's edges (0,1), (1,2), (2,0).
    for start in range(0, count, CHUNK_FACES):
        rows = np.arange(start, min(count, start + CHUNK_FACES))
        corners = np.asarray(vertices[faces[rows]], dtype=np.float64)
        area[rows] = .5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0],
                                                 corners[:, 2] - corners[:, 0]), axis=1)
        boundary[rows] = np.linalg.norm(corners - corners[:, [1, 2, 0]], axis=2) * weights[rows]
        local_rows, slots = np.nonzero(neighbors[rows] >= 0)
        current = rows[local_rows]
        other = neighbors[current, slots]
        edge = np.sort(faces[current][:, [[0, 1], [1, 2], [2, 0]]][np.arange(len(current)), slots], axis=1)
        other_edges = np.sort(faces[other][:, [[0, 1], [1, 2], [2, 0]]], axis=2)
        # A valid triangle dual is a multigraph: two opposed triangles can
        # share all three edges. Reciprocity belongs to the exact edge slot,
        # not merely to a face pair, which may occur in several slots.
        reciprocal = ((neighbors[other] == current[:, None])
                      & (other_edges == edge[:, None, :]).all(2))
        if np.any(current == other) or np.any(reciprocal.sum(1) != 1):
            raise ValueError("surface adjacency must bind one distinct reciprocal shared edge")
        reverse = reciprocal.argmax(1)
        other_edge = other_edges[np.arange(len(other)), reverse]
        if (not np.array_equal(edge, other_edge)
                or not np.allclose(weights[current, slots], weights[other, reverse], atol=1e-7, rtol=0)):
            raise ValueError("surface adjacency must bind the same weighted shared edge")
    rows, slots = np.nonzero((neighbors >= 0) & (np.arange(count)[:, None] < neighbors))
    return (faces, neighbors, area, boundary, rows.astype(np.int32),
            neighbors[rows, slots].astype(np.int32), boundary[rows, slots])


def _regions(labels, left, right):
    same = (labels[left] >= 0) & (labels[left] == labels[right])
    graph = sparse.csr_matrix((np.ones(int(same.sum()), dtype=np.uint8),
                               (left[same], right[same])), shape=(len(labels), len(labels)))
    return connected_components(graph, directed=False)


def _objective(labels, candidates, unary, area, left, right, length, smoothness):
    value = 0.0
    for start in range(0, len(labels), CHUNK_FACES):
        rows = np.arange(start, min(len(labels), start + CHUNK_FACES))
        rows = rows[labels[rows] >= 0]
        slot = (candidates[rows] == labels[rows, None]).argmax(1)
        value += float(np.dot(area[rows], unary[rows, slot].astype(np.float64)))
    different = ((labels[left] >= 0) & (labels[right] >= 0) & (labels[left] != labels[right]))
    boundary = float(length[different].sum(dtype=np.float64))
    return value + smoothness * boundary, boundary


def _region_costs(rows, region, components, labels, candidates, unary, area,
                  neighbors, boundary, source_lut, source_count):
    counts = np.zeros(source_count, dtype=np.int64)
    costs = np.zeros(source_count, dtype=np.float64)
    agreement = np.zeros(source_count, dtype=np.float64)
    total_boundary = 0.0
    for start in range(0, len(rows), CHUNK_FACES):
        part = rows[start:start + CHUNK_FACES]
        choices = candidates[part]
        valid = choices >= 0
        mapped = source_lut[choices[valid]]
        counts += np.bincount(mapped, minlength=source_count)
        data = unary[part][valid].astype(np.float64) * np.broadcast_to(area[part, None], choices.shape)[valid]
        costs += np.bincount(mapped, weights=data, minlength=source_count)
        adjacent = neighbors[part]
        safe = np.maximum(adjacent, 0)
        external = ((adjacent >= 0) & (components[safe] != region) & (labels[safe] >= 0))
        lengths = boundary[part][external]
        agreement += np.bincount(source_lut[labels[safe][external]], weights=lengths, minlength=source_count)
        total_boundary += float(lengths.sum(dtype=np.float64))
    return counts, costs, agreement, total_boundary


def coherent_labels(
    vertices: np.ndarray, faces: np.ndarray, candidates: np.ndarray, scores: np.ndarray,
    neighbors: np.ndarray, dihedral_weights: np.ndarray, *, smoothness_m: float = .05,
    iterations: int = 8, minimum_patch_area_m2: float | None = .01,
    max_faces: int | None = None, max_vertices: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select only original valid sources using decreasing whole-region moves.

    Energy = sum(area_m2 * relative_log_score) + smoothness_m *
    sum(shared_edge_length_m * dihedral_weight * source_disagreement).
    No boundary penalty crosses neutral faces. The optional minimum patch area
    sets small-region-first priority and diagnostics; it does not force a merge
    or exempt larger regions from energy-decreasing optimization.

    Components are frozen within each pass to keep work linear in faces times
    at most 12 candidates. Every region move evaluates CURRENT boundary labels;
    accepted merges are incorporated into recomputed components next pass.
    """
    if (not isinstance(iterations, (int, np.integer)) or not 1 <= iterations <= 20
            or not math.isfinite(smoothness_m) or not 0 <= smoothness_m <= 10
            or (minimum_patch_area_m2 is not None
                and (not math.isfinite(minimum_patch_area_m2) or not 0 <= minimum_patch_area_m2 <= 100))):
        raise ValueError("unbounded physical source-label parameters")
    candidates, scores = np.asarray(candidates), np.asarray(scores)
    if (candidates.ndim != 2 or candidates.shape != scores.shape or len(candidates) != len(faces)
            or not 1 <= candidates.shape[1] <= 12 or not np.issubdtype(candidates.dtype, np.integer)
            or not np.isfinite(scores).all() or np.any(scores < 0)
            or np.any((candidates < -1) | (candidates > 32767))
            or np.any((candidates >= 0) != (scores > 0))):
        raise ValueError("malformed original valid photo candidates")
    for start in range(0, len(candidates), CHUNK_FACES):
        ordered = np.sort(candidates[start:start + CHUNK_FACES], axis=1)
        if np.any((ordered[:, 1:] >= 0) & (ordered[:, 1:] == ordered[:, :-1])):
            raise ValueError("duplicate original photo candidates")
    faces, neighbors, area, boundary, left, right, edge_length = _geometry(
        vertices, faces, neighbors, dihedral_weights, max_faces=max_faces, max_vertices=max_vertices,
    )
    sources = np.unique(candidates[candidates >= 0])
    if len(sources) > MAX_SOURCES:
        raise ValueError("source photograph count exceeds finite work bound")
    best = scores.max(1)
    labels = np.full(len(faces), -1, dtype=np.int16)
    eligible = best > 0
    if np.any((area <= 0) & eligible):
        raise ValueError("a degenerate face cannot have a valid photograph")
    labels[eligible] = candidates[np.flatnonzero(eligible), scores[eligible].argmax(1)]
    unary = -np.log(np.maximum(scores / np.maximum(best[:, None], 1e-30), 1e-8)).astype(np.float32)
    unary[candidates < 0] = np.inf
    source_lut = np.full(int(sources.max()) + 1 if len(sources) else 0, -1, dtype=np.int16)
    source_lut[sources] = np.arange(len(sources), dtype=np.int16)
    before = labels.copy()
    initial_energy, initial_boundary = _objective(labels, candidates, unary, area, left, right, edge_length, smoothness_m)
    threshold = 0.0 if minimum_patch_area_m2 is None else minimum_patch_area_m2
    history = []
    first_stats = None
    move_count = 0
    relabeled_faces = 0
    termination = "iteration_bound"
    for pass_index in range(iterations):
        count, components = _regions(labels, left, right)
        sizes = np.bincount(components, minlength=count)
        patch_area = np.bincount(components, weights=area, minlength=count)
        active = np.unique(components[labels >= 0])
        if first_stats is None:
            first_stats = {"photo_patch_count": int(len(active)),
                           "small_photo_patch_count": int(np.count_nonzero(patch_area[active] < threshold))}
        order = np.argsort(components, kind="stable")
        ends = np.cumsum(sizes)
        starts = ends - sizes
        # Small physical regions first, then deterministic area/component order.
        regions = active[np.lexsort((active, patch_area[active], patch_area[active] >= threshold))]
        pass_moves = 0
        accepted_delta = 0.0
        energy_before, _ = _objective(labels, candidates, unary, area, left, right, edge_length, smoothness_m)
        for region in regions:
            rows = order[starts[region]:ends[region]]
            old_source = int(labels[rows[0]])
            counts, costs, agreement, total = _region_costs(
                rows, region, components, labels, candidates, unary, area,
                neighbors, boundary, source_lut, len(sources),
            )
            common = np.flatnonzero(counts == len(rows))
            old = int(source_lut[old_source])
            delta = costs[common] - costs[old] + smoothness_m * (agreement[old] - agreement[common])
            if not len(common):
                raise ValueError("current region lost its original valid source")
            chosen = int(common[delta.argmin()])
            decrease = float(delta.min())
            numerical_bound = 32 * np.finfo(np.float64).eps * max(
                float(costs[old]), smoothness_m * total, np.finfo(np.float64).tiny,
            )
            if decrease < -numerical_bound:
                labels[rows] = sources[chosen]
                accepted_delta += decrease
                relabeled_faces += len(rows)
                pass_moves += 1
        energy_after, _ = _objective(labels, candidates, unary, area, left, right, edge_length, smoothness_m)
        numerical_bound = 128 * np.finfo(np.float64).eps * max(energy_before, np.finfo(np.float64).tiny)
        if energy_after > energy_before + numerical_bound:
            raise ValueError("physical source-label objective increased")
        history.append({"pass": pass_index + 1, "region_count": int(len(active)), "accepted_region_moves": pass_moves,
                        "energy_before_m2": energy_before, "energy_after_m2": energy_after,
                        "sum_accepted_delta_m2": accepted_delta})
        move_count += pass_moves
        if not pass_moves:
            termination = "no_decreasing_valid_region_move"
            break
    count, components = _regions(labels, left, right)
    patch_area = np.bincount(components, weights=area, minlength=count)
    active = np.unique(components[labels >= 0])
    final_energy, final_boundary = _objective(labels, candidates, unary, area, left, right, edge_length, smoothness_m)
    if np.any(labels[~eligible] != -1):
        raise ValueError("neutral photo faces changed")
    for start in range(0, len(labels), CHUNK_FACES):
        part = slice(start, min(len(labels), start + CHUNK_FACES))
        if not np.all((labels[part] < 0) | (candidates[part] == labels[part, None]).any(1)):
            raise ValueError("selected photo was not an original valid candidate")
    metrics = {
        "method": "physical_area_edge_length_decreasing_valid_region_moves",
        "energy_units": "m2", "smoothness_m": float(smoothness_m),
        "minimum_patch_area_m2": None if minimum_patch_area_m2 is None else float(minimum_patch_area_m2),
        "minimum_patch_area_policy": "small_region_priority_and_diagnostics_not_forced_merge",
        "before": {**first_stats, "energy_m2": initial_energy, "weighted_boundary_length_m": initial_boundary},
        "after": {"photo_patch_count": int(len(active)),
                  "small_photo_patch_count": int(np.count_nonzero(patch_area[active] < threshold)),
                  "energy_m2": final_energy, "weighted_boundary_length_m": final_boundary,
                  "patch_area_m2_p0_p50_p100": np.percentile(patch_area[active], [0, 50, 100]).tolist() if len(active) else []},
        "iterations": len(history), "iteration_bound": int(iterations), "termination": termination,
        "accepted_region_moves": move_count, "relabeled_face_updates": relabeled_faces,
        "changed_face_count": int(np.count_nonzero(labels != before)),
        "photo_face_count": int(np.count_nonzero(labels >= 0)), "neutral_face_count": int(np.count_nonzero(labels < 0)),
        "photo_area_m2": float(area[labels >= 0].sum()), "neutral_area_m2": float(area[labels < 0].sum()),
        "candidate_validity_preserved": True, "neutral_labels_preserved": True,
        "colour_blending": False, "candidate_propagation": False,
        "component_recompute": "each_pass_current_boundary_labels_for_each_move",
        "energy_history": history,
        "work_bounds": {"max_faces": MAX_FACES if max_faces is None else int(max_faces),
                        "max_vertices": MAX_VERTICES if max_vertices is None else int(max_vertices),
                        "max_sources": MAX_SOURCES, "max_candidates_per_face": 12, "chunk_faces": CHUNK_FACES},
    }
    return labels, metrics
