"""Physical, bounded label moves preserve original photo support and neutrality."""
from __future__ import annotations

import json

import numpy as np
import pytest

from tools.mapanything_phone_scan import coherent_surface_photo_labels as selection


def _grid(cells=32, extent=1.0, tiles=16, ratio=.99):
    x, y = np.meshgrid(np.arange(cells + 1), np.arange(cells + 1))
    vertices = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))) * extent / cells
    first = np.arange((cells + 1) ** 2).reshape(cells + 1, cells + 1)[:-1, :-1].ravel()
    faces = np.vstack((np.column_stack((first, first + 1, first + cells + 2)),
                       np.column_stack((first, first + cells + 2, first + cells + 1))))
    centers = vertices[faces].mean(1)
    preferred = ((np.floor(centers[:, 0] / extent * tiles)
                  + np.floor(centers[:, 1] / extent * tiles)).astype(int) % 2)
    candidates = np.column_stack((preferred, 1 - preferred)).astype(np.int16)
    scores = np.tile([1., ratio], (len(faces), 1)).astype(np.float32)
    neighbors = np.full((len(faces), 3), -1, dtype=np.int32)
    weights = np.zeros(neighbors.shape, dtype=np.float32)
    edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    ordered = edges[order]
    starts = np.flatnonzero(np.r_[True, np.any(ordered[1:] != ordered[:-1], axis=1)])
    counts = np.diff(np.r_[starts, len(order)])
    shared = starts[counts == 2]
    left, right = order[shared] // 3, order[shared + 1] // 3
    a, b = order[shared] % 3, order[shared + 1] % 3
    neighbors[left, a], neighbors[right, b] = right, left
    weights[left, a], weights[right, b] = 1., 1.
    return vertices, faces, candidates, scores, neighbors, weights


@pytest.mark.parametrize("extent", [1., .1, .01])
def test_one_percent_all_valid_preferences_escape_single_face_local_minimum(extent):
    inputs = _grid(extent=extent)
    labels, metrics = selection.coherent_labels(*inputs)
    assert metrics["before"]["photo_patch_count"] == 256
    assert metrics["after"]["photo_patch_count"] == 1
    assert metrics["after"]["weighted_boundary_length_m"] == 0
    assert metrics["after"]["energy_m2"] < metrics["before"]["energy_m2"]
    assert np.all((inputs[2] == labels[:, None]).any(1))
    assert metrics["accepted_region_moves"] > 0


def test_whole_region_candidate_intersection_cannot_cross_one_invalid_face():
    inputs = list(_grid(cells=16, tiles=4))
    centers = inputs[0][inputs[1]].mean(1)
    inside = ((centers[:, 0] > .25) & (centers[:, 0] < .5)
              & (centers[:, 1] > .25) & (centers[:, 1] < .5))
    inputs[2][:] = [0, 1]
    inputs[2][inside] = [1, 0]
    inputs[3][:] = [1., .99]
    forced = np.flatnonzero(inside)[0]
    inputs[2][forced] = [1, -1]
    inputs[3][forced] = [1., 0.]
    # The connected initial view1 region cannot move to view0 because even ONE
    # affected face lacks that source. Its neighboring view0 region is fixed too.
    outside = np.flatnonzero(~inside)[0]
    inputs[2][outside] = [0, -1]
    inputs[3][outside] = [1., 0.]
    labels, metrics = selection.coherent_labels(*inputs)
    assert np.all(labels[inside] == 1)
    assert np.all(labels[~inside] == 0)
    assert metrics["accepted_region_moves"] == 0
    assert metrics["candidate_validity_preserved"]


def test_neutral_strip_is_immutable_and_not_a_boundary_bridge():
    inputs = list(_grid(cells=16, tiles=2))
    centers = inputs[0][inputs[1]].mean(1)
    neutral = (centers[:, 0] > .4375) & (centers[:, 0] < .5625)
    left = centers[:, 0] <= .4375
    right = centers[:, 0] >= .5625
    inputs[2][left], inputs[2][right], inputs[2][neutral] = [0, 1], [1, 0], [-1, -1]
    inputs[3][neutral] = 0
    labels, metrics = selection.coherent_labels(*inputs)
    assert np.all(labels[neutral] == -1)
    assert np.all(labels[left] == 0) and np.all(labels[right] == 1)
    assert metrics["neutral_face_count"] == int(neutral.sum())
    assert metrics["after"]["photo_patch_count"] == 2
    assert metrics["after"]["weighted_boundary_length_m"] == 0


def test_every_accepted_move_and_pass_decreases_full_physical_energy():
    inputs = _grid(extent=.1)
    _, metrics = selection.coherent_labels(*inputs)
    previous = metrics["before"]["energy_m2"]
    for step in metrics["energy_history"]:
        assert step["energy_before_m2"] == previous
        assert step["energy_after_m2"] <= step["energy_before_m2"]
        np.testing.assert_allclose(step["energy_after_m2"],
                                   step["energy_before_m2"] + step["sum_accepted_delta_m2"], atol=1e-14)
        if step["accepted_region_moves"]:
            assert step["sum_accepted_delta_m2"] < 0
            assert step["energy_after_m2"] < step["energy_before_m2"]
        previous = step["energy_after_m2"]
    assert metrics["after"]["energy_m2"] == previous
    json.dumps(metrics, allow_nan=False)


def test_physical_scale_changes_tradeoff_not_a_face_count_threshold():
    _, large = selection.coherent_labels(*_grid(extent=10., ratio=.1), smoothness_m=.05)
    _, tiny = selection.coherent_labels(*_grid(extent=.01, ratio=.1), smoothness_m=.05)
    assert large["after"]["photo_patch_count"] == 256
    assert tiny["after"]["photo_patch_count"] == 1
    assert large["accepted_region_moves"] == 0
    # Changing the diagnostic area threshold cannot force an unjustified merge.
    _, forced = selection.coherent_labels(*_grid(extent=10., ratio=.1),
                                          smoothness_m=.05, minimum_patch_area_m2=100.)
    assert forced["after"]["photo_patch_count"] == 256


def test_retessellation_preserves_physical_boundary_energy_for_same_preferences():
    _, coarse = selection.coherent_labels(*_grid(cells=8, tiles=4, ratio=.1))
    _, fine = selection.coherent_labels(*_grid(cells=32, tiles=4, ratio=.1))
    assert coarse["after"]["photo_patch_count"] == fine["after"]["photo_patch_count"] == 16
    assert coarse["before"]["energy_m2"] == fine["before"]["energy_m2"]
    assert coarse["after"]["energy_m2"] == fine["after"]["energy_m2"]


def test_deterministic_bounded_termination_and_no_input_mutation():
    inputs = _grid()
    originals = [value.copy() for value in inputs]
    first, a = selection.coherent_labels(*inputs, iterations=1)
    second, b = selection.coherent_labels(*inputs, iterations=1)
    np.testing.assert_array_equal(first, second)
    assert a == b and a["iterations"] == 1 and a["termination"] == "iteration_bound"
    for actual, original in zip(inputs, originals):
        np.testing.assert_array_equal(actual, original)


def test_zero_smoothing_keeps_best_valid_source_without_arbitrary_region_changes():
    inputs = _grid()
    labels, metrics = selection.coherent_labels(*inputs, smoothness_m=0)
    np.testing.assert_array_equal(labels, inputs[2][:, 0])
    assert metrics["accepted_region_moves"] == 0
    assert metrics["after"]["energy_m2"] == 0


def test_noncontiguous_source_identity_and_zero_dihedral_barrier_are_preserved():
    inputs = list(_grid(cells=8, tiles=4))
    inputs[2] = np.where(inputs[2] == 0, 47, 106).astype(np.int16)
    inputs[5][:] = 0
    labels, metrics = selection.coherent_labels(*inputs)
    np.testing.assert_array_equal(labels, inputs[2][:, 0])
    assert set(labels.tolist()) == {47, 106}
    assert metrics["accepted_region_moves"] == 0
    assert metrics["after"]["weighted_boundary_length_m"] == 0


def test_all_neutral_and_empty_graph_are_finite_json_safe():
    inputs = list(_grid(cells=2, tiles=2))
    inputs[2][:], inputs[3][:] = -1, 0
    labels, metrics = selection.coherent_labels(*inputs, minimum_patch_area_m2=None)
    assert np.all(labels == -1) and metrics["after"]["photo_patch_count"] == 0
    assert metrics["after"]["energy_m2"] == 0
    json.dumps(metrics, allow_nan=False)
    empty = [np.empty((0, 3)), np.empty((0, 3), np.int32), np.empty((0, 2), np.int16),
             np.empty((0, 2)), np.empty((0, 3), np.int32), np.empty((0, 3))]
    labels, metrics = selection.coherent_labels(*empty)
    assert labels.size == 0 and metrics["neutral_face_count"] == 0
    json.dumps(metrics, allow_nan=False)


@pytest.mark.parametrize("kwargs", [{"iterations": 0}, {"iterations": 21}, {"smoothness_m": -1},
                                    {"smoothness_m": np.nan}, {"minimum_patch_area_m2": -1},
                                    {"minimum_patch_area_m2": np.inf}])
def test_parameter_bounds_fail_closed(kwargs):
    with pytest.raises(ValueError, match="parameters"):
        selection.coherent_labels(*_grid(cells=2, tiles=2), **kwargs)


@pytest.mark.parametrize("malformation", ["duplicate_candidate", "missing_score", "nonfinite_score",
                                          "unreciprocated", "asymmetric_weight", "wrong_edge"])
def test_malformed_candidate_or_surface_binding_fails_closed(malformation):
    inputs = list(_grid(cells=2, tiles=2))
    if malformation == "duplicate_candidate":
        inputs[2][0] = [0, 0]
    elif malformation == "missing_score":
        inputs[3][0, 0] = 0
    elif malformation == "nonfinite_score":
        inputs[3][0, 0] = np.nan
    else:
        row, slot = np.argwhere(inputs[4] >= 0)[0]
        if malformation == "unreciprocated":
            inputs[4][row, slot] = -1
        elif malformation == "asymmetric_weight":
            inputs[5][row, slot] = .5
        else:
            other = inputs[4][row, slot]
            reverse = int(np.flatnonzero(inputs[4][other] == row)[0])
            inputs[4][row, slot] = -1
            inputs[4][row, (slot + 1) % 3] = other
            inputs[4][other, reverse] = row
    with pytest.raises(ValueError):
        selection.coherent_labels(*inputs)


def test_dense_face_work_cap_is_explicit(monkeypatch):
    monkeypatch.setattr(selection, "MAX_FACES", 1)
    with pytest.raises(ValueError, match="unbounded"):
        selection.coherent_labels(*_grid(cells=2, tiles=2))


def _opposed_triangle_shell():
    vertices = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    faces = np.asarray([[0, 1, 2], [2, 1, 0]], dtype=np.int32)
    candidates = np.asarray([[0, 1], [1, 0]], dtype=np.int16)
    scores = np.asarray([[1., .99], [1., .99]], dtype=np.float32)
    neighbors = np.asarray([[1, 1, 1], [0, 0, 0]], dtype=np.int32)
    # Edge identities, not face-slot numbers, determine reciprocal weights.
    weights = np.asarray([[.2, .6, .9], [.6, .2, .9]])
    return vertices, faces, candidates, scores, neighbors, weights


def test_opposed_triangle_closed_shell_accepts_three_reciprocal_edges():
    inputs = _opposed_triangle_shell()
    labels, metrics = selection.coherent_labels(*inputs)
    np.testing.assert_allclose(metrics["before"]["weighted_boundary_length_m"], 1.1 + .6 * np.sqrt(2.))
    assert metrics["before"]["photo_patch_count"] == 2
    assert metrics["after"]["photo_patch_count"] == 1
    assert labels[0] == labels[1]
    assert np.all((inputs[2] == labels[:, None]).any(1))
    assert metrics["after"]["energy_m2"] < metrics["before"]["energy_m2"]
    assert all(step["energy_after_m2"] <= step["energy_before_m2"]
               for step in metrics["energy_history"])


@pytest.mark.parametrize("malformation", ["self", "wrong_slot", "asymmetric_weight"])
def test_parallel_edges_do_not_excuse_wrong_edge_or_weight(malformation):
    inputs = list(_opposed_triangle_shell())
    if malformation == "self":
        inputs[4][0, 0] = 0
    elif malformation == "wrong_slot":
        # Other face slots still point to this face, but the exact (0,1) edge
        # loses its reciprocal slot; face-pair matching alone must not pass.
        inputs[4][1, 1] = -1
    else:
        inputs[5][1, 1] = .3
    with pytest.raises(ValueError, match="shared edge"):
        selection.coherent_labels(*inputs)


def test_actual_closed_envelope_opposed_pair_has_edge_identity_regression():
    # surface_closed_envelope_v4 faces54658/63539, vertices26970/26976/26977.
    # Retained coordinates reproduce the failure without loading the 74MB GLB.
    inputs = list(_opposed_triangle_shell())
    inputs[0] = np.asarray([[7.527094841003418, 1.186765193939209, -.2588024139404297],
                           [7.540250301361084, 1.1992459297180176, -.2588024139404297],
                           [7.526476860046387, 1.2025599479675293, -.2588024139404297]])
    inputs[5][:] = 1.
    labels, metrics = selection.coherent_labels(*inputs)
    assert labels[0] == labels[1]
    assert metrics["accepted_region_moves"] == 1
    assert metrics["after"]["energy_m2"] < metrics["before"]["energy_m2"]
