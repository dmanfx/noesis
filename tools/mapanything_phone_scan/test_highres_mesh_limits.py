"""Small CPU-only regressions for explicitly raised offline mesh work bounds."""

from __future__ import annotations

import inspect
import subprocess
from unittest.mock import Mock

import numpy as np
import pytest

from tools.mapanything_phone_scan import coherent_surface_photo_labels as selection
from tools.room_mesh_export import build_browser_mesh as browser


@pytest.fixture(autouse=True)
def block_external_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("mesh-limit tests must not launch runtime or subprocesses")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)


@pytest.fixture
def small_surface():
    vertices = np.asarray([[0., 0., 0.], [.1, 0., 0.], [.1, .1, 0.], [0., .1, 0.]])
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    candidates = np.asarray([[47, 106], [106, 47]], dtype=np.int16)
    scores = np.asarray([[1., .99], [1., .99]], dtype=np.float32)
    # Shared diagonal: first face edge (2,0), second face edge (0,2).
    neighbors = np.asarray([[-1, -1, 1], [0, -1, -1]], dtype=np.int32)
    weights = np.asarray([[0., 0., 1.], [1., 0., 0.]], dtype=np.float32)
    return [vertices, faces, candidates, scores, neighbors, weights]


def _without_bounds(metrics):
    return {key: value for key, value in metrics.items() if key != "work_bounds"}


@pytest.mark.parametrize("neutral", [False, True], ids=["decreasing-energy", "neutral-barrier"])
@pytest.mark.parametrize(
    "limits",
    [{"max_faces": None, "max_vertices": None},
     {"max_faces": 3_000_000, "max_vertices": 2_000_000},
     {"max_faces": 5_000_000, "max_vertices": 3_000_000},
     {"max_faces": 5_000_000}, {"max_vertices": 3_000_000},
     {"max_faces": np.int64(5_000_000), "max_vertices": np.int64(3_000_000)}],
    ids=["explicit-none", "baseline", "largest", "faces-only", "vertices-only", "numpy-integers"],
)
def test_optional_caps_preserve_exact_labels_energy_and_inputs(small_surface, neutral, limits):
    if neutral:
        small_surface[2][1] = [-1, -1]
        small_surface[3][1] = 0.
    originals = [value.copy() for value in small_surface]
    for value in small_surface:
        value.setflags(write=False)
    expected_labels, expected_metrics = selection.coherent_labels(*small_surface)
    labels, metrics = selection.coherent_labels(*small_surface, **limits)

    assert np.array_equal(labels, expected_labels)
    # Compare all diagnostics, including exact energy history, not only labels.
    assert _without_bounds(metrics) == _without_bounds(expected_metrics)
    assert metrics["work_bounds"] == {
        **expected_metrics["work_bounds"],
        "max_faces": int(limits.get("max_faces") or 3_000_000),
        "max_vertices": int(limits.get("max_vertices") or 2_000_000),
    }
    assert expected_metrics["work_bounds"]["max_faces"] == 3_000_000
    assert expected_metrics["work_bounds"]["max_vertices"] == 2_000_000
    if neutral:
        assert labels.tolist() == [47, -1]
        assert metrics["neutral_labels_preserved"] is True
    else:
        assert metrics["accepted_region_moves"] > 0
        assert metrics["after"]["energy_m2"] < metrics["before"]["energy_m2"]
    for value, original in zip(small_surface, originals, strict=True):
        assert np.array_equal(value, original)


@pytest.mark.parametrize(
    ("strict", "larger"),
    [({"max_faces": 1}, {"max_faces": 2}),
     ({"max_vertices": 3}, {"max_vertices": 4}),
     ({"max_faces": 1, "max_vertices": 3}, {"max_faces": 2, "max_vertices": 4})],
    ids=["faces", "vertices", "both"],
)
def test_tiny_caps_reject_and_explicit_sufficient_caps_preserve_result(small_surface, strict, larger):
    expected_labels, expected_metrics = selection.coherent_labels(*small_surface)
    with pytest.raises(ValueError, match="malformed or unbounded physical surface graph"):
        selection.coherent_labels(*small_surface, **strict)
    labels, metrics = selection.coherent_labels(*small_surface, **larger)
    assert np.array_equal(labels, expected_labels)
    assert _without_bounds(metrics) == _without_bounds(expected_metrics)


def test_explicit_override_does_not_change_subsequent_default_caps(small_surface, monkeypatch):
    expected_labels, expected_metrics = selection.coherent_labels(*small_surface)
    monkeypatch.setattr(selection, "MAX_FACES", 1)
    monkeypatch.setattr(selection, "MAX_VERTICES", 3)
    with pytest.raises(ValueError, match="unbounded"):
        selection.coherent_labels(*small_surface)
    labels, metrics = selection.coherent_labels(*small_surface, max_faces=2, max_vertices=4)
    assert np.array_equal(labels, expected_labels)
    assert _without_bounds(metrics) == _without_bounds(expected_metrics)
    assert selection.MAX_FACES == 1 and selection.MAX_VERTICES == 3
    with pytest.raises(ValueError, match="unbounded"):
        selection.coherent_labels(*small_surface)


BAD_LIMITS = [True, False, 0, -1, 1.5, "2", np.nan, np.inf, np.bool_(True)]


@pytest.mark.parametrize("name", ["max_faces", "max_vertices"])
@pytest.mark.parametrize("limit", BAD_LIMITS)
def test_malformed_label_caps_are_rejected(small_surface, name, limit):
    with pytest.raises(ValueError, match="work bounds exceed supported limits"):
        selection.coherent_labels(*small_surface, **{name: limit})


@pytest.mark.parametrize("name", ["max_faces", "max_vertices"])
def test_label_caps_above_supported_ceiling_are_rejected(small_surface, name):
    upper = 5_000_000 if name == "max_faces" else 3_000_000
    with pytest.raises(ValueError, match="work bounds exceed supported limits"):
        selection.coherent_labels(*small_surface, **{name: upper + 1})


def test_legacy_mesh_work_defaults_remain_3m_faces_2m_vertices():
    assert selection.MAX_FACES == 3_000_000
    assert selection.MAX_VERTICES == 2_000_000
    selection_parameters = inspect.signature(selection.coherent_labels).parameters
    assert selection_parameters["max_faces"].default is None
    assert selection_parameters["max_vertices"].default is None
    browser_parameters = inspect.signature(browser.read_source).parameters
    assert browser_parameters["max_faces"].default == 3_000_000
    assert browser_parameters["max_vertices"].default == 2_000_000


@pytest.mark.parametrize("name", ["max_faces", "max_vertices"])
@pytest.mark.parametrize("limit", BAD_LIMITS + [None, np.int64(2)])
def test_bad_browser_caps_fail_before_missing_source_access(tmp_path, monkeypatch, name, limit):
    missing = tmp_path / "missing.glb"
    record = Mock(side_effect=AssertionError("invalid caps must not read or hash a source"))
    monkeypatch.setattr(browser, "record", record)
    with pytest.raises(ValueError, match="source mesh work bounds exceed supported limits"):
        browser.read_source(missing, {"output": {}}, **{name: limit})
    record.assert_not_called()
    assert not missing.exists()


@pytest.mark.parametrize("name", ["max_faces", "max_vertices"])
def test_browser_caps_above_ceiling_fail_before_source_access(tmp_path, monkeypatch, name):
    upper = 5_000_000 if name == "max_faces" else 3_000_000
    record = Mock(side_effect=AssertionError("unsupported caps must not access the source"))
    monkeypatch.setattr(browser, "record", record)
    with pytest.raises(ValueError, match="source mesh work bounds exceed supported limits"):
        browser.read_source(tmp_path / "missing.glb", {"output": {}}, **{name: upper + 1})
    record.assert_not_called()


@pytest.mark.parametrize(
    "limits", [{}, {"max_faces": 3_000_000, "max_vertices": 2_000_000},
               {"max_faces": 5_000_000, "max_vertices": 3_000_000}],
    ids=["omitted", "explicit-defaults", "explicit-larger"],
)
def test_valid_browser_caps_reach_the_original_source_binding_check(tmp_path, monkeypatch, limits):
    missing = tmp_path / "missing.glb"
    record = Mock(wraps=browser.record)
    monkeypatch.setattr(browser, "record", record)
    with pytest.raises(FileNotFoundError):
        browser.read_source(missing, {"output": {}}, **limits)
    record.assert_called_once_with(missing)
