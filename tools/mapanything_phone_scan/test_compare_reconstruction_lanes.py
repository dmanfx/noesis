"""CPU-only comparison tests, with no real scan/runtime access.

Set NOESIS_PCF_COMPARISON_REFERENCE_HELPERS to the supplied compare_cross_walk.py
to enable reference-helper tests. All generated input/output files are temporary.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from tools.mapanything_phone_scan import compare_reconstruction_lanes as comparison


@pytest.fixture(autouse=True)
def block_external_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("comparison tests must not launch runtime, inference, or subprocesses")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)


@pytest.fixture(scope="module")
def reference_helpers():
    value = os.environ.get("NOESIS_PCF_COMPARISON_REFERENCE_HELPERS")
    if not value:
        pytest.skip("supply NOESIS_PCF_COMPARISON_REFERENCE_HELPERS for reference integration")
    return comparison._load_helpers(Path(value).resolve())


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _dataset(root, *, edge=8):
    target = root / "target"
    target.mkdir()
    intrinsics = np.asarray([[80., 0., 40.], [0., 80., 40.], [0., 0., 1.]])
    yy, xx = np.indices((80, 80))
    target_points = np.stack(
        ((xx - 40) * 2 / 80, (yy - 40) * 2 / 80, np.full(xx.shape, 2)), axis=-1
    ).reshape(-1, 3)
    np.savez_compressed(target / "room_points.npz", points=target_points,
                        colors=np.zeros((len(target_points), 3), np.uint8))
    assert cv2.imwrite(str(target / "rgb.png"), np.zeros((80, 80, 3), np.uint8))
    mask = np.full((80, 80), 255, np.uint8)
    mask[0, 0] = 0
    assert cv2.imwrite(str(target / "valid.png"), mask)
    calibration = root / "calibration.json"
    _write_json(calibration, {"cameras": {"living-room": {
        "E": np.eye(4).reshape(-1, order="F").tolist(),
    }}})
    binding = {
        "world_frame": "backend_world_m", "world_frame_revision": "synthetic-review",
        "camera_frame_binding": {"camera_calibration_sha256": "physical-calibration"},
        "reference_kind": "paired_static", "companion_session_id": "synthetic-session",
    }
    meta = {**binding, "camera": "living-room", "coordinate_frame": "backend_world_m_stream_points",
            "rgb_keyframes": {"living-room": "rgb.png"},
            "rgb_keyframe_valid_masks": {"living-room": "valid.png"},
            "intrinsics": intrinsics.tolist(),
            "floor_alignment": {"world_correction_col_major": np.eye(4).reshape(-1, order="F").tolist()}}
    _write_json(target / "room_points_meta.json", meta)
    target_inputs = {key: comparison._identity(path)["sha256"] for key, path in {
        "target_points_sha256": target / "room_points.npz",
        "target_meta_sha256": target / "room_points_meta.json",
        "fixed_camera_keyframe_sha256": target / "rgb.png",
        "camera_calibration_sha256": calibration,
        "fixed_camera_rgb_valid_mask_sha256": target / "valid.png",
    }.items()}
    args = {"target_revision": target, "calibration": calibration, "camera": "living-room"}
    for name, z in (("baseline", 2.0), ("candidate", 2.1)):
        consensus, alignment = root / f"{name}-consensus", root / f"{name}-alignment"
        consensus.mkdir()
        alignment.mkdir()
        raw = consensus / "raw"
        raw.mkdir()
        args[f"{name}_consensus"], args[f"{name}_alignment"] = consensus, alignment
        frame = "consensus_phone_metric_world_unaligned_to_noesis"
        source_path = consensus / "scan_outputs_manifest.json"
        _write_json(source_path, {"view_count": 2, "coordinate_frame": frame})
        _write_json(consensus / "consensus_manifest.json", {
            "view_count": 2, "multiview_consistency": {"consensus": {"median_error_m": 0.0}},
            "surfel_fusion": {"voxel_size_m": 0.04},
        })
        matrix = np.eye(4)
        if name == "candidate":
            matrix[:3, 3] = [0.2, -0.3, 0.4]
        y, x = np.indices((edge, edge))
        depth = np.full(x.shape, z, np.float32)
        world = np.stack(((x / edge * 80 - 40) * z / 80,
                          (y / edge * 80 - 40) * z / 80, depth), axis=-1) - matrix[:3, 3]
        views = []
        for index in range(2):
            path = raw / f"view_{index:04d}.npz"
            np.savez_compressed(path, world_points=world.astype(np.float32), depth_z=depth,
                                confidence=np.ones(x.shape, np.float32), mask=np.ones(x.shape, bool),
                                model_rgb=np.zeros((*x.shape, 3), np.uint8), camera_pose=np.linalg.inv(matrix))
            views.append({"name": path.name, **comparison._identity(path)})
        records = [{key: row[key] for key in ("name", "sha256", "size_bytes")} for row in views]
        set_hash = hashlib.sha256(json.dumps(
            records, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        manifest_identity = {key: comparison._identity(source_path)[key] for key in ("path", "sha256")}
        provenance = {"raw_root": str(raw), "output_manifest": manifest_identity,
                      "view_count": 2, "views": views, "raw_view_set_sha256": set_hash}
        transform = {"target_binding": binding, "source_coordinate_frame": frame,
                     "target_coordinate_frame": "backend_world_m_stream_points", "scale": 1.0,
                     "world_from_mapanything_row_major": matrix.tolist(), "source_raw_root": str(raw),
                     "source_output_manifest": manifest_identity, "source_raw_view_set_sha256": set_hash}
        _write_json(alignment / "phone_ma_to_noesis_world.json", transform)
        _write_json(alignment / "alignment_report.json", {
            "status": "passed", "quality_gate": {"passed": True, "checks": {"rigid": True}},
            "transform": transform, "target": {"camera_id": "living-room"},
            "inputs": {**target_inputs, "phone_source": provenance,
                       "phone_output_manifest_sha256": manifest_identity["sha256"]},
        })
    heldout = root / "heldout"
    heldout.mkdir()
    artifact = heldout / "heldout_0001.npz"
    np.savez_compressed(artifact, depth_z=np.full((80, 80), 2.), mask=mask > 0,
                        intrinsics=intrinsics, camera_to_world=np.eye(4))
    manifest = {
        "status": "complete", "camera_id": "living-room", "world_frame_revision": "synthetic-review",
        "calibration_sha256": "physical-calibration", "camera_to_world": np.eye(4).tolist(),
        "used_for_alignment": False, "used_for_reference_inference": False, "fitting_performed": False,
        "independent_of_phone_geometry": True, "heldout_limitations": "Synthetic, not surveyed geometry.",
        "model": {"id": "synthetic-fixture"},
        "files": [{"path": artifact.name, "sha256": comparison._identity(artifact)["sha256"]}],
        "source": {"pixel_decode": "synthetic-bounded-fixture", "frames": [{"artifact": artifact.name,
                   "source_frame_index": 1, "encoded_pts_ns": 123456, "intrinsics": intrinsics.tolist(),
                   "source_exclusion_mask_sha256": "frozen-exclusion"}]},
    }
    args["heldout_manifest"] = heldout / "manifest.json"
    _write_json(args["heldout_manifest"], manifest)
    value = os.environ.get("NOESIS_PCF_COMPARISON_REFERENCE_HELPERS")
    args["reference_helpers"] = Path(value).resolve() if value else root / "helpers.py"
    if not value:
        args["reference_helpers"].write_text("# temporary mocked helper\n")
    args["output_dir"] = root / "comparison"
    return SimpleNamespace(args=SimpleNamespace(**args), manifest=manifest)


@pytest.fixture
def dataset(tmp_path):
    return _dataset(tmp_path)


def _edit_alignment(dataset, edit):
    root = dataset.args.candidate_alignment
    report = comparison._json(root / "alignment_report.json")
    edit(report, report["transform"])
    _write_json(root / "phone_ma_to_noesis_world.json", report["transform"])
    _write_json(root / "alignment_report.json", report)


def _stub_helpers(dataset, monkeypatch):
    helpers = SimpleNamespace(
        masked_depth_cells=Mock(return_value=np.ones((10, 10))),
        load_heldouts=Mock(return_value=(dataset.manifest, [SimpleNamespace()])),
        exact_cloud=Mock(side_effect=AssertionError("must reject before sampling")),
    )
    monkeypatch.setattr(comparison, "_load_helpers", lambda path: helpers)
    return helpers


@pytest.mark.parametrize("case", ["status", "gate", "check", "frame", "count", "matrix", "scale", "view-order", "set-hash"])
def test_lane_rejects_failed_or_mismatched_alignment(dataset, case):
    def edit(report, transform):
        if case == "status": report["status"] = "rejected"
        elif case == "gate": report["quality_gate"]["passed"] = False
        elif case == "check": report["quality_gate"]["checks"]["rigid"] = False
        elif case == "frame": transform["source_coordinate_frame"] = "another-frame"
        elif case == "count": report["inputs"]["phone_source"]["view_count"] = 3
        elif case == "matrix": transform["world_from_mapanything_row_major"][0][0] = 2
        elif case == "scale": transform["scale"] = 2.0
        elif case == "view-order": report["inputs"]["phone_source"]["views"].reverse()
        elif case == "set-hash": transform["source_raw_view_set_sha256"] = "f" * 64
    _edit_alignment(dataset, edit)
    with pytest.raises(ValueError):
        comparison._load_lane(dataset.args.candidate_consensus, dataset.args.candidate_alignment)


@pytest.mark.parametrize("artifact", ["scan_outputs_manifest.json", "raw/view_0001.npz"])
def test_lane_rejects_changed_bound_source_content(dataset, artifact):
    path = dataset.args.candidate_consensus / artifact
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="different source manifest|no longer matches"):
        comparison._load_lane(dataset.args.candidate_consensus, dataset.args.candidate_alignment)


def test_transform_file_must_match_passed_report(dataset):
    path = dataset.args.candidate_alignment / "phone_ma_to_noesis_world.json"
    transform = comparison._json(path)
    transform["world_from_mapanything_row_major"][0][3] += .01
    _write_json(path, transform)
    with pytest.raises(ValueError, match="report and transform disagree"):
        comparison._load_lane(dataset.args.candidate_consensus, dataset.args.candidate_alignment)


@pytest.mark.parametrize("case", ["target-binding", "camera", "target-points", "target-meta", "rgb", "calibration", "mask"])
def test_comparison_rejects_target_identity_mismatch_before_sampling(dataset, monkeypatch, case):
    helpers = _stub_helpers(dataset, monkeypatch)
    def edit(report, transform):
        if case == "target-binding":
            transform["target_binding"]["world_frame_revision"] = "another-revision"
        elif case == "camera":
            report["target"]["camera_id"] = "another-camera"
        else:
            key = {"target-points": "target_points_sha256", "target-meta": "target_meta_sha256",
                   "rgb": "fixed_camera_keyframe_sha256", "calibration": "camera_calibration_sha256",
                   "mask": "fixed_camera_rgb_valid_mask_sha256"}[case]
            report["inputs"][key] = "f" * 64
    _edit_alignment(dataset, edit)
    with pytest.raises(ValueError, match="different target bindings|another camera|target input differs"):
        comparison.compare(dataset.args)
    helpers.exact_cloud.assert_not_called()
    assert not dataset.args.output_dir.exists()


@pytest.mark.parametrize("case", ["existing-output", "output-in-input", "missing-mask", "wrong-mask-shape"])
def test_comparison_output_and_mask_guards(dataset, monkeypatch, case):
    _stub_helpers(dataset, monkeypatch)
    if case == "existing-output":
        dataset.args.output_dir.mkdir()
    elif case == "output-in-input":
        dataset.args.output_dir = dataset.args.baseline_consensus / "forbidden-output"
    elif case == "missing-mask":
        path = dataset.args.target_revision / "room_points_meta.json"
        meta = comparison._json(path)
        meta["rgb_keyframe_valid_masks"] = {}
        _write_json(path, meta)
    else:
        assert cv2.imwrite(str(dataset.args.target_revision / "valid.png"), np.ones((8, 8), np.uint8))
    with pytest.raises(FileExistsError if case == "existing-output" else ValueError):
        comparison.compare(dataset.args)


@pytest.mark.parametrize("case", ["undershoot", "audit-budget", "audit-views"])
def test_sampler_budget_and_declared_view_audit_fail_closed(dataset, monkeypatch, case):
    helpers = _stub_helpers(dataset, monkeypatch)
    count = 599_999 if case == "undershoot" else 600_000
    helpers.exact_cloud.side_effect = None
    helpers.exact_cloud.return_value = SimpleNamespace(points=np.broadcast_to(np.zeros(3), (count, 3)))
    helpers.exact_cloud.audit = [
        {"view": "view_0000.npz", "sampled_points": 300_000},
        {"view": "view_0001.npz", "sampled_points": 300_000},
    ]
    if case == "audit-budget": helpers.exact_cloud.audit[1]["sampled_points"] -= 1
    if case == "audit-views": helpers.exact_cloud.audit[1]["view"] = "view_9999.npz"
    with pytest.raises(ValueError, match="equal point budget|Sampler audit differs"):
        comparison.compare(dataset.args)
    assert not dataset.args.output_dir.exists()


def test_exact_view_container_uses_no_directory_inventory(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("declared view container must not scan the filesystem")
    monkeypatch.setattr(Path, "glob", forbidden)
    monkeypatch.setattr(Path, "rglob", forbidden)
    paths = [tmp_path / "raw/view_0000.npz", tmp_path / "raw/view_0001.npz"]
    assert list(comparison._BoundRawViews(paths).glob("view_*.npz")) == paths
    with pytest.raises(ValueError, match="declared raw view set"):
        list(comparison._BoundRawViews(paths).glob("**/*"))
    with pytest.raises(ValueError, match="escapes"):
        comparison._artifact(tmp_path, "../escape.npz")


def test_common_support_keeps_identical_cells_but_retains_own_coverage(reference_helpers):
    target = np.asarray([[2., 3.], [np.inf, 4.]])
    grids = {"baseline": np.asarray([[2., np.inf], [9., 4.]]),
             "candidate": np.asarray([[2.1, 3.], [np.inf, 4.2]])}
    common, scores, errors = comparison._score(grids, target, reference_helpers)
    assert np.array_equal(common, [[True, False], [False, True]])
    assert scores["baseline"]["common_support"]["count"] == scores["candidate"]["common_support"]["count"] == 2
    assert scores["baseline"]["own_support"]["count"] == 2
    assert scores["candidate"]["own_support"]["count"] == 3
    assert np.allclose(errors["candidate"]["common"], [.1, .2])
    with pytest.raises(ValueError, match="different shapes"):
        comparison._score({"baseline": np.ones((3, 3))}, target, reference_helpers)


@pytest.mark.parametrize("cell_px", [4, 8])
def test_reference_cells_conservatively_exclude_mask_edges(reference_helpers, cell_px):
    depth, mask = np.ones((cell_px * 2 + 1, cell_px * 2 + 1)), np.ones((cell_px * 2 + 1, cell_px * 2 + 1), bool)
    mask[0, 0] = False
    cells = reference_helpers.masked_depth_cells(depth, mask, cell_px)
    assert cells.shape == (3, 3)
    assert np.isfinite(cells).sum() == 3
    assert not np.isfinite(cells[0, 0])
    assert not np.isfinite(cells[2]).any()
    assert not np.isfinite(cells[:, 2]).any()


@pytest.mark.parametrize("key", ["used_for_alignment", "used_for_reference_inference", "fitting_performed", "calibration_sha256"])
def test_reference_rejects_non_withheld_or_mismatched_evidence(dataset, reference_helpers, key):
    manifest = copy.deepcopy(dataset.manifest)
    manifest[key] = "another-calibration" if key == "calibration_sha256" else True
    _write_json(dataset.args.heldout_manifest, manifest)
    static = comparison._load_static_reference(dataset.args.target_revision, dataset.args.calibration, "living-room")
    binding = comparison._json(dataset.args.baseline_alignment / "phone_ma_to_noesis_world.json")["target_binding"]
    with pytest.raises(ValueError):
        reference_helpers.load_heldouts(dataset.args.heldout_manifest, static, binding, "living-room")


def test_heldout_report_pools_cells_not_frame_means_and_preserves_provenance(tmp_path, reference_helpers):
    path = tmp_path / "heldout.json"
    manifest = {"independent_of_phone_geometry": True, "heldout_limitations": "model-based",
                "source": {"pixel_decode": "frozen-decoder", "timing": {"verified": False}}}
    _write_json(path, manifest)
    rows = []
    for index, errors in enumerate(([.1], [1., 1., 1.])):
        target = np.ones((1, len(errors)))
        rows.append({"source_frame_index": index, "encoded_pts_ns": index * 100,
                     "artifact": {"sha256": f"frame-{index}"}, "valid_model_pixels": len(errors),
                     "image_width": len(errors), "image_height": 1, "target_grid": target,
                     "grids": {"baseline": target.copy(), "candidate": target + np.asarray([errors])}})
    result = comparison._heldout_report(path, manifest, rows, reference_helpers)
    assert result["manifest_provenance"] == manifest
    assert result["metric_ground_truth"] is False
    assert result["cell_px"] == 4
    assert result["pooled"]["candidate"]["common_support"]["median_m"] == 1.0
    assert result["pooled"]["candidate"]["common_support"]["count"] == 4
    assert result["pooled"]["candidate"]["coverage"] == 1.0


def test_real_helpers_end_to_end_exact_600k_outputs_and_unchanged_inputs(tmp_path, reference_helpers):
    dataset = _dataset(tmp_path, edge=548)
    raw_paths = [getattr(dataset.args, f"{name}_consensus") / "raw" / f"view_{index:04d}.npz"
                 for name in comparison.LANES for index in range(2)]
    before = {path: comparison._identity(path)["sha256"] for path in raw_paths}
    result = comparison.compare(dataset.args)
    assert result["point_budget_each"] == 600_000
    assert result["common_static_cells"] == 99
    assert result["metric_ground_truth"] is False
    assert result["heldout"]["manifest_provenance"] == dataset.manifest
    assert result["heldout"]["frames"][0]["encoded_pts_ns"] == 123456
    for name, expected in (("baseline", 0.), ("candidate", .1)):
        row = result["candidates"][name]
        assert row["sampled_points"] == 600_000
        assert [item["sampled_points"] for item in row["per_view_sampling"]] == [300_000] * 2
        for support in ("static_own_support", "static_common_support"):
            assert row[support]["median_m"] == pytest.approx(expected, abs=1e-6)
        pooled = result["heldout"]["pooled"][name]
        assert pooled["common_support"]["median_m"] == pytest.approx(expected, abs=1e-6)
        assert pooled["common_support"]["count"] == 399
        assert pooled["coverage"] == 1.0
    output = dataset.args.output_dir
    assert comparison._json(output / "metrics.json") == result
    assert "Metric ground truth: **false**" in (output / "COMPARISON.md").read_text()
    with np.load(output / "static_depth_grids.npz") as grids:
        assert set(grids.files) == {"target", "common", "baseline", "candidate"}
        assert grids["common"].sum() == 99
        assert not grids["common"][0, 0]
    assert {path: comparison._identity(path)["sha256"] for path in raw_paths} == before
