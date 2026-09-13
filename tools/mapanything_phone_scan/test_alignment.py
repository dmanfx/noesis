from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.mapanything_phone_scan import alignment


def test_explicit_static_rgb_exclusion_removes_features_before_matching(tmp_path):
    image = np.random.default_rng(7).integers(0, 256, (192, 256, 3), dtype=np.uint8)
    keyframe = tmp_path / "keyframe.png"
    assert cv2.imwrite(str(keyframe), image)
    ys, xs = np.indices((40, 50))
    points = np.column_stack((xs.ravel() / 25, ys.ravel() / 25, np.ones(xs.size)))
    k = np.asarray([[50, 0, 20], [0, 50, 20], [0, 0, 1]], dtype=float)
    result, report = alignment._visual_alignment_anchor(
        tmp_path, points, keyframe, np.eye(4), k, np.empty((0, 4, 4)), lambda *_: None,
        target_valid_mask=np.zeros((192, 256), dtype=np.uint8),
    )
    assert result is None and report["target_feature_count"] == 0
    assert report["status"] == "insufficient_target_features"
    with pytest.raises(alignment.NoesisAlignmentError, match="mask dimensions"):
        alignment._visual_alignment_anchor(
            tmp_path, points, keyframe, np.eye(4), k, np.empty((0, 4, 4)), lambda *_: None,
            target_valid_mask=np.zeros((10, 10), dtype=np.uint8),
        )


def _write_raw_view(path: Path, marker: float) -> None:
    points = np.asarray(
        [
            [[marker, 1.0, 2.0], [marker + 0.1, 1.0, 2.0]],
            [[marker, 1.1, 2.0], [marker + 0.1, 1.1, 2.0]],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        path,
        world_points=points,
        depth_z=np.full((2, 2), 2.0, dtype=np.float32),
        confidence=np.ones((2, 2), dtype=np.float32),
        mask=np.ones((2, 2), dtype=bool),
        camera_pose=np.eye(4, dtype=np.float64),
        model_rgb=np.full((2, 2, 3), 0.5, dtype=np.float32),
        intrinsics=np.asarray([[50.0, 0.0, 4.0], [0.0, 50.0, 4.0], [0.0, 0.0, 1.0]]),
    )


def _make_source(root: Path, manifest: Path, marker: float) -> None:
    root.mkdir(parents=True)
    _write_raw_view(root / "view_0000.npz", marker)
    _write_raw_view(root / "view_0001.npz", marker + 1.0)
    manifest.write_text(
        json.dumps(
            {
                "schema": "noesis.phone_scan.outputs.v2",
                "view_count": 2,
            }
        ),
        encoding="utf-8",
    )


def test_explicit_source_raw_root_reaches_real_loader(tmp_path: Path) -> None:
    scan_dir = tmp_path / "scan"
    default_raw = scan_dir / "outputs" / "raw"
    default_raw.mkdir(parents=True)
    _write_raw_view(default_raw / "view_0000.npz", 90.0)
    _write_raw_view(default_raw / "view_0001.npz", 91.0)
    default_manifest = scan_dir / "outputs" / "scan_outputs_manifest.json"
    default_manifest.write_text(json.dumps({"view_count": 2}), encoding="utf-8")

    explicit_raw = tmp_path / "refined-carrier" / "raw"
    explicit_manifest = tmp_path / "refined-carrier" / "outputs.json"
    _make_source(explicit_raw, explicit_manifest, 7.0)
    outputs = {"view_count": 2}
    settings = alignment.NoesisAlignmentSettings(
        camera_id="camera0",
        target_revision=tmp_path / "target",
        calibration_path=tmp_path / "calibration.json",
        review_point_budget=1_000,
    )

    default_registration, _, _, _ = alignment._load_phone_clouds(
        scan_dir,
        outputs,
        settings,
    )
    assert float(default_registration[0, 0]) == 90.0
    assert alignment._resolve_source_output_manifest(scan_dir, outputs, None) == (
        default_manifest.resolve()
    )
    registration, _, _, _ = alignment._load_phone_clouds(
        scan_dir,
        outputs,
        settings,
        source_raw_root=explicit_raw,
    )
    provenance = alignment._source_raw_provenance(
        explicit_raw,
        explicit_manifest,
        2,
    )

    assert float(registration[0, 0]) == 7.0
    assert provenance["raw_root"] == str(explicit_raw.resolve())
    assert provenance["output_manifest"]["path"] == str(explicit_manifest.resolve())
    assert [row["path"] for row in provenance["views"]] == [
        str((explicit_raw / "view_0000.npz").resolve()),
        str((explicit_raw / "view_0001.npz").resolve()),
    ]
    assert len(provenance["raw_view_set_sha256"]) == 64


def test_alignment_report_binds_explicit_source_and_output_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scan_dir = tmp_path / "scan"
    scan_dir.mkdir()
    explicit_raw = tmp_path / "post-scale-carrier" / "raw"
    explicit_manifest = tmp_path / "post-scale-carrier" / "outputs.json"
    _make_source(explicit_raw, explicit_manifest, 3.0)

    target_revision = tmp_path / "target-revision"
    target_revision.mkdir()
    target_points = np.asarray(
        [[0.0, 1.0, 2.0], [0.1, 1.0, 2.0], [0.0, 1.1, 2.0]],
        dtype=np.float64,
    )
    np.savez_compressed(target_revision / "room_points.npz", points=target_points)
    target_meta = {
        "camera": "camera0",
        "coordinate_frame": "backend_world_m_stream_points",
        "revision_id": "target-r1",
        "intrinsics": [[50.0, 0.0, 4.0], [0.0, 50.0, 4.0], [0.0, 0.0, 1.0]],
        "rgb_keyframes": {"camera0": "keyframe.png"},
        "floor_alignment": {
            "world_correction_col_major": np.eye(4).reshape(-1, order="F").tolist()
        },
        "floor_y": 0.0,
    }
    (target_revision / "room_points_meta.json").write_text(
        json.dumps(target_meta),
        encoding="utf-8",
    )
    assert cv2.imwrite(
        str(target_revision / "keyframe.png"),
        np.zeros((8, 8, 3), dtype=np.uint8),
    )
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps({"cameras": {"camera0": {"E": np.eye(4).reshape(-1, order="F").tolist()}}}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "alignment-output"
    output_dir.mkdir()
    settings = alignment.NoesisAlignmentSettings(
        camera_id="camera0",
        target_revision=target_revision,
        calibration_path=calibration_path,
        review_point_budget=1_000,
    )

    def level(points, poses, _settings):
        return points, poses, np.eye(4), {"test": True}

    def vertical(points, _voxel_size):
        return points, np.tile(np.asarray([1.0, 0.0, 0.0]), (len(points), 1))

    quality = {
        "objective": 0.01,
        "target_overlap_0_30m": 0.8,
        "source_overlap_0_30m": 0.8,
        "plane_residual_median_m": 0.01,
        "plane_residual_p80_m": 0.02,
    }
    visible_structure = {
        "comparable_point_count": 1_000.0,
        "source_point_count": 1_000.0,
        "source_overlap_0_30m": 0.8,
        "plane_residual_median_m": 0.01,
        "plane_residual_p80_m": 0.02,
    }
    visible_cloud = {
        "comparable_point_count": 6_000.0,
        "source_point_count": 6_000.0,
        "source_overlap_0_30m": 0.8,
    }
    monkeypatch.setattr(alignment, "_level_phone_floor", level)
    monkeypatch.setattr(alignment, "_voxel_points", lambda points, _size: points)
    monkeypatch.setattr(alignment, "_vertical_structure", vertical)
    monkeypatch.setattr(
        alignment,
        "_structure_refine",
        lambda *_args, **_kwargs: (np.zeros(3, dtype=np.float64), quality),
    )
    monkeypatch.setattr(
        alignment,
        "_full_cloud_metrics",
        lambda *_args: {
            "source_median_m": 0.01,
            "target_median_m": 0.01,
            "source_overlap_0_30m": 0.8,
            "target_overlap_0_30m": 0.8,
        },
    )
    monkeypatch.setattr(
        alignment,
        "_fixed_camera_visible_structure_metrics",
        lambda *_args, **_kwargs: visible_structure,
    )
    monkeypatch.setattr(
        alignment,
        "_fixed_camera_visible_cloud_metrics",
        lambda *_args, **_kwargs: visible_cloud,
    )
    monkeypatch.setattr(
        alignment,
        "_project_depth_grid",
        lambda *_args, **_kwargs: np.ones((1, 1), dtype=np.float64),
    )
    monkeypatch.setattr(
        alignment,
        "_write_glbs",
        lambda aligned, comparison, *_args: (
            aligned.write_bytes(b"aligned"),
            comparison.write_bytes(b"comparison"),
        ),
    )
    monkeypatch.setattr(
        alignment,
        "_write_topdown",
        lambda path, *_args: path.write_bytes(b"topdown"),
    )
    monkeypatch.setattr(
        alignment,
        "_write_reprojection",
        lambda path, *_args: (path.write_bytes(b"reprojection") or {"ok": 1.0}),
    )

    diagnostics = []
    result = alignment.run_noesis_alignment(
        scan_dir,
        output_dir,
        {"view_count": 2, "anchor_view_index": 0},
        settings,
        lambda *_args: None,
        source_raw_root=explicit_raw,
        source_output_manifest=explicit_manifest,
        diagnostic_callback=lambda report, arrays: diagnostics.append((report, arrays)),
    )

    report_path = output_dir / "alignment_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    source = report["inputs"]["phone_source"]
    assert source["raw_root"] == str(explicit_raw.resolve())
    assert source["output_manifest"]["path"] == str(explicit_manifest.resolve())
    assert {Path(row["path"]).parent for row in source["views"]} == {
        explicit_raw.resolve()
    }
    assert result["artifact_paths"] == {
        "transform": str((output_dir / "phone_ma_to_noesis_world.json").resolve()),
        "report": str(report_path.resolve()),
    }
    assert report["artifact_paths"] == result["artifact_paths"]
    assert result["global_vertical_structure"] == report["global_vertical_structure"]
    assert result["global_vertical_structure"]["metric_domain"] == "all_source_structure"
    transform = json.loads(
        (output_dir / "phone_ma_to_noesis_world.json").read_text(encoding="utf-8")
    )
    assert transform["source_raw_root"] == str(explicit_raw.resolve())
    assert transform["source_output_manifest"]["path"] == str(
        explicit_manifest.resolve()
    )
    assert diagnostics[0][0]["quality_gate"]["passed"] is True
    assert diagnostics[0][0]["source_provenance"] == source
    assert diagnostics[0][0]["input_hashes"] == {
        name: value for name, value in report["inputs"].items() if name.endswith("_sha256")
    }
    np.testing.assert_array_equal(diagnostics[0][1]["best_parameters"], np.zeros(3))

    visible_structure["plane_residual_median_m"] = 0.20
    failed_dir = tmp_path / "failed-fit"
    failed_dir.mkdir()
    with pytest.raises(alignment.NoesisAlignmentError, match="vertical_plane_residual"):
        alignment.run_noesis_alignment(
            scan_dir, failed_dir, {"view_count": 2, "anchor_view_index": 0}, settings,
            lambda *_args: None, source_raw_root=explicit_raw,
            source_output_manifest=explicit_manifest,
            diagnostic_callback=lambda report, arrays: diagnostics.append((report, arrays)),
        )
    assert diagnostics[1][0]["quality_gate"]["passed"] is False
    assert diagnostics[1][0]["failed_checks"] == ["vertical_plane_residual"]
    assert diagnostics[1][0]["input_hashes"] == diagnostics[0][0]["input_hashes"]
    assert not (failed_dir / "phone_ma_to_noesis_world.json").exists()
