from __future__ import annotations

import json

import numpy as np
import pytest

from tools.mapanything_phone_scan.build_dense_review_surfels import build
from tools.mapanything_phone_scan.build_single_room_review_surface import SingleRoomSurfaceError, prepare_input
from tools.mapanything_phone_scan.test_build_single_room_review_surface import _sha256, _write_alignment_fixture, _write_surfels


def _fixture(tmp_path):
    consensus, alignment = _write_alignment_fixture(tmp_path)
    _write_surfels(consensus)
    report_path = alignment / "alignment_report.json"
    report = json.loads(report_path.read_text())
    for index in range(2):
        path = consensus / "raw" / f"view_{index:04d}.npz"
        np.savez(path, camera_pose=np.eye(4), depth_z=np.full((4, 4), 2., dtype=np.float32),
                 mask=np.ones((4, 4), dtype=bool), confidence=np.ones((4, 4), dtype=np.float32),
                 model_rgb=np.full((4, 4, 3), 128, dtype=np.uint8),
                 intrinsics=np.asarray([[80., 0., 1.5], [0., 80., 1.5], [0., 0., 1.]]))
        report["inputs"]["phone_source"]["views"][index].update(sha256=_sha256(path), size_bytes=path.stat().st_size)
    report_path.write_text(json.dumps(report))
    return consensus, alignment


def test_actual_cached_export_and_surface_consumer_preserve_sources_and_support(tmp_path):
    consensus, alignment = _fixture(tmp_path)
    hashes = [_sha256(consensus / "raw" / f"view_{index:04d}.npz") for index in range(2)]
    dense = tmp_path / "dense"
    report = build(consensus, alignment, dense, camera_id="living-room")
    assert report["metrics"]["input_sample_count"] == 32
    assert report["metrics"]["surfel_count"] == 16
    assert report["metrics"]["support_view_count_min"] == 2
    assert report["raw_geometry_or_mask_changed"] is False
    with np.load(dense / "surfel_points.npz") as rows:
        assert len(rows["points"]) == 16
        np.testing.assert_array_equal(rows["support_view_ids"][:, :2], np.tile([0, 1], (16, 1)))
    adapted = prepare_input(consensus_dir=consensus, alignment_dir=alignment, room_name="living",
                            owner=3, output_dir=tmp_path / "mesh_input", dense_surfel_dir=dense)
    assert adapted["manifest"]["support_metrics"]["accepted_multi_view_count"] == 16
    assert adapted["manifest"]["source"]["dense_surfel_export"]["sha256"] == _sha256(dense / "dense_surfel_manifest.json")
    assert hashes == [_sha256(consensus / "raw" / f"view_{index:04d}.npz") for index in range(2)]
    with pytest.raises(ValueError, match="overwrite"):
        build(consensus, alignment, dense, camera_id="living-room")


@pytest.mark.parametrize("mutation", ["raw_identity", "support_gate", "artifact_digest", "static_points"])
def test_dense_surface_consumer_rejects_changed_authority_or_evidence(tmp_path, mutation):
    consensus, alignment = _fixture(tmp_path)
    dense = tmp_path / "dense"
    build(consensus, alignment, dense, camera_id="living-room")
    path = dense / "dense_surfel_manifest.json"
    report = json.loads(path.read_text())
    if mutation == "raw_identity":
        report["source"]["raw_views"][0]["sha256"] = "0" * 64
    elif mutation == "support_gate":
        report["metrics"]["minimum_distinct_views_per_surfel"] = 1
    elif mutation == "artifact_digest":
        report["artifacts"]["accepted_surfels"]["sha256"] = "0" * 64
    else:
        report["static_camera_points_included"] = True
    path.write_text(json.dumps(report))
    with pytest.raises(SingleRoomSurfaceError):
        prepare_input(consensus_dir=consensus, alignment_dir=alignment, room_name="living", owner=3,
                      output_dir=tmp_path / "bad_surface", dense_surfel_dir=dense)
    assert not (tmp_path / "bad_surface").exists()


def test_cached_dense_export_respects_memory_and_camera_bounds(tmp_path):
    consensus, alignment = _fixture(tmp_path)
    with pytest.raises(ValueError, match="memory budget"):
        build(consensus, alignment, tmp_path / "too_large", camera_id="living-room", max_total_pixels=20)
    assert not (tmp_path / "too_large").exists()
    with pytest.raises(SingleRoomSurfaceError, match="camera"):
        build(consensus, alignment, tmp_path / "wrong_camera", camera_id="kitchen")
