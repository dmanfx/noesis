from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from tools.mapanything_phone_scan.build_single_room_review_surface import (
    SingleRoomSurfaceError,
    prepare_input,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_alignment_fixture(
    root: Path,
    *,
    status: str = "passed",
    wrong_digest: bool = False,
    missing_raw_root: bool = False,
    missing_manifest_sha: bool = False,
    target_camera_id: str = "living-room",
    target_world_frame_revision: str | None = "test-world-revision",
    payload_world_frame_revision: str | None = "test-world-revision",
) -> tuple[Path, Path]:
    consensus = root / "consensus"
    alignment_dir = root / "alignment"
    raw = consensus / "raw"
    raw.mkdir(parents=True)
    alignment_dir.mkdir(parents=True)
    for index in range(2):
        np.savez(
            raw / f"view_{index:04d}.npz",
            camera_pose=np.eye(4, dtype=np.float32),
        )
    output_manifest = consensus / "scan_outputs_manifest.json"
    output_manifest.write_text(json.dumps({"view_count": 2}) + "\n")
    (consensus / "consensus_manifest.json").write_text("{}\n")
    manifest_binding = {
        "path": str(output_manifest.resolve()),
        "sha256": _sha256(output_manifest),
    }
    if missing_manifest_sha:
        manifest_binding.pop("sha256")
    raw_root = None if missing_raw_root else str(raw.resolve())
    identity = np.eye(4, dtype=float).tolist()
    camera_frame_binding = {
        "frame_id": "backend_world_m",
        "revision": "test-camera-frame-revision",
    }
    views = []
    for index in range(2):
        path = raw / f"view_{index:04d}.npz"
        views.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": ("0" * 64 if wrong_digest else _sha256(path)),
            }
        )
    alignment = {
        "status": status,
        "quality_gate": {"passed": status == "passed"},
        "target": {
            "camera_id": target_camera_id,
            "revision_id": "test-revision",
            "coordinate_frame": "backend_world_m_stream_points",
            "world_frame": "backend_world_m",
            "world_frame_revision": target_world_frame_revision,
            "camera_frame_binding": camera_frame_binding,
        },
        "transform": {
            "source_raw_root": raw_root,
            "source_output_manifest": manifest_binding,
            "world_from_mapanything_row_major": identity,
            "source_raw_view_set_sha256": "source-view-set",
        },
        "inputs": {
            "phone_source": {
                "raw_root": raw_root,
                "output_manifest": manifest_binding,
                "views": views,
            }
        },
    }
    (alignment_dir / "alignment_report.json").write_text(json.dumps(alignment) + "\n")
    transform = {
        "source_coordinate_frame": "consensus_phone_metric_world_unaligned_to_noesis",
        "target_coordinate_frame": "backend_world_m_stream_points",
        "target_binding": {
            "world_frame": "backend_world_m",
            "world_frame_revision": payload_world_frame_revision,
            "camera_frame_binding": camera_frame_binding,
        },
        "world_from_mapanything_row_major": identity,
        "source_raw_root": raw_root,
        "source_output_manifest": manifest_binding,
        "source_raw_view_set_sha256": "source-view-set",
    }
    (alignment_dir / "phone_ma_to_noesis_world.json").write_text(json.dumps(transform) + "\n")
    return consensus, alignment_dir


def _write_surfels(consensus: Path) -> None:
    np.savez_compressed(
        consensus / "surfel_points.npz",
        points=np.asarray([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32),
        colors=np.asarray([[10, 20, 30], [40, 50, 60]], dtype=np.uint8),
        support_view_count=np.asarray([2, 3], dtype=np.uint16),
    )
    np.savez_compressed(
        consensus / "surfel_points_single_view.npz",
        points=np.asarray([[3.0, 3.0, 3.0]], dtype=np.float32),
        colors=np.asarray([[70, 80, 90]], dtype=np.uint8),
        support_view_count=np.asarray([1], dtype=np.uint16),
    )


def test_rejects_unpassed_alignment(tmp_path: Path) -> None:
    consensus, alignment = _write_alignment_fixture(tmp_path, status="inconsistent")
    with pytest.raises(SingleRoomSurfaceError, match="status is not passed"):
        prepare_input(
            consensus_dir=consensus,
            alignment_dir=alignment,
            room_name="living-room",
            owner=3,
            output_dir=tmp_path / "output",
        )


def test_rejects_mismatched_raw_view_digest(tmp_path: Path) -> None:
    consensus, alignment = _write_alignment_fixture(tmp_path, wrong_digest=True)
    with pytest.raises(SingleRoomSurfaceError, match="view 0 digest mismatch"):
        prepare_input(
            consensus_dir=consensus,
            alignment_dir=alignment,
            room_name="living",
            owner=3,
            output_dir=tmp_path / "output",
        )


def test_rejects_missing_source_binding_hash(tmp_path: Path) -> None:
    consensus, alignment = _write_alignment_fixture(tmp_path, missing_manifest_sha=True)
    with pytest.raises(SingleRoomSurfaceError, match="source output manifest binding requires path and sha256"):
        prepare_input(
            consensus_dir=consensus,
            alignment_dir=alignment,
            room_name="living",
            owner=3,
            output_dir=tmp_path / "output",
        )


def test_rejects_missing_source_raw_root_binding(tmp_path: Path) -> None:
    consensus, alignment = _write_alignment_fixture(tmp_path, missing_raw_root=True)
    with pytest.raises(SingleRoomSurfaceError, match="source raw root binding is missing"):
        prepare_input(
            consensus_dir=consensus,
            alignment_dir=alignment,
            room_name="living",
            owner=3,
            output_dir=tmp_path / "output",
        )


def test_rejects_wrong_alignment_target_camera(tmp_path: Path) -> None:
    consensus, alignment = _write_alignment_fixture(tmp_path, target_camera_id="kitchen")
    with pytest.raises(SingleRoomSurfaceError, match="target camera id 'kitchen'.*'living-room'"):
        prepare_input(
            consensus_dir=consensus,
            alignment_dir=alignment,
            room_name="living",
            owner=3,
            output_dir=tmp_path / "output",
        )


def test_rejects_missing_alignment_world_frame_revision(tmp_path: Path) -> None:
    consensus, alignment = _write_alignment_fixture(
        tmp_path,
        target_world_frame_revision=None,
        payload_world_frame_revision=None,
    )
    with pytest.raises(SingleRoomSurfaceError, match="target has no world frame revision"):
        prepare_input(
            consensus_dir=consensus,
            alignment_dir=alignment,
            room_name="living",
            owner=3,
            output_dir=tmp_path / "output",
        )


def test_rejects_mismatched_world_frame_revision_binding(tmp_path: Path) -> None:
    consensus, alignment = _write_alignment_fixture(
        tmp_path,
        payload_world_frame_revision="wrong-world-revision",
    )
    with pytest.raises(SingleRoomSurfaceError, match="world frame revision disagrees"):
        prepare_input(
            consensus_dir=consensus,
            alignment_dir=alignment,
            room_name="living",
            owner=3,
            output_dir=tmp_path / "output",
        )


def test_rejects_mismatched_camera_frame_binding(tmp_path: Path) -> None:
    consensus, alignment = _write_alignment_fixture(tmp_path)
    transform_path = alignment / "phone_ma_to_noesis_world.json"
    transform = json.loads(transform_path.read_text())
    transform["target_binding"]["camera_frame_binding"]["revision"] = "wrong-camera-frame"
    transform_path.write_text(json.dumps(transform) + "\n")
    with pytest.raises(SingleRoomSurfaceError, match="camera frame binding disagrees"):
        prepare_input(
            consensus_dir=consensus,
            alignment_dir=alignment,
            room_name="living",
            owner=3,
            output_dir=tmp_path / "output",
        )


def test_excludes_single_view_points_and_computes_full_height(tmp_path: Path) -> None:
    consensus, alignment = _write_alignment_fixture(tmp_path)
    _write_surfels(consensus)
    output = prepare_input(
        consensus_dir=consensus,
        alignment_dir=alignment,
        room_name="living-room",
        owner=3,
        output_dir=tmp_path / "output",
        ceiling_mode="full_height",
        full_height_margin_m=0.07,
    )
    with np.load(output["output_npz"], allow_pickle=False) as arrays:
        assert arrays.files == [
            "points",
            "colors",
            "owner_room_id",
            "view_count",
            "geometry_status",
            "living_camera_positions",
        ]
        assert arrays["points"].shape == (2, 3)
        assert np.all(arrays["owner_room_id"] == 3)
    scene = trimesh.load(output["withheld_output"], force="scene", process=False)
    point_count = sum(len(geometry.vertices) for geometry in scene.geometry.values())
    assert point_count == 1
    manifest = output["manifest"]
    assert manifest["support_metrics"]["accepted_mesh_input_excludes_withheld_single_view"] is True
    assert manifest["support_metrics"]["withheld_single_view_count"] == 1
    assert manifest["support_metrics"]["unknown_count"] is None
    assert manifest["support_metrics"]["unknown_count_status"] == "not_assessed"
    assert manifest["support_metrics"]["contradiction_count"] is None
    assert manifest["sources"]["living"]["camera_id"] == "living-room"
    assert manifest["ceiling"]["finite_measured_upper_y_m"] == 2.0
    assert manifest["ceiling"]["effective_ceiling_cutaway_m"] == pytest.approx(2.07)


def test_rejects_any_non_single_view_withheld_support(tmp_path: Path) -> None:
    consensus, alignment = _write_alignment_fixture(tmp_path)
    _write_surfels(consensus)
    np.savez_compressed(
        consensus / "surfel_points_single_view.npz",
        points=np.asarray([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]], dtype=np.float32),
        colors=np.asarray([[70, 80, 90], [71, 81, 91]], dtype=np.uint8),
        support_view_count=np.asarray([1, 2], dtype=np.uint16),
    )
    with pytest.raises(SingleRoomSurfaceError, match="not all single-view support"):
        prepare_input(
            consensus_dir=consensus,
            alignment_dir=alignment,
            room_name="living",
            owner=3,
            output_dir=tmp_path / "output",
        )
