"""Focused checks for retained phone-carrier structural point evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from noesis.calibration.structural_carrier_evidence import (
    project_carrier_image_point, verify_carrier_image_point,
)
from noesis.calibration.structural_registration import StructuralRegistrationError


def _ref(path: Path, root: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size}


def _point_fixture(root: Path) -> tuple[dict, dict, dict]:
    manifest = root / "source_manifest.json"
    manifest.write_text(json.dumps({
        "schema": "noesis.phone_walk.static_world_point_cloud_review.v1",
        "source_coordinate_frame": "da3_relative_metric_world",
        "output_coordinate_frame": "backend_world_m_stream_points",
        "static_reference_points_included": False,
        "alignment": {"method": "all_phone_pose_correspondence_sim3_to_validated_da3_backend_path",
                      "rotation_row_major": np.eye(3).tolist(), "scale": 1.0,
                      "translation": [0, 0, 0]},
    }))
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb[:, :, 0] = 135
    image = root / "review.png"
    Image.fromarray(rgb).save(image)
    view = root / "view_0001.npz"
    K = np.asarray([[10, 0, 5], [0, 10, 5], [0, 0, 1]], dtype=np.float32)
    np.savez(view, model_rgb=rgb, intrinsics=K, camera_pose=np.eye(4, dtype=np.float32))
    point = {"point_id": "reviewed_jamb", "kind": "opening", "xz": [0.0, 5.0],
             "source_wall_ids": ["measured_wall"], "evidence_refs": ["carrier_view:0001"]}
    binding = {"world_frame": {"revision": "prior-v1"}, "transform_sha256": "a" * 64,
               "camera_calibration_sha256": "b" * 64}
    evidence = {"kind": "retained_carrier_pcf_roi_plane", "point": point,
                "view_index": 1, "pixel_uv": [5, 5],
                "pixel_bounds": [[4.5, 4.5], [5.5, 5.5]],
                "segment_id": "measured_wall", "identity_review": "visible jamb edge",
                "image_file": _ref(image, root),
                "image_evidence": {"pixel_uv": [5, 5], "image": {"width": 10, "height": 10},
                                   "wall_plane": {"segment_id": "measured_wall"}, "surface_y_m": None},
                "projection": {"view": _ref(view, root), "source_manifest": _ref(manifest, root),
                               "normal_xyz": [0, 0, 1], "offset_m": -5.0,
                               "reference_y_m": 1.0},
                "target_world_revision": "prior-v1", "target_transform_sha256": "a" * 64,
                "target_calibration_sha256": "b" * 64}
    return evidence, point, binding


def test_carrier_ray_replays_frozen_point_and_annotation_variant(tmp_path: Path) -> None:
    evidence, point, binding = _point_fixture(tmp_path)
    verify_carrier_image_point(evidence, point, binding, tmp_path)
    varied = project_carrier_image_point(evidence, [6, 5], point, tmp_path)
    assert varied["point_id"] == point["point_id"]
    assert varied["xz"] == pytest.approx([0.5, 5.0])
    assert point["xz"] == [0.0, 5.0]


def test_carrier_point_rejects_changed_raster_and_frame(tmp_path: Path) -> None:
    evidence, point, binding = _point_fixture(tmp_path)
    bad_binding = {**binding, "world_frame": {"revision": "another-prior"}}
    with pytest.raises(StructuralRegistrationError, match="revision"):
        verify_carrier_image_point(evidence, point, bad_binding, tmp_path)
    Image.fromarray(np.full((10, 10, 3), 240, dtype=np.uint8)).save(tmp_path / "review.png")
    with pytest.raises(StructuralRegistrationError, match="bytes differ"):
        verify_carrier_image_point(evidence, point, binding, tmp_path)


def test_carrier_point_rejects_guessed_coordinate(tmp_path: Path) -> None:
    evidence, point, binding = _point_fixture(tmp_path)
    guessed = {**point, "xz": [0.3, 5.0]}
    evidence["point"] = guessed
    with pytest.raises(StructuralRegistrationError, match="does not replay"):
        verify_carrier_image_point(evidence, guessed, binding, tmp_path)
