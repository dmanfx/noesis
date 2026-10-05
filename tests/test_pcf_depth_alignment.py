"""Offline PCF scale review must never turn background depth into person depth."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from noesis.calibration.pcf_depth_alignment import (
    AlignmentError, CONTRACT, EXCLUSIONS_CONTRACT, OCCUPIED_CONTRACT, OPTICAL_Z,
    _binding, _exclusion_masks, project_pcf_optical_z, review_static_scale,
    scale_occupied_person_samples,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _record(path: Path, root: Path) -> dict:
    return {"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _bindings() -> tuple[dict, dict]:
    source = {"frame_id": "backend_world_m", "revision": "source-rev"}
    target = {"frame_id": "backend_world_m", "revision": "pcf-rev"}
    floor = {"normal": [0, 1, 0], "offset_m": 0}
    transform = np.eye(4).reshape(-1, order="F").tolist()
    catalog_binding = {
        "source_camera_calibration_sha256": "a" * 64,
        "source_world_alignment_sha256": "b" * 64,
        "source_frame": source, "target_frame": target,
        "target_from_source_col_major": transform,
        "target_from_source_sha256": "c" * 64,
        "target_revision_id": "target-rev",
        "source_floor_plane": floor, "target_floor_plane": floor,
    }
    static_binding = {
        "camera_calibration_sha256": "a" * 64,
        "world_alignment_sha256": "b" * 64,
        "calibration_frame": source, "world_frame": target,
        "target_from_calibration_col_major": transform,
        "transform_sha256": "c" * 64,
        "target_revision_id": "target-rev",
        "calibration_floor_plane": floor, "world_floor_plane": floor,
        "scene_prior_id": "pcf-rev",
    }
    return catalog_binding, static_binding


def test_pcf_projection_returns_optical_z_not_ray_range() -> None:
    K = np.array([[10, 0, 5], [0, 10, 5], [0, 0, 1]], dtype=float)
    depth = project_pcf_optical_z(np.array([[0, 0, 4], [2, 0, 4]]), np.eye(4), K, (11, 11))
    assert depth[5, 5] == 4
    assert depth[5, 10] == 4  # range at this pixel would be sqrt(20)
    reflection = np.eye(4)
    reflection[0, 0] = -1
    with pytest.raises(AlignmentError):
        project_pcf_optical_z(np.array([[0, 0, 4]]), reflection, K, (11, 11))


def test_frame_binding_requires_exact_revision_and_transform() -> None:
    catalog_binding, static_binding = _bindings()
    catalog = {"camera_bindings": [{"camera_id": "living-room", "mode": "shadow",
                                    "prior_id": "pcf-rev", "frame_binding": catalog_binding}]}
    static = {"camera_id": "living-room", "world_frame": "backend_world_m",
              "world_frame_revision": "pcf-rev", "status": "complete", "reference_kind": "paired_static",
              "camera_frame_binding": static_binding,
              "source": {"frames": [{"depth_semantics":
                  "model_metric_range_on_captured_calibrated_rays_converted_to_z"}] * 4}}
    assert _binding(catalog, static, "living-room")[1] == "pcf-rev"
    static_binding["transform_sha256"] = "d" * 64
    with pytest.raises(AlignmentError, match="binding disagrees"):
        _binding(catalog, static, "living-room")


def test_exclusions_are_bound_to_each_exact_source_frame() -> None:
    frame = {"raw_artifact": "static_0000.npz", "source_frame_index": 19,
             "raw_frame_sha256": "a" * 64}
    static = {"camera_id": "living-room", "source": {"frames": [frame]}}
    exclusions = {"contract": EXCLUSIONS_CONTRACT, "camera_id": "living-room",
                  "static_manifest_sha256": "b" * 64,
                  "frames": [{**frame, "rectangles_xyxy": [[1, 1, 3, 3]]}]}
    mask = _exclusion_masks(exclusions, static, "b" * 64, (4, 4))["static_0000.npz"]
    assert int(mask.sum()) == 4
    exclusions["frames"][0]["raw_frame_sha256"] = "c" * 64
    with pytest.raises(AlignmentError, match="identity mismatch"):
        _exclusion_masks(exclusions, static, "b" * 64, (4, 4))


def test_occupied_application_uses_only_exact_person_sample() -> None:
    report = {"contract": CONTRACT, "runtime_eligible": False, "fit": {"scale": 1.2},
              "camera_id": "living-room", "world_frame": "backend_world_m",
              "world_frame_revision": "pcf-rev", "camera_calibration_sha256": "a" * 64,
              "world_transform_sha256": "c" * 64}
    row = {"sample_id": "person-1", "anchor_source": "torso_core",
           "support_kind": "current_person_silhouette_or_pose_capsule",
           "person_support_valid": True, "person_support_count": 12,
           "person_support_sha256": "d" * 64,
           "source_frame_id": "frame-1", "mapanything_frame_id": "frame-1", "dav2_frame_id": "frame-1",
           "source_frame_index": 15, "mapanything_frame_index": 15, "dav2_frame_index": 15,
           "source_encoded_pts_us": 500000, "mapanything_encoded_pts_us": 500000,
           "dav2_encoded_pts_us": 500000,
           "anchor_uv_normalized": [0.3, 0.4], "mapanything_anchor_uv_normalized": [0.3, 0.4],
           "dav2_anchor_uv_normalized": [0.3, 0.4],
           "raw_dav2_optical_z_m": 2.0, "mapanything_person_optical_z_m": 3.0}
    document = {**{k: report[k] for k in (
        "camera_id", "world_frame", "world_frame_revision", "camera_calibration_sha256",
        "world_transform_sha256")}, "contract": OCCUPIED_CONTRACT,
        "optical_depth_space": OPTICAL_Z, "samples": [row]}
    pair = scale_occupied_person_samples(report, document)[0]
    assert pair["target_mapanything_pcf_scaled_optical_z_m"] == pytest.approx(3.6)
    assert "pcf_depth_m" not in pair
    row["support_kind"] = "empty_room_background"
    with pytest.raises(AlignmentError, match="person support"):
        scale_occupied_person_samples(report, document)
    row["support_kind"] = "current_person_silhouette_or_pose_capsule"
    row["dav2_encoded_pts_us"] += 1
    with pytest.raises(AlignmentError, match="PTS mismatch"):
        scale_occupied_person_samples(report, document)


def test_synthetic_static_review_uses_disjoint_frames_and_people_exclusion(tmp_path: Path) -> None:
    catalog_binding, static_binding = _bindings()
    catalog_path = tmp_path / "catalog.json"
    static_root = tmp_path / "static"
    static_root.mkdir()
    scene_root = tmp_path / "revisions" / "pcf-rev"
    scene_root.mkdir(parents=True)
    h = w = 72
    K = np.array([[72, 0, 35.5], [0, 72, 35.5], [0, 0, 1]], dtype=float)
    yy, xx = np.indices((h, w))
    z = np.full((h, w), 4.0, dtype=np.float32)
    points = np.stack(((xx - K[0, 2]) * z / K[0, 0],
                       (yy - K[1, 2]) * z / K[1, 1], z), axis=-1).reshape(-1, 3)
    pcf_path = scene_root / "points.glb"
    trimesh.points.PointCloud(points).export(pcf_path)
    scene = {"prior_id": "pcf-rev", "quality": {"passed": True,
             "selected_point_count": len(points)}, "artifacts": [_record(pcf_path, scene_root)]}
    scene_path = scene_root / "manifest.json"
    _write_json(scene_path, scene)
    catalog = {"camera_bindings": [{"camera_id": "living-room", "mode": "shadow",
                "prior_id": "pcf-rev", "frame_binding": catalog_binding}],
                "revisions": [{"prior_id": "pcf-rev", **_record(scene_path, tmp_path)}]}
    _write_json(catalog_path, catalog)
    fused = static_root / "fused_depth.npz"
    np.savez_compressed(fused, intrinsics=K, camera_to_world=np.eye(4), depth_z=z)
    files = [_record(fused, static_root)]
    frames = []
    excluded = []
    for index in range(6):
        name = f"static_{index:04d}.npz"
        model = np.full((h, w), 3.7, dtype=np.float32)
        model[24:48, 24:48] = 1.0  # Person-sized depth must not enter the static fit.
        path = static_root / name
        np.savez_compressed(path, depth_z=model, mask=np.ones((h, w), dtype=bool),
                            confidence=np.ones((h, w)), intrinsics=K,
                            source_frame_index=np.array(index),
                            encoded_pts_ns=np.array(index * 1_000_000_000))
        files.append(_record(path, static_root))
        frame = {"raw_artifact": name, "source_frame_index": index,
                 "encoded_pts_ns": str(index * 1_000_000_000),
                 "raw_frame_sha256": f"{index + 1:064x}",
                 "depth_semantics": "model_metric_range_on_captured_calibrated_rays_converted_to_z"}
        frames.append(frame)
        excluded.append({"raw_artifact": name, "source_frame_index": index,
                         "raw_frame_sha256": frame["raw_frame_sha256"],
                         "rectangles_xyxy": [[24, 24, 48, 48]]})
    static = {"camera_id": "living-room", "world_frame": "backend_world_m",
              "world_frame_revision": "pcf-rev", "status": "complete", "reference_kind": "paired_static",
              "revision_id": "static-rev", "camera_frame_binding": static_binding,
              "source": {"frames": frames}, "files": files}
    static_path = static_root / "manifest.json"
    _write_json(static_path, static)
    exclusions_path = tmp_path / "exclusions.json"
    _write_json(exclusions_path, {"contract": EXCLUSIONS_CONTRACT, "camera_id": "living-room",
                "static_manifest_sha256": _record(static_path, static_root)["sha256"], "frames": excluded})
    report = review_static_scale(catalog_path, static_path, exclusions_path, "living-room")
    assert report["runtime_eligible"] is False
    assert report["fit"]["scale"] == pytest.approx(4 / 3.7, rel=1e-5)
    assert len(report["holdout"]) == 3
    assert all(row["after"]["median_abs_error_m"] < 1e-4 for row in report["holdout"])
