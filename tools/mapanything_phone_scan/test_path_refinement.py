from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from tools.mapanything_phone_scan.prepared_frame_identity import prepared_frame_identity
from tools.mapanything_phone_scan.trajectory_motion_review import _load_provider
from tools.mapanything_phone_scan.path_refinement import refine_review_path


POSE_CONVENTION = "opencv_cam2world_x_right_y_down_z_forward"


def _fixture(tmp_path: Path, count: int = 10) -> tuple[Path, dict, dict, Path, dict[str, bytes]]:
    scan = tmp_path / "scan"
    frames_root = scan / "frames"
    raw_root = scan / "outputs" / "raw"
    frames_root.mkdir(parents=True)
    raw_root.mkdir(parents=True)
    prepared_rows = []
    poses = []
    raw_before: dict[str, bytes] = {}
    for index in range(count):
        image = np.zeros((16, 20, 3), dtype=np.uint8)
        cv2.circle(image, (3 + (index * 2) % 14, 8), 3, (220, 120, 40), -1)
        image_path = frames_root / f"frame_{index:04d}.jpg"
        assert cv2.imwrite(str(image_path), image)
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        prepared_rows.append(
            {
                "index": index,
                "frame": f"frames/{image_path.name}",
                "frame_id": prepared_frame_identity(index, digest),
                "timestamp_s": float(index),
                "capture_time_ns": (index + 1) * 1_000,
                "source_frame_index": index,
                "sha256": digest,
            }
        )
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = index * 0.05
        poses.append(pose.tolist())
        np.savez_compressed(
            raw_root / f"view_{index:04d}.npz",
            world_points=np.zeros((16, 20, 3), dtype=np.float32),
            depth_z=np.ones((16, 20), dtype=np.float32),
            confidence=np.ones((16, 20), dtype=np.float32),
            mask=np.ones((16, 20), dtype=np.uint8),
            camera_pose=pose,
            intrinsics=np.eye(3, dtype=np.float32),
            model_rgb=image,
        )
        raw_before[f"view_{index:04d}.npz"] = (raw_root / f"view_{index:04d}.npz").read_bytes()

    (scan / "prepared_frames_manifest.json").write_text(
        json.dumps({"frames": prepared_rows}), encoding="utf-8"
    )
    trajectory = scan / "outputs" / "camera_trajectory.json"
    trajectory.write_text(
        json.dumps(
            {
                "schema": "noesis.da3.phone_scan.camera_trajectory.v1",
                "provider": "da3",
                "coordinate_frame": "da3_metric_world_unaligned_to_noesis",
                "pose_convention": POSE_CONVENTION,
                "camera_to_world": poses,
            }
        ),
        encoding="utf-8",
    )
    provider_manifest = scan / "outputs" / "scan_outputs_manifest.json"
    provider_manifest.write_text(
        json.dumps(
            {
                "schema": "noesis.da3.phone_scan.outputs.v1",
                "provider": "da3",
                "model_id": "fixture/da3",
                "coordinate_frame": "da3_metric_world_unaligned_to_noesis",
                "pose_convention": POSE_CONVENTION,
                "view_count": count,
                "window_count": 1,
                "anchor_view_index": None,
                "artifacts": {"trajectory_json": "outputs/camera_trajectory.json"},
                "frames": [
                    {
                        "index": index,
                        "source_frame": row["frame"],
                        "timestamp_s": row["timestamp_s"],
                        "fixed_camera_anchor": False,
                        "adaptive_window_index": 0,
                    }
                    for index, row in enumerate(prepared_rows)
                ],
            }
        ),
        encoding="utf-8",
    )
    source = _load_provider(scan, provider_manifest, allow_partial=False)
    state = {"id": "path-scan", "vio": {"status": "failed", "error": "not run"}}
    return scan, source, state, raw_root, raw_before


def _install_accepted_visual_edges(monkeypatch, *, count: int = 10) -> None:
    import tools.mapanything_phone_scan.build_consensus_fusion as fusion
    import tools.mapanything_phone_scan.trajectory_refinement as refinement

    monkeypatch.setattr(
        refinement,
        "_retrieve_nonadjacent_pairs",
        lambda _images, _settings: [
            {"source_view": 0, "target_view": 6, "retrieval_score": 1.0, "withheld_from_fit": False},
            {"source_view": 7, "target_view": count - 1, "retrieval_score": 1.0, "withheld_from_fit": True},
        ],
    )

    def verify(source_index, target_index, *_args):
        transform = np.eye(4, dtype=np.float64)
        transform[0, 3] = (target_index - source_index) * 0.05
        return (
            {
                "source_view": source_index,
                "target_view": target_index,
                "status": "accepted",
                "rejection_reason": None,
                "match_count": 32,
                "two_d_inlier_count": 24,
                "three_d_match_count": 16,
                "three_d_geometry": {"residual_p80_m": 0.01},
            },
            {
                "source_view": source_index,
                "target_view": target_index,
                "transform": transform.tolist(),
                "label": "verified_visual_loop",
                "translation_sigma_m": 0.03,
                "rotation_sigma_deg": 1.0,
            },
        )

    monkeypatch.setattr(refinement, "_verify_visual_revisit", verify)

    def pose_graph(first, _second, edges, *, single_carrier=False, single_carrier_name="da3"):
        assert single_carrier is True
        assert single_carrier_name == "da3"
        relative = np.stack([np.linalg.inv(first[0]) @ pose for pose in first])
        return relative, {"solver_success": True, "edge_count": len(edges)}

    monkeypatch.setattr(fusion, "_pose_graph", pose_graph)


def test_accepted_path_executes_existing_engine_and_returns_poses(tmp_path: Path, monkeypatch) -> None:
    scan, source, state, raw_root, raw_before = _fixture(tmp_path)
    _install_accepted_visual_edges(monkeypatch)

    result = refine_review_path(scan, source, state, tmp_path / "review")

    assert result["status"] == "accepted"
    poses = result["accepted_camera_poses"]
    assert isinstance(poses, np.ndarray)
    assert poses.shape == (10, 4, 4)
    report = result["report"]
    assert report["accepted_constraint_count"] == 1
    assert report["accepted_visual_constraint_count"] == 1
    assert report["accepted_vio_constraint_count"] == 0
    assert report["refinement"]["withheld_evaluation"]["fit_excluded"] is True
    assert tuple(report["settings"]["withheld_ranges"]) == ((7, 10),)
    assert report["vio_constraint_admission"]["status"] == "missing_admission"
    assert {path.name: path.read_bytes() for path in raw_root.iterdir()} == raw_before


def test_static_scale_revalidation_callback_reaches_existing_engine(tmp_path: Path, monkeypatch) -> None:
    import tools.mapanything_phone_scan.path_refinement as adapter
    scan, source, state, _, _ = _fixture(tmp_path)
    _install_accepted_visual_edges(monkeypatch)
    original = adapter.run_trajectory_refinement
    callback = lambda *_args: None
    seen = []
    def run(*args, **kwargs):
        seen.append(kwargs["revalidate_scaled_carrier"])
        return original(*args, **kwargs)
    monkeypatch.setattr(adapter, "run_trajectory_refinement", run)
    refine_review_path(scan, source, state, tmp_path / "review", revalidate_scaled_carrier=callback)
    assert seen == [callback]


def test_qualified_vio_is_passed_to_existing_constraint_adapter(tmp_path: Path, monkeypatch) -> None:
    scan, source, state, _raw_root, _raw_before = _fixture(tmp_path)
    _install_accepted_visual_edges(monkeypatch)
    frame_ids = [row["frame_id"] for row in source["prepared"]["frames"]]
    poses = []
    for index, frame_id in enumerate(frame_ids):
        pose = np.eye(4).tolist()
        pose[0][3] = index * 0.05
        poses.append(
            {
                "capture_time_ns": (index + 1) * 1_000,
                "prepared_frame_id": frame_id,
                "T_vio_world_camera": pose,
                "velocity_mps": [0, 0, 0],
                "gravity_mps2": [0, 0, -9.81],
                "gyro_bias_rads": [0, 0, 0],
                "accel_bias_mps2": [0, 0, 0],
                "covariance": np.eye(6).tolist(),
                "segment_id": "segment-0",
            }
        )
    vio = {
        "schema": "noesis.phone_capture.vio_result.v1",
        "estimator": "openvins",
        "accepted_for_metric_vio": True,
        "frame": {
            "source": "camera", "target": "vio_world", "pose_convention": "T_vio_world_camera",
            "units": "meters", "capture_id": "capture", "camera_sensor_id": "camera0",
            "time_domain": "camera2_hardware", "camera_axes": "x_right_y_down_z_forward",
            "world_axes": "z_up_gravity_up", "pose_origin": "camera_optical_center",
            "velocity_origin": "imu_center", "gravity_frame": "vio_world",
            "gravity_semantics": "physical_world_acceleration",
        },
        "scale": {"mode": "metric", "source": "imu_camera_calibration"},
        "covariance_frame": "camera_pose_tangent_se3_row_major",
        "covariance_tangent_frame": "vio_world_rotation_additive_position",
        "poses": poses,
        "quality": {"initialized": True, "tracking_ratio": 1.0},
        "segments": [{"id": "segment-0", "reset": False}],
    }
    vio_path = scan / "vio" / "run" / "vio_result.json"
    vio_path.parent.mkdir(parents=True)
    vio_path.write_text(json.dumps(vio), encoding="utf-8")
    state["vio"] = {"status": "complete", "results": {"artifact": "vio/run/vio_result.json"}}

    result = refine_review_path(scan, source, state, tmp_path / "review")

    assert result["status"] == "accepted"
    assert result["report"]["accepted_vio_constraint_count"] == 9
    assert result["report"]["vio_constraint_admission"]["status"] == "accepted"
    assert result["report"]["constraint_contributions"]["qualified_vio_relative_constraints"] == 9
    assert "qualified_vio_relative_constraints" in result["report"]["position_refinement_algorithm"]


def test_rejected_visual_gates_do_not_rewrite_original_raw_or_return_poses(tmp_path: Path, monkeypatch) -> None:
    scan, source, state, raw_root, raw_before = _fixture(tmp_path)
    import tools.mapanything_phone_scan.trajectory_refinement as refinement

    monkeypatch.setattr(refinement, "_retrieve_nonadjacent_pairs", lambda _images, _settings: [])

    result = refine_review_path(scan, source, state, tmp_path / "review")

    assert result["status"] == "rejected"
    assert result["accepted_camera_poses"] is None
    assert result["report"]["accepted_constraint_count"] == 0
    assert result["report"]["refinement"]["raw_materialized"] is False
    assert not (tmp_path / "review" / "raw").exists()
    assert {path.name: path.read_bytes() for path in raw_root.iterdir()} == raw_before


def test_missing_raw_is_reported_without_rewrite_and_missing_vio_is_exact(tmp_path: Path) -> None:
    scan, source, state, raw_root, _raw_before = _fixture(tmp_path)
    (raw_root / "view_0009.npz").unlink()
    state["vio"] = {"status": "failed", "error": "calibration incomplete"}

    result = refine_review_path(scan, source, state, tmp_path / "review")

    assert result["status"] == "needs_evidence"
    assert result["accepted_camera_poses"] is None
    assert "DA3 raw view 9 is missing" in result["report"]["rejection_reason"]
    assert result["report"]["vio_constraint_admission"]["status"] == "missing_admission"
    assert "status is 'failed'" in result["report"]["vio_constraint_admission"]["reason"]


def test_provider_manifest_hash_change_is_rejected_before_refinement(tmp_path: Path) -> None:
    scan, source, state, _raw_root, _raw_before = _fixture(tmp_path)
    provider_manifest = scan / "outputs" / "scan_outputs_manifest.json"
    provider_manifest.write_text(provider_manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    result = refine_review_path(scan, source, state, tmp_path / "review")

    assert result["status"] == "needs_evidence"
    assert "exact evidence binding failed" in result["report"]["rejection_reason"]
    assert result["accepted_camera_poses"] is None


def test_partial_provider_passes_exact_original_ids_to_existing_engine(tmp_path: Path, monkeypatch) -> None:
    scan, _source, state, raw_root, _raw_before = _fixture(tmp_path)
    provider_manifest = scan / "outputs" / "scan_outputs_manifest.json"
    payload = json.loads(provider_manifest.read_text(encoding="utf-8"))
    payload["frames"].pop(7)
    for index, row in enumerate(payload["frames"]):
        row["index"] = index
    payload["view_count"] = 9
    payload["partial_reconstruction"] = {
        "included_global_indices": [0, 1, 2, 3, 4, 5, 6, 8, 9],
        "omitted_global_indices": [7],
        "parent_view_count": 10,
    }
    provider_manifest.write_text(json.dumps(payload), encoding="utf-8")
    trajectory = scan / "outputs" / "camera_trajectory.json"
    trajectory_payload = json.loads(trajectory.read_text(encoding="utf-8"))
    trajectory_payload["camera_to_world"].pop(7)
    trajectory.write_text(json.dumps(trajectory_payload), encoding="utf-8")
    source = _load_provider(scan, provider_manifest, allow_partial=True)
    # Provider output index 7/8 now represent original prepared indices 8/9.
    for output_index, prepared_index in ((7, 8), (8, 9)):
        with np.load(raw_root / f"view_{output_index:04d}.npz", allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
        arrays["camera_pose"] = np.asarray(_source["poses"][prepared_index], dtype=np.float32)
        np.savez_compressed(raw_root / f"view_{output_index:04d}.npz", **arrays)
    _install_accepted_visual_edges(monkeypatch, count=9)

    result = refine_review_path(scan, source, state, tmp_path / "review")

    assert result["status"] == "accepted"
    assert result["accepted_camera_poses"].shape == (9, 4, 4)
    assert result["report"]["input_budget"]["partial_provider"] is True
    assert result["report"]["input_budget"]["prepared_indices"] == [0, 1, 2, 3, 4, 5, 6, 8, 9]
    assert result["report"]["provenance"]["included_prepared_indices"] == [0, 1, 2, 3, 4, 5, 6, 8, 9]
    assert result["report"]["constraints"][0]["source_frame_id"] == source["prepared"]["frames"][0]["frame_id"]
