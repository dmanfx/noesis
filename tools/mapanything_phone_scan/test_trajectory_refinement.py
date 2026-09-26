from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.mapanything_phone_scan import app as phone_app
from tools.mapanything_phone_scan.prepared_frame_identity import prepared_frame_identity
from tools.mapanything_phone_scan.vio import VIOSettings
from tools.mapanything_phone_scan.trajectory_refinement import (
    TrajectoryRefinementError,
    TrajectoryRefinementSettings,
    _fit_3d_rigid,
    _materialize_refined_raw,
    _retrieve_nonadjacent_pairs,
    _raw_gray,
    _source_identity,
    run_trajectory_refinement,
)


def _vio_payload(
    frame_count: int, *, step_m: float, frame_ids: list[str] | None = None
) -> dict:
    poses = []
    for index in range(frame_count):
        pose = np.eye(4).tolist()
        pose[0][3] = index * step_m
        poses.append(
            {
                "capture_time_ns": (index + 1) * 10,
                "prepared_frame_id": (
                    frame_ids[index] if frame_ids is not None else f"frame-{index}"
                ),
                "T_vio_world_camera": pose,
                "velocity_mps": [0, 0, 0],
                "gravity_mps2": [0, 0, -9.81],
                "gyro_bias_rads": [0, 0, 0],
                "accel_bias_mps2": [0, 0, 0],
                "covariance": np.eye(6).tolist(),
                "segment_id": "segment-0",
            }
        )
    return {
        "schema": "noesis.phone_capture.vio_result.v1",
        "estimator": "openvins",
        "accepted_for_metric_vio": True,
        "frame": {
            "source": "camera",
            "target": "vio_world",
            "pose_convention": "T_vio_world_camera",
            "units": "meters",
            "capture_id": "capture",
            "camera_sensor_id": "camera0",
            "time_domain": "camera2_hardware",
            "camera_axes": "x_right_y_down_z_forward",
            "world_axes": "z_up_gravity_up",
            "pose_origin": "camera_optical_center",
            "velocity_origin": "imu_center",
            "gravity_frame": "vio_world",
            "gravity_semantics": "physical_world_acceleration",
        },
        "scale": {"mode": "metric", "source": "imu_camera_calibration"},
        "covariance_frame": "camera_pose_tangent_se3_row_major",
        "covariance_tangent_frame": "vio_world_rotation_additive_position",
        "poses": poses,
        "quality": {"initialized": True, "tracking_ratio": 1.0},
        "segments": [{"id": "segment-0", "reset": False}],
    }


def test_nonadjacent_retrieval_is_bounded_and_respects_gap() -> None:
    images = [np.full((24, 32), index * 11, dtype=np.uint8) for index in range(12)]
    settings = TrajectoryRefinementSettings(
        min_nonadjacent_gap=4,
        max_candidate_pairs=5,
        withheld_ranges=((8, 12),),
    )
    rows = _retrieve_nonadjacent_pairs(images, settings)
    assert len(rows) == 5
    assert all(row["target_view"] - row["source_view"] >= 4 for row in rows)
    assert any(row["withheld_from_fit"] for row in rows)


def test_depth_backed_rigid_fit_requires_real_geometry() -> None:
    rng = np.random.default_rng(4)
    source = rng.normal(size=(32, 3))
    rotation = cv2.Rodrigues(np.asarray([0.08, -0.03, 0.04]))[0]
    translation = np.asarray([0.20, -0.10, 0.05])
    target = (rotation @ source.T).T + translation
    target += rng.normal(scale=0.005, size=target.shape)
    transform, metrics = _fit_3d_rigid(
        source,
        target,
        TrajectoryRefinementSettings(
            min_3d_matches=8,
            min_3d_inliers=8,
            min_3d_inlier_fraction=0.7,
            max_3d_residual_m=0.05,
            max_3d_p80_residual_m=0.05,
        ),
        seed=9,
    )
    assert transform is not None
    assert metrics["status"] == "ok"
    np.testing.assert_allclose(transform[:3, :3], rotation, atol=0.03)
    np.testing.assert_allclose(transform[:3, 3], translation, atol=0.03)

    degenerate, degenerate_metrics = _fit_3d_rigid(
        np.zeros((8, 3)),
        np.zeros((8, 3)),
        TrajectoryRefinementSettings(min_3d_matches=8, min_3d_inliers=8),
        seed=9,
    )
    assert degenerate is None
    assert degenerate_metrics["status"] == "insufficient_3d_inliers"


def test_refinement_keeps_ordinary_da3_path_when_no_revisit_exists(tmp_path: Path, monkeypatch) -> None:
    scan = tmp_path / "scan"
    (scan / "frames").mkdir(parents=True)
    frames = []
    for index in range(2):
        image_path = scan / "frames" / f"frame_{index:04d}.jpg"
        cv2.imwrite(str(image_path), np.full((16, 24), 80 + index, dtype=np.uint8))
        frames.append(
            {
                "index": index,
                "frame": f"frames/frame_{index:04d}.jpg",
                "timestamp_s": float(index),
                "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            }
        )
    (scan / "prepared_frames_manifest.json").write_text(
        json.dumps({"frames": frames}), encoding="utf-8"
    )
    raw = scan / "outputs" / "raw"
    raw.mkdir(parents=True)
    for index in range(2):
        np.savez_compressed(
            raw / f"view_{index:04d}.npz",
            world_points=np.zeros((4, 6, 3), dtype=np.float32),
            depth_z=np.ones((4, 6), dtype=np.float32),
            confidence=np.ones((4, 6), dtype=np.float32),
            mask=np.ones((4, 6), dtype=np.uint8),
            camera_pose=np.eye(4, dtype=np.float32),
            intrinsics=np.eye(3, dtype=np.float32),
            model_rgb=np.zeros((4, 6, 3), dtype=np.float32),
        )
    import tools.mapanything_phone_scan.trajectory_refinement as refinement_module

    matching_calls = []

    def inspect_matching_projection(source, target, source_image, target_image, *_args):
        # Prepared captures above are nonzero; retained RGB on the depth grid
        # is zero. Sampling the resized capture would bind different pixels.
        assert not np.any(source_image)
        assert not np.any(target_image)
        matching_calls.append((source, target))
        return {"rejection_reason": "missing_descriptors"}, None

    monkeypatch.setattr(refinement_module, "_verify_visual_revisit", inspect_matching_projection)
    report = run_trajectory_refinement(
        scan, raw, tmp_path / "refinement",
        TrajectoryRefinementSettings(min_nonadjacent_gap=1, max_candidate_pairs=1),
    )
    assert matching_calls == [(0, 1)]
    assert report["refinement"]["status"] == "no_verified_constraints"
    assert report["refinement"]["raw_materialized"] is False
    assert not (tmp_path / "refinement" / "raw").exists()


def test_materialized_scale_rebuilds_depth_points_and_poses_together(tmp_path: Path) -> None:
    pose = np.eye(4, dtype=np.float64)
    data = {
        "world_points": np.asarray([[[0.0, 0.0, 2.0]]], dtype=np.float64),
        "depth_z": np.asarray([[2.0]], dtype=np.float32),
        "confidence": np.ones((1, 1), dtype=np.float32),
        "mask": np.ones((1, 1), dtype=bool),
        "camera_pose": pose,
        "intrinsics": np.eye(3, dtype=np.float64),
        "model_rgb": np.zeros((1, 1, 3), dtype=np.float32),
    }
    _materialize_refined_raw(
        [{"path": tmp_path / "source.npz", "data": data}],
        np.asarray([[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]),
        tmp_path / "refined",
        scale=1.5,
    )
    with np.load(tmp_path / "refined" / "raw" / "view_0000.npz") as row:
        np.testing.assert_allclose(row["depth_z"], [[3.0]])
        np.testing.assert_allclose(row["world_points"], [[[1.0, 0.0, 3.0]]])
        np.testing.assert_allclose(row["camera_pose"][:3, 3], [1.0, 0.0, 0.0])



def test_actual_refinement_scales_carrier_before_alignment_callback(
    tmp_path: Path, monkeypatch
) -> None:
    scan = tmp_path / "scan"
    frames_root = scan / "frames"
    frames_root.mkdir(parents=True)
    frames = []
    for index in range(4):
        image_path = frames_root / f"frame_{index:04d}.jpg"
        cv2.imwrite(str(image_path), np.full((8, 12), 90, dtype=np.uint8))
        frame_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
        frames.append(
            {
                "index": index,
                "frame": f"frames/frame_{index:04d}.jpg",
                "timestamp_s": float(index),
                "capture_time_ns": (index + 1) * 10,
                "frame_id": prepared_frame_identity(index, frame_sha256),
                "sha256": frame_sha256,
            }
        )
    (scan / "prepared_frames_manifest.json").write_text(
        json.dumps({"frames": frames}), encoding="utf-8"
    )
    raw = scan / "outputs" / "raw"
    raw.mkdir(parents=True)
    for index in range(4):
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = index * 0.10
        np.savez_compressed(
            raw / f"view_{index:04d}.npz",
            world_points=np.zeros((4, 6, 3), dtype=np.float32),
            depth_z=np.ones((4, 6), dtype=np.float32),
            confidence=np.ones((4, 6), dtype=np.float32),
            mask=np.ones((4, 6), dtype=np.uint8),
            camera_pose=pose,
            intrinsics=np.eye(3, dtype=np.float32),
            model_rgb=np.zeros((4, 6, 3), dtype=np.float32),
        )
    vio_path = scan / "vio_result.json"
    vio_path.write_text(
        json.dumps(
            _vio_payload(
                4,
                step_m=0.15,
                frame_ids=[str(row["frame_id"]) for row in frames],
            )
        ),
        encoding="utf-8",
    )

    import tools.mapanything_phone_scan.build_consensus_fusion as fusion
    import tools.mapanything_phone_scan.trajectory_refinement as refinement_module

    def fake_retrieval(_images, _settings):
        return [
            {
                "source_view": 0,
                "target_view": 2,
                "retrieval_score": 1.0,
                "withheld_from_fit": False,
            },
            {
                "source_view": 0,
                "target_view": 3,
                "retrieval_score": 1.0,
                "withheld_from_fit": True,
            },
        ]

    def fake_verify(source_index, target_index, *_args):
        transform = np.eye(4, dtype=np.float64)
        # Visual edges originate in the DA3 metric gauge and are scaled by the
        # refinement before the graph and holdout comparisons.
        transform[0, 3] = target_index * 0.10
        metrics = {
            "source_view": source_index,
            "target_view": target_index,
            "status": "accepted",
            "rejection_reason": None,
            "match_count": 32,
            "two_d_inlier_count": 24,
            "three_d_match_count": 16,
            "three_d_geometry": {"residual_p80_m": 0.01},
        }
        edge = {
            "source_view": source_index,
            "target_view": target_index,
            "transform": transform.tolist(),
            "label": "verified_visual_loop",
            "translation_sigma_m": 0.03,
            "rotation_sigma_deg": 1.0,
        }
        return metrics, edge

    def fake_pose_graph(first, _second, edges, *, single_carrier=False, single_carrier_name="da3"):
        assert single_carrier is True
        assert single_carrier_name == "da3"
        relative = np.stack([np.linalg.inv(first[0]) @ pose for pose in first])
        return relative, {
            "solver_success": True,
            "edge_count": len(edges),
            "carrier_mode": "single_da3",
        }

    monkeypatch.setattr(refinement_module, "_retrieve_nonadjacent_pairs", fake_retrieval)
    monkeypatch.setattr(refinement_module, "_verify_visual_revisit", fake_verify)
    monkeypatch.setattr(fusion, "_pose_graph", fake_pose_graph)

    callback_calls = {}

    def revalidate(candidate_raw, candidate_manifest, result_dir, scale_factor):
        callback_calls.update(
            {
                "raw": candidate_raw,
                "manifest": candidate_manifest,
                "scale": scale_factor,
            }
        )
        assert candidate_raw != raw
        assert candidate_raw.is_dir()
        assert candidate_manifest.is_file()
        manifest_payload = json.loads(candidate_manifest.read_text(encoding="utf-8"))
        assert manifest_payload["geometry_scale_factor"] == pytest.approx(1.5)
        with np.load(candidate_raw / "view_0003.npz") as row:
            np.testing.assert_allclose(row["depth_z"], 1.5)
            np.testing.assert_allclose(row["camera_pose"][:3, 3], [0.45, 0.0, 0.0])
        result_dir.mkdir(parents=True)
        transform_path = result_dir / "phone_ma_to_noesis_world.json"
        transform_path.write_text(
            json.dumps(
                {
                    "schema": "noesis.mapanything.phone_scan.world_alignment.v1",
                    "scale": 1.0,
                    "world_from_mapanything_row_major": np.eye(4).tolist(),
                }
            ),
            encoding="utf-8",
        )
        return {
            "quality_gate": {"passed": True},
            "artifact_paths": {"transform": str(transform_path)},
        }

    report = run_trajectory_refinement(
        scan,
        raw,
        tmp_path / "refinement",
        TrajectoryRefinementSettings(
            min_scale_excitation_m=0.05,
            withheld_ranges=((3, 4),),
        ),
        vio_constraints=vio_path,
        revalidate_scaled_carrier=revalidate,
    )
    refinement = report["refinement"]
    assert callback_calls["scale"] == pytest.approx(1.5)
    assert refinement["scale_change"] == pytest.approx(1.5)
    assert refinement["raw_materialized"] is True
    assert refinement["world_alignment_revalidation"]["alignment_fit_used"] is True
    assert refinement["pose_deformation"]["translation_change_max_m"] == pytest.approx(0.0)
    assert refinement["withheld_evaluation"]["baseline_translation_residual_p80_m"] == pytest.approx(0.0, abs=1e-6)
    assert refinement["withheld_evaluation"]["translation_residual_p80_m"] == pytest.approx(0.0, abs=1e-6)


def test_nonunit_scale_without_revalidation_callback_stays_review_only(
    tmp_path: Path, monkeypatch
) -> None:
    scan = tmp_path / "scan"
    frames_root = scan / "frames"
    frames_root.mkdir(parents=True)
    frames = []
    for index in range(4):
        image_path = frames_root / f"frame_{index:04d}.jpg"
        cv2.imwrite(str(image_path), np.full((8, 12), 90, dtype=np.uint8))
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        frames.append(
            {
                "index": index,
                "frame": f"frames/frame_{index:04d}.jpg",
                "timestamp_s": float(index),
                "capture_time_ns": (index + 1) * 10,
                "frame_id": prepared_frame_identity(index, digest),
                "sha256": digest,
            }
        )
    (scan / "prepared_frames_manifest.json").write_text(
        json.dumps({"frames": frames}), encoding="utf-8"
    )
    raw = scan / "outputs" / "raw"
    raw.mkdir(parents=True)
    for index in range(4):
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = index * 0.10
        np.savez_compressed(
            raw / f"view_{index:04d}.npz",
            world_points=np.zeros((4, 6, 3), dtype=np.float32),
            depth_z=np.ones((4, 6), dtype=np.float32),
            confidence=np.ones((4, 6), dtype=np.float32),
            mask=np.ones((4, 6), dtype=np.uint8),
            camera_pose=pose,
            intrinsics=np.eye(3, dtype=np.float32),
            model_rgb=np.zeros((4, 6, 3), dtype=np.float32),
        )
    vio_path = scan / "vio_result.json"
    vio_path.write_text(
        json.dumps(
            _vio_payload(
                4,
                step_m=0.15,
                frame_ids=[str(row["frame_id"]) for row in frames],
            )
        ),
        encoding="utf-8",
    )
    import tools.mapanything_phone_scan.build_consensus_fusion as fusion
    import tools.mapanything_phone_scan.trajectory_refinement as refinement_module

    def fake_retrieval(_images, _settings):
        return [
            {
                "source_view": 0,
                "target_view": 3,
                "retrieval_score": 1.0,
                "withheld_from_fit": True,
            }
        ]

    def fake_verify(source_index, target_index, *_args):
        transform = np.eye(4, dtype=np.float64)
        transform[0, 3] = target_index * 0.10
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

    def fake_pose_graph(first, _second, edges, *, single_carrier=False, single_carrier_name="da3"):
        assert single_carrier is True
        assert single_carrier_name == "da3"
        relative = np.stack([np.linalg.inv(first[0]) @ pose for pose in first])
        return relative, {"solver_success": True, "edge_count": len(edges)}

    monkeypatch.setattr(refinement_module, "_retrieve_nonadjacent_pairs", fake_retrieval)
    monkeypatch.setattr(refinement_module, "_verify_visual_revisit", fake_verify)
    monkeypatch.setattr(fusion, "_pose_graph", fake_pose_graph)
    report = run_trajectory_refinement(
        scan,
        raw,
        tmp_path / "refinement",
        TrajectoryRefinementSettings(
            min_scale_excitation_m=0.05,
            withheld_ranges=((3, 4),),
        ),
        vio_constraints=vio_path,
    )
    assert report["refinement"]["status"] == "rejected_world_alignment_revalidation_required"
    assert report["refinement"]["withheld_evaluation"]["status"] == "passed"
    assert report["refinement"]["world_alignment_revalidation_required"] is True
    assert report["refinement"]["raw_materialized"] is False
    assert (tmp_path / "refinement" / "raw").is_dir()

def test_normal_app_vio_adapter_binds_prepared_manifest_to_trajectory(tmp_path: Path, monkeypatch) -> None:
    """The app adapter and trajectory consumer share the real index/hash identity."""
    scan = tmp_path / "scan"
    frames_root = scan / "frames"
    frames_root.mkdir(parents=True)
    frame_rows = []
    for index in range(4):
        image_path = frames_root / f"frame_{index:04d}.jpg"
        cv2.imwrite(str(image_path), np.full((8, 12), 90 + index, dtype=np.uint8))
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        frame_rows.append(
            {
                "index": index,
                "frame": f"frames/frame_{index:04d}.jpg",
                "timestamp_s": float(index),
                "capture_time_ns": (index + 1) * 10,
                "source_frame_index": index,
                "sha256": digest,
            }
        )
    manifest = scan / "prepared_frames_manifest.json"
    manifest.write_text(json.dumps({"frames": frame_rows}), encoding="utf-8")

    native_result = _vio_payload(4, step_m=0.15)
    for index, pose in enumerate(native_result["poses"]):
        # Native dense results carry source-frame identities; the application
        # separately binds their exact times to the selected prepared views.
        pose["prepared_frame_id"] = f"capture:source:{index}"
        pose["source_frame_index"] = index
    generated_config = tmp_path / "vio-config.yaml"
    generated_config.write_text("fixture", encoding="utf-8")

    monkeypatch.setattr(phone_app, "validate_vio_input", lambda *_args: None)
    monkeypatch.setattr(
        phone_app,
        "materialize_openvins_input",
        lambda _capture, _report, output, _settings, _progress: (output, generated_config),
    )
    monkeypatch.setattr(phone_app, "run_openvins", lambda *_args: native_result)
    prepared = {"frames": frame_rows}
    adapted = phone_app._default_vio_runner(
        scan,
        scan / "vio",
        {},
        prepared,
        VIOSettings(),
        lambda *_args: None,
    )
    dense = json.loads((scan / "vio/dense_camera_trajectory.json").read_text())
    assert dense["poses"][0]["prepared_frame_id"] == "capture:source:0"
    vio_path = scan / "vio_result.json"
    vio_path.write_text(json.dumps(adapted), encoding="utf-8")

    from tools.mapanything_phone_scan.trajectory_refinement import (
        _load_prepared_frames,
        _validate_vio_constraints,
    )

    loaded_rows = _load_prepared_frames(scan)
    edges, info = _validate_vio_constraints(vio_path, loaded_rows)
    assert info["relative_constraints_use_global_axis_map"] is False
    assert len(edges) == 3
    assert all(
        row["prepared_frame_id"]
        == prepared_frame_identity(index, frame_rows[index]["sha256"])
        for index, row in enumerate(adapted["poses"])
    )

    # A changed referenced image cannot be silently accepted under the same
    # VIO result, even if its dimensions and timestamp remain unchanged.
    (frames_root / "frame_0000.jpg").write_bytes(b"changed")
    with pytest.raises(TrajectoryRefinementError, match="digest"):
        _load_prepared_frames(scan)


def test_short_profile_edges_preserve_capture_identity_and_distinct_pose_time(tmp_path):
    from .motion_profile import validate_short_walk_runtime
    from .trajectory_refinement import _validate_vio_constraints
    from .vio import SHORT_CONSUMER_VERSION, SHORT_IMAGE_MODEL, POSE_TIME_REFERENCE, _short_runtime_quality

    payload = _vio_payload(4, step_m=0.015)
    payload["status"] = "completed"
    payload["quality"]["reset_count"] = 0
    payload["frame"].update(pose_time_reference=POSE_TIME_REFERENCE,
                            capture_time_reference="original_camera_sensor_timestamp")
    payload["short_session_consumer"] = {
        "consumer_version": SHORT_CONSUMER_VERSION, "validation_run": False,
        "fixed_camera_imu_calibration": True, "native_imu_corrections": "identity",
        "dense_frame_mapping_verified": True, "image_motion_model": SHORT_IMAGE_MODEL,
        "rolling_shutter_compensated": False, "profile_id": "short-test",
    }
    frames, mappings = [], []
    for i, row in enumerate(payload["poses"]):
        row.update(capture_time_ns=1_000_000_000 + i * 100_000_000,
                   pose_time_ns=1_007_000_000 + i * 100_000_000)
        frames.append({"index": i, "frame_id": row["prepared_frame_id"], "capture_time_ns": row["capture_time_ns"]})
        mappings.append({"pose_time_ns": row["pose_time_ns"]})
    envelope = {"maximum_rotation_during_exposure_rad": 0.005, "maximum_rotation_during_readout_rad": 0.02}
    short = {"consumer_version": SHORT_CONSUMER_VERSION, "validation_run": False,
             "profile_validation_sha256": "a" * 64, "motion_envelope": envelope,
             "validated_motion_envelope": envelope}
    metadata = {"short_session": short, "frame_mapping": mappings}
    payload["runtime_quality_control"] = validate_short_walk_runtime(payload, metadata)
    payload["direct_runtime_quality"] = _short_runtime_quality(payload, metadata)
    path = tmp_path / "vio.json"
    path.write_text(json.dumps(payload))
    edges, info = _validate_vio_constraints(path, frames)
    assert len(edges) == 3 and info["pose_time_reference"] == POSE_TIME_REFERENCE
    assert edges[0]["source_time_ns"] == 1_000_000_000
    assert edges[0]["source_pose_time_ns"] == 1_007_000_000
    assert edges[0]["target_pose_time_ns"] == 1_107_000_000
    payload["poses"][0].pop("pose_time_ns")
    path.write_text(json.dumps(payload))
    with pytest.raises(TrajectoryRefinementError, match="pose timestamps"):
        _validate_vio_constraints(path, frames)


def test_matching_uses_retained_rgb_projection_not_resized_capture() -> None:
    # A common-ray warp moves this landmark. Its depth belongs at the new
    # pixel, regardless of the landmark's position in the prepared image.
    raw_rgb = np.zeros((12, 16, 3), dtype=np.uint8)
    raw_rgb[4:7, 9:12] = [240, 100, 20]
    raw = {"model_rgb": raw_rgb, "depth_z": np.ones((12, 16))}
    gray = _raw_gray(raw)
    assert gray[5, 10] > 100
    assert gray[5, 4] == 0
    with pytest.raises(TrajectoryRefinementError, match="depth grid"):
        _raw_gray({"model_rgb": raw_rgb[:6], "depth_z": raw["depth_z"]})


def test_source_frame_identity_survives_refined_materialization(tmp_path: Path) -> None:
    raw_dir = tmp_path / "source" / "raw"
    raw_dir.mkdir(parents=True)
    source_manifest = raw_dir.parent / "scan_outputs_manifest.json"
    identity = "consensus_phone_metric_world_unaligned_to_noesis"
    source_manifest.write_text(json.dumps({
        "coordinate_frame": identity, "view_count": 1, "provider": "consensus_fusion",
    }))
    source = _source_identity(raw_dir, 1)
    row = {"data": {
        "camera_pose": np.eye(4), "world_points": np.ones((2, 3, 3)),
        "depth_z": np.ones((2, 3)), "intrinsics": np.eye(3),
    }}
    output = tmp_path / "refined"
    output.mkdir()
    result = _materialize_refined_raw([row], np.eye(4)[None], output, source_identity=source)
    manifest = json.loads(Path(result["source_manifest"]).read_text())
    assert manifest["coordinate_frame"] == identity
    assert manifest["source_identity"]["manifest_sha256"] == hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    with np.load(result["camera_solution"]) as arrays:
        assert arrays["coordinate_frame"].item() == identity
    with pytest.raises(TrajectoryRefinementError, match="view count"):
        _source_identity(raw_dir, 2)
