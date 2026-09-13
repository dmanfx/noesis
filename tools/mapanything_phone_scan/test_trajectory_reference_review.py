from __future__ import annotations

import csv
import hashlib
import json

import numpy as np
from PIL import Image
import pytest

from tools.mapanything_phone_scan import trajectory_reference_review as review
from tools.mapanything_phone_scan.prepared_frame_identity import prepared_frame_identity
from tools.mapanything_phone_scan.trajectory_motion_review import POSE_CONVENTION


def write_json(path, value):
    path.write_text(json.dumps(value))


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometry():
    points = np.array([[x, y, 3.] for x in np.linspace(-.8, .8, 8) for y in np.linspace(-.8, .8, 8)])
    return {"source_structure": points.copy(), "target_structure": points.copy(),
            "leveled_registration": points.copy(), "target_points": points.copy(),
            "target_normals": np.tile([0., 0., 1.], (len(points), 1)), "best_parameters": np.zeros(3),
            "target_camera_from_world": np.eye(4), "intrinsics": np.array([[50., 0, 32], [0, 50., 32], [0, 0, 1.]]),
            "target_depth_grid": np.full((8, 8), 3.), "floor_transform": np.eye(4)}


@pytest.fixture
def fixture(tmp_path):
    scan = tmp_path / "scan"; scan.mkdir()
    provider = tmp_path / "provider"; provider.mkdir()
    alignment = tmp_path / "alignment"; alignment.mkdir()
    prepared, frames, poses = [], [], []
    for i in range(8):
        image = scan / f"frame{i}.jpg"; image.write_bytes(f"rgb{i}".encode())
        prepared.append({"index": i, "frame": image.name, "sha256": digest(image), "frame_id": prepared_frame_identity(i, digest(image)), "timestamp_s": i * .5})
        frames.append({"index": i, "source_frame": image.name, "timestamp_s": i * .5, "fixed_camera_anchor": False})
        pose = np.eye(4); pose[:3, 3] = [-.4 + i * .1, -.2 + (i % 3) * .2, 2. + i * .1]; poses.append(pose)
    poses = np.array(poses)
    write_json(scan / "prepared_frames_manifest.json", {"frames": prepared})
    manifest = provider / "scan_outputs_manifest.json"
    write_json(manifest, {"provider": "da3", "model_id": "fixture/da3", "coordinate_frame": "da3_metric_world_unaligned_to_noesis", "pose_convention": POSE_CONVENTION,
                          "view_count": 8, "frames": frames, "artifacts": {"trajectory_json": "outputs/trajectory.json"}})
    write_json(provider / "trajectory.json", {"pose_convention": POSE_CONVENTION, "camera_to_world": poses.tolist()})
    frame_binding = {"world_frame": {"frame_id": "backend_world_m", "revision": "world1"}, "target_from_calibration_col_major": np.eye(4).ravel(order="F").tolist()}
    target = {"camera_id": "cam", "revision_id": "ref1", "world_frame": "backend_world_m", "world_frame_revision": "world1", "companion_session_id": "session1",
              "coordinate_frame": "backend_world_m_stream_points", "camera_frame_binding": frame_binding}
    calibration = tmp_path / "calibration.json"
    write_json(calibration, {"E_semantics": "camera_from_calibration_frame_raw", "frame_binding": frame_binding,
                             "cameras": {"cam": {"E": np.eye(4).ravel(order="F").tolist(), "K": [50, 50, 32, 32]}}})
    write_json(alignment / "alignment_report.json", {"status": "passed", "target": target, "inputs": {"phone_source": {"output_manifest": {"sha256": digest(manifest)}}}})
    write_json(alignment / "diagnostic_report.json", {"camera_id": "cam", "target_revision_id": "ref1", "input_hashes": {"phone_output_manifest_sha256": digest(manifest), "camera_calibration_sha256": digest(calibration)}})
    write_json(alignment / "phone_ma_to_noesis_world.json", {"target_binding": target, "source_coordinate_frame": "da3_metric_world_unaligned_to_noesis", "target_coordinate_frame": target["coordinate_frame"],
               "scale": 1., "world_from_mapanything_row_major": np.eye(4).tolist(), "source_output_manifest": {"sha256": digest(manifest)}})
    di = geometry(); np.savez(alignment / "diagnostic_inputs.npz", **di); np.savez(alignment / "aligned_camera_solution.npz", camera_poses=poses)
    temporal = tmp_path / "temporal.json"
    write_json(temporal, {"scan_id": "scan", "camera_id": "cam", "companion_session_id": "session1", "light_alignment": {"equation": "static_mkv_pts_s = phone_mp4_pts_s + offset_s", "offset_s": 1., "clock_rate": 1., "local_review_allowance_s": .1},
                          "static_ds9_mapping": {"static_decoded_frame_index_equals_ds9_frame_id_minus": 100}})
    annotation = tmp_path / "annotations.csv"; timing_map = tmp_path / "timing.csv"
    rows, timing = [], []
    projected, _ = review.project(poses[:, :3, 3] + [.03, .02, .08], di["target_camera_from_world"], di["intrinsics"])
    for i in range(8):
        image = tmp_path / f"image{i}.png"; Image.new("RGB", (64, 64)).save(image)
        rows.append({"frame_id": 100 + i, "phone_pts_s": i * .5, "u_px": projected[i, 0], "v_px": projected[i, 1], "uncertainty_px": 2,
            "uncertainty_semantics": "visual device tolerance", "target": "visible_phone_device_center", "visibility": "visible", "source_image": image.name,
            "image_width_px": 64, "image_height_px": 64, "image_sha256": digest(image)})
        timing.append({"ds9_frame_id": 100 + i, "derived_static_frame_index": i, "static_decoded_pts_s": i * .5 + 1, "estimated_phone_pts_s": i * .5,
                       "recorded_static_frame_exists": "True", "mapping_status": "exact_cohort_to_unique_differential_pts_pair"})
    write_csv(annotation, rows); write_csv(timing_map, timing)
    binding = {**{k:target[k] for k in ("camera_id", "revision_id", "world_frame", "world_frame_revision", "companion_session_id")},
        "schema": review.BINDING_SCHEMA, "annotation_csv_sha256": digest(annotation), "independent_of_reconstruction_and_projection": True,
        "pixel_frame": "rectified_static_image_px", "pixel_axes": "origin_top_left_u_right_v_down", "intrinsics": di["intrinsics"].tolist(), "image_size_px": [64, 64], "image_root": "."}
    for key, path in {"diagnostic_inputs": alignment / "diagnostic_inputs.npz", "aligned_camera_solution": alignment / "aligned_camera_solution.npz", "calibration": calibration,
                      "temporal_alignment": temporal, "frame_timing_map": timing_map}.items():
        binding[key] = {"path": str(path), "sha256": digest(path)}
    for key in ("rectification_producer", "rectification_config", "annotation_procedure"):
        p = tmp_path / key; p.write_text("retained fixture provenance")
        binding[key] = {"path": str(p), "sha256": digest(p)}
    binding_path = tmp_path / "binding.json"; write_json(binding_path, binding)
    return dict(scan=scan, provider_manifest=manifest, alignment_dir=alignment, annotation_csv=annotation, annotation_binding=binding_path, output_dir=tmp_path / "result")


def run(fixture, candidate=False):
    return review.review_trajectory_reference(**fixture, settings=review.TrajectoryReferenceReviewSettings(heldout_start_s=2., timing_allowance_s=.1, fit_translation_candidate=candidate))


def test_complete_review_retains_heldout_and_rejects_structure_conflict(fixture):
    before = fixture["alignment_dir"].joinpath("aligned_camera_solution.npz").read_bytes()
    result = run(fixture, True)
    assert result["original"]["supported_count"] == 8
    candidate = result["translation_candidate"]
    assert candidate["training_frame_ids"] == [100, 101, 102, 103]
    assert candidate["heldout_frame_ids"] == [104, 105, 106, 107]
    np.testing.assert_allclose(candidate["world_translation_m"], [.03, .02, .08], atol=1e-7)
    assert candidate["heldout_error_px"]["rms"] < 1e-6
    assert "structure_fixed_domain_residual_regression" in candidate["conflicts"]
    assert candidate["ready_to_replace_alignment"] is False
    assert result["runtime_admission_ready"] is False
    assert result["timing_sensitivity"]["common_supported_count"] == 6
    assert fixture["alignment_dir"].joinpath("aligned_camera_solution.npz").read_bytes() == before
    with pytest.raises(review.Error, match="new output directory"):
        run(fixture)


@pytest.mark.parametrize("key,value", [("world_frame_revision", "wrong"), ("camera_id", "wrong"), ("revision_id", "wrong"),
    ("companion_session_id", "wrong"), ("pixel_frame", "raw_fisheye"), ("pixel_axes", "v_up"), ("independent_of_reconstruction_and_projection", False), ("image_size_px", [32, 32])])
def test_binding_mismatch_fails_closed(fixture, key, value):
    path = fixture["annotation_binding"]; binding = json.loads(path.read_text()); binding[key] = value; write_json(path, binding)
    with pytest.raises(review.Error): run(fixture)
    assert not fixture["output_dir"].exists()


def test_image_hash_mismatch(fixture):
    fixture["annotation_binding"].parent.joinpath("image0.png").write_bytes(b"altered")
    with pytest.raises(review.Error, match="image hash mismatch"): run(fixture)


def test_annotation_time_mismatch_even_with_updated_csv_hash(fixture):
    path = fixture["annotation_csv"]; rows = list(csv.DictReader(path.open())); rows[1]["phone_pts_s"] = .51; write_csv(path, rows)
    bpath = fixture["annotation_binding"]; binding = json.loads(bpath.read_text()); binding["annotation_csv_sha256"] = digest(path); write_json(bpath, binding)
    with pytest.raises(review.Error, match="frame/time identity mismatch"): run(fixture)


def test_pose_transform_mismatch_even_with_updated_archive_hash(fixture):
    path = fixture["alignment_dir"] / "aligned_camera_solution.npz"
    poses = np.load(path)["camera_poses"]; poses[2, 0, 3] += .01; np.savez(path, camera_poses=poses)
    bpath = fixture["annotation_binding"]; binding = json.loads(bpath.read_text()); binding["aligned_camera_solution"]["sha256"] = digest(path); write_json(bpath, binding)
    with pytest.raises(review.Error, match="aligned poses disagree"): run(fixture)


def test_projection_sign_and_behind_camera():
    uv, valid = review.project(np.array([[1., 2., 4.], [0, 0, -1.], [0, 0, .01]]), np.eye(4), np.diag([100., 100., 1.]))
    np.testing.assert_allclose(uv[0], [25, 50]); assert valid.tolist() == [True, False, False]
    assert np.isnan(uv[1:]).all()


def test_time_bounds_gaps_omissions_and_exact_endpoints():
    times = np.array([0., .5, 2., 2.5]); positions = np.column_stack([times, times, times])
    p, reasons, _ = review.interpolate_positions(times, positions, np.array([-.1, 0., .25, 1., 2.25, 2.5, 2.6]), [0, 1, 2, 4], .85)
    assert reasons == ["outside_pose_time_range", "supported_exact_pose", "supported_interpolation", "pose_gap_exceeds_limit", "omitted_prepared_view_gap", "supported_exact_pose", "outside_pose_time_range"]
    assert np.isnan(p[[0, 3, 4, 6]]).all(); np.testing.assert_allclose(p[2], [.25] * 3)


def test_heldout_pixel_values_cannot_influence_translation():
    points = np.array([[-.5, 0, 2], [.2, -.5, 3], [.5, .3, 4], [0, .1, 2.5], [.1, .2, 3.]])
    K = np.diag([100., 100., 1.]); uv, _ = review.project(points + [.1, -.05, .2], np.eye(4), K)
    train = np.array([True, True, True, False, False])
    first = review.fit_translation(points, uv, np.ones(5), train, np.eye(4), K, .5)[0]
    uv[~train] += 10000
    second = review.fit_translation(points, uv, np.ones(5), train, np.eye(4), K, .5)[0]
    np.testing.assert_array_equal(first, second)


def test_frozen_geometry_cannot_hide_coverage_loss():
    di = geometry()
    original, conflicts = review.assess_geometry(di, np.zeros(3)); assert conflicts == []
    trial, conflicts = review.assess_geometry(di, np.array([0., 0., .4]))
    assert "structure_coverage_loss" in conflicts
    assert "structure_fixed_domain_residual_regression" in conflicts
    assert trial["structure"]["retained_original_comparable_count"] == 0
    assert trial["structure"]["fixed_original_points_and_references"]["candidate"]["median"] == pytest.approx(.4)


@pytest.mark.parametrize("kwargs", [{"timing_allowance_s": float("nan")}, {"heldout_start_s": -1}, {"max_translation_axis_m": 2}, {"max_pose_gap_s": 0}])
def test_setting_bounds(kwargs):
    settings = {"heldout_start_s": 2, "timing_allowance_s": .1, **kwargs}
    with pytest.raises(review.Error): review.TrajectoryReferenceReviewSettings(**settings).validate()


def test_provider_window_boundary_is_not_interpolated():
    points, reasons, _ = review.interpolate_positions(np.array([0., .5]), np.zeros((2, 3)), np.array([.25]), [0, 1], .85, np.array([0, 1]))
    assert reasons == ["provider_window_transition"]
    assert np.isnan(points).all()
