from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tools.mapanything_phone_scan import trajectory_motion_review as review
from tools.mapanything_phone_scan.prepared_frame_identity import prepared_frame_identity


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value))


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def capture(tmp_path):
    scan = tmp_path / "scan"
    native = scan / "capture"
    native.mkdir(parents=True)
    (scan / "frames").mkdir()
    (scan / "scan_state.json").write_text('{"status":"untouched"}')
    provider = tmp_path / "provider"
    provider.mkdir()
    # Deliberately above float64's exact integer range: deltas must stay exact.
    origin = 9_100_000_000_000_037
    times = np.arange(40) * .25
    omega = np.array([.1, -.2, .3])
    gyro_bias = np.array([.011, -.009, .005])
    prepared, outputs, camera, encoder, association = [], [], [], [], []
    camera_to_imu = Rotation.from_euler("xyz", [70, -20, 100], degrees=True)
    global_gauge = Rotation.from_euler("xyz", [10, 45, 30], degrees=True)
    poses = []
    for index, time in enumerate(times):
        stamp = origin + index * 250_000_000
        image = f"frames/frame_{index:04d}.jpg"
        image_bytes = f"fixture-rgb-{index}".encode()
        (scan / image).write_bytes(image_bytes)
        digest = hashlib.sha256(image_bytes).hexdigest()
        prepared.append({"index": index, "frame": image, "sha256": digest,
                         "frame_id": prepared_frame_identity(index, digest),
                         "timestamp_s": float(time), "capture_time_ns": stamp,
                         "source_frame_index": index,
                         "timestamp_source": "android_camera2_sensor_timestamp_exact_encoder_association"})
        outputs.append({"index": index, "source_frame": image, "timestamp_s": float(time),
                        "fixed_camera_anchor": False, "adaptive_window_index": int(index >= 20)})
        pose = np.eye(4)
        pose[:3, :3] = (global_gauge * Rotation.from_rotvec(omega * time) * camera_to_imu).as_matrix()
        pose[:3, 3] = [time * .2, .1 * np.sin(time), 0]
        poses.append(pose.tolist())
        camera.append({"frame_number": index, "sensor_timestamp_ns": stamp, "exposure_time_ns": 10_000_000})
        encoder.append({"encoded_index": index, "encoded_pts_us": stamp // 1000})
        association.append({**encoder[-1], "frame_number": index, "timestamp_ns": stamp})
    _write_json(scan / "prepared_frames_manifest.json", {"frames": prepared})
    _write_json(provider / "camera_trajectory.json", {"pose_convention": review.POSE_CONVENTION, "camera_to_world": poses})
    _write_json(provider / "scan_outputs_manifest.json", {
        "provider": "da3", "model_id": "fixture/da3", "coordinate_frame": "da3_metric_world_window_registered_unaligned_to_noesis",
        "pose_convention": review.POSE_CONVENTION, "view_count": len(outputs), "window_count": 2,
        "anchor_view_index": None, "frames": outputs, "artifacts": {"trajectory_json": "outputs/camera_trajectory.json"}})
    (native / "camera_results.jsonl").write_text("\n".join(json.dumps(r) for r in camera))
    _write_csv(native / "encoder_pts.csv", encoder)
    _write_csv(native / "timestamps.csv", association)
    sensors = {}
    for key, label, unit in (("gyro", "gyroscope", "rad/s"), ("accel", "accelerometer", "m/s^2")):
        rows = []
        for offset in range(-100, 1101):
            time = offset / 100
            vector = omega + gyro_bias if key == "gyro" else np.array([0, 9.81 + .2 * np.sin(time * 2 * np.pi * 1.4), 0])
            bias = gyro_bias if key == "gyro" else np.zeros(3)
            rows.append({"timestamp_ns": origin + offset * 10_000_000,
                         **dict(zip(("x", "y", "z"), vector)),
                         **dict(zip(("bias_x", "bias_y", "bias_z"), bias)), "accuracy": 3})
        _write_csv(native / f"{key}.csv", rows)
        sensors[label] = {"file": f"{key}.csv", "units": unit,
                          "axes": "android_device_x_right_y_up_z_out_of_screen",
                          "timestamp_source": "android_elapsed_realtime_ns", "uncalibrated": True,
                          "bias_fields_are_sensor_estimates": True, "sample_count": len(rows),
                          "first_timestamp_ns": rows[0]["timestamp_ns"], "last_timestamp_ns": rows[-1]["timestamp_ns"]}
    _write_json(native / "capture_import.json", {
        "metric_vio_allowed": False, "android_capture": {
            "camera_acquisition_timestamp_verified": True,
            "association_method": "encoder_pts_us_equals_sensor_timestamp_ns_div_1000",
            "association_row_count": len(times), "recorder": {"sensors": sensors},
            "evidence_paths": {"encoder_pts_path": "encoder_pts.csv", "camera_results_path": "camera_results.jsonl"}},
        "manifest": {"capture_id": "fixture", "calibration": {},
                     "clocks": {"camera_domain": "android.elapsedRealtimeNanos", "imu_domain": "android.elapsedRealtimeNanos"},
                     "video": {"frame_timestamps_path": "timestamps.csv"},
                     "imu": {"gyro_path": "gyro.csv", "accel_path": "accel.csv", "gyro_unit": "rad/s", "accel_unit": "m/s^2"}}})
    return {"scan": scan, "native": native, "provider": provider,
            "manifest": provider / "scan_outputs_manifest.json", "origin": origin, "tmp": tmp_path}


def _run(capture, **settings):
    return review.review_trajectory_motion(capture["scan"], capture["manifest"], capture["tmp"] / "review",
        review.TrajectoryMotionReviewSettings(heldout_start_s=5.125, **settings))


def _edit_json(path: Path, edit) -> None:
    value = json.loads(path.read_text())
    edit(value)
    _write_json(path, value)


def _subset(capture, indices):
    def manifest(value):
        value["frames"] = [{**value["frames"][old], "index": new} for new, old in enumerate(indices)]
        value["view_count"] = len(indices)
    _edit_json(capture["manifest"], manifest)
    _edit_json(capture["provider"] / "camera_trajectory.json",
               lambda value: value.update(camera_to_world=[value["camera_to_world"][i] for i in indices]))


def _edit_sensor(capture, name, transform):
    path = capture["native"] / f"{name}.csv"
    rows = list(csv.DictReader(path.open()))
    rows = transform(rows)
    _write_csv(path, rows)
    label = "gyroscope" if name == "gyro" else "accelerometer"
    def descriptor(value):
        value["android_capture"]["recorder"]["sensors"][label].update(
            sample_count=len(rows), first_timestamp_ns=int(rows[0]["timestamp_ns"]),
            last_timestamp_ns=int(rows[-1]["timestamp_ns"]))
    _edit_json(capture["native"] / "capture_import.json", descriptor)


def test_cli_exact_timing_gauge_and_bias_without_admission(capture):
    state = (capture["scan"] / "scan_state.json").read_bytes()
    args = ["--scan-dir", str(capture["scan"]), "--source-output-manifest", str(capture["manifest"]),
            "--output-dir", str(capture["tmp"] / "review"), "--heldout-start-s", "5.125", "--fit-rotation-candidate"]
    assert review.main(args) == 0
    result = json.loads((capture["tmp"] / "review/motion_review.json").read_text())
    assert result["rotation"]["all_supported"]["n"] == 38  # 39 intervals minus the provider boundary.
    assert result["rotation"]["all_supported"]["absolute_increment_error_deg_max"] < 1e-10
    assert result["timing"]["phone_native_origin_ns"] == capture["origin"]
    assert result["prepared_view_identities"][1]["capture_time_ns"] == capture["origin"] + 250_000_000
    assert result["rotation_candidate"]["status"] == "insufficient_rotational_excitation"
    assert result["metric_vio_admission_changed"] is False
    assert "camera_to_imu_extrinsics" in result["calibration"]["missing_or_unverified"]
    assert result["imu"]["gyro"]["recorded_bias_estimates_subtracted"] is True
    assert (capture["scan"] / "scan_state.json").read_bytes() == state
    intervals = list(csv.DictReader((capture["tmp"] / "review/intervals.csv").open()))
    assert intervals[19]["exclusions"] == "provider_window_transition"
    assert intervals[20]["split"] == "split_boundary"  # No training/heldout interval overlap.


def test_partial_scope_is_explicit_and_retains_missing_identities(capture):
    _subset(capture, list(range(30)))
    with pytest.raises(review.TrajectoryMotionReviewError, match="explicitly allow a partial"):
        _run(capture)
    assert not (capture["tmp"] / "review").exists()
    result = _run(capture, allow_partial=True)
    assert result["scope"]["omitted_prepared_indices"] == list(range(30, 40))
    assert result["scope"]["status"] == "partial"


def test_hole_in_provider_views_is_not_joined(capture):
    _subset(capture, [i for i in range(40) if i != 10])
    _run(capture, allow_partial=True)
    intervals = list(csv.DictReader((capture["tmp"] / "review/intervals.csv").open()))
    assert intervals[9]["rotation_supported"] == "False"
    assert intervals[9]["exclusions"] == "omitted_prepared_views"


@pytest.mark.parametrize("mutation,match", [
    (lambda m: m["frames"][1].update(timestamp_s=.251), "source timestamp"),
    (lambda m: m.update(pose_convention="world_to_camera"), "camera-to-world"),
    (lambda m: m.update(coordinate_frame="backend_world_m"), "unaligned metric"),
    (lambda m: m["frames"][0].update(fixed_camera_anchor=True), "phone source identity"),
    (lambda m: m["frames"][25].pop("adaptive_window_index"), "exact integer"),
    (lambda m: m["frames"][0].update(index=False), "exact integer"),
    (lambda m: m.pop("model_id"), "provider/model identity"),
    (lambda m: m.update(partial_reconstruction={"omitted_global_indices": [1]}), "scope disagrees"),
    (lambda m: m.update(partial_reconstruction={"parent_prepared_manifest_sha256": "0" * 64}), "parent-manifest"),
])
def test_provider_binding_failures(capture, mutation, match):
    _edit_json(capture["manifest"], mutation)
    with pytest.raises(review.TrajectoryMotionReviewError, match=match):
        _run(capture)


def test_changed_rgb_is_rejected(capture):
    (capture["scan"] / "frames/frame_0000.jpg").write_bytes(b"changed")
    with pytest.raises(review.TrajectoryMotionReviewError, match="RGB bytes"):
        _run(capture)


@pytest.mark.parametrize("value", [1.0, True, "9000000000000000.5"])
def test_native_acquisition_requires_integer_not_float(capture, value):
    _edit_json(capture["scan"] / "prepared_frames_manifest.json", lambda x: x["frames"][1].update(capture_time_ns=value))
    with pytest.raises(review.TrajectoryMotionReviewError, match="exact integer"):
        _run(capture)


def test_encoder_reassociation_and_unverified_clocks_are_rejected(capture):
    path = capture["native"] / "encoder_pts.csv"
    rows = list(csv.DictReader(path.open()))
    rows[3]["encoded_pts_us"] = str(int(rows[3]["encoded_pts_us"]) + 1)
    _write_csv(path, rows)
    with pytest.raises(review.TrajectoryMotionReviewError, match="association no longer matches"):
        _run(capture)
    _edit_json(capture["native"] / "capture_import.json", lambda x: x["android_capture"].update(camera_acquisition_timestamp_verified=False))
    with pytest.raises(review.TrajectoryMotionReviewError, match="verified native"):
        _run(capture)


def test_trajectory_reflection_is_not_normalized_into_a_rotation(capture):
    def reflect(value):
        value["camera_to_world"][3][0][0] *= -1
    _edit_json(capture["provider"] / "camera_trajectory.json", reflect)
    with pytest.raises(review.TrajectoryMotionReviewError, match="proper SO"):
        _run(capture)


def test_malformed_nested_evidence_fails_as_input_error(capture):
    _edit_json(capture["native"] / "capture_import.json", lambda x: x["android_capture"].update(evidence_paths=[]))
    with pytest.raises(review.TrajectoryMotionReviewError, match="must be an object"):
        _run(capture)


def test_pose_displacement_overflow_is_rejected_before_output(capture):
    def change(value):
        value["camera_to_world"][0][0][3] = 1e308
        value["camera_to_world"][1][0][3] = -1e308
    _edit_json(capture["provider"] / "camera_trajectory.json", change)
    with pytest.raises(review.TrajectoryMotionReviewError, match="speed is not finite"):
        _run(capture)
    assert not (capture["tmp"] / "review").exists()


def test_missing_and_unreliable_gyro_samples_leave_gaps_but_later_motion_works(capture):
    origin = capture["origin"]
    def remove(rows):
        rows = [r for r in rows if not origin + 2_000_000_000 < int(r["timestamp_ns"]) < origin + 2_400_000_000]
        for row in rows:
            if int(row["timestamp_ns"]) == origin + 7_100_000_000:
                row["accuracy"] = "0"
        return rows
    _edit_sensor(capture, "gyro", remove)
    result = _run(capture)
    assert result["imu"]["gyro"]["supported_segments"] == 3
    assert result["imu"]["gyro"]["unreliable_samples_excluded"] == 1
    assert result["rotation"]["heldout"]["absolute_increment_error_deg_max"] < 1e-10
    rows = list(csv.DictReader((capture["tmp"] / "review/intervals.csv").open()))
    assert "gyro_gap" in rows[8]["exclusions"]
    assert rows[35]["rotation_supported"] == "True"


def test_missing_coverage_never_clamps_gyro_to_a_boundary(capture):
    origin = capture["origin"]
    _edit_sensor(capture, "gyro", lambda rows: [r for r in rows if int(r["timestamp_ns"]) >= origin + 400_000_000])
    _run(capture)
    rows = list(csv.DictReader((capture["tmp"] / "review/intervals.csv").open()))
    assert rows[0]["gyro_net_rotation_deg"] == ""
    assert rows[1]["rotation_supported"] == "False"
    assert rows[2]["rotation_supported"] == "True"


@pytest.mark.parametrize("kind", ["nan", "duplicate", "bias"])
def test_malformed_sensor_data_fails_closed(capture, kind):
    def mutate(rows):
        if kind == "nan": rows[100]["x"] = "nan"
        if kind == "duplicate": rows[100]["timestamp_ns"] = rows[99]["timestamp_ns"]
        if kind == "bias": rows[100]["bias_y"] = "inf"
        return rows
    _edit_sensor(capture, "gyro", mutate)
    with pytest.raises(review.TrajectoryMotionReviewError):
        _run(capture)


def test_protected_output_and_escaping_artifacts(capture):
    settings = review.TrajectoryMotionReviewSettings(heldout_start_s=5)
    for out in (capture["scan"] / "review", capture["provider"] / "review"):
        with pytest.raises(review.TrajectoryMotionReviewError, match="outside"):
            review.review_trajectory_motion(capture["scan"], capture["manifest"], out, settings)
    out = capture["tmp"] / "review"
    out.mkdir()
    (out / "keep").write_text("previous")
    with pytest.raises(review.TrajectoryMotionReviewError):
        _run(capture)
    assert (out / "keep").read_text() == "previous"
    _edit_json(capture["manifest"], lambda x: x["artifacts"].update(trajectory_json="outputs/../outside.json"))
    with pytest.raises(review.TrajectoryMotionReviewError, match="escapes"):
        review.review_trajectory_motion(capture["scan"], capture["manifest"], capture["tmp"] / "another", settings)


def test_gyro_composition_order_and_gap_local_gauge():
    timestamps = np.array([0, 10, 20, 30], dtype=np.int64) * 1_000_000
    stream = review.SensorStream(timestamps, np.array([[0, 0, 0], [20, 0, 0], [0, 20, 0], [0, 0, 20.]]), np.ones(4, bool), False)
    integrator = review.GyroOrientations(stream, 0, 15_000_000)
    vectors, valid = integrator.relative(np.array([0, 30_000_000], dtype=np.int64))
    expected = Rotation.from_rotvec([.1, 0, 0]) * Rotation.from_rotvec([.1, .1, 0]) * Rotation.from_rotvec([0, .1, .1])
    assert valid.tolist() == [True]
    np.testing.assert_allclose(vectors[0], expected.as_rotvec(), atol=1e-12)
    gap_ts = np.array([0, 10, 100, 110], dtype=np.int64) * 1_000_000
    stream = review.SensorStream(gap_ts, np.tile([0, 0, 1.], (4, 1)), np.ones(4, bool), False)
    vectors, valid = review.GyroOrientations(stream, 0, 15_000_000).relative(gap_ts)
    assert valid.tolist() == [True, False, True]
    assert np.isnan(vectors[1]).all()
    np.testing.assert_allclose(vectors[[0, 2]], [[0, 0, .01], [0, 0, .01]], atol=1e-12)


def test_accel_activity_does_not_remove_gravity_into_position_or_bridge_gaps():
    ts = np.arange(1001, dtype=np.int64) * 10_000_000
    valid = np.ones(len(ts), bool)
    valid[450:551] = False
    stream = review.SensorStream(ts, np.tile([0, 9.81, 0], (len(ts), 1)), valid, False)
    activity = review.acceleration_activity(stream, 0, np.array([2., 4., 5., 6., 8.]), 50_000_000)
    np.testing.assert_allclose(activity[[0, 4]], 0, atol=1e-10)
    assert np.isnan(activity[1:4]).all()


def test_rotation_candidate_convention_heldout_and_single_axis_observability():
    random = np.random.default_rng(17)
    camera = random.normal(0, .15, (40, 3))
    expected = Rotation.from_euler("xyz", [80, -20, 35], degrees=True)
    gyro = expected.apply(camera)
    train = np.arange(40) < 25
    result = review.rotation_candidate(camera, gyro, train, ~train)
    fitted = Rotation.from_matrix(result["camera_to_imu_rotation_row_major"])
    assert (expected.inv() * fitted).magnitude() < 1e-12
    assert result["heldout_residual"]["full_rotation_error_deg_rms"] < 1e-10
    changed = gyro.copy()
    changed[~train] = Rotation.from_euler("z", 30, degrees=True).apply(changed[~train])
    other = review.rotation_candidate(camera, changed, train, ~train)
    np.testing.assert_allclose(other["camera_to_imu_rotation_row_major"], result["camera_to_imu_rotation_row_major"], atol=1e-12)
    assert other["heldout_residual"]["full_rotation_error_deg_rms"] > 1
    single_axis = np.column_stack([np.zeros(40), np.linspace(.01, .2, 40), np.zeros(40)])
    refused = review.rotation_candidate(single_axis, expected.apply(single_axis), train, ~train)
    assert refused["status"] == "insufficient_rotational_excitation"
    assert "camera_to_imu_rotation_row_major" not in refused


@pytest.mark.parametrize("kwargs", [
    {"heldout_start_s": float("nan")}, {"max_sample_gap_s": 1}, {"max_pose_interval_s": float("inf")},
    {"sensitivity_offsets_s": (0., 0.)}, {"sensitivity_offsets_s": (0., float("nan"))},
])
def test_settings_are_bounded(kwargs):
    with pytest.raises(review.TrajectoryMotionReviewError):
        review.TrajectoryMotionReviewSettings(**{"heldout_start_s": 2, **kwargs}).validate()
