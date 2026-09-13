from __future__ import annotations

import hashlib
import copy
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.mapanything_phone_scan.capture import (
    CaptureImportError,
    import_capture_bundle,
    parse_imu_axis_csv,
    validate_capture_manifest,
)
from tools.mapanything_phone_scan.processing import (
    FramePreparationSettings,
    _extract_candidates,
)
from tools.mapanything_phone_scan.vio import (
    VIOError,
    VIOSettings,
    CALIBRATION_BOUNDS,
    CALIBRATION_COVARIANCE_CONVENTION,
    VIO_CALIBRATION_PRIOR_SCHEMA,
    VIO_CALIBRATION_SCHEMA,
    _interpolate_stream,
    _openvins_camera_projection,
    _rectify_dense_camera_images,
    materialize_openvins_input,
    materialize_openvins_calibration_input,
    run_openvins,
    run_openvins_calibration,
    validate_vio_calibration_prior,
    validate_vio_calibration_result,
    validate_vio_result,
)


def _manifest() -> dict:
    return {
        "schema": "noesis.phone_capture.v1",
        "capture_id": "test-capture",
        "device": {"id": "test-device", "model": "test", "os": "Android"},
        "video": {"path": "walk.mp4", "timestamp_source": "camera2_hardware"},
        "imu": {
            "path": "imu.csv",
            "sensor_id": "imu0",
            "time_domain": "camera2_hardware",
            "axes": "x,y,z",
            "accel_unit": "m/s^2",
            "gyro_unit": "rad/s",
            "timestamp_unit": "ns",
            "noise": {
                "gyroscope_noise_density": 1.7e-4,
                "gyroscope_random_walk": 1.9e-5,
                "accelerometer_noise_density": 2.0e-3,
                "accelerometer_random_walk": 3.0e-3,
            },
        },
        "camera": {
            "id": "camera0",
            "intrinsics": [[400, 0, 48], [0, 400, 32], [0, 0, 1]],
            "distortion": [0, 0, 0, 0],
            "distortion_model": "plumb_bob",
            "resolution_px": [96, 64],
            "orientation_deg": 0,
            "stabilization": "off",
        },
        "extrinsics": {"T_imu_camera": np.eye(4).tolist()},
        "clocks": {
            "camera_domain": "camera2_hardware",
            "imu_domain": "camera2_hardware",
            "imu_to_camera_offset_ns": 0,
            "timestamp_source": "hardware",
        },
    }


def _valid_vio_result() -> dict:
    pose = np.eye(4).tolist()
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
        "poses": [
            {
                "capture_time_ns": 10,
                "prepared_frame_id": "frame-0",
                "T_vio_world_camera": pose,
                "velocity_mps": [0, 0, 0],
                "gravity_mps2": [0, 0, -9.81],
                "gyro_bias_rads": [0, 0, 0],
                "accel_bias_mps2": [0, 0, 0],
                "covariance": np.eye(6).tolist(),
            },
            {
                "capture_time_ns": 20,
                "prepared_frame_id": "frame-1",
                "T_vio_world_camera": pose,
                "velocity_mps": [0, 0, 0],
                "gravity_mps2": [0, 0, -9.81],
                "gyro_bias_rads": [0, 0, 0],
                "accel_bias_mps2": [0, 0, 0],
                "covariance": np.eye(6).tolist(),
            },
        ],
        "quality": {"initialized": True, "tracking_ratio": 1.0},
        "segments": [{"id": "openvins-0", "reset": False}],
    }


def test_separate_opencamera_stream_preserves_clocked_rows_and_si_units() -> None:
    rows, metrics = parse_imu_axis_csv(
        b"1,2,3,1000000000\n2,3,4,1005000000\n",
        {"accel_unit": "g", "gyro_unit": "rad/s", "timestamp_unit": "ns"},
        kind="accel",
    )
    assert rows[0]["timestamp_ns"] == 1_000_000_000
    assert rows[0]["raw"] == [1.0, 2.0, 3.0]
    assert rows[0]["si"] == pytest.approx([9.80665, 19.6133, 29.41995])
    assert metrics["sample_count"] == 2


def test_missing_camera_calibration_is_null_and_blocks_metric_admission() -> None:
    manifest = _manifest()
    manifest["camera"].update({"intrinsics_source": "missing", "intrinsics": None, "resolution_px": None})
    normalized = validate_capture_manifest(manifest)
    assert normalized["camera"]["intrinsics"] is None
    assert normalized["camera"]["resolution_px"] is None
    assert normalized["calibration"]["complete_for_metric_vio"] is False


def test_higher_order_distortion_is_retained_but_blocks_metric_admission() -> None:
    manifest = _manifest()
    manifest["camera"]["distortion"] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    normalized = validate_capture_manifest(manifest)
    assert normalized["camera"]["distortion"][-1] == 0.6
    assert normalized["camera"]["distortion_supported"] is False
    assert normalized["calibration"]["complete_for_metric_vio"] is False


def _d5_camera() -> dict:
    camera = _manifest()["camera"]
    camera.update({
        "intrinsics": [[55.0, 0.0, 47.5], [0.0, 57.0, 31.5], [0.0, 0.0, 1.0]],
        "distortion": [-0.08, 0.02, 0.002, -0.003, 0.2],
    })
    return camera


def _brown_d5_source_maps(camera: dict) -> tuple[np.ndarray, np.ndarray]:
    """Independent Brown-Conrady projection of each ideal output-image ray."""
    width, height = camera["resolution_px"]
    matrix = np.asarray(camera["intrinsics"])
    pixels_y, pixels_x = np.indices((height, width), dtype=np.float64)
    x = (pixels_x - matrix[0, 2]) / matrix[0, 0]
    y = (pixels_y - matrix[1, 2]) / matrix[1, 1]
    k1, k2, p1, p2, k3 = camera["distortion"]
    radius2 = x * x + y * y
    radial = 1 + k1 * radius2 + k2 * radius2**2 + k3 * radius2**3
    source_x = matrix[0, 0] * (x * radial + 2 * p1 * x * y + p2 * (radius2 + 2 * x * x)) + matrix[0, 2]
    source_y = matrix[1, 1] * (y * radial + p1 * (radius2 + 2 * y * y) + 2 * p2 * x * y) + matrix[1, 2]
    return source_x.astype(np.float32), source_y.astype(np.float32)


def test_openvins_d5_map_preserves_nonzero_k3_and_exact_camera_geometry() -> None:
    camera = _d5_camera()
    projection, maps = _openvins_camera_projection(camera)
    assert maps is not None
    expected = _brown_d5_source_maps(camera)
    np.testing.assert_allclose(maps[0], expected[0], rtol=0, atol=1e-5)
    np.testing.assert_allclose(maps[1], expected[1], rtol=0, atol=1e-5)
    truncated_camera = {**camera, "distortion": [*camera["distortion"][:4], 0.0]}
    truncated_maps = _brown_d5_source_maps(truncated_camera)
    assert float(np.max(np.abs(maps[0] - truncated_maps[0]))) > 5.0
    assert projection["source_camera"]["distortion"] == camera["distortion"]
    assert projection["estimator_camera"]["K"] == camera["intrinsics"]
    assert projection["estimator_camera"]["resolution_px"] == [96, 64]
    assert projection["estimator_camera"]["distortion"] == [0, 0, 0, 0]
    assert projection["camera_axes_unchanged"] is True


@pytest.mark.parametrize("model", ["plumb_bob", "radtan", "brown_conrady"])
def test_admitted_pinhole_d5_retains_every_coefficient(model: str) -> None:
    manifest = _manifest()
    manifest["camera"] = {**_d5_camera(), "distortion_model": model}
    normalized = validate_capture_manifest(manifest)
    assert normalized["camera"]["distortion"] == manifest["camera"]["distortion"]
    assert normalized["camera"]["distortion_supported"] is True
    assert normalized["calibration"]["complete_for_metric_vio"] is True


@pytest.mark.parametrize("model,distortion", [
    ("fisheye", [0.1] * 5), ("equidistant", [0.1] * 5),
    ("radtan", [0.1] * 6), ("unknown", [0.0] * 4),
])
def test_openvins_rejects_distortion_models_without_exact_support(model: str, distortion: list[float]) -> None:
    camera = {**_d5_camera(), "distortion_model": model, "distortion": distortion}
    with pytest.raises(VIOError, match="cannot preserve"):
        _openvins_camera_projection(camera)
    if model in {"fisheye", "equidistant"}:
        manifest = _manifest()
        manifest["camera"] = camera
        assert validate_capture_manifest(manifest)["calibration"]["complete_for_metric_vio"] is False


def test_openvins_rejects_skew_instead_of_silently_changing_K() -> None:
    camera = _d5_camera()
    camera["intrinsics"][0][1] = 0.1
    with pytest.raises(VIOError, match="zero-skew"):
        _openvins_camera_projection(camera)


def _dense_capture_fixture(tmp_path: Path) -> tuple[Path, dict, VIOSettings, list[np.ndarray]]:
    capture_dir = tmp_path / "capture"
    source_dir = tmp_path / "source-images"
    capture_dir.mkdir()
    source_dir.mkdir()
    yy, xx = np.indices((64, 96))
    source_images = []
    for index in range(3):
        image = np.stack(((xx * 7 + index * 23) % 256, (yy * 9 + index * 31) % 256, ((xx + yy) * 5) % 256), axis=-1).astype(np.uint8)
        source_images.append(image)
        assert cv2.imwrite(str(source_dir / f"frame-{index:02d}.png"), image)
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "10",
        "-i", str(source_dir / "frame-%02d.png"), "-c:v", "libx264rgb", "-crf", "0",
        "-pix_fmt", "rgb24", str(capture_dir / "walk.mp4"),
    ], check=True)
    manifest = _manifest()
    manifest["camera"] = _d5_camera()
    normalized = validate_capture_manifest(manifest)
    assert normalized["calibration"]["complete_for_metric_vio"] is True
    timestamps = [1_000_000_000, 1_100_000_000, 1_200_000_000]
    (capture_dir / "video_timestamps_ns.json").write_text(json.dumps(timestamps))
    (capture_dir / "imu_normalized.json").write_text(json.dumps([
        {"timestamp_ns": value, "accel_mps2": [0, 0, 9.81], "gyro_rads": [0, 0, 0]}
        for value in range(900_000_000, 1_300_000_001, 50_000_000)
    ]))
    report = {"metric_vio_allowed": True, "manifest": normalized, "video": {"frame_count": 3, "timestamp_source": "fixture_acquisition_ns"}}
    settings = VIOSettings(config=Path(__file__).parent / "native/openvins_phone_base_config.yaml", timeout_s=30)
    return capture_dir, report, settings, source_images


def test_dense_d5_decode_rectifies_actual_images_and_binds_mask_K_and_frame_identity(tmp_path: Path) -> None:
    capture_dir, report, settings, source_images = _dense_capture_fixture(tmp_path)
    original_video = hashlib.sha256((capture_dir / "walk.mp4").read_bytes()).hexdigest()
    input_root, _ = materialize_openvins_input(capture_dir, report, tmp_path / "run", settings, lambda *_: None)
    metadata = json.loads((input_root / "openvins_input.json").read_text())
    projection = metadata["camera_preprocessing"]
    maps = _brown_d5_source_maps(_d5_camera())
    invalid = (maps[0] < 0) | (maps[0] > 95) | (maps[1] < 0) | (maps[1] > 63)
    mask = cv2.imread(str(input_root / "cam0/rectification_mask.png"), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(mask, np.where(invalid, 255, 0).astype(np.uint8))
    assert 0 < projection["invalid_ray_mask"]["invalid_pixel_count"] < 96 * 64
    assert projection["source_camera"]["distortion"] == _d5_camera()["distortion"]
    assert projection["estimator_camera"]["distortion"] == [0, 0, 0, 0]
    for index, row in enumerate(metadata["frame_mapping"]):
        path = input_root / "cam0/data" / row["filename"]
        actual = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        expected = cv2.remap(source_images[index], *maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        expected[invalid] = 0
        np.testing.assert_array_equal(actual, expected)
        assert row["source_frame_index"] == index
        assert row["capture_time_ns"] == 1_000_000_000 + index * 100_000_000
        assert row["estimator_image_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert len(row["decoded_source_sha256"]) == 64
        assert row["decoded_source_sha256"] != row["estimator_image_sha256"]
    assert hashlib.sha256((capture_dir / "walk.mp4").read_bytes()).hexdigest() == original_video
    yaml = cv2.FileStorage(str(input_root / "kalibr_imucam_chain.yaml"), cv2.FILE_STORAGE_READ)
    node = yaml.getNode("cam0")
    assert [node.getNode("intrinsics").at(i).real() for i in range(4)] == [55.0, 57.0, 47.5, 31.5]
    assert [node.getNode("resolution").at(i).real() for i in range(2)] == [96, 64]
    assert [node.getNode("distortion_coeffs").at(i).real() for i in range(4)] == [0, 0, 0, 0]
    yaml.release()


@pytest.mark.parametrize("mismatch", ["rotation", "resolution", "crop"])
def test_dense_d5_rejects_encoded_geometry_mismatch_before_writing_inputs(tmp_path: Path, mismatch: str) -> None:
    capture_dir, report, settings, _ = _dense_capture_fixture(tmp_path)
    camera = report["manifest"]["camera"]
    if mismatch == "rotation":
        camera["orientation_deg"] = 90
    elif mismatch == "resolution":
        camera["resolution_px"] = [96, 62]
    else:
        camera["crop"] = {"left": 1, "top": 0, "width": 95, "height": 64}
    with pytest.raises(VIOError, match="orientation_deg|geometry does not match"):
        materialize_openvins_input(capture_dir, report, tmp_path / "run", settings, lambda *_: None)
    assert not (tmp_path / "run").exists()


def test_d5_rectification_rejects_wrong_decoded_image_size_without_overwriting_it(tmp_path: Path) -> None:
    image_dir = tmp_path / "cam0/data"
    image_dir.mkdir(parents=True)
    path = image_dir / "frame-00000000.png"
    assert cv2.imwrite(str(path), np.full((62, 96, 3), 200, dtype=np.uint8))
    original = path.read_bytes()
    projection, maps = _openvins_camera_projection(_d5_camera())
    assert maps is not None
    with pytest.raises(VIOError, match="decoded image does not match"):
        _rectify_dense_camera_images([path], image_dir, projection, maps)
    assert path.read_bytes() == original


def _masked_runner_fixture(tmp_path: Path) -> tuple[Path, VIOSettings, dict]:
    capture_dir = tmp_path / "input"
    mask_path = capture_dir / "cam0/rectification_mask.png"
    mask_path.parent.mkdir(parents=True)
    mask = np.zeros((64, 96), dtype=np.uint8)
    mask[:, :2] = 255
    assert cv2.imwrite(str(mask_path), mask)
    confirmation = {"applied": True, "invalid_pixel_count": 128, "resolution_px": [96, 64]}
    metadata = {
        "capture_id": "capture", "camera_sensor_id": "camera0", "time_domain": "camera2_hardware",
        "camera_preprocessing": {
            "rectified": True,
            "estimator_camera": {"resolution_px": [96, 64]},
            "invalid_ray_mask": {
                "path": "cam0/rectification_mask.png",
                "sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                "invalid_pixel_count": confirmation["invalid_pixel_count"],
            },
        },
    }
    (capture_dir / "openvins_input.json").write_text(json.dumps(metadata))
    settings = VIOSettings(
        executable=Path(sys.executable),
        config=Path(__file__).parent / "native/openvins_phone_base_config.yaml",
    )
    return capture_dir, settings, confirmation


@pytest.mark.parametrize("reported", ["correct", "missing", "not_applied", "wrong_count", "wrong_resolution"])
def test_rectified_runner_requires_native_mask_confirmation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reported: str) -> None:
    capture_dir, settings, confirmation = _masked_runner_fixture(tmp_path)
    payload = _valid_vio_result()
    if reported != "missing":
        payload["camera_mask"] = dict(confirmation)
        if reported == "not_applied":
            payload["camera_mask"]["applied"] = False
        elif reported == "wrong_count":
            payload["camera_mask"]["invalid_pixel_count"] += 1
        elif reported == "wrong_resolution":
            payload["camera_mask"]["resolution_px"] = [96, 62]

    class ReportedProcess:
        def __init__(self, command: list[str], **_: object) -> None:
            assert command[command.index("--camera-mask") + 1] == str(capture_dir / "cam0/rectification_mask.png")
            Path(command[command.index("--output") + 1]).write_text(json.dumps(payload))

        def wait(self, **_: object) -> int:
            return 0

    monkeypatch.setattr(subprocess, "Popen", ReportedProcess)
    if reported == "correct":
        result = run_openvins(capture_dir, tmp_path / "output", settings, lambda *_: None)
        assert result["camera_mask"] == confirmation
    else:
        with pytest.raises(VIOError, match="did not confirm the required rectification mask"):
            run_openvins(capture_dir, tmp_path / "output", settings, lambda *_: None)


def test_rectified_runner_rejects_changed_mask_before_starting_estimator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    capture_dir, settings, _ = _masked_runner_fixture(tmp_path)
    assert cv2.imwrite(str(capture_dir / "cam0/rectification_mask.png"), np.zeros((64, 96), dtype=np.uint8))

    def unexpected_process(*_: object, **__: object) -> None:
        pytest.fail("estimator must not start after the required mask changes")

    monkeypatch.setattr(subprocess, "Popen", unexpected_process)
    with pytest.raises(VIOError, match="mask does not match its provenance"):
        run_openvins(capture_dir, tmp_path / "output", settings, lambda *_: None)


def test_separate_stream_interpolation_uses_precomputed_timestamp_arrays() -> None:
    times = [1_000_000_000, 1_010_000_000]
    values = [[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]]
    assert _interpolate_stream(times, values, 1_005_000_000) == pytest.approx([1.0, 2.0, 3.0])


def test_capture_archive_rejects_traversal_and_symlink_members(tmp_path: Path) -> None:
    traversal = tmp_path / "traversal.zip"
    with zipfile.ZipFile(traversal, "w") as archive:
        archive.writestr("../escape.txt", "blocked")
    with pytest.raises(CaptureImportError, match="unsafe member path"):
        import_capture_bundle(traversal, tmp_path / "scan-traversal")

    symlink = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("capture_manifest.json")
    info.create_system = 3
    info.external_attr = (0o120777 << 16)
    with zipfile.ZipFile(symlink, "w") as archive:
        archive.writestr(info, "target")
    with pytest.raises(CaptureImportError, match="regular file"):
        import_capture_bundle(symlink, tmp_path / "scan-symlink")


def test_import_keeps_integer_video_times_and_rejects_unequal_sensor_endpoints(tmp_path: Path) -> None:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    for index in range(3):
        image = np.full((64, 96, 3), 30 + index * 50, dtype=np.uint8)
        assert cv2.imwrite(str(frame_dir / f"frame-{index:02d}.png"), image)
    video = tmp_path / "walk.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-framerate", "10", "-i", str(frame_dir / "frame-%02d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-video_track_timescale", "1000",
            str(video),
        ],
        check=True,
    )
    manifest = _manifest()
    manifest["video"].update(
        {
            "path": "walk.mp4",
            "frame_timestamps_path": "walk_timestamps.csv",
            "timestamp_unit": "ns",
        }
    )
    manifest["imu"].pop("path")
    manifest["imu"].update({"accel_path": "walk_accel.csv", "gyro_path": "walk_gyro.csv"})
    manifest["clocks"]["camera_start_time_ns"] = 1_000_000_000
    timestamps = "0\n100000000\n200000000\n"
    accel = "0,0,9.81,900000000\n0,0,9.81,1100000000\n0,0,9.81,1300000000\n"
    gyro = "0,0,0,900000000\n0,0,0,1100000000\n"
    archive = tmp_path / "capture.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("capture_manifest.json", json.dumps(manifest))
        bundle.write(video, "walk.mp4")
        bundle.writestr("walk_timestamps.csv", timestamps)
        bundle.writestr("walk_accel.csv", accel)
        bundle.writestr("walk_gyro.csv", gyro)
    report = import_capture_bundle(archive, tmp_path / "scan")
    imported_times = json.loads((tmp_path / "scan" / "capture" / "video_timestamps_ns.json").read_text())
    assert imported_times == [1_000_000_000, 1_100_000_000, 1_200_000_000]
    assert report["video"]["duration_s"] == pytest.approx(0.2, abs=0.02)
    assert report["coverage"]["imu_start_ns"] == 900_000_000
    assert report["coverage"]["imu_end_ns"] == 1_100_000_000
    assert report["coverage"]["ends_after_video"] is False
    assert report["metric_vio_allowed"] is False


def test_vio_result_rejects_indefinite_covariance() -> None:
    payload = _valid_vio_result()
    payload["poses"][0]["covariance"][0][0] = -1
    with pytest.raises(VIOError, match="positive semidefinite"):
        validate_vio_result(payload)

    payload = _valid_vio_result()
    payload["poses"][0]["covariance"][0][1] = 0.5
    with pytest.raises(VIOError, match="symmetric"):
        validate_vio_result(payload)


def test_exact_vfr_nonzero_pts_selection_preserves_source_frame_identity(tmp_path: Path) -> None:
    frame_dir = tmp_path / "source"
    frame_dir.mkdir()
    source_frames: list[np.ndarray] = []
    for index, color in enumerate(((20, 40, 200), (40, 180, 30), (200, 50, 50), (180, 180, 20))):
        image = np.full((64, 96, 3), color, dtype=np.uint8)
        cv2.putText(image, str(index), (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        source_frames.append(image)
        assert cv2.imwrite(str(frame_dir / f"frame-{index}.png"), image)
    concat = tmp_path / "frames.txt"
    concat.write_text(
        "\n".join(
            [*(f"file '{frame_dir / f'frame-{index}.png'}'\nduration {duration}" for index, duration in enumerate((0.11, 0.27, 0.19, 0.33))),
             f"file '{frame_dir / 'frame-3.png'}'"]
        ),
        encoding="utf-8",
    )
    video = tmp_path / "vfr_nonzero_pts.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat), "-vf", "setpts=PTS+2/TB", "-fps_mode", "vfr", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-video_track_timescale", "1000", str(video),
        ],
        check=True,
    )
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(video)],
        check=True, capture_output=True, text=True,
    )
    source_timestamps = [
        float(row["best_effort_timestamp_time"])
        for row in json.loads(probe.stdout)["frames"]
    ]
    assert len(source_timestamps) >= 4
    assert source_timestamps[0] >= 2.0
    assert len({round(source_timestamps[index + 1] - source_timestamps[index], 4) for index in range(len(source_timestamps) - 1)}) > 1
    capture_timestamps = [int(round((value + 10.0) * 1e9)) for value in source_timestamps]
    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    settings = FramePreparationSettings(
        candidate_fps=1000.0, max_candidate_frames=16, candidate_edge_px=96, max_edge_px=96,
    )
    candidates = _extract_candidates(video, candidate_dir, 4.0, settings, capture_timestamps)
    indices = json.loads((candidate_dir / "candidate_source_indices.json").read_text())
    timestamps = json.loads((candidate_dir / "candidate_timestamps_ns.json").read_text())
    assert indices == sorted(set(indices))
    assert all(0 <= index < len(source_timestamps) for index in indices)
    assert timestamps == [capture_timestamps[index] for index in indices]
    decoded = cv2.VideoCapture(str(video))
    decoded_frames: list[np.ndarray] = []
    while True:
        ok, image = decoded.read()
        if not ok:
            break
        decoded_frames.append(image)
    decoded.release()
    assert len(decoded_frames) > max(indices)
    for candidate, source_index in zip(candidates, indices, strict=True):
        extracted = cv2.imread(str(candidate), cv2.IMREAD_COLOR)
        assert extracted is not None
        assert float(np.mean(np.abs(extracted.astype(np.int16) - decoded_frames[source_index].astype(np.int16)))) < 35.0


def _calibration_prior(manifest: dict) -> dict:
    return {
        "schema": VIO_CALIBRATION_PRIOR_SCHEMA, "status": "provisional",
        "capture_id": manifest["capture_id"],
        **{key: copy.deepcopy(manifest[key]) for key in ("camera", "extrinsics", "clocks", "imu")},
        "provenance": {"source": "explicit test assumptions; not admitted calibration"},
    }


def _calibration_result() -> dict:
    result = _valid_vio_result()
    sigma = [0.1] * 3 + [0.05] * 3 + [0.02]
    calibration = {
        "T_imu_camera": np.eye(4).tolist(), "camera_to_imu_offset_s": 0.0, "imu_to_camera_offset_ns": 0.0,
        "covariance": np.diag(np.square(sigma)).tolist(),
        "covariance_convention": CALIBRATION_COVARIANCE_CONVENTION,
        "covariance_order": ["theta_x_rad", "theta_y_rad", "theta_z_rad", "p_I_in_C_x_m", "p_I_in_C_y_m", "p_I_in_C_z_m", "camera_to_imu_offset_s"],
    }
    result.update({
        "schema": VIO_CALIBRATION_SCHEMA, "accepted_for_metric_vio": False, "status": "provisional",
        "scale": {"mode": "provisional_metric", "source": "provisional_imu_camera_prior_online_calibration"},
        "bounds": dict(CALIBRATION_BOUNDS), "camera_intrinsics_optimized": False,
        "initial_calibration": copy.deepcopy(calibration), "calibration": copy.deepcopy(calibration),
    })
    for row in result["poses"]:
        row["calibration"] = copy.deepcopy(calibration)
    return result


@pytest.mark.parametrize("invalid", ["capture", "status", "uncertainty", "offset", "rotation", "noise", "provenance"])
def test_online_calibration_prior_rejects_unbound_or_unbounded_assumptions(invalid: str) -> None:
    prior = _calibration_prior(_manifest())
    if invalid == "capture":
        prior["capture_id"] = "another-capture"
    elif invalid == "status":
        prior["status"] = "admitted"
    elif invalid == "uncertainty":
        prior["uncertainty"] = {"time_offset_std_s": 0.2}
    elif invalid == "offset":
        prior["clocks"]["imu_to_camera_offset_ns"] = 200_000_000
    elif invalid == "rotation":
        prior["extrinsics"]["T_imu_camera"][0][0] = -1
    elif invalid == "noise":
        prior["imu"]["noise"]["gyroscope_noise_density"] = 0
    else:
        prior["provenance"] = {}
    with pytest.raises(VIOError):
        validate_vio_calibration_prior(prior, "test-capture")


def test_online_calibration_materializes_without_mutating_capture_admission(tmp_path: Path) -> None:
    capture_dir, report, settings, _ = _dense_capture_fixture(tmp_path)
    report["schema"] = "noesis.phone_capture.v1"
    prior = _calibration_prior(report["manifest"])
    report["metric_vio_allowed"] = False
    report["manifest"]["extrinsics"]["T_imu_camera"] = None
    report["manifest"]["clocks"]["imu_to_camera_offset_ns"] = None
    report["manifest"]["imu"]["noise"] = {}
    before = copy.deepcopy(report)
    input_root, config = materialize_openvins_calibration_input(capture_dir, report, tmp_path / "calibration", settings, lambda *_: None, prior=prior)
    assert report == before
    assert report["metric_vio_allowed"] is False
    metadata = json.loads((input_root / "openvins_input.json").read_text())
    assert metadata["schema"] == "noesis.phone_capture.openvins_calibration_input.v1"
    assert metadata["mode"] == "calibration"
    assert metadata["accepted_for_metric_vio"] is False
    assert metadata["capture_metric_vio_allowed"] is False
    assert metadata["calibration_prior"]["bounds"] == CALIBRATION_BOUNDS
    config_text = config.read_text()
    for text in ("calib_cam_extrinsics: true", "calib_cam_timeoffset: true", "calib_cam_intrinsics: false", "init_dyn_use: true", "init_dyn_mle_opt_calib: false", "multi_threading_subs: false"):
        assert text in config_text
    with pytest.raises(VIOError, match="blocked"):
        materialize_openvins_input(capture_dir, report, tmp_path / "metric", settings, lambda *_: None)
    with pytest.raises(VIOError, match="cannot run as admitted"):
        run_openvins(input_root, tmp_path / "wrong-mode", VIOSettings(executable=Path(sys.executable), config=config), lambda *_: None)


@pytest.mark.parametrize("invalid", ["admission", "missing_trace", "offset_sign", "covariance", "bound", "final_state"])
def test_online_calibration_result_rejects_unsafe_or_incomplete_trace(invalid: str) -> None:
    result = _calibration_result()
    if invalid == "admission":
        result["accepted_for_metric_vio"] = True
    elif invalid == "missing_trace":
        del result["poses"][0]["calibration"]
    elif invalid == "offset_sign":
        result["poses"][0]["calibration"]["imu_to_camera_offset_ns"] = 1_000_000
    elif invalid == "covariance":
        result["poses"][0]["calibration"]["covariance"][0][0] = -1
    elif invalid == "bound":
        result["poses"][0]["calibration"]["T_imu_camera"][0][3] = 0.3
    else:
        result["calibration"]["T_imu_camera"][0][3] = 0.01
    with pytest.raises(VIOError):
        validate_vio_calibration_result(result)


def test_provisional_trajectory_cannot_be_consumed_as_admitted_vio() -> None:
    result = _calibration_result()
    assert validate_vio_calibration_result(result)["accepted_for_metric_vio"] is False
    with pytest.raises(VIOError, match="must use"):
        validate_vio_result(result)


def test_online_calibration_runner_binds_native_prior_and_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    capture_dir, report, settings, _ = _dense_capture_fixture(tmp_path)
    report.update({"schema": "noesis.phone_capture.v1", "metric_vio_allowed": False})
    input_root, config = materialize_openvins_calibration_input(capture_dir, report, tmp_path / "calibration", settings, lambda *_: None, prior=_calibration_prior(report["manifest"]))
    metadata = json.loads((input_root / "openvins_input.json").read_text())
    output = _calibration_result()
    for key in ("capture_id", "camera_sensor_id", "time_domain"):
        output["frame"][key] = metadata[key]
    for index, row in enumerate(output["poses"]):
        row.update({"capture_time_ns": metadata["frame_mapping"][index]["capture_time_ns"], "source_frame_index": index})
    mask = metadata["camera_preprocessing"]["invalid_ray_mask"]
    output["camera_mask"] = {"applied": True, "invalid_pixel_count": mask["invalid_pixel_count"], "resolution_px": [96, 64]}

    class Process:
        def __init__(self, command, **_kwargs):
            assert command[command.index("--mode") + 1] == "calibration"
            assert command[command.index("--calibration-rotation-std-rad") + 1] == "0.1"
            Path(command[command.index("--output") + 1]).write_text(json.dumps(output))

        def wait(self, **_kwargs):
            return 0

    monkeypatch.setattr(subprocess, "Popen", Process)
    result = run_openvins_calibration(input_root, tmp_path / "result", VIOSettings(executable=Path(sys.executable), config=config), lambda *_: None)
    assert result["accepted_for_metric_vio"] is False
    assert result["calibration_prior"]["capture_id"] == report["manifest"]["capture_id"]
    assert (tmp_path / "result/vio_calibration_result.json").is_file()
    output["initial_calibration"]["covariance"][0][0] = 0.02
    with pytest.raises(VIOError, match="did not apply"):
        run_openvins_calibration(input_root, tmp_path / "wrong-prior", VIOSettings(executable=Path(sys.executable), config=config), lambda *_: None)
