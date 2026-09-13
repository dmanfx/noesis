from __future__ import annotations

import io
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from .app import create_app
from .capture import CaptureImportError, import_capture_bundle, validate_capture_manifest
from .test_browser_capture import _settings, _wait_for_ready
from .test_companion_upload import CAMERA_ID, CAPTURE_ID, SESSION_ID, _FakeCompanionManager


SENSOR_START_NS = 18_900_000_000_000_789
FRAME_TIMES = [SENSOR_START_NS + index * 200_000_000 for index in range(6)]


@pytest.fixture(scope="module")
def encoded_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    video = tmp_path_factory.mktemp("android-video") / "camera.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=128x96:rate=5:duration=1.2",
            "-c:v", "libx264", "-bf", "0", "-pix_fmt", "yuv420p",
            "-video_track_timescale", "1000000", str(video),
        ],
        check=True,
    )
    return video


def _manifest(*, acquisition_mapping: bool = True) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema": "noesis.phone_capture.v1",
        "capture_id": CAPTURE_ID,
        "device": {"id": "android-installation-test", "model": "test", "os": "Android 15"},
        "video": {
            "path": "camera.mp4", "timestamp_unit": "ns",
            "timestamp_source": "camera2_sensor_timestamp",
        },
        "imu": {
            "accel_path": "accel.csv", "gyro_path": "gyro.csv",
            "axes": "x,y,z", "accel_unit": "m/s^2", "gyro_unit": "rad/s",
            "timestamp_unit": "ns", "noise": {},
        },
        "camera": {
            "id": "0", "intrinsics_source": "missing", "intrinsics": None,
            "distortion_model": "unknown", "distortion": [], "resolution_px": [128, 96],
            "orientation_deg": 0, "stabilization": "unknown",
        },
        "extrinsics": {"T_imu_camera": None},
        "clocks": {
            "camera_domain": "android.elapsedRealtimeNanos",
            "imu_domain": "android.elapsedRealtimeNanos",
            "imu_to_camera_offset_ns": None, "timestamp_source": "unverified",
        },
        "android_capture": {
            "schema": "noesis.phone_capture.android.v1",
            "capture_result_path": "capture_result.json",
            "encoder_pts_path": "encoder_pts.csv", "camera_results_path": "camera_results.jsonl",
        },
    }
    if acquisition_mapping:
        manifest["video"]["frame_timestamps_path"] = "timestamps.csv"
    return manifest


def _bundle(
    video: Path, *, acquisition_mapping: bool = True, tamper: str | None = None,
    paired: bool = False, calibrated: bool = False,
    stabilization_modes: tuple[Any, Any] = (0, 0),
    motion_evidence: bool = False,
) -> bytes:
    manifest = _manifest(acquisition_mapping=acquisition_mapping)
    if calibrated:
        manifest["camera"].update({
            "intrinsics_source": "provided",
            "intrinsics": [[100, 0, 64], [0, 100, 48], [0, 0, 1]],
            "distortion_model": "plumb_bob", "distortion": [0.01, 0, 0, 0, 0.005],
            "stabilization": "off",
        })
        manifest["extrinsics"]["T_imu_camera"] = [
            [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
        ]
        manifest["clocks"]["imu_to_camera_offset_ns"] = 0
        manifest["imu"]["noise"] = {
            "gyroscope_noise_density": 0.001, "gyroscope_random_walk": 0.0001,
            "accelerometer_noise_density": 0.01, "accelerometer_random_walk": 0.001,
        }
    if paired:
        manifest["companion_capture"] = {
            "session_id": SESSION_ID, "camera_id": CAMERA_ID, "phone_capture_id": CAPTURE_ID,
        }
    result = {
        "schema": "noesis.phone_capture.android_result.v1", "android_api_level": 35,
        "camera": {
            "id": "0", "width": 128, "height": 96,
            "timestamp_source": "REALTIME", "timestamp_base": "SENSOR", "timestamp_base_configured": True,
        },
        "timing": {
            "association_method": "encoder_pts_us_equals_sensor_timestamp_ns_div_1000",
            "exact_frame_association_verified": acquisition_mapping,
            "encoded_frame_count": 6, "matched_frame_count": 6,
            "unmatched_encoded_frame_count": 0, "duplicate_encoded_pts_count": 0,
            "duplicate_sensor_timestamp_us_count": 0,
            "sensor_timestamps_monotonic": True, "encoder_pts_monotonic": True,
            "imu_coverage_verified": True,
        },
        "status": "complete", "partial": False, "failures": [],
        "stop_reason": "user", "dropped_metadata_records": 0,
        "metric_vio_allowed": False,
    }
    times = list(FRAME_TIMES)
    if tamper == "container_cadence":
        times[3] += 1_000_000
    encoder_rows = [f"{index},{timestamp // 1000},0,1000" for index, timestamp in enumerate(times)]
    camera_rows = [
        {"frame_number": index + 100, "sensor_timestamp_ns": timestamp,
         "received_elapsed_realtime_ns": timestamp + 10_000_000,
         "ois_mode": stabilization_modes[0], "eis_mode": stabilization_modes[1]}
        for index, timestamp in enumerate(times)
    ]
    if motion_evidence:
        for row in camera_rows:
            row["exposure_time_ns"] = 10_000_000
        result["sensors"] = {
            name: {"units": unit, "timestamp_source": "android_elapsed_realtime_ns",
                   "axes": "android_device_x_right_y_up_z_out_of_screen"}
            for name, unit in (("accelerometer", "m/s^2"), ("gyroscope", "rad/s"))
        }
    associations = [
        f"{index},{timestamp // 1000},{index + 100},{timestamp}"
        for index, timestamp in enumerate(times)
    ]
    if tamper == "association":
        associations[0] = f"0,{times[0] // 1000},100,{times[0] + 1}"
    elif tamper == "encoder_count":
        encoder_rows.pop()
    elif tamper == "result_count":
        result["timing"]["encoded_frame_count"] = 100
    elif tamper == "camera_timestamp":
        camera_rows[0]["sensor_timestamp_ns"] += 1000
    elif tamper == "timestamp_base":
        result["camera"]["timestamp_base_configured"] = False
    elif tamper == "android_api":
        result["android_api_level"] = 32
    elif tamper == "timestamp_source":
        result["camera"]["timestamp_source"] = "UNKNOWN"
    files = {
        "capture_manifest.json": json.dumps(manifest),
        "capture_result.json": json.dumps(result),
        "encoder_pts.csv": "encoded_index,encoded_pts_us,flags,size_bytes\n" + "\n".join(encoder_rows) + "\n",
        "camera_results.jsonl": "\n".join(json.dumps(row) for row in camera_rows) + "\n",
    }
    if acquisition_mapping:
        files["timestamps.csv"] = "encoded_index,encoded_pts_us,frame_number,timestamp_ns\n" + "\n".join(associations) + "\n"
    for kind in ("accel", "gyro"):
        step = 5_000_000 if motion_evidence else 100_000_000
        count = 280 if motion_evidence else 14
        sensor_rows = [
            f"{SENSOR_START_NS - 100_000_000 + index * step},0,0,{9.81 if kind == 'accel' else 0},0,0,0,3,{SENSOR_START_NS + index * step}"
            for index in range(count)
        ]
        files[f"{kind}.csv"] = "timestamp_ns,x,y,z,bias_x,bias_y,bias_z,accuracy,received_elapsed_realtime_ns\n" + "\n".join(sensor_rows) + "\n"
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as bundle:
        bundle.write(video, "camera.mp4")
        for name, content in files.items():
            bundle.writestr(name, content)
    return output.getvalue()


@pytest.mark.parametrize(
    ("modes", "tamper", "expected_stabilization", "admitted"),
    [((0, 0), None, "off", True), ((1, 0), None, "on", False),
     ((0, None), None, "unknown", False), ((False, 0), None, "unknown", False),
     ((0, 0), "association", "unknown", False)],
)
def test_android_calibrated_admission_uses_verified_timing_and_actual_modes(
    tmp_path: Path, encoded_video: Path, modes: tuple[Any, Any], tamper: str | None,
    expected_stabilization: str, admitted: bool,
) -> None:
    archive = tmp_path / "calibrated.zip"
    archive.write_bytes(_bundle(
        encoded_video, calibrated=True, stabilization_modes=modes, tamper=tamper,
    ))
    report = import_capture_bundle(archive, tmp_path / "scan")
    assert report["manifest"]["camera"]["stabilization"] == expected_stabilization
    assert report["calibration"]["distortion_supported"] is True
    assert report["metric_vio_allowed"] is admitted
    # Verification is derived; do not rewrite the recorder's original manifest.
    original = json.loads((tmp_path / "scan/capture/capture_manifest.json").read_text())
    assert original["clocks"]["timestamp_source"] == "unverified"


def test_unknown_android_calibration_remains_unknown() -> None:
    normalized = validate_capture_manifest(_manifest())
    assert normalized["camera"]["intrinsics"] is None
    assert normalized["camera"]["intrinsics_source"] == "missing"
    assert normalized["camera"]["distortion_model"] == "unknown"
    assert normalized["camera"]["distortion_supported"] is False
    assert normalized["extrinsics"]["T_imu_camera"] is None
    assert normalized["clocks"]["imu_to_camera_offset_ns"] is None
    assert normalized["imu"]["noise"] == {}
    assert normalized["calibration"]["complete_for_metric_vio"] is False
    manifest = _manifest()
    manifest["camera"]["distortion"] = [0.1]
    with pytest.raises(CaptureImportError, match="calibrated model"):
        validate_capture_manifest(manifest)


@pytest.mark.parametrize(
    ("acquisition_mapping", "tamper", "verified"),
    [(False, None, False), (True, None, True), (True, "association", False)],
)
def test_android_zip_http_import_and_rgb_frame_preparation(
    tmp_path: Path, encoded_video: Path, acquisition_mapping: bool, tamper: str | None, verified: bool,
) -> None:
    payload = _bundle(encoded_video, acquisition_mapping=acquisition_mapping, tamper=tamper)
    settings = _settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/scans/sensor-bundle?name=Android%20walk", content=payload,
            headers={"Content-Type": "application/zip", "X-File-Name": "android-capture.zip"},
        )
        assert response.status_code == 201, response.text
        initial = response.json()
        capture = initial["capture"]
        assert capture["capture_kind"] == "android_camera_imu"
        assert capture["camera_acquisition_timestamp_verified"] is verified
        assert capture["metric_vio_allowed"] is False
        assert capture["raw_streams_preserved"] is True
        assert "companion_capture" not in initial
        assert client.get(capture["manifest_url"]).status_code == 200
        raw_root = settings.storage_root / initial["id"] / "capture"
        assert (raw_root / "camera.mp4").read_bytes() == encoded_video.read_bytes()
        assert (raw_root / "camera_results.jsonl").is_file()
        imu = json.loads((raw_root / "imu_normalized.json").read_text())
        assert imu["accel"][0]["timestamp_ns"] == SENSOR_START_NS - 100_000_000
        state = _wait_for_ready(client, initial["id"])
        assert state["status"] == "ready", state.get("error")
        frames = state["prepared"]["frames"]
        assert len(frames) >= 2
        if verified:
            assert json.loads((raw_root / "video_timestamps_ns.json").read_text()) == FRAME_TIMES
            assert all(row["capture_time_ns"] == FRAME_TIMES[row["source_frame_index"]] for row in frames)
        else:
            assert all(row["capture_time_ns"] is None for row in frames)
            assert all(row["timestamp_source"] == "encoded_pts" for row in frames)
        assert client.post(f"/api/scans/{initial['id']}/initiate-vio").status_code == 409


@pytest.mark.parametrize(
    "tamper",
    ["association", "encoder_count", "result_count", "camera_timestamp", "timestamp_base", "android_api", "timestamp_source", "container_cadence"],
)
def test_android_timing_contradictions_preserve_raw_rgb_without_admission(
    tmp_path: Path, encoded_video: Path, tamper: str,
) -> None:
    archive = tmp_path / "capture.zip"
    archive.write_bytes(_bundle(encoded_video, tamper=tamper))
    report = import_capture_bundle(archive, tmp_path / "scan")
    assert report["android_capture"]["camera_acquisition_timestamp_verified"] is False
    assert report["android_capture"]["errors"]
    assert report["metric_vio_allowed"] is False
    assert report["video"]["timestamp_source"] == "ffprobe.best_effort_timestamp_time"
    assert (tmp_path / "scan" / "capture" / "timestamps.csv").is_file()
    assert (tmp_path / "scan" / "capture" / "camera.mp4").read_bytes() == encoded_video.read_bytes()


def test_android_native_upload_preserves_explicit_static_companion_binding(
    tmp_path: Path, encoded_video: Path,
) -> None:
    app = create_app(_settings(tmp_path))
    manager = _FakeCompanionManager()
    app.state.phone_scan_service.companion_capture = manager
    payload = _bundle(encoded_video, paired=True, motion_evidence=True)
    with TestClient(app) as client:
        response = client.post(
            "/api/scans/sensor-bundle", content=payload,
            headers={
                "Content-Type": "application/zip", "X-File-Name": "android-capture.zip",
                "X-Companion-Session": SESSION_ID, "X-Phone-Capture-ID": CAPTURE_ID,
                "X-Companion-Camera": CAMERA_ID,
            },
        )
        assert response.status_code == 201, response.text
        state = response.json()
        assert state["companion_capture"]["session_id"] == SESSION_ID
        assert state["companion_capture"]["camera_id"] == CAMERA_ID
        assert manager.phone["capture_id"] == CAPTURE_ID
        assert manager.phone["scan_id"] == state["id"]
        assert state["capture"]["metric_vio_allowed"] is False
        ready = _wait_for_ready(client, state["id"])
        assert ready["prepared"]["imu_motion"]["candidate_count_with_motion"] > 0
        assert all(row["imu_motion"]["status"] == "available" for row in ready["prepared"]["frames"])
        assert ready["companion_capture"]["session_id"] == SESSION_ID
        assert client.get(ready["capture"]["imu_normalized_url"]).status_code == 200
        retry = client.post("/api/scans/sensor-bundle", content=payload,
                            headers={"Content-Type": "application/zip", "X-File-Name": "saved-native.zip"})
        assert retry.status_code == 200, retry.text
        assert retry.json()["id"] == state["id"]
