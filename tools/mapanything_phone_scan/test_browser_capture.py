from __future__ import annotations

import io
import json
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from .alignment import NoesisAlignmentSettings
from .app import APP_ROOT, REPO_ROOT, PhoneScanSettings, create_app
from .browser_capture import (
    BROWSER_CAPTURE_SCHEMA,
    validate_browser_capture_manifest,
)
from .capture import CaptureImportError, CaptureImportLimits, import_capture_bundle
from .da3_inference import DA3PhoneScanSettings
from .inference import MapAnythingScanSettings
from .processing import FramePreparationSettings, prepare_video_frames, probe_video


def _settings(tmp_path: Path) -> PhoneScanSettings:
    alignment = NoesisAlignmentSettings(
        camera_id="living-room",
        target_revision=tmp_path / "living-room-revision",
        calibration_path=tmp_path / "camera_calibration.json",
        review_point_budget=10_000,
    )
    return PhoneScanSettings(
        storage_root=tmp_path / "scans",
        static_root=APP_ROOT / "static",
        three_root=REPO_ROOT / "oai2-fe" / "node_modules" / "three",
        max_upload_bytes=32 * 1024 * 1024,
        frame=FramePreparationSettings(
            candidate_fps=2.0,
            max_candidate_frames=16,
            max_selected_frames=8,
            candidate_edge_px=320,
            feature_edge_px=320,
            min_keyframe_interval_s=0.1,
            max_keyframe_interval_s=2.0,
            max_edge_px=320,
        ),
        mapanything=MapAnythingScanSettings(point_budget=10_000),
        da3=DA3PhoneScanSettings(
            point_budget=10_000,
            metric_engine_path=tmp_path / "da3metric.engine",
        ),
        alignment=alignment,
        alignment_targets=(alignment,),
        alignment_release_id="test-browser-capture",
        pcf_storage_root=tmp_path / "pcf",
        capture_limits=CaptureImportLimits(
            max_archive_bytes=32 * 1024 * 1024,
            max_uncompressed_bytes=32 * 1024 * 1024,
            max_member_bytes=32 * 1024 * 1024,
            max_files=16,
            max_imu_rows=20_000,
            max_video_timestamps=10_000,
        ),
    )


def _video(tmp_path: Path, *, no_duration: bool = False) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    for index in range(6):
        image = np.full((96, 128, 3), 25 + index * 30, dtype=np.uint8)
        cv2.rectangle(image, (8 + index * 5, 12), (68 + index * 5, 76), (220, 220, 220), 2)
        cv2.putText(image, str(index), (78, 54), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        assert cv2.imwrite(str(source / f"frame-{index:02d}.png"), image)
    output = tmp_path / "phone_walk.webm"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", "5", "-i", str(source / "frame-%02d.png"),
        "-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p",
    ]
    if no_duration:
        command.extend(["-f", "webm", "-live", "1"])
    command.append(str(output))
    subprocess.run(
        command,
        check=True,
    )
    return output


def _manifest(
    capture_id: str = "browser-test",
    *,
    samples: int = 12,
    video_frames: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    accel = [
        {
            "timestamp_ms": index * 16.666,
            "received_ms": index * 16.8,
            "x": 0.0,
            "y": 0.0,
            "z": 9.81,
        }
        for index in range(samples)
    ]
    gyro = [
        {
            "timestamp_ms": index * 16.666,
            "received_ms": index * 16.9,
            "x": 0.0,
            "y": 0.01,
            "z": 0.02,
        }
        for index in range(samples)
    ]
    return {
        "schema": BROWSER_CAPTURE_SCHEMA,
        "capture_id": capture_id,
        "video": {"path": "phone_walk.webm", "mime_type": "video/webm;codecs=vp9"},
        "device": {
            "id": f"browser-session:{capture_id}",
            "model": "Test browser",
            "user_agent": "Mozilla/5.0 test",
        },
        "camera": {"settings": {"width": 128}, "capabilities": {}},
        "timing": {"time_origin_ms": 1000, "started_ms": 10, "stopped_ms": 3000},
        "sensors": {
            "accelerometer": {
                "api": "Generic Sensor Accelerometer (including gravity)",
                "units": "m/s^2",
                "reference_frame": "device",
                "samples": accel,
            },
            "gyroscope": {
                "api": "Generic Sensor Gyroscope",
                "units": "rad/s",
                "reference_frame": "device",
                "samples": gyro,
            },
        },
        "video_frames": video_frames if video_frames is not None else [],
        "events": [],
        "stop_reason": "user",
    }


def _archive(tmp_path: Path, manifest: dict[str, Any], video: Path) -> bytes:
    archive_path = tmp_path / "browser_capture.tar"
    with tarfile.open(archive_path, "w") as archive:
        manifest_data = json.dumps(manifest).encode("utf-8")
        info = tarfile.TarInfo("capture_manifest.json")
        info.size = len(manifest_data)
        archive.addfile(info, io.BytesIO(manifest_data))
        archive.add(video, arcname="phone_walk.webm")
    return archive_path.read_bytes()


def _wait_for_ready(client: TestClient, scan_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        response = client.get(f"/api/scans/{scan_id}")
        assert response.status_code == 200
        state = response.json()
        if state["status"] in {"ready", "frame_failed"}:
            return state
        time.sleep(0.03)
    raise AssertionError("browser capture did not finish frame preparation")


def test_browser_tar_http_import_preserves_evidence_and_uses_encoded_pts(tmp_path: Path) -> None:
    video = _video(tmp_path)
    payload = _archive(
        tmp_path,
        _manifest(
            video_frames=[
                {
                    "media_time_s": 0.0,
                    "callback_time_ms": 20.0,
                    "capture_time_ms": None,
                    "frame_index_claimed": False,
                }
            ]
        ),
        video,
    )
    with TestClient(create_app(_settings(tmp_path))) as client:
        response = client.post(
            "/api/scans/sensor-bundle?name=Browser%20walk",
            content=payload,
            headers={
                "Content-Type": "application/x-tar",
                "X-File-Name": "browser-capture.tar",
            },
        )
        assert response.status_code == 201, response.text
        initial = response.json()
        capture = initial["capture"]
        assert capture["schema"] == BROWSER_CAPTURE_SCHEMA
        assert capture["capture_kind"] == "browser_camera_imu"
        assert capture["capture_source"] == "browser camera + IMU"
        assert capture["metric_vio_allowed"] is False
        assert capture["camera_acquisition_timestamp_verified"] is False
        assert capture["sensors"]["accelerometer"]["sample_count"] == 12
        assert capture["sensors"]["gyroscope"]["timestamp_source"] == "Sensor.timestamp"
        assert client.get(capture["manifest_url"]).status_code == 200
        assert client.get(capture["sensor_samples_url"]).status_code == 200

        state = _wait_for_ready(client, initial["id"])
        assert state["status"] == "ready"
        assert state["prepared"]["frame_count"] >= 2
        assert all(row["capture_time_ns"] is None for row in state["prepared"]["frames"])
        assert all(row["source_frame_index"] is None for row in state["prepared"]["frames"])
        assert all(row["timestamp_source"] == "encoded_pts" for row in state["prepared"]["frames"])

        vio = client.post(f"/api/scans/{initial['id']}/initiate-vio")
        assert vio.status_code == 409
        assert "browser" in vio.json()["detail"].lower()


def test_browser_partial_sensor_rows_are_retained_and_reported(tmp_path: Path) -> None:
    manifest = _manifest()
    malformed = dict(manifest["sensors"]["accelerometer"]["samples"][3])
    malformed["x"] = "not-a-number"
    manifest["sensors"]["accelerometer"]["samples"].insert(3, malformed)
    normalized, parsed, interruptions = validate_browser_capture_manifest(
        manifest,
        member_names={"phone_walk.webm"},
    )
    assert normalized["schema"] == BROWSER_CAPTURE_SCHEMA
    assert parsed["sensors"]["accelerometer"]["sample_count"] == 12
    assert parsed["sensors"]["accelerometer"]["invalid_sample_count"] == 1
    assert any(row["source"] == "accelerometer" for row in interruptions)


def test_browser_epoch_device_motion_timestamps_do_not_fake_callback_lag(tmp_path: Path) -> None:
    manifest = _manifest()
    for kind in ("accelerometer", "gyroscope"):
        stream = manifest["sensors"][kind]
        stream["api"] = "DeviceMotionEvent fallback"
        for index, sample in enumerate(stream["samples"]):
            sample["timestamp_ms"] = 1_700_000_000_000 + index * 16.666
            sample["timestamp_domain"] = "event.timeStamp_epoch_ms"
    _, parsed, _ = validate_browser_capture_manifest(manifest, member_names={"phone_walk.webm"})
    summary = parsed["sensors"]["accelerometer"]
    assert summary["timestamp_source"] == "DeviceMotionEvent.timestamp"
    assert summary["callback_lag_comparable"] is False
    assert summary["max_callback_lag_ms"] is None
    assert summary["callback_lag_unavailable_reason"]


def test_generic_sensor_without_clock_evidence_does_not_infer_browser_origin() -> None:
    manifest = _manifest()
    _, parsed, _ = validate_browser_capture_manifest(
        manifest, member_names={"phone_walk.webm"}
    )
    summary = parsed["sensors"]["accelerometer"]
    assert summary["timestamp_domain"] == "sensor_timestamp_domain_unverified"
    assert summary["callback_lag_comparable"] is False
    assert summary["min_callback_lag_ms"] is None


@pytest.mark.parametrize("sensor_shift_ms", [0.0, 1_416_896_128.0])
def test_declared_browser_clock_is_checked_against_actual_receipt_times(
    sensor_shift_ms: float,
) -> None:
    manifest = _manifest()
    for kind in ("accelerometer", "gyroscope"):
        for sample in manifest["sensors"][kind]["samples"]:
            sample["received_ms"] = sample["timestamp_ms"] + 5.0
            sample["timestamp_ms"] += sensor_shift_ms
            sample["timestamp_domain"] = "performance_time_origin"
    original_samples = [dict(row) for row in manifest["sensors"]["accelerometer"]["samples"]]
    _, parsed, _ = validate_browser_capture_manifest(
        manifest, member_names={"phone_walk.webm"}
    )
    summary = parsed["sensors"]["accelerometer"]
    assert summary["sample_count"] == len(original_samples)
    assert summary["timestamp_start_ms"] == original_samples[0]["timestamp_ms"]
    assert manifest["sensors"]["accelerometer"]["samples"] == original_samples
    if sensor_shift_ms:
        assert summary["callback_lag_comparable"] is False
        assert summary["max_callback_lag_ms"] is None
        assert summary["min_callback_lag_ms"] is None
        assert "contradicted" in summary["callback_lag_unavailable_reason"]
    else:
        assert summary["callback_lag_comparable"] is True
        assert summary["max_callback_lag_ms"] == 5.0
        assert summary["min_callback_lag_ms"] == 5.0


def test_browser_manifest_over_native_limit_is_admitted_only_for_browser_schema(tmp_path: Path) -> None:
    video = _video(tmp_path)
    manifest = _manifest(samples=8_000)
    _archive(tmp_path, manifest, video)
    assert len(json.dumps(manifest).encode("utf-8")) > 512 * 1024
    report = import_capture_bundle(
        tmp_path / "browser_capture.tar",
        tmp_path / "scan",
        limits=CaptureImportLimits(
            max_archive_bytes=32 * 1024 * 1024,
            max_uncompressed_bytes=32 * 1024 * 1024,
            max_member_bytes=32 * 1024 * 1024,
            max_files=16,
            max_imu_rows=10_000,
            max_video_timestamps=10_000,
        ),
    )
    assert report["schema"] == BROWSER_CAPTURE_SCHEMA
    assert report["sensors"]["accelerometer"]["sample_count"] == 8_000


def test_browser_manifest_rejects_missing_sensor_and_unsafe_video_path() -> None:
    manifest = _manifest()
    del manifest["sensors"]["gyroscope"]
    with pytest.raises(CaptureImportError, match="missing sensors.gyroscope"):
        validate_browser_capture_manifest(manifest, member_names={"phone_walk.webm"})

    manifest = _manifest()
    manifest["video"]["path"] = "../phone_walk.webm"
    with pytest.raises(CaptureImportError, match="unsafe member path"):
        validate_browser_capture_manifest(manifest, member_names={"phone_walk.webm"})


def test_strict_8k_capture_rejects_negotiated_and_encoded_downgrades(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["camera"]["requested"] = {"mode": "8k"}
    with pytest.raises(CaptureImportError, match="strict 8K capture"):
        validate_browser_capture_manifest(manifest)
    manifest["camera"]["settings"] = {
        "width": 7680, "height": 4320, "resizeMode": "none", "facingMode": "environment"
    }
    normalized, _, _ = validate_browser_capture_manifest(manifest)
    assert normalized["camera"]["requested"]["mode"] == "8k"
    _archive(tmp_path, manifest, _video(tmp_path))
    with pytest.raises(CaptureImportError, match="strict 8K recording encoded 128x96"):
        import_capture_bundle(tmp_path / "browser_capture.tar", tmp_path / "scan")
    assert not (tmp_path / "scan" / "capture").exists()


def test_browser_preparation_uses_validated_encoded_duration_for_live_webm(tmp_path: Path) -> None:
    source_video = _video(tmp_path, no_duration=True)
    scan_dir = tmp_path / "scan"
    capture_dir = scan_dir / "capture"
    capture_dir.mkdir(parents=True)
    video = capture_dir / "phone_walk.webm"
    source_video.replace(video)
    assert probe_video(video)["duration_s"] is None
    (capture_dir / "capture_import.json").write_text(
        json.dumps(
            {
                "schema": BROWSER_CAPTURE_SCHEMA,
                "video": {
                    "encoded_duration_s": 1.0,
                    "encoded_timestamp_source": "ffprobe.best_effort_timestamp_time",
                    "timestamp_source": "browser_callback_observation",
                },
            }
        ),
        encoding="utf-8",
    )

    prepared = prepare_video_frames(
        video,
        scan_dir,
        FramePreparationSettings(
            candidate_fps=2.0,
            max_candidate_frames=16,
            max_selected_frames=8,
            candidate_edge_px=320,
            feature_edge_px=320,
            min_keyframe_interval_s=0.1,
            max_keyframe_interval_s=2.0,
            max_edge_px=320,
        ),
        lambda _fraction, _message: None,
    )

    assert prepared["probe"]["duration_s"] == pytest.approx(1.0)
    assert prepared["probe"]["duration_source"] == "capture_import.video.encoded_duration_s"
    assert prepared["effective_fps"] == pytest.approx(prepared["frame_count"] / 1.0)
    assert all(row["capture_time_ns"] is None for row in prepared["frames"])
    assert all(row["source_frame_index"] is None for row in prepared["frames"])
    assert all(row["timestamp_source"] == "encoded_pts" for row in prepared["frames"])


def test_health_advertises_secure_capture_setup_and_ca_endpoint_is_public_cert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = tmp_path / "ca.key"
    generated_ca = tmp_path / "generated-ca.pem"
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-keyout", str(key), "-out", str(generated_ca), "-days", "1",
            "-subj", "/CN=test-browser-ca",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    ca = tmp_path / "ca.pem"
    ca.write_bytes(generated_ca.read_bytes())
    monkeypatch.setenv("NOESIS_PHONE_SCAN_TLS_CA_CERT_FILE", str(ca))
    with TestClient(create_app(_settings(tmp_path))) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["secure_capture_url"].startswith("https://TauntonMainframe.local:")
        assert health.json()["ca_certificate_url"] == "/api/browser-capture/ca-certificate"
        response = client.get(health.json()["ca_certificate_url"])
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/x-x509-ca-cert"
        assert response.headers["content-disposition"] == 'attachment; filename="Noesis-Room-Walk-CA.crt"'
        assert b"BEGIN CERTIFICATE" in response.content
        assert b"PRIVATE KEY" not in response.content

        ca.write_bytes(generated_ca.read_bytes() + b"\n" + key.read_bytes())
        response = client.get(health.json()["ca_certificate_url"])
        assert response.status_code == 500
        assert "private-key" in response.json()["detail"]

    monkeypatch.setenv("NOESIS_PHONE_SCAN_TLS_CA_CERT_FILE", str(tmp_path / "missing.pem"))
    with TestClient(create_app(_settings(tmp_path / "missing-app"))) as client:
        response = client.get("/api/browser-capture/ca-certificate")
        assert response.status_code == 404
