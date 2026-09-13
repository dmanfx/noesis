from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile
from dataclasses import replace

import cv2
import numpy as np
import pytest

from tools.mapanything_phone_scan.phone_calibration import (
    PhoneCalibrationError,
    calibration_for_video,
    import_calibration_bundle,
    load_phone_calibration,
    rectification_maps,
    rectify_selected_frames,
)


def _bundle(tmp_path: Path) -> Path:
    camera = {
        "camera_id": "test-phone",
        "model": "opencv_pinhole",
        "resolution": [7680, 4320],
        "K": [[6613.925977748088, 0.0, 3817.3486099259267],
              [0.0, 6607.022109180938, 2117.928902931894], [0.0, 0.0, 1.0]],
        "D": [-0.030249635365637628, -0.005489054814002665, 0.0, 0.0, 0.0582832677148535],
        "runtime": {"use_undistorted_frames": False},
        "status": "acceptable",
    }
    numeric = io.BytesIO()
    np.savez(numeric, camera_matrix=camera["K"], distortion_coefficients=camera["D"], image_size=camera["resolution"])
    members = {
        "camera_noesis.json": json.dumps(camera).encode(),
        "camera_opencv.npz": numeric.getvalue(),
        "capture_mode.json": json.dumps({"camera_id": "test-phone", "calibration_resolution": camera["resolution"]}).encode(),
    }
    members["manifest.json"] = json.dumps({
        "camera_id": "test-phone", "session": "test-session",
        "files": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in members.items()},
    }).encode()
    result = tmp_path / "bundle.zip"
    with zipfile.ZipFile(result, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return result


@pytest.fixture
def profile_path(tmp_path: Path) -> Path:
    return import_calibration_bundle(_bundle(tmp_path), tmp_path / "profiles")


def test_import_preserves_original_bundle_and_all_five_coefficients(tmp_path: Path, profile_path: Path) -> None:
    profile = load_phone_calibration(profile_path)
    assert len(profile["D"]) == 5
    assert profile["D"][4] == 0.0582832677148535
    assert profile["metric_vio_allowed"] is False
    assert (profile_path.parent / "source_bundle.zip").read_bytes() == (tmp_path / "bundle.zip").read_bytes()
    assert import_calibration_bundle(tmp_path / "bundle.zip", tmp_path / "profiles") == profile_path


def test_import_rejects_corrupted_calibration_before_creating_profile(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    with zipfile.ZipFile(bundle) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members["camera_noesis.json"] += b" "
    with zipfile.ZipFile(bundle, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    with pytest.raises(PhoneCalibrationError, match="checksum/size mismatch"):
        import_calibration_bundle(bundle, tmp_path / "profiles")
    assert not (tmp_path / "profiles").exists()


def test_import_rejects_archive_path_traversal(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    with zipfile.ZipFile(bundle, "a") as archive:
        archive.writestr("../outside", b"unsafe")
    with pytest.raises(PhoneCalibrationError, match="plain filenames"):
        import_calibration_bundle(bundle, tmp_path / "profiles")
    assert not (tmp_path / "outside").exists()


def test_profile_parameter_edit_cannot_change_verified_calibration(profile_path: Path) -> None:
    profile = json.loads(profile_path.read_text())
    profile["K"][0][0] += 1.0
    profile_path.write_text(json.dumps(profile))
    with pytest.raises(PhoneCalibrationError, match="differs from its source: K"):
        load_phone_calibration(profile_path)


@pytest.mark.parametrize(("bound_mode", "recorded_mode", "size", "rotation", "reason"), [
    ("unbound", "uploaded_video", (7680, 4320), 0, "capture_mode_not_bound"),
    ("uploaded_video", "browser", (7680, 4320), 0, "capture_mode_mismatch"),
    ("uploaded_video", "uploaded_video", (1920, 1080), 0, "native_resolution_mismatch"),
    ("uploaded_video", "uploaded_video", (7680, 4320), 90, "encoded_rotation_requires_separate_calibration_transform"),
])
def test_profile_requires_capture_mode_and_native_pixel_match(
    profile_path: Path, bound_mode: str, recorded_mode: str, size: tuple[int, int], rotation: int, reason: str,
) -> None:
    profile, summary = calibration_for_video(
        profile_path, configured_capture_mode=bound_mode, capture_mode=recorded_mode,
        video={"width": size[0], "height": size[1], "rotation_degrees": rotation},
    )
    assert profile is None
    assert summary["status"] == "not_applied"
    assert reason in summary["reason_codes"]
    assert summary["metric_vio_allowed"] is False


def test_rectification_uses_nonzero_k3_and_preserves_provenance(profile_path: Path, tmp_path: Path) -> None:
    profile, summary = calibration_for_video(
        profile_path, configured_capture_mode="uploaded_video", capture_mode="uploaded_video",
        video={"width": 7680, "height": 4320, "rotation_degrees": 0},
    )
    assert profile is not None and summary["status"] == "applied"
    yy, xx = np.indices((720, 1280))
    image = np.repeat((((xx // 12 + yy // 12) % 2) * 255).astype(np.uint8)[..., None], 3, axis=2)
    frame = tmp_path / "frame.jpg"
    assert cv2.imwrite(str(frame), image)
    before = hashlib.sha256(frame.read_bytes()).hexdigest()
    rows = rectify_selected_frames([frame], profile, summary)
    spec = rows[0]["camera_intrinsics"]
    assert spec["source_distortion"] == profile["D"]
    assert spec["distortion_model"] == "none"
    np.testing.assert_allclose(np.asarray(spec["K"])[0], np.asarray(profile["K"])[0] / 6)
    assert rows[0]["calibration_processing"]["source_frame_sha256"] == before
    assert hashlib.sha256(frame.read_bytes()).hexdigest() != before
    assert rows[0]["calibration_processing"]["border_policy"] == "reject_rays_outside_distorted_source_in_inference"
    map_x, map_y, _ = rectification_maps(profile, (1280, 720))
    truncated = {**profile, "D": [*profile["D"][:4], 0.0]}
    wrong_x, wrong_y, _ = rectification_maps(truncated, (1280, 720))
    assert np.max(np.hypot(map_x - wrong_x, map_y - wrong_y)) > 3.0


def test_rotated_prepared_pixels_cannot_receive_landscape_calibration(profile_path: Path, tmp_path: Path) -> None:
    profile = load_phone_calibration(profile_path)
    frame = tmp_path / "portrait.jpg"
    assert cv2.imwrite(str(frame), np.zeros((1280, 720, 3), dtype=np.uint8))
    with pytest.raises(PhoneCalibrationError, match="native framing"):
        rectify_selected_frames([frame], profile, {"profile_sha256": "test"})


def test_service_exposes_imported_profile_without_admitting_metric_vio(profile_path: Path, tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from tools.mapanything_phone_scan.app import create_app
    from tools.mapanything_phone_scan.test_phone_scan import _settings

    settings = _settings(tmp_path)
    settings = replace(settings, frame=replace(
        settings.frame, phone_camera_calibration=profile_path,
        phone_camera_capture_mode="unbound",
    ))
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        calibration = response.json()["phone_camera_calibration"]
        assert calibration["status"] == "registered_capture_mode_unbound"
        assert calibration["resolution"] == [7680, 4320]
        assert calibration["distortion_coefficient_count"] == 5
        assert calibration["metric_vio_allowed"] is False
        assert "phone-calibration-note" in client.get("/").text
