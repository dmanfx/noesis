"""Retired camera-marker mapping must not return as presentation authority."""
import json
from pathlib import Path
import pytest
from noesis.calibration.presentation_registration import PresentationRegistrationError, load_registration
from noesis.calibration.manager import CalibrationManager, load_camera_labels

ROOT = Path(__file__).resolve().parents[1]

def test_legacy_marker_artifact_is_rejected(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"contract":"noesis.scene_presentation.registration","contract_version":1}))
    with pytest.raises(PresentationRegistrationError, match="camera-marker registration is retired"):
        load_registration(path, authored_obj_path=tmp_path / "unused.obj")

def test_historical_marker_bytes_do_not_publish_or_supply_a_fallback():
    manager = CalibrationManager(
        cameras_yaml_path=ROOT / "config/cameras.yaml",
        camera_calibration_json_path=ROOT / "config/camera_calibration.json",
        ply_alignment_json_path=ROOT / "config/ply_alignment.json",
    )
    manager.set_camera_labels(load_camera_labels(ROOT / "config/cameras.yaml"))
    assert "scene_similarity" not in manager.calibration_bundle()["align"]
    assert manager.world_frame_presentations() == {}
    assert manager.set_align({"scene_similarity":{}})["error"] == "camera_marker_presentation_retired"
