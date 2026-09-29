from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from noesis.calibration.pose_v1 import pose_to_E_col_major
from noesis.calibration.manager import CalibrationValidationError


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "generate_v3dt_caminfo.py"
SPEC = importlib.util.spec_from_file_location("generate_v3dt_caminfo_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


@pytest.fixture(autouse=True)
def canonical_projection_environment(monkeypatch):
    for name in (
        "NOESIS_V3DT_CAMINFO_WORLD_AXES",
        "NOESIS_V3DT_CAMINFO_WORLD_SCALE",
        "NOESIS_V3DT_CAMINFO_MATRIX_TYPE",
        "NOESIS_V3DT_CAMINFO_INVERT_E",
        "NOESIS_V3DT_CAMINFO_Y_FLIP",
    ):
        monkeypatch.delenv(name, raising=False)


def test_rectified_upward_person_projects_above_ground_contact() -> None:
    # Canonical backend Y is up; OpenCV pixel Y is down. This known camera
    # pose is independent of generated camInfo and exposes a second image flip.
    pose = {
        "position": [0.0, 2.5, 0.0],
        "yaw_pitch_roll_deg": [0.0, 15.0, 0.0],
        "rotation_order": "YXZ",
        "frame": "backend_world_m",
    }
    intrinsics = np.array([[625.0, 0.0, 960.0], [0.0, 625.0, 540.0], [0, 0, 1]])
    projection = generator._projection_matrix(
        intrinsics, pose_to_E_col_major(pose), target_h=1080
    )
    # The tracker receives x,z,y: its positive Z is canonical positive Y.
    foot = projection @ np.array([0.0, 5.0, 0.0, 1.0])
    head = projection @ np.array([0.0, 5.0, 1.7, 1.0])
    foot_uv, head_uv = foot[:2] / foot[2], head[:2] / head[2]
    assert foot[2] > 0 and head[2] > 0
    assert 0 < head_uv[1] < foot_uv[1] < 1080
    assert head_uv[0] == pytest.approx(foot_uv[0])


def test_shipped_caminfo_matches_current_rectified_camera_contract(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--cameras-config", str(REPO_ROOT / "DS9/config/cameras_v3dt.yaml"),
            "--calibration", str(REPO_ROOT / "config/camera_calibration.json"),
            "--pipeline-config", str(REPO_ROOT / "DS9/config/infer_v3dt.yaml"),
            "--alignment", str(REPO_ROOT / "config/ply_alignment.json"),
            "--output-dir", str(tmp_path),
            "--model-height", "1.7",
            "--model-radius", "0.35",
        ],
    )
    generator.main()
    cameras = yaml.safe_load((REPO_ROOT / "DS9/config/cameras_v3dt.yaml").read_text())
    pipeline = yaml.safe_load((REPO_ROOT / "DS9/config/infer_v3dt.yaml").read_text())
    manager = generator._projection_calibration(
        REPO_ROOT / "DS9/config/infer_v3dt.yaml",
        REPO_ROOT / "DS9/config/cameras_v3dt.yaml",
        REPO_ROOT / "config/camera_calibration.json",
        REPO_ROOT / "config/ply_alignment.json",
    )
    for source_id, camera in cameras["cameras"].items():
        name = camera["name"]
        filename = f"camInfo_{name}.yml"
        generated = yaml.safe_load((tmp_path / filename).read_text())
        shipped = yaml.safe_load(
            (REPO_ROOT / "DS9/config/v3dt/caminfo_baseline" / filename).read_text()
        )
        snapshot = manager.world_snapshot(int(source_id), name)
        assert snapshot is not None
        k = snapshot.intrinsics
        e = np.array(snapshot.extrinsics_col_major).reshape(4, 4, order="F")
        # Camera-model matrices must project the same pixel observations as
        # canonical calibration after only the declared tracker axis swap.
        axis_swap = np.eye(4)[:, [0, 2, 1, 3]]
        expected = k @ e[:3] @ axis_swap
        expected_native = expected
        expected_binding = {
            "world_frame_id": snapshot.world_frame_id,
            "world_frame_revision": snapshot.world_frame_revision,
            "frame_transform_sha256": snapshot.frame_transform_sha256,
            "camera_calibration_sha256": snapshot.camera_calibration_sha256,
            "image_size": list(snapshot.image_size),
            "floor_y": snapshot.floor_y,
        }
        if pipeline["v3dt"].get("caminfo_pixel_space") == "tracker":
            tracker_size = [pipeline["tracker"]["tracker-width"], pipeline["tracker"]["tracker-height"]]
            expected_native = np.diag([
                tracker_size[0] / snapshot.image_size[0],
                tracker_size[1] / snapshot.image_size[1], 1.0,
            ]) @ expected
            expected_binding.update(projection_pixel_space="tracker", tracker_image_size=tracker_size)
        for payload in (generated, shipped):
            np.testing.assert_allclose(
                np.array(payload["projectionMatrix_3x4_w2p"]).reshape(3, 4),
                expected_native,
                atol=1e-10,
                rtol=1e-12,
            )
            assert payload["modelInfo"] == {"height": 1.7, "radius": 0.35}
            assert payload["noesis_frame_binding"] == expected_binding
        if name == "living-room":
            # The accepted edge has a tilted, elevated source floor. Merely
            # swapping raw calibration axes must not reproduce target P.
            raw = manager.snapshot(int(source_id), name)
            assert raw is not None
            raw_e = np.array(raw.extrinsics_col_major).reshape(4, 4, order="F")
            assert not np.allclose(expected, k @ raw_e[:3] @ axis_swap)
            point = np.array([3.5, 0.0, 4.0, 1.0])
            canonical_pixel = k @ e[:3] @ point
            tracker_pixel = expected @ point[[0, 2, 1, 3]]
            np.testing.assert_allclose(canonical_pixel, tracker_pixel)


def test_generator_rejects_legacy_flip_before_writing(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("NOESIS_V3DT_CAMINFO_Y_FLIP", "1")
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--output-dir", str(tmp_path)])
    with pytest.raises(ValueError, match="canonical camInfo frame contract"):
        generator.main()
    assert list(tmp_path.iterdir()) == []


def test_generator_rejects_stale_accepted_frame_binding(tmp_path) -> None:
    calibration = json.loads((REPO_ROOT / "config/camera_calibration.json").read_text())
    calibration["cameras"]["living-room"]["pose"]["position"][0] += 0.25
    changed = tmp_path / "camera_calibration.json"
    changed.write_text(json.dumps(calibration))
    with pytest.raises(CalibrationValidationError, match="calibration revision mismatch"):
        generator._projection_calibration(
            REPO_ROOT / "DS9/config/infer_v3dt.yaml",
            REPO_ROOT / "DS9/config/cameras_v3dt.yaml",
            changed,
            REPO_ROOT / "config/ply_alignment.json",
        )
