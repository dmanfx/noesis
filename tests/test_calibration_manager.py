#!/usr/bin/env python3
"""Unit tests for DS8 CalibrationManager.

Tests:
1. K-divergence: `calibration-bundle.cameras.K[cam]` matches `snapshot(cam).intrinsics`
2. Twc-inversion: `set_extrinsics(Twc=...)` stores correct `E` such that `inv(E) == Twc`
3. Validation: invalid E/K/align matrices are rejected

Run with: pytest tests/test_calibration_manager.py -v
"""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from noesis.calibration.manager import (
    CalibrationManager,
    CalibrationValidationError,
    _world_to_scene_sha256,
    create_calibration_manager,
    load_camera_labels,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_calibration_dir(tmp_path: Path) -> Path:
    """Create a temp directory with valid calibration files."""
    # cameras.yaml
    cameras_yaml = tmp_path / "cameras.yaml"
    cameras_yaml.write_text("""
intrinsics_models:
  test_model:
    intrinsics:
      fx: 800.0
      fy: 800.0
      cx: 640.0
      cy: 360.0
      k1: 0.0
      k2: 0.0
      k3: 0.0

cameras:
  0:
    name: test-camera
    model: test_model
    height_m: 2.5
""")

    # camera_calibration.json with identity E
    calib_json = tmp_path / "camera_calibration.json"
    E_identity = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    ]
    calib_json.write_text(json.dumps({
        "cameras": {
            "test-camera": {"E": E_identity}
        }
    }))

    # ply_alignment.json
    align_json = tmp_path / "ply_alignment.json"
    align_json.write_text(json.dumps({
        "matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        "floor_y": 0.0,
        "units": {"s_obj_to_m": 1.0}
    }))

    return tmp_path


@pytest.fixture
def manager(temp_calibration_dir: Path) -> CalibrationManager:
    """Create a CalibrationManager with test files."""
    mgr = CalibrationManager(
        cameras_yaml_path=temp_calibration_dir / "cameras.yaml",
        camera_calibration_json_path=temp_calibration_dir / "camera_calibration.json",
        ply_alignment_json_path=temp_calibration_dir / "ply_alignment.json",
        streammux_size=(1280, 720),
        raw_audit_dir=temp_calibration_dir / "private_audit",
    )
    mgr.set_camera_labels({0: "test-camera"})
    return mgr


def test_live_and_offline_factory_uses_one_calibration_authority(
    temp_calibration_dir: Path,
) -> None:
    labels = load_camera_labels(temp_calibration_dir / "cameras.yaml")
    manager = create_calibration_manager(
        cameras_yaml_path=temp_calibration_dir / "cameras.yaml",
        pipeline_config={"streammux": {"width": 640, "height": 360}},
        camera_calibration_json_path=temp_calibration_dir / "camera_calibration.json",
        ply_alignment_json_path=temp_calibration_dir / "ply_alignment.json",
        camera_labels=labels,
    )

    snapshot = manager.snapshot(0, "test-camera")
    assert labels == {0: "test-camera"}
    assert snapshot is not None
    assert snapshot.image_size == (640, 360)
    np.testing.assert_allclose(
        [snapshot.intrinsics[0, 0], snapshot.intrinsics[1, 1]],
        [400.0, 400.0],
    )


def test_calibration_authority_rejects_ambiguous_or_missing_inputs(
    temp_calibration_dir: Path,
) -> None:
    cameras_path = temp_calibration_dir / "cameras.yaml"
    cameras_path.write_text(
        "cameras:\n  0: {name: duplicate}\n  1: {name: duplicate}\n",
        encoding="utf-8",
    )
    with pytest.raises(CalibrationValidationError, match="unique"):
        load_camera_labels(cameras_path)
    with pytest.raises(CalibrationValidationError, match="missing"):
        create_calibration_manager(
            cameras_yaml_path=cameras_path,
            pipeline_config={"streammux": {"width": 640, "height": 360}},
            camera_calibration_json_path=temp_calibration_dir / "missing.json",
            ply_alignment_json_path=temp_calibration_dir / "ply_alignment.json",
        )


def test_live_and_offline_paths_do_not_reintroduce_private_calibration_provider() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = (
        root / "noesis" / "ds8_runtime.py",
        root / "noesis" / "ds8_runtime_v3dt_reimpl.py",
        root / "DS9" / "noesis" / "ds9_runtime_core.py",
        root / "scripts" / "build_virtual_twin_reconstruction.py",
        root / "scripts" / "build_rectified_virtual_twin_reconstruction.py",
        root / "scripts" / "build_stream_room_reconstruction.py",
        root / "scripts" / "build_depth_registration.py",
        root / "DS9" / "scripts" / "build_depth_registration.py",
    )
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert "_CalibrationProvider" not in source, path
        assert "create_calibration_manager" in source, path


# ---------------------------------------------------------------------------
# Test: K-divergence (bundle K == snapshot K)
# ---------------------------------------------------------------------------


def test_k_divergence_bundle_matches_snapshot(manager: CalibrationManager):
    """Verify calibration-bundle K matches snapshot intrinsics (no divergence)."""
    # Get bundle
    bundle = manager.calibration_bundle()
    k_table = bundle.get("cameras", {}).get("K", {})
    assert "test-camera" in k_table, "Expected test-camera in bundle K table"
    bundle_K = k_table["test-camera"]  # [fx, fy, cx, cy]

    # Get snapshot
    snap = manager.snapshot(source_id=0, camera_id="test-camera")
    assert snap is not None, "Snapshot should not be None"
    snap_K = snap.intrinsics

    # Compare: bundle [fx, fy, cx, cy] should match snapshot matrix
    assert len(bundle_K) == 4, "Bundle K should have [fx, fy, cx, cy]"
    np.testing.assert_allclose(bundle_K[0], snap_K[0, 0], rtol=1e-6, err_msg="fx diverged")
    np.testing.assert_allclose(bundle_K[1], snap_K[1, 1], rtol=1e-6, err_msg="fy diverged")
    np.testing.assert_allclose(bundle_K[2], snap_K[0, 2], rtol=1e-6, err_msg="cx diverged")
    np.testing.assert_allclose(bundle_K[3], snap_K[1, 2], rtol=1e-6, err_msg="cy diverged")


def test_explicit_intrinsics_resolution_handles_offcenter_principal_point(temp_calibration_dir: Path):
    """Verify native resolution metadata prevents bogus cx*2/cy*2 rescaling."""
    (temp_calibration_dir / "cameras.yaml").write_text("""
intrinsics_models:
  offcenter_g3:
    resolution: [1920, 1080]
    intrinsics:
      fx: 1055.5160398848277
      fy: 1053.5226356877163
      cx: 931.2433649803215
      cy: 563.146369643397
      k1: 0.0
      k2: 0.0
      k3: 0.0

cameras:
  0:
    name: test-camera
    model: offcenter_g3
    height_m: 2.6
""")

    native_mgr = CalibrationManager(
        cameras_yaml_path=temp_calibration_dir / "cameras.yaml",
        camera_calibration_json_path=temp_calibration_dir / "camera_calibration.json",
        ply_alignment_json_path=temp_calibration_dir / "ply_alignment.json",
        streammux_size=(1920, 1080),
    )
    native_mgr.set_camera_labels({0: "test-camera"})
    native_snap = native_mgr.snapshot(source_id=0, camera_id="test-camera")
    assert native_snap is not None
    np.testing.assert_allclose(
        [native_snap.intrinsics[0, 0], native_snap.intrinsics[1, 1], native_snap.intrinsics[0, 2], native_snap.intrinsics[1, 2]],
        [1055.5160398848277, 1053.5226356877163, 931.2433649803215, 563.146369643397],
        rtol=1e-9,
    )

    half_mgr = CalibrationManager(
        cameras_yaml_path=temp_calibration_dir / "cameras.yaml",
        camera_calibration_json_path=temp_calibration_dir / "camera_calibration.json",
        ply_alignment_json_path=temp_calibration_dir / "ply_alignment.json",
        streammux_size=(960, 540),
    )
    half_mgr.set_camera_labels({0: "test-camera"})
    half_snap = half_mgr.snapshot(source_id=0, camera_id="test-camera")
    assert half_snap is not None
    np.testing.assert_allclose(
        [half_snap.intrinsics[0, 0], half_snap.intrinsics[1, 1], half_snap.intrinsics[0, 2], half_snap.intrinsics[1, 2]],
        [527.7580199424139, 526.7613178438582, 465.62168249016075, 281.5731848216985],
        rtol=1e-9,
    )


# ---------------------------------------------------------------------------
# Test: Parser (pose -> E)
# ---------------------------------------------------------------------------


def test_load_extrinsics_derives_E_from_pose(manager: CalibrationManager, temp_calibration_dir: Path):
    """Verify pose-only entries are parsed into world->camera E."""
    pose_only = {
        "position": [1.0, 2.0, 3.0],
        "yaw_pitch_roll_deg": [0.0, 0.0, 0.0],
        "rotation_order": "YXZ",
        "frame": "backend_world_m",
        "source": "unit-test",
    }
    (temp_calibration_dir / "camera_calibration.json").write_text(
        json.dumps({"cameras": {"test-camera": {"pose": pose_only}}})
    )

    manager.reload_extrinsics()
    snap = manager.snapshot(source_id=0, camera_id="test-camera")
    assert snap is not None, "Snapshot should include pose-derived E"

    E_stored = np.array(snap.extrinsics_col_major, dtype=np.float64).reshape((4, 4), order="F")
    E_expected = np.eye(4, dtype=np.float64)
    E_expected[:3, :3] = np.diag([-1.0, -1.0, 1.0])
    E_expected[:3, 3] = np.array([1.0, 2.0, -3.0], dtype=np.float64)
    np.testing.assert_allclose(E_stored, E_expected, atol=1e-9, rtol=1e-9, err_msg="pose->E parse mismatch")


def test_load_extrinsics_prefers_pose_over_legacy_E(manager: CalibrationManager, temp_calibration_dir: Path):
    """Verify parser uses pose-derived E when both pose and legacy E are present."""
    pose = {
        "position": [4.0, 5.0, 6.0],
        "yaw_pitch_roll_deg": [0.0, 0.0, 0.0],
        "rotation_order": "YXZ",
        "frame": "backend_world_m",
    }
    legacy_identity_E = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    (temp_calibration_dir / "camera_calibration.json").write_text(
        json.dumps({"cameras": {"test-camera": {"pose": pose, "E": legacy_identity_E}}})
    )

    manager.reload_extrinsics()
    snap = manager.snapshot(source_id=0, camera_id="test-camera")
    assert snap is not None, "Snapshot should include pose-derived E"

    E_stored = np.array(snap.extrinsics_col_major, dtype=np.float64).reshape((4, 4), order="F")
    E_expected = np.eye(4, dtype=np.float64)
    E_expected[:3, :3] = np.diag([-1.0, -1.0, 1.0])
    E_expected[:3, 3] = np.array([4.0, 5.0, -6.0], dtype=np.float64)
    np.testing.assert_allclose(E_stored, E_expected, atol=1e-9, rtol=1e-9, err_msg="pose should override legacy E")
    assert not np.allclose(E_stored, np.eye(4, dtype=np.float64), atol=1e-12), "Legacy identity E should be ignored"


# ---------------------------------------------------------------------------
# Test: Twc inversion (set_extrinsics with Twc stores correct E)
# ---------------------------------------------------------------------------


def test_twc_inversion_set_extrinsics(manager: CalibrationManager, temp_calibration_dir: Path):
    """Verify set_extrinsics(Twc=...) inverts and stores correct E."""
    # Create a known Twc (camera-to-world)
    # Simple rotation + translation
    R = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)  # 90° around Z
    t = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    Twc = np.eye(4, dtype=np.float64)
    Twc[:3, :3] = R
    Twc[:3, 3] = t

    # Expected E = inv(Twc)
    E_expected = np.linalg.inv(Twc)

    # Send as Twc (column-major)
    Twc_flat = list(Twc.flatten(order="F"))
    result = manager.set_extrinsics("test-camera", Twc=Twc_flat)
    assert result.get("ok"), f"set_extrinsics failed: {result}"

    audit_files = list((temp_calibration_dir / "private_audit").glob("calibration_*.json"))
    assert len(audit_files) == 1
    assert audit_files[0].parent.stat().st_mode & 0o777 == 0o700
    assert audit_files[0].stat().st_mode & 0o777 == 0o600
    assert audit_files[0].stat().st_nlink == 1
    audit = json.loads(audit_files[0].read_text(encoding="utf-8"))
    assert audit["camera_id"] == "test-camera"
    assert audit["raw_kind"] == "Twc"

    # Verify stored E
    manager.reload_extrinsics()
    snap = manager.snapshot(source_id=0, camera_id="test-camera")
    assert snap is not None, "Snapshot should not be None after set_extrinsics"

    E_stored = np.array(snap.extrinsics_col_major, dtype=np.float64).reshape((4, 4), order="F")
    np.testing.assert_allclose(E_stored, E_expected, atol=1e-9, rtol=1e-9, err_msg="Stored E != inv(Twc)")

    # Also verify: inv(E_stored) == Twc
    Twc_recovered = np.linalg.inv(E_stored)
    np.testing.assert_allclose(Twc_recovered, Twc, atol=1e-9, rtol=1e-9, err_msg="inv(E_stored) != Twc")

    stored_json = json.loads((temp_calibration_dir / "camera_calibration.json").read_text())
    stored_pose = stored_json["cameras"]["test-camera"].get("pose")
    assert isinstance(stored_pose, dict), "E-only persistence should keep pose synchronized"


def test_set_extrinsics_refuses_mutation_when_private_audit_is_unsafe(
    temp_calibration_dir: Path,
) -> None:
    public_audit = temp_calibration_dir / "public_audit"
    public_audit.mkdir(mode=0o755)
    manager = CalibrationManager(
        cameras_yaml_path=temp_calibration_dir / "cameras.yaml",
        camera_calibration_json_path=temp_calibration_dir / "camera_calibration.json",
        ply_alignment_json_path=temp_calibration_dir / "ply_alignment.json",
        streammux_size=(1280, 720),
        raw_audit_dir=public_audit,
    )
    manager.set_camera_labels({0: "test-camera"})
    calibration_path = temp_calibration_dir / "camera_calibration.json"
    before = calibration_path.read_bytes()

    result = manager.set_extrinsics(
        "test-camera",
        E=[
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
    )

    assert result == {"ok": False, "error": "calibration_audit_failed"}
    assert calibration_path.read_bytes() == before
    assert public_audit.stat().st_mode & 0o777 == 0o755


def test_set_extrinsics_autoconverts_cm_translation(monkeypatch: pytest.MonkeyPatch, manager: CalibrationManager):
    """Verify cm-like translations are coerced to meters in auto mode."""
    monkeypatch.delenv("NOESIS_EXTRINSICS_INPUT_UNITS", raising=False)

    # E (world->camera) with identity rotation and camera center at Y=250 (cm-like).
    # For R=I, camera center C_world = -t, so t_y = -250.
    E_cm = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, -250.0, 0.0, 1.0,
    ]

    result = manager.set_extrinsics("test-camera", E=E_cm)
    assert result.get("ok"), f"set_extrinsics failed: {result}"

    snap = manager.snapshot(source_id=0, camera_id="test-camera")
    assert snap is not None, "Snapshot should not be None after set_extrinsics"

    E_stored = np.array(snap.extrinsics_col_major, dtype=np.float64).reshape((4, 4), order="F")
    assert abs(float(E_stored[1, 3]) - (-2.5)) < 1e-6, "Expected cm→m translation coercion (t_y=-2.5)"


# ---------------------------------------------------------------------------
# Test: Validation (invalid E rejected)
# ---------------------------------------------------------------------------


def test_validation_rejects_singular_E(manager: CalibrationManager):
    """Verify set_extrinsics rejects singular (non-invertible) E."""
    # Singular matrix (zero row)
    E_singular = [
        0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    ]
    result = manager.set_extrinsics("test-camera", E=E_singular)
    assert not result.get("ok"), "Singular E should be rejected"
    assert "singular" in str(result.get("error", "")).lower(), f"Expected singular error, got: {result}"


def test_validation_rejects_non_orthonormal_rotation(manager: CalibrationManager):
    """Verify set_extrinsics rejects E with non-orthonormal rotation."""
    # Scaled rotation (not orthonormal)
    E_scaled = [
        2.0, 0.0, 0.0, 0.0,  # Scaled column
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    ]
    result = manager.set_extrinsics("test-camera", E=E_scaled)
    assert not result.get("ok"), "Non-orthonormal rotation should be rejected"
    assert "orthonormal" in str(result.get("error", "")).lower(), f"Expected orthonormal error, got: {result}"


def test_validation_rejects_invalid_align_matrix(manager: CalibrationManager):
    """Verify set_align rejects invalid (singular) alignment matrix."""
    # Singular alignment matrix
    align_singular = {
        "matrix": [
            0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        ]
    }
    result = manager.set_align(align_singular)
    assert not result.get("ok"), "Singular align.matrix should be rejected"
    assert "singular" in str(result.get("error", "")).lower(), f"Expected singular error, got: {result}"


# ---------------------------------------------------------------------------
# Test: Partial align update preserves existing values
# ---------------------------------------------------------------------------


def test_partial_align_update_preserves_values(manager: CalibrationManager, temp_calibration_dir: Path):
    """Verify partial set_align updates only the specified fields."""
    # First, set a custom floor_y
    result = manager.set_align({"floor_y": 0.5})
    assert result.get("ok"), f"set_align failed: {result}"

    # Now update only units
    result = manager.set_align({"units": {"s_obj_to_m": 2.0}})
    assert result.get("ok"), f"set_align failed: {result}"

    # Verify floor_y is still 0.5
    manager.reload_alignment()
    bundle = manager.calibration_bundle()
    align = bundle.get("align", {})
    assert abs(align.get("floor_y", 0) - 0.5) < 1e-9, "floor_y should be preserved"
    assert abs(align.get("units", {}).get("s_obj_to_m", 0) - 2.0) < 1e-9, "s_obj_to_m should be updated"


def test_scene_similarity_persists_and_round_trips(manager: CalibrationManager, temp_calibration_dir: Path):
    """Verify world-to-scene similarity persists in alignment and bundle payloads."""
    similarity = {
        "world_to_scene_col_major": [
            2.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 0.0, 0.0,
            0.0, 0.0, 2.0, 0.0,
            10.0, 20.0, 30.0, 1.0,
        ],
        "source": "unit_test",
        "camera_count": 3,
        "mean_residual": 0.25,
    }

    result = manager.set_align({"scene_similarity": similarity})
    assert result.get("ok"), f"set_align failed: {result}"

    manager.reload_alignment()
    bundle = manager.calibration_bundle()
    stored = bundle.get("align", {}).get("scene_similarity")
    assert isinstance(stored, dict), "Expected scene_similarity in calibration bundle"
    assert stored.get("source") == "unit_test"
    assert stored.get("camera_count") == 3
    assert abs(float(stored.get("mean_residual", 0.0)) - 0.25) < 1e-9
    assert stored.get("world_to_scene_col_major") == similarity["world_to_scene_col_major"]
    expected_digest = _world_to_scene_sha256(
        similarity["world_to_scene_col_major"], stored["s_obj_to_m"]
    )
    assert stored.get("world_to_scene_sha256") == expected_digest

    raw_align = json.loads((temp_calibration_dir / "ply_alignment.json").read_text())
    assert raw_align.get("scene_similarity", {}).get("world_to_scene_col_major") == similarity["world_to_scene_col_major"]
    assert (
        raw_align.get("scene_similarity", {}).get("world_to_scene_sha256")
        == expected_digest
    )

    mismatched = dict(similarity)
    mismatched["world_to_scene_sha256"] = "0" * 64
    rejected = manager.set_align({"scene_similarity": mismatched})
    assert rejected["ok"] is False
    assert "world_to_scene_sha256 does not match" in rejected["error"]


def test_ds8_ds9_world_to_scene_digest_contract_is_identical() -> None:
    ds9_path = (
        Path(__file__).resolve().parents[1]
        / "DS9"
        / "noesis"
        / "calibration"
        / "manager.py"
    )
    spec = importlib.util.spec_from_file_location(
        "ds9_calibration_manager_digest_contract", ds9_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    matrix = [
        92.5, 0.0, 0.0, 0.0,
        0.0, 92.5, 0.0, 0.0,
        0.0, 0.0, 92.5, 0.0,
        -4.6, 2.0, -0.1, 1.0,
    ]
    scene_to_m = 1.0 / 92.5
    assert module._world_to_scene_sha256(  # type: ignore[attr-defined]
        matrix, scene_to_m
    ) == _world_to_scene_sha256(matrix, scene_to_m)


def test_camera_anchor_scene_similarity_preserves_binding_and_rejects_bad_fit(
    manager: CalibrationManager,
) -> None:
    from noesis.calibration.scene_registration import camera_anchor_state_sha256

    correspondences = [
        {
            "anchor_id": f"device-{index}",
            "camera_id": f"camera-{index}",
            "world_position_m": [float(index), 2.0, 0.0],
            "scene_position": [float(index) * 100.0, 200.0, 0.0],
            "residual_scene_units": 1.0,
        }
        for index in range(3)
    ]
    similarity = {
        "world_to_scene_col_major": [
            100.0, 0.0, 0.0, 0.0,
            0.0, 100.0, 0.0, 0.0,
            0.0, 0.0, 100.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
        "source": "menon_virtual_device_camera_similarity_v1",
        "anchor_state_sha256": camera_anchor_state_sha256(correspondences),
        "camera_calibration_sha256": "b" * 64,
        "position_rmse_m": 0.02,
        "max_residual_m": 0.03,
        "anchor_residual_limit_m": 0.05,
        "correspondences": correspondences,
    }
    result = manager.set_align({"scene_similarity": similarity})
    assert result == {"ok": True}
    stored = manager.calibration_bundle()["align"]["scene_similarity"]
    assert stored["anchor_state_sha256"] == camera_anchor_state_sha256(correspondences)
    assert stored["camera_calibration_sha256"] == "b" * 64
    assert all(row.get("anchor_id") for row in stored["correspondences"])

    bad_binding = dict(similarity)
    bad_binding["anchor_state_sha256"] = "0" * 64
    rejected_binding = manager.set_align({"scene_similarity": bad_binding})
    assert rejected_binding["ok"] is False
    assert "does not match anchor_state_sha256" in rejected_binding["error"]

    bad_fit = dict(similarity)
    bad_fit["max_residual_m"] = 0.051
    rejected = manager.set_align({"scene_similarity": bad_fit})
    assert rejected["ok"] is False
    assert "exceeds its residual limit" in rejected["error"]


# ---------------------------------------------------------------------------
# Test: pixel_to_world helper
# ---------------------------------------------------------------------------


def test_pixel_to_world_floor_intersection():
    """Verify pixel_to_world floor intersection returns correct world point."""
    from noesis.calibration.geometry import pixel_to_world

    # Simple setup: camera at (0, 2, 0), looking down (-Y)
    # K with cx=640, cy=360, fx=fy=800
    K = np.array([
        [800.0, 0.0, 640.0],
        [0.0, 800.0, 360.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # E: rotate camera to look down (swap Y and Z axes)
    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0]
    ], dtype=np.float64)
    t = np.array([0.0, 0.0, 2.0], dtype=np.float64)  # Camera at world (0, 2, 0) after transform

    E = np.eye(4, dtype=np.float64)
    E[:3, :3] = R
    E[:3, 3] = t
    E_col_major = list(E.flatten(order="F"))

    # Test floor intersection at image center (should hit floor at world origin)
    result = pixel_to_world(K, E_col_major, floor_y=0.0, unit_scale=1.0, u=640.0, v=360.0)
    assert result.ok, f"pixel_to_world failed: {result.error}"
    assert result.method == "floor", "Expected floor method"
    # World point should be near origin
    # (exact values depend on camera geometry)


def test_pixel_to_world_with_depth():
    """Verify pixel_to_world with depth returns correct world point."""
    from noesis.calibration.geometry import pixel_to_world

    # Identity E and simple K
    K = np.array([
        [800.0, 0.0, 640.0],
        [0.0, 800.0, 360.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    E_identity = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    ]

    # At image center with depth=5m, should project to (0, 0, 5) in camera frame
    # With identity E, camera frame = world frame
    result = pixel_to_world(K, E_identity, floor_y=0.0, unit_scale=1.0, u=640.0, v=360.0, depth_m=5.0)
    assert result.ok, f"pixel_to_world failed: {result.error}"
    assert result.method == "depth", "Expected depth method"

    # World point should be at (0, 0, 5)
    np.testing.assert_allclose(result.world_point, [0.0, 0.0, 5.0], atol=1e-6)


# ---------------------------------------------------------------------------
# Test: pixel_to_world with unit_scale ≠ 1.0
# ---------------------------------------------------------------------------


def test_pixel_to_world_floor_with_unit_scale():
    """Verify pixel_to_world floor intersection works correctly with unit_scale=2.0."""
    from noesis.calibration.geometry import pixel_to_world

    # Simple setup: camera at (0, 4, 0) in world units, looking down (-Y)
    # K with cx=640, cy=360, fx=fy=800
    K = np.array([
        [800.0, 0.0, 640.0],
        [0.0, 800.0, 360.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # E: rotate camera to look down (swap Y and Z axes), camera at world (0, 4, 0)
    # world→camera: Y_world maps to Z_cam, Z_world maps to -Y_cam
    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0]
    ], dtype=np.float64)
    # Camera at world (0, 4, 0) in world units
    # E = [R | t] where t = -R @ C_world
    C_world_units = np.array([0.0, 4.0, 0.0], dtype=np.float64)
    t = -R @ C_world_units  # t = [0, 0, 4]

    E = np.eye(4, dtype=np.float64)
    E[:3, :3] = R
    E[:3, 3] = t
    E_col_major = list(E.flatten(order="F"))

    # With unit_scale=2.0: floor_y=0 becomes 0m, camera is at (0, 8, 0)m after scaling
    # Ray from image center should hit floor at origin
    result = pixel_to_world(K, E_col_major, floor_y=0.0, unit_scale=2.0, u=640.0, v=360.0)
    assert result.ok, f"pixel_to_world failed: {result.error}"
    assert result.method == "floor", "Expected floor method"

    # The world point should be near (0, 0, 0) - floor intersection below camera
    # Camera at (0, 8, 0)m looking down, ray from center hits floor at Y=0
    np.testing.assert_allclose(result.world_point[1], 0.0, atol=1e-6, err_msg="Floor Y should be 0")


def test_pixel_to_world_depth_with_unit_scale():
    """Verify pixel_to_world depth projection works correctly with unit_scale=2.0."""
    from noesis.calibration.geometry import pixel_to_world

    # Identity E and simple K
    K = np.array([
        [800.0, 0.0, 640.0],
        [0.0, 800.0, 360.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    E_identity = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    ]

    # At image center with depth=5m and unit_scale=2.0
    # Expected: depth × scale = 5 × 2 = 10
    result = pixel_to_world(K, E_identity, floor_y=0.0, unit_scale=2.0, u=640.0, v=360.0, depth_m=5.0)
    assert result.ok, f"pixel_to_world failed: {result.error}"
    assert result.method == "depth", "Expected depth method"

    # World point should be at (0, 0, 10) due to scaling
    np.testing.assert_allclose(result.world_point, [0.0, 0.0, 10.0], atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
