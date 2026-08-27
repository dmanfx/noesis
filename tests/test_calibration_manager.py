#!/usr/bin/env python3
"""Unit tests for the canonical CalibrationManager.

Tests:
1. K-divergence: `calibration-bundle.cameras.K[cam]` matches `snapshot(cam).intrinsics`
2. Twc-inversion: `set_extrinsics(Twc=...)` stores correct `E` such that `inv(E) == Twc`
3. Validation: invalid E/K/align matrices are rejected

Run with: pytest tests/test_calibration_manager.py -v
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from noesis.calibration.manager import (
    CalibrationManager,
    CalibrationValidationError,
    _world_to_scene_sha256,
    create_calibration_manager,
    load_camera_labels,
)
from noesis_core.contracts.scene_prior import (
    ScenePriorFloorPlane,
    ScenePriorFrameBinding,
    ScenePriorFrameRef,
)
from noesis_core.coordinate_frames import (
    BACKEND_WORLD_FRAME_ID,
    RevisionedFrame,
    revisioned_frame_sha256,
    revisioned_transform_sha256,
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
        root / "DS9" / "noesis" / "ds9_runtime_core.py",
        root / "scripts" / "build_virtual_twin_reconstruction.py",
        root / "scripts" / "build_rectified_virtual_twin_reconstruction.py",
        root / "scripts" / "build_stream_room_reconstruction.py",
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


def test_snapshot_and_world_snapshot_reuse_one_immutable_generation(
    manager: CalibrationManager,
) -> None:
    """Hot tracking/BEV callers receive the exact same immutable objects."""

    raw_first = manager.snapshot(source_id=0, camera_id="test-camera")
    raw_second = manager.snapshot(source_id=0, camera_id="test-camera")
    world_first = manager.world_snapshot(source_id=0, camera_id="test-camera")
    world_second = manager.world_snapshot(source_id=0, camera_id="test-camera")

    assert raw_first is not None and raw_second is raw_first
    assert world_first is not None and world_second is world_first
    assert world_first.intrinsics is raw_first.intrinsics
    assert raw_first.intrinsics.flags.writeable is False
    with pytest.raises(ValueError):
        raw_first.intrinsics[0, 0] = 123.0
    assert isinstance(raw_first.extrinsics_col_major, tuple)


def test_snapshot_cache_is_invalidated_by_every_public_calibration_reload_or_setter(
    manager: CalibrationManager,
    temp_calibration_dir: Path,
) -> None:
    """Each calibration generation gets new raw and active snapshot objects."""

    def assert_replaced() -> None:
        old_raw = manager.snapshot(source_id=0, camera_id="test-camera")
        old_world = manager.world_snapshot(source_id=0, camera_id="test-camera")
        assert old_raw is not None and old_world is not None
        manager._load_frame_contracts()
        new_raw = manager.snapshot(source_id=0, camera_id="test-camera")
        new_world = manager.world_snapshot(source_id=0, camera_id="test-camera")
        assert new_raw is not None and new_world is not None
        assert new_raw is not old_raw
        assert new_world is not old_world

    # Camera-label changes are a public setter and must not leave a snapshot
    # from a previous source/camera mapping in the live cache.
    manager.set_camera_labels({0: "test-camera"})
    assert_replaced()

    extrinsics = json.loads(
        (temp_calibration_dir / "camera_calibration.json").read_text(
            encoding="utf-8"
        )
    )
    extrinsics["cameras"]["test-camera"]["E"][12] = 0.25
    (temp_calibration_dir / "camera_calibration.json").write_text(
        json.dumps(extrinsics), encoding="utf-8"
    )
    manager.reload_extrinsics()
    assert_replaced()

    alignment = json.loads(
        (temp_calibration_dir / "ply_alignment.json").read_text(encoding="utf-8")
    )
    alignment["floor_y"] = 0.25
    (temp_calibration_dir / "ply_alignment.json").write_text(
        json.dumps(alignment), encoding="utf-8"
    )
    manager.reload_alignment()
    assert_replaced()

    cameras_path = temp_calibration_dir / "cameras.yaml"
    cameras_path.write_text(
        cameras_path.read_text(encoding="utf-8").replace("fx: 800.0", "fx: 900.0"),
        encoding="utf-8",
    )
    manager.reload_intrinsics()
    assert_replaced()

    manager.reload_all()
    assert_replaced()

    updated_E = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.5, 0.0, 0.0, 1.0,
    ]
    assert manager.set_extrinsics("test-camera", E=updated_E)["ok"] is True
    assert_replaced()

    assert manager.set_align({"floor_y": 0.5})["ok"] is True
    assert_replaced()


def test_snapshot_failure_is_not_cached_across_extrinsics_reload(
    manager: CalibrationManager,
    temp_calibration_dir: Path,
) -> None:
    """A transient missing camera can recover on the next reload."""

    assert manager.snapshot(source_id=0, camera_id="test-camera") is not None
    (temp_calibration_dir / "camera_calibration.json").write_text(
        json.dumps({"cameras": {}}), encoding="utf-8"
    )
    manager.reload_extrinsics()
    assert manager.snapshot(source_id=0, camera_id="test-camera") is None

    identity = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    (temp_calibration_dir / "camera_calibration.json").write_text(
        json.dumps({"cameras": {"test-camera": {"E": identity}}}),
        encoding="utf-8",
    )
    manager.reload_extrinsics()
    recovered = manager.snapshot(source_id=0, camera_id="test-camera")
    assert recovered is not None


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


def _family_active_frame_binding(repo_root: Path) -> ScenePriorFrameBinding:
    """Exact active Family PCF edge captured from its revision metadata."""

    calibration_sha256 = hashlib.sha256(
        (repo_root / "config" / "camera_calibration.json").read_bytes()
    ).hexdigest()
    alignment_sha256 = hashlib.sha256(
        (repo_root / "config" / "ply_alignment.json").read_bytes()
    ).hexdigest()
    assert calibration_sha256 == "9b34e3ea8a3808475584a55383afcff96373a9e0845d33dc026a6659c975be93"
    assert alignment_sha256 == "88e05c3259bde42265c8bdc97e1c86762ca7540402041ebcceadb407d72df82b"
    source = RevisionedFrame(
        BACKEND_WORLD_FRAME_ID,
        revisioned_frame_sha256(
            BACKEND_WORLD_FRAME_ID,
            artifact_sha256s=(calibration_sha256, alignment_sha256),
        ),
    )
    assert source.revision == "b73b6a742e2d0d5936876875d904aa74c98996088f06efd04d552dac3f96a214"
    target = RevisionedFrame(
        BACKEND_WORLD_FRAME_ID,
        "sceneprior_family-room_20260811T015847Z_8b80dc69a7c4",
    )
    world_correction = (
        0.9467470066425219, -0.09059934064225462, -0.30896903548513316, 0.0,
        0.0649604489274947, 0.9936095819714693, -0.09230459733740415, 0.0,
        0.31535732984752884, 0.06731833397872138, 0.9465822713434718, 0.0,
        -3.1508635068958624, 0.17446172385443262, 5.462715819965061, 1.0,
    )
    transform_sha256 = revisioned_transform_sha256(
        source,
        target,
        world_correction,
    )
    assert transform_sha256 == "5bec221621ad20d64d5fc8477945b3d33ca90b75264184ea59194575167de217"
    return ScenePriorFrameBinding(
        contract="noesis.scene_prior.frame_binding",
        contract_version=1,
        source_frame=ScenePriorFrameRef(
            frame_id=source.frame_id,
            revision=source.revision,
        ),
        target_frame=ScenePriorFrameRef(
            frame_id=target.frame_id,
            revision=target.revision,
        ),
        source_camera_calibration_sha256=calibration_sha256,
        source_world_alignment_sha256=alignment_sha256,
        target_revision_id="vt_family_room_stream_rgbmesh_20260623T215850_645134993",
        target_revision_metadata_sha256="67dee858ad3c5b0564e7d364b9522e59842f14b7c7de1c07659216e1defedd9c",
        target_from_source_col_major=world_correction,
        target_from_source_sha256=transform_sha256,
        source_floor_plane=ScenePriorFloorPlane(
            frame=ScenePriorFrameRef(
                frame_id=source.frame_id,
                revision=source.revision,
            ),
            normal=(-0.09059934064225462, 0.9936095819714694, 0.06731833397872138),
            offset_m=0.17558377759227575,
        ),
        target_floor_plane=ScenePriorFloorPlane(
            frame=ScenePriorFrameRef(
                frame_id=target.frame_id,
                revision=target.revision,
            ),
            normal=(0.0, 1.0, 0.0),
            offset_m=0.0,
        ),
    )


def test_family_active_world_view_intersects_revision_floor_at_reported_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from noesis.calibration.geometry import pixel_to_world

    repo_root = Path(__file__).resolve().parents[1]
    binding = _family_active_frame_binding(repo_root)
    manager = CalibrationManager(
        cameras_yaml_path=repo_root / "config" / "cameras.yaml",
        camera_calibration_json_path=repo_root / "config" / "camera_calibration.json",
        ply_alignment_json_path=repo_root / "config" / "ply_alignment.json",
        streammux_size=(1920, 1080),
        scene_prior_frame_bindings={"family-room": binding},
    )
    manager.set_camera_labels({0: "living-room", 1: "kitchen", 2: "family-room"})

    raw = manager.snapshot(2, "family-room")
    active = manager.world_snapshot(2, "family-room")
    assert raw is not None and active is not None
    assert raw.world_frame is not None
    assert raw.world_frame.frame_id == BACKEND_WORLD_FRAME_ID
    assert raw.world_frame.revision == binding.target_frame.revision
    assert tuple(raw.extrinsics_col_major) == active.calibration_extrinsics_col_major
    assert active.world_frame_id == BACKEND_WORLD_FRAME_ID
    assert active.world_frame_revision == binding.target_frame.revision
    assert active.floor_y == pytest.approx(0.0)
    assert tuple(active.extrinsics_col_major) != tuple(raw.extrinsics_col_major)

    raw_matrix = np.asarray(raw.extrinsics_col_major).reshape((4, 4), order="F")
    active_matrix = np.asarray(active.extrinsics_col_major).reshape((4, 4), order="F")
    world_correction = raw.frame_contract.matrix  # type: ignore[union-attr]
    np.testing.assert_allclose(
        active_matrix,
        raw_matrix @ np.linalg.inv(world_correction),
        atol=1e-12,
    )
    raw_camera = np.linalg.inv(raw_matrix)[:3, 3]

    for v, expected_range_m, minimum_displacement_m in (
        (653.0, 5.902, 2.0),
        (550.0, 9.395, 4.0),
    ):
        raw_hit = pixel_to_world(
            raw.intrinsics,
            raw.extrinsics_col_major,
            raw.floor_y,
            raw.unit_scale,
            1100.0,
            v,
        )
        active_hit = pixel_to_world(
            active.intrinsics,
            active.extrinsics_col_major,
            active.floor_y,
            active.unit_scale,
            1100.0,
            v,
        )
        assert raw_hit.ok and raw_hit.world_point is not None
        assert active_hit.ok and active_hit.world_point is not None
        raw_point = np.asarray(raw_hit.world_point, dtype=np.float64)
        active_point = np.asarray(active_hit.world_point, dtype=np.float64)
        assert np.linalg.norm(raw_point - raw_camera) == pytest.approx(
            expected_range_m,
            abs=0.02,
        )
        raw_point_in_active = (world_correction @ np.r_[raw_point, 1.0])[:3]
        assert np.linalg.norm(raw_point_in_active - active_point) > minimum_displacement_m
        assert active_point[1] == pytest.approx(0.0, abs=1e-9)
        active_point_in_source = (
            np.linalg.inv(world_correction) @ np.r_[active_point, 1.0]
        )[:3]
        source_plane = binding.source_floor_plane
        assert (
            np.dot(np.asarray(source_plane.normal), active_point_in_source)
            + source_plane.offset_m
        ) == pytest.approx(0.0, abs=0.002)

    living_raw = manager.snapshot(0, "living-room")
    living_active = manager.world_snapshot(0, "living-room")
    assert living_raw is not None and living_active is not None
    assert living_active.world_frame_id == BACKEND_WORLD_FRAME_ID
    assert tuple(living_active.extrinsics_col_major) == tuple(
        living_raw.extrinsics_col_major
    )

    def _unexpected_write(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("revision-bound calibration mutation reached persistence")

    monkeypatch.setattr(manager, "_log_raw_extrinsics_payload", _unexpected_write)
    monkeypatch.setattr(manager, "_save_extrinsics", _unexpected_write)
    monkeypatch.setattr(manager, "_save_alignment", _unexpected_write)
    expected_rejection = {
        "ok": False,
        "error": "revision_bound_frame_contract_requires_atomic_regeneration",
    }
    assert manager.set_extrinsics(
        "family-room",
        E=list(raw.extrinsics_col_major),
    ) == expected_rejection
    assert manager.set_align({"floor_y": 0.0}) == expected_rejection


def test_calibration_manager_rejects_stale_frame_binding() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    stale = _family_active_frame_binding(repo_root).model_copy(
        update={"source_camera_calibration_sha256": "0" * 64}
    )
    with pytest.raises(CalibrationValidationError, match="calibration revision mismatch"):
        CalibrationManager(
            cameras_yaml_path=repo_root / "config" / "cameras.yaml",
            camera_calibration_json_path=repo_root / "config" / "camera_calibration.json",
            ply_alignment_json_path=repo_root / "config" / "ply_alignment.json",
            streammux_size=(1920, 1080),
            scene_prior_frame_bindings={"family-room": stale},
        )


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
