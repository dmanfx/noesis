from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from noesis.calibration.depth_registration import (
    DepthRegistrationError,
    DepthRegistrationManager,
    calibration_fingerprint_from_snapshot,
    model_profile_fingerprint,
    write_depth_registration_bundle,
)
from noesis.calibration.depth_registration_builder import build_registration_entry
from noesis.calibration.manager import CalibrationSnapshot
from noesis.ds9_runtime_core import _build_depth_registration_profile_fingerprints
from noesis.ds9_runtime_core import _load_depth_registration_manager
from noesis.ds9_runtime_core import _resolve_depth_registration_path


ROOT = Path(__file__).resolve().parents[1]


def _legacy_snapshot_fingerprint(snapshot: CalibrationSnapshot) -> dict[str, object]:
    payload = {
        "camera_id": str(snapshot.camera_id),
        "intrinsics": [[float(snapshot.intrinsics[r, c]) for c in range(3)] for r in range(3)],
        "extrinsics_col_major": [float(v) for v in list(snapshot.extrinsics_col_major)],
        "floor_y": float(snapshot.floor_y),
        "image_size": [int(snapshot.image_size[0]), int(snapshot.image_size[1])],
        "unit_scale": float(snapshot.unit_scale),
    }
    payload["fingerprint_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return payload


def _snapshot(camera_id: str = "cam0") -> CalibrationSnapshot:
    intrinsics = np.array(
        [
            [800.0, 0.0, 640.0],
            [0.0, 800.0, 360.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    extrinsics = np.eye(4, dtype=np.float64)
    return CalibrationSnapshot(
        camera_id=camera_id,
        intrinsics=intrinsics,
        extrinsics_col_major=list(extrinsics.flatten(order="F")),
        floor_y=0.0,
        image_size=(1280, 720),
        unit_scale=1.0,
    )


def _entry(camera_id: str = "cam0") -> tuple[CalibrationSnapshot, dict[str, object], dict[str, object], object]:
    snapshot = _snapshot(camera_id)
    dav2_profile = model_profile_fingerprint({"name": "depth_tracking_fullframe", "engine": "models/engines/da2.engine"}, repo_root=Path.cwd())
    mapanything_profile = model_profile_fingerprint({"name": "mapanything_fullframe", "engine": "models/mapanything_depth/1/model.plan"}, repo_root=Path.cwd())
    raw_depth = np.linspace(2.0, 10.0, 64, dtype=np.float64)
    registered_depth = (raw_depth * 0.92) + 0.12
    entry = build_registration_entry(
        camera_id=camera_id,
        calibration_fingerprint=calibration_fingerprint_from_snapshot(snapshot),
        dav2_profile=dav2_profile,
        mapanything_profile=mapanything_profile,
        provenance={"source_uri": "file:///tmp/example.mp4"},
        raw_depth_m=raw_depth,
        registered_depth_m=registered_depth,
        min_samples=4,
        num_knots=4,
    )
    return snapshot, dav2_profile, mapanything_profile, entry


def test_model_profile_fingerprint_keeps_absolute_repo_paths_relative_through_symlink(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    engine_storage = tmp_path / "engine_storage"
    repo_root.mkdir()
    engine_storage.mkdir()
    (repo_root / "models").symlink_to(engine_storage, target_is_directory=True)
    engine_path = repo_root / "models" / "engines" / "da2.engine"

    absolute_profile = model_profile_fingerprint(
        {"name": "depth_tracking_fullframe", "engine": str(engine_path)},
        repo_root=repo_root,
    )
    relative_profile = model_profile_fingerprint(
        {"name": "depth_tracking_fullframe", "engine": "models/engines/da2.engine"},
        repo_root=repo_root,
    )

    assert absolute_profile["engine"] == "models/engines/da2.engine"
    assert absolute_profile["fingerprint_sha256"] == relative_profile["fingerprint_sha256"]


def test_ds9_registration_profile_relocation_is_limited_to_reviewed_roots(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_engine = artifact_root / "models" / "engines" / "depth.engine"
    untrusted_engine = tmp_path / "untrusted" / "models" / "engines" / "depth.engine"
    wrong_subdir_engine = artifact_root / "other" / "engines" / "depth.engine"
    wrong_name_engine = artifact_root / "models" / "engines" / "different.engine"
    repo_engine = ROOT / "DS9" / "models" / "engines" / "depth.engine"
    code = """
import json
import os
import sys

from noesis import ds9_runtime_core as runtime
from noesis.calibration.depth_registration import _dav2_profile_matches

artifact_engine, repo_engine, untrusted_engine, wrong_subdir, wrong_name = sys.argv[1:]
logical = 'DS9/models/engines/depth.engine'
base_cfg = {
    'enable': True,
    'name': 'depth_tracking_fullframe',
    'config-file-path': 'DS9/build/depth.ini',
    'engine': artifact_engine,
    'batch_size': 3,
    'gie_id': 5,
    'attach_tensor_meta': True,
}
allowed, _ = runtime._build_depth_registration_profile_fingerprints(
    {'models': {'depth_tracking': base_cfg, 'mapanything': {}}},
    pipeline_path=runtime.DS9_ROOT / 'config' / 'infer.yaml',
)
stored = dict(allowed)
stored['fingerprint_sha256'] = 'stored-profile'
bad_cfg = dict(base_cfg)
bad_cfg['engine'] = untrusted_engine
untrusted, _ = runtime._build_depth_registration_profile_fingerprints(
    {'models': {'depth_tracking': bad_cfg, 'mapanything': {}}},
    pipeline_path=runtime.DS9_ROOT / 'config' / 'infer.yaml',
)
wrong_name_cfg = dict(base_cfg)
wrong_name_cfg['engine'] = wrong_name
wrong_name_profile, _ = runtime._build_depth_registration_profile_fingerprints(
    {'models': {'depth_tracking': wrong_name_cfg, 'mapanything': {}}},
    pipeline_path=runtime.DS9_ROOT / 'config' / 'infer.yaml',
)
mapped = {
    'logical': runtime._logical_ds9_model_reference(logical),
    'repo': runtime._logical_ds9_model_reference(repo_engine),
    'artifact': runtime._logical_ds9_model_reference(artifact_engine),
    'untrusted': runtime._logical_ds9_model_reference(untrusted_engine),
    'wrong_subdir': runtime._logical_ds9_model_reference(wrong_subdir),
    'relative_other': runtime._logical_ds9_model_reference('models/engines/depth.engine'),
}
os.environ.pop('NOESIS_DS9_ARTIFACT_ROOT', None)
mapped['artifact_without_authority'] = runtime._logical_ds9_model_reference(artifact_engine)
print(json.dumps({
    'mapped': mapped,
    'allowed_engine': allowed['engine'],
    'untrusted_engine': untrusted['engine'],
    'wrong_name_engine': wrong_name_profile['engine'],
    'allowed_match': _dav2_profile_matches(stored, allowed),
    'untrusted_match': _dav2_profile_matches(stored, untrusted),
    'wrong_name_match': _dav2_profile_matches(stored, wrong_name_profile),
}))
"""
    env = dict(os.environ)
    env.update(
        {
            "NOESIS_DS9_ARTIFACT_ROOT": str(artifact_root),
            "PYTHONPATH": os.pathsep.join(
                (str(ROOT / "DS9"), str(ROOT), env.get("PYTHONPATH", ""))
            ),
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(artifact_engine),
            str(repo_engine),
            str(untrusted_engine),
            str(wrong_subdir_engine),
            str(wrong_name_engine),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    logical = "DS9/models/engines/depth.engine"
    assert payload["mapped"]["logical"] == logical
    assert payload["mapped"]["repo"] == logical
    assert payload["mapped"]["artifact"] == logical
    assert payload["mapped"]["untrusted"] == str(untrusted_engine)
    assert payload["mapped"]["wrong_subdir"] == str(wrong_subdir_engine)
    assert payload["mapped"]["relative_other"] == "models/engines/depth.engine"
    assert payload["mapped"]["artifact_without_authority"] == str(artifact_engine)
    assert payload["allowed_engine"] == logical
    assert payload["untrusted_engine"] == "depth.engine"
    assert payload["wrong_name_engine"] == "DS9/models/engines/different.engine"
    assert payload["allowed_match"] is True
    assert payload["untrusted_match"] is False
    assert payload["wrong_name_match"] is False


def test_depth_registration_profiles_are_stable_for_generated_configs() -> None:
    repo_root = Path.cwd().resolve()
    pipeline_cfg = {
        "models": {
            "depth_tracking": {
                "name": "depth_tracking_fullframe",
                "engine": "models/engines/depth_anything_v2_metric_hypersim_vits_294x518_b3_fp16.engine",
                "config-file-path": "build/config_infer_depth_tracking_da2_vits_294x518_b3_i1.ini",
                "batch_size": 3,
                "gie_id": 5,
                "attach_tensor_meta": True,
            },
            "mapanything": {
                "name": "mapanything_fullframe",
                "engine": str(repo_root / "models/mapanything_depth/1/model.plan"),
                "config-file-path": "DS9/pipelines/config_infer_secondary_mapanything.ini",
                "batch_size": 3,
                "gie_id": 2,
                "attach_tensor_meta": True,
            },
        }
    }

    active_profiles = _build_depth_registration_profile_fingerprints(
        pipeline_cfg,
        pipeline_path=repo_root / "DS9/config/infer.yaml",
    )
    generated_profiles = _build_depth_registration_profile_fingerprints(
        pipeline_cfg,
        pipeline_path=repo_root / "diagnostics/bev_alignment/run/infer_file_sources.yaml",
    )

    assert generated_profiles == active_profiles
    assert generated_profiles[1]["engine"] == "models/mapanything_depth/1/model.plan"

    with_first_camera_secret = {
        **pipeline_cfg,
        "sources": [
            {
                "uri_secret": "living-room",
                "uri": "rtsp://camera.invalid/private-one",
            }
        ],
    }
    with_rotated_camera_secret = {
        **pipeline_cfg,
        "sources": [
            {
                "uri_secret": "living-room",
                "uri": "rtsp://camera.invalid/private-two",
            }
        ],
    }
    assert _build_depth_registration_profile_fingerprints(
        with_first_camera_secret,
        pipeline_path=repo_root / "DS9/config/infer.yaml",
    ) == _build_depth_registration_profile_fingerprints(
        with_rotated_camera_secret,
        pipeline_path=repo_root / "DS9/config/infer.yaml",
    )


def test_depth_registration_bundle_round_trip_and_apply(tmp_path: Path) -> None:
    snapshot, dav2_profile, mapanything_profile, entry = _entry()
    bundle_path = write_depth_registration_bundle(path=tmp_path / "depth_registration.json", entries={"cam0": entry})
    manager = DepthRegistrationManager.load(bundle_path)
    loaded = manager.validate_runtime(
        camera_id="cam0",
        snapshot=snapshot,
        dav2_profile=dav2_profile,
        mapanything_profile=mapanything_profile,
    )
    assert loaded.camera_id == "cam0"
    corrected, status, reg_id = manager.apply(camera_id="cam0", raw_depth_m=5.0)
    assert status == "ok"
    assert reg_id is not None
    assert corrected is not None
    assert corrected == pytest.approx(np.interp(5.0, entry.knots_raw_m, entry.knots_registered_m))


def test_depth_registration_accepts_dav2_cadence_only_profile_changes(tmp_path: Path) -> None:
    snapshot = _snapshot("cam0")
    repo_root = Path.cwd()
    dav2_profile_i1 = model_profile_fingerprint(
        {
            "name": "depth_tracking_fullframe",
            "engine": "models/engines/depth_anything_v2_metric_hypersim_vits_294x518_b3_fp16.engine",
            "config-file-path": "build/config_infer_depth_tracking_da2_vits_294x518_b3_i1.ini",
            "batch_size": 3,
            "gie_id": 5,
            "attach_tensor_meta": True,
        },
        repo_root=repo_root,
        extra={
            "model_name": "depth-anything-v2-metric-hypersim-vits",
            "input_size": [518, 294],
            "batch_size": 3,
            "interval": 1,
            "gie_id": 5,
        },
    )
    dav2_profile_i3 = model_profile_fingerprint(
        {
            "name": "depth_tracking_fullframe",
            "engine": "models/engines/depth_anything_v2_metric_hypersim_vits_294x518_b3_fp16.engine",
            "config-file-path": "build/config_infer_depth_tracking_da2_vits_294x518_b3_i3.ini",
            "batch_size": 3,
            "gie_id": 5,
            "attach_tensor_meta": True,
        },
        repo_root=repo_root,
        extra={
            "model_name": "depth-anything-v2-metric-hypersim-vits",
            "input_size": [518, 294],
            "batch_size": 3,
            "interval": 3,
            "gie_id": 5,
        },
    )
    assert dav2_profile_i1["fingerprint_sha256"] != dav2_profile_i3["fingerprint_sha256"]
    mapanything_profile = model_profile_fingerprint(
        {"name": "mapanything_fullframe", "engine": "models/mapanything_depth/1/model.plan"},
        repo_root=repo_root,
    )
    entry = build_registration_entry(
        camera_id="cam0",
        calibration_fingerprint=calibration_fingerprint_from_snapshot(snapshot),
        dav2_profile=dav2_profile_i1,
        mapanything_profile=mapanything_profile,
        provenance={"source_uri": "file:///tmp/example.mp4"},
        raw_depth_m=np.linspace(2.0, 10.0, 64, dtype=np.float64),
        registered_depth_m=(np.linspace(2.0, 10.0, 64, dtype=np.float64) * 0.92) + 0.12,
        min_samples=4,
        num_knots=4,
    )
    bundle_path = write_depth_registration_bundle(path=tmp_path / "depth_registration.json", entries={"cam0": entry})
    manager = DepthRegistrationManager.load(bundle_path)

    loaded = manager.validate_runtime(
        camera_id="cam0",
        snapshot=snapshot,
        dav2_profile=dav2_profile_i3,
        mapanything_profile=mapanything_profile,
    )

    assert loaded.camera_id == "cam0"


def test_depth_registration_accepts_dav2_yaml_to_nvinfer_profile_aliases(tmp_path: Path) -> None:
    snapshot = _snapshot("cam0")
    repo_root = Path.cwd()
    stored_dav2_profile = {
        "attach_tensor_meta": True,
        "batch_size": 3,
        "builder_runtime_device": "cuda",
        "config-file-path": "build/config_infer_depth_tracking_da2_vits_294x518_b3_i1.ini",
        "enable": True,
        "engine": "models/engines/depth_anything_v2_metric_hypersim_vits_294x518_b3_fp16.engine",
        "fingerprint_sha256": "stored-i1",
        "gie_id": 5,
        "input_size": [518, 294],
        "interval": 1,
        "model_name": "depth-anything-v2-metric-hypersim-vits",
        "name": "depth_tracking_fullframe",
    }
    runtime_dav2_profile = model_profile_fingerprint(
        {
            "config-file-path": "build/config_infer_depth_tracking_da2_vits_294x518_b3_i3.ini",
            "model-engine-file": "models/engines/depth_anything_v2_metric_hypersim_vits_294x518_b3_fp16.engine",
            "onnx-file": "models/onnx/depth_anything_v2_metric_hypersim_vits_294x518_b3.onnx",
            "model_name": "depth-anything-v2-metric-hypersim-vits",
        },
        repo_root=repo_root,
        extra={
            "model_name": "depth-anything-v2-metric-hypersim-vits",
            "input_size": [518, 294],
            "batch_size": 3,
            "interval": 3,
            "gie_id": 5,
        },
    )
    mapanything_profile = model_profile_fingerprint(
        {"name": "mapanything_fullframe", "engine": "models/mapanything_depth/1/model.plan"},
        repo_root=repo_root,
    )
    entry = build_registration_entry(
        camera_id="cam0",
        calibration_fingerprint=calibration_fingerprint_from_snapshot(snapshot),
        dav2_profile=stored_dav2_profile,
        mapanything_profile=mapanything_profile,
        provenance={"source_uri": "file:///tmp/example.mp4"},
        raw_depth_m=np.linspace(2.0, 10.0, 64, dtype=np.float64),
        registered_depth_m=(np.linspace(2.0, 10.0, 64, dtype=np.float64) * 0.92) + 0.12,
        min_samples=4,
        num_knots=4,
    )
    bundle_path = write_depth_registration_bundle(path=tmp_path / "depth_registration.json", entries={"cam0": entry})
    manager = DepthRegistrationManager.load(bundle_path)

    loaded = manager.validate_runtime(
        camera_id="cam0",
        snapshot=snapshot,
        dav2_profile=runtime_dav2_profile,
        mapanything_profile=mapanything_profile,
    )

    assert loaded.camera_id == "cam0"


def _mapanything_registration_fixture(
    tmp_path: Path,
) -> tuple[CalibrationSnapshot, dict[str, object], dict[str, object], object]:
    snapshot, dav2_profile, _mapanything_profile, _entry0 = _entry()
    stored_mapanything = model_profile_fingerprint(
        {
            "enable": True,
            "name": "mapanything_fullframe",
            "config-file-path": "DS9/pipelines/config_infer_secondary_mapanything.ini",
            "engine": "models/mapanything_depth/1/model.plan",
            "batch_size": 3,
            "gie_id": 2,
            "attach_tensor_meta": True,
        },
        repo_root=Path.cwd(),
        extra={"scope": "mapanything_reference_depth"},
    )
    entry = build_registration_entry(
        camera_id="cam0",
        calibration_fingerprint=calibration_fingerprint_from_snapshot(snapshot),
        dav2_profile=dav2_profile,
        mapanything_profile=stored_mapanything,
        provenance={"source_uri": "file:///tmp/example.mp4"},
        raw_depth_m=np.linspace(2.0, 10.0, 64, dtype=np.float64),
        registered_depth_m=(np.linspace(2.0, 10.0, 64, dtype=np.float64) * 0.92) + 0.12,
        min_samples=4,
        num_knots=4,
    )
    bundle_path = write_depth_registration_bundle(
        path=tmp_path / "depth_registration.json",
        entries={"cam0": entry},
    )
    return snapshot, dav2_profile, stored_mapanything, DepthRegistrationManager.load(bundle_path)


def test_depth_registration_accepts_derived_mapanything_runtime_config_path(
    tmp_path: Path,
) -> None:
    snapshot, dav2_profile, _stored, manager = _mapanything_registration_fixture(tmp_path)
    runtime_mapanything = model_profile_fingerprint(
        {
            "enable": True,
            "name": "mapanything_fullframe",
            "config-file-path": str(
                tmp_path
                / "build/runtime_inference/nvinfer/mapanything_fullframe/abcd/config.ini"
            ),
            "engine": "models/mapanything_depth/1/model.plan",
            "batch_size": 3,
            "gie_id": 2,
            "attach_tensor_meta": True,
            "interval": 4,
        },
        repo_root=Path.cwd(),
        extra={"scope": "mapanything_reference_depth"},
    )

    loaded = manager.validate_runtime(
        camera_id="cam0",
        snapshot=snapshot,
        dav2_profile=dav2_profile,
        mapanything_profile=runtime_mapanything,
    )

    assert loaded.camera_id == "cam0"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("engine", "models/mapanything_depth/1/different.plan"),
        ("name", "different_mapanything"),
        ("batch_size", 1),
        ("gie_id", 99),
    ],
)
def test_depth_registration_rejects_mapanything_semantic_identity_changes(
    tmp_path: Path,
    key: str,
    value: object,
) -> None:
    snapshot, dav2_profile, _stored, manager = _mapanything_registration_fixture(tmp_path)
    runtime_cfg: dict[str, object] = {
        "enable": True,
        "name": "mapanything_fullframe",
        "config-file-path": str(tmp_path / "runtime/config.ini"),
        "engine": "models/mapanything_depth/1/model.plan",
        "batch_size": 3,
        "gie_id": 2,
        "attach_tensor_meta": True,
    }
    runtime_cfg[key] = value
    runtime_mapanything = model_profile_fingerprint(
        runtime_cfg,
        repo_root=Path.cwd(),
        extra={"scope": "mapanything_reference_depth"},
    )

    with pytest.raises(
        DepthRegistrationError,
        match="mapanything_profile_fingerprint_mismatch",
    ):
        manager.validate_runtime(
            camera_id="cam0",
            snapshot=snapshot,
            dav2_profile=dav2_profile,
            mapanything_profile=runtime_mapanything,
        )


def test_depth_registration_rejects_fingerprint_mismatch(tmp_path: Path) -> None:
    _snapshot0, dav2_profile, mapanything_profile, entry = _entry()
    bundle_path = write_depth_registration_bundle(path=tmp_path / "depth_registration.json", entries={"cam0": entry})
    manager = DepthRegistrationManager.load(bundle_path)
    mismatched = _snapshot("cam0")
    mismatched = CalibrationSnapshot(
        camera_id=mismatched.camera_id,
        intrinsics=np.array([[810.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float64),
        extrinsics_col_major=mismatched.extrinsics_col_major,
        floor_y=mismatched.floor_y,
        image_size=mismatched.image_size,
        unit_scale=mismatched.unit_scale,
    )
    with pytest.raises(DepthRegistrationError, match="calibration_fingerprint_mismatch"):
        manager.validate_runtime(
            camera_id="cam0",
            snapshot=mismatched,
            dav2_profile=dav2_profile,
            mapanything_profile=mapanything_profile,
        )


def test_depth_registration_accepts_pose_only_changes_for_runtime_validation(tmp_path: Path) -> None:
    snapshot, dav2_profile, mapanything_profile, entry = _entry()
    legacy_entry = build_registration_entry(
        camera_id="cam0",
        calibration_fingerprint=_legacy_snapshot_fingerprint(snapshot),
        dav2_profile=dav2_profile,
        mapanything_profile=mapanything_profile,
        provenance={"source_uri": "file:///tmp/example.mp4"},
        raw_depth_m=np.linspace(2.0, 10.0, 64, dtype=np.float64),
        registered_depth_m=(np.linspace(2.0, 10.0, 64, dtype=np.float64) * 0.92) + 0.12,
        min_samples=4,
        num_knots=4,
    )
    bundle_path = write_depth_registration_bundle(path=tmp_path / "depth_registration.json", entries={"cam0": legacy_entry})
    manager = DepthRegistrationManager.load(bundle_path)

    pose_changed = CalibrationSnapshot(
        camera_id=snapshot.camera_id,
        intrinsics=np.array(snapshot.intrinsics, dtype=np.float64),
        extrinsics_col_major=[
            0.0, 1.0, 0.0, 0.0,
            -1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.5, 1.0, 1.5, 1.0,
        ],
        floor_y=2.0,
        image_size=snapshot.image_size,
        unit_scale=3.0,
    )

    loaded = manager.validate_runtime(
        camera_id="cam0",
        snapshot=pose_changed,
        dav2_profile=dav2_profile,
        mapanything_profile=mapanything_profile,
    )
    assert loaded.camera_id == "cam0"


def test_depth_registration_rejects_malformed_knots(tmp_path: Path) -> None:
    snapshot, dav2_profile, mapanything_profile, entry = _entry()
    payload = {
        "type": "depth_registration_bundle",
        "contract_version": 1,
        "scope": "people_tracking_depth_registration",
        "source_space": "dav2_anchor_range_raw_m",
        "target_space": "mapanything_room_range_m",
        "cameras": {
            "cam0": {
                **entry.to_dict(),
                "knots_registered_m": [5.0, 4.0, 6.0][: len(entry.knots_raw_m)],
            }
        },
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DepthRegistrationError, match="monotonically non-decreasing"):
        DepthRegistrationManager.load(path)
    # Prove the happy-path profiles still validate for completeness.
    fingerprint = calibration_fingerprint_from_snapshot(snapshot)
    assert fingerprint["fingerprint_sha256"]
    assert fingerprint["basis"] == "image_space_intrinsics_v1"
    assert dav2_profile["fingerprint_sha256"]
    assert mapanything_profile["fingerprint_sha256"]


def test_depth_registration_apply_rejects_out_of_domain(tmp_path: Path) -> None:
    _snapshot0, _dav2_profile, _mapanything_profile, entry = _entry()
    bundle_path = write_depth_registration_bundle(path=tmp_path / "depth_registration.json", entries={"cam0": entry})
    manager = DepthRegistrationManager.load(bundle_path)
    corrected, status, _reg_id = manager.apply(camera_id="cam0", raw_depth_m=99.0)
    assert corrected is None
    assert status == "out_of_domain_or_invalid"


def test_depth_registration_runtime_loader_rejects_legacy_contract(tmp_path: Path) -> None:
    snapshot, _dav2_profile, _mapanything_profile, entry = _entry()
    bundle_path = write_depth_registration_bundle(path=tmp_path / "depth_registration.json", entries={"cam0": entry})

    class _Provider:
        def snapshot(self, source_id: int, camera_id: str) -> CalibrationSnapshot:
            assert source_id == 0
            assert camera_id == "cam0"
            return CalibrationSnapshot(
                camera_id=snapshot.camera_id,
                intrinsics=np.array([[820.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float64),
                extrinsics_col_major=snapshot.extrinsics_col_major,
                floor_y=snapshot.floor_y,
                image_size=snapshot.image_size,
                unit_scale=snapshot.unit_scale,
            )

    pipeline_cfg = {
        "sources": [{"uri": "file:///tmp/example.mp4"}],
        "models": {
            "depth_tracking": {"name": "depth_tracking_fullframe", "engine": "models/engines/da2.engine"},
            "mapanything": {"name": "mapanything_fullframe", "engine": "models/mapanything_depth/1/model.plan"},
        },
    }
    with pytest.raises(
        DepthRegistrationError,
        match="depth_registration_contract_version_mismatch",
    ):
        _load_depth_registration_manager(
            path=bundle_path,
            pipeline_path=Path.cwd() / "config" / "infer.yaml",
            pipeline_cfg=pipeline_cfg,
            calibration_provider=_Provider(),
            camera_labels={0: "cam0"},
            logger=__import__("logging").getLogger(__name__),
        )


def test_depth_registration_path_resolves_from_pipeline_config(tmp_path: Path) -> None:
    pipeline_path = tmp_path / "infer.yaml"
    pipeline_path.write_text(
        "depth_registration:\n  path: DS9/config/depth_registration.json\n",
        encoding="utf-8",
    )
    args = __import__("argparse").Namespace(depth_registration_config=None)
    resolved = _resolve_depth_registration_path(args, pipeline_path=pipeline_path)
    assert resolved == (Path.cwd() / "DS9" / "config" / "depth_registration.json").resolve()
