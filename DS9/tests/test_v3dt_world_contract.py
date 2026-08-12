from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "DS9" / "scripts" / "v3dt_world_contract_smoke_test.py"
SPEC = importlib.util.spec_from_file_location("ds9_v3dt_world_contract_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
world_gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = world_gate
SPEC.loader.exec_module(world_gate)


def _digest(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _config_binding() -> dict[str, object]:
    projection = [
        100.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        100.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
    ]
    return {
        "schema_version": 1,
        "contract": world_gate.CONFIG_BINDING_CONTRACT,
        "contract_version": 1,
        "session": {
            "session_id": "v3dt-world-test",
            "runtime_lane": "v3dt",
            "build_mount": "/var/lib/noesis/build",
            "effective_pipeline_container_path": "/var/lib/noesis/build/effective_pipeline_yolo26_seg.yaml",
            "tracker_container_path": "/var/lib/noesis/build/config/v3dt/nvtracker_v3dt_runtime.yaml",
        },
        "files": {
            key: {
                "path": path,
                "sha256": _digest(key),
            }
            for key, path in sorted(world_gate.EXPECTED_BINDING_PATHS.items())
        },
        "caminfo": [
            {
                "camera_id": camera,
                "path": world_gate.EXPECTED_CAMINFO_PATHS[camera],
                "sha256": _digest(f"caminfo-{camera}"),
                "projection_key": "projectionMatrix_3x4_w2p",
                "projection_matrix_3x4": projection,
                "model_height_m": 2.2,
                "model_radius_m": 0.35,
            }
            for camera in world_gate.EXPECTED_CAMERAS
        ],
        "semantics": {
            "profile": "sv3dt",
            "tracking_mode": "v3dt",
            "world_frame": "backend_world_m",
            "caminfo_world_axes": "xzy",
            "camera_order": list(world_gate.EXPECTED_CAMERAS),
            "stream_size": [1920, 1080],
            "enable_padding": 0,
            "state_estimator_type": 3,
            "output_foot_location": 1,
            "output_visibility": 1,
            "floor_y_m": 0.0,
            "world_unit_scale_m": 1.0,
            "calibration_pose_frame": "backend_world_m",
            "minimum_camera_center_separation_m": 7.0,
        },
    }


def _raw_track(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "stable_id": 7,
        "resident_uuid": "must-not-persist",
        "tracker_id": 11,
        "frame_id": 7,
        "camera_id": "kitchen",
        "image_size": [1920, 1080],
        "bbox3d": {
            "xCentre": 1.0,
            "yCentre": 2.0,
            "zCentre": 1.0,
            "xLen": 0.5,
            "yLen": 0.4,
            "zLen": 1.8,
            "xRot": 0.0,
            "yRot": 0.0,
            "zRot": 0.0,
        },
        # Tracker foot [1, 2, 0.1] -> canonical xzy [1, 0.1, 2].
        # The ground-state filter snaps public Y to floor 0.
        "world": [1.0, 0.0, 2.0],
        "world_valid": True,
        "world_frame": "backend_world_m",
        "world_source": "bbox3d",
        "visibility": 0.9,
        "image_foot": [50.0, 5.0],
        # Opposite tracker endpoint [1, 2, 1.9].
        "image_base": [50.0, 95.0],
    }
    value.update(overrides)
    return value


def _source_track(**overrides: object) -> dict[str, object]:
    return world_gate._source_track(
        _raw_track(**overrides), session_id="v3dt-world-test"
    )


def _passing_tracks() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for camera in world_gate.EXPECTED_CAMERAS:
        rows.extend(
            (
                _source_track(camera_id=camera, frame_id=7),
                _source_track(camera_id=camera, frame_id=8),
            )
        )
    return rows


def _analyze(
    tracks: list[dict[str, object]],
    *,
    tracking_messages: int = 2,
    min_bbox3d_coverage: float = 0.95,
) -> dict[str, object]:
    return world_gate.analyze_tracks(
        tracks,
        session_id="v3dt-world-test",
        runtime_lane="v3dt",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        tracking_messages=tracking_messages,
        config_binding=_config_binding(),
        min_bbox3d_coverage=min_bbox3d_coverage,
    )


def test_locked_ds9_global_world_contract_passes_without_mv3dt_claim() -> None:
    result = _analyze(_passing_tracks())

    assert result["ok"] is True
    assert result["status"] == "pass"
    assert result["bbox3d_coverage"] == 1.0
    assert result["native_bridge_coverage"] == 1.0
    assert result["contract_coverage"] == 1.0
    assert result["advancing_trackers_by_camera"] == {
        camera: 1 for camera in world_gate.EXPECTED_CAMERAS
    }
    assert result["locked_world_contract"] == {
        "world_frame": "backend_world_m",
        "world_source": "bbox3d",
        "caminfo_world_axes": "xzy",
        "mv3dt_overlap_promotion": "not_claimed",
        "time_sync_promotion": "not_claimed",
    }
    assert result["old_raw_tuple_rejection"]["rejection_fraction"] == 1.0
    assert result["native_image_foot_reprojection_px"]["p95"] == pytest.approx(
        0.0, abs=1e-12
    )
    assert result["derived_image_base_reprojection_px"]["p95"] == pytest.approx(
        0.0, abs=1e-12
    )


def test_old_raw_tracker_tuple_is_decisively_rejected_even_if_relabelled() -> None:
    rows = _passing_tracks()
    for row in rows:
        row["world"] = [1.0, 2.0, 0.1]

    result = _analyze(rows)

    assert result["ok"] is False
    assert "public_world_axis_error" in result["failures"]
    assert "public_floor_error" in result["failures"]
    assert "old_raw_axis_tuple_not_rejected" in result["failures"]


def test_camera_local_label_and_missing_axis_binding_fail_closed() -> None:
    rows = _passing_tracks()
    rows[0]["world_frame"] = "unexpected"
    result = _analyze(rows)
    assert result["wrong_world_frame"] == 1
    assert "wrong_world_frame" in result["failures"]

    binding = _config_binding()
    binding["semantics"]["caminfo_world_axes"] = "xyz"
    with pytest.raises(ValueError, match="locked global-world profile"):
        world_gate.analyze_tracks(
            _passing_tracks(),
            session_id="v3dt-world-test",
            runtime_lane="v3dt",
            runtime_instance_id="runtime-instance-test",
            runtime_run_id="runtime-run-test",
            tracking_messages=2,
            config_binding=binding,
            min_bbox3d_coverage=0.95,
        )


def test_gate_rejects_empty_partial_and_invalid_bbox3d_evidence() -> None:
    empty = _analyze([], tracking_messages=0, min_bbox3d_coverage=0.5)
    assert empty["ok"] is False
    assert empty["status"] == "blocked"
    assert empty["occupied_scene_observed"] is False
    assert "insufficient_tracking_messages" in empty["failures"]
    assert "no_tracks" in empty["failures"]

    rows = _passing_tracks()
    rows[0]["bbox3d"] = {"xCentre": 1.0}
    result = _analyze(rows)
    assert result["ok"] is False
    assert result["invalid_bbox3d"] >= 1
    assert "bbox3d_coverage" in result["failures"]


def test_gate_requires_occupied_continuity_on_every_configured_camera() -> None:
    rows = [
        row for row in _passing_tracks() if row["camera_id"] != "family-room"
    ]
    result = _analyze(rows)

    assert result["ok"] is False
    assert "camera_coverage" in result["failures"]
    assert "missing_per_camera_tracker_continuity" in result["failures"]


def test_global_stable_id_repetition_cannot_replace_camera_tracker_continuity() -> None:
    rows = _passing_tracks()
    # One-frame tokens on every camera never prove within-camera continuity.
    for index, row in enumerate(rows):
        row["track_token"] = _digest(f"different-token-{index}")
    result = _analyze(rows)

    assert result["ok"] is False
    assert result["advancing_trackers_by_camera"] == {
        camera: 0 for camera in world_gate.EXPECTED_CAMERAS
    }
    assert "missing_per_camera_tracker_continuity" in result["failures"]


def test_native_metadata_and_both_projection_contracts_are_required() -> None:
    rows = _passing_tracks()
    rows[0]["visibility"] = None
    rows[1]["image_foot"] = [999.0, 999.0]
    rows[2]["image_base"] = [999.0, 999.0]
    result = _analyze(rows)

    assert result["ok"] is False
    assert result["invalid_visibility"] == 1
    assert "native_bridge_metadata_coverage" in result["failures"]
    assert "native_image_foot_reprojection" in result["failures"]
    assert "derived_image_base_reprojection" in result["failures"]


def test_cli_binds_all_canonical_runtime_geometry_inputs() -> None:
    args = world_gate._parse_args(
        [
            "--session-id",
            "v3dt-world-test",
            "--runtime-lane",
            "v3dt",
            "--runtime-instance-id",
            "runtime-instance-test",
            "--runtime-run-id",
            "runtime-run-test",
            "--source-out",
            "/tmp/v3dt-world-source.json",
            "--launcher-evidence-dir",
            "/tmp/v3dt-world-launcher",
        ]
    )
    assert args.min_bbox3d_coverage == 0.95
    assert args.pipeline_config == REPO_ROOT / "DS9/config/infer_v3dt.yaml"
    assert args.cameras_config == REPO_ROOT / "DS9/config/cameras_v3dt.yaml"
    assert args.calibration_config == REPO_ROOT / "config/camera_calibration.json"
    assert args.alignment_config == REPO_ROOT / "config/ply_alignment.json"
    assert args.launcher_evidence_dir == Path("/tmp/v3dt-world-launcher")


def test_config_binding_maps_exact_supervisor_external_build_mount(
    tmp_path: Path,
) -> None:
    session_id = "v3dt-world-test"
    runtime_root = tmp_path / "runtime"
    build_root = runtime_root / "build" / session_id
    launcher = runtime_root / "evidence" / session_id / "launcher"
    tracker_path = build_root / "config/v3dt/nvtracker_v3dt_runtime.yaml"
    build_root.mkdir(parents=True)
    tracker_path.parent.mkdir(parents=True)
    launcher.mkdir(parents=True)
    build_root.chmod(0o700)
    launcher.chmod(0o700)

    effective = yaml.safe_load(
        (REPO_ROOT / "DS9/config/infer_v3dt.yaml").read_text(encoding="utf-8")
    )
    effective["tracking_mode"] = "v3dt"
    effective["tracker"]["config-file"] = (
        "/var/lib/noesis/build/config/v3dt/nvtracker_v3dt_runtime.yaml"
    )
    (build_root / "effective_pipeline_yolo26_seg.yaml").write_text(
        yaml.safe_dump(effective, sort_keys=False), encoding="utf-8"
    )
    (build_root / "effective_pipeline_yolo26_seg.yaml").chmod(0o600)

    tracker = yaml.safe_load(
        (REPO_ROOT / "DS9/config/v3dt/nvtracker_v3dt.yaml").read_text(
            encoding="utf-8"
        )
    )
    tracker["ObjectModelProjection"]["cameraModelFilepath"] = [
        f"/workspace/{world_gate.EXPECTED_CAMINFO_PATHS[camera]}"
        for camera in world_gate.EXPECTED_CAMERAS
    ]
    tracker_path.write_text(
        yaml.safe_dump(tracker, sort_keys=False), encoding="utf-8"
    )
    tracker_path.chmod(0o600)

    launch_plan = {
        "schema_version": 1,
        "contract": "noesis.ds9.canonical_runtime_container",
        "mode": "plan",
        "ready_for_explicit_run": True,
        "session_id": session_id,
        "runtime_lane": "v3dt",
        "session_paths": {
            "build": str(build_root),
            "state": str(runtime_root / "state" / session_id),
            "depth": str(runtime_root / "depth" / session_id),
            "runtime_evidence": str(
                runtime_root / "evidence" / session_id / "runtime"
            ),
            "launcher_evidence": str(launcher),
        },
        "canonical_runtime": {
            "lane": "v3dt",
            "pipeline": "DS9/config/infer_v3dt.yaml",
            "cameras": "DS9/config/cameras_v3dt.yaml",
            "pgie_profile": "yolo26_seg",
            "model_size": "s",
            "tracking_mode": "v3dt",
            "source_ids": ["0", "1", "2"],
        },
    }
    (launcher / "launch-plan.json").write_text(
        json.dumps(launch_plan), encoding="utf-8"
    )
    (launcher / "launch-plan.json").chmod(0o600)

    binding = world_gate.build_config_binding(
        pipeline_config=REPO_ROOT / "DS9/config/infer_v3dt.yaml",
        cameras_config=REPO_ROOT / "DS9/config/cameras_v3dt.yaml",
        calibration_config=REPO_ROOT / "config/camera_calibration.json",
        alignment_config=REPO_ROOT / "config/ply_alignment.json",
        launcher_dir=launcher,
        session_id=session_id,
        runtime_lane="v3dt",
    )

    assert binding["session"] == {
        "session_id": session_id,
        "runtime_lane": "v3dt",
        "build_mount": "/var/lib/noesis/build",
        "effective_pipeline_container_path": "/var/lib/noesis/build/effective_pipeline_yolo26_seg.yaml",
        "tracker_container_path": "/var/lib/noesis/build/config/v3dt/nvtracker_v3dt_runtime.yaml",
    }
    assert binding["files"]["effective_pipeline"]["path"] == (
        "build/effective_pipeline_yolo26_seg.yaml"
    )
    assert binding["files"]["tracker"]["path"] == (
        "build/config/v3dt/nvtracker_v3dt_runtime.yaml"
    )
    assert binding["files"]["launch_plan"]["path"] == "launcher/launch-plan.json"

    alternate_build = runtime_root / "alternate-build" / session_id
    launch_plan["session_paths"]["build"] = str(alternate_build)
    (launcher / "launch-plan.json").write_text(
        json.dumps(launch_plan), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="canonical session"):
        world_gate.build_config_binding(
            pipeline_config=REPO_ROOT / "DS9/config/infer_v3dt.yaml",
            cameras_config=REPO_ROOT / "DS9/config/cameras_v3dt.yaml",
            calibration_config=REPO_ROOT / "config/camera_calibration.json",
            alignment_config=REPO_ROOT / "config/ply_alignment.json",
            launcher_dir=launcher,
            session_id=session_id,
            runtime_lane="v3dt",
        )
    launch_plan["session_paths"]["build"] = str(build_root)
    launch_plan["canonical_runtime"]["source_ids"] = ["0", "1"]
    (launcher / "launch-plan.json").write_text(
        json.dumps(launch_plan), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="canonical session"):
        world_gate.build_config_binding(
            pipeline_config=REPO_ROOT / "DS9/config/infer_v3dt.yaml",
            cameras_config=REPO_ROOT / "DS9/config/cameras_v3dt.yaml",
            calibration_config=REPO_ROOT / "config/camera_calibration.json",
            alignment_config=REPO_ROOT / "config/ply_alignment.json",
            launcher_dir=launcher,
            session_id=session_id,
            runtime_lane="v3dt",
        )
    launch_plan["canonical_runtime"]["source_ids"] = ["0", "1", "2"]
    (launcher / "launch-plan.json").write_text(
        json.dumps(launch_plan), encoding="utf-8"
    )

    effective["tracker"]["config-file"] = "/var/lib/noesis/build/other.yaml"
    (build_root / "effective_pipeline_yolo26_seg.yaml").write_text(
        yaml.safe_dump(effective, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="did not load the bound tracker"):
        world_gate.build_config_binding(
            pipeline_config=REPO_ROOT / "DS9/config/infer_v3dt.yaml",
            cameras_config=REPO_ROOT / "DS9/config/cameras_v3dt.yaml",
            calibration_config=REPO_ROOT / "config/camera_calibration.json",
            alignment_config=REPO_ROOT / "config/ply_alignment.json",
            launcher_dir=launcher,
            session_id=session_id,
            runtime_lane="v3dt",
        )

    launcher_link = runtime_root / "linked-launcher"
    launcher_link.symlink_to(launcher, target_is_directory=True)
    with pytest.raises(ValueError, match="owner-private directory"):
        world_gate.build_config_binding(
            pipeline_config=REPO_ROOT / "DS9/config/infer_v3dt.yaml",
            cameras_config=REPO_ROOT / "DS9/config/cameras_v3dt.yaml",
            calibration_config=REPO_ROOT / "config/camera_calibration.json",
            alignment_config=REPO_ROOT / "config/ply_alignment.json",
            launcher_dir=launcher_link,
            session_id=session_id,
            runtime_lane="v3dt",
        )

    plan_path = launcher / "launch-plan.json"
    plan_backing = launcher / "launch-plan.backing.json"
    plan_path.rename(plan_backing)
    os.link(plan_backing, plan_path)
    with pytest.raises(ValueError, match="exactly one hard link"):
        world_gate.build_config_binding(
            pipeline_config=REPO_ROOT / "DS9/config/infer_v3dt.yaml",
            cameras_config=REPO_ROOT / "DS9/config/cameras_v3dt.yaml",
            calibration_config=REPO_ROOT / "config/camera_calibration.json",
            alignment_config=REPO_ROOT / "config/ply_alignment.json",
            launcher_dir=launcher,
            session_id=session_id,
            runtime_lane="v3dt",
        )


def test_v2_source_transcript_is_private_minimal_typed_and_replayable() -> None:
    message = {
        "type": "tracking",
        "observed_at_us": 1_000_000,
        "tracks": [_source_track()],
    }
    source = world_gate._source_transcript_document(
        session_id="v3dt-world-test",
        runtime_lane="v3dt",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        config_binding=_config_binding(),
        min_bbox3d_coverage=0.95,
        messages=[message],
    )
    tracks, count = world_gate._replay_source_messages(
        source["messages"], session_id="v3dt-world-test"
    )
    assert source["schema_version"] == 2
    assert source["contract_version"] == 2
    assert count == 1
    assert len(tracks) == 1
    encoded_tracks = json.dumps(source["messages"], sort_keys=True)
    for forbidden in (
        "stable_id",
        "resident_uuid",
        "tracker_id",
        "embedding",
        "image_b64",
        "camera-secret",
    ):
        assert forbidden not in encoded_tracks

    poisoned = dict(message)
    poisoned_track = dict(poisoned["tracks"][0])
    poisoned_track["embedding"] = [0.1, 0.2]
    poisoned["tracks"] = [poisoned_track]
    with pytest.raises(ValueError, match="schema drifted"):
        world_gate._replay_source_messages(
            [poisoned], session_id="v3dt-world-test"
        )


def test_v1_source_envelope_is_rejected_not_accepted_as_compatibility() -> None:
    source = world_gate._source_transcript_document(
        session_id="v3dt-world-test",
        runtime_lane="v3dt",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        config_binding=_config_binding(),
        min_bbox3d_coverage=0.95,
        messages=[],
    )
    source["schema_version"] = 1
    source["contract_version"] = 1
    assert source["schema_version"] != world_gate.SOURCE_TRANSCRIPT_VERSION
    assert source["contract_version"] != world_gate.SOURCE_TRANSCRIPT_VERSION


def test_world_source_track_redacts_invalid_field_payloads() -> None:
    poisoned = _raw_track(
        tracker_id="secret-tracker",
        frame_id="secret-frame",
        camera_id="secret-camera",
        image_size=["secret-size", "secret-size"],
        bbox3d={key: "secret-bbox" for key in world_gate.EXPECTED_BBOX3D_KEYS},
        world=["secret-world"] * 3,
        world_valid="secret-valid",
        world_frame="secret-frame-name",
        world_source="secret-source",
        visibility="secret-visibility",
        image_foot=["secret-foot", "secret-foot"],
        image_base=["secret-base", "secret-base"],
    )

    sanitized = world_gate._source_track(
        poisoned, session_id="v3dt-world-test"
    )
    assert "secret" not in json.dumps(sanitized, sort_keys=True)
    assert sanitized["track_token"] is None
    assert sanitized["frame_id"] is None
    assert sanitized["camera_id"] == "unexpected"
    assert sanitized["image_size"] == [None, None]
    assert all(value is None for value in sanitized["bbox3d"].values())
    assert sanitized["world"] == [None, None, None]
    assert sanitized["visibility"] is None
    assert sanitized["image_foot"] == [None, None]
    assert sanitized["image_base"] == [None, None]


def test_world_numeric_normalization_rejects_overflow_without_crashing() -> None:
    huge = 10**10_000
    sanitized = world_gate._source_track(
        _raw_track(
            bbox3d={key: huge for key in world_gate.EXPECTED_BBOX3D_KEYS},
            world=[huge, huge, huge],
            visibility=huge,
            image_foot=[huge, huge],
            image_base=[huge, huge],
        ),
        session_id="v3dt-world-test",
    )

    assert all(value is None for value in sanitized["bbox3d"].values())
    assert sanitized["world"] == [None, None, None]
    assert sanitized["visibility"] is None
    assert sanitized["image_foot"] == [None, None]
    assert sanitized["image_base"] == [None, None]
