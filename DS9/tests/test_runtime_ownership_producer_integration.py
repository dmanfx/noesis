from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SESSION_ID = "authentic-producer-session"


def _load(name: str, relative: str) -> ModuleType:
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ownership = _load(
    "ownership_producer_integration_validator",
    "DS9/scripts/validate_runtime_ownership.py",
)
wholebody = _load(
    "ownership_producer_integration_wholebody",
    "DS9/scripts/wholebody49_occupied_scene_smoke_test.py",
)
wholebody_media = _load(
    "ownership_producer_integration_wholebody_media",
    "DS9/scripts/wholebody49_media_decode_gate.py",
)
identity = _load(
    "ownership_producer_integration_identity",
    "DS9/scripts/ds9_identity_shadow_live_gate.py",
)
floorplan = _load(
    "ownership_producer_integration_floorplan",
    "DS9/scripts/ds9_floorplan_live_gate.py",
)
semantic = _load(
    "ownership_producer_integration_semantic",
    "DS9/scripts/ds9_semantic_observation_smoke_test.py",
)
semantic_fixture = _load(
    "ownership_producer_integration_semantic_fixture",
    "DS9/tests/test_semantic_observation_gate.py",
)
live_acceptance_fixture = _load(
    "ownership_producer_integration_live_fixture",
    "DS9/tests/test_live_acceptance_gates.py",
)
v3dt_world = _load(
    "ownership_producer_integration_v3dt_world",
    "DS9/scripts/v3dt_world_contract_smoke_test.py",
)
resource = _load(
    "ownership_producer_integration_resource",
    "DS9/scripts/run_canonical_runtime_container.py",
)
resource_fixture = _load(
    "ownership_producer_integration_resource_fixture",
    "DS9/tests/test_runtime_container_boundary.py",
)


def _assert_registered_report(
    report: Mapping[str, Any],
    behavior_id: str,
    *,
    lane: str,
) -> None:
    spec = ownership.BEHAVIOR_CONTRACTS[behavior_id]
    assert report["schema_version"] == int(spec.get("schema_version", 1))
    assert type(report["schema_version"]) is int
    assert report["contract"] == spec["contract"]
    assert report["contract_version"] == int(spec.get("contract_version", 1))
    assert type(report["contract_version"]) is int
    assert report["session_id"] == SESSION_ID
    assert report["runtime_lane"] == lane
    assert report["runtime_instance_id"] == "runtime-instance-test"
    assert report["runtime_run_id"] == "runtime-run-test"
    assert lane in spec["lanes"]
    for assertion in spec["assertions"]:
        actual = ownership._json_pointer(
            report,
            assertion["pointer"],
            f"authentic producer {behavior_id}",
        )
        assert type(actual) is assertion["type"]
        if assertion["op"] == "equals":
            assert actual == assertion["value"]
        else:
            assert assertion["op"] == "minimum"
            assert actual >= assertion["value"]


def _identity_bundle(lane: str) -> tuple[Mapping[str, Any], bytes]:
    collector = identity.IdentityEvidenceCollector()
    collector.observe_payload(
        {
            "type": "tracking",
            "tracks": [
                live_acceptance_fixture._identity_track(
                    camera="kitchen",
                    tracker=1,
                    frame=1,
                    observed_at_us=1_783_749_601_000_000,
                    state="unknown", subject=None, sid=None,
                    reason="provisional_evidence_pending",
                ),
                live_acceptance_fixture._identity_track(
                    camera="kitchen",
                    tracker=2,
                    frame=10,
                    observed_at_us=1_783_749_601_100_000,
                    state="visitor", subject="visitor:runtime:generation:0", sid=1000,
                    reason="visitor_provisional_confirmed", visitor_generation=0,
                ),
                live_acceptance_fixture._identity_track(
                    camera="kitchen",
                    tracker=2,
                    frame=11,
                    observed_at_us=1_783_749_601_200_000,
                    state="visitor", subject="visitor:runtime:generation:0", sid=1000,
                    reason="matched", visitor_generation=0,
                ),
            ],
        }
    )
    raw_health = live_acceptance_fixture._health()
    health = identity._validate_health_snapshot(
        raw_health,
        expected_layer="fc_pred",
        expected_dimension=256,
    )
    source_document = identity._source_transcript_document(
        session_id=SESSION_ID,
        runtime_lane=lane,
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        collector=collector,
        health_before=raw_health,
        health_after=raw_health,
        require_cross_camera=False,
        require_open_set=True,
        min_fresh_embeddings=2,
    )
    source_encoded = identity._encoded_private_json(source_document)
    source_filename = (
        identity.BASELINE_SOURCE_TRANSCRIPT_FILENAME
        if lane == "baseline"
        else identity.V3DT_SOURCE_TRANSCRIPT_FILENAME
    )
    source_evidence = identity._source_evidence_metadata(
        filename=source_filename,
        encoded=source_encoded,
        document=source_document,
        collector=collector,
    )
    report = identity._build_report(
        session_id=SESSION_ID,
        runtime_lane=lane,
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        collector=collector,
        health_before=health,
        health_after=health,
        require_cross_camera=False,
        require_open_set=True,
        min_fresh_embeddings=2,
        source_evidence=source_evidence,
        errors=[],
    )
    return report, source_encoded


def _identity_report(lane: str) -> Mapping[str, Any]:
    return _identity_bundle(lane)[0]


def _wholebody_bundle(
    *,
    lane: str,
    mode: str,
    counters: Mapping[str, int],
    source_ids: tuple[int, ...] = (0, 1, 2),
) -> tuple[Mapping[str, Any], bytes]:
    events: list[dict[str, object]] = []
    base = 1_783_749_601_000_000
    frame_ids = {source_id: 0 for source_id in source_ids}
    for seconds in range(0, 33, 2):
        for source_id in source_ids:
            events.append(
                {
                    "type": "tracking",
                    "observed_at_us": base + seconds * 1_000_000 + source_id + 1,
                    "source_id": source_id,
                    "frame_id": frame_ids[source_id],
                    "person_track_count": 1 if source_id == 0 else 0,
                }
            )
            frame_ids[source_id] += 6
        events.append(
            {
                "type": "stats",
                "observed_at_us": (
                    base + seconds * 1_000_000 + max(source_ids) + 2
                ),
                "counters": {
                    wholebody.MASK_COUNTER: int(
                        counters.get(wholebody.MASK_COUNTER, 0)
                    ),
                    wholebody.BBOX_COUNTER: int(
                        counters.get(wholebody.BBOX_COUNTER, 0)
                    ),
                    wholebody.CPU_VIOLATION_COUNTER: int(
                        counters.get(wholebody.CPU_VIOLATION_COUNTER, 0)
                    ),
                },
                "pipeline_errors": [],
                "pipeline_prepared": True,
                "pipeline_activated": True,
                "application_running": True,
            }
        )
    source_document = wholebody._source_transcript_document(
        session_id=SESSION_ID,
        runtime_lane=lane,
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        mode=mode,
        expected_source_ids=source_ids,
        events=events,
    )
    source_encoded = wholebody._encoded_private_json(source_document)
    source_evidence = wholebody._source_evidence_metadata(
        encoded=source_encoded,
        document=source_document,
    )
    return (
        wholebody.analyze_evidence(
            session_id=SESSION_ID,
            runtime_lane=lane,
            runtime_instance_id="runtime-instance-test",
            runtime_run_id="runtime-run-test",
            mode=mode,
            expected_source_ids=source_ids,
            events=events,
            source_evidence=source_evidence,
        ),
        source_encoded,
    )


@pytest.mark.parametrize(
    ("lane", "mode", "counter_name"),
    (
        ("wholebody49-s", "masks", wholebody.MASK_COUNTER),
        ("wholebody49-x", "boxes", wholebody.BBOX_COUNTER),
    ),
)
@pytest.mark.parametrize(
    "mutation",
    ("subset", "superset", "reordered", "boolean"),
)
def test_wholebody_ownership_binds_exact_reviewed_source_inventory(
    lane: str,
    mode: str,
    counter_name: str,
    mutation: str,
) -> None:
    producer_source_ids = {
        "subset": (0, 1),
        "superset": (0, 1, 2, 3),
    }.get(mutation, (0, 1, 2))
    report, source_raw = _wholebody_bundle(
        lane=lane,
        mode=mode,
        counters={
            wholebody.MASK_COUNTER: 2 if counter_name == wholebody.MASK_COUNTER else 0,
            wholebody.BBOX_COUNTER: 2 if counter_name == wholebody.BBOX_COUNTER else 0,
            wholebody.CPU_VIOLATION_COUNTER: 0,
        },
        source_ids=producer_source_ids,
    )
    report = copy.deepcopy(report)
    if mutation == "reordered":
        report["expected_source_ids"] = [2, 1, 0]
    elif mutation == "boolean":
        report["expected_source_ids"] = [False, 1, 2]

    with pytest.raises(ValueError, match="differs from reviewed lane"):
        ownership._validate_wholebody_behavior(
            report,
            "Wholebody adversarial source inventory",
            covered={wholebody.CANONICAL_SOURCE_TRANSCRIPT_FILENAME: source_raw},
            session_id=SESSION_ID,
            lane=lane,
            runtime_instance_id="runtime-instance-test",
            runtime_run_id="runtime-run-test",
            inspect_started=ownership._parse_rfc3339_utc(
                "2026-07-11T05:59:59Z", "fixture start"
            ),
            inspect_finished=ownership._parse_rfc3339_utc(
                "2026-07-11T06:05:02Z", "fixture finish"
            ),
        )


def _wholebody_media_bundle(
    lane: str,
) -> tuple[Mapping[str, Any], bytes]:
    source = wholebody_media._source_document(
        session_id=SESSION_ID,
        runtime_lane=lane,
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        started_at_us=1_783_749_650_000_000,
        finished_at_us=1_783_749_662_000_000,
        probe_result={
            "ok": True,
            "answer_video_direction": "sendonly",
            "ice_state": "connected",
            "peer_state": "connected",
            "rtp_packets": 20,
            "decoded_frames": 3,
            "saw_src_pad": True,
            "src_caps": "application/x-rtp, encoding-name=H264",
            "min_rtp": 10,
            "min_decoded": 1,
            "rtsp_decoded_frames": 4,
            "rtsp_min_decoded": 1,
        },
    )
    source_raw = wholebody_media._encoded(source)
    report = wholebody_media.analyze_source(
        source,
        source_evidence=wholebody_media._source_evidence(source_raw, source),
    )
    return report, source_raw


def _v3dt_track(
    frame_id: int,
    camera_id: str,
    config_binding: Mapping[str, object],
) -> dict[str, object]:
    bbox3d = {
        "xCentre": 1.0,
        "yCentre": 2.0,
        "zCentre": 0.9,
        "xLen": 0.5,
        "yLen": 0.4,
        "zLen": 1.8,
        "xRot": 0.0,
        "yRot": 0.0,
        "zRot": 0.0,
    }
    caminfo = next(
        row
        for row in config_binding["caminfo"]
        if row["camera_id"] == camera_id
    )
    image_foot = v3dt_world._project_tracker_point(
        (1.0, 2.0, 0.0),
        caminfo=caminfo,
        image_size=(1920, 1080),
    )
    image_base = v3dt_world._project_tracker_point(
        (1.0, 2.0, 1.8),
        caminfo=caminfo,
        image_size=(1920, 1080),
    )
    assert image_foot is not None and image_base is not None
    return {
        "stable_id": 7,
        "tracker_id": 11,
        "frame_id": frame_id,
        "camera_id": camera_id,
        "image_size": [1920, 1080],
        "bbox3d": bbox3d,
        "world": [1.0, 0.0, 2.0],
        "world_valid": True,
        "world_frame": "backend_world_m",
        "world_source": "bbox3d",
        "visibility": 0.9,
        "image_foot": list(image_foot),
        "image_base": list(image_base),
    }


def test_authentic_gate_and_supervisor_outputs_satisfy_behavior_registry(
    tmp_path: Path,
) -> None:
    assert ownership.EXPECTED_SECURITY_OPTIONS == list(
        resource.EXPECTED_SECURITY_OPTIONS
    )
    assert ownership.EXPECTED_APPARMOR_PROFILE == resource.EXPECTED_APPARMOR_PROFILE
    assert dict(ownership.EXPECTED_SENSITIVE_HOST_DEFAULTS) == dict(
        resource.EXPECTED_SENSITIVE_HOST_DEFAULTS
    )
    assert ownership.EXPECTED_ABSENT_SENSITIVE_HOST_FIELDS == (
        resource.EXPECTED_ABSENT_SENSITIVE_HOST_FIELDS
    )
    assert ownership.EXPECTED_READONLY_PATHS == list(resource.EXPECTED_READONLY_PATHS)
    assert ownership.EXPECTED_MASKED_PATHS == list(resource.EXPECTED_MASKED_PATHS)
    wholebody_s, wholebody_s_source = _wholebody_bundle(
        lane="wholebody49-s",
        mode="masks",
        counters={
            wholebody.MASK_COUNTER: 2,
            wholebody.BBOX_COUNTER: 0,
            wholebody.CPU_VIOLATION_COUNTER: 0,
        },
    )
    wholebody_x, wholebody_x_source = _wholebody_bundle(
        lane="wholebody49-x",
        mode="boxes",
        counters={
            wholebody.MASK_COUNTER: 0,
            wholebody.BBOX_COUNTER: 2,
            wholebody.CPU_VIOLATION_COUNTER: 0,
        },
    )
    calibration_fingerprints = {
        "family-room": "ee22aa6b6367aabf31944d53e0a8edc13e8dfe2a06f589b23bc8e522f71d0bc6",
        "kitchen": "16654dd3b282a3a574bdf9a7299f5671804c4dab97276934627252a86182850e",
        "living-room": "2e6253444412b292a6c41061be530e4db218d2128b6093c1ccbf9bf2886d102c",
    }
    camera_results: dict[str, Mapping[str, object]] = {}
    floorplan_payloads: dict[str, dict[str, object]] = {}
    cache_results: dict[str, Mapping[str, object]] = {}
    for camera_id, calibration_fingerprint in calibration_fingerprints.items():
        request_id = f"request-{camera_id}"
        payload = live_acceptance_fixture._floorplan_payload(
            camera=camera_id,
            request_id=request_id,
        )
        payload["calibration_fingerprint"] = calibration_fingerprint
        floorplan_payloads[camera_id] = payload
        camera_results[camera_id] = floorplan._validate_floorplan_payload(
            payload,
            camera_id=camera_id,
            request_id=request_id,
            max_age_sec=120.0,
            now_us=10_000_000,
        )
        cache_payload = live_acceptance_fixture._cache_payload(
            payload,
            request_id=f"request-cache-{camera_id}",
        )
        cache_results[camera_id] = floorplan._validate_floorplan_payload(
            cache_payload,
            camera_id=camera_id,
            request_id=f"request-cache-{camera_id}",
            max_age_sec=120.0,
            now_us=10_000_000,
            mode="cache_only",
            expected_fresh=camera_results[camera_id],
        )
    floorplan_health = live_acceptance_fixture._health_result(
        {key: dict(value) for key, value in camera_results.items()},
        inactive_cameras=(tuple(calibration_fingerprints)[-1],),
    )
    floorplan_events: list[dict[str, object]] = [
        {
            "type": "validated_exact_floorplan_capture",
            "observed_at_us": 1_783_749_601_300_000 + index,
            "request_id": f"request-{camera_id}",
            "attempt": 1,
            "camera_id": camera_id,
            "result": dict(camera_results[camera_id]),
            "capture_event": copy.deepcopy(
                floorplan_payloads[camera_id]["capture_event"]
            ),
        }
        for index, camera_id in enumerate(calibration_fingerprints)
    ]
    floorplan_events.append(
        {
            "type": "validated_floorplan_runtime_health",
            "observed_at_us": 1_783_749_601_300_010,
            "phase": "after_fresh",
            "result": copy.deepcopy(floorplan_health),
        }
    )
    floorplan_events.extend(
        {
            "type": "validated_cache_only_floorplan",
            "observed_at_us": 1_783_749_601_300_020 + index,
            "request_id": f"request-cache-{camera_id}",
            "attempt": 1,
            "camera_id": camera_id,
            "result": dict(cache_results[camera_id]),
        }
        for index, camera_id in enumerate(calibration_fingerprints)
    )
    floorplan_events.append(
        {
            "type": "validated_floorplan_runtime_health",
            "observed_at_us": 1_783_749_601_300_030,
            "phase": "after_cache_only",
            "result": copy.deepcopy(floorplan_health),
        }
    )
    floorplan_source_document = floorplan._source_transcript_document(
        session_id=SESSION_ID,
        runtime_lane="baseline",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        camera_ids=tuple(calibration_fingerprints),
        max_snapshot_age_s=120.0,
        events=floorplan_events,
    )
    floorplan_source_encoded = floorplan._encoded_private_json(
        floorplan_source_document
    )
    floorplan_source_evidence = floorplan._source_evidence_metadata(
        encoded=floorplan_source_encoded,
        document=floorplan_source_document,
    )
    depth_quality = floorplan._build_report(
        session_id=SESSION_ID,
        runtime_lane="baseline",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        camera_ids=tuple(calibration_fingerprints),
        results=camera_results,
        cache_results=cache_results,
        after_fresh_health=floorplan_health,
        after_cache_only_health=copy.deepcopy(floorplan_health),
        max_snapshot_age_s=120.0,
        errors=[],
        source_evidence=floorplan_source_evidence,
    )
    world_runtime = tmp_path / "v3dt-runtime"
    world_build = world_runtime / "build" / SESSION_ID
    world_launcher = world_runtime / "evidence" / SESSION_ID / "launcher"
    world_tracker = world_build / "config/v3dt/nvtracker_v3dt_runtime.yaml"
    world_tracker.parent.mkdir(parents=True)
    world_launcher.mkdir(parents=True)
    world_build.chmod(0o700)
    world_launcher.chmod(0o700)
    effective_pipeline = yaml.safe_load(
        (REPO_ROOT / "DS9/config/infer_v3dt.yaml").read_text(encoding="utf-8")
    )
    effective_pipeline["tracking_mode"] = "v3dt"
    effective_pipeline["tracker"]["config-file"] = (
        "/var/lib/noesis/build/config/v3dt/nvtracker_v3dt_runtime.yaml"
    )
    (world_build / "effective_pipeline_yolo26_seg.yaml").write_text(
        yaml.safe_dump(effective_pipeline, sort_keys=False), encoding="utf-8"
    )
    (world_build / "effective_pipeline_yolo26_seg.yaml").chmod(0o600)
    tracker_payload = yaml.safe_load(
        (REPO_ROOT / "DS9/config/v3dt/nvtracker_v3dt.yaml").read_text(
            encoding="utf-8"
        )
    )
    tracker_payload["ObjectModelProjection"]["cameraModelFilepath"] = [
        f"/workspace/{v3dt_world.EXPECTED_CAMINFO_PATHS[camera]}"
        for camera in v3dt_world.EXPECTED_CAMERAS
    ]
    world_tracker.write_text(
        yaml.safe_dump(tracker_payload, sort_keys=False), encoding="utf-8"
    )
    world_tracker.chmod(0o600)
    launch_plan = {
        "schema_version": 1,
        "contract": "noesis.ds9.canonical_runtime_container",
        "mode": "plan",
        "ready_for_explicit_run": True,
        "session_id": SESSION_ID,
        "runtime_lane": "v3dt",
        "image": {"id": resource.IMAGE_ID},
        "session_paths": {
            "build": str(world_build),
            "state": str(world_runtime / "state" / SESSION_ID),
            "depth": str(world_runtime / "depth" / SESSION_ID),
            "runtime_evidence": str(
                world_runtime / "evidence" / SESSION_ID / "runtime"
            ),
            "launcher_evidence": str(world_launcher),
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
    launch_plan_raw = (
        json.dumps(launch_plan, sort_keys=True) + "\n"
    ).encode("utf-8")
    (world_launcher / "launch-plan.json").write_bytes(launch_plan_raw)
    (world_launcher / "launch-plan.json").chmod(0o600)
    world_binding = v3dt_world.build_config_binding(
        pipeline_config=REPO_ROOT / "DS9/config/infer_v3dt.yaml",
        cameras_config=REPO_ROOT / "DS9/config/cameras_v3dt.yaml",
        calibration_config=REPO_ROOT / "config/camera_calibration.json",
        alignment_config=REPO_ROOT / "config/ply_alignment.json",
        launcher_dir=world_launcher,
        session_id=SESSION_ID,
        runtime_lane="v3dt",
    )
    world_source_messages = []
    world_tracks = []
    for camera_index, camera in enumerate(v3dt_world.EXPECTED_CAMERAS):
        for frame_offset, frame_id in enumerate((7, 8)):
            raw_track = _v3dt_track(frame_id, camera, world_binding)
            source_track = v3dt_world._source_track(
                raw_track, session_id=SESSION_ID
            )
            world_tracks.append(source_track)
            world_source_messages.append(
                {
                    "type": "tracking",
                    "observed_at_us": (
                        1_783_749_601_400_000
                        + (camera_index * 10)
                        + frame_offset
                    ),
                    "tracks": [source_track],
                }
            )
    world_source_document = v3dt_world._source_transcript_document(
        session_id=SESSION_ID,
        runtime_lane="v3dt",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        config_binding=world_binding,
        min_bbox3d_coverage=0.95,
        messages=world_source_messages,
    )
    world_source_encoded = v3dt_world._encoded_private_json(world_source_document)
    world_source_evidence = v3dt_world._source_evidence_metadata(
        encoded=world_source_encoded,
        document=world_source_document,
    )
    world = v3dt_world.analyze_tracks(
        world_tracks,
        session_id=SESSION_ID,
        runtime_lane="v3dt",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        tracking_messages=len(world_source_messages),
        config_binding=world_binding,
        min_bbox3d_coverage=0.95,
        source_evidence=world_source_evidence,
    )
    world_report_path = world_launcher / v3dt_world.CANONICAL_REPORT_FILENAME
    world_source_path = (
        world_launcher / v3dt_world.CANONICAL_SOURCE_TRANSCRIPT_FILENAME
    )
    canonical_world_report_raw = v3dt_world._encoded_private_json(world)
    world_report_path.write_bytes(canonical_world_report_raw)
    world_source_path.write_bytes(world_source_encoded)
    world_report_path.chmod(0o600)
    world_source_path.chmod(0o600)
    assert v3dt_world.validate_sealed_v3dt_world_report(
        world_report_path,
        world_source_path,
        pipeline_config=REPO_ROOT / "DS9/config/infer_v3dt.yaml",
        cameras_config=REPO_ROOT / "DS9/config/cameras_v3dt.yaml",
        calibration_config=REPO_ROOT / "config/camera_calibration.json",
        alignment_config=REPO_ROOT / "config/ply_alignment.json",
        launcher_dir=world_launcher,
        session_id=SESSION_ID,
        runtime_lane="v3dt",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
    )["ok"] is True
    tampered_world_report = copy.deepcopy(world)
    tampered_world_report["tracks_seen"] += 1
    world_report_path.write_bytes(
        v3dt_world._encoded_private_json(tampered_world_report)
    )
    world_report_path.chmod(0o600)
    with pytest.raises(ValueError, match="exactly replay"):
        v3dt_world.validate_sealed_v3dt_world_report(
            world_report_path,
            world_source_path,
            pipeline_config=REPO_ROOT / "DS9/config/infer_v3dt.yaml",
            cameras_config=REPO_ROOT / "DS9/config/cameras_v3dt.yaml",
            calibration_config=REPO_ROOT / "config/camera_calibration.json",
            alignment_config=REPO_ROOT / "config/ply_alignment.json",
            launcher_dir=world_launcher,
            session_id=SESSION_ID,
            runtime_lane="v3dt",
            runtime_instance_id="runtime-instance-test",
            runtime_run_id="runtime-run-test",
        )
    world_report_path.write_bytes(canonical_world_report_raw)
    world_report_path.chmod(0o600)
    world_source_path.write_bytes(
        world_source_encoded.replace(
            b'"schema_version": 2,',
            b'"schema_version": 2,\n  "schema_version": 2,',
            1,
        )
    )
    world_source_path.chmod(0o600)
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        v3dt_world.validate_sealed_v3dt_world_report(
            world_report_path,
            world_source_path,
            pipeline_config=REPO_ROOT / "DS9/config/infer_v3dt.yaml",
            cameras_config=REPO_ROOT / "DS9/config/cameras_v3dt.yaml",
            calibration_config=REPO_ROOT / "config/camera_calibration.json",
            alignment_config=REPO_ROOT / "config/ply_alignment.json",
            launcher_dir=world_launcher,
            session_id=SESSION_ID,
            runtime_lane="v3dt",
            runtime_instance_id="runtime-instance-test",
            runtime_run_id="runtime-run-test",
        )
    world_source_path.write_bytes(world_source_encoded)
    world_source_path.chmod(0o600)
    semantic_collector = semantic.SemanticObservationCollector()
    semantic_collector.observe_payload(semantic_fixture._stats())
    semantic_tracking = semantic_fixture._tracking_payload()
    semantic_tracking["captured_at_us"] = 1_783_749_602_000_000
    semantic_tracking["observed_at_us"] = 1_783_749_602_000_000
    semantic_tracking["media_pts_ns"] = 1_783_749_602_000_000_000
    semantic_tracking["tracks"][0]["observed_at_us"] = 1_783_749_602_000_000
    semantic_observation = semantic_tracking["observations"][0]
    semantic_observation["captured_at_us"] = 1_783_749_602_000_000
    semantic_observation["observed_at_us"] = 1_783_749_602_000_000
    semantic_observation["published_at_us"] = 1_783_749_602_000_001
    semantic_observation["media_pts_ns"] = 1_783_749_602_000_000_000
    semantic_observation["payload"]["tracklet"]["observed_at_us"] = (
        1_783_749_602_000_000
    )
    semantic_collector.observe_payload(semantic_tracking)
    semantic_acquisition_started_at_us = 1_783_749_601_900_000
    semantic_acquisition_finished_at_us = 1_783_749_602_100_000
    semantic_source_document = semantic._source_transcript_document(
        session_id=SESSION_ID,
        runtime_lane="v3dt",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        acquisition_started_at_us=semantic_acquisition_started_at_us,
        acquisition_finished_at_us=semantic_acquisition_finished_at_us,
        collector=semantic_collector,
    )
    semantic_source_encoded = semantic._encoded_source_transcript(
        semantic_source_document
    )
    semantic_snapshot = semantic_fixture._snapshot(
        session_id=SESSION_ID,
        observed_at_us=1_783_749_602_000_000,
    )
    semantic_report = semantic._evaluate(
        semantic_collector,
        semantic_snapshot,
        session_id=SESSION_ID,
        runtime_lane="v3dt",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        acquisition_started_at_us=semantic_acquisition_started_at_us,
        acquisition_finished_at_us=semantic_acquisition_finished_at_us,
        expected_model_layer="fc_pred",
        expected_embedding_dimension=256,
        sealed_snapshot_filename=semantic.CANONICAL_IDENTITY_SNAPSHOT_FILENAME,
        source_evidence=semantic._source_evidence_metadata(
            filename=semantic.CANONICAL_SOURCE_TRANSCRIPT_FILENAME,
            encoded=semantic_source_encoded,
            document=semantic_source_document,
        ),
    )
    baseline_semantic_source_document = semantic._source_transcript_document(
        session_id=SESSION_ID,
        runtime_lane="baseline",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        acquisition_started_at_us=semantic_acquisition_started_at_us,
        acquisition_finished_at_us=semantic_acquisition_finished_at_us,
        collector=semantic_collector,
    )
    baseline_semantic_source_encoded = semantic._encoded_source_transcript(
        baseline_semantic_source_document
    )
    baseline_semantic_report = semantic._evaluate(
        semantic_collector,
        semantic_snapshot,
        session_id=SESSION_ID,
        runtime_lane="baseline",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        acquisition_started_at_us=semantic_acquisition_started_at_us,
        acquisition_finished_at_us=semantic_acquisition_finished_at_us,
        expected_model_layer="fc_pred",
        expected_embedding_dimension=256,
        sealed_snapshot_filename=semantic.CANONICAL_IDENTITY_SNAPSHOT_FILENAME,
        source_evidence=semantic._source_evidence_metadata(
            filename=semantic.CANONICAL_SOURCE_TRANSCRIPT_FILENAME,
            encoded=baseline_semantic_source_encoded,
            document=baseline_semantic_source_document,
        ),
    )
    resource_samples = resource_fixture._resource_samples(
        lambda _elapsed: 4 * 1024 * 1024 * 1024
    )
    container_id = "d" * 64
    primary_engine_sha256 = {
        "v3dt": "c" * 64,
        "wholebody49-s": "d" * 64,
        "wholebody49-x": "e" * 64,
    }

    def resource_bundle(lane: str) -> tuple[Mapping[str, Any], bytes]:
        lane_policy = resource.RUNTIME_LANES[lane]
        binding = {
            "container_id": container_id,
            "runtime_image_id": resource.IMAGE_ID,
            "checkout_sha256": "a" * 64,
            "realization_sha256": "b" * 64,
            "primary_engine_artifact_id": resource.RESOURCE_SOAK_PRIMARY_ENGINE_ID[
                lane
            ],
            "primary_engine_sha256": primary_engine_sha256[lane],
            "pipeline_config": lane_policy.pipeline_config,
            "pipeline_config_sha256": resource._sha256_file(
                resource.REPO_ROOT / lane_policy.pipeline_config
            ),
            "cameras_config": lane_policy.cameras_config,
            "cameras_config_sha256": resource._sha256_file(
                resource.REPO_ROOT / lane_policy.cameras_config
            ),
        }
        report = resource.evaluate_resource_soak_samples(
            resource_samples,
            session_id=SESSION_ID,
            runtime_lane=lane,
            runtime_instance_id="runtime-instance-test",
            runtime_run_id="runtime-run-test",
            requested_duration_seconds=resource.RESOURCE_SOAK_MIN_DURATION_SECONDS,
            stop_reason="duration_complete",
            runtime_binding=binding,
        )
        report.update(
            {
                "started_at_utc": "2026-07-11T06:00:00Z",
                "finished_at_utc": "2026-07-11T06:05:01Z",
                "container_cgroup_v2_path": (
                    f"/sys/fs/cgroup/docker-{container_id}.scope"
                ),
            }
        )
        launcher = tmp_path / f"launcher-{lane}"
        launcher.mkdir(mode=0o700)
        persisted = resource.persist_resource_soak_evidence(
            launcher, resource_samples, report
        )
        return (
            persisted,
            (launcher / resource.RESOURCE_SOAK_SAMPLES_FILENAME).read_bytes(),
        )

    soak, soak_samples_raw = resource_bundle("v3dt")
    wholebody_s_soak, wholebody_s_soak_raw = resource_bundle("wholebody49-s")
    wholebody_x_soak, wholebody_x_soak_raw = resource_bundle("wholebody49-x")
    wholebody_s_media, wholebody_s_media_source = _wholebody_media_bundle(
        "wholebody49-s"
    )
    wholebody_x_media, wholebody_x_media_source = _wholebody_media_bundle(
        "wholebody49-x"
    )

    reports = (
        (wholebody_s, "wholebody49_occupied_s_v2", "wholebody49-s"),
        (wholebody_x, "wholebody49_occupied_x_v2", "wholebody49-x"),
        (wholebody_s_media, "wholebody49_media_decode_v1", "wholebody49-s"),
        (wholebody_x_media, "wholebody49_media_decode_v1", "wholebody49-x"),
        (_identity_report("baseline"), "reid_open_set_occupied_v1", "baseline"),
        (baseline_semantic_report, "semantic_gate_v3", "baseline"),
        (depth_quality, "mapanything_depth_quality_v4", "baseline"),
        (world, "v3dt_world_gate_v2", "v3dt"),
        (semantic_report, "semantic_gate_v3", "v3dt"),
        (_identity_report("v3dt"), "v3dt_identity_gate_v1", "v3dt"),
        (soak, "runtime_resource_soak_v2", "v3dt"),
        (wholebody_s_soak, "runtime_resource_soak_v2", "wholebody49-s"),
        (wholebody_x_soak, "runtime_resource_soak_v2", "wholebody49-x"),
    )
    for report, behavior_id, lane in reports:
        _assert_registered_report(report, behavior_id, lane=lane)

    inspect_started = ownership._parse_rfc3339_utc(
        "2026-07-11T05:59:59Z", "fixture inspect start"
    )
    inspect_finished = ownership._parse_rfc3339_utc(
        "2026-07-11T06:05:02Z", "fixture inspect finish"
    )

    def encoded(report: Mapping[str, Any]) -> bytes:
        return (
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")

    def compact_reseal(raw: bytes) -> bytes:
        resealed = (
            json.dumps(
                json.loads(raw),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        assert resealed != raw
        return resealed

    def assert_report_reseal_rejected(
        *,
        behavior_ids: list[str],
        covered: Mapping[str, bytes],
        capability_id: str,
        lane: str,
        report_filename: str,
    ) -> None:
        probe = dict(covered)
        probe[report_filename] = compact_reseal(probe[report_filename])
        with pytest.raises(ValueError, match="report bytes differ from the producer encoding"):
            ownership._validate_behavior_documents(
                behavior_ids,
                covered=probe,
                capability_id=capability_id,
                lane=lane,
                **common,
            )

    def assert_source_reseal_rejected(
        *,
        behavior_id: str,
        behavior_ids: list[str],
        covered: Mapping[str, bytes],
        capability_id: str,
        lane: str,
        report_filename: str,
        source_filename: str,
        evidence_field: str = "source_evidence",
    ) -> None:
        probe = dict(covered)
        resealed_source = compact_reseal(probe[source_filename])
        report = json.loads(probe[report_filename])
        report[evidence_field]["sha256"] = ownership._sha256_bytes(resealed_source)
        probe[source_filename] = resealed_source
        probe[report_filename] = ownership._canonical_behavior_report_bytes(
            behavior_id, report
        )
        with pytest.raises(ValueError, match="bytes differ from the producer encoding"):
            ownership._validate_behavior_documents(
                behavior_ids,
                covered=probe,
                capability_id=capability_id,
                lane=lane,
                **common,
            )

    common = {
        "label": "authentic producer integration",
        "launcher_dir": world_launcher,
        "session_id": SESSION_ID,
        "runtime_instance_id": "runtime-instance-test",
        "runtime_run_id": "runtime-run-test",
        "inspect_started": inspect_started,
        "inspect_finished": inspect_finished,
        "expected_identity_evidence_path": Path(
            "/private/session/runtime/identity_v2.jsonl"
        ),
        "artifact_binding": {
            "realization_sha256": "b" * 64,
            "output_sha256": {
                ownership.REID_ARTIFACT_ID: "a" * 64,
                "engine.v3dt_tracker_reid": "c" * 64,
                "engine.wholebody49_s_masks": "d" * 64,
                "engine.wholebody49_x_boxes": "e" * 64,
            },
        },
    }
    runtime_authorities = {
        "launch-plan.json": launch_plan_raw,
        "checkout-before.json": encoded({"sha256": "a" * 64}),
        "container-inspect.json": encoded([{"Id": container_id}]),
    }

    def assert_behavior_rejected(
        behavior_id: str,
        document: Mapping[str, Any],
        *,
        lane: str,
        covered: Mapping[str, bytes],
        probe: str,
        expected_session_id: str = SESSION_ID,
    ) -> None:
        try:
            ownership._validate_behavior_structure(
                behavior_id,
                document,
                covered=covered,
                launcher_dir=world_launcher,
                artifact_binding=common["artifact_binding"],
                label=f"{probe} adversarial probe",
                session_id=expected_session_id,
                lane=lane,
                runtime_instance_id="runtime-instance-test",
                runtime_run_id="runtime-run-test",
                inspect_started=inspect_started,
                inspect_finished=inspect_finished,
                expected_identity_evidence_path=common[
                    "expected_identity_evidence_path"
                ],
            )
        except ValueError:
            return
        pytest.fail(f"{probe} adversarial evidence was unexpectedly accepted")

    wholebody_s_covered = {
        **runtime_authorities,
        "wholebody49-occupied-scene.json": encoded(wholebody_s),
        wholebody.CANONICAL_SOURCE_TRANSCRIPT_FILENAME: wholebody_s_source,
        "wholebody49-media-decode.json": encoded(wholebody_s_media),
        wholebody_media.CANONICAL_SOURCE_FILENAME: wholebody_s_media_source,
        resource.RESOURCE_SOAK_REPORT_FILENAME: encoded(wholebody_s_soak),
        resource.RESOURCE_SOAK_SAMPLES_FILENAME: wholebody_s_soak_raw,
    }
    ownership._validate_behavior_documents(
        sorted(
            (
                "runtime_resource_soak_v2",
                "wholebody49_media_decode_v1",
                "wholebody49_occupied_s_v2",
            )
        ),
        covered=wholebody_s_covered,
        capability_id="model.wholebody49_profile",
        lane="wholebody49-s",
        **common,
    )
    for probe, mutate in (
        (
            "wholebody mask counter/check mismatch",
            lambda row: row["counters"].__setitem__(wholebody.MASK_COUNTER, 0),
        ),
        (
            "wholebody occupied counter/check mismatch",
            lambda row: row.__setitem__("tracks_seen", 0),
        ),
        (
            "wholebody render policy drift",
            lambda row: row.__setitem__("render_evidence_policy", "subjective"),
        ),
    ):
        wholebody_probe = copy.deepcopy(wholebody_s)
        mutate(wholebody_probe)
        assert_behavior_rejected(
            "wholebody49_occupied_s_v2",
            wholebody_probe,
            lane="wholebody49-s",
            covered={
                wholebody.CANONICAL_SOURCE_TRANSCRIPT_FILENAME: wholebody_s_source
            },
            probe=probe,
        )
    wholebody_relabel = copy.deepcopy(wholebody_s)
    wholebody_relabel["session_id"] = "other-session"
    assert_behavior_rejected(
        "wholebody49_occupied_s_v2",
        wholebody_relabel,
        lane="wholebody49-s",
        covered={
            wholebody.CANONICAL_SOURCE_TRANSCRIPT_FILENAME: wholebody_s_source
        },
        probe="wholebody report-only session relabel",
        expected_session_id="other-session",
    )
    media_relabel = copy.deepcopy(wholebody_s_media)
    media_relabel["session_id"] = "other-session"
    assert_behavior_rejected(
        "wholebody49_media_decode_v1",
        media_relabel,
        lane="wholebody49-s",
        covered={
            wholebody_media.CANONICAL_SOURCE_FILENAME: wholebody_s_media_source
        },
        probe="decoded-media report-only session relabel",
        expected_session_id="other-session",
    )
    media_source_probe = json.loads(wholebody_s_media_source)
    media_source_probe["sample"]["decoded_frames"] = 0
    media_source_probe_raw = wholebody_media._encoded(media_source_probe)
    media_report_probe = copy.deepcopy(wholebody_s_media)
    media_report_probe["source_evidence"] = wholebody_media._source_evidence(
        media_source_probe_raw,
        media_source_probe,
    )
    assert_behavior_rejected(
        "wholebody49_media_decode_v1",
        media_report_probe,
        lane="wholebody49-s",
        covered={
            wholebody_media.CANONICAL_SOURCE_FILENAME: media_source_probe_raw
        },
        probe="decoded-media source/report drift",
    )

    envelope_only = {
        "schema_version": 1,
        "contract": ownership.BEHAVIOR_CONTRACTS["v3dt_world_gate_v2"]["contract"],
        "contract_version": 1,
        "session_id": SESSION_ID,
        "runtime_lane": "v3dt",
        "runtime_instance_id": "runtime-instance-test",
        "runtime_run_id": "runtime-run-test",
        "ok": True,
        "status": "pass",
        "occupied_scene_observed": True,
        "locked_world_contract": {
            "world_frame": "camera_local",
            "global_world_promotion": "not_claimed",
        },
        "tracks_seen": 1,
    }
    with pytest.raises(ValueError, match="producer schema"):
        ownership._validate_behavior_structure(
            "v3dt_world_gate_v2",
            envelope_only,
            covered={},
            artifact_binding=common["artifact_binding"],
            label="generic envelope probe",
            session_id=SESSION_ID,
            lane="v3dt",
            runtime_instance_id="runtime-instance-test",
            runtime_run_id="runtime-run-test",
            inspect_started=inspect_started,
            inspect_finished=inspect_finished,
            expected_identity_evidence_path=common[
                "expected_identity_evidence_path"
            ],
        )

    extra_field_probe = copy.deepcopy(world)
    extra_field_probe["unregistered_claim"] = True
    with pytest.raises(ValueError, match="producer schema"):
        ownership._validate_behavior_structure(
            "v3dt_world_gate_v2",
            extra_field_probe,
            covered={},
            artifact_binding=common["artifact_binding"],
            label="extra-field producer probe",
            session_id=SESSION_ID,
            lane="v3dt",
            runtime_instance_id="runtime-instance-test",
            runtime_run_id="runtime-run-test",
            inspect_started=inspect_started,
            inspect_finished=inspect_finished,
            expected_identity_evidence_path=common[
                "expected_identity_evidence_path"
            ],
        )

    authority_probe = copy.deepcopy(_identity_report("baseline"))
    authority_probe["claims"]["semantic_accuracy"]["status"] = "pass"
    with pytest.raises(ValueError, match="accuracy/authority"):
        ownership._validate_behavior_structure(
            "reid_open_set_occupied_v1",
            authority_probe,
            covered={},
            artifact_binding=common["artifact_binding"],
            label="identity authority probe",
            session_id=SESSION_ID,
            lane="baseline",
            runtime_instance_id="runtime-instance-test",
            runtime_run_id="runtime-run-test",
            inspect_started=inspect_started,
            inspect_finished=inspect_finished,
            expected_identity_evidence_path=common[
                "expected_identity_evidence_path"
            ],
        )

    wholebody_x_covered = {
        **runtime_authorities,
        "wholebody49-occupied-scene.json": encoded(wholebody_x),
        wholebody.CANONICAL_SOURCE_TRANSCRIPT_FILENAME: wholebody_x_source,
        "wholebody49-media-decode.json": encoded(wholebody_x_media),
        wholebody_media.CANONICAL_SOURCE_FILENAME: wholebody_x_media_source,
        resource.RESOURCE_SOAK_REPORT_FILENAME: encoded(wholebody_x_soak),
        resource.RESOURCE_SOAK_SAMPLES_FILENAME: wholebody_x_soak_raw,
    }
    ownership._validate_behavior_documents(
        sorted(
            (
                "runtime_resource_soak_v2",
                "wholebody49_media_decode_v1",
                "wholebody49_occupied_x_v2",
            )
        ),
        covered=wholebody_x_covered,
        capability_id="model.wholebody49_profile",
        lane="wholebody49-x",
        **common,
    )
    wholebody_x_probe = copy.deepcopy(wholebody_x)
    wholebody_x_probe["counters"][wholebody.BBOX_COUNTER] = 0
    assert_behavior_rejected(
        "wholebody49_occupied_x_v2",
        wholebody_x_probe,
        lane="wholebody49-x",
        covered={
            wholebody.CANONICAL_SOURCE_TRANSCRIPT_FILENAME: wholebody_x_source
        },
        probe="wholebody bbox counter/check mismatch",
    )
    baseline_identity, baseline_identity_source = _identity_bundle("baseline")
    assert ownership.BEHAVIOR_CONTRACTS["semantic_gate_v3"]["lanes"] == {
        "baseline",
        "v3dt",
    }
    assert ownership.CAPABILITY_REGISTRY["model.reid_profile"][
        "runtime_requirements"
    ]["baseline"] == ["reid_open_set_occupied_v1", "semantic_gate_v3"]
    baseline_covered = {
        "identity-open-set-occupied.json": encoded(baseline_identity),
        identity.BASELINE_SOURCE_TRANSCRIPT_FILENAME: baseline_identity_source,
        "semantic-observation.json": encoded(baseline_semantic_report),
        semantic.CANONICAL_IDENTITY_SNAPSHOT_FILENAME: semantic_snapshot.payload,
        semantic.CANONICAL_SOURCE_TRANSCRIPT_FILENAME:
            baseline_semantic_source_encoded,
    }
    ownership._validate_behavior_documents(
        ["reid_open_set_occupied_v1", "semantic_gate_v3"],
        covered=baseline_covered,
        capability_id="model.reid_profile",
        lane="baseline",
        **common,
    )
    for probe, path, value in (
        ("identity tracking count", ("counts", "tracking_messages"), 0),
        ("identity fresh evidence count", ("counts", "fresh_embedding_rows"), 1),
        (
            "identity continuity claim count",
            ("claims", "tracker_subject_continuity", "evidence_count"),
            99,
        ),
        (
            "identity cross-camera status",
            ("claims", "cross_camera_assignment_continuity", "status"),
            "observed",
        ),
        (
            "identity post-gate cache occupancy",
            ("health_after", "observation_cache_entries"),
            0,
        ),
        (
            "identity realized model binding",
            ("health_after", "runtime_model_fingerprint"),
            "b" * 64,
        ),
    ):
        identity_probe = copy.deepcopy(baseline_identity)
        target: Any = identity_probe
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        assert_behavior_rejected(
            "reid_open_set_occupied_v1",
            identity_probe,
            lane="baseline",
            covered={
                identity.BASELINE_SOURCE_TRANSCRIPT_FILENAME:
                    baseline_identity_source,
            },
            probe=probe,
        )

    identity_source_probe = json.loads(baseline_identity_source)
    identity_source_probe["session_id"] = "other-session"
    identity_source_probe_raw = identity._encoded_private_json(identity_source_probe)
    identity_report_probe = copy.deepcopy(baseline_identity)
    identity_report_probe["source_evidence"]["sha256"] = ownership._sha256_bytes(
        identity_source_probe_raw
    )
    assert_behavior_rejected(
        "reid_open_set_occupied_v1",
        identity_report_probe,
        lane="baseline",
        covered={
            identity.BASELINE_SOURCE_TRANSCRIPT_FILENAME: identity_source_probe_raw,
        },
        probe="identity transcript session splice",
    )
    floorplan_covered = {
        "mapanything-depth-quality.json": encoded(depth_quality),
        floorplan.CANONICAL_SOURCE_TRANSCRIPT_FILENAME: floorplan_source_encoded,
    }
    ownership._validate_behavior_documents(
        ["mapanything_depth_quality_v4"],
        covered=floorplan_covered,
        capability_id="model.mapanything_validated_fp32_builder",
        lane="baseline",
        **common,
    )
    for probe, mutate in (
        (
            "floorplan canonical freshness bound",
            lambda row: row.__setitem__("max_snapshot_age_s", 121.0),
        ),
        (
            "floorplan stale camera",
            lambda row: row["cameras"][0].__setitem__("snapshot_age_s", 120.1),
        ),
        (
            "floorplan camera inventory",
            lambda row: row["cameras"][0].__setitem__("camera_id", "garage"),
        ),
        (
            "floorplan calibration binding",
            lambda row: row["cameras"][0].__setitem__(
                "calibration_fingerprint", "f" * 64
            ),
        ),
        (
            "floorplan camera count relationship",
            lambda row: row.__setitem__("validated_camera_count", 2),
        ),
        (
            "floorplan v1 downgrade",
            lambda row: (
                row.__setitem__("schema_version", 1),
                row.__setitem__("contract_version", 1),
            ),
        ),
        (
            "floorplan cache-only claim",
            lambda row: row.__setitem__("cache_only_zero_mutation", False),
        ),
        (
            "floorplan RGB substitution",
            lambda row: row["cameras"][0].__setitem__(
                "capture_event_rgb_status", "invalid"
            ),
        ),
        (
            "floorplan cache identity substitution",
            lambda row: row["cache_only_cameras"][0].__setitem__(
                "snapshot_id", "substituted"
            ),
        ),
        (
            "floorplan controller mutation",
            lambda row: row["runtime_health"]["after_cache_only"][
                "capture_event_controller_health"
            ]["counters"].__setitem__("requests_total", 99),
        ),
    ):
        floorplan_probe = copy.deepcopy(depth_quality)
        mutate(floorplan_probe)
        assert_behavior_rejected(
            "mapanything_depth_quality_v4",
            floorplan_probe,
            lane="baseline",
            covered={
                floorplan.CANONICAL_SOURCE_TRANSCRIPT_FILENAME:
                    floorplan_source_encoded
            },
            probe=probe,
        )
    floorplan_relabel = copy.deepcopy(depth_quality)
    floorplan_relabel["session_id"] = "other-session"
    assert_behavior_rejected(
        "mapanything_depth_quality_v4",
        floorplan_relabel,
        lane="baseline",
        covered={
            floorplan.CANONICAL_SOURCE_TRANSCRIPT_FILENAME:
                floorplan_source_encoded
        },
        probe="floorplan report-only session relabel",
        expected_session_id="other-session",
    )
    v3dt_covered = {
        **runtime_authorities,
        "v3dt-world-contract.json": encoded(world),
        v3dt_world.CANONICAL_SOURCE_TRANSCRIPT_FILENAME: world_source_encoded,
        "semantic-observation.json": encoded(semantic_report),
        semantic.CANONICAL_IDENTITY_SNAPSHOT_FILENAME: semantic_snapshot.payload,
        semantic.CANONICAL_SOURCE_TRANSCRIPT_FILENAME: semantic_source_encoded,
        "v3dt-identity-open-set-occupied.json": encoded(_identity_report("v3dt")),
        identity.V3DT_SOURCE_TRANSCRIPT_FILENAME: _identity_bundle("v3dt")[1],
        resource.RESOURCE_SOAK_REPORT_FILENAME: encoded(soak),
        resource.RESOURCE_SOAK_SAMPLES_FILENAME: soak_samples_raw,
    }
    for probe, path, value in (
        ("V3DT camera/track count", ("cameras", "kitchen"), 1),
        ("V3DT bbox count relationship", ("bbox3d_valid",), 1),
        ("V3DT fatal world counter", ("invalid_world",), 1),
        (
            "V3DT advancing tracker proof",
            ("advancing_trackers_by_camera", "kitchen"),
            0,
        ),
        (
            "V3DT minimum coverage policy",
            ("policy", "min_bbox3d_coverage"),
            0.9,
        ),
        (
            "V3DT axis characterization count",
            ("public_world_axis_error_m", "count"),
            1,
        ),
        (
            "V3DT old tuple rejection policy",
            ("old_raw_tuple_rejection", "policy"),
            "informational",
        ),
    ):
        world_probe = copy.deepcopy(world)
        target = world_probe
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        assert_behavior_rejected(
            "v3dt_world_gate_v2",
            world_probe,
            lane="v3dt",
            covered={
                v3dt_world.CANONICAL_SOURCE_TRANSCRIPT_FILENAME:
                    world_source_encoded
            },
            probe=probe,
        )
    world_relabel = copy.deepcopy(world)
    world_relabel["session_id"] = "other-session"
    assert_behavior_rejected(
        "v3dt_world_gate_v2",
        world_relabel,
        lane="v3dt",
        covered={
            v3dt_world.CANONICAL_SOURCE_TRANSCRIPT_FILENAME: world_source_encoded
        },
        probe="V3DT world report-only session relabel",
        expected_session_id="other-session",
    )

    v1_source = json.loads(world_source_encoded)
    v1_source["schema_version"] = 1
    v1_source["contract_version"] = 1
    v1_source_raw = v3dt_world._encoded_private_json(v1_source)
    v1_source_report = copy.deepcopy(world)
    v1_source_report["source_evidence"]["sha256"] = ownership._sha256_bytes(
        v1_source_raw
    )
    assert_behavior_rejected(
        "v3dt_world_gate_v2",
        v1_source_report,
        lane="v3dt",
        covered={
            "launch-plan.json": launch_plan_raw,
            v3dt_world.CANONICAL_SOURCE_TRANSCRIPT_FILENAME: v1_source_raw,
        },
        probe="V3DT v1 source compatibility rejection",
    )

    v1_report = copy.deepcopy(world)
    v1_report["schema_version"] = 1
    v1_report["contract"] = "noesis.ds9.v3dt_camera_local_live_gate"
    v1_report["contract_version"] = 1
    v1_covered = dict(v3dt_covered)
    v1_covered["v3dt-world-contract.json"] = encoded(v1_report)
    with pytest.raises(ValueError, match="schema/contract/session/lane binding drift"):
        ownership._validate_behavior_documents(
            sorted(
                (
                    "v3dt_world_gate_v2",
                    "semantic_gate_v3",
                    "v3dt_identity_gate_v1",
                    "runtime_resource_soak_v2",
                )
            ),
            covered=v1_covered,
            capability_id="tracking.v3dt",
            lane="v3dt",
            **common,
        )

    for probe, path, value in (
        (
            "semantic required check",
            ("checks", "persisted_embedding_anchor"),
            False,
        ),
        (
            "semantic persisted/public count relationship",
            ("counts", "persisted_anchors"),
            2,
        ),
        (
            "semantic identity snapshot source path",
            ("evidence_snapshot", "source_path"),
            "/private/foreign/identity_v2.jsonl",
        ),
        (
            "semantic realized model binding",
            ("evidence_snapshot", "model_sha256"),
            "b" * 64,
        ),
        (
            "semantic sample sequence binding",
            ("sample_cohort", "anchor", "embedding_sequence"),
            999,
        ),
        (
            "semantic sample runtime binding",
            ("sample_cohort", "run_id"),
            "other-runtime-run",
        ),
        (
            "semantic temporal upper endpoint",
            ("sample_cohort", "span_us"),
            1_500_001,
        ),
        (
            "semantic fingerprint binding",
            ("sample_cohort", "fingerprints", "config_sha256"),
            "f" * 64,
        ),
        (
            "semantic immutable identity binding",
            ("sample_cohort", "identity", "compatibility_sid"),
            2,
        ),
    ):
        semantic_probe = copy.deepcopy(semantic_report)
        target = semantic_probe
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        assert_behavior_rejected(
            "semantic_gate_v3",
            semantic_probe,
            lane="v3dt",
            covered=v3dt_covered,
            probe=probe,
        )

    semantic_source_probe = json.loads(semantic_source_encoded)
    semantic_source_probe["runtime_run_id"] = "other-runtime-run"
    semantic_source_probe_raw = semantic._encoded_source_transcript(
        semantic_source_probe
    )
    semantic_report_probe = copy.deepcopy(semantic_report)
    semantic_report_probe["source_evidence"]["sha256"] = ownership._sha256_bytes(
        semantic_source_probe_raw
    )
    semantic_covered_probe = dict(v3dt_covered)
    semantic_covered_probe[
        semantic.CANONICAL_SOURCE_TRANSCRIPT_FILENAME
    ] = semantic_source_probe_raw
    assert_behavior_rejected(
        "semantic_gate_v3",
        semantic_report_probe,
        lane="v3dt",
        covered=semantic_covered_probe,
        probe="semantic transcript runtime splice",
    )

    semantic_duplicate_source_raw = semantic_source_encoded.replace(
        b'"runtime_run_id": "runtime-run-test",',
        (
            b'"runtime_run_id": "runtime-run-test",\n'
            b'  "runtime_run_id": "runtime-run-test",'
        ),
        1,
    )
    assert semantic_duplicate_source_raw != semantic_source_encoded
    semantic_duplicate_report = copy.deepcopy(semantic_report)
    semantic_duplicate_report["source_evidence"]["sha256"] = (
        ownership._sha256_bytes(semantic_duplicate_source_raw)
    )
    semantic_duplicate_covered = dict(v3dt_covered)
    semantic_duplicate_covered[
        semantic.CANONICAL_SOURCE_TRANSCRIPT_FILENAME
    ] = semantic_duplicate_source_raw
    assert_behavior_rejected(
        "semantic_gate_v3",
        semantic_duplicate_report,
        lane="v3dt",
        covered=semantic_duplicate_covered,
        probe="semantic duplicate-key ownership transcript",
    )

    semantic_source_probe = json.loads(semantic_source_encoded)
    semantic_source_probe["acquisition_window"]["started_at_us"] = (
        int(inspect_started.timestamp() * 1_000_000) - 1
    )
    semantic_source_probe_raw = semantic._encoded_source_transcript(
        semantic_source_probe
    )
    semantic_report_probe = copy.deepcopy(semantic_report)
    semantic_report_probe["source_evidence"].update(
        {
            "sha256": ownership._sha256_bytes(semantic_source_probe_raw),
            "acquisition_started_at_us": semantic_source_probe[
                "acquisition_window"
            ]["started_at_us"],
        }
    )
    semantic_covered_probe = dict(v3dt_covered)
    semantic_covered_probe[
        semantic.CANONICAL_SOURCE_TRANSCRIPT_FILENAME
    ] = semantic_source_probe_raw
    assert_behavior_rejected(
        "semantic_gate_v3",
        semantic_report_probe,
        lane="v3dt",
        covered=semantic_covered_probe,
        probe="semantic acquisition starts before inspected runtime",
    )

    semantic_source_probe = json.loads(semantic_source_encoded)
    semantic_source_probe["acquisition_window"][
        "capture_pre_window_leeway_us"
    ] += 1
    semantic_source_probe_raw = semantic._encoded_source_transcript(
        semantic_source_probe
    )
    semantic_report_probe = copy.deepcopy(semantic_report)
    semantic_report_probe["source_evidence"]["sha256"] = ownership._sha256_bytes(
        semantic_source_probe_raw
    )
    semantic_covered_probe = dict(v3dt_covered)
    semantic_covered_probe[
        semantic.CANONICAL_SOURCE_TRANSCRIPT_FILENAME
    ] = semantic_source_probe_raw
    assert_behavior_rejected(
        "semantic_gate_v3",
        semantic_report_probe,
        lane="v3dt",
        covered=semantic_covered_probe,
        probe="semantic acquisition leeway policy tamper",
    )

    semantic_snapshot_row = json.loads(semantic_snapshot.payload)
    semantic_snapshot_row["session_id"] = "other-session"
    semantic_snapshot_body = dict(semantic_snapshot_row)
    semantic_snapshot_body.pop("event_id")
    semantic_snapshot_row["event_id"] = hashlib.sha256(
        b"noesis-identity-shadow-evidence-v2\0"
        + semantic._canonical_bytes(semantic_snapshot_body)
    ).hexdigest()
    semantic_snapshot_probe_raw = (
        json.dumps(semantic_snapshot_row, sort_keys=True) + "\n"
    ).encode()
    semantic_report_probe = copy.deepcopy(semantic_report)
    semantic_report_probe["evidence_snapshot"]["bytes"] = len(
        semantic_snapshot_probe_raw
    )
    semantic_report_probe["evidence_snapshot"]["sha256"] = (
        ownership._sha256_bytes(semantic_snapshot_probe_raw)
    )
    semantic_report_probe["sample_cohort"]["anchor"]["event_id"] = (
        semantic_snapshot_row["event_id"]
    )
    semantic_covered_probe = dict(v3dt_covered)
    semantic_covered_probe[
        semantic.CANONICAL_IDENTITY_SNAPSHOT_FILENAME
    ] = semantic_snapshot_probe_raw
    assert_behavior_rejected(
        "semantic_gate_v3",
        semantic_report_probe,
        lane="v3dt",
        covered=semantic_covered_probe,
        probe="semantic identity snapshot session splice",
    )

    resource_relabel = copy.deepcopy(soak)
    resource_relabel["session_id"] = "other-session"
    assert_behavior_rejected(
        "runtime_resource_soak_v2",
        resource_relabel,
        lane="v3dt",
        covered=v3dt_covered,
        probe="resource report-only session relabel",
        expected_session_id="other-session",
    )
    legacy_v3dt_soak = copy.deepcopy(soak)
    legacy_v3dt_soak["schema_version"] = 1
    legacy_v3dt_soak["contract"] = "noesis.ds9.v3dt_resource_soak"
    legacy_v3dt_soak["contract_version"] = 1
    assert_behavior_rejected(
        "runtime_resource_soak_v2",
        legacy_v3dt_soak,
        lane="v3dt",
        covered=v3dt_covered,
        probe="legacy V3DT-only soak contract",
    )
    for mutation in (
        "sample_hash",
        "sample_count",
        "raw_sample_count",
        "sample_session",
        "sample_lane",
        "sample_runtime_instance",
        "sample_runtime_run",
        "sample_window",
        "source_binding",
        "coordinated_binding_relabel",
        "metric",
        "report_binding",
        "report_start_window",
        "report_finish_window",
    ):
        report_probe = copy.deepcopy(soak)
        covered_probe = dict(v3dt_covered)
        if mutation == "sample_hash":
            report_probe["samples_evidence"]["sha256"] = "f" * 64
        elif mutation == "sample_count":
            report_probe["samples_evidence"]["sample_count"] += 1
        elif mutation == "metric":
            report_probe["metrics"]["observed_duration_seconds"] += 5.0
        elif mutation == "report_binding":
            report_probe["runtime_binding"]["primary_engine_sha256"] = "f" * 64
        elif mutation == "report_start_window":
            report_probe["started_at_utc"] = "2026-07-11T05:59:58Z"
        elif mutation == "report_finish_window":
            report_probe["finished_at_utc"] = "2026-07-11T06:05:03Z"
        else:
            samples_probe = json.loads(
                covered_probe[resource.RESOURCE_SOAK_SAMPLES_FILENAME]
            )
            if mutation == "sample_session":
                samples_probe["session_id"] = "other-session"
            elif mutation == "raw_sample_count":
                samples_probe["sample_count"] += 1
            elif mutation == "sample_lane":
                samples_probe["runtime_lane"] = "baseline"
            elif mutation == "sample_runtime_instance":
                samples_probe["runtime_instance_id"] = "other-runtime-instance"
            elif mutation == "sample_runtime_run":
                samples_probe["runtime_run_id"] = "other-runtime-run"
            elif mutation in {"source_binding", "coordinated_binding_relabel"}:
                samples_probe["runtime_binding"]["primary_engine_sha256"] = (
                    "f" * 64
                )
                if mutation == "coordinated_binding_relabel":
                    report_probe["runtime_binding"]["primary_engine_sha256"] = (
                        "f" * 64
                    )
            else:
                samples_probe["samples"][0]["captured_at_utc"] = (
                    "2026-07-11T05:59:58Z"
                )
            raw_probe = (json.dumps(samples_probe, sort_keys=True) + "\n").encode()
            covered_probe[resource.RESOURCE_SOAK_SAMPLES_FILENAME] = raw_probe
            report_probe["samples_evidence"]["sha256"] = ownership._sha256_bytes(
                raw_probe
            )
        with pytest.raises(ValueError):
            ownership._validate_behavior_structure(
                "runtime_resource_soak_v2",
                report_probe,
                covered=covered_probe,
                artifact_binding=common["artifact_binding"],
                label=f"resource {mutation} probe",
                session_id=SESSION_ID,
                lane="v3dt",
                runtime_instance_id="runtime-instance-test",
                runtime_run_id="runtime-run-test",
                inspect_started=inspect_started,
                inspect_finished=inspect_finished,
                expected_identity_evidence_path=common[
                    "expected_identity_evidence_path"
                ],
            )
    ownership._validate_behavior_documents(
        sorted(
            (
                "v3dt_world_gate_v2", "semantic_gate_v3",
                "v3dt_identity_gate_v1", "runtime_resource_soak_v2",
            )
        ),
        covered=v3dt_covered,
        capability_id="tracking.v3dt",
        lane="v3dt",
        **common,
    )

    wholebody_s_behaviors = sorted(
        [
            "runtime_resource_soak_v2",
            "wholebody49_media_decode_v1",
            "wholebody49_occupied_s_v2",
        ]
    )
    wholebody_x_behaviors = sorted(
        [
            "runtime_resource_soak_v2",
            "wholebody49_media_decode_v1",
            "wholebody49_occupied_x_v2",
        ]
    )
    baseline_behaviors = ["reid_open_set_occupied_v1", "semantic_gate_v3"]
    floorplan_behaviors = ["mapanything_depth_quality_v4"]
    v3dt_behaviors = sorted(
        [
            "v3dt_world_gate_v2",
            "semantic_gate_v3",
            "v3dt_identity_gate_v1",
            "runtime_resource_soak_v2",
        ]
    )
    for behavior_ids, covered, capability_id, lane, report_filename in (
        (
            wholebody_s_behaviors,
            wholebody_s_covered,
            "model.wholebody49_profile",
            "wholebody49-s",
            wholebody.CANONICAL_REPORT_FILENAME,
        ),
        (
            wholebody_s_behaviors,
            wholebody_s_covered,
            "model.wholebody49_profile",
            "wholebody49-s",
            wholebody_media.CANONICAL_REPORT_FILENAME,
        ),
        (
            wholebody_s_behaviors,
            wholebody_s_covered,
            "model.wholebody49_profile",
            "wholebody49-s",
            resource.RESOURCE_SOAK_REPORT_FILENAME,
        ),
        (
            wholebody_x_behaviors,
            wholebody_x_covered,
            "model.wholebody49_profile",
            "wholebody49-x",
            wholebody.CANONICAL_REPORT_FILENAME,
        ),
        (
            baseline_behaviors,
            baseline_covered,
            "model.reid_profile",
            "baseline",
            identity.BASELINE_CANONICAL_REPORT_FILENAME,
        ),
        (
            baseline_behaviors,
            baseline_covered,
            "model.reid_profile",
            "baseline",
            semantic.CANONICAL_REPORT_FILENAME,
        ),
        (
            floorplan_behaviors,
            floorplan_covered,
            "model.mapanything_validated_fp32_builder",
            "baseline",
            floorplan.CANONICAL_REPORT_FILENAME,
        ),
        (
            v3dt_behaviors,
            v3dt_covered,
            "tracking.v3dt",
            "v3dt",
            v3dt_world.CANONICAL_REPORT_FILENAME,
        ),
        (
            v3dt_behaviors,
            v3dt_covered,
            "tracking.v3dt",
            "v3dt",
            identity.V3DT_CANONICAL_REPORT_FILENAME,
        ),
    ):
        assert_report_reseal_rejected(
            behavior_ids=behavior_ids,
            covered=covered,
            capability_id=capability_id,
            lane=lane,
            report_filename=report_filename,
        )

    for source_probe in (
        {
            "behavior_id": "wholebody49_occupied_s_v2",
            "behavior_ids": wholebody_s_behaviors,
            "covered": wholebody_s_covered,
            "capability_id": "model.wholebody49_profile",
            "lane": "wholebody49-s",
            "report_filename": wholebody.CANONICAL_REPORT_FILENAME,
            "source_filename": wholebody.CANONICAL_SOURCE_TRANSCRIPT_FILENAME,
        },
        {
            "behavior_id": "wholebody49_occupied_x_v2",
            "behavior_ids": wholebody_x_behaviors,
            "covered": wholebody_x_covered,
            "capability_id": "model.wholebody49_profile",
            "lane": "wholebody49-x",
            "report_filename": wholebody.CANONICAL_REPORT_FILENAME,
            "source_filename": wholebody.CANONICAL_SOURCE_TRANSCRIPT_FILENAME,
        },
        {
            "behavior_id": "wholebody49_media_decode_v1",
            "behavior_ids": wholebody_s_behaviors,
            "covered": wholebody_s_covered,
            "capability_id": "model.wholebody49_profile",
            "lane": "wholebody49-s",
            "report_filename": wholebody_media.CANONICAL_REPORT_FILENAME,
            "source_filename": wholebody_media.CANONICAL_SOURCE_FILENAME,
        },
        {
            "behavior_id": "runtime_resource_soak_v2",
            "behavior_ids": wholebody_s_behaviors,
            "covered": wholebody_s_covered,
            "capability_id": "model.wholebody49_profile",
            "lane": "wholebody49-s",
            "report_filename": resource.RESOURCE_SOAK_REPORT_FILENAME,
            "source_filename": resource.RESOURCE_SOAK_SAMPLES_FILENAME,
            "evidence_field": "samples_evidence",
        },
        {
            "behavior_id": "reid_open_set_occupied_v1",
            "behavior_ids": baseline_behaviors,
            "covered": baseline_covered,
            "capability_id": "model.reid_profile",
            "lane": "baseline",
            "report_filename": identity.BASELINE_CANONICAL_REPORT_FILENAME,
            "source_filename": identity.BASELINE_SOURCE_TRANSCRIPT_FILENAME,
        },
        {
            "behavior_id": "semantic_gate_v3",
            "behavior_ids": baseline_behaviors,
            "covered": baseline_covered,
            "capability_id": "model.reid_profile",
            "lane": "baseline",
            "report_filename": semantic.CANONICAL_REPORT_FILENAME,
            "source_filename": semantic.CANONICAL_SOURCE_TRANSCRIPT_FILENAME,
        },
        {
            "behavior_id": "mapanything_depth_quality_v4",
            "behavior_ids": floorplan_behaviors,
            "covered": floorplan_covered,
            "capability_id": "model.mapanything_validated_fp32_builder",
            "lane": "baseline",
            "report_filename": floorplan.CANONICAL_REPORT_FILENAME,
            "source_filename": floorplan.CANONICAL_SOURCE_TRANSCRIPT_FILENAME,
        },
        {
            "behavior_id": "v3dt_world_gate_v2",
            "behavior_ids": v3dt_behaviors,
            "covered": v3dt_covered,
            "capability_id": "tracking.v3dt",
            "lane": "v3dt",
            "report_filename": v3dt_world.CANONICAL_REPORT_FILENAME,
            "source_filename": v3dt_world.CANONICAL_SOURCE_TRANSCRIPT_FILENAME,
        },
        {
            "behavior_id": "v3dt_identity_gate_v1",
            "behavior_ids": v3dt_behaviors,
            "covered": v3dt_covered,
            "capability_id": "tracking.v3dt",
            "lane": "v3dt",
            "report_filename": identity.V3DT_CANONICAL_REPORT_FILENAME,
            "source_filename": identity.V3DT_SOURCE_TRANSCRIPT_FILENAME,
        },
    ):
        assert_source_reseal_rejected(**source_probe)

    assert wholebody.CANONICAL_REPORT_FILENAME == ownership.BEHAVIOR_CONTRACTS[
        "wholebody49_occupied_s_v2"
    ]["filename"]
    assert floorplan.CANONICAL_REPORT_FILENAME == ownership.BEHAVIOR_CONTRACTS[
        "mapanything_depth_quality_v4"
    ]["filename"]
    assert identity.BASELINE_CANONICAL_REPORT_FILENAME == ownership.BEHAVIOR_CONTRACTS[
        "reid_open_set_occupied_v1"
    ]["filename"]
    assert identity.V3DT_CANONICAL_REPORT_FILENAME == ownership.BEHAVIOR_CONTRACTS[
        "v3dt_identity_gate_v1"
    ]["filename"]
    assert semantic.CANONICAL_REPORT_FILENAME == ownership.BEHAVIOR_CONTRACTS[
        "semantic_gate_v3"
    ]["filename"]
    assert v3dt_world.CANONICAL_REPORT_FILENAME == ownership.BEHAVIOR_CONTRACTS[
        "v3dt_world_gate_v2"
    ]["filename"]
    assert resource.RESOURCE_SOAK_REPORT_FILENAME == ownership.BEHAVIOR_CONTRACTS[
        "runtime_resource_soak_v2"
    ]["filename"]
