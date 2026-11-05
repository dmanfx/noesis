from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from noesis.server.alignment_walk_api import (
    AlignmentWalkApiError,
    AlignmentWalkController,
    install_alignment_walk_api,
)
from noesis.server.internal_auth import InternalAuthConfig
from noesis.validation.alignment_walk import payload_sha256


def _target_owned_fixture() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Published CalibrationManager shape, with nonidentity source and render edges."""
    from noesis_core.coordinate_frames import RevisionedFrame, revisioned_transform_sha256
    from noesis_core.contracts.scene_prior import (
        ScenePriorFrameRef, ScenePriorWorldToScenePresentation,
        world_to_scene_presentation_sha256, scene_revision_sha256,
    )
    source = RevisionedFrame("backend_world_m", "source-physical-v1")
    target = RevisionedFrame("backend_world_m", "target-metric-v1")
    edge = np.eye(4)
    edge[0, 3] = 10.0
    edge_values = edge.flatten(order="F").tolist()
    edge_sha = revisioned_transform_sha256(source, target, edge_values)
    render = np.diag([2.0, 2.0, 2.0, 1.0])
    render[0, 3] = 100.0
    render_values = render.flatten(order="F").tolist()
    frame = ScenePriorFrameRef(frame_id=target.frame_id, revision=target.revision)
    presentation = ScenePriorWorldToScenePresentation(
        contract="noesis.scene_prior.world_to_scene_presentation", contract_version=1,
        world_frame=frame, render_frame="menon_scene", world_to_scene_col_major=render_values,
        world_to_scene_sha256=world_to_scene_presentation_sha256(frame, "menon_scene", render_values),
        scene_revision_id=scene_revision_sha256(frame, "menon_scene", render_values),
        source_transform_sha256s=[edge_sha],
        provenance={"role": "world_to_scene_presentation", "sha256": "c" * 64},
        accepted=True, authored_scene_sha256="e" * 64, registration_sha256="d" * 64,
    ).model_dump(mode="json")
    refs = {"calibration_frame": {"frame_id": source.frame_id, "revision": source.revision},
            "world_frame": frame.model_dump(mode="json"), "transform_sha256": edge_sha}
    bundle = _calibration_message()["data"]
    bundle["cameras"]["frame_bindings"] = {"family-room": dict(refs,
        contract="noesis.calibration.frame_binding", contract_version=2,
        target_from_calibration_col_major=edge_values,
        calibration_floor_plane={"normal": [0.0, 1.0, 0.0], "offset_m": 0.0},
        world_floor_plane={"normal": [0.0, 1.0, 0.0], "offset_m": 0.0})}
    bundle["world_frame_presentations"] = {target.revision: presentation}
    binding = {"contract_version": 2, "release_id": "menon-target-v2", "authored_scene_sha256": "e" * 64,
        "targets": {target.revision: {k: presentation[k] for k in (
            "world_frame", "world_to_scene_sha256", "scene_revision_id", "registration_sha256")}},
        "cameras": {"family-room": refs}}
    release = {"authority": "noesis.scenes.current", "release_id": binding["release_id"],
               "authored_scene_sha256": "e" * 64, "release_sha256": "f" * 64}
    return json.loads(json.dumps(bundle)), json.loads(json.dumps(binding)), release


def test_target_owned_api_capture_analysis_trace(tmp_path: Path) -> None:
    bundle, binding, release = _target_owned_fixture()
    tracking = _tracking_message()
    tracking["tracks"][0].update(world=[10.0, 0.0, 2.0], world_valid=True,
        world_frame="backend_world_m", world_frame_revision="target-metric-v1",
        world_transform_sha256=binding["cameras"]["family-room"]["transform_sha256"],
        world_floor_candidate=[10.0, 0.0, 2.0], world_depth_candidate=[10.0, 0.0, 2.0],
        world_prefilter_measurement=[10.0, 0.0, 2.0], world_filter_prediction=[10.0, 0.0, 2.0])
    connector = _FakeConnector([{"type": "calibration-bundle", "data": bundle}, tracking])
    app = FastAPI()
    controller = install_alignment_walk_api(app, ws_host="127.0.0.1", ws_port=6008,
        auth_config=InternalAuthConfig(mode="disabled", token_file=None, token=None),
        output_root=tmp_path / "capture", websocket_connect=connector,
        scene_release_resolver=lambda: release)
    request = _create_payload("target-owned-session")
    request["duration_s"] = 60
    request["scene_binding"] = binding
    request["waypoints"]["waypoints"][0]["expected_scene_xyz"] = [120.0, 0.0, 4.0]
    request["waypoints"]["waypoints"].extend([
        {"id": "family-fit-2", "camera_id": "family-room", "expected_scene_xyz": [122.0, 0.0, 4.0], "split": "fit"},
        {"id": "family-fit-3", "camera_id": "family-room", "expected_scene_xyz": [120.0, 0.0, 6.0], "split": "fit"},
        {"id": "family-holdout", "camera_id": "family-room", "expected_scene_xyz": [122.0, 0.0, 6.0], "split": "holdout"},
    ])
    with TestClient(app) as client:
        created = client.post("/api/v1/alignment-walk/sessions", json=request)
        assert created.status_code == 201, created.text
        status_payload = _wait_for_sample(client, "target-owned-session")
        assert status_payload["scene_binding_verification"]["status"] == "matched"
        marker_responses = []
        for waypoint in request["waypoints"]["waypoints"]:
            for phase in ("arrived", "leave"):
                marker = client.post("/api/v1/alignment-walk/sessions/target-owned-session/markers",
                    json={"waypoint_id": waypoint["id"], "phase": phase, "tracklet_key": "tracklet-1"})
                assert marker.status_code == 200, marker.text
                marker_responses.append(marker.json())
        post_marker_status = client.get("/api/v1/alignment-walk/sessions/target-owned-session").json()
        assert len(post_marker_status["markers"]) == 8
        finished = client.post("/api/v1/alignment-walk/sessions/target-owned-session/finish")
        assert finished.status_code == 200, finished.text
        analyzed = client.post("/api/v1/alignment-walk/sessions/target-owned-session/analysis", json={})
        assert analyzed.status_code == 200, analyzed.text
        results = client.get("/api/v1/alignment-walk/sessions/target-owned-session/results")
        assert results.status_code == 200, results.text
        result = results.json()
        assert result["candidate_summary"]["advisory_only"] is True
        assert result["candidate_summary"]["active_config_modified"] is False
        assert result["candidate_similarity"]["status"] == "not_applicable"
        assert result["scene_binding"] == binding
        evidence = json.loads((tmp_path / "capture" / "target-owned-session" /
            "waypoint_calibration_evidence.json").read_text())
        for stage in evidence[0]["stages"]["active"].values():
            assert stage["error_m"] == pytest.approx(0.0)
        trace = {"fixture_kind": "synthetic published producer shape; actual API handlers; no live capture",
            "request": request, "calibration_bundle": bundle, "tracking_message": tracking,
            "created": created.json(), "verified_status": status_payload,
            "post_marker_status": post_marker_status, "marker_responses": marker_responses,
            "finished": finished.json(), "analysis": analyzed.json(), "results": result,
            "waypoint_evidence": evidence}
        path = tmp_path / "target-owned-api-trace.json"
        path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
        print(f"E03_TRACE={path}")
        release["release_sha256"] = "a" * 64
        stale = client.get("/api/v1/alignment-walk/sessions/target-owned-session/results")
        assert stale.status_code == 409
    controller.close()


@pytest.mark.parametrize("change", ["revision", "model", "digest", "edge", "missing", "source"])
def test_target_owned_published_authority_negative_controls(change: str) -> None:
    from noesis.validation.alignment_walk import _target_owned_context, validate_scene_binding
    bundle, binding, release = _target_owned_fixture()
    if change == "revision":
        bundle["cameras"]["frame_bindings"]["family-room"]["world_frame"]["revision"] = "stale"
    elif change == "model":
        release["authored_scene_sha256"] = "a" * 64
    elif change == "digest":
        bundle["world_frame_presentations"]["target-metric-v1"]["world_to_scene_sha256"] = "a" * 64
    elif change == "edge":
        bundle["cameras"]["frame_bindings"]["family-room"]["target_from_calibration_col_major"][12] += 1.0
    elif change == "source":
        bundle["world_frame_presentations"]["target-metric-v1"]["source_transform_sha256s"] = ["a" * 64]
    else:
        bundle.pop("world_frame_presentations")
    with pytest.raises((RuntimeError, ValueError)):
        _target_owned_context(bundle, validate_scene_binding(binding), release)


def test_target_owned_one_transform_and_explicit_schema() -> None:
    from noesis.validation.alignment_walk import _target_owned_context, validate_scene_binding
    from noesis.server.alignment_walk_api import TargetSceneBindingRequest
    bundle, binding, release = _target_owned_fixture()
    context = _target_owned_context(bundle, validate_scene_binding(binding), release)["family-room"]
    assert np.allclose(context["world_to_scene"] @ [10, 0, 2, 1], [120, 0, 4, 1])
    assert np.allclose(context["calibration_to_scene"] @ [0, 0, 2, 1], [120, 0, 4, 1])
    del binding["targets"]["target-metric-v1"]["registration_sha256"]
    with pytest.raises(ValueError):
        TargetSceneBindingRequest.model_validate(binding)
    with pytest.raises(ValueError, match="unsupported_scene_binding_schema"):
        validate_scene_binding({"contract_version": 3})


def test_target_owned_default_resolver_uses_canonical_current_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from noesis.server import scene_api
    _bundle, binding, _release = _target_owned_fixture()
    release = SimpleNamespace(release_id=binding["release_id"],
        authored_scene=SimpleNamespace(sha256=binding["authored_scene_sha256"]),
        model_dump=lambda **_kwargs: {"release_id": binding["release_id"], "authored_scene_sha256": "e" * 64})
    calls = []
    def snapshot():
        calls.append(True)
        return object(), release
    monkeypatch.setattr(scene_api, "_current_snapshot", snapshot)
    controller = AlignmentWalkController(ws_host="127.0.0.1", ws_port=6008,
        auth_config=InternalAuthConfig(mode="disabled", token_file=None, token=None), output_root=tmp_path)
    evidence = controller._verified_release(binding)
    assert calls == [True]
    assert evidence["authority"] == "noesis.scenes.current"
    assert evidence["release_sha256"] == payload_sha256(release.model_dump())
    release.authored_scene.sha256 = "a" * 64
    with pytest.raises(AlignmentWalkApiError, match="scene_binding_mismatch:"):
        controller._verified_release(binding)


def test_target_owned_capture_never_falls_back_and_rejects_stale_tracking(tmp_path: Path) -> None:
    from noesis.validation.alignment_walk import AlignmentWalkCapture, create_run_directory
    bundle, binding, release = _target_owned_fixture()
    capture = AlignmentWalkCapture(create_run_directory(tmp_path, "target"), run_id="target", ws_uri="ws://test.invalid", scene_binding=binding,
        scene_release_resolver=lambda: release)
    try:
        capture.process_message({"type": "calibration-bundle", "data": bundle})
        tracking = _tracking_message()
        tracking["tracks"][0].update(world_valid=True, world=[10, 0, 2],
            world_frame="backend_world_m", world_frame_revision="target-metric-v1",
            world_transform_sha256=binding["cameras"]["family-room"]["transform_sha256"])
        third_camera = dict(tracking["tracks"][0], camera_id="unselected-third-camera",
                            world_frame_revision="foreign-revision")
        tracking["tracks"].append(third_camera)
        rows = capture.process_message(tracking)
        assert len(rows) == 1 and rows[0]["camera_id"] == "family-room"
        assert capture.counts["unselected_camera_tracks_discarded"] == 1
        tracking["tracks"][0]["world_frame_revision"] = "stale"
        with pytest.raises(RuntimeError, match="tracking target revision/source edge"):
            capture.process_message(tracking)
        assert capture.counts.get("track_samples", 0) == 1
    finally:
        capture.finish()
    legacy = AlignmentWalkCapture(create_run_directory(tmp_path, "legacy"), run_id="legacy", ws_uri="ws://test.invalid", scene_binding=_create_payload()["scene_binding"])
    try:
        with pytest.raises(RuntimeError, match="unsupported_scene_binding_schema"):
            legacy.process_message({"type": "calibration-bundle", "data": bundle})
    finally:
        legacy.finish()


@pytest.mark.parametrize("stage", ["world_floor_candidate", "world_depth_candidate",
                                  "world_prefilter_measurement", "world_filter_prediction"])
@pytest.mark.parametrize("provenance", ["stale", "missing"])
def test_target_owned_invalid_final_stage_requires_exact_provenance(
    tmp_path: Path, stage: str, provenance: str,
) -> None:
    from noesis.validation.alignment_walk import AlignmentWalkCapture, create_run_directory
    bundle, binding, release = _target_owned_fixture()
    capture = AlignmentWalkCapture(create_run_directory(tmp_path, "invalid-final"),
        run_id="invalid-final", ws_uri="ws://test.invalid", scene_binding=binding,
        scene_release_resolver=lambda: release)
    try:
        capture.process_message({"type": "calibration-bundle", "data": bundle})
        tracking = _tracking_message()
        track = tracking["tracks"][0]
        track.update(world_valid=False, world_frame="backend_world_m",
            world_frame_revision="target-metric-v1",
            world_transform_sha256=binding["cameras"]["family-room"]["transform_sha256"])
        track[stage] = [10.0, 0.0, 2.0]
        # Invalid final admission does not invalidate correctly bound diagnostics.
        accepted = capture.process_message(tracking)
        assert len(accepted) == 1 and accepted[0]["world_valid"] is False
        assert any(accepted[0].get(field) == [10.0, 0.0, 2.0] for field in (
            "floor_world_raw_m", "depth_world_raw_m", "world_measurement_raw_m", "world_prediction_m"))
        if provenance == "stale":
            track["world_frame_revision"] = "stale"
        else:
            for key in ("world_frame", "world_frame_revision", "world_transform_sha256"):
                track.pop(key)
        with pytest.raises(RuntimeError, match="tracking target revision/source edge"):
            capture.process_message(tracking)
        assert capture.counts["track_samples"] == 1
        # With no finite target stage, missing frame provenance is only 2-D data.
        track.pop(stage)
        only_2d = capture.process_message(tracking)
        assert len(only_2d) == 1 and only_2d[0]["world_valid"] is False
        assert all(only_2d[0].get(field) is None for field in (
            "floor_world_raw_m", "depth_world_raw_m", "world_measurement_raw_m", "world_prediction_m"))
    finally:
        capture.finish()


def _identity_col_major() -> list[float]:
    return [float(value) for value in np.eye(4).flatten(order="F")]


def _world_digest() -> str:
    return payload_sha256(
        {
            "world_to_scene_col_major": _identity_col_major(),
            "s_obj_to_m": 1.0,
        }
    )


def _calibration_message() -> dict[str, Any]:
    return {
        "type": "calibration-bundle",
        "data": {
            "align": {
                "matrix": _identity_col_major(),
                "units": {"s_obj_to_m": 99.0},
                "scene_similarity": {
                    "world_to_scene_col_major": _identity_col_major(),
                    "s_obj_to_m": 1.0,
                    "world_to_scene_sha256": _world_digest(),
                },
                "floor_y": 0.0,
            },
            "cameras": {
                "K": {"family-room": [100.0, 100.0, 50.0, 40.0]},
                "E": {"family-room": _identity_col_major()},
            },
            "meta": {"world_frame": "backend_world_m"},
        },
    }


def _tracking_message(frame_id: int = 1) -> dict[str, Any]:
    return {
        "type": "tracking",
        "camera_id": "family-room",
        "source_id": 0,
        "frame_id": frame_id,
        "media_pts_ns": frame_id * 1_000_000,
        "captured_at_us": time.time_ns() // 1_000,
        "observed_at_us": time.time_ns() // 1_000,
        "tracks": [
            {
                "tracker_id": 101,
                "tracker_lifecycle_generation": 1,
                "camera_id": "family-room",
                "frame_id": frame_id,
                "bbox": [10.0, 20.0, 30.0, 40.0],
                "image_size": [1280.0, 720.0],
                "image_foot": [50.0, 40.0],
                "confidence": 0.9,
                "tracker_confidence": 0.8,
                "depth_anchor_m": 2.0,
                "depth_registered_m": 2.0,
                "depth_used_m": 2.0,
                "world": [9000.0, 9000.0, 9000.0],
                "world_valid": False,
            }
        ],
    }


def _create_payload(session_id: str = "operator-session") -> dict[str, Any]:
    return {
        "session_id": session_id,
        "duration_s": 30,
        "scene_binding": {
            "release_id": "menon-scene-v1",
            "authored_scene_sha256": "e" * 64,
            "world_to_scene_sha256": _world_digest(),
        },
        "waypoints": {
            "contract": "noesis.alignment.walk_waypoints",
            "contract_version": 1,
            "waypoints": [
                {
                    "id": "family-fit",
                    "camera_id": "family-room",
                    "label": "Family fit",
                    "expected_scene_xyz": [0.0, 0.0, 2.0],
                    "split": "fit",
                    "pause_s": 1.0,
                }
            ],
        },
    }


class _FakeWebSocket:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.messages = deque(json.dumps(message) for message in messages)

    async def __aenter__(self) -> "_FakeWebSocket":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def recv(self) -> str:
        if self.messages:
            return self.messages.popleft()
        await asyncio.sleep(60.0)
        raise AssertionError("unreachable")


class _FakeConnector:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.websocket = _FakeWebSocket(messages)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, uri: str, **kwargs: Any) -> _FakeWebSocket:
        self.calls.append((uri, kwargs))
        return self.websocket


def _wait_for_sample(client: TestClient, session_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/alignment-walk/sessions/{session_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["counts"].get("track_samples", 0) >= 1:
            return payload
        time.sleep(0.02)
    raise AssertionError("alignment capture did not ingest its test sample")


def test_operator_api_matches_manifest_contract_and_reflects_marker_state(
    tmp_path: Path,
) -> None:
    connector = _FakeConnector([_calibration_message(), _tracking_message()])
    app = FastAPI()
    controller = install_alignment_walk_api(
        app,
        ws_host="0.0.0.0",
        ws_port=6008,
        auth_config=InternalAuthConfig(
            mode="required",
            token_file=tmp_path / "token",
            token="t" * 48,
        ),
        output_root=tmp_path / "alignment-walk",
        websocket_connect=connector,
    )
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/alignment-walk/sessions", json=_create_payload()
        )
        assert created.status_code == 201, created.text
        assert created.json()["session_id"] == "operator-session"
        assert created.json()["state"] == "recording"
        assert created.json()["deadline_at_us"] > created.json()["started_at_us"]

        status_payload = _wait_for_sample(client, "operator-session")
        assert connector.calls[0][0] == "ws://127.0.0.1:6008"
        assert connector.calls[0][1]["additional_headers"] == {
            "Authorization": "Bearer " + "t" * 48
        }
        assert status_payload["scene_binding_verification"]["status"] == "partial"
        assert status_payload["active_tracks"][0]["tracklet_key"] == "tracklet-1"
        assert set(status_payload["active_tracks"][0]).isdisjoint(
            {
                "world",
                "world_valid",
                "world_quality",
                "depth_anchor_m",
                "depth_registered_m",
                "depth_used_m",
            }
        )

        missing_track = client.post(
            "/api/v1/alignment-walk/sessions/operator-session/markers",
            json={"waypoint_id": "family-fit", "phase": "arrived"},
        )
        assert missing_track.status_code == 422
        marker = client.post(
            "/api/v1/alignment-walk/sessions/operator-session/markers",
            json={
                "waypoint_id": "family-fit",
                "phase": "arrived",
                "tracklet_key": "tracklet-1",
            },
        )
        assert marker.status_code == 200, marker.text
        reflected = client.get(
            "/api/v1/alignment-walk/sessions/operator-session"
        ).json()
        assert reflected["markers"] == [
            {
                "waypoint_id": "family-fit",
                "phase": "arrived",
                "tracklet_key": "tracklet-1",
                "recorded_at_us": reflected["markers"][0]["recorded_at_us"],
            }
        ]
        assert reflected["waypoints"][0]["state"] == "arrived"

        finished = client.post(
            "/api/v1/alignment-walk/sessions/operator-session/finish"
        )
        assert finished.status_code == 200, finished.text
        assert finished.json()["state"] == "complete"

        analyzed = client.post(
            "/api/v1/alignment-walk/sessions/operator-session/analysis",
            json={},
        )
        assert analyzed.status_code == 200, analyzed.text
        assert "advisory_similarity_status" in analyzed.json()
        assert analyzed.json()["candidate_summary"] == {
            "fit_solver_status": "blocked",
            "calibration_candidate_status": "blocked",
            "advisory_only": True,
            "active_config_modified": False,
            "camera_candidate_status": "blocked",
            "advisory_similarity_status": "blocked",
        }

        results = client.get(
            "/api/v1/alignment-walk/sessions/operator-session/results"
        )
        assert results.status_code == 200, results.text
        result_payload = results.json()
        assert result_payload["session_id"] == "operator-session"
        assert result_payload["state"] == "results_ready"
        assert set(result_payload) >= {
            "report",
            "checks",
            "metrics",
            "candidate_similarity",
            "candidate_summary",
            "camera_candidates",
        }
        assert result_payload["candidate_summary"] == {
            "fit_solver_status": "blocked",
            "calibration_candidate_status": "blocked",
            "advisory_only": True,
            "active_config_modified": False,
            "camera_candidate_status": "blocked",
            "advisory_similarity_status": "blocked",
        }
        assert "artifacts" not in result_payload
        assert len(results.content) < 2_000_000

    controller.close()


def test_operator_api_rejects_flat_legacy_create_and_second_active_session(
    tmp_path: Path,
) -> None:
    connector = _FakeConnector([_calibration_message(), _tracking_message()])
    app = FastAPI()
    install_alignment_walk_api(
        app,
        ws_host="127.0.0.1",
        ws_port=6008,
        auth_config=InternalAuthConfig(
            mode="disabled", token_file=None, token=None
        ),
        output_root=tmp_path / "alignment-walk",
        websocket_connect=connector,
    )
    with TestClient(app) as client:
        flat = _create_payload("flat")
        flat["waypoints"] = flat["waypoints"]["waypoints"]
        assert (
            client.post("/api/v1/alignment-walk/sessions", json=flat).status_code
            == 422
        )
        assert (
            client.post(
                "/api/v1/alignment-walk/sessions",
                json=_create_payload("first"),
            ).status_code
            == 201
        )
        second = client.post(
            "/api/v1/alignment-walk/sessions",
            json=_create_payload("second"),
        )
        assert second.status_code == 409
        client.post("/api/v1/alignment-walk/sessions/first/finish")


def test_finish_timeout_is_a_bounded_conflict_not_a_type_error(
    tmp_path: Path,
) -> None:
    class _NeverStops:
        def join(self, timeout: float) -> None:
            assert timeout == 8.0

        def is_alive(self) -> bool:
            return True

    controller = AlignmentWalkController(
        ws_host="127.0.0.1",
        ws_port=6008,
        auth_config=InternalAuthConfig(
            mode="disabled", token_file=None, token=None
        ),
        output_root=tmp_path,
    )
    controller._current = SimpleNamespace(  # type: ignore[assignment]
        session_id="stuck",
        status="recording",
        stop_event=threading.Event(),
        thread=_NeverStops(),
    )
    with pytest.raises(AlignmentWalkApiError, match="did not stop") as raised:
        controller.finish("stuck")
    assert raised.value.status_code == 409


def test_ds9_mounts_the_shared_alignment_controller() -> None:
    root = Path(__file__).resolve().parents[1]
    for runtime_path in (root / "DS9" / "noesis" / "ds9_runtime_core.py",):
        source = runtime_path.read_text(encoding="utf-8")
        assert "alignment_walk_api.install_alignment_walk_api(" in source
        assert "auth_config=auth_config" in source
        assert "ws_host=args.ws_host" in source
        assert "ws_port=int(args.ws_port)" in source
