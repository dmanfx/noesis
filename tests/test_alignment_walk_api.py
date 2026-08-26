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
