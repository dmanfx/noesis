from __future__ import annotations

import copy
import importlib.util
import json
import sqlite3
import stat
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from noesis.server import metric_scale_requests as api
from noesis.server.scene_api import router
from noesis_core.scene_prior import ScenePriorSet
from noesis_core.contracts.scene_prior import ScenePriorCatalog


def _helper(name):
    spec = importlib.util.spec_from_file_location("metric_fixture_" + name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def journey(tmp_path, monkeypatch):
    # Real scene artifacts/register/promote/current validation; only the runtime
    # provider is injected, with the established nonidentity target-owned bundle.
    client, release = _helper("test_scene_api")._client(tmp_path, monkeypatch)
    assert client.post("/api/v1/scenes/releases", json=release).status_code == 201
    assert client.post("/api/v1/scenes/releases/release-1/promote", json={"actor_id": "owner", "expected_current_release_id": None}).status_code == 200
    bundle, binding, _ = _helper("test_alignment_walk_api")._target_owned_fixture()
    binding["release_id"] = release["release_id"]
    binding["authored_scene_sha256"] = release["authored_scene"]["sha256"]
    binding["cameras"]["kitchen"] = binding["cameras"].pop("family-room")
    bundle["cameras"]["frame_bindings"]["kitchen"] = bundle["cameras"]["frame_bindings"].pop("family-room")
    bundle["world_frame_presentations"]["target-metric-v1"]["authored_scene_sha256"] = binding["authored_scene_sha256"]
    priors = ScenePriorSet(ScenePriorCatalog.model_construct(camera_bindings=[]), {})
    calibration = SimpleNamespace(calibration_bundle=lambda: copy.deepcopy(bundle))
    client.app.state.scene_prior_runtime_provider = lambda: (priors, calibration, {0: "kitchen"})
    body = {"request_id": str(uuid.uuid4()), "actor_id": "owner-1", "scene_binding": binding,
            "points_scene": [[0, -2, 0], [3, -2, 4]], "distance_metres": 2.5, "note": "Measured reference"}
    return SimpleNamespace(client=client, body=body, bundle=bundle, calibration=calibration, tmp=tmp_path)


POST = "/api/v1/scenes/current/metric-scale-requests"
def get_path(body):
    return f'/api/v1/scenes/metric-scale-requests/{body["request_id"]}?actor_id={body["actor_id"]}'


def test_actual_current_scene_proposal_is_persisted_exact_and_never_applied(journey):
    f = journey; original_bundle = copy.deepcopy(f.bundle)
    current_release = f.client.get("/api/v1/scenes/current").json()
    result = f.client.post(POST, json=f.body)
    assert result.status_code == 201, result.text
    saved = result.json()
    assert set(saved) == {"contract", "contract_version", "request_id", "actor_id", "created_at_us", "status", "applied", "request", "scene_distance", "proposed_metres_per_scene_unit"}
    assert saved["request"] == {k: f.body[k] for k in ("scene_binding", "points_scene", "distance_metres", "note")}
    assert saved["scene_distance"] == 5 and saved["proposed_metres_per_scene_unit"] == 0.5
    assert saved["status"] == "submitted_for_review" and saved["applied"] is False
    assert result.headers["cache-control"] == "no-store"
    assert f.client.get(get_path(f.body)).json() == saved
    assert stat.S_IMODE(api._path().stat().st_mode) == 0o600
    assert stat.S_IMODE(api._path().parent.stat().st_mode) == 0o700
    assert f.bundle == original_bundle
    assert f.client.get("/api/v1/scenes/current").json() == current_release
    # A fresh ASGI router/process facade reads the same private database.
    app = FastAPI(); app.include_router(router)
    assert TestClient(app).get(get_path(f.body)).json() == saved
    f.bundle.clear()
    replay = f.client.post(POST, json=f.body)
    assert replay.status_code == 200 and replay.json() == saved


def test_original_uuid_actor_and_fingerprint_conflicts_never_replace(journey):
    f = journey; saved = f.client.post(POST, json=f.body).json()
    for key, value in [("note", "Different review"), ("actor_id", "other-owner"), ("distance_metres", 3)]:
        changed = {**f.body, key: value}
        assert f.client.post(POST, json=changed).status_code == 409
    assert f.client.get(get_path(f.body)).json() == saved
    assert f.client.get(get_path({**f.body, "actor_id": "other-owner"})).status_code == 404


@pytest.mark.parametrize("change", ["release", "model", "frame", "matrix", "mapping", "registration", "source", "missing"])
def test_current_authority_mismatches_refuse_before_any_proposal(journey, change):
    f = journey
    if change == "release": f.body["scene_binding"]["release_id"] = "old-release"
    if change == "model": f.body["scene_binding"]["authored_scene_sha256"] = "a" * 64
    if change == "frame": f.bundle["cameras"]["frame_bindings"]["kitchen"]["world_frame"]["revision"] = "other-target"
    if change == "matrix": f.bundle["cameras"]["frame_bindings"]["kitchen"]["target_from_calibration_col_major"][12] += 1
    if change == "mapping": f.bundle["world_frame_presentations"]["target-metric-v1"]["world_to_scene_col_major"][12] += 1
    if change == "registration": f.bundle["world_frame_presentations"]["target-metric-v1"]["registration_sha256"] = "b" * 64
    if change == "source": f.bundle["world_frame_presentations"]["target-metric-v1"]["source_transform_sha256s"] = ["b" * 64]
    if change == "missing": f.bundle.clear()
    assert f.client.post(POST, json=f.body).status_code == 409
    assert not api._path().exists()


@pytest.mark.parametrize("key,value", [("points_scene", [[0, 0, 0], [0, 0, 0]]), ("points_scene", [[False, 0, 0], [1, 0, 0]]),
    ("points_scene", [["0", 0, 0], [1, 0, 0]]), ("points_scene", [[None, 0, 0], [1, 0, 0]]),
    ("distance_metres", False), ("distance_metres", "2"), ("distance_metres", 0), ("distance_metres", -1), ("distance_metres", 100001),
    ("note", ""), ("note", "x" * 501), ("note", " padded "), ("note", "control\x00text"),
    ("points_scene", [[1e9 + 1, 0, 0], [1, 0, 0]]), ("actor_id", ""), ("request_id", "not-uuid")])
def test_strict_geometry_text_and_identity(journey, key, value):
    f = journey; f.body[key] = value
    assert f.client.post(POST, json=f.body).status_code == 422
    assert not api._path().exists()


def test_closed_request_v2_only_wire_limits_nonfinite_and_duplicate_fields(journey):
    f = journey
    for body in [{**f.body, "apply": True}, {**f.body, "scene_binding": {"release_id": "old", "authored_scene_sha256": "a" * 64, "world_to_scene_sha256": "b" * 64}}]:
        assert f.client.post(POST, json=body).status_code == 422
    encoded = json.dumps(f.body)
    assert f.client.post(POST, content=encoded.replace('2.5', 'NaN')).status_code == 422
    assert f.client.post(POST, content=encoded[:-1] + ',"actor_id":"second"}').status_code == 422
    assert f.client.post(POST, content=" " * (api.MAX_BODY + 1)).status_code == 413
    assert f.client.post(POST + "?apply=true", json=f.body).status_code == 422
    assert f.client.get(get_path(f.body) + "&actor_id=other").status_code == 422
    assert not api._path().exists()


def test_bounded_capacity_refuses_new_but_preserves_original_and_lookup(journey, monkeypatch):
    f = journey; monkeypatch.setattr(api, "MAX_ACTOR_RECORDS", 1)
    saved = f.client.post(POST, json=f.body).json()
    changed = {**f.body, "request_id": str(uuid.uuid4())}
    assert f.client.post(POST, json=changed).status_code == 409
    assert f.client.post(POST, json=f.body).json() == saved
    assert f.client.get(get_path(f.body)).json() == saved


def test_missing_lookup_does_not_create_state_and_corruption_is_unavailable(journey):
    f = journey
    assert f.client.get(get_path(f.body)).status_code == 404
    assert not api._path().exists()
    saved = f.client.post(POST, json=f.body).json()
    with sqlite3.connect(api._path()) as connection:
        saved["request"]["note"] = "corrupt replacement"
        connection.execute("UPDATE metric_scale_requests SET response=?", (json.dumps(saved),))
    assert f.client.get(get_path(f.body)).status_code == 503
    assert f.client.post(POST, json=f.body).status_code == 503


def test_multiline_note_is_preserved_exactly_not_used_as_authority(journey):
    f = journey; f.body["note"] = "First reference\nSecond line\twith a tab"
    result = f.client.post(POST, json=f.body)
    assert result.status_code == 201
    assert result.json()["request"]["note"] == f.body["note"]
    assert f.client.get(get_path(f.body)).json() == result.json()


def test_source_change_before_insert_is_revalidated_and_rolls_back(journey, monkeypatch):
    f = journey; original = api._validate_current; calls = []
    def changed(request, binding):
        original(request, binding); calls.append(True)
        if len(calls) == 1:
            f.bundle.clear()
    monkeypatch.setattr(api, "_validate_current", changed)
    assert f.client.post(POST, json=f.body).status_code == 409
    with sqlite3.connect(api._path()) as connection:
        assert connection.execute("SELECT count(*) FROM metric_scale_requests").fetchone()[0] == 0


def test_missing_or_malformed_runtime_authority_is_unavailable_without_fallback(journey):
    f = journey
    for provider in [lambda: (None, f.calibration, {0: "kitchen"}), lambda: ("bad",), lambda: (_ for _ in ()).throw(RuntimeError("private runtime error"))]:
        f.client.app.state.scene_prior_runtime_provider = provider
        result = f.client.post(POST, json=f.body)
        assert result.status_code == 503 and "private runtime error" not in result.text
    assert not api._path().exists()


@pytest.mark.parametrize("limit", ["MAX_RECORDS", "MAX_STORED_BYTES"])
def test_total_capacity_and_byte_caps_retain_existing_records(journey, monkeypatch, limit):
    f = journey; saved = f.client.post(POST, json=f.body).json()
    monkeypatch.setattr(api, limit, 1)
    assert f.client.post(POST, json={**f.body, "request_id": str(uuid.uuid4())}).status_code == 409
    assert f.client.get(get_path(f.body)).json() == saved


def test_unsafe_private_database_is_refused_without_permission_repair(journey):
    f = journey; assert f.client.post(POST, json=f.body).status_code == 201
    path = api._path(); path.chmod(0o644); before = path.read_bytes()
    assert f.client.get(get_path(f.body)).status_code == 503
    assert f.client.post(POST, json=f.body).status_code == 503
    assert path.read_bytes() == before and stat.S_IMODE(path.stat().st_mode) == 0o644
