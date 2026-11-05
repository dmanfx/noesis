from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
import sqlite3
import stat
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from noesis.server import semantic_space as api
from noesis.server.internal_auth import InternalBearerAuthMiddleware
from noesis.server.scene_api import _database_path, _store, router
from noesis_core.contracts.scene import SceneRelease
from noesis_core.contracts.semantic_space import SpaceWrite
from noesis_core.semantic_space_store import REPLAY_US, SemanticSpaceStore

CURRENT = "/api/v1/scenes/current/semantic-space"
LOOKUP = "/api/v1/scenes/semantic-space/lookup"


def _id():
    return str(uuid.uuid4())


def _fixture(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("semantic_scene_fixture", Path(__file__).with_name("test_scene_api.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._client(tmp_path, monkeypatch)


def _object(binding, kind="area", label="Family area"):
    return {"id": _id(), "kind": kind, "label": label, "description": "Owner taught label",
        "retired": False, "reviewed_binding": copy.deepcopy(binding), "relations": []}


@pytest.fixture
def journey(tmp_path, monkeypatch):
    client, release = _fixture(tmp_path, monkeypatch)
    assert client.post("/api/v1/scenes/releases", json=release).status_code == 201
    assert client.post("/api/v1/scenes/releases/release-1/promote", json={"actor_id": "owner", "expected_current_release_id": None}).status_code == 200
    snapshot = client.get(CURRENT).json()
    body = {"actor_id": "owner-1", "operation_id": _id(), "expected_revision": None,
        "expected_binding": snapshot["binding"], "objects": [_object(snapshot["binding"])]}
    return SimpleNamespace(client=client, body=body, release=release, path=_database_path())


def _next(body, result):
    return {**copy.deepcopy(body), "operation_id": _id(), "expected_revision": result["record"]["revision"]}


def _unchanged(path):
    return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns


def test_actual_http_create_edit_relations_retire_restart_and_no_geometry_effect(journey):
    f = journey
    release_before = f.client.get("/api/v1/scenes/current").json()
    initial = f.client.get(CURRENT)
    assert initial.headers["cache-control"] == "private, no-store"
    assert initial.json()["status"] == "unknown"
    assert initial.json()["record"] == {"revision": None, "updated_at_us": None, "actor_id": None, "objects": []}
    binding = f.body["expected_binding"]
    assert binding == {"scope": "scene_release", "release_id": "release-1",
        "release_sha256": binding["release_sha256"],
        "authored_scene_sha256": f.release["authored_scene"]["sha256"],
        "model_sha256": f.release["model"]["sha256"], "calibration_sha256": f.release["calibration"]["sha256"],
        "config_sha256": f.release["config"]["sha256"]}
    canonical_release = json.dumps(SceneRelease.model_validate(f.release).model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert binding["release_sha256"] == hashlib.sha256(canonical_release.encode()).hexdigest()
    seat = _object(binding, "seat", "Window chair")
    seat["relations"] = [{"kind": "part_of", "target_id": f.body["objects"][0]["id"]}]
    f.body["objects"].append(seat)
    created = f.client.put(CURRENT, json=f.body)
    assert created.status_code == 200, created.text
    saved = created.json()
    assert saved["replayed"] is False and saved["record"]["objects"] == f.body["objects"]
    assert {item["status"] for item in saved["review_states"]} == {"reviewed"}
    edit = _next(f.body, saved)
    edit["objects"][1]["label"] = "Reading chair"
    edit["objects"][1]["description"] = "Owner note\nsecond line"
    edited = f.client.put(CURRENT, json=edit).json()
    retired = _next(edit, edited)
    retired["objects"][1]["retired"] = True
    retired["objects"][1]["relations"] = []
    final = f.client.put(CURRENT, json=retired).json()
    assert final["review_states"][1]["status"] == "retired"
    app = FastAPI(); app.include_router(router)
    read = TestClient(app).get(CURRENT).json()
    assert read["record"] == final["record"] and read["status"] == "available"
    assert f.client.get("/api/v1/scenes/current").json() == release_before
    with sqlite3.connect(f.path) as connection:
        assert connection.execute("SELECT count(*) FROM promotion_history").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM releases").fetchone()[0] == 1
    assert stat.S_IMODE(f.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(f.path.parent.stat().st_mode) == 0o700


def test_original_lookup_and_replay_keep_original_after_newer_edit(journey):
    f = journey
    first = f.client.put(CURRENT, json=f.body).json()
    edit = _next(f.body, first); edit["objects"][0]["label"] = "Newer name"
    newer = f.client.put(CURRENT, json=edit).json()
    before = _unchanged(f.path)
    for method, path in [("post", LOOKUP), ("put", CURRENT)]:
        replay = getattr(f.client, method)(path, json=f.body)
        assert replay.status_code == 200 and replay.json() == {**first, "replayed": True}
        assert _unchanged(f.path) == before
    assert f.client.get(CURRENT).json()["record"] == newer["record"]
    wrong_actor = {**f.body, "actor_id": "other-owner"}
    assert f.client.post(LOOKUP, json=wrong_actor).status_code == 404
    changed = copy.deepcopy(f.body); changed["objects"][0]["label"] = "Different original"
    assert f.client.post(LOOKUP, json=changed).status_code == 409
    assert f.client.put(CURRENT, json=changed).status_code == 409
    assert _unchanged(f.path) == before


def test_lookup_miss_before_schema_and_missing_database_never_write(journey, tmp_path, monkeypatch):
    f = journey; before = _unchanged(f.path)
    assert f.client.post(LOOKUP, json=f.body).status_code == 404
    assert _unchanged(f.path) == before
    with sqlite3.connect(f.path) as connection:
        assert not connection.execute("SELECT 1 FROM sqlite_master WHERE name LIKE 'semantic_space_%'").fetchall()
    absent = tmp_path / "absent" / "scenes.sqlite3"
    monkeypatch.setenv("NOESIS_SCENE_STORE_PATH", str(absent))
    assert f.client.post(LOOKUP, json=f.body).status_code == 404
    assert f.client.get(CURRENT).status_code == 409
    assert f.client.put(CURRENT, json=f.body).status_code == 409
    assert not absent.parent.exists()


def test_expired_original_is_read_only_and_never_reissued(journey, monkeypatch):
    f = journey; saved = f.client.put(CURRENT, json=f.body).json()
    late = saved["record"]["updated_at_us"] + REPLAY_US
    monkeypatch.setattr(api, "_read_store", lambda: SemanticSpaceStore(database_path=f.path, now_us=lambda: late))
    before = _unchanged(f.path)
    assert f.client.post(LOOKUP, json=f.body).status_code == 410
    assert f.client.put(CURRENT, json=f.body).status_code == 410
    assert _unchanged(f.path) == before


@pytest.mark.parametrize("change", ["revision", "binding", "kind", "remove", "resurrect"])
def test_compare_and_swap_and_immutable_stable_ids(journey, change):
    f = journey; saved = f.client.put(CURRENT, json=f.body).json()
    edit = _next(f.body, saved)
    if change == "revision": edit["expected_revision"] = None
    if change == "binding": edit["expected_binding"]["model_sha256"] = "0" * 64
    if change == "kind": edit["objects"][0]["kind"] = "door"
    if change == "remove": edit["objects"] = []
    if change == "resurrect":
        edit["objects"][0]["retired"] = True
        retired = f.client.put(CURRENT, json=edit).json()
        edit = _next(edit, retired); edit["objects"][0]["retired"] = False
    before = _unchanged(f.path)
    assert f.client.put(CURRENT, json=edit).status_code == 409
    assert _unchanged(f.path) == before


@pytest.mark.parametrize("change", ["duplicate", "dangling", "self", "cycle", "retired_target", "duplicate_relation", "coordinates", "boolean", "actor", "label", "uuid", "too_many"])
def test_strict_bounded_graph_and_nonspatial_fields(journey, change):
    f = journey; body = copy.deepcopy(f.body); item = body["objects"][0]
    other = _object(body["expected_binding"], "surface", "Desk")
    if change == "duplicate": body["objects"].append(copy.deepcopy(item))
    if change == "dangling": item["relations"] = [{"kind": "covers", "target_id": _id()}]
    if change == "self": item["relations"] = [{"kind": "part_of", "target_id": item["id"]}]
    if change in {"cycle", "retired_target", "duplicate_relation"}:
        body["objects"].append(other)
        item["relations"] = [{"kind": "part_of", "target_id": other["id"]}]
        if change == "cycle": other["relations"] = [{"kind": "part_of", "target_id": item["id"]}]
        if change == "retired_target": other["retired"] = True
        if change == "duplicate_relation": item["relations"] *= 2
    if change == "coordinates": item["position"] = [0, 0, 0]
    if change == "boolean": item["retired"] = 0
    if change == "actor": body["actor_id"] = " \t"
    if change == "label": item["label"] = "Name\nspoof"
    if change == "uuid": item["id"] = item["id"].upper()
    if change == "too_many": body["objects"] = [_object(body["expected_binding"]) for _ in range(129)]
    before = _unchanged(f.path)
    assert f.client.put(CURRENT, json=body).status_code == 422
    assert _unchanged(f.path) == before


def test_new_current_release_marks_old_review_stale_and_requires_explicit_rebase(journey):
    f = journey; saved = f.client.put(CURRENT, json=f.body).json()
    store = _store()
    payload = copy.deepcopy(f.release)
    payload["release_id"] = "release-2"
    payload["authored_scene_path"] = "releases/release-2/home.obj"
    payload["authored_scene_dependencies"] = []
    payload["validation_report_path"] = "releases/release-2/validation.json"
    release = SceneRelease.model_validate(payload)
    # Register/promote a second valid row without filesystem realization: this
    # fixture tests selector CAS, not geometry/artifact publication.
    raw_store = type(store)(f.path)
    raw_store.register(release)
    raw_store.promote("release-2", actor_id="fixture", occurred_at_us=300, expected_current_release_id="release-1")
    snapshot = f.client.get(CURRENT).json()
    assert snapshot["binding"]["release_id"] == "release-2"
    assert snapshot["review_states"][0]["status"] == "stale"
    stale = _next(f.body, saved)
    assert f.client.put(CURRENT, json=stale).status_code == 409
    stale["expected_binding"] = snapshot["binding"]
    stale["objects"][0]["label"] = "Changed without fresh review"
    assert f.client.put(CURRENT, json=stale).status_code == 409
    stale["objects"][0]["reviewed_binding"] = snapshot["binding"]
    rebased = f.client.put(CURRENT, json=stale)
    assert rebased.status_code == 200 and rebased.json()["review_states"][0]["status"] == "reviewed"
    assert f.client.post(LOOKUP, json=f.body).json() == {**saved, "replayed": True}


def test_unreviewed_and_all_kinds_are_explicit(journey):
    f = journey
    f.body["objects"] = [_object(f.body["expected_binding"], kind, kind.title()) for kind in ("area", "seat", "surface", "door", "coverage")]
    for item in f.body["objects"]: item["reviewed_binding"] = None
    result = f.client.put(CURRENT, json=f.body)
    assert result.status_code == 200
    assert {r["status"] for r in result.json()["review_states"]} == {"unreviewed"}


def test_actual_private_bearer_boundary_and_malformed_requests(journey, monkeypatch):
    f = journey; token = "x" * 48
    app = FastAPI(); app.include_router(router)
    app.add_middleware(InternalBearerAuthMiddleware, token=token)
    client = TestClient(app)
    before = _unchanged(f.path)
    for method, path in [("get", CURRENT), ("put", CURRENT), ("post", LOOKUP)]:
        kwargs = {} if method == "get" else {"json": f.body}
        assert getattr(client, method)(path, **kwargs).status_code == 401
        assert getattr(client, method)(path, headers={"Authorization": "Bearer wrong", "X-Actor-Id": "owner-1"}, **kwargs).status_code == 401
    assert _unchanged(f.path) == before
    headers = {"Authorization": "Bearer " + token, "Content-Type": "application/json"}
    assert client.get(CURRENT, headers=headers).status_code == 200
    for value in ['{"actor_id":"owner","actor_id":"spoof"}', '{"value":NaN}', '{', '[]']:
        assert client.put(CURRENT, content=value, headers=headers).status_code == 422
    assert client.put(CURRENT + "?extra=1", json=f.body, headers=headers).status_code == 422
    assert client.put(CURRENT, content="{}", headers={"Authorization": "Bearer " + token}).status_code == 415
    monkeypatch.setattr(api, "MAX_BODY", 4)
    assert client.put(CURRENT, json=f.body, headers=headers).status_code == 413
    assert _unchanged(f.path) == before


def test_bounded_workers_and_retention_preserve_existing_original(journey, monkeypatch):
    f = journey
    assert api._workers.acquire(False) and api._workers.acquire(False)
    try:
        assert f.client.get(CURRENT).status_code == 503
    finally:
        api._workers.release(); api._workers.release()
    import noesis_core.semantic_space_store as module
    monkeypatch.setattr(module, "MAX_OPERATIONS", 1)
    first = f.client.put(CURRENT, json=f.body).json()
    edit = _next(f.body, first)
    before = _unchanged(f.path)
    assert f.client.put(CURRENT, json=edit).status_code == 409
    assert _unchanged(f.path) == before
    assert f.client.post(LOOKUP, json=f.body).json() == {**first, "replayed": True}


def test_concurrent_originals_share_one_revision_cas(journey):
    f = journey
    other = copy.deepcopy(f.body)
    other["operation_id"] = _id()
    other["objects"][0]["label"] = "Competing label"
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda body: f.client.put(CURRENT, json=body), (f.body, other)))
    assert sorted(result.status_code for result in results) == [200, 409]
    saved = next(result.json() for result in results if result.status_code == 200)
    assert f.client.get(CURRENT).json()["record"] == saved["record"]
    with sqlite3.connect(f.path) as connection:
        assert connection.execute("SELECT count(*) FROM semantic_space_operations").fetchone()[0] == 1


@pytest.mark.parametrize("part", ["state", "original"])
def test_malformed_saved_graph_or_original_fails_closed(journey, part):
    f = journey; saved = f.client.put(CURRENT, json=f.body).json()
    with sqlite3.connect(f.path) as connection:
        if part == "state":
            record = copy.deepcopy(saved["record"])
            record["objects"][0]["relations"] = [{"kind": "part_of", "target_id": record["objects"][0]["id"]}]
            connection.execute("UPDATE semantic_space_state SET payload_json=?", (json.dumps(record),))
        else:
            saved["record"]["actor_id"] = "other-owner"
            connection.execute("UPDATE semantic_space_operations SET result_json=?", (json.dumps(saved),))
    before = _unchanged(f.path)
    response = f.client.get(CURRENT) if part == "state" else f.client.post(LOOKUP, json=f.body)
    assert response.status_code == 503
    assert _unchanged(f.path) == before


def test_relationship_kinds_are_only_explicit_owner_declarations(journey):
    f = journey
    objects = [_object(f.body["expected_binding"], kind, kind) for kind in ("area", "seat", "surface", "door", "coverage")]
    for index, kind in enumerate(("part_of", "adjacent_to", "supports", "covers"), 1):
        objects[index]["relations"] = [{"kind": kind, "target_id": objects[0]["id"]}]
    f.body["objects"] = objects
    result = f.client.put(CURRENT, json=f.body)
    assert result.status_code == 200 and result.json()["record"]["objects"] == objects
    assert not any(key in json.dumps(result.json()) for key in ('"position"', '"occupancy"', '"room_id"', '"world"'))


@pytest.mark.parametrize("limit", ["MAX_BYTES", "MAX_STORED_BYTES"])
def test_json_byte_caps_roll_back_without_creating_semantic_state(journey, monkeypatch, limit):
    import noesis_core.semantic_space_store as module
    f = journey; before = _unchanged(f.path)
    monkeypatch.setattr(module, limit, 1)
    result = f.client.put(CURRENT, json=f.body)
    assert result.status_code == (503 if limit == "MAX_BYTES" else 409)
    assert _unchanged(f.path) == before
    with sqlite3.connect(f.path) as connection:
        assert not connection.execute("SELECT 1 FROM sqlite_master WHERE name LIKE 'semantic_space_%'").fetchall()


@pytest.mark.parametrize("link", ["symbolic", "hard"])
def test_private_store_link_aliases_are_rejected_without_readback(journey, tmp_path, monkeypatch, link):
    f = journey; f.client.put(CURRENT, json=f.body)
    alias = tmp_path / "alias.sqlite3"
    if link == "symbolic": alias.symlink_to(f.path)
    else: os.link(f.path, alias)
    before = _unchanged(f.path)
    monkeypatch.setenv("NOESIS_SCENE_STORE_PATH", str(alias))
    assert f.client.get(CURRENT).status_code == 503
    assert f.client.post(LOOKUP, json=f.body).status_code == 503
    assert _unchanged(f.path) == before
