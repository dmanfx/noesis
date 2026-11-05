from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from noesis.server import semantic_anchor as api
from noesis.server.internal_auth import InternalBearerAuthMiddleware
from noesis.server.scene_api import _database_path, _store, router
from noesis_core.contracts.scene import SceneRelease
from noesis_core.contracts.semantic_anchor import AnchorWrite
from noesis_core.contracts.semantic_space import SpaceWrite
from noesis_core.scene_store import SceneReleaseConflict, SceneReleaseStore
from noesis_core.semantic_anchor_store import REPLAY_US, SemanticAnchorStore
from noesis_core.semantic_space_store import SemanticSpaceStore

CURRENT = "/api/v1/scenes/current/semantic-anchors"
LOOKUP = "/api/v1/scenes/semantic-anchors/lookup"
TOKEN = "synthetic-internal-bearer-" + "x" * 48
HEADERS = {"Authorization": "Bearer " + TOKEN}


def _id():
    return str(uuid.uuid4())


def _unchanged(path):
    return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns


def _next(body, result):
    return {**copy.deepcopy(body), "operation_id": _id(), "expected_revision": result["record"]["revision"]}


@pytest.fixture
def journey(tmp_path, monkeypatch):
    # Exact fixture import avoids accidentally importing an unrelated tests package.
    spec = importlib.util.spec_from_file_location("anchor_scene_fixture", Path(__file__).with_name("test_scene_api.py"))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    client, release = module._client(tmp_path, monkeypatch)
    authored = tmp_path / "virtual-twin" / release["authored_scene_path"]
    authored.write_bytes(b"mtllib home.mtl\no synthetic-home\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    release["authored_scene"]["sha256"] = hashlib.sha256(authored.read_bytes()).hexdigest()
    release["authored_scene_size_bytes"] = authored.stat().st_size
    assert client.post("/api/v1/scenes/releases", json=release).status_code == 201
    assert client.post("/api/v1/scenes/releases/release-1/promote", json={"actor_id": "fixture", "expected_current_release_id": None}).status_code == 200
    scene_store = _store(); semantic_store = SemanticSpaceStore(scene_store)
    binding = semantic_store.read().binding.model_dump(mode="json")
    semantic_command = {"actor_id": "owner-1", "operation_id": _id(), "expected_revision": None,
        "expected_binding": binding, "objects": [{"id": _id(), "kind": "surface", "label": "Declared shelf", "description": "Owner-taught object",
            "retired": False, "reviewed_binding": binding, "relations": []}]}
    semantic = semantic_store.replace(SpaceWrite.model_validate(semantic_command))
    app = FastAPI(); app.include_router(router); app.add_middleware(InternalBearerAuthMiddleware, token=TOKEN)
    private = TestClient(app)
    snapshot = private.get(CURRENT, headers=HEADERS).json()
    body = {"actor_id": "owner-1", "operation_id": _id(), "expected_revision": None, "expected_binding": binding,
        "expected_artifact": snapshot["artifact"], "expected_semantic_revision": semantic.record.revision,
        "anchors": [{"id": _id(), "semantic_id": semantic.record.objects[0].id, "semantic_revision": semantic.record.revision,
            "binding": binding, "artifact": snapshot["artifact"], "declaration": "owner_declared", "retired": False,
            "geometry": {"space": "authored_scene_geometry_local", "units": "artifact_units", "point": [0.0, 0.0, 0.0], "normal": [0.0, 1.0, 0.0]}}]}
    return SimpleNamespace(client=private, body=body, path=_database_path(), scene_store=scene_store,
        semantic_store=semantic_store, semantic_command=semantic_command, semantic=semantic, release=release, authored=authored)


def test_actual_scene_semantic_store_private_api_edit_retire_and_restart(journey):
    f = journey; before_release = f.scene_store.current(); before_semantic = f.semantic_store.read()
    initial = f.client.get(CURRENT, headers=HEADERS)
    assert initial.headers["cache-control"] == "private, no-store"
    assert initial.json()["status"] == "unknown" and initial.json()["record"]["anchors"] == []
    assert initial.json()["eligibility"] == {"status": "supported", "reason": "authored_obj_contract"}
    response = f.client.put(CURRENT, json=f.body, headers=HEADERS)
    assert response.status_code == 200, response.text
    first = response.json(); assert first["record"]["anchors"] == f.body["anchors"] and not first["replayed"]
    assert first["review_states"][0]["status"] == "reviewed"
    edit = _next(f.body, first); edit["anchors"][0]["geometry"]["point"] = [0.25, 0.5, -0.25]
    second = f.client.put(CURRENT, json=edit, headers=HEADERS).json()
    retired = _next(edit, second); retired["anchors"][0]["retired"] = True
    final = f.client.put(CURRENT, json=retired, headers=HEADERS).json()
    assert final["review_states"][0]["status"] == "retired"
    restarted = SemanticAnchorStore(database_path=f.path).read()
    assert restarted.record.model_dump(mode="json") == final["record"]
    assert f.scene_store.current() == before_release and f.semantic_store.read() == before_semantic
    with sqlite3.connect(f.path) as connection:
        assert connection.execute("SELECT count(*) FROM promotion_history").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM releases").fetchone()[0] == 1


def test_original_lookup_and_matching_put_are_actor_bound_read_only_after_newer_state(journey):
    f = journey; first = f.client.put(CURRENT, json=f.body, headers=HEADERS).json()
    edit = _next(f.body, first); edit["anchors"][0]["geometry"]["point"] = [1.0, 0.0, 0.0]
    newer = f.client.put(CURRENT, json=edit, headers=HEADERS).json(); before = _unchanged(f.path)
    for method, path in [("post", LOOKUP), ("put", CURRENT)]:
        original = getattr(f.client, method)(path, json=f.body, headers=HEADERS)
        assert original.status_code == 200 and original.json() == {**first, "replayed": True}
        assert _unchanged(f.path) == before
    other_actor = {**f.body, "actor_id": "another-owner"}
    assert f.client.post(LOOKUP, json=other_actor, headers=HEADERS).status_code == 404
    changed = copy.deepcopy(f.body); changed["anchors"][0]["geometry"]["point"] = [2.0, 0.0, 0.0]
    assert f.client.post(LOOKUP, json=changed, headers=HEADERS).status_code == 409
    assert _unchanged(f.path) == before
    assert f.client.get(CURRENT, headers=HEADERS).json()["record"] == newer["record"]


def test_lookup_missing_expired_and_absent_database_are_read_only(journey, tmp_path, monkeypatch):
    f = journey; before = _unchanged(f.path)
    assert f.client.post(LOOKUP, json=f.body, headers=HEADERS).status_code == 404
    assert _unchanged(f.path) == before
    with sqlite3.connect(f.path) as connection:
        assert not connection.execute("SELECT 1 FROM sqlite_master WHERE name LIKE 'semantic_anchor_%'").fetchall()
    saved = f.client.put(CURRENT, json=f.body, headers=HEADERS).json()
    late = saved["record"]["updated_at_us"] + REPLAY_US
    monkeypatch.setattr(api, "_read_store", lambda: SemanticAnchorStore(database_path=f.path, now_us=lambda: late))
    before = _unchanged(f.path)
    for method, path in [("post", LOOKUP), ("put", CURRENT)]:
        assert getattr(f.client, method)(path, json=f.body, headers=HEADERS).status_code == 410
        assert _unchanged(f.path) == before
    absent = tmp_path / "never-created" / "scene.sqlite3"
    monkeypatch.setattr(api, "_read_store", lambda: SemanticAnchorStore(database_path=absent))
    assert f.client.post(LOOKUP, json=f.body, headers=HEADERS).status_code == 404
    assert f.client.get(CURRENT, headers=HEADERS).status_code == 409
    assert f.client.put(CURRENT, json=f.body, headers=HEADERS).status_code == 409
    assert not absent.parent.exists()


@pytest.mark.parametrize("change", ["anchor_revision", "semantic_revision", "binding", "artifact", "semantic_id", "remove", "revive"])
def test_cas_and_stable_id_invariants(journey, change):
    f = journey; first = f.client.put(CURRENT, json=f.body, headers=HEADERS).json(); edit = _next(f.body, first)
    if change == "anchor_revision": edit["expected_revision"] = None
    if change == "semantic_revision": edit["expected_semantic_revision"] = _id()
    if change == "binding": edit["expected_binding"]["model_sha256"] = "0" * 64
    if change == "artifact": edit["expected_artifact"]["relative_path"] = "releases/release-1/manifest.json"
    if change == "semantic_id": edit["anchors"][0]["semantic_id"] = _id()
    if change == "remove": edit["anchors"] = []
    if change == "revive":
        edit["anchors"][0]["retired"] = True
        retired = f.client.put(CURRENT, json=edit, headers=HEADERS).json()
        edit = _next(edit, retired); edit["anchors"][0]["retired"] = False
    before = _unchanged(f.path)
    assert f.client.put(CURRENT, json=edit, headers=HEADERS).status_code == 409
    assert _unchanged(f.path) == before


def test_concurrent_cas_has_one_winner_using_same_scene_sqlite_transaction(journey):
    f = journey; store = SemanticAnchorStore(f.scene_store)
    commands = [AnchorWrite.model_validate({**f.body, "operation_id": _id()}) for _ in range(2)]
    def write(command):
        try: return store.replace(command).record.revision
        except SceneReleaseConflict: return "conflict"
    with ThreadPoolExecutor(max_workers=2) as workers:
        values = list(workers.map(write, commands))
    assert values.count("conflict") == 1 and len(set(values)) == 2


@pytest.mark.parametrize("change", ["rename", "retire", "unreview"])
def test_semantic_replacement_retires_review_but_original_lookup_survives(journey, change):
    f = journey; saved = f.client.put(CURRENT, json=f.body, headers=HEADERS).json()
    semantic = copy.deepcopy(f.semantic_command); semantic["operation_id"] = _id(); semantic["expected_revision"] = f.semantic.record.revision
    if change == "rename": semantic["objects"][0]["label"] = "New declaration name"
    if change == "retire": semantic["objects"][0]["retired"] = True
    if change == "unreview": semantic["objects"][0]["reviewed_binding"] = None
    new_semantic = f.semantic_store.replace(SpaceWrite.model_validate(semantic))
    snapshot = f.client.get(CURRENT, headers=HEADERS).json()
    expected = {"rename": "stale_semantic", "retire": "semantic_retired", "unreview": "semantic_unreviewed"}[change]
    assert snapshot["review_states"][0]["status"] == expected
    stale = _next(f.body, saved); before = _unchanged(f.path)
    assert f.client.put(CURRENT, json=stale, headers=HEADERS).status_code == 409
    stale["expected_semantic_revision"] = new_semantic.record.revision
    stale["anchors"][0]["semantic_revision"] = new_semantic.record.revision
    if change != "rename": assert f.client.put(CURRENT, json=stale, headers=HEADERS).status_code == 409
    assert f.client.post(LOOKUP, json=f.body, headers=HEADERS).json() == {**saved, "replayed": True}
    assert _unchanged(f.path) == before


def test_replaced_scene_is_stale_and_original_lookup_does_not_revalidate_current_artifact(journey):
    f = journey; saved = f.client.put(CURRENT, json=f.body, headers=HEADERS).json()
    release = copy.deepcopy(f.release); release["release_id"] = "release-2"
    release["authored_scene_path"] = "releases/release-2/home.obj"; release["authored_scene_dependencies"] = []
    release["validation_report_path"] = "releases/release-2/validation.json"
    selector = SceneReleaseStore(f.path); selector.register(SceneRelease.model_validate(release))
    selector.promote("release-2", actor_id="fixture", occurred_at_us=300, expected_current_release_id="release-1")
    snapshot = f.client.get(CURRENT, headers=HEADERS).json()
    assert snapshot["review_states"][0]["status"] == "stale_scene"
    before = _unchanged(f.path)
    assert f.client.put(CURRENT, json=_next(f.body, saved), headers=HEADERS).status_code == 409
    assert f.client.post(LOOKUP, json=f.body, headers=HEADERS).json() == {**saved, "replayed": True}
    assert _unchanged(f.path) == before


@pytest.mark.parametrize("replacement", ["retired_target", "semantic_revision", "scene"])
def test_exact_historical_retirement_preserves_declaration_under_current_top_level_cas(journey, replacement):
    f = journey; first = f.client.put(CURRENT, json=f.body, headers=HEADERS).json()
    if replacement == "scene":
        release = copy.deepcopy(f.release); release["release_id"] = "replacement"
        # There is no new geometry declaration in a retirement. Its top-level
        # selector binding is current, while its original artifact stays exact;
        # this deliberately absent metadata file must never be read as a model.
        release["authored_scene_path"] = "releases/replacement/manifest.json"
        release["authored_scene_dependencies"] = []; release["validation_report_path"] = "releases/replacement/validation.json"
        selector = SceneReleaseStore(f.path); selector.register(SceneRelease.model_validate(release))
        selector.promote("replacement", actor_id="fixture", occurred_at_us=300, expected_current_release_id="release-1")
    else:
        semantic = copy.deepcopy(f.semantic_command); semantic["operation_id"] = _id(); semantic["expected_revision"] = f.semantic.record.revision
        if replacement == "retired_target": semantic["objects"][0]["retired"] = True
        else: semantic["objects"][0]["label"] = "New semantic review"
        f.semantic_store.replace(SpaceWrite.model_validate(semantic))
    current = f.client.get(CURRENT, headers=HEADERS).json()
    retirement = _next(f.body, first)
    retirement.update(expected_binding=current["binding"], expected_artifact=current["artifact"], expected_semantic_revision=current["semantic_revision"])
    retirement["anchors"][0]["retired"] = True
    before = _unchanged(f.path)
    for field in ("expected_revision", "expected_semantic_revision", "expected_binding", "expected_artifact"):
        wrong = copy.deepcopy(retirement)
        if field == "expected_revision": wrong[field] = _id()
        elif field == "expected_semantic_revision": wrong[field] = _id()
        elif field == "expected_binding": wrong[field]["config_sha256"] = "0" * 64
        else: wrong[field]["relative_path"] = "releases/wrong/home.obj"
        assert f.client.put(CURRENT, json=wrong, headers=HEADERS).status_code == 409
    for mutation in ("point", "signed_zero", "normal", "artifact", "binding", "semantic_revision"):
        wrong = copy.deepcopy(retirement); anchor = wrong["anchors"][0]
        if mutation == "point": anchor["geometry"]["point"][0] = 0.25
        if mutation == "signed_zero": anchor["geometry"]["point"][0] = -0.0
        if mutation == "normal": anchor["geometry"]["normal"] = [1.0, 0.0, 0.0]
        if mutation == "artifact": anchor["artifact"]["relative_path"] = "releases/changed/home.obj"
        if mutation == "binding": anchor["binding"]["model_sha256"] = "0" * 64
        if mutation == "semantic_revision": anchor["semantic_revision"] = _id()
        assert f.client.put(CURRENT, json=wrong, headers=HEADERS).status_code == 409, mutation
    assert _unchanged(f.path) == before
    response = f.client.put(CURRENT, json=retirement, headers=HEADERS)
    assert response.status_code == 200, response.text
    saved = response.json()
    assert saved["record"]["anchors"] == [{**f.body["anchors"][0], "retired": True}]
    assert saved["review_states"][0]["status"] == "retired"
    before = _unchanged(f.path)
    assert f.client.post(LOOKUP, json=f.body, headers=HEADERS).json() == {**first, "replayed": True}
    assert f.client.post(LOOKUP, json=retirement, headers=HEADERS).json() == {**saved, "replayed": True}
    assert _unchanged(f.path) == before
    # Already-retired declarations remain retainable without rebinding, while
    # immutable retirement still prohibits revival.
    retained = f.client.put(CURRENT, json=_next(retirement, saved), headers=HEADERS)
    assert retained.status_code == 200
    revived = _next(retirement, retained.json()); revived["anchors"][0]["retired"] = False
    before = _unchanged(f.path)
    assert f.client.put(CURRENT, json=revived, headers=HEADERS).status_code == 409
    assert _unchanged(f.path) == before


def test_fabricated_retired_anchor_is_not_a_historical_declaration(journey):
    f = journey; body = copy.deepcopy(f.body); body["anchors"][0]["retired"] = True
    before = _unchanged(f.path)
    assert f.client.put(CURRENT, json=body, headers=HEADERS).status_code == 409
    assert _unchanged(f.path) == before
    saved = f.client.put(CURRENT, json=f.body, headers=HEADERS).json()
    body = _next(f.body, saved); extra = copy.deepcopy(body["anchors"][0]); extra.update(id=_id(), retired=True)
    body["anchors"].append(extra); before = _unchanged(f.path)
    assert f.client.put(CURRENT, json=body, headers=HEADERS).status_code == 409
    assert _unchanged(f.path) == before


@pytest.mark.parametrize("change", ["duplicate", "too_many", "extra", "frame", "meters", "string", "boolean", "nan", "normal", "actor", "uuid", "retired"])
def test_strict_schema_and_bounds_before_storage(journey, change):
    f = journey; body = copy.deepcopy(f.body); anchor = body["anchors"][0]
    if change == "duplicate": body["anchors"].append(copy.deepcopy(anchor))
    if change == "too_many": body["anchors"] = [{**copy.deepcopy(anchor), "id": _id()} for _ in range(129)]
    if change == "extra": anchor["room_id"] = "invented-room"
    if change == "frame": anchor["geometry"]["space"] = "backend_world_m"
    if change == "meters": anchor["geometry"]["units"] = "meters"
    if change == "string": anchor["geometry"]["point"][0] = "1.5"
    if change == "boolean": anchor["geometry"]["point"][0] = True
    if change == "nan": anchor["geometry"]["point"][0] = float("nan")
    if change == "normal": anchor["geometry"]["normal"] = [0, 0, 0]
    if change == "actor": body["actor_id"] = " owner "
    if change == "uuid": anchor["id"] = anchor["id"].upper()
    if change == "retired": anchor["retired"] = 1
    before = _unchanged(f.path)
    response = f.client.put(CURRENT, content=json.dumps(body), headers={**HEADERS, "Content-Type": "application/json"})
    assert response.status_code == 422, response.text
    assert _unchanged(f.path) == before


def test_private_bearer_closed_requests_and_byte_limit(journey):
    f = journey; before = _unchanged(f.path)
    for method, path in [("get", CURRENT), ("put", CURRENT), ("post", LOOKUP)]:
        kwargs = {} if method == "get" else {"json": f.body}
        assert getattr(f.client, method)(path, **kwargs).status_code == 401
        assert getattr(f.client, method)(path, headers={"Authorization": "Bearer wrong"}, **kwargs).status_code == 401
    assert f.client.get(CURRENT + "?ref=1", headers=HEADERS).status_code == 422
    assert f.client.request("GET", CURRENT, content="{}", headers=HEADERS).status_code == 422
    assert f.client.put(CURRENT, content="{}", headers=HEADERS).status_code == 415
    headers = {**HEADERS, "Content-Type": "application/json"}
    assert f.client.put(CURRENT, content='{"actor_id":"a","actor_id":"b"}', headers=headers).status_code == 422
    assert f.client.put(CURRENT, content=b" " * (256 * 1024 + 1), headers=headers).status_code == 413
    assert _unchanged(f.path) == before


def test_exact_write_readback_detects_trigger_corruption_and_rolls_back(journey):
    f = journey; first = f.client.put(CURRENT, json=f.body, headers=HEADERS).json()
    with sqlite3.connect(f.path) as connection:
        connection.execute("CREATE TRIGGER corrupt_anchor AFTER INSERT ON semantic_anchor_state BEGIN UPDATE semantic_anchor_state SET payload_json='{}' WHERE singleton=1; END")
    before = _unchanged(f.path)
    assert f.client.put(CURRENT, json=_next(f.body, first), headers=HEADERS).status_code == 503
    assert _unchanged(f.path) == before
    assert f.client.get(CURRENT, headers=HEADERS).json()["record"] == first["record"]


def test_no_cached_substitute_on_corrupt_geometry_but_original_lookup_survives(journey):
    f = journey; saved = f.client.put(CURRENT, json=f.body, headers=HEADERS).json()
    f.authored.write_bytes(b"tampered")
    before = _unchanged(f.path)
    assert f.client.put(CURRENT, json=_next(f.body, saved), headers=HEADERS).status_code == 503
    assert f.client.post(LOOKUP, json=f.body, headers=HEADERS).json() == {**saved, "replayed": True}
    assert _unchanged(f.path) == before


@pytest.mark.parametrize("suffix,content", [("json", b'{"model":"manifest only"}'), ("obj", b'{"not":"geometry"}')])
def test_metadata_or_json_disguised_as_mesh_is_explicitly_unsupported(journey, suffix, content):
    f = journey; release = copy.deepcopy(f.release); release["release_id"] = "unsupported"
    release["authored_scene_path"] = f"releases/unsupported/scene.{suffix}"
    release["authored_scene"]["sha256"] = hashlib.sha256(content).hexdigest(); release["authored_scene_size_bytes"] = len(content)
    release["authored_scene_dependencies"] = []; release["validation_report_path"] = "releases/unsupported/validation.json"
    path = f.authored.parents[2] / release["authored_scene_path"]; path.parent.mkdir(); path.write_bytes(content)
    selector = SceneReleaseStore(f.path); selector.register(SceneRelease.model_validate(release))
    selector.promote("unsupported", actor_id="fixture", occurred_at_us=300, expected_current_release_id="release-1")
    semantic = copy.deepcopy(f.semantic_command); semantic["operation_id"] = _id(); semantic["expected_revision"] = f.semantic.record.revision
    snapshot = f.client.get(CURRENT, headers=HEADERS).json(); semantic["expected_binding"] = snapshot["binding"]
    semantic["objects"][0]["reviewed_binding"] = snapshot["binding"]
    updated = f.semantic_store.replace(SpaceWrite.model_validate(semantic))
    body = copy.deepcopy(f.body); body["expected_binding"] = snapshot["binding"]; body["expected_artifact"] = snapshot["artifact"]
    body["expected_semantic_revision"] = updated.record.revision
    body["anchors"][0].update(binding=snapshot["binding"], artifact=snapshot["artifact"], semantic_revision=updated.record.revision)
    if suffix == "json": assert snapshot["eligibility"] == {"status": "unsupported", "reason": "unsupported_authored_scene_format"}
    response = f.client.put(CURRENT, json=body, headers=HEADERS)
    assert response.status_code == 409 and "unsupported" in response.text
