"""Synthetic actual metadata -> private bearer HTTP -> private SQLite only."""
from __future__ import annotations

import copy
import hashlib
import sqlite3
import sys
import time
import uuid
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from noesis.server import object_associations as api
from noesis.server.object_observations import router
from noesis.server.internal_auth import InternalBearerAuthMiddleware
from noesis_core.contracts.object_associations import MAX_BYTES
from noesis_core.object_association_store import ObjectAssociationStore
from noesis_core.object_observations import ObjectObservationService

PATH = api.PATH
TOKEN = "synthetic-associations-private-" + "x" * 48
HEADERS = {"Authorization": "Bearer " + TOKEN}


def ident():
    return str(uuid.uuid4())


def row(ordinal=0):
    return SimpleNamespace(class_id=56, label=" chair <literal> ", confidence=.85,
        rect_params=SimpleNamespace(left=10., top=20., width=30., height=40.))


@pytest.fixture
def journey(tmp_path, monkeypatch):
    stamp, mono, services = [1_000_000], [10.], []
    path = tmp_path / "private" / "state.sqlite3"
    store = ObjectAssociationStore(path, now_us=lambda: stamp[0])
    monkeypatch.setattr(api, "_store", lambda: store)
    app = FastAPI(); app.include_router(router); app.add_middleware(InternalBearerAuthMiddleware, token=TOKEN)
    def source(run="synthetic-association-run"):
        service = ObjectObservationService(run, monotonic=lambda: mono[0], clock_us=lambda: stamp[0]); services.append(service)
        return service
    service = source(); app.state.object_observation_service = service
    def capture(*, camera="camera-a", frame=1, count=1):
        service = app.state.object_observation_service
        cap = service.begin_frame(SimpleNamespace(pipeline_width=640, pipeline_height=480, source_width=1920, source_height=1080),
            camera_id=camera, source_id=3, source_epoch=2, frame_id=frame,
            temporal={"observed_at_us": stamp[0], "captured_at_us": stamp[0], "capture_time_status": "estimated", "media_pts_ns": frame * 1234567})
        assert cap is not None
        for ordinal in range(count): cap.capture(row(), ordinal)
        assert cap.submit()
        deadline = time.monotonic() + 2
        while service._queue.unfinished_tasks and time.monotonic() < deadline: time.sleep(.002)
        assert service._queue.unfinished_tasks == 0 and service._worker.is_alive()
    def advance(seconds):
        stamp[0] += int(seconds * 1_000_000); mono[0] += seconds
    def count(table):
        if not path.exists(): return 0
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
            if not connection.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone(): return 0
            return connection.execute("SELECT count(*) FROM " + table).fetchone()[0]
    try:
        with TestClient(app) as client:
            def post(route, body, status=200):
                response = client.post(PATH + route, json=body, headers=HEADERS)
                assert response.status_code == status, response.text
                return response.json()
            def inspect(actor="owner-a", operation=None):
                body = {"actor_id": actor, "operation_id": operation or ident()}
                return body, post("/inspect", body)
            def prepare(body, inspection, *, association_id=None, label="Named chair", retired=False, observation=True, revision="inspection", status=200):
                capture = inspection["captures"][0] if inspection["captures"] else None
                reference = None if not observation else {key: capture[key] for key in ("run_id", "camera_id", "source_id", "source_epoch", "frame_id")}
                if reference is not None: reference["observation_id"] = capture["observation"]["observation_id"]
                command = {**body, "inspection_token": inspection["inspection_token"], "expected_revision": inspection["registry_revision"] if revision == "inspection" else revision,
                    "association_id": association_id or ident(), "label": label, "retired": retired, "observation": reference}
                return post("/prepare", command, status)
            def confirm(review, status=200):
                response = client.put(PATH, json={"review_token": review["review_token"], "command": review["command"]}, headers=HEADERS)
                assert response.status_code == status, response.text
                return response.json()
            yield SimpleNamespace(client=client, app=app, service=service, source=source, path=path, store=store, capture=capture,
                advance=advance, count=count, post=post, inspect=inspect, prepare=prepare, confirm=confirm)
    finally:
        for service in services:
            service.close(); assert not service._worker.is_alive(); assert service._queue.unfinished_tasks == 0
        assert api._workers._value == 2
        assert not any(name == "ultralytics" or name.startswith("ultralytics.") for name in sys.modules)


def unchanged(path):
    return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns


def test_exact_server_inspection_survives_normal_frames_and_expiry_then_historical_confirm_reload_original_lookup(journey):
    f = journey
    assert f.client.get(PATH).status_code == 401
    assert f.client.get(PATH, headers=HEADERS).json()["status"] == "unknown"
    assert not f.path.exists(), "GET does not create private state"
    f.capture(); original = f.client.get("/api/v1/objects/observations", headers=HEADERS).json()["cameras"][0]
    body, inspection = f.inspect(); capture = inspection["captures"][0]
    assert capture["observation"] == original["observations"][0]
    assert {key: value for key, value in capture.items() if key not in {"run_id", "observation"}} == {key: value for key, value in original.items() if key != "observations"}
    assert capture["observation"]["label"] == " chair <literal> "
    assert inspection["remaining_ms"] == 120000
    f.advance(6); f.capture(frame=2)
    before = unchanged(f.path); repeated = f.post("/inspect", body)
    assert repeated["inspection_token"] == inspection["inspection_token"] and repeated["captures"] == inspection["captures"]
    assert repeated["expires_at_us"] == inspection["expires_at_us"] and repeated["remaining_ms"] == 114000
    assert unchanged(f.path) == before, "inspection replay never writes or renews"
    review = f.prepare(body, inspection); assert review["remaining_ms"] == 114000 and review["expires_at_us"] == inspection["expires_at_us"]
    f.advance(6)
    assert f.client.get("/api/v1/objects/observations", headers=HEADERS).json()["cameras"] == []
    saved = f.confirm(review); declared = saved["record"]["associations"][0]
    assert declared["capture"] == capture and declared["capture"]["valid_until_us"] == 6_000_000
    assert declared["declaration"] == "owner_declared" and not declared["retired"] and not saved["replayed"]
    assert not {"tracker_id", "world", "room_id", "physical_id", "current_presence"}.intersection(declared)
    assert f.count("object_association_operations") == 1
    assert f.count("object_association_inspections") == f.count("object_association_reviews") == 0
    reloaded = ObjectAssociationStore(f.path).read().model_dump(mode="json")
    assert reloaded["record"] == saved["record"]
    del f.app.state.object_observation_service
    before = unchanged(f.path); found = f.post("/lookup", review["command"])
    assert found == {**saved, "replayed": True}
    assert f.confirm(review) == found and unchanged(f.path) == before
    assert f.client.get(PATH, headers=HEADERS).headers["cache-control"] == "private, no-store"


def test_actor_operation_revision_conflicts_and_stable_label_edit_retirement(journey):
    f = journey; f.capture()
    a, ia = f.inspect(); b, ib = f.inspect()
    ra, rb = f.prepare(a, ia), f.prepare(b, ib)
    saved = f.confirm(ra); f.confirm(rb, 409)
    wrong = copy.deepcopy(ra["command"]); wrong["actor_id"] = "owner-b"; f.post("/lookup", wrong, 404)
    forged = copy.deepcopy(ra["command"]); forged["association"]["label"] = "Altered original"; f.post("/lookup", forged, 409)
    foreign = {**a, "actor_id": "owner-b"}; f.prepare(foreign, ia, status=404)
    stable = saved["association_id"]
    c, ic = f.inspect(); edit = f.prepare(c, ic, association_id=stable, label="Owner renamed chair", observation=False)
    edited = f.confirm(edit); assert edited["record"]["associations"][0]["capture"] == saved["record"]["associations"][0]["capture"]
    d, id_ = f.inspect(); retire = f.prepare(d, id_, association_id=stable, label="Owner renamed chair", retired=True, observation=False)
    retired = f.confirm(retire); assert retired["record"]["associations"][0]["retired"] is True
    e, ie = f.inspect(); f.prepare(e, ie, association_id=stable, retired=False, observation=False, status=409)
    f.prepare(e, ie, association_id=stable, retired=True, observation=True, status=409)
    before = unchanged(f.path); assert f.post("/lookup", ra["command"])["record"] == saved["record"]
    assert unchanged(f.path) == before and f.count("object_association_operations") == 3


def test_fixed_inspection_lease_and_new_service_run_retire_pending_but_not_completed_original(journey):
    f = journey; f.capture(); body, inspection = f.inspect(); review = f.prepare(body, inspection)
    f.advance(119); repeated = f.prepare(body, inspection, association_id=review["command"]["association"]["id"])
    assert repeated["review_token"] == review["review_token"] and repeated["remaining_ms"] == 1000
    f.advance(1); f.confirm(review, 410); f.post("/inspect", body, 410)
    assert f.count("object_association_operations") == 0
    f.capture(frame=2); body, inspection = f.inspect(); review = f.prepare(body, inspection)
    replacement = f.source("synthetic-replacement-run"); f.app.state.object_observation_service = replacement
    f.confirm(review, 409); f.prepare(body, inspection, association_id=review["command"]["association"]["id"], status=409)
    assert f.count("object_association_operations") == 0


def test_owner_replaced_during_actual_snapshot_cannot_issue_inspection(journey, monkeypatch):
    f = journey; f.capture(); original = f.service.snapshot
    replacement = f.source("replacement-during-capture")
    def changed():
        value = original(); f.app.state.object_observation_service = replacement; return value
    monkeypatch.setattr(f.service, "snapshot", changed)
    f.post("/inspect", {"actor_id": "owner-a", "operation_id": ident()}, 409)
    assert f.count("object_association_inspections") == f.count("object_association_operations") == 0


def test_closed_private_requests_and_bounded_inspections_rows_retention(journey, monkeypatch):
    f = journey; f.capture(count=32); f.capture(camera="camera-b", count=32)
    body, inspection = f.inspect(); assert len(inspection["captures"]) == 50 and inspection["omitted_count"] == 14
    assert len(f.client.get(PATH, headers=HEADERS).content) <= MAX_BYTES
    for _ in range(15): f.inspect()
    f.post("/inspect", {"actor_id": "owner-a", "operation_id": ident()}, 409)
    assert f.count("object_association_inspections") == 16
    f.advance(120); f.capture(frame=2)
    body, inspection = f.inspect(); assert f.count("object_association_inspections") == 1
    assert f.client.get(PATH + "?query=chair", headers=HEADERS).status_code == 422
    assert f.client.request("GET", PATH, content=b"{}", headers=HEADERS).status_code == 422
    assert f.client.post(PATH + "/inspect", content='{"actor_id":"owner-a","actor_id":"owner-b"}', headers={**HEADERS, "Content-Type": "application/json"}).status_code == 422
    assert f.client.post(PATH + "/inspect", json={**body, "tracker_id": 7}, headers=HEADERS).status_code == 422
    assert f.client.post(PATH + "/inspect", content=b"x" * (MAX_BYTES + 1), headers={**HEADERS, "Content-Type": "application/json"}).status_code == 413
    from noesis_core import object_association_store as module
    monkeypatch.setattr(module, "MAX_OPERATIONS", 1)
    review = f.prepare(body, inspection); saved = f.confirm(review)
    other, issued = f.inspect(); review2 = f.prepare(other, issued, association_id=saved["association_id"], observation=False, label="Changed label")
    before = f.store.read(); f.confirm(review2, 409)
    assert f.store.read() == before and f.count("object_association_operations") == 1
    before_bytes = unchanged(f.path); f.post("/lookup", review["command"])
    assert unchanged(f.path) == before_bytes


def test_exact_readback_corruption_rolls_back_registry_and_operation(journey):
    f = journey; f.capture(); body, inspection = f.inspect(); review = f.prepare(body, inspection)
    with sqlite3.connect(f.path) as connection:
        connection.execute("CREATE TRIGGER corrupt_association AFTER INSERT ON object_association_state BEGIN UPDATE object_association_state SET payload_json='{}'; END")
    f.confirm(review, 503)
    assert f.count("object_association_state") == f.count("object_association_operations") == 0
