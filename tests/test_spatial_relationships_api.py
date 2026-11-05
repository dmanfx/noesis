from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from noesis.server import world_read_api as api
from noesis_core.contracts.spatial_relationships import MAX_RESPONSE_BYTES, SpatialRelationshipsRead
from noesis_core.spatial_relationships import build_spatial_relationships
from tests.test_spatial_relationships import supported_snapshot, quarantined_snapshot
from tests.test_world_read_api import HEADERS, TOKEN, world


def read(world):
    response = world.client.get(api.RELATIONSHIPS_PATH, headers=HEADERS)
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "private, no-store" and len(response.content) <= MAX_RESPONSE_BYTES
    assert TOKEN not in response.text
    value = SpatialRelationshipsRead.model_validate(response.json())
    if value.status == "unknown": assert value.entities == () and value.valid_until_us is None and value.snapshot_id is None
    return value


def test_actual_private_service_to_relationships_route_preserves_original_clock_evidence_and_zero_writes(world):
    world.service._current_snapshot = supported_snapshot(world); original = world.service.current_snapshot()
    before = (world.service._revision, world.service._event_sequence, dict(world.service._sequence_by_source), world.service._fusion)
    assert world.client.get(api.RELATIONSHIPS_PATH).status_code == 401
    assert world.client.get(api.RELATIONSHIPS_PATH, headers={"Authorization": "Bearer wrong"}).status_code == 401
    result = read(world); assert result == build_spatial_relationships(original, world.clock[0])
    assert result.entities[0].assigned_to_room.room_id == "Kitchen" and result.entities[0].supported_by.state == "known"
    world.clock[0] += 100_000; again = read(world)
    assert again.valid_until_us == result.valid_until_us and again.published_at_us == result.published_at_us
    assert again.generated_at_us == world.clock[0] and again.entities[0].observed_at_us == result.entities[0].observed_at_us
    assert before == (world.service._revision, world.service._event_sequence, world.service._sequence_by_source, world.service._fusion)
    assert world.service.current_snapshot() is original


def test_private_unknown_branches_and_parameterless_get_cannot_acquire_or_mutate(world):
    before = world.service.current_snapshot()
    for response in [world.client.get(api.RELATIONSHIPS_PATH + "?room=Kitchen", headers=HEADERS),
        world.client.request("GET", api.RELATIONSHIPS_PATH, headers=HEADERS, content=b"{}")]:
        assert response.status_code == 400 and response.headers["cache-control"] == "private, no-store"
        assert SpatialRelationshipsRead.model_validate(response.json()).status == "unknown"
    for method in ("POST", "PUT", "DELETE"):
        assert world.client.request(method, api.RELATIONSHIPS_PATH, headers=HEADERS).status_code == 405
    del world.app.state.canonical_world_service; assert read(world).reason == "world_unbound"
    world.app.state.canonical_world_service = world.service
    world.service._current_snapshot = None; assert read(world).reason == "snapshot_unavailable"
    world.service._current_snapshot = before; world.service.close(); assert read(world).reason == "world_closed"


@pytest.mark.parametrize("at", [1_999_999, 4_000_000, 4_000_001, 2**53, None, True])
def test_future_expired_and_bad_own_clock_fail_closed(world, at):
    world.clock[0] = at
    assert read(world).status == "unknown"


@pytest.mark.parametrize("change", ["close", "owner", "run", "expiry", "entity_expiry", "missing"])
def test_final_owner_clock_guard_retires_replaced_or_expired_response(world, monkeypatch, change):
    world.service._current_snapshot = supported_snapshot(world)
    original = api.encode_spatial_relationships; changed = [False]
    def encode(value):
        body = original(value)
        if value.status == "available" and not changed[0]:
            changed[0] = True
            if change == "close": world.service.close()
            elif change == "owner": world.app.state.canonical_world_service = object()
            elif change == "run": world.service.producer = world.service.producer.model_copy(update={"run_id": "new-run"})
            elif change == "missing": world.service._current_snapshot = None
            elif change == "entity_expiry": world.clock[0] = value.entities[0].stale_after_us
            else: world.clock[0] = value.valid_until_us
        return body
    monkeypatch.setattr(api, "encode_spatial_relationships", encode)
    result = read(world); assert result.status == "unknown"
    assert result.reason in {"world_closed", "world_owner_changed", "snapshot_not_fresh", "entity_expired_during_read", "snapshot_unavailable"}


def test_same_run_newer_commit_remains_original_as_of_read_not_a_silent_rebase(world, monkeypatch):
    world.service._current_snapshot = supported_snapshot(world); original = world.service.current_snapshot(); encode = api.encode_spatial_relationships; changed = [False]
    def during(value):
        body = encode(value)
        if value.status == "available" and not changed[0]:
            changed[0] = True; world.clock[0] += 1; world.service.publish(0, [], metadata={"camera_id": "kitchen"})
        return body
    monkeypatch.setattr(api, "encode_spatial_relationships", during)
    result = read(world); assert result.snapshot_id == original.snapshot_id and result.entities
    assert result.valid_until_us == original.published_at_us + 2_000_000
    assert world.service.current_snapshot() is not original


def test_busy_publication_lock_and_shared_reader_limit_are_immediate_unknown(world, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def hold():
        with world.service._lock: entered.set(); assert release.wait(3)
    thread = threading.Thread(target=hold); thread.start(); assert entered.wait(2)
    try: assert read(world).reason == "world_busy"
    finally: release.set(); thread.join(2)
    original = api.build_spatial_relationships; entered.clear(); release.clear(); count = [0]; lock = threading.Lock()
    def build(snapshot, at):
        with lock:
            count[0] += 1
            if count[0] >= 2: entered.set()
        assert release.wait(3)
        return original(snapshot, at)
    monkeypatch.setattr(api, "build_spatial_relationships", build)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(read, world) for _ in range(2)]
        try:
            assert entered.wait(2)
            assert read(world).reason == "reader_limit_exceeded"
            response = world.client.get(api.PATH, headers=HEADERS); assert response.json()["reason"] == "reader_limit_exceeded"
        finally: release.set()
        assert all(future.result(timeout=3).status == "available" for future in futures)


def test_api_retains_quarantine_and_wire_bound_unknown_no_partial_entities(world):
    world.service._current_snapshot = quarantined_snapshot(world); result = read(world)
    assert all(entity.assigned_to_room.reason == "attribution_quarantined" and entity.dwell.duration_us is None for entity in result.entities)
    original = world.service._current_snapshot; entity = original.entities[0]
    world.service._current_snapshot = original.model_copy(update={"entities": (entity,) * 257})
    assert read(world).reason == "entity_limit_exceeded"


def test_projection_and_response_serialization_are_outside_publication_lock(world, monkeypatch):
    original = SpatialRelationshipsRead.model_dump; acquired = []
    def dump(value, *args, **kwargs):
        def check():
            held = world.service._lock.acquire(blocking=False); acquired.append(held)
            if held: world.service._lock.release()
        thread = threading.Thread(target=check); thread.start(); thread.join(1)
        return original(value, *args, **kwargs)
    monkeypatch.setattr(SpatialRelationshipsRead, "model_dump", dump)
    assert read(world).status == "available" and acquired and all(acquired)
