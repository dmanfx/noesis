from __future__ import annotations

import copy
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from noesis.server import world_read_api as api
from noesis.server.internal_auth import InternalBearerAuthMiddleware
from noesis_core.contracts.base import ArtifactFingerprint, ProducerRef
from noesis_core.contracts.world import WorldAttributionClaim, WorldAttributionQuarantine, WorldSnapshot
from noesis_core.world_service import CanonicalWorldService, WorldArtifacts

TOKEN = "disposable-world-read-gateway-" + "x" * 48
HEADERS = {"Authorization": "Bearer " + TOKEN}
KEYS = {"contract", "version", "status", "reason", "generated_at_us", "valid_until_us", "max_snapshot_age_us", "snapshot"}


def _track():
    return {"camera_id": "kitchen", "tracker_id": 7, "frame_id": 1, "observed_at_us": 1_900_000,
        "capture_time_status": "estimated", "media_pts_ns": 1000, "bbox": [10, 20, 30, 40], "image_size": [1920, 1080],
        "confidence": 0.91, "tracker_confidence": 0.88, "stable_id": 1, "identity_kind": "resident", "resident_uuid": "fixture-resident",
        "world": [-1.0, 0.0, 2.0], "world_valid": True, "world_frame": "backend_world_m", "world_frame_revision": "fixture-world",
        "world_transform_sha256": "d" * 64, "calibration_revision": "fixture-calibration", "world_quality": "good", "world_source": "pose_depth_fused",
        "zone": "Kitchen", "zone_source": "nvdsanalytics_roi", "zone_authoritative": True}


@pytest.fixture
def world():
    clock = [2_000_000]
    producer = ProducerRef(runtime="test", instance_id="disposable-native-owner", run_id="run-world-read", software_revision="fixture")
    artifacts = WorldArtifacts(calibration=ArtifactFingerprint(role="camera_calibration", sha256="a" * 64),
        model=ArtifactFingerprint(role="tracking_models", sha256="b" * 64), config=ArtifactFingerprint(role="runtime_config", sha256="c" * 64))
    service = CanonicalWorldService(producer=producer, artifacts=artifacts, clock_us=lambda: clock[0])
    publication = service.publish(0, [_track()], metadata={"camera_id": "kitchen"})
    app = FastAPI(); app.include_router(api.router); app.add_middleware(InternalBearerAuthMiddleware, token=TOKEN)
    app.state.canonical_world_service = service
    try:
        with TestClient(app) as client:
            yield SimpleNamespace(service=service, snapshot=publication.snapshot, publication=publication, clock=clock, app=app, client=client)
    finally:
        service.close()


def _read(world):
    response = world.client.get(api.PATH, headers=HEADERS)
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "private, no-store"
    assert len(response.content) <= api.MAX_RESPONSE_BYTES
    body = response.json()
    assert set(body) == KEYS
    assert body["contract"] == "noesis.world.current-read" and body["version"] == 1
    assert body["max_snapshot_age_us"] == 2_000_000
    if body["status"] == "unknown":
        assert body["snapshot"] is None and body["valid_until_us"] is None
    return body


def test_private_bearer_actual_committed_snapshot_exact_evidence_original_timestamps_and_zero_world_writes(world):
    before = (world.service._revision, world.service._event_sequence, dict(world.service._sequence_by_source), world.service._fusion)
    assert world.client.get(api.PATH).status_code == 401
    assert world.client.get(api.PATH, headers={"Authorization": "Bearer wrong"}).status_code == 401
    result = _read(world)
    assert result["status"] == "available"
    assert result["snapshot"] == world.snapshot.model_dump(mode="json")
    assert WorldSnapshot.model_validate(result["snapshot"]) == world.snapshot
    assert result["snapshot"]["entities"][0]["room_id"] == "Kitchen"
    assert result["snapshot"]["entities"][0]["sources"][0]["zone_source"] == "nvdsanalytics_roi"
    assert result["snapshot"]["entities"][0]["position"]["x"] == -1.0
    assert result["generated_at_us"] == 2_000_000 and result["valid_until_us"] == 4_000_000
    world.clock[0] += 50_000
    again = _read(world)
    assert again["snapshot"] == result["snapshot"]
    assert again["valid_until_us"] == result["valid_until_us"] and again["generated_at_us"] == 2_050_000
    assert before == (world.service._revision, world.service._event_sequence, world.service._sequence_by_source, world.service._fusion)
    assert world.service.current_snapshot() is world.snapshot
    assert TOKEN not in json.dumps(result)


def test_absent_uncommitted_and_closed_owners_are_unknown_without_fallback(world):
    del world.app.state.canonical_world_service
    assert _read(world)["reason"] == "world_unbound"
    world.app.state.canonical_world_service = world.service
    committed = world.service._current_snapshot; world.service._current_snapshot = None
    assert _read(world)["reason"] == "snapshot_unavailable"
    world.service._current_snapshot = committed; world.service.close()
    assert _read(world)["reason"] == "world_closed"


@pytest.mark.parametrize("at", [1_999_999, 4_000_000, 4_000_001])
def test_future_or_exact_two_second_boundary_is_unknown_and_never_renews(world, at):
    world.clock[0] = at
    assert _read(world)["reason"] == "snapshot_not_fresh"
    assert world.service.current_snapshot() is world.snapshot


@pytest.mark.parametrize("at", [None, -1, 0, True, 2.5, float("nan"), float("inf"), 2**53])
def test_clock_is_own_finite_native_safe_integer_or_explicit_unknown(world, at):
    world.clock[0] = at
    result = _read(world)
    assert result["reason"] == "invalid_clock" and result["generated_at_us"] is None


def test_authoritative_empty_is_exact_available_evidence_not_an_invented_room_list(world):
    world.clock[0] += 1
    empty = world.service.publish(0, [], metadata={"camera_id": "kitchen", "observed_at_us": world.clock[0]})
    result = _read(world)
    assert result["status"] == "available" and result["snapshot"] == empty.snapshot.model_dump(mode="json")
    assert result["snapshot"]["entities"] == []


def test_query_body_and_non_get_cannot_mutate_or_change_read_parameters(world):
    before = world.service.current_snapshot()
    for response in [world.client.get(api.PATH + "?renew=true", headers=HEADERS),
        world.client.request("GET", api.PATH, headers=HEADERS, content=b"{}")]:
        assert response.status_code == 400 and set(response.json()) == KEYS
        assert response.headers["cache-control"] == "private, no-store"
        assert response.json()["status"] == "unknown"
    for method in ["POST", "PUT", "DELETE"]:
        assert world.client.request(method, api.PATH, headers=HEADERS).status_code == 405
    assert world.service.current_snapshot() is before


def test_contended_publication_lock_returns_unknown_immediately(world):
    entered, release = threading.Event(), threading.Event()
    def hold():
        with world.service._lock:
            entered.set(); assert release.wait(3)
    thread = threading.Thread(target=hold); thread.start(); assert entered.wait(2)
    try:
        start = time.monotonic(); result = _read(world)
        assert result["reason"] == "world_busy" and result["generated_at_us"] is None
        assert time.monotonic() - start < 1
    finally:
        release.set(); thread.join(2)
    assert not thread.is_alive()


def test_serialization_and_own_clock_are_outside_publication_lock(world, monkeypatch):
    original = WorldSnapshot.model_dump
    acquired = []
    def check_lock():
        value = world.service._lock.acquire(blocking=False); acquired.append(value)
        if value: world.service._lock.release()
    def dump(snapshot, *args, **kwargs):
        thread = threading.Thread(target=check_lock); thread.start(); thread.join(1)
        return original(snapshot, *args, **kwargs)
    monkeypatch.setattr(WorldSnapshot, "model_dump", dump)
    def clock():
        thread = threading.Thread(target=check_lock); thread.start(); thread.join(1)
        return world.clock[0]
    world.service._clock_us = clock
    assert _read(world)["status"] == "available"
    assert acquired and all(acquired)


def test_at_most_two_actual_serializers_and_no_waiting_third_reader(world, monkeypatch):
    original = WorldSnapshot.model_dump; lock = threading.Lock(); active = [0, 0]; entered, release = threading.Event(), threading.Event()
    def dump(snapshot, *args, **kwargs):
        with lock:
            active[0] += 1; active[1] = max(active); current = active[0]
        if current == 2: entered.set()
        try:
            assert release.wait(3)
            return original(snapshot, *args, **kwargs)
        finally:
            with lock: active[0] -= 1
    monkeypatch.setattr(WorldSnapshot, "model_dump", dump)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_read, world) for _ in range(2)]
        try:
            assert entered.wait(2)
            assert _read(world)["reason"] == "reader_limit_exceeded"
        finally: release.set()
        assert all(future.result(timeout=3)["status"] == "available" for future in futures)
    assert active[1] == 2 and active[0] == 0


@pytest.mark.parametrize("change", ["close", "service", "run", "expiry", "missing"])
def test_final_guard_retires_closed_replaced_run_or_expired_read_after_dump(world, monkeypatch, change):
    original = WorldSnapshot.model_dump
    def dump(snapshot, *args, **kwargs):
        result = original(snapshot, *args, **kwargs)
        if change == "close": world.service.close()
        elif change == "service": world.app.state.canonical_world_service = object()
        elif change == "run": world.service.producer = world.service.producer.model_copy(update={"run_id": "replacement-run"})
        elif change == "missing": world.service._current_snapshot = None
        else: world.clock[0] = 4_000_000
        return result
    monkeypatch.setattr(WorldSnapshot, "model_dump", dump)
    result = _read(world)
    assert result["status"] == "unknown"
    assert result["reason"] in {"world_closed", "world_owner_changed", "snapshot_not_fresh", "snapshot_unavailable"}


def test_newer_same_run_commit_can_leave_captured_snapshot_as_explicit_as_of_read(world, monkeypatch):
    original = WorldSnapshot.model_dump; changed = [False]
    def dump(snapshot, *args, **kwargs):
        result = original(snapshot, *args, **kwargs)
        if not changed[0]:
            changed[0] = True; world.clock[0] += 1000
            world.service.publish(0, [], metadata={"camera_id": "kitchen"})
        return result
    monkeypatch.setattr(WorldSnapshot, "model_dump", dump)
    result = _read(world)
    assert result["status"] == "available" and result["snapshot"] == original(world.snapshot, mode="json")
    assert world.service.current_snapshot() is not world.snapshot
    assert result["valid_until_us"] == world.snapshot.published_at_us + 2_000_000


def test_entity_and_source_preflights_refuse_before_model_dump(world, monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError("Oversized model must not be dumped")
    monkeypatch.setattr(WorldSnapshot, "model_dump", forbidden)
    entity = world.snapshot.entities[0]
    world.service._current_snapshot = world.snapshot.model_copy(update={"entities": (entity,) * 257})
    assert _read(world)["reason"] == "entity_limit_exceeded"
    excessive = entity.model_copy(update={"sources": entity.sources * 4097})
    world.service._current_snapshot = world.snapshot.model_copy(update={"entities": (excessive,)})
    assert _read(world)["reason"] == "source_evidence_limit_exceeded"


def test_two_mib_wire_bound_preserves_original_or_returns_small_unknown(world):
    payload = world.snapshot.model_dump(mode="json"); entity = payload["entities"][0]; source = copy.deepcopy(entity["sources"][0])
    source.update(camera_id="🟦" * 160, observation_id="🟦" * 200, world_frame_revision="🟦" * 200, calibration_revision="🟦" * 200)
    entity.update(world_frame_revision="🟦" * 200, calibration_revision="🟦" * 200, sources=[source] * 700)
    oversized = WorldSnapshot.model_validate(payload)
    assert api._bounded_model(oversized)
    world.service._current_snapshot = oversized
    result = _read(world)
    assert result["reason"] == "response_byte_limit_exceeded" and result["snapshot"] is None


def test_final_clock_callback_cannot_close_or_replace_owner_after_guard(world):
    calls = [0]
    def clock():
        calls[0] += 1
        if calls[0] == 3: world.service.close()
        return world.clock[0]
    world.service._clock_us = clock
    assert _read(world)["reason"] == "world_closed"


def test_valid_retained_quarantine_is_lossless_not_reinterpreted_as_settled_room_occupancy(world):
    run = world.snapshot.producer.run_id
    owners = [WorldAttributionClaim(run_id=run, physical_serial=serial, camera_id="kitchen", source_id=0,
        source_epoch=0, raw_id=serial, lifecycle_generation=1, lifecycle_started_frame=1, frame_id=2,
        observation_id=f"fixture:{serial}", acquired_at_s=1.9, clock_domain="synthetic-native", media_time_s=0.1,
        clock_basis="synthetic", household_state="resident", household_identity_key="contested-fixture",
        household_stable_id=9, household_visitor_generation=None, household_constraint="eligible") for serial in (1, 2)]
    payload = dict(status="quarantined", purpose="physical_only_quarantine", reason="settled_household_claim_conflict",
        run_id=run, association_revision=3, subject_ids=["resident:contested-fixture"], stable_ids=[9],
        owners=[owner.model_dump(mode="json") for owner in owners], retained_owners=[], component_entity_ids=[f"physical:{run}:{serial}" for serial in (1, 2)])
    record = WorldAttributionQuarantine(conflict_id=WorldAttributionQuarantine.digest(payload), **payload)
    data = world.snapshot.model_dump(mode="json"); base = data["entities"][0]; data["entities"] = []
    for serial in (1, 2):
        entity = copy.deepcopy(base)
        entity.update(entity_id=f"physical:{run}:{serial}", subject={"subject_id": f"quarantine:{run}:{serial}", "kind": "provisional", "generation": 0,
            "resident_uuid": None, "display_name": None, "stable_id": None}, attribution_quarantine=record.model_dump(mode="json"))
        data["entities"].append(entity)
    retained = WorldSnapshot.model_validate(data)
    world.service._current_snapshot = retained  # Typed synthetic retained fixture, not a runtime publication.
    result = _read(world)
    assert result["status"] == "available" and result["snapshot"] == retained.model_dump(mode="json")
    assert result["snapshot"]["entities"][0]["subject"]["kind"] == "provisional"
    assert result["snapshot"]["entities"][0]["attribution_quarantine"] == record.model_dump(mode="json")


def test_nonfinite_or_unsafe_retained_fields_and_failing_clock_are_unknown(world):
    world.service._current_snapshot = world.snapshot.model_copy(update={"observed_start_us": 2**53})
    assert _read(world)["reason"] == "snapshot_bound_exceeded"
    world.service._current_snapshot = world.snapshot.model_copy(update={"observed_start_us": True})
    assert _read(world)["reason"] == "snapshot_bound_exceeded"
    entity = world.snapshot.entities[0]
    position = entity.position.model_copy(update={"x": float("nan")})
    world.service._current_snapshot = world.snapshot.model_copy(update={"entities": (entity.model_copy(update={"position": position}),)})
    assert _read(world)["reason"] == "snapshot_bound_exceeded"
    world.service._current_snapshot = world.snapshot
    def failed_clock(): raise ValueError("synthetic clock failure")
    world.service._clock_us = failed_clock
    assert _read(world)["reason"] == "invalid_clock"
