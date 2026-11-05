from __future__ import annotations

import copy
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from noesis.server import world_read_api as api
from noesis.server.internal_auth import InternalBearerAuthMiddleware
from noesis_core import temporal_relationships as temporal
from noesis_core.contracts.temporal_relationships import TemporalRelationshipsRead
from noesis_core.contracts.world import WorldSnapshot
from noesis_core.journal import AsyncContractJournal, ContractJournal, ContractJournalError
from tests.test_world_history import service_for, track

TOKEN = "disposable-temporal-owner-" + "t" * 48
HEADERS = {"Authorization": "Bearer " + TOKEN}


@pytest.fixture
def world(tmp_path, monkeypatch):
    journal = ContractJournal(tmp_path / "private" / "world.sqlite", max_records=1000)
    clock = [2_000_000]; service = service_for(journal, clock)
    publication = service.publish(0, [track()], metadata={"camera_id": "kitchen"})
    app = FastAPI(); app.include_router(api.router); app.add_middleware(InternalBearerAuthMiddleware, token=TOKEN)
    app.state.canonical_world_service = service
    monkeypatch.setattr(api, "temporal_relationships_reader", temporal.TemporalRelationshipsReader())
    with TestClient(app) as client:
        yield SimpleNamespace(service=service, journal=journal, clock=clock, publication=publication, app=app, client=client)
    try:
        service.close()
    except ContractJournalError:
        assert type(service._journal) is AsyncContractJournal and service._journal._failure is not None
    replacement = app.state.canonical_world_service
    if replacement is not service and hasattr(replacement, "close"):
        replacement.close()


def publish(world, frame=2, observed=1_905_001):
    world.clock[0] += 100_000
    row = track(frame); row["observed_at_us"] = observed
    publication = world.service.publish(0, [row], metadata={"camera_id": "kitchen"})
    return publication.snapshot


def inject(world, mutate=None):
    data = world.service.current_snapshot().model_dump(mode="json")
    data["sequence"] += 1; data["snapshot_id"] = "fixture-injected-" + str(data["sequence"])
    data["published_at_us"] += 100_000
    entity = data["entities"][0]
    entity["observed_at_us"] += 5000
    for source in entity["sources"]:
        source["observed_at_us"] = entity["observed_at_us"]
    if mutate:
        mutate(data)
    snapshot = WorldSnapshot.model_validate(data)
    world.journal.append_many([snapshot.model_dump(mode="json")], recorded_at_us=snapshot.published_at_us)
    world.service._current_snapshot = snapshot; world.clock[0] = snapshot.published_at_us
    return snapshot


def read(world, suffix=""):
    response = world.client.get(api.TEMPORAL_RELATIONSHIPS_PATH + suffix, headers=HEADERS)
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "private, no-store" and len(response.content) <= 128 * 1024
    data = response.json(); TemporalRelationshipsRead.model_validate(data)
    return data


def span(data):
    return data["entities"][0]["observed_assignment_span"]


def test_actual_world_service_journal_private_api_two_distinct_reports_exact_span_and_original_evidence_never_dwell(world):
    assert world.client.get(api.TEMPORAL_RELATIONSHIPS_PATH).status_code == 401
    assert span(read(world))["state"] == "unknown"
    head = publish(world)
    before = (world.service._revision, world.service._event_sequence, world.service.current_snapshot(), world.journal.records())
    result = read(world); value = span(result)
    assert result["status"] == "available" and result["source"]["scope"] == "persisted_as_of"
    assert value["state"] == "known" and value["room_id"] == "Kitchen"
    assert value["observation_count"] == value["snapshot_report_count"] == 2
    assert value["first_observed_at_us"] == 1_900_001 and value["last_observed_at_us"] == 1_905_001
    assert value["span_us"] == 5000 and value["span_us"] != head.published_at_us - world.publication.snapshot.published_at_us
    records = {row.sequence: row for row in world.journal.records()}
    for evidence in value["evidence"]:
        record = records[evidence["journal_sequence"]]
        assert evidence["record_sha256"] == record.record_sha256 and evidence["previous_sha256"] == record.previous_sha256
        assert evidence["recorded_at_us"] == record.recorded_at_us
        assert evidence["snapshot_id"] == record.payload["snapshot_id"] and evidence["producer"] == record.payload["producer"]
        assert evidence["sources"] == record.payload["entities"][0]["sources"]
    assert result["entities"][0]["dwell"] == {"state": "unknown", "reason": "point_observations_do_not_prove_continuous_occupancy", "duration_us": None}
    assert before == (world.service._revision, world.service._event_sequence, world.service.current_snapshot(), world.journal.records())
    assert result["valid_until_us"] <= head.published_at_us + 2_000_000


def test_duplicate_observation_publications_never_extend_observed_span(world):
    publish(world, observed=1_900_001)
    value = span(read(world)); assert value["state"] == "unknown" and value["observation_count"] == 1
    assert value["snapshot_report_count"] == 2 and value["span_us"] is None
    publish(world, frame=3, observed=1_905_001)
    value = span(read(world)); assert value["state"] == "known" and value["observation_count"] == 2
    assert value["snapshot_report_count"] == 3 and value["span_us"] == 5000


@pytest.mark.parametrize(("mode", "reason"), [
    ("room", "room_assignment_changed"), ("binding", "identity_or_frame_binding_changed"),
    ("identity", "identity_or_frame_binding_changed"), ("held", "entity_held"), ("conflict", "entity_conflict"),
    ("sequence", "snapshot_sequence_gap"), ("clock", "snapshot_clock_regression"), ("observation_clock", "entity_observation_clock_regression"),
    ("gap", "observation_gap"), ("missing", "entity_missing"),
])
def test_report_boundaries_break_instead_of_synthesizing_arrivals_or_continuity(world, mode, reason):
    def mutate(data):
        entity = data["entities"][0]
        if mode == "room":
            entity["room_id"] = "Other exact room"
            for source in entity["sources"]: source["zone"] = entity["room_id"]
        elif mode == "binding":
            entity["world_frame_revision"] = "next-frame"
            for source in entity["sources"]: source["world_frame_revision"] = "next-frame"
        elif mode == "identity": entity["subject"]["generation"] += 1
        elif mode == "sequence": data["sequence"] += 1
        elif mode == "clock": data["published_at_us"] = world.publication.snapshot.published_at_us
        elif mode == "observation_clock":
            entity["observed_at_us"] = 1_899_000
            for source in entity["sources"]: source["observed_at_us"] = 1_899_000
        elif mode == "gap":
            data["published_at_us"] += 2_000_000
            entity["observed_at_us"] += 2_000_000; entity["stale_after_us"] += 2_000_000
            data["observed_start_us"] = data["observed_end_us"] = entity["observed_at_us"]
            for source in entity["sources"]: source["observed_at_us"] = entity["observed_at_us"]
    if mode in {"held", "conflict", "missing"}:
        # Put the break between an older measured report and a newer measured
        # report; a later valid report cannot reach across the unsettled gap.
        def middle(data):
            if mode == "missing": data["entities"] = []
            elif mode == "held": data["entities"][0]["lifecycle"] = "held"
            else: data["entities"][0].update(conflict=True, conflict_reason="Original recorded disagreement")
        broken = inject(world, middle)
        base = world.publication.snapshot.model_dump(mode="json")
        base.update(sequence=broken.sequence, snapshot_id=broken.snapshot_id, published_at_us=broken.published_at_us)
        world.service._current_snapshot = WorldSnapshot.model_validate(base)
    inject(world, mutate)
    result = read(world); value = span(result)
    assert value["state"] == "unknown" and value["boundary_reason"] == reason
    assert value["span_us"] is None and result["entities"][0]["dwell"]["duration_us"] is None


def test_head_held_conflict_quarantine_and_expiry_never_make_known_spans(world):
    publish(world)
    original = world.service.current_snapshot()
    for mode in ["held", "conflict", "quarantine"]:
        if mode == "quarantine":
            from tests.test_spatial_relationships import quarantined_snapshot
            # Use the actual strict original attribution fixture, without
            # inventing a settled subject for the quarantined component.
            fake = SimpleNamespace(snapshot=original)
            snapshot = quarantined_snapshot(fake)
            world.journal.append_many([snapshot.model_dump(mode="json")], recorded_at_us=snapshot.published_at_us)
            world.service._current_snapshot = snapshot
        else:
            world.service._current_snapshot = original
            inject(world, lambda data: data["entities"][0].update({"lifecycle": "held"} if mode == "held"
                else {"conflict": True, "conflict_reason": "Recorded conflict"}))
        result = read(world)
        assert all(row["observed_assignment_span"]["state"] == "unknown" for row in result["entities"])
    world.service._current_snapshot = original; inject(world)
    world.clock[0] = world.service.current_snapshot().entities[0].stale_after_us
    result = read(world)
    assert result["status"] == "unknown" or span(result)["state"] == "unknown"


def test_room_change_starts_a_new_observed_tail_without_inventing_transition_or_physical_dwell(world):
    def other(data):
        entity = data["entities"][0]; entity["room_id"] = "Other exact room"
        for source in entity["sources"]: source["zone"] = "Other exact room"
    inject(world, other); inject(world)
    value = span(read(world)); assert value["state"] == "known" and value["room_id"] == "Other exact room"
    assert value["observation_count"] == 2 and value["boundary_reason"] == "room_assignment_changed"


def test_run_change_restart_retention_and_persisted_head_lag_are_explicit(world):
    publish(world); path = world.journal.path
    world.service.close(); reopened = ContractJournal(path, max_records=1000)
    replacement = service_for(reopened, world.clock, run="replacement-run"); world.app.state.canonical_world_service = replacement
    assert read(world)["status"] == "unknown"
    replacement.publish(0, [track()], metadata={"camera_id": "kitchen"})
    result = read(world); assert span(result)["state"] == "unknown" and span(result)["boundary_reason"] == "producer_changed"
    # The same owner can inspect its last durable head while a newer current
    # publication is queued; that is explicitly partial and persisted-as-of.
    world.app.state.canonical_world_service = world.service
    world.service._closed = False; world.service._journal = reopened; world.journal = reopened
    world.service.producer = replacement.producer; world.service._current_snapshot = replacement.current_snapshot()
    current = inject(world); newer = current.model_copy(update={"sequence": current.sequence + 1, "snapshot_id": "not-yet-durable"})
    world.service._current_snapshot = newer
    result = read(world); assert result["status"] == "partial" and result["source"]["head_matches_current_at_read"] is False
    reopened.max_records = 1; reopened.append_many([current.model_dump(mode="json")], recorded_at_us=current.published_at_us)
    world.service._current_snapshot = current
    result = read(world); assert result["status"] == "partial" and result["source"]["retention_pruned"] is True
    assert span(result)["state"] == "unknown"
    world.app.state.canonical_world_service = replacement


def test_queued_inflight_persistence_is_partial_without_flush_and_corruption_is_whole_unknown(world, monkeypatch):
    publish(world); entered, release = threading.Event(), threading.Event(); original = world.journal.append_entries
    def held(entries): entered.set(); assert release.wait(3); return original(entries)
    monkeypatch.setattr(world.journal, "append_entries", held)
    asynchronous = AsyncContractJournal(world.journal); world.service._journal = asynchronous
    monkeypatch.setattr(asynchronous, "flush", lambda: pytest.fail("Read must never flush"))
    publish(world, frame=3, observed=1_910_001); assert entered.wait(2)
    try:
        result = read(world); assert result["status"] == "partial" and result["persistence"]["inflight_batches"] >= 1
        assert result["source"]["head_matches_current_at_read"] is False
    finally: release.set()
    # Complete the disposable worker before directly corrupting its private DB.
    asynchronous._queue.join()
    connection = sqlite3.connect(world.journal.path)
    connection.execute("UPDATE records SET record_sha256=? WHERE sequence=(SELECT MAX(sequence) FROM records)", ("f" * 64,)); connection.commit(); connection.close()
    assert read(world)["status"] == "unknown"


def test_snapshot_eligibility_window_limits_source_output_and_deadline_are_bounded(world, monkeypatch):
    for _ in range(34): inject(world)
    result = read(world)
    assert result["status"] == "partial" and result["source"]["inspected_snapshot_count"] == 32
    assert result["source"]["earlier_snapshots_omitted"] is True
    assert len(span(result)["evidence"]) <= 32
    with monkeypatch.context() as scoped:
        scoped.setattr(temporal, "READ_DEADLINE_SECONDS", 0)
        assert read(world)["reason"] == "temporal_read_deadline_exceeded"
    with monkeypatch.context() as scoped:
        scoped.setattr(temporal, "MAX_TEMPORAL_BYTES", 500)
        assert read(world)["reason"] == "temporal_response_byte_limit_exceeded"
    assert world.client.get(api.TEMPORAL_RELATIONSHIPS_PATH + "?limit=1", headers=HEADERS).status_code == 400
    assert world.client.request("GET", api.TEMPORAL_RELATIONSHIPS_PATH, headers=HEADERS, content=b"{}").status_code == 400
    assert world.client.post(api.TEMPORAL_RELATIONSHIPS_PATH, headers=HEADERS).status_code == 405


def test_media_writer_lock_nonblocking_capture_and_one_reader_limit(world, monkeypatch):
    publish(world); entered, release = threading.Event(), threading.Event(); original = world.journal.inspect_world_snapshots
    def hold(**kwargs):
        checked = []
        def check():
            acquired = world.service._lock.acquire(blocking=False); checked.append(acquired)
            if acquired: world.service._lock.release()
        thread = threading.Thread(target=check); thread.start(); thread.join(1); assert checked == [True]
        entered.set(); assert release.wait(3); return original(**kwargs)
    monkeypatch.setattr(world.journal, "inspect_world_snapshots", hold)
    with ThreadPoolExecutor(max_workers=1) as executor:
        active = executor.submit(read, world)
        try:
            assert entered.wait(2); assert read(world)["reason"] == "temporal_reader_limit_exceeded"
        finally: release.set()
        assert active.result(timeout=3)["status"] == "available"
    monkeypatch.setattr(world.journal, "inspect_world_snapshots", original)
    entered.clear(); release.clear()
    def lock_writer():
        with world.journal._lock: entered.set(); release.wait(3)
    thread = threading.Thread(target=lock_writer); thread.start(); assert entered.wait(2)
    try: assert read(world)["status"] == "available"
    finally: release.set(); thread.join(2)


@pytest.mark.parametrize("mode", ["expiry", "owner"])
def test_original_head_expiry_and_owner_replacement_during_encoding_refuse_delivery(world, monkeypatch, mode):
    publish(world); original = temporal.encode_temporal_relationships
    def expire(value):
        body = original(value)
        if value.status != "unknown":
            if mode == "expiry": world.clock[0] = value.valid_until_us
            else: world.app.state.canonical_world_service = object()
        return body
    monkeypatch.setattr(temporal, "encode_temporal_relationships", expire)
    assert read(world)["reason"] == ("temporal_head_expired_during_read" if mode == "expiry" else "world_owner_changed")


def test_entity_overflow_refuses_whole_result_instead_of_selecting_a_subset(world):
    def overflow(data):
        base = data["entities"][0]; data["entities"] = []
        for index in range(13):
            entity = copy.deepcopy(base); entity["entity_id"] = f"fixture-entity-{index}"
            entity["subject"].update(subject_id=f"fixture-subject-{index}", stable_id=index + 1)
            data["entities"].append(entity)
    inject(world, overflow)
    result = read(world); assert result["status"] == "unknown" and result["reason"] == "temporal_entity_limit_exceeded"
    assert result["entities"] == []


def test_snapshot_source_byte_preflight_precedes_full_payload_materialization(world, monkeypatch):
    publish(world)
    connection = sqlite3.connect(world.journal.path)
    connection.execute("UPDATE records SET payload_json=? WHERE contract='noesis.world.snapshot'", ("x" * 1_100_000,))
    connection.commit(); connection.close()
    monkeypatch.setattr(world.journal, "_validated_records", lambda *args, **kwargs: pytest.fail("Oversized source must not be materialized"))
    result = read(world); assert result["status"] == "unknown" and result["reason"] == "temporal_snapshot_byte_limit_exceeded"
