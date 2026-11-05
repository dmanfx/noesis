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
from noesis_core import world_history as history
from noesis_core.contracts.base import ArtifactFingerprint, ProducerRef
from noesis_core.contracts.world_history import WorldHistoryRead
from noesis_core.journal import AsyncContractJournal, ContractJournal, ContractJournalError
from noesis_core.world_service import CanonicalWorldService, WorldArtifacts

TOKEN = "disposable-world-history-" + "h" * 48
HEADERS = {"Authorization": "Bearer " + TOKEN}


def track(frame=1):
    return {"camera_id": "kitchen", "tracker_id": 7, "frame_id": frame, "observed_at_us": 1_900_000 + frame,
        "capture_time_status": "estimated", "media_pts_ns": frame * 1000, "bbox": [10, 20, 30, 40], "image_size": [1920, 1080],
        "confidence": 0.91, "tracker_confidence": 0.88, "stable_id": 1, "identity_kind": "resident", "resident_uuid": "fixture-resident",
        "world": [-1.0, 0.0, 2.0], "world_valid": True, "world_frame": "backend_world_m", "world_frame_revision": "fixture-world",
        "world_transform_sha256": "d" * 64, "calibration_revision": "fixture-calibration", "world_quality": "good", "world_source": "pose_depth_fused",
        "zone": "Kitchen", "zone_source": "nvdsanalytics_roi", "zone_authoritative": True}


def service_for(journal, clock, run="history-run"):
    return CanonicalWorldService(producer=ProducerRef(runtime="test", instance_id="history-owner", run_id=run, software_revision="fixture"),
        artifacts=WorldArtifacts(calibration=ArtifactFingerprint(role="camera_calibration", sha256="a" * 64),
            model=ArtifactFingerprint(role="tracking_models", sha256="b" * 64), config=ArtifactFingerprint(role="runtime_config", sha256="c" * 64)),
        clock_us=lambda: clock[0], journal=journal)


@pytest.fixture
def world(tmp_path, monkeypatch):
    durable = ContractJournal(tmp_path / "private" / "world.sqlite", max_records=1000)
    clock = [2_000_000]; service = service_for(durable, clock)
    publication = service.publish(0, [track()], metadata={"camera_id": "kitchen"})
    assert publication.events
    app = FastAPI(); app.include_router(api.router); app.add_middleware(InternalBearerAuthMiddleware, token=TOKEN)
    app.state.canonical_world_service = service
    reader = history.WorldHistoryReader(); monkeypatch.setattr(api, "world_history_reader", reader)
    with TestClient(app) as client:
        value = SimpleNamespace(service=service, durable=durable, clock=clock, publication=publication, app=app, client=client, reader=reader)
        yield value
    if type(service._journal) is AsyncContractJournal and service._journal._failure is not None:
        with pytest.raises(ContractJournalError):
            service.close()
    else:
        service.close()
    if app.state.canonical_world_service is not service and isinstance(app.state.canonical_world_service, CanonicalWorldService):
        app.state.canonical_world_service.close()


def read(world, suffix=""):
    response = world.client.get(api.HISTORY_PATH + suffix, headers=HEADERS)
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "private, no-store"
    assert len(response.content) <= 256 * 1024
    result = response.json(); WorldHistoryRead.model_validate(result)
    return result


def append_events(world, count, start=100):
    original = world.publication.events[0].model_dump(mode="json")
    for index in range(start, start + count):
        event = copy.deepcopy(original); event.update(event_id=f"fixture-event-{index}", sequence=index)
        # Existing strict journal owns persistence; the reader creates no rows.
        world.durable.append_many([event, world.publication.snapshot.model_dump(mode="json")], recorded_at_us=2_000_001 + index)


def test_actual_world_service_persisted_events_private_auth_original_provenance_and_no_world_mutation(world):
    assert world.client.get(api.HISTORY_PATH).status_code == 401
    assert world.client.get(api.HISTORY_PATH, headers={"Authorization": "Bearer wrong"}).status_code == 401
    stored = world.durable.records(); expected = [r for r in stored if r.payload["contract"] == "noesis.world.event"]
    before = (world.service._revision, world.service._event_sequence, world.service.current_snapshot(), world.service._fusion)
    result = read(world)
    assert result["status"] == "available" and result["coverage"] == "retained_persisted_events_only"
    assert result["persistence"]["status"] == "synchronous"
    assert len(result["events"]) == len(expected)
    for actual, record in zip(result["events"], reversed(expected)):
        assert actual == {"sequence": record.sequence, "recorded_at_us": record.recorded_at_us, "previous_sha256": record.previous_sha256,
            "record_sha256": record.record_sha256, "payload": record.payload}
        assert actual["payload"]["producer"]["run_id"] == "history-run"
        assert "room_id" not in actual["payload"] and "stale_after_us" not in actual["payload"]
    assert result["inspection"]["created_at_us"] == world.clock[0]
    assert result["inspection"]["expires_at_us"] == world.clock[0] + 30_000_000
    world.clock[0] += 10_000
    assert read(world)["events"] == result["events"]
    assert before == (world.service._revision, world.service._event_sequence, world.service.current_snapshot(), world.service._fusion)
    assert stored == world.durable.records() and TOKEN not in json.dumps(result)


def test_sql_event_eligibility_before_limit_exact_backwards_pages_fixed_anchor_exclude_new_appends(world):
    append_events(world, 5)
    head = read(world, "?limit=2"); assert len(head["events"]) == 2 and head["next_cursor"]
    original_inspection = head["inspection"]; original_sequences = [r.sequence for r in world.durable.records() if r.payload["contract"] == "noesis.world.event"]
    append_events(world, 1, start=999)
    pages = [head]; token = head["next_cursor"]
    while token:
        page = read(world, "?limit=2&cursor=" + token); assert page["inspection"] == original_inspection
        pages.append(page); token = page["next_cursor"]
    assert [r["sequence"] for p in pages for r in p["events"]] == list(reversed(original_sequences))
    assert all(len(p["events"]) <= 2 for p in pages)
    assert read(world)["inspection"]["anchor_sequence"] > original_inspection["anchor_sequence"]


def test_restart_keeps_original_events_but_a_new_reader_never_reconstructs_inspection_cursor(world, monkeypatch):
    append_events(world, 2); original = read(world, "?limit=1")
    path = world.durable.path; world.service.close()
    durable = ContractJournal(path, max_records=1000); replacement = service_for(durable, world.clock, run="new-process-run")
    world.app.state.canonical_world_service = replacement
    monkeypatch.setattr(api, "world_history_reader", history.WorldHistoryReader())
    assert read(world, "?cursor=" + original["next_cursor"])["reason"] == "inspection_expired"
    current = read(world); assert current["events"][0] == original["events"][0]
    assert current["events"][0]["payload"]["producer"]["run_id"] == "history-run"
    assert replacement.current_snapshot() is None


def test_pruned_floor_and_original_owner_clock_expiry_refuse_existing_inspection(world):
    append_events(world, 4); head = read(world, "?limit=1")
    world.durable.max_records = 8; append_events(world, 6, start=900)
    result = read(world, "?limit=1&cursor=" + head["next_cursor"])
    assert result["status"] == "unavailable" and result["reason"] == "inspection_expired" and not result["events"]
    fresh = read(world, "?limit=1"); world.clock[0] += 30_000_000
    assert read(world, "?cursor=" + fresh["next_cursor"])["reason"] == "inspection_expired"


@pytest.mark.parametrize("corruption", ["checksum", "payload", "schema"])
def test_hash_payload_schema_corruption_refuses_whole_page_without_partial_data(world, corruption):
    connection = sqlite3.connect(world.durable.path)
    if corruption == "checksum": connection.execute("UPDATE records SET record_sha256=? WHERE sequence=(SELECT MAX(sequence) FROM records)", ("e" * 64,))
    elif corruption == "payload": connection.execute("UPDATE records SET payload_json='{}' WHERE sequence=(SELECT MAX(sequence) FROM records)")
    else: connection.execute("CREATE TABLE alien(value TEXT)")
    connection.commit(); connection.close()
    result = read(world); assert result["status"] == "unavailable" and result["events"] == [] and result["inspection"] is None
    assert str(world.durable.path) not in json.dumps(result)


def test_async_dequeued_and_queued_batches_are_explicit_partial_without_flush_or_wait(world, monkeypatch):
    original = world.durable.append_entries; entered, release = threading.Event(), threading.Event()
    def hold(entries):
        entered.set(); assert release.wait(3); return original(entries)
    monkeypatch.setattr(world.durable, "append_entries", hold)
    asynchronous = AsyncContractJournal(world.durable, max_pending_batches=4, max_transaction_batches=1)
    world.service._journal = asynchronous
    world.service.publish(0, [track(2)], metadata={"camera_id": "kitchen"}); assert entered.wait(2)
    def forbidden(): raise AssertionError("Read cannot flush queued persistence")
    monkeypatch.setattr(asynchronous, "flush", forbidden)
    try:
        result = read(world)
        assert result["status"] == "partial" and result["persistence"]["inflight_batches"] == 1
        assert result["persistence"]["pending_batches"] == 0
        world.service.publish(0, [track(3)], metadata={"camera_id": "kitchen"})
        result = read(world); assert result["status"] == "partial" and result["persistence"]["pending_batches"] >= 1
        assert result["events"]
    finally:
        release.set()


def test_async_failure_and_rejected_cohorts_remain_partial_with_original_durable_rows(world, monkeypatch):
    failed = threading.Event()
    def fail(_entries): failed.set(); raise ContractJournalError("disposable private worker failure")
    monkeypatch.setattr(world.durable, "append_entries", fail)
    asynchronous = AsyncContractJournal(world.durable); world.service._journal = asynchronous
    world.service.publish(0, [track(2)], metadata={"camera_id": "kitchen"}); assert failed.wait(2)
    for _ in range(100):
        if asynchronous._failure is not None: break
        time.sleep(0.001)
    world.service.publish(0, [track(3)], metadata={"camera_id": "kitchen"})
    result = read(world)
    assert result["status"] == "partial" and result["persistence"]["status"] == "failed"
    assert result["persistence"]["rejected_cohorts"] >= 1 and result["persistence"]["error_reported"] is True
    assert result["events"] and "private worker failure" not in json.dumps(result)


def test_media_and_journal_writer_locks_are_never_held_or_waited_for_during_request_io_and_clock(world, monkeypatch):
    checked = []; original = world.durable.inspect_world_events
    def check_lock():
        acquired = world.service._lock.acquire(blocking=False); checked.append(acquired)
        if acquired: world.service._lock.release()
    def inspect(**kwargs):
        thread = threading.Thread(target=check_lock); thread.start(); thread.join(1)
        return original(**kwargs)
    monkeypatch.setattr(world.durable, "inspect_world_events", inspect)
    original_clock = world.service._clock_us
    def clock():
        thread = threading.Thread(target=check_lock); thread.start(); thread.join(1); return original_clock()
    world.service._clock_us = clock
    entered, release = threading.Event(), threading.Event()
    def hold_writer():
        with world.durable._lock: entered.set(); release.wait(3)
    thread = threading.Thread(target=hold_writer); thread.start(); assert entered.wait(2)
    try:
        start = time.monotonic(); assert read(world)["status"] == "available"; assert time.monotonic() - start < 1
    finally: release.set(); thread.join(2)
    assert checked and all(checked)


def test_one_reader_and_contended_world_health_locks_refuse_without_waiting(world, monkeypatch):
    entered, release = threading.Event(), threading.Event(); original = world.durable.inspect_world_events
    def hold(**kwargs): entered.set(); assert release.wait(3); return original(**kwargs)
    monkeypatch.setattr(world.durable, "inspect_world_events", hold)
    with ThreadPoolExecutor(max_workers=1) as executor:
        active = executor.submit(read, world)
        try:
            assert entered.wait(2); start = time.monotonic()
            assert read(world)["reason"] == "reader_limit_exceeded"; assert time.monotonic() - start < 1
        finally: release.set()
        assert active.result(timeout=3)["status"] == "available"
    entered.clear(); release.clear()
    def hold_world():
        with world.service._lock: entered.set(); release.wait(3)
    thread = threading.Thread(target=hold_world); thread.start(); assert entered.wait(2)
    try: assert read(world)["reason"] == "world_busy"
    finally: release.set(); thread.join(2)


@pytest.mark.parametrize("bound", ["rows", "bytes"])
def test_scan_preflight_refuses_before_any_full_json_materialization(world, monkeypatch, bound):
    def forbidden(*args, **kwargs): raise AssertionError("Preflight must precede full JSON validation")
    monkeypatch.setattr(world.durable, "_validated_records", forbidden)
    monkeypatch.setattr(history, "MAX_SCAN_ROWS" if bound == "rows" else "MAX_SCAN_BYTES", 1)
    assert read(world)["reason"] == "history_scan_bound_exceeded"


def test_deadline_response_bytes_cursor_capacity_and_strict_request_bounds_are_finite(world, monkeypatch):
    for suffix in ["?limit=0", "?limit=51", "?limit=01", "?limit=1&limit=2", "?other=1", "?cursor=bad", "?cursor=" + "x" * 300]:
        assert world.client.get(api.HISTORY_PATH + suffix, headers=HEADERS).status_code == 400
    assert world.client.request("GET", api.HISTORY_PATH, headers=HEADERS, content=b"{}").status_code == 400
    assert world.client.post(api.HISTORY_PATH, headers=HEADERS).status_code == 405
    with monkeypatch.context() as scoped:
        scoped.setattr(history, "READ_DEADLINE_SECONDS", 0)
        assert read(world)["reason"] == "history_deadline_exceeded"
    with monkeypatch.context() as scoped:
        scoped.setattr(history, "MAX_HISTORY_BYTES", 500)
        assert read(world)["reason"] == "response_byte_limit_exceeded"
    append_events(world, 2)
    with monkeypatch.context() as scoped:
        scoped.setattr(history, "MAX_INSPECTIONS", 1)
        assert read(world, "?limit=1")["next_cursor"]
        assert read(world, "?limit=1")["reason"] == "inspection_capacity_exceeded"


@pytest.mark.parametrize("replacement", ["service", "journal", "path"])
def test_replacement_during_serialization_refuses_retired_source(world, monkeypatch, replacement):
    original = history.encode_world_history; alternate = ContractJournal(world.durable.path.parent / "replacement.sqlite")
    def encode(value):
        result = original(value)
        if value.status != "unavailable":
            if replacement == "service": world.app.state.canonical_world_service = object()
            elif replacement == "journal": world.service._journal = alternate
            else:
                world.durable.path.rename(world.durable.path.with_name("old.sqlite"))
                # Copy no live state: merely a different private inode at the exact
                # configured name, which the original binding must refuse.
                world.durable.path.write_bytes(b"disposable changed inode"); world.durable.path.chmod(0o600)
        return result
    monkeypatch.setattr(history, "encode_world_history", encode)
    try:
        result = read(world); assert result["status"] == "unavailable" and not result["events"]
    finally:
        world.service._journal = world.durable; alternate.close()
        if replacement == "path":
            world.durable.path.unlink(); world.durable.path.with_name("old.sqlite").rename(world.durable.path)


def test_disabled_closed_and_empty_known_journal_never_fabricate_events(world):
    world.service._journal = None; assert read(world)["reason"] == "persistence_disabled"
    world.service._journal = world.durable
    other = ContractJournal(world.durable.path.parent / "empty.sqlite")
    world.service._journal = other
    try:
        result = read(world); assert result["status"] == "available" and result["events"] == []
        assert result["inspection"]["anchor_sequence"] == -1
    finally: other.close(); world.service._journal = world.durable
    world.service.close(); assert read(world)["reason"] == "world_closed"


def test_large_retained_journal_reads_original_pages_without_full_payload_scan(world, monkeypatch):
    # Retention size is independent of the finite JSON work allowed per request.
    world.durable.max_records = 12_000
    payload = world.durable.records()[0].payload
    world.durable.append_many([payload] * 10_010, recorded_at_us=world.clock[0])
    monkeypatch.setattr(world.durable, '_validated_records', lambda *a, **k: pytest.fail('request attempted full journal validation'))
    result = read(world, '?limit=1')
    assert result['status'] == 'available'
    assert len(result['events']) == 1
    records, first, previous, omitted = world.durable.inspect_world_snapshots(deadline=time.monotonic()+0.75)
    assert len(records) == 1 and first == 0 and len(previous) == 64 and not omitted
    # A rewritten, self-consistent payload checksum is still not a committed row.
    sequence = result['events'][0]['sequence']
    connection = sqlite3.connect(world.durable.path)
    connection.execute('UPDATE records SET recorded_at_us=recorded_at_us+1 WHERE sequence=?', (sequence,))
    connection.commit(); connection.close()
    assert read(world)['reason'] == 'journal_integrity_unavailable'
