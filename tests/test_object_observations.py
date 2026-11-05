from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from noesis.server.object_observations import PATH, router
from noesis.server.internal_auth import InternalBearerAuthMiddleware
from noesis_core.object_observations import ObjectObservationService, install_object_observation_owner, clear_object_observation_owner, close_object_observation_owner
from noesis_core.startup_lifecycle import RuntimeStartupTransaction

TOKEN = "synthetic-objects-private-" + "x" * 48


def obj(**changes):
    return SimpleNamespace(class_id=56, label="chair", confidence=.85,
        rect_params=SimpleNamespace(left=10., top=20., width=30., height=40.), **changes)


@pytest.fixture
def service():
    stamp = [1_000_000]
    mono = [10.]
    value = ObjectObservationService("synthetic-run", monotonic=lambda: mono[0], clock_us=lambda: stamp[0])
    yield SimpleNamespace(value=value, stamp=stamp, mono=mono)
    value.close()
    assert not value._worker.is_alive()
    assert value._queue.unfinished_tasks == 0


def capture(service, *, camera="camera-a", frame_id=1):
    return service.begin_frame(SimpleNamespace(pipeline_width=640, pipeline_height=480, source_width=1920, source_height=1080),
        camera_id=camera, source_id=3, source_epoch=2, frame_id=frame_id,
        temporal={"observed_at_us": 1_000_000, "captured_at_us": 1_000_000, "capture_time_status": "estimated", "media_pts_ns": 1234567})


def settled(service):
    deadline = time.monotonic() + 2
    while service._queue.unfinished_tasks and time.monotonic() < deadline:
        time.sleep(.002)
    assert service._queue.unfinished_tasks == 0
    assert service._worker.is_alive()


def test_original_detector_metadata_private_read_original_clocks_and_expiry_no_renewal(service):
    cap = capture(service.value); source = obj(); cap.capture(source, 7); assert cap.submit(); assert not cap.submit()
    source.label = "changed SDK lifetime"; settled(service.value)
    app = FastAPI(); app.include_router(router); app.add_middleware(InternalBearerAuthMiddleware, token=TOKEN)
    app.state.object_observation_service = service.value
    with TestClient(app) as client:
        assert client.get(PATH).status_code == 401
        assert client.get(PATH, headers={"Authorization": "Bearer wrong"}).status_code == 401
        headers = {"Authorization": "Bearer " + TOKEN}
        response = client.get(PATH, headers=headers); assert response.status_code == 200
        assert response.headers["cache-control"] == "private, no-store"
        body = response.json(); assert body["status"] == "available"; frame = body["cameras"][0]
        row = frame["observations"][0]
        assert row["label"] == "chair" and row["class_id"] == 56 and row["confidence"] == .85
        assert row["observation_id"] == "synthetic-run:camera-a:3:2:1:7"
        assert frame["coordinate_frame"] == "pipeline_image_px" and frame["image_width"] == 640 and frame["source_image_width"] == 1920
        assert frame["media_pts_ns"] == 1234567 and frame["observed_at_us"] == frame["captured_at_us"] == 1_000_000
        assert frame["valid_until_us"] == 6_000_000 and frame["capture_time_status"] == "estimated"
        assert not {"tracker_id", "world", "room_id", "scene", "object_id"}.intersection(row)
        service.mono[0] += 1; service.stamp[0] += 1_000_000
        assert client.get(PATH, headers=headers).json()["cameras"][0] == frame
        assert client.get(PATH + "?query=chair", headers=headers).status_code == 400
        service.mono[0] += 4
        expired = client.get(PATH, headers=headers).json(); assert expired["reason"] == "expired" and expired["cameras"] == []
        del app.state.object_observation_service
        assert client.get(PATH, headers=headers).json()["reason"] == "unbound"


def test_person_tracked_only_nonfinite_bbox_missing_label_and_32_copy_cap_are_explicit(service):
    cap = capture(service.value)
    person = obj(); person.class_id = 0; cap.capture(person, 0)
    for index, change in enumerate([{"confidence": -.1}, {"confidence": float("nan")}, {"label": ""}, {"class_id": 80},
        {"rect_params": SimpleNamespace(left=-1., top=0., width=1., height=1.)}], 1):
        item = obj()
        for key, value in change.items(): setattr(item, key, value)
        cap.capture(item, index)
    for index in range(40): cap.capture(obj(), index + 10)
    assert cap.inspected == 32 and len(cap.rows) == 27; cap.submit(); settled(service.value)
    state = service.value.snapshot(); frame = state["cameras"][0]
    assert state["status"] == "partial" and frame["invalid_count"] == 5 and frame["omitted_count"] == 13
    assert len(frame["observations"]) == 27


def test_queue_full_is_nonblocking_invalidates_only_optional_object_freshness(service):
    first = capture(service.value); first.capture(obj(), 0); first.submit(); settled(service.value)
    assert service.value.snapshot()["status"] == "available"
    original = service.value._queue.put_nowait
    def full(_frame):
        import queue
        raise queue.Full
    service.value._queue.put_nowait = full
    dropped = capture(service.value, frame_id=2)
    assert not dropped.submit()
    assert service.value.snapshot()["reason"] == "queue_dropped"
    service.value._queue.put_nowait = original
    fresh = capture(service.value, frame_id=3); fresh.capture(obj(), 0); fresh.submit(); settled(service.value)
    assert service.value.snapshot()["cameras"][0]["frame_id"] == 3
    assert service.value.snapshot()["dropped_frames"] == 1


def test_camera_slots_rows_bytes_and_late_worker_cannot_renew(service):
    for camera in range(16):
        cap = capture(service.value, camera=f"camera-{camera}")
        for index in range(32): cap.capture(obj(), index)
        cap.submit(); settled(service.value)
    state = service.value.snapshot()
    assert len(state["cameras"]) <= 16
    assert sum(len(frame["observations"]) for frame in state["cameras"]) <= 512
    import json
    assert len(json.dumps(state).encode()) <= 256 * 1024
    held = capture(service.value, camera="held")
    held.capture(obj(), 0); service.mono[0] += 5; held.submit(); settled(service.value)
    assert service.value.snapshot()["cameras"] == []


def test_transaction_abort_and_exact_cleanup_preserve_replacement_owner():
    pipeline, app = SimpleNamespace(), FastAPI()
    transaction = RuntimeStartupTransaction(on_ambiguous=lambda reason: pytest.fail(reason))
    service = install_object_observation_owner(transaction, pipeline, app, "startup-run")
    assert pipeline.object_observation_service is app.state.object_observation_service is service
    replacement = object(); app.state.object_observation_service = replacement
    receipt = transaction.abort_reversible()
    assert receipt.completed and not service._worker.is_alive()
    assert not hasattr(pipeline, "object_observation_service") and app.state.object_observation_service is replacement
    normal = ObjectObservationService("normal-run"); pipeline.object_observation_service = normal; app.state.object_observation_service = normal
    close_object_observation_owner(pipeline, app, normal)
    assert not hasattr(pipeline, "object_observation_service") and not hasattr(app.state, "object_observation_service")
    assert normal.snapshot()["reason"] == "closed"


def test_private_read_rechecks_exact_owner_after_serialization(service, monkeypatch):
    from noesis.server import object_observations as api
    cap = capture(service.value); cap.capture(obj(), 0); cap.submit(); settled(service.value)
    app = FastAPI(); app.include_router(router); app.add_middleware(InternalBearerAuthMiddleware, token=TOKEN)
    app.state.object_observation_service = service.value
    original = api._encode
    def replace(value):
        response = original(value)
        if value["status"] == "available": app.state.object_observation_service = object()
        return response
    monkeypatch.setattr(api, "_encode", replace)
    with TestClient(app) as client:
        response = client.get(PATH, headers={"Authorization": "Bearer " + TOKEN})
        assert response.status_code == 200
        assert response.json()["reason"] == "owner_changed" and response.json()["cameras"] == []


@pytest.mark.parametrize("stamp", [999999, 6000000, 6000001])
def test_original_wall_clock_bounds_fail_closed_without_renewing_monotonic_ttl(service, stamp):
    cap = capture(service.value); cap.capture(obj(), 0); cap.submit(); settled(service.value)
    service.stamp[0] = stamp
    assert service.value.snapshot()["reason"] == "invalid_clock"
    assert service.value.snapshot()["cameras"] == []
    service.stamp[0] = 1_000_000
    assert service.value.snapshot()["cameras"][0]["valid_until_us"] == 6_000_000
    service.mono[0] += 5
    assert service.value.snapshot()["reason"] == "expired"
