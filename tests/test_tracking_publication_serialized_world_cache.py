from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest


class _Admission:
    def __init__(self, owner, messages):
        self.receipt = SimpleNamespace(
            submission_id=1,
            message_count=len(messages),
        )
        self._owner = owner
        self._messages = messages

    def commit_then_release(self, commit):
        result = commit()
        self._owner.messages.extend(self._messages)
        return result


class _WebSocketRecorder:
    def __init__(self) -> None:
        self.messages = []

    def admit_broadcast_batch_sync(self, messages, **_kwargs):
        return _Admission(self, messages)


def _publisher_module(name: str):
    if name == "shared":
        from noesis.telemetry import publishers

        return publishers
    path = Path(__file__).resolve().parents[1] / "DS9" / "noesis" / "telemetry" / "publishers.py"
    spec = importlib.util.spec_from_file_location(
        "ds9_publishers_cache_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("publisher_name", ("shared", "ds9"))
def test_tracking_publisher_reuses_durable_world_json_without_model_dump(
    publisher_name: str,
) -> None:
    publishers = _publisher_module(publisher_name)
    TrackingTelemetryPublisher = publishers.TrackingTelemetryPublisher

    class NoDump:
        def model_dump(self, **_kwargs):
            raise AssertionError("prepared world JSON should bypass model_dump")

    observation = NoDump()
    snapshot = NoDump()
    event = NoDump()
    publication = SimpleNamespace(
        observations=(observation,),
        snapshot=snapshot,
        events=(event,),
    )
    observation_payload = {
        "contract": "noesis.observation.person",
        "contract_version": 1,
        "observation_id": "observation-1",
    }
    snapshot_payload = {
        "contract": "noesis.world.snapshot",
        "contract_version": 1,
        "snapshot_id": "snapshot-1",
    }
    event_payload = {
        "contract": "noesis.world.event",
        "contract_version": 1,
        "event_id": "event-1",
    }
    preparation = SimpleNamespace(
        publication=publication,
        serialized_payloads=(
            observation_payload,
            snapshot_payload,
            event_payload,
        ),
    )

    class WorldService:
        def prepare(self, *_args, **_kwargs):
            return preparation

        def commit(self, candidate):
            assert candidate is preparation
            return publication

        def discard(self, _candidate):
            return None

    websocket = _WebSocketRecorder()
    publisher = TrackingTelemetryPublisher(
        websocket,
        world_service=WorldService(),
    )

    publisher.publish(0, [], frame_metadata={"frame_id": 1})

    tracking, world_snapshot, world_event = websocket.messages
    assert tracking["observations"] == [observation_payload]
    assert tracking["world_snapshot"] is snapshot_payload
    assert tracking["world_events"] == [event_payload]
    assert world_snapshot["payload"] is snapshot_payload
    assert world_event["payload"] is event_payload
