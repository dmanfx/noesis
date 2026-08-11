from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from noesis.server import health_api
from noesis_core.appliance import (
    DeploymentHealthBinding,
    optional_runtime_context_binding,
    runtime_context_environment,
    runtime_inventory_digest,
    runtime_snapshot_digest,
)
from noesis_core.health import CapabilityMonitor, CapabilityPolicy
from websocket_server import WebSocketServer


SHA = "a" * 64
REVISION = "b" * 40


class _Socket:
    path = "/healthz"

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed: list[tuple[int, str]] = []

    async def send(self, value: str) -> None:
        self.sent.append(value)

    async def close(self, *, code: int, reason: str) -> None:
        self.closed.append((code, reason))


def _binding() -> DeploymentHealthBinding:
    return DeploymentHealthBinding(
        deployment_id="deploy-alpha",
        selector_sha256=SHA,
        state_release_id="release-alpha",
        runtime_family="ds9",
        runtime_variant="ds9:baseline",
        software_revision=REVISION,
        boot_id="00000000-0000-0000-0000-000000000001",
    )


def _monitor() -> CapabilityMonitor:
    return CapabilityMonitor(
        instance_id="appliance-instance",
        run_id="runtime-run",
        policies={
            "tracking_observations": CapabilityPolicy(
                stale_after_us=2_000_000, fail_after_us=10_000_000
            ),
            "global_world": CapabilityPolicy(
                stale_after_us=2_000_000, fail_after_us=10_000_000
            ),
        },
    )


def _record_ready(monitor: CapabilityMonitor) -> None:
    now = time.time_ns() // 1_000
    for index, capability in enumerate(("tracking_observations", "global_world")):
        monitor.record_success(
            capability,
            producer_run_id="runtime-run",
            sequence=index + 1,
            observed_at_us=now - 10,
            checked_at_us=now,
            contract_compatible=True,
        )


def test_runtime_context_round_trip_is_closed_and_digest_bound() -> None:
    binding = _binding()
    env = runtime_context_environment(binding)
    loaded = optional_runtime_context_binding(env)
    assert loaded is not None
    assert loaded.deployment_id == binding.deployment_id
    assert loaded.runtime_variant == "ds9:baseline"

    poisoned = dict(env)
    payload = json.loads(poisoned["NOESIS_APPLIANCE_RUNTIME_CONTEXT"])
    payload["runtime_variant"] = "ds9:v3dt"
    poisoned["NOESIS_APPLIANCE_RUNTIME_CONTEXT"] = json.dumps(
        payload, separators=(",", ":")
    )
    try:
        optional_runtime_context_binding(poisoned)
    except ValueError as exc:
        assert "digest" in str(exc)
    else:  # pragma: no cover - a digest splice must never be accepted.
        raise AssertionError("spliced appliance runtime context was accepted")


def test_runtime_context_rejects_nested_state_lease_environment() -> None:
    env = runtime_context_environment(_binding())
    env["MENON_STATE_RELEASE_LEASE_FILE"] = "/tmp/forbidden.lease"
    try:
        optional_runtime_context_binding(env)
    except ValueError as exc:
        assert "must not receive or acquire" in str(exc)
    else:  # pragma: no cover - nested lease ownership must remain impossible.
        raise AssertionError("nested appliance state lease was accepted")


def test_noesis_runtime_snapshot_digest_vector_is_cross_language_stable() -> None:
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "contracts/fixtures/v1/noesis_runtime_snapshot_vector.json"
        ).read_text(encoding="utf-8")
    )
    for path, payload in fixture["payloads"].items():
        row = next(row for row in fixture["inventory_rows"] if row["path"] == path)
        encoded = payload.encode("utf-8")
        assert row["bytes"] == len(encoded)
        assert row["sha256"] == hashlib.sha256(encoded).hexdigest()
    assert runtime_inventory_digest(fixture["inventory_rows"]) == fixture["descriptor"][
        "runtime_inventory_sha256"
    ]
    assert runtime_snapshot_digest(fixture["descriptor"]) == fixture["snapshot_sha256"]


def test_deployment_health_is_503_until_both_capabilities_are_current() -> None:
    binding = _binding()
    monitor = _monitor()
    binding.bind_producer(instance_id="appliance-instance", run_id="runtime-run")
    health_api.register_capability_monitor_getter(lambda: monitor)
    health_api.register_deployment_binding_getter(lambda: binding)
    app = FastAPI()
    app.include_router(health_api.router)
    client = TestClient(app)

    assert client.get("/api/v1/health/deployment").status_code == 503
    _record_ready(monitor)
    first = client.get("/api/v1/health/deployment")
    second = client.get("/api/v1/health/deployment")
    assert first.status_code == 200
    assert second.status_code == 200
    first_payload = first.json()
    second_payload = second.json()
    assert set(first_payload) == {
        "contract",
        "contract_version",
        "deployment_id",
        "selector_sha256",
        "state_release_id",
        "runtime_family",
        "runtime_variant",
        "instance_id",
        "run_id",
        "boot_id",
        "software_revision",
        "generated_at_us",
        "ready",
    }
    assert first_payload["contract"] == "noesis.appliance.deployment_health"
    assert first_payload["ready"] is True
    assert second_payload["generated_at_us"] > first_payload["generated_at_us"]


def test_appliance_websocket_health_v2_matches_rest_identity() -> None:
    binding = _binding()
    monitor = _monitor()
    binding.bind_producer(instance_id="appliance-instance", run_id="runtime-run")
    _record_ready(monitor)
    server = WebSocketServer(
        health_payload_getter=lambda: binding.websocket_health(monitor).model_dump(
            mode="json"
        )
    )
    socket = _Socket()
    asyncio.run(server.handle_client(socket, "/healthz"))
    assert socket.closed == [(1000, "health_complete")]
    payload = json.loads(socket.sent[0])
    assert payload["type"] == "health"
    assert payload["contract"] == "noesis.ws.health"
    assert payload["contract_version"] == 2
    assert payload["deployment_id"] == "deploy-alpha"
    assert payload["instance_id"] == "appliance-instance"
    assert payload["run_id"] == "runtime-run"


def test_appliance_websocket_health_fails_closed_before_progress() -> None:
    binding = _binding()
    monitor = _monitor()
    binding.bind_producer(instance_id="appliance-instance", run_id="runtime-run")
    server = WebSocketServer(
        health_payload_getter=lambda: binding.websocket_health(monitor).model_dump(
            mode="json"
        )
    )
    socket = _Socket()
    asyncio.run(server.handle_client(socket, "/healthz"))
    assert socket.sent == []
    assert socket.closed == [(1013, "health_not_ready")]
