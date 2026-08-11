from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from noesis.server import health_api
from noesis_core.health import CapabilityMonitor, CapabilityPolicy


def _client() -> tuple[TestClient, CapabilityMonitor]:
    monitor = CapabilityMonitor(
        instance_id="appliance",
        run_id="run-1",
        policies={"global_world": CapabilityPolicy(stale_after_us=100, fail_after_us=200)},
    )
    health_api.register_capability_monitor_getter(lambda: monitor)
    app = FastAPI()
    app.include_router(health_api.router)
    return TestClient(app), monitor


def test_health_endpoint_is_unknown_without_producer_progress() -> None:
    client, _monitor = _client()
    payload = client.get("/api/v1/health/capabilities").json()
    assert payload["contract"] == "noesis.capability.health"
    assert payload["capabilities"][0]["status"] == "unknown"
    assert payload["capabilities"][0]["blockers"] == ["no_producer_progress"]


def test_health_endpoint_reports_recorded_contract_progress(monkeypatch) -> None:
    client, monitor = _client()
    monitor.record_success(
        "global_world",
        producer_run_id="producer-1",
        sequence=1,
        observed_at_us=900,
        checked_at_us=1_000,
        contract_compatible=True,
        evidence={"entity_count": 2},
    )
    monkeypatch.setattr(health_api.time, "time_ns", lambda: 1_050_000)
    payload = client.get("/api/v1/health/capabilities").json()
    assert payload["capabilities"][0]["status"] == "healthy"
    assert payload["capabilities"][0]["evidence"]["entity_count"] == 2
