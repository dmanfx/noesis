from __future__ import annotations

import importlib
from pathlib import Path
import sys
import time
from typing import Any

from fastapi.testclient import TestClient
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from noesis.pipelines import ds8_pipeline  # noqa: E402
from noesis.server import boundary_metrics  # noqa: E402

PIPELINE_CONFIG_ENV = "NOESIS_DS8_PIPELINE_CONFIG"
FORCE_STUB_ENV = "NOESIS_DEPTH_API_FORCE_STUB"


def _load_depth_api(monkeypatch: pytest.MonkeyPatch, force_stub: bool) -> Any:
    """Import depth_api with a clean module cache, optionally forcing stub mode."""
    monkeypatch.delenv(FORCE_STUB_ENV, raising=False)
    if force_stub:
        monkeypatch.setenv(FORCE_STUB_ENV, "1")

    for name in list(sys.modules):
        if name.startswith("noesis.server.depth_api"):
            sys.modules.pop(name)

    module = importlib.import_module("noesis.server.depth_api")
    return module


def _reset_pipeline() -> None:
    """Clear the global pipeline singleton between tests."""
    try:
        graph = ds8_pipeline.get_pipeline()
    except Exception:
        graph = None

    if graph is not None and getattr(graph, "_timer", None):
        graph._timer.cancel()

    ds8_pipeline._PIPELINE_SINGLETON = None  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def pipeline_guard(monkeypatch: pytest.MonkeyPatch):
    """Reset pipeline state before each test to keep globals isolated."""
    _reset_pipeline()
    boundary_metrics.reset_boundary_serialization_metrics()
    yield
    _reset_pipeline()
    boundary_metrics.reset_boundary_serialization_metrics()
    monkeypatch.delenv(PIPELINE_CONFIG_ENV, raising=False)
    monkeypatch.delenv(FORCE_STUB_ENV, raising=False)


def test_depth_refresh_uses_controller_owned_provider(
    monkeypatch: pytest.MonkeyPatch,
):
    depth_api = _load_depth_api(monkeypatch, force_stub=False)
    cfg_path = Path("config/infer.yaml")
    assert cfg_path.exists()
    monkeypatch.setenv(PIPELINE_CONFIG_ENV, str(cfg_path))
    assert depth_api.ensure_pipeline_ready() is True

    graph = ds8_pipeline.get_pipeline()
    assert graph.depth_enabled is False

    calls: list[int] = []

    def provider(seconds: int) -> dict[str, object]:
        calls.append(int(seconds))
        now = int(time.time())
        graph.mark_depth_enabled(True)
        return {
            "started_at": now,
            "will_disable_at": now + int(seconds),
            "enabled": True,
            "seconds": int(seconds),
        }

    depth_api.app.state.depth_refresh_provider = provider

    with TestClient(depth_api.app) as client:
        response = client.post("/api/v1/depth/refresh", params={"seconds": 5})

    assert response.status_code == 200
    data = response.json()
    assert data["seconds"] == 5
    assert data["will_disable_at"] >= data["started_at"]
    assert graph.depth_enabled is True
    assert calls == [5]
    metrics = boundary_metrics.get_boundary_serialization_metrics()
    assert int(metrics.get("count", 0)) >= 1
    assert int(metrics["last_payload_bytes"]) == len(response.content)
    stages = metrics.get("stages", {})
    assert "rest|/api/v1/depth/refresh|DepthRefreshResponse|response_model|ok" in stages
    assert "rest|/api/v1/depth/refresh|DepthRefreshResponse|json_encode|ok" in stages
    assert "rest|/api/v1/depth/refresh|DepthRefreshResponse|total|ok" in stages
    assert int(
        stages["rest|/api/v1/depth/refresh|DepthRefreshResponse|total|ok"][
            "last_payload_bytes"
        ]
    ) == len(response.content)


def test_depth_refresh_rejects_missing_or_busy_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    depth_api = _load_depth_api(monkeypatch, force_stub=False)
    monkeypatch.setenv(PIPELINE_CONFIG_ENV, "config/infer.yaml")

    with TestClient(depth_api.app) as client:
        missing = client.post(
            "/api/v1/depth/refresh",
            params={"seconds": 5},
        )
    assert missing.status_code == 503

    class _Busy(RuntimeError):
        code = "capture_event_busy"

    depth_api.app.state.depth_refresh_provider = (
        lambda _seconds: (_ for _ in ()).throw(_Busy("owned"))
    )
    with TestClient(depth_api.app) as client:
        busy = client.post(
            "/api/v1/depth/refresh",
            params={"seconds": 5},
        )
    assert busy.status_code == 409
    assert busy.json()["detail"] == "capture_event_busy"


def test_depth_refresh_stub_mode(monkeypatch: pytest.MonkeyPatch):
    depth_api = _load_depth_api(monkeypatch, force_stub=True)
    assert depth_api.USING_PIPELINE_STUB is True

    with TestClient(depth_api.app) as client:
        response = client.post("/api/v1/depth/refresh", params={"seconds": 2})

    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is True
    assert payload["will_disable_at"] >= payload["started_at"]
    pipeline = depth_api.get_pipeline()
    assert pipeline.depth_enabled is True

    timer = getattr(pipeline, "_timer", None)
    if timer is not None:
        timer.cancel()

    # Restore the real implementation for the rest of the suite.
    _load_depth_api(monkeypatch, force_stub=False)
