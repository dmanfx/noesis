from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import pytest

from noesis import ds8_runtime
from noesis.pipelines import hooks
from noesis.server import boundary_metrics
from noesis.telemetry import publishers
from websocket_server import WebSocketServer, _serialize_boundary_json


REPO_ROOT = Path(__file__).resolve().parents[1]
THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _import_runtime_env(extra_env: dict[str, str] | None = None) -> dict[str, str | None]:
    env = os.environ.copy()
    for name in THREAD_ENV_VARS + ("NOESIS_CPU_MATH_THREADS",):
        env.pop(name, None)
    env.update(extra_env or {})
    env["PYTHONPATH"] = str(REPO_ROOT)

    code = """
import json
import os
from noesis import ds8_runtime

print("CPU_CAPS_JSON=" + json.dumps({name: os.environ.get(name) for name in ds8_runtime._CPU_MATH_THREAD_ENV_VARS}, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    for line in reversed(result.stdout.splitlines()):
        if line.startswith("CPU_CAPS_JSON="):
            return json.loads(line.split("=", 1)[1])
    raise AssertionError(f"missing CPU_CAPS_JSON in stdout={result.stdout!r} stderr={result.stderr!r}")


class _DummyPipeline:
    def __init__(self) -> None:
        self.prepared = True
        self.activated = True
        self.depth_enabled = False
        self.analytics_reload_count = 0
        self.errors = []
        self.frame_size = (1920, 1080)
        self.latency_collector = None
        self.components: Dict[str, Any] = {
            "tiler": type("Tiler", (), {"config": {"width": 1920, "height": 1080, "columns": 1, "rows": 1}})()
        }

    def depth_fps(self) -> float:
        return 0.0


def test_zero_copy_stats_fields_present() -> None:
    hooks.reset_core_path_instrumentation()
    hooks._serialize_compact_json_with_metrics({"type": "probe"}, metric="test.metric")
    hooks._CORE_PATH_INSTRUMENTATION.record_cpu_copy_violation(  # type: ignore[attr-defined]
        location="test.location",
        reason="unit-test",
    )

    ws = WebSocketServer(stats_callback=None)
    ws._record_boundary_serialization(1.2, 128)  # type: ignore[attr-defined]
    boundary_metrics.reset_boundary_serialization_metrics()
    boundary_metrics.record_boundary_stage(
        0.8,
        64,
        channel="rest",
        route="/unit",
        message_type="UnitResponse",
        stage="total",
        include_budget=True,
    )

    callback = ds8_runtime._build_stats_callback(  # type: ignore[attr-defined]
        _DummyPipeline(),
        {0: "cam0"},
        ws_metrics_getter=ws.get_boundary_serialization_metrics,
        ws_metrics_resetter=ws.reset_boundary_serialization_metrics,
    )

    payload = callback()
    pipeline_payload = payload["pipeline"]

    assert pipeline_payload["zero_copy_core_enabled"] is True
    assert int(pipeline_payload["zero_copy_violations"]) >= 1
    assert pipeline_payload["boundary_cpu_serialization_p99_ms"] is not None
    assert "boundary_cpu_serialization_ws_p99_ms" in pipeline_payload
    assert "boundary_cpu_serialization_rest_p99_ms" in pipeline_payload
    assert "stableid_backend_mode" in pipeline_payload
    assert "stableid_gpu_match_p95_ms" in pipeline_payload
    assert "zero_copy_core" in pipeline_payload

    callback.clear_stats()  # type: ignore[attr-defined]
    assert ws.get_boundary_serialization_metrics()["count"] == 0
    assert boundary_metrics.get_boundary_serialization_metrics()["count"] == 0


def test_ws_port_selection_fails_when_required_port_is_busy(monkeypatch) -> None:
    availability = {
        6008: False,
        6009: False,
        6010: True,
    }

    def _fake_bindable(_host: str, port: int) -> bool:
        return bool(availability.get(int(port), False))

    monkeypatch.setattr(ds8_runtime, "_port_bindable", _fake_bindable)
    with pytest.raises(RuntimeError, match="required WebSocket endpoint"):
        ds8_runtime._select_ws_port(  # type: ignore[attr-defined]
            "127.0.0.1",
            6008,
            8,
            logging.getLogger("test.zero_copy"),
        )


def test_rest_server_refuses_an_occupied_required_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(ds8_runtime, "_port_bindable", lambda _host, _port: False)
    app = SimpleNamespace(
        state=SimpleNamespace(noesis_internal_auth={"mode": "required"})
    )
    with pytest.raises(RuntimeError, match="required REST endpoint"):
        ds8_runtime._start_rest_server(app, "127.0.0.1", 8080)  # type: ignore[attr-defined]


def test_internal_control_defaults_are_loopback() -> None:
    source = Path(ds8_runtime.__file__).read_text(encoding="utf-8")
    assert 'os.environ.get("NOESIS_WS_HOST", "127.0.0.1")' in source
    assert 'os.environ.get("NOESIS_REST_HOST", "127.0.0.1")' in source


def test_sigterm_uses_cleanup_path_with_bounded_watchdog() -> None:
    source = Path(ds8_runtime.__file__).read_text(encoding="utf-8")
    assert "signal.signal(signal.SIGTERM, _signal_handler)" in source
    assert "NOESIS_SHUTDOWN_GRACE_SECONDS" in source
    assert "signal.alarm(0)" in source
    assert "sigterm hard exit" not in source


def test_ds8_runtime_caps_cpu_math_threads_before_numpy_import() -> None:
    values = _import_runtime_env()

    assert values == {name: "1" for name in THREAD_ENV_VARS}


def test_ds8_runtime_cpu_math_threads_override_preserves_explicit_pool_values() -> None:
    values = _import_runtime_env(
        {
            "NOESIS_CPU_MATH_THREADS": "3",
            "OPENBLAS_NUM_THREADS": "7",
        }
    )

    assert values["OPENBLAS_NUM_THREADS"] == "7"
    assert values["OMP_NUM_THREADS"] == "3"
    assert values["MKL_NUM_THREADS"] == "3"
    assert values["NUMEXPR_NUM_THREADS"] == "3"
    assert values["VECLIB_MAXIMUM_THREADS"] == "3"
    assert values["BLIS_NUM_THREADS"] == "3"


class _Ws:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def admit_broadcast_batch_sync(
        self,
        messages: list[dict[str, Any]],
        **_kwargs: Any,
    ) -> Any:
        receipt = SimpleNamespace(
            submission_id=len(self.messages) + 1,
            message_count=len(messages),
        )

        class _Admission:
            def __init__(self, owner: _Ws) -> None:
                self.receipt = receipt
                self._owner = owner

            def commit_then_release(self, commit: Any) -> Any:
                result = commit()
                self._owner.messages.extend(messages)
                return result

        return _Admission(self)


def test_tracking_publisher_preserves_json_native_track_identity() -> None:
    tracks = [
        {
            "stable_id": 42,
            "bbox": [1.0, 2.0, 3.0, 4.0],
            "center": [2.5, 4.0],
            "zone": "kitchen",
            "analytics": {"roiStatus": ["kitchen"]},
        }
    ]
    ws = _Ws()

    publishers.TrackingTelemetryPublisher(ws).publish(0, tracks)

    assert ws.messages[0]["tracks"] is tracks


def test_tracking_publisher_defers_numpy_and_tuple_conversion_to_one_pass_boundary() -> None:
    ws = _Ws()

    bbox = (np.float32(1.5), np.float64(2.5))
    publishers.TrackingTelemetryPublisher(ws).publish(
        0,
        [{"stable_id": np.int64(7), "bbox": bbox}],
    )

    message = ws.messages[0]
    assert isinstance(message["tracks"][0]["stable_id"], np.integer)
    assert message["tracks"][0]["bbox"] is bbox
    encoded, payload_bytes, _, convert_ms, encode_ms = _serialize_boundary_json(message)
    decoded = json.loads(encoded)
    assert decoded["tracks"][0] == {"stable_id": 7, "bbox": [1.5, 2.5]}
    assert payload_bytes == len(encoded)
    assert convert_ms >= 0.0
    assert encode_ms >= 0.0
