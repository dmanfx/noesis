from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from noesis.metadata.depth_result import DepthResult
from noesis.pipelines import hooks
from noesis.telemetry.publishers import DepthTelemetryPublisher
from noesis_core.runtime_publication import RuntimePublicationGate


REPO_ROOT = Path(__file__).resolve().parents[2]


def _depth_result() -> DepthResult:
    return DepthResult(
        source_id=0,
        frame_id=1,
        ts=1,
        width=2,
        height=2,
        depth_map_ref="private-depth-reference",
        minmax=(1.0, 2.0),
    )


def test_runtime_publication_gate_is_required_by_ds9_publishers_and_hooks() -> None:
    for callable_object in (
        DepthTelemetryPublisher,
        hooks.attach_analytics_telemetry_hook,
    ):
        parameter = inspect.signature(callable_object).parameters["publication_gate"]
        assert parameter.default is inspect.Parameter.empty
    # Direct SDK-neutral processor construction remains compatible with
    # focused callers; the runtime attachment path still requires the one
    # shared gate explicitly.
    parameter = inspect.signature(
        hooks._AnalyticsTelemetryProcessor  # type: ignore[attr-defined]
    ).parameters["publication_gate"]
    assert parameter.default is not inspect.Parameter.empty


def test_closed_gate_drops_late_depth_and_analytics_callbacks() -> None:
    gate = RuntimePublicationGate()
    gate.close_and_wait(timeout_s=0.1)

    class _PoisonWebSocket:
        def broadcast_sync(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("closed depth gate touched WebSocket")

    DepthTelemetryPublisher(
        _PoisonWebSocket(),
        publication_gate=gate,
    ).publish(_depth_result())

    class _PoisonFrame:
        def __getattribute__(self, name: str) -> Any:
            raise AssertionError(f"closed analytics gate touched frame field {name}")

    processor = hooks._AnalyticsTelemetryProcessor(  # type: ignore[attr-defined]
        pipeline=SimpleNamespace(config={}, errors=[]),
        tracking_pub=SimpleNamespace(),
        camera_labels={},
        sensor_id_map={},
        publication_gate=gate,
    )
    processor.handle_servicemaker_frame(_PoisonFrame())

    receipt = gate.snapshot()
    assert receipt.admitted == receipt.completed == 0
    assert receipt.rejected == 2


def test_open_depth_gate_accounts_for_completed_publication() -> None:
    gate = RuntimePublicationGate()
    messages: list[dict[str, object]] = []

    class _WebSocket:
        def broadcast_sync(self, message: dict[str, object]) -> None:
            messages.append(message)

    DepthTelemetryPublisher(
        _WebSocket(),
        publication_gate=gate,
    ).publish(_depth_result())

    assert [message["type"] for message in messages] == ["depth_result"]
    receipt = gate.snapshot()
    assert receipt.admitted == receipt.completed == 1
    assert receipt.active == 0


def test_ds9_runtime_passes_one_gate_to_depth_and_analytics() -> None:
    source = (
        REPO_ROOT / "DS9" / "noesis" / "ds9_runtime_core.py"
    ).read_text(encoding="utf-8")
    assert source.count("runtime_publication_gate = RuntimePublicationGate()") == 1
    assert source.count("publication_gate=runtime_publication_gate") == 2
    assert "runtime_publication_gate.close_and_wait(" in source
    assert "runtime_publication_gate.snapshot()" in source
    assert source.index("runtime_publication_gate.close_and_wait(") < source.index(
        "_stop_websocket_server(ws_server, ws_thread, ws_loop)"
    )
