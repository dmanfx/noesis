from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from pathlib import Path
import threading
import time

import pytest
from websockets.exceptions import ConnectionClosedOK
from websockets.frames import Close

from noesis.server.websocket import (
    BoundaryResponseModelContractError,
    CanonicalOutboundRouteRequired,
    FrozenOutboundJSON,
    FrozenOutboundOwnershipError,
    GatedOutboundAdmission,
    OutboundAdmissionClosed,
    OutboundAdmissionReceipt,
    OutboundCapacityExceeded,
    OutboundGateResolutionError,
    OutboundQuiescenceTimeout,
    ProviderAdmissionClosed,
    ProviderCapacityExceeded,
    ProviderQuiescenceTimeout,
    WebSocketServer,
)


ROOT = Path(__file__).resolve().parents[1]


class _FakeSocket:
    def __init__(self) -> None:
        self.sent: list[object] = []
        self.remote_address = ("127.0.0.1", 6008)

    async def send(self, payload: object) -> None:
        self.sent.append(payload)


class _CapturedFuture:
    def __init__(self) -> None:
        self.callback = None

    def add_done_callback(self, callback) -> None:
        self.callback = callback


class _InboundSocket(_FakeSocket):
    def __init__(self, messages: list[dict[str, object]]) -> None:
        super().__init__()
        self._messages = iter(json.dumps(message) for message in messages)

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        try:
            return next(self._messages)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class _CapacitySocket(_FakeSocket):
    def __init__(self, *, path: str | None = None, blocking: bool = False) -> None:
        super().__init__()
        self.path = path
        self.blocking = blocking
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.close_calls: list[tuple[int | None, str | None]] = []

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        self.entered.set()
        if self.blocking:
            await self.release.wait()
        raise StopAsyncIteration

    async def close(
        self,
        code: int | None = None,
        reason: str | None = None,
    ) -> None:
        self.close_calls.append((code, reason))


def _timed_payload(ws: WebSocketServer, factory):
    started_ns = time.perf_counter_ns()
    return ws.timed_payload_since(factory(), started_ns)


def _canonical_tracking_batch(
    ws: WebSocketServer,
    *,
    sequence: int,
    tracks: list[dict[str, object]] | None = None,
    with_world: bool = True,
) -> list[dict[str, object]]:
    cohort = {
        "source_id": 0,
        "frame_id": sequence + 1,
        "observed_at_us": 1_000_000 + sequence,
        "tracking_publication_sequence": sequence,
    }
    tracking: dict[str, object] = {
        "type": "tracking",
        **cohort,
        "cohort": dict(cohort),
        "tracks": tracks or [],
    }
    messages: list[dict[str, object]] = [tracking]
    if with_world:
        snapshot = {"sequence": sequence}
        events = [{"sequence": sequence, "event_type": "appeared"}]
        tracking["world_snapshot"] = snapshot
        tracking["world_events"] = events
        messages.extend(
            [
                {
                    "type": "world_snapshot",
                    "payload": snapshot,
                    "cohort": dict(cohort),
                    **cohort,
                },
                {
                    "type": "world_event",
                    "payload": events[0],
                    "cohort": dict(cohort),
                    **cohort,
                },
            ]
        )
    return [
        _timed_payload(ws, lambda message=message: message)
        for message in messages
    ]


def test_webrtc_gateway_does_not_persist_or_log_full_sdp() -> None:
    source = (ROOT / "noesis" / "mosaic_webrtc_gateway.py").read_text(
        encoding="utf-8"
    )
    assert ".cursor/debug.log" not in source
    assert '"message": "offer sdp"' not in source
    assert "Full SDP" not in source
    assert "stun.l.google.com" not in source
    assert "candidate[:" not in source

    websocket_source = (ROOT / "noesis" / "server" / "websocket.py").read_text(
        encoding="utf-8"
    )
    assert "candidate[:" not in websocket_source

    replay_source = (
        ROOT / "scripts" / "webrtc_gateway_browser_offer_replay_test.py"
    ).read_text(encoding="utf-8")
    assert "/home/" not in replay_source
    assert ".cursor" not in replay_source
    assert "read_private_file" in replay_source
    assert "MAX_OFFER_BYTES" in replay_source


def test_telemetry_client_capacity_releases_reconnects_and_bypasses_health() -> None:
    async def _run() -> dict[str, object]:
        ws = WebSocketServer(stats_callback=None)
        ws._max_telemetry_clients = 1
        first = _CapacitySocket(blocking=True)
        first_task = asyncio.create_task(ws.handle_client(first))
        await asyncio.wait_for(first.entered.wait(), timeout=1.0)
        assert first in ws.connected_clients

        rejected = _CapacitySocket()
        await ws.handle_client(rejected)
        assert rejected not in ws.connected_clients
        assert rejected.sent == []
        assert rejected.close_calls == [
            (
                ws.TELEMETRY_CAPACITY_CLOSE_CODE,
                ws.TELEMETRY_CAPACITY_CLOSE_REASON,
            )
        ]

        first.release.set()
        await first_task
        assert ws.connected_clients == set()

        reconnected = _CapacitySocket()
        await ws.handle_client(reconnected)
        assert reconnected.close_calls == []
        assert reconnected.sent
        assert ws.connected_clients == set()

        occupied_slot = _FakeSocket()
        ws.connected_clients.add(occupied_slot)
        health = _CapacitySocket(path=ws.HEALTH_PATH)
        await ws.handle_client(health)
        assert health.sent == [ws.HEALTH_PAYLOAD]
        assert health.close_calls == [(1000, "health_complete")]
        assert ws.connected_clients == {occupied_slot}
        metrics = ws.get_boundary_serialization_metrics()[
            "telemetry_clients"
        ]
        ws.connected_clients.clear()
        return metrics

    metrics = asyncio.run(_run())
    assert metrics == {
        "connected": 1,
        "peak": 1,
        "max": 1,
        "rejections": 1,
    }


def test_telemetry_client_capacity_environment_is_hard_clamped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NOESIS_WS_MAX_TELEMETRY_CLIENTS", "999")
    assert WebSocketServer(stats_callback=None)._max_telemetry_clients == 16
    monkeypatch.setenv("NOESIS_WS_MAX_TELEMETRY_CLIENTS", "0")
    assert WebSocketServer(stats_callback=None)._max_telemetry_clients == 1


def test_retired_noop_websocket_controls_are_not_public_routes() -> None:
    source = (ROOT / "noesis" / "server" / "websocket.py").read_text(
        encoding="utf-8"
    )
    for retired_type in (
        "update_detection_config",
        "detection_config_update",
        "detection_config_sync",
        "set_detection_toggle",
        "detection_toggle_update",
        "ma_heatmap_ready",
    ):
        assert retired_type not in source


def test_get_ma_depth_cache_forwards_cache_only_to_provider() -> None:
    observed: list[tuple[str, object, object, bool]] = []

    def provider(
        camera: str,
        ts_max_us: object,
        request_id: object,
        *,
        cache_only: bool = False,
    ) -> dict[str, object]:
        observed.append((camera, ts_max_us, request_id, cache_only))
        return {
            "type": "ma_depth_response",
            "camera": camera,
            "request_id": request_id,
            "cache_only": cache_only,
            "served_from_cache": False,
            "error": "no_cached_depth",
            "ok": False,
        }

    async def _run() -> tuple[WebSocketServer, _InboundSocket]:
        ws = WebSocketServer(stats_callback=None)
        ws.ma_depth_provider = provider
        socket = _InboundSocket(
            [
                {
                    "type": "get_ma_depth_cache",
                    "camera": "living-room",
                    "request_id": "cache-request",
                }
            ]
        )
        await ws.handle_client(socket)
        return ws, socket

    ws, socket = asyncio.run(_run())
    receipt = ws.quiesce_blocking_providers(timeout_s=1.0)
    assert receipt.quiesced is True
    assert observed == [("living-room", None, "cache-request", True)]
    responses = [
        json.loads(str(payload))
        for payload in socket.sent
        if isinstance(payload, str)
    ]
    response = next(
        payload
        for payload in responses
        if payload.get("type") == "ma_depth_response"
    )
    assert response["cache_only"] is True
    assert response["error"] == "no_cached_depth"


def test_get_floorplan_forwards_exact_snapshot_identity_to_provider() -> None:
    observed: list[tuple[object, ...]] = []

    def provider(
        camera: str,
        max_age_sec: float,
        grid_res_m: float,
        max_extent_m: float,
        *,
        cache_only: bool = False,
        snapshot_ref: object = None,
        snapshot_id: object = None,
        snapshot_content_sha256: object = None,
    ) -> dict[str, object]:
        observed.append((
            camera,
            max_age_sec,
            grid_res_m,
            max_extent_m,
            cache_only,
            snapshot_ref,
            snapshot_id,
            snapshot_content_sha256,
        ))
        return {
            "camera_id": camera,
            "snapshot_ref": snapshot_ref,
            "snapshot_id": snapshot_id,
            "snapshot_content_sha256": snapshot_content_sha256,
            "ok": True,
        }

    digest = "a" * 64

    async def _run() -> tuple[WebSocketServer, _InboundSocket]:
        ws = WebSocketServer(stats_callback=None)
        ws.floorplan_provider = provider
        socket = _InboundSocket([
            {
                "type": "get_floorplan",
                "camera": "living-room",
                "request_id": "exact-floorplan",
                "max_age_sec": 0,
                "grid_res_m": 0.15,
                "max_extent_m": 20,
                "cache_only": False,
                "snapshot_ref": "living-room/fused.zarr",
                "snapshot_id": "fused-1",
                "snapshot_content_sha256": digest,
            }
        ])
        await ws.handle_client(socket)
        return ws, socket

    ws, socket = asyncio.run(_run())
    receipt = ws.quiesce_blocking_providers(timeout_s=1.0)
    assert receipt.quiesced is True
    assert observed == [(
        "living-room",
        0.0,
        0.15,
        20.0,
        False,
        "living-room/fused.zarr",
        "fused-1",
        digest,
    )]
    response = next(
        json.loads(str(payload))
        for payload in socket.sent
        if isinstance(payload, str)
        and json.loads(str(payload)).get("type") == "floorplan_response"
    )
    assert response["snapshot_id"] == "fused-1"


def test_get_floorplan_forwards_explicit_scene_prior_only_mode() -> None:
    observed: list[tuple[str, bool, bool]] = []

    def provider(
        camera: str,
        _max_age_sec: float,
        _grid_res_m: float,
        _max_extent_m: float,
        *,
        cache_only: bool = False,
        scene_prior_only: bool = False,
    ) -> dict[str, object]:
        observed.append((camera, cache_only, scene_prior_only))
        return {
            "camera_id": camera,
            "scene_prior_only": scene_prior_only,
            "display_source": "pcf",
            "ok": True,
        }

    async def _run() -> tuple[WebSocketServer, _InboundSocket]:
        ws = WebSocketServer(stats_callback=None)
        ws.floorplan_provider = provider
        socket = _InboundSocket([
            {
                "type": "get_floorplan",
                "camera": "living-room",
                "request_id": "pcf-floorplan",
                "cache_only": True,
                "scene_prior_only": True,
            }
        ])
        await ws.handle_client(socket)
        return ws, socket

    ws, socket = asyncio.run(_run())
    receipt = ws.quiesce_blocking_providers(timeout_s=1.0)
    assert receipt.quiesced is True
    assert observed == [("living-room", True, True)]
    response = next(
        json.loads(str(payload))
        for payload in socket.sent
        if isinstance(payload, str)
        and json.loads(str(payload)).get("type") == "floorplan_response"
    )
    assert response["request_id"] == "pcf-floorplan"
    assert response["scene_prior_only"] is True
    assert response["display_source"] == "pcf"


def test_blocking_calibration_rpc_runs_off_event_loop() -> None:
    ticks = 0
    heartbeat_done = asyncio.Event()

    def pixel_to_world(_request: dict[str, object]) -> dict[str, object]:
        time.sleep(0.05)
        return {"ok": True, "world": [1.0, 2.0, 3.0]}

    async def _run() -> tuple[WebSocketServer, _InboundSocket, int]:
        nonlocal ticks
        ws = WebSocketServer(stats_callback=None)
        ws.pixel_to_world_handler = pixel_to_world
        socket = _InboundSocket(
            [
                {
                    "type": "pixel_to_world",
                    "request_id": "calibration-request",
                }
            ]
        )

        async def _heartbeat() -> None:
            nonlocal ticks
            while not heartbeat_done.is_set():
                ticks += 1
                await asyncio.sleep(0.005)

        heartbeat = asyncio.create_task(_heartbeat())
        await ws.handle_client(socket)
        heartbeat_done.set()
        await heartbeat
        return ws, socket, ticks

    ws, socket, observed_ticks = asyncio.run(_run())
    receipt = ws.quiesce_blocking_providers(timeout_s=1.0)
    assert receipt.quiesced is True
    assert observed_ticks >= 5
    responses = [
        json.loads(str(payload))
        for payload in socket.sent
        if isinstance(payload, str)
    ]
    response = next(
        payload
        for payload in responses
        if payload.get("type") == "pixel_to_world_response"
    )
    assert response["ok"] is True
    assert response["world"] == {"x": 1.0, "y": 2.0, "z": 3.0}


def test_all_calibration_handlers_use_owned_provider_admission() -> None:
    source = (ROOT / "noesis" / "server" / "websocket.py").read_text(
        encoding="utf-8"
    )
    message_types = (
        "pixel_to_world",
        "set_extrinsics",
        "solve_pnp",
        "set_align",
        "get_ma_depth",
    )
    for message_type in message_types[:-1]:
        start = source.index(f"elif data.get('type') == '{message_type}':")
        end = source.index("elif data.get('type')", start + 1)
        branch = source[start:end]
        assert "await asyncio.wait_for(" in branch
        assert "self._run_blocking_provider(" in branch
        assert "except ProviderAdmissionClosed:" in branch


def test_send_to_client_records_stage_metrics() -> None:
    async def _run() -> dict:
        ws = WebSocketServer(stats_callback=None)
        client = _FakeSocket()
        ws.connected_clients.add(client)
        await ws._send_to_client(
            client,
            _timed_payload(
                ws,
                lambda: {"type": "stats", "payload": {"ok": True}},
            ),
        )
        return ws.get_boundary_serialization_metrics()

    metrics = asyncio.run(_run())
    assert int(metrics.get("count", 0)) >= 1
    stages = metrics.get("stages", {})
    assert any("ws|send_to_client|stats|numpy_convert|ok" == key for key in stages)
    assert any("ws|send_to_client|stats|json_encode|ok" == key for key in stages)
    assert any("ws|send_to_client|stats|send_dispatch|ok" == key for key in stages)
    assert any("ws|send_to_client|stats|total|ok" == key for key in stages)


def test_non_budget_stage_does_not_increment_global_count() -> None:
    ws = WebSocketServer(stats_callback=None)
    ws._record_boundary_serialization_stage(
        7.5,
        0,
        channel="ws",
        route="get_ma_depth",
        message_type="ma_depth_response",
        stage="provider_wait",
        outcome="ok",
        include_budget=False,
    )
    metrics = ws.get_boundary_serialization_metrics()
    assert int(metrics.get("count", 0)) == 0
    stages = metrics.get("stages", {})
    key = "ws|get_ma_depth|ma_depth_response|provider_wait|ok"
    assert key in stages
    assert int(stages[key].get("count", 0)) == 1


def test_ws_boundary_uses_true_rolling_windows_and_sparse_path_authority() -> None:
    now_s = [100.0]
    ws = WebSocketServer(stats_callback=None)
    ws._boundary_clock = lambda: now_s[0]
    for _ in range(200):
        ws._record_boundary_serialization_stage(
            0.2,
            32,
            route="fast",
            message_type="stats",
            stage="total",
            include_budget=True,
        )
    ws._record_boundary_serialization_stage(
        10.0,
        64,
        route="sparse_slow",
        message_type="world_snapshot",
        stage="total",
        include_budget=True,
    )

    metrics = ws.get_boundary_serialization_metrics()
    assert metrics["p99_10s_ms"] == 0.2
    assert metrics["p99_60s_ms"] == 0.2
    assert metrics["max_path_p99_10s_ms"] == 10.0
    assert metrics["max_path_p99_60s_ms"] == 10.0
    assert metrics["max_path_p99_ms"] == 10.0

    now_s[0] = 111.0
    ws._record_boundary_serialization_stage(
        0.3,
        32,
        route="fast",
        message_type="stats",
        stage="total",
        include_budget=True,
    )
    metrics = ws.get_boundary_serialization_metrics()
    assert metrics["max_path_p99_10s_ms"] == 0.3
    assert metrics["max_path_p99_60s_ms"] == 10.0

    now_s[0] = 172.0
    metrics = ws.get_boundary_serialization_metrics()
    assert metrics["windows"]["10s"]["count"] == 0
    assert metrics["windows"]["60s"]["count"] == 0
    assert metrics["max_path_p99_ms"] is None


def test_ws_boundary_sample_caps_retain_live_slow_offender_fail_closed() -> None:
    ws = WebSocketServer(stats_callback=None)
    ws.BOUNDARY_AGGREGATE_SAMPLE_LIMIT = 8
    ws.BOUNDARY_DETAIL_SAMPLE_LIMIT = 4
    ws._boundary_clock = lambda: 100.0
    ws._record_boundary_serialization_stage(
        12.0,
        64,
        route="burst",
        message_type="tracking",
        stage="total",
        include_budget=True,
    )
    for _ in range(20):
        ws._record_boundary_serialization_stage(
            0.1,
            64,
            route="burst",
            message_type="tracking",
            stage="total",
            include_budget=True,
        )

    assert len(ws._boundary_samples_ms) == 8
    stage_key = "ws|burst|tracking|total|ok"
    assert len(ws._boundary_stage_metrics[stage_key]["samples"]) == 4
    metrics = ws.get_boundary_serialization_metrics()
    assert metrics["detail_truncated"] is True
    assert metrics["max_path_p99_ms"] == 12.0
    assert metrics["windows"]["60s"]["truncated"] is True

    ws._boundary_clock = lambda: 161.0
    expired = ws.get_boundary_serialization_metrics()
    assert expired["max_path_p99_ms"] is None
    assert expired["windows"]["60s"]["truncated"] is False


def test_compact_ws_metrics_omit_detail_sort_and_payload() -> None:
    ws = WebSocketServer(stats_callback=None)
    for route_idx in range(12):
        for _ in range(128):
            ws._record_boundary_serialization_stage(
                0.2 + route_idx / 100.0,
                64,
                route=f"route_{route_idx}",
                message_type="tracking",
                stage="total",
                include_budget=True,
            )
            ws._record_boundary_serialization_stage(
                0.05,
                0,
                route=f"route_{route_idx}",
                message_type="tracking",
                stage="json_encode",
                include_budget=False,
            )
    started_ns = time.perf_counter_ns()
    metrics = ws.get_boundary_serialization_metrics_compact()
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
    assert metrics["details_included"] is False
    assert metrics["routes"] == {}
    assert metrics["stages"] == {}
    assert len(metrics["top_budget_paths"]) <= 8
    assert elapsed_ms < 30.0


def test_worker_queue_delay_is_visible_and_fails_boundary_budget() -> None:
    async def _run() -> dict:
        ws = WebSocketServer(stats_callback=None)
        client = _FakeSocket()
        blocker_started = threading.Event()
        release_blocker = threading.Event()

        def _occupy_serializer() -> None:
            blocker_started.set()
            release_blocker.wait(timeout=2.0)

        executor = ws._ensure_serializer_executor()
        blocker = executor.submit(_occupy_serializer)
        assert blocker_started.wait(timeout=1.0)
        try:
            send_task = asyncio.create_task(
                ws._send_json_with_boundary_metrics(
                    client,
                    _timed_payload(
                        ws,
                        lambda: {
                            "type": "floorplan_response",
                            "error": "no_depth",
                        },
                    ),
                    route="get_floorplan",
                    message_type="floorplan_response",
                    use_to_thread_json=True,
                )
            )
            await asyncio.sleep(0.02)
            release_blocker.set()
            await send_task
            return ws.get_boundary_serialization_metrics()
        finally:
            release_blocker.set()
            blocker.result(timeout=1.0)
            await ws.stop()

    metrics = asyncio.run(_run())
    assert int(metrics["count"]) == 1
    assert int(metrics["violations"]) == 1
    stages = metrics["stages"]
    wait = stages[
        "ws|get_floorplan|floorplan_response|worker_dispatch_wait|ok"
    ]
    total = stages["ws|get_floorplan|floorplan_response|total|ok"]
    assert float(wait["p99_ms"]) >= 15.0
    assert float(total["p99_ms"]) >= float(wait["p99_ms"])
    assert float(metrics["p99_ms"]) == float(total["p99_ms"])


def test_single_client_transport_delay_is_awaited_but_excluded_from_budget() -> None:
    class _DelayedSocket(_FakeSocket):
        async def send(self, payload: object) -> None:
            await asyncio.sleep(0.02)
            await super().send(payload)

    async def _run() -> dict:
        ws = WebSocketServer(stats_callback=None)
        client = _DelayedSocket()
        started_ns = time.perf_counter_ns()
        await ws._send_json_with_boundary_metrics(
            client,
            _timed_payload(
                ws,
                lambda: {"type": "ma_depth_response", "error": "no_depth"},
            ),
            route="get_ma_depth",
            message_type="ma_depth_response",
        )
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        return ws.get_boundary_serialization_metrics(), elapsed_ms, client

    metrics, elapsed_ms, client = asyncio.run(_run())
    stages = metrics["stages"]
    send = stages["ws|get_ma_depth|ma_depth_response|send_dispatch|ok"]
    total = stages["ws|get_ma_depth|ma_depth_response|total|ok"]
    assert elapsed_ms >= 15.0
    assert client.sent
    assert float(send["p99_ms"]) < 15.0
    assert float(total["p99_ms"]) >= float(send["p99_ms"])
    assert int(metrics["violations"]) == 0


def test_broadcast_transport_delay_is_awaited_but_excluded_from_budget() -> None:
    class _DelayedSocket(_FakeSocket):
        async def send(self, payload: object) -> None:
            await asyncio.sleep(0.02)
            await super().send(payload)

    async def _run() -> dict:
        ws = WebSocketServer(stats_callback=None)
        client = _DelayedSocket()
        ws.connected_clients.add(client)
        started_ns = time.perf_counter_ns()
        await ws.broadcast(
            _timed_payload(
                ws,
                lambda: {"type": "stats", "payload": {"ok": True}},
            ),
        )
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        return ws.get_boundary_serialization_metrics(), elapsed_ms, client

    metrics, elapsed_ms, client = asyncio.run(_run())
    stages = metrics["stages"]
    send = stages["ws|broadcast|stats|send_dispatch|ok"]
    total = stages["ws|broadcast|stats|total|ok"]
    assert elapsed_ms >= 15.0
    assert client.sent
    assert float(send["p99_ms"]) < 15.0
    assert float(total["p99_ms"]) >= float(send["p99_ms"])
    assert int(metrics["violations"]) == 0


def test_broadcast_local_dispatch_failure_cancels_already_created_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_tasks: list[asyncio.Task[object]] = []
    real_create_task = asyncio.create_task

    def _tracked_create_task(
        coroutine: object,
        *args: object,
        **kwargs: object,
    ) -> asyncio.Task[object]:
        task = real_create_task(coroutine, *args, **kwargs)  # type: ignore[arg-type]
        created_tasks.append(task)
        return task

    monkeypatch.setattr(asyncio, "create_task", _tracked_create_task)

    class _BlockingSocket(_FakeSocket):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def send(self, payload: object) -> None:
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled.set()

    class _RaisingSocket(_FakeSocket):
        def send(self, _payload: object) -> None:
            raise RuntimeError("synthetic local dispatch failure")

    class _UntouchedSocket(_FakeSocket):
        calls = 0

        async def send(self, payload: object) -> None:
            self.calls += 1
            await super().send(payload)

    async def _run() -> tuple[WebSocketServer, _BlockingSocket, _UntouchedSocket]:
        ws = WebSocketServer(stats_callback=None)
        first = _BlockingSocket()
        third = _UntouchedSocket()
        ws.connected_clients = [first, _RaisingSocket(), third]  # type: ignore[assignment]
        with pytest.raises(RuntimeError, match="synthetic local dispatch failure"):
            await ws.broadcast(
                _timed_payload(
                    ws,
                    lambda: {"type": "stats", "payload": {}},
                ),
            )
        await asyncio.sleep(0)
        return ws, first, third

    ws, first, third = asyncio.run(_run())
    assert created_tasks
    assert all(task.done() for task in created_tasks)
    assert created_tasks[0].cancelled()
    assert third.calls == 0
    metrics = ws.get_boundary_serialization_metrics()
    key = "ws|broadcast|stats|send_dispatch|RuntimeError"
    assert metrics["boundary_serialization_errors"] == {key: 1}


def test_response_model_duration_is_additive_and_budgeted() -> None:
    async def _run() -> dict:
        ws = WebSocketServer(stats_callback=None)
        client = _FakeSocket()
        ws.connected_clients.add(client)
        await ws.broadcast(
            {"type": "stats", "payload": {}},
            response_model_timing=ws.response_model_timing_since(
                time.perf_counter_ns() - 20_000_000
            ),
        )
        return ws.get_boundary_serialization_metrics()

    metrics = asyncio.run(_run())
    response_model = metrics["stages"][
        "ws|broadcast|stats|response_model|ok"
    ]
    total = metrics["stages"]["ws|broadcast|stats|total|ok"]
    assert response_model["p99_ms"] >= 20.0
    assert total["p99_ms"] >= response_model["p99_ms"]
    assert metrics["violations"] == 1


def test_json_boundary_without_response_model_timing_fails_closed() -> None:
    async def _run() -> tuple[WebSocketServer, _FakeSocket, list[BaseException]]:
        ws = WebSocketServer(stats_callback=None)
        client = _FakeSocket()
        failures: list[BaseException] = []
        ws.connected_clients.add(client)
        ws.boundary_failure_callback = failures.append

        with pytest.raises(BoundaryResponseModelContractError):
            await ws.broadcast({"type": "stats", "payload": {}})
        with pytest.raises(BoundaryResponseModelContractError):
            await ws._coalesce_json_and_maybe_flush(
                {"type": "test_latest", "payload": {"sequence": 1}}
            )
        return ws, client, failures

    ws, client, failures = asyncio.run(_run())
    assert client.sent == []
    assert len(failures) == 2
    assert all(
        isinstance(error, BoundaryResponseModelContractError)
        for error in failures
    )
    metrics = ws.get_boundary_serialization_metrics()
    assert metrics["boundary_serialization_errors"] == {
        "ws|broadcast|stats|response_model|BoundaryResponseModelContractError": 1,
        "ws|broadcast|test_latest|response_model|BoundaryResponseModelContractError": 1,
    }


def test_provider_wait_before_response_assembly_is_excluded() -> None:
    async def _run() -> tuple[dict, float]:
        ws = WebSocketServer(stats_callback=None)
        client = _FakeSocket()
        started_ns = time.perf_counter_ns()
        await asyncio.sleep(0.02)
        response_started_ns = time.perf_counter_ns()
        response = {"type": "ma_depth_response", "ok": True}
        await ws._send_json_with_boundary_metrics(
            client,
            response,
            route="get_ma_depth",
            message_type="ma_depth_response",
            response_model_timing=ws.response_model_timing_since(
                response_started_ns
            ),
        )
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        return ws.get_boundary_serialization_metrics(), elapsed_ms

    metrics, elapsed_ms = asyncio.run(_run())
    assert elapsed_ms >= 15.0
    assert metrics["p99_ms"] < 15.0
    assert metrics["violations"] == 0


def test_broadcast_sync_loop_dispatch_delay_fails_boundary_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduled: list[object] = []

    def _capture(coroutine: object, _loop: object) -> _CapturedFuture:
        scheduled.append(coroutine)
        return _CapturedFuture()

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", _capture)
    ws = WebSocketServer(stats_callback=None)
    ws.event_loop = type("OpenLoop", (), {"is_closed": lambda self: False})()
    ws.connected_clients.add(_FakeSocket())

    ws.broadcast_sync(
        _timed_payload(
            ws,
            lambda: {
                "type": "stats",
                "payload": {"sequence": 1},
            },
        ),
    )
    time.sleep(0.02)
    assert len(scheduled) == 1
    asyncio.run(scheduled.pop())

    metrics = ws.get_boundary_serialization_metrics()
    stages = metrics["stages"]
    enqueue = stages["ws|broadcast|stats|publisher_enqueue|ok"]
    total = stages["ws|broadcast|stats|total|ok"]
    assert float(enqueue["p99_ms"]) >= 15.0
    assert float(total["p99_ms"]) >= float(enqueue["p99_ms"])
    assert int(metrics["violations"]) == 1


def test_generic_coalesced_broadcast_preserves_dispatch_delay_in_eventual_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduled: list[object] = []

    def _capture(coroutine: object, _loop: object) -> _CapturedFuture:
        scheduled.append(coroutine)
        return _CapturedFuture()

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", _capture)
    ws = WebSocketServer(stats_callback=None)
    ws.event_loop = type("OpenLoop", (), {"is_closed": lambda self: False})()
    ws._json_coalesce_interval_by_type["test_latest"] = 0.001
    ws.connected_clients.add(_FakeSocket())

    ws.broadcast_sync(
        _timed_payload(
            ws,
            lambda: {"type": "test_latest", "seq": 1},
        ),
    )
    time.sleep(0.02)

    async def _drain() -> None:
        assert len(scheduled) == 1
        await scheduled.pop()
        assert ws._json_flush_task is not None
        await ws._json_flush_task

    asyncio.run(_drain())
    metrics = ws.get_boundary_serialization_metrics()
    stages = metrics["stages"]
    enqueue = stages["ws|broadcast|test_latest|publisher_enqueue|ok"]
    total = stages["ws|broadcast|test_latest|total|ok"]
    assert float(enqueue["p99_ms"]) >= 15.0
    assert float(total["p99_ms"]) >= float(enqueue["p99_ms"])
    assert int(metrics["violations"]) == 1


def test_generic_json_coalescing_dwell_is_not_in_boundary_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduled: list[object] = []

    def _capture(coroutine: object, _loop: object) -> _CapturedFuture:
        scheduled.append(coroutine)
        return _CapturedFuture()

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", _capture)
    ws = WebSocketServer(stats_callback=None)
    ws.event_loop = type("OpenLoop", (), {"is_closed": lambda self: False})()
    ws._json_coalesce_interval_by_type["test_latest"] = 0.05
    ws._json_last_sent["test_latest:global"] = time.time()
    ws.connected_clients.add(_FakeSocket())

    started_ns = time.perf_counter_ns()
    ws.broadcast_sync(
        _timed_payload(
            ws,
            lambda: {"type": "test_latest", "seq": 2},
        ),
    )

    async def _drain() -> None:
        assert len(scheduled) == 1
        await scheduled.pop()
        assert ws._json_flush_task is not None
        await ws._json_flush_task

    asyncio.run(_drain())
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
    metrics = ws.get_boundary_serialization_metrics()
    stages = metrics["stages"]
    enqueue_ms = float(
        stages["ws|broadcast|test_latest|publisher_enqueue|ok"]["p99_ms"]
    )
    response_model_ms = float(
        stages["ws|broadcast|test_latest|response_model|ok"]["p99_ms"]
    )
    convert_ms = float(
        stages[
            "ws|broadcast|test_latest|admission_freeze_numpy_convert|ok"
        ]["p99_ms"]
    )
    encode_ms = float(
        stages[
            "ws|broadcast|test_latest|admission_freeze_json_encode|ok"
        ]["p99_ms"]
    )
    send_ms = float(
        stages["ws|broadcast|test_latest|send_dispatch|ok"]["p99_ms"]
    )
    total_ms = float(stages["ws|broadcast|test_latest|total|ok"]["p99_ms"])

    assert elapsed_ms >= 35.0
    assert total_ms == pytest.approx(
        response_model_ms + enqueue_ms + convert_ms + encode_ms + send_ms,
    )
    assert total_ms < elapsed_ms - 20.0


def test_owned_serializer_executor_is_joined_during_stop() -> None:
    async def _run() -> tuple[WebSocketServer, list[threading.Thread]]:
        ws = WebSocketServer(stats_callback=None)
        executor = ws._ensure_serializer_executor()
        await asyncio.get_running_loop().run_in_executor(executor, lambda: None)
        worker_threads = list(executor._threads)
        await ws.stop()
        return ws, worker_threads

    ws, worker_threads = asyncio.run(_run())
    assert worker_threads
    assert ws._serializer_executor is None
    assert all(not worker.is_alive() for worker in worker_threads)


def test_provider_quiescence_waits_for_admitted_callback_and_joins_pool() -> None:
    ws = WebSocketServer(stats_callback=None)
    provider_started = threading.Event()
    release_provider = threading.Event()
    provider_finished = threading.Event()
    receipt_holder: list[object] = []

    def _blocked_provider() -> str:
        provider_started.set()
        release_provider.wait(timeout=2.0)
        provider_finished.set()
        return "complete"

    def _invoke_provider() -> None:
        assert asyncio.run(ws._run_blocking_provider(_blocked_provider)) == "complete"

    provider_thread = threading.Thread(target=_invoke_provider)
    provider_thread.start()
    assert provider_started.wait(timeout=1.0)
    executor = ws._provider_executor
    assert executor is not None
    worker_threads = list(executor._threads)

    def _quiesce() -> None:
        receipt_holder.append(ws.quiesce_blocking_providers(timeout_s=1.0))

    quiesce_thread = threading.Thread(target=_quiesce)
    quiesce_thread.start()
    time.sleep(0.03)
    assert quiesce_thread.is_alive()
    assert not provider_finished.is_set()
    with pytest.raises(ProviderAdmissionClosed):
        asyncio.run(ws._run_blocking_provider(lambda: None))

    release_provider.set()
    provider_thread.join(timeout=1.0)
    quiesce_thread.join(timeout=1.0)
    assert not provider_thread.is_alive()
    assert not quiesce_thread.is_alive()
    receipt = receipt_holder[0]
    assert receipt.quiesced is True
    assert receipt.admitted_calls == 1
    assert receipt.completed_calls == 1
    assert receipt.active_calls == 0
    assert receipt.pending_futures == 0
    assert all(not worker.is_alive() for worker in worker_threads)


def test_provider_quiescence_timeout_fails_closed_until_callback_returns() -> None:
    ws = WebSocketServer(stats_callback=None)
    provider_started = threading.Event()
    release_provider = threading.Event()

    def _blocked_provider() -> None:
        provider_started.set()
        release_provider.wait(timeout=2.0)

    provider_thread = threading.Thread(
        target=lambda: asyncio.run(ws._run_blocking_provider(_blocked_provider))
    )
    provider_thread.start()
    assert provider_started.wait(timeout=1.0)
    try:
        with pytest.raises(ProviderQuiescenceTimeout) as exc_info:
            ws.quiesce_blocking_providers(timeout_s=0.02)
        receipt = exc_info.value.receipt
        assert receipt.admission_closed is True
        assert receipt.active_calls == 1
        assert receipt.pending_futures == 1
        assert receipt.executor_joined is False
        assert receipt.quiesced is False
        with pytest.raises(ProviderAdmissionClosed):
            asyncio.run(ws._run_blocking_provider(lambda: None))
    finally:
        release_provider.set()
        provider_thread.join(timeout=1.0)
        ws.quiesce_blocking_providers(timeout_s=1.0)
    assert not provider_thread.is_alive()


def test_admitted_provider_cooperates_with_shutdown_and_receipt_is_complete() -> None:
    ws = WebSocketServer(stats_callback=None)
    provider_started = threading.Event()
    shutdown_requested = threading.Event()
    provider_result: list[str] = []

    def _cooperative_provider() -> str:
        provider_started.set()
        while not shutdown_requested.wait(0.01):
            pass
        return "shutting_down"

    provider_thread = threading.Thread(
        target=lambda: provider_result.append(
            asyncio.run(ws._run_blocking_provider(_cooperative_provider))
        )
    )
    provider_thread.start()
    assert provider_started.wait(timeout=1.0)

    shutdown_requested.set()
    receipt = ws.quiesce_blocking_providers(timeout_s=1.0)
    provider_thread.join(timeout=1.0)

    assert provider_result == ["shutting_down"]
    assert receipt.admitted_calls == receipt.completed_calls == 1
    assert receipt.active_calls == 0
    assert receipt.pending_futures == 0
    assert receipt.executor_joined is True
    assert receipt.quiesced is True


def test_cancelled_provider_waiter_does_not_release_shutdown_lease() -> None:
    ws = WebSocketServer(stats_callback=None)
    provider_started = threading.Event()
    release_provider = threading.Event()

    def _blocked_provider() -> None:
        provider_started.set()
        release_provider.wait(timeout=2.0)

    async def _time_out_waiter() -> None:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                ws._run_blocking_provider(_blocked_provider),
                timeout=0.01,
            )

    asyncio.run(_time_out_waiter())
    assert provider_started.is_set()
    try:
        with pytest.raises(ProviderQuiescenceTimeout) as exc_info:
            ws.quiesce_blocking_providers(timeout_s=0.02)
        assert exc_info.value.receipt.active_calls == 1
    finally:
        release_provider.set()
        ws.quiesce_blocking_providers(timeout_s=1.0)


def test_all_blocking_websocket_callbacks_use_owned_provider_executor() -> None:
    source = (ROOT / "noesis" / "server" / "websocket.py").read_text(
        encoding="utf-8"
    )
    assert "asyncio.to_thread(" not in source
    handle_start = source.index("    async def handle_client(")
    handler_patterns = (
        "if callable(self.auto_calibrate_handler):",
        "if callable(self.pixel_to_world_handler):",
        "if callable(self.set_extrinsics_handler):",
        "if callable(self.solve_pnp_handler):",
        "if callable(self.set_align_handler):",
        "provider = self.ma_depth_provider",
        "provider = getattr(self, 'floorplan_provider', None)",
    )
    for pattern in handler_patterns:
        handler_pos = source.index(pattern, handle_start)
        branch_end = source.index("elif data.get('type')", handler_pos)
        assert "self._run_blocking_provider(" in source[handler_pos:branch_end]


def test_provider_capacity_is_bounded_before_executor_submit() -> None:
    ws = WebSocketServer(stats_callback=None)
    release = threading.Event()

    def _blocked_provider(index: int) -> int:
        release.wait(timeout=2.0)
        return index

    async def _run() -> list[int]:
        tasks = [
            asyncio.create_task(ws._run_blocking_provider(_blocked_provider, index))
            for index in range(ws.PROVIDER_MAX_INFLIGHT)
        ]
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            with ws._provider_condition:
                if ws._provider_active_calls == ws.PROVIDER_MAX_INFLIGHT:
                    break
            await asyncio.sleep(0.001)
        with ws._provider_condition:
            assert ws._provider_active_calls == ws.PROVIDER_MAX_INFLIGHT
            assert len(ws._provider_futures) == ws.PROVIDER_MAX_INFLIGHT
        with pytest.raises(ProviderCapacityExceeded):
            await ws._run_blocking_provider(_blocked_provider, 999)
        release.set()
        return list(await asyncio.gather(*tasks))

    results = asyncio.run(_run())
    assert results == list(range(ws.PROVIDER_MAX_INFLIGHT))
    receipt = ws.quiesce_blocking_providers(timeout_s=1.0)
    assert receipt.quiesced is True
    assert receipt.admitted_calls == ws.PROVIDER_MAX_INFLIGHT


def test_provider_capacity_has_stable_public_error_for_every_rpc() -> None:
    message_types = [
        {"type": "auto_calibrate_pose", "camera": "living-room"},
        {"type": "pixel_to_world", "request_id": "pixel"},
        {"type": "set_extrinsics"},
        {"type": "solve_pnp"},
        {"type": "set_align"},
        {
            "type": "get_ma_depth",
            "camera": "living-room",
            "request_id": "depth",
        },
        {
            "type": "get_floorplan",
            "camera": "living-room",
            "request_id": "floorplan",
        },
    ]

    async def _run() -> tuple[WebSocketServer, _InboundSocket]:
        ws = WebSocketServer(stats_callback=None)
        ws.PROVIDER_MAX_INFLIGHT = 0
        ws.auto_calibrate_handler = lambda _camera: {"ok": True}
        ws.pixel_to_world_handler = lambda _request: {"ok": True}
        ws.set_extrinsics_handler = lambda _request: {"ok": True}
        ws.solve_pnp_handler = lambda _request: {"ok": True}
        ws.set_align_handler = lambda _request: {"ok": True}
        ws.ma_depth_provider = lambda *_args, **_kwargs: {"ok": True}
        ws.floorplan_provider = lambda *_args, **_kwargs: {"ok": True}
        socket = _InboundSocket(message_types)
        await ws.handle_client(socket)
        return ws, socket

    ws, socket = asyncio.run(_run())
    responses = [
        json.loads(str(payload))
        for payload in socket.sent
        if isinstance(payload, str)
    ]
    expected_types = {
        "auto_calibrate_result",
        "pixel_to_world_response",
        "set_extrinsics_result",
        "solve_pnp_result",
        "set_align_result",
        "ma_depth_response",
        "floorplan_response",
    }
    capacity_responses = [
        payload for payload in responses if payload.get("type") in expected_types
    ]
    assert {payload["type"] for payload in capacity_responses} == expected_types
    assert {
        payload.get("error") for payload in capacity_responses
    } == {"provider_capacity_exceeded"}
    receipt = ws.quiesce_blocking_providers(timeout_s=1.0)
    assert receipt.quiesced is True
    assert receipt.admitted_calls == 0


def test_calibration_rpc_timeout_and_shutdown_errors_are_stable() -> None:
    def _slow_pixel_to_world(_request: dict[str, object]) -> dict[str, object]:
        time.sleep(0.03)
        return {"ok": True}

    async def _timeout() -> tuple[WebSocketServer, _InboundSocket]:
        ws = WebSocketServer(stats_callback=None)
        ws._calibration_rpc_timeout = 0.005
        ws.pixel_to_world_handler = _slow_pixel_to_world
        socket = _InboundSocket([{"type": "pixel_to_world"}])
        await ws.handle_client(socket)
        return ws, socket

    timeout_ws, timeout_socket = asyncio.run(_timeout())
    timeout_response = next(
        json.loads(str(payload))
        for payload in timeout_socket.sent
        if isinstance(payload, str)
        and json.loads(str(payload)).get("type") == "pixel_to_world_response"
    )
    assert timeout_response["error"] == "timeout"
    assert timeout_ws.quiesce_blocking_providers(timeout_s=1.0).quiesced is True

    async def _shutdown() -> tuple[WebSocketServer, _InboundSocket]:
        ws = WebSocketServer(stats_callback=None)
        ws.pixel_to_world_handler = lambda _request: {"ok": True}
        assert ws.quiesce_blocking_providers(timeout_s=1.0).quiesced is True
        socket = _InboundSocket([{"type": "pixel_to_world"}])
        await ws.handle_client(socket)
        return ws, socket

    _, shutdown_socket = asyncio.run(_shutdown())
    shutdown_response = next(
        json.loads(str(payload))
        for payload in shutdown_socket.sent
        if isinstance(payload, str)
        and json.loads(str(payload)).get("type") == "pixel_to_world_response"
    )
    assert shutdown_response["error"] == "shutting_down"


def test_concurrent_stats_clear_requests_coalesce_to_one_executor_job() -> None:
    clear_entered = threading.Event()
    release_clear = threading.Event()

    class _Stats:
        calls = 0
        clears = 0

        def __call__(self) -> dict[str, int]:
            self.calls += 1
            return {"sequence": self.calls}

        def clear_stats(self) -> None:
            self.clears += 1
            clear_entered.set()
            release_clear.wait(timeout=2.0)

    async def _run() -> tuple[WebSocketServer, _Stats, list[dict[str, int]]]:
        callback = _Stats()
        ws = WebSocketServer(stats_callback=callback)
        tasks = [
            asyncio.create_task(ws._clear_and_refresh_stats())
            for _ in range(12)
        ]
        assert await asyncio.to_thread(clear_entered.wait, 1.0)
        await asyncio.sleep(0.02)
        with ws._stats_condition:
            assert len(ws._stats_futures) == 1
            assert ws._stats_admitted_snapshots == 1
            assert ws._stats_inflight_kind == "clear"
        release_clear.set()
        results = list(await asyncio.gather(*tasks))
        return ws, callback, results

    ws, callback, results = asyncio.run(_run())
    assert callback.clears == 1
    assert callback.calls == 1
    assert results == [{"sequence": 1}] * 12
    receipt = ws.quiesce_stats_collector(timeout_s=1.0)
    assert receipt.quiesced is True


def test_stats_callback_failure_is_observable_and_never_cached_as_empty() -> None:
    def _failed_stats() -> dict[str, object]:
        raise ValueError("synthetic stats failure")

    ws = WebSocketServer(stats_callback=_failed_stats)
    with pytest.raises(ValueError, match="synthetic stats failure"):
        asyncio.run(ws._get_stats_snapshot())

    metrics = ws.get_boundary_serialization_metrics_compact()
    assert metrics["stats_collection_failures_total"] == 1
    assert metrics["stats_collection_last_failure_type"] == "ValueError"
    assert ws._stats_cache is None
    receipt = ws.quiesce_stats_collector(timeout_s=1.0)
    assert receipt.quiesced is True
    assert receipt.failed_snapshots == 1


def test_closed_broadcast_client_is_lifecycle_noise_not_runtime_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _ClosedSocket(_FakeSocket):
        async def send(self, _payload: object) -> None:
            raise ConnectionClosedOK(Close(1000, "complete"), None, None)

    async def _run() -> WebSocketServer:
        ws = WebSocketServer(stats_callback=None)
        client = _ClosedSocket()
        ws.connected_clients.add(client)
        await ws.broadcast(
            _timed_payload(
                ws,
                lambda: {"type": "stats", "payload": {"ok": True}},
            ),
        )
        return ws

    with caplog.at_level(logging.DEBUG, logger="WebSocketServer"):
        ws = asyncio.run(_run())
    assert not ws.connected_clients
    assert not [row for row in caplog.records if row.levelno >= logging.ERROR]


def test_websocket_stop_propagates_unproven_listener_quiescence() -> None:
    class _BrokenServer:
        sockets = ()

        def close(self) -> None:
            return None

        async def wait_closed(self) -> None:
            raise RuntimeError("synthetic listener failure")

    ws = WebSocketServer(stats_callback=None)
    ws.server = _BrokenServer()
    with pytest.raises(RuntimeError, match="synthetic listener failure"):
        asyncio.run(ws.stop())


def test_websocket_stop_aborts_hostile_client_and_drains_boundary_handler() -> None:
    class _Transport:
        def __init__(self) -> None:
            self.aborted = False
            self.abort_event = asyncio.Event()

        def abort(self) -> None:
            self.aborted = True
            self.abort_event.set()

    class _HostileClient(_FakeSocket):
        def __init__(self) -> None:
            super().__init__()
            self.transport = _Transport()

        async def close(self) -> None:
            await self.transport.abort_event.wait()

    class _BoundaryServer:
        sockets = ()

        def __init__(self) -> None:
            self.closed = False
            self.handler_drained = False
            self._late_handler = None

        def close(self) -> None:
            self.closed = True

            async def _handler_admitted_at_close() -> None:
                await asyncio.sleep(0.03)
                self.handler_drained = True

            self._late_handler = asyncio.create_task(_handler_admitted_at_close())

        async def wait_closed(self) -> None:
            assert self._late_handler is not None
            await self._late_handler

    async def _run() -> tuple[_HostileClient, _BoundaryServer, WebSocketServer]:
        ws = WebSocketServer(stats_callback=None)
        ws.CLIENT_CLOSE_GRACE_S = 0.01
        ws.CLIENT_ABORT_DRAIN_S = 0.1
        ws.LISTENER_SHUTDOWN_TIMEOUT_S = 0.2
        client = _HostileClient()
        server = _BoundaryServer()
        ws.connected_clients.add(client)
        ws.server = server
        await ws.stop()
        return client, server, ws

    client, server, ws = asyncio.run(_run())
    assert client.transport.aborted is True
    assert server.closed is True
    assert server.handler_drained is True
    assert ws.connected_clients == set()
    assert ws.server is None


def test_websocket_stop_rejects_server_handler_past_total_deadline() -> None:
    class _NeverClosedServer:
        sockets = ()

        def close(self) -> None:
            return None

        async def wait_closed(self) -> None:
            await asyncio.Event().wait()

    async def _run() -> None:
        ws = WebSocketServer(stats_callback=None)
        ws.LISTENER_SHUTDOWN_TIMEOUT_S = 0.02
        ws.server = _NeverClosedServer()
        await ws.stop()

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(_run())


@pytest.mark.parametrize("method_name", ["broadcast_sync", "send_to_client_sync"])
def test_stop_waits_cross_thread_submission_admitted_before_barrier(
    method_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop_ready = threading.Event()
    stop_loop = threading.Event()
    loop_holder: dict[str, asyncio.AbstractEventLoop] = {}

    def _run_loop() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop_holder["loop"] = loop
        loop_ready.set()
        loop.run_until_complete(asyncio.to_thread(stop_loop.wait))
        loop.close()

    loop_thread = threading.Thread(target=_run_loop)
    loop_thread.start()
    assert loop_ready.wait(timeout=1.0)
    ws = WebSocketServer(event_loop=loop_holder["loop"], stats_callback=None)
    admitted = threading.Event()
    release_schedule = threading.Event()
    original_submit = asyncio.run_coroutine_threadsafe

    def _paused_submit(coroutine, loop):
        admitted.set()
        release_schedule.wait(timeout=2.0)
        return original_submit(coroutine, loop)

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", _paused_submit)
    message = _timed_payload(
        ws,
        lambda: {"type": "stats", "payload": {}},
    )
    if method_name == "broadcast_sync":
        submit_args = (message,)
    else:
        submit_args = (_FakeSocket(), message)
    submission_thread = threading.Thread(
        target=getattr(ws, method_name),
        args=submit_args,
    )
    submission_thread.start()
    assert admitted.wait(timeout=1.0)
    quiesce_result: list[object] = []

    async def _quiesce() -> None:
        quiesce_result.append(
            await ws.quiesce_outbound_submissions(timeout_s=1.0)
        )

    quiesce_future = original_submit(_quiesce(), loop_holder["loop"])
    time.sleep(0.03)
    assert not quiesce_future.done()

    release_schedule.set()
    submission_thread.join(timeout=1.0)
    assert not submission_thread.is_alive()
    quiesce_future.result(timeout=1.0)
    receipt = quiesce_result[0]
    assert receipt.quiesced is True
    assert receipt.admitted_submissions == 1
    assert receipt.completed_submissions == 1
    assert receipt.pending_futures == 0
    assert ws._outbound_futures == set()
    stop_loop.set()
    loop_thread.join(timeout=1.0)
    assert not loop_thread.is_alive()


def test_outbound_quiescence_fails_closed_for_live_submission() -> None:
    async def _run() -> None:
        ws = WebSocketServer(stats_callback=None)
        loop = asyncio.get_running_loop()
        ws.event_loop = loop
        release = asyncio.Event()

        async def _blocked() -> None:
            await release.wait()

        admission = ws._submit_outbound_coroutine(_blocked)
        assert isinstance(admission, OutboundAdmissionReceipt)
        assert admission.submission_id == 1
        assert admission.message_count == 1
        assert admission.admitted_at_ns > 0
        try:
            with pytest.raises(OutboundQuiescenceTimeout) as exc_info:
                await ws.quiesce_outbound_submissions(timeout_s=0.01)
            assert exc_info.value.receipt.pending_futures == 1
        finally:
            release.set()
            await asyncio.sleep(0)
            await asyncio.sleep(0)

    asyncio.run(_run())


def test_outbound_admission_raises_typed_lifecycle_and_capacity_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduled: list[object] = []

    def _capture(coroutine: object, _loop: object) -> _CapturedFuture:
        scheduled.append(coroutine)
        return _CapturedFuture()

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", _capture)
    open_loop = type("OpenLoop", (), {"is_closed": lambda self: False})()

    no_loop = WebSocketServer(stats_callback=None)
    with pytest.raises(OutboundAdmissionClosed, match="event loop is unavailable"):
        no_loop.broadcast_sync(
            _timed_payload(no_loop, lambda: {"type": "stats", "payload": {}})
        )

    closed = WebSocketServer(event_loop=open_loop, stats_callback=None)
    closed._outbound_admission_open = False
    with pytest.raises(OutboundAdmissionClosed, match="admission is closed"):
        closed.broadcast_sync(
            _timed_payload(closed, lambda: {"type": "stats", "payload": {}})
        )

    stopped = WebSocketServer(event_loop=open_loop, stats_callback=None)
    stopped.running = False
    with pytest.raises(OutboundAdmissionClosed, match="server is stopped"):
        stopped.broadcast_sync(
            _timed_payload(stopped, lambda: {"type": "stats", "payload": {}})
        )

    saturated = WebSocketServer(event_loop=open_loop, stats_callback=None)
    saturated._outbound_max_inflight = 0
    with pytest.raises(OutboundCapacityExceeded, match="capacity is exhausted"):
        saturated.broadcast_sync(
            _timed_payload(
                saturated,
                lambda: {"type": "stats", "payload": {}},
            )
        )

    assert scheduled == []


def test_broadcast_batch_sync_is_all_or_none_and_preserves_exact_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduled: list[object] = []

    def _capture(coroutine: object, _loop: object) -> _CapturedFuture:
        scheduled.append(coroutine)
        return _CapturedFuture()

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", _capture)
    ws = WebSocketServer(stats_callback=None)
    ws.event_loop = type("OpenLoop", (), {"is_closed": lambda self: False})()
    client = _FakeSocket()
    ws.connected_clients.add(client)
    valid = _timed_payload(
        ws,
        lambda: {"type": "test_batch", "sequence": 1},
    )

    with pytest.raises(TypeError, match="batch messages"):
        ws.broadcast_batch_sync([valid, object()])
    assert scheduled == []
    assert client.sent == []

    batch = [
        _timed_payload(
            ws,
            lambda: {"type": "test_batch_a", "sequence": 1},
        ),
        _timed_payload(
            ws,
            lambda: {"type": "test_batch_b", "sequence": 1},
        ),
        _timed_payload(
            ws,
            lambda: {"type": "test_batch_c", "sequence": 1},
        ),
    ]
    receipt = ws.broadcast_batch_sync(batch)
    assert isinstance(receipt, OutboundAdmissionReceipt)
    assert receipt.submission_id == 1
    assert receipt.message_count == 3
    assert len(scheduled) == 1

    asyncio.run(scheduled.pop())
    sent = [json.loads(str(payload)) for payload in client.sent]
    assert sent == [dict(message) for message in batch]


def test_broadcast_batch_sync_freezes_caller_values_at_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduled: list[object] = []

    def _capture(coroutine: object, _loop: object) -> _CapturedFuture:
        scheduled.append(coroutine)
        return _CapturedFuture()

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", _capture)
    ws = WebSocketServer(stats_callback=None)
    ws.event_loop = type("OpenLoop", (), {"is_closed": lambda self: False})()
    client = _FakeSocket()
    ws.connected_clients.add(client)
    tracks = [{"track_id": 7, "labels": ["resident"]}]
    message = _timed_payload(
        ws,
        lambda: {"type": "test_batch", "tracks": tracks},
    )

    receipt = ws.broadcast_batch_sync([message])
    tracks[0]["track_id"] = 99
    tracks[0]["labels"].append("mutated")
    message["tracks"].append({"track_id": 100})

    assert receipt.message_count == 1
    asyncio.run(scheduled.pop())
    assert [json.loads(str(payload)) for payload in client.sent] == [
        {
            "type": "test_batch",
            "tracks": [{"track_id": 7, "labels": ["resident"]}],
        }
    ]


def test_gated_batch_withholds_exact_bytes_until_commit_then_releases_in_order() -> None:
    async def _run() -> tuple[list[dict[str, object]], object, int]:
        ws = WebSocketServer(stats_callback=None)
        ws.event_loop = asyncio.get_running_loop()
        client = _FakeSocket()
        ws.connected_clients.add(client)
        tracks = [{"track_id": 7}]
        messages = _canonical_tracking_batch(
            ws,
            sequence=1,
            tracks=tracks,
        )

        admission = ws.admit_broadcast_batch_sync(messages)
        assert isinstance(admission, GatedOutboundAdmission)
        assert admission.receipt.message_count == 3
        assert ws._outbound_inflight_bytes == admission.receipt.payload_bytes
        tracks[0]["track_id"] = 99
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert client.sent == []

        committed = admission.commit_then_release(lambda: "committed")
        assert committed == "committed"
        shutdown = await ws.quiesce_outbound_submissions(timeout_s=1.0)
        return (
            [json.loads(str(item)) for item in client.sent],
            shutdown,
            admission.receipt.payload_bytes,
        )

    sent, shutdown, payload_bytes = asyncio.run(_run())
    assert [item["type"] for item in sent] == [
        "tracking",
        "world_snapshot",
        "world_event",
    ]
    assert sent[0]["tracks"] == [{"track_id": 7}]
    assert shutdown.quiesced is True
    assert shutdown.aborted_submissions == 0
    assert shutdown.inflight_bytes == 0
    assert shutdown.completed_bytes == payload_bytes


def test_concurrent_gated_batches_hold_admission_order_without_interleaving() -> None:
    class _YieldingSocket(_FakeSocket):
        async def send(self, payload: object) -> None:
            self.sent.append(payload)
            await asyncio.sleep(0)

    async def _run() -> list[dict[str, object]]:
        ws = WebSocketServer(stats_callback=None)
        ws.event_loop = asyncio.get_running_loop()
        client = _YieldingSocket()
        ws.connected_clients.add(client)
        first = ws.admit_broadcast_batch_sync(
            _canonical_tracking_batch(ws, sequence=1)
        )
        second = ws.admit_broadcast_batch_sync(
            _canonical_tracking_batch(ws, sequence=2)
        )

        # Let the first admitted task own the canonical lane, then resolve the
        # second gate first. It must remain behind the unresolved first cohort.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        second.commit_then_release(lambda: None)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert client.sent == []

        first.commit_then_release(lambda: None)
        await ws.quiesce_outbound_submissions(timeout_s=1.0)
        return [json.loads(str(item)) for item in client.sent]

    sent = asyncio.run(_run())
    assert [(item["type"], item["tracking_publication_sequence"]) for item in sent] == [
        ("tracking", 1),
        ("world_snapshot", 1),
        ("world_event", 1),
        ("tracking", 2),
        ("world_snapshot", 2),
        ("world_event", 2),
    ]


def test_gated_batch_commit_failure_aborts_without_delivery_or_byte_leak() -> None:
    async def _run() -> tuple[GatedOutboundAdmission, object, list[object]]:
        ws = WebSocketServer(stats_callback=None)
        ws.event_loop = asyncio.get_running_loop()
        client = _FakeSocket()
        ws.connected_clients.add(client)
        admission = ws.admit_broadcast_batch_sync(
            _canonical_tracking_batch(
                ws,
                sequence=0,
                with_world=False,
            )
        )
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="journal failed"):
            admission.commit_then_release(
                lambda: (_ for _ in ()).throw(RuntimeError("journal failed"))
            )
        shutdown = await ws.quiesce_outbound_submissions(timeout_s=1.0)
        return admission, shutdown, client.sent

    admission, shutdown, sent = asyncio.run(_run())
    assert admission.state == "aborted"
    assert sent == []
    assert shutdown.quiesced is True
    assert shutdown.failed_submissions == 0
    assert shutdown.aborted_submissions == 1
    assert shutdown.aborted_bytes == admission.receipt.payload_bytes
    assert shutdown.inflight_bytes == 0


def test_gated_batch_shutdown_waits_for_resolution_and_closes_new_admission() -> None:
    async def _run() -> object:
        ws = WebSocketServer(stats_callback=None)
        ws.event_loop = asyncio.get_running_loop()
        admission = ws.admit_broadcast_batch_sync(
            _canonical_tracking_batch(
                ws,
                sequence=0,
                with_world=False,
            )
        )
        quiesce = asyncio.create_task(
            ws.quiesce_outbound_submissions(timeout_s=1.0)
        )
        await asyncio.sleep(0)
        assert not quiesce.done()
        with pytest.raises(OutboundAdmissionClosed):
            ws.broadcast_sync(
                _timed_payload(
                    ws,
                    lambda: {"type": "stats", "payload": {}},
                )
            )
        with pytest.raises(RuntimeError, match="shutdown abort"):
            admission.commit_then_release(
                lambda: (_ for _ in ()).throw(
                    RuntimeError("shutdown abort")
                )
            )
        return await quiesce

    shutdown = asyncio.run(_run())
    assert shutdown.quiesced is True
    assert shutdown.aborted_submissions == 1
    assert shutdown.inflight_bytes == 0


def test_cancelled_gated_batch_is_not_releasable_and_releases_bytes() -> None:
    async def _run() -> tuple[GatedOutboundAdmission, WebSocketServer]:
        ws = WebSocketServer(stats_callback=None)
        ws.event_loop = asyncio.get_running_loop()
        admission = ws.admit_broadcast_batch_sync(
            _canonical_tracking_batch(
                ws,
                sequence=0,
                with_world=False,
            )
        )
        future = next(iter(ws._outbound_futures))
        assert future.cancel() is True
        for _ in range(10):
            await asyncio.sleep(0)
        with pytest.raises(OutboundGateResolutionError):
            admission.commit_then_release(lambda: None)
        await ws.quiesce_outbound_submissions(timeout_s=1.0)
        return admission, ws

    admission, ws = asyncio.run(_run())
    assert admission.state == "resolution_failed"
    assert ws._outbound_failed_submissions == 1
    assert ws._outbound_inflight_bytes == 0
    assert ws._outbound_completed_bytes == admission.receipt.payload_bytes


@pytest.mark.parametrize(
    ("message_type", "required_route"),
    [
        ("tracking", "admit_broadcast_batch_sync"),
        ("world_snapshot", "admit_broadcast_batch_sync"),
        ("world_event", "admit_broadcast_batch_sync"),
        ("bev-frame", "broadcast_bev_sync"),
        ("bev-status", "broadcast_bev_sync"),
    ],
)
def test_canonical_message_routing_table_rejects_generic_paths(
    message_type: str,
    required_route: str,
) -> None:
    ws = WebSocketServer(stats_callback=None)
    ws.event_loop = type("OpenLoop", (), {"is_closed": lambda self: False})()
    ws._json_coalesce_interval_by_type[message_type] = 0.01
    message = _timed_payload(
        ws,
        lambda: {"type": message_type, "payload": {}},
    )
    expected = f"use {required_route}"

    with pytest.raises(CanonicalOutboundRouteRequired, match=expected):
        ws.broadcast_sync(message)
    with pytest.raises(CanonicalOutboundRouteRequired, match=expected):
        ws.broadcast_batch_sync([message])
    with pytest.raises(CanonicalOutboundRouteRequired, match=expected):
        ws.send_to_client_sync(_FakeSocket(), message)
    with pytest.raises(CanonicalOutboundRouteRequired, match=expected):
        asyncio.run(ws.broadcast(message))
    assert ws._should_coalesce_json_message(message) is False
    assert ws._outbound_admitted_submissions == 0


@pytest.mark.parametrize(
    "raw_message",
    [
        ' {"type":"world_event","payload":{}}',
        b'\t{"type":"tracking","tracks":[]}',
        b'\xef\xbb\xbf{"type":"world_snapshot","payload":{}}',
    ],
)
def test_raw_encoded_canonical_json_cannot_bypass_dedicated_routes(
    raw_message: str | bytes,
) -> None:
    ws = WebSocketServer(stats_callback=None)
    ws.event_loop = type("OpenLoop", (), {"is_closed": lambda self: False})()

    with pytest.raises(CanonicalOutboundRouteRequired):
        ws.broadcast_sync(raw_message)
    with pytest.raises(CanonicalOutboundRouteRequired):
        ws.broadcast_batch_sync([raw_message])
    with pytest.raises(CanonicalOutboundRouteRequired):
        ws.send_to_client_sync(_FakeSocket(), raw_message)
    with pytest.raises(CanonicalOutboundRouteRequired):
        asyncio.run(ws.broadcast(raw_message))
    assert ws._outbound_admitted_submissions == 0


def test_noncanonical_raw_text_and_binary_traffic_remains_available() -> None:
    async def _run() -> list[object]:
        ws = WebSocketServer(stats_callback=None)
        ws.event_loop = asyncio.get_running_loop()
        client = _FakeSocket()
        ws.connected_clients.add(client)
        text_receipt = ws.send_to_client_sync(
            client,
            '{"type":"operator_note","value":1}',
        )
        binary_receipt = ws.send_to_client_sync(client, b"\x00\xffraw")
        assert text_receipt.submission_id < binary_receipt.submission_id
        await ws.quiesce_outbound_submissions(timeout_s=1.0)
        return client.sent

    assert asyncio.run(_run()) == [
        '{"type":"operator_note","value":1}',
        b"\x00\xffraw",
    ]


def test_caller_constructed_frozen_json_is_not_trusted() -> None:
    ws = WebSocketServer(stats_callback=None)
    encoded = '{"type":"world_event","payload":{}}'
    forged = FrozenOutboundJSON(
        encoded=encoded,
        payload_bytes=len(encoded),
        message_type="stats",
        convert_ms=0.0,
        encode_ms=0.0,
        route="broadcast",
        coalesce_key=None,
        telemetry_payload=None,
        owner_token=object(),
    )
    with pytest.raises(FrozenOutboundOwnershipError, match="does not belong"):
        asyncio.run(ws.broadcast(forged))

    mismatched = forged._replace(owner_token=ws._frozen_outbound_token)
    with pytest.raises(FrozenOutboundOwnershipError, match="type does not match"):
        asyncio.run(ws.broadcast(mismatched))


def test_dedicated_bev_route_returns_typed_admission_and_is_noncoalesced() -> None:
    async def _run() -> tuple[object, list[dict[str, object]]]:
        ws = WebSocketServer(stats_callback=None)
        ws.event_loop = asyncio.get_running_loop()
        ws._json_coalesce_interval_by_type["bev-frame"] = 0.01
        client = _FakeSocket()
        ws.connected_clients.add(client)
        receipt = ws.broadcast_bev_sync(
            _timed_payload(
                ws,
                lambda: {
                    "type": "bev-frame",
                    "cameraId": "kitchen",
                    "trackingPublicationSequence": 7,
                },
            )
        )
        await ws.quiesce_outbound_submissions(timeout_s=1.0)
        return receipt, [json.loads(str(item)) for item in client.sent]

    receipt, sent = asyncio.run(_run())
    assert receipt.message_count == 1
    assert sent == [
        {
            "type": "bev-frame",
            "cameraId": "kitchen",
            "trackingPublicationSequence": 7,
        }
    ]


def test_dedicated_bev_status_route_preserves_type_and_response_timing() -> None:
    async def _run() -> tuple[object, list[dict[str, object]]]:
        ws = WebSocketServer(stats_callback=None)
        ws.event_loop = asyncio.get_running_loop()
        ws._json_coalesce_interval_by_type["bev-status"] = 0.01
        client = _FakeSocket()
        ws.connected_clients.add(client)
        receipt = ws.broadcast_bev_sync(
            _timed_payload(
                ws,
                lambda: {
                    "type": "bev-status",
                    "cameraId": "kitchen",
                    "sourceId": 1,
                    "frameId": 5,
                    "observedAtUs": 1_000_005,
                    "trackingPublicationSequence": 4,
                    "trackingOutboundSubmissionId": 9,
                    "cohort": {
                        "source_id": 1,
                        "frame_id": 5,
                        "observed_at_us": 1_000_005,
                        "tracking_publication_sequence": 4,
                    },
                    "error": "homography_failed",
                },
            )
        )
        await ws.quiesce_outbound_submissions(timeout_s=1.0)
        return receipt, [json.loads(str(item)) for item in client.sent]

    receipt, sent = asyncio.run(_run())
    assert receipt.message_count == 1
    assert sent[0]["type"] == "bev-status"
    assert sent[0]["trackingOutboundSubmissionId"] == 9


def test_authority_gated_batch_rejects_mixed_or_drifted_cohorts() -> None:
    ws = WebSocketServer(stats_callback=None)
    ws.event_loop = type("OpenLoop", (), {"is_closed": lambda self: False})()

    not_tracking_first = _canonical_tracking_batch(ws, sequence=1)
    not_tracking_first[0], not_tracking_first[1] = (
        not_tracking_first[1],
        not_tracking_first[0],
    )
    with pytest.raises(ValueError, match="begin with tracking"):
        ws.admit_broadcast_batch_sync(not_tracking_first)

    mixed = _canonical_tracking_batch(ws, sequence=1, with_world=False)
    mixed.append(
        _timed_payload(ws, lambda: {"type": "bev-frame", "cameraId": "kitchen"})
    )
    with pytest.raises(ValueError, match="cannot contain extra messages"):
        ws.admit_broadcast_batch_sync(mixed)

    drifted = _canonical_tracking_batch(ws, sequence=1)
    drifted[2]["cohort"] = {
        **dict(drifted[2]["cohort"]),
        "frame_id": 999,
    }
    with pytest.raises(ValueError, match="event message is not exact"):
        ws.admit_broadcast_batch_sync(drifted)

    assert ws._outbound_admitted_submissions == 0


def test_authority_gated_api_has_only_publisher_production_call_sites() -> None:
    hits: set[str] = set()
    bev_hits: set[str] = set()
    for path in ROOT.rglob("*.py"):
        relative = path.relative_to(ROOT)
        if any(part in {"tests", "archive", ".git"} for part in relative.parts):
            continue
        if "admit_broadcast_batch_sync" in path.read_text(
            encoding="utf-8",
            errors="ignore",
        ):
            hits.add(relative.as_posix())
        if "broadcast_bev_sync" in path.read_text(
            encoding="utf-8",
            errors="ignore",
        ):
            bev_hits.add(relative.as_posix())
    assert hits == {
        "noesis/server/websocket.py",
        "noesis/telemetry/publishers.py",
        "DS9/noesis/telemetry/publishers.py",
    }
    assert bev_hits == {
        "noesis/server/websocket.py",
        "noesis/telemetry/bev.py",
    }


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_single_json_rejects_nonfinite_before_outbound_admission(
    monkeypatch: pytest.MonkeyPatch,
    nonfinite: float,
) -> None:
    scheduled: list[object] = []

    def _capture(coroutine: object, _loop: object) -> _CapturedFuture:
        scheduled.append(coroutine)
        return _CapturedFuture()

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", _capture)
    ws = WebSocketServer(stats_callback=None)
    ws.event_loop = type("OpenLoop", (), {"is_closed": lambda self: False})()
    failures: list[BaseException] = []
    ws.boundary_failure_callback = failures.append

    with pytest.raises(ValueError, match="Out of range float values"):
        ws.broadcast_sync(
            _timed_payload(
                ws,
                lambda: {"type": "stats", "value": nonfinite},
            )
        )

    assert scheduled == []
    assert ws._outbound_admitted_submissions == 0
    assert len(failures) == 1
    assert isinstance(failures[0], ValueError)


def test_batch_json_rejects_nested_nonfinite_all_or_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduled: list[object] = []

    def _capture(coroutine: object, _loop: object) -> _CapturedFuture:
        scheduled.append(coroutine)
        return _CapturedFuture()

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", _capture)
    ws = WebSocketServer(stats_callback=None)
    ws.event_loop = type("OpenLoop", (), {"is_closed": lambda self: False})()
    client = _FakeSocket()
    ws.connected_clients.add(client)
    batch = [
        _timed_payload(ws, lambda: {"type": "test_batch_a", "sequence": 0}),
        _timed_payload(
            ws,
            lambda: {
                "type": "test_batch_b",
                "payload": {"position": [1.0, float("nan")]},
            },
        ),
    ]

    with pytest.raises(ValueError, match="Out of range float values"):
        ws.broadcast_batch_sync(batch)

    assert scheduled == []
    assert client.sent == []
    assert ws._outbound_admitted_submissions == 0


def test_sync_json_is_encoded_once_and_reused_for_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from noesis.server import websocket as websocket_module

    scheduled: list[object] = []
    encode_calls = 0
    original = websocket_module._serialize_boundary_json

    def _capture(coroutine: object, _loop: object) -> _CapturedFuture:
        scheduled.append(coroutine)
        return _CapturedFuture()

    def _counted(*args, **kwargs):
        nonlocal encode_calls
        encode_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", _capture)
    monkeypatch.setattr(websocket_module, "_serialize_boundary_json", _counted)
    ws = WebSocketServer(stats_callback=None)
    ws.event_loop = type("OpenLoop", (), {"is_closed": lambda self: False})()
    client = _FakeSocket()
    ws.connected_clients.add(client)

    receipt = ws.broadcast_sync(
        _timed_payload(
            ws,
            lambda: {"type": "stats", "sequence": 0},
        )
    )
    asyncio.run(scheduled.pop())

    assert receipt.message_count == 1
    assert encode_calls == 1
    assert [json.loads(str(item)) for item in client.sent] == [
        {"type": "stats", "sequence": 0}
    ]


def test_send_to_client_sync_freezes_caller_values_and_route_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduled: list[object] = []

    def _capture(coroutine: object, _loop: object) -> _CapturedFuture:
        scheduled.append(coroutine)
        return _CapturedFuture()

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", _capture)
    ws = WebSocketServer(stats_callback=None)
    ws.event_loop = type("OpenLoop", (), {"is_closed": lambda self: False})()
    client = _FakeSocket()
    ws.connected_clients.add(client)
    payload = {"type": "webrtc_answer", "nested": {"sequence": 1}}
    message = _timed_payload(ws, lambda: payload)

    receipt = ws.send_to_client_sync(client, message)
    payload["nested"]["sequence"] = 99
    message["nested"]["extra"] = True
    asyncio.run(scheduled.pop())

    assert receipt.message_count == 1
    assert [json.loads(str(item)) for item in client.sent] == [
        {"type": "webrtc_answer", "nested": {"sequence": 1}}
    ]
    stages = ws.get_boundary_serialization_metrics()["stages"]
    assert (
        "ws|send_to_client|webrtc_answer|admission_freeze_json_encode|ok"
        in stages
    )
    assert "ws|send_to_client|webrtc_answer|total|ok" in stages


def test_send_to_client_sync_rejects_nonfinite_before_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduled: list[object] = []

    def _capture(coroutine: object, _loop: object) -> _CapturedFuture:
        scheduled.append(coroutine)
        return _CapturedFuture()

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", _capture)
    ws = WebSocketServer(stats_callback=None)
    ws.event_loop = type("OpenLoop", (), {"is_closed": lambda self: False})()
    client = _FakeSocket()
    ws.connected_clients.add(client)

    with pytest.raises(ValueError, match="Out of range float values"):
        ws.send_to_client_sync(
            client,
            _timed_payload(
                ws,
                lambda: {"type": "webrtc_answer", "value": float("nan")},
            ),
        )

    assert scheduled == []
    assert client.sent == []
    assert ws._outbound_admitted_submissions == 0


def test_async_outbound_failure_is_accounted_and_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduled: list[object] = []
    future: concurrent.futures.Future[None] = concurrent.futures.Future()

    def _capture(coroutine: object, _loop: object):
        scheduled.append(coroutine)
        return future

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", _capture)
    ws = WebSocketServer(stats_callback=None)
    ws.event_loop = type("OpenLoop", (), {"is_closed": lambda self: False})()
    failures: list[BaseException] = []
    ws.boundary_failure_callback = failures.append

    admission = ws.broadcast_sync(
        _timed_payload(ws, lambda: {"type": "stats", "payload": {}})
    )
    failure = RuntimeError("synthetic async send failure")
    future.set_exception(failure)

    assert admission.submission_id == 1
    assert failures == [failure]
    assert ws._outbound_completed_submissions == 1
    assert ws._outbound_failed_submissions == 1
    assert ws._outbound_last_failure_type == "RuntimeError"
    assert ws._outbound_futures == set()
    assert admission.payload_bytes > 0
    assert ws._outbound_inflight_bytes == 0
    assert ws._outbound_completed_bytes == admission.payload_bytes
    quiescence = asyncio.run(ws.quiesce_outbound_submissions(timeout_s=0.01))
    assert quiescence.failed_submissions == 1
    assert quiescence.inflight_bytes == 0
    assert quiescence.quiesced is True
    scheduled.pop().close()


def test_outbound_byte_capacity_releases_and_readmits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduled: list[object] = []
    futures: list[concurrent.futures.Future[None]] = []

    def _capture(coroutine: object, _loop: object):
        scheduled.append(coroutine)
        future: concurrent.futures.Future[None] = concurrent.futures.Future()
        futures.append(future)
        return future

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", _capture)
    ws = WebSocketServer(stats_callback=None)
    ws.event_loop = type("OpenLoop", (), {"is_closed": lambda self: False})()
    ws._outbound_max_inflight_bytes = 200
    message = _timed_payload(
        ws,
        lambda: {"type": "stats", "value": "x" * 120},
    )

    first = ws.broadcast_sync(message)
    assert 0 < first.payload_bytes <= 200
    assert first.inflight_bytes == first.payload_bytes
    with pytest.raises(OutboundCapacityExceeded, match="byte capacity"):
        ws.broadcast_sync(message)
    assert len(scheduled) == 1
    assert ws._outbound_inflight_bytes == first.payload_bytes

    futures[0].set_result(None)
    scheduled[0].close()
    assert ws._outbound_inflight_bytes == 0
    assert ws._outbound_completed_bytes == first.payload_bytes

    second = ws.broadcast_sync(message)
    assert second.submission_id == 2
    assert second.payload_bytes == first.payload_bytes
    futures[1].set_result(None)
    scheduled[1].close()
    shutdown = asyncio.run(
        ws.quiesce_outbound_submissions(timeout_s=0.01)
    )
    assert shutdown.quiesced is True
    assert shutdown.inflight_bytes == 0
    assert shutdown.admitted_bytes == first.payload_bytes * 2
    assert shutdown.completed_bytes == shutdown.admitted_bytes
    metrics = ws.get_boundary_serialization_metrics()["outbound_admission"]
    assert metrics["inflight_bytes"] == 0
    assert metrics["peak_inflight_bytes"] == first.payload_bytes
    assert metrics["max_inflight_bytes"] == 200


def test_cancelled_outbound_submission_releases_bytes_without_underflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduled: list[object] = []
    future: concurrent.futures.Future[None] = concurrent.futures.Future()

    def _capture(coroutine: object, _loop: object):
        scheduled.append(coroutine)
        return future

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", _capture)
    ws = WebSocketServer(stats_callback=None)
    ws.event_loop = type("OpenLoop", (), {"is_closed": lambda self: False})()

    admission = ws.broadcast_sync(
        _timed_payload(
            ws,
            lambda: {"type": "stats", "sequence": 1},
        )
    )
    assert ws._outbound_inflight_bytes == admission.payload_bytes
    assert future.cancel() is True

    assert ws._outbound_inflight_bytes == 0
    assert ws._outbound_completed_bytes == admission.payload_bytes
    assert ws._outbound_failed_submissions == 1
    assert ws._outbound_bytes_by_future == {}
    scheduled.pop().close()


def test_client_handler_re_raises_closed_connections_to_outer_lifecycle() -> None:
    source = (ROOT / "noesis" / "server" / "websocket.py").read_text(
        encoding="utf-8"
    )
    assert "except websockets.exceptions.ConnectionClosed:\n" in source
    assert "Treating it as a request" in source


def test_tracking_json_broadcasts_are_lossless_even_if_coalescing_is_configured() -> None:
    async def _run() -> tuple[list[dict], bool]:
        ws = WebSocketServer(stats_callback=None)
        ws.event_loop = asyncio.get_running_loop()
        ws._json_coalesce_interval_by_type["tracking"] = 0.01
        client = _FakeSocket()
        ws.connected_clients.add(client)
        for sequence in (1, 2, 3):
            admission = ws.admit_broadcast_batch_sync(
                _canonical_tracking_batch(
                    ws,
                    sequence=sequence,
                    with_world=False,
                )
            )
            admission.commit_then_release(lambda: None)
        await ws.quiesce_outbound_submissions(timeout_s=1.0)
        return (
            [json.loads(str(item)) for item in client.sent],
            ws._should_coalesce_json_message(
                {"type": "tracking", "source_id": 0}
            ),
        )

    sent, is_coalesced = asyncio.run(_run())
    assert [item["tracking_publication_sequence"] for item in sent] == [1, 2, 3]
    assert all(item["type"] == "tracking" for item in sent)
    assert is_coalesced is False


def test_world_snapshot_and_world_event_are_lossless_and_non_coalesced() -> None:
    async def _run() -> tuple[list[dict], bool, bool]:
        ws = WebSocketServer(stats_callback=None)
        ws.event_loop = asyncio.get_running_loop()
        ws._json_coalesce_interval_by_type["world_snapshot"] = 0.01
        ws._json_coalesce_interval_by_type["world_event"] = 0.01
        client = _FakeSocket()
        ws.connected_clients.add(client)
        for sequence in (1, 2, 3):
            admission = ws.admit_broadcast_batch_sync(
                _canonical_tracking_batch(
                    ws,
                    sequence=sequence,
                )
            )
            admission.commit_then_release(lambda: None)
        snapshot_is_coalesced = ws._should_coalesce_json_message(
            {"type": "world_snapshot", "payload": {"sequence": 2}}
        )
        event_is_coalesced = ws._should_coalesce_json_message(
            {"type": "world_event", "payload": {"sequence": 2}}
        )
        await ws.quiesce_outbound_submissions(timeout_s=1.0)
        return (
            [json.loads(str(item)) for item in client.sent],
            snapshot_is_coalesced,
            event_is_coalesced,
        )

    sent, snapshot_is_coalesced, event_is_coalesced = asyncio.run(_run())
    snapshots = [item for item in sent if item["type"] == "world_snapshot"]
    events = [item for item in sent if item["type"] == "world_event"]
    assert [item["payload"]["sequence"] for item in snapshots] == [1, 2, 3]
    assert [item["payload"]["sequence"] for item in events] == [1, 2, 3]
    assert snapshot_is_coalesced is False
    assert event_is_coalesced is False


def test_stats_snapshot_worker_does_not_block_event_loop_and_joins() -> None:
    entered = threading.Event()
    release = threading.Event()

    def _slow_stats() -> dict:
        entered.set()
        release.wait(timeout=2.0)
        return {"sequence": 1}

    async def _run() -> WebSocketServer:
        ws = WebSocketServer(stats_callback=_slow_stats)
        snapshot_task = asyncio.create_task(ws._get_stats_snapshot())
        assert await asyncio.to_thread(entered.wait, 1.0)
        client = _FakeSocket()
        ws.connected_clients.add(client)
        started_ns = time.perf_counter_ns()
        await ws.broadcast(
            _timed_payload(
                ws,
                lambda: {"type": "pong", "timestamp": 1},
            ),
        )
        dispatch_elapsed_ms = (
            time.perf_counter_ns() - started_ns
        ) / 1_000_000.0
        assert dispatch_elapsed_ms < 50.0
        assert client.sent
        release.set()
        assert await snapshot_task == {"sequence": 1}
        receipt = await asyncio.to_thread(
            ws.quiesce_stats_collector,
            timeout_s=1.0,
        )
        assert receipt.quiesced is True
        return ws

    ws = asyncio.run(_run())
    assert ws._stats_futures == set()
    assert not [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith("NoesisWSStats")
    ]


def test_clear_stats_generation_never_republishes_stale_inflight_snapshot() -> None:
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()
    release_second = threading.Event()

    class _Stats:
        calls = 0
        clears = 0

        def __call__(self) -> dict:
            self.calls += 1
            if self.calls == 1:
                first_entered.set()
                release_first.wait(timeout=2.0)
                return {"sequence": 1}
            second_entered.set()
            release_second.wait(timeout=2.0)
            return {"sequence": 2}

        def clear_stats(self) -> None:
            self.clears += 1

    async def _run() -> None:
        callback = _Stats()
        ws = WebSocketServer(stats_callback=callback)
        first = asyncio.create_task(ws._get_stats_snapshot())
        assert await asyncio.to_thread(first_entered.wait, 1.0)
        clear = asyncio.create_task(ws._clear_and_refresh_stats())
        release_first.set()
        assert await first == {"sequence": 1}
        assert await asyncio.to_thread(second_entered.wait, 1.0)
        read_during_clear = asyncio.create_task(ws._get_stats_snapshot())
        await asyncio.sleep(0.02)
        assert not read_during_clear.done()
        release_second.set()
        assert await clear == {"sequence": 2}
        assert await read_during_clear == {"sequence": 2}
        assert callback.clears == 1
        receipt = await asyncio.to_thread(
            ws.quiesce_stats_collector,
            timeout_s=1.0,
        )
        assert receipt.quiesced is True

    asyncio.run(_run())


def test_webrtc_gateway_factory_creates_extra_slot_on_demand() -> None:
    class _Gateway:
        def __init__(self) -> None:
            self.stopped = False

        def stop(self) -> None:
            self.stopped = True

    ws = WebSocketServer(stats_callback=None)
    ws.register_webrtc_gateway(_Gateway())
    created: list[_Gateway] = []

    def _factory() -> _Gateway:
        gateway = _Gateway()
        created.append(gateway)
        return gateway

    ws.register_webrtc_gateway_factory(_factory, max_clients=2, initial_clients=1)
    first_owner = _FakeSocket()
    second_owner = _FakeSocket()
    ws.connected_clients.add(first_owner)
    ws.connected_clients.add(second_owner)
    first = ws._select_gateway_for_client(first_owner)
    assert first is ws.webrtc_gateways[0]
    ws._set_gateway_owner(first, first_owner, "127.0.0.1")

    second = ws._select_gateway_for_client(second_owner)

    assert second is created[0]
    assert len(ws.webrtc_gateways) == 2


def test_webrtc_gateway_factory_stops_unregistered_slot_on_registration_failure(
    monkeypatch,
) -> None:
    class _Gateway:
        def __init__(self) -> None:
            self.stop_calls = 0

        def stop(self) -> None:
            self.stop_calls += 1

    ws = WebSocketServer(stats_callback=None)
    gateway = _Gateway()
    ws.register_webrtc_gateway_factory(lambda: gateway, max_clients=1)

    def _reject(_gateway) -> None:
        raise RuntimeError("registration closed")

    monkeypatch.setattr(ws, "register_webrtc_gateway", _reject)

    assert ws._create_webrtc_gateway() is None
    assert gateway.stop_calls == 1


def test_webrtc_owner_disconnect_resets_warm_peer_before_slot_reuse() -> None:
    class _Gateway:
        def __init__(self) -> None:
            self.reset_reasons: list[str] = []
            self.completion = None

        def reset_peer(self, *, reason: str, on_complete) -> None:
            self.reset_reasons.append(reason)
            self.completion = on_complete

        def stop(self) -> None:
            raise AssertionError("bounded warm gateway must be reset, not stopped")

    ws = WebSocketServer(stats_callback=None)
    gateway = _Gateway()
    ws.register_webrtc_gateway(gateway)
    ws.register_webrtc_gateway_factory(lambda: None, max_clients=1, initial_clients=1)
    owner = _FakeSocket()
    replacement = _FakeSocket()
    ws.connected_clients.add(owner)
    ws.connected_clients.add(replacement)
    ws._set_gateway_owner(gateway, owner, "127.0.0.1")

    ws.connected_clients.discard(owner)
    ws._clear_gateway_owner_for_client(owner)

    assert gateway.reset_reasons == ["owner_disconnected"]
    assert gateway in ws.webrtc_gateways
    assert ws._select_gateway_for_client(replacement) is None
    assert callable(gateway.completion)
    gateway.completion(True)
    assert ws._select_gateway_for_client(replacement) is gateway


def test_failed_webrtc_peer_reset_removes_unsafe_slot() -> None:
    class _Gateway:
        def __init__(self) -> None:
            self.stopped = False

        def reset_peer(self, *, reason: str, on_complete) -> None:
            assert reason == "owner_disconnected"
            on_complete(False)

        def stop(self) -> None:
            self.stopped = True

    ws = WebSocketServer(stats_callback=None)
    gateway = _Gateway()
    ws.register_webrtc_gateway(gateway)
    owner = _FakeSocket()
    ws.connected_clients.add(owner)
    ws._set_gateway_owner(gateway, owner, "127.0.0.1")
    ws.connected_clients.discard(owner)

    ws._clear_gateway_owner_for_client(owner)

    assert gateway not in ws.webrtc_gateways
    assert ws.webrtc_gateway is None
