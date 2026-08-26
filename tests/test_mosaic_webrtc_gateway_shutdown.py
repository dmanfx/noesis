from __future__ import annotations

import threading
import time

import pytest

from noesis import mosaic_webrtc_gateway as gateway_module
from noesis.mosaic_webrtc_gateway import MosaicWebRTCGateway
from noesis.server.websocket import WebSocketServer


Gst = gateway_module.Gst


class _WebSocketStub:
    def __init__(self) -> None:
        self.registration_calls = 0
        self.webrtc_errors: list[tuple[str, object]] = []
        self.gateway_failures: list[tuple[object, str]] = []

    def register_webrtc_gateway(self, _gateway: object) -> None:
        self.registration_calls += 1

    def send_webrtc_error(self, error: str, *, gateway: object) -> None:
        self.webrtc_errors.append((error, gateway))

    def report_webrtc_gateway_failure(self, gateway: object, reason: str) -> None:
        self.gateway_failures.append((gateway, reason))


class _NullPipeline:
    def __init__(self) -> None:
        self.set_state_calls: list[object] = []
        self.get_state_calls: list[int] = []

    def set_state(self, state: object) -> object:
        self.set_state_calls.append(state)
        return Gst.StateChangeReturn.ASYNC

    def get_state(self, timeout: int) -> tuple[object, object, object]:
        self.get_state_calls.append(timeout)
        current = (
            self.set_state_calls[-1]
            if self.set_state_calls
            else Gst.State.NULL
        )
        return (
            Gst.StateChangeReturn.SUCCESS,
            current,
            Gst.State.VOID_PENDING,
        )

    @staticmethod
    def get_bus() -> None:
        return None


def _started_gateway() -> MosaicWebRTCGateway:
    gateway = MosaicWebRTCGateway(_WebSocketStub())
    gateway.pipeline = _NullPipeline()  # type: ignore[assignment]
    gateway._started = True
    return gateway


def test_stop_drains_admitted_callback_before_pipeline_null() -> None:
    gateway = _started_gateway()
    pipeline = gateway.pipeline
    assert isinstance(pipeline, _NullPipeline)
    callback_entered = threading.Event()
    release_callback = threading.Event()

    def _callback() -> None:
        assert gateway._begin_lifecycle_callback() is True
        callback_entered.set()
        release_callback.wait(timeout=2.0)
        gateway._end_lifecycle_callback()

    callback_thread = threading.Thread(target=_callback)
    callback_thread.start()
    assert callback_entered.wait(timeout=1.0)

    stop_thread = threading.Thread(target=gateway.stop)
    stop_thread.start()
    time.sleep(0.03)
    assert pipeline.set_state_calls == []
    assert stop_thread.is_alive()

    release_callback.set()
    callback_thread.join(timeout=1.0)
    stop_thread.join(timeout=1.0)
    assert not callback_thread.is_alive()
    assert not stop_thread.is_alive()
    assert pipeline.set_state_calls == [Gst.State.NULL]
    assert pipeline.get_state_calls == [3 * Gst.SECOND]
    assert gateway._started is False


def test_stop_rejects_unproven_pipeline_null_and_retains_ownership() -> None:
    class _StuckPipeline(_NullPipeline):
        def get_state(self, timeout: int) -> tuple[object, object, object]:
            self.get_state_calls.append(timeout)
            return (
                Gst.StateChangeReturn.ASYNC,
                Gst.State.PLAYING,
                Gst.State.NULL,
            )

    gateway = _started_gateway()
    pipeline = _StuckPipeline()
    gateway.pipeline = pipeline  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="failed to enter NULL"):
        gateway.stop()

    assert gateway.pipeline is pipeline
    assert gateway._started is True
    assert gateway._stopping is True


def test_tracked_glib_timeout_cannot_fire_after_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callbacks: list[object] = []
    removed: list[int] = []
    fired: list[str] = []

    def _timeout_add(_interval: int, callback: object) -> int:
        callbacks.append(callback)
        return 77

    monkeypatch.setattr(gateway_module.GLib, "timeout_add", _timeout_add)
    monkeypatch.setattr(gateway_module.GLib, "source_remove", removed.append)
    gateway = _started_gateway()
    assert gateway._schedule_glib_timeout(5, lambda: fired.append("late") or True) == 77

    gateway.stop()

    assert removed == [77]
    assert callbacks
    assert callbacks[0]() is False  # type: ignore[operator]
    assert fired == []


def test_pending_relink_probe_is_removed_and_generation_guarded_on_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Pad:
        def __init__(self) -> None:
            self.callback = None
            self.removed: list[int] = []
            self.link_calls = 0

        def add_probe(self, _kind: object, callback: object, _data: object) -> int:
            self.callback = callback
            return 41

        def remove_probe(self, probe_id: int) -> None:
            self.removed.append(probe_id)

        @staticmethod
        def get_peer() -> None:
            return None

        def link(self, _sink: object) -> object:
            self.link_calls += 1
            return Gst.PadLinkReturn.OK

    class _Queue:
        def __init__(self, pad: _Pad) -> None:
            self.pad = pad

        def get_static_pad(self, _name: str) -> _Pad:
            return self.pad

    pad = _Pad()
    gateway = _started_gateway()
    gateway.queue = _Queue(pad)  # type: ignore[assignment]
    gateway.webrtc = object()  # type: ignore[assignment]
    gateway.drain = object()  # type: ignore[assignment]
    gateway.webrtc_sink_pad = object()  # type: ignore[assignment]
    monkeypatch.setattr(gateway, "_queue_payload_matches_offer", lambda: True)
    monkeypatch.setattr(gateway, "_ensure_webrtc_sender_pad", lambda: None)

    gateway._link_sender_into_webrtc()
    assert gateway._relink_probe_id == 41

    gateway.stop()

    assert pad.removed == [41]
    assert pad.callback is not None
    result = pad.callback(pad, object(), None)
    assert result == Gst.PadProbeReturn.REMOVE
    assert pad.link_calls == 0


def test_relink_probe_failure_never_uses_unsafe_direct_link_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Pad:
        def __init__(self) -> None:
            self.link_calls = 0

        @staticmethod
        def add_probe(_kind: object, _callback: object, _data: object) -> int:
            raise RuntimeError("probe unavailable")

        @staticmethod
        def get_peer() -> None:
            return None

        def link(self, _sink: object) -> object:
            self.link_calls += 1
            return Gst.PadLinkReturn.OK

    class _Queue:
        def __init__(self, pad: _Pad) -> None:
            self.pad = pad

        def get_static_pad(self, _name: str) -> _Pad:
            return self.pad

    pad = _Pad()
    gateway = _started_gateway()
    gateway.queue = _Queue(pad)  # type: ignore[assignment]
    gateway.webrtc = object()  # type: ignore[assignment]
    gateway.drain = object()  # type: ignore[assignment]
    gateway.webrtc_sink_pad = object()  # type: ignore[assignment]
    monkeypatch.setattr(gateway, "_queue_payload_matches_offer", lambda: True)
    monkeypatch.setattr(gateway, "_ensure_webrtc_sender_pad", lambda: None)

    with pytest.raises(RuntimeError, match="required blocking WebRTC relink probe"):
        gateway._link_sender_into_webrtc()

    assert pad.link_calls == 0
    assert gateway._sender_linked is False
    assert gateway._sender_link_in_progress is False


def test_reset_after_stop_cannot_rebuild_or_resurrect_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _started_gateway()
    rebuilt: list[bool] = []
    completions: list[bool] = []
    monkeypatch.setattr(
        gateway,
        "_rebuild_pipeline_for_new_peer",
        lambda: rebuilt.append(True),
    )

    gateway.stop()
    gateway.reset_peer(on_complete=completions.append)

    assert rebuilt == []
    assert completions == [False]


def test_stop_waits_admitted_rebuild_then_nulls_replacement_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _started_gateway()
    old_pipeline = gateway.pipeline
    assert isinstance(old_pipeline, _NullPipeline)
    replacement = _NullPipeline()
    build_entered = threading.Event()
    release_build = threading.Event()

    def _paused_build() -> None:
        build_entered.set()
        release_build.wait(timeout=2.0)
        gateway.pipeline = replacement  # type: ignore[assignment]
        gateway.webrtc = object()  # type: ignore[assignment]

    monkeypatch.setattr(gateway, "build", _paused_build)

    def _leased_rebuild() -> None:
        assert gateway._begin_lifecycle_callback() is True
        try:
            gateway._rebuild_pipeline_for_new_peer()
        finally:
            gateway._end_lifecycle_callback()

    rebuild_thread = threading.Thread(target=_leased_rebuild)
    rebuild_thread.start()
    assert build_entered.wait(timeout=1.0)
    stop_thread = threading.Thread(target=gateway.stop)
    stop_thread.start()
    time.sleep(0.03)
    assert stop_thread.is_alive()
    assert replacement.set_state_calls == []

    release_build.set()
    rebuild_thread.join(timeout=1.0)
    stop_thread.join(timeout=1.0)
    assert not rebuild_thread.is_alive()
    assert not stop_thread.is_alive()
    assert old_pipeline.set_state_calls == [Gst.State.NULL]
    assert replacement.set_state_calls == [Gst.State.PLAYING, Gst.State.NULL]
    assert gateway.pipeline is None
    assert gateway._started is False


def test_slow_peer_drops_incoming_whole_au_at_appsrc_limit() -> None:
    class _AppSrc:
        pushes = 0

        @staticmethod
        def get_property(name: str) -> int:
            assert name == "current-level-buffers"
            return gateway_module.APPSRC_MAX_AUS

        def emit(self, _name: str, _buffer: object) -> object:
            self.pushes += 1
            return Gst.FlowReturn.OK

    gateway = _started_gateway()
    appsrc = _AppSrc()
    gateway.appsrc = appsrc  # type: ignore[assignment]

    accepted = gateway.push_h264_au(Gst.Buffer.new_allocate(None, 16, None))

    assert accepted is False
    assert appsrc.pushes == 0
    assert gateway._push_drop_count == 1


def test_peer_push_retimestamps_au_in_gateway_clock_domain() -> None:
    class _AppSrc:
        pushed: list[object] = []

        @staticmethod
        def get_property(name: str) -> int:
            assert name == "current-level-buffers"
            return 0

        def emit(self, name: str, buffer: object) -> object:
            assert name == "push-buffer"
            self.pushed.append(buffer)
            return Gst.FlowReturn.OK

    gateway = _started_gateway()
    appsrc = _AppSrc()
    gateway.appsrc = appsrc  # type: ignore[assignment]
    buffer = Gst.Buffer.new_allocate(None, 16, None)
    buffer.pts = 1234
    buffer.dts = 1200
    buffer.duration = 33 * Gst.MSECOND

    assert gateway.push_h264_au(buffer) is True
    assert appsrc.pushed == [buffer]
    assert buffer.pts == Gst.CLOCK_TIME_NONE
    assert buffer.dts == Gst.CLOCK_TIME_NONE
    assert buffer.duration == 33 * Gst.MSECOND


def test_gateway_build_has_bounded_au_and_rtp_queues() -> None:
    ws = _WebSocketStub()
    gateway = MosaicWebRTCGateway(ws)
    gateway.build()
    try:
        assert ws.registration_calls == 0
        assert gateway.appsrc is not None
        assert gateway.queue is not None
        assert int(gateway.appsrc.get_property("max-buffers")) == gateway_module.APPSRC_MAX_AUS
        assert int(gateway.appsrc.get_property("leaky-type")) == 1
        assert gateway.appsrc.get_property("block") is False
        assert int(gateway.queue.get_property("leaky")) == 0
        assert int(gateway.queue.get_property("max-size-buffers")) == gateway_module.RTP_QUEUE_MAX_BUFFERS
        assert int(gateway.queue.get_property("max-size-bytes")) == gateway_module.RTP_QUEUE_MAX_BYTES
        assert int(gateway.queue.get_property("max-size-time")) == gateway_module.RTP_QUEUE_MAX_TIME_NS
    finally:
        if gateway.pipeline is not None:
            bus = gateway.pipeline.get_bus()
            if bus is not None:
                bus.remove_signal_watch()
            gateway.pipeline.set_state(Gst.State.NULL)


def test_gateway_reports_terminal_pipeline_failure_only_once() -> None:
    ws = _WebSocketStub()
    gateway = MosaicWebRTCGateway(ws)
    gateway._started = True

    gateway._report_terminal_pipeline_failure("webrtc_gateway_pipeline_error")
    gateway._report_terminal_pipeline_failure("webrtc_gateway_unexpected_eos")

    assert ws.webrtc_errors == [("webrtc_gateway_pipeline_error", gateway)]
    assert ws.gateway_failures == [
        (gateway, "webrtc_gateway_pipeline_error")
    ]


def test_server_terminal_gateway_failure_retires_and_stops_slot_once() -> None:
    class _Gateway:
        def __init__(self) -> None:
            self.stop_calls = 0
            self.stopped = threading.Event()

        def stop(self) -> None:
            self.stop_calls += 1
            self.stopped.set()

    ws = WebSocketServer(stats_callback=None)
    gateway = _Gateway()
    ws.register_webrtc_gateway(gateway)

    ws.report_webrtc_gateway_failure(gateway, "pipeline_error")
    ws.report_webrtc_gateway_failure(gateway, "duplicate_error")

    assert gateway.stopped.wait(timeout=1.0)
    assert gateway.stop_calls == 1
    assert gateway not in ws.webrtc_gateways
    assert ws.webrtc_gateway is None
    assert ws.begin_webrtc_shutdown(timeout_s=1.0) == []


def test_idle_retirement_transfer_is_visible_to_concurrent_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Gateway:
        pass

    ws = WebSocketServer(stats_callback=None)
    ws._webrtc_initial_clients = 1
    first = _Gateway()
    retired = _Gateway()
    ws.register_webrtc_gateway(first)
    ws.register_webrtc_gateway(retired)
    retirement_registered = threading.Event()
    release_retirement = threading.Event()

    def _pause_schedule(
        gateway: object,
        *,
        reason: str,
        ownership_registered: bool = False,
    ) -> None:
        assert gateway is retired
        assert reason == "idle_retirement"
        assert ownership_registered is True
        assert retired in ws._webrtc_retired_gateways
        retirement_registered.set()
        release_retirement.wait(timeout=2.0)

    monkeypatch.setattr(ws, "_schedule_gateway_stop", _pause_schedule)
    retire_thread = threading.Thread(target=ws._retire_extra_idle_gateway, args=(retired,))
    retire_thread.start()
    assert retirement_registered.wait(timeout=1.0)

    detached = ws.begin_webrtc_shutdown(timeout_s=0.2)

    release_retirement.set()
    retire_thread.join(timeout=1.0)
    assert not retire_thread.is_alive()
    assert retired in detached


def test_shutdown_drains_owned_retirement_worker_exactly_once() -> None:
    class _Gateway:
        def __init__(self, *, blocked: bool = False) -> None:
            self.blocked = blocked
            self.stop_entered = threading.Event()
            self.release_stop = threading.Event()
            self.stop_calls = 0

        def stop(self) -> None:
            self.stop_calls += 1
            self.stop_entered.set()
            if self.blocked:
                self.release_stop.wait(timeout=2.0)

    ws = WebSocketServer(stats_callback=None)
    ws._webrtc_initial_clients = 1
    first = _Gateway()
    retired = _Gateway(blocked=True)
    ws.register_webrtc_gateway(first)
    ws.register_webrtc_gateway(retired)
    assert ws._retire_extra_idle_gateway(retired) is True
    assert retired.stop_entered.wait(timeout=1.0)
    result: list[list[object]] = []

    shutdown_thread = threading.Thread(
        target=lambda: result.append(ws.begin_webrtc_shutdown(timeout_s=1.0))
    )
    shutdown_thread.start()
    time.sleep(0.03)
    assert shutdown_thread.is_alive()

    retired.release_stop.set()
    shutdown_thread.join(timeout=1.0)
    assert not shutdown_thread.is_alive()
    assert retired.stop_calls == 1
    assert result == [[first]]
