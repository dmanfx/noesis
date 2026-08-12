from __future__ import annotations

import importlib
import logging
import re
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNTIMES = (
    ROOT / "noesis" / "ds8_runtime.py",
    ROOT / "noesis" / "ds8_runtime_v3dt_reimpl.py",
    ROOT / "DS9" / "noesis" / "ds9_runtime_core.py",
)


@pytest.mark.parametrize(
    "module_name",
    (
        "noesis.ds8_runtime",
        "noesis.ds8_runtime_v3dt_reimpl",
    ),
)
def test_runtime_rejects_unknown_tracking_mode(module_name: str) -> None:
    runtime = importlib.import_module(module_name)

    with pytest.raises(SystemExit, match="Unsupported .* tracking mode"):
        runtime._normalize_tracking_mode("v3dtt")


@pytest.mark.parametrize("runtime_path", RUNTIMES, ids=lambda path: path.name)
def test_runtime_closes_callback_resources_only_after_pipeline_quiescence(
    runtime_path: Path,
) -> None:
    source = runtime_path.read_text(encoding="utf-8")
    shutdown = source[source.index('logger.info("Shutting down') :]

    ordered_markers = (
        "_arm_shutdown_watchdog()",
        "rest_shutdown_receipt = _stop_rest_server(",
        "if not rest_shutdown_receipt.quiesced:",
        "stats_shutdown_receipt = ws_server.quiesce_stats_collector(",
        "provider_shutdown_receipt = ws_server.quiesce_blocking_providers(",
        "detached_gateways = ws_server.begin_webrtc_shutdown(",
        "gateway_executor.submit(gateway.stop)",
        "_stop_websocket_server(ws_server, ws_thread, ws_loop)",
        "pipeline.cancel_control_timers()",
        "pipeline.mark_depth_enabled(False)",
        "request_orderly_eos(pipeline)",
        "wait_thread.join(",
        "if not pipeline_quiesced:",
        "shutdown_mapanything(wait=True, timeout_s=5.0)",
        "mgr.save_gallery()",
        "identity_v2_service.close()",
        "world_service.close()",
        "storage_shutdown_receipt = close_depth_storage(",
        "diagnostics_logger.close()",
        "signal.alarm(0)",
    )
    positions = [shutdown.index(marker) for marker in ordered_markers]
    assert positions == sorted(positions)
    rest_failure = shutdown[
        shutdown.index("if not rest_shutdown_receipt.quiesced:") : shutdown.index(
            "    gateways_to_stop:"
        )
    ]
    assert 'runtime_state["pipeline_failed"] = True' in rest_failure
    assert "preserving callback-owned " in rest_failure
    assert "native resources until watchdog exit" in rest_failure
    assert "while True:" in rest_failure
    assert "signal.pause()" in rest_failure
    assert "preserving callback-owned resources until watchdog exit" in shutdown
    assert 'runtime_state["ws_provider_shutdown_receipt"]' in shutdown
    assert 'runtime_state["websocket_shutdown_quiesced"] = True' in shutdown
    assert 'runtime_state["mapanything_shutdown_receipt"]' in shutdown
    assert "MapAnything worker ownership remains unresolved" in shutdown
    assert "ds.stop()" not in shutdown
    assert "signal.pause()" in shutdown


@pytest.mark.parametrize("runtime_path", RUNTIMES, ids=lambda path: path.name)
def test_runtime_shutdown_watchdog_is_independent_of_shutdown_event(
    runtime_path: Path,
) -> None:
    source = runtime_path.read_text(encoding="utf-8")
    assert "shutdown_watchdog_armed = False" in source
    assert "if shutdown_watchdog_armed:" in source
    assert "if signum == signal.SIGTERM:" in source
    signal_handler = source[
        source.index("    def _signal_handler") : source.index(
            "    # Some DS/GStreamer backends manipulate signal masks"
        )
    ]
    assert "and not shutdown_event.is_set()" not in signal_handler
    assert 'runtime_state["expected_eos"] = True' in source


@pytest.mark.parametrize("runtime_path", RUNTIMES, ids=lambda path: path.name)
def test_runtime_watchdog_cannot_be_configured_below_proven_phase_budget(
    runtime_path: Path,
) -> None:
    source = runtime_path.read_text(encoding="utf-8")
    assert "SHUTDOWN_WATCHDOG_DEFAULT_S = 75" in source
    assert "SHUTDOWN_EXTERNAL_SUPERVISOR_TIMEOUT_S = 90" in source
    start = source.index("    def _arm_shutdown_watchdog()")
    end = source.index("    def _signal_handler", start)
    watchdog = source[start:end]
    assert "max(\n                SHUTDOWN_WATCHDOG_DEFAULT_S," in watchdog
    assert "grace_s = SHUTDOWN_WATCHDOG_DEFAULT_S" in watchdog
    assert "grace_s = 30" not in watchdog
    assert "max(\n                5," not in watchdog


@pytest.mark.parametrize(
    "runtime_path",
    RUNTIMES,
    ids=lambda path: path.name,
)
def test_runtime_treats_eos_during_expected_shutdown_as_nonfatal(
    runtime_path: Path,
) -> None:
    source = runtime_path.read_text(encoding="utf-8")
    assert 'state["pipeline_eos_seen"] = True' in source
    assert 'logger.info("EOS received on pipeline (reason=%s)", eos_reason)' in source
    assert 'state["pipeline_eos_reason"] = eos_reason' in source


@pytest.mark.parametrize("runtime_path", RUNTIMES, ids=lambda path: path.name)
def test_runtime_never_treats_wait_failure_as_pipeline_quiescence(
    runtime_path: Path,
) -> None:
    source = runtime_path.read_text(encoding="utf-8")

    assert 'state["wait_failed"] = True' in source
    assert 'state["pipeline_failed"] = True' in source
    assert 'wait_failed = bool(runtime_state.get("wait_failed"))' in source
    assert "and not wait_failed" in source
    assert "wait_failed=%s" in source


def test_wait_loop_exception_sets_explicit_failure_state() -> None:
    from noesis import ds8_runtime

    class _FailingPipeline:
        @staticmethod
        def wait() -> None:
            raise RuntimeError("synthetic wait failure")

    state: dict[str, object] = {}
    shutdown_event = threading.Event()
    thread = ds8_runtime._start_pyservicemaker_wait_loop(
        _FailingPipeline(),
        shutdown_event,
        logging.getLogger("test.runtime-shutdown"),
        state,
    )

    assert thread is not None
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert shutdown_event.is_set()
    assert state["wait_failed"] is True
    assert state["pipeline_failed"] is True


def test_websocket_startup_timeout_cancels_and_joins_owner_thread() -> None:
    from noesis import ds8_runtime
    from websocket_server import WebSocketStartupError

    class _BlockedStartupServer:
        def __init__(self) -> None:
            self.server = None
            self.event_loop = None
            self._shutdown_quiesced = False
            self.start_entered = threading.Event()
            self.stop_calls = 0
            self.lifecycle_failures: list[BaseException] = []

        async def start(self) -> None:
            self.start_entered.set()
            await __import__("asyncio").Event().wait()

        async def stop(self) -> None:
            self.stop_calls += 1
            self._shutdown_quiesced = True

        def report_lifecycle_failure(self, error: BaseException) -> None:
            self.lifecycle_failures.append(error)

    server = _BlockedStartupServer()
    with pytest.raises(WebSocketStartupError) as raised:
        ds8_runtime._start_websocket_server(server, timeout_s=0.05)

    receipt = raised.value.receipt
    assert server.start_entered.is_set()
    assert server.stop_calls == 1
    assert server.lifecycle_failures
    assert receipt.startup_signaled is True
    assert receipt.server_bound is False
    assert receipt.start_task_done is True
    assert receipt.thread_stopped is True
    assert receipt.event_loop_closed is True
    assert receipt.quiesced is True
    assert not any(
        thread.name == "DS8-WebSocket" and thread.is_alive()
        for thread in threading.enumerate()
    )


@pytest.mark.parametrize("runtime_path", RUNTIMES, ids=lambda path: path.name)
def test_websocket_startup_uses_bounded_task_and_cleanup_receipt(
    runtime_path: Path,
) -> None:
    source = runtime_path.read_text(encoding="utf-8")
    start = source.index("def _start_websocket_server(")
    end = source.index("\ndef _stop_websocket_server(", start)
    helper = source[start:end]

    assert "asyncio.wait_for(start_task, timeout=timeout)" in helper
    assert "started.wait(timeout=timeout + 0.5)" in helper
    assert "start_task.cancel" in helper
    assert "thread.join(timeout=WebSocketServer.RUNTIME_SHUTDOWN_TIMEOUT_S)" in helper
    assert "WebSocketStartupReceipt(" in helper
    assert "raise WebSocketStartupError(" in helper


@pytest.mark.parametrize("runtime_path", RUNTIMES, ids=lambda path: path.name)
def test_post_bind_startup_failures_follow_startup_ownership_transaction(
    runtime_path: Path,
) -> None:
    source = runtime_path.read_text(encoding="utf-8")
    start = source.index("        ws_thread, ws_loop = _start_websocket_server(")
    end = source.index("    wait_thread = _start_pyservicemaker_wait_loop(", start)
    startup = source[start:end]

    assert "except WebSocketStartupError as exc:" in startup
    assert "if not exc.receipt.quiesced:" in startup
    acquisition = startup.index('"websocket_listener",')
    listener_cleanup = startup.index(
        "_stop_websocket_server(ws_server, owner[0], owner[1])",
        acquisition,
    )
    listener_missing = startup.index("WebSocket server failed to bind required endpoint")
    listener_abort = startup.index(
        'return _abort_startup("websocket_listener_missing")',
        listener_missing,
    )
    assert acquisition < listener_cleanup < listener_missing < listener_abort

    registration = acquisition
    stdin_failure = startup.index(
        "Failed to establish Service Maker stdin lifecycle contract"
    )
    stdin_abort = startup.index(
        'return _abort_startup("servicemaker_stdin_contract_failed")',
        stdin_failure,
    )
    assert registration < stdin_failure < stdin_abort
    assert "ingress=True" in startup[registration:stdin_failure]
    assert 'return _abort_startup("shutdown_requested_before_activation")' in startup

    attempted = startup.index("startup_transaction.mark_activation_attempted()")
    activation = startup.index("ds8_pipeline.activate()")
    activation_failure = startup.index("pipeline activation failed")
    activation_abort = startup.index(
        '_abort_ambiguous_and_wait("pipeline_activation_failed")',
        activation_failure,
    )
    missing_owner = startup.index("pipeline activated but ds_pipeline is missing")
    missing_abort = startup.index(
        '_abort_ambiguous_and_wait("activated_pipeline_owner_missing")',
        missing_owner,
    )
    assert attempted < activation < activation_failure < activation_abort
    assert activation_abort < missing_owner < missing_abort


class _FakeRestServer:
    def __init__(self) -> None:
        self.should_exit = False


def test_rest_shutdown_receipt_proves_server_and_analytics_quiescence() -> None:
    from noesis import ds8_runtime

    server = _FakeRestServer()
    analytics_lock = threading.Lock()

    def _serve_until_stopped() -> None:
        while not server.should_exit:
            time.sleep(0.001)

    thread = threading.Thread(target=_serve_until_stopped, name="test-rest-server")
    thread.start()
    receipt = ds8_runtime._stop_rest_server(
        server,
        thread,
        analytics_lock,
        timeout=1.0,
    )

    assert receipt.rest_pair_consistent is True
    assert receipt.stop_requested is True
    assert receipt.server_thread_stopped is True
    assert receipt.analytics_transaction_lock_retained is True
    assert receipt.quiesced is True
    assert not thread.is_alive()
    assert analytics_lock.locked()
    analytics_lock.release()


def test_rest_shutdown_receipt_rejects_blocked_server_thread() -> None:
    from noesis import ds8_runtime

    server = _FakeRestServer()
    release_server = threading.Event()
    thread = threading.Thread(
        target=release_server.wait,
        name="test-blocked-rest-server",
    )
    thread.start()
    analytics_lock = threading.Lock()
    try:
        receipt = ds8_runtime._stop_rest_server(
            server,
            thread,
            analytics_lock,
            timeout=0.02,
        )

        assert receipt.stop_requested is True
        assert receipt.server_thread_stopped is False
        assert receipt.analytics_transaction_lock_retained is False
        assert receipt.quiesced is False
        assert not analytics_lock.locked()
    finally:
        release_server.set()
        thread.join(timeout=1.0)


def test_rest_shutdown_receipt_rejects_blocked_roi_reload_transaction() -> None:
    from noesis import ds8_runtime

    analytics_lock = threading.Lock()
    transaction_entered = threading.Event()
    release_transaction = threading.Event()

    def _blocked_roi_reload() -> None:
        with analytics_lock:
            transaction_entered.set()
            release_transaction.wait()

    worker = threading.Thread(
        target=_blocked_roi_reload,
        name="test-blocked-roi-reload",
    )
    worker.start()
    assert transaction_entered.wait(timeout=1.0)
    try:
        receipt = ds8_runtime._stop_rest_server(
            None,
            None,
            analytics_lock,
            timeout=0.02,
        )

        assert receipt.rest_pair_consistent is True
        assert receipt.server_thread_stopped is True
        assert receipt.analytics_transaction_lock_retained is False
        assert receipt.quiesced is False
    finally:
        release_transaction.set()
        worker.join(timeout=1.0)


def test_real_rest_server_does_not_abandon_blocked_sync_handler() -> None:
    from fastapi import FastAPI
    from noesis import ds8_runtime

    app = FastAPI()
    app.state.noesis_internal_auth = {"mode": "disabled"}
    handler_entered = threading.Event()
    release_handler = threading.Event()

    @app.get("/blocked")
    def _blocked_handler() -> dict[str, bool]:
        handler_entered.set()
        release_handler.wait(timeout=2.0)
        return {"ok": True}

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])

    server, server_thread = ds8_runtime._start_rest_server(
        app,
        "127.0.0.1",
        port,
    )
    request_thread = threading.Thread(
        target=lambda: urlopen(  # noqa: S310 - loopback test endpoint
            f"http://127.0.0.1:{port}/blocked",
            timeout=2.0,
        ).read()
    )
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.05):
                    break
            except OSError:
                time.sleep(0.01)
        request_thread.start()
        assert handler_entered.wait(timeout=1.0)

        receipt = ds8_runtime._stop_rest_server(
            server,
            server_thread,
            threading.Lock(),
            timeout=0.02,
        )
        assert receipt.stop_requested is True
        assert receipt.server_thread_stopped is False
        assert receipt.quiesced is False
        assert server_thread is not None and server_thread.is_alive()
    finally:
        release_handler.set()
        if request_thread.ident is not None:
            request_thread.join(timeout=2.0)
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=2.0)
    assert server_thread is not None and not server_thread.is_alive()


@pytest.mark.parametrize(
    "module_name",
    (
        "noesis.ds8_runtime",
        "noesis.ds8_runtime_v3dt_reimpl",
    ),
)
def test_rest_startup_timeout_retains_cleanly_stopped_owner(
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi import FastAPI
    import uvicorn

    runtime = importlib.import_module(module_name)

    class _Server:
        def __init__(self, config: object) -> None:
            self.config = config
            self.started = False
            self.should_exit = False

        def run(self) -> None:
            while not self.should_exit:
                time.sleep(0.001)

    monkeypatch.setattr(uvicorn, "Server", _Server)
    app = FastAPI()
    app.state.noesis_internal_auth = {"mode": "disabled"}
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])

    with pytest.raises(runtime.RestStartupError) as exc_info:
        runtime._start_rest_server(
            app,
            "127.0.0.1",
            port,
            startup_timeout_s=0.01,
            cleanup_timeout_s=0.2,
        )

    assert exc_info.value.cleanup_proven is True
    assert exc_info.value.server.should_exit is True
    assert not exc_info.value.thread.is_alive()


@pytest.mark.parametrize(
    "module_name",
    (
        "noesis.ds8_runtime",
        "noesis.ds8_runtime_v3dt_reimpl",
    ),
)
def test_rest_startup_timeout_retains_unresolved_live_owner(
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi import FastAPI
    import uvicorn

    runtime = importlib.import_module(module_name)
    release = threading.Event()

    class _Server:
        def __init__(self, config: object) -> None:
            self.config = config
            self.started = False
            self.should_exit = False

        def run(self) -> None:
            release.wait()

    monkeypatch.setattr(uvicorn, "Server", _Server)
    app = FastAPI()
    app.state.noesis_internal_auth = {"mode": "disabled"}
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])

    try:
        with pytest.raises(runtime.RestStartupError) as exc_info:
            runtime._start_rest_server(
                app,
                "127.0.0.1",
                port,
                startup_timeout_s=0.01,
                cleanup_timeout_s=0.01,
            )

        assert exc_info.value.cleanup_proven is False
        assert exc_info.value.server.should_exit is True
        assert exc_info.value.thread.is_alive()
    finally:
        release.set()
        if "exc_info" in locals():
            exc_info.value.thread.join(timeout=1.0)


def _rest_test_app() -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(noesis_internal_auth={"mode": "disabled"})
    )


def test_rest_startup_early_exit_retains_owned_thread_and_proves_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uvicorn

    from noesis import ds8_runtime

    class _Server:
        started = False
        should_exit = False

        @staticmethod
        def run() -> None:
            return None

    server = _Server()
    monkeypatch.setattr(ds8_runtime, "_port_bindable", lambda *_args: True)
    monkeypatch.setattr(uvicorn, "Config", lambda **_kwargs: object())
    monkeypatch.setattr(uvicorn, "Server", lambda **_kwargs: server)

    with pytest.raises(ds8_runtime.RestStartupError) as raised:
        ds8_runtime._start_rest_server(_rest_test_app(), "127.0.0.1", 18080)

    assert raised.value.server is server
    assert raised.value.thread.name == "DS8-REST"
    assert raised.value.cleanup_proven is True
    assert not raised.value.thread.is_alive()


def test_rest_startup_timeout_cancels_and_joins_owned_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uvicorn

    from noesis import ds8_runtime

    class _Server:
        started = False
        should_exit = False

        def run(self) -> None:
            while not self.should_exit:
                threading.Event().wait(0.001)

    ticks = iter((0.0, 11.0))
    server = _Server()
    monkeypatch.setattr(ds8_runtime, "_port_bindable", lambda *_args: True)
    monkeypatch.setattr(ds8_runtime.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(uvicorn, "Config", lambda **_kwargs: object())
    monkeypatch.setattr(uvicorn, "Server", lambda **_kwargs: server)

    with pytest.raises(ds8_runtime.RestStartupError) as raised:
        ds8_runtime._start_rest_server(_rest_test_app(), "127.0.0.1", 18080)

    assert server.should_exit is True
    assert raised.value.cleanup_proven is True
    assert not raised.value.thread.is_alive()


def test_rest_startup_timeout_never_loses_unjoined_thread_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uvicorn

    from noesis import ds8_runtime

    release = threading.Event()

    class _Server:
        started = False
        should_exit = False

        @staticmethod
        def run() -> None:
            release.wait()

    ticks = iter((0.0, 11.0))
    server = _Server()
    monkeypatch.setattr(ds8_runtime, "_port_bindable", lambda *_args: True)
    monkeypatch.setattr(ds8_runtime.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(uvicorn, "Config", lambda **_kwargs: object())
    monkeypatch.setattr(uvicorn, "Server", lambda **_kwargs: server)
    try:
        with pytest.raises(ds8_runtime.RestStartupError) as raised:
            ds8_runtime._start_rest_server(
                _rest_test_app(),
                "127.0.0.1",
                18080,
            )

        assert raised.value.server is server
        assert raised.value.thread.is_alive()
        assert raised.value.cleanup_proven is False
        assert "cleanup was not proven" in str(raised.value)
    finally:
        release.set()
        if "raised" in locals():
            raised.value.thread.join(timeout=1.0)


@pytest.mark.parametrize("runtime_path", RUNTIMES, ids=lambda path: path.name)
def test_runtime_rest_partial_startup_is_watchdog_fatal(runtime_path: Path) -> None:
    source = runtime_path.read_text(encoding="utf-8")

    assert "class RestStartupError(RuntimeError):" in source
    assert "server: object" in source
    assert "thread: threading.Thread" in source
    assert "cleanup_proven = bool(not thread_started or not thread.is_alive())" in source
    start = source.index("        except RestStartupError as exc:")
    end = source.index("        except Exception:", start)
    handler = source[start:end]
    assert 'runtime_state["rest_startup_receipt"]' in handler
    assert "if not exc.cleanup_proven:" in handler
    assert "_arm_shutdown_watchdog()" in handler
    assert "while True:" in handler
    assert "signal.pause()" in handler


def test_websocket_shutdown_rejects_missing_event_loop() -> None:
    from noesis import ds8_runtime

    with pytest.raises(RuntimeError, match="event loop was unavailable"):
        ds8_runtime._stop_websocket_server(  # type: ignore[attr-defined]
            object(),
            threading.current_thread(),
            None,
            timeout=0.01,
        )


def test_websocket_shutdown_rejects_live_worker_after_listener_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from noesis import ds8_runtime

    class _Future:
        @staticmethod
        def result(*, timeout: float) -> None:
            assert 0.0 < timeout <= 0.01

    class _Loop:
        stopped = False

        @staticmethod
        def is_running() -> bool:
            return True

        def stop(self) -> None:
            self.stopped = True

        @staticmethod
        def call_soon_threadsafe(callback) -> None:
            callback()

    class _Thread:
        joined = False

        def join(self, *, timeout: float) -> None:
            assert 0.0 <= timeout <= 0.01
            self.joined = True

        @staticmethod
        def is_alive() -> bool:
            return True

    class _Server:
        async def stop(self) -> None:
            return None

    def _schedule(coroutine, _loop):
        coroutine.close()
        return _Future()

    monkeypatch.setattr(ds8_runtime.asyncio, "run_coroutine_threadsafe", _schedule)
    loop = _Loop()
    thread = _Thread()
    with pytest.raises(RuntimeError, match="thread remained alive"):
        ds8_runtime._stop_websocket_server(  # type: ignore[attr-defined]
            _Server(),
            thread,
            loop,
            timeout=0.01,
        )
    assert loop.stopped is True
    assert thread.joined is True


@pytest.mark.parametrize("runtime_path", RUNTIMES, ids=lambda path: path.name)
def test_runtime_records_required_rest_shutdown_receipt(runtime_path: Path) -> None:
    source = runtime_path.read_text(encoding="utf-8")

    assert "class RestShutdownReceipt(NamedTuple):" in source
    assert "analytics_transaction_lock.acquire(" in source
    assert "retaining the" in source
    assert "analytics lock" in source
    assert 'runtime_state["rest_shutdown_receipt"]' in source
    assert '"quiesced": rest_shutdown_receipt.quiesced' in source
    assert "if not rest_shutdown_receipt.quiesced:" in source
    assert "timeout_graceful_shutdown=None" in source


@pytest.mark.parametrize("runtime_path", RUNTIMES, ids=lambda path: path.name)
def test_runtime_providers_cooperate_with_shutdown_before_depth_enable(
    runtime_path: Path,
) -> None:
    source = runtime_path.read_text(encoding="utf-8")
    provider_region = source[
        source.index("    def _ds8_auto_calibrate_handler") : source.index(
            "    pipeline = ds8_pipeline.build_pipeline"
        )
    ]

    assert provider_region.count("if shutdown_event.is_set():") >= 2
    assert '"error": "shutting_down"' in provider_region
    assert "CaptureEventRuntimeProviders(" in source
    assert "shutdown_requested=shutdown_event.is_set" in source
    assert "burst_waiter=shutdown_event.wait" in source


@pytest.mark.parametrize("runtime_path", RUNTIMES, ids=lambda path: path.name)
def test_ma_depth_cache_only_contract_never_opens_inference_gate(
    runtime_path: Path,
) -> None:
    source = runtime_path.read_text(encoding="utf-8")
    start = source.index("    def _ds8_ma_depth_provider(")
    end = source.index("    def _ds8_floorplan_provider(", start)
    provider = source[start:end]

    assert "cache_only: bool = False" in provider
    assert '"cache_only": bool(cache_only)' in provider
    assert "return providers.depth_provider(" in provider
    assert "cache_only=cache_only" in provider
    assert "enable_depth(" not in provider


def test_shared_ma_depth_cache_only_contract_precedes_capture_admission() -> None:
    source = (ROOT / "noesis" / "capture_event_runtime.py").read_text(
        encoding="utf-8"
    )
    start = source.index("    def depth_provider(")
    end = source.index("    def _record_floorplan(", start)
    provider = source[start:end]

    cache_branch = provider.index("        if cache_only:")
    branch_check = provider.index("        if not self.depth_branch_available():")
    controller = provider.index("            controller = self._controller()")
    assert cache_branch < branch_check < controller
    assert 'error="no_cached_depth"' in provider


@pytest.mark.parametrize("runtime_path", RUNTIMES, ids=lambda path: path.name)
def test_boundary_metrics_getter_failures_are_not_replaced_with_empty_success(
    runtime_path: Path,
) -> None:
    source = runtime_path.read_text(encoding="utf-8")
    start = source.index("def _build_stats_callback(")
    end = source.index("\ndef _start_websocket_server(", start)
    stats = source[start:end]

    assert "ws_boundary_metrics = ws_metrics_getter()" in stats
    assert "rest_boundary_metrics = compact_getter()" in stats
    assert "WebSocket boundary metrics getter must return a dict" in stats
    assert "REST boundary metrics getter must return a dict" in stats
    assert "ws_boundary_metrics = ws_metrics_getter() or {}" not in stats
    assert "rest_boundary_metrics = compact_getter() or {}" not in stats


@pytest.mark.parametrize("runtime_path", RUNTIMES, ids=lambda path: path.name)
def test_runtime_map_worker_shutdown_is_after_probe_quiescence_before_storage(
    runtime_path: Path,
) -> None:
    source = runtime_path.read_text(encoding="utf-8")
    shutdown = source[source.index('logger.info("Shutting down') :]

    assert shutdown.index("if not pipeline_quiesced:") < shutdown.index(
        "shutdown_mapanything(wait=True, timeout_s=5.0)"
    )
    assert shutdown.index(
        "shutdown_mapanything(wait=True, timeout_s=5.0)"
    ) < shutdown.index("storage_shutdown_receipt = close_depth_storage(")


def test_ds8_v3dt_and_ds9_share_the_same_rest_shutdown_gate() -> None:
    gates: list[str] = []
    for runtime_path in RUNTIMES:
        source = runtime_path.read_text(encoding="utf-8")
        start = source.index("    rest_shutdown_receipt = _stop_rest_server(")
        end = source.index("    gateways_to_stop:", start)
        gates.append(source[start:end])

    assert len(set(gates)) == 1


@pytest.mark.parametrize("runtime_path", RUNTIMES, ids=lambda path: path.name)
def test_runtime_has_no_successful_interpreter_exit_bypass(
    runtime_path: Path,
) -> None:
    source = runtime_path.read_text(encoding="utf-8")
    hard_exit_codes = re.findall(r"os\._exit\((\d+)\)", source)

    # The only hard exit is the non-zero, fail-closed SIGALRM watchdog used
    # after quiescence has already failed. It is not a successful teardown path.
    assert hard_exit_codes == ["2"]
    assert "NOESIS_DS9_BYPASS_NATIVE_GC_ON_EXIT" not in source
    assert "if not pipeline_quiesced:" in source
    assert "waiting for shutdown watchdog" in source


@pytest.mark.parametrize("runtime_path", RUNTIMES, ids=lambda path: path.name)
def test_runtime_assembly_is_guarded_until_native_wait_handoff(
    runtime_path: Path,
) -> None:
    source = runtime_path.read_text(encoding="utf-8")

    assert "def _run_main(startup_main_guard: StartupMainGuard) -> int:" in source
    armed = source.index(
        "startup_main_guard.arm(_handle_unexpected_startup_exception)"
    )
    first_owned_binding = source.index(
        '"analytics_runtime_hooks",',
        armed,
    )
    wait_start = source.index(
        "wait_thread = _start_pyservicemaker_wait_loop(",
        first_owned_binding,
    )
    handoff = source.index("startup_transaction.handoff_to_runtime()", wait_start)
    disarmed = source.index("startup_main_guard.disarm()", handoff)
    assert armed < first_owned_binding < wait_start < handoff < disarmed

    assembly = source[first_owned_binding:handoff]
    assert re.search(r"return\s+(?:1|78)\b", assembly) is None
    assert "except StartupOwnershipAmbiguous as exc:" in source[wait_start:handoff]
    assert "_preserve_ambiguous_and_wait(" in source[wait_start:handoff]


@pytest.mark.parametrize("runtime_path", RUNTIMES, ids=lambda path: path.name)
def test_runtime_uses_atomic_startup_ownership_helpers(runtime_path: Path) -> None:
    source = runtime_path.read_text(encoding="utf-8")

    assert "startup_transaction.register(" not in source
    assert 'startup_transaction.bind(\n            "analytics_runtime_hooks"' in source
    assert 'startup_transaction.acquire(\n            "depth_storage"' in source
    assert 'startup_transaction.bind(\n            "reid_api_binding"' in source
    assert 'startup_transaction.acquire(\n        "canonical_world"' in source
    assert 'startup_transaction.bind(\n        "health_api_bindings"' in source
    assert 'startup_transaction.bind(\n        "identity_v2_api_bindings"' in source
    assert 'startup_transaction.bind(\n                "mapanything_processor"' in source
    assert 'startup_transaction.acquire(\n        "websocket_listener"' in source
    assert 'startup_transaction.bind(\n            "servicemaker_stdin_keepalive"' in source


@pytest.mark.parametrize("runtime_path", RUNTIMES, ids=lambda path: path.name)
def test_runtime_rejects_preactivation_shutdown_and_hidden_startup_fallbacks(
    runtime_path: Path,
) -> None:
    source = runtime_path.read_text(encoding="utf-8")

    assert 'return _abort_startup("shutdown_requested_before_prepare")' in source
    assert 'return _abort_startup("shutdown_requested_after_prepare")' in source
    assert source.count(
        'return _abort_startup("shutdown_requested_before_activation")'
    ) >= 2
    assert 'return _abort_startup("repo_root_chdir_failed")' in source
    assert 'return _abort_startup("reid_api_binding_failed")' in source
    assert "ReID API registration skipped" not in source
    assert "defaulting to baseline" not in source


@pytest.mark.parametrize("runtime_path", RUNTIMES, ids=lambda path: path.name)
def test_runtime_finalization_cannot_claim_success_after_any_failure(
    runtime_path: Path,
) -> None:
    source = runtime_path.read_text(encoding="utf-8")
    shutdown = source[source.index('logger.info("Shutting down') :]

    assert "if not mgr.save_gallery():" in shutdown
    assert 'raise RuntimeError("StableID gallery save returned false")' in shutdown
    assert "except BaseException as exc:" in shutdown
    assert 'if not runtime_finalization_failures:' in shutdown
    assert "startup_transaction.mark_quiesced()" in shutdown
    failure_gate = shutdown.index("if runtime_finalization_failures:")
    watchdog_cancel = shutdown.index("signal.alarm(0)", failure_gate)
    complete = shutdown.index('logger.info("Shutdown complete")', watchdog_cancel)
    assert failure_gate < watchdog_cancel < complete
    assert "while True:\n                signal.pause()" in shutdown[
        failure_gate:watchdog_cancel
    ]
    assert "for binding_name, clear_binding in (" in shutdown
    assert '"runtime_finalization_failures"' in shutdown


def test_ds9_rest_prebuild_has_no_post_binding_return_or_rebuild_fallback() -> None:
    source = (ROOT / "DS9" / "noesis" / "ds9_runtime_core.py").read_text(
        encoding="utf-8"
    )

    assert 'return _abort_startup("rest_runtime_preload_failed")' in source
    assert 'return _abort_startup("rest_app_prebuild_failed")' in source
    assert "prebuilt_rest_app if prebuilt_rest_app is not None" not in source
    assert 'raise RuntimeError("required prebuilt DS9 REST app is unavailable")' in source


def test_ds9_launcher_does_not_bypass_verified_native_teardown() -> None:
    source = (ROOT / "DS9" / "noesis" / "ds9_runtime.py").read_text(
        encoding="utf-8"
    )

    assert "os._exit" not in source
    assert "NOESIS_DS9_BYPASS_NATIVE_GC_ON_EXIT" not in source
    assert 'raise SystemExit(int(main()))' in source
