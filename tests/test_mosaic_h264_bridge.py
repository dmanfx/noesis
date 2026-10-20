from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import pytest

from noesis import mosaic_h264_bridge as bridge_module
from noesis.mosaic_glib_context import SharedDefaultGlibContext
from noesis.mosaic_h264_bridge import MosaicH264ShmFeeder, ensure_parent_dir


Gst = bridge_module.Gst


def test_shared_glib_context_uses_one_reference_counted_thread() -> None:
    driver = SharedDefaultGlibContext()

    driver.acquire()
    first_thread = driver.thread
    driver.acquire()

    assert first_thread is not None
    assert first_thread.is_alive()
    assert driver.thread is first_thread
    assert driver.reference_count == 2

    driver.release()
    assert driver.reference_count == 1
    assert first_thread.is_alive()

    driver.release()
    assert driver.reference_count == 0
    assert driver.thread is None
    assert not first_thread.is_alive()


def test_shm_path_cleanup_refuses_non_socket(tmp_path: Path) -> None:
    path = tmp_path / "mosaic.sock"
    path.write_text("operator data", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Refusing to replace non-socket"):
        ensure_parent_dir(str(path))

    assert path.read_text(encoding="utf-8") == "operator data"


def test_shm_path_cleanup_removes_only_stale_socket(tmp_path: Path) -> None:
    path = tmp_path / "mosaic.sock"
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(path))
    stale.close()

    ensure_parent_dir(str(path))

    assert not path.exists()


def test_shm_path_cleanup_refuses_active_socket(tmp_path: Path) -> None:
    path = tmp_path / "mosaic.sock"
    active = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    active.bind(str(path))
    active.listen(1)
    try:
        with pytest.raises(RuntimeError, match="already active"):
            ensure_parent_dir(str(path))
        assert path.exists()
    finally:
        active.close()
        path.unlink(missing_ok=True)


def test_feeder_fatal_error_is_observable_and_notified_once() -> None:
    failures: list[BaseException] = []
    feeder = MosaicH264ShmFeeder(
        "/tmp/not-started-mosaic.sock",
        on_fatal_error=failures.append,
    )
    feeder._started = True
    first = RuntimeError("shm disconnected")

    feeder._record_fatal_error(first)
    feeder._record_fatal_error(RuntimeError("duplicate"))

    assert feeder.ready is False
    assert feeder.fatal_error is first
    assert failures == [first]


@pytest.mark.parametrize('receipt', [None, lambda:False, lambda:None])
def test_unexpected_eos_remains_fatal_without_exact_producer_receipt(receipt):
    failures=[]
    feeder=MosaicH264ShmFeeder('/tmp/not-started-mosaic.sock',on_fatal_error=failures.append,
                             is_expected_eos=receipt)
    feeder._on_bus_eos(None,None)
    assert len(failures)==1 and feeder.fatal_error is failures[0]


def test_confirmed_finite_eos_does_not_override_an_earlier_transport_failure():
    state={'pipeline_eos_seen':False,'pipeline_eos_reason':None}
    failures=[]
    feeder=MosaicH264ShmFeeder('/tmp/not-started-mosaic.sock',on_fatal_error=failures.append,
        is_expected_eos=lambda:state['pipeline_eos_seen'] is True and state['pipeline_eos_reason']=='finite_source')
    state.update(pipeline_eos_seen=True,pipeline_eos_reason='finite_source')
    feeder._on_bus_eos(None,None)
    assert not failures and feeder.fatal_error is None
    error=RuntimeError('actual transport failure')
    feeder._record_fatal_error(error)
    feeder._on_bus_eos(None,None)
    assert feeder.fatal_error is error and failures==[error]


def test_eos_receipt_exception_fails_closed():
    def unavailable():raise ValueError('no producer receipt')
    feeder=MosaicH264ShmFeeder('/tmp/not-started-mosaic.sock',is_expected_eos=unavailable)
    feeder._on_bus_eos(None,None)
    assert feeder.fatal_error is not None


@pytest.mark.parametrize('confirmed,source,domain,code,expected',[
    (True,'mosaic_h264_shmsrc','gst-resource-error-quark',9,True),
    (False,'mosaic_h264_shmsrc','gst-resource-error-quark',9,False),
    (True,'other_source','gst-resource-error-quark',9,False),
    (True,'mosaic_h264_shmsrc','gst-resource-error-quark',1,False),
    (True,'mosaic_h264_shmsrc','gst-stream-error-quark',1,False),
])
def test_only_confirmed_finite_shm_read_close_is_normal(confirmed,source,domain,code,expected):
    from types import SimpleNamespace
    feeder=MosaicH264ShmFeeder('/tmp/not-started-mosaic.sock',is_expected_eos=lambda:confirmed)
    error=SimpleNamespace(domain=domain,code=code)
    message=SimpleNamespace(src=SimpleNamespace(get_name=lambda:source),parse_error=lambda:(error,'test'))
    feeder._on_bus_error(None,message)
    assert (feeder.fatal_error is None)==expected
    if expected:
        message=SimpleNamespace(src=message.src,
            parse_error=lambda:(SimpleNamespace(domain='gst-stream-error-quark',code=1),'derivative'))
        feeder._on_bus_error(None,message)
        assert feeder.fatal_error is None


def test_feeder_start_proves_first_au_and_fans_out_complete_buffers(
    tmp_path: Path,
) -> None:
    if Gst.ElementFactory.find("x264enc") is None:
        pytest.skip("x264enc is unavailable")
    if Gst.ElementFactory.find("shmsink") is None or Gst.ElementFactory.find("shmsrc") is None:
        pytest.skip("GStreamer SHM elements are unavailable")

    socket_path = tmp_path / "mosaic-h264.sock"
    producer = Gst.parse_launch(
        "videotestsrc is-live=true pattern=ball "
        "! video/x-raw,width=320,height=180,framerate=30/1 "
        "! x264enc tune=zerolatency key-int-max=10 bitrate=1000 "
        "! h264parse config-interval=-1 "
        "! video/x-h264,stream-format=byte-stream,alignment=au "
        f"! shmsink socket-path={socket_path} shm-size=8388608 "
        "wait-for-connection=false sync=false"
    )
    keyframe_reasons: list[str] = []
    received: list[Gst.Buffer] = []

    class _Consumer:
        @staticmethod
        def push_h264_au(buffer: Gst.Buffer) -> bool:
            received.append(buffer)
            return True

    feeder = MosaicH264ShmFeeder(
        str(socket_path),
        request_keyframe=keyframe_reasons.append,
        startup_timeout_s=5.0,
    )
    feeder.register_consumer(_Consumer())
    start_failures: list[BaseException] = []

    def _start_feeder() -> None:
        try:
            feeder.start()
        except BaseException as exc:
            start_failures.append(exc)

    feeder_thread = threading.Thread(target=_start_feeder)
    try:
        feeder_thread.start()
        time.sleep(0.05)
        assert feeder_thread.is_alive()

        assert producer.set_state(Gst.State.PLAYING) != Gst.StateChangeReturn.FAILURE
        state_result, current, _pending = producer.get_state(3 * Gst.SECOND)
        assert state_result != Gst.StateChangeReturn.FAILURE
        assert current == Gst.State.PLAYING

        feeder_thread.join(timeout=6.0)
        assert not feeder_thread.is_alive()
        assert start_failures == []
        deadline = time.monotonic() + 2.0
        while len(received) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)

        assert feeder.ready is True
        assert feeder.fatal_error is None
        assert len(received) >= 3
        assert all(buffer.get_size() > 0 for buffer in received)
        assert keyframe_reasons == ["h264_shm_feeder_start"]
    finally:
        feeder_thread.join(timeout=1.0)
        if feeder.started:
            feeder.stop()
        producer.set_state(Gst.State.NULL)
        producer.get_state(3 * Gst.SECOND)
