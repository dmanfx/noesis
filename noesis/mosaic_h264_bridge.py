"""Shared-memory H.264 access-unit bridge for mosaic WebRTC.

The DS pipeline encodes the mosaic once and publishes byte-stream AUs through
``shmsink``. This module owns the single ``shmsrc`` reader and fans each AU out
to zero or more WebRTC gateways. Gateways then perform the *only* RTP
packetization step (into webrtcbin).

This design deliberately avoids:

- localhost UDP loopback into Service Maker's RTSP server
- RTSP depay → re-pay double packetization
- post-encode leaky queues that drop mid-GOP access units
"""

from __future__ import annotations

import logging
import os
import socket
import stat
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Protocol

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from noesis.mosaic_glib_context import shared_default_glib_context

Gst.init(None)

logger = logging.getLogger(__name__)

DEFAULT_SHM_SOCKET = "/tmp/noesis-mosaic-h264"
DEFAULT_SHM_SIZE_BYTES = 64 * 1024 * 1024
DEFAULT_STARTUP_TIMEOUT_SECONDS = 10.0


class H264AuConsumer(Protocol):
    """Gateway-facing consumer of encoded H.264 access units."""

    def push_h264_au(self, buffer: Gst.Buffer) -> bool:
        """Accept one AU. Return False if the consumer dropped it."""


def resolve_mosaic_h264_shm_socket(config: Optional[dict] = None) -> str:
    """Resolve the shared-memory socket path from env/config with a stable default."""
    env = str(os.environ.get("NOESIS_MOSAIC_H264_SHM", "") or "").strip()
    if env:
        return env
    if isinstance(config, dict):
        cfg = str(config.get("mosaic_h264_shm_socket", "") or "").strip()
        if cfg:
            return cfg
    return DEFAULT_SHM_SOCKET


class MosaicH264ShmFeeder:
    """Single SHM reader that fans H.264 AUs out to registered gateways."""

    def __init__(
        self,
        socket_path: str,
        *,
        request_keyframe: Optional[Callable[[str], None]] = None,
        on_fatal_error: Optional[Callable[[BaseException], None]] = None,
        startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
    ) -> None:
        self.socket_path = str(socket_path or DEFAULT_SHM_SOCKET)
        self._request_keyframe = request_keyframe
        self._on_fatal_error = on_fatal_error
        self._startup_timeout_s = max(1.0, min(30.0, float(startup_timeout_s)))
        self.pipeline: Optional[Gst.Pipeline] = None
        self.appsink: Optional[Gst.Element] = None
        self._bus: Optional[Gst.Bus] = None
        self._consumers: list[H264AuConsumer] = []
        self._consumer_lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._first_au_event = threading.Event()
        self._started = False
        self._stopping = False
        self._context_acquired = False
        self._fatal_error: Optional[BaseException] = None
        self._fatal_error_notified = False
        self._frame_count = 0
        self._drop_count = 0
        self._last_keyframe_request = 0.0

    @property
    def started(self) -> bool:
        with self._state_lock:
            return bool(self._started)

    @property
    def ready(self) -> bool:
        with self._state_lock:
            return bool(
                self._started
                and not self._stopping
                and self._fatal_error is None
                and self._frame_count > 0
            )

    @property
    def fatal_error(self) -> Optional[BaseException]:
        with self._state_lock:
            return self._fatal_error

    def register_consumer(self, consumer: H264AuConsumer) -> None:
        with self._consumer_lock:
            if consumer not in self._consumers:
                self._consumers.append(consumer)

    def unregister_consumer(self, consumer: H264AuConsumer) -> None:
        with self._consumer_lock:
            if consumer in self._consumers:
                self._consumers.remove(consumer)

    def build(self) -> None:
        if self.pipeline is not None:
            return

        pipeline = Gst.Pipeline.new("mosaic-h264-shm-feeder")
        shmsrc = Gst.ElementFactory.make("shmsrc", "mosaic_h264_shmsrc")
        parse = Gst.ElementFactory.make("h264parse", "mosaic_h264_parse")
        capsfilter = Gst.ElementFactory.make("capsfilter", "mosaic_h264_caps")
        appsink = Gst.ElementFactory.make("appsink", "mosaic_h264_appsink")
        if not all((pipeline, shmsrc, parse, capsfilter, appsink)):
            raise RuntimeError("Failed to create mosaic H.264 SHM feeder elements")

        shmsrc.set_property("socket-path", self.socket_path)
        shmsrc.set_property("is-live", True)
        try:
            shmsrc.set_property("do-timestamp", True)
        except Exception:
            pass

        try:
            parse.set_property("config-interval", -1)
        except Exception:
            pass

        capsfilter.set_property(
            "caps",
            Gst.Caps.from_string(
                "video/x-h264,stream-format=byte-stream,alignment=au"
            ),
        )

        # Non-dropping AU queue at the reader edge. Backpressure propagates to
        # the pipeline's pre-encode leaky queue rather than mid-GOP AU drops.
        appsink.set_property("emit-signals", True)
        appsink.set_property("sync", False)
        appsink.set_property("async", False)
        appsink.set_property("max-buffers", 4)
        appsink.set_property("drop", False)
        appsink.set_property(
            "caps",
            Gst.Caps.from_string(
                "video/x-h264,stream-format=byte-stream,alignment=au"
            ),
        )
        appsink.connect("new-sample", self._on_new_sample)

        pipeline.add(shmsrc)
        pipeline.add(parse)
        pipeline.add(capsfilter)
        pipeline.add(appsink)
        if not shmsrc.link(parse):
            raise RuntimeError("Failed to link shmsrc → h264parse")
        if not parse.link(capsfilter):
            raise RuntimeError("Failed to link h264parse → capsfilter")
        if not capsfilter.link(appsink):
            raise RuntimeError("Failed to link capsfilter → appsink")

        self.pipeline = pipeline
        self.appsink = appsink
        bus = pipeline.get_bus()
        if bus is None:
            raise RuntimeError("Mosaic H.264 SHM feeder has no GStreamer bus")
        bus.add_signal_watch()
        bus.connect("message::error", self._on_bus_error)
        bus.connect("message::eos", self._on_bus_eos)
        self._bus = bus
        logger.info(
            "MosaicH264ShmFeeder built: socket_path=%s",
            self.socket_path,
        )

    def start(self) -> None:
        if self.started:
            return
        deadline = time.monotonic() + self._startup_timeout_s
        try:
            self._wait_for_socket(deadline)
            self.build()
            assert self.pipeline is not None
            self._first_au_event.clear()
            with self._state_lock:
                self._stopping = False
                self._fatal_error = None
                self._fatal_error_notified = False
                self._frame_count = 0
                self._drop_count = 0
                self._started = True

            shared_default_glib_context.acquire()
            self._context_acquired = True
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Failed to set mosaic H.264 SHM feeder to PLAYING")
            state_result, current, _pending = self.pipeline.get_state(3 * Gst.SECOND)
            if (
                state_result == Gst.StateChangeReturn.FAILURE
                or current != Gst.State.PLAYING
            ):
                raise RuntimeError(
                    "Mosaic H.264 SHM feeder did not reach PLAYING "
                    f"(result={state_result}, current={current})"
                )

            if self._request_keyframe is not None:
                self._request_keyframe("h264_shm_feeder_start")

            remaining = max(0.0, deadline - time.monotonic())
            if not self._first_au_event.wait(timeout=remaining):
                raise RuntimeError(
                    "Mosaic H.264 SHM feeder received no access unit within "
                    f"{self._startup_timeout_s:.1f}s"
                )
            fatal_error = self.fatal_error
            if fatal_error is not None:
                raise RuntimeError(
                    f"Mosaic H.264 SHM feeder failed during startup: {fatal_error}"
                ) from fatal_error
        except Exception:
            self._cleanup_failed_start()
            raise

        logger.info(
            "MosaicH264ShmFeeder ready: socket_path=%s first_au_received=true",
            self.socket_path,
        )

    def stop(self) -> None:
        if not self._started and self.pipeline is None:
            return
        with self._state_lock:
            self._stopping = True
        pipeline = self.pipeline
        if pipeline is not None:
            result = pipeline.set_state(Gst.State.NULL)
            if result == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("GStreamer rejected mosaic SHM feeder NULL state")
            state_result, current, pending = pipeline.get_state(3 * Gst.SECOND)
            if (
                state_result == Gst.StateChangeReturn.FAILURE
                or current != Gst.State.NULL
                or pending != Gst.State.VOID_PENDING
            ):
                raise RuntimeError(
                    "Mosaic H.264 SHM feeder did not prove terminal NULL state "
                    f"(result={state_result}, current={current}, pending={pending})"
                )
        self._remove_bus_watch()
        if self._context_acquired:
            shared_default_glib_context.release()
            self._context_acquired = False
        self.pipeline = None
        self.appsink = None
        with self._consumer_lock:
            self._consumers.clear()
        with self._state_lock:
            self._started = False
            self._stopping = False
        self._first_au_event.clear()
        logger.info(
            "MosaicH264ShmFeeder stopped frames=%d drops=%d",
            int(self._frame_count),
            int(self._drop_count),
        )

    def _on_new_sample(self, sink: Gst.Element) -> Gst.FlowReturn:
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR
        buf = sample.get_buffer()
        if buf is None:
            return Gst.FlowReturn.OK

        with self._state_lock:
            self._frame_count += 1
            self._first_au_event.set()
        with self._consumer_lock:
            consumers = list(self._consumers)
        if not consumers:
            return Gst.FlowReturn.OK

        any_drop = False
        for consumer in consumers:
            try:
                # Each consumer needs an independent buffer ownership.
                copy = buf.copy()
                ok = bool(consumer.push_h264_au(copy))
                if not ok:
                    any_drop = True
                    self._drop_count += 1
            except Exception:
                any_drop = True
                self._drop_count += 1
                logger.debug("H.264 AU fanout consumer failed", exc_info=True)

        if any_drop:
            self._maybe_request_keyframe("consumer_drop")
        if self._frame_count == 1:
            logger.info(">>> First mosaic H.264 AU received on SHM feeder")
        elif self._frame_count % 300 == 0:
            logger.debug(
                "Mosaic H.264 SHM feeder frames=%d consumers=%d drops=%d",
                int(self._frame_count),
                len(consumers),
                int(self._drop_count),
            )
        return Gst.FlowReturn.OK

    def _wait_for_socket(self, deadline: float) -> None:
        path = Path(self.socket_path)
        while time.monotonic() < deadline:
            try:
                mode = os.lstat(path).st_mode
            except FileNotFoundError:
                time.sleep(0.025)
                continue
            if not stat.S_ISSOCK(mode):
                raise RuntimeError(
                    f"Mosaic H.264 SHM path exists but is not a socket: {path}"
                )
            return
        raise RuntimeError(
            "Mosaic H.264 shmsink did not bind its control socket within "
            f"{self._startup_timeout_s:.1f}s: {path}"
        )

    def _record_fatal_error(self, error: BaseException) -> None:
        callback: Optional[Callable[[BaseException], None]] = None
        with self._state_lock:
            if self._fatal_error is None:
                self._fatal_error = error
            self._first_au_event.set()
            if not self._stopping and not self._fatal_error_notified:
                self._fatal_error_notified = True
                callback = self._on_fatal_error
        if callback is not None:
            try:
                callback(error)
            except Exception:
                logger.exception("Mosaic H.264 SHM fatal-error callback failed")

    def _on_bus_error(self, _bus: Gst.Bus, message: Gst.Message) -> None:
        error, debug = message.parse_error()
        failure = RuntimeError(f"{error}; debug={debug or 'unavailable'}")
        logger.critical("Mosaic H.264 SHM feeder pipeline failed: %s", failure)
        self._record_fatal_error(failure)

    def _on_bus_eos(self, _bus: Gst.Bus, _message: Gst.Message) -> None:
        with self._state_lock:
            stopping = self._stopping
        if not stopping:
            failure = RuntimeError("Mosaic H.264 SHM feeder received unexpected EOS")
            logger.critical("%s", failure)
            self._record_fatal_error(failure)

    def _remove_bus_watch(self) -> None:
        bus = self._bus
        self._bus = None
        if bus is not None:
            try:
                bus.remove_signal_watch()
            except Exception:
                logger.debug("Failed to remove mosaic SHM bus watch", exc_info=True)

    def _cleanup_failed_start(self) -> None:
        with self._state_lock:
            self._stopping = True
        if self.pipeline is not None:
            try:
                self.pipeline.set_state(Gst.State.NULL)
                self.pipeline.get_state(3 * Gst.SECOND)
            except Exception:
                logger.debug("Failed to null mosaic SHM feeder after startup error", exc_info=True)
        self._remove_bus_watch()
        if self._context_acquired:
            try:
                shared_default_glib_context.release()
            finally:
                self._context_acquired = False
        self.pipeline = None
        self.appsink = None
        self._first_au_event.clear()
        with self._state_lock:
            self._started = False
            self._stopping = False

    def _maybe_request_keyframe(self, reason: str) -> None:
        if self._request_keyframe is None:
            return
        now = time.monotonic()
        if (now - self._last_keyframe_request) < 0.25:
            return
        self._last_keyframe_request = now
        try:
            self._request_keyframe(reason)
        except Exception:
            logger.debug("Keyframe request failed (%s)", reason, exc_info=True)


def ensure_parent_dir(socket_path: str) -> None:
    path = Path(socket_path)
    if path.parent and str(path.parent) not in ("", "."):
        path.parent.mkdir(parents=True, exist_ok=True)
    if not os.path.lexists(path):
        return

    mode = os.lstat(path).st_mode
    if not stat.S_ISSOCK(mode):
        raise RuntimeError(
            f"Refusing to replace non-socket mosaic H.264 SHM path: {path}"
        )

    # A successful connection proves another runtime owns the socket. Only an
    # unconnectable Unix socket is stale crash residue and safe to remove.
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(0.1)
    try:
        probe.connect(str(path))
    except (ConnectionRefusedError, FileNotFoundError):
        path.unlink(missing_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot prove mosaic H.264 SHM socket is stale: {path}: {exc}"
        ) from exc
    else:
        raise RuntimeError(f"Mosaic H.264 SHM socket is already active: {path}")
    finally:
        probe.close()
