"""Process-wide GLib default-context driver for mosaic media pipelines.

The SHM feeder and every WebRTC peer attach bus watches, timers, and invoked
callbacks to GLib's default main context.  Driving that one context from one
thread avoids callback affinity changing between independently polling gateway
threads while still keeping GStreamer's event work off the WebSocket thread.
"""

from __future__ import annotations

import logging
import threading

import gi

gi.require_version("GLib", "2.0")
from gi.repository import GLib  # noqa: E402


logger = logging.getLogger(__name__)


class SharedDefaultGlibContext:
    """Reference-counted owner of one thread driving GLib's default context."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._references = 0
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None

    @property
    def reference_count(self) -> int:
        with self._lock:
            return int(self._references)

    @property
    def thread(self) -> threading.Thread | None:
        with self._lock:
            return self._thread

    def acquire(self) -> None:
        with self._lock:
            self._references += 1
            if self._thread is not None and self._thread.is_alive():
                return

            stop_event = threading.Event()
            thread = threading.Thread(
                target=self._run,
                args=(stop_event,),
                daemon=False,
                name="NoesisMosaicGLib",
            )
            self._stop_event = stop_event
            self._thread = thread
            try:
                thread.start()
            except Exception:
                self._references -= 1
                self._stop_event = None
                self._thread = None
                raise

    def release(self) -> None:
        thread: threading.Thread | None = None
        stop_event: threading.Event | None = None
        with self._lock:
            if self._references <= 0:
                raise RuntimeError("GLib context driver release without acquire")
            self._references -= 1
            if self._references != 0:
                return
            thread = self._thread
            stop_event = self._stop_event
            self._thread = None
            self._stop_event = None
            if stop_event is not None:
                stop_event.set()

        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=3.0)
            if thread.is_alive():
                raise RuntimeError("Shared GLib context thread did not stop")

    @staticmethod
    def _run(stop_event: threading.Event) -> None:
        context = GLib.MainContext.default()
        try:
            while not stop_event.is_set():
                while context.pending():
                    context.iteration(False)
                stop_event.wait(0.005)
        except Exception:
            logger.exception("Shared mosaic GLib context thread failed")


shared_default_glib_context = SharedDefaultGlibContext()
