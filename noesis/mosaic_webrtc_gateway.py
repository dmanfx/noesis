"""
Mosaic WebRTC Gateway

Ultra-light GStreamer WebRTC gateway that consumes already-encoded mosaic H.264
access units from the shared-memory feeder and exposes them as a WebRTC video
track via the existing WebSocketServer signaling path.

Pipeline: appsrc (H.264 AU) → rtph264pay → webrtcbin

No RTSP hop. No depay/re-pay cycle. No transcoding. No decode. No GPU load.
The only RTP packetization happens here, once, for the browser peer.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Optional

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstSdp", "1.0")
gi.require_version("GstWebRTC", "1.0")

from gi.repository import GLib, Gst, GstSdp, GstWebRTC  # noqa: E402

from noesis.mosaic_glib_context import shared_default_glib_context

if TYPE_CHECKING:
    from noesis.mosaic_h264_bridge import MosaicH264ShmFeeder
    from noesis.server.websocket import WebSocketServer

Gst.init(None)

logger = logging.getLogger(__name__)

APPSRC_MAX_AUS = 8
RTP_QUEUE_MAX_BUFFERS = 512
RTP_QUEUE_MAX_BYTES = 4 * 1024 * 1024
RTP_QUEUE_MAX_TIME_NS = 250 * Gst.MSECOND


def _guard_gateway_callback(
    default_factory: Optional[Callable[[], Any]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Lease an SDK/GLib callback against gateway shutdown."""

    def _decorate(callback: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(callback)
        def _wrapped(self: "MosaicWebRTCGateway", *args: Any, **kwargs: Any) -> Any:
            if not self._begin_lifecycle_callback():
                return default_factory() if default_factory is not None else None
            try:
                return callback(self, *args, **kwargs)
            finally:
                self._end_lifecycle_callback()

        return _wrapped

    return _decorate


class MosaicWebRTCGateway:
    """
    Ultra-light GStreamer WebRTC gateway that consumes mosaic H.264 access units
    from :class:`MosaicH264ShmFeeder` and exposes them as a WebRTC video track.
    """

    def __init__(
        self,
        ws_server: "WebSocketServer",
        *,
        h264_feeder: Optional["MosaicH264ShmFeeder"] = None,
        request_keyframe: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.ws = ws_server
        self._h264_feeder = h264_feeder
        self._request_keyframe = request_keyframe
        self.pipeline: Optional[Gst.Pipeline] = None
        self.webrtc: Optional[Gst.Element] = None
        self.appsrc: Optional[Gst.Element] = None
        self.pay: Optional[Gst.Element] = None
        self.queue: Optional[Gst.Element] = None
        self.drain: Optional[Gst.Element] = None
        self.webrtc_sink_pad: Optional[Gst.Pad] = None
        self.webrtc_transceiver: Optional[Any] = None
        self._started = False
        self._peer_count = 0
        self._frame_count = 0
        self._keyframe_count = 0
        self._push_drop_count = 0
        self._offer_h264_pt: Optional[int] = None
        self._remote_description_set = False
        self._pending_create_answer = False
        self._answer_create_started = False
        self._pending_answer_started_at: Optional[float] = None
        self._context_acquired = False
        self._rtp_in_packets = 0
        self._sender_linked = False
        self._sender_link_in_progress = False
        self._pending_ice: list[tuple[str, int]] = []
        self._lifecycle_condition = threading.Condition(threading.RLock())
        self._lifecycle_generation = 0
        self._stopping = False
        self._active_callbacks = 0
        self._glib_source_ids: set[int] = set()
        # Pipeline rebuilds and terminal NULL transition are mutually exclusive.
        # Callback admission is closed and drained before stop acquires this lock,
        # so a peer reset can never publish a replacement PLAYING pipeline after
        # shutdown has already nulled the previous instance.
        self._pipeline_mutation_lock = threading.RLock()
        self._relink_probe_pad: Optional[Gst.Pad] = None
        self._relink_probe_id: Optional[int] = None
        self._appsrc_lock = threading.RLock()
        self._terminal_failure_reported = False

    def _current_generation(self) -> int:
        with self._lifecycle_condition:
            return int(self._lifecycle_generation)

    def _begin_lifecycle_callback(self, generation: Optional[int] = None) -> bool:
        with self._lifecycle_condition:
            if self._stopping or not self._started:
                return False
            if (
                generation is not None
                and int(generation) != int(self._lifecycle_generation)
            ):
                return False
            self._active_callbacks += 1
            return True

    def _end_lifecycle_callback(self) -> None:
        with self._lifecycle_condition:
            self._active_callbacks -= 1
            self._lifecycle_condition.notify_all()

    def _accepts_async_work(self) -> bool:
        with self._lifecycle_condition:
            return bool(self._started and not self._stopping)

    def _invoke_glib(self, callback: Callable[[object], bool]) -> bool:
        with self._lifecycle_condition:
            if self._stopping or not self._started:
                return False
            generation = int(self._lifecycle_generation)

        def _guarded(data: object) -> bool:
            if not self._begin_lifecycle_callback(generation):
                return False
            try:
                return bool(callback(data))
            finally:
                self._end_lifecycle_callback()

        try:
            GLib.MainContext.default().invoke_full(
                GLib.PRIORITY_DEFAULT,
                _guarded,
                None,
            )
            return True
        except Exception:
            logger.exception("Failed to schedule work on the mosaic GLib context")
            return False

    def _schedule_glib_timeout(
        self,
        interval_ms: int,
        callback: Callable[[], bool],
    ) -> Optional[int]:
        source_id_holder: list[int] = []
        with self._lifecycle_condition:
            if self._stopping or not self._started:
                return None
            generation = int(self._lifecycle_generation)

            def _guarded() -> bool:
                if not self._begin_lifecycle_callback(generation):
                    keep = False
                else:
                    try:
                        keep = bool(callback())
                    finally:
                        self._end_lifecycle_callback()
                with self._lifecycle_condition:
                    if self._stopping or generation != self._lifecycle_generation:
                        keep = False
                    if not keep and source_id_holder:
                        self._glib_source_ids.discard(source_id_holder[0])
                return keep

            source_id = int(GLib.timeout_add(int(interval_ms), _guarded))
            source_id_holder.append(source_id)
            self._glib_source_ids.add(source_id)
            return source_id

    def _new_guarded_promise(
        self,
        callback: Callable[[Gst.Promise], None],
    ) -> Gst.Promise:
        generation = self._current_generation()

        def _guarded(promise: Gst.Promise) -> None:
            if not self._begin_lifecycle_callback(generation):
                return
            try:
                callback(promise)
            finally:
                self._end_lifecycle_callback()

        return Gst.Promise.new_with_change_func(_guarded)

    def _cancel_glib_sources(self) -> None:
        with self._lifecycle_condition:
            source_ids = list(self._glib_source_ids)
            self._glib_source_ids.clear()
        for source_id in source_ids:
            try:
                GLib.source_remove(source_id)
            except Exception:
                logger.debug(
                    "Failed to remove WebRTC GLib source %d",
                    source_id,
                    exc_info=True,
                )

    def _cancel_relink_probe(self) -> None:
        """Remove the one-shot blocking relink probe before pipeline teardown."""

        with self._lifecycle_condition:
            pad = self._relink_probe_pad
            probe_id = self._relink_probe_id
            self._relink_probe_pad = None
            self._relink_probe_id = None
            self._sender_link_in_progress = False
        if pad is None or probe_id is None:
            return
        try:
            pad.remove_probe(probe_id)
        except Exception:
            logger.debug(
                "Failed to remove WebRTC sender relink probe %d",
                probe_id,
                exc_info=True,
            )

    def _wait_for_callbacks(self, *, timeout_s: float, phase: str) -> None:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._lifecycle_condition:
            while self._active_callbacks > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise RuntimeError(
                        "MosaicWebRTCGateway callbacks remained active "
                        f"during {phase}"
                    )
                self._lifecycle_condition.wait(timeout=remaining)

    def _set_pipeline_null_and_wait(self, pipeline: Gst.Pipeline) -> None:
        """Require a bounded, observed transition to the terminal NULL state."""

        result = pipeline.set_state(Gst.State.NULL)
        if result == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("GStreamer rejected WebRTC gateway NULL state")
        state_result, current, pending = pipeline.get_state(3 * Gst.SECOND)
        if (
            state_result == Gst.StateChangeReturn.FAILURE
            or current != Gst.State.NULL
            or pending != Gst.State.VOID_PENDING
        ):
            raise RuntimeError(
                "WebRTC gateway did not prove terminal NULL state "
                f"(result={state_result}, current={current}, pending={pending})"
            )

    @staticmethod
    def _set_pipeline_playing_and_wait(pipeline: Gst.Pipeline) -> None:
        """Require a bounded, observed transition to PLAYING."""

        result = pipeline.set_state(Gst.State.PLAYING)
        if result == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("GStreamer rejected WebRTC gateway PLAYING state")
        state_result, current, _pending = pipeline.get_state(3 * Gst.SECOND)
        if (
            state_result == Gst.StateChangeReturn.FAILURE
            or current != Gst.State.PLAYING
        ):
            raise RuntimeError(
                "WebRTC gateway did not reach PLAYING "
                f"(result={state_result}, current={current})"
            )

    @staticmethod
    def _extract_h264_payload_type(sdp_offer: str) -> Optional[int]:
        """Extract the H264 payload type from a browser SDP offer (answer must match offer PT)."""
        if not sdp_offer:
            return None

        lines = [line.strip() for line in sdp_offer.splitlines() if line.strip()]
        video_mline = next((line for line in lines if line.startswith("m=video ")), None)
        if not video_mline:
            return None

        parts = video_mline.split()
        if len(parts) < 4:
            return None

        payload_types: list[int] = []
        for token in parts[3:]:
            try:
                payload_types.append(int(token))
            except Exception:
                continue

        rtpmap_by_pt: dict[int, str] = {}
        fmtp_by_pt: dict[int, str] = {}
        for line in lines:
            if not line.lower().startswith("a=rtpmap:"):
                continue
            try:
                prefix, mapping = line.split(None, 1)
            except ValueError:
                continue
            try:
                pt_str = prefix.split(":", 1)[1]
                pt = int(pt_str)
            except Exception:
                continue
            rtpmap_by_pt[pt] = mapping.strip()

        for line in lines:
            if not line.lower().startswith("a=fmtp:"):
                continue
            # a=fmtp:<pt> <params>
            body = line.split(":", 1)[1]
            try:
                pt_str, params = body.split(None, 1)
                pt = int(pt_str)
            except Exception:
                continue
            fmtp_by_pt[pt] = params.strip()

        h264_pts = [pt for pt in payload_types if rtpmap_by_pt.get(pt, "").lower().startswith("h264/")]
        if not h264_pts:
            return None

        def _score_h264(pt: int) -> tuple[int, int]:
            fmtp = fmtp_by_pt.get(pt, "")
            norm = fmtp.replace(" ", "").lower()
            packetization_1 = "packetization-mode=1" in norm

            # Default profile preference:
            # Prefer constrained-baseline-ish variants first for browser compatibility.
            profile = ""
            if "profile-level-id=" in norm:
                try:
                    profile = norm.split("profile-level-id=", 1)[1].split(";", 1)[0]
                except Exception:
                    profile = ""

            profile_score = 0
            if profile.startswith("42e0"):
                profile_score = 50
            elif profile.startswith("42c0"):
                profile_score = 45
            elif profile.startswith("4200"):
                profile_score = 40
            elif profile.startswith("4d00"):
                profile_score = 30
            elif profile.startswith("6400"):
                profile_score = 20
            elif profile.startswith("f400"):
                profile_score = 10
            else:
                profile_score = 0

            # Primary: packetization-mode=1; Secondary: profile preference; Tertiary: stable ordering.
            return (100 if packetization_1 else 0) + profile_score, -pt

        best = sorted(h264_pts, key=_score_h264, reverse=True)[0]
        return int(best)

    @staticmethod
    def _find_video_direction(sdp_text: str) -> Optional[str]:
        in_video = False
        for line in (sdp_text or "").splitlines():
            line = line.strip()
            if line.startswith("m="):
                in_video = line.startswith("m=video ")
                continue
            if not in_video:
                continue
            if line in ("a=sendonly", "a=recvonly", "a=sendrecv", "a=inactive"):
                return line.split("=", 1)[1]
        return None

    @staticmethod
    def _find_h264_payload_type(sdp_text: str) -> Optional[int]:
        for line in (sdp_text or "").splitlines():
            line = line.strip()
            if not line.lower().startswith("a=rtpmap:"):
                continue
            try:
                prefix, mapping = line.split(None, 1)
                pt = int(prefix.split(":", 1)[1])
            except Exception:
                continue
            if mapping.lower().startswith("h264/"):
                return int(pt)
        return None

    def build(self) -> None:
        """Construct the GStreamer pipeline: appsrc AUs → single RTP pay → webrtcbin."""
        self.pipeline = Gst.Pipeline.new("webrtc-gateway")

        self.appsrc = Gst.ElementFactory.make("appsrc", "h264_appsrc")
        self.pay = Gst.ElementFactory.make("rtph264pay", "pay")
        # RTP-level queue used as the re-link point (drain → webrtcbin). Keep it
        # non-leaky so we never drop individual RTP packets mid-frame.
        self.queue = Gst.ElementFactory.make("queue", "webrtc_queue")
        self.webrtc = Gst.ElementFactory.make("webrtcbin", "webrtc")
        # Drain sink: keep AU ingest flowing before a peer offers WebRTC.
        self.drain = Gst.ElementFactory.make("fakesink", "webrtc_drain")

        for name, elem in (
            ("appsrc", self.appsrc),
            ("pay", self.pay),
            ("queue", self.queue),
            ("webrtc", self.webrtc),
            ("drain", self.drain),
        ):
            if elem is None:
                raise RuntimeError(f"Failed to create {name} element")

        # Live AU source. The upstream leaky policy drops a newly arriving whole
        # AU when this peer's bounded queue is full. push_h264_au checks the
        # current level first so every such drop is counted and reported.
        self.appsrc.set_property("is-live", True)
        self.appsrc.set_property("format", Gst.Format.TIME)
        self.appsrc.set_property("stream-type", 0)  # GST_APP_STREAM_TYPE_STREAM
        self.appsrc.set_property("do-timestamp", True)
        self.appsrc.set_property("block", False)
        self.appsrc.set_property("max-buffers", APPSRC_MAX_AUS)
        self.appsrc.set_property("max-bytes", 0)
        self.appsrc.set_property("max-time", 0)
        self.appsrc.set_property("leaky-type", 1)  # upstream: drop incoming AU
        if self.appsrc.find_property("current-level-buffers") is None:
            raise RuntimeError(
                "GStreamer appsrc lacks required current-level-buffers accounting"
            )
        self.appsrc.set_property(
            "caps",
            Gst.Caps.from_string(
                "video/x-h264,stream-format=byte-stream,alignment=au"
            ),
        )

        # Single packetization point for WebRTC. Repeat SPS/PPS with every IDR
        # so late-joining browsers can start decoding immediately.
        self.pay.set_property("config-interval", -1)
        try:
            self.pay.set_property("aggregate-mode", 1)  # zero-latency
        except Exception:
            pass

        self.queue.set_property("leaky", 0)
        self.queue.set_property("max-size-buffers", RTP_QUEUE_MAX_BUFFERS)
        self.queue.set_property("max-size-bytes", RTP_QUEUE_MAX_BYTES)
        self.queue.set_property("max-size-time", RTP_QUEUE_MAX_TIME_NS)

        self.webrtc.set_property("bundle-policy", 3)  # max-bundle
        self.drain.set_property("sync", False)
        self.drain.set_property("async", False)

        self.pipeline.add(self.appsrc)
        self.pipeline.add(self.pay)
        self.pipeline.add(self.queue)
        self.pipeline.add(self.webrtc)
        self.pipeline.add(self.drain)

        if not self.appsrc.link(self.pay):
            raise RuntimeError("Failed to link appsrc → pay")
        if not self.pay.link(self.queue):
            raise RuntimeError("Failed to link pay → webrtc_queue")

        self._sender_linked = False
        self._sender_link_in_progress = False
        self.webrtc_sink_pad = None
        self.webrtc_transceiver = None
        if not self.queue.link(self.drain):
            raise RuntimeError("Failed to link webrtc_queue → webrtc_drain")

        # Count RTP packets entering webrtcbin (should be >0 before we answer).
        try:
            queue_src_pad = self.queue.get_static_pad("src")
            if queue_src_pad is not None:
                queue_src_pad.add_probe(Gst.PadProbeType.BUFFER, self._on_rtp_in_probe, None)
        except Exception:
            pass

        # Count AUs pre-payload (not per-RTP-packet) to avoid GIL starvation.
        try:
            appsrc_src = self.appsrc.get_static_pad("src")
            if appsrc_src is not None:
                appsrc_src.add_probe(Gst.PadProbeType.BUFFER, self._on_frame_probe, None)
                logger.info("Added frame probe on appsrc src pad")
        except Exception:
            pass

        # A single-home LAN does not need an external ICE discovery service.
        # Supplying an explicit server is the opt-in for deployments that do.
        stun_server = os.environ.get("NOESIS_MOSAIC_WEBRTC_STUN_SERVER", "").strip()
        if stun_server:
            stun_server = stun_server.split(",", 1)[0].strip()
        if stun_server:
            self.webrtc.set_property("stun-server", stun_server)
            logger.info("WebRTC STUN server: %s", stun_server)

        turn_server = os.environ.get("NOESIS_MOSAIC_WEBRTC_TURN_SERVER", "").strip()
        if turn_server:
            try:
                self.webrtc.set_property("turn-server", turn_server)
                logger.info("WebRTC TURN server configured")
            except Exception as exc:
                logger.warning("Failed to set webrtcbin turn-server: %s", exc)

        self.webrtc.connect("on-ice-candidate", self._on_ice_candidate)
        self.webrtc.connect("on-negotiation-needed", self._on_negotiation_needed)
        self.webrtc.connect("notify::ice-connection-state", self._on_ice_connection_state)
        self.webrtc.connect("notify::connection-state", self._on_connection_state)

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self._on_bus_error)
        bus.connect("message::eos", self._on_bus_eos)
        bus.connect("message::state-changed", self._on_bus_state_changed)

        logger.info(
            "MosaicWebRTCGateway built: source=h264_shm_au feeder=%s",
            "yes" if self._h264_feeder is not None else "no",
        )

    def push_h264_au(self, buffer: Gst.Buffer) -> bool:
        """Push one encoded H.264 access unit into this gateway's appsrc."""
        try:
            with self._appsrc_lock:
                if not self._started or self._stopping:
                    return False
                appsrc = self.appsrc
                if appsrc is None:
                    return False
                queued_aus = int(appsrc.get_property("current-level-buffers"))
                if queued_aus >= APPSRC_MAX_AUS:
                    self._push_drop_count += 1
                    if self._push_drop_count == 1 or self._push_drop_count % 30 == 0:
                        logger.warning(
                            "Dropping whole H.264 AU for slow WebRTC peer: "
                            "queued_aus=%d limit=%d drops=%d",
                            queued_aus,
                            APPSRC_MAX_AUS,
                            self._push_drop_count,
                        )
                    return False

                # SHM timestamps belong to the feeder pipeline's clock domain.
                # Let this live appsrc stamp the AU in the peer pipeline's own
                # running-time domain; duration and keyframe flags are retained.
                buffer.pts = Gst.CLOCK_TIME_NONE
                buffer.dts = Gst.CLOCK_TIME_NONE
                ret = appsrc.emit("push-buffer", buffer)
        except Exception:
            self._push_drop_count += 1
            logger.exception("Failed to enqueue H.264 AU for WebRTC peer")
            return False
        if ret == Gst.FlowReturn.OK:
            return True
        self._push_drop_count += 1
        if self._push_drop_count == 1 or self._push_drop_count % 30 == 0:
            logger.warning(
                "WebRTC appsrc rejected whole H.264 AU: flow=%s drops=%d",
                ret,
                self._push_drop_count,
            )
        return False

    @_guard_gateway_callback()
    def _on_ice_connection_state(self, webrtc: Gst.Element, pspec: object) -> None:
        """Log ICE connection state changes."""
        if webrtc is not self.webrtc:
            return
        state = webrtc.get_property("ice-connection-state")
        state_name = GstWebRTC.WebRTCICEConnectionState(state).value_nick if state else "unknown"
        logger.info("!!! ICE connection state: %s", state_name)

    @_guard_gateway_callback()
    def _on_connection_state(self, webrtc: Gst.Element, pspec: object) -> None:
        """Log peer connection state changes."""
        if webrtc is not self.webrtc:
            return
        state = webrtc.get_property("connection-state")
        state_name = GstWebRTC.WebRTCPeerConnectionState(state).value_nick if state else "unknown"
        logger.info("!!! Peer connection state: %s", state_name)
        if state_name == "connected" and self._request_keyframe is not None:
            try:
                self._request_keyframe("webrtc_connected")
            except Exception:
                logger.debug("Mosaic keyframe request failed on connected", exc_info=True)

    @_guard_gateway_callback(lambda: Gst.PadProbeReturn.OK)
    def _on_frame_probe(self, pad: Gst.Pad, info: Gst.PadProbeInfo, user_data: object) -> Gst.PadProbeReturn:
        """Count H.264 access units flowing into the single RTP payloader."""
        try:
            buf = info.get_buffer()
        except Exception:
            buf = None

        # Track keyframes (non-delta units) to correlate with "bytesReceived but framesDecoded=0" cases.
        is_keyframe = False
        try:
            if buf is not None and not buf.has_flags(Gst.BufferFlags.DELTA_UNIT):
                is_keyframe = True
        except Exception:
            is_keyframe = False
        if is_keyframe:
            self._keyframe_count += 1

        self._frame_count += 1
        if self._frame_count == 1:
            logger.info(">>> First video frame received in gateway!")
            # If we delayed answering because AUs hadn't started yet, kick answer creation now.
            if (
                self._pending_create_answer
                and self._remote_description_set
                and not self._answer_create_started
                and self._sender_linked
            ):
                def _do(_: object) -> bool:
                    try:
                        self._maybe_start_create_answer(force=False)
                    except Exception:
                        logger.debug("Failed to start answer after first frame", exc_info=True)
                    return False

                self._invoke_glib(_do)
        if self._keyframe_count == 1 and is_keyframe:
            logger.info(">>> First keyframe observed in gateway")
        elif self._frame_count % 300 == 0:
            logger.debug(
                "Gateway video frames=%d keyframes=%d appsrc_drops=%d",
                int(self._frame_count),
                int(self._keyframe_count),
                int(self._push_drop_count),
            )
        return Gst.PadProbeReturn.OK

    def _queue_payload_matches_offer(self) -> bool:
        if self._offer_h264_pt is None:
            return True
        if self.queue is None:
            return False
        pad = self.queue.get_static_pad("src")
        if pad is None:
            return False
        caps = pad.get_current_caps() or pad.query_caps(None)
        try:
            if caps is None or caps.get_size() <= 0:
                return False
            s = caps.get_structure(0)
            if s is None:
                return False
            ok, payload = s.get_int("payload")
            if not ok:
                return False
            return int(payload) == int(self._offer_h264_pt)
        except Exception:
            return False

    def _record_transceivers(self, *, stage: str, location: str) -> None:
        """Log a concise, non-secret webrtcbin transceiver inventory at DEBUG."""
        if self.webrtc is None:
            return
        try:
            transceivers_summary: list[dict[str, Any]] = []
            arr = self.webrtc.emit("get-transceivers")
            n = int(arr.len) if arr is not None else 0
            for i in range(n):
                try:
                    tr = self.webrtc.emit("get-transceiver", int(i))
                except Exception:
                    tr = None
                if tr is None:
                    continue
                try:
                    transceivers_summary.append(
                        {
                            "i": i,
                            "kind": tr.get_property("kind").value_nick,
                            "direction": tr.get_property("direction").value_nick,
                            "current_direction": tr.get_property("current-direction").value_nick,
                            "mlineindex": int(tr.get_property("mlineindex")),
                            "mid": tr.get_property("mid"),
                        }
                    )
                except Exception:
                    continue

            sink_tr_info: Optional[dict[str, Any]] = None
            if self.webrtc_sink_pad is not None:
                try:
                    sink_tr = self.webrtc_sink_pad.get_property("transceiver")
                    sink_tr_info = {
                        "kind": sink_tr.get_property("kind").value_nick,
                        "direction": sink_tr.get_property("direction").value_nick,
                        "current_direction": sink_tr.get_property("current-direction").value_nick,
                        "mlineindex": int(sink_tr.get_property("mlineindex")),
                        "mid": sink_tr.get_property("mid"),
                    }
                except Exception:
                    sink_tr_info = None

            logger.debug(
                "WebRTC transceivers location=%s stage=%s count=%d "
                "transceivers=%s sink_transceiver=%s sender_linked=%s "
                "rtp_in_packets=%d frame_count=%d offer_h264_pt=%s",
                location,
                stage,
                n,
                transceivers_summary,
                sink_tr_info,
                self._sender_linked,
                self._rtp_in_packets,
                self._frame_count,
                self._offer_h264_pt,
            )
        except Exception:
            logger.debug("Failed to inspect transceivers (%s)", stage, exc_info=True)

    def _ensure_webrtc_sender_pad(self) -> None:
        """Ensure a webrtcbin sink pad (and its transceiver) exists for video sending."""
        if self.webrtc is None:
            return
        if self.webrtc_sink_pad is not None:
            # Update codec preferences if we now know the offer payload type.
            pt = int(self._offer_h264_pt) if self._offer_h264_pt is not None else 96
            caps = Gst.Caps.from_string(
                f"application/x-rtp,media=video,encoding-name=H264,clock-rate=90000,payload={pt},packetization-mode=(string)1"
            )
            if self.webrtc_transceiver is None:
                try:
                    self.webrtc_transceiver = self.webrtc_sink_pad.get_property("transceiver")
                except Exception:
                    self.webrtc_transceiver = None
            if self.webrtc_transceiver is not None:
                try:
                    self.webrtc_transceiver.set_property(
                        "direction", GstWebRTC.WebRTCRTPTransceiverDirection.SENDONLY
                    )
                except Exception:
                    pass
                try:
                    self.webrtc_transceiver.set_property("codec-preferences", caps)
                except Exception:
                    pass
            return

        pt = int(self._offer_h264_pt) if self._offer_h264_pt is not None else 96
        caps = Gst.Caps.from_string(
            f"application/x-rtp,media=video,encoding-name=H264,clock-rate=90000,payload={pt},packetization-mode=(string)1"
        )

        template = self.webrtc.get_pad_template("sink_%u")
        sink_pad = None
        if template is not None:
            try:
                sink_pad = self.webrtc.request_pad(template, None, caps)
            except Exception:
                sink_pad = None
        if sink_pad is None:
            try:
                sink_pad = self.webrtc.request_pad_simple("sink_%u")
            except Exception:
                sink_pad = None
        if sink_pad is None:
            raise RuntimeError("Failed to request webrtcbin sink pad")

        self.webrtc_sink_pad = sink_pad
        try:
            sink_pad.set_property("msid", "noesis-mosaic")
        except Exception:
            pass
        try:
            self.webrtc_transceiver = sink_pad.get_property("transceiver")
        except Exception:
            self.webrtc_transceiver = None
        if self.webrtc_transceiver is not None:
            try:
                self.webrtc_transceiver.set_property("direction", GstWebRTC.WebRTCRTPTransceiverDirection.SENDONLY)
            except Exception:
                pass
            try:
                self.webrtc_transceiver.set_property("codec-preferences", caps)
            except Exception:
                pass

    def _link_sender_into_webrtc(self) -> None:
        """Switch queue output from drain sink to webrtcbin (must run in GLib thread)."""
        if self.queue is None or self.webrtc is None or self.drain is None:
            raise RuntimeError("Cannot link sender: missing queue/webrtc/drain")
        if self._sender_linked:
            return
        if self._sender_link_in_progress:
            return
        if not self._queue_payload_matches_offer():
            # Wait until rtph264pay has renegotiated its payload type.
            return

        # Ensure we create the webrtcbin sink pad/transceiver *before* setting remote SDP
        # to avoid webrtcbin creating a separate recvonly transceiver for the offer and
        # later answering with a=inactive due to transceiver mismatch.
        try:
            self._ensure_webrtc_sender_pad()
        except Exception as exc:
            raise RuntimeError("Failed to ensure WebRTC sender pad") from exc

        queue_src = self.queue.get_static_pad("src")
        if queue_src is None:
            raise RuntimeError("webrtc_queue src pad missing")

        sink_pad = self.webrtc_sink_pad
        if sink_pad is None:
            raise RuntimeError("webrtcbin sink pad missing after ensure")

        # Relink while the src pad is blocked to avoid transient "not-linked" stream errors.
        with self._lifecycle_condition:
            if self._stopping or not self._started:
                return
            generation = int(self._lifecycle_generation)
            self._sender_link_in_progress = True

        probe_id_holder: list[int] = []
        probe_finished = False

        def _do_relink(pad: Gst.Pad, info: Gst.PadProbeInfo, user_data: object) -> Gst.PadProbeReturn:
            nonlocal probe_finished
            if not self._begin_lifecycle_callback(generation):
                with self._lifecycle_condition:
                    probe_finished = True
                    self._sender_link_in_progress = False
                return Gst.PadProbeReturn.REMOVE
            try:
                if pad is not queue_src:
                    return Gst.PadProbeReturn.REMOVE
                if sink_pad is not self.webrtc_sink_pad:
                    return Gst.PadProbeReturn.REMOVE
                peer = pad.get_peer()
                if peer is not None:
                    try:
                        pad.unlink(peer)
                    except Exception:
                        pass

                ret = pad.link(sink_pad)
                if ret != Gst.PadLinkReturn.OK:
                    logger.error("Failed to link webrtc_queue → webrtcbin: %s", ret)
                    return Gst.PadProbeReturn.REMOVE

                self._sender_linked = True
                logger.info("Linked pay → webrtc_queue → webrtcbin sink pad: %s", sink_pad.get_name())
                if self._request_keyframe is not None:
                    try:
                        self._request_keyframe("webrtc_sender_linked")
                    except Exception:
                        logger.debug(
                            "Mosaic keyframe request failed after sender link",
                            exc_info=True,
                        )
                return Gst.PadProbeReturn.REMOVE
            finally:
                with self._lifecycle_condition:
                    probe_finished = True
                    if (
                        probe_id_holder
                        and self._relink_probe_id == probe_id_holder[0]
                    ):
                        self._relink_probe_pad = None
                        self._relink_probe_id = None
                    self._sender_link_in_progress = False
                self._end_lifecycle_callback()

        try:
            probe_id = int(
                queue_src.add_probe(
                    Gst.PadProbeType.BLOCK | Gst.PadProbeType.BUFFER,
                    _do_relink,
                    None,
                )
            )
            probe_id_holder.append(probe_id)
            remove_immediately = False
            with self._lifecycle_condition:
                if (
                    probe_finished
                    or self._stopping
                    or generation != self._lifecycle_generation
                ):
                    remove_immediately = not probe_finished
                    self._sender_link_in_progress = False
                else:
                    self._relink_probe_pad = queue_src
                    self._relink_probe_id = probe_id
            if remove_immediately:
                queue_src.remove_probe(probe_id)
        except Exception as exc:
            with self._lifecycle_condition:
                self._sender_link_in_progress = False
            raise RuntimeError(
                "Failed to install the required blocking WebRTC relink probe"
            ) from exc

    @_guard_gateway_callback(lambda: Gst.PadProbeReturn.OK)
    def _on_rtp_in_probe(self, pad: Gst.Pad, info: Gst.PadProbeInfo, user_data: object) -> Gst.PadProbeReturn:
        self._rtp_in_packets += 1
        if self._rtp_in_packets == 1:
            # If we delayed answering until RTP is present, kick answer creation now.
            if self._pending_create_answer and self._remote_description_set and not self._answer_create_started:
                def _do(_: object) -> bool:
                    try:
                        self._maybe_start_create_answer(force=False)
                    except Exception:
                        logger.debug("Failed to start answer after first RTP packet", exc_info=True)
                    return False

                self._invoke_glib(_do)
        elif self._rtp_in_packets % 200 == 0:
            logger.info(">>> Gateway RTP packets into webrtcbin: %d", self._rtp_in_packets)
        return Gst.PadProbeReturn.OK

    def _switch_to_webrtc_sender(self) -> None:
        """
        Ensure the RTP sender is linked into webrtcbin.

        Must be called from the GLib main context (we use invoke_full for entrypoints).
        """
        if not self._accepts_async_work():
            return
        if self._sender_linked:
            return
        if self.pipeline is None or self.webrtc is None or self.queue is None:
            return

        # Keep the payloader payload type aligned with the offer and trigger a caps reconfigure.
        if self.pay is not None and self._offer_h264_pt is not None:
            try:
                self.pay.set_property("pt", int(self._offer_h264_pt))
            except Exception:
                pass
            try:
                pay_src = self.pay.get_static_pad("src")
                if pay_src is not None:
                    pay_src.send_event(Gst.Event.new_reconfigure())
            except Exception:
                pass

        # Attempt the link (will no-op until the queue caps show the negotiated payload).
        try:
            self._link_sender_into_webrtc()
        except Exception:
            logger.exception("Failed to link sender into webrtcbin")
            self._report_terminal_pipeline_failure(
                "webrtc_gateway_sender_link_failed"
            )
            return

        if not self._sender_linked:
            # Retry shortly; payload/caps renegotiation can lag behind pt property updates.
            def _retry() -> bool:
                try:
                    self._switch_to_webrtc_sender()
                except Exception:
                    logger.debug("Sender link retry failed", exc_info=True)
                return not self._sender_linked

            self._schedule_glib_timeout(50, _retry)
            return

        logger.debug("WebRTC sender linked with H264 payload type %s", self._offer_h264_pt)

    def _maybe_start_create_answer(self, *, force: bool) -> None:
        """Start create-answer if we're ready (or forced). Must run in GLib context."""
        if not self._accepts_async_work():
            return
        if self.webrtc is None:
            return
        if self._answer_create_started:
            return
        if not self._remote_description_set:
            return
        if not self._sender_linked:
            # Switch to WebRTC output once the payloader is emitting the offer PT.
            try:
                self._switch_to_webrtc_sender()
            except Exception:
                logger.debug("Failed to switch gateway output to webrtcbin", exc_info=True)
            if self._terminal_failure_reported:
                return
            if not self._sender_linked:
                self._pending_create_answer = True
                return
        # Wait until we have at least one encoded H264 AU (via appsrc probe).
        # This avoids webrtcbin answering with a=inactive when no sender stream is observed yet.
        if not force and self._frame_count <= 0:
            self._pending_create_answer = True
            return

        self._pending_create_answer = False
        self._answer_create_started = True
        self._pending_answer_started_at = None

        logger.info("    Creating WebRTC answer...")
        self._record_transceivers(stage="before create-answer", location="mosaic_webrtc_gateway.py:_maybe_start_create_answer")
        promise = self._new_guarded_promise(self._on_answer_created)
        self.webrtc.emit("create-answer", None, promise)

    def start(self) -> None:
        """Start the gateway and join the process-wide GLib context driver."""
        if self._started:
            logger.warning("WebRTC gateway already started")
            return

        with self._lifecycle_condition:
            self._stopping = False
            self._lifecycle_generation += 1
            self._active_callbacks = 0

        try:
            if self.pipeline is None:
                # Start in drain mode so AU ingest remains live before an offer.
                self.build()
            assert self.pipeline is not None
            shared_default_glib_context.acquire()
            self._context_acquired = True
            with self._lifecycle_condition:
                self._started = True
            self._set_pipeline_playing_and_wait(self.pipeline)
            if self._h264_feeder is not None:
                self._h264_feeder.register_consumer(self)
        except Exception:
            with self._lifecycle_condition:
                self._stopping = True
            self._cancel_glib_sources()
            self._cancel_relink_probe()
            if self.pipeline is not None:
                try:
                    self.pipeline.set_state(Gst.State.NULL)
                    self.pipeline.get_state(3 * Gst.SECOND)
                except Exception:
                    logger.debug("Failed to null partially started gateway", exc_info=True)
                try:
                    bus = self.pipeline.get_bus()
                    if bus is not None:
                        bus.remove_signal_watch()
                except Exception:
                    logger.debug(
                        "Failed to remove partially started gateway bus watch",
                        exc_info=True,
                    )
            if self._context_acquired:
                try:
                    shared_default_glib_context.release()
                finally:
                    self._context_acquired = False
            self.pipeline = None
            self.webrtc = None
            self.appsrc = None
            self.pay = None
            self.queue = None
            self.drain = None
            self.webrtc_sink_pad = None
            self.webrtc_transceiver = None
            with self._lifecycle_condition:
                self._started = False
                self._stopping = False
            raise
        logger.info("MosaicWebRTCGateway started")

    def stop(self) -> None:
        """Stop and prove quiescence of every gateway-owned async resource."""
        with self._lifecycle_condition:
            if not self._started:
                return
            self._stopping = True
            self._lifecycle_generation += 1

        logger.info("Stopping MosaicWebRTCGateway...")
        if self._h264_feeder is not None:
            try:
                self._h264_feeder.unregister_consumer(self)
            except Exception:
                logger.debug("Failed to unregister H.264 feeder consumer", exc_info=True)
        self._cancel_glib_sources()
        self._cancel_relink_probe()

        # Let callbacks admitted before the barrier finish while the GLib
        # context and current pipeline are still intact. No callback admitted
        # after `_stopping` may enter.
        self._wait_for_callbacks(timeout_s=1.0, phase="pre-NULL drain")

        pipeline_error: Optional[BaseException] = None
        try:
            with self._pipeline_mutation_lock, self._appsrc_lock:
                if self.pipeline:
                    self._set_pipeline_null_and_wait(self.pipeline)
        except Exception as e:
            pipeline_error = e
            logger.exception("Error stopping WebRTC gateway pipeline")

        self._wait_for_callbacks(timeout_s=1.0, phase="post-NULL drain")

        if pipeline_error is not None:
            raise RuntimeError(
                "MosaicWebRTCGateway pipeline failed to enter NULL state"
            ) from pipeline_error

        try:
            if self.pipeline is not None:
                bus = self.pipeline.get_bus()
                if bus is not None:
                    bus.remove_signal_watch()
        except Exception:
            logger.debug("Failed to remove WebRTC bus signal watch", exc_info=True)

        if self._context_acquired:
            shared_default_glib_context.release()
            self._context_acquired = False

        self._pending_create_answer = False
        self._answer_create_started = False
        self._pending_answer_started_at = None
        self._remote_description_set = False
        self._pending_ice.clear()
        self._sender_linked = False
        self._sender_link_in_progress = False
        self.webrtc_sink_pad = None
        self.webrtc_transceiver = None
        self.pipeline = None
        self.webrtc = None
        self.appsrc = None
        self.pay = None
        self.queue = None
        self.drain = None

        with self._lifecycle_condition:
            self._started = False
        logger.info("MosaicWebRTCGateway stopped")

    def _rebuild_pipeline_for_new_peer(self) -> None:
        """Tear down and rebuild the gateway pipeline to handle a new PeerConnection cleanly."""
        with self._pipeline_mutation_lock, self._appsrc_lock:
            if not self._accepts_async_work():
                raise RuntimeError("WebRTC gateway is not accepting peer rebuilds")
            with self._lifecycle_condition:
                self._lifecycle_generation += 1
            self._cancel_glib_sources()
            self._cancel_relink_probe()
            # Stop the old pipeline (keep the gateway thread alive).
            if self.pipeline is not None:
                try:
                    bus = self.pipeline.get_bus()
                    if bus is not None:
                        bus.remove_signal_watch()
                except Exception:
                    logger.debug("Failed to remove old WebRTC bus watch", exc_info=True)
                self._set_pipeline_null_and_wait(self.pipeline)

            self.pipeline = None
            self.webrtc = None
            self.appsrc = None
            self.pay = None
            self.queue = None
            self.drain = None
            self.webrtc_sink_pad = None
            self.webrtc_transceiver = None
            self._sender_linked = False
            self._sender_link_in_progress = False
            self._remote_description_set = False
            self._pending_create_answer = False
            self._answer_create_started = False
            self._pending_answer_started_at = None
            self._pending_ice.clear()
            self._rtp_in_packets = 0
            self._frame_count = 0
            self._offer_h264_pt = None
            self._terminal_failure_reported = False

            # Rebuild and start playing only while admission remains open. Stop
            # closes admission before waiting on this mutation lock.
            if not self._accepts_async_work():
                raise RuntimeError("WebRTC gateway shutdown interrupted peer rebuild")
            self.build()
            if self.pipeline is not None:
                self._set_pipeline_playing_and_wait(self.pipeline)

    def reset_peer(
        self,
        reason: str = "owner_disconnected",
        on_complete: Optional[Callable[[bool], None]] = None,
    ) -> None:
        """Revoke the current peer while keeping this bounded gateway slot warm.

        A WebRTC media path remains viable after its signaling socket disappears,
        so clearing WebSocket ownership alone is not revocation. Rebuilding the
        peer pipeline tears down ICE/DTLS/RTP state and returns the same gateway
        object to drain mode for the next authenticated owner.
        """

        if not self._accepts_async_work():
            if callable(on_complete):
                on_complete(False)
            return

        def _complete(ok: bool) -> None:
            if not callable(on_complete):
                return
            try:
                on_complete(ok)
            except Exception:
                logger.debug("WebRTC peer-reset completion callback failed", exc_info=True)

        def _reset(_: object) -> bool:
            ok = False
            try:
                if not self._accepts_async_work():
                    return False
                logger.info("Resetting WebRTC peer after %s", str(reason or "owner_disconnected"))
                self._peer_count = 0
                self._rebuild_pipeline_for_new_peer()
                ok = self.pipeline is not None and self.webrtc is not None
            except Exception:
                logger.exception("Failed to reset revoked WebRTC peer")
            finally:
                _complete(ok)
            return False

        if not self._invoke_glib(_reset):
            _complete(False)

    # ========== WebSocket signaling methods (called by WebSocketServer) ==========

    def accept_offer(self, sdp_offer: str) -> None:
        """Handle incoming WebRTC offer from browser client."""
        if not self._accepts_async_work():
            logger.warning("Ignoring WebRTC offer during gateway shutdown")
            return
        logger.info(">>> Gateway accept_offer called (offer length=%d)", len(sdp_offer) if sdp_offer else 0)
        if self.webrtc is None:
            logger.error("Cannot accept offer: webrtcbin not initialized")
            return

        # Marshal into the GLib context driving the gateway to avoid cross-thread
        # interaction with GStreamer/webrtcbin.
        def _do_offer(_: object) -> bool:
            try:
                self._accept_offer_impl(sdp_offer)
            except Exception:
                logger.exception("Error processing WebRTC offer")
            return False

        self._invoke_glib(_do_offer)

    def _accept_offer_impl(self, sdp_offer: str) -> None:
        if not self._accepts_async_work():
            return
        if self.webrtc is None:
            logger.error("Cannot accept offer: webrtcbin not initialized")
            return

        if self._request_keyframe is not None:
            try:
                self._request_keyframe("webrtc_offer")
            except Exception:
                logger.debug("Mosaic keyframe request failed on offer", exc_info=True)

        # webrtcbin represents a single PeerConnection. If a client reconnects (new offer),
        # rebuild the gateway pipeline so ICE/DTLS state is clean.
        if self._peer_count > 0:
            self._rebuild_pipeline_for_new_peer()
        self._peer_count += 1

        if self.webrtc is None:
            logger.error("Cannot accept offer: webrtcbin not initialized after rebuild")
            return

        # Reset answer gating for a new offer.
        self._remote_description_set = False
        self._pending_create_answer = False
        self._answer_create_started = False
        self._pending_answer_started_at = None
        self._keyframe_count = 0
        self._pending_ice.clear()

        # Refresh transceiver reference (can change across negotiation cycles).
        try:
            if self.webrtc_sink_pad is not None:
                self.webrtc_transceiver = self.webrtc_sink_pad.get_property("transceiver")
        except Exception:
            pass

        offer_pt = self._extract_h264_payload_type(sdp_offer)
        logger.debug(
            "WebRTC offer summary lines=%d video_direction=%s h264_pt=%s",
            len((sdp_offer or "").splitlines()),
            self._find_video_direction(sdp_offer),
            offer_pt,
        )
        if offer_pt is not None:
            self._offer_h264_pt = offer_pt
        else:
            logger.warning("No H264 payload type found in offer")
            try:
                self.ws.send_webrtc_error("webrtc_gateway_no_h264_in_offer", gateway=self)
            except Exception:
                pass
            return

        # Reconfigure rtph264pay payload type to match the offer, and request renegotiation.
        if self.pay is not None and self._offer_h264_pt is not None:
            try:
                self.pay.set_property("pt", int(self._offer_h264_pt))
            except Exception:
                pass
            try:
                pay_src = self.pay.get_static_pad("src")
                if pay_src is not None:
                    pay_src.send_event(Gst.Event.new_reconfigure())
            except Exception:
                pass

        # Ensure the send transceiver exists before setting the remote description.
        # This avoids webrtcbin creating a separate recvonly transceiver for the offer
        # and later producing an answer with a=inactive due to transceiver mismatch.
        try:
            self._ensure_webrtc_sender_pad()
        except Exception:
            logger.debug("Failed to pre-create webrtc sender pad", exc_info=True)

        logger.info("    Parsing SDP and setting remote description...")
        try:
            res, sdpmsg = GstSdp.SDPMessage.new_from_text(sdp_offer)
            if res != GstSdp.SDPResult.OK:
                raise ValueError(f"Failed to parse SDP offer: {res}")

            offer = GstWebRTC.WebRTCSessionDescription.new(GstWebRTC.WebRTCSDPType.OFFER, sdpmsg)

            # Set remote description
            promise = self._new_guarded_promise(
                self._on_set_remote_description_done
            )
            self.webrtc.emit("set-remote-description", offer, promise)
        except Exception as e:
            logger.exception("Error processing WebRTC offer: %s", e)
            self.ws.send_webrtc_error(str(e), gateway=self)

    @_guard_gateway_callback()
    def _on_set_remote_description_done(self, promise: Gst.Promise) -> None:
        """Callback after setting remote description."""
        logger.info("    _on_set_remote_description_done callback fired")
        promise.wait()
        reply = promise.get_reply()
        if reply is None:
            logger.info("    Remote description set successfully (no reply structure)")
        else:
            logger.info("    Remote description set: %s", reply)

        # Refresh transceiver reference (can change across negotiation cycles).
        try:
            if self.webrtc_sink_pad is not None:
                self.webrtc_transceiver = self.webrtc_sink_pad.get_property("transceiver")
        except Exception:
            pass

        self._remote_description_set = True

        # Apply any ICE candidates received before remote description completed.
        if self.webrtc is not None and self._pending_ice:
            for cand, mline in list(self._pending_ice):
                try:
                    self.webrtc.emit("add-ice-candidate", int(mline), str(cand))
                except Exception:
                    pass
            self._pending_ice.clear()

        # Log transceiver inventory after remote description is applied.
        # This is critical for diagnosing why m=video becomes a=inactive.
        self._record_transceivers(
            stage="after remote-description", location="mosaic_webrtc_gateway.py:_on_set_remote_description_done"
        )

        # Create answer only after RTP is flowing and the sender is linked into webrtcbin.
        self._maybe_start_create_answer(force=False)

        if self._pending_create_answer and self._pending_answer_started_at is None:
            self._pending_answer_started_at = time.monotonic()

            def _poll_start_answer() -> bool:
                try:
                    if not self._accepts_async_work():
                        return False
                    if not self._pending_create_answer or self._answer_create_started:
                        return False

                    # Keep trying until the sender is linked and we have frames.
                    # Negotiation/caps updates can lag behind pt updates and pad relinking.
                    self._maybe_start_create_answer(force=False)
                    if self._answer_create_started:
                        return False

                    started_at = self._pending_answer_started_at or time.monotonic()
                    # Cold SHM/AU startup can lag the signaling connection.
                    if time.monotonic() - started_at > 15.0:
                        logger.warning("No H.264 access units yet; refusing to answer after 15s")
                        self._pending_create_answer = False
                        try:
                            self.ws.send_webrtc_error("webrtc_gateway_no_frames", gateway=self)
                        except Exception:
                            pass
                        return False

                    return True
                except Exception:
                    logger.debug("Error while waiting for first frame before answering", exc_info=True)
                    return False

            self._schedule_glib_timeout(100, _poll_start_answer)

    @_guard_gateway_callback()
    def _on_answer_created(self, promise: Gst.Promise) -> None:
        """Callback when answer is created."""
        logger.info("    _on_answer_created callback fired")
        promise.wait()
        reply = promise.get_reply()
        if reply is None:
            logger.error("    create-answer returned no reply - FAILED!")
            return

        answer = reply.get_value("answer")
        if answer is None:
            logger.error("    No 'answer' in create-answer reply - FAILED!")
            return

        sdp_text = answer.sdp.as_text()
        sdp_lines = len(sdp_text.split('\n')) if sdp_text else 0
        
        # Log SDP summary to debug video inclusion
        has_video = "m=video" in sdp_text
        has_audio = "m=audio" in sdp_text
        direction = self._find_video_direction(sdp_text)
        pt = self._find_h264_payload_type(sdp_text)
        logger.info("<<< Answer SDP: %d lines, video=%s, audio=%s", sdp_lines, has_video, has_audio)
        logger.info(
            "    Answer video_direction=%s h264_pt=%s packetization_mode_1=%s",
            direction,
            pt,
            "packetization-mode=1" in (sdp_text or ""),
        )
        # Keep RTP caps aligned with the answer we are about to send. Offer-side
        # alignment usually wins, but this catches webrtcbin choosing a different
        # H264 PT during answer creation.
        if pt is not None and self.pay is not None:
            try:
                self.pay.set_property("pt", int(pt))
                pay_src = self.pay.get_static_pad("src")
                if pay_src is not None:
                    pay_src.send_event(Gst.Event.new_reconfigure())
                logger.info("Configured rtph264pay payload type from answer: pt=%d", int(pt))
            except Exception as exc:
                logger.warning("Failed to set rtph264pay pt=%d from answer: %s", int(pt), exc)

        # Hard guardrail: never send an answer that disables the video m= section.
        # This is the "ICE connected but no video" failure mode in browsers.
        if has_video and direction == "inactive":
            logger.error("Refusing to send SDP answer with video_direction=inactive")
            self._record_transceivers(stage="inactive-answer", location="mosaic_webrtc_gateway.py:_on_answer_created")
            try:
                self.ws.send_webrtc_error("webrtc_gateway_inactive_answer", gateway=self)
            except Exception:
                pass
            try:
                def _rebuild(_: object) -> bool:
                    try:
                        self._peer_count = 0
                        self._rebuild_pipeline_for_new_peer()
                    except Exception:
                        logger.debug("Failed to rebuild pipeline after inactive answer", exc_info=True)
                    return False

                self._invoke_glib(_rebuild)
            except Exception:
                pass
            return
        
        logger.info("    Answer created successfully, setting local description...")

        def _on_local_description_set(p: Gst.Promise) -> None:
            try:
                p.wait()
            except Exception:
                pass
            if not self._accepts_async_work():
                return
            logger.info("    Local description set")
            self.ws.send_webrtc_answer(sdp_text, gateway=self)

        promise2 = self._new_guarded_promise(_on_local_description_set)
        self.webrtc.emit("set-local-description", answer, promise2)

    def accept_ice(self, candidate: str, sdp_mline_index: int) -> None:
        """Handle incoming ICE candidate from browser client."""
        if not self._accepts_async_work():
            logger.debug("Ignoring ICE candidate during gateway shutdown")
            return
        logger.info(">>> Gateway accept_ice called (mline=%d)", sdp_mline_index)
        if self.webrtc is None:
            logger.warning("Cannot accept ICE: webrtcbin not initialized")
            return

        def _do_ice(_: object) -> bool:
            try:
                self._accept_ice_impl(candidate, sdp_mline_index)
            except Exception:
                logger.exception("Error adding ICE candidate")
            return False

        self._invoke_glib(_do_ice)

    def _accept_ice_impl(self, candidate: str, sdp_mline_index: int) -> None:
        if not self._accepts_async_work():
            return
        if self.webrtc is None or not self._remote_description_set:
            # Queue until we have a webrtcbin and remote description.
            try:
                self._pending_ice.append((str(candidate), int(sdp_mline_index)))
            except Exception:
                pass
            return
        try:
            self.webrtc.emit("add-ice-candidate", int(sdp_mline_index), candidate)
            logger.info("    ICE candidate added successfully")
        except Exception as e:
            logger.warning("Error adding ICE candidate: %s", e)

    # ========== GStreamer signal handlers ==========

    @_guard_gateway_callback()
    def _on_ice_candidate(
        self, element: Gst.Element, mline_index: int, candidate: str
    ) -> None:
        """Called when webrtcbin has a local ICE candidate to send to peer."""
        if element is not self.webrtc:
            return
        logger.info(
            "<<< Sending local ICE candidate (mline=%d bytes=%d)",
            mline_index,
            len(candidate) if candidate else 0,
        )
        try:
            self.ws.send_webrtc_ice(mline_index, candidate, gateway=self)
        except Exception as e:
            logger.warning("Failed to send ICE candidate: %s", e)

    @_guard_gateway_callback()
    def _on_negotiation_needed(self, element: Gst.Element) -> None:
        """Called when webrtcbin needs negotiation."""
        # In our use case, the browser sends the offer, so we wait for that
        logger.debug("webrtcbin: on-negotiation-needed (waiting for browser offer)")

    # ========== Bus message handlers ==========

    @_guard_gateway_callback()
    def _on_bus_error(self, bus: Gst.Bus, message: Gst.Message) -> None:
        """Handle pipeline errors."""
        err, debug = message.parse_error()
        logger.error("!!! WebRTC gateway pipeline ERROR: %s", err)
        logger.error("    Debug info: %s", debug)
        self._report_terminal_pipeline_failure("webrtc_gateway_pipeline_error")

    @_guard_gateway_callback()
    def _on_bus_eos(self, bus: Gst.Bus, message: Gst.Message) -> None:
        """Handle end-of-stream."""
        logger.warning("WebRTC gateway pipeline received EOS")
        self._report_terminal_pipeline_failure("webrtc_gateway_unexpected_eos")

    def _report_terminal_pipeline_failure(self, reason: str) -> None:
        with self._lifecycle_condition:
            if self._stopping or self._terminal_failure_reported:
                return
            self._terminal_failure_reported = True
        try:
            self.ws.send_webrtc_error(reason, gateway=self)
        except Exception:
            logger.debug("Failed to notify WebRTC owner of pipeline failure", exc_info=True)
        self.ws.report_webrtc_gateway_failure(self, reason)

    @_guard_gateway_callback()
    def _on_bus_state_changed(self, bus: Gst.Bus, message: Gst.Message) -> None:
        """Handle state changes."""
        if message.src != self.pipeline:
            return
        old, new, pending = message.parse_state_changed()
        logger.debug(
            "WebRTC gateway state: %s -> %s",
            Gst.Element.state_get_name(old),
            Gst.Element.state_get_name(new),
        )
