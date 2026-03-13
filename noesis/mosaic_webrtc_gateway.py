"""
Mosaic WebRTC Gateway

Ultra-light GStreamer WebRTC gateway that consumes DS8's RTSP mosaic output
and exposes it as a WebRTC video track using the existing WebSocketServer
for signaling.

Pipeline: rtspsrc → rtph264depay → h264parse → rtph264pay → webrtcbin

No transcoding. No decode. No GPU load. Passthrough only.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Optional

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstSdp", "1.0")
gi.require_version("GstWebRTC", "1.0")

from gi.repository import GLib, Gst, GstSdp, GstWebRTC

if TYPE_CHECKING:
    from websocket_server import WebSocketServer

Gst.init(None)

logger = logging.getLogger(__name__)


class MosaicWebRTCGateway:
    """
    Ultra-light GStreamer WebRTC gateway that consumes the DS8 RTSP mosaic output
    and exposes it as a WebRTC video track using the existing WebSocketServer
    for signaling.
    """

    def __init__(
        self,
        ws_server: "WebSocketServer",
        rtsp_uri: str,
        request_rtsp_keyframe: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.ws = ws_server
        self.rtsp_uri = rtsp_uri
        self._request_rtsp_keyframe = request_rtsp_keyframe
        self.pipeline: Optional[Gst.Pipeline] = None
        self.webrtc: Optional[Gst.Element] = None
        self.rtspsrc: Optional[Gst.Element] = None
        self.depay: Optional[Gst.Element] = None
        self.parse: Optional[Gst.Element] = None
        self.h264_queue: Optional[Gst.Element] = None
        self.pay: Optional[Gst.Element] = None
        self.queue: Optional[Gst.Element] = None
        self.drain: Optional[Gst.Element] = None
        self.webrtc_sink_pad: Optional[Gst.Pad] = None
        self.webrtc_transceiver: Optional[Any] = None
        self.loop: Optional[GLib.MainLoop] = None
        self.thread: Optional[threading.Thread] = None
        self._started = False
        self._peer_count = 0
        self._frame_count = 0
        self._keyframe_count = 0
        self._offer_h264_pt: Optional[int] = None
        self._remote_description_set = False
        self._pending_create_answer = False
        self._answer_create_started = False
        self._pending_answer_started_at: Optional[float] = None
        self._stop_event = threading.Event()
        self._rtp_in_packets = 0
        self._sender_linked = False
        self._sender_link_in_progress = False
        self._pending_ice: list[tuple[str, int]] = []

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

    def build(self) -> None:
        """Construct the GStreamer pipeline with proper dynamic pad handling."""
        # rtspsrc creates pads dynamically, so we need to build the pipeline manually
        # and connect pads via signals
        
        self.pipeline = Gst.Pipeline.new("webrtc-gateway")
        
        # Create elements
        self.rtspsrc = Gst.ElementFactory.make("rtspsrc", "rtspsrc")
        if self.rtspsrc is None:
            raise RuntimeError("Failed to create rtspsrc element")
        self.rtspsrc.set_property("location", self.rtsp_uri)
        self.rtspsrc.set_property("latency", 200)
        self.rtspsrc.set_property("buffer-mode", 0)  # auto
        try:
            # Prefer TCP for local RTSP to avoid UDP quirks.
            # GstRTSPLowerTrans bitmask: tcp=0x4.
            self.rtspsrc.set_property("protocols", 4)
        except Exception as exc:
            logger.debug("Failed to set rtspsrc protocols=tcp: %s", exc)
        
        self.depay = Gst.ElementFactory.make("rtph264depay", "depay")
        self.parse = Gst.ElementFactory.make("h264parse", "parse")
        if self.parse:
            # Ensure SPS/PPS are re-inserted periodically downstream (helps WebRTC peers
            # that join after the RTSP stream has already started).
            try:
                # -1 = send SPS/PPS with every IDR frame (best for late-join WebRTC peers).
                self.parse.set_property("config-interval", -1)
            except Exception as exc:
                logger.debug("Failed to set h264parse config-interval: %s", exc)
        self.pay = Gst.ElementFactory.make("rtph264pay", "pay")
        if self.pay:
            # Do not hardcode payload type; answer must match the browser offer PT.
            #
            # Important for late-joining peers: repeat SPS/PPS periodically so the
            # browser can start decoding even if it missed the initial IDR/config.
            # -1 = send SPS/PPS with every IDR frame (best for late-join WebRTC peers).
            self.pay.set_property("config-interval", -1)
        # Frame-level queue (H264 access units). This protects RTSP ingest from downstream
        # stalls without dropping individual RTP packets (dropping RTP packets can corrupt
        # keyframes and lead to "bytesReceived but framesDecoded=0" in browsers).
        self.h264_queue = Gst.ElementFactory.make("queue", "h264_queue")
        if self.h264_queue:
            self.h264_queue.set_property("leaky", 2)  # drop oldest when full
            self.h264_queue.set_property("max-size-buffers", 30)
            self.h264_queue.set_property("max-size-bytes", 0)
            self.h264_queue.set_property("max-size-time", 0)

        # RTP-level queue used as the re-link point (drain → webrtcbin). Keep it non-leaky
        # so we don't drop RTP packets mid-frame.
        self.queue = Gst.ElementFactory.make("queue", "webrtc_queue")
        if self.queue:
            self.queue.set_property("leaky", 0)
        
        self.webrtc = Gst.ElementFactory.make("webrtcbin", "webrtc")
        if self.webrtc:
            self.webrtc.set_property("bundle-policy", 3)  # max-bundle

        # Drain sink: keep RTSP ingest running even before a peer offers WebRTC.
        self.drain = Gst.ElementFactory.make("fakesink", "webrtc_drain")
        if self.drain:
            self.drain.set_property("sync", False)
        
        # Verify all elements created
        for name, elem in [
            ("rtspsrc", self.rtspsrc),
            ("depay", self.depay),
            ("parse", self.parse),
            ("h264_queue", self.h264_queue),
            ("pay", self.pay),
            ("queue", self.queue),
            ("webrtc", self.webrtc),
            ("drain", self.drain),
        ]:
            if elem is None:
                raise RuntimeError(f"Failed to create {name} element")
        
        # Add elements to pipeline
        self.pipeline.add(self.rtspsrc)
        self.pipeline.add(self.depay)
        self.pipeline.add(self.parse)
        self.pipeline.add(self.h264_queue)
        self.pipeline.add(self.pay)
        self.pipeline.add(self.queue)
        self.pipeline.add(self.webrtc)
        self.pipeline.add(self.drain)
        
        # Link static elements: depay → parse → h264_queue → pay → webrtc_queue
        if not self.depay.link(self.parse):
            raise RuntimeError("Failed to link depay → parse")
        if not self.parse.link(self.h264_queue):
            raise RuntimeError("Failed to link parse → h264_queue")
        if not self.h264_queue.link(self.pay):
            raise RuntimeError("Failed to link h264_queue → pay")
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
        
        # Add probe to count encoded frames (pre-RTP-payload) to avoid per-RTP-packet overhead
        # which can starve other Python threads (WS server, signal handlers).
        parse_src_pad = self.parse.get_static_pad("src")
        if parse_src_pad is not None:
            parse_src_pad.add_probe(Gst.PadProbeType.BUFFER, self._on_frame_probe, None)
            logger.info("Added frame probe on parse src pad")
        
        # rtspsrc has dynamic pads - connect via pad-added signal
        self.rtspsrc.connect("pad-added", self._on_rtspsrc_pad_added)
        
        # Configure ICE servers (env override).
        stun_server = os.environ.get("NOESIS_MOSAIC_WEBRTC_STUN_SERVER", "").strip()
        if not stun_server:
            stun_server = "stun://stun.l.google.com:19302"
        else:
            # Allow comma-separated list; webrtcbin accepts a single stun-server string.
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
        
        # Connect webrtcbin signals
        self.webrtc.connect("on-ice-candidate", self._on_ice_candidate)
        self.webrtc.connect("on-negotiation-needed", self._on_negotiation_needed)
        self.webrtc.connect("notify::ice-connection-state", self._on_ice_connection_state)
        self.webrtc.connect("notify::connection-state", self._on_connection_state)
        
        # Connect to bus for error/state handling
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self._on_bus_error)
        bus.connect("message::eos", self._on_bus_eos)
        bus.connect("message::state-changed", self._on_bus_state_changed)
        
        # Register gateway with WebSocket server
        self.ws.register_webrtc_gateway(self)
        logger.info("MosaicWebRTCGateway built: %s", self.rtsp_uri)

    def _on_ice_connection_state(self, webrtc: Gst.Element, pspec: object) -> None:
        """Log ICE connection state changes."""
        state = webrtc.get_property("ice-connection-state")
        state_name = GstWebRTC.WebRTCICEConnectionState(state).value_nick if state else "unknown"
        logger.info("!!! ICE connection state: %s", state_name)
        try:
            import json, time

            with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                _f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H4",
                            "location": "mosaic_webrtc_gateway.py:_on_ice_connection_state",
                            "message": "ice state",
                            "data": {"state": state_name},
                            "timestamp": int(time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass

    def _on_connection_state(self, webrtc: Gst.Element, pspec: object) -> None:
        """Log peer connection state changes."""
        state = webrtc.get_property("connection-state")
        state_name = GstWebRTC.WebRTCPeerConnectionState(state).value_nick if state else "unknown"
        logger.info("!!! Peer connection state: %s", state_name)
        if state_name == "connected" and self._request_rtsp_keyframe is not None:
            try:
                self._request_rtsp_keyframe("webrtc_connected")
            except Exception:
                logger.debug("RTSP keyframe request failed on connected", exc_info=True)
        try:
            import json, time

            with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                _f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H4",
                            "location": "mosaic_webrtc_gateway.py:_on_connection_state",
                            "message": "peer state",
                            "data": {"state": state_name},
                            "timestamp": int(time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass

    def _on_rtspsrc_pad_added(self, src: Gst.Element, pad: Gst.Pad) -> None:
        """Handle dynamic pad from rtspsrc."""
        caps = pad.get_current_caps()
        if caps is None:
            caps = pad.query_caps(None)
        
        caps_str = caps.to_string() if caps else "unknown"
        logger.info(">>> rtspsrc pad added: %s (caps: %s...)", pad.get_name(), caps_str[:60])
        
        # rtspsrc can expose multiple pads and caps may be generic early (e.g., only "application/x-rtp").
        # Link the first RTP pad and let depay/parser enforce H264 later.
        media = ""
        encoding = ""
        try:
            if caps is not None and caps.get_size() > 0:
                s = caps.get_structure(0)
                media = (s.get_string("media") or "") if s is not None else ""
                encoding = (s.get_string("encoding-name") or "") if s is not None else ""
        except Exception:
            media = ""
            encoding = ""

        is_rtp = caps_str.startswith("application/x-rtp")
        is_rtp_src = pad.get_name().startswith("recv_rtp_src")

        if is_rtp and is_rtp_src:
            sink_pad = self.depay.get_static_pad("sink")
            if sink_pad and not sink_pad.is_linked():
                ret = pad.link(sink_pad)
                if ret == Gst.PadLinkReturn.OK:
                    logger.info("    Linked rtspsrc pad to depay successfully")
                    try:
                        import json, time

                        with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                            _f.write(
                                json.dumps(
                                    {
                                        "sessionId": "debug-session",
                                        "runId": "run1",
                                        "hypothesisId": "H4",
                                        "location": "mosaic_webrtc_gateway.py:_on_rtspsrc_pad_added",
                                        "message": "rtspsrc pad linked",
                                        "data": {"pad": pad.get_name(), "caps": caps_str},
                                        "timestamp": int(time.time() * 1000),
                                    }
                                )
                                + "\n"
                            )
                    except Exception:
                        pass
                else:
                    logger.error("    Failed to link rtspsrc pad to depay: %s", ret)
            else:
                logger.warning("    depay sink pad already linked or not found")
        else:
            logger.info(
                "    Ignoring pad (name=%s, media=%s, encoding=%s)",
                pad.get_name(),
                media or "?",
                encoding or "?",
            )

    def _on_frame_probe(self, pad: Gst.Pad, info: Gst.PadProbeInfo, user_data: object) -> Gst.PadProbeReturn:
        """Count frames flowing through gateway pipeline."""
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
            #region agent log
            try:
                import json, time  # local import to avoid module-level impact

                with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                    _f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H4",
                                "location": "mosaic_webrtc_gateway.py:_on_frame_probe",
                                "message": "gateway first frame",
                                "data": {"count": self._frame_count, "rtsp_uri": self.rtsp_uri},
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            #endregion
            # If we delayed answering because RTSP hadn't started yet, kick answer creation now.
            if (
                self._pending_create_answer
                and self._remote_description_set
                and not self._answer_create_started
                and self._sender_linked
            ):
                ctx = GLib.MainContext.default()

                def _do(_: object) -> bool:
                    try:
                        self._maybe_start_create_answer(force=False)
                    except Exception:
                        logger.debug("Failed to start answer after first frame", exc_info=True)
                    return False

                try:
                    ctx.invoke_full(GLib.PRIORITY_DEFAULT, _do, None)
                except Exception:
                    _do(None)
        if self._keyframe_count == 1 and is_keyframe:
            logger.info(">>> First keyframe observed in gateway")
            try:
                import json, time  # local import to avoid module-level impact

                with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                    _f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H4",
                                "location": "mosaic_webrtc_gateway.py:_on_frame_probe",
                                "message": "gateway first keyframe",
                                "data": {"frame_count": int(self._frame_count), "rtsp_uri": self.rtsp_uri},
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
        elif self._frame_count % 10 == 0:
            #region agent log
            try:
                import json, time  # local import to avoid module-level impact

                with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                    _f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H4",
                                "location": "mosaic_webrtc_gateway.py:_on_frame_probe",
                                "message": "gateway frame count",
                                "data": {
                                    "count": self._frame_count,
                                    "keyframes": int(self._keyframe_count),
                                    "rtsp_uri": self.rtsp_uri,
                                },
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            #endregion
        elif self._frame_count % 100 == 0:
            logger.info(">>> Gateway video frames: %d", self._frame_count)
            #region agent log
            try:
                import json, time  # local import to avoid module-level impact

                with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                    _f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H4",
                                "location": "mosaic_webrtc_gateway.py:_on_frame_probe",
                                "message": "gateway frame milestone",
                                "data": {"count": self._frame_count, "rtsp_uri": self.rtsp_uri},
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            #endregion
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
        """Persist a concise webrtcbin transceiver inventory snapshot to debug.log."""
        if self.webrtc is None:
            return
        try:
            import json, time

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

            with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                _f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H4",
                            "location": location,
                            "message": "webrtc transceivers",
                            "data": {
                                "stage": stage,
                                "count": n,
                                "transceivers": transceivers_summary,
                                "sink_transceiver": sink_tr_info,
                                "sender_linked": bool(self._sender_linked),
                                "rtp_in_packets": int(self._rtp_in_packets),
                                "frame_count": int(self._frame_count),
                                "offer_h264_pt": int(self._offer_h264_pt) if self._offer_h264_pt is not None else None,
                            },
                            "timestamp": int(time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            logger.debug("Failed to record transceivers (%s)", stage, exc_info=True)

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
        except Exception:
            logger.debug("Failed to ensure webrtc sender pad", exc_info=True)
            return

        queue_src = self.queue.get_static_pad("src")
        if queue_src is None:
            raise RuntimeError("webrtc_queue src pad missing")

        sink_pad = self.webrtc_sink_pad
        if sink_pad is None:
            raise RuntimeError("webrtcbin sink pad missing after ensure")

        # Relink while the src pad is blocked to avoid transient "not-linked" stream errors.
        self._sender_link_in_progress = True

        def _do_relink(pad: Gst.Pad, info: Gst.PadProbeInfo, user_data: object) -> Gst.PadProbeReturn:
            try:
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
                return Gst.PadProbeReturn.REMOVE
            finally:
                self._sender_link_in_progress = False

        try:
            queue_src.add_probe(Gst.PadProbeType.BLOCK | Gst.PadProbeType.BUFFER, _do_relink, None)
        except Exception:
            self._sender_link_in_progress = False
            # Fall back to direct relink (best-effort).
            peer = queue_src.get_peer()
            if peer is not None:
                try:
                    queue_src.unlink(peer)
                except Exception:
                    pass
            ret = queue_src.link(sink_pad)
            if ret != Gst.PadLinkReturn.OK:
                raise RuntimeError(f"Failed to link webrtc_queue → webrtcbin: {ret}")
            self._sender_linked = True
            logger.info("Linked pay → webrtc_queue → webrtcbin sink pad: %s", sink_pad.get_name())

    def _on_rtp_in_probe(self, pad: Gst.Pad, info: Gst.PadProbeInfo, user_data: object) -> Gst.PadProbeReturn:
        self._rtp_in_packets += 1
        if self._rtp_in_packets == 1:
            #region agent log
            try:
                import json, time  # local import to avoid module-level impact

                with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                    _f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H4",
                                "location": "mosaic_webrtc_gateway.py:_on_rtp_in_probe",
                                "message": "gateway first rtp in",
                                "data": {"count": int(self._rtp_in_packets), "rtsp_uri": self.rtsp_uri},
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            #endregion
            # If we delayed answering until RTP is present, kick answer creation now.
            if self._pending_create_answer and self._remote_description_set and not self._answer_create_started:
                ctx = GLib.MainContext.default()

                def _do(_: object) -> bool:
                    try:
                        self._maybe_start_create_answer(force=False)
                    except Exception:
                        logger.debug("Failed to start answer after first RTP packet", exc_info=True)
                    return False

                try:
                    ctx.invoke_full(GLib.PRIORITY_DEFAULT, _do, None)
                except Exception:
                    _do(None)
        elif self._rtp_in_packets % 200 == 0:
            logger.info(">>> Gateway RTP packets into webrtcbin: %d", self._rtp_in_packets)
        return Gst.PadProbeReturn.OK

    def _switch_to_webrtc_sender(self) -> None:
        """
        Ensure the RTP sender is linked into webrtcbin.

        Must be called from the GLib main context (we use invoke_full for entrypoints).
        """
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
            logger.debug("Failed to link sender into webrtcbin", exc_info=True)
            return

        if not self._sender_linked:
            # Retry shortly; payload/caps renegotiation can lag behind pt property updates.
            def _retry() -> bool:
                try:
                    self._switch_to_webrtc_sender()
                except Exception:
                    logger.debug("Sender link retry failed", exc_info=True)
                return not self._sender_linked

            try:
                GLib.timeout_add(50, _retry)
            except Exception:
                pass
            return

        # Linked successfully; record for debugging.
        try:
            import json, time

            with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                _f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H4",
                            "location": "mosaic_webrtc_gateway.py:_switch_to_webrtc_sender",
                            "message": "sender linked",
                            "data": {"pt": int(self._offer_h264_pt) if self._offer_h264_pt is not None else None},
                            "timestamp": int(time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass

    def _maybe_start_create_answer(self, *, force: bool) -> None:
        """Start create-answer if we're ready (or forced). Must run in GLib context."""
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
            if not self._sender_linked:
                self._pending_create_answer = True
                return
        # Wait until we have at least one encoded H264 frame (via h264parse probe).
        # This avoids webrtcbin answering with a=inactive when no sender stream is observed yet.
        if not force and self._frame_count <= 0:
            self._pending_create_answer = True
            return

        self._pending_create_answer = False
        self._answer_create_started = True
        self._pending_answer_started_at = None

        logger.info("    Creating WebRTC answer...")
        self._record_transceivers(stage="before create-answer", location="mosaic_webrtc_gateway.py:_maybe_start_create_answer")
        promise = Gst.Promise.new_with_change_func(self._on_answer_created)
        self.webrtc.emit("create-answer", None, promise)

    def start(self) -> None:
        """Start the gateway pipeline in a background thread."""
        if self._started:
            logger.warning("WebRTC gateway already started")
            return

        if self.pipeline is None:
            # Start in drain mode (keep RTSP ingest negotiated even before any peer offers WebRTC).
            self.build()
        self._stop_event.clear()

        def _run() -> None:
            try:
                ret = self.pipeline.set_state(Gst.State.PLAYING)
                if ret == Gst.StateChangeReturn.FAILURE:
                    logger.error("Failed to set WebRTC gateway pipeline to PLAYING")
                    return
                logger.info("MosaicWebRTCGateway pipeline started")
                #region agent log
                try:
                    import json, time  # local import to avoid module-level impact

                    with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                        _f.write(
                            json.dumps(
                                {
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "H4",
                                    "location": "mosaic_webrtc_gateway.py:start",
                                    "message": "gateway started",
                                    "data": {"rtsp_uri": self.rtsp_uri},
                                    "timestamp": int(time.time() * 1000),
                                }
                            )
                            + "\n"
                        )
                except Exception:
                    pass
                #endregion
                # Drive the default GLib main context without monopolizing the GIL.
                # A blocking GLib.MainLoop.run() in a Python thread can starve other Python
                # threads (WebSocket server, signal handlers), causing "stuck" behavior.
                context = GLib.MainContext.default()
                while not self._stop_event.is_set():
                    try:
                        while context.pending():
                            context.iteration(False)
                    except Exception:
                        # Never crash the gateway thread due to main-context issues.
                        pass
                    time.sleep(0.01)
            except Exception as e:
                logger.exception("WebRTC gateway thread error: %s", e)
            finally:
                logger.info("MosaicWebRTCGateway main loop exited")

        self.thread = threading.Thread(target=_run, daemon=True, name="WebRTCGateway")
        self.thread.start()
        self._started = True
        logger.info("MosaicWebRTCGateway started")

    def stop(self) -> None:
        """Stop the gateway pipeline."""
        if not self._started:
            return

        logger.info("Stopping MosaicWebRTCGateway...")
        self._stop_event.set()

        try:
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
        except Exception as e:
            logger.warning("Error stopping pipeline: %s", e)

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)

        self._started = False
        logger.info("MosaicWebRTCGateway stopped")

    def _rebuild_pipeline_for_new_peer(self) -> None:
        """Tear down and rebuild the gateway pipeline to handle a new PeerConnection cleanly."""
        # Stop the old pipeline (keep the gateway thread alive).
        if self.pipeline is not None:
            try:
                bus = self.pipeline.get_bus()
                if bus is not None:
                    try:
                        bus.remove_signal_watch()
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                self.pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass

        self.pipeline = None
        self.webrtc = None
        self.rtspsrc = None
        self.depay = None
        self.parse = None
        self.h264_queue = None
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

        # Rebuild and start playing.
        self.build()
        if self.pipeline is not None:
            try:
                ret = self.pipeline.set_state(Gst.State.PLAYING)
                if ret == Gst.StateChangeReturn.FAILURE:
                    logger.error("Failed to set rebuilt WebRTC gateway pipeline to PLAYING")
            except Exception:
                logger.exception("Failed to start rebuilt WebRTC gateway pipeline")

    # ========== WebSocket signaling methods (called by WebSocketServer) ==========

    def accept_offer(self, sdp_offer: str) -> None:
        """Handle incoming WebRTC offer from browser client."""
        logger.info(">>> Gateway accept_offer called (offer length=%d)", len(sdp_offer) if sdp_offer else 0)
        if self.webrtc is None:
            logger.error("Cannot accept offer: webrtcbin not initialized")
            return

        # Marshal into the GLib context driving the gateway to avoid cross-thread
        # interaction with GStreamer/webrtcbin.
        ctx = GLib.MainContext.default()

        def _do_offer(_: object) -> bool:
            try:
                self._accept_offer_impl(sdp_offer)
            except Exception:
                logger.exception("Error processing WebRTC offer")
            return False

        try:
            ctx.invoke_full(GLib.PRIORITY_DEFAULT, _do_offer, None)
        except Exception:
            # Fallback to direct call (best-effort).
            _do_offer(None)

    def _accept_offer_impl(self, sdp_offer: str) -> None:
        if self.webrtc is None:
            logger.error("Cannot accept offer: webrtcbin not initialized")
            return

        if self._request_rtsp_keyframe is not None:
            try:
                self._request_rtsp_keyframe("webrtc_offer")
            except Exception:
                logger.debug("RTSP keyframe request failed on offer", exc_info=True)

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

        # Log offer direction + H264 fmtp summary for diagnosis.
        try:
            import json, time

            offer_dir = None
            in_video = False
            for line in (sdp_offer or "").splitlines():
                line = line.strip()
                if line.startswith("m="):
                    in_video = line.startswith("m=video ")
                    continue
                if not in_video:
                    continue
                if line in ("a=sendonly", "a=recvonly", "a=sendrecv", "a=inactive"):
                    offer_dir = line.split("=", 1)[1]
                    break

            offer_pts: list[dict[str, Any]] = []
            if sdp_offer:
                lines = [ln.strip() for ln in sdp_offer.splitlines() if ln.strip()]
                video_mline = next((ln for ln in lines if ln.startswith("m=video ")), None)
                if video_mline:
                    parts = video_mline.split()
                    pts: list[int] = []
                    for token in parts[3:]:
                        try:
                            pts.append(int(token))
                        except Exception:
                            continue

                    rtpmap: dict[int, str] = {}
                    fmtp: dict[int, str] = {}
                    for ln in lines:
                        if ln.lower().startswith("a=rtpmap:"):
                            try:
                                prefix, mapping = ln.split(None, 1)
                                pt = int(prefix.split(":", 1)[1])
                                rtpmap[pt] = mapping.strip()
                            except Exception:
                                pass
                        if ln.lower().startswith("a=fmtp:"):
                            try:
                                body = ln.split(":", 1)[1]
                                pt_str, params = body.split(None, 1)
                                pt = int(pt_str)
                                fmtp[pt] = params.strip()
                            except Exception:
                                pass

                    for pt in pts:
                        m = rtpmap.get(pt, "")
                        if m.lower().startswith("h264/"):
                            offer_pts.append({"pt": pt, "rtpmap": m, "fmtp": fmtp.get(pt, "")})

            with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                _f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H4",
                            "location": "mosaic_webrtc_gateway.py:accept_offer",
                            "message": "offer summary",
                            "data": {
                                "video_direction": offer_dir,
                                "h264_pts": offer_pts,
                            },
                            "timestamp": int(time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            logger.debug("Failed to record offer summary", exc_info=True)

        # Persist full offer SDP for postmortem analysis (browser offers can be large).
        try:
            import json, time

            with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                _f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H4",
                            "location": "mosaic_webrtc_gateway.py:accept_offer",
                            "message": "offer sdp",
                            "data": {"sdp": sdp_offer},
                            "timestamp": int(time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            logger.debug("Failed to record offer sdp", exc_info=True)

        offer_pt = self._extract_h264_payload_type(sdp_offer)
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
            promise = Gst.Promise.new_with_change_func(self._on_set_remote_description_done)
            self.webrtc.emit("set-remote-description", offer, promise)
        except Exception as e:
            logger.exception("Error processing WebRTC offer: %s", e)
            self.ws.send_webrtc_error(str(e), gateway=self)

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
                    if not self._pending_create_answer or self._answer_create_started:
                        return False

                    # Keep trying until the sender is linked and we have frames.
                    # Negotiation/caps updates can lag behind pt updates and pad relinking.
                    self._maybe_start_create_answer(force=False)
                    if self._answer_create_started:
                        return False

                    started_at = self._pending_answer_started_at or time.monotonic()
                    # RTSP startup can be slow on cold start (DESCRIBE/SETUP/PLAY + first keyframe).
                    # Keep waiting long enough to avoid spurious "no frames" errors on the dashboard.
                    if time.monotonic() - started_at > 15.0:
                        logger.warning("No RTSP frames yet; refusing to answer after 15s")
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

            try:
                GLib.timeout_add(100, _poll_start_answer)
            except Exception:
                pass

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
        direction: Optional[str] = None
        pt: Optional[int] = None
        logger.info("<<< Answer SDP: %d lines, video=%s, audio=%s", sdp_lines, has_video, has_audio)
        # Always log full SDP for debugging
        logger.info("    Full SDP:\n%s", sdp_text)

        #region agent log
        try:
            import json, time
            import re

            def _find_video_direction(text: str) -> Optional[str]:
                in_video = False
                for line in (text or "").splitlines():
                    line = line.strip()
                    if line.startswith("m="):
                        in_video = line.startswith("m=video ")
                        continue
                    if not in_video:
                        continue
                    if line in ("a=sendonly", "a=recvonly", "a=sendrecv", "a=inactive"):
                        return line.split("=", 1)[1]
                return None

            def _find_h264_pt(text: str) -> Optional[int]:
                for line in (text or "").splitlines():
                    m = re.match(r"^a=rtpmap:(\d+)\s+H264/", line.strip(), flags=re.IGNORECASE)
                    if m:
                        try:
                            return int(m.group(1))
                        except Exception:
                            return None
                return None

            direction = _find_video_direction(sdp_text)
            pt = _find_h264_pt(sdp_text)
            has_msid = "a=msid:" in (sdp_text or "")
            has_packetization_mode_1 = "packetization-mode=1" in (sdp_text or "")

            # Ensure the RTP payload type we emit matches the negotiated SDP answer.
            # Setting this based on the offer can break negotiation if webrtcbin chooses a different PT.
            if pt is not None and self.pay is not None:
                try:
                    self.pay.set_property("pt", int(pt))
                    logger.info("Configured rtph264pay payload type from answer: pt=%d", int(pt))
                except Exception as exc:
                    logger.warning("Failed to set rtph264pay pt=%d from answer: %s", int(pt), exc)

            with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                _f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H4",
                            "location": "mosaic_webrtc_gateway.py:_on_answer_created",
                            "message": "answer sdp summary",
                            "data": {
                                "lines": sdp_lines,
                                "has_video": bool(has_video),
                                "video_direction": direction,
                                "h264_pt": pt,
                                "has_msid": bool(has_msid),
                                "packetization_mode_1": bool(has_packetization_mode_1),
                            },
                            "timestamp": int(time.time() * 1000),
                        }
                    )
                    + "\n"
                )

            if pt is not None:
                with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                    _f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H4",
                                "location": "mosaic_webrtc_gateway.py:_on_answer_created",
                                "message": "configured pay pt from answer",
                                "data": {"pt": int(pt)},
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                        + "\n"
                    )

            # Also record the full SDP (20-ish lines) for postmortem.
            with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                _f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H4",
                            "location": "mosaic_webrtc_gateway.py:_on_answer_created",
                            "message": "answer sdp",
                            "data": {"sdp": sdp_text},
                            "timestamp": int(time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        #endregion

        # Hard guardrail: never send an answer that disables the video m= section.
        # This is the "ICE connected but no video" failure mode in browsers.
        if has_video and direction == "inactive":
            logger.error("Refusing to send SDP answer with video_direction=inactive")
            self._record_transceivers(stage="inactive-answer", location="mosaic_webrtc_gateway.py:_on_answer_created")
            try:
                import json, time

                with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                    _f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H4",
                                "location": "mosaic_webrtc_gateway.py:_on_answer_created",
                                "message": "refused inactive answer",
                                "data": {"video_direction": direction, "h264_pt": pt, "lines": sdp_lines},
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            try:
                self.ws.send_webrtc_error("webrtc_gateway_inactive_answer", gateway=self)
            except Exception:
                pass
            try:
                ctx = GLib.MainContext.default()

                def _rebuild(_: object) -> bool:
                    try:
                        self._peer_count = 0
                        self._rebuild_pipeline_for_new_peer()
                    except Exception:
                        logger.debug("Failed to rebuild pipeline after inactive answer", exc_info=True)
                    return False

                ctx.invoke_full(GLib.PRIORITY_DEFAULT, _rebuild, None)
            except Exception:
                pass
            return
        
        logger.info("    Answer created successfully, setting local description...")

        def _on_local_description_set(p: Gst.Promise) -> None:
            try:
                p.wait()
            except Exception:
                pass
            logger.info("    Local description set")
            self.ws.send_webrtc_answer(sdp_text, gateway=self)

        promise2 = Gst.Promise.new_with_change_func(_on_local_description_set)
        self.webrtc.emit("set-local-description", answer, promise2)

    def accept_ice(self, candidate: str, sdp_mline_index: int) -> None:
        """Handle incoming ICE candidate from browser client."""
        logger.info(">>> Gateway accept_ice called (mline=%d)", sdp_mline_index)
        if self.webrtc is None:
            logger.warning("Cannot accept ICE: webrtcbin not initialized")
            return

        ctx = GLib.MainContext.default()

        def _do_ice(_: object) -> bool:
            try:
                self._accept_ice_impl(candidate, sdp_mline_index)
            except Exception:
                logger.exception("Error adding ICE candidate")
            return False

        try:
            ctx.invoke_full(GLib.PRIORITY_DEFAULT, _do_ice, None)
        except Exception:
            _do_ice(None)

    def _accept_ice_impl(self, candidate: str, sdp_mline_index: int) -> None:
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
            #region agent log
            try:
                import json, time  # local import to avoid module-level impact

                with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                    _f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H4",
                                "location": "mosaic_webrtc_gateway.py:accept_ice",
                                "message": "accept ice",
                                "data": {"mline": int(sdp_mline_index)},
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            #endregion
        except Exception as e:
            logger.warning("Error adding ICE candidate: %s", e)

    # ========== GStreamer signal handlers ==========

    def _on_ice_candidate(
        self, element: Gst.Element, mline_index: int, candidate: str
    ) -> None:
        """Called when webrtcbin has a local ICE candidate to send to peer."""
        logger.info("<<< Sending local ICE candidate (mline=%d): %s...", mline_index, candidate[:50] if candidate else '')
        try:
            self.ws.send_webrtc_ice(mline_index, candidate, gateway=self)
        except Exception as e:
            logger.warning("Failed to send ICE candidate: %s", e)

    def _on_negotiation_needed(self, element: Gst.Element) -> None:
        """Called when webrtcbin needs negotiation."""
        # In our use case, the browser sends the offer, so we wait for that
        logger.debug("webrtcbin: on-negotiation-needed (waiting for browser offer)")

    # ========== Bus message handlers ==========

    def _on_bus_error(self, bus: Gst.Bus, message: Gst.Message) -> None:
        """Handle pipeline errors."""
        err, debug = message.parse_error()
        logger.error("!!! WebRTC gateway pipeline ERROR: %s", err)
        logger.error("    Debug info: %s", debug)
        #region agent log
        try:
            import json, time  # local import to avoid module-level impact

            with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                _f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H4",
                            "location": "mosaic_webrtc_gateway.py:_on_bus_error",
                            "message": "gateway bus error",
                            "data": {"error": str(err), "debug": debug},
                            "timestamp": int(time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        #endregion

    def _on_bus_eos(self, bus: Gst.Bus, message: Gst.Message) -> None:
        """Handle end-of-stream."""
        logger.warning("WebRTC gateway pipeline received EOS")
        #region agent log
        try:
            import json, time  # local import to avoid module-level impact

            with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                _f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H4",
                            "location": "mosaic_webrtc_gateway.py:_on_bus_eos",
                            "message": "gateway eos",
                            "data": {},
                            "timestamp": int(time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        #endregion

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
