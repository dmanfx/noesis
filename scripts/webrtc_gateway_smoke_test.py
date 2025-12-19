#!/usr/bin/env python3
"""
Headless WebRTC smoke test for the Noesis RTSP→WebRTC gateway.

This connects to the existing WebSocket signaling server, acts as a WebRTC
offerer (recvonly video), and validates that we receive RTP and decoded frames.

Usage:
  python3 scripts/webrtc_gateway_smoke_test.py --ws ws://127.0.0.1:6008 --duration 6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstSdp", "1.0")
gi.require_version("GstWebRTC", "1.0")

from gi.repository import GLib, Gst, GstSdp, GstWebRTC  # noqa: E402

import websockets  # noqa: E402


Gst.init(None)


def _sdp_video_direction(sdp: str) -> Optional[str]:
    in_video = False
    for line in (sdp or "").splitlines():
        line = line.strip()
        if line.startswith("m="):
            in_video = line.startswith("m=video ")
            continue
        if not in_video:
            continue
        if line in ("a=sendonly", "a=recvonly", "a=sendrecv", "a=inactive"):
            return line.split("=", 1)[1]
    return None


@dataclass
class WebRTCStats:
    ice_state: str = "unknown"
    peer_state: str = "unknown"
    rtp_packets: int = 0
    decoded_frames: int = 0


class GstWebRTCRecvClient:
    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        outgoing: "asyncio.Queue[dict[str, Any]]",
        stun_server: Optional[str],
        h264_pt: int,
    ) -> None:
        self._loop = loop
        self._outgoing = outgoing
        self._stun_server = stun_server

        self._stats = WebRTCStats()
        self._started = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._ctx = GLib.MainContext.default()
        self.pipeline = Gst.Pipeline.new("noesis-webrtc-smoke-client")
        self.webrtc = Gst.ElementFactory.make("webrtcbin", "webrtc")
        if self.webrtc is None:
            raise RuntimeError("Failed to create webrtcbin")

        self.webrtc.set_property("bundle-policy", 3)  # max-bundle
        if self._stun_server:
            self.webrtc.set_property("stun-server", self._stun_server)

        self.pipeline.add(self.webrtc)

        # Match the FE: recvonly video transceiver (H264).
        # Note: GStreamer webrtcbin (1.24.x) will not include an H264 m=video line
        # in offers unless payload (and usually packetization-mode) are present.
        # This is only for SDP/codec advertisement; the server will answer with the
        # same PT and configure its rtph264pay accordingly.
        if not (0 <= int(h264_pt) <= 127):
            raise ValueError(f"Invalid H264 payload type: {h264_pt}")
        self._h264_caps = Gst.Caps.from_string(
            f"application/x-rtp,media=video,encoding-name=H264,clock-rate=90000,payload={int(h264_pt)},packetization-mode=(string)1"
        )
        self._recv_transceiver = self.webrtc.emit(
            "add-transceiver",
            GstWebRTC.WebRTCRTPTransceiverDirection.RECVONLY,
            self._h264_caps,
        )
        try:
            self._recv_transceiver.set_property("codec-preferences", self._h264_caps)
        except Exception:
            pass

        self.webrtc.connect("on-negotiation-needed", self._on_negotiation_needed)
        self.webrtc.connect("on-ice-candidate", self._on_ice_candidate)
        self.webrtc.connect("pad-added", self._on_pad_added)
        self.webrtc.connect("notify::ice-connection-state", self._on_ice_state)
        self.webrtc.connect("notify::connection-state", self._on_peer_state)

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self._on_bus_error)

        self._linked_src = False
        self._saw_src_pad = False
        self._src_caps: Optional[str] = None

    def _push_outgoing(self, msg: dict[str, Any]) -> None:
        self._loop.call_soon_threadsafe(self._outgoing.put_nowait, msg)

    def start(self) -> None:
        if self._started:
            return
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Failed to set WebRTC client pipeline to PLAYING")

        self._stop_event.clear()

        def _run() -> None:
            ctx = GLib.MainContext.default()
            while not self._stop_event.is_set():
                try:
                    while ctx.pending():
                        ctx.iteration(False)
                except Exception:
                    pass
                time.sleep(0.01)

        self._thread = threading.Thread(target=_run, name="NoesisWebRTCSmokeClient", daemon=True)
        self._thread.start()
        self._started = True

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self.pipeline.set_state(Gst.State.NULL)
        except Exception:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._started = False

    def stats(self) -> WebRTCStats:
        return self._stats
    
    def saw_src_pad(self) -> bool:
        return self._saw_src_pad

    def src_caps(self) -> Optional[str]:
        return self._src_caps

    def set_remote_answer(self, sdp: str) -> None:
        def _do(_: object) -> bool:
            res, sdpmsg = GstSdp.SDPMessage.new_from_text(sdp or "")
            if res != GstSdp.SDPResult.OK:
                raise RuntimeError(f"Failed to parse SDP answer: {res}")
            answer = GstWebRTC.WebRTCSessionDescription.new(GstWebRTC.WebRTCSDPType.ANSWER, sdpmsg)
            self.webrtc.emit("set-remote-description", answer, Gst.Promise.new())
            return False

        self._ctx.invoke_full(GLib.PRIORITY_DEFAULT, _do, None)

    def add_ice_candidate(self, candidate: str, mline_index: int) -> None:
        def _do(_: object) -> bool:
            try:
                self.webrtc.emit("add-ice-candidate", int(mline_index), str(candidate))
            except Exception:
                pass
            return False

        self._ctx.invoke_full(GLib.PRIORITY_DEFAULT, _do, None)

    # ---- GStreamer callbacks (GLib thread) ----
    def _on_negotiation_needed(self, element: Gst.Element) -> None:
        promise = Gst.Promise.new_with_change_func(self._on_offer_created)
        element.emit("create-offer", None, promise)

    def _on_offer_created(self, promise: Gst.Promise) -> None:
        promise.wait()
        reply = promise.get_reply()
        offer = reply.get_value("offer") if reply is not None else None
        if offer is None:
            self._push_outgoing({"type": "webrtc_error", "error": "client_create_offer_failed"})
            return

        sdp_text = offer.sdp.as_text()

        def _on_local_set(p: Gst.Promise) -> None:
            try:
                p.wait()
            except Exception:
                pass
            self._push_outgoing({"type": "webrtc_offer", "sdp": sdp_text})

        self.webrtc.emit("set-local-description", offer, Gst.Promise.new_with_change_func(_on_local_set))

    def _on_ice_candidate(self, element: Gst.Element, mline_index: int, candidate: str) -> None:
        self._push_outgoing(
            {
                "type": "webrtc_ice_candidate",
                "candidate": str(candidate),
                "sdpMLineIndex": int(mline_index),
            }
        )

    def _on_ice_state(self, webrtc: Gst.Element, pspec: object) -> None:
        state = webrtc.get_property("ice-connection-state")
        try:
            self._stats.ice_state = GstWebRTC.WebRTCICEConnectionState(state).value_nick
        except Exception:
            self._stats.ice_state = str(state)

    def _on_peer_state(self, webrtc: Gst.Element, pspec: object) -> None:
        state = webrtc.get_property("connection-state")
        try:
            self._stats.peer_state = GstWebRTC.WebRTCPeerConnectionState(state).value_nick
        except Exception:
            self._stats.peer_state = str(state)

    def _on_pad_added(self, element: Gst.Element, pad: Gst.Pad) -> None:
        self._saw_src_pad = True
        if self._linked_src:
            return

        caps = pad.get_current_caps() or pad.query_caps(None)
        caps_s = caps.to_string() if caps is not None else ""
        self._src_caps = caps_s
        if "application/x-rtp" not in caps_s or "media=(string)video" not in caps_s:
            return

        self._linked_src = True

        pad.add_probe(Gst.PadProbeType.BUFFER, self._on_rtp_probe, None)

        depay = Gst.ElementFactory.make("rtph264depay", None)
        parse = Gst.ElementFactory.make("h264parse", None)
        dec = Gst.ElementFactory.make("avdec_h264", None)
        sink = Gst.ElementFactory.make("fakesink", None)
        if depay is None or parse is None or dec is None or sink is None:
            raise RuntimeError("Missing GStreamer elements for decode chain (rtph264depay/h264parse/avdec_h264/fakesink)")

        sink.set_property("sync", False)
        sink.set_property("signal-handoffs", True)
        sink.connect("handoff", self._on_decoded_handoff)

        for e in (depay, parse, dec, sink):
            self.pipeline.add(e)
            e.sync_state_with_parent()

        if not depay.link(parse):
            raise RuntimeError("Failed to link depay->parse")
        if not parse.link(dec):
            raise RuntimeError("Failed to link parse->dec")
        if not dec.link(sink):
            raise RuntimeError("Failed to link dec->sink")

        depay_sink = depay.get_static_pad("sink")
        if depay_sink is None:
            raise RuntimeError("Failed to get depay sink pad")
        ret = pad.link(depay_sink)
        if ret != Gst.PadLinkReturn.OK:
            raise RuntimeError(f"Failed to link webrtc src pad -> depay: {ret}")

    def _on_rtp_probe(self, pad: Gst.Pad, info: Gst.PadProbeInfo, user_data: object) -> Gst.PadProbeReturn:
        self._stats.rtp_packets += 1
        return Gst.PadProbeReturn.OK

    def _on_decoded_handoff(self, sink: Gst.Element, buffer: Gst.Buffer, pad: Gst.Pad) -> None:
        self._stats.decoded_frames += 1

    def _on_bus_error(self, bus: Gst.Bus, message: Gst.Message) -> None:
        err, debug = message.parse_error()
        self._push_outgoing({"type": "webrtc_error", "error": f"client_bus_error: {err}", "debug": debug})


async def run_smoke_test(
    *,
    ws_url: str,
    duration_s: float,
    min_rtp: int,
    min_decoded: int,
    stun_server: Optional[str],
    h264_pt: int,
) -> int:
    outgoing: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    client = GstWebRTCRecvClient(loop=loop, outgoing=outgoing, stun_server=stun_server, h264_pt=h264_pt)

    ws: Optional[websockets.WebSocketClientProtocol] = None
    connect_deadline = time.monotonic() + 15.0
    last_exc: Optional[BaseException] = None
    while ws is None and time.monotonic() < connect_deadline:
        try:
            ws = await websockets.connect(ws_url, max_size=None)
        except Exception as exc:
            last_exc = exc
            await asyncio.sleep(0.25)

    if ws is None:
        raise RuntimeError(f"Failed to connect to WS {ws_url}: {last_exc}")

    async with ws:
        answer_seen = asyncio.Event()
        answer_dir: Optional[str] = None

        async def _sender() -> None:
            while True:
                msg = await outgoing.get()
                await ws.send(json.dumps(msg))

        sender_task = asyncio.create_task(_sender(), name="webrtc-smoke-sender")

        try:
            client.start()

            # Wait for answer (or a client-side error) before starting the frame timer.
            answer_deadline = time.monotonic() + 12.0
            while not answer_seen.is_set():
                remaining = answer_deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("Timed out waiting for webrtc_answer from server")

                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=min(0.5, remaining))
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed as exc:
                    raise RuntimeError(f"WebSocket closed while waiting for answer: {exc}") from exc

                if isinstance(raw, (bytes, bytearray)):
                    continue
                msg = json.loads(raw)
                t = msg.get("type")
                if t == "webrtc_answer":
                    sdp = msg.get("sdp", "")
                    answer_dir = _sdp_video_direction(sdp) or "unknown"
                    client.set_remote_answer(sdp)
                    answer_seen.set()
                elif t == "webrtc_ice_candidate":
                    client.add_ice_candidate(msg.get("candidate", ""), int(msg.get("sdpMLineIndex", 0)))
                elif t == "webrtc_error":
                    raise RuntimeError(f"Server webrtc_error: {msg.get('error')}")

            start = time.monotonic()
            while time.monotonic() - start < duration_s:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=0.25)
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed as exc:
                    raise RuntimeError(f"WebSocket closed during media loop: {exc}") from exc
                if isinstance(raw, (bytes, bytearray)):
                    continue
                msg = json.loads(raw)
                t = msg.get("type")
                if t == "webrtc_ice_candidate":
                    client.add_ice_candidate(msg.get("candidate", ""), int(msg.get("sdpMLineIndex", 0)))
                elif t == "webrtc_error":
                    raise RuntimeError(f"Server webrtc_error: {msg.get('error')}")

            st = client.stats()
            print(
                json.dumps(
                    {
                        "ok": st.rtp_packets >= min_rtp and st.decoded_frames >= min_decoded,
                        "answer_video_direction": answer_dir,
                        "ice_state": st.ice_state,
                        "peer_state": st.peer_state,
                        "rtp_packets": st.rtp_packets,
                        "decoded_frames": st.decoded_frames,
                        "saw_src_pad": client.saw_src_pad(),
                        "src_caps": client.src_caps(),
                        "min_rtp": min_rtp,
                        "min_decoded": min_decoded,
                    },
                    indent=2,
                )
            )

            if answer_dir not in ("sendonly", "sendrecv"):
                return 3
            if st.rtp_packets < min_rtp:
                return 4
            if st.decoded_frames < min_decoded:
                return 5
            return 0

        finally:
            client.stop()
            sender_task.cancel()
            try:
                await sender_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", dest="ws_url", default="ws://127.0.0.1:6008")
    ap.add_argument("--duration", dest="duration_s", type=float, default=6.0)
    ap.add_argument("--min-rtp", dest="min_rtp", type=int, default=10)
    ap.add_argument("--min-decoded", dest="min_decoded", type=int, default=1)
    ap.add_argument("--stun", dest="stun_server", default=None)
    ap.add_argument("--pt", dest="h264_pt", type=int, default=96, help="H264 RTP payload type to advertise in offer")
    args = ap.parse_args()
    return asyncio.run(
        run_smoke_test(
            ws_url=args.ws_url,
            duration_s=args.duration_s,
            min_rtp=args.min_rtp,
            min_decoded=args.min_decoded,
            stun_server=args.stun_server,
            h264_pt=args.h264_pt,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
