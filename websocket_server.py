from typing import Dict, Optional, Any, Callable, List
import asyncio
import concurrent.futures
from websockets.legacy.server import serve
import json
import logging
import os
from collections import deque
import threading
import time
import uuid

from models import convert_numpy_types


class WebSocketServer:
    """Manages WebSocket server for broadcasting data to clients"""
    
    def __init__(
        self, 
        host: str = "0.0.0.0", 
        port: int = 6008,
        event_loop: Optional[asyncio.AbstractEventLoop] = None,
        stats_callback: Optional[Callable[[], Dict[str, Any]]] = None,
        toggle_callback: Optional[Callable[[str, bool], None]] = None,
        initial_trail_state: bool = True
    ):
        """Initialize the WebSocket server
        
        Args:
            host: Server host address
            port: Server port
            event_loop: Optional asyncio event loop
            stats_callback: Optional callback to get statistics
            toggle_callback: Optional callback to handle toggle updates
            initial_trail_state: The initial state of the trail visualization
        """
        self.host = host
        self.port = port
        self.event_loop = event_loop
        self.stats_callback = stats_callback
        self.toggle_callback = toggle_callback
        self.initial_trail_state = initial_trail_state
        self.connected_clients = set()
        self.server = None
        self.server_task = None
        self.running = True
        self.logger = logging.getLogger("WebSocketServer")
        self._stats_task = None # Added reference for the periodic stats task
        self._last_stats_info_log: float = 0.0
        # Binary frame coalescer state: keep only latest per camera
        self._latest_binary_by_cam: Dict[str, bytes] = {}
        self._binary_flush_task: Optional[asyncio.Task] = None
        self._binary_sending: bool = False
        # Lightweight telemetry for Menon calibration/coordinate RPCs
        self._telemetry: Dict[str, Dict[str, Any]] = {"rx": {}, "tx": {}}
        self._telemetry_task: Optional[asyncio.Task] = None
        # Optional BEV control callbacks
        self.bev_config_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
        self.bev_overlay_callback: Optional[Callable[[str, bool], None]] = None
        # Optional MA heatmap ready callback (called with cameraId)
        self.ma_ready_callback: Optional[Callable[[str], None]] = None
        # RPC guardrails
        self._depth_rpc_tracker: Dict[str, float] = {}
        self._floorplan_rpc_tracker: Dict[str, float] = {}
        self._depth_rate_limit_window = 0.5  # seconds per camera/client
        self._floorplan_rate_limit_window = 2.0
        self._depth_rpc_timeout = 4.0  # seconds
        self._floorplan_rpc_timeout = 6.0
        self._tracker_prune_window = 30.0
        # Optional calibration + RPC callbacks
        self.calibration_getter: Optional[Callable[[], Dict[str, Any]]] = None
        self.pixel_to_world_handler: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
        self.set_extrinsics_handler: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
        self.solve_pnp_handler: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
        self.set_align_handler: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
        self.ma_depth_provider: Optional[Callable[[str, Optional[Any], Optional[str]], Optional[Dict[str, Any]]]] = None
        self.floorplan_provider: Optional[Callable[[Optional[list], float, float, float, bool], Optional[Dict[str, Any]]]] = None
        # Optional auto-calibration handler (cameraId -> result)
        self.auto_calibrate_handler: Optional[Callable[[Optional[str]], Dict[str, Any]]] = None
        # WebRTC signaling: webrtcbin element reference (legacy direct approach)
        self.webrtc_elem: Optional[Any] = None
        # WebRTC gateway reference (new RTSP-based gateway approach)
        self.webrtc_gateway: Optional[Any] = None
        # WebRTC signaling ownership: only the connection that most recently sent a
        # webrtc_offer should receive webrtc_answer / server ICE candidates.
        self._webrtc_owner: Optional[Any] = None
        self._webrtc_owner_ip: Optional[str] = None
        # Boundary serialization metrics (JSON conversion at WS boundary).
        self._boundary_lock = threading.Lock()
        self._boundary_samples_ms = deque(maxlen=4096)
        self._boundary_count = 0
        self._boundary_total_ms = 0.0
        self._boundary_max_ms = 0.0
        self._boundary_last_ms = 0.0
        self._boundary_total_bytes = 0
        self._boundary_last_bytes = 0
        self._boundary_budget_ms = 3.0
        self._boundary_violations = 0
        self._boundary_route_metrics: Dict[str, Dict[str, Any]] = {}
        self._boundary_stage_metrics: Dict[str, Dict[str, Any]] = {}

    def _new_boundary_bucket(self, maxlen: int = 1024) -> Dict[str, Any]:
        return {
            "samples": deque(maxlen=maxlen),
            "count": 0,
            "total_ms": 0.0,
            "max_ms": 0.0,
            "last_ms": 0.0,
            "total_bytes": 0,
            "last_bytes": 0,
        }

    def _update_boundary_bucket(self, bucket: Dict[str, Any], duration_ms: float, payload_bytes: int) -> None:
        bucket["samples"].append(duration_ms)
        bucket["count"] = int(bucket.get("count", 0)) + 1
        bucket["total_ms"] = float(bucket.get("total_ms", 0.0)) + float(duration_ms)
        bucket["max_ms"] = max(float(bucket.get("max_ms", 0.0)), float(duration_ms))
        bucket["last_ms"] = float(duration_ms)
        bucket["total_bytes"] = int(bucket.get("total_bytes", 0)) + int(payload_bytes)
        bucket["last_bytes"] = int(payload_bytes)

    def _record_boundary_serialization_stage(
        self,
        duration_ms: float,
        payload_bytes: int,
        *,
        channel: str = "ws",
        route: str = "broadcast",
        message_type: str = "unknown",
        stage: str = "total",
        outcome: str = "ok",
        include_budget: bool = True,
    ) -> None:
        with self._boundary_lock:
            d = max(0.0, float(duration_ms))
            b = max(0, int(payload_bytes))
            if include_budget and stage == "total":
                self._boundary_samples_ms.append(d)
                self._boundary_count += 1
                self._boundary_total_ms += d
                self._boundary_max_ms = max(self._boundary_max_ms, d)
                self._boundary_last_ms = d
                self._boundary_total_bytes += b
                self._boundary_last_bytes = b
                if d > float(self._boundary_budget_ms):
                    self._boundary_violations += 1

            route_key = f"{channel}|{route}|{message_type}|{outcome}"
            route_bucket = self._boundary_route_metrics.get(route_key)
            if route_bucket is None:
                route_bucket = self._new_boundary_bucket()
                self._boundary_route_metrics[route_key] = route_bucket
            self._update_boundary_bucket(route_bucket, d, b)

            stage_key = f"{channel}|{route}|{message_type}|{stage}|{outcome}"
            stage_bucket = self._boundary_stage_metrics.get(stage_key)
            if stage_bucket is None:
                stage_bucket = self._new_boundary_bucket()
                self._boundary_stage_metrics[stage_key] = stage_bucket
            self._update_boundary_bucket(stage_bucket, d, b)

    def _record_boundary_serialization(self, duration_ms: float, payload_bytes: int) -> None:
        self._record_boundary_serialization_stage(
            duration_ms,
            payload_bytes,
            channel="ws",
            route="legacy",
            message_type="legacy",
            stage="total",
            outcome="ok",
            include_budget=True,
        )

    @staticmethod
    def _percentile(values: List[float], q: float) -> Optional[float]:
        if not values:
            return None
        q = max(0.0, min(1.0, float(q)))
        ordered = sorted(values)
        if len(ordered) == 1:
            return float(ordered[0])
        idx = int(round(q * (len(ordered) - 1)))
        idx = max(0, min(len(ordered) - 1, idx))
        return float(ordered[idx])

    def get_boundary_serialization_metrics(self) -> Dict[str, Any]:
        with self._boundary_lock:
            samples = list(self._boundary_samples_ms)
            count = int(self._boundary_count)
            total_ms = float(self._boundary_total_ms)
            max_ms = float(self._boundary_max_ms)
            last_ms = float(self._boundary_last_ms)
            total_bytes = int(self._boundary_total_bytes)
            last_bytes = int(self._boundary_last_bytes)
            violations = int(self._boundary_violations)
            route_snapshot = {
                str(k): {
                    "samples": list(v.get("samples", [])),
                    "count": int(v.get("count", 0)),
                    "total_ms": float(v.get("total_ms", 0.0)),
                    "max_ms": float(v.get("max_ms", 0.0)),
                    "last_ms": float(v.get("last_ms", 0.0)),
                    "total_bytes": int(v.get("total_bytes", 0)),
                    "last_bytes": int(v.get("last_bytes", 0)),
                }
                for k, v in self._boundary_route_metrics.items()
            }
            stage_snapshot = {
                str(k): {
                    "samples": list(v.get("samples", [])),
                    "count": int(v.get("count", 0)),
                    "total_ms": float(v.get("total_ms", 0.0)),
                    "max_ms": float(v.get("max_ms", 0.0)),
                    "last_ms": float(v.get("last_ms", 0.0)),
                    "total_bytes": int(v.get("total_bytes", 0)),
                    "last_bytes": int(v.get("last_bytes", 0)),
                }
                for k, v in self._boundary_stage_metrics.items()
            }
        avg_ms = (total_ms / float(count)) if count > 0 else None
        def _summarize(snapshot: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
            out: Dict[str, Dict[str, Any]] = {}
            for key, item in snapshot.items():
                values = [float(x) for x in item.get("samples", [])]
                icount = int(item.get("count", 0))
                itotal = float(item.get("total_ms", 0.0))
                out[key] = {
                    "count": icount,
                    "avg_ms": (itotal / float(icount)) if icount > 0 else None,
                    "p50_ms": self._percentile(values, 0.50),
                    "p95_ms": self._percentile(values, 0.95),
                    "p99_ms": self._percentile(values, 0.99),
                    "max_ms": float(item.get("max_ms", 0.0)) if icount > 0 else None,
                    "last_ms": float(item.get("last_ms", 0.0)) if icount > 0 else None,
                    "total_bytes": int(item.get("total_bytes", 0)),
                    "last_payload_bytes": int(item.get("last_bytes", 0)),
                }
            return out
        return {
            "count": count,
            "avg_ms": avg_ms,
            "p50_ms": self._percentile(samples, 0.50),
            "p95_ms": self._percentile(samples, 0.95),
            "p99_ms": self._percentile(samples, 0.99),
            "max_ms": max_ms if count > 0 else None,
            "last_ms": last_ms if count > 0 else None,
            "total_bytes": total_bytes,
            "last_payload_bytes": last_bytes,
            "budget_ms": float(self._boundary_budget_ms),
            "violations": violations,
            "routes": _summarize(route_snapshot),
            "stages": _summarize(stage_snapshot),
        }

    def reset_boundary_serialization_metrics(self) -> None:
        with self._boundary_lock:
            self._boundary_samples_ms.clear()
            self._boundary_count = 0
            self._boundary_total_ms = 0.0
            self._boundary_max_ms = 0.0
            self._boundary_last_ms = 0.0
            self._boundary_total_bytes = 0
            self._boundary_last_bytes = 0
            self._boundary_violations = 0
            self._boundary_route_metrics.clear()
            self._boundary_stage_metrics.clear()

    async def _send_json_with_boundary_metrics(
        self,
        websocket: Any,
        payload: Dict[str, Any],
        *,
        route: str,
        message_type: str = "unknown",
        use_to_thread_json: bool = False,
    ) -> None:
        convert_start_ns = time.perf_counter_ns()
        safe_payload = convert_numpy_types(payload)
        convert_ms = (time.perf_counter_ns() - convert_start_ns) / 1_000_000.0
        self._record_boundary_serialization_stage(
            convert_ms,
            0,
            channel="ws",
            route=route,
            message_type=message_type,
            stage="numpy_convert",
            outcome="ok",
            include_budget=False,
        )

        encode_start_ns = time.perf_counter_ns()
        if use_to_thread_json:
            message_text = await asyncio.to_thread(json.dumps, safe_payload, separators=(",", ":"))
        else:
            message_text = json.dumps(safe_payload, separators=(",", ":"))
        payload_bytes = len(message_text.encode("utf-8"))
        encode_ms = (time.perf_counter_ns() - encode_start_ns) / 1_000_000.0
        self._record_boundary_serialization_stage(
            encode_ms,
            payload_bytes,
            channel="ws",
            route=route,
            message_type=message_type,
            stage="json_encode",
            outcome="ok",
            include_budget=False,
        )

        send_start_ns = time.perf_counter_ns()
        await websocket.send(message_text)
        send_ms = (time.perf_counter_ns() - send_start_ns) / 1_000_000.0
        self._record_boundary_serialization_stage(
            send_ms,
            payload_bytes,
            channel="ws",
            route=route,
            message_type=message_type,
            stage="send_dispatch",
            outcome="ok",
            include_budget=False,
        )

        # Keep gate metric aligned with historical budget semantics:
        # conversion + JSON encoding only. Send dispatch is reported separately.
        total_ms = float(convert_ms + encode_ms)
        self._record_boundary_serialization_stage(
            total_ms,
            payload_bytes,
            channel="ws",
            route=route,
            message_type=message_type,
            stage="total",
            outcome="ok",
            include_budget=True,
        )

    def _get_webrtc_owner(self) -> Optional[Any]:
        owner = self._webrtc_owner
        if owner is None:
            return None
        if owner in self.connected_clients:
            return owner
        # Owner disconnected; clear to avoid sending signaling to stale sockets.
        self._webrtc_owner = None
        self._webrtc_owner_ip = None
        return None

    async def _send_to_client(self, websocket, message) -> None:
        """Send a message to one client, mirroring broadcast() semantics."""
        if not websocket or websocket not in self.connected_clients:
            return

        try:
            if isinstance(message, dict):
                message_type = str(message.get("type", "unknown"))
                try:
                    self._record_tx(message)
                except Exception:
                    pass
                await self._send_json_with_boundary_metrics(
                    websocket,
                    message,
                    route="send_to_client",
                    message_type=message_type,
                )
            elif isinstance(message, str):
                await websocket.send(message)
            elif isinstance(message, bytes):
                await websocket.send(message)
            else:
                self.logger.warning("Unknown message type for send_to_client: %s", type(message))
        except Exception as exc:
            client_ip = websocket.remote_address if hasattr(websocket, 'remote_address') else "Unknown"
            self.logger.debug("Failed to send message to %s: %s", client_ip, exc)
            try:
                self.connected_clients.discard(websocket)
            except Exception:
                pass

    def send_to_client_sync(self, websocket, message) -> None:
        """Thread-safe one-client send for use from other threads."""
        if not self.event_loop:
            self.logger.error("No event loop available for send_to_client_sync")
            return
        if not self.running:
            return
        try:
            if self.event_loop.is_closed():
                self.logger.warning("Event loop is closed; dropping send_to_client")
                return
        except Exception:
            pass

        asyncio.run_coroutine_threadsafe(
            self._send_to_client(websocket, message),
            self.event_loop
        )

    # ---------------- Menon telemetry helpers ----------------
    def _telemetry_now(self) -> float:
        try:
            return time.time()
        except Exception:
            return 0.0

    def _short_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Produce a compact view of calibration/coordinate payloads for logging."""
        try:
            t = d.get('type') if isinstance(d, dict) else None
            out: Dict[str, Any] = {'type': t}
            if t == 'set_align':
                al = d.get('align', {}) if isinstance(d, dict) else {}
                mx = al.get('matrix') if isinstance(al, dict) else None
                out.update({
                    'floor_y': al.get('floor_y'),
                    's': (al.get('units') or {}).get('s_obj_to_m') if isinstance(al.get('units'), dict) else None,
                    'matrix': f"len={len(mx)}" if isinstance(mx, list) else None
                })
            elif t == 'set_extrinsics':
                out.update({
                    'cameraId': d.get('cameraId'),
                    'E': f"len={len(d.get('E'))}" if isinstance(d.get('E'), list) else None,
                    'Twc': f"len={len(d.get('Twc'))}" if isinstance(d.get('Twc'), list) else None,
                })
            elif t == 'solve_pnp':
                pts2 = d.get('points2D') if isinstance(d.get('points2D'), list) else None
                pts3 = d.get('points3D') if isinstance(d.get('points3D'), list) else None
                out.update({'cameraId': d.get('cameraId'), 'points2D': len(pts2) if pts2 else 0, 'points3D': len(pts3) if pts3 else 0})
            elif t == 'pixel_to_world':
                out.update({'camId': d.get('camId') or d.get('cameraId'), 'u': d.get('u'), 'v': d.get('v'), 'request_id': d.get('request_id') or d.get('reqId')})
            elif t == 'pixel_to_world_response':
                out.update({'ok': d.get('ok'), 'world': d.get('world'), 'request_id': d.get('request_id') or d.get('reqId')})
            elif t == 'set_extrinsics_result' or t == 'set_align_result' or t == 'solve_pnp_result':
                out.update({'ok': d.get('ok'), 'error': d.get('error')})
            elif t == 'calibration-bundle':
                data = d.get('data') if isinstance(d.get('data'), dict) else {}
                cams = data.get('cameras') if isinstance(data.get('cameras'), dict) else {}
                al = data.get('align') if isinstance(data.get('align'), dict) else {}
                out.update({'cameras': len(cams), 'floor_y': al.get('floor_y'), 's': (al.get('units') or {}).get('s_obj_to_m') if isinstance(al.get('units'), dict) else None})
            else:
                # Default: include a shallow subset
                for k in ('cameraId', 'camId', 'u', 'v', 'ok'):
                    if k in d:
                        out[k] = d.get(k)
            return out
        except Exception:
            return {'type': d.get('type') if isinstance(d, dict) else None}

    def _record_rx(self, msg: Dict[str, Any]) -> None:
        try:
            t = msg.get('type') if isinstance(msg, dict) else None
            if t in {'set_align', 'set_extrinsics', 'solve_pnp', 'pixel_to_world'}:
                self._telemetry['rx'][t] = {'t': self._telemetry_now(), 'data': self._short_dict(msg)}
        except Exception:
            pass

    def _record_tx(self, msg: Dict[str, Any]) -> None:
        try:
            t = msg.get('type') if isinstance(msg, dict) else None
            if t in {'calibration-bundle', 'set_align_result', 'set_extrinsics_result', 'solve_pnp_result', 'pixel_to_world_response'}:
                self._telemetry['tx'][t] = {'t': self._telemetry_now(), 'data': self._short_dict(msg)}
        except Exception:
            pass

    # ---------------- WebRTC signaling helpers ----------------
    def attach_webrtc_endpoint(self, webrtc_elem: Any) -> None:
        """Register webrtcbin element for signaling and attach GStreamer callbacks."""
        self.webrtc_elem = webrtc_elem
        self.logger.info("WebRTC endpoint attached to WebSocketServer")

        # Try to connect GStreamer signal handlers for negotiation and ICE candidates
        try:
            # on-negotiation-needed: webrtcbin requests offer creation
            webrtc_elem.connect("on-negotiation-needed", self._on_webrtc_negotiation_needed)
            # on-ice-candidate: webrtcbin has a local ICE candidate to send
            webrtc_elem.connect("on-ice-candidate", self._on_webrtc_ice_candidate)
            self.logger.info("Connected GStreamer webrtcbin signals")
        except Exception as exc:
            self.logger.warning("Failed to connect webrtcbin signals: %s", exc)

    def _on_webrtc_negotiation_needed(self, *args) -> None:
        """Called when webrtcbin needs to create/send an offer."""
        self.logger.debug("webrtcbin: on-negotiation-needed")
        # In most server-as-sender scenarios, we wait for browser to send offer
        # This callback is here for future use if server initiates

    def _on_webrtc_ice_candidate(self, webrtc, mline_index: int, candidate: str) -> None:
        """Called when webrtcbin has a local ICE candidate to send to peer."""
        try:
            msg = {
                'type': 'webrtc_ice_candidate',
                'candidate': candidate,
                'sdpMLineIndex': mline_index,
            }
            owner = self._get_webrtc_owner()
            if owner is not None:
                self.send_to_client_sync(owner, msg)
            else:
                self.broadcast_sync(msg)
            self.logger.debug("Sent ICE candidate to clients: mline=%d", mline_index)
        except Exception as exc:
            self.logger.warning("Failed to send ICE candidate: %s", exc)

    async def _handle_webrtc_offer(self, websocket, data: Dict[str, Any]) -> None:
        """Handle incoming WebRTC offer from browser client."""
        if self.webrtc_elem is None:
            await websocket.send(json.dumps({'type': 'webrtc_error', 'error': 'no_webrtc_element'}))
            return

        sdp = data.get('sdp')
        if not sdp:
            await websocket.send(json.dumps({'type': 'webrtc_error', 'error': 'missing_sdp'}))
            return

        try:
            from gi.repository import Gst, GstSdp, GstWebRTC

            # Parse and set remote description
            res, sdpmsg = GstSdp.SDPMessage.new_from_text(sdp)
            if res != GstSdp.SDPResult.OK:
                raise ValueError("Failed to parse SDP")

            offer = GstWebRTC.WebRTCSessionDescription.new(
                GstWebRTC.WebRTCSDPType.OFFER, sdpmsg
            )
            self.webrtc_elem.emit("set-remote-description", offer, None)

            # Create answer
            promise = Gst.Promise.new()
            self.webrtc_elem.emit("create-answer", None, promise)
            promise.wait()
            reply = promise.get_reply()
            answer = reply.get_value("answer")
            if answer is None:
                raise ValueError("Failed to create answer")

            self.webrtc_elem.emit("set-local-description", answer, None)

            # Send answer to client
            response = {
                'type': 'webrtc_answer',
                'sdp': answer.sdp.as_text(),
            }
            await websocket.send(json.dumps(response))
            self.logger.info("Sent WebRTC answer to client")

        except Exception as exc:
            self.logger.error("WebRTC offer handling failed: %s", exc)
            await websocket.send(json.dumps({'type': 'webrtc_error', 'error': str(exc)}))

    async def _handle_webrtc_ice_candidate(self, websocket, data: Dict[str, Any]) -> None:
        """Handle incoming ICE candidate from browser client."""
        if self.webrtc_elem is None:
            return

        candidate = data.get('candidate')
        sdp_mline_index = data.get('sdpMLineIndex', 0)

        if candidate:
            try:
                self.webrtc_elem.emit("add-ice-candidate", sdp_mline_index, candidate)
                self.logger.debug("Added ICE candidate from client: mline=%d", sdp_mline_index)
            except Exception as exc:
                self.logger.warning("Failed to add ICE candidate: %s", exc)

    # ---------------- WebRTC Gateway methods (new RTSP-based approach) ----------------

    def register_webrtc_gateway(self, gateway: Any) -> None:
        """Register the MosaicWebRTCGateway for signaling."""
        self.webrtc_gateway = gateway
        self.logger.info("WebRTC gateway registered with WebSocketServer")

    def send_webrtc_answer(self, sdp: str) -> None:
        """Send WebRTC answer SDP to the owning client (fallback: broadcast)."""
        msg = {"type": "webrtc_answer", "sdp": sdp}
        owner = self._get_webrtc_owner()
        if owner is not None:
            self.logger.info("<<< Sending webrtc_answer to WebRTC owner %s", self._webrtc_owner_ip or "unknown")
        else:
            self.logger.info("<<< Sending webrtc_answer to %d connected clients", len(self.connected_clients))
        try:
            import json as _json, time as _time

            with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                _f.write(
                    _json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H4",
                            "location": "websocket_server.py:send_webrtc_answer",
                            "message": "send webrtc_answer",
                            "data": {"clients": len(self.connected_clients), "sdp_lines": len((sdp or '').splitlines())},
                            "timestamp": int(_time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        if owner is not None:
            self.send_to_client_sync(owner, msg)
        else:
            self.broadcast_sync(msg)

    def send_webrtc_ice(self, mline_index: int, candidate: str) -> None:
        """Send WebRTC ICE candidate to the owning client (fallback: broadcast)."""
        msg = {
            "type": "webrtc_ice_candidate",
            "candidate": candidate,
            "sdpMLineIndex": mline_index,
        }
        owner = self._get_webrtc_owner()
        if owner is not None:
            self.logger.info(
                "<<< Sending webrtc_ice_candidate (mline=%d) to WebRTC owner %s",
                mline_index,
                self._webrtc_owner_ip or "unknown",
            )
        else:
            self.logger.info("<<< Sending webrtc_ice_candidate (mline=%d) to %d clients", mline_index, len(self.connected_clients))
        try:
            import json as _json, time as _time

            with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                _f.write(
                    _json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H4",
                            "location": "websocket_server.py:send_webrtc_ice",
                            "message": "send webrtc_ice_candidate",
                            "data": {"clients": len(self.connected_clients), "mline": int(mline_index)},
                            "timestamp": int(_time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        if owner is not None:
            self.send_to_client_sync(owner, msg)
        else:
            self.broadcast_sync(msg)

    def send_webrtc_error(self, error: str) -> None:
        """Send WebRTC error to the owning client (fallback: broadcast)."""
        msg = {"type": "webrtc_error", "error": error}
        owner = self._get_webrtc_owner()
        if owner is not None:
            self.send_to_client_sync(owner, msg)
            self.logger.warning("<<< WebRTC error (owner=%s): %s", self._webrtc_owner_ip or "unknown", error)
        else:
            self.broadcast_sync(msg)
            self.logger.warning("<<< Broadcast WebRTC error: %s", error)

    async def _periodic_menon_telemetry_log(self, interval_seconds: float = 1.0) -> None:
        """Emit a concise 1 Hz INFO log with latest Menon calibration/coordinate RX/TX."""
        keys_rx = ['set_align', 'set_extrinsics', 'solve_pnp', 'pixel_to_world']
        keys_tx = ['calibration-bundle', 'set_align_result', 'set_extrinsics_result', 'solve_pnp_result', 'pixel_to_world_response']
        while self.running:
            try:
                parts = []
                # RX summary
                rx_items = []
                for k in keys_rx:
                    entry = self._telemetry['rx'].get(k)
                    if entry:
                        rx_items.append(f"{k}:{entry['data']}")
                parts.append(f"rx=[{'; '.join(rx_items) if rx_items else '-'}]")
                # TX summary
                tx_items = []
                for k in keys_tx:
                    entry = self._telemetry['tx'].get(k)
                    if entry:
                        tx_items.append(f"{k}:{entry['data']}")
                parts.append(f"tx=[{'; '.join(tx_items) if tx_items else '-'}]")
                env_flag = str(os.environ.get("NOESIS_MENON_IO_DEBUG", "")).strip().lower()
                if env_flag in ("1", "true", "yes", "on"):
                    self.logger.debug(f"MENON I/O | {' | '.join(parts)}")
            except asyncio.CancelledError:
                # Properly handle cancellation
                break
            except Exception:
                # Never fail loop due to logging issues
                pass
            finally:
                # Always yield to avoid event loop starvation
                try:
                    await asyncio.sleep(min(interval_seconds, 1.0))
                except asyncio.CancelledError:
                    break
    
    async def _cleanup_stale_connections(self):
        """Periodically clean up any stale or closed connections"""
        try:
            # Since the broadcast method already handles cleanup when connections fail,
            # we'll make this cleanup much simpler and safer
            stale_clients = set()

            for client in self.connected_clients:
                try:
                    # Check various attributes that might indicate a closed connection
                    is_closed = False

                    # Try the most common attributes first
                    if hasattr(client, 'closed') and getattr(client, 'closed', False):
                        is_closed = True
                    elif hasattr(client, 'close_code') and getattr(client, 'close_code', None) is not None:
                        is_closed = True
                    elif hasattr(client, 'state') and getattr(client, 'state', None) == 'CLOSED':
                        is_closed = True

                    if is_closed:
                        stale_clients.add(client)

                except Exception:
                    # If we can't safely inspect the client, assume it's stale
                    stale_clients.add(client)

            if stale_clients:
                for client in stale_clients:
                    if client in self.connected_clients:
                        self.connected_clients.remove(client)
                        client_ip = getattr(client, 'remote_address', 'Unknown') if hasattr(client, 'remote_address') else "Unknown"
                        self.logger.info(f"Cleaned up stale connection for client {client_ip}")
                self.logger.debug(f"Cleaned up {len(stale_clients)} stale connections")

        except Exception as e:
            self.logger.error(f"Error during connection cleanup: {e}")

    async def _periodic_stats_broadcast(self, interval_seconds: float = 1.0):
        """Periodically fetches and broadcasts stats."""
        self.logger.info(f"Starting periodic stats broadcast every {interval_seconds} seconds.")
        cleanup_counter = 0

        while self.running:
            try:
                # Periodic cleanup of stale connections (every 60 seconds, reduced frequency)
                cleanup_counter += 1
                if cleanup_counter >= 60:
                    await self._cleanup_stale_connections()
                    cleanup_counter = 0

                if self.stats_callback and self.connected_clients: # Only send if callback exists and clients are connected
                    self.logger.debug(f"📊 Broadcasting stats to {len(self.connected_clients)} clients")
                    stats_payload = self.stats_callback()
                    if stats_payload: # Ensure callback returned something
                        stats_message = {
                            'type': 'stats',
                            'payload': stats_payload
                        }
                        await self.broadcast(stats_message)
                        self.logger.debug("✅ Stats broadcast completed")

                        # Periodic INFO log (every ~5s) to surface telemetry presence
                        now = time.time()
                        if now - self._last_stats_info_log >= 5.0:
                            try:
                                cam_count = len((stats_payload or {}).get('cameras', {}))
                                uptime = (stats_payload or {}).get('uptime', 0)
                                # Prepare compact payload dump (truncate to keep logs readable)
                                try:
                                    stats_json = json.dumps(stats_payload, separators=(',', ':'), default=str)
                                except Exception:
                                    stats_json = str(stats_payload)
                                max_len = 4096
                                if len(stats_json) > max_len:
                                    extra = len(stats_json) - max_len
                                    stats_json = stats_json[:max_len] + f"...(+{extra} chars)"
                                self.logger.debug(
                                    f"📡 Stats sent to {len(self.connected_clients)} clients | cameras={cam_count} | uptime={uptime:.1f}s | payload={stats_json}"
                                )
                            except Exception:
                                # Defensive: never break the loop due to logging
                                pass
                            self._last_stats_info_log = now
                    else:
                        self.logger.debug("Stats callback returned empty payload, skipping broadcast.")
                elif not self.connected_clients:
                    #self.logger.debug("No clients connected, skipping stats broadcast.")
                    pass

                # Wait for the next interval - use shorter intervals to allow for cancellation
                sleep_time = 5.0 if not self.connected_clients else interval_seconds
                await asyncio.sleep(min(sleep_time, 1.0))  # Cap at 1 second for responsiveness

            except asyncio.CancelledError:
                self.logger.info("Periodic stats broadcast task cancelled.")
                break # Exit loop if task is cancelled
            except Exception as e:
                self.logger.error(f"Error during periodic stats broadcast: {e}")
                # Avoid tight loop on persistent error
                await asyncio.sleep(interval_seconds) 

    async def start(self):
        """Start the WebSocket server and the periodic stats broadcast."""
        try:
            # Capture the running loop for cross-thread callbacks
            try:
                self.event_loop = asyncio.get_running_loop()
            except RuntimeError:
                self.event_loop = None

            # Start the server (legacy API for compatibility)
            self.server = await serve(
                self.handle_client,
                self.host,
                self.port,
                max_size=None,       # allow large binary frames
                max_queue=1,         # minimize buffering to reduce latency
                ping_interval=60,    # keep-alive pings every 60 seconds (was 20)
                ping_timeout=30      # wait 30 seconds for pong response (was 20)
            )

            # If bound to port 0 (ephemeral), capture the actual port
            try:
                if self.port in (0, None) and self.server and getattr(self.server, 'sockets', None):
                    sock = self.server.sockets[0]
                    bound = sock.getsockname()
                    if isinstance(bound, tuple) and len(bound) >= 2:
                        self.port = int(bound[1])
            except Exception:
                pass

            # Start periodic stats broadcast task if callback is provided
            if self.stats_callback:
                self._stats_task = asyncio.create_task(
                    self._periodic_stats_broadcast(),
                    name="PeriodicStatsBroadcast"
                )

            # Always start Menon I/O telemetry logger (1 Hz)
            self._telemetry_task = asyncio.create_task(
                self._periodic_menon_telemetry_log(),
                name="MenonTelemetryLogger"
            )

            self.logger.info(f"WebSocket server running on {self.host}:{self.port}")
            print(f"🚀 WebSocket server is LIVE on {self.host}:{self.port}")
            print(f"📡 Server listening for connections on ws://{self.host}:{self.port}")
            print(f"🔄 Periodic stats broadcast: {'ENABLED' if self.stats_callback else 'DISABLED'}")
            print(f"💓 Keep-alive: ping every 60s, timeout 30s")
            self.logger.info("WebSocket server startup completed successfully")

        except OSError as e:
            self.logger.error(f"Failed to start server (Port {self.port} likely in use): {e}")
            raise
        except Exception as e:
            self.logger.error(f"Server failed: {e}")
            raise
    async def stop(self):
        """Gracefully stop the WebSocket server and the periodic stats broadcast."""
        self.logger.info("Stopping WebSocket server...")
        self.running = False

        # Immediately cancel all background tasks to prevent them from continuing
        tasks_to_cancel = []

        # Cancel any pending binary flush task
        if self._binary_flush_task and not self._binary_flush_task.done():
            self._binary_flush_task.cancel()
            tasks_to_cancel.append(self._binary_flush_task)

        # Cancel the periodic stats task
        if self._stats_task and not self._stats_task.done():
            self.logger.info("Cancelling periodic stats broadcast task...")
            self._stats_task.cancel()
            tasks_to_cancel.append(self._stats_task)

        # Cancel Menon telemetry task
        if self._telemetry_task and not self._telemetry_task.done():
            self.logger.info("Cancelling Menon telemetry task...")
            self._telemetry_task.cancel()
            tasks_to_cancel.append(self._telemetry_task)

        # Cancel server task if it exists
        if self.server_task and not self.server_task.done():
            self.server_task.cancel()
            tasks_to_cancel.append(self.server_task)

        # Wait for all tasks to cancel
        if tasks_to_cancel:
            try:
                await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
                self.logger.info(f"Successfully cancelled {len(tasks_to_cancel)} background tasks")
            except Exception as e:
                self.logger.warning(f"Error waiting for task cancellation: {e}")

        # Close all client connections
        if self.connected_clients:
            try:
                close_tasks = [client.close() for client in self.connected_clients]
                await asyncio.gather(*close_tasks, return_exceptions=True)
                self.connected_clients.clear()
                self.logger.info(f"Closed {len(close_tasks)} client connections")
            except Exception as e:
                self.logger.warning(f"Error closing client connections: {e}")

        # Close server
        if self.server:
            try:
                self.server.close()
                await self.server.wait_closed()
                self.server = None
                self.logger.info("WebSocket server stopped successfully")
            except Exception as e:
                self.logger.error(f"Error stopping WebSocket server: {e}")

    def _force_close_server(self) -> bool:
        """Best-effort fallback to close listening sockets when the event loop is unavailable."""
        server = self.server
        if not server:
            return True

        try:
            # Close the asyncio server without awaiting wait_closed (loop may be unavailable)
            server.close()
        except Exception as exc:
            self.logger.error(f"Forced WebSocket server close failed: {exc}")
            return False

        # Explicitly close bound sockets to free the port immediately
        sockets = getattr(server, "sockets", None)
        if sockets:
            for sock in sockets:
                try:
                    sock.close()
                except Exception:
                    pass

        self.server = None
        self.connected_clients.clear()
        self.logger.info("WebSocket server sockets closed via fallback path")
        return True

    def stop_sync(self, timeout: float = 2.0) -> bool:
        """Synchronous stop method for use from other threads"""
        self.logger.info("Stopping WebSocket server (sync mode)...")
        self.running = False

        # Attempt to schedule the async stop on an active event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        # If we're already running inside the websocket event loop thread, just schedule the coroutine
        if loop is not None:
            try:
                loop.create_task(self.stop())
                return True
            except Exception as exc:
                self.logger.warning(f"Could not schedule WebSocket stop on running loop: {exc}")
                return False

        target_loop = getattr(self, 'event_loop', None)
        deadline = time.time() + timeout

        # Allow the worker thread a moment to publish its loop reference
        while (target_loop is None or target_loop.is_closed()) and time.time() < deadline:
            time.sleep(0.05)
            target_loop = getattr(self, 'event_loop', None)

        if not target_loop or target_loop.is_closed():
            self.logger.warning("No event loop available for WebSocket server shutdown; forcing close")
            return self._force_close_server()

        if not target_loop.is_running():
            self.logger.warning("WebSocket event loop not running during shutdown; forcing close")
            return self._force_close_server()

        try:
            future = asyncio.run_coroutine_threadsafe(self.stop(), target_loop)
        except Exception as exc:
            self.logger.warning(f"Could not stop WebSocket server via event loop: {exc}")
            return self._force_close_server()

        try:
            remaining = max(0.1, deadline - time.time())
            future.result(timeout=remaining)
            return True
        except concurrent.futures.TimeoutError:
            self.logger.warning(f"WebSocket server stop did not complete within {timeout}s; forcing close")
            return self._force_close_server()
        except Exception as exc:
            self.logger.error(f"Error waiting for WebSocket server stop: {exc}")
            # Try force close on any exception
            force_result = self._force_close_server()
            self.logger.info(f"Force close attempted: {force_result}")
            return force_result

    async def handle_client(self, websocket, path=None):
        """Handle incoming WebSocket connections and messages."""
        try:
            client_ip = websocket.remote_address[0] if hasattr(websocket, 'remote_address') else "Unknown"
        except Exception as exc:
            self.logger.error(f"handle_client: unable to read remote address: {exc}")
            client_ip = "Unknown"
        self.connected_clients.add(websocket)
        self.logger.info(f"Client {client_ip} connected. Total clients: {len(self.connected_clients)}")
        print(f"✅ Client {client_ip} connected! Total clients: {len(self.connected_clients)}")

        try:
            # Send initial detection configuration to new client
            if hasattr(self, 'detection_config_getter') and self.detection_config_getter:
                try:
                    # Get current detection config
                    config = self.detection_config_getter()
                    if config:
                        initial_config_message = {
                            'type': 'detection_config_sync',
                            'config': config
                        }
                        await websocket.send(json.dumps(initial_config_message))
                        self.logger.info(f"Sent initial detection config to {client_ip}")
                except Exception as e:
                    self.logger.warning(f"Could not send initial detection config to {client_ip}: {e}")

            # Send initial trail visualization state to new client
            try:
                initial_trail_message = {
                    'type': 'trail_visualization_enabled_update',
                    'enabled': self.initial_trail_state
                }
                await websocket.send(json.dumps(initial_trail_message))
                self.logger.info(f"Sent initial trail visualization state ({self.initial_trail_state}) to {client_ip}")
            except Exception as e:
                self.logger.warning(f"Could not send initial trail visualization state to {client_ip}: {e}")

            # Send initial calibration bundle if available
            try:
                if callable(self.calibration_getter):
                    bundle = self.calibration_getter() or {}
                    if bundle:
                        msg = {'type': 'calibration-bundle', 'data': bundle}
                        await websocket.send(json.dumps(msg))
                        try:
                            self._record_tx(msg)
                        except Exception:
                            pass
                        self.logger.info(f"Sent calibration-bundle to {client_ip}")
            except Exception as e:
                self.logger.warning(f"Could not send calibration-bundle to {client_ip}: {e}")

            # Send an immediate stats snapshot on connect for quicker UI readiness
            try:
                if self.stats_callback:
                    snap = self.stats_callback() or {}
                    if snap:
                        safe_snap = convert_numpy_types(snap)
                        stats_message = {'type': 'stats', 'payload': safe_snap}
                        await websocket.send(json.dumps(stats_message))
                        self.logger.info(f"Sent initial stats snapshot to {client_ip}")
            except Exception as e:
                self.logger.debug(f"Could not send initial stats snapshot to {client_ip}: {e}")

            # Process messages from client
            async for message in websocket:
                try:
                    data = json.loads(message)
                    # Record RX telemetry for Menon RPCs
                    try:
                        self._record_rx(data)
                    except Exception:
                        pass

                    # Handle clear_stats command
                    if data.get('type') == 'clear_stats':
                        self.logger.info(f"Received clear_stats request from {client_ip}")
                        # Call clear stats function if registered
                        if self.stats_callback and hasattr(self.stats_callback, 'clear_stats'):
                            self.stats_callback.clear_stats()
                            self.logger.info("Stats cleared. Broadcasting updated stats...")
                            # Broadcast updated stats
                            if self.stats_callback:
                                stats_payload = self.stats_callback()
                                stats_message = {
                                    'type': 'stats',
                                    'payload': stats_payload
                                }
                                await self.broadcast(stats_message)
                        else:
                            self.logger.warning("Stats callback not available to clear stats.")

                    # Handle visualization toggle command
                    elif data.get('type') == 'set_vis_toggle':
                        toggle_name = data.get('toggle_name')
                        enabled = data.get('enabled')

                        if toggle_name is not None and isinstance(enabled, bool):
                            self.logger.info(f"Received set_vis_toggle from {client_ip}: {toggle_name} = {enabled}")

                            # Call toggle callback if available
                            if self.toggle_callback:
                                try:
                                    self.toggle_callback(toggle_name, enabled)
                                except Exception as e:
                                    self.logger.error(f"Error in toggle callback: {e}")

                            # Broadcast to all clients including the sender
                            broadcast_message = {
                                'type': 'toggle_update',
                                'toggle_name': toggle_name,
                                'enabled': enabled
                            }
                            await self.broadcast(broadcast_message)
                        else:
                            self.logger.warning(f"Invalid set_vis_toggle message from {client_ip}: {data}")

                    # Handle client heartbeat ping messages
                    elif data.get('type') == 'ping':
                        timestamp = data.get('timestamp')
                        self.logger.debug(f"Received ping from {client_ip} (timestamp: {timestamp})")
                        # Send pong response
                        try:
                            pong_message = {
                                'type': 'pong',
                                'timestamp': timestamp
                            }
                            await websocket.send(json.dumps(pong_message))
                        except Exception as e:
                            self.logger.warning(f"Failed to send pong response to {client_ip}: {e}")

                    # Handle detection configuration updates
                    elif data.get('type') == 'update_detection_config':
                        config_data = data.get('config', {})

                        if config_data:
                            self.logger.info(f"Received detection config update from {client_ip}: {config_data}")

                            # Call detection config callback if available
                            if hasattr(self, 'detection_config_callback') and self.detection_config_callback:
                                try:
                                    self.detection_config_callback(config_data)
                                except Exception as e:
                                    self.logger.error(f"Error in detection config callback: {e}")

                            # Broadcast to all clients including the sender
                            broadcast_message = {
                                'type': 'detection_config_update',
                                'config': config_data
                            }
                            await self.broadcast(broadcast_message)
                        else:
                            self.logger.warning(f"Invalid detection config message from {client_ip}: {data}")

                    # Handle BEV config overrides
                    elif data.get('type') == 'bev-config':
                        cam_id = data.get('cameraId') or data.get('camId')
                        cfg = data.get('config') or {}
                        if isinstance(cam_id, str) and isinstance(cfg, dict):
                            self.logger.info("Received BEV config for %s: %s", cam_id, cfg)
                            if callable(self.bev_config_callback):
                                try:
                                    self.bev_config_callback(cam_id, cfg)
                                except Exception as exc:
                                    self.logger.error("BEV config callback failed: %s", exc)
                            ack = {
                                'type': 'bev-config-ack',
                                'cameraId': cam_id,
                                'config': cfg,
                            }
                            await self.broadcast(ack)
                        else:
                            self.logger.warning("Invalid BEV config payload from %s: %s", client_ip, data)

                    elif data.get('type') == 'bev-overlay':
                        cam_id = data.get('cameraId') or data.get('camId')
                        enabled = data.get('enabled')
                        if isinstance(cam_id, str) and isinstance(enabled, bool):
                            self.logger.info("Received BEV overlay toggle for %s: %s", cam_id, enabled)
                            if callable(self.bev_overlay_callback):
                                try:
                                    self.bev_overlay_callback(cam_id, enabled)
                                except Exception as exc:
                                    self.logger.error("BEV overlay callback failed: %s", exc)
                            await self.broadcast({'type': 'bev-overlay-update', 'cameraId': cam_id, 'enabled': enabled})
                        else:
                            self.logger.warning("Invalid BEV overlay message from %s: %s", client_ip, data)

                    elif data.get('type') == 'ma_heatmap_ready':
                        cam_id = data.get('cameraId') or data.get('camId')
                        if isinstance(cam_id, str) and callable(self.ma_ready_callback):
                            try:
                                self.ma_ready_callback(cam_id)
                            except Exception:
                                pass
                        # no ack required; server action is side-effect only

                    elif data.get('type') == 'auto_calibrate_pose':
                        cam_id = data.get('camera') or data.get('cameraId') or data.get('camId')
                        result = {'type': 'auto_calibrate_result'}
                        if callable(self.auto_calibrate_handler):
                            try:
                                out = await asyncio.wait_for(
                                    asyncio.to_thread(self.auto_calibrate_handler, cam_id if isinstance(cam_id, str) else None),
                                    timeout=30.0
                                )
                                if isinstance(out, dict):
                                    result.update(out)
                            except asyncio.TimeoutError:
                                result.update({'ok': False, 'error': 'timeout'})
                            except Exception as exc:
                                result.update({'ok': False, 'error': str(exc)})
                        else:
                            result.update({'ok': False, 'error': 'no_handler'})
                        try:
                            await websocket.send(json.dumps(result))
                        except Exception:
                            pass

                    # ---- Spatial & calibration RPCs ----
                    # legacy 'get_transformation' removed; calibration-bundle is source of truth

                    elif data.get('type') == 'pixel_to_world':
                        req = data
                        req_id = req.get('request_id') or req.get('reqId')
                        result = {'type': 'pixel_to_world_response'}
                        if req_id is not None:
                            result['request_id'] = req_id
                            result.setdefault('reqId', req_id)
                        try:
                            if callable(self.pixel_to_world_handler):
                                out = self.pixel_to_world_handler(req) or {}
                                result.update(out)
                            else:
                                result.update({'ok': False, 'error': 'no_handler'})
                        except Exception as e:
                            result.update({'ok': False, 'error': str(e)})

                        world_val = result.get('world')
                        if isinstance(world_val, (list, tuple)) and len(world_val) >= 3:
                            try:
                                result['world'] = {
                                    'x': float(world_val[0]),
                                    'y': float(world_val[1]),
                                    'z': float(world_val[2]),
                                }
                            except Exception:
                                pass

                        try:
                            # Record TX telemetry and send
                            self._record_tx(result)
                        except Exception:
                            pass
                        try:
                            await websocket.send(json.dumps(result))
                        except Exception:
                            pass

                    elif data.get('type') == 'set_extrinsics':
                        req = data
                        result = {'type': 'set_extrinsics_result'}
                        try:
                            if callable(self.set_extrinsics_handler):
                                out = self.set_extrinsics_handler(req) or {}
                                result.update(out)
                            else:
                                result.update({'ok': False, 'error': 'no_handler'})
                        except Exception as e:
                            result.update({'ok': False, 'error': str(e)})
                        try:
                            try:
                                self._record_tx(result)
                            except Exception:
                                pass
                            await websocket.send(json.dumps(result))
                        except Exception:
                            pass

                    elif data.get('type') == 'solve_pnp':
                        req = data
                        result = {'type': 'solve_pnp_result'}
                        try:
                            if callable(self.solve_pnp_handler):
                                out = self.solve_pnp_handler(req) or {}
                                result.update(out)
                            else:
                                result.update({'ok': False, 'error': 'no_handler'})
                        except Exception as e:
                            result.update({'ok': False, 'error': str(e)})
                        try:
                            try:
                                self._record_tx(result)
                            except Exception:
                                pass
                            await websocket.send(json.dumps(result))
                        except Exception:
                            pass

                    elif data.get('type') == 'set_align':
                        req = data
                        result = {'type': 'set_align_result'}
                        try:
                            if callable(self.set_align_handler):
                                out = self.set_align_handler(req) or {}
                                result.update(out)
                            else:
                                result.update({'ok': False, 'error': 'no_handler'})
                        except Exception as e:
                            result.update({'ok': False, 'error': str(e)})
                        try:
                            try:
                                self._record_tx(result)
                            except Exception:
                                pass
                            await websocket.send(json.dumps(result))
                        except Exception:
                            pass

                    elif data.get('type') == 'get_ma_depth':
                        camera = (
                            data.get('camera')
                            or data.get('cameraId')
                            or data.get('camera_id')
                            or data.get('camId')
                        )
                        request_id = (
                            data.get('request_id')
                            or data.get('requestId')
                            or data.get('requestID')
                        )
                        if not request_id:
                            request_id = str(uuid.uuid4())
                        ts_max_us = (
                            data.get('ts_max_us')
                            or data.get('tsMaxUs')
                            or data.get('ts_maxUS')
                            or data.get('tsMax_us')
                        )
                        if ts_max_us is None:
                            ts_max_us = data.get('ts_max') or data.get('tsMax')

                        provider = self.ma_depth_provider if callable(self.ma_depth_provider) else None
                        if not camera or provider is None:
                            result = {
                                'type': 'ma_depth_response',
                                'camera': camera,
                                'request_id': request_id,
                                'served_from_cache': False,
                                'error': 'no_provider',
                                'ok': False,
                            }
                            await self._send_json_with_boundary_metrics(
                                websocket,
                                result,
                                route="get_ma_depth",
                                message_type="ma_depth_response",
                            )
                            continue

                        rate_key = f"{client_ip}:{camera}"
                        now = time.time()
                        last = self._depth_rpc_tracker.get(rate_key, 0.0)
                        if now - last < self._depth_rate_limit_window:
                            self.logger.debug(f"Depth RPC throttled for {rate_key}")
                            continue
                        self._depth_rpc_tracker[rate_key] = now
                        if len(self._depth_rpc_tracker) > 256:
                            self._depth_rpc_tracker = {
                                k: v for k, v in self._depth_rpc_tracker.items() if now - v <= self._tracker_prune_window
                            }

                        try:
                            provider_start_ns = time.perf_counter_ns()
                            payload = await asyncio.wait_for(
                                asyncio.to_thread(provider, camera, ts_max_us, request_id),
                                timeout=self._depth_rpc_timeout,
                            )
                            provider_ms = (time.perf_counter_ns() - provider_start_ns) / 1_000_000.0
                            self._record_boundary_serialization_stage(
                                provider_ms,
                                0,
                                channel="ws",
                                route="get_ma_depth",
                                message_type="ma_depth_response",
                                stage="provider_wait",
                                outcome="ok",
                                include_budget=False,
                            )
                            if payload:
                                if 'type' not in payload:
                                    payload['type'] = 'ma_depth_response'
                                payload.setdefault('camera', camera)
                                if request_id and 'request_id' not in payload:
                                    payload['request_id'] = request_id
                                payload.setdefault('served_from_cache', False)
                                payload.setdefault('ts_us', 0)
                                payload.setdefault('ok', 'error' not in payload)
                                await self._send_json_with_boundary_metrics(
                                    websocket,
                                    payload,
                                    route="get_ma_depth",
                                    message_type="ma_depth_response",
                                    use_to_thread_json=True,
                                )
                            else:
                                result = {
                                    'type': 'ma_depth_response',
                                    'camera': camera,
                                    'request_id': request_id,
                                    'served_from_cache': False,
                                    'error': 'not_available',
                                    'ok': False,
                                }
                                await self._send_json_with_boundary_metrics(
                                    websocket,
                                    result,
                                    route="get_ma_depth",
                                    message_type="ma_depth_response",
                                )
                        except asyncio.TimeoutError:
                            self.logger.warning(f"Depth RPC timed out for {camera} from {client_ip}")
                            result = {
                                'type': 'ma_depth_response',
                                'camera': camera,
                                'request_id': request_id,
                                'served_from_cache': False,
                                'error': 'timeout',
                                'ok': False,
                            }
                            await self._send_json_with_boundary_metrics(
                                websocket,
                                result,
                                route="get_ma_depth",
                                message_type="ma_depth_response",
                            )
                        except Exception as exc:
                            result = {
                                'type': 'ma_depth_response',
                                'camera': camera,
                                'request_id': request_id,
                                'served_from_cache': False,
                                'error': str(exc),
                                'ok': False,
                            }
                            await self._send_json_with_boundary_metrics(
                                websocket,
                                result,
                                route="get_ma_depth",
                                message_type="ma_depth_response",
                            )

                    elif data.get('type') == 'get_floorplan':
                        request_id = data.get('request_id') or data.get('requestId') or str(uuid.uuid4())
                        camera = data.get('camera')
                        if not camera and isinstance(data.get('cameras'), list) and data['cameras']:
                            camera = str(data['cameras'][0])
                        camera = str(camera) if camera else ''
                        max_age_sec = float(data.get('max_age_sec', data.get('maxAgeSec', 60.0)))
                        grid_res_m = float(data.get('grid_res_m', data.get('gridResM', 0.5)))
                        max_extent_m = float(data.get('max_extent_m', data.get('maxExtentM', 20.0)))
                        cache_only = bool(data.get('cache_only', data.get('cacheOnly', False)))

                        provider = getattr(self, 'floorplan_provider', None)
                        result = {
                            'type': 'floorplan_response',
                            'request_id': request_id,
                            'camera_id': camera,
                            'cache_only': cache_only,
                        }
                        if not callable(provider):
                            result['error'] = 'no_provider'
                            await self._send_json_with_boundary_metrics(
                                websocket,
                                result,
                                route="get_floorplan",
                                message_type="floorplan_response",
                            )
                            continue

                        rate_key = f"{client_ip}:{camera or 'unknown'}"
                        now = time.time()
                        last = self._floorplan_rpc_tracker.get(rate_key, 0.0)
                        if now - last < self._floorplan_rate_limit_window:
                            self.logger.debug(f"Floorplan RPC throttled for {rate_key}")
                            continue
                        self._floorplan_rpc_tracker[rate_key] = now
                        if len(self._floorplan_rpc_tracker) > 256:
                            self._floorplan_rpc_tracker = {
                                k: v for k, v in self._floorplan_rpc_tracker.items() if now - v <= self._tracker_prune_window
                            }

                        try:
                            provider_start_ns = time.perf_counter_ns()
                            payload = await asyncio.wait_for(
                                asyncio.to_thread(provider, camera, max_age_sec, grid_res_m, max_extent_m, cache_only=cache_only),
                                timeout=self._floorplan_rpc_timeout
                            )
                            provider_ms = (time.perf_counter_ns() - provider_start_ns) / 1_000_000.0
                            self._record_boundary_serialization_stage(
                                provider_ms,
                                0,
                                channel="ws",
                                route="get_floorplan",
                                message_type="floorplan_response",
                                stage="provider_wait",
                                outcome="ok",
                                include_budget=False,
                            )
                            if payload:
                                result.update(payload)
                                await self._send_json_with_boundary_metrics(
                                    websocket,
                                    result,
                                    route="get_floorplan",
                                    message_type="floorplan_response",
                                    use_to_thread_json=True,
                                )
                            else:
                                result['error'] = 'no_payload'
                                await self._send_json_with_boundary_metrics(
                                    websocket,
                                    result,
                                    route="get_floorplan",
                                    message_type="floorplan_response",
                                )
                        except asyncio.TimeoutError:
                            self.logger.warning(f"Floorplan RPC timed out for {camera} from {client_ip}")
                            result['error'] = 'timeout'
                            await self._send_json_with_boundary_metrics(
                                websocket,
                                result,
                                route="get_floorplan",
                                message_type="floorplan_response",
                            )
                        except Exception as exc:
                            result['error'] = str(exc)
                            self.logger.error(f"Floorplan generation error: {exc}")
                            await self._send_json_with_boundary_metrics(
                                websocket,
                                result,
                                route="get_floorplan",
                                message_type="floorplan_response",
                            )

                    # Handle individual detection toggles
                    elif data.get('type') == 'set_detection_toggle':
                        toggle_name = data.get('toggle_name')
                        enabled = data.get('enabled')

                        if toggle_name is not None and isinstance(enabled, bool):
                            self.logger.info(f"Received detection toggle from {client_ip}: {toggle_name} = {enabled}")

                            # Call detection toggle callback if available
                            if hasattr(self, 'detection_toggle_callback') and self.detection_toggle_callback:
                                try:
                                    self.detection_toggle_callback(toggle_name, enabled)
                                except Exception as e:
                                    self.logger.error(f"Error in detection toggle callback: {e}")

                            # Broadcast to all clients including the sender
                            broadcast_message = {
                                'type': 'detection_toggle_update',
                                'toggle_name': toggle_name,
                                'enabled': enabled
                            }
                            await self.broadcast(broadcast_message)
                        else:
                            self.logger.warning(f"Invalid detection toggle message from {client_ip}: {data}")

                    # Handle WebRTC signaling: offer from browser
                    elif data.get('type') == 'webrtc_offer':
                        self.logger.info(">>> Received webrtc_offer from client %s", client_ip)
                        if self.webrtc_gateway is not None or self.webrtc_elem is not None:
                            prev_owner = self._get_webrtc_owner()
                            if prev_owner is not None and prev_owner is not websocket:
                                try:
                                    await prev_owner.send(json.dumps({'type': 'webrtc_error', 'error': 'webrtc_taken_over'}))
                                except Exception:
                                    pass
                            self._webrtc_owner = websocket
                            self._webrtc_owner_ip = client_ip
                        # Prefer gateway if registered, otherwise fall back to legacy approach
                        if self.webrtc_gateway is not None:
                            sdp = data.get('sdp', '')
                            sdp_lines = len(sdp.split('\n')) if sdp else 0
                            self.logger.info("    Gateway registered, forwarding offer (%d SDP lines)", sdp_lines)
                            try:
                                import json as _json, time as _time

                                with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                                    _f.write(
                                        _json.dumps(
                                            {
                                                "sessionId": "debug-session",
                                                "runId": "run1",
                                                "hypothesisId": "H4",
                                                "location": "websocket_server.py:handle_client",
                                                "message": "rx webrtc_offer",
                                                "data": {"client": client_ip, "sdp_lines": sdp_lines},
                                                "timestamp": int(_time.time() * 1000),
                                            }
                                        )
                                        + "\n"
                                    )
                            except Exception:
                                pass
                            try:
                                from pathlib import Path

                                offer_dir = Path("/home/mayor/Noesis_Devel/.cursor/webrtc_offers")
                                offer_dir.mkdir(parents=True, exist_ok=True)
                                ts_ms = int(_time.time() * 1000)
                                safe_client = str(client_ip).replace(":", "_")
                                offer_path = offer_dir / f"offer_{ts_ms}_{safe_client}.sdp"
                                offer_path.write_text(sdp or "", encoding="utf-8")
                                with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                                    _f.write(
                                        _json.dumps(
                                            {
                                                "sessionId": "debug-session",
                                                "runId": "run1",
                                                "hypothesisId": "H4",
                                                "location": "websocket_server.py:handle_client",
                                                "message": "saved webrtc_offer",
                                                "data": {
                                                    "client": client_ip,
                                                    "path": str(offer_path),
                                                    "sdp_lines": sdp_lines,
                                                },
                                                "timestamp": ts_ms,
                                            }
                                        )
                                        + "\n"
                                    )
                            except Exception:
                                pass
                            self.webrtc_gateway.accept_offer(sdp)
                        else:
                            self.logger.warning("    No gateway registered, using legacy handler")
                            await self._handle_webrtc_offer(websocket, data)

                    # Handle WebRTC signaling: ICE candidate from browser
                    elif data.get('type') == 'webrtc_ice_candidate':
                        self.logger.info(">>> Received webrtc_ice_candidate from client %s", client_ip)
                        owner = self._get_webrtc_owner()
                        if owner is not None and owner is not websocket:
                            try:
                                await websocket.send(json.dumps({'type': 'webrtc_error', 'error': 'webrtc_not_owner'}))
                            except Exception:
                                pass
                            continue
                        if owner is None and (self.webrtc_gateway is not None or self.webrtc_elem is not None):
                            self._webrtc_owner = websocket
                            self._webrtc_owner_ip = client_ip
                        # Prefer gateway if registered
                        if self.webrtc_gateway is not None:
                            candidate = data.get('candidate', '')
                            mline_index = data.get('sdpMLineIndex', 0)
                            self.logger.info("    Adding ICE candidate (mline=%d): %s...", mline_index, candidate[:50] if candidate else '')
                            try:
                                import json as _json, time as _time

                                with open("/home/mayor/Noesis_Devel/.cursor/debug.log", "a", encoding="utf-8") as _f:
                                    _f.write(
                                        _json.dumps(
                                            {
                                                "sessionId": "debug-session",
                                                "runId": "run1",
                                                "hypothesisId": "H4",
                                                "location": "websocket_server.py:handle_client",
                                                "message": "rx webrtc_ice_candidate",
                                                "data": {"client": client_ip, "mline": int(mline_index)},
                                                "timestamp": int(_time.time() * 1000),
                                            }
                                        )
                                        + "\n"
                                    )
                            except Exception:
                                pass
                            self.webrtc_gateway.accept_ice(candidate, mline_index)
                        else:
                            await self._handle_webrtc_ice_candidate(websocket, data)

                except json.JSONDecodeError:
                    self.logger.warning(f"Received non-JSON message from {client_ip}. Ignoring.")
                except Exception as e:
                    self.logger.error(f"Error processing message from {client_ip}: {e}")

        except websockets.exceptions.ConnectionClosedOK:
            self.logger.info(f"Client {client_ip} disconnected normally (code: 1000).")
            print(f"❌ Client {client_ip} disconnected normally.")
        except websockets.exceptions.ConnectionClosedError as e:
            # Log ping timeout errors as info instead of warning to reduce noise
            if "keepalive ping timeout" in str(e).lower():
                self.logger.info(f"Client {client_ip} disconnected due to ping timeout - connection cleaned up.")
            else:
                self.logger.warning(f"Client {client_ip} disconnected with error: {e}")
                print(f"❌ Client {client_ip} disconnected with error: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error with client {client_ip}: {e}")
        finally:
            # Ensure client is removed from set - use discard to avoid KeyError if already removed
            try:
                self.connected_clients.discard(websocket)
                if self._webrtc_owner is websocket:
                    self._webrtc_owner = None
                    self._webrtc_owner_ip = None
                self.logger.info(f"Client {client_ip} removed. Total clients: {len(self.connected_clients)}")
                print(f"👋 Client {client_ip} removed. Total clients: {len(self.connected_clients)}")
            except Exception as e:
                self.logger.warning(f"Error removing client {client_ip} from connected clients: {e}")
    
    async def broadcast(self, message):
        """Broadcast a message to all connected WebSocket clients

        Args:
            message: Message to broadcast (dict, bytes, or string)
        """
        if not self.connected_clients:
            return

        message_str = ""
        disconnected_clients = []
        # Use a list to preserve order for correct result-to-client mapping
        active_clients = list(self.connected_clients)
        is_dict_message = isinstance(message, dict)
        message_type = "unknown"
        payload_bytes = 0

        try:
            # Prepare message based on type
            if isinstance(message, dict):
                # Convert to JSON string
                message_type = str(message.get("type", "unknown"))
                convert_start_ns = time.perf_counter_ns()
                message = convert_numpy_types(message)
                convert_ms = (time.perf_counter_ns() - convert_start_ns) / 1_000_000.0
                self._record_boundary_serialization_stage(
                    convert_ms,
                    0,
                    channel="ws",
                    route="broadcast",
                    message_type=message_type,
                    stage="numpy_convert",
                    outcome="ok",
                    include_budget=False,
                )
                # Record TX telemetry for tracked messages (e.g., calibration-bundle)
                try:
                    self._record_tx(message)
                except Exception:
                    pass
                encode_start_ns = time.perf_counter_ns()
                message_str = json.dumps(message, separators=(",", ":"))
                payload_bytes = len(message_str.encode("utf-8"))
                encode_ms = (time.perf_counter_ns() - encode_start_ns) / 1_000_000.0
                self._record_boundary_serialization_stage(
                    encode_ms,
                    payload_bytes,
                    channel="ws",
                    route="broadcast",
                    message_type=message_type,
                    stage="json_encode",
                    outcome="ok",
                    include_budget=False,
                )
            elif isinstance(message, bytes):
                # Route binary frames into coalescer; actual sending is handled elsewhere
                await self._coalesce_binary_and_maybe_flush(message)
                return
            elif isinstance(message, str):
                # String message
                message_str = message
            else:
                self.logger.warning(f"Unknown message type: {type(message)}")
                return

            # Send string message to active clients only
            send_start_ns = time.perf_counter_ns()
            results = await asyncio.gather(
                *[client.send(message_str) for client in active_clients],
                return_exceptions=True
            )
            if is_dict_message:
                send_ms = (time.perf_counter_ns() - send_start_ns) / 1_000_000.0
                self._record_boundary_serialization_stage(
                    send_ms,
                    payload_bytes,
                    channel="ws",
                    route="broadcast",
                    message_type=message_type,
                    stage="send_dispatch",
                    outcome="ok",
                    include_budget=False,
                )
                # Keep gate metric aligned with conversion+encode budget only.
                total_ms = float(convert_ms + encode_ms)
                self._record_boundary_serialization_stage(
                    total_ms,
                    payload_bytes,
                    channel="ws",
                    route="broadcast",
                    message_type=message_type,
                    stage="total",
                    outcome="ok",
                    include_budget=True,
                )

            # Check for errors and mark disconnected clients
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    client = active_clients[i] if i < len(active_clients) else None
                    if client:
                        disconnected_clients.append(client)
                    client_ip = client.remote_address if client and hasattr(client, 'remote_address') else "Unknown"
                    # Only log ping timeout errors as debug to reduce noise
                    if "keepalive ping timeout" in str(result):
                        self.logger.debug(f"Client {client_ip} ping timeout - will be cleaned up")
                    else:
                        self.logger.error(f"Failed to send message to {client_ip}: {result}")

        except Exception as e:
            self.logger.error(f"Broadcast error: {e}")
        finally:
            # Immediately clean up disconnected clients from the set
            if disconnected_clients:
                for client in disconnected_clients:
                    if client in self.connected_clients:
                        self.connected_clients.remove(client)
                        client_ip = client.remote_address if hasattr(client, 'remote_address') else "Unknown"
                        self.logger.info(f"Removed disconnected client {client_ip} from connected clients")
                self.logger.debug(f"Cleaned up {len(disconnected_clients)} disconnected clients during broadcast")
    
    def broadcast_sync(self, message):
        """Synchronous version of broadcast for use from other threads
        
        Args:
            message: Message to broadcast
        """
        if not self.event_loop:
            self.logger.error("No event loop available for broadcast_sync")
            return
            
        if not self.running:
            return
        # Guard against closed event loop
        try:
            if self.event_loop.is_closed():
                self.logger.warning("Event loop is closed; dropping broadcast")
                return
        except Exception:
            # If the loop object doesn't implement is_closed, proceed defensively
            pass
            
        # Create a task in the event loop
        asyncio.run_coroutine_threadsafe(
            self.broadcast(message),
            self.event_loop
        )

    def broadcast_frame(self, frame_data: Dict[str, Any]) -> None:
        """
        Synchronous helper so DeepStream pipeline can push frames without asyncio context.
        Converts payload to JSON and re-uses existing broadcast_sync().
        """
        from models import convert_numpy_types
        payload = {
            "type": "frame",
            "payload": convert_numpy_types(frame_data),
        }
        self.broadcast_sync(payload)

    # ---------------- Binary coalescer helpers ----------------
    def _extract_cam_id_from_binary(self, data: bytes) -> Optional[str]:
        try:
            if not data or len(data) < 1:
                return None
            id_len = data[0]
            if len(data) < 1 + id_len:
                return None
            cam_id_bytes = data[1:1 + id_len]
            return cam_id_bytes.decode('utf-8', errors='ignore')
        except Exception:
            return None

    async def _coalesce_binary_and_maybe_flush(self, data: bytes) -> None:
        """Keep only the latest binary frame per camera and schedule a flush if idle."""
        try:
            cam_id = self._extract_cam_id_from_binary(data)
            if cam_id is None:
                # If we cannot parse, fallback to immediate broadcast
                await self._broadcast_binary_immediate(data)
                return
            # Coalesce latest per camera
            self._latest_binary_by_cam[cam_id] = data

            # Schedule a flush if not currently sending
            if not self._binary_sending:
                # Avoid creating multiple concurrent flushers
                if self._binary_flush_task is None or self._binary_flush_task.done():
                    self._binary_flush_task = asyncio.create_task(self._flush_binary_queue(), name="BinaryFlush")
        except Exception as e:
            self.logger.debug(f"Coalescer error: {e}")

    async def _broadcast_binary_immediate(self, data: bytes) -> None:
        """Fallback path to send a single binary payload to all clients."""
        if not self.connected_clients:
            return
        active_clients = list(self.connected_clients)
        results = await asyncio.gather(
            *[client.send(data) for client in active_clients],
            return_exceptions=True
        )
        disconnected_clients = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                client = active_clients[i] if i < len(active_clients) else None
                if client:
                    disconnected_clients.append(client)
                client_ip = client.remote_address if client and hasattr(client, 'remote_address') else "Unknown"
                if "keepalive ping timeout" in str(result).lower():
                    self.logger.debug(f"Client {client_ip} ping timeout - will be cleaned up")
                else:
                    self.logger.error(f"Failed to send binary message to {client_ip}: {result}")
        # Cleanup
        if disconnected_clients:
            for client in disconnected_clients:
                if client in self.connected_clients:
                    self.connected_clients.remove(client)
                    client_ip = client.remote_address if hasattr(client, 'remote_address') else "Unknown"
                    self.logger.info(f"Removed disconnected client {client_ip} from connected clients")

    async def _flush_binary_queue(self) -> None:
        """Flush latest binary frames per camera to all clients, dropping superseded frames."""
        if self._binary_sending:
            return
        self._binary_sending = True
        try:
            # Loop while there is data and server running
            while self.running and self._latest_binary_by_cam:
                if not self.connected_clients:
                    # No clients; keep latest for later but don't spin
                    await asyncio.sleep(0.05)
                    continue
                # Snapshot current latest and clear for coalescing of new arrivals
                batch = list(self._latest_binary_by_cam.items())
                self._latest_binary_by_cam.clear()

                # Send each camera's latest frame once
                for cam_id, payload in batch:
                    await self._broadcast_binary_immediate(payload)

                # Yield control to allow new coalescing
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Binary flush error: {e}")
        finally:
            self._binary_sending = False


class WebSocketClient:
    """WebSocket client for testing the server"""
    
    def __init__(self, uri: str = "ws://localhost:6008"):
        """Initialize the WebSocket client
        
        Args:
            uri: WebSocket server URI
        """
        self.uri = uri
        self.websocket = None
        self.running = False
        self.logger = logging.getLogger("WebSocketClient")
    
    async def connect(self):
        """Connect to the WebSocket server"""
        try:
            self.websocket = await websockets.connect(self.uri)
            self.running = True
            self.logger.info(f"Connected to {self.uri}")
            return True
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from the WebSocket server"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            self.running = False
            self.logger.info("Disconnected")
    
    async def send(self, message):
        """Send a message to the server
        
        Args:
            message: Message to send (dict or string)
        """
        if not self.websocket:
            self.logger.error("Not connected")
            return
            
        try:
            # Convert dict to JSON string
            if isinstance(message, dict):
                message = json.dumps(message)
                
            await self.websocket.send(message)
        except Exception as e:
            self.logger.error(f"Send error: {e}")
    
    async def receive(self):
        """Receive a message from the server
        
        Returns:
            The received message
        """
        if not self.websocket:
            self.logger.error("Not connected")
            return None
            
        try:
            message = await self.websocket.recv()
            
            # Try to parse as JSON
            try:
                return json.loads(message)
            except json.JSONDecodeError:
                return message
        except Exception as e:
            self.logger.error(f"Receive error: {e}")
            return None
    
    async def listen(self, callback):
        """Listen for messages from the server
        
        Args:
            callback: Function to call with received messages
        """
        if not self.websocket:
            self.logger.error("Not connected")
            return
            
        self.running = True
        
        try:
            while self.running:
                message = await self.receive()
                if message:
                    callback(message)
        except Exception as e:
            self.logger.error(f"Listen error: {e}")
        finally:
            self.running = False 
    
