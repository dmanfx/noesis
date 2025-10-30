from typing import Dict, Optional, Any, Callable
import asyncio
import concurrent.futures
import websockets
import json
import logging
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
        self.ma_depth_provider: Optional[Callable[[str, Optional[int]], Optional[Dict[str, Any]]]] = None
        self.floorplan_provider: Optional[Callable[[Optional[list], float, float, float, bool], Optional[Dict[str, Any]]]] = None

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
                self.logger.debug(f"MENON I/O | {' | '.join(parts)}")
            except asyncio.CancelledError:
                break
            except Exception:
                # Never fail loop due to logging issues
                pass
            finally:
                try:
                    await asyncio.sleep(interval_seconds)
                except Exception:
                    await asyncio.sleep(1.0)
    
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
                    self.logger.debug("No clients connected, skipping stats broadcast.")

                # Wait for the next interval - use longer interval when no clients
                if not self.connected_clients:
                    await asyncio.sleep(5.0)  # 5 second interval when no clients
                else:
                    await asyncio.sleep(interval_seconds)  # Normal 1 second interval with clients

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
            # Remember the running loop for cross-thread broadcasts
            try:
                self.event_loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop; will be set by caller if needed
                self.event_loop = None

            # Create server with backpressure and heartbeat settings
            # Increased ping interval and timeout for better stability
            self.server = await websockets.serve(
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

        # Cancel any pending binary flush task
        if self._binary_flush_task and not self._binary_flush_task.done():
            try:
                self._binary_flush_task.cancel()
            except Exception:
                pass

        # Cancel the periodic stats task first
        if self._stats_task and not self._stats_task.done():
            self._stats_task.cancel()
            try:
                await self._stats_task # Wait for cancellation
            except asyncio.CancelledError:
                self.logger.info("Periodic stats broadcast task successfully cancelled.")
            except Exception as e:
                 self.logger.error(f"Error waiting for stats task cancellation: {e}")

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

        # Cancel Menon telemetry task
        if self._telemetry_task and not self._telemetry_task.done():
            self._telemetry_task.cancel()
            try:
                await self._telemetry_task
            except asyncio.CancelledError:
                pass

        # Cancel server task if it exists
        if self.server_task and not self.server_task.done():
            self.server_task.cancel()
            try:
                await self.server_task
            except asyncio.CancelledError:
                pass

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

    def stop_sync(self, timeout: float = 5.0) -> bool:
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
            return self._force_close_server()

    async def handle_client(self, websocket, path=None):
        """Handle incoming WebSocket connections and messages

        Args:
            websocket: WebSocket connection
            path: WebSocket path
        """
        client_ip = websocket.remote_address[0] if hasattr(websocket, 'remote_address') else "Unknown"
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
                        cam_id = data.get('camId') or data.get('cameraId')
                        ts_max_val = data.get('ts_max') if data.get('ts_max') is not None else data.get('tsMax')
                        ts_max = None
                        if ts_max_val is not None:
                            try:
                                ts = int(ts_max_val)
                                # Normalize to microseconds:
                                # <1e10 => seconds, <1e13 => milliseconds, else assume microseconds
                                if ts < 10_000_000_000:
                                    ts *= 1_000_000
                                elif ts < 10_000_000_000_000:
                                    ts *= 1_000
                                ts_max = ts
                            except Exception:
                                ts_max = None

                        rate_key = f"{client_ip}:{cam_id or 'unknown'}"
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

                        result = {'type': 'ma_depth_response', 'cam_id': cam_id}
                        if cam_id and callable(self.ma_depth_provider):
                            try:
                                payload = await asyncio.wait_for(
                                    asyncio.to_thread(self.ma_depth_provider, cam_id, ts_max),
                                    timeout=self._depth_rpc_timeout
                                )
                                if payload:
                                    result.update(payload)
                                    result['ok'] = True
                                    # Pre-serialize heavy payload off loop
                                    message_text = await asyncio.to_thread(json.dumps, result)
                                    await websocket.send(message_text)
                                else:
                                    result.update({'ok': False, 'error': 'not_available'})
                                    await websocket.send(json.dumps(result))
                            except asyncio.TimeoutError:
                                self.logger.warning(f"Depth RPC timed out for {cam_id} from {client_ip}")
                                result.update({'ok': False, 'error': 'timeout'})
                                await websocket.send(json.dumps(result))
                            except Exception as exc:
                                result.update({'ok': False, 'error': str(exc)})
                                await websocket.send(json.dumps(result))
                        else:
                            result.update({'ok': False, 'error': 'no_provider'})
                            await websocket.send(json.dumps(result))

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

                        result = {
                            'type': 'floorplan_response',
                            'request_id': request_id,
                            'camera_id': camera,
                            'cache_only': cache_only,
                        }

                        provider = getattr(self, 'floorplan_provider', None)
                        if callable(provider):
                            try:
                                payload = await asyncio.wait_for(
                                    asyncio.to_thread(provider, camera, max_age_sec, grid_res_m, max_extent_m, cache_only=cache_only),
                                    timeout=self._floorplan_rpc_timeout
                                )
                                if payload:
                                    result.update(payload)
                                    message_text = await asyncio.to_thread(json.dumps, result)
                                    await websocket.send(message_text)
                                else:
                                    result['error'] = 'no_payload'
                                    await websocket.send(json.dumps(result))
                            except asyncio.TimeoutError:
                                self.logger.warning(f"Floorplan RPC timed out for {camera} from {client_ip}")
                                result['error'] = 'timeout'
                                await websocket.send(json.dumps(result))
                            except Exception as exc:
                                result['error'] = str(exc)
                                self.logger.error(f"Floorplan generation error: {exc}")
                                await websocket.send(json.dumps(result))
                        else:
                            result['error'] = 'no_provider'
                            await websocket.send(json.dumps(result))

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

        try:
            # Prepare message based on type
            if isinstance(message, dict):
                # Convert to JSON string
                message = convert_numpy_types(message)
                # Record TX telemetry for tracked messages (e.g., calibration-bundle)
                try:
                    self._record_tx(message)
                except Exception:
                    pass
                message_str = json.dumps(message)
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
            results = await asyncio.gather(
                *[client.send(message_str) for client in active_clients],
                return_exceptions=True
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
    
