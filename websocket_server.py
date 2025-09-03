from typing import Dict, Optional, Any, Callable
import asyncio
import websockets
import json
import logging
import time

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
                                self.logger.info(f"📡 Stats sent to {len(self.connected_clients)} clients | cameras={cam_count} | uptime={uptime:.1f}s")
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

            # Start periodic stats broadcast task if callback is provided
            if self.stats_callback:
                self._stats_task = asyncio.create_task(
                    self._periodic_stats_broadcast(),
                    name="PeriodicStatsBroadcast"
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

        # Cancel server task if it exists
        if self.server_task and not self.server_task.done():
            self.server_task.cancel()
            try:
                await self.server_task
            except asyncio.CancelledError:
                pass
    
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

            # Process messages from client
            async for message in websocket:
                try:
                    data = json.loads(message)

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
        disconnected_clients = set()
        active_clients = set(self.connected_clients)  # Create a copy to iterate over

        try:
            # Prepare message based on type
            if isinstance(message, dict):
                # Convert to JSON string
                message = convert_numpy_types(message)
                message_str = json.dumps(message)
            elif isinstance(message, bytes):
                # Binary message - send to active clients only
                results = await asyncio.gather(
                    *[client.send(message) for client in active_clients],
                    return_exceptions=True
                )
                # Check for errors and mark disconnected clients
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        client = list(active_clients)[i] if i < len(active_clients) else None
                        if client:
                            disconnected_clients.add(client)
                        client_ip = client.remote_address if client and hasattr(client, 'remote_address') else "Unknown"
                        # Only log ping timeout errors to reduce noise, but still handle all exceptions
                        if "keepalive ping timeout" in str(result):
                            self.logger.debug(f"Client {client_ip} ping timeout - will be cleaned up")
                        else:
                            self.logger.error(f"Failed to send binary message to {client_ip}: {result}")
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
                    client = list(active_clients)[i] if i < len(active_clients) else None
                    if client:
                        disconnected_clients.add(client)
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
    
