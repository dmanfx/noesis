from typing import Dict, Optional, Any, Callable
import asyncio
import websockets
import json
import logging

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
    
    async def _periodic_stats_broadcast(self, interval_seconds: float = 1.0):
        """Periodically fetches and broadcasts stats."""
        self.logger.info(f"Starting periodic stats broadcast every {interval_seconds} seconds.")
        while self.running:
            try:
                if self.stats_callback and self.connected_clients: # Only send if callback exists and clients are connected
                    stats_payload = self.stats_callback()
                    if stats_payload: # Ensure callback returned something
                        stats_message = {
                            'type': 'stats',
                            'payload': stats_payload
                        }
                        await self.broadcast(stats_message)
                    else:
                        self.logger.debug("Stats callback returned empty payload, skipping broadcast.")
                elif not self.connected_clients:
                    self.logger.debug("No clients connected, skipping stats broadcast.")
                    
                # Wait for the next interval
                await asyncio.sleep(interval_seconds)
                
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
            # Create server
            self.server = await websockets.serve(
                self.handle_client,
                self.host,
                self.port
            )
            
            # Create server task
            self.server_task = asyncio.create_task(
                self.server.wait_closed(), 
                name="WebSocketServerWaitClosed"
            )

            # Start periodic stats broadcast task if callback is provided
            if self.stats_callback:
                self._stats_task = asyncio.create_task(
                    self._periodic_stats_broadcast(),
                    name="PeriodicStatsBroadcast"
                )
            
            self.logger.info(f"WebSocket server running on {self.host}:{self.port}")
            print(f"🚀 WebSocket server is LIVE on {self.host}:{self.port}")
            
        except OSError as e:
            self.logger.error(f"Failed to start server (Port {self.port} likely in use): {e}")
            raise
        except Exception as e:
            self.logger.error(f"Server failed: {e}")
            raise
    
    async def stop(self):
        """Gracefully stop the WebSocket server and the periodic stats broadcast."""
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
            close_tasks = [client.close() for client in self.connected_clients]
            await asyncio.gather(*close_tasks, return_exceptions=True)
            self.connected_clients.clear()
        
        # Close server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.server = None
            self.logger.info("WebSocket server stopped")
    
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
            self.logger.info(f"Client {client_ip} disconnected normally.")
            print(f"❌ Client {client_ip} disconnected normally.")
        except websockets.exceptions.ConnectionClosedError as e:
            self.logger.info(f"Client {client_ip} disconnected with error: {e}")
            print(f"❌ Client {client_ip} disconnected with error: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error with client {client_ip}: {e}")
        finally:
            # Ensure client is removed from set
            if websocket in self.connected_clients:
                self.connected_clients.remove(websocket)
                self.logger.info(f"Client {client_ip} removed. Total clients: {len(self.connected_clients)}")
                print(f"👋 Client {client_ip} removed. Total clients: {len(self.connected_clients)}")
    
    async def broadcast(self, message):
        """Broadcast a message to all connected WebSocket clients
        
        Args:
            message: Message to broadcast (dict, bytes, or string)
        """
        if not self.connected_clients:
            return
            
        message_str = ""
        
        try:
            # Prepare message based on type
            if isinstance(message, dict):
                # Convert to JSON string
                message = convert_numpy_types(message)
                message_str = json.dumps(message)
            elif isinstance(message, bytes):
                # Binary message
                results = await asyncio.gather(
                    *[client.send(message) for client in self.connected_clients],
                    return_exceptions=True
                )
                # Check for errors
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        client = list(self.connected_clients)[i] if i < len(self.connected_clients) else None
                        client_ip = client.remote_address if client and hasattr(client, 'remote_address') else "Unknown"
                        self.logger.error(f"Failed to send binary message to {client_ip}: {result}")
                return
            elif isinstance(message, str):
                # String message
                message_str = message
            else:
                self.logger.warning(f"Unknown message type: {type(message)}")
                return
                
            # Send string message to all clients
            results = await asyncio.gather(
                *[client.send(message_str) for client in self.connected_clients],
                return_exceptions=True
            )
            
            # Check for errors
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    client = list(self.connected_clients)[i] if i < len(self.connected_clients) else None
                    client_ip = client.remote_address if client and hasattr(client, 'remote_address') else "Unknown"
                    self.logger.error(f"Failed to send message to {client_ip}: {result}")
        
        except Exception as e:
            self.logger.error(f"Broadcast error: {e}")
    
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
    