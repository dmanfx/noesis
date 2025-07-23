"""
Event-Driven I/O Module for GPU Pipeline
========================================

This module provides event-driven networking and I/O operations to replace
polling-based approaches, reducing CPU usage by eliminating busy-wait loops.

Key Features:
- Event-driven RTSP reconnection
- Async I/O for network operations
- Callback-based frame processing
- Automatic backoff strategies
- Zero CPU usage when idle
"""

import asyncio
import threading
import time
import logging
import queue
from typing import Optional, Callable, Dict, Any, Tuple
from dataclasses import dataclass
from collections import deque
import subprocess
import select
import numpy as np
import torch

# Import backoff strategies if available
try:
    from exponential_backoff import ExponentialBackoff
    BACKOFF_AVAILABLE = True
except ImportError:
    BACKOFF_AVAILABLE = False


@dataclass
class ConnectionState:
    """Tracks connection state for event-driven management"""
    connected: bool = False
    connecting: bool = False
    last_attempt: float = 0.0
    attempts: int = 0
    last_error: Optional[str] = None
    backoff_delay: float = 1.0


class EventDrivenVideoReader:
    """
    Event-driven video reader that eliminates polling for frame availability.
    Uses async I/O and callbacks for efficient CPU usage.
    """
    
    def __init__(self, 
                 source: str,
                 width: int = 1920,
                 height: int = 1080,
                 frame_callback: Optional[Callable[[np.ndarray], None]] = None,
                 error_callback: Optional[Callable[[str], None]] = None,
                 use_cuda: bool = True):
        """
        Initialize event-driven video reader.
        
        Args:
            source: Video source (RTSP URL or file path)
            width: Frame width
            height: Frame height
            frame_callback: Callback for processed frames
            error_callback: Callback for errors
            use_cuda: Whether to use CUDA acceleration
        """
        self.source = source
        self.width = width
        self.height = height
        self.frame_callback = frame_callback
        self.error_callback = error_callback
        self.use_cuda = use_cuda
        
        # State management
        self.state = ConnectionState()
        self.running = False
        
        # Event loop for async operations
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.loop_thread: Optional[threading.Thread] = None
        
        # FFmpeg process
        self.process: Optional[subprocess.Popen] = None
        
        # Frame processing
        self.frame_size = width * height * 3  # BGR24
        self.frame_buffer = bytearray()
        
        # Backoff strategy
        if BACKOFF_AVAILABLE:
            self.backoff = ExponentialBackoff(
                initial_delay=1.0,
                max_delay=30.0,
                factor=2.0
            )
        else:
            self.backoff = None
        
        # Logger
        self.logger = logging.getLogger(f"EventDrivenVideoReader.{source}")
        
    def start(self):
        """Start the event-driven video reader"""
        if self.running:
            return
            
        self.running = True
        
        # Start event loop in separate thread
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(
            target=self._run_event_loop,
            daemon=True
        )
        self.loop_thread.start()
        
        # Schedule initial connection
        asyncio.run_coroutine_threadsafe(
            self._connect(),
            self.loop
        )
        
        self.logger.info("Started event-driven video reader")
        
    def _run_event_loop(self):
        """Run the async event loop"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
        
    async def _connect(self):
        """Async connection with automatic retry and backoff"""
        while self.running and not self.state.connected:
            try:
                self.state.connecting = True
                self.state.last_attempt = time.time()
                
                # Start FFmpeg process
                await self._start_ffmpeg_async()
                
                # Start reading frames
                asyncio.create_task(self._read_frames_async())
                
                self.state.connected = True
                self.state.connecting = False
                self.state.attempts = 0
                self.logger.info("Successfully connected to video source")
                
            except Exception as e:
                self.state.connecting = False
                self.state.attempts += 1
                self.state.last_error = str(e)
                
                # Calculate backoff delay
                if self.backoff:
                    delay = self.backoff.get_delay(self.state.attempts)
                else:
                    delay = min(30.0, 2.0 ** self.state.attempts)
                
                self.logger.warning(
                    f"Connection failed (attempt {self.state.attempts}): {e}. "
                    f"Retrying in {delay:.1f}s"
                )
                
                # Schedule retry with backoff
                await asyncio.sleep(delay)
                
    async def _start_ffmpeg_async(self):
        """Start FFmpeg process asynchronously"""
        cmd = self._build_ffmpeg_command()
        
        # Create process with async stdout
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Start stderr monitoring
        asyncio.create_task(self._monitor_stderr_async())
        
        # Wait a bit to ensure process started
        await asyncio.sleep(0.5)
        
        # Check if process is still running
        if self.process.returncode is not None:
            stderr = await self._read_stderr_async()
            raise RuntimeError(f"FFmpeg process died immediately: {stderr}")
            
    def _build_ffmpeg_command(self) -> list:
        """Build FFmpeg command for hardware acceleration"""
        cmd = ['ffmpeg']
        
        if self.use_cuda:
            cmd.extend([
                '-hwaccel', 'cuda',
                '-hwaccel_device', '0',
                '-c:v', 'h264_cuvid'
            ])
        
        cmd.extend([
            '-rtsp_transport', 'tcp',
            '-i', self.source,
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-an',
            '-sn',
            '-v', 'error',
            '-'
        ])
        
        return cmd
        
    async def _read_frames_async(self):
        """Async frame reading - no polling!"""
        try:
            while self.running and self.state.connected:
                # Read data when available (async, no polling)
                chunk = await self.process.stdout.read(65536)
                
                if not chunk:
                    # EOF reached
                    self.logger.warning("EOF reached on video stream")
                    self.state.connected = False
                    await self._handle_disconnection()
                    break
                    
                # Add to buffer
                self.frame_buffer.extend(chunk)
                
                # Extract complete frames
                while len(self.frame_buffer) >= self.frame_size:
                    frame_data = self.frame_buffer[:self.frame_size]
                    self.frame_buffer = self.frame_buffer[self.frame_size:]
                    
                    # Process frame
                    try:
                        frame = np.frombuffer(frame_data, dtype=np.uint8)
                        frame = frame.reshape((self.height, self.width, 3))
                        
                        # Call frame callback in thread pool to avoid blocking
                        if self.frame_callback:
                            await self.loop.run_in_executor(
                                None,
                                self.frame_callback,
                                frame
                            )
                            
                    except Exception as e:
                        self.logger.error(f"Error processing frame: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error in frame reading: {e}")
            self.state.connected = False
            await self._handle_disconnection()
            
    async def _monitor_stderr_async(self):
        """Monitor FFmpeg stderr asynchronously"""
        try:
            while self.process and self.process.returncode is None:
                line = await self.process.stderr.readline()
                if line:
                    stderr_msg = line.decode().strip()
                    if stderr_msg and any(
                        keyword in stderr_msg.lower() 
                        for keyword in ['error', 'failed', 'invalid']
                    ):
                        self.logger.warning(f"FFmpeg: {stderr_msg}")
                        
        except Exception as e:
            self.logger.debug(f"Error monitoring stderr: {e}")
            
    async def _read_stderr_async(self) -> str:
        """Read all available stderr"""
        if self.process and self.process.stderr:
            try:
                stderr_data = await self.process.stderr.read()
                return stderr_data.decode() if stderr_data else ""
            except:
                return ""
        return ""
        
    async def _handle_disconnection(self):
        """Handle disconnection and schedule reconnection"""
        self.logger.warning("Handling disconnection")
        
        # Clean up current process
        if self.process:
            try:
                self.process.terminate()
                await self.process.wait()
            except:
                pass
            self.process = None
            
        # Clear buffers
        self.frame_buffer.clear()
        
        # Notify error callback
        if self.error_callback:
            await self.loop.run_in_executor(
                None,
                self.error_callback,
                "Connection lost"
            )
            
        # Schedule reconnection
        if self.running:
            asyncio.create_task(self._connect())
            
    def stop(self):
        """Stop the event-driven reader"""
        self.running = False
        self.state.connected = False
        
        # Stop FFmpeg process
        if self.process:
            try:
                self.process.terminate()
            except:
                pass
                
        # Stop event loop
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
            
        # Wait for thread
        if self.loop_thread:
            self.loop_thread.join(timeout=2.0)
            
        self.logger.info("Stopped event-driven video reader")
        
    def get_state(self) -> Dict[str, Any]:
        """Get current connection state"""
        return {
            'connected': self.state.connected,
            'connecting': self.state.connecting,
            'attempts': self.state.attempts,
            'last_error': self.state.last_error,
            'uptime': time.time() - self.state.last_attempt if self.state.connected else 0
        }


class EventDrivenGPUVideoReader(EventDrivenVideoReader):
    """
    GPU-optimized event-driven video reader that outputs GPU tensors.
    """
    
    def __init__(self,
                 source: str,
                 width: int = 1920,
                 height: int = 1080,
                 tensor_callback: Optional[Callable[[torch.Tensor], None]] = None,
                 error_callback: Optional[Callable[[str], None]] = None):
        """
        Initialize GPU-optimized event-driven reader.
        
        Args:
            source: Video source
            width: Frame width
            height: Frame height
            tensor_callback: Callback for GPU tensors
            error_callback: Callback for errors
        """
        # Override frame callback with tensor conversion
        super().__init__(
            source=source,
            width=width,
            height=height,
            frame_callback=None,  # We'll handle this internally
            error_callback=error_callback,
            use_cuda=True
        )
        
        self.tensor_callback = tensor_callback
        self._setup_gpu_processing()
        
    def _setup_gpu_processing(self):
        """Setup GPU processing pipeline"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available for GPU video reader")
            
        # Pre-allocate GPU tensors for zero-copy
        self.gpu_tensor_pool = []
        for _ in range(3):  # Pool of 3 tensors
            tensor = torch.empty(
                (3, self.height, self.width),
                dtype=torch.float32,
                device='cuda'
            )
            self.gpu_tensor_pool.append(tensor)
            
        self.tensor_pool_idx = 0
        
    async def _read_frames_async(self):
        """Override to process frames as GPU tensors"""
        try:
            while self.running and self.state.connected:
                # Read data when available (async, no polling)
                chunk = await self.process.stdout.read(65536)
                
                if not chunk:
                    self.logger.warning("EOF reached on video stream")
                    self.state.connected = False
                    await self._handle_disconnection()
                    break
                    
                # Add to buffer
                self.frame_buffer.extend(chunk)
                
                # Extract complete frames
                while len(self.frame_buffer) >= self.frame_size:
                    frame_data = self.frame_buffer[:self.frame_size]
                    self.frame_buffer = self.frame_buffer[self.frame_size:]
                    
                    # Process frame as GPU tensor
                    try:
                        # Get tensor from pool
                        gpu_tensor = self.gpu_tensor_pool[self.tensor_pool_idx]
                        self.tensor_pool_idx = (self.tensor_pool_idx + 1) % len(self.gpu_tensor_pool)
                        
                        # Convert to numpy and copy to GPU tensor
                        frame = np.frombuffer(frame_data, dtype=np.uint8)
                        frame = frame.reshape((self.height, self.width, 3))
                        
                        # BGR to RGB and normalize
                        frame_rgb = frame[:, :, [2, 1, 0]].astype(np.float32) / 255.0
                        
                        # Copy to GPU tensor (CHW format)
                        gpu_tensor.copy_(
                            torch.from_numpy(frame_rgb).permute(2, 0, 1)
                        )
                        
                        # Call tensor callback
                        if self.tensor_callback:
                            await self.loop.run_in_executor(
                                None,
                                self.tensor_callback,
                                gpu_tensor
                            )
                            
                    except Exception as e:
                        self.logger.error(f"Error processing GPU tensor: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error in GPU frame reading: {e}")
            self.state.connected = False
            await self._handle_disconnection()


def integrate_event_driven_reader(pipeline, reader_class=EventDrivenGPUVideoReader):
    """
    Integrate event-driven reader into existing pipeline.
    
    Args:
        pipeline: Pipeline instance to integrate with
        reader_class: Reader class to use (default: EventDrivenGPUVideoReader)
    """
    # Replace polling-based reader with event-driven
    if hasattr(pipeline, 'nvdec_reader'):
        # Stop old reader
        if pipeline.nvdec_reader:
            pipeline.nvdec_reader.stop()
            
        # Create frame queue for compatibility
        frame_queue = queue.Queue(maxsize=10)
        
        # Tensor callback that adds to queue
        def tensor_callback(tensor):
            try:
                frame_queue.put(tensor.clone(), block=False)
            except queue.Full:
                pass  # Drop frame if queue is full
                
        # Error callback
        def error_callback(error):
            pipeline.logger.error(f"Event-driven reader error: {error}")
            
        # Create event-driven reader
        event_reader = reader_class(
            source=pipeline.source,
            width=pipeline.config.cameras.CAMERA_WIDTH,
            height=pipeline.config.cameras.CAMERA_HEIGHT,
            tensor_callback=tensor_callback,
            error_callback=error_callback
        )
        
        # Start reader
        event_reader.start()
        
        # Replace read method
        def read_gpu_tensor():
            try:
                tensor = frame_queue.get(timeout=1.0)
                return True, tensor
            except queue.Empty:
                return False, None
                
        # Monkey-patch the reader
        pipeline.nvdec_reader = type('EventDrivenReaderWrapper', (), {
            'read_gpu_tensor': read_gpu_tensor,
            'stop': event_reader.stop,
            'get_stats': lambda: event_reader.get_state()
        })()
        
        pipeline.logger.info("Integrated event-driven reader into pipeline")


# Example usage
if __name__ == "__main__":
    import cv2
    
    # Frame counter for testing
    frame_count = 0
    
    def frame_callback(frame):
        global frame_count
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Received frame {frame_count}: {frame.shape}")
            
    def error_callback(error):
        print(f"Error: {error}")
        
    # Create event-driven reader
    reader = EventDrivenVideoReader(
        source="rtsp://example.com/stream",
        frame_callback=frame_callback,
        error_callback=error_callback
    )
    
    # Start reader
    reader.start()
    
    # Run for a while
    try:
        time.sleep(30)
    except KeyboardInterrupt:
        pass
        
    # Stop reader
    reader.stop()
    print(f"Total frames received: {frame_count}") 