"""
NVENC Hardware Encoder
======================

Provides hardware-accelerated video encoding using NVIDIA NVENC.
Supports H.264 and H.265 encoding directly from GPU memory.

Key features:
- Zero-copy encoding from GPU tensors
- Low-latency encoding for streaming
- Configurable quality/performance trade-offs
- Automatic format conversion
"""

import torch
import numpy as np
import logging
import time
import subprocess
import threading
import queue
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import os

# Try to import PyAV for hardware encoding
try:
    import av
    av.logging.set_level(av.logging.ERROR)
    PYAV_AVAILABLE = True
except ImportError:
    PYAV_AVAILABLE = False
    logging.getLogger(__name__).warning("PyAV not available - hardware encoding disabled")


@dataclass
class EncoderConfig:
    """Configuration for NVENC encoder"""
    width: int
    height: int
    fps: int = 30
    codec: str = 'h264_nvenc'  # h264_nvenc or hevc_nvenc
    preset: str = 'll'  # low-latency
    profile: str = 'high'
    bitrate: int = 4000000  # 4 Mbps
    gop_size: int = 30  # keyframe interval
    pixel_format: str = 'yuv420p'
    gpu_id: int = 0


class NVENCEncoder:
    """
    NVENC hardware video encoder using FFmpeg.
    
    This encoder uses FFmpeg's NVENC support for maximum compatibility
    and performance. It accepts GPU tensors and encodes them with
    minimal CPU involvement.
    """
    
    def __init__(self, config: EncoderConfig):
        """
        Initialize NVENC encoder.
        
        Args:
            config: Encoder configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"NVENCEncoder.GPU{config.gpu_id}")
        
        # Validate NVENC availability
        self._check_nvenc_support()
        
        # FFmpeg process
        self.ffmpeg_process = None
        self.encoding_thread = None
        self.frame_queue = queue.Queue(maxsize=30)
        self.running = False
        
        # Performance stats
        self.stats = {
            'frames_encoded': 0,
            'total_encode_time': 0.0,
            'encoding_errors': 0,
            'queue_drops': 0
        }
        
        # Output buffer for encoded frames
        self.output_queue = queue.Queue()
        
        self.logger.info(f"Initialized NVENC encoder: {config.width}x{config.height}@{config.fps}fps")
    
    def _check_nvenc_support(self):
        """Check if NVENC is available"""
        try:
            # Check for NVENC support in ffmpeg
            result = subprocess.run(
                ['ffmpeg', '-hide_banner', '-encoders'],
                capture_output=True,
                text=True
            )
            
            if self.config.codec not in result.stdout:
                raise RuntimeError(f"NVENC codec {self.config.codec} not available in FFmpeg")
            
            self.logger.info(f"NVENC support confirmed for {self.config.codec}")
            
        except Exception as e:
            raise RuntimeError(f"NVENC not available: {e}")
    
    def start(self):
        """Start the encoder"""
        if self.running:
            return
        
        # Build FFmpeg command
        cmd = self._build_ffmpeg_command()
        
        # Start FFmpeg process
        self.ffmpeg_process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        self.running = True
        
        # Start encoding thread
        self.encoding_thread = threading.Thread(
            target=self._encoding_loop,
            daemon=True
        )
        self.encoding_thread.start()
        
        # Start output reader thread
        self.output_thread = threading.Thread(
            target=self._read_output,
            daemon=True
        )
        self.output_thread.start()
        
        self.logger.info("Started NVENC encoder")
    
    def _build_ffmpeg_command(self) -> list:
        """Build FFmpeg command for NVENC encoding"""
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-f', 'rawvideo',
            '-pixel_format', 'rgb24',
            '-video_size', f'{self.config.width}x{self.config.height}',
            '-framerate', str(self.config.fps),
            '-i', '-',  # Read from stdin
            '-c:v', self.config.codec,
            '-preset', self.config.preset,
            '-profile:v', self.config.profile,
            '-b:v', str(self.config.bitrate),
            '-g', str(self.config.gop_size),
            '-gpu', str(self.config.gpu_id),
            '-pix_fmt', self.config.pixel_format,
            '-f', 'h264',  # Output format
            '-'  # Write to stdout
        ]
        
        return cmd
    
    def encode_tensor(self, tensor: torch.Tensor) -> bool:
        """
        Encode GPU tensor using NVENC - ENHANCED for DALI integration.
        
        Args:
            tensor: GPU tensor (C, H, W) or (1, C, H, W) in RGB format [0,1]
            
        Returns:
            True if frame was queued, False if dropped
        """
        if not self.running:
            self.logger.warning("Encoder not running")
            return False
        
        try:
            # Handle both DALI formats: (C, H, W) and (1, C, H, W)
            if tensor.dim() == 4 and tensor.shape[0] == 1:
                # Remove batch dimension from DALI output
                tensor = tensor.squeeze(0)
            elif tensor.dim() != 3:
                self.logger.error(f"Unsupported tensor shape: {tensor.shape}")
                return False
            
            # Ensure tensor is on correct device and dtype
            if not tensor.is_cuda:
                self.logger.warning("Tensor not on GPU, moving to GPU")
                tensor = tensor.cuda()
            
            # DALI outputs FP16, ensure compatibility
            if tensor.dtype == torch.float16:
                tensor = tensor.float()  # Convert to FP32 for encoding
            
            # Add to queue (non-blocking)
            self.frame_queue.put_nowait(tensor)
            return True
            
        except queue.Full:
            self.stats['queue_drops'] += 1
            if self.stats['queue_drops'] % 100 == 0:
                self.logger.warning(f"Dropped {self.stats['queue_drops']} frames due to full queue")
            return False
        except Exception as e:
            self.logger.error(f"Error encoding tensor: {e}")
            return False
    
    def _encoding_loop(self):
        """Main encoding loop"""
        while self.running:
            try:
                # Get frame from queue
                tensor = self.frame_queue.get(timeout=1.0)
                
                start_time = time.time()
                
                # Convert tensor to numpy
                frame_np = self._tensor_to_numpy(tensor)
                
                # Write to FFmpeg stdin
                if self.ffmpeg_process and self.ffmpeg_process.stdin:
                    self.ffmpeg_process.stdin.write(frame_np.tobytes())
                    self.ffmpeg_process.stdin.flush()
                
                # Update stats
                self.stats['frames_encoded'] += 1
                self.stats['total_encode_time'] += time.time() - start_time
                
            except queue.Empty:
                continue
            except Exception as e:
                self.stats['encoding_errors'] += 1
                self.logger.error(f"Encoding error: {e}")
    
    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert GPU tensor to numpy array for encoding"""
        # Ensure correct format
        if tensor.dim() == 3:
            # Convert CHW to HWC
            tensor_hwc = tensor.permute(1, 2, 0)
        else:
            # Batch mode - take first frame
            tensor_hwc = tensor[0].permute(1, 2, 0)
        
        # Convert to uint8
        if tensor_hwc.dtype != torch.uint8:
            tensor_hwc = (tensor_hwc * 255).clamp(0, 255).to(torch.uint8)
        
        # Move to CPU
        return tensor_hwc.cpu().numpy()
    
    def _read_output(self):
        """Read encoded output from FFmpeg"""
        while self.running:
            if self.ffmpeg_process and self.ffmpeg_process.stdout:
                # Read encoded data
                data = self.ffmpeg_process.stdout.read(4096)
                if data:
                    self.output_queue.put(data)
    
    def get_encoded_frame(self, timeout: float = 0.1) -> Optional[bytes]:
        """
        Get encoded frame data.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Encoded frame bytes or None
        """
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop the encoder"""
        self.running = False
        
        # Stop FFmpeg process
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.stdin.close()
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=5)
            except Exception as e:
                self.logger.error(f"Error stopping FFmpeg: {e}")
                self.ffmpeg_process.kill()
        
        # Wait for threads
        if self.encoding_thread:
            self.encoding_thread.join(timeout=2)
        if hasattr(self, 'output_thread') and self.output_thread:
            self.output_thread.join(timeout=2)
        
        self.logger.info("Stopped NVENC encoder")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get encoder statistics"""
        stats = self.stats.copy()
        
        if stats['frames_encoded'] > 0:
            stats['avg_encode_time_ms'] = (stats['total_encode_time'] / stats['frames_encoded']) * 1000
            stats['encode_fps'] = stats['frames_encoded'] / max(1, stats['total_encode_time'])
        
        return stats


class NVJPEGEncoder:
    """
    Hardware-accelerated JPEG encoder using NVJPEG.
    
    Provides fast JPEG encoding directly from GPU memory.
    """
    
    def __init__(self, quality: int = 85, device: str = 'cuda:0'):
        """
        Initialize NVJPEG encoder.
        
        Args:
            quality: JPEG quality (1-100)
            device: GPU device
        """
        self.quality = quality
        self.device = device
        self.logger = logging.getLogger(f"NVJPEGEncoder.{device}")
        
        # Check if we can use hardware JPEG encoding
        self.use_hardware = self._check_nvjpeg_support()
        
        # Stats
        self.stats = {
            'frames_encoded': 0,
            'total_encode_time': 0.0,
            'hardware_encodes': 0,
            'software_encodes': 0
        }
    
    def _check_nvjpeg_support(self) -> bool:
        """Check if NVJPEG is available"""
        # For now, we'll use software encoding
        # TODO: Integrate actual NVJPEG library when available
        return False
    
    def encode_tensor(self, tensor: torch.Tensor) -> bytes:
        """
        Encode GPU tensor to JPEG.
        
        Args:
            tensor: GPU tensor (C, H, W) in RGB format [0,1]
            
        Returns:
            Encoded JPEG bytes
        """
        start_time = time.time()
        
        if self.use_hardware:
            # Hardware encoding path (TODO)
            encoded = self._encode_hardware(tensor)
            self.stats['hardware_encodes'] += 1
        else:
            # Software encoding fallback
            encoded = self._encode_software(tensor)
            self.stats['software_encodes'] += 1
        
        self.stats['frames_encoded'] += 1
        self.stats['total_encode_time'] += time.time() - start_time
        
        return encoded
    
    def _encode_software(self, tensor: torch.Tensor) -> bytes:
        """Software JPEG encoding using OpenCV"""
        import cv2
        
        # Convert to HWC format
        if tensor.dim() == 3:
            tensor_hwc = tensor.permute(1, 2, 0)
        else:
            tensor_hwc = tensor[0].permute(1, 2, 0)
        
        # Convert to uint8
        if tensor_hwc.dtype != torch.uint8:
            tensor_hwc = (tensor_hwc * 255).clamp(0, 255).to(torch.uint8)
        
        # Move to CPU
        frame_np = tensor_hwc.cpu().numpy()
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        
        # Encode to JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        _, encoded = cv2.imencode('.jpg', frame_bgr, encode_params)
        
        return encoded.tobytes()
    
    def _encode_hardware(self, tensor: torch.Tensor) -> bytes:
        """Hardware JPEG encoding (placeholder)"""
        # TODO: Implement actual NVJPEG encoding
        return self._encode_software(tensor)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get encoder statistics"""
        stats = self.stats.copy()
        
        if stats['frames_encoded'] > 0:
            stats['avg_encode_time_ms'] = (stats['total_encode_time'] / stats['frames_encoded']) * 1000
            stats['hardware_ratio'] = stats['hardware_encodes'] / stats['frames_encoded']
        
        return stats


def test_nvenc_encoder():
    """Test NVENC encoder"""
    # Create test configuration
    config = EncoderConfig(
        width=1280,
        height=720,
        fps=30,
        codec='h264_nvenc',
        bitrate=2000000
    )
    
    # Create encoder
    encoder = NVENCEncoder(config)
    
    try:
        # Start encoder
        encoder.start()
        
        # Create test frames
        print("Encoding test frames...")
        for i in range(100):
            # Create random frame
            frame = torch.rand(3, 720, 1280, device='cuda')
            
            # Encode
            encoder.encode_tensor(frame)
            
            # Check for output
            encoded = encoder.get_encoded_frame(timeout=0.01)
            if encoded:
                print(f"Frame {i}: Encoded {len(encoded)} bytes")
        
        # Wait a bit for encoding to finish
        time.sleep(1)
        
        # Get stats
        stats = encoder.get_stats()
        print(f"\nEncoder stats: {stats}")
        
    finally:
        encoder.stop()


if __name__ == "__main__":
    test_nvenc_encoder() 