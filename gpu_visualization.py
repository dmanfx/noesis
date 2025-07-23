"""
GPU-Accelerated Visualization
==============================

Implements GPU-based drawing operations for video analytics visualization.
Replaces CPU-based OpenCV operations with PyTorch tensor operations.

Key features:
- GPU-based bounding box drawing
- GPU-based text rendering (simplified)
- Zero-copy operations until final encoding
- Hardware-accelerated JPEG/H264 encoding
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
import time
from typing import List, Tuple, Dict, Optional, Union, Any
from dataclasses import dataclass
import colorsys

# Try to import NVJPEG for hardware JPEG encoding
try:
    import nvjpeg
    NVJPEG_AVAILABLE = True
except ImportError:
    NVJPEG_AVAILABLE = False
    logging.getLogger(__name__).info("NVJPEG not available - using CPU JPEG encoding")

# Try to import PyNVENC for hardware video encoding
try:
    import pynvenc
    NVENC_AVAILABLE = True
except ImportError:
    NVENC_AVAILABLE = False
    logging.getLogger(__name__).info("PyNVENC not available - using CPU video encoding")

# Import memory pool if available
try:
    from gpu_memory_pool import get_global_memory_pool
    GPU_MEMORY_POOL_AVAILABLE = True
except ImportError:
    GPU_MEMORY_POOL_AVAILABLE = False

# Import models for type hints
from models import DetectionResult, TrackingResult


@dataclass
class BoundingBox:
    """Bounding box for GPU rendering"""
    x1: float
    y1: float
    x2: float
    y2: float
    color: Tuple[float, float, float]  # RGB normalized [0,1]
    thickness: int = 2
    label: Optional[str] = None
    confidence: Optional[float] = None


class GPUVisualizer:
    """
    GPU-accelerated visualization for video analytics.
    
    This visualizer performs all drawing operations on GPU tensors,
    avoiding CPU transfers until final encoding.
    """
    
    def __init__(self, device: str = 'cuda:0', use_fp16: bool = True,
                 enable_nvjpeg: bool = True, enable_nvenc: bool = True):
        """
        Initialize GPU visualizer.
        
        Args:
            device: GPU device to use
            use_fp16: Whether to use FP16 for visualization
            enable_nvjpeg: Whether to use NVJPEG for encoding
            enable_nvenc: Whether to use NVENC for video encoding
        """
        self.device = device
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.logger = logging.getLogger(f"GPUVisualizer.{device}")
        
        # Memory pool
        self.memory_pool = None
        if GPU_MEMORY_POOL_AVAILABLE:
            try:
                self.memory_pool = get_global_memory_pool(device=device)
                self.logger.info("Using unified GPU memory pool")
            except Exception as e:
                self.logger.warning(f"Failed to initialize memory pool: {e}")
        
        # Hardware encoding
        self.nvjpeg_encoder = None
        if enable_nvjpeg and NVJPEG_AVAILABLE:
            try:
                self.nvjpeg_encoder = nvjpeg.Encoder()
                self.logger.info("NVJPEG hardware encoding enabled")
            except Exception as e:
                self.logger.warning(f"Failed to initialize NVJPEG: {e}")
        
        self.nvenc_encoder = None
        if enable_nvenc and NVENC_AVAILABLE:
            try:
                self.nvenc_encoder = pynvenc.NVEncoder()
                self.logger.info("NVENC hardware video encoding enabled")
            except Exception as e:
                self.logger.warning(f"Failed to initialize NVENC: {e}")
        
        # Color palette for tracking
        self.color_palette = self._generate_color_palette(100)
        
        # Performance stats
        self.stats = {
            'frames_processed': 0,
            'total_draw_time': 0.0,
            'total_encode_time': 0.0,
            'gpu_draw_operations': 0,
            'cpu_fallback_operations': 0
        }
        
        self.logger.info(f"Initialized GPU visualizer on {device}")
    
    def _generate_color_palette(self, num_colors: int) -> torch.Tensor:
        """Generate distinct colors for tracking visualization"""
        colors = []
        for i in range(num_colors):
            hue = i / num_colors
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
            colors.append(rgb)
        
        # Convert to GPU tensor
        palette = torch.tensor(colors, device=self.device, dtype=self.dtype)
        return palette
    
    def draw_detections(self, frame_tensor: torch.Tensor,
                       detections: List[DetectionResult],
                       tracks: Optional[List[TrackingResult]] = None) -> torch.Tensor:
        """
        Draw detections and tracks on GPU tensor.
        
        Args:
            frame_tensor: GPU tensor (C, H, W) or (B, C, H, W) in RGB format [0,1]
            detections: List of detection results
            tracks: Optional list of tracking results
            
        Returns:
            Annotated frame tensor on GPU
        """
        start_time = time.time()
        
        with torch.inference_mode():
            # Ensure correct format
            if frame_tensor.dim() == 3:
                frame = frame_tensor
                C, H, W = frame.shape
            else:
                frame = frame_tensor[0]  # Take first in batch
                C, H, W = frame.shape
            
            # Clone to avoid modifying original
            if self.memory_pool:
                try:
                    output_frame, alloc_id = self.memory_pool.get_tensor(
                        frame.shape, dtype=self.dtype
                    )
                    output_frame.copy_(frame)
                except Exception:
                    output_frame = frame.clone()
                    alloc_id = None
            else:
                output_frame = frame.clone()
                alloc_id = None
            
            # Convert to FP16 if needed
            if output_frame.dtype != self.dtype:
                output_frame = output_frame.to(self.dtype)
            
            # Draw detections
            for i, detection in enumerate(detections):
                # Get color based on track or detection index
                if tracks and i < len(tracks):
                    color_idx = tracks[i].track_id % len(self.color_palette)
                    color = self.color_palette[color_idx]
                else:
                    color = torch.tensor([0.0, 1.0, 0.0], device=self.device, dtype=self.dtype)
                
                # Extract bbox coordinates
                x1, y1, x2, y2 = detection.bbox
                
                # Get label from class_id
                label = f"class_{detection.class_id}"
                
                # Draw bounding box
                bbox = BoundingBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    color=tuple(color.cpu().numpy()),
                    label=label,
                    confidence=detection.confidence
                )
                
                output_frame = self._draw_bbox_gpu(output_frame, bbox, H, W)
                self.stats['gpu_draw_operations'] += 1
            
            # Update stats
            self.stats['frames_processed'] += 1
            self.stats['total_draw_time'] += time.time() - start_time
            
            # Return tensor to pool if used
            if self.memory_pool and alloc_id is not None:
                self.memory_pool.return_tensor(output_frame, alloc_id)
            
            return output_frame
    
    def _draw_bbox_gpu(self, frame: torch.Tensor, bbox: BoundingBox,
                      height: int, width: int) -> torch.Tensor:
        """
        Draw bounding box on GPU tensor.
        
        Args:
            frame: GPU tensor (C, H, W) in RGB format [0,1]
            bbox: Bounding box to draw
            height: Frame height
            width: Frame width
            
        Returns:
            Frame with bounding box drawn
        """
        # Convert normalized coordinates to pixel coordinates
        x1 = int(bbox.x1 * width)
        y1 = int(bbox.y1 * height)
        x2 = int(bbox.x2 * width)
        y2 = int(bbox.y2 * height)
        
        # Clamp to frame boundaries
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        
        # Create color tensor
        color = torch.tensor(bbox.color, device=self.device, dtype=self.dtype).view(3, 1, 1)
        
        # Draw rectangle edges
        thickness = bbox.thickness
        
        # Top edge
        if y1 >= 0 and y1 + thickness < height:
            frame[:, y1:y1+thickness, x1:x2] = color
        
        # Bottom edge
        if y2 - thickness >= 0 and y2 < height:
            frame[:, y2-thickness:y2, x1:x2] = color
        
        # Left edge
        if x1 >= 0 and x1 + thickness < width:
            frame[:, y1:y2, x1:x1+thickness] = color
        
        # Right edge
        if x2 - thickness >= 0 and x2 < width:
            frame[:, y1:y2, x2-thickness:x2] = color
        
        # Draw label background if provided
        if bbox.label:
            # Simplified label drawing - just a colored bar above bbox
            label_height = 20
            label_y1 = max(0, y1 - label_height)
            label_y2 = y1
            
            if label_y1 < label_y2:
                # Darken background for label
                frame[:, label_y1:label_y2, x1:x2] *= 0.3
                # Add color tint
                frame[:, label_y1:label_y2, x1:x2] += color * 0.7
        
        return frame
    
    def encode_frame_gpu(self, frame_tensor: torch.Tensor,
                        quality: int = 85) -> bytes:
        """
        Encode GPU tensor to JPEG using hardware acceleration.
        
        Args:
            frame_tensor: GPU tensor (C, H, W) in RGB format [0,1]
            quality: JPEG quality (1-100)
            
        Returns:
            Encoded JPEG bytes
        """
        start_time = time.time()
        
        try:
            # Ensure correct format (H, W, C) for encoding
            if frame_tensor.dim() == 3:
                # Convert CHW to HWC
                frame_hwc = frame_tensor.permute(1, 2, 0)
            else:
                # Batch mode - take first frame
                frame_hwc = frame_tensor[0].permute(1, 2, 0)
            
            # Convert to uint8
            if frame_hwc.dtype != torch.uint8:
                frame_hwc = (frame_hwc * 255).clamp(0, 255).to(torch.uint8)
            
            if self.nvjpeg_encoder:
                # Use NVJPEG hardware encoding
                try:
                    encoded = self.nvjpeg_encoder.encode(
                        frame_hwc.cpu().numpy(),  # NVJPEG needs numpy
                        quality=quality
                    )
                    self.stats['total_encode_time'] += time.time() - start_time
                    return encoded
                except Exception as e:
                    self.logger.debug(f"NVJPEG encoding failed: {e}")
            
            # Fallback to CPU encoding
            import cv2
            frame_np = frame_hwc.cpu().numpy()
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            
            # Encode to JPEG
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            _, encoded = cv2.imencode('.jpg', frame_bgr, encode_params)
            
            self.stats['total_encode_time'] += time.time() - start_time
            self.stats['cpu_fallback_operations'] += 1
            
            return encoded.tobytes()
            
        except Exception as e:
            self.logger.error(f"Frame encoding failed: {e}")
            raise
    
    def create_overlay_tensor(self, base_frame: torch.Tensor,
                            overlay_info: Dict[str, Any]) -> torch.Tensor:
        """
        Create overlay information on GPU tensor.
        
        Args:
            base_frame: Base frame tensor on GPU
            overlay_info: Dictionary with overlay information
            
        Returns:
            Frame with overlay applied
        """
        with torch.inference_mode():
            output = base_frame.clone()
            
            # Add FPS counter
            if 'fps' in overlay_info:
                # Simple FPS indicator - colored corner
                fps = overlay_info['fps']
                color = torch.tensor(
                    [0.0, 1.0, 0.0] if fps > 25 else [1.0, 0.0, 0.0],
                    device=self.device, dtype=self.dtype
                ).view(3, 1, 1)
                
                # Draw in top-left corner
                output[:, :30, :100] = output[:, :30, :100] * 0.3 + color * 0.7
            
            # Add detection count
            if 'detection_count' in overlay_info:
                count = overlay_info['detection_count']
                # Simple count indicator - bar width
                bar_width = min(200, count * 20)
                if bar_width > 0:
                    output[:, -30:, :bar_width] = torch.tensor(
                        [0.0, 0.5, 1.0], device=self.device, dtype=self.dtype
                    ).view(3, 1, 1)
            
            return output
    
    def batch_visualize(self, frame_tensors: List[torch.Tensor],
                       detections_list: List[List[DetectionResult]],
                       tracks_list: Optional[List[List[TrackingResult]]] = None) -> List[torch.Tensor]:
        """
        Batch visualization for multiple frames.
        
        Args:
            frame_tensors: List of GPU tensors
            detections_list: List of detection lists
            tracks_list: Optional list of tracking lists
            
        Returns:
            List of annotated frame tensors
        """
        results = []
        
        for i, frame in enumerate(frame_tensors):
            detections = detections_list[i] if i < len(detections_list) else []
            tracks = tracks_list[i] if tracks_list and i < len(tracks_list) else None
            
            annotated = self.draw_detections(frame, detections, tracks)
            results.append(annotated)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get visualization statistics."""
        stats = self.stats.copy()
        
        if stats['frames_processed'] > 0:
            stats['avg_draw_time_ms'] = (stats['total_draw_time'] / stats['frames_processed']) * 1000
            stats['avg_encode_time_ms'] = (stats['total_encode_time'] / stats['frames_processed']) * 1000
        
        return stats


class HardwareEncoder:
    """
    Hardware-accelerated video encoder using NVENC.
    
    Provides H.264/H.265 encoding directly from GPU tensors.
    """
    
    def __init__(self, width: int, height: int, fps: int = 30,
                 codec: str = 'h264', bitrate: int = 4000000,
                 device: str = 'cuda:0'):
        """
        Initialize hardware encoder.
        
        Args:
            width: Video width
            height: Video height
            fps: Frame rate
            codec: Codec to use ('h264' or 'h265')
            bitrate: Target bitrate in bits/second
            device: GPU device
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self.bitrate = bitrate
        self.device = device
        self.logger = logging.getLogger("HardwareEncoder")
        
        self.encoder = None
        self._init_encoder()
    
    def _init_encoder(self):
        """Initialize NVENC encoder"""
        if not NVENC_AVAILABLE:
            self.logger.warning("NVENC not available - hardware encoding disabled")
            return
        
        try:
            # Initialize NVENC encoder
            self.encoder = pynvenc.NVEncoder(
                width=self.width,
                height=self.height,
                fps=self.fps,
                codec=self.codec,
                bitrate=self.bitrate,
                preset='llhq',  # Low latency high quality
                profile='high'
            )
            self.logger.info(f"Initialized NVENC {self.codec} encoder: {self.width}x{self.height}@{self.fps}fps")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize NVENC: {e}")
            self.encoder = None
    
    def encode_frame(self, frame_tensor: torch.Tensor) -> Optional[bytes]:
        """
        Encode GPU tensor to video frame.
        
        Args:
            frame_tensor: GPU tensor (C, H, W) in RGB format [0,1]
            
        Returns:
            Encoded frame bytes or None if encoding failed
        """
        if not self.encoder:
            return None
        
        try:
            # Convert to HWC format
            if frame_tensor.dim() == 3:
                frame_hwc = frame_tensor.permute(1, 2, 0)
            else:
                frame_hwc = frame_tensor[0].permute(1, 2, 0)
            
            # Convert to uint8
            if frame_hwc.dtype != torch.uint8:
                frame_hwc = (frame_hwc * 255).clamp(0, 255).to(torch.uint8)
            
            # Encode frame
            encoded = self.encoder.encode_frame(frame_hwc)
            return encoded
            
        except Exception as e:
            self.logger.error(f"Frame encoding failed: {e}")
            return None
    
    def get_headers(self) -> bytes:
        """Get video stream headers (SPS/PPS for H.264)"""
        if self.encoder:
            return self.encoder.get_headers()
        return b''
    
    def close(self):
        """Close encoder and release resources"""
        if self.encoder:
            try:
                self.encoder.close()
                self.logger.info("Closed NVENC encoder")
            except Exception as e:
                self.logger.error(f"Error closing encoder: {e}")
            self.encoder = None


def test_gpu_visualization():
    """Test GPU visualization"""
    import time
    from models import DetectionResult
    
    # Create visualizer
    visualizer = GPUVisualizer()
    
    # Create test frame
    test_frame = torch.rand(3, 720, 1280, device='cuda', dtype=torch.float32)
    
    # Create test detections
    test_detections = [
        DetectionResult(
            id=0,
            class_id=0,  # person
            confidence=0.95,
            bbox=(0.1, 0.1, 0.3, 0.4)
        ),
        DetectionResult(
            id=1,
            class_id=2,  # car
            confidence=0.87,
            bbox=(0.5, 0.2, 0.7, 0.6)
        )
    ]
    
    # Test drawing
    print("Testing GPU visualization...")
    
    for i in range(10):
        start = time.time()
        annotated = visualizer.draw_detections(test_frame, test_detections)
        draw_time = (time.time() - start) * 1000
        
        # Test encoding
        start = time.time()
        encoded = visualizer.encode_frame_gpu(annotated)
        encode_time = (time.time() - start) * 1000
        
        print(f"Frame {i}: Draw: {draw_time:.2f}ms, Encode: {encode_time:.2f}ms, Size: {len(encoded)/1024:.1f}KB")
    
    # Print stats
    stats = visualizer.get_stats()
    print(f"\nStats: {stats}")


if __name__ == "__main__":
    test_gpu_visualization() 