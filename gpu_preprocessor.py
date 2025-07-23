"""
GPU Frame Preprocessor
=====================

Handles GPU-based frame preprocessing for unified pipeline.
All operations remain on GPU to maintain zero-copy behavior.

Key features:
- GPU-only tensor operations
- Memory pool integration
- FP16 optimization
- Zero-copy processing throughout
- Advanced resize optimization with benchmarking
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
import time
from typing import Optional, Tuple, Union, Dict, List
from collections import deque

# Import memory pool if available
try:
    from gpu_memory_pool import get_global_memory_pool
    GPU_MEMORY_POOL_AVAILABLE = True
except ImportError:
    GPU_MEMORY_POOL_AVAILABLE = False

# Import advanced resize optimizer if available
try:
    from advanced_resize_optimizer import AdvancedResizeOptimizer, integrate_advanced_resize
    ADVANCED_RESIZE_AVAILABLE = True
except ImportError:
    ADVANCED_RESIZE_AVAILABLE = False
    logging.getLogger(__name__).info("Advanced resize optimizer not available - using PyTorch resize")


class GPUFramePreprocessor:
    """
    GPU-based frame preprocessing for video analytics.
    
    This preprocessor handles all frame transformations on GPU:
    - Resizing (with advanced optimization)
    - Normalization
    - Color space conversion
    - Format conversion
    
    All operations are performed without CPU memory transfers.
    """
    
    def __init__(self, target_width: int = 640, target_height: int = 640, 
                 device: str = 'cuda:0', use_fp16: bool = True,
                 enable_advanced_resize: bool = True,
                 benchmark_resize_on_init: bool = False):
        """
        Initialize GPU frame preprocessor.
        
        Args:
            target_width: Target width for resizing
            target_height: Target height for resizing
            device: GPU device to use
            use_fp16: Whether to use FP16 precision
            enable_advanced_resize: Whether to use advanced resize optimization
            benchmark_resize_on_init: Whether to benchmark resize methods on initialization
        """
        self.target_width = target_width
        self.target_height = target_height
        self.device = device
        self.use_fp16 = use_fp16
        self.logger = logging.getLogger(f"GPUFramePreprocessor.{device}")
        
        # Statistics tracking
        self.stats = {
            'frames_processed': 0,
            'total_preprocess_time': 0.0,
            'resize_time': 0.0,
            'normalize_time': 0.0,
            'color_convert_time': 0.0,
            'avg_preprocess_time': 0.0,
            'memory_pool_hits': 0,
            'memory_pool_misses': 0
        }
        
        # Memory pool
        self.memory_pool = None
        if GPU_MEMORY_POOL_AVAILABLE:
            try:
                self.memory_pool = get_global_memory_pool(device=device)
                self.logger.info("Using unified GPU memory pool")
            except Exception as e:
                self.logger.warning(f"Failed to initialize memory pool: {e}")
        
        # Advanced resize optimizer
        self.resize_optimizer = None
        if enable_advanced_resize and ADVANCED_RESIZE_AVAILABLE:
            try:
                self.resize_optimizer = AdvancedResizeOptimizer(
                    target_size=(target_height, target_width),
                    device=device
                )
                
                # Benchmark resize methods if requested
                if benchmark_resize_on_init:
                    self.logger.info("Benchmarking resize methods...")
                    self.resize_optimizer.benchmark_resize_methods(
                        test_sizes=[(1080, 1920), (720, 1280), (480, 640)],
                        num_iterations=50
                    )
                    report = self.resize_optimizer.generate_report()
                    self.logger.info(f"Resize benchmark results:\n{report}")
                
                # Integrate with preprocessor
                integrate_advanced_resize(self, self.resize_optimizer)
                self.logger.info(f"Using advanced resize optimizer: {self.resize_optimizer.selected_method}")
                
            except Exception as e:
                self.logger.warning(f"Failed to initialize advanced resize: {e}")
                self.resize_optimizer = None
        
        # Performance tracking
        self.preprocess_times = deque(maxlen=100)
        
        # Warm up GPU operations
        self._warmup_gpu_operations()
        
        self.logger.info(f"Initialized GPU preprocessor on {device}")
    
    def _warmup_gpu_operations(self):
        """Warm up GPU operations to avoid cold start latency."""
        self.logger.debug("Warming up GPU operations...")
        
        with torch.inference_mode():
            # Create dummy tensor
            dummy_tensor = torch.randn(3, 1080, 1920, device=self.device)
            
            # Warm up resize operation
            for _ in range(3):
                resized = self._resize_tensor_gpu(dummy_tensor, self.target_height, self.target_width)
                # Stay on GPU for warmup - no CPU transfer needed
            
            # Return tensor to pool
            del dummy_tensor
            if resized is not None:
                del resized
            
            torch.cuda.synchronize()
    
    def preprocess_tensor_gpu(self, tensor: torch.Tensor, 
                            normalize: bool = True,
                            output_format: str = 'CHW') -> torch.Tensor:
        """
        Preprocess tensor entirely on GPU (zero-copy).
        
        Args:
            tensor: Input tensor on GPU (C, H, W) or (B, C, H, W)
            normalize: Whether to normalize to [0, 1]
            output_format: Output format ('CHW' or 'HWC')
            
        Returns:
            Preprocessed tensor on GPU
        """
        start_time = time.time()
        
        with torch.inference_mode():
            # Validate tensor is on GPU
            if not tensor.is_cuda:
                raise RuntimeError("GPU-only preprocessor received CPU tensor!")
            
            # Ensure correct dimensions
            if tensor.dim() == 3:
                batch_mode = False
                C, H, W = tensor.shape
            elif tensor.dim() == 4:
                batch_mode = True
                B, C, H, W = tensor.shape
            else:
                raise ValueError(f"Expected 3D or 4D tensor, got {tensor.dim()}D")
            
            # Resize if needed
            if H != self.target_height or W != self.target_width:
                resize_start = time.time()
                
                if self.resize_optimizer and hasattr(self, '_optimized_resize'):
                    # Use advanced resize optimizer
                    tensor = self._optimized_resize(tensor, self.target_height, self.target_width)
                else:
                    # Use standard resize
                    tensor = self._resize_tensor_gpu(tensor, self.target_height, self.target_width)
                
                self.stats['resize_time'] += (time.time() - resize_start)
            
            # Normalize if requested
            if normalize:
                norm_start = time.time()
                tensor = self._normalize_tensor_gpu(tensor)
                self.stats['normalize_time'] += (time.time() - norm_start)
            
            # Convert to FP16 if requested
            if self.use_fp16 and tensor.dtype != torch.float16:
                tensor = tensor.half()
            
            # Format conversion if needed
            if output_format == 'HWC':
                if batch_mode:
                    tensor = tensor.permute(0, 2, 3, 1)  # BCHW -> BHWC
                else:
                    tensor = tensor.permute(1, 2, 0)  # CHW -> HWC
            
            # Update statistics
            preprocess_time = time.time() - start_time
            self.stats['frames_processed'] += 1
            self.stats['total_preprocess_time'] += preprocess_time
            self.stats['avg_preprocess_time'] = self.stats['total_preprocess_time'] / self.stats['frames_processed']
            self.preprocess_times.append(preprocess_time)
            
            return tensor
    
    def _resize_tensor_gpu(self, tensor: torch.Tensor, 
                          target_height: int, target_width: int) -> torch.Tensor:
        """
        Resize tensor on GPU using PyTorch interpolation (fallback method).
        
        Args:
            tensor: Input tensor (C, H, W) or (B, C, H, W)
            target_height: Target height
            target_width: Target width
            
        Returns:
            Resized tensor
        """
        # Add batch dimension if needed
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
            remove_batch = True
        else:
            remove_batch = False
        
        # Use memory pool if available
        output_tensor = None
        alloc_id = None
        
        if self.memory_pool:
            try:
                output_shape = (tensor.shape[0], tensor.shape[1], target_height, target_width)
                output_tensor, alloc_id = self.memory_pool.get_tensor(
                    output_shape, 
                    dtype=torch.float16 if self.use_fp16 else torch.float32
                )
                self.memory_pool.touch_tensor(alloc_id)  # Mark as actively used
                self.stats['memory_pool_hits'] += 1
            except Exception as e:
                self.logger.debug(f"Memory pool allocation failed: {e}")
                self.stats['memory_pool_misses'] += 1
        
        # Perform resize
        if output_tensor is not None:
            # F.interpolate doesn't support 'out' parameter, so we resize and copy
            resized = F.interpolate(
                tensor,
                size=(target_height, target_width),
                mode='bilinear',
                align_corners=False
            )
            # Copy to pre-allocated tensor
            output_tensor.copy_(resized)
            resized = output_tensor
        else:
            # Standard allocation
            resized = F.interpolate(
                tensor,
                size=(target_height, target_width),
                mode='bilinear',
                align_corners=False
            )
        
        # Remove batch dimension if we added it
        if remove_batch:
            resized = resized.squeeze(0)
        
        # Return tensor to pool if used
        if self.memory_pool and alloc_id is not None:
            self.memory_pool.return_tensor(output_tensor, alloc_id)
        
        return resized
    
    def _normalize_tensor_gpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize tensor to [0, 1] range on GPU.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Normalized tensor
        """
        # Efficient in-place normalization if possible
        if tensor.dtype == torch.uint8:
            return tensor.float() / 255.0
        elif tensor.max() > 1.0:
            return tensor / 255.0
        else:
            return tensor  # Already normalized
    
    def convert_bgr_to_rgb_gpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert BGR to RGB on GPU.
        
        Args:
            tensor: Input tensor in BGR format (C, H, W) or (B, C, H, W)
            
        Returns:
            Tensor in RGB format
        """
        if tensor.shape[-3] == 3:  # Check channel dimension
            # Flip channel order: BGR -> RGB
            if tensor.dim() == 3:
                return tensor[[2, 1, 0], :, :]
            else:  # batch mode
                return tensor[:, [2, 1, 0], :, :]
        else:
            self.logger.warning(f"Unexpected channel count: {tensor.shape[-3]}")
            return tensor
    
    def benchmark_resize_methods(self, test_sizes: List[Tuple[int, int]] = None,
                               num_iterations: int = 100) -> Optional[str]:
        """
        Benchmark available resize methods and return report.
        
        Args:
            test_sizes: List of (height, width) tuples to test
            num_iterations: Number of iterations per test
            
        Returns:
            Benchmark report string or None if not available
        """
        if not self.resize_optimizer:
            return "Advanced resize optimizer not available"
        
        # Run benchmarks
        self.resize_optimizer.benchmark_resize_methods(test_sizes, num_iterations)
        
        # Generate and return report
        return self.resize_optimizer.generate_report()
    
    def get_stats(self) -> Dict[str, float]:
        """Get preprocessing statistics."""
        stats = self.stats.copy()
        
        # Add percentiles for preprocess times
        if self.preprocess_times:
            times_array = np.array(self.preprocess_times)
            stats['preprocess_p50_ms'] = np.percentile(times_array, 50) * 1000
            stats['preprocess_p90_ms'] = np.percentile(times_array, 90) * 1000
            stats['preprocess_p99_ms'] = np.percentile(times_array, 99) * 1000
        
        # Add resize method info
        if self.resize_optimizer:
            stats['resize_method'] = self.resize_optimizer.selected_method or 'pytorch_bilinear'
        else:
            stats['resize_method'] = 'pytorch_bilinear'
        
        return stats
    
    def preprocess_frame_gpu(self, frame: Union[np.ndarray, torch.Tensor],
                           color_format: str = 'BGR',
                           normalize: bool = True) -> np.ndarray:
        """
        Legacy interface for compatibility - converts numpy to tensor and back.
        
        WARNING: This method performs GPU->CPU transfer and should be avoided
        in the unified GPU pipeline. Use preprocess_tensor_gpu() instead.
        
        Args:
            frame: Input frame as numpy array or tensor
            color_format: Color format of input ('BGR' or 'RGB')
            normalize: Whether to normalize to [0, 1]
            
        Returns:
            Preprocessed frame as numpy array
        """
        self.logger.warning("Using legacy preprocess_frame_gpu() - consider using preprocess_tensor_gpu()")
        
        # Convert to tensor if needed
        if isinstance(frame, np.ndarray):
            # Convert HWC -> CHW and move to GPU
            if frame.ndim == 3:
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).to(self.device)
            else:
                frame_tensor = torch.from_numpy(frame).to(self.device)
        else:
            frame_tensor = frame
        
        # Convert color if needed
        if color_format == 'BGR':
            frame_tensor = self.convert_bgr_to_rgb_gpu(frame_tensor)
        
        # Preprocess
        processed_tensor = self.preprocess_tensor_gpu(
            frame_tensor,
            normalize=normalize,
            output_format='CHW'
        )
        
        # Convert back to numpy - MINIMAL CPU for final output only 
        result = frame_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        
        return result
    
    def preprocess_batch_gpu(self, frames: List[np.ndarray],
                           color_format: str = 'BGR',
                           normalize: bool = True) -> List[np.ndarray]:
        """
        Legacy batch interface for compatibility.
        
        WARNING: This method performs GPU->CPU transfer and should be avoided
        in the unified GPU pipeline.
        
        Args:
            frames: List of input frames as numpy arrays
            color_format: Color format of input ('BGR' or 'RGB')
            normalize: Whether to normalize to [0, 1]
            
        Returns:
            List of preprocessed frames as numpy arrays
        """
        self.logger.warning("Using legacy preprocess_batch_gpu() - consider using tensor operations")
        
        # Stack frames into batch tensor
        batch_list = []
        for frame in frames:
            if frame.ndim == 3:
                batch_list.append(torch.from_numpy(frame).permute(2, 0, 1))
            else:
                batch_list.append(torch.from_numpy(frame))
        
        batch_tensor = torch.stack(batch_list).to(self.device)
        
        # Convert color if needed
        if color_format == 'BGR':
            batch_tensor = self.convert_bgr_to_rgb_gpu(batch_tensor)
        
        # Preprocess batch
        processed_batch = self.preprocess_tensor_gpu(
            batch_tensor,
            normalize=normalize,
            output_format='CHW'
        )
        
        # Convert back to numpy - MINIMAL CPU for final output only
        results = []
        batch_numpy = batch_tensor.cpu().numpy()
        for i in range(batch_numpy.shape[0]):
            frame = batch_numpy[i].transpose(1, 2, 0).astype(np.uint8)
            results.append(frame)
        
        return results 