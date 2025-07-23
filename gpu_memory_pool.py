#!/usr/bin/env python3
"""
Unified GPU Memory Pool Manager

This module provides a centralized GPU memory pool for all pipeline components
to prevent memory fragmentation and optimize memory usage across multi-engine
FP16 workloads.

Key features:
- Pre-allocated memory pools for common tensor sizes
- Memory pool recycling with size-based buckets
- Anti-fragmentation strategies
- Memory leak detection and monitoring
"""

import torch
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import threading
import time


class UnifiedGPUMemoryPool:
    """
    Centralized GPU memory pool manager for all pipeline components.
    
    This class manages pre-allocated GPU memory blocks to reduce allocation
    overhead and prevent fragmentation, especially important for FP16 workloads.
    """
    
    def __init__(self, device: str = "cuda:0", enable_monitoring: bool = True):
        """
        Initialize unified GPU memory pool.
        
        Args:
            device: GPU device to manage memory for
            enable_monitoring: Enable memory usage monitoring
        """
        self.device = torch.device(device)
        self.enable_monitoring = enable_monitoring
        self.logger = logging.getLogger("UnifiedGPUMemoryPool")
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Memory pools organized by size buckets
        # Key: (dtype, shape) -> Value: deque of available tensors
        self._pools: Dict[Tuple[torch.dtype, Tuple[int, ...]], deque] = defaultdict(lambda: deque(maxlen=10))
        
        # Common tensor sizes for pre-allocation (based on typical pipeline usage)
        self._common_sizes = {
            # Detection/Segmentation input sizes
            (torch.float16, (1, 3, 640, 640)): 5,      # YOLO input
            (torch.float16, (1, 3, 256, 128)): 10,     # ReID input
            (torch.float16, (1, 3, 1080, 1920)): 3,    # Full frame
            (torch.float16, (3, 1080, 1920)): 5,       # Single frame tensor
            (torch.float16, (3, 640, 640)): 10,        # Resized frame
            
            # Output sizes
            (torch.float16, (1, 25200, 85)): 5,        # YOLO output
            (torch.float16, (1, 512)): 10,             # ReID features
            (torch.float16, (1, 17, 3)): 10,           # Pose keypoints
        }
        
        # Memory usage statistics
        self._stats = {
            'allocations': 0,
            'reuses': 0,
            'returns': 0,
            'leaks': 0,
            'total_allocated_mb': 0.0,
            'peak_allocated_mb': 0.0,
            'fragmentation_events': 0
        }
        
        # Memory allocation tracking for leak detection
        self._allocated_tensors: Dict[int, Dict] = {}  # tensor_id -> allocation info
        self._allocation_id_counter = 0
        self._warned_allocations: set = set()  # Track which allocations have been warned about
        
        # Pre-allocate common sizes
        self._preallocate_common_sizes()
        
        # Start monitoring thread if enabled
        if self.enable_monitoring:
            self._monitoring_thread = threading.Thread(target=self._monitor_memory, daemon=True)
            self._monitoring_thread.start()
        
        self.logger.info(f"Initialized unified GPU memory pool on {device}")
    
    def _preallocate_common_sizes(self):
        """Pre-allocate memory blocks for common tensor sizes."""
        self.logger.info("Pre-allocating common tensor sizes...")
        
        try:
            torch.cuda.set_device(self.device)
            
            for (dtype, shape), count in self._common_sizes.items():
                pool_key = (dtype, shape)
                
                for _ in range(count):
                    tensor = torch.empty(shape, dtype=dtype, device=self.device)
                    self._pools[pool_key].append(tensor)
                
                allocated_mb = (count * torch.empty(shape, dtype=dtype).element_size() * 
                              torch.empty(shape, dtype=dtype).numel()) / (1024 * 1024)
                
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"Pre-allocated {count} tensors of shape {shape} "
                                    f"dtype {dtype} ({allocated_mb:.2f} MB)")
                
                self._stats['total_allocated_mb'] += allocated_mb
            
            self._stats['peak_allocated_mb'] = self._stats['total_allocated_mb']
            self.logger.info(f"Pre-allocation complete: {self._stats['total_allocated_mb']:.2f} MB")
            
        except Exception as e:
            self.logger.error(f"Failed to pre-allocate memory: {e}")
            raise RuntimeError(f"GPU memory pool pre-allocation failed: {e}")
    
    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float16) -> Tuple[torch.Tensor, int]:
        """
        Get a tensor from the pool or allocate a new one.
        
        Args:
            shape: Desired tensor shape
            dtype: Tensor data type (default: float16 for FP16 optimization)
            
        Returns:
            Tuple of (tensor, allocation_id) for tracking
        """
        with self._lock:
            pool_key = (dtype, shape)
            pool = self._pools[pool_key]
            
            # Try to get from pool
            if pool:
                tensor = pool.popleft()
                self._stats['reuses'] += 1
                
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"Reused tensor from pool: shape={shape}, dtype={dtype}")
            else:
                # Allocate new tensor
                try:
                    tensor = torch.empty(shape, dtype=dtype, device=self.device)
                    self._stats['allocations'] += 1
                    
                    # Update memory stats
                    allocated_mb = (tensor.element_size() * tensor.numel()) / (1024 * 1024)
                    self._stats['total_allocated_mb'] += allocated_mb
                    self._stats['peak_allocated_mb'] = max(
                        self._stats['peak_allocated_mb'], 
                        self._stats['total_allocated_mb']
                    )
                    
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(f"Allocated new tensor: shape={shape}, dtype={dtype}, "
                                        f"size={allocated_mb:.2f} MB")
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        self._handle_oom_error(shape, dtype)
                    raise
            
            # Track allocation for leak detection
            allocation_id = self._allocation_id_counter
            self._allocation_id_counter += 1
            
            self._allocated_tensors[allocation_id] = {
                'tensor_id': id(tensor),
                'shape': shape,
                'dtype': dtype,
                'timestamp': time.time(),
                'last_used': time.time(),  # Track when tensor was last used
                'returned': False
            }
            
            return tensor, allocation_id
    
    def return_tensor(self, tensor: torch.Tensor, allocation_id: int):
        """
        Return a tensor to the pool for reuse.
        
        Args:
            tensor: Tensor to return
            allocation_id: ID from get_tensor() for tracking
        """
        with self._lock:
            # Validate tensor device
            if tensor.device != self.device:
                self.logger.warning(f"Tensor returned on wrong device: {tensor.device} != {self.device}")
                return
            
            # Check allocation tracking
            if allocation_id in self._allocated_tensors:
                alloc_info = self._allocated_tensors[allocation_id]
                
                if alloc_info['returned']:
                    self.logger.warning(f"Tensor already returned: allocation_id={allocation_id}")
                    return
                
                alloc_info['returned'] = True
                alloc_info['return_time'] = time.time()
            else:
                self.logger.warning(f"Unknown allocation_id: {allocation_id}")
            
            # Return to pool
            pool_key = (tensor.dtype, tensor.shape)
            pool = self._pools[pool_key]
            
            if len(pool) < pool.maxlen:
                pool.append(tensor)
                self._stats['returns'] += 1
                
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"Returned tensor to pool: shape={tensor.shape}, "
                                    f"dtype={tensor.dtype}, pool_size={len(pool)}")
            else:
                # Pool is full, let tensor be garbage collected
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"Pool full, releasing tensor: shape={tensor.shape}")
    
    def touch_tensor(self, allocation_id: int):
        """
        Update the last_used timestamp for a tensor to indicate it's being actively used.
        
        Args:
            allocation_id: ID from get_tensor() for tracking
        """
        with self._lock:
            if allocation_id in self._allocated_tensors:
                self._allocated_tensors[allocation_id]['last_used'] = time.time()
                # Clear warning flag since tensor is being used again
                self._warned_allocations.discard(allocation_id)
            else:
                self.logger.debug(f"Unknown allocation_id for touch: {allocation_id}")
    
    def _handle_oom_error(self, shape: Tuple[int, ...], dtype: torch.dtype):
        """Handle out-of-memory errors with defragmentation."""
        self.logger.warning(f"GPU OOM when allocating {shape} {dtype}")
        
        # Clear unused tensors from all pools
        cleared_mb = 0.0
        
        with self._lock:
            for pool_key, pool in self._pools.items():
                pool_dtype, pool_shape = pool_key
                pool_size = len(pool)
                
                if pool_size > 0:
                    # Calculate memory to be freed
                    tensor_size_mb = (torch.empty(pool_shape, dtype=pool_dtype).element_size() * 
                                    torch.empty(pool_shape, dtype=pool_dtype).numel()) / (1024 * 1024)
                    cleared_mb += tensor_size_mb * pool_size
                    
                    # Clear pool
                    pool.clear()
                    
                    self.logger.info(f"Cleared {pool_size} tensors from pool {pool_key} "
                                   f"({tensor_size_mb * pool_size:.2f} MB)")
            
            self._stats['fragmentation_events'] += 1
        
        # Force garbage collection
        torch.cuda.empty_cache()
        
        self.logger.info(f"Cleared {cleared_mb:.2f} MB from memory pools")
    
    def _monitor_memory(self):
        """Background thread to monitor memory usage and detect leaks."""
        while True:
            try:
                time.sleep(300)  # Check every 5 minutes (reduced frequency)
                
                with self._lock:
                    # Check for memory leaks
                    current_time = time.time()
                    leak_threshold = 1800  # 30 minutes (increased threshold)
                    
                    for alloc_id, alloc_info in list(self._allocated_tensors.items()):
                        if not alloc_info['returned']:
                            age = current_time - alloc_info['timestamp']
                            time_since_last_use = current_time - alloc_info.get('last_used', alloc_info['timestamp'])
                            
                            # Only warn if threshold exceeded AND not already warned about this allocation
                            if time_since_last_use > leak_threshold and alloc_id not in self._warned_allocations:
                                self.logger.warning(f"Potential memory leak detected: "
                                                  f"allocation_id={alloc_id}, "
                                                  f"shape={alloc_info['shape']}, "
                                                  f"age={age:.1f}s, "
                                                  f"unused_for={time_since_last_use:.1f}s")
                                self._stats['leaks'] += 1
                                # Mark as warned to prevent repeated warnings
                                self._warned_allocations.add(alloc_id)
                    
                    # Log memory statistics
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(f"Memory pool stats: {self.get_stats()}")
                
            except Exception as e:
                self.logger.error(f"Error in memory monitoring: {e}")
    
    def get_stats(self) -> Dict[str, any]:
        """Get memory pool statistics."""
        with self._lock:
            stats = self._stats.copy()
            
            # Add current pool sizes
            stats['pools'] = {}
            for pool_key, pool in self._pools.items():
                dtype, shape = pool_key
                key_str = f"{dtype}_{shape}"
                stats['pools'][key_str] = len(pool)
            
            # Add active allocations
            stats['active_allocations'] = sum(
                1 for alloc in self._allocated_tensors.values() 
                if not alloc.get('returned', False)
            )
            
            return stats
    
    def clear_pools(self):
        """Clear all memory pools and force garbage collection."""
        self.logger.info("Clearing all memory pools...")
        
        with self._lock:
            for pool in self._pools.values():
                pool.clear()
            
            self._stats['total_allocated_mb'] = 0.0
        
        torch.cuda.empty_cache()
        self.logger.info("Memory pools cleared")


# Global singleton instance
_global_memory_pool: Optional[UnifiedGPUMemoryPool] = None


def get_global_memory_pool(device: str = "cuda:0") -> UnifiedGPUMemoryPool:
    """Get or create the global memory pool instance."""
    global _global_memory_pool
    
    if _global_memory_pool is None:
        _global_memory_pool = UnifiedGPUMemoryPool(device=device)
    
    return _global_memory_pool


# Test function
def test_memory_pool():
    """Test the unified GPU memory pool."""
    logging.basicConfig(level=logging.DEBUG)
    
    pool = get_global_memory_pool()
    
    # Test allocation and reuse
    print("Testing memory pool allocation and reuse...")
    
    # Allocate some tensors
    tensors = []
    for i in range(5):
        tensor, alloc_id = pool.get_tensor((3, 640, 640), torch.float16)
        tensors.append((tensor, alloc_id))
        print(f"Allocated tensor {i}: shape={tensor.shape}, id={alloc_id}")
    
    # Return some tensors
    for i in range(3):
        tensor, alloc_id = tensors[i]
        pool.return_tensor(tensor, alloc_id)
        print(f"Returned tensor {i}")
    
    # Allocate again (should reuse)
    for i in range(3):
        tensor, alloc_id = pool.get_tensor((3, 640, 640), torch.float16)
        print(f"Re-allocated tensor: shape={tensor.shape}, id={alloc_id}")
    
    # Print statistics
    stats = pool.get_stats()
    print(f"\nMemory pool statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_memory_pool() 