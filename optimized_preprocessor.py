#!/usr/bin/env python3
"""
Optimized CPU-based frame preprocessing with better algorithms and threading
"""

import cv2
import numpy as np
import threading
import queue
import time
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class OptimizedFramePreprocessor:
    """
    CPU-optimized frame preprocessor with:
    - Fast resizing algorithms
    - Memory pool for reduced allocations
    - Multi-threaded processing
    - Batch processing capabilities
    """
    
    def __init__(
        self, 
        target_width: int = 640,
        target_height: int = 480,
        num_threads: int = 2,
        enable_batch_processing: bool = True,
        interpolation: int = cv2.INTER_LINEAR
    ):
        """
        Initialize optimized preprocessor
        
        Args:
            target_width: Target frame width
            target_height: Target frame height  
            num_threads: Number of processing threads
            enable_batch_processing: Enable batch processing
            interpolation: OpenCV interpolation method
        """
        self.target_width = target_width
        self.target_height = target_height
        self.target_size = (target_width, target_height)
        self.num_threads = num_threads
        self.enable_batch_processing = enable_batch_processing
        self.interpolation = interpolation
        
        # Performance tracking
        self.processed_frames = 0
        self.total_time = 0.0
        self.lock = threading.Lock()
        
        # Memory pool for common frame sizes
        self.memory_pool = {}
        self.pool_lock = threading.Lock()
        
        # Batch processing queue
        if enable_batch_processing:
            self.batch_queue = queue.Queue(maxsize=16)
            self.result_queue = queue.Queue()
            self.processing_threads = []
            self.stop_event = threading.Event()
            self._start_processing_threads()
        
        logger.info(f"OptimizedFramePreprocessor initialized: {target_width}x{target_height}, {num_threads} threads")
    
    def _start_processing_threads(self):
        """Start background processing threads"""
        for i in range(self.num_threads):
            thread = threading.Thread(
                target=self._processing_worker,
                name=f"OptPreprocess-{i}",
                daemon=True
            )
            thread.start()
            self.processing_threads.append(thread)
    
    def _processing_worker(self):
        """Background processing worker thread"""
        while not self.stop_event.is_set():
            try:
                # Get batch from queue with timeout
                batch_data = self.batch_queue.get(timeout=0.1)
                if batch_data is None:  # Sentinel to stop
                    break
                
                frame, frame_id, result_queue = batch_data
                
                # Process frame
                start_time = time.time()
                processed = self._resize_optimized(frame)
                process_time = time.time() - start_time
                
                # Update stats
                with self.lock:
                    self.processed_frames += 1
                    self.total_time += process_time
                
                # Return result
                result_queue.put((frame_id, processed, process_time))
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing worker: {e}")
    
    def _get_memory_pool_buffer(self, shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
        """Get buffer from memory pool"""
        with self.pool_lock:
            if shape in self.memory_pool:
                return self.memory_pool[shape].pop() if self.memory_pool[shape] else None
        return None
    
    def _return_memory_pool_buffer(self, buffer: np.ndarray):
        """Return buffer to memory pool"""
        shape = buffer.shape
        with self.pool_lock:
            if shape not in self.memory_pool:
                self.memory_pool[shape] = []
            if len(self.memory_pool[shape]) < 4:  # Limit pool size
                self.memory_pool[shape].append(buffer)
    
    def _resize_optimized(self, frame: np.ndarray) -> np.ndarray:
        """
        Optimized frame resizing with memory pooling
        """
        # Check if resize is needed
        if frame.shape[:2] == (self.target_height, self.target_width):
            return frame.copy()
        
        # Try to get buffer from memory pool
        target_shape = (self.target_height, self.target_width, frame.shape[2])
        output_buffer = self._get_memory_pool_buffer(target_shape)
        
        if output_buffer is not None:
            # Use pre-allocated buffer
            cv2.resize(frame, self.target_size, dst=output_buffer, interpolation=self.interpolation)
            return output_buffer
        else:
            # Allocate new buffer
            resized = cv2.resize(frame, self.target_size, interpolation=self.interpolation)
            return resized
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess single frame (synchronous)
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame
        """
        start_time = time.time()
        processed = self._resize_optimized(frame)
        process_time = time.time() - start_time
        
        # Update stats
        with self.lock:
            self.processed_frames += 1
            self.total_time += process_time
        
        return processed
    
    def preprocess_frame_async(self, frame: np.ndarray, frame_id: int = 0) -> queue.Queue:
        """
        Preprocess frame asynchronously (returns result queue)
        
        Args:
            frame: Input frame
            frame_id: Frame identifier
            
        Returns:
            Queue that will contain (frame_id, processed_frame, process_time)
        """
        if not self.enable_batch_processing:
            # Fallback to synchronous processing
            result_queue = queue.Queue()
            processed = self.preprocess_frame(frame)
            result_queue.put((frame_id, processed, 0.0))
            return result_queue
        
        # Submit to batch processing
        result_queue = queue.Queue()
        try:
            self.batch_queue.put((frame, frame_id, result_queue), block=False)
        except queue.Full:
            # Fallback to synchronous if queue is full
            processed = self.preprocess_frame(frame)
            result_queue.put((frame_id, processed, 0.0))
        
        return result_queue
    
    def preprocess_batch(self, frames: list) -> list:
        """
        Preprocess batch of frames
        
        Args:
            frames: List of input frames
            
        Returns:
            List of preprocessed frames
        """
        if not self.enable_batch_processing or len(frames) == 1:
            return [self.preprocess_frame(frame) for frame in frames]
        
        # Submit all frames for async processing
        result_queues = []
        for i, frame in enumerate(frames):
            result_queue = self.preprocess_frame_async(frame, i)
            result_queues.append(result_queue)
        
        # Collect results
        results = [None] * len(frames)
        for i, result_queue in enumerate(result_queues):
            frame_id, processed, _ = result_queue.get(timeout=5.0)  # 5 second timeout
            results[frame_id] = processed
        
        return results
    
    def get_stats(self) -> dict:
        """Get performance statistics"""
        with self.lock:
            avg_time = (self.total_time / self.processed_frames * 1000) if self.processed_frames > 0 else 0
            return {
                'processed_frames': self.processed_frames,
                'avg_time_ms': avg_time,
                'total_time_s': self.total_time,
                'threads': self.num_threads,
                'batch_enabled': self.enable_batch_processing,
                'memory_pool_sizes': {str(k): len(v) for k, v in self.memory_pool.items()}
            }
    
    def reset_stats(self):
        """Reset performance statistics"""
        with self.lock:
            self.processed_frames = 0
            self.total_time = 0.0
    
    def cleanup(self):
        """Cleanup resources"""
        if self.enable_batch_processing:
            # Stop processing threads
            self.stop_event.set()
            
            # Send sentinel values
            for _ in range(self.num_threads):
                try:
                    self.batch_queue.put(None, timeout=1.0)
                except queue.Full:
                    pass
            
            # Wait for threads to finish
            for thread in self.processing_threads:
                thread.join(timeout=2.0)
        
        # Clear memory pool
        with self.pool_lock:
            self.memory_pool.clear()
        
        logger.info("OptimizedFramePreprocessor cleanup completed")
    
    def __del__(self):
        """Destructor"""
        try:
            self.cleanup()
        except:
            pass


class FastResizePreprocessor:
    """
    Ultra-fast resize preprocessor using optimized algorithms
    """
    
    def __init__(self, target_width: int = 640, target_height: int = 480):
        self.target_width = target_width
        self.target_height = target_height
        self.target_size = (target_width, target_height)
        
        # Performance tracking
        self.processed_frames = 0
        self.total_time = 0.0
        
        logger.info(f"FastResizePreprocessor initialized: {target_width}x{target_height}")
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Ultra-fast frame preprocessing using INTER_NEAREST for speed
        """
        start_time = time.time()
        
        # Use fastest interpolation for maximum speed
        if frame.shape[:2] != (self.target_height, self.target_width):
            # Use INTER_NEAREST for maximum speed (sacrificing some quality)
            processed = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_NEAREST)
        else:
            processed = frame
        
        process_time = time.time() - start_time
        self.processed_frames += 1
        self.total_time += process_time
        
        return processed
    
    def get_stats(self) -> dict:
        """Get performance statistics"""
        avg_time = (self.total_time / self.processed_frames * 1000) if self.processed_frames > 0 else 0
        return {
            'processed_frames': self.processed_frames,
            'avg_time_ms': avg_time,
            'total_time_s': self.total_time,
            'algorithm': 'INTER_NEAREST'
        } 