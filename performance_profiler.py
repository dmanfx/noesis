#!/usr/bin/env python3
"""
Optimized Performance Profiler

This module provides lightweight profiling with minimal CPU impact:
- Sampling-based profiling (every Nth frame)
- Async logging to reduce main thread impact
- Production-safe lightweight mode
- Performance impact measurement
"""

import time
import logging
import threading
import queue
import asyncio
from typing import Dict, Any, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass
import json
import os


@dataclass
class ProfileEvent:
    """Single profiling event"""
    name: str
    start_time: float
    end_time: float
    thread_id: int
    metadata: Dict[str, Any] = None


class AsyncProfilerLogger:
    """Asynchronous logger for profiling data to minimize main thread impact"""
    
    def __init__(self, log_file: str = "logs/profiler.jsonl", max_queue_size: int = 1000):
        self.log_file = log_file
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.running = False
        self.worker_thread = None
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def start(self):
        """Start async logging worker"""
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker, daemon=True)
            self.worker_thread.start()
    
    def stop(self):
        """Stop async logging worker"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
    
    def log_event(self, event: ProfileEvent):
        """Queue an event for async logging"""
        try:
            event_data = {
                'name': event.name,
                'duration_ms': (event.end_time - event.start_time) * 1000,
                'start_time': event.start_time,
                'thread_id': event.thread_id,
                'metadata': event.metadata or {}
            }
            self.queue.put_nowait(json.dumps(event_data))
        except queue.Full:
            # Drop events if queue is full to avoid blocking
            pass
    
    def _worker(self):
        """Background worker for writing log events"""
        with open(self.log_file, 'a') as f:
            while self.running:
                try:
                    event_json = self.queue.get(timeout=1.0)
                    f.write(event_json + '\n')
                    f.flush()
                except queue.Empty:
                    continue
                except Exception:
                    # Continue on errors to avoid crashing
                    continue


class LightweightProfiler:
    """
    Lightweight profiler with minimal CPU impact for production use.
    
    Features:
    - Sampling-based profiling (configurable)
    - Async logging
    - Performance impact measurement
    - Production-safe operation
    """
    
    def __init__(self, 
                 enable_profiling: bool = False,
                 sampling_rate: int = 10,  # Profile every Nth frame
                 lightweight_mode: bool = True,
                 async_logging: bool = True):
        
        self.enable_profiling = enable_profiling
        self.sampling_rate = sampling_rate
        self.lightweight_mode = lightweight_mode
        self.frame_counter = 0
        
        # Async logger
        self.async_logger = AsyncProfilerLogger() if async_logging else None
        if self.async_logger:
            self.async_logger.start()
        
        # Lightweight stats (minimal memory overhead)
        self.stats = defaultdict(lambda: {
            'count': 0, 
            'total_time': 0.0, 
            'max_time': 0.0,
            'samples': deque(maxlen=100)  # Keep last 100 samples
        })
        
        # Performance impact measurement
        self.profiler_overhead = {
            'total_overhead_ms': 0.0,
            'sample_count': 0
        }
        
        self.logger = logging.getLogger("LightweightProfiler")
        
        if self.enable_profiling:
            self.logger.info(f"Profiler enabled: sampling_rate={sampling_rate}, "
                           f"lightweight={lightweight_mode}, async={async_logging}")
    
    def should_profile(self) -> bool:
        """Determine if this frame should be profiled based on sampling rate"""
        if not self.enable_profiling:
            return False
        
        self.frame_counter += 1
        return (self.frame_counter % self.sampling_rate) == 0
    
    def profile_step(self, name: str, metadata: Dict[str, Any] = None):
        """Context manager for profiling a step with minimal overhead"""
        return ProfileStep(self, name, metadata)
    
    def record_event(self, event: ProfileEvent):
        """Record a profiling event"""
        if not self.enable_profiling:
            return
        
        # Measure profiler overhead
        overhead_start = time.perf_counter()
        
        # Update lightweight stats
        duration = event.end_time - event.start_time
        stats = self.stats[event.name]
        stats['count'] += 1
        stats['total_time'] += duration
        stats['max_time'] = max(stats['max_time'], duration)
        
        if not self.lightweight_mode:
            stats['samples'].append(duration)
        
        # Async logging if enabled
        if self.async_logger:
            self.async_logger.log_event(event)
        
        # Measure overhead
        overhead_end = time.perf_counter()
        overhead_ms = (overhead_end - overhead_start) * 1000
        self.profiler_overhead['total_overhead_ms'] += overhead_ms
        self.profiler_overhead['sample_count'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current profiling statistics"""
        result = {}
        
        for name, stats in self.stats.items():
            avg_time = stats['total_time'] / max(1, stats['count'])
            result[name] = {
                'count': stats['count'],
                'avg_time_ms': avg_time * 1000,
                'max_time_ms': stats['max_time'] * 1000,
                'total_time_ms': stats['total_time'] * 1000
            }
            
            # Add percentiles if samples available
            if not self.lightweight_mode and stats['samples']:
                samples_ms = [s * 1000 for s in sorted(stats['samples'])]
                result[name].update({
                    'p50_ms': samples_ms[len(samples_ms) // 2],
                    'p95_ms': samples_ms[int(len(samples_ms) * 0.95)],
                    'p99_ms': samples_ms[int(len(samples_ms) * 0.99)]
                })
        
        # Add profiler overhead stats
        avg_overhead = (self.profiler_overhead['total_overhead_ms'] / 
                       max(1, self.profiler_overhead['sample_count']))
        
        result['_profiler_overhead'] = {
            'avg_overhead_us': avg_overhead * 1000,  # Convert to microseconds
            'total_overhead_ms': self.profiler_overhead['total_overhead_ms'],
            'sample_count': self.profiler_overhead['sample_count']
        }
        
        return result
    
    def reset_stats(self):
        """Reset all statistics"""
        self.stats.clear()
        self.profiler_overhead = {'total_overhead_ms': 0.0, 'sample_count': 0}
        self.frame_counter = 0
    
    def __del__(self):
        """Cleanup async logger"""
        if self.async_logger:
            self.async_logger.stop()


class ProfileStep:
    """Context manager for profiling individual steps"""
    
    def __init__(self, profiler: LightweightProfiler, name: str, metadata: Dict[str, Any] = None):
        self.profiler = profiler
        self.name = name
        self.metadata = metadata
        self.start_time = None
        self.should_profile = profiler.should_profile()
    
    def __enter__(self):
        if self.should_profile:
            self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.should_profile and self.start_time is not None:
            end_time = time.perf_counter()
            event = ProfileEvent(
                name=self.name,
                start_time=self.start_time,
                end_time=end_time,
                thread_id=threading.get_ident(),
                metadata=self.metadata
            )
            self.profiler.record_event(event)


# Global profiler instance
_global_profiler: Optional[LightweightProfiler] = None


def get_profiler() -> LightweightProfiler:
    """Get or create global profiler instance"""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = LightweightProfiler()
    return _global_profiler


def configure_profiler(enable_profiling: bool = False, 
                      sampling_rate: int = 10, 
                      lightweight_mode: bool = True,
                      async_logging: bool = True):
    """Configure global profiler settings"""
    global _global_profiler
    _global_profiler = LightweightProfiler(
        enable_profiling=enable_profiling,
        sampling_rate=sampling_rate,
        lightweight_mode=lightweight_mode,
        async_logging=async_logging
    )
    return _global_profiler


def profile_step(name: str, metadata: Dict[str, Any] = None):
    """Decorator/context manager for profiling steps"""
    profiler = get_profiler()
    return profiler.profile_step(name, metadata)


# Compatibility functions for existing code
def profile_function(func: Callable) -> Callable:
    """Decorator for profiling entire functions"""
    def wrapper(*args, **kwargs):
        with profile_step(func.__name__):
            return func(*args, **kwargs)
    return wrapper


# Test function
def test_lightweight_profiler():
    """Test the lightweight profiler"""
    import random
    
    # Configure profiler
    profiler = configure_profiler(
        enable_profiling=True,
        sampling_rate=2,  # Profile every 2nd operation
        lightweight_mode=False,  # Get detailed stats for testing
        async_logging=True
    )
    
    print("Testing Lightweight Profiler...")
    
    # Simulate processing multiple frames
    for frame_id in range(20):
        # Simulate frame processing steps
        with profile_step("frame_decode", {"frame_id": frame_id}):
            time.sleep(random.uniform(0.001, 0.005))  # 1-5ms
        
        with profile_step("frame_preprocess", {"frame_id": frame_id}):
            time.sleep(random.uniform(0.002, 0.008))  # 2-8ms
        
        with profile_step("inference", {"frame_id": frame_id}):
            time.sleep(random.uniform(0.010, 0.020))  # 10-20ms
        
        with profile_step("postprocess", {"frame_id": frame_id}):
            time.sleep(random.uniform(0.001, 0.003))  # 1-3ms
    
    # Get and print stats
    stats = profiler.get_stats()
    print("\nProfiling Results:")
    print(json.dumps(stats, indent=2))
    
    # Check profiler overhead
    overhead = stats.get('_profiler_overhead', {})
    print(f"\nProfiler Overhead: {overhead.get('avg_overhead_us', 0):.2f} μs per sample")


if __name__ == "__main__":
    test_lightweight_profiler() 