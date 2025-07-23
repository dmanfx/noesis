"""
Comprehensive CPU Profiling Utilities

This module provides extensive CPU profiling capabilities for the video processing pipeline
to identify sources of CPU utilization. It includes:

- Per-thread CPU monitoring
- Per-process CPU monitoring  
- Function-level timing and CPU usage
- System-wide resource monitoring
- Real-time profiling data collection and reporting

The profiling system tracks:
- Individual thread CPU usage
- Process CPU usage breakdown
- Memory usage patterns
- Queue sizes and throughput
- Frame processing bottlenecks
- WebSocket streaming overhead
"""

import psutil
import threading
import time
import logging
import os
import sys
import functools
from contextlib import contextmanager
from typing import Dict, Any, Optional, List, Callable, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass, field
import multiprocessing
import queue
import json
from pathlib import Path


@dataclass
class CPUProfileData:
    """Data structure for CPU profiling information"""
    timestamp: float
    process_id: int
    thread_id: int
    thread_name: str
    cpu_percent: float
    memory_mb: float
    function_name: Optional[str] = None
    duration_ms: Optional[float] = None
    context: Optional[str] = None


@dataclass
class SystemResourceData:
    """System-wide resource usage data"""
    timestamp: float
    total_cpu_percent: float
    available_memory_mb: float
    used_memory_mb: float
    memory_percent: float
    active_processes: int
    active_threads: int
    gpu_utilization: Optional[float] = None
    gpu_memory_used: Optional[float] = None


class ThreadCPUMonitor:
    """Monitors CPU usage for individual threads"""
    
    def __init__(self, sample_interval: float = 0.1):
        self.sample_interval = sample_interval
        self.running = False
        self.monitor_thread = None
        self.thread_data = defaultdict(deque)
        self.lock = threading.Lock()
        self.logger = logging.getLogger("ThreadCPUMonitor")
        
        # Track thread CPU times
        self.thread_cpu_times = {}
        self.last_sample_time = time.time()
        
    def start(self):
        """Start thread CPU monitoring"""
        if self.running:
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="ThreadCPUMonitor",
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info("Thread CPU monitoring started")
        
    def stop(self):
        """Stop thread CPU monitoring"""
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        self.logger.info("Thread CPU monitoring stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        process = psutil.Process()
        
        while self.running:
            try:
                current_time = time.time()
                
                # Get all threads for current process
                threads = process.threads()
                
                # Sample thread CPU usage
                for thread_info in threads:
                    thread_id = thread_info.id
                    
                    # Get thread name if available
                    thread_name = self._get_thread_name(thread_id)
                    
                    # Calculate CPU usage for this thread
                    cpu_percent = self._calculate_thread_cpu(thread_id, thread_info, current_time)
                    
                    # Get memory info (approximated per thread)
                    memory_info = process.memory_info()
                    memory_mb = memory_info.rss / (1024 * 1024) / len(threads)
                    
                    # Store profiling data
                    profile_data = CPUProfileData(
                        timestamp=current_time,
                        process_id=os.getpid(),
                        thread_id=thread_id,
                        thread_name=thread_name,
                        cpu_percent=cpu_percent,
                        memory_mb=memory_mb
                    )
                    
                    with self.lock:
                        self.thread_data[thread_id].append(profile_data)
                        # Keep only last 1000 samples per thread
                        if len(self.thread_data[thread_id]) > 1000:
                            self.thread_data[thread_id].popleft()
                
                self.last_sample_time = current_time
                time.sleep(self.sample_interval)
                
            except Exception as e:
                self.logger.error(f"Error in thread CPU monitoring: {e}")
                time.sleep(self.sample_interval)
                
    def _get_thread_name(self, thread_id: int) -> str:
        """Get thread name from thread ID"""
        try:
            # Try to find thread by ID in current process
            for thread in threading.enumerate():
                if hasattr(thread, 'ident') and thread.ident == thread_id:
                    return thread.name
            return f"Thread-{thread_id}"
        except:
            return f"Thread-{thread_id}"
            
    def _calculate_thread_cpu(self, thread_id: int, thread_info, current_time: float) -> float:
        """Calculate CPU percentage for a specific thread"""
        try:
            # Get current CPU times for thread
            current_user_time = thread_info.user_time
            current_system_time = thread_info.system_time
            current_total = current_user_time + current_system_time
            
            # Get previous times
            if thread_id in self.thread_cpu_times:
                prev_time, prev_total = self.thread_cpu_times[thread_id]
                time_delta = current_time - prev_time
                cpu_delta = current_total - prev_total
                
                if time_delta > 0:
                    cpu_percent = (cpu_delta / time_delta) * 100.0
                else:
                    cpu_percent = 0.0
            else:
                cpu_percent = 0.0
                
            # Store current times for next calculation
            self.thread_cpu_times[thread_id] = (current_time, current_total)
            
            return min(cpu_percent, 100.0)  # Cap at 100%
            
        except Exception as e:
            self.logger.debug(f"Error calculating CPU for thread {thread_id}: {e}")
            return 0.0
            
    def get_thread_stats(self) -> Dict[int, Dict[str, Any]]:
        """Get CPU statistics for all monitored threads"""
        stats = {}
        
        with self.lock:
            for thread_id, data_deque in self.thread_data.items():
                if not data_deque:
                    continue
                    
                recent_data = list(data_deque)[-50:]  # Last 50 samples
                
                cpu_values = [d.cpu_percent for d in recent_data]
                memory_values = [d.memory_mb for d in recent_data]
                
                stats[thread_id] = {
                    'thread_name': recent_data[-1].thread_name,
                    'avg_cpu_percent': sum(cpu_values) / len(cpu_values),
                    'max_cpu_percent': max(cpu_values),
                    'current_cpu_percent': cpu_values[-1] if cpu_values else 0,
                    'avg_memory_mb': sum(memory_values) / len(memory_values),
                    'sample_count': len(recent_data),
                    'last_update': recent_data[-1].timestamp
                }
                
        return stats


class SystemResourceMonitor:
    """Monitors system-wide resource usage"""
    
    def __init__(self, sample_interval: float = 1.0):
        self.sample_interval = sample_interval
        self.running = False
        self.monitor_thread = None
        self.resource_data = deque(maxlen=300)  # Keep 5 minutes of data
        self.lock = threading.Lock()
        self.logger = logging.getLogger("SystemResourceMonitor")
        
        # Try to import GPU monitoring
        self.gpu_available = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.gpu_available = True
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.logger.info("GPU monitoring enabled")
        except:
            self.logger.info("GPU monitoring not available")
            
    def start(self):
        """Start system resource monitoring"""
        if self.running:
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="SystemResourceMonitor", 
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info("System resource monitoring started")
        
    def stop(self):
        """Stop system resource monitoring"""
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        self.logger.info("System resource monitoring stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                current_time = time.time()
                
                # Get CPU usage
                cpu_percent = psutil.cpu_percent(interval=None)
                
                # Get memory usage
                memory = psutil.virtual_memory()
                
                # Count active processes and threads
                active_processes = len(psutil.pids())
                active_threads = sum(proc.num_threads() for proc in psutil.process_iter(['num_threads']) if proc.info['num_threads'])
                
                # Get GPU usage if available
                gpu_util = None
                gpu_memory = None
                if self.gpu_available:
                    try:
                        import pynvml
                        gpu_util_info = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                        gpu_util = gpu_util_info.gpu
                        
                        gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                        gpu_memory = gpu_mem_info.used / (1024 * 1024)  # MB
                    except:
                        pass
                
                # Create resource data
                resource_data = SystemResourceData(
                    timestamp=current_time,
                    total_cpu_percent=cpu_percent,
                    available_memory_mb=memory.available / (1024 * 1024),
                    used_memory_mb=memory.used / (1024 * 1024),
                    memory_percent=memory.percent,
                    active_processes=active_processes,
                    active_threads=active_threads,
                    gpu_utilization=gpu_util,
                    gpu_memory_used=gpu_memory
                )
                
                with self.lock:
                    self.resource_data.append(resource_data)
                    
                time.sleep(self.sample_interval)
                
            except Exception as e:
                self.logger.error(f"Error in system resource monitoring: {e}")
                time.sleep(self.sample_interval)
                
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current system resource statistics"""
        with self.lock:
            if not self.resource_data:
                return {}
                
            recent_data = list(self.resource_data)[-10:]  # Last 10 samples
            current = recent_data[-1]
            
            return {
                'current_cpu_percent': current.total_cpu_percent,
                'avg_cpu_percent': sum(d.total_cpu_percent for d in recent_data) / len(recent_data),
                'current_memory_percent': current.memory_percent,
                'used_memory_mb': current.used_memory_mb,
                'available_memory_mb': current.available_memory_mb,
                'active_processes': current.active_processes,
                'active_threads': current.active_threads,
                'gpu_utilization': current.gpu_utilization,
                'gpu_memory_used': current.gpu_memory_used,
                'last_update': current.timestamp
            }


class FunctionProfiler:
    """Profiles individual functions for CPU usage and timing"""
    
    def __init__(self):
        self.function_stats = defaultdict(lambda: {
            'call_count': 0,
            'total_time': 0.0,
            'total_cpu_time': 0.0,
            'max_time': 0.0,
            'min_time': float('inf'),
            'recent_calls': deque(maxlen=100)
        })
        self.lock = threading.Lock()
        self.logger = logging.getLogger("FunctionProfiler")
        
    def profile_function(self, func_name: str = None):
        """Decorator to profile a function"""
        def decorator(func):
            nonlocal func_name
            if func_name is None:
                func_name = f"{func.__module__}.{func.__qualname__}"
                
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self._profile_call(func_name, func, args, kwargs)
            return wrapper
        return decorator
        
    def _profile_call(self, func_name: str, func: Callable, args: tuple, kwargs: dict):
        """Profile a function call"""
        # Get current process for CPU timing
        process = psutil.Process()
        
        # Record start times
        start_time = time.time()
        start_cpu_times = process.cpu_times()
        start_cpu_total = start_cpu_times.user + start_cpu_times.system
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # Record end times
            end_time = time.time()
            end_cpu_times = process.cpu_times()
            end_cpu_total = end_cpu_times.user + end_cpu_times.system
            
            # Calculate durations
            wall_time = end_time - start_time
            cpu_time = end_cpu_total - start_cpu_total
            
            # Update statistics
            with self.lock:
                stats = self.function_stats[func_name]
                stats['call_count'] += 1
                stats['total_time'] += wall_time
                stats['total_cpu_time'] += cpu_time
                stats['max_time'] = max(stats['max_time'], wall_time)
                stats['min_time'] = min(stats['min_time'], wall_time)
                
                # Store recent call data
                call_data = {
                    'timestamp': end_time,
                    'wall_time': wall_time,
                    'cpu_time': cpu_time,
                    'thread_id': threading.get_ident(),
                    'thread_name': threading.current_thread().name
                }
                stats['recent_calls'].append(call_data)
                
            return result
            
        except Exception as e:
            # Still record timing for failed calls
            end_time = time.time()
            wall_time = end_time - start_time
            
            with self.lock:
                stats = self.function_stats[func_name]
                stats['call_count'] += 1
                stats['total_time'] += wall_time
                
            raise
            
    def get_function_stats(self) -> Dict[str, Any]:
        """Get statistics for all profiled functions"""
        stats = {}
        
        with self.lock:
            for func_name, func_stats in self.function_stats.items():
                if func_stats['call_count'] == 0:
                    continue
                    
                avg_time = func_stats['total_time'] / func_stats['call_count']
                avg_cpu_time = func_stats['total_cpu_time'] / func_stats['call_count']
                
                # Get recent performance
                recent_calls = list(func_stats['recent_calls'])
                recent_avg_time = 0.0
                recent_avg_cpu = 0.0
                if recent_calls:
                    recent_avg_time = sum(c['wall_time'] for c in recent_calls) / len(recent_calls)
                    recent_avg_cpu = sum(c['cpu_time'] for c in recent_calls) / len(recent_calls)
                
                stats[func_name] = {
                    'call_count': func_stats['call_count'],
                    'total_time_sec': func_stats['total_time'],
                    'total_cpu_time_sec': func_stats['total_cpu_time'],
                    'avg_time_ms': avg_time * 1000,
                    'avg_cpu_time_ms': avg_cpu_time * 1000,
                    'max_time_ms': func_stats['max_time'] * 1000,
                    'min_time_ms': func_stats['min_time'] * 1000 if func_stats['min_time'] != float('inf') else 0,
                    'recent_avg_time_ms': recent_avg_time * 1000,
                    'recent_avg_cpu_ms': recent_avg_cpu * 1000,
                    'recent_calls_count': len(recent_calls)
                }
                
        return stats


class ComprehensiveCPUProfiler:
    """Main CPU profiler that coordinates all monitoring components"""
    
    def __init__(self, 
                 thread_sample_interval: float = 0.1,
                 system_sample_interval: float = 1.0,
                 enable_function_profiling: bool = True):
        self.thread_monitor = ThreadCPUMonitor(thread_sample_interval)
        self.system_monitor = SystemResourceMonitor(system_sample_interval)
        self.function_profiler = FunctionProfiler() if enable_function_profiling else None
        
        self.running = False
        self.logger = logging.getLogger("ComprehensiveCPUProfiler")
        
        # Performance data storage
        self.profile_data_file = None
        self.save_interval = 30.0  # Save every 30 seconds
        self.last_save_time = time.time()
        
    def start(self, profile_data_file: Optional[str] = None):
        """Start comprehensive CPU profiling"""
        if self.running:
            return
            
        self.running = True
        self.profile_data_file = profile_data_file
        
        # Start all monitors
        self.thread_monitor.start()
        self.system_monitor.start()
        
        self.logger.info("Comprehensive CPU profiling started")
        
    def stop(self):
        """Stop comprehensive CPU profiling"""
        if not self.running:
            return
            
        self.running = False
        
        # Stop all monitors
        self.thread_monitor.stop()
        self.system_monitor.stop()
        
        # Save final data
        if self.profile_data_file:
            self.save_profile_data()
            
        self.logger.info("Comprehensive CPU profiling stopped")
        
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive profiling statistics"""
        stats = {
            'timestamp': time.time(),
            'system': self.system_monitor.get_current_stats(),
            'threads': self.thread_monitor.get_thread_stats(),
            'functions': self.function_profiler.get_function_stats() if self.function_profiler else {}
        }
        
        # Add summary statistics
        thread_stats = stats['threads']
        if thread_stats:
            total_thread_cpu = sum(t['current_cpu_percent'] for t in thread_stats.values())
            max_thread_cpu = max(t['current_cpu_percent'] for t in thread_stats.values())
            active_threads = len([t for t in thread_stats.values() if t['current_cpu_percent'] > 1.0])
            
            stats['summary'] = {
                'total_thread_cpu_percent': total_thread_cpu,
                'max_thread_cpu_percent': max_thread_cpu,
                'active_high_cpu_threads': active_threads,
                'total_monitored_threads': len(thread_stats)
            }
        else:
            stats['summary'] = {
                'total_thread_cpu_percent': 0,
                'max_thread_cpu_percent': 0,
                'active_high_cpu_threads': 0,
                'total_monitored_threads': 0
            }
            
        return stats
        
    def save_profile_data(self):
        """Save current profiling data to file"""
        if not self.profile_data_file:
            return
            
        try:
            stats = self.get_comprehensive_stats()
            
            # Ensure directory exists
            Path(self.profile_data_file).parent.mkdir(parents=True, exist_ok=True)
            
            # Append to file
            with open(self.profile_data_file, 'a') as f:
                f.write(json.dumps(stats) + '\n')
                
            self.last_save_time = time.time()
            self.logger.debug(f"Profile data saved to {self.profile_data_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving profile data: {e}")
            
    def should_save_data(self) -> bool:
        """Check if it's time to save profile data"""
        return (self.profile_data_file and 
                time.time() - self.last_save_time >= self.save_interval)
                
    def profile_function(self, func_name: str = None):
        """Decorator to profile a function (if function profiling is enabled)"""
        if self.function_profiler:
            return self.function_profiler.profile_function(func_name)
        else:
            # Return pass-through decorator if function profiling disabled
            def decorator(func):
                return func
            return decorator


# Global profiler instance
_global_profiler: Optional[ComprehensiveCPUProfiler] = None


def get_global_profiler() -> Optional[ComprehensiveCPUProfiler]:
    """Get the global CPU profiler instance"""
    return _global_profiler


def start_global_profiling(profile_data_file: Optional[str] = None,
                          thread_sample_interval: float = 0.1,
                          system_sample_interval: float = 1.0,
                          enable_function_profiling: bool = True):
    """Start global CPU profiling"""
    global _global_profiler
    
    if _global_profiler is not None:
        return _global_profiler
        
    _global_profiler = ComprehensiveCPUProfiler(
        thread_sample_interval=thread_sample_interval,
        system_sample_interval=system_sample_interval,
        enable_function_profiling=enable_function_profiling
    )
    
    _global_profiler.start(profile_data_file)
    return _global_profiler


def stop_global_profiling():
    """Stop global CPU profiling"""
    global _global_profiler
    
    if _global_profiler is not None:
        _global_profiler.stop()
        _global_profiler = None


@contextmanager
def cpu_profile_context(func_name: str):
    """Context manager for profiling a code block"""
    profiler = get_global_profiler()
    if profiler and profiler.function_profiler:
        # Create a dummy function to profile
        def dummy_func():
            pass
        
        # Use the profiler's internal method
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            # Manually record the timing
            with profiler.function_profiler.lock:
                stats = profiler.function_profiler.function_stats[func_name]
                stats['call_count'] += 1
                duration = end_time - start_time
                stats['total_time'] += duration
                stats['max_time'] = max(stats['max_time'], duration)
                stats['min_time'] = min(stats['min_time'], duration)
    else:
        yield


def profile_function(func_name: str = None):
    """Decorator to profile a function using global profiler"""
    profiler = get_global_profiler()
    if profiler:
        return profiler.profile_function(func_name)
    else:
        # Return pass-through decorator if no global profiler
        def decorator(func):
            return func
        return decorator 