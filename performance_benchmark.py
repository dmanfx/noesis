"""
Performance Benchmarking Framework
==================================

Comprehensive benchmarking tools for validating GPU pipeline performance,
measuring CPU/GPU utilization, memory usage, and end-to-end latency.

Features:
- Automated baseline measurement
- Real-time performance tracking
- Comparative analysis
- Performance regression detection
"""

import time
import torch
import psutil
import pynvml
import threading
import queue
import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import subprocess
import logging
from datetime import datetime
import os


@dataclass
class PerformanceMetrics:
    """Container for performance measurements"""
    timestamp: float
    cpu_percent: float
    cpu_per_core: List[float]
    memory_percent: float
    memory_mb: float
    gpu_utilization: float
    gpu_memory_percent: float
    gpu_memory_mb: float
    gpu_decode_percent: float
    gpu_temperature: float
    frame_rate: float
    latency_ms: float
    memory_transfers_mb: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class GPUMonitor:
    """Monitors GPU performance metrics using pynvml"""
    
    def __init__(self, device_id: int = 0):
        """Initialize GPU monitor"""
        self.device_id = device_id
        self.logger = logging.getLogger("GPUMonitor")
        
        # Initialize NVML
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        
        # Get GPU info
        self.gpu_name = pynvml.nvmlDeviceGetName(self.handle).decode()
        self.total_memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle).total
        
        self.logger.info(f"Monitoring GPU: {self.gpu_name} ({self.total_memory / 1e9:.1f} GB)")
        
    def get_metrics(self) -> Dict[str, float]:
        """Get current GPU metrics"""
        try:
            # Basic utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            
            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            
            # Temperature
            temp = pynvml.nvmlDeviceGetTemperature(
                self.handle, 
                pynvml.NVML_TEMPERATURE_GPU
            )
            
            # Decoder utilization (if available)
            try:
                decoder_util = pynvml.nvmlDeviceGetDecoderUtilization(self.handle)
            except:
                decoder_util = 0
                
            return {
                'gpu_utilization': util.gpu,
                'gpu_memory_percent': (mem_info.used / mem_info.total) * 100,
                'gpu_memory_mb': mem_info.used / 1e6,
                'gpu_decode_percent': decoder_util,
                'gpu_temperature': temp
            }
            
        except Exception as e:
            self.logger.error(f"Error getting GPU metrics: {e}")
            return {
                'gpu_utilization': 0,
                'gpu_memory_percent': 0,
                'gpu_memory_mb': 0,
                'gpu_decode_percent': 0,
                'gpu_temperature': 0
            }
            
    def cleanup(self):
        """Cleanup NVML"""
        try:
            pynvml.nvmlShutdown()
        except:
            pass


class MemoryTransferMonitor:
    """Monitors CPU-GPU memory transfers"""
    
    def __init__(self):
        """Initialize memory transfer monitor"""
        self.logger = logging.getLogger("MemoryTransferMonitor")
        self.last_h2d_bytes = 0
        self.last_d2h_bytes = 0
        self.last_time = time.time()
        
    def get_transfer_rate(self) -> float:
        """Get memory transfer rate in MB/s"""
        try:
            # Use nvidia-smi to get memory transfer info
            result = subprocess.run(
                ['nvidia-smi', 'dmon', '-c', '1', '-s', 'mu'],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if result.returncode == 0:
                # Parse output to get memory utilization
                lines = result.stdout.strip().split('\n')
                if len(lines) > 2:
                    # Extract memory bandwidth usage
                    data = lines[-1].split()
                    if len(data) > 7:
                        # Memory utilization percentage
                        mem_util = float(data[7])
                        # Estimate transfer rate (simplified)
                        return mem_util * 10  # MB/s estimate
            
            return 0.0
            
        except Exception as e:
            self.logger.debug(f"Error monitoring memory transfers: {e}")
            return 0.0


class PerformanceBenchmark:
    """Comprehensive performance benchmarking framework"""
    
    def __init__(self, 
                 output_dir: str = "benchmarks",
                 sample_interval: float = 1.0,
                 warmup_time: float = 10.0):
        """
        Initialize performance benchmark.
        
        Args:
            output_dir: Directory for benchmark results
            sample_interval: Sampling interval in seconds
            warmup_time: Warmup period before measurements
        """
        self.output_dir = output_dir
        self.sample_interval = sample_interval
        self.warmup_time = warmup_time
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize monitors
        self.gpu_monitor = GPUMonitor()
        self.memory_monitor = MemoryTransferMonitor()
        
        # Performance tracking
        self.metrics_queue = queue.Queue()
        self.metrics_history = deque(maxlen=3600)  # 1 hour at 1Hz
        self.running = False
        self.monitor_thread = None
        
        # Frame rate tracking
        self.frame_times = deque(maxlen=100)
        self.last_frame_time = None
        
        # Logger
        self.logger = logging.getLogger("PerformanceBenchmark")
        
    def start(self):
        """Start performance monitoring"""
        self.running = True
        self.start_time = time.time()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.info(f"Started performance benchmark (warmup: {self.warmup_time}s)")
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        # Warmup period
        warmup_end = time.time() + self.warmup_time
        while time.time() < warmup_end and self.running:
            time.sleep(0.1)
            
        self.logger.info("Warmup complete, starting measurements")
        
        while self.running:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                
                # Store metrics
                self.metrics_history.append(metrics)
                self.metrics_queue.put(metrics)
                
                # Wait for next sample
                time.sleep(self.sample_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}")
                
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect all performance metrics"""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
        
        # Memory metrics
        mem = psutil.virtual_memory()
        memory_percent = mem.percent
        memory_mb = mem.used / 1e6
        
        # GPU metrics
        gpu_metrics = self.gpu_monitor.get_metrics()
        
        # Memory transfer rate
        memory_transfers = self.memory_monitor.get_transfer_rate()
        
        # Frame rate
        frame_rate = self._calculate_frame_rate()
        
        # Latency (if available)
        latency_ms = self._get_latency()
        
        return PerformanceMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            cpu_per_core=cpu_per_core,
            memory_percent=memory_percent,
            memory_mb=memory_mb,
            frame_rate=frame_rate,
            latency_ms=latency_ms,
            memory_transfers_mb=memory_transfers,
            **gpu_metrics
        )
        
    def record_frame(self):
        """Record frame processing timestamp"""
        current_time = time.time()
        if self.last_frame_time is not None:
            frame_time = current_time - self.last_frame_time
            self.frame_times.append(frame_time)
        self.last_frame_time = current_time
        
    def _calculate_frame_rate(self) -> float:
        """Calculate current frame rate"""
        if len(self.frame_times) < 2:
            return 0.0
            
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        
    def _get_latency(self) -> float:
        """Get end-to-end latency (placeholder for actual measurement)"""
        # This would be implemented based on actual pipeline latency measurement
        return 0.0
        
    def stop(self):
        """Stop monitoring and save results"""
        self.running = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
            
        # Save results
        self._save_results()
        
        # Cleanup
        self.gpu_monitor.cleanup()
        
        self.logger.info("Performance benchmark stopped")
        
    def _save_results(self):
        """Save benchmark results to file"""
        if not self.metrics_history:
            return
            
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"benchmark_{timestamp}.json")
        
        # Convert metrics to dict
        results = {
            'metadata': {
                'start_time': self.start_time,
                'end_time': time.time(),
                'duration': time.time() - self.start_time,
                'sample_interval': self.sample_interval,
                'gpu_name': self.gpu_monitor.gpu_name,
                'total_samples': len(self.metrics_history)
            },
            'metrics': [m.to_dict() for m in self.metrics_history],
            'summary': self._calculate_summary()
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
            
        self.logger.info(f"Saved benchmark results to {filename}")
        
    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics"""
        if not self.metrics_history:
            return {}
            
        # Convert to numpy arrays for easy calculation
        cpu_values = [m.cpu_percent for m in self.metrics_history]
        gpu_util_values = [m.gpu_utilization for m in self.metrics_history]
        gpu_mem_values = [m.gpu_memory_percent for m in self.metrics_history]
        fps_values = [m.frame_rate for m in self.metrics_history if m.frame_rate > 0]
        transfer_values = [m.memory_transfers_mb for m in self.metrics_history]
        
        return {
            'cpu': {
                'mean': np.mean(cpu_values),
                'min': np.min(cpu_values),
                'max': np.max(cpu_values),
                'std': np.std(cpu_values),
                'p50': np.percentile(cpu_values, 50),
                'p90': np.percentile(cpu_values, 90),
                'p99': np.percentile(cpu_values, 99)
            },
            'gpu_utilization': {
                'mean': np.mean(gpu_util_values),
                'min': np.min(gpu_util_values),
                'max': np.max(gpu_util_values),
                'std': np.std(gpu_util_values)
            },
            'gpu_memory': {
                'mean': np.mean(gpu_mem_values),
                'min': np.min(gpu_mem_values),
                'max': np.max(gpu_mem_values)
            },
            'frame_rate': {
                'mean': np.mean(fps_values) if fps_values else 0,
                'min': np.min(fps_values) if fps_values else 0,
                'max': np.max(fps_values) if fps_values else 0
            },
            'memory_transfers': {
                'mean': np.mean(transfer_values),
                'max': np.max(transfer_values)
            }
        }
        
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get most recent metrics"""
        try:
            return self.metrics_queue.get_nowait()
        except queue.Empty:
            return None
            
    def print_summary(self):
        """Print performance summary to console"""
        summary = self._calculate_summary()
        
        if not summary:
            print("No data collected yet")
            return
            
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("="*60)
        
        print(f"\nCPU Usage:")
        print(f"  Mean: {summary['cpu']['mean']:.1f}%")
        print(f"  Range: {summary['cpu']['min']:.1f}% - {summary['cpu']['max']:.1f}%")
        print(f"  P90: {summary['cpu']['p90']:.1f}%")
        print(f"  P99: {summary['cpu']['p99']:.1f}%")
        
        print(f"\nGPU Utilization:")
        print(f"  Mean: {summary['gpu_utilization']['mean']:.1f}%")
        print(f"  Range: {summary['gpu_utilization']['min']:.1f}% - {summary['gpu_utilization']['max']:.1f}%")
        
        print(f"\nGPU Memory:")
        print(f"  Mean: {summary['gpu_memory']['mean']:.1f}%")
        print(f"  Max: {summary['gpu_memory']['max']:.1f}%")
        
        print(f"\nFrame Rate:")
        print(f"  Mean: {summary['frame_rate']['mean']:.1f} FPS")
        print(f"  Range: {summary['frame_rate']['min']:.1f} - {summary['frame_rate']['max']:.1f} FPS")
        
        print(f"\nMemory Transfers:")
        print(f"  Mean: {summary['memory_transfers']['mean']:.1f} MB/s")
        print(f"  Max: {summary['memory_transfers']['max']:.1f} MB/s")
        
        print("="*60 + "\n")


class PerformanceValidator:
    """Validates performance against target metrics"""
    
    def __init__(self, targets: Dict[str, Dict[str, float]]):
        """
        Initialize validator with target metrics.
        
        Args:
            targets: Target performance metrics
        """
        self.targets = targets
        self.logger = logging.getLogger("PerformanceValidator")
        
    def validate(self, summary: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate performance against targets.
        
        Returns:
            Tuple of (success, list of failures)
        """
        failures = []
        
        # CPU usage validation
        if 'cpu' in self.targets and 'cpu' in summary:
            cpu_mean = summary['cpu']['mean']
            cpu_target = self.targets['cpu']['max']
            
            if cpu_mean > cpu_target:
                failures.append(
                    f"CPU usage {cpu_mean:.1f}% exceeds target {cpu_target:.1f}%"
                )
                
        # GPU utilization validation
        if 'gpu' in self.targets and 'gpu_utilization' in summary:
            gpu_mean = summary['gpu_utilization']['mean']
            gpu_min_target = self.targets['gpu'].get('min', 0)
            
            if gpu_mean < gpu_min_target:
                failures.append(
                    f"GPU utilization {gpu_mean:.1f}% below target {gpu_min_target:.1f}%"
                )
                
        # Memory transfer validation
        if 'memory_transfers' in self.targets and 'memory_transfers' in summary:
            transfer_mean = summary['memory_transfers']['mean']
            transfer_target = self.targets['memory_transfers']['max']
            
            if transfer_mean > transfer_target:
                failures.append(
                    f"Memory transfers {transfer_mean:.1f} MB/s exceed target {transfer_target:.1f} MB/s"
                )
                
        # Frame rate validation
        if 'frame_rate' in self.targets and 'frame_rate' in summary:
            fps_mean = summary['frame_rate']['mean']
            fps_target = self.targets['frame_rate']['min']
            
            if fps_mean < fps_target:
                failures.append(
                    f"Frame rate {fps_mean:.1f} FPS below target {fps_target:.1f} FPS"
                )
                
        success = len(failures) == 0
        return success, failures


# Default performance targets
DEFAULT_TARGETS = {
    'cpu': {'max': 10.0},  # 10% max CPU usage
    'gpu': {'min': 60.0},  # 60% min GPU utilization
    'memory_transfers': {'max': 1.0},  # 1 MB/s max transfers
    'frame_rate': {'min': 25.0}  # 25 FPS minimum
}


def run_benchmark(duration: int = 300, targets: Optional[Dict] = None) -> bool:
    """
    Run performance benchmark.
    
    Args:
        duration: Benchmark duration in seconds
        targets: Performance targets (uses defaults if None)
        
    Returns:
        True if performance meets targets
    """
    print(f"Starting {duration}s performance benchmark...")
    
    # Create benchmark
    benchmark = PerformanceBenchmark()
    
    # Create validator
    validator = PerformanceValidator(targets or DEFAULT_TARGETS)
    
    try:
        # Start benchmark
        benchmark.start()
        
        # Run for specified duration
        end_time = time.time() + duration
        while time.time() < end_time:
            metrics = benchmark.get_current_metrics()
            if metrics:
                print(f"\rCPU: {metrics.cpu_percent:5.1f}% | "
                      f"GPU: {metrics.gpu_utilization:5.1f}% | "
                      f"FPS: {metrics.frame_rate:5.1f}", end='')
            time.sleep(1)
            
        print()  # New line
        
        # Stop and get results
        benchmark.stop()
        
        # Print summary
        benchmark.print_summary()
        
        # Validate results
        summary = benchmark._calculate_summary()
        success, failures = validator.validate(summary)
        
        if success:
            print("✅ Performance validation PASSED")
        else:
            print("❌ Performance validation FAILED:")
            for failure in failures:
                print(f"  - {failure}")
                
        return success
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted")
        benchmark.stop()
        return False
        
    except Exception as e:
        print(f"Benchmark error: {e}")
        benchmark.stop()
        return False


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU Pipeline Performance Benchmark")
    parser.add_argument('--duration', type=int, default=300,
                        help='Benchmark duration in seconds (default: 300)')
    parser.add_argument('--cpu-target', type=float, default=10.0,
                        help='Maximum CPU usage target (default: 10.0%)')
    parser.add_argument('--gpu-target', type=float, default=60.0,
                        help='Minimum GPU utilization target (default: 60.0%)')
    
    args = parser.parse_args()
    
    # Custom targets
    targets = {
        'cpu': {'max': args.cpu_target},
        'gpu': {'min': args.gpu_target},
        'memory_transfers': {'max': 1.0},
        'frame_rate': {'min': 25.0}
    }
    
    # Run benchmark
    success = run_benchmark(duration=args.duration, targets=targets)
    exit(0 if success else 1) 