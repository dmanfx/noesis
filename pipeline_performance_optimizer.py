"""
Pipeline Performance Optimizer
===============================

Optimizes UnifiedGPUPipeline performance by:
- Minimizing tensor copying between pipeline stages
- Implementing efficient pipeline scheduling
- Adding performance profiling hooks (DEBUG LEVEL ONLY)
- Monitoring GPU utilization and resource usage
"""

import torch
import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from collections import deque, defaultdict
from dataclasses import dataclass
import numpy as np

# GPU monitoring imports
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    logging.getLogger(__name__).warning("pynvml not available - GPU monitoring limited")


@dataclass
class PipelineStageMetrics:
    """Metrics for a single pipeline stage"""
    name: str
    execution_times: deque  # Rolling window of execution times
    tensor_copies: int = 0
    gpu_memory_used: int = 0
    last_update: float = 0.0
    
    def __post_init__(self):
        if not isinstance(self.execution_times, deque):
            self.execution_times = deque(maxlen=100)


@dataclass
class TensorFlowOptimization:
    """Tracks tensor flow to minimize copies"""
    tensor_id: int
    device: str
    shape: Tuple[int, ...]
    dtype: torch.dtype
    copy_count: int = 0
    last_stage: str = ""
    creation_time: float = 0.0


class ZeroCopyTensorTracker:
    """Tracks tensors through pipeline to ensure zero-copy operations"""
    
    def __init__(self):
        self.active_tensors: Dict[int, TensorFlowOptimization] = {}
        self.copy_violations: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("ZeroCopyTensorTracker")
        self._lock = threading.Lock()
    
    def register_tensor(self, tensor: torch.Tensor, stage: str) -> int:
        """Register a tensor entering the pipeline"""
        tensor_id = id(tensor)
        
        with self._lock:
            if tensor_id not in self.active_tensors:
                self.active_tensors[tensor_id] = TensorFlowOptimization(
                    tensor_id=tensor_id,
                    device=str(tensor.device),
                    shape=tensor.shape,
                    dtype=tensor.dtype,
                    last_stage=stage,
                    creation_time=time.time()
                )
            else:
                # Tensor already exists - check if it's been copied
                existing = self.active_tensors[tensor_id]
                if existing.device != str(tensor.device):
                    self.copy_violations.append({
                        'tensor_id': tensor_id,
                        'from_device': existing.device,
                        'to_device': str(tensor.device),
                        'stage': stage,
                        'timestamp': time.time()
                    })
                    if self.logger.isEnabledFor(logging.WARNING):
                        self.logger.warning(
                            f"Tensor copy detected! {existing.device} -> {tensor.device} in {stage}"
                        )
                existing.last_stage = stage
        
        return tensor_id
    
    def validate_zero_copy(self, tensor: torch.Tensor, stage: str) -> bool:
        """Validate tensor hasn't been copied"""
        tensor_id = id(tensor)
        
        with self._lock:
            if tensor_id in self.active_tensors:
                existing = self.active_tensors[tensor_id]
                return existing.device == str(tensor.device)
        
        return True
    
    def release_tensor(self, tensor_id: int):
        """Release tensor from tracking"""
        with self._lock:
            self.active_tensors.pop(tensor_id, None)
    
    def get_copy_violations(self) -> List[Dict[str, Any]]:
        """Get list of copy violations"""
        with self._lock:
            return self.copy_violations.copy()


class PipelineScheduler:
    """Efficient pipeline scheduling to maximize GPU utilization"""
    
    def __init__(self, num_stages: int = 4):
        self.num_stages = num_stages
        self.stage_queues: List[deque] = [deque() for _ in range(num_stages)]
        self.stage_ready: List[threading.Event] = [threading.Event() for _ in range(num_stages)]
        self.stage_metrics: Dict[str, PipelineStageMetrics] = {}
        self.logger = logging.getLogger("PipelineScheduler")
        self._lock = threading.Lock()
        
        # Scheduling parameters
        self.batch_timeout_ms = 10.0  # Max wait time for batching
        self.min_batch_size = 1
        self.max_batch_size = 4
        
    def schedule_work(self, stage_idx: int, work_item: Any) -> None:
        """Schedule work for a pipeline stage"""
        with self._lock:
            self.stage_queues[stage_idx].append(work_item)
            self.stage_ready[stage_idx].set()
    
    def get_batch(self, stage_idx: int, timeout: Optional[float] = None) -> List[Any]:
        """Get batch of work items for a stage"""
        if timeout is None:
            timeout = self.batch_timeout_ms / 1000.0
        
        # Wait for work to be available
        if not self.stage_ready[stage_idx].wait(timeout):
            return []
        
        with self._lock:
            batch = []
            queue = self.stage_queues[stage_idx]
            
            # Collect up to max_batch_size items
            while queue and len(batch) < self.max_batch_size:
                batch.append(queue.popleft())
            
            # Clear ready flag if queue is empty
            if not queue:
                self.stage_ready[stage_idx].clear()
            
            return batch
    
    def update_stage_metrics(self, stage_name: str, execution_time: float, 
                           tensor_copies: int = 0, gpu_memory: int = 0):
        """Update metrics for a pipeline stage"""
        with self._lock:
            if stage_name not in self.stage_metrics:
                self.stage_metrics[stage_name] = PipelineStageMetrics(
                    name=stage_name,
                    execution_times=deque(maxlen=100)
                )
            
            metrics = self.stage_metrics[stage_name]
            metrics.execution_times.append(execution_time)
            metrics.tensor_copies += tensor_copies
            metrics.gpu_memory_used = gpu_memory
            metrics.last_update = time.time()


class GPUResourceMonitor:
    """Monitors GPU utilization and resource usage"""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.logger = logging.getLogger("GPUResourceMonitor")
        self.metrics_history = deque(maxlen=1000)  # Keep last 1000 samples
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        self._lock = threading.Lock()
        
        # Initialize NVML handle if available
        self.nvml_handle = None
        if NVML_AVAILABLE:
            try:
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            except Exception as e:
                self.logger.warning(f"Failed to get NVML handle: {e}")
    
    def start_monitoring(self, interval: float = 1.0):
        """Start background GPU monitoring"""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._stop_monitoring.clear()
            self._monitoring_thread = threading.Thread(
                target=self._monitor_loop,
                args=(interval,),
                daemon=True
            )
            self._monitoring_thread.start()
            self.logger.info("Started GPU resource monitoring")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=2.0)
    
    def _monitor_loop(self, interval: float):
        """Background monitoring loop"""
        while not self._stop_monitoring.is_set():
            try:
                metrics = self.get_current_metrics()
                with self._lock:
                    self.metrics_history.append(metrics)
                
                # Only log if DEBUG level
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        f"GPU Metrics - Util: {metrics['gpu_utilization']:.1f}%, "
                        f"Mem: {metrics['memory_used_mb']:.0f}MB/{metrics['memory_total_mb']:.0f}MB, "
                        f"Temp: {metrics['temperature']}°C"
                    )
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
            
            time.sleep(interval)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current GPU metrics"""
        metrics = {
            'timestamp': time.time(),
            'gpu_utilization': 0.0,
            'memory_used_mb': 0.0,
            'memory_total_mb': 0.0,
            'memory_utilization': 0.0,
            'temperature': 0,
            'power_draw': 0.0
        }
        
        # PyTorch memory stats
        if torch.cuda.is_available():
            metrics['memory_used_mb'] = torch.cuda.memory_allocated(self.device_id) / 1e6
            metrics['memory_total_mb'] = torch.cuda.get_device_properties(self.device_id).total_memory / 1e6
            metrics['memory_utilization'] = (metrics['memory_used_mb'] / metrics['memory_total_mb']) * 100
        
        # NVML stats if available
        if NVML_AVAILABLE and self.nvml_handle:
            try:
                # GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                metrics['gpu_utilization'] = util.gpu
                
                # Temperature
                metrics['temperature'] = pynvml.nvmlDeviceGetTemperature(
                    self.nvml_handle, pynvml.NVML_TEMPERATURE_GPU
                )
                
                # Power draw
                metrics['power_draw'] = pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle) / 1000.0
                
            except Exception as e:
                self.logger.debug(f"NVML query failed: {e}")
        
        return metrics
    
    def get_average_metrics(self, window_seconds: float = 60.0) -> Dict[str, float]:
        """Get average metrics over time window"""
        with self._lock:
            if not self.metrics_history:
                return {}
            
            current_time = time.time()
            cutoff_time = current_time - window_seconds
            
            # Filter metrics within window
            recent_metrics = [
                m for m in self.metrics_history 
                if m['timestamp'] >= cutoff_time
            ]
            
            if not recent_metrics:
                return {}
            
            # Calculate averages
            avg_metrics = {
                'gpu_utilization': np.mean([m['gpu_utilization'] for m in recent_metrics]),
                'memory_used_mb': np.mean([m['memory_used_mb'] for m in recent_metrics]),
                'memory_utilization': np.mean([m['memory_utilization'] for m in recent_metrics]),
                'temperature': np.mean([m['temperature'] for m in recent_metrics]),
                'power_draw': np.mean([m['power_draw'] for m in recent_metrics]),
                'sample_count': len(recent_metrics)
            }
            
            return avg_metrics
    
    def generate_report(self) -> str:
        """Generate resource usage report (DEBUG LEVEL ONLY)"""
        if not self.logger.isEnabledFor(logging.DEBUG):
            return ""
        
        current = self.get_current_metrics()
        avg_1min = self.get_average_metrics(60.0)
        avg_5min = self.get_average_metrics(300.0)
        
        report = [
            "\n========== GPU Resource Report ==========",
            f"Device: cuda:{self.device_id}",
            f"\nCurrent:",
            f"  GPU Utilization: {current['gpu_utilization']:.1f}%",
            f"  Memory: {current['memory_used_mb']:.0f}/{current['memory_total_mb']:.0f} MB ({current['memory_utilization']:.1f}%)",
            f"  Temperature: {current['temperature']}°C",
            f"  Power Draw: {current['power_draw']:.1f}W"
        ]
        
        if avg_1min:
            report.extend([
                f"\n1-Minute Average ({avg_1min['sample_count']} samples):",
                f"  GPU Utilization: {avg_1min['gpu_utilization']:.1f}%",
                f"  Memory: {avg_1min['memory_used_mb']:.0f} MB ({avg_1min['memory_utilization']:.1f}%)",
                f"  Temperature: {avg_1min['temperature']:.1f}°C",
                f"  Power Draw: {avg_1min['power_draw']:.1f}W"
            ])
        
        if avg_5min:
            report.extend([
                f"\n5-Minute Average ({avg_5min['sample_count']} samples):",
                f"  GPU Utilization: {avg_5min['gpu_utilization']:.1f}%",
                f"  Memory: {avg_5min['memory_used_mb']:.0f} MB ({avg_5min['memory_utilization']:.1f}%)",
                f"  Temperature: {avg_5min['temperature']:.1f}°C", 
                f"  Power Draw: {avg_5min['power_draw']:.1f}W"
            ])
        
        report.append("=========================================\n")
        
        return "\n".join(report)


class PipelinePerformanceOptimizer:
    """Main optimizer for UnifiedGPUPipeline performance"""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.logger = logging.getLogger("PipelinePerformanceOptimizer")
        
        # Initialize components
        self.tensor_tracker = ZeroCopyTensorTracker()
        self.scheduler = PipelineScheduler()
        self.gpu_monitor = GPUResourceMonitor(device_id)
        
        # Performance metrics
        self.optimization_metrics = {
            'tensor_copies_prevented': 0,
            'batches_optimized': 0,
            'memory_saved_mb': 0.0,
            'optimization_start_time': time.time()
        }
        
        # Start GPU monitoring
        self.gpu_monitor.start_monitoring(interval=1.0)
    
    def optimize_tensor_flow(self, tensor: torch.Tensor, stage: str) -> torch.Tensor:
        """Optimize tensor flow through pipeline"""
        # Register and validate tensor
        tensor_id = self.tensor_tracker.register_tensor(tensor, stage)
        
        # Ensure tensor stays on GPU
        if not tensor.is_cuda:
            self.logger.error(f"CPU tensor detected in {stage}! This violates GPU-only policy.")
            raise RuntimeError(f"GPU-only violation: CPU tensor in {stage}")
        
        # Track zero-copy success
        if self.tensor_tracker.validate_zero_copy(tensor, stage):
            self.optimization_metrics['tensor_copies_prevented'] += 1
        
        return tensor
    
    def schedule_stage_work(self, stage_idx: int, work_item: Any):
        """Schedule work for optimal GPU utilization"""
        self.scheduler.schedule_work(stage_idx, work_item)
    
    def get_stage_batch(self, stage_idx: int) -> List[Any]:
        """Get optimized batch for stage processing"""
        batch = self.scheduler.get_batch(stage_idx)
        if len(batch) > 1:
            self.optimization_metrics['batches_optimized'] += 1
        return batch
    
    def profile_stage(self, stage_name: str):
        """Context manager for profiling pipeline stages (DEBUG LEVEL ONLY)"""
        class StageProfiler:
            def __init__(self, optimizer, stage_name):
                self.optimizer = optimizer
                self.stage_name = stage_name
                self.start_time = None
                self.start_memory = None
            
            def __enter__(self):
                if self.optimizer.logger.isEnabledFor(logging.DEBUG):
                    self.start_time = time.time()
                    self.start_memory = torch.cuda.memory_allocated()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.start_time is not None:
                    execution_time = time.time() - self.start_time
                    memory_used = torch.cuda.memory_allocated() - self.start_memory
                    
                    self.optimizer.scheduler.update_stage_metrics(
                        self.stage_name,
                        execution_time,
                        gpu_memory=memory_used
                    )
                    
                    if self.optimizer.logger.isEnabledFor(logging.DEBUG):
                        self.optimizer.logger.debug(
                            f"Stage '{self.stage_name}' - Time: {execution_time*1000:.2f}ms, "
                            f"Memory: {memory_used/1e6:.2f}MB"
                        )
        
        return StageProfiler(self, stage_name)
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report (DEBUG LEVEL ONLY)"""
        if not self.logger.isEnabledFor(logging.DEBUG):
            return {}
        
        runtime = time.time() - self.optimization_metrics['optimization_start_time']
        
        report = {
            'runtime_seconds': runtime,
            'optimization_metrics': self.optimization_metrics.copy(),
            'tensor_flow': {
                'active_tensors': len(self.tensor_tracker.active_tensors),
                'copy_violations': len(self.tensor_tracker.copy_violations),
                'violation_details': self.tensor_tracker.get_copy_violations()
            },
            'pipeline_scheduling': {
                'stage_metrics': {}
            },
            'gpu_resources': {
                'current': self.gpu_monitor.get_current_metrics(),
                'average_1min': self.gpu_monitor.get_average_metrics(60.0),
                'average_5min': self.gpu_monitor.get_average_metrics(300.0)
            }
        }
        
        # Add stage metrics
        for stage_name, metrics in self.scheduler.stage_metrics.items():
            if metrics.execution_times:
                report['pipeline_scheduling']['stage_metrics'][stage_name] = {
                    'avg_time_ms': np.mean(metrics.execution_times) * 1000,
                    'min_time_ms': np.min(metrics.execution_times) * 1000,
                    'max_time_ms': np.max(metrics.execution_times) * 1000,
                    'tensor_copies': metrics.tensor_copies,
                    'gpu_memory_mb': metrics.gpu_memory_used / 1e6
                }
        
        return report
    
    def cleanup(self):
        """Clean up optimizer resources"""
        self.gpu_monitor.stop_monitoring()
        self.logger.info("Pipeline optimizer cleaned up")


# Singleton instance management
_optimizer_instance: Optional[PipelinePerformanceOptimizer] = None
_optimizer_lock = threading.Lock()


def get_pipeline_optimizer(device_id: int = 0) -> PipelinePerformanceOptimizer:
    """Get or create singleton pipeline optimizer"""
    global _optimizer_instance
    
    with _optimizer_lock:
        if _optimizer_instance is None:
            _optimizer_instance = PipelinePerformanceOptimizer(device_id)
        return _optimizer_instance


def cleanup_pipeline_optimizer():
    """Clean up singleton optimizer"""
    global _optimizer_instance
    
    with _optimizer_lock:
        if _optimizer_instance is not None:
            _optimizer_instance.cleanup()
            _optimizer_instance = None 