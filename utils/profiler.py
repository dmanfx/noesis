"""
Profiling Utilities Module

This module provides lightweight profiling utilities for the video processing pipeline.
It includes context managers for timing individual processing steps and functions
for aggregating performance statistics.

The profiling system is designed to be:
- Lightweight and opt-in (controlled by config.processing.ENABLE_PROFILING)
- Thread-safe and process-safe
- Non-intrusive (failures don't affect main processing)
"""

import time
import logging
from contextlib import contextmanager
from typing import Dict, Any, Optional
from collections import deque
import sys
import os

# Import PerformanceMonitor from the root utils.py file
# We need to import it directly since we're in a utils subdirectory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
utils_file = os.path.join(parent_dir, 'utils.py')

# Import the module spec and load it
import importlib.util
spec = importlib.util.spec_from_file_location("root_utils", utils_file)
root_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(root_utils)

PerformanceMonitor = root_utils.PerformanceMonitor


@contextmanager
def profile_step(name: str, monitor: PerformanceMonitor):
    """Context manager for profiling a processing step
    
    Times the execution of a code block and records the elapsed time
    in the provided PerformanceMonitor instance.
    
    Args:
        name: Name of the processing step being profiled
        monitor: PerformanceMonitor instance to record timing data
        
    Yields:
        None
        
    Example:
        with profile_step("detection", self.perf_monitor):
            detections = self.detector.process_frame(frame)
    """
    logger = logging.getLogger("profiler")
    
    try:
        monitor.start_timer(name)
        yield
    except Exception as e:
        # Log the error but don't let profiling failures break the pipeline
        logger.warning(f"Profiling step '{name}' failed: {e}")
        raise  # Re-raise the original exception
    finally:
        try:
            elapsed_ms = monitor.stop_timer(name)
            logger.debug(f"Step '{name}' took {elapsed_ms:.2f}ms")
        except Exception as e:
            logger.warning(f"Failed to stop timer for step '{name}': {e}")


def aggregate_stats(monitor: PerformanceMonitor) -> Dict[str, Any]:
    """Aggregate performance statistics from a PerformanceMonitor
    
    Extracts timing data from the monitor and returns a dictionary
    with average and last execution times for each profiled step.
    
    Args:
        monitor: PerformanceMonitor instance containing timing data
        
    Returns:
        Dictionary with profiling statistics in the format:
        {
            "step_name": {
                "avg_ms": float,  # Average execution time in milliseconds
                "last_ms": float, # Last execution time in milliseconds
                "count": int      # Number of samples
            }
        }
    """
    stats = {}
    
    try:
        # Get all timing data from the monitor
        with monitor.lock:
            for step_name, times_list in monitor.averages.items():
                if times_list:
                    avg_time = sum(times_list) / len(times_list)
                    last_time = times_list[-1]
                    count = len(times_list)
                    
                    stats[step_name] = {
                        "avg_ms": round(avg_time, 2),
                        "last_ms": round(last_time, 2), 
                        "count": count
                    }
                    
    except Exception as e:
        logger = logging.getLogger("profiler")
        logger.warning(f"Failed to aggregate profiling stats: {e}")
        
    return stats 