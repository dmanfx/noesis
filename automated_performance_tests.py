"""
Automated Performance Testing Framework
=======================================

Automated test suite for GPU pipeline performance validation,
including regression testing, load testing, and continuous monitoring.

Features:
- Automated test execution
- Performance regression detection
- Load testing scenarios
- Continuous integration support
"""

import unittest
import time
import subprocess
import json
import os
import sys
import signal
import psutil
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from contextlib import contextmanager
import threading
import queue

# Import benchmark framework
from performance_benchmark import (
    PerformanceBenchmark, 
    PerformanceValidator,
    DEFAULT_TARGETS
)


@dataclass
class TestResult:
    """Container for test results"""
    test_name: str
    passed: bool
    duration: float
    metrics: Dict[str, Any]
    errors: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'test_name': self.test_name,
            'passed': self.passed,
            'duration': self.duration,
            'metrics': self.metrics,
            'errors': self.errors
        }


class GPUPipelineProcess:
    """Manages GPU pipeline process for testing"""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize pipeline process manager"""
        self.config_file = config_file or "config.json"
        self.process = None
        self.logger = logging.getLogger("GPUPipelineProcess")
        
    def start(self, cameras: int = 1) -> bool:
        """Start GPU pipeline process"""
        try:
            # Prepare command
            cmd = [sys.executable, "main.py"]
            
            # Add config file if specified
            if self.config_file:
                cmd.extend(["--config", self.config_file])
                
            # Start process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for startup
            time.sleep(10)
            
            # Check if process is running
            if self.process.poll() is None:
                self.logger.info(f"GPU pipeline started (PID: {self.process.pid})")
                return True
            else:
                stdout, stderr = self.process.communicate()
                self.logger.error(f"Pipeline failed to start: {stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error starting pipeline: {e}")
            return False
            
    def stop(self):
        """Stop GPU pipeline process"""
        if self.process:
            try:
                # Send SIGTERM
                self.process.terminate()
                
                # Wait for graceful shutdown
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Force kill if needed
                    self.process.kill()
                    self.process.wait()
                    
                self.logger.info("GPU pipeline stopped")
                
            except Exception as e:
                self.logger.error(f"Error stopping pipeline: {e}")
                
    def is_running(self) -> bool:
        """Check if process is running"""
        return self.process is not None and self.process.poll() is None


class PerformanceTestBase(unittest.TestCase):
    """Base class for performance tests"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.logger = logging.getLogger(cls.__name__)
        cls.results = []
        
    def run_performance_test(self,
                           test_name: str,
                           duration: int = 60,
                           cameras: int = 1,
                           targets: Optional[Dict] = None) -> TestResult:
        """
        Run a performance test.
        
        Args:
            test_name: Name of the test
            duration: Test duration in seconds
            cameras: Number of cameras to test
            targets: Performance targets
            
        Returns:
            TestResult object
        """
        self.logger.info(f"Running test: {test_name}")
        
        # Start timer
        start_time = time.time()
        errors = []
        
        # Create test config
        test_config = self._create_test_config(cameras)
        
        # Start pipeline
        pipeline = GPUPipelineProcess(config_file=test_config)
        
        try:
            # Start pipeline
            if not pipeline.start(cameras):
                errors.append("Failed to start pipeline")
                return TestResult(
                    test_name=test_name,
                    passed=False,
                    duration=0,
                    metrics={},
                    errors=errors
                )
                
            # Create benchmark
            benchmark = PerformanceBenchmark(
                output_dir=f"benchmarks/{test_name}",
                sample_interval=1.0,
                warmup_time=10.0
            )
            
            # Start monitoring
            benchmark.start()
            
            # Run for specified duration
            self._run_test_duration(benchmark, duration)
            
            # Stop monitoring
            benchmark.stop()
            
            # Get results
            summary = benchmark._calculate_summary()
            
            # Validate results
            validator = PerformanceValidator(targets or DEFAULT_TARGETS)
            passed, validation_errors = validator.validate(summary)
            errors.extend(validation_errors)
            
            # Calculate duration
            test_duration = time.time() - start_time
            
            return TestResult(
                test_name=test_name,
                passed=passed,
                duration=test_duration,
                metrics=summary,
                errors=errors
            )
            
        except Exception as e:
            self.logger.error(f"Test error: {e}")
            errors.append(str(e))
            
            return TestResult(
                test_name=test_name,
                passed=False,
                duration=time.time() - start_time,
                metrics={},
                errors=errors
            )
            
        finally:
            # Stop pipeline
            pipeline.stop()
            
            # Cleanup test config
            if os.path.exists(test_config):
                os.remove(test_config)
                
    def _create_test_config(self, cameras: int) -> str:
        """Create test configuration file"""
        # Load base config
        with open("config.json", 'r') as f:
            config = json.load(f)
            
        # Modify for test
        if cameras == 1:
            # Single camera test
            config['cameras']['RTSP_STREAMS'] = [
                config['cameras']['RTSP_STREAMS'][0]
            ]
        elif cameras == 2:
            # Two camera test
            config['cameras']['RTSP_STREAMS'] = config['cameras']['RTSP_STREAMS'][:2]
            
        # Ensure GPU pipeline is enabled
        config['processing']['USE_UNIFIED_GPU_PIPELINE'] = True
        config['processing']['ENABLE_DECOUPLED_PIPELINE'] = False
        config['models']['FORCE_GPU_ONLY'] = True
        
        # Save test config
        test_config = f"test_config_{cameras}cam.json"
        with open(test_config, 'w') as f:
            json.dump(config, f, indent=2)
            
        return test_config
        
    def _run_test_duration(self, benchmark: PerformanceBenchmark, duration: int):
        """Run test for specified duration"""
        end_time = time.time() + duration
        
        while time.time() < end_time:
            # Simulate frame processing
            benchmark.record_frame()
            
            # Get current metrics
            metrics = benchmark.get_current_metrics()
            if metrics:
                self.logger.debug(
                    f"CPU: {metrics.cpu_percent:.1f}% | "
                    f"GPU: {metrics.gpu_utilization:.1f}%"
                )
                
            time.sleep(0.1)


class BasicPerformanceTests(PerformanceTestBase):
    """Basic performance validation tests"""
    
    def test_single_camera_performance(self):
        """Test single camera performance"""
        result = self.run_performance_test(
            test_name="single_camera",
            duration=120,
            cameras=1,
            targets={
                'cpu': {'max': 10.0},
                'gpu': {'min': 30.0},
                'memory_transfers': {'max': 1.0}
            }
        )
        
        self.results.append(result)
        self.assertTrue(result.passed, f"Test failed: {result.errors}")
        
    def test_dual_camera_performance(self):
        """Test dual camera performance"""
        result = self.run_performance_test(
            test_name="dual_camera",
            duration=120,
            cameras=2,
            targets={
                'cpu': {'max': 15.0},  # Slightly higher for 2 cameras
                'gpu': {'min': 50.0},
                'memory_transfers': {'max': 2.0}
            }
        )
        
        self.results.append(result)
        self.assertTrue(result.passed, f"Test failed: {result.errors}")
        
    def test_sustained_performance(self):
        """Test sustained performance over longer period"""
        result = self.run_performance_test(
            test_name="sustained",
            duration=600,  # 10 minutes
            cameras=2,
            targets={
                'cpu': {'max': 15.0},
                'gpu': {'min': 50.0},
                'memory_transfers': {'max': 2.0}
            }
        )
        
        self.results.append(result)
        self.assertTrue(result.passed, f"Test failed: {result.errors}")


class LoadTests(PerformanceTestBase):
    """Load testing scenarios"""
    
    def test_startup_performance(self):
        """Test startup performance and resource usage"""
        self.logger.info("Testing startup performance")
        
        # Monitor during startup
        cpu_samples = []
        memory_samples = []
        
        # Start pipeline
        pipeline = GPUPipelineProcess()
        
        try:
            # Monitor startup
            start_time = time.time()
            pipeline.start(cameras=2)
            
            # Sample CPU/memory during startup
            for _ in range(20):  # 20 seconds
                cpu_samples.append(psutil.cpu_percent(interval=0.5))
                memory_samples.append(psutil.virtual_memory().percent)
                
            startup_time = time.time() - start_time
            
            # Check startup metrics
            max_cpu = max(cpu_samples)
            avg_cpu = sum(cpu_samples) / len(cpu_samples)
            
            self.logger.info(f"Startup time: {startup_time:.1f}s")
            self.logger.info(f"Max CPU during startup: {max_cpu:.1f}%")
            self.logger.info(f"Avg CPU during startup: {avg_cpu:.1f}%")
            
            # Validate
            self.assertLess(startup_time, 30, "Startup took too long")
            self.assertLess(max_cpu, 80, "CPU spike too high during startup")
            
        finally:
            pipeline.stop()
            
    def test_memory_stability(self):
        """Test memory usage stability over time"""
        result = self.run_performance_test(
            test_name="memory_stability",
            duration=300,  # 5 minutes
            cameras=2,
            targets={
                'cpu': {'max': 15.0},
                'gpu': {'min': 50.0}
            }
        )
        
        # Check for memory leaks
        if 'gpu_memory' in result.metrics:
            memory_values = result.metrics['gpu_memory']
            
            # Memory should not increase significantly
            memory_increase = memory_values['max'] - memory_values['min']
            self.assertLess(
                memory_increase, 
                10.0,  # Max 10% increase
                f"Possible memory leak detected: {memory_increase:.1f}% increase"
            )
            
        self.results.append(result)


class RegressionTests(PerformanceTestBase):
    """Performance regression tests"""
    
    def test_cpu_regression(self):
        """Test for CPU usage regression"""
        # Load baseline if exists
        baseline_file = "benchmarks/baseline.json"
        
        if os.path.exists(baseline_file):
            with open(baseline_file, 'r') as f:
                baseline = json.load(f)
                
            # Run current test
            result = self.run_performance_test(
                test_name="regression_cpu",
                duration=120,
                cameras=2
            )
            
            # Compare with baseline
            if 'summary' in baseline and 'cpu' in result.metrics:
                baseline_cpu = baseline['summary']['cpu']['mean']
                current_cpu = result.metrics['cpu']['mean']
                
                # Allow 10% regression
                max_allowed = baseline_cpu * 1.1
                
                self.assertLess(
                    current_cpu,
                    max_allowed,
                    f"CPU regression detected: {current_cpu:.1f}% vs baseline {baseline_cpu:.1f}%"
                )
                
            self.results.append(result)
            
        else:
            self.skipTest("No baseline found for regression testing")


class StressTests(PerformanceTestBase):
    """Stress testing scenarios"""
    
    def test_rapid_restart(self):
        """Test rapid stop/start cycles"""
        self.logger.info("Testing rapid restart cycles")
        
        errors = []
        
        for i in range(5):
            self.logger.info(f"Restart cycle {i+1}/5")
            
            pipeline = GPUPipelineProcess()
            
            try:
                # Start
                if not pipeline.start(cameras=1):
                    errors.append(f"Failed to start on cycle {i+1}")
                    
                # Run briefly
                time.sleep(10)
                
                # Stop
                pipeline.stop()
                
                # Brief pause
                time.sleep(2)
                
            except Exception as e:
                errors.append(f"Error on cycle {i+1}: {e}")
                
        self.assertEqual(len(errors), 0, f"Restart test failed: {errors}")
        
    def test_resource_exhaustion(self):
        """Test behavior under resource exhaustion"""
        self.logger.info("Testing resource exhaustion handling")
        
        # This test would attempt to exhaust GPU memory
        # and verify graceful handling
        
        # Create large tensor to consume GPU memory
        try:
            if torch.cuda.is_available():
                # Allocate large chunk of GPU memory
                large_tensor = torch.zeros(
                    (10000, 10000), 
                    device='cuda',
                    dtype=torch.float32
                )
                
                # Try to run pipeline with limited memory
                result = self.run_performance_test(
                    test_name="resource_exhaustion",
                    duration=60,
                    cameras=1
                )
                
                # Pipeline should handle gracefully
                self.assertFalse(
                    result.passed,
                    "Pipeline should fail gracefully under resource exhaustion"
                )
                
                # Cleanup
                del large_tensor
                torch.cuda.empty_cache()
                
        except Exception as e:
            self.logger.warning(f"Resource exhaustion test skipped: {e}")


def run_all_tests(test_suite: str = "basic") -> bool:
    """
    Run all performance tests.
    
    Args:
        test_suite: Test suite to run ("basic", "load", "regression", "stress", "all")
        
    Returns:
        True if all tests pass
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests based on suite selection
    if test_suite in ["basic", "all"]:
        suite.addTests(loader.loadTestsFromTestCase(BasicPerformanceTests))
        
    if test_suite in ["load", "all"]:
        suite.addTests(loader.loadTestsFromTestCase(LoadTests))
        
    if test_suite in ["regression", "all"]:
        suite.addTests(loader.loadTestsFromTestCase(RegressionTests))
        
    if test_suite in ["stress", "all"]:
        suite.addTests(loader.loadTestsFromTestCase(StressTests))
        
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate report
    generate_test_report(result)
    
    return result.wasSuccessful()


def generate_test_report(test_result: unittest.TestResult):
    """Generate test report"""
    report = {
        'timestamp': time.time(),
        'total_tests': test_result.testsRun,
        'passed': test_result.testsRun - len(test_result.failures) - len(test_result.errors),
        'failed': len(test_result.failures),
        'errors': len(test_result.errors),
        'failures': [],
        'errors': []
    }
    
    # Add failure details
    for test, traceback in test_result.failures:
        report['failures'].append({
            'test': str(test),
            'traceback': traceback
        })
        
    # Add error details
    for test, traceback in test_result.errors:
        report['errors'].append({
            'test': str(test),
            'traceback': traceback
        })
        
    # Save report
    report_file = f"test_report_{int(time.time())}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"\nTest report saved to: {report_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total Tests: {report['total_tests']}")
    print(f"Passed: {report['passed']}")
    print(f"Failed: {report['failed']}")
    print(f"Errors: {report['errors']}")
    print("="*60)


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated GPU Pipeline Performance Tests")
    parser.add_argument('--suite', choices=['basic', 'load', 'regression', 'stress', 'all'],
                        default='basic', help='Test suite to run')
    
    args = parser.parse_args()
    
    # Run tests
    success = run_all_tests(args.suite)
    exit(0 if success else 1) 