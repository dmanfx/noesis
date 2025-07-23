#!/usr/bin/env python3
"""
TensorRT Engine Validator

This module provides validation and warm-up procedures for TensorRT engines
to ensure proper loading and FP16 precision consistency.

Key features:
- Engine loading validation
- Warm-up procedures for optimal performance
- FP16 precision validation
- Performance benchmarking
"""

import torch
import tensorrt as trt
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional
from config import AppConfig
from tensorrt_inference import TensorRTInferenceEngine, TensorRTModelManager
from gpu_memory_pool import get_global_memory_pool


class TensorRTEngineValidator:
    """Validates TensorRT engines and performs warm-up procedures"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.device = torch.device(config.models.DEVICE)
        self.logger = logging.getLogger("TensorRTValidator")
        
        # Ensure CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for TensorRT validation")
        
        torch.cuda.set_device(self.device)
        
        self.logger.info(f"TensorRT Engine Validator initialized on {self.device}")
    
    def validate_all_engines(self) -> Dict[str, bool]:
        """
        Validate all TensorRT engines specified in config.
        
        Returns:
            Dict mapping engine names to validation status
        """
        engines = {
            "detection": self.config.models.DETECTION_ENGINE_PATH,
            "pose": self.config.models.POSE_ENGINE_PATH,
            "segmentation": self.config.models.SEGMENTATION_ENGINE_PATH,
            "reid": self.config.models.REID_ENGINE_PATH
        }
        
        results = {}
        
        for engine_name, engine_path in engines.items():
            self.logger.info(f"Validating {engine_name} engine...")
            
            try:
                # Validate engine loading
                validation_result = self.validate_engine(engine_path)
                results[engine_name] = validation_result
                
                if validation_result:
                    self.logger.info(f"✅ {engine_name} engine validation passed")
                else:
                    self.logger.error(f"❌ {engine_name} engine validation failed")
                    
            except Exception as e:
                self.logger.error(f"❌ {engine_name} engine validation error: {e}")
                results[engine_name] = False
        
        return results
    
    def validate_engine(self, engine_path: str) -> bool:
        """
        Validate a single TensorRT engine.
        
        Args:
            engine_path: Path to the engine file
            
        Returns:
            True if validation passes
        """
        try:
            # Test engine loading
            engine = TensorRTInferenceEngine(engine_path, str(self.device))
            
            # Create test input tensor
            input_shape = engine.inputs[0]['shape']
            input_dtype = torch.float16 if engine.inputs[0]['dtype'] == np.float16 else torch.float32
            
            self.logger.debug(f"Engine input shape: {input_shape}, dtype: {input_dtype}")
            
            # Create test input
            test_input = torch.randn(input_shape, dtype=input_dtype, device=self.device)
            
            # Test inference
            output = engine.infer(test_input)
            
            # Validate output
            if output is None:
                self.logger.error("Engine returned None output")
                return False
            
            if not output.is_cuda:
                self.logger.error(f"Engine output not on GPU: {output.device}")
                return False
            
            # Validate FP16 precision if enabled
            if self.config.models.TENSORRT_FP16:
                if output.dtype != torch.float16:
                    self.logger.error(f"Expected FP16 output, got {output.dtype}")
                    return False
                self.logger.debug("✅ FP16 precision validation passed")
            
            self.logger.debug(f"Engine output shape: {output.shape}, dtype: {output.dtype}")
            
            # Perform warm-up
            self.warm_up_engine(engine)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Engine validation failed: {e}")
            return False
    
    def warm_up_engine(self, engine: TensorRTInferenceEngine, iterations: int = None) -> Dict[str, float]:
        """
        Warm up TensorRT engine for optimal performance.
        
        Args:
            engine: TensorRT inference engine
            iterations: Number of warm-up iterations (default from config)
            
        Returns:
            Dict with warm-up performance metrics
        """
        if iterations is None:
            iterations = self.config.models.WARM_UP_ITERATIONS
        
        self.logger.info(f"Warming up engine with {iterations} iterations...")
        
        # Get input shape and dtype
        input_shape = engine.inputs[0]['shape']
        input_dtype = torch.float16 if engine.inputs[0]['dtype'] == np.float16 else torch.float32
        
        # Performance tracking
        times = []
        
        # Warm-up iterations
        for i in range(iterations):
            # Create random input
            test_input = torch.randn(input_shape, dtype=input_dtype, device=self.device)
            
            # Time the inference
            start_time = time.perf_counter()
            
            with torch.cuda.device(self.device):
                output = engine.infer(test_input)
                torch.cuda.synchronize()  # Ensure completion
            
            end_time = time.perf_counter()
            iteration_time = (end_time - start_time) * 1000  # Convert to ms
            times.append(iteration_time)
            
            if i == 0:
                self.logger.debug(f"First inference: {iteration_time:.2f}ms")
            elif i == iterations - 1:
                self.logger.debug(f"Last inference: {iteration_time:.2f}ms")
        
        # Calculate statistics
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)
        
        # Performance after warm-up (exclude first few iterations)
        stable_times = times[3:] if len(times) > 5 else times
        stable_avg = np.mean(stable_times) if stable_times else avg_time
        
        performance_metrics = {
            'iterations': iterations,
            'avg_time_ms': avg_time,
            'min_time_ms': min_time,
            'max_time_ms': max_time,
            'std_time_ms': std_time,
            'stable_avg_ms': stable_avg,
            'first_vs_stable_improvement': times[0] - stable_avg if len(times) > 3 else 0
        }
        
        self.logger.info(f"Warm-up complete: {stable_avg:.2f}ms avg (stable), "
                        f"{times[0] - stable_avg:.2f}ms improvement")
        
        return performance_metrics
    
    def validate_fp16_pipeline(self) -> bool:
        """
        Validate FP16 precision consistency throughout the pipeline.
        
        Returns:
            True if FP16 validation passes
        """
        if not self.config.models.TENSORRT_FP16:
            self.logger.info("FP16 not enabled, skipping FP16 validation")
            return True
        
        self.logger.info("Validating FP16 pipeline consistency...")
        
        try:
            # Create test model manager
            model_manager = TensorRTModelManager(self.config)
            
            # Test each available engine with FP16
            test_cases = [
                ("detection", (1, 3, 640, 640)),
                ("pose", (1, 3, 640, 640)),
                ("segmentation", (1, 3, 640, 640)),
                ("reid", (1, 3, 256, 128))
            ]
            
            for engine_name, input_shape in test_cases:
                if engine_name in model_manager.engines:
                    self.logger.debug(f"Testing FP16 consistency for {engine_name}...")
                    
                    # Create FP16 input
                    test_input = torch.randn(input_shape, dtype=torch.float16, device=self.device)
                    
                    # Test inference
                    if engine_name == "detection":
                        output = model_manager.detect_objects(test_input)
                    elif engine_name == "pose":
                        output = model_manager.estimate_pose(test_input)
                    elif engine_name == "segmentation":
                        output = model_manager.segment_objects(test_input)
                    elif engine_name == "reid":
                        output = model_manager.extract_features(test_input)
                    
                    # Validate output is FP16
                    if output.dtype != torch.float16:
                        self.logger.error(f"FP16 validation failed for {engine_name}: "
                                        f"expected float16, got {output.dtype}")
                        return False
                    
                    # Validate output is on GPU
                    if not output.is_cuda:
                        self.logger.error(f"FP16 validation failed for {engine_name}: "
                                        f"output not on GPU: {output.device}")
                        return False
                    
                    self.logger.debug(f"✅ {engine_name} FP16 validation passed")
            
            self.logger.info("✅ FP16 pipeline validation completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"FP16 pipeline validation failed: {e}")
            return False
    
    def benchmark_engines(self, iterations: int = 100) -> Dict[str, Dict[str, float]]:
        """
        Benchmark all TensorRT engines for performance.
        
        Args:
            iterations: Number of benchmark iterations
            
        Returns:
            Dict mapping engine names to performance metrics
        """
        self.logger.info(f"Benchmarking engines with {iterations} iterations...")
        
        results = {}
        
        try:
            model_manager = TensorRTModelManager(self.config)
            
            # Standard input shapes for each engine type
            benchmark_configs = {
                "detection": (1, 3, 640, 640),
                "pose": (1, 3, 640, 640),
                "segmentation": (1, 3, 640, 640),
                "reid": (1, 3, 256, 128)
            }
            
            for engine_name, input_shape in benchmark_configs.items():
                if engine_name in model_manager.engines:
                    self.logger.info(f"Benchmarking {engine_name} engine...")
                    
                    # Create test input
                    test_input = torch.randn(input_shape, dtype=torch.float16, device=self.device)
                    
                    # Warm up first
                    for _ in range(10):
                        if engine_name == "detection":
                            _ = model_manager.detect_objects(test_input)
                        elif engine_name == "pose":
                            _ = model_manager.estimate_pose(test_input)
                        elif engine_name == "segmentation":
                            _ = model_manager.segment_objects(test_input)
                        elif engine_name == "reid":
                            _ = model_manager.extract_features(test_input)
                    
                    torch.cuda.synchronize()
                    
                    # Benchmark
                    times = []
                    for _ in range(iterations):
                        start_time = time.perf_counter()
                        
                        if engine_name == "detection":
                            _ = model_manager.detect_objects(test_input)
                        elif engine_name == "pose":
                            _ = model_manager.estimate_pose(test_input)
                        elif engine_name == "segmentation":
                            _ = model_manager.segment_objects(test_input)
                        elif engine_name == "reid":
                            _ = model_manager.extract_features(test_input)
                        
                        torch.cuda.synchronize()
                        end_time = time.perf_counter()
                        
                        times.append((end_time - start_time) * 1000)  # Convert to ms
                    
                    # Calculate metrics
                    results[engine_name] = {
                        'avg_time_ms': np.mean(times),
                        'min_time_ms': np.min(times),
                        'max_time_ms': np.max(times),
                        'std_time_ms': np.std(times),
                        'fps': 1000 / np.mean(times),
                        'p95_time_ms': np.percentile(times, 95),
                        'p99_time_ms': np.percentile(times, 99)
                    }
                    
                    self.logger.info(f"{engine_name}: {results[engine_name]['avg_time_ms']:.2f}ms avg, "
                                   f"{results[engine_name]['fps']:.1f} FPS")
        
        except Exception as e:
            self.logger.error(f"Benchmarking failed: {e}")
        
        return results


def validate_tensorrt_setup(config: AppConfig) -> bool:
    """
    Complete TensorRT setup validation.
    
    Args:
        config: Application configuration
        
    Returns:
        True if all validations pass
    """
    logger = logging.getLogger("TensorRTSetupValidator")
    
    try:
        validator = TensorRTEngineValidator(config)
        
        # Validate all engines
        engine_results = validator.validate_all_engines()
        
        # Check if all required engines passed
        required_engines = ["detection"]  # Minimum required
        if config.visualization.SHOW_KEYPOINTS:
            required_engines.append("pose")
        if config.models.ENABLE_SEGMENTATION:
            required_engines.append("segmentation")
        
        all_required_passed = all(
            engine_results.get(engine, False) for engine in required_engines
        )
        
        if not all_required_passed:
            logger.error("❌ Required TensorRT engines failed validation")
            return False
        
        # Validate FP16 pipeline
        if not validator.validate_fp16_pipeline():
            logger.error("❌ FP16 pipeline validation failed")
            return False
        
        # Optional: Run benchmarks
        if config.processing.ENABLE_PROFILING:
            benchmarks = validator.benchmark_engines(iterations=50)
            logger.info(f"Benchmark results: {benchmarks}")
        
        logger.info("✅ TensorRT setup validation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ TensorRT setup validation failed: {e}")
        return False


if __name__ == "__main__":
    from config import config
    
    logging.basicConfig(level=logging.INFO)
    
    # Run validation
    success = validate_tensorrt_setup(config)
    
    if success:
        print("✅ All TensorRT validations passed!")
    else:
        print("❌ TensorRT validation failed!")
        exit(1) 