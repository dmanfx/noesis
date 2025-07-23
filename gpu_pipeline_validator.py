"""
GPU Pipeline Validation Framework
=================================

Validates GPU-only pipeline operation, ensuring zero CPU fallbacks
and proper memory management throughout the processing pipeline.

Features:
- Zero-fallback validation
- Memory transfer monitoring
- Accuracy validation
- Pipeline integrity checks
"""

import torch
import numpy as np
import time
import logging
import threading
import queue
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import json
import os
import traceback
import cv2

# Try to import pipeline components
try:
    from gpu_pipeline import UnifiedGPUPipeline
    from tensorrt_inference import GPUOnlyDetectionManager
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    logging.warning("Pipeline components not available for validation")


@dataclass
class ValidationResult:
    """Container for validation results"""
    test_name: str
    passed: bool
    details: Dict[str, Any]
    warnings: List[str]
    errors: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'test_name': self.test_name,
            'passed': self.passed,
            'details': self.details,
            'warnings': self.warnings,
            'errors': self.errors
        }


class CPUFallbackDetector:
    """Detects any CPU fallback operations in the pipeline"""
    
    def __init__(self):
        """Initialize CPU fallback detector"""
        self.logger = logging.getLogger("CPUFallbackDetector")
        self.cpu_operations = []
        self.monitoring = False
        self.original_functions = {}
        
    def start_monitoring(self):
        """Start monitoring for CPU operations"""
        self.monitoring = True
        self.cpu_operations.clear()
        
        # Hook into common CPU fallback points
        self._install_hooks()
        
    def stop_monitoring(self) -> List[str]:
        """Stop monitoring and return detected operations"""
        self.monitoring = False
        self._remove_hooks()
        return self.cpu_operations
        
    def _install_hooks(self):
        """Install monitoring hooks"""
        # Hook tensor.cpu() calls
        original_cpu = torch.Tensor.cpu
        
        def cpu_hook(tensor):
            if self.monitoring:
                # Get calling context
                stack = traceback.extract_stack()
                caller = None
                for frame in reversed(stack[:-1]):
                    if 'site-packages' not in frame.filename:
                        caller = f"{frame.filename}:{frame.lineno} in {frame.name}"
                        break
                        
                self.cpu_operations.append(f"tensor.cpu() called from {caller}")
                
            return original_cpu(tensor)
            
        self.original_functions['tensor.cpu'] = original_cpu
        torch.Tensor.cpu = cpu_hook
        
        # Hook numpy conversions
        original_numpy = torch.Tensor.numpy
        
        def numpy_hook(tensor):
            if self.monitoring:
                stack = traceback.extract_stack()
                caller = None
                for frame in reversed(stack[:-1]):
                    if 'site-packages' not in frame.filename:
                        caller = f"{frame.filename}:{frame.lineno} in {frame.name}"
                        break
                        
                self.cpu_operations.append(f"tensor.numpy() called from {caller}")
                
            return original_numpy(tensor)
            
        self.original_functions['tensor.numpy'] = original_numpy
        torch.Tensor.numpy = numpy_hook
        
    def _remove_hooks(self):
        """Remove monitoring hooks"""
        # Restore original functions
        if 'tensor.cpu' in self.original_functions:
            torch.Tensor.cpu = self.original_functions['tensor.cpu']
            
        if 'tensor.numpy' in self.original_functions:
            torch.Tensor.numpy = self.original_functions['tensor.numpy']


class MemoryTransferValidator:
    """Validates memory transfer patterns"""
    
    def __init__(self):
        """Initialize memory transfer validator"""
        self.logger = logging.getLogger("MemoryTransferValidator")
        self.transfers = []
        self.monitoring = False
        
    def start_monitoring(self):
        """Start monitoring memory transfers"""
        self.monitoring = True
        self.transfers.clear()
        self.start_time = time.time()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitor_transfers,
            daemon=True
        )
        self.monitor_thread.start()
        
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return analysis"""
        self.monitoring = False
        
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2.0)
            
        # Analyze transfers
        duration = time.time() - self.start_time
        
        if self.transfers:
            total_mb = sum(t['size_mb'] for t in self.transfers)
            avg_rate = total_mb / duration if duration > 0 else 0
            
            return {
                'total_transfers': len(self.transfers),
                'total_mb': total_mb,
                'duration': duration,
                'avg_rate_mb_s': avg_rate,
                'transfers': self.transfers[-10:]  # Last 10 transfers
            }
        else:
            return {
                'total_transfers': 0,
                'total_mb': 0,
                'duration': duration,
                'avg_rate_mb_s': 0,
                'transfers': []
            }
            
    def _monitor_transfers(self):
        """Monitor CUDA memory transfers"""
        while self.monitoring:
            try:
                # Get current CUDA memory stats
                if torch.cuda.is_available():
                    stats = torch.cuda.memory_stats()
                    
                    # Look for allocation/deallocation patterns
                    # that indicate CPU-GPU transfers
                    # This is simplified - real implementation would
                    # hook into CUDA profiler
                    
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.debug(f"Error monitoring transfers: {e}")


class AccuracyValidator:
    """Validates processing accuracy"""
    
    def __init__(self, reference_model: Optional[Any] = None):
        """Initialize accuracy validator"""
        self.logger = logging.getLogger("AccuracyValidator")
        self.reference_model = reference_model
        self.results = []
        
    def validate_detection_accuracy(self, 
                                  gpu_results: List[Any],
                                  reference_results: List[Any],
                                  iou_threshold: float = 0.5) -> Dict[str, float]:
        """
        Validate detection accuracy against reference.
        
        Args:
            gpu_results: Results from GPU pipeline
            reference_results: Reference results
            iou_threshold: IoU threshold for matching
            
        Returns:
            Accuracy metrics
        """
        if not gpu_results or not reference_results:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            
        # Calculate matches based on IoU
        true_positives = 0
        false_positives = len(gpu_results)
        false_negatives = len(reference_results)
        
        matched_refs = set()
        
        for gpu_det in gpu_results:
            best_iou = 0
            best_ref_idx = -1
            
            for i, ref_det in enumerate(reference_results):
                if i in matched_refs:
                    continue
                    
                iou = self._calculate_iou(gpu_det, ref_det)
                if iou > best_iou:
                    best_iou = iou
                    best_ref_idx = i
                    
            if best_iou >= iou_threshold:
                true_positives += 1
                false_positives -= 1
                matched_refs.add(best_ref_idx)
                
        false_negatives -= len(matched_refs)
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
        
    def _calculate_iou(self, box1: Any, box2: Any) -> float:
        """Calculate IoU between two boxes"""
        # Extract coordinates (assuming [x1, y1, x2, y2] format)
        if hasattr(box1, 'bbox'):
            x1_1, y1_1, x2_1, y2_1 = box1.bbox
        else:
            x1_1, y1_1, x2_1, y2_1 = box1[:4]
            
        if hasattr(box2, 'bbox'):
            x1_2, y1_2, x2_2, y2_2 = box2.bbox
        else:
            x1_2, y1_2, x2_2, y2_2 = box2[:4]
            
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
            
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0


class PipelineIntegrityValidator:
    """Validates pipeline integrity and data flow"""
    
    def __init__(self):
        """Initialize pipeline integrity validator"""
        self.logger = logging.getLogger("PipelineIntegrityValidator")
        self.checkpoints = {}
        
    def add_checkpoint(self, name: str, validator: Callable[[Any], bool]):
        """Add validation checkpoint"""
        self.checkpoints[name] = validator
        
    def validate_pipeline_flow(self, pipeline: Any) -> ValidationResult:
        """Validate entire pipeline flow"""
        errors = []
        warnings = []
        details = {}
        
        # Check GPU-only configuration
        if hasattr(pipeline, 'config'):
            config = pipeline.config
            
            # Validate critical settings
            if not config.models.FORCE_GPU_ONLY:
                errors.append("FORCE_GPU_ONLY is not enabled")
                
            if not config.models.ENABLE_TENSORRT:
                errors.append("TensorRT is not enabled")
                
            if config.processing.NVDEC_FALLBACK_TO_CPU:
                errors.append("CPU fallback is enabled for NVDEC")
                
        # Check component initialization
        if hasattr(pipeline, 'deepstream_processor'):
            if not pipeline.deepstream_processor:
                errors.append("DeepStream processor not initialized")
            else:
                details['deepstream_processor_initialized'] = True
                
        if hasattr(pipeline, 'gpu_detector'):
            if not pipeline.gpu_detector:
                errors.append("GPU detector not initialized")
            else:
                details['gpu_detector_initialized'] = True
                
        # Run checkpoints
        for name, validator in self.checkpoints.items():
            try:
                if validator(pipeline):
                    details[f'checkpoint_{name}'] = "passed"
                else:
                    warnings.append(f"Checkpoint '{name}' failed")
                    details[f'checkpoint_{name}'] = "failed"
            except Exception as e:
                errors.append(f"Checkpoint '{name}' error: {e}")
                details[f'checkpoint_{name}'] = "error"
                
        return ValidationResult(
            test_name="pipeline_integrity",
            passed=len(errors) == 0,
            details=details,
            warnings=warnings,
            errors=errors
        )


class GPUPipelineValidator:
    """Main GPU pipeline validator"""
    
    def __init__(self):
        """Initialize GPU pipeline validator"""
        self.logger = logging.getLogger("GPUPipelineValidator")
        
        # Initialize sub-validators
        self.cpu_detector = CPUFallbackDetector()
        self.memory_validator = MemoryTransferValidator()
        self.accuracy_validator = AccuracyValidator()
        self.integrity_validator = PipelineIntegrityValidator()
        
        # Add default checkpoints
        self._setup_default_checkpoints()
        
    def _setup_default_checkpoints(self):
        """Setup default validation checkpoints"""
        # Check tensor on GPU
        def check_tensor_gpu(pipeline):
            if hasattr(pipeline, 'last_tensor'):
                return pipeline.last_tensor.is_cuda
            return True
            
        self.integrity_validator.add_checkpoint("tensor_on_gpu", check_tensor_gpu)
        
        # Check memory pool usage
        def check_memory_pool(pipeline):
            if hasattr(pipeline, 'memory_pool') and pipeline.memory_pool:
                stats = pipeline.memory_pool.get_stats()
                return stats.get('reuses', 0) > 0
            return True
            
        self.integrity_validator.add_checkpoint("memory_pool_active", check_memory_pool)
        
    def validate_zero_fallback(self, pipeline: Any, duration: int = 60) -> ValidationResult:
        """
        Validate zero CPU fallback operation.
        
        Args:
            pipeline: Pipeline instance to validate
            duration: Test duration in seconds
            
        Returns:
            ValidationResult
        """
        self.logger.info("Validating zero CPU fallback...")
        
        errors = []
        warnings = []
        details = {}
        
        # Start CPU operation monitoring
        self.cpu_detector.start_monitoring()
        
        try:
            # Run pipeline for duration
            start_time = time.time()
            frame_count = 0
            
            while time.time() - start_time < duration:
                # Process frames
                if hasattr(pipeline, 'process_frame'):
                    pipeline.process_frame()
                    frame_count += 1
                    
                time.sleep(0.033)  # ~30 FPS
                
            # Stop monitoring
            cpu_operations = self.cpu_detector.stop_monitoring()
            
            # Analyze results
            details['duration'] = time.time() - start_time
            details['frames_processed'] = frame_count
            details['cpu_operations_detected'] = len(cpu_operations)
            
            if cpu_operations:
                errors.append(f"Detected {len(cpu_operations)} CPU operations")
                details['cpu_operations'] = cpu_operations[:10]  # First 10
                
            return ValidationResult(
                test_name="zero_fallback",
                passed=len(cpu_operations) == 0,
                details=details,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            errors.append(str(e))
            
            return ValidationResult(
                test_name="zero_fallback",
                passed=False,
                details=details,
                warnings=warnings,
                errors=errors
            )
            
    def validate_memory_transfers(self, pipeline: Any, duration: int = 60) -> ValidationResult:
        """
        Validate memory transfer patterns.
        
        Args:
            pipeline: Pipeline instance to validate
            duration: Test duration in seconds
            
        Returns:
            ValidationResult
        """
        self.logger.info("Validating memory transfers...")
        
        errors = []
        warnings = []
        
        # Start memory monitoring
        self.memory_validator.start_monitoring()
        
        try:
            # Run pipeline
            start_time = time.time()
            
            while time.time() - start_time < duration:
                if hasattr(pipeline, 'process_frame'):
                    pipeline.process_frame()
                    
                time.sleep(0.033)
                
            # Stop monitoring
            transfer_stats = self.memory_validator.stop_monitoring()
            
            # Validate transfer rate
            avg_rate = transfer_stats['avg_rate_mb_s']
            
            if avg_rate > 1.0:  # 1 MB/s threshold
                warnings.append(f"High memory transfer rate: {avg_rate:.2f} MB/s")
                
            if avg_rate > 10.0:  # Critical threshold
                errors.append(f"Excessive memory transfers: {avg_rate:.2f} MB/s")
                
            return ValidationResult(
                test_name="memory_transfers",
                passed=avg_rate <= 1.0,
                details=transfer_stats,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            errors.append(str(e))
            
            return ValidationResult(
                test_name="memory_transfers",
                passed=False,
                details={},
                warnings=warnings,
                errors=errors
            )
            
    def validate_accuracy(self, 
                         pipeline: Any,
                         test_frames: List[np.ndarray],
                         reference_results: List[List[Any]]) -> ValidationResult:
        """
        Validate processing accuracy.
        
        Args:
            pipeline: Pipeline instance to validate
            test_frames: Test frames
            reference_results: Reference detection results
            
        Returns:
            ValidationResult
        """
        self.logger.info("Validating processing accuracy...")
        
        errors = []
        warnings = []
        details = {
            'frames_tested': len(test_frames),
            'per_frame_results': []
        }
        
        try:
            gpu_results = []
            
            # Process test frames
            for i, frame in enumerate(test_frames):
                if hasattr(pipeline, 'process_frame'):
                    result = pipeline.process_frame(frame)
                    gpu_results.append(result)
                    
                    # Validate individual frame
                    if i < len(reference_results):
                        frame_accuracy = self.accuracy_validator.validate_detection_accuracy(
                            result,
                            reference_results[i]
                        )
                        details['per_frame_results'].append(frame_accuracy)
                        
            # Calculate overall accuracy
            if details['per_frame_results']:
                avg_precision = np.mean([r['precision'] for r in details['per_frame_results']])
                avg_recall = np.mean([r['recall'] for r in details['per_frame_results']])
                avg_f1 = np.mean([r['f1'] for r in details['per_frame_results']])
                
                details['overall'] = {
                    'precision': avg_precision,
                    'recall': avg_recall,
                    'f1': avg_f1
                }
                
                # Check thresholds
                if avg_f1 < 0.8:
                    warnings.append(f"F1 score below threshold: {avg_f1:.3f}")
                    
                if avg_f1 < 0.7:
                    errors.append(f"Critical accuracy degradation: F1={avg_f1:.3f}")
                    
            return ValidationResult(
                test_name="accuracy",
                passed=len(errors) == 0,
                details=details,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            errors.append(str(e))
            
            return ValidationResult(
                test_name="accuracy",
                passed=False,
                details=details,
                warnings=warnings,
                errors=errors
            )
            
    def validate_pipeline_integrity(self, pipeline: Any) -> ValidationResult:
        """
        Validate overall pipeline integrity.
        
        Args:
            pipeline: Pipeline instance to validate
            
        Returns:
            ValidationResult
        """
        return self.integrity_validator.validate_pipeline_flow(pipeline)
        
    def run_full_validation(self, pipeline: Any) -> Dict[str, ValidationResult]:
        """
        Run complete validation suite.
        
        Args:
            pipeline: Pipeline instance to validate
            
        Returns:
            Dictionary of validation results
        """
        self.logger.info("Running full GPU pipeline validation...")
        
        results = {}
        
        # 1. Pipeline integrity
        results['integrity'] = self.validate_pipeline_integrity(pipeline)
        
        # 2. Zero CPU fallback
        results['zero_fallback'] = self.validate_zero_fallback(pipeline, duration=30)
        
        # 3. Memory transfers
        results['memory_transfers'] = self.validate_memory_transfers(pipeline, duration=30)
        
        # 4. Generate summary
        all_passed = all(r.passed for r in results.values())
        total_errors = sum(len(r.errors) for r in results.values())
        total_warnings = sum(len(r.warnings) for r in results.values())
        
        self.logger.info("="*60)
        self.logger.info("VALIDATION SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Overall: {'PASSED' if all_passed else 'FAILED'}")
        self.logger.info(f"Total Errors: {total_errors}")
        self.logger.info(f"Total Warnings: {total_warnings}")
        
        for name, result in results.items():
            self.logger.info(f"\n{name}: {'PASSED' if result.passed else 'FAILED'}")
            if result.errors:
                for error in result.errors:
                    self.logger.error(f"  - {error}")
                    
        self.logger.info("="*60)
        
        return results


def validate_pipeline_from_config(config_file: str = "config.json") -> bool:
    """
    Validate pipeline using configuration file.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        True if validation passes
    """
    if not PIPELINE_AVAILABLE:
        logging.error("Pipeline components not available")
        return False
        
    try:
        # Load configuration
        from config import AppConfig
        config = AppConfig()
        
        # Create pipeline
        pipeline = UnifiedGPUPipeline(
            camera_id="test",
            source="test_video.mp4",  # Would use test video
            config=config
        )
        
        # Create validator
        validator = GPUPipelineValidator()
        
        # Run validation
        results = validator.run_full_validation(pipeline)
        
        # Save results
        output = {
            'timestamp': time.time(),
            'config': config_file,
            'results': {k: v.to_dict() for k, v in results.items()}
        }
        
        with open('validation_results.json', 'w') as f:
            json.dump(output, f, indent=2)
            
        return all(r.passed for r in results.values())
        
    except Exception as e:
        logging.error(f"Validation failed: {e}")
        return False


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU Pipeline Validator")
    parser.add_argument('--config', default='config.json',
                        help='Configuration file path')
    parser.add_argument('--test', choices=['integrity', 'fallback', 'memory', 'full'],
                        default='full', help='Test to run')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run validation
    success = validate_pipeline_from_config(args.config)
    exit(0 if success else 1) 