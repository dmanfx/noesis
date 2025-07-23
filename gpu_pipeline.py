#!/usr/bin/env python3
"""
Unified GPU Pipeline

This module provides a unified GPU-only video processing pipeline that eliminates
CPU bottlenecks by keeping all operations on GPU memory:

Pipeline: DeepStream (GPU) -> GPU Preprocessing -> TensorRT Inference (GPU) -> Results

Key features:
- Zero-copy GPU memory operations
- No CPU fallbacks - fails hard if GPU operations fail
- Direct tensor processing between components
- Maximum performance with minimal memory transfers
- Resource management and conflict prevention
- Enhanced error handling and recovery
"""

import os
os.environ['no_proxy'] = '*'
import torch
import numpy as np
import time
import logging
import threading
import queue
from typing import Union, Optional, Dict, Any
import re

from config import AppConfig
from deepstream_video_pipeline import create_deepstream_video_processor
# Remove redundant TensorRT inference imports - DeepStream handles inference natively
# from deepstream_inference_bridge import DeepStreamInferenceBridge  # Removed - redundant
# from tensorrt_inference import GPUOnlyDetectionManager, TensorRTModelManager  # Removed - redundant
from detection import PoseEstimator
from tracking import TrackingSystem
from models import AnalysisFrame, convert_detection_to_detection_result

# Ensure convert_detection_to_detection_result is always available
if convert_detection_to_detection_result is None:
    try:
        from models import convert_detection_to_detection_result
    except ImportError:
        # Create a fallback function if import fails
        def convert_detection_to_detection_result(detection, detection_id: int = 0):
            from models import DetectionResult
            return DetectionResult(
                id=detection_id,
                class_id=getattr(detection, 'class_id', 0),
                confidence=getattr(detection, 'confidence', 0.0),
                bbox=getattr(detection, 'bbox', (0.0, 0.0, 0.0, 0.0))
            )
from performance_profiler import profile_step, get_profiler, configure_profiler
from utils.interrupt import safe_join
from utils import RateLimitedLogger

# Import GPU memory pool for resource management
try:
    from gpu_memory_pool import get_global_memory_pool
    GPU_MEMORY_POOL_AVAILABLE = True
except ImportError:
    GPU_MEMORY_POOL_AVAILABLE = False
    logging.getLogger(__name__).warning("GPU memory pool not available - using fallback memory management")

# Import pipeline performance optimizer
try:
    from pipeline_performance_optimizer import get_pipeline_optimizer
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False
    logging.getLogger(__name__).warning("Pipeline optimizer not available - performance optimization disabled")

logger = logging.getLogger(__name__)

def validate_gpu_tensor(tensor: torch.Tensor, operation_name: str) -> torch.Tensor:
    """Validate that a tensor is on GPU. Fails hard if not."""
    if not tensor.is_cuda:
        raise RuntimeError(f"GPU-only violation in {operation_name}: tensor on {tensor.device}")
    return tensor


class UnifiedGPUPipeline:
    """
    Unified GPU-only video processing pipeline for maximum performance.
    
    This pipeline processes video entirely on GPU memory:
    DeepStream Decode -> GPU Preprocess -> TensorRT Inference -> Results
    
    Enhanced with:
    - Resource management and conflict prevention
    - GPU memory pool integration
    - Enhanced error handling and recovery
    - Process lifecycle management
    - Performance optimization and monitoring
    """
    
    def __init__(
        self,
        camera_id: str,
        source: Union[str, int],
        config: AppConfig,
        output_queue: Optional[queue.Queue] = None,
        shared_data: Optional[Dict] = None,
        stream_width: Optional[int] = None,
        stream_height: Optional[int] = None
    ):
        """
        Initialize unified GPU pipeline.
        
        Args:
            camera_id: Camera identifier
            source: Video source (RTSP URL, file path, or device index)
            config: Application configuration
            output_queue: Queue for processed analysis frames
            shared_data: Optional shared data dictionary
        """
        self.camera_id = camera_id
        self.source = source
        self.config = config
        self.output_queue = output_queue
        self.shared_data = shared_data or {}
        
        # Use per-stream dimensions if provided, otherwise fall back to config defaults
        self.stream_width = stream_width or config.cameras.CAMERA_WIDTH
        self.stream_height = stream_height or config.cameras.CAMERA_HEIGHT
        
        # Initialize logger
        self.logger = logging.getLogger(f"UnifiedGPUPipeline-{camera_id}")
        # Use rate-limited logger for frame-level messages
        self.rate_limited_logger = RateLimitedLogger(self.logger, rate_limit_seconds=5.0)
        
        # Validate GPU-only configuration
        self._validate_gpu_config()
        
        # Initialize GPU memory pool if available
        if GPU_MEMORY_POOL_AVAILABLE:
            self.memory_pool = get_global_memory_pool(device=config.models.DEVICE)
        else:
            self.memory_pool = None
        
        # Initialize performance optimizer if available
        if OPTIMIZER_AVAILABLE:
            self.performance_optimizer = get_pipeline_optimizer(
                device_id=int(config.models.DEVICE.split(':')[-1]) if ':' in config.models.DEVICE else 0
            )
        else:
            self.performance_optimizer = None
        
        # Initialize GPU components (DeepStream-only pipeline)
        self.deepstream_processor = None  # DeepStream processor (primary and only video reader)
        self.gpu_detector = None
        self.pose_estimator = None
        self.tracking_system = None
        
        # Pipeline state
        self.running = False
        self.stop_event = threading.Event()
        self.pipeline_thread = None
        
        # Performance tracking
        self.frame_count = 0
        self.processed_count = 0
        self.fps = 0.0
        self.start_time = time.time()
        
        # Enhanced profiler configuration
        if config.processing.ENABLE_PROFILING:
            self.profiler = configure_profiler(
                enable_profiling=True,
                sampling_rate=getattr(config.processing, 'PROFILING_SAMPLING_RATE', 10),
                lightweight_mode=True,
                async_logging=True
            )
        else:
            self.profiler = get_profiler()  # Disabled profiler
        
        # Error handling state
        self.consecutive_errors = 0
        self.max_consecutive_errors = 10
        
        # Store staggered delay for execution during start()
        self.staggered_delay = 0
        try:
            # Try to extract numeric part from camera_id for staggered startup
            match = re.search(r'(\d+)', str(camera_id))
            if match:
                stream_index = int(match.group(1))
                self.staggered_delay = stream_index * 3.0  # 3 second stagger between streams
                self.logger.info(f"🔧 Staggered startup: Camera {camera_id} (index {stream_index}) will delay {self.staggered_delay:.1f}s during start()")
        except Exception:
            # Fallback to hash-based delay if camera_id parsing fails
            self.staggered_delay = (hash(str(camera_id)) % 5) * 2.0  # 0-8 second random delay
            self.logger.info(f"🔧 Staggered startup: Camera {camera_id} (hash-based) will delay {self.staggered_delay:.1f}s during start()")
        
        if self.staggered_delay == 0:
            self.logger.info(f"🔧 No staggered delay for {self.camera_id} (first stream or no numeric index)")
        
        self.logger.info(f"Initialized unified GPU pipeline for camera {camera_id}")
    
    def _validate_gpu_config(self):
        """Validate that all required GPU components are enabled."""
        if not self.config.processing.ENABLE_DEEPSTREAM:
            raise RuntimeError("Unified GPU Pipeline: DeepStream must be enabled")
            
        if not self.config.processing.ENABLE_GPU_PREPROCESSING:
            raise RuntimeError("Unified GPU Pipeline: GPU preprocessing must be enabled")
            
        if not self.config.models.ENABLE_TENSORRT:
            raise RuntimeError("Unified GPU Pipeline: TensorRT must be enabled")
            
        if not self.config.models.FORCE_GPU_ONLY:
            raise RuntimeError("Unified GPU Pipeline: GPU-only mode must be enabled")
            
        if not torch.cuda.is_available():
            raise RuntimeError("Unified GPU Pipeline: CUDA is not available")
    
    def start(self) -> bool:
        """Start the unified GPU pipeline with enhanced resource management."""
        try:
            # Execute staggered startup delay (non-blocking for main thread)
            if self.staggered_delay > 0:
                startup_timestamp = time.time()
                self.logger.info(f"⏱️  Staggered startup delay initiated for {self.camera_id} at {startup_timestamp:.3f}")
                time.sleep(self.staggered_delay)
                delay_completed_timestamp = time.time()
                actual_delay = delay_completed_timestamp - startup_timestamp
                self.logger.info(f"⏱️  Staggered startup delay completed for {self.camera_id}: {actual_delay:.3f}s (target: {self.staggered_delay:.1f}s)")
            
            # Initialize DeepStream processor (required - no fallback)
            self.logger.info(f"Initializing DeepStream video processor for {self.camera_id}")
            self.deepstream_processor = create_deepstream_video_processor(
                camera_id=self.camera_id,
                source=self.source,
                config=self.config
            )
            if not self.deepstream_processor.start():
                raise RuntimeError(f"DeepStream processor failed to start for {self.camera_id} - no fallback available")
            
            # DeepStream handles preprocessing internally - no separate preprocessor needed
            self.logger.info(f"Using DeepStream integrated preprocessing for {self.camera_id}")
            
            # Phase 2: Remove redundant TensorRT inference - DeepStream handles this natively
            # DeepStream already does YOLOv11 inference with parse-bbox-func-name=NvDsInferParseYOLO11
            self.logger.info(f"Using DeepStream native YOLOv11 inference for {self.camera_id}")
            self.gpu_detector = None  # No longer needed - DeepStream handles inference
            
            # Initialize pose estimator
            self.logger.info(f"Initializing pose estimator for {self.camera_id}")
            self.pose_estimator = PoseEstimator(
                model_path=self.config.models.POSE_MODEL_PATH,
                confidence_threshold=self.config.models.MODEL_CONFIDENCE_THRESHOLD
            )
            
            # Initialize tracking system
            self.logger.info(f"Initializing tracking system for {self.camera_id}")
            self.tracking_system = TrackingSystem(
                tracker_config=self.config.tracking.get_tracker_config(),
                inactive_threshold_seconds=self.config.tracking.INACTIVE_THRESHOLD_SECONDS,
                trace_persistence_seconds=self.config.tracking.TRACE_PERSISTENCE_SECONDS
            )
            
            # Start pipeline thread
            self.running = True
            self.pipeline_thread = threading.Thread(target=self._pipeline_loop, daemon=True)
            self.pipeline_thread.start()
            
            self.logger.info(f"✅ Unified GPU pipeline started for camera {self.camera_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start unified GPU pipeline: {e}")
            # Cleanup on failure
            self._cleanup_resources()
            raise RuntimeError(f"Unified GPU Pipeline startup failed: {e}")
    
    def _pipeline_loop(self):
        """Main pipeline processing loop - entirely on GPU with enhanced performance optimization."""
        self.logger.info(f"Starting unified GPU pipeline loop for {self.camera_id}")
        
        consecutive_failures = 0
        max_consecutive_failures = 10
        last_fps_update = time.time()
        processed_count_interval = 0
        
        while self.running and not self.stop_event.is_set():
            try:
                with torch.inference_mode():  # Prevent autograd memory allocation
                    # Stage 1: Read frame data from DeepStream (includes native inference results)
                    stage_context = self.performance_optimizer.profile_stage("deepstream_read") if self.performance_optimizer else None
                    if stage_context:
                        with stage_context:
                            ret, frame_data = self.deepstream_processor.read_gpu_tensor()
                    else:
                        ret, frame_data = self.deepstream_processor.read_gpu_tensor()
                    
                    if not ret or frame_data is None:
                        consecutive_failures += 1
                        if self.logger.isEnabledFor(logging.DEBUG):
                            self.logger.debug(f"Failed to read frame data from DeepStream for {self.camera_id} (failure {consecutive_failures}/{max_consecutive_failures})")
                        
                        if consecutive_failures >= max_consecutive_failures:
                            self.logger.error(f"Too many consecutive DeepStream read failures for {self.camera_id}")
                            break
                        
                        time.sleep(0.1)
                        continue
                    
                    # Phase 2: DeepStream provides both frame data and inference results
                    # Extract frame information
                    gpu_frame_tensor = frame_data.get('tensor')
                    detections = frame_data.get('detections', [])  # DeepStream native detections
                    # source_id = frame_data.get('source_id', 0)  # Unused
                    # frame_num = frame_data.get('frame_num', self.frame_count)  # Unused
                    
                    # Reset failure counter on successful read
                    consecutive_failures = 0
                    
                    # Update frame count
                    self.frame_count += 1
                    frame_timestamp = time.time()
                    
                    # Apply frame filtering based on configuration
                    # Only process frames that meet the analysis interval criteria
                    if self.config.processing.ANALYSIS_FRAME_INTERVAL > 1:
                        if (self.frame_count - 1) % self.config.processing.ANALYSIS_FRAME_INTERVAL != 0:
                            # Skip this frame - don't process or save
                            continue
                    
                    # Memory monitoring
                    if self.frame_count % 100 == 0:
                        allocated_mb = torch.cuda.memory_allocated() // 1e6
                        cached_mb = torch.cuda.memory_reserved() // 1e6
                        self.logger.debug(f"GPU mem: {allocated_mb:.0f}MB allocated, {cached_mb:.0f}MB cached")
                        
                        # Warning if memory usage is high
                        if cached_mb > 8000:  # 8GB threshold for RTX 3060
                            self.logger.warning(f"High GPU memory usage: {cached_mb:.0f}MB")
                    
                    # Phase 2: Use DeepStream native inference results (no redundant TensorRT processing)
                    process_start_time = time.time()
                    self.rate_limited_logger.debug(f"Using DeepStream native detections: {len(detections)} objects found")
                    
                    # Convert GPU tensor to numpy for legacy components (visualization) if tensor is available
                    frame_numpy = None
                    if gpu_frame_tensor is not None:
                        frame_numpy = self._tensor_to_numpy_bgr(gpu_frame_tensor)
                    else:
                        # Create dummy frame if no tensor available
                        frame_numpy = np.zeros((self.stream_height, self.stream_width, 3), dtype=np.uint8)
                    
                    # Convert detection dictionaries to Detection objects for compatibility
                    detection_objects = []
                    for det in detections:
                        # Convert DeepStream detection dict to Detection object
                        detection_obj = self._convert_deepstream_detection(det)
                        detection_objects.append(detection_obj)
                    
                    # Stage 3: Tracking (if enabled)
                    tracks = []
                    if self.config.tracking.ENABLE_TRACKING and detection_objects:
                        if getattr(self.config.tracking, 'USE_NATIVE_DEEPSTREAM_TRACKER', False):
                            # Native DeepStream tracker: use object_id directly, skip custom tracking
                            self.logger.info("Using native DeepStream tracker (nvtracker) for object IDs.")
                            tracks = []
                            try:
                                from models import TrackingResult, convert_detection_to_detection_result
                            except ImportError:
                                TrackingResult = None
                                convert_detection_to_detection_result = None
                            for i, det in enumerate(detection_objects):
                                # Ensure bbox is a tuple of floats for DetectionResult
                                bbox = tuple(float(x) for x in det.bbox) if hasattr(det, 'bbox') else (0.0, 0.0, 0.0, 0.0)
                                if TrackingResult and convert_detection_to_detection_result:
                                    try:
                                        # Pass bbox as tuple to convert_detection_to_detection_result
                                        det_for_result = det
                                        if hasattr(det, 'bbox') and not isinstance(det.bbox, tuple):
                                            # Create a shallow copy with bbox as tuple if needed
                                            from copy import copy
                                            det_for_result = copy(det)
                                            det_for_result.bbox = bbox
                                        detection_result = convert_detection_to_detection_result(det_for_result, detection_id=i)
                                        tracks.append(TrackingResult(
                                            track_id=getattr(det, 'object_id', -1),
                                            camera_id=self.camera_id,
                                            detection=detection_result
                                        ))
                                    except Exception as e:
                                        self.logger.error(f"Failed to convert detection for tracking {i}: {e}")
                                        # Fallback to simple dict format
                                        tracks.append({
                                            'track_id': getattr(det, 'object_id', -1),
                                            'camera_id': self.camera_id,
                                            'detection': det
                                        })
                                else:
                                    tracks.append({
                                        'track_id': getattr(det, 'object_id', -1),
                                        'camera_id': self.camera_id,
                                        'detection': det
                                    })
                        elif self.tracking_system:
                            # Custom Python tracking
                            self.logger.info("Using custom Python tracking system.")
                            stage_context = self.performance_optimizer.profile_stage("tracking") if self.performance_optimizer else None
                            if stage_context:
                                with stage_context:
                                    tracks = self.tracking_system.update(detection_objects, frame_numpy, self.camera_id)
                            else:
                                tracks = self.tracking_system.update(detection_objects, frame_numpy, self.camera_id)
                    
                    # Convert Detection objects to DetectionResult objects for visualization
                    detection_results = []
                    for i, detection in enumerate(detection_objects):
                        try:
                            # Ensure convert_detection_to_detection_result is available
                            if convert_detection_to_detection_result is None:
                                # Re-import if it was set to None
                                from models import convert_detection_to_detection_result
                            
                            detection_result = convert_detection_to_detection_result(detection, detection_id=i)
                            # Validate conversion worked properly
                            if not hasattr(detection_result, 'bbox') or detection_result.bbox is None:
                                self.logger.error(f"Detection conversion failed for detection {i}: no bbox")
                            detection_results.append(detection_result)
                        except Exception as e:
                            self.logger.error(f"Failed to convert detection {i}: {e}")
                            # Create a fallback DetectionResult
                            from models import DetectionResult
                            detection_results.append(DetectionResult(
                                id=i,
                                class_id=getattr(detection, 'class_id', 0),
                                confidence=getattr(detection, 'confidence', 0.0),
                                bbox=getattr(detection, 'bbox', (0.0, 0.0, 0.0, 0.0))
                            ))
                    
                    # Log detection conversion for debugging
                    if detection_results and self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(f"Converted {len(detection_results)} detections to DetectionResult format for {self.camera_id}")
                    
                    # Create analysis frame
                    analysis_frame = AnalysisFrame(
                        frame_id=self.frame_count,
                        camera_id=self.camera_id,
                        timestamp=frame_timestamp,
                        frame=frame_numpy,
                        detections=detection_results,  # Use converted DetectionResult objects
                        tracks=tracks,
                        frame_width=self.stream_width,
                        frame_height=self.stream_height,
                        processing_time=(time.time() - process_start_time) * 1000
                    )
                    
                    # Output to queue (non-blocking)
                    if self.output_queue:
                        try:
                            self.output_queue.put(analysis_frame, block=False)
                        except queue.Full:
                            if self.logger.isEnabledFor(logging.DEBUG):
                                self.logger.debug("Output queue full, dropping analysis frame")
                    
                    # Update processed count
                    processed_count_interval += 1
                    
                    # ADD EXPLICIT CLEANUP HERE
                    # Clean up GPU tensors to prevent memory leak
                    
                    # Periodic cache clearing to prevent GPU OOM
                    if self.frame_count % 300 == 0:  # Every 300 frames (~10s at 30fps)
                        torch.cuda.empty_cache()
                        self.logger.debug(f"GPU cache cleared at frame {self.frame_count}")
                    
                    # Calculate FPS
                    current_time = time.time()
                    if current_time - last_fps_update >= 5.0:  # Update every 5 seconds
                        self.fps = processed_count_interval / (current_time - last_fps_update)
                        processed_count_interval = 0
                        last_fps_update = current_time
                        
                        if self.logger.isEnabledFor(logging.DEBUG):
                            self.logger.debug(f"Unified GPU Pipeline FPS: {self.fps:.2f} for {self.camera_id}")
                    
                    # Reset consecutive errors on successful processing
                    self.consecutive_errors = 0
                
            except RuntimeError as e:
                # Handle CUDA errors with fail-hard behavior
                if "CUDA error" in str(e):
                    self.logger.error(f"CUDA error in GPU pipeline for {self.camera_id}: {e}")
                    self.logger.error("GPU-only mode: Terminating pipeline due to CUDA failure")
                    self.running = False
                    raise RuntimeError(f"GPU-only pipeline failure for {self.camera_id}: {e}")
                else:
                    # Handle other RuntimeErrors normally
                    self.consecutive_errors += 1
                    self.logger.error(f"Runtime error in GPU pipeline (error {self.consecutive_errors}/{self.max_consecutive_errors}): {e}")
                    
                    if self.consecutive_errors >= self.max_consecutive_errors:
                        self.logger.error(f"Too many consecutive runtime errors ({self.consecutive_errors}), stopping pipeline for {self.camera_id}")
                        break
                    
                    time.sleep(0.1)
            except Exception as e:
                self.consecutive_errors += 1
                self.logger.error(f"Unexpected error in unified GPU pipeline (error {self.consecutive_errors}/{self.max_consecutive_errors}): {e}")
                
                if self.consecutive_errors >= self.max_consecutive_errors:
                    self.logger.error(f"Too many consecutive errors ({self.consecutive_errors}), stopping pipeline for {self.camera_id}")
                    break
                
                # Brief pause before retrying
                time.sleep(0.1)
        
        self.logger.info(f"Unified GPU pipeline loop stopped for {self.camera_id}")
    
    def _convert_deepstream_detection(self, detection_dict: Dict[str, Any]) -> Any:
        """Convert DeepStream detection dictionary to Detection object for compatibility"""
        try:
            # Import Detection class here to avoid circular imports
            from detection import Detection
            
            # Extract detection data from DeepStream format
            bbox = detection_dict.get('bbox', [0, 0, 0, 0])
            class_id = detection_dict.get('class_id', 0)
            confidence = detection_dict.get('confidence', 0.0)
            object_id = detection_dict.get('object_id', -1)
            
            # Create Detection object (object_id is not a parameter of Detection.__init__)
            detection_obj = Detection(
                bbox=bbox,
                confidence=confidence,
                class_id=class_id,
                class_name=f"class_{class_id}"
            )
            
            # Add object_id as an attribute after creation if needed
            if object_id != -1:
                detection_obj.object_id = object_id
            
            return detection_obj
            
        except Exception as e:
            self.logger.error(f"Failed to convert DeepStream detection: {e}")
            # Return a dummy detection to prevent pipeline failure
            from detection import Detection
            return Detection(
                bbox=[0, 0, 0, 0],
                confidence=0.0,
                class_id=0,
                class_name="unknown"
            )

    def _tensor_to_numpy_bgr(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert GPU tensor to numpy array for legacy components.
        
        CRITICAL FIX: Handles DeepStream memory stride padding properly to prevent
        visual corruption from memory layout mismatches.
        
        This method is needed only for components that haven't been converted
        to full tensor processing yet. It will be removed as more components
        become tensor-native.
        """
        try:
            with profile_step("tensor_to_numpy_conversion"):
                # PHASE 1: Ensure tensor is contiguous to eliminate stride padding
                # DeepStream often outputs tensors with memory padding that must be removed
                if not tensor.is_contiguous():
                    self.logger.debug("DeepStream tensor not contiguous, making contiguous to fix stride")
                    tensor = tensor.contiguous()
                
                # PHASE 2: Validate tensor dimensions
                if tensor.dim() != 3 or tensor.shape[0] != 3:
                    raise RuntimeError(f"Expected (C, H, W) tensor with C=3, got {tensor.shape}")
                
                # PHASE 3: Move to CPU with contiguous guarantee
                tensor_cpu = tensor.cpu().contiguous()
                
                # PHASE 4: Scale to [0,255] with explicit dtype conversion
                # Ensure we're working with proper data types
                if tensor_cpu.dtype in [torch.float16, torch.float32]:
                    tensor_scaled = (tensor_cpu * 255.0).clamp(0, 255).to(torch.uint8)
                else:
                    # Already uint8, just clamp
                    tensor_scaled = tensor_cpu.clamp(0, 255).to(torch.uint8)
                
                # PHASE 5: Convert from (C, H, W) to (H, W, C) with contiguous guarantee
                tensor_hwc = tensor_scaled.permute(1, 2, 0).contiguous()
                
                # PHASE 6: Create numpy array and validate buffer size
                numpy_rgb = tensor_hwc.numpy()
                
                # BUFFER SIZE VALIDATION: Critical check for stride issues
                expected_size = tensor.shape[1] * tensor.shape[2] * 3  # H * W * C
                actual_size = numpy_rgb.size
                if actual_size != expected_size:
                    self.logger.error("Buffer size mismatch detected!")
                    self.logger.error(f"Expected: {expected_size} bytes ({tensor.shape[1]}x{tensor.shape[2]}x3)")
                    self.logger.error(f"Actual: {actual_size} bytes")
                    self.logger.error(f"Tensor shape: {tensor.shape}, numpy shape: {numpy_rgb.shape}")
                    raise RuntimeError(f"Memory stride corruption: expected {expected_size} bytes, got {actual_size}")
                
                # PHASE 7: Convert RGB to BGR for OpenCV compatibility
                numpy_bgr = numpy_rgb[:, :, [2, 1, 0]].copy()  # Force copy for contiguous memory
                
                # FINAL VALIDATION: Ensure output array is contiguous
                if not numpy_bgr.flags['C_CONTIGUOUS']:
                    self.logger.warning("Output array not contiguous, fixing")
                    numpy_bgr = np.ascontiguousarray(numpy_bgr)
                
                # LOG SUCCESS: Buffer size validation passed
                self.logger.debug(f"Tensor conversion successful: {tensor.shape} -> {numpy_bgr.shape}")
                self.logger.debug(f"Buffer size validated: {numpy_bgr.size} bytes as expected")
                
                return numpy_bgr
                
        except Exception as e:
            self.logger.error(f"Tensor to numpy conversion failed: {e}")
            self.logger.error(f"Input tensor: shape={tensor.shape}, device={tensor.device}, dtype={tensor.dtype}")
            self.logger.error(f"Is contiguous: {tensor.is_contiguous()}")
            raise RuntimeError(f"GPU-only mode: Tensor to numpy conversion failed: {e}")
    
    def _cleanup_resources(self):
        """Clean up pipeline resources"""
        try:
            # Clean up DeepStream processor
            if self.deepstream_processor:
                # Check if DeepStream processor has GPU reader that needs cleanup
                if hasattr(self.deepstream_processor, 'gpu_reader') and self.deepstream_processor.gpu_reader:
                    try:
                        self.deepstream_processor.gpu_reader.close()
                        self.logger.info(f"GPU reader cleaned up for {self.camera_id}")
                    except Exception as e:
                        self.logger.error(f"Error cleaning up GPU reader: {e}")
                
                self.deepstream_processor.stop()
                self.deepstream_processor = None
            
            # Clean up GPU memory if pool is available
            if self.memory_pool:
                # Force garbage collection
                torch.cuda.empty_cache()
            
            self.logger.info(f"Resources cleaned up for {self.camera_id}")
            
        except Exception as e:
            self.logger.error(f"Error during resource cleanup: {e}")
    
    def stop(self):
        """Stop the unified GPU pipeline with enhanced cleanup."""
        self.logger.info(f"Stopping unified GPU pipeline for {self.camera_id}")
        
        # Set stop flag
        self.running = False
        self.stop_event.set()
        
        # Wait for pipeline thread
        if self.pipeline_thread and self.pipeline_thread.is_alive():
            safe_join(self.pipeline_thread, timeout=2.0, name=f"pipeline_{self.camera_id}")
        
        # Clean up resources
        self._cleanup_resources()
        
        self.logger.info(f"Stopped unified GPU pipeline for {self.camera_id}")
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics including optimization metrics."""
        # Get DeepStream processor stats
        deepstream_stats = {}
        if self.deepstream_processor:
            deepstream_stats = self.deepstream_processor.get_stats()
        
        # Add memory pool stats if available
        memory_stats = {}
        if self.memory_pool:
            try:
                memory_stats = self.memory_pool.get_stats()
            except Exception as e:
                self.logger.debug(f"Could not get memory pool stats: {e}")
        
        # Add optimization stats if available
        optimization_stats = {}
        if self.performance_optimizer:
            try:
                optimization_stats = self.performance_optimizer.get_optimization_report()
            except Exception as e:
                self.logger.debug(f"Could not get optimization stats: {e}")
        
        return {
            'camera_id': self.camera_id,
            'frames_processed': self.frame_count,
            'fps': self.fps,
            'running_time': time.time() - self.start_time,
            'pipeline_type': 'unified_gpu_deepstream_only',
            'deepstream_stats': deepstream_stats,
            'memory_pool_stats': memory_stats,
            'consecutive_errors': self.consecutive_errors,
            'running': self.running,
            'optimization_stats': optimization_stats
        }
    
    def get_performance_data(self) -> Dict[str, Any]:
        """Get performance data (alias for get_pipeline_stats for compatibility)"""
        return self.get_pipeline_stats()


# Global cleanup function for resource management
def cleanup_all_gpu_resources():
    """Cleanup all global GPU resources"""
    logger = logging.getLogger("GPUResourceCleanup")
    logger.info("Cleaning up all GPU resources...")
    
    # Force GPU cache cleanup
    try:
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared")
    except Exception as e:
        logger.error(f"Error clearing GPU cache: {e}")
    
    logger.info("✅ All GPU resources cleanup complete")


# For backward compatibility - export the main pipeline class
__all__ = ['UnifiedGPUPipeline', 'cleanup_all_gpu_resources'] 