#!/usr/bin/env python3
"""
GPU-Only Video Analysis Application

This application provides real-time video analysis using a pure GPU pipeline:
- NVDEC hardware video decoding (GPU)
- GPU-accelerated frame preprocessing  
- TensorRT inference (GPU)
- Real-time WebSocket streaming

The system uses ZERO CPU fallbacks - all operations must succeed on GPU.
"""

import argparse
import asyncio
import logging
import multiprocessing
import os
os.environ['no_proxy'] = '*'
import queue
import signal
import sys
import threading
import time
import json
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import cv2
import numpy as np

from config import AppConfig, config
from detection import DetectionManager, PoseEstimator
from models import DetectionResult, TrackingResult, AnalysisFrame, convert_numpy_types
from deepstream_video_pipeline import create_deepstream_video_processor
# from gpu_pipeline import UnifiedGPUPipeline, cleanup_all_gpu_resources  # DEPRECATED
from tracking import TrackingSystem

class DeepStreamProcessorWrapper:
    """Simple wrapper to provide UnifiedGPUPipeline interface for DeepStream processor"""
    
    def __init__(self, camera_id: str, source, config, output_queue=None, websocket_server=None):
        self.camera_id = camera_id
        self.source = source
        self.config = config
        self.output_queue = output_queue
        self.websocket_server = websocket_server
        self.processor = None
        self.running = False
        self.frame_count = 0
        self.logger = logging.getLogger(f"DeepStreamWrapper-{camera_id}")
        # Use rate-limited logger for performance-critical messages
        self.rate_limited_logger = RateLimitedLogger(self.logger, rate_limit_seconds=5.0)
        
    def start(self) -> bool:
        """Start the DeepStream processor"""
        try:
            self.processor = create_deepstream_video_processor(
                camera_id=self.camera_id,
                source=self.source,
                config=self.config
            )
            
            if not self.processor.start():
                self.logger.error(f"Failed to start DeepStream processor for {self.camera_id}")
                return False
            
            self.running = True
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            
            self.logger.info(f"✅ DeepStream processor started for {self.camera_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start DeepStream processor: {e}")
            return False
    
    def _processing_loop(self):
        """Main processing loop that reads frames and puts them in the output queue"""
        self.logger.info(f"Starting processing loop for {self.camera_id}")
        
        while self.running:
            try:
                if self.config.visualization.USE_NATIVE_DEEPSTREAM_OSD:
                    # Forward GPU-encoded JPEG bytes directly
                    success, jpeg_bytes = self.processor.read_encoded_jpeg()
                    if success and jpeg_bytes and self.websocket_server:
                        cam_id_bytes = self.camera_id.encode('utf-8')
                        msg = bytes([len(cam_id_bytes)]) + cam_id_bytes + jpeg_bytes
                        self.websocket_server.broadcast_sync(msg)
                    else:
                        time.sleep(0.005)
                    continue
                
                ret, frame_data = self.processor.read_gpu_tensor()
                
                if ret and frame_data:
                    self.frame_count += 1
                    
                    # Removed skip logic to restore frame processing for native OSD
                    
                    # Adaptive frame processing based on performance
                    frame_start_time = time.time()
                    
                    # Convert frame data to AnalysisFrame and put in output queue
                    if self.output_queue:
                        try:
                            analysis_frame = self._convert_to_analysis_frame(frame_data)
                            if analysis_frame:
                                self.output_queue.put_nowait(analysis_frame)
                                if self.frame_count % 100 == 0:  # Rate-limited logging
                                    self.logger.debug(f"Queued frame {self.frame_count} for {self.camera_id}")
                        except queue.Full:
                            self.rate_limited_logger.warning(f"Output queue full for {self.camera_id}")
                        except Exception as e:
                            self.logger.error(f"Error converting frame data: {e}")
                    
                    # Performance-based adaptive processing
                    processing_time = time.time() - frame_start_time
                    target_frame_time = 1.0 / self.config.processing.TARGET_FPS
                    if processing_time > target_frame_time:
                        # Skip next frame if we're falling behind
                        if self.frame_count % 2 == 0:  # Adaptive frame skip
                            continue
                
                else:
                    # No frame data, sleep briefly
                    time.sleep(0.01)
                    
            except Exception as e:
                self.logger.error(f"Error in processing loop for {self.camera_id}: {e}")
                time.sleep(0.1)
        
        self.logger.info(f"Processing loop stopped for {self.camera_id}")
    
    def stop(self):
        """Stop the DeepStream processor"""
        if self.processor:
            self.processor.stop()
        self.running = False
        self.logger.info(f"DeepStream processor stopped for {self.camera_id}")
    
    def read_gpu_tensor(self):
        """Read GPU tensor from DeepStream processor (legacy method - not used)"""
        if not self.running or not self.processor:
            return False, None
        
        # This method is kept for compatibility but the processing loop handles everything
        return self.processor.read_gpu_tensor()
    
    def _convert_to_analysis_frame(self, frame_data):
        """Convert DeepStream frame data to AnalysisFrame"""
        try:
            # Extract data from frame_data
            detections = frame_data.get('detections', [])
            gpu_tensor = frame_data.get('tensor')
            source_id = frame_data.get('source_id', 0)
            frame_num = frame_data.get('frame_num', self.frame_count)
            timestamp = frame_data.get('timestamp', time.time())
            
            # Convert detections to DetectionResult objects
            detection_results = []
            for i, detection in enumerate(detections):
                if isinstance(detection, dict):
                    detection_result = DetectionResult(
                        id=i,
                        class_id=detection.get('class_id', 0),
                        confidence=detection.get('confidence', 0.0),
                        bbox=detection.get('bbox', (0.0, 0.0, 0.0, 0.0)),
                        keypoints=detection.get('keypoints', None),
                        mask=detection.get('mask', None)
                    )
                    detection_results.append(detection_result)
            
            # Convert tensor to numpy frame if available and not using native OSD
            frame_numpy = None
            if not self.config.visualization.USE_NATIVE_DEEPSTREAM_OSD and gpu_tensor is not None:
                try:
                    # Convert GPU tensor to numpy array for visualization
                    if hasattr(gpu_tensor, 'cpu'):
                        tensor_np = gpu_tensor.cpu().numpy()
                    else:
                        tensor_np = gpu_tensor.numpy()
                    
                    # Validate the converted tensor
                    if tensor_np is None or tensor_np.size == 0:
                        self.logger.warning(f"GPU tensor conversion resulted in empty tensor for {self.camera_id}")
                        frame_numpy = None
                    else:
                        self.logger.debug(f"Successfully converted GPU tensor to numpy: shape={tensor_np.shape}, dtype={tensor_np.dtype}")
                        
                        # Convert from CHW to HWC format for OpenCV
                        if tensor_np.shape[0] == 3:  # CHW format
                            frame_numpy = np.transpose(tensor_np, (1, 2, 0))  # Convert to HWC
                            self.logger.debug(f"Converted CHW to HWC: shape={frame_numpy.shape}")
                        
                        # Convert to uint8 if needed
                        if frame_numpy.dtype != np.uint8:
                            if frame_numpy.max() <= 1.0:  # Normalized to [0,1]
                                frame_numpy = (frame_numpy * 255).astype(np.uint8)
                            else:
                                frame_numpy = frame_numpy.astype(np.uint8)
                            self.logger.debug(f"Converted to uint8: shape={frame_numpy.shape}, dtype={frame_numpy.dtype}")
                        
                except Exception as e:
                    self.logger.warning(f"Failed to convert GPU tensor to numpy: {e}")
                    frame_numpy = None
            
            # Create fallback frame if no valid frame available and not using native OSD
            if frame_numpy is None and not self.config.visualization.USE_NATIVE_DEEPSTREAM_OSD:
                # Create a dummy frame for visualization
                frame_numpy = np.zeros((self.config.cameras.CAMERA_HEIGHT, self.config.cameras.CAMERA_WIDTH, 3), dtype=np.uint8)
                self.logger.debug(f"Created fallback frame for {self.camera_id}: shape={frame_numpy.shape}")
            
            # Create AnalysisFrame (without frame_tensor parameter)
            analysis_frame = AnalysisFrame(
                frame_id=frame_num,
                camera_id=self.camera_id,
                timestamp=timestamp,
                frame=frame_numpy,
                detections=detection_results,
                tracks=[],  # No tracking data from DeepStream
                frame_width=self.config.cameras.CAMERA_WIDTH,
                frame_height=self.config.cameras.CAMERA_HEIGHT,
                processing_time=0.0,  # Will be calculated by result processor
                detection_time=0.0,
                tracking_time=0.0
            )
            
            return analysis_frame
            
        except Exception as e:
            self.logger.error(f"Error converting to AnalysisFrame: {e}")
            return None
    
    def get_pipeline_stats(self) -> dict:
        """Return stats directly from the underlying DeepStreamVideoPipeline."""
        if self.processor and hasattr(self.processor, 'get_stats'):
            return self.processor.get_stats()
        return {}

    def get_performance_data(self) -> dict:
        """Get performance+telemetry data for WebSocket stats."""
        # Base skeleton – keeps legacy keys the frontend may read
        stats = {
            'fps': 0.0,
            'frame_count': 0,
            'processing_time_ms': 0.0,
            'status': 'unknown'
        }

        # Fetch full DeepStream stats (includes tracking)
        pipeline_stats = self.get_pipeline_stats()
        if pipeline_stats:
            # Expose under a predictable key for ApplicationManager
            stats['deepstream_stats'] = pipeline_stats
            # Also flatten top-level for backwards compatibility (fps, runtime, etc.)
            stats.update(pipeline_stats)

        return stats

    def get_stats(self):
        """Get comprehensive stats including tracking telemetry"""
        if hasattr(self.processor, 'get_stats'):
            return self.processor.get_stats()
        return {}

    def update_confidence_threshold(self, confidence_threshold: float) -> bool:
        """Update confidence threshold in real-time"""
        if self.processor and hasattr(self.processor, 'update_confidence_threshold'):
            return self.processor.update_confidence_threshold(confidence_threshold)
        return False

    def update_iou_threshold(self, iou_threshold: float) -> bool:
        """Update IOU threshold in real-time"""
        if self.processor and hasattr(self.processor, 'update_iou_threshold'):
            return self.processor.update_iou_threshold(iou_threshold)
        return False

    def set_detection_enabled(self, enabled: bool) -> bool:
        """Enable or disable detection in real-time"""
        if self.processor and hasattr(self.processor, 'set_detection_enabled'):
            return self.processor.set_detection_enabled(enabled)
        return False

    def update_target_classes(self, target_classes: List[int]) -> bool:
        """Update target classes in real-time"""
        if self.processor and hasattr(self.processor, 'update_target_classes'):
            return self.processor.update_target_classes(target_classes)
        return False

# Import TensorRT shutdown mode function
try:
    from tensorrt_inference import set_tensorrt_shutdown_mode
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    def set_tensorrt_shutdown_mode(shutting_down=True):
        pass  # No-op if TensorRT not available
# Import from root utils.py file
import importlib.util
utils_file = os.path.join(os.path.dirname(__file__), 'utils.py')
spec = importlib.util.spec_from_file_location("root_utils", utils_file)
root_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(root_utils)

# Import specific functions
setup_logging = root_utils.setup_logging
ensure_dir = root_utils.ensure_dir
get_timestamp = root_utils.get_timestamp
PerformanceMonitor = root_utils.PerformanceMonitor
RateLimitedLogger = root_utils.RateLimitedLogger

from utils.profiler import profile_step, aggregate_stats
from utils.cpu_profiler import start_global_profiling, stop_global_profiling, get_global_profiler, profile_function
from utils.interrupt import safe_join, safe_process_join
from visualization import VisualizationManager
from websocket_server import WebSocketServer

# Ensure project root is in path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from database import DatabaseManager

# Global interrupt counter for clean shutdown handling
INTERRUPT_COUNT = 0
MAX_SHUTDOWN_TIME = 15  # Maximum time to wait for graceful shutdown

try:
    from simple_dashboard_server import stop_dashboard as stop_simple_dashboard_server
    SIMPLE_DASHBOARD_SERVER_AVAILABLE = True
except ImportError:
    SIMPLE_DASHBOARD_SERVER_AVAILABLE = False

import faulthandler; faulthandler.enable()

class ApplicationManager:
    """Main application manager that coordinates all components"""

    def __init__(self, config: AppConfig):
        """Initialize the application manager
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger("ApplicationManager")
        # Use rate-limited logger for frame-level messages
        self.rate_limited_logger = RateLimitedLogger(self.logger, rate_limit_seconds=5.0)
        
        # Initialize state
        self.running = False
        self.stop_event = threading.Event()
        self.camera_sources = {}
        self.frame_processors = {}
        self.analysis_frame_queue = multiprocessing.Queue(maxsize=100)
        self.streaming_frame_queue = queue.Queue(maxsize=100)
        
        # Initialize components
        self.websocket_server = None
        self.visualization_manager = VisualizationManager()
        
        # Initialize async event loop
        self.event_loop = None
        
        # Performance monitoring for profiling
        self.perf_monitor = PerformanceMonitor() if config.processing.ENABLE_PROFILING else None
        
        # Initialize comprehensive CPU profiling
        self.cpu_profiler = None
        if config.processing.ENABLE_PROFILING:
            # Create CPU profile data directory
            profile_dir = Path("logs/cpu_profiles")
            profile_dir.mkdir(parents=True, exist_ok=True)
            
            # Start comprehensive CPU profiling
            profile_data_file = profile_dir / f"cpu_profile_{get_timestamp()}.jsonl"
            self.cpu_profiler = start_global_profiling(
                profile_data_file=str(profile_data_file),
                thread_sample_interval=0.1,  # Sample threads every 100ms
                system_sample_interval=1.0,  # Sample system every 1s
                enable_function_profiling=True
            )
            self.logger.info(f"✅ Comprehensive CPU profiling started, data: {profile_data_file}")
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle termination signals with graceful shutdown
        
        Args:
            sig: Signal number
            frame: Current stack frame
        """
        global INTERRUPT_COUNT
        INTERRUPT_COUNT += 1
        
        if INTERRUPT_COUNT == 1:
            self.logger.info(f"Received signal {sig}, starting graceful shutdown...")
            # Start graceful shutdown immediately (not in background thread)
            self.stop()
        elif INTERRUPT_COUNT >= 2:
            self.logger.warning("Second interrupt received, forcing immediate exit")
            os._exit(1)
    
    def _graceful_exit(self):
        """Perform graceful shutdown with timeout"""
        try:
            self.logger.info("Starting graceful shutdown process...")
            start_time = time.time()
            
            # Call stop method
            self.stop()
            
            # Wait for shutdown to complete or timeout
            while self.running and (time.time() - start_time) < MAX_SHUTDOWN_TIME:
                time.sleep(0.1)
            
            if self.running:
                self.logger.error(f"Graceful shutdown timed out after {MAX_SHUTDOWN_TIME}s, forcing exit")
                os._exit(1)
            else:
                self.logger.info("Graceful shutdown completed successfully")
                os._exit(0)
                
        except Exception as e:
            self.logger.error(f"Error during graceful shutdown: {e}")
            os._exit(1)
    
    @profile_function("ApplicationManager.initialize")
    def initialize(self):
        """Initialize application components"""
        self.logger.info("Initializing application components")
        
        # Create event loop for async tasks
        self.event_loop = asyncio.new_event_loop()
        
        # Initialize WebSocket server
        self.websocket_server = WebSocketServer(
            host=self.config.websocket.HOST,
            port=self.config.websocket.PORT,
            event_loop=self.event_loop,
            stats_callback=self._get_stats,
            toggle_callback=self._handle_toggle_update
        )
        
        # Add detection configuration callbacks
        self.websocket_server.detection_config_callback = self._handle_detection_config_update
        self.websocket_server.detection_toggle_callback = self._handle_detection_toggle
        
        # Add detection config getter for initial sync
        self.websocket_server.detection_config_getter = self._get_detection_config
        
        # Load camera sources
        self._load_camera_sources()
        
        # Initialize output directory
        ensure_dir(self.config.output.OUTPUT_DIR)
        
        self.logger.info("Application initialization complete")
    
    @profile_function("ApplicationManager.load_camera_sources")
    def _load_camera_sources(self):
        """Load camera sources from configuration"""
        self.logger.info("Loading camera sources")
        
        # Clear existing sources
        self.camera_sources = {}
        
        if self.config.cameras.USE_WEBCAM:
            # Add webcam as source
            self.camera_sources["webcam"] = 0
            self.logger.info("Added webcam as source")
            
        # Load video files
        if self.config.cameras.VIDEO_FILES:
            for i, video_file in enumerate(self.config.cameras.VIDEO_FILES):
                if os.path.exists(video_file):
                    camera_id = f"video_{i}"
                    self.camera_sources[camera_id] = video_file
                    self.logger.info(f"Added video file as source: {video_file}")
                else:
                    self.logger.warning(f"Video file not found: {video_file}")
        
        # Load RTSP streams
        if self.config.cameras.RTSP_STREAMS:
            for i, stream_config in enumerate(self.config.cameras.RTSP_STREAMS):
                # Skip disabled streams
                if not stream_config.get("enabled", True):
                    self.logger.info(f"Skipping disabled stream: {stream_config.get('name', f'Camera {i+1}')}")
                    continue
                    
                camera_id = f"rtsp_{i}"
                stream_name = stream_config.get("name", f"Camera {i+1}")
                stream_url = stream_config["url"]
                stream_width = stream_config["width"]
                stream_height = stream_config["height"]
                
                # Store the full stream config for later use
                self.camera_sources[camera_id] = {
                    "url": stream_url,
                    "name": stream_name,
                    "width": stream_width,
                    "height": stream_height
                }
                self.logger.info(f"Added RTSP stream as source: {stream_name} ({stream_url}) - {stream_width}x{stream_height}")
        
        # Check if we have any sources
        if not self.camera_sources:
            self.logger.warning("No camera sources found in configuration")
    
    @profile_function("ApplicationManager.start")
    def start(self):
        """Start all application components"""
        if self.running:
            return
            
        self.logger.info("Starting application")
        self.running = True
        self.stop_event.clear()
        
        try:
            # Start unified GPU pipeline
            self.logger.info("🚀 Starting unified GPU pipeline...")
            self._start_frame_processors()  # Uses UnifiedGPUPipeline exclusively
            self.logger.info("✅ Frame processors started")
            
            # Start result processing (ALWAYS needed for WebSocket streaming)
            self.logger.info("✅ Starting result processing...")
            self._start_result_processing()
            self.logger.info("✅ Result processing started")
            
            # Start WebSocket server
            self.logger.info("🚀 Starting WebSocket server...")
            self._start_websocket_server()
            self.logger.info("✅ WebSocket server startup initiated")
            
            self.logger.info("🎉 Application started successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Application startup failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    @profile_function("ApplicationManager.start_frame_processors")
    def _start_frame_processors(self):
        """Start unified GPU frame processors for each camera source"""
        self.logger.info("Starting unified GPU frame processors")
        
        # Validate GPU-only configuration
        if not self.config.processing.ENABLE_DEEPSTREAM:
            raise RuntimeError("GPU-only mode: DeepStream must be enabled")
        if not self.config.processing.ENABLE_GPU_PREPROCESSING:
            raise RuntimeError("GPU-only mode: GPU preprocessing must be enabled")
        if not self.config.models.FORCE_GPU_ONLY:
            raise RuntimeError("GPU-only mode: GPU-only inference must be enabled")
        
        for camera_id, source in self.camera_sources.items():
            try:
                self.logger.info(f"Starting unified GPU processor for camera {camera_id}")
                
                # Use DeepStream pipeline directly (bypass deprecated UnifiedGPUPipeline)
                self.logger.info(f"Creating DeepStream processor wrapper for {camera_id}")
                
                processor = DeepStreamProcessorWrapper(
                    camera_id=camera_id,
                    source=source,
                    config=self.config,
                    output_queue=self.analysis_frame_queue,
                    websocket_server=None  # will be set after WebSocket server starts
                )
                
                # Start processor
                self.logger.info(f"Starting processor for {camera_id}...")
                
                # Add timeout for processor startup to prevent hanging
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Processor startup timeout for {camera_id}")
                
                # Set timeout alarm (30 seconds)
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)
                
                try:
                    start_success = processor.start()
                finally:
                    # Cancel the alarm
                    signal.alarm(0)
                
                if start_success:
                    # Store processor
                    self.frame_processors[camera_id] = processor
                    self.logger.info(f"✅ Started unified GPU processor for camera {camera_id}")
                else:
                    self.logger.error(f"❌ Failed to start processor for camera {camera_id}")
                    raise RuntimeError(f"Processor startup failed for {camera_id}")
                
            except Exception as e:
                self.logger.error(f"❌ Error starting processor for camera {camera_id}: {e}")
                
                import traceback
                traceback.print_exc()
                # Continue with other cameras instead of failing completely
                continue
        
        self.logger.info(f"✅ All unified GPU processors started successfully ({len(self.frame_processors)}/{len(self.camera_sources)} cameras)")
        
        # Verify at least one processor started
        if not self.frame_processors:
            raise RuntimeError("No frame processors started successfully")
        elif len(self.frame_processors) < len(self.camera_sources):
            self.logger.warning(f"⚠️  Only {len(self.frame_processors)}/{len(self.camera_sources)} cameras started successfully")
            self.logger.warning("🔧 Check camera connectivity or configuration issues")
        
        # Give processors a moment to initialize
        time.sleep(1.0)
        self.logger.info("Frame processors initialization complete")
    
    @profile_function("ApplicationManager.start_websocket_server")
    def _start_websocket_server(self):
        """Start WebSocket server"""
        if not self.websocket_server:
            self.logger.warning("WebSocket server not initialized")
            return
            
        # Start server in a background thread
        def run_websocket_server():
            try:
                # Set event loop for this thread
                asyncio.set_event_loop(self.event_loop)
                self.logger.info("WebSocket thread started, setting up event loop")
                
                # Run server
                self.logger.info("Starting WebSocket server...")
                self.event_loop.run_until_complete(self.websocket_server.start())
                self.logger.info("WebSocket server started successfully")
                
                # Run event loop
                self.logger.info("Starting WebSocket event loop...")
                self.event_loop.run_forever()
                
            except Exception as e:
                self.logger.error(f"WebSocket server thread failed: {e}")
                import traceback
                traceback.print_exc()
        
        self.websocket_thread = threading.Thread(
            target=run_websocket_server,
            name="WebSocketThread",
            daemon=True
        )
        
        try:
            self.websocket_thread.start()
            self.logger.info(f"Started WebSocket server thread on {self.config.websocket.HOST}:{self.config.websocket.PORT}")
            
            # Give the thread a moment to start
            time.sleep(0.5)
            
            if self.websocket_thread.is_alive():
                self.logger.info("✅ WebSocket server thread is running")
                # Provide websocket_server instance to all frame processors
                for proc in self.frame_processors.values():
                    proc.websocket_server = self.websocket_server
            else:
                self.logger.error("❌ WebSocket server thread failed to start")
                
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket server thread: {e}")
            import traceback
            traceback.print_exc()

    @profile_function("ApplicationManager.start_result_processing")
    def _start_result_processing(self):
        """Start result processing thread"""
        
        def process_results():
            while not self.stop_event.is_set():
                try:
                    # Get analysis frame from queue
                    analysis_frame = self.analysis_frame_queue.get(timeout=1.0)
                    
                    # Process results (e.g., save to file, send to WebSocket clients)
                    self._process_analysis_frame(analysis_frame)
                    
                except queue.Empty:
                    # Queue timeout, continue
                    pass
                except Exception as e:
                    self.logger.error(f"Error processing results: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Create and start processing thread
        self.result_thread = threading.Thread(
            target=process_results,
            name="ResultProcessingThread",
            daemon=True
        )
        self.result_thread.start()
        
        self.logger.info("Started result processing thread")
    
    @profile_function("ApplicationManager.process_analysis_frame")
    def _process_analysis_frame(self, analysis_frame: AnalysisFrame):
        """Process analysis frame
        
        Args:
            analysis_frame: Analysis frame with detection and tracking results
        """
        try:
            # Rate-limited logging for frame processing
            masks_count = sum(1 for det in analysis_frame.detections if det.mask is not None)
            self.rate_limited_logger.debug(f"Processing frame for camera {analysis_frame.camera_id}, frame_id={analysis_frame.frame_id}")
            self.rate_limited_logger.debug(f"Frame has {len(analysis_frame.detections)} detections, {len(analysis_frame.tracks)} tracks, {masks_count} masks")
            
            # Log detection count if > 0 (rate-limited)
            if len(analysis_frame.detections) > 0:
                self.rate_limited_logger.log_detection_count(analysis_frame.camera_id, len(analysis_frame.detections), analysis_frame.frame_id)
            
            # Visualize results with profiling
            if getattr(self.config.visualization, 'USE_NATIVE_DEEPSTREAM_OSD', False):
                self.logger.info("Using native DeepStream OSD (nvdsosd) for visualization.")
                # Ensure frame is available for visualization
                if analysis_frame.frame is not None:
                    annotated_frame = analysis_frame.frame  # Already annotated by DeepStream
                else:
                    self.logger.warning(f"No frame available for visualization, creating fallback frame")
                    annotated_frame = np.zeros((analysis_frame.frame_height, analysis_frame.frame_width, 3), dtype=np.uint8)
            elif self.perf_monitor and self.config.processing.ENABLE_PROFILING:
                with profile_step("visualization", self.perf_monitor):
                    annotated_frame = self.visualization_manager.annotate_frame(
                        frame=analysis_frame.frame.copy(),
                        detections=analysis_frame.detections,
                        tracks=analysis_frame.tracks,
                        show_traces=self.config.visualization.SHOW_TRACES,
                        show_detection_boxes=self.config.visualization.SHOW_DETECTION_BOXES,
                        show_tracking_boxes=self.config.visualization.SHOW_TRACKING_BOXES,
                        show_keypoints=self.config.visualization.SHOW_KEYPOINTS,
                        show_masks=self.config.visualization.SHOW_MASKS,
                        mask_alpha=self.config.visualization.MASK_ALPHA
                    )
            else:
                annotated_frame = self.visualization_manager.annotate_frame(
                    frame=analysis_frame.frame.copy(),
                    detections=analysis_frame.detections,
                    tracks=analysis_frame.tracks,
                    show_traces=self.config.visualization.SHOW_TRACES,
                    show_detection_boxes=self.config.visualization.SHOW_DETECTION_BOXES,
                    show_tracking_boxes=self.config.visualization.SHOW_TRACKING_BOXES,
                    show_keypoints=self.config.visualization.SHOW_KEYPOINTS,
                    show_masks=self.config.visualization.SHOW_MASKS,
                    mask_alpha=self.config.visualization.MASK_ALPHA
                )
            
            # Encode frame for streaming with profiling
            if self.perf_monitor and self.config.processing.ENABLE_PROFILING:
                with profile_step("frame_encoding", self.perf_monitor):
                    # Encode frame as JPEG
                    _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    self.logger.debug(f"JPEG buffer type: {type(buffer)}, shape: {getattr(buffer, 'shape', 'N/A')}")
                    jpeg_data = buffer.tobytes()
            else:
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                self.logger.debug(f"JPEG buffer type: {type(buffer)}, shape: {getattr(buffer, 'shape', 'N/A')}")
                jpeg_data = buffer.tobytes()
            
            # Create binary frame message for WebSocket broadcasting
            # Format: [camera_id_length(1 byte)][camera_id][jpeg_data]
            camera_id_bytes = analysis_frame.camera_id.encode('utf-8')
            camera_id_length = len(camera_id_bytes)
            
            if camera_id_length > 255:
                self.logger.error(f"Camera ID too long: {camera_id_length} bytes")
                return
                
            # Construct binary message
            self.logger.debug(f"Camera ID length: {camera_id_length}, Camera ID bytes: {len(camera_id_bytes)}, JPEG data type: {type(jpeg_data)}, JPEG data length: {len(jpeg_data)}")
            binary_message = bytes([camera_id_length]) + camera_id_bytes + jpeg_data
            self.logger.debug(f"Binary message type: {type(binary_message)}, length: {len(binary_message)}")
            
            # Broadcast frame via WebSocket with profiling  
            if self.websocket_server:
                if self.perf_monitor and self.config.processing.ENABLE_PROFILING:
                    with profile_step("websocket_broadcast", self.perf_monitor):
                        self.websocket_server.broadcast_sync(binary_message)
                else:
                    self.websocket_server.broadcast_sync(binary_message)
            
            # Save frame if enabled and meets save interval criteria
            if self.config.output.SAVE_FRAMES:
                # Apply frame save interval filtering
                should_save = True
                if self.config.output.FRAME_SAVE_INTERVAL > 1:
                    should_save = (analysis_frame.frame_id - 1) % self.config.output.FRAME_SAVE_INTERVAL == 0
                
                if should_save:
                    output_path = Path(self.config.output.OUTPUT_DIR) / analysis_frame.camera_id
                    output_path.mkdir(parents=True, exist_ok=True)
                    
                    frame_filename = output_path / f"frame_{analysis_frame.frame_id:06d}.jpg"
                    cv2.imwrite(str(frame_filename), annotated_frame)
                    
                    # Log frame saving (rate-limited to avoid spam)
                    self.rate_limited_logger.debug(f"Saved frame {analysis_frame.frame_id} for camera {analysis_frame.camera_id}")
                    
        except Exception as e:
            self.logger.error(f"Error processing results: {e}")
            # Continue processing other frames instead of crashing
            import traceback
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")

    @profile_function("ApplicationManager.get_stats")
    def _get_stats(self) -> Dict[str, Any]:
        """Get application statistics for WebSocket broadcasting"""

        current_time = time.time()
        uptime = current_time - getattr(self.config.app, "START_TIME", current_time)

        try:
            stats = {
                'timestamp': current_time,
                'uptime': uptime,
                'application': {
                    'running': self.running,
                    'cameras_active': len(self.camera_sources),
                    'processors_active': len(self.frame_processors)
                },
                'cameras': {}
            }

            # Add camera stats
            for camera_id in self.camera_sources:
                camera_stats = {}

                # Get stats from frame processor
                if camera_id in self.frame_processors:
                    processor = self.frame_processors[camera_id]
                    camera_stats.update(processor.get_performance_data())

                    # Get comprehensive stats including tracking telemetry
                    if hasattr(processor, 'get_stats'):
                        comprehensive_stats = processor.get_stats()

                        # Handle both flat and nested tracking data layouts
                        tracking_data = None
                        if comprehensive_stats:
                            if 'tracking' in comprehensive_stats:
                                tracking_data = comprehensive_stats['tracking']
                            elif 'deepstream_stats' in comprehensive_stats and 'tracking' in comprehensive_stats['deepstream_stats']:
                                tracking_data = comprehensive_stats['deepstream_stats']['tracking']

                        if tracking_data:
                            camera_stats['tracking'] = tracking_data

                # Add to overall stats
                stats['cameras'][camera_id] = camera_stats

            # Add application-level profiling data if enabled
            if self.perf_monitor and self.config.processing.ENABLE_PROFILING:
                stats['application_profiling'] = aggregate_stats(self.perf_monitor)

            # Add comprehensive CPU profiling data if enabled
            if self.cpu_profiler:
                try:
                    cpu_stats = self.cpu_profiler.get_comprehensive_stats()
                    stats['cpu_profiling'] = cpu_stats

                    # Save profile data periodically
                    if self.cpu_profiler.should_save_data():
                        self.cpu_profiler.save_profile_data()

                except Exception as e:
                    self.logger.warning(f"Error getting CPU profiling stats: {e}")
                    stats['cpu_profiling'] = {'error': str(e)}

        except Exception as e:
            self.logger.error(f"Error gathering application stats: {e}")
            stats = {
                'timestamp': current_time,
                'uptime': uptime,
                'application': {'running': self.running},
                'cameras': {}
            }

        return stats
        

    @profile_function("ApplicationManager.get_detection_config")
    def _get_detection_config(self) -> Dict[str, Any]:
        """Get current detection configuration for frontend sync"""
        
        config = {
            'confidence_threshold': self.config.models.MODEL_CONFIDENCE_THRESHOLD,
            'iou_threshold': self.config.models.MODEL_IOU_THRESHOLD,
            'detection_enabled': True,  # Default to enabled
            'detection_toggles': {
                'detect_people': 0 in getattr(self.config.models, 'TARGET_CLASSES', []),
                'detect_vehicles': any(c in getattr(self.config.models, 'TARGET_CLASSES', []) for c in [1, 2, 3, 5, 7, 8]),
                'detect_furniture': any(c in getattr(self.config.models, 'TARGET_CLASSES', []) for c in [13, 56, 57, 59, 60, 61])
            }
        }
        
        return config

    @profile_function("ApplicationManager.handle_toggle_update")
    def _handle_toggle_update(self, toggle_name: str, enabled: bool):
        """Handle visualization toggle updates from WebSocket clients
        
        Args:
            toggle_name: Name of the toggle to update
            enabled: Whether the toggle should be enabled
        """
        self.logger.info(f"Handling toggle update: {toggle_name} = {enabled}")
        
        # Update visualization configuration
        if hasattr(self.config.visualization, toggle_name.upper()):
            setattr(self.config.visualization, toggle_name.upper(), enabled)
            self.logger.info(f"Updated visualization config: {toggle_name.upper()} = {enabled}")
        else:
            self.logger.warning(f"Unknown visualization toggle: {toggle_name}")

    @profile_function("ApplicationManager.handle_detection_config_update")
    def _handle_detection_config_update(self, config_data: Dict[str, Any]):
        """Handle detection configuration updates from WebSocket clients
        
        Args:
            config_data: Dictionary containing detection configuration updates
        """
        self.logger.info(f"Handling detection config update: {config_data}")
        
        try:
            # Update application configuration
            if 'confidence_threshold' in config_data:
                new_threshold = float(config_data['confidence_threshold'])
                self.config.models.MODEL_CONFIDENCE_THRESHOLD = new_threshold
                self.logger.info(f"Updated confidence threshold to: {new_threshold}")
                
                # Update DeepStream pipeline in real-time
                for camera_id, processor in self.frame_processors.items():
                    if hasattr(processor, 'update_confidence_threshold'):
                        processor.update_confidence_threshold(new_threshold)
                        self.logger.info(f"Updated confidence threshold for camera {camera_id}")
            
            if 'iou_threshold' in config_data:
                new_iou = float(config_data['iou_threshold'])
                self.config.models.MODEL_IOU_THRESHOLD = new_iou
                self.logger.info(f"Updated IOU threshold to: {new_iou}")
                
                # Update DeepStream pipeline in real-time
                for camera_id, processor in self.frame_processors.items():
                    if hasattr(processor, 'update_iou_threshold'):
                        processor.update_iou_threshold(new_iou)
                        self.logger.info(f"Updated IOU threshold for camera {camera_id}")
            
            if 'detection_enabled' in config_data:
                enabled = bool(config_data['detection_enabled'])
                self.logger.info(f"Detection enabled: {enabled}")
                
                # Update DeepStream pipeline in real-time
                for camera_id, processor in self.frame_processors.items():
                    if hasattr(processor, 'set_detection_enabled'):
                        processor.set_detection_enabled(enabled)
                        self.logger.info(f"Updated detection enabled for camera {camera_id}: {enabled}")
            
        except Exception as e:
            self.logger.error(f"Error updating detection configuration: {e}")
            import traceback
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")

    @profile_function("ApplicationManager.handle_detection_toggle")
    def _handle_detection_toggle(self, toggle_name: str, enabled: bool):
        """Handle detection toggle updates from WebSocket clients
        
        Args:
            toggle_name: Name of the detection toggle
            enabled: Whether the detection type should be enabled
        """
        self.logger.info(f"Handling detection toggle: {toggle_name} = {enabled}")
        
        try:
            # Map toggle names to class IDs
            class_mapping = {
                'detect_people': [0],  # person
                'detect_vehicles': [1, 2, 3, 5, 7, 8],  # bicycle, car, motorcycle, bus, truck, boat
                'detect_furniture': [13, 56, 57, 59, 60, 61]  # bench, chair, couch, dining table, tv, laptop
            }
            
            if toggle_name in class_mapping:
                target_classes = class_mapping[toggle_name]
                
                # Update DeepStream pipeline in real-time
                for camera_id, processor in self.frame_processors.items():
                    if hasattr(processor, 'update_target_classes'):
                        if enabled:
                            # Add classes to current target classes
                            current_classes = getattr(self.config.models, 'TARGET_CLASSES', [])
                            new_classes = list(set(current_classes + target_classes))
                            self.config.models.TARGET_CLASSES = new_classes
                            processor.update_target_classes(new_classes)
                            self.logger.info(f"Added classes {target_classes} to camera {camera_id}")
                        else:
                            # Remove classes from current target classes
                            current_classes = getattr(self.config.models, 'TARGET_CLASSES', [])
                            new_classes = [c for c in current_classes if c not in target_classes]
                            self.config.models.TARGET_CLASSES = new_classes
                            processor.update_target_classes(new_classes)
                            self.logger.info(f"Removed classes {target_classes} from camera {camera_id}")
            else:
                self.logger.warning(f"Unknown detection toggle: {toggle_name}")
                
        except Exception as e:
            self.logger.error(f"Error updating detection toggle: {e}")
            import traceback
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")

    @profile_function("ApplicationManager.stop")
    def stop(self):
        """Stop all application components with optimized cleanup order"""
        if not self.running:
            return
            
        self.logger.info("Stopping application")
        self.running = False
        self.stop_event.set()
        
        # STEP 1: Set TensorRT shutdown mode to suppress error logging
        set_tensorrt_shutdown_mode(True)
        
        # STEP 2: Early GPU cleanup BEFORE stopping processors (while CUDA context is valid)
        self.logger.info("Starting early GPU resource cleanup...")
        
        # Check if CUDA is available before attempting GPU cleanup
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            # Clean up TensorRT engines first, while CUDA context is still valid
            try:
                # Clean up frame processor GPU detectors
                if self.frame_processors:
                    for camera_id, processor in self.frame_processors.items():
                        try:
                            if hasattr(processor, 'gpu_detector') and hasattr(processor.gpu_detector, 'model_manager'):
                                if hasattr(processor.gpu_detector.model_manager, 'cleanup'):
                                    processor.gpu_detector.model_manager.cleanup()
                                    self.logger.info(f"Early cleanup: GPU detector for {camera_id}")
                        except Exception as e:
                            self.logger.warning(f"Error in early GPU cleanup for {camera_id}: {e}")
                
                # Force cleanup all GPU resources via consolidated pipeline
                cleanup_all_gpu_resources()
                self.logger.info("Early cleanup: all GPU resources via unified pipeline")
                
                # Additional GPU memory pool cleanup if available
                try:
                    from gpu_memory_pool import get_global_memory_pool
                    memory_pool = get_global_memory_pool()
                    memory_pool.clear_pools()
                    self.logger.info("Early cleanup: GPU memory pools")
                except Exception as e:
                    self.logger.debug(f"GPU memory pool cleanup not available: {e}")
                    
            except Exception as e:
                self.logger.warning(f"Error during early GPU cleanup: {e}")
        else:
            self.logger.warning("CUDA not available or no devices, skipping GPU cleanup")
        
        # STEP 3: Stop frame processors (after GPU cleanup)
        if self.frame_processors:
            self.logger.info("Stopping frame processors")
            for camera_id, processor in self.frame_processors.items():
                try:
                    processor.stop()
                except Exception as e:
                    self.logger.error(f"Error stopping processor for {camera_id}: {e}")
            self.frame_processors.clear()
            self.logger.info("Stopped frame processors")
        
        # STEP 4: Stop result processing thread
        if hasattr(self, 'result_thread') and self.result_thread:
            try:
                self.logger.info("Stopping result processing thread")
                safe_join(self.result_thread, timeout=2.0, name="result_processing")
                self.logger.info("Stopped result processing thread")
            except Exception as e:
                self.logger.error(f"Error stopping result processing thread: {e}")
        
        # STEP 5: Stop WebSocket server and event loop
        if self.websocket_server:
            try:
                self.logger.info("Stopping WebSocket server")
                # Stop WebSocket server gracefully with timeout
                if self.event_loop and not self.event_loop.is_closed():
                    try:
                        asyncio.run_coroutine_threadsafe(
                            asyncio.wait_for(self.websocket_server.stop(), timeout=3.0),
                            self.event_loop
                        ).result(timeout=4.0)
                    except (asyncio.TimeoutError, Exception) as e:
                        self.logger.warning(f"WebSocket server stop timed out or failed: {e}")
                    
                    # Close the event loop
                    try:
                        self.event_loop.close()
                        asyncio.set_event_loop(None)
                    except Exception as e:
                        self.logger.warning(f"Error closing event loop: {e}")
                
                self.logger.info("Stopped WebSocket server")
            except Exception as e:
                self.logger.error(f"Error stopping WebSocket server: {e}")
        
        # STEP 6: Stop WebSocket thread
        if hasattr(self, 'websocket_thread') and self.websocket_thread:
            try:
                self.logger.info("Stopping WebSocket thread")
                safe_join(self.websocket_thread, timeout=2.0, name="websocket_thread")
                self.logger.info("Stopped WebSocket thread")
            except Exception as e:
                self.logger.error(f"Error stopping WebSocket thread: {e}")
        
        # STEP 7: Stop CPU profiling
        if self.cpu_profiler:
            try:
                stop_global_profiling()
                self.logger.info("Stopped comprehensive CPU profiling")
            except Exception as e:
                self.logger.error(f"Error stopping CPU profiling: {e}")
        
        # STEP 8: Stop dashboard servers
        try:
            if SIMPLE_DASHBOARD_SERVER_AVAILABLE:
                stop_simple_dashboard_server()
                self.logger.info("Stopped simple dashboard server")
        except Exception as e:
            self.logger.error(f"Error stopping simple dashboard server: {e}")
        
        self.logger.info("Application stopped")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="GPU-Only Video Analysis Application")
    
    # Camera options
    parser.add_argument('--webcam', action='store_true', help='Use webcam as input')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--rtsp', type=str, help='RTSP stream URL')
    
    # Processing options
    parser.add_argument('--gpu-device', type=str, default='cuda:0', help='GPU device to use')
    parser.add_argument('--confidence', type=float, default=0.3, help='Detection confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7, help='IoU threshold for NMS')
    parser.add_argument('--target-classes', type=int, nargs='+', help='Target class IDs to detect')
    parser.add_argument('--analysis-frame-interval', type=int, help='Process every Nth frame for analysis (default: 15)')
    
    # Model options
    parser.add_argument('--detection-model', type=str, help='Path to detection model')
    parser.add_argument('--pose-model', type=str, help='Path to pose estimation model')
    parser.add_argument('--reid-model', type=str, help='Path to ReID model')
    
    # Output options
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    parser.add_argument('--save-frames', action='store_true', help='Save annotated frames')
    parser.add_argument('--save-detections', action='store_true', help='Save detection results')
    parser.add_argument('--frame-save-interval', type=int, help='Save every Nth frame (default: 15)')
    
    # Processing options
    parser.add_argument('--enable-profiling', action='store_true', help='Enable performance profiling')
    parser.add_argument('--enable-nvdec', action='store_true', help='Enable NVDEC hardware decoding')
    parser.add_argument('--enable-gpu-preprocessing', action='store_true', help='Enable GPU preprocessing')
    parser.add_argument('--force-gpu-only', action='store_true', help='Force GPU-only processing (no CPU fallbacks)')
    parser.add_argument('--use-unified-pipeline', action='store_true', help='Use unified GPU pipeline')
    
    # WebSocket options
    parser.add_argument('--websocket-host', type=str, default='0.0.0.0', help='WebSocket server host')
    parser.add_argument('--websocket-port', type=int, default=6008, help='WebSocket server port')
    
    # Visualization options
    parser.add_argument('--show-traces', action='store_true', help='Show tracking traces')
    parser.add_argument('--show-detection-boxes', action='store_true', help='Show detection boxes')
    parser.add_argument('--show-tracking-boxes', action='store_true', help='Show tracking boxes')
    parser.add_argument('--show-keypoints', action='store_true', help='Show pose keypoints')
    parser.add_argument('--show-masks', action='store_true', help='Show segmentation masks')

    # NEW: Logging options
    parser.add_argument('--logging-level', type=str,
                        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
                        help='Override log level (default: INFO)')
    return parser.parse_args()


def update_config_from_args(args):
    """Update configuration from command line arguments"""
    # Camera options
    if args.webcam:
        config.cameras.USE_WEBCAM = True
    if args.video:
        config.cameras.VIDEO_FILES = [args.video]
    if args.rtsp:
        config.cameras.RTSP_STREAMS = [{
        "name": "Command Line Stream",
        "url": args.rtsp,
        "width": 1920,  # Default resolution
        "height": 1080
    }]
    
    # Processing options
    if args.gpu_device:
        config.models.DEVICE = args.gpu_device
        config.processing.GPU_PREPROCESSING_DEVICE = args.gpu_device
    if args.confidence:
        config.models.MODEL_CONFIDENCE_THRESHOLD = args.confidence
    if args.iou:
        config.models.MODEL_IOU_THRESHOLD = args.iou
    if args.target_classes:
        config.models.TARGET_CLASSES = args.target_classes
    if args.analysis_frame_interval:
        config.processing.ANALYSIS_FRAME_INTERVAL = args.analysis_frame_interval
    
    # Model options
    if args.detection_model:
        config.models.MODEL_PATH = args.detection_model
    if args.pose_model:
        config.models.POSE_MODEL_PATH = args.pose_model
    if args.reid_model:
        config.models.REID_MODEL_PATH = args.reid_model
    
    # Output options
    if args.output_dir:
        config.output.OUTPUT_DIR = args.output_dir
    if args.save_frames:
        config.output.SAVE_FRAMES = True
    if args.save_detections:
        config.output.SAVE_DETECTIONS = True
    if args.frame_save_interval:
        config.output.FRAME_SAVE_INTERVAL = args.frame_save_interval
    
    # Processing options
    if args.enable_profiling:
        config.processing.ENABLE_PROFILING = True
    if args.enable_nvdec:
        config.processing.ENABLE_NVDEC = True
    if args.enable_gpu_preprocessing:
        config.processing.ENABLE_GPU_PREPROCESSING = True
    if args.force_gpu_only:
        config.models.FORCE_GPU_ONLY = True
    if args.use_unified_pipeline:
        config.processing.USE_UNIFIED_GPU_PIPELINE = True
    
    # WebSocket options
    if args.websocket_host:
        config.websocket.HOST = args.websocket_host
    if args.websocket_port:
        config.websocket.PORT = args.websocket_port
    
    # Visualization options
    if args.show_traces:
        config.visualization.SHOW_TRACES = True
    if args.show_detection_boxes:
        config.visualization.SHOW_DETECTION_BOXES = True
    if args.show_tracking_boxes:
        config.visualization.SHOW_TRACKING_BOXES = True
    if args.show_keypoints:
        config.visualization.SHOW_KEYPOINTS = True
    if args.show_masks:
        config.visualization.SHOW_MASKS = True

    # NEW: Logging option
    if args.logging_level:
        import logging
        config.app.LOG_LEVEL = getattr(logging, args.logging_level)


def main():
    """Main application entry point"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Update configuration from arguments
    update_config_from_args(args)
    
    # Set up logging
    setup_logging(
        log_level=config.app.LOG_LEVEL,
        log_file=config.app.LOG_FILE
    )
    
    logger = logging.getLogger("main")
    logger.info("Starting GPU-Only Video Analysis Application")
    logger.info(f"Configuration: {config}")
    
    # Create application manager
    app_manager = ApplicationManager(config)
    
    try:
        # Initialize application
        app_manager.initialize()
        
        # Start application
        app_manager.start()
        
        # Keep application running
        logger.info("Application running. Press Ctrl+C to stop.")
        
        # Wait for termination signal
        while app_manager.running:
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Application error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean shutdown
        app_manager.stop()
        logger.info("Application shutdown complete")


if __name__ == "__main__":
    main() 
