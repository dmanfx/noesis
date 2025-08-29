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

import sys
print("🚀 main.py script started!", flush=True)
sys.stdout.flush()
import argparse
print("✅ argparse imported", flush=True)
sys.stdout.flush()
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
from models import DetectionResult, TrackingResult, AnalysisFrame, convert_numpy_types
from deepstream_video_pipeline import create_deepstream_video_processor
# from gpu_pipeline import UnifiedGPUPipeline, cleanup_all_gpu_resources  # DEPRECATED
from utils import RateLimitedLogger

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

# Import specific functions from root utils (keep RateLimitedLogger from package)
setup_logging = root_utils.setup_logging
ensure_dir = root_utils.ensure_dir
get_timestamp = root_utils.get_timestamp
PerformanceMonitor = root_utils.PerformanceMonitor
encode_frame = root_utils.encode_frame

from utils.profiler import profile_step, aggregate_stats
from utils.cpu_profiler import start_global_profiling, stop_global_profiling, get_global_profiler, profile_function
from utils.interrupt import safe_join, safe_process_join
from visualization import VisualizationManager
from websocket_server import WebSocketServer

# Ensure project root is in path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Database integration is currently not active; DatabaseManager import removed

# Global interrupt counter for clean shutdown handling
INTERRUPT_COUNT = 0
MAX_SHUTDOWN_TIME = 15  # Maximum time to wait for graceful shutdown



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
        self.multi_stream_processor = None  # Single multi-stream processor
        self.analysis_frame_queue = multiprocessing.Queue(maxsize=100)
        self.streaming_frame_queue = queue.Queue(maxsize=100)
        # Cache for precomputed WebSocket headers per camera id
        self._ws_header_cache: Dict[str, bytes] = {}
        
        # Initialize components
        self.websocket_server = None
        self.websocket_loop = None
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
        
        # Add immediate print for visibility
        print(f"\n🛑 Signal {sig} received (interrupt #{INTERRUPT_COUNT})")
        
        if INTERRUPT_COUNT == 1:
            self.logger.info(f"Received signal {sig}, starting graceful shutdown...")
            print("🔄 Starting graceful shutdown...")
            # Start graceful shutdown immediately (not in background thread)
            self.stop()
        elif INTERRUPT_COUNT >= 2:
            self.logger.warning("Second interrupt received, forcing immediate exit")
            print("💥 Second interrupt - forcing immediate exit!")
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
        print("🔧 ApplicationManager.initialize() called", flush=True)
        sys.stdout.flush()
        self.logger.info("Initializing application components")
        print("✅ Logger initialized", flush=True)
        sys.stdout.flush()

        try:
            print("🔧 Starting try block in initialize()", flush=True)
            sys.stdout.flush()
            # Create event loop for async tasks
            print("🔧 About to create event loop...", flush=True)
            sys.stdout.flush()
            self.event_loop = asyncio.new_event_loop()
            print("✅ Event loop created", flush=True)
            sys.stdout.flush()

            # Test accessing config properties
            print("🔧 About to access config.websocket properties...", flush=True)
            sys.stdout.flush()
            host = self.config.websocket.HOST
            port = self.config.websocket.PORT
            trail_state = self.config.visualization.TRAIL_VISUALIZATION_ENABLED
            print(f"✅ Config properties accessed: host={host}, port={port}, trail_state={trail_state}", flush=True)
            sys.stdout.flush()

            # Initialize WebSocket server
            self.websocket_server = WebSocketServer(
                host=host,
                port=port,
                event_loop=None,  # will run on thread's own loop
                stats_callback=self._get_stats,
                toggle_callback=self._handle_toggle_update,
                initial_trail_state=trail_state
            )

            # Add detection configuration callbacks
            self.websocket_server.detection_config_callback = self._handle_detection_config_update
            self.websocket_server.detection_toggle_callback = self._handle_detection_toggle

            # Add detection config getter for initial sync
            self.websocket_server.detection_config_getter = self._get_detection_config

            # Debug: Write to stderr instead of stdout
            import os
            os.write(2, b"DEBUG: About to load camera sources...\n")

            # Load camera sources
            os.write(2, b"DEBUG: Loading camera sources...\n")
            self._load_camera_sources()
            os.write(2, b"DEBUG: Camera sources loaded\n")

            # Initialize output directory
            ensure_dir(self.config.output.OUTPUT_DIR)

            self.logger.info("Application initialization complete")
            print("🎉 ApplicationManager.initialize() completed successfully")

        except Exception as e:
            print(f"❌ Exception in initialize(): {e}")
            import traceback
            traceback.print_exc()
            raise
    
    @profile_function("ApplicationManager.load_camera_sources")
    def _load_camera_sources(self):
        """Load camera sources from configuration"""
        self.logger.info("Loading camera sources")
        
        # Clear existing sources
        self.camera_sources = {}
        
        if self.config.cameras.USE_WEBCAM:
            # Add webcam as a source entry
            self.camera_sources["webcam"] = {
                "url": "v4l2:///dev/video0",  # Example for V4L2 webcam
                "name": "Webcam",
                "width": self.config.cameras.CAMERA_WIDTH,
                "height": self.config.cameras.CAMERA_HEIGHT,
                "enabled": True
            }
            self.logger.info("Added webcam as a source")
            
        # Load video files
        if self.config.cameras.VIDEO_FILES:
            for i, video_file in enumerate(self.config.cameras.VIDEO_FILES):
                if os.path.exists(video_file):
                    camera_id = f"video_{i}"
                    self.camera_sources[camera_id] = {
                        "url": f"file://{os.path.abspath(video_file)}",
                        "name": f"Video File {i}",
                        "width": self.config.cameras.CAMERA_WIDTH,
                        "height": self.config.cameras.CAMERA_HEIGHT,
                        "enabled": True
                    }
                    self.logger.info(f"Added video file as source: {video_file} with ID {camera_id}")
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
        print("🎯 ApplicationManager.start() called!")
        if self.running:
            print("⚠️ Application already running, returning")
            return

        print("🚀 Starting application components...")
        self.logger.info("Starting application")
        self.running = True
        self.stop_event.clear()

        try:
            # Start WebSocket server FIRST so frontend can connect while DS initializes
            self.logger.info("🚀 Starting WebSocket server...")
            self._start_websocket_server()
            self.logger.info("✅ WebSocket server startup initiated")

            # Start unified GPU pipeline
            self.logger.info("🚀 Starting unified GPU pipeline...")
            self._start_multi_stream_processor()
            self.logger.info("✅ Multi-stream processor started")

            # Now that processor exists, wire WebSocket server to it and start JPEG loop
            try:
                if self.websocket_server and self.multi_stream_processor:
                    self.multi_stream_processor.websocket_server = self.websocket_server
                    self.logger.info("✅ Provided WebSocket server to multi-stream processor (post-start)")
                    # Avoid starting duplicate JPEG processing loop
                    if not hasattr(self, 'jpeg_thread') or not getattr(self, 'jpeg_thread').is_alive():
                        self._start_jpeg_processing_loop()
            except Exception as e:
                self.logger.warning(f"Unable to attach WebSocket server to processor: {e}")

            # Start result processing (ALWAYS needed for WebSocket streaming)
            self.logger.info("✅ Starting result processing...")
            self._start_result_processing()
            self.logger.info("✅ Result processing started")

            self.logger.info("🎉 Application started successfully")
            print("🎉 Application started successfully")
            print("🌐 WebSocket server ready - frontend can now connect!")
            print(f"📡 Connect to: ws://{self.config.websocket.HOST}:{self.config.websocket.PORT}")
            
        except Exception as e:
            self.logger.error(f"❌ Application startup failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    @profile_function("ApplicationManager.start_multi_stream_processor")
    def _start_multi_stream_processor(self):
        """Start single multi-stream DeepStream processor"""
        self.logger.info("Starting multi-stream DeepStream processor")
        
        # Validate GPU-only configuration
        if not self.config.processing.ENABLE_DEEPSTREAM:
            raise RuntimeError("GPU-only mode: DeepStream must be enabled")
        if not self.config.processing.ENABLE_GPU_PREPROCESSING:
            raise RuntimeError("GPU-only mode: GPU preprocessing must be enabled")
        if not self.config.models.FORCE_GPU_ONLY:
            raise RuntimeError("GPU-only mode: GPU-only inference must be enabled")
        
        # Prepare sources list from all enabled camera sources
        enabled_sources = [
            source_config for source_config in self.camera_sources.values()
            if isinstance(source_config, dict) and source_config.get("enabled", True)
        ]
        
        if not enabled_sources:
            raise RuntimeError("No enabled camera sources found")
        
        self.logger.info(f"🎥 Creating single multi-stream DeepStream processor for {len(enabled_sources)} sources")
        
        try:
            # Create single multi-stream processor
            processor = create_deepstream_video_processor(
                sources=enabled_sources,
                config=self.config
            )
            
            # Start the processor
            self.logger.info("🚀 Starting multi-stream DeepStream processor...")
            if not processor.start():
                raise RuntimeError("Failed to start multi-stream DeepStream processor")
            
            # Store single processor (not per-camera)
            self.multi_stream_processor = processor
            self.logger.info(f"✅ Multi-stream DeepStream processor started successfully with {len(enabled_sources)} streams")
            
            # Give processor a moment to initialize
            time.sleep(1.0)
            self.logger.info("Multi-stream processor initialization complete")
            
        except Exception as e:
            self.logger.error(f"❌ Error starting multi-stream processor: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Multi-stream processor startup failed: {e}")
    
    @profile_function("ApplicationManager.start_websocket_server")
    def _start_websocket_server(self):
        """Start WebSocket server"""
        if not self.websocket_server:
            self.logger.warning("WebSocket server not initialized")
            return
            
        # Start server in a background thread
        def run_websocket_server():
            print("🎯 WebSocket server thread function called!")
            try:
                # Create new event loop for this thread (don't share with main thread)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                print("🔧 WebSocket thread event loop created")
                self.logger.info("WebSocket thread started, created new event loop")
                # Keep reference for shutdown
                self.websocket_loop = loop

                # Provide the event loop to the WebSocket server so broadcast_sync works
                try:
                    if self.websocket_server is not None:
                        self.websocket_server.event_loop = loop
                except Exception as e:
                    self.logger.warning(f"Unable to set WebSocket server event loop: {e}")

                # Check if websocket server exists
                if self.websocket_server is None:
                    print("⚠️ WebSocket server is None, skipping startup")
                    return

                # Run server
                self.logger.info("Starting WebSocket server...")
                loop.run_until_complete(self.websocket_server.start())
                self.logger.info("WebSocket server started successfully")

                # Run event loop to handle connections
                self.logger.info("Starting WebSocket event loop...")
                try:
                    loop.run_forever()
                except KeyboardInterrupt:
                    self.logger.info("WebSocket server received keyboard interrupt")
                finally:
                    # Clean up the event loop
                    loop.close()

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
            print("🔧 About to start WebSocket server thread...")
            self.websocket_thread.start()
            print("✅ WebSocket server thread started")
            self.logger.info(f"Started WebSocket server thread on {self.config.websocket.HOST}:{self.config.websocket.PORT}")

            # Give the thread a moment to start
            print("⏳ Waiting for WebSocket thread to initialize...")
            time.sleep(0.5)

            if self.websocket_thread.is_alive():
                print("✅ WebSocket server thread is alive")
                self.logger.info("✅ WebSocket server thread is running")

                # Wait a bit more for the websocket server to actually start listening
                print("⏳ Waiting for WebSocket server to bind to port...")
                time.sleep(2.0)  # Give it time to bind
                print("✅ WebSocket server should be ready now")

                # Provide websocket_server instance to the multi-stream processor
                if hasattr(self, 'multi_stream_processor') and self.multi_stream_processor:
                    self.multi_stream_processor.websocket_server = self.websocket_server
                    self.logger.info("✅ WebSocket server instance provided to multi-stream processor")
                    print("✅ WebSocket server instance provided to multi-stream processor")

                    # Start JPEG processing loop for native DeepStream OSD mode
                    self._start_jpeg_processing_loop()
                else:
                    self.logger.warning("⚠️ Multi-stream processor not available to assign WebSocket server")
                    print("⚠️ Multi-stream processor not available to assign WebSocket server")
            else:
                print("❌ WebSocket server thread is NOT alive")
                self.logger.error("❌ WebSocket server thread failed to start")
                
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket server thread: {e}")
            import traceback
            traceback.print_exc()

    def _start_jpeg_processing_loop(self):
        """Start JPEG processing loop for native DeepStream OSD mode"""
        self.logger.info("Starting JPEG processing loop for native DeepStream OSD")
        
        def jpeg_processing_loop():
            """Process JPEG frames from multi-stream processor and broadcast them"""
            self.logger.info("JPEG processing loop started")
            
            # Get source information from multi-stream processor
            source_info = getattr(self.multi_stream_processor, 'source_info', {})
            if not source_info:
                self.logger.error("No source info available from multi-stream processor")
                return
            # Per-source counters to surface activity
            frames_sent = {sid: 0 for sid in source_info.keys()}
            last_info_log = time.time()
            self.logger.info(f"Processing JPEG data for {len(source_info)} sources: {list(source_info.keys())}")
            # Precompute headers for each camera clean name
            for sid, info in source_info.items():
                camera_id = info['clean_name']
                cam_id_bytes = camera_id.encode('utf-8')
                if len(cam_id_bytes) <= 255:
                    self._ws_header_cache[camera_id] = bytes([len(cam_id_bytes)]) + cam_id_bytes
                else:
                    self.logger.error(f"Camera ID too long to cache header: {camera_id}")
            
            heartbeat_t = time.time()
            while self.running:
                try:
                    # Log JPEG loop tick for debugging
                    #self.logger.debug("TRACE JPEG loop tick")
                    
                    # Heartbeat every 5 s
                    if time.time() - heartbeat_t > 5:
                        try:
                            qsizes = {sid:q.qsize() for sid,q in self.multi_stream_processor.jpeg_queues.items()}
                            self.logger.debug(f"JPEG loop heartbeat – queue sizes: {qsizes}")
                        except Exception:
                            pass
                        heartbeat_t = time.time()
                        # Process each source
                    for source_id, info in source_info.items():
                        if not self.running:
                            break
                            
                        # Read JPEG data from multi-stream processor
                        success, jpeg_bytes = self.multi_stream_processor.read_encoded_jpeg(source_id, timeout=0.1)
                        
                        if success and jpeg_bytes and self.websocket_server:
                            # Use clean camera name for frontend
                            camera_id = info['clean_name']
                            # Get or compute header
                            header = self._ws_header_cache.get(camera_id)
                            if header is None:
                                cam_id_bytes = camera_id.encode('utf-8')
                                if len(cam_id_bytes) <= 255:
                                    header = bytes([len(cam_id_bytes)]) + cam_id_bytes
                                    self._ws_header_cache[camera_id] = header
                                else:
                                    self.logger.error(f"Camera ID too long: {len(cam_id_bytes)} bytes for {camera_id}")
                                    continue
                            # Send precomputed header + payload
                            msg = header + jpeg_bytes
                            self.websocket_server.broadcast_sync(msg)
                            # Count frames per source and occasionally surface INFO logs
                            try:
                                frames_sent[source_id] += 1
                            except Exception:
                                frames_sent[source_id] = 1
                            self.rate_limited_logger.debug(f"Broadcast JPEG frame for {camera_id}: {len(jpeg_bytes)} bytes")

                    # Periodic INFO summary: frames sent and JPEG queue sizes
                    if time.time() - last_info_log >= 5.0:
                        try:
                            qsizes = {sid: q.qsize() for sid, q in self.multi_stream_processor.jpeg_queues.items()}
                        except Exception:
                            qsizes = {}
                        summary_counts = {source_id: frames_sent.get(source_id, 0) for source_id in source_info.keys()}
                        self.logger.info(f"📤 JPEG broadcast summary (last 5s): sent={summary_counts} | queues={qsizes} | clients={len(getattr(self.websocket_server, 'connected_clients', []))}")
                        # Reset counters for next window
                        frames_sent = {sid: 0 for sid in source_info.keys()}
                        last_info_log = time.time()
                
                except Exception as e:
                    self.logger.error(f"Error in JPEG processing loop: {e}")
                    if not self.running:
                        break
                    time.sleep(0.1)  # Brief pause on error

                # Rate limit the loop to ~30 FPS instead of spinning at 100% CPU
                time.sleep(0.033)

            self.logger.info("JPEG processing loop stopped")
        
        # Start JPEG processing thread
        self.jpeg_thread = threading.Thread(
            target=jpeg_processing_loop,
            name="JPEGProcessingThread", 
            daemon=True
        )
        self.jpeg_thread.start()
        self.logger.info("✅ JPEG processing thread started")

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
            
            # Encode frame as JPEG (single operation, remove duplication)
            jpeg_quality = int(getattr(self.config.visualization, 'JPEG_QUALITY', 85))
            if self.perf_monitor and self.config.processing.ENABLE_PROFILING:
                with profile_step("frame_encoding", self.perf_monitor):
                    jpeg_data = encode_frame(annotated_frame, quality=jpeg_quality)
            else:
                jpeg_data = encode_frame(annotated_frame, quality=jpeg_quality)
            if not jpeg_data:
                self.logger.warning("JPEG encoding returned no data; skipping frame broadcast")
                return
            
            # Create binary frame message for WebSocket broadcasting
            # Format: [camera_id_length(1 byte)][camera_id][jpeg_data]
            camera_id_bytes = analysis_frame.camera_id.encode('utf-8')
            camera_id_length = len(camera_id_bytes)
            
            if camera_id_length > 255:
                self.logger.error(f"Camera ID too long: {camera_id_length} bytes")
                return
                
            # Construct binary message using cached header per camera
            if analysis_frame.camera_id not in self._ws_header_cache:
                header = bytes([camera_id_length]) + camera_id_bytes
                self._ws_header_cache[analysis_frame.camera_id] = header
            header = self._ws_header_cache[analysis_frame.camera_id]
            self.logger.debug(f"Camera ID length: {camera_id_length}, JPEG length: {len(jpeg_data)}")
            binary_message = header + jpeg_data
            #self.logger.debug(f"Binary message type: {type(binary_message)}, length: {len(binary_message)}")
            
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
                    'cameras_active': len([s for s in self.config.cameras.RTSP_STREAMS if s.get('enabled', True)]),
                    'processors_active': 1 if hasattr(self, 'multi_stream_processor') and self.multi_stream_processor else 0
                },
                'cameras': {}
            }

            # Get stats from multi-stream processor
            if hasattr(self, 'multi_stream_processor') and self.multi_stream_processor:
                try:
                    comprehensive_stats = self.multi_stream_processor.get_stats()
                    self.logger.debug(f"📊 Multi-stream processor stats: {comprehensive_stats}")
                    
                    if comprehensive_stats and 'tracking' in comprehensive_stats:
                        # comprehensive_stats['tracking'] is now {source_id: {occupancy, active_tracks, transitions}}
                        per_stream_tracking = comprehensive_stats['tracking']
                        
                        # Map source_id back to camera names for UI
                        source_info = getattr(self.multi_stream_processor, 'source_info', {})
                        
                        for source_id, stream_tracking_data in per_stream_tracking.items():
                            if source_id in source_info:
                                camera_name = source_info[source_id]['clean_name']  # e.g., 'living-room', 'kitchen', 'family-room'
                                
                                # Create camera stats for this stream
                                camera_stats = {
                                    'fps': comprehensive_stats.get('fps', 0) / len(per_stream_tracking) if per_stream_tracking else 0,
                                    'frames_processed': comprehensive_stats.get('frames_processed', 0),
                                    'status': 'running' if comprehensive_stats.get('running', False) else 'stopped',
                                    'tracking': stream_tracking_data
                                }
                                
                                stats['cameras'][camera_name] = camera_stats
                                self.logger.debug(f"📊 Camera {camera_name} (source_id={source_id}) tracking data: {stream_tracking_data}")
                            else:
                                self.logger.warning(f"⚠️ Unknown source_id {source_id} in tracking data")
                                
                except Exception as e:
                    self.logger.error(f"Error getting multi-stream processor stats: {e}")
                    # Add empty camera stats for enabled streams
                    for i, stream_config in enumerate(self.config.cameras.RTSP_STREAMS):
                        if stream_config.get("enabled", True):
                            camera_name = stream_config.get('name', f'Camera_{i}')
                            if 'Living Room' in camera_name:
                                clean_name = 'living-room'
                            elif 'Kitchen' in camera_name:
                                clean_name = 'kitchen'
                            elif 'Family Room' in camera_name:
                                clean_name = 'family-room'
                            else:
                                clean_name = camera_name.lower().replace(' ', '-').replace('_', '-')
                            
                            stats['cameras'][clean_name] = {
                                'fps': 0,
                                'frames_processed': 0,
                                'status': 'error',
                                'tracking': {'occupancy': {}, 'active_tracks': [], 'transitions': []}
                            }

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
        if toggle_name == "trail_visualization_enabled":
            if hasattr(self, 'multi_stream_processor') and self.multi_stream_processor:
                self.multi_stream_processor.set_trail_visualization(enabled)
        elif hasattr(self.config.visualization, toggle_name.upper()):
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
                if hasattr(self, 'multi_stream_processor') and self.multi_stream_processor:
                    self.multi_stream_processor.update_confidence_threshold(new_threshold)
                    self.logger.info("Updated confidence threshold for multi-stream processor")
            
            if 'iou_threshold' in config_data:
                new_iou = float(config_data['iou_threshold'])
                self.config.models.MODEL_IOU_THRESHOLD = new_iou
                self.logger.info(f"Updated IOU threshold to: {new_iou}")
                
                # Update DeepStream pipeline in real-time
                if hasattr(self, 'multi_stream_processor') and self.multi_stream_processor:
                    self.multi_stream_processor.update_iou_threshold(new_iou)
                    self.logger.info("Updated IOU threshold for multi-stream processor")
            
            if 'detection_enabled' in config_data:
                enabled = bool(config_data['detection_enabled'])
                self.logger.info(f"Detection enabled: {enabled}")
                
                # Update DeepStream pipeline in real-time
                if hasattr(self, 'multi_stream_processor') and self.multi_stream_processor:
                    self.multi_stream_processor.set_detection_enabled(enabled)
                    self.logger.info(f"Updated detection enabled for multi-stream processor: {enabled}")
            
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
                if hasattr(self, 'multi_stream_processor') and self.multi_stream_processor:
                    if enabled:
                        # Add classes to current target classes
                        current_classes = getattr(self.config.models, 'TARGET_CLASSES', [])
                        new_classes = list(set(current_classes + target_classes))
                        self.config.models.TARGET_CLASSES = new_classes
                        self.multi_stream_processor.update_target_classes(new_classes)
                        self.logger.info(f"Added classes {target_classes} to multi-stream processor")
                    else:
                        # Remove classes from current target classes
                        current_classes = getattr(self.config.models, 'TARGET_CLASSES', [])
                        new_classes = [c for c in current_classes if c not in target_classes]
                        self.config.models.TARGET_CLASSES = new_classes
                        self.multi_stream_processor.update_target_classes(new_classes)
                        self.logger.info(f"Removed classes {target_classes} from multi-stream processor")
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
                # Clean up multi-stream processor GPU resources
                if hasattr(self, 'multi_stream_processor') and self.multi_stream_processor:
                    try:
                        # DeepStream handles its own GPU cleanup internally
                        self.logger.info("Early cleanup: Multi-stream processor GPU resources")
                    except Exception as e:
                        self.logger.warning(f"Error in early GPU cleanup for multi-stream processor: {e}")
                
                # Force cleanup all GPU resources via consolidated pipeline
                # cleanup_all_gpu_resources() # This function is deprecated
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
        
        # STEP 3: Stop multi-stream processor (after GPU cleanup)
        if hasattr(self, 'multi_stream_processor') and self.multi_stream_processor:
            self.logger.info("Stopping multi-stream processor")
            try:
                self.multi_stream_processor.stop()
                self.multi_stream_processor = None
                self.logger.info("Stopped multi-stream processor")
            except Exception as e:
                self.logger.error(f"Error stopping multi-stream processor: {e}")
        
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
                # If websocket server thread loop exists, stop via that loop
                if hasattr(self, 'websocket_loop') and self.websocket_loop:
                    try:
                        fut = asyncio.run_coroutine_threadsafe(self.websocket_server.stop(), self.websocket_loop)
                        fut.result(timeout=4.0)
                    except Exception as e:
                        self.logger.warning(f"WebSocket server stop via thread loop failed/timed out: {e}")
                    try:
                        self.websocket_loop.call_soon_threadsafe(self.websocket_loop.stop)
                    except Exception:
                        pass
                else:
                    # Fallback: mark not running
                    try:
                        self.websocket_server.running = False
                    except Exception:
                        pass
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
        config.cameras.RTSP_STREAMS.append({
            "name": "Command Line Stream",
            "url": args.rtsp,
            "width": 1920,  # Default resolution
            "height": 1080,
            "enabled": True
        })
    
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
    # DEPRECATED: Unified pipeline option removed - DeepStream-only now
    # if args.use_unified_pipeline:
    #     config.processing.USE_UNIFIED_GPU_PIPELINE = True
    
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
    print("🔧 Starting main() function...")
    # Parse command line arguments
    args = parse_arguments()
    print("✅ Arguments parsed")

    # Update configuration from arguments
    update_config_from_args(args)
    print("✅ Configuration updated")

    # Set up logging
    setup_logging(
        log_level=config.app.LOG_LEVEL,
        log_file=config.app.LOG_FILE
    )
    print("✅ Logging setup complete")

    logger = logging.getLogger("main")
    print("🔧 Logger created")
    logger.info("Starting GPU-Only Video Analysis Application")
    logger.info(f"Configuration: {config}")

    print("🔧 Creating application manager...")
    # Create application manager
    app_manager = ApplicationManager(config)
    print("✅ Application manager created")
    
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
