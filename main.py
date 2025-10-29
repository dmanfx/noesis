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

# Set GStreamer debug BEFORE any other imports
import os
# Respect pre-set GST_DEBUG if provided; default to *:4 otherwise
os.environ['GST_DEBUG'] = os.environ.get('GST_DEBUG', '*:4')
os.environ['GST_DEBUG_NO_COLOR'] = '1'  # Disable colored output
os.environ['no_proxy'] = '*'

# Configure NVIDIA DeepStream/nvinfer logging to reduce INFO noise
os.environ['NVDS_DEBUG_LEVEL'] = '0'  # Minimal DeepStream logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow/TensorRT INFO
os.environ['TRT_LOGGER_LEVEL'] = '2'  # Suppress TensorRT logger INFO

import argparse
print("✅ argparse imported", flush=True)
sys.stdout.flush()
import asyncio
import http.client
import logging
import multiprocessing
import subprocess

from concurrent.futures import Future, ThreadPoolExecutor
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

from config import AppConfig, config, load_ma_config
from geometry.depth_source import DepthResult, DepthSummary, MapAnythingDepthSource
from geometry.depth_publisher import DepthDiagnosticsPublisher, DiagnosticsConfig
from models import DetectionResult, TrackingResult, AnalysisFrame, convert_numpy_types
# DS8 adapter for GStreamer-based pipeline (no DS7 instantiation)
from noesis.adapters.ds8_adapter import DS8Adapter
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

# Optional Menon config backup integration
try:
    from scripts.backup_configs import backup_loop as menon_backup_loop  # type: ignore
except Exception:  # noqa: BLE001
    menon_backup_loop = None

from utils.profiler import profile_step, aggregate_stats
from utils.cpu_profiler import start_global_profiling, stop_global_profiling, get_global_profiler, profile_function
from utils.interrupt import safe_join, safe_process_join
from visualization import VisualizationManager
from websocket_server import WebSocketServer
from calibration_bundle import (
    load_intrinsics,
    load_alignment,
    load_extrinsics,
    assemble_calibration_bundle,
    save_extrinsics,
    save_alignment,
)
from geometry.transform import pixel_to_world as pixel_to_world_aligned, build_align_matrix
from pixel_to_world import K_from_intrinsics, E_to_world_and_R, ray_from_pixel, intersect_floor, bbox_bottom_center

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
        self.multi_stream_processor = None  # Single multi-stream processor
        try:
            self.analysis_frame_queue = multiprocessing.Queue(maxsize=100)
        except Exception:
            # Fallback in restricted environments (e.g., no semaphores)
            self.analysis_frame_queue = queue.Queue(maxsize=100)
        self.streaming_frame_queue = queue.Queue(maxsize=100)
        # Cache for precomputed WebSocket headers per camera id
        self._ws_header_cache: Dict[str, bytes] = {}
        
        # Initialize components
        self.websocket_server = None
        self.websocket_loop = None
        self.visualization_manager = VisualizationManager()
        self._backup_task: Optional[asyncio.Task] = None
        # Calibration state
        self.calibration_bundle = None
        self._intrinsics_models = None
        self._calib_paths = {}
        # MapAnything microservice process handle
        self._mapanything_process: Optional[subprocess.Popen[str]] = None
        # Depth inference integration
        self.depth_source = MapAnythingDepthSource()
        self._depth_executor = ThreadPoolExecutor(max_workers=2)
        self._depth_warmup_executor = ThreadPoolExecutor(max_workers=1)
        self._depth_warmup_future: Optional[Future] = None
        self._pending_depth_futures: Dict[str, Future] = {}
        self._latest_depth_results: Dict[str, DepthResult] = {}
        self._camera_room_map: Dict[str, str] = {}
        self._last_depth_publish: Dict[str, float] = {}
        self._last_depth_summary: Dict[str, DepthSummary] = {}
        self._depth_publisher = self._create_depth_publisher()
        
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
        # Integrations
        self._occupancy_publisher = None
    
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

            # Load calibration & wire WebSocket RPCs
            try:
                self._load_calibration()
                if self.websocket_server:
                    self.websocket_server.calibration_getter = self._get_calibration_bundle
                    self.websocket_server.pixel_to_world_handler = self._pixel_to_world_rpc
                    self.websocket_server.set_extrinsics_handler = self._set_extrinsics_rpc
                    self.websocket_server.set_align_handler = self._set_align_rpc
                    if self.depth_source and self.websocket_server:
                        self.websocket_server.ma_depth_provider = self.depth_source.load_latest_depth
                        self.websocket_server.floorplan_provider = (
                            lambda cam=None, max_age=60.0, grid_res=0.5, max_extent=20.0, cache_only=False, **_:
                                self.depth_source.generate_topdown_floorplan(
                                    str(cam) if cam else '',
                                    max_age_sec=max_age,
                                    grid_res_m=grid_res,
                                    max_extent_m=max_extent,
                                    cache_only=cache_only,
                                )
                        )
            except Exception as e:
                self.logger.warning(f"Calibration init failed: {e}")

            # Initialize output directory
            ensure_dir(self.config.output.OUTPUT_DIR)

            self.logger.info("Application initialization complete")
            print("🎉 ApplicationManager.initialize() completed successfully")

        except Exception as e:
            print(f"❌ Exception in initialize(): {e}")
            import traceback
            traceback.print_exc()
            raise
    
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
            # Launch MapAnything inference microservice
            self._start_mapanything_service()

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
                        # Start mosaic publisher if configured (maps mosaic to a single camera id)
                        if getattr(self.config.websocket, 'MOSAIC_BROADCAST', False):
                            self._start_mosaic_broadcast()
                        self._start_mapanything_scheduler()
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

            # Start Menon config backup loop if available
            if menon_backup_loop and self.event_loop:
                try:
                    self._backup_task = asyncio.ensure_future(menon_backup_loop())
                    self.logger.info("Menon config backup task scheduled")
                except Exception as exc:
                    self.logger.warning(f"Unable to start config backup task: {exc}")
            
        except Exception as e:
            self.logger.error(f"❌ Application startup failed: {e}")
            import traceback
            traceback.print_exc()
            self._terminate_mapanything_process()
            raise
    
    @profile_function("ApplicationManager.start_multi_stream_processor")
    def _start_multi_stream_processor(self):
        """Start single multi-stream video processor (DS8 by default)."""
        self.logger.info("Starting multi-stream video processor (DS8)")
        
        # Validate GPU-only configuration
        if not self.config.processing.ENABLE_DEEPSTREAM:
            raise RuntimeError("GPU-only mode: DeepStream must be enabled")
        if not self.config.processing.ENABLE_GPU_PREPROCESSING:
            raise RuntimeError("GPU-only mode: GPU preprocessing must be enabled")
        if not self.config.models.FORCE_GPU_ONLY:
            raise RuntimeError("GPU-only mode: GPU-only inference must be enabled")
        
        try:
            # Choose DS8 by default; no DS7 import/instantiation in this module
            if getattr(self.config.processing, 'USE_DS8', True):
                processor = DS8Adapter(config=self.config)
            else:
                raise RuntimeError("DS7 pipeline fallback disabled in this build; set processing.USE_DS8=True")

            source_count = len(getattr(processor, "source_info", {}) or {})
            if source_count == 0:
                raise RuntimeError("No enabled sources available for DS8Adapter pipeline")
            self.logger.info(f"🎥 Creating single multi-stream processor for {source_count} sources")

            # Start the processor
            self.logger.info("🚀 Starting multi-stream processor...")
            if not processor.start():
                raise RuntimeError("Failed to start multi-stream processor")

            # Store single processor (not per-camera)
            self.multi_stream_processor = processor
            self.logger.info(f"✅ Multi-stream processor started successfully with {source_count} streams")
            try:
                source_info = getattr(self.multi_stream_processor, 'source_info', None)
                if isinstance(source_info, dict):
                    for info in source_info.values():
                        clean_name = info.get('clean_name') or info.get('name')
                        if isinstance(clean_name, str):
                            self._camera_room_map[clean_name] = clean_name
            except Exception as exc:
                self.logger.debug(f"Unable to build camera-room mapping: {exc}")

            # Wire OccupancyPublisher if enabled
            try:
                if getattr(self.config.integrations, 'ENABLE_OCCUPANCY_PUBLISH', False):
                    self.logger.info("🔌 Initializing OccupancyPublisher (MQTT + Influx)")
                    try:
                        print(f"🔌 OccupancyPublisher enabled: MQTT {self.config.integrations.MQTT_HOST}:{self.config.integrations.MQTT_PORT} base={self.config.integrations.BASE_TOPIC} | Influx bucket={self.config.integrations.INFLUX_BUCKET_RAW}")
                    except Exception:
                        pass
                    from occupancy_publisher import OccupancyPublisher, OccupancyConfig
                    occ_cfg = OccupancyConfig(
                        enabled=True,
                        heartbeat_sec=int(self.config.integrations.HEARTBEAT_SEC),
                        base_topic=str(self.config.integrations.BASE_TOPIC),
                        status_topic=str(self.config.integrations.STATUS_TOPIC),
                        mqtt_host=str(self.config.integrations.MQTT_HOST),
                        mqtt_port=int(self.config.integrations.MQTT_PORT),
                        mqtt_username=str(self.config.integrations.MQTT_USERNAME),
                        mqtt_password=str(self.config.integrations.MQTT_PASSWORD),
                        mqtt_qos=int(self.config.integrations.MQTT_QOS),
                        mqtt_retain=bool(self.config.integrations.MQTT_RETAIN),
                        influx_url=str(self.config.integrations.INFLUX_URL),
                        influx_org=str(self.config.integrations.INFLUX_ORG),
                        influx_token=str(self.config.integrations.INFLUX_TOKEN),
                        influx_bucket_raw=str(self.config.integrations.INFLUX_BUCKET_RAW),
                    )
                    pub = OccupancyPublisher(occ_cfg, logger=logging.getLogger("OccupancyPublisher"))
                    self._occupancy_publisher = pub
                    # attach to processor so DS code can emit from probe
                    try:
                        setattr(self.multi_stream_processor, 'occupancy_publisher', pub)
                        self.logger.info("✅ Attached OccupancyPublisher to DeepStream processor")
                        try:
                            print("✅ OccupancyPublisher attached to DeepStream processor")
                        except Exception:
                            pass
                    except Exception as e:
                        self.logger.warning(f"Unable to attach OccupancyPublisher to processor: {e}")
                else:
                    self.logger.info("Occupancy publishing disabled via config")
            except Exception as e:
                self.logger.error(f"Error initializing OccupancyPublisher: {e}")
            
            # Give processor a moment to initialize
            time.sleep(1.0)
            self.logger.info("Multi-stream processor initialization complete")

            # Rebuild calibration now that DeepStream has canonical camera IDs
            try:
                self._load_calibration()
                if self.websocket_server and (self.calibration_bundle is not None):
                    self.websocket_server.broadcast_sync({'type': 'calibration-bundle', 'data': self.calibration_bundle})
                    self.logger.info("📡 Re-broadcast calibration-bundle with canonical camera IDs")
            except Exception as e:
                self.logger.warning(f"Calibration rebuild after DeepStream init failed: {e}")
            
        except Exception as e:
            self.logger.error(f"❌ Error starting multi-stream processor: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Multi-stream processor startup failed: {e}")

    def _start_mapanything_service(self) -> None:
        """Launch the MapAnything FastAPI microservice if not already running."""
        script_path = (project_root / "services" / "mapanything_svc" / "run.sh").resolve()
        if not script_path.exists():
            self.logger.warning("MapAnything run script missing at %s; skipping service startup", script_path)
            return

        if self._mapanything_process and self._mapanything_process.poll() is None:
            self.logger.info("MapAnything service already running (pid=%s)", self._mapanything_process.pid)
            return

        ma_cfg = load_ma_config()
        service_cfg = ma_cfg.get('service', {}) if isinstance(ma_cfg, dict) else {}

        def _parse(value: Any, default: str) -> str:
            raw = str(value) if value is not None else default
            return raw.split('#', 1)[0].strip()

        host = _parse(service_cfg.get('host'), '127.0.0.1')
        port_str = _parse(service_cfg.get('port'), '8001')
        try:
            port = int(port_str)
        except ValueError:
            port = 8001

        env = os.environ.copy()
        env.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

        self.logger.info("Starting MapAnything service via %s", script_path)
        process = subprocess.Popen(
            ["bash", str(script_path)],
            cwd=str(project_root),
            env=env,
        )
        self._mapanything_process = process

        try:
            self._wait_for_mapanything_ready(host, port)
        except Exception:
            self.logger.error("MapAnything service failed to report healthy; terminating process")
            self._terminate_mapanything_process()
            raise

    def _wait_for_mapanything_ready(self, host: str, port: int, timeout: float = 30.0) -> None:
        """Poll the service health endpoint until it responds or timeout expires."""
        deadline = time.time() + timeout
        delay = 0.5
        while time.time() < deadline:
            proc = self._mapanything_process
            if proc and proc.poll() is not None:
                raise RuntimeError("MapAnything process exited prematurely")
            try:
                conn = http.client.HTTPConnection(host, port, timeout=2.0)
                try:
                    conn.request("GET", "/health")
                    response = conn.getresponse()
                    if response.status == 200:
                        self.logger.info("MapAnything service healthy at %s:%s", host, port)
                        return
                finally:
                    conn.close()
            except Exception:
                pass
            time.sleep(delay)
            delay = min(delay * 2.0, 4.0)
        raise RuntimeError("Timed out waiting for MapAnything service health check")

    def _terminate_mapanything_process(self) -> None:
        """Terminate MapAnything service process if running."""
        proc = self._mapanything_process
        if not proc:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                self.logger.warning("MapAnything service did not terminate in time; killing")
                proc.kill()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self.logger.error("Unable to kill MapAnything service process cleanly")
        self._mapanything_process = None

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

                # Set up shutdown handling
                def shutdown_handler():
                    """Handle shutdown signal for this thread"""
                    self.logger.info("WebSocket thread received shutdown signal")
                    try:
                        # Stop the WebSocket server gracefully
                        loop.run_until_complete(self.websocket_server.stop())
                    except Exception as e:
                        self.logger.warning(f"Error stopping WebSocket server in thread: {e}")
                    finally:
                        # Stop the event loop
                        loop.stop()

                # Run event loop to handle connections
                self.logger.info("Starting WebSocket event loop...")
                try:
                    loop.run_forever()
                except KeyboardInterrupt:
                    self.logger.info("WebSocket server received keyboard interrupt")
                    shutdown_handler()
                finally:
                    # Ensure cleanup happens even if shutdown_handler wasn't called
                    try:
                        if not loop.is_closed():
                            loop.run_until_complete(self.websocket_server.stop())
                    except Exception as e:
                        self.logger.debug(f"WebSocket server already stopped or error during final cleanup: {e}")
                    finally:
                        # Clean up the event loop
                        if not loop.is_closed():
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

                # WebSocket server wiring to multi-stream processor will happen later in start() method
                # after the processor is created
            else:
                print("❌ WebSocket server thread is NOT alive")
                self.logger.error("❌ WebSocket server thread failed to start")
                
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket server thread: {e}")
            import traceback
            traceback.print_exc()

    def _start_mapanything_scheduler(self):
        """Background loop to drive MapAnything mono inference from latest JPEGs."""
        if not self.multi_stream_processor:
            self.logger.warning("MapAnything scheduler not started (no multi-stream processor)")
            return

        self.logger.info("Starting MapAnything mono scheduler loop")

        def loop():
            last_pub = self._last_depth_publish
            try:
                mono_interval = getattr(self.depth_source, 'mono_interval', 0.5)
            except Exception:
                mono_interval = 0.5

            dry_last_log: Dict[str, float] = {}
            while self.running and not self.stop_event.is_set():
                try:
                    # Snapshot source_info each tick to tolerate dynamic sources
                    source_info = getattr(self.multi_stream_processor, 'source_info', {}) or {}
                    now = time.time()
                    for source_id, info in source_info.items():
                        cam_id = str(info.get('clean_name') or info.get('name') or source_id)
                        # Respect per-camera rate gating
                        try:
                            if not self.depth_source.should_infer(cam_id, now):
                                continue
                        except Exception:
                            continue

                        ok, jpeg_bytes = self.multi_stream_processor.read_encoded_jpeg(source_id, timeout=0.05)
                        if not ok or not jpeg_bytes:
                            # Rate-limited dryness log per camera
                            t0 = dry_last_log.get(cam_id, 0.0)
                            if (now - t0) >= 5.0:
                                self.logger.debug(f"MDE scheduler: no frame available for {cam_id} (sid={source_id})")
                                dry_last_log[cam_id] = now
                            continue

                        # Decode JPEG -> BGR
                        try:
                            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                            frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if frame_bgr is None:
                                continue
                        except Exception:
                            continue

                        # Run mono inference
                        try:
                            result = self.depth_source.maybe_infer_mono(cam_id, frame_bgr, self.calibration_bundle, now)
                        except Exception as e:
                            self.logger.debug(f"MDE mono error for {cam_id}: {e}")
                            result = None
                        if result is None:
                            continue

                        # Cache latest result and summary
                        self._latest_depth_results[cam_id] = result
                        self._last_depth_summary[cam_id] = result.summary

                        # Throttle diagnostics to ~5s per camera
                        if (now - last_pub.get(cam_id, 0.0)) >= 5.0:
                            last_pub[cam_id] = now
                            room_id = self._camera_room_map.get(cam_id, cam_id)
                            if self._depth_publisher:
                                try:
                                    self._depth_publisher.publish_depth_summary(result, room_id)
                                except Exception as e:
                                    self.logger.debug(f"Depth publisher error: {e}")

                            summary_message = {
                                'type': 'ma_diagnostics',
                                'cam_id': cam_id,
                                'summary': {
                                    'median': result.summary.median,
                                    'p10': result.summary.p10,
                                    'p90': result.summary.p90,
                                    'conf_mean': result.summary.conf_mean,
                                    'valid_ratio': result.summary.valid_ratio,
                                    'sample_count': result.summary.sample_count,
                                    'method': 'mde' if result.summary.conf_mean >= getattr(self.depth_source, 'min_conf', 0.5) else 'floor'
                                },
                                'ts': result.ts_us
                            }
                            self._schedule_ws_broadcast(summary_message)
                except Exception as e:
                    self.logger.debug(f"MDE scheduler tick error: {e}")
                finally:
                    try:
                        time.sleep(max(0.05, mono_interval * 0.5))
                    except Exception:
                        time.sleep(0.1)

        t = threading.Thread(target=loop, name="MapAnythingScheduler", daemon=True)
        t.start()
        self.logger.info("✅ MapAnything mono scheduler loop started")

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
                            
                        # Read and coalesce to latest JPEG for this source (drain queue)
                        jpeg_bytes = None
                        try:
                            q = getattr(self.multi_stream_processor, 'jpeg_queues', {}).get(source_id)
                        except Exception:
                            q = None
                        if q is not None:
                            try:
                                # Blocking read for first item
                                jpeg_bytes = q.get(timeout=0.1)
                                # Drain any additional queued frames to keep only the latest
                                drained = 0
                                while True:
                                    try:
                                        more = q.get_nowait()
                                        jpeg_bytes = more
                                        drained += 1
                                    except queue.Empty:
                                        break
                                # Optional: debug drain count at low rate
                                # if drained > 0:
                                #     self.rate_limited_logger.debug(f"Drained {drained} frames for source {source_id}")
                            except queue.Empty:
                                jpeg_bytes = None

                        # If mosaic mode is enabled, skip per-camera WS broadcast
                        if getattr(self.config.websocket, 'MOSAIC_BROADCAST', False):
                            # Optionally, we could cache latest JPEGs here for reuse
                            continue

                        if jpeg_bytes and self.websocket_server:
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
                        self.logger.debug(f"📤 JPEG broadcast summary (last 5s): sent={summary_counts} | queues={qsizes} | clients={len(getattr(self.websocket_server, 'connected_clients', []))}")
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

    def _start_mosaic_broadcast(self):
        """Start a background thread that composites a mosaic from latest JPEGs and broadcasts it under a target camera id."""
        if not getattr(self.config.websocket, 'MOSAIC_BROADCAST', False):
            return

        target_cam = getattr(self.config.websocket, 'MOSAIC_TARGET_CAMERA', 'living-room')
        fps_limit = max(1, int(getattr(self.config.websocket, 'MAX_FPS', 10)))
        period = 1.0 / float(fps_limit)

        # Resolve sensor ordering and prepare cache for last-decoded frames
        try:
            source_info = getattr(self.multi_stream_processor, 'source_info', {}) or {}
            sensor_ids = list(source_info.keys())
        except Exception:
            sensor_ids = []

        last_frames: Dict[int, Any] = {}

        def _decode(b: bytes):
            import numpy as _np
            a = _np.frombuffer(b, dtype=_np.uint8)
            return cv2.imdecode(a, cv2.IMREAD_COLOR)

        def _mosaic_loop():
            next_tick = time.time()
            header = None
            if self.websocket_server:
                cam_id_bytes = target_cam.encode('utf-8')
                if len(cam_id_bytes) <= 255:
                    header = bytes([len(cam_id_bytes)]) + cam_id_bytes
            sent = 0
            t_start = time.time()
            last_log = t_start
            last_bytes = 0
            while self.running and not self.stop_event.is_set():
                ok = False
                data = None
                try:
                    if hasattr(self.multi_stream_processor, 'read_mosaic_jpeg'):
                        ok, data = self.multi_stream_processor.read_mosaic_jpeg(timeout=0.1)
                except Exception:
                    ok, data = False, None
                if ok and data and header and self.websocket_server:
                    self.websocket_server.broadcast_sync(header + data)
                    sent += 1
                    last_bytes = len(data)
                # Log every ~2s
                now = time.time()
                if (now - last_log) >= 2.0:
                    try:
                        qsz = getattr(self.multi_stream_processor, 'mosaic_queue', queue.Queue()).qsize()
                    except Exception:
                        qsz = -1
                    elapsed = max(1e-3, now - t_start)
                    fps = sent / elapsed
                    self.logger.debug(f"Mosaic feed: {fps:.1f} fps, last={last_bytes} bytes, q={qsz}")
                    last_log = now
                # Throttle to target FPS
                next_tick += period
                sleep_for = next_tick - time.time()
                if sleep_for > 0:
                    time.sleep(min(sleep_for, period))

        t = threading.Thread(target=_mosaic_loop, name="MosaicPublisher", daemon=True)
        t.start()
        self.logger.info("✅ Mosaic broadcast thread started (target=%s)", target_cam)

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
    
    # ---------------------- Calibration helpers ----------------------
    def _get_camera_ids(self) -> list:
        ids = []
        try:
            if self.multi_stream_processor and getattr(self.multi_stream_processor, 'source_info', None):
                for sid, info in self.multi_stream_processor.source_info.items():
                    ids.append(str(info.get('clean_name') or info.get('name') or f"camera-{sid}"))
                return ids
        except Exception:
            pass
        try:
            for i, stream in enumerate(self.config.cameras.RTSP_STREAMS):
                if stream.get('enabled', True):
                    name = stream.get('name', f'Camera {i+1}')
                    clean = name.lower().replace(' ', '-').replace('_', '-')
                    ids.append(clean)
        except Exception:
            pass
        return ids

    def _load_calibration(self) -> None:
        # Resolve paths
        try:
            root_dir = str(Path(__file__).parent)
        except Exception:
            root_dir = os.getcwd()
        intr_path = self.config.calibration.INTRINSICS_PATH
        if not os.path.isabs(intr_path):
            intr_path = os.path.join(root_dir, intr_path)
        align_path = self.config.calibration.PLY_ALIGNMENT_PATH
        if not os.path.isabs(align_path):
            align_path = os.path.join(root_dir, align_path)
        extr_path = self.config.calibration.CAMERA_CALIBRATION_PATH
        if not os.path.isabs(extr_path):
            extr_path = os.path.join(root_dir, extr_path)

        # Load components
        self._intrinsics_models = load_intrinsics(intr_path)
        align = load_alignment(align_path)
        extr = load_extrinsics(extr_path)

        cam_ids = self._get_camera_ids()
        model_map = dict(self.config.calibration.CAMERA_INTRINSICS_MODEL_MAP)
        camera_specs = dict(getattr(self.config.calibration, 'CAMERA_SPECS', {}) or {})
        self.calibration_bundle = assemble_calibration_bundle(
            cam_ids,
            self._intrinsics_models,
            model_map,
            extr,
            align,
            camera_specs
        )
        self._calib_paths = {'intrinsics': intr_path, 'alignment': align_path, 'extrinsics': extr_path}
        camera_keys = sorted(list((self.calibration_bundle.get('cameras') or {}).get('K', {}).keys()))
        self.logger.info(f"Calibration ready for cameras (K): {camera_keys}")
        if self.depth_source is not None:
            try:
                self.depth_source.calibration_bundle = self.calibration_bundle
            except Exception:
                pass

    def _get_calibration_bundle(self) -> dict:
        return self.calibration_bundle or {}

    def _schedule_ws_broadcast(self, message: Dict[str, Any]) -> None:
        if not self.websocket_server:
            return
        try:
            loop = self.websocket_server.event_loop or self.event_loop
            if loop is None:
                loop = asyncio.get_event_loop()
            asyncio.run_coroutine_threadsafe(self.websocket_server.broadcast(message), loop)
        except RuntimeError as exc:
            self.logger.debug(f"WebSocket broadcast failed (no loop): {exc}")
        except Exception as exc:
            self.logger.debug(f"Failed to schedule WebSocket broadcast: {exc}")

    def _pixel_to_world_rpc(self, req: dict) -> dict:
        try:
            cam_id = str(req.get('camId') or req.get('cameraId') or '')
            u = float(req.get('u'))
            v = float(req.get('v'))
        except Exception:
            return {'ok': False, 'error': 'invalid_args'}

        depth_raw = req.get('depth') if 'depth' in req else req.get('depth_m')
        depth_m = None
        if depth_raw is not None:
            try:
                depth_m = float(depth_raw)
            except Exception:
                return {'ok': False, 'error': 'invalid_depth'}

        calib = self.calibration_bundle or {}
        cameras_node = calib.get('cameras') or {}
        k_table = cameras_node.get('K') if isinstance(cameras_node, dict) else {}
        e_table = cameras_node.get('E') if isinstance(cameras_node, dict) else {}
        intr = k_table.get(cam_id) if isinstance(k_table, dict) else None
        E = e_table.get(cam_id) if isinstance(e_table, dict) else None

        # Legacy fallback: old schema {cam_id:{intrinsics:{}, extrinsics:{E}}}
        if intr is None or E is None:
            legacy_cam = cameras_node.get(cam_id) if isinstance(cameras_node, dict) else None
            if isinstance(legacy_cam, dict):
                if intr is None:
                    intr = legacy_cam.get('intrinsics')
                if E is None:
                    extr = legacy_cam.get('extrinsics') or {}
                    E = extr.get('E') if isinstance(extr, dict) else None

        align_node = calib.get('align')
        align = align_node if isinstance(align_node, dict) else {}
        depth_result = self._latest_depth_results.get(cam_id)
        conf_threshold = getattr(self.depth_source, 'min_conf', 0.5)
        if depth_result and depth_result.intrinsics is not None:
            K = np.array(depth_result.intrinsics, dtype=float)
        else:
            K = K_from_intrinsics(intr)
        if K is None or not isinstance(E, list) or len(E) != 16:
            return {'ok': False, 'error': 'calibration_missing'}

        try:
            E_matrix = np.array(E, dtype=float).reshape((4, 4), order='F')
            T_cam2world = np.linalg.inv(E_matrix)
        except Exception:
            return {'ok': False, 'error': 'bad_extrinsics'}

        depth_used = depth_m if depth_m is not None and depth_m > 0.0 else None
        conf_value = None
        method = 'floor'

        if depth_result and depth_result.depth.size > 0:
            h, w = depth_result.depth.shape[:2]
            x_idx = int(round(u))
            y_idx = int(round(v))
            if 0 <= x_idx < w and 0 <= y_idx < h:
                candidate_depth = float(depth_result.depth[y_idx, x_idx])
                candidate_conf = float(depth_result.conf[y_idx, x_idx])
                if candidate_depth > 0.0 and np.isfinite(candidate_depth) and candidate_conf >= conf_threshold:
                    depth_used = candidate_depth
                    conf_value = candidate_conf
                    method = 'mde'
                    # Override intrinsics with depth result if available
                    if depth_result.intrinsics is not None:
                        K = np.array(depth_result.intrinsics, dtype=float)

        world_point = None
        if depth_used is not None:
            try:
                world_point = pixel_to_world_aligned(u, v, depth_used, cam_id, K, T_cam2world, align)
            except Exception as exc:
                try:
                    self.logger.debug('pixel_to_world depth transform failed for %s: %s', cam_id, exc)
                except Exception:
                    pass
                world_point = None
                method = 'floor'
                conf_value = None

        if world_point is None:
            pose = E_to_world_and_R(E)
            if pose is None:
                return {'ok': False, 'error': 'bad_extrinsics'}
            Cw, Rwc = pose
            O, D = ray_from_pixel(u, v, K, Cw, Rwc)
            floor_y = float((align or {}).get('floor_y') or 0.0)
            hit = intersect_floor(O, D, floor_y)
            if hit is None:
                return {'ok': False, 'error': 'no_intersection'}
            align_matrix = build_align_matrix(align)
            p_world = np.array([hit[0], hit[1], hit[2], 1.0], dtype=float)
            world_point = (align_matrix @ p_world)[:3]
            method = 'floor'
            conf_value = conf_value if conf_value is not None else 0.0
        response = {
            'ok': True,
            'world': [float(world_point[0]), float(world_point[1]), float(world_point[2])],
            'method': method,
        }
        if depth_used is not None:
            response['depth'] = float(depth_used)
        if conf_value is not None:
            response['conf'] = float(conf_value)
        return response

    def _set_extrinsics_rpc(self, req: dict) -> dict:
        try:
            cam_id = str(req.get('cameraId'))
        except Exception:
            return {'ok': False, 'error': 'cameraId_required'}
        E = None
        try:
            if isinstance(req.get('E'), list) and len(req['E']) == 16:
                E = [float(x) for x in req['E']]
            elif isinstance(req.get('Twc'), list) and len(req['Twc']) == 16:
                Twc = np.array(req['Twc'], dtype=float).reshape((4, 4), order='F')
                Emat = np.linalg.inv(Twc)
                E = list(np.array(Emat, dtype=float).reshape(-1, order='F'))
            else:
                return {'ok': False, 'error': 'E_or_Twc_required'}
        except Exception as e:
            return {'ok': False, 'error': f'parse_error: {e}'}
        ok = save_extrinsics(self._calib_paths.get('extrinsics', ''), cam_id, E)
        if not ok:
            return {'ok': False, 'error': 'persist_failed'}
        # Rebuild bundle with updated extrinsics
        extr = load_extrinsics(self._calib_paths.get('extrinsics', ''))
        align = self.calibration_bundle.get('align', {}) if self.calibration_bundle else load_alignment(self._calib_paths.get('alignment', ''))
        cam_ids = self._get_camera_ids()
        model_map = dict(self.config.calibration.CAMERA_INTRINSICS_MODEL_MAP)
        camera_specs = dict(getattr(self.config.calibration, 'CAMERA_SPECS', {}) or {})
        self.calibration_bundle = assemble_calibration_bundle(
            cam_ids,
            self._intrinsics_models or {},
            model_map,
            extr,
            align,
            camera_specs
        )
        if self.depth_source is not None:
            try:
                self.depth_source.calibration_bundle = self.calibration_bundle
            except Exception:
                pass
        # Broadcast updated bundle
        try:
            if self.websocket_server:
                self.websocket_server.broadcast_sync({'type': 'calibration-bundle', 'data': self.calibration_bundle})
        except Exception:
            pass
        return {'ok': True}

    def _set_align_rpc(self, req: dict) -> dict:
        try:
            align_update = req.get('align', {})
            if not isinstance(align_update, dict):
                return {'ok': False, 'error': 'align_required'}
            
            # Basic validation
            matrix = align_update.get('matrix')
            if matrix is not None and (not isinstance(matrix, list) or len(matrix) != 16):
                return {'ok': False, 'error': 'invalid_matrix'}
            
            floor_y = align_update.get('floor_y')
            if floor_y is not None and not isinstance(floor_y, (int, float)):
                return {'ok': False, 'error': 'invalid_floor_y'}
            
            s_obj_to_m = align_update.get('units', {}).get('s_obj_to_m') if isinstance(align_update.get('units'), dict) else None
            if s_obj_to_m is not None:
                try:
                    sval = float(s_obj_to_m)
                except Exception:
                    return {'ok': False, 'error': 'invalid_s_obj_to_m'}
                if sval <= 0:
                    return {'ok': False, 'error': 'invalid_s_obj_to_m'}
            
            # Save (merges with existing)
            ok = save_alignment(self._calib_paths.get('alignment', ''), align_update)
            if not ok:
                return {'ok': False, 'error': 'persist_failed'}
            
            # Reload and rebuild bundle
            align = load_alignment(self._calib_paths.get('alignment', ''))
            extr = load_extrinsics(self._calib_paths.get('extrinsics', ''))
            cam_ids = self._get_camera_ids()
            model_map = dict(self.config.calibration.CAMERA_INTRINSICS_MODEL_MAP)
            camera_specs = dict(getattr(self.config.calibration, 'CAMERA_SPECS', {}) or {})
            self.calibration_bundle = assemble_calibration_bundle(
                cam_ids,
                self._intrinsics_models or {},
                model_map,
                extr,
                align,
                camera_specs
            )
            if self.depth_source is not None:
                try:
                    self.depth_source.calibration_bundle = self.calibration_bundle
                except Exception:
                    pass
            
            # Re-broadcast to all clients
            try:
                if self.websocket_server:
                    self.websocket_server.broadcast_sync({'type': 'calibration-bundle', 'data': self.calibration_bundle})
            except Exception:
                pass  # Non-blocking
            
            self.logger.info(f"Updated alignment: matrix={bool(matrix)}, floor_y={floor_y}, s_obj_to_m={s_obj_to_m}")
            return {'ok': True}
        except Exception as e:
            self.logger.error(f"set_align RPC error: {e}")
            return {'ok': False, 'error': str(e)}

    def _create_depth_publisher(self) -> Optional[DepthDiagnosticsPublisher]:
        try:
            integrations = self.config.integrations
            if not getattr(integrations, 'ENABLE_OCCUPANCY_PUBLISH', True):
                return None
            base_root = str(integrations.BASE_TOPIC).split('/')[0] or 'noesis'
            diag_cfg = DiagnosticsConfig(
                base_topic=base_root,
                mqtt_host=str(integrations.MQTT_HOST),
                mqtt_port=int(integrations.MQTT_PORT),
                mqtt_username=str(integrations.MQTT_USERNAME),
                mqtt_password=str(integrations.MQTT_PASSWORD),
                mqtt_qos=int(integrations.MQTT_QOS),
                mqtt_retain=False,
                influx_url=str(integrations.INFLUX_URL),
                influx_org=str(integrations.INFLUX_ORG),
                influx_token=str(integrations.INFLUX_TOKEN),
                influx_bucket=str(integrations.INFLUX_BUCKET_RAW),
            )
            return DepthDiagnosticsPublisher(diag_cfg)
        except Exception as exc:
            self.logger.warning(f"Depth diagnostics publisher unavailable: {exc}")
            return None

    def _schedule_depth_inference(self, analysis_frame: AnalysisFrame) -> None:
        if self.depth_source is None:
            return
        if analysis_frame.frame is None:
            return
        timestamp = analysis_frame.timestamp or time.time()
        camera_id = analysis_frame.camera_id
        if not self.depth_source.should_infer(camera_id, timestamp):
            return
        last_summary = self._last_depth_summary.get(camera_id)
        if last_summary and last_summary.conf_mean is not None and last_summary.conf_mean >= 0.9:
            try:
                if not analysis_frame.tracks:
                    self.logger.debug("Skipping MapAnything mono inference for %s (confidence %.2f, no motion)", camera_id, last_summary.conf_mean)
                    return
            except Exception:
                pass
        pending = self._pending_depth_futures.get(camera_id)
        if pending and not pending.done():
            return

        frame_copy = analysis_frame.frame.copy()
        future = self._depth_executor.submit(
            self.depth_source.maybe_infer_mono,
            camera_id,
            frame_copy,
            self.calibration_bundle,
            timestamp,
        )
        self._pending_depth_futures[camera_id] = future
        future.add_done_callback(lambda fut, cam=camera_id: self._handle_depth_future(cam, fut))

    def _handle_depth_future(self, camera_id: str, future: Future) -> None:
        try:
            result = future.result()
            if result is not None:
                self._latest_depth_results[camera_id] = result
                self._last_depth_summary[camera_id] = result.summary
                try:
                    self.depth_source.update_depth_cache(result)
                except Exception as exc:
                    self.logger.debug(f"Depth cache update failed for {camera_id}: {exc}")
                room_id = self._camera_room_map.get(camera_id, camera_id)
                now = time.time()
                last_pub = self._last_depth_publish.get(camera_id, 0.0)
                if (now - last_pub) >= 5.0:
                    self._last_depth_publish[camera_id] = now
                    if self._depth_publisher:
                        self._depth_publisher.publish_depth_summary(result, room_id)
                    summary_message = {
                        'type': 'ma_diagnostics',
                        'cam_id': camera_id,
                        'summary': {
                            'median': result.summary.median,
                            'p10': result.summary.p10,
                            'p90': result.summary.p90,
                            'conf_mean': result.summary.conf_mean,
                            'valid_ratio': result.summary.valid_ratio,
                            'sample_count': result.summary.sample_count,
                            'method': 'mde' if result.summary.conf_mean >= getattr(self.depth_source, 'min_conf', 0.5) else 'floor'
                        },
                        'ts': result.ts_us
                    }
                    self._schedule_ws_broadcast(summary_message)
        except Exception as exc:
            self.logger.error(f"Depth future for {camera_id} failed: {exc}")
        finally:
            self._pending_depth_futures.pop(camera_id, None)

    @profile_function("ApplicationManager.process_analysis_frame")
    def _process_analysis_frame(self, analysis_frame: AnalysisFrame):
        """Process analysis frame
        
        Args:
            analysis_frame: Analysis frame with detection and tracking results
        """
        try:
            self._schedule_depth_inference(analysis_frame)
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
            # Helper: normalize backend room keys (e.g., "LivingRoom", "living-room")
            # to Menon-friendly names with spaces (e.g., "Living Room").
            def _normalize_room_name(name: Any) -> str:
                try:
                    s = str(name)
                    if not s:
                        return s
                    # Replace common separators with space
                    s2 = s.replace('_', ' ').replace('-', ' ')
                    # Insert spaces before capital letters for CamelCase
                    import re
                    if ' ' not in s2 and any(c.islower() for c in s2) and any(c.isupper() for c in s2):
                        s2 = re.sub(r'(?<!^)(?=[A-Z])', ' ', s2)
                    # Normalize whitespace and Title Case
                    s2 = ' '.join(s2.split())
                    return s2.title()
                except Exception:
                    return str(name)

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
                                
                                # Create camera stats for this stream with normalized occupancy keys for frontend
                                try:
                                    raw_occ = (stream_tracking_data or {}).get('occupancy', {}) or {}
                                    norm_occ = {}
                                    for k, v in raw_occ.items():
                                        try:
                                            cnt = int(v) if v is not None else 0
                                        except Exception:
                                            cnt = 0
                                        norm_occ[_normalize_room_name(k)] = max(0, cnt)
                                    tracking_payload = {
                                        'active_tracks': (stream_tracking_data or {}).get('active_tracks', []),
                                        'transitions': (stream_tracking_data or {}).get('transitions', []),
                                        'occupancy': norm_occ,
                                    }
                                    # Augment active tracks with world coordinates if calibration available
                                    try:
                                        calib = self.calibration_bundle or {}
                                        cameras_node = calib.get('cameras') or {}
                                        k_table = cameras_node.get('K') if isinstance(cameras_node, dict) else None
                                        e_table = cameras_node.get('E') if isinstance(cameras_node, dict) else None
                                        intr = (k_table or {}).get(camera_name) if isinstance(k_table, dict) else None
                                        E = (e_table or {}).get(camera_name) if isinstance(e_table, dict) else None
                                        if (intr is None or E is None) and isinstance(cameras_node, dict):
                                            legacy_cam = cameras_node.get(camera_name)
                                            if isinstance(legacy_cam, dict):
                                                if intr is None:
                                                    intr = legacy_cam.get('intrinsics')
                                                if E is None:
                                                    extr = legacy_cam.get('extrinsics') or {}
                                                    if isinstance(extr, dict):
                                                        E = extr.get('E')
                                        align = calib.get('align', {})
                                        floor_y = float(align.get('floor_y') or 0.0)
                                        K = K_from_intrinsics(intr)
                                        if K is not None and isinstance(E, list) and len(E) == 16:
                                            pose = E_to_world_and_R(E)
                                            if pose is not None:
                                                Cw, Rwc = pose
                                                for t in tracking_payload.get('active_tracks', []) or []:
                                                    try:
                                                        bb = t.get('bbox')
                                                        if isinstance(bb, list) and len(bb) == 4:
                                                            fp = bbox_bottom_center(bb)
                                                            if fp is not None:
                                                                u, v = fp
                                                                O, D = ray_from_pixel(u, v, K, Cw, Rwc)
                                                                hit = intersect_floor(O, D, floor_y)
                                                                if hit is not None:
                                                                    t['world'] = [float(hit[0]), float(hit[1]), float(hit[2])]
                                                                    t['world_valid'] = True
                                                                else:
                                                                    t['world_valid'] = False
                                                    except Exception:
                                                        try:
                                                            t['world_valid'] = False
                                                        except Exception:
                                                            pass
                                    except Exception:
                                        pass
                                except Exception:
                                    tracking_payload = stream_tracking_data

                                camera_stats = {
                                    'fps': comprehensive_stats.get('fps', 0) / len(per_stream_tracking) if per_stream_tracking else 0,
                                    'frames_processed': comprehensive_stats.get('frames_processed', 0),
                                    'status': 'running' if comprehensive_stats.get('running', False) else 'stopped',
                                    'tracking': tracking_payload
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

        # Stop integrations (OccupancyPublisher)
        if getattr(self, '_occupancy_publisher', None) is not None:
            try:
                self.logger.info("Stopping OccupancyPublisher")
                self._occupancy_publisher.close()
                self._occupancy_publisher = None
                self.logger.info("Stopped OccupancyPublisher")
            except Exception as e:
                self.logger.error(f"Error stopping OccupancyPublisher: {e}")
        
        # STEP 4: Stop result processing thread
        if hasattr(self, 'result_thread') and self.result_thread:
            try:
                self.logger.info("Stopping result processing thread")
                safe_join(self.result_thread, timeout=2.0, name="result_processing")
                self.logger.info("Stopped result processing thread")
            except Exception as e:
                self.logger.error(f"Error stopping result processing thread: {e}")
        
        # STEP 5: Stop WebSocket server and event loop
        if self._backup_task:
            try:
                self.logger.info("Cancelling Menon config backup task")
                self._backup_task.cancel()
                if self.event_loop:
                    self.event_loop.run_until_complete(asyncio.sleep(0))
            except Exception as e:
                self.logger.debug(f"Error cancelling backup task: {e}")

        if self.websocket_server:
            try:
                self.logger.info("Stopping WebSocket server")
                stop_ok = False
                # Use the improved sync stop method
                try:
                    stop_ok = self.websocket_server.stop_sync()
                except Exception as e:
                    self.logger.warning(f"WebSocket server stop_sync failed: {e}")

                # If websocket server thread loop exists, stop via that loop
                if hasattr(self, 'websocket_loop') and self.websocket_loop:
                    try:
                        # Schedule the event loop to stop
                        self.websocket_loop.call_soon_threadsafe(self.websocket_loop.stop)
                    except Exception as e:
                        self.logger.debug(f"Could not stop event loop: {e}")
                else:
                    # Fallback: mark not running
                    try:
                        self.websocket_server.running = False
                    except Exception as exc:
                        self.logger.debug(f"Could not mark WebSocket server as not running: {exc}")

                if stop_ok:
                    self.logger.info("✅ WebSocket server shutdown initiated")
                else:
                    self.logger.warning("⚠️ WebSocket server stop did not confirm completion")
            except Exception as e:
                self.logger.error(f"Error stopping WebSocket server: {e}")

        # STEP 6: Stop WebSocket thread
        if hasattr(self, 'websocket_thread') and self.websocket_thread:
            try:
                self.logger.info("Waiting for WebSocket thread to complete...")
                # Give more time for graceful shutdown
                thread_joined = safe_join(self.websocket_thread, timeout=5.0, name="websocket_thread")
                if thread_joined:
                    self.logger.info("✅ WebSocket thread stopped gracefully")
                else:
                    self.logger.warning("⚠️ WebSocket thread did not stop within timeout, will be abandoned")
            except Exception as e:
                self.logger.error(f"Error stopping WebSocket thread: {e}")
        
        # STEP 7: Stop CPU profiling
        if self.cpu_profiler:
            try:
                stop_global_profiling()
                self.logger.info("Stopped comprehensive CPU profiling")
            except Exception as e:
                self.logger.error(f"Error stopping CPU profiling: {e}")

        try:
            self._depth_executor.shutdown(wait=False, cancel_futures=True)
        except Exception as exc:
            self.logger.debug(f"Depth executor shutdown error: {exc}")

        if getattr(self, '_depth_publisher', None) is not None:
            try:
                self._depth_publisher.close()
            except Exception as exc:
                self.logger.debug(f"Depth publisher shutdown error: {exc}")

        if getattr(self, 'depth_source', None) is not None:
            try:
                self.depth_source.close()
            except Exception as exc:
                self.logger.debug(f"Depth storage shutdown error: {exc}")

        # Always terminate the MapAnything microservice
        self._terminate_mapanything_process()

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
