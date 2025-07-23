#!/usr/bin/env python3
"""
DeepStream Video Pipeline

This module provides a high-performance video processing pipeline using NVIDIA DeepStream
for GPU-accelerated video reading, decoding, and preprocessing. It replaces the DALI
pipeline with DeepStream's optimized GStreamer elements.

Key Features:
- GPU-native video decoding with nvurisrcbin
- Batch processing with nvstreammux
- GPU preprocessing with nvdspreprocess
- Zero-copy tensor output via appsink
- Support for RTSP, file, and camera sources
"""

# Core imports
import sys
import os
os.environ['no_proxy'] = '*'
import logging
import threading
import time
import queue
import numpy as np  # Added for shape calculations

# Bypass libproxy issues by disabling GIO proxy resolver
import os

# Add ctypes imports for PyCapsule handling
import ctypes
from ctypes import c_void_p, c_uint64, c_uint32, c_int, POINTER, Structure, c_char_p, c_float

# Add missing imports for tensor operations
import torch
from typing import Optional, Tuple, Dict, Any, List, Union

# GStreamer imports
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GLib, GstApp  # type: ignore  # noqa: E402

# DeepStream imports
sys.path.append('/opt/nvidia/deepstream/deepstream/lib')
import pyds  # type: ignore  # noqa: E402

# Local imports
from websocket_server import WebSocketServer  # noqa: E402
from exponential_backoff import ExponentialBackoff  # noqa: E402
from utils import RateLimitedLogger  # noqa: E402

# Initialize GStreamer
Gst.init(None)

from config import AppConfig  # noqa: E402

# DeepStream metadata type constants
NVDS_PREPROCESS_BATCH_META = pyds.NvDsMetaType.NVDS_PREPROCESS_BATCH_META if hasattr(pyds.NvDsMetaType, 'NVDS_PREPROCESS_BATCH_META') else 27
NVDSINFER_TENSOR_OUTPUT_META = pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META if hasattr(pyds.NvDsMetaType, 'NVDSINFER_TENSOR_OUTPUT_META') else 12

# Define ctypes structures matching nvdspreprocess_meta.h (for fallback if pyds insufficient)
class StdVectorInt(Structure):
    _fields_ = [
        ("begin", c_void_p),
        ("end", c_void_p),
        ("capacity", c_void_p)
    ]

class NvDsPreProcessTensorMeta(ctypes.Structure):
    _fields_ = [
        ("raw_tensor_buffer", ctypes.c_void_p),
        ("buffer_size", ctypes.c_uint64),
        ("tensor_shape", StdVectorInt),
        ("data_type", ctypes.c_uint32),
        ("tensor_name", ctypes.c_char_p),
        ("gpu_id", ctypes.c_uint32),
        ("private_data", ctypes.c_void_p),
        ("meta_id", ctypes.c_uint32),
        ("maintain_aspect_ratio", ctypes.c_int),
        ("aspect_ratio", ctypes.c_float * 4),
    ]

class GstNvDsPreProcessBatchMeta(ctypes.Structure):
    _fields_ = [
        ("tensor_meta", ctypes.POINTER(NvDsPreProcessTensorMeta)),
        ("private_data", ctypes.c_void_p),
        ("meta_id", ctypes.c_uint32),
        ("batch_size", ctypes.c_uint32),
        ("gpu_id", ctypes.c_uint32),
        ("roi_vector", ctypes.c_void_p),  # Pointer to vector; access later if needed
        ("roi_vector_size", ctypes.c_uint32),
        ("target_unique_ids", ctypes.c_uint32 * 4),  # Fixed max 4
    ]


class DeepStreamVideoPipeline:
    """
    DeepStream-based video pipeline for GPU-accelerated video processing.
    
    This pipeline uses:
    - nvurisrcbin for source handling (RTSP/file/camera)
    - nvstreammux for batching
    - nvdspreprocess for GPU preprocessing
    - appsink for tensor output
    """
    
    def __init__(self, rtsp_url: str, config: AppConfig, websocket_port: int = 8765, 
                 config_file: str = "config_infer_primary_yolo11.txt",
                 preproc_config: str = "config_preproc.txt"):
        self.rtsp_url = rtsp_url
        self.websocket_port = websocket_port
        self.config_file = config_file
        self.preproc_config = preproc_config
        self.config = config  # Store config as instance variable
        self.logger = logging.getLogger(__name__)
        # Use rate-limited logger for frame-level messages
        self.rate_limited_logger = RateLimitedLogger(self.logger, rate_limit_seconds=5.0)
        
        # Initialize pipeline components
        self.pipeline: Optional[Gst.Pipeline] = None
        self.loop: Optional[GLib.MainLoop] = None
        self.websocket_server: Optional[WebSocketServer] = None
        self.backoff = ExponentialBackoff()
        # self.cuda_manager = CudaContextManager()  # Commented out - not available
        
        # Threading and state management
        self.running = False
        self.pipeline_thread: Optional[threading.Thread] = None
        self.websocket_thread: Optional[threading.Thread] = None
        
        # Error tracking
        self.consecutive_no_meta_count = 0
        self.max_consecutive_no_meta = 10
        
        # Phase 3.4: Enhanced fail-safe mechanisms
        self.pipeline_health_check_interval = 30  # seconds
        self.last_health_check = time.time()
        self.pipeline_restart_count = 0
        self.max_pipeline_restarts = 3
        self.error_recovery_enabled = True
        
        # Pipeline performance monitoring
        self.fps_monitor = {
            'last_frame_time': time.time(),
            'frame_intervals': [],
            'avg_fps': 0.0,
            'low_fps_count': 0,
            'max_low_fps_count': 5
        }
        
        # Pipeline configuration - read batch size from config
        config_batch_size = self.config.processing.DEEPSTREAM_MUX_BATCH_SIZE
        self.batch_size = config_batch_size if config_batch_size > 0 else 1
        self.max_width = 1920
        self.max_height = 1080
        self.device_id = 0
        
        # Create sources list from single RTSP URL
        self.sources = [{'url': rtsp_url, 'name': 'rtsp_source', 'width': 1920, 'height': 1080, 'enabled': True}]
        
        # Add missing attributes for compatibility
        self.tensor_queue = queue.Queue(maxsize=30)
        self.frame_count = 0
        self.start_time = time.time()
        
        # Initialize GStreamer
        Gst.init(None)
        
        # Create pipeline elements
        self._create_pipeline()

    def _validate_deepstream_config(self) -> bool:
        """Validate DeepStream configuration before pipeline creation"""
        try:
            # Check preprocessing config file
            if not os.path.exists(self.preproc_config):
                self.logger.error(f"Preprocessing config file not found: {self.preproc_config}")
                return False
            
            # Check inference config file
            if not os.path.exists(self.config_file):
                self.logger.error(f"Inference config file not found: {self.config_file}")
                return False
            
            # Check custom library
            custom_lib_path = "/opt/nvidia/deepstream/deepstream/lib/gst-plugins/libcustom2d_preprocess.so"
            if not os.path.exists(custom_lib_path):
                self.logger.error(f"Custom preprocessing library not found: {custom_lib_path}")
                return False
            
            self.logger.info("✅ DeepStream configuration validation passed")
            return True
        except Exception as e:
            self.logger.error(f"Error validating DeepStream config: {e}")
            return False

    def _check_for_engine_file(self, config_file_path: str):
        """Checks for a pre-built TensorRT engine file and logs whether a rebuild is required.
        
        This method intelligently determines the expected engine path based on:
        1. The actual model file specified in config.models.MODEL_PATH
        2. The engine path specified in the nvinfer config file
        3. Common engine naming patterns
        
        Search locations:
        1. ./models/engines/ (primary location)
        2. ./models/ (fallback location)
        """
        try:
            # Get workspace root for path resolution
            workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
            
            # Collect candidate engine paths -------------------------------------------------------
            candidate_paths: List[str] = []
            
            # 1) Engine path from nvinfer config file
            engine_path_from_cfg: Optional[str] = None
            try:
                with open(config_file_path, "r", encoding="utf-8") as cfg_fd:
                    for line in cfg_fd:
                        if line.strip().startswith("model-engine-file="):
                            engine_path_from_cfg = line.split("=", 1)[1].strip()
                            break
            except FileNotFoundError:
                self.logger.warning(f"⚠️  nvinfer config file not found: {config_file_path}")
            
            if engine_path_from_cfg:
                candidate_paths.append(engine_path_from_cfg)
            
            # 2) Generate engine path based on actual model file
            if hasattr(self.config, "models") and hasattr(self.config.models, "MODEL_PATH"):
                model_path = self.config.models.MODEL_PATH
                if model_path:
                    # Extract model name and generate engine path
                    model_basename = os.path.splitext(os.path.basename(model_path))[0]
                    
                    # Generate multiple possible engine names
                    engine_names = [
                        f"{model_basename}_fp16.engine",  # yolo11m_fp16.engine
                        f"{model_basename}.engine",       # yolo11m.engine
                        f"{model_basename}_b1_gpu0_fp16.engine",  # yolo11m_b1_gpu0_fp16.engine
                        f"{model_basename}.onnx_b1_gpu0_fp16.engine",  # yolo11m.onnx_b1_gpu0_fp16.engine
                        f"detection_fp16.engine",         # fallback
                    ]
                    
                    # Add to search paths in both locations
                    for engine_name in engine_names:
                        candidate_paths.extend([
                            os.path.join("models", "engines", engine_name),
                            os.path.join("models", engine_name)
                        ])
            
            # 3) Config override (if specified)
            if hasattr(self.config, "models") and hasattr(self.config.models, "DETECTION_ENGINE_PATH"):
                config_engine_path = self.config.models.DETECTION_ENGINE_PATH
                if config_engine_path:
                    candidate_paths.insert(0, config_engine_path)  # Priority override
            
            # Resolve & test ----------------------------------------------------------------
            for path in candidate_paths:
                # Skip empty paths
                if not path:
                    continue
                    
                # Resolve relative paths against workspace root
                abs_path = path if os.path.isabs(path) else os.path.join(workspace_root, path)
                
                if os.path.exists(abs_path):
                    self.logger.info(f"✅ Found existing TensorRT engine: {abs_path}")
                    self.logger.info(f"✅ Engine file size: {os.path.getsize(abs_path) / (1024*1024):.1f} MB")
                    return  # Engine found – no build required
            
            # None of the candidates exist --------------------------------------------------
            self.logger.warning("⚠️  No existing TensorRT engine found. nvinfer will build a new one (this may take several minutes)...")
            self.logger.info(f"   Searched locations:")
            for path in candidate_paths:
                if path:
                    abs_path = path if os.path.isabs(path) else os.path.join(workspace_root, path)
                    self.logger.info(f"   - {abs_path}")
            
            # Log the model that will be used for building
            if hasattr(self.config, "models") and hasattr(self.config.models, "MODEL_PATH"):
                model_path = self.config.models.MODEL_PATH
                if model_path:
                    abs_model_path = model_path if os.path.isabs(model_path) else os.path.join(workspace_root, model_path)
                    if os.path.exists(abs_model_path):
                        self.logger.info(f"✅ Will build engine from model: {abs_model_path}")
                        self.logger.info(f"   Model file size: {os.path.getsize(abs_model_path) / (1024*1024):.1f} MB")
                    else:
                        self.logger.error(f"❌ Model file not found: {abs_model_path}")
            
        except Exception as e:
            self.logger.error(f"Error during engine-file check: {e}")


    def _create_pipeline(self) -> bool:
        """Create the DeepStream GStreamer pipeline."""
        try:
            # Validate DeepStream configuration first
            if not self._validate_deepstream_config():
                return False
            
            # Check if nvdspreprocess is available
            test_preprocess = Gst.ElementFactory.make("nvdspreprocess", "test")
            if not test_preprocess:
                self.logger.error("❌ nvdspreprocess element not available - check DeepStream installation")
                raise RuntimeError("nvdspreprocess element not found")
            else:
                self.logger.info("✅ nvdspreprocess element available")
                # Clean up test element
                test_preprocess = None
            
            # Create pipeline
            self.pipeline = Gst.Pipeline()
            if not self.pipeline:
                raise RuntimeError("Failed to create pipeline")
            self.logger.info(f"✅ Created pipeline: {self.pipeline}")
            
            # Set up bus message handling
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self._on_bus_message)
            
            # Create nvstreammux
            streammux = Gst.ElementFactory.make("nvstreammux", "mux")
            if not streammux:
                raise RuntimeError("Failed to create nvstreammux")
            self.logger.info(f"✅ Created nvstreammux: {streammux}")
            
            # Configure streammux for batch processing
            streammux.set_property("batch-size", self.batch_size)
            streammux.set_property("width", 1280)
            streammux.set_property("height", 720)
            streammux.set_property("batched-push-timeout", 4000000)
            streammux.set_property("live-source", 1)
            streammux.set_property("sync-inputs", 0)  # Disable input synchronization for better batch performance
            streammux.set_property("drop-pipeline-eos", 1)  # Drop EOS events to prevent pipeline stalls
            
            # Normal pipeline with preprocessing and inference
            # Create nvdspreprocess
            preprocess = Gst.ElementFactory.make("nvdspreprocess", "preprocess")
            if not preprocess:
                raise RuntimeError("Failed to create nvdspreprocess")
            self.logger.info(f"✅ Created nvdspreprocess: {preprocess}")

            # Set nvdspreprocess properties for batch processing
            preprocess.set_property("config-file", self.config.processing.DEEPSTREAM_PREPROCESS_CONFIG)
            preprocess.set_property("gpu-id", self.device_id)
            preprocess.set_property("enable", True)
            preprocess.set_property("process-on-frame", True)

            self.logger.info(f"✅ nvdspreprocess configured: config={self.config.processing.DEEPSTREAM_PREPROCESS_CONFIG}, gpu-id={self.device_id}")

            self.pipeline.add(preprocess)

            # Phase 1: Add pad probe on nvdspreprocess sink pad to log buffer metadata
            preprocess_sink_pad = preprocess.get_static_pad("sink")
            if preprocess_sink_pad:
                preprocess_sink_pad.add_probe(Gst.PadProbeType.BUFFER, self._preprocess_probe, 0)
                self.logger.info("✅ Added metadata probe to nvdspreprocess sink pad")
            else:
                self.logger.warning("❌ Failed to get nvdspreprocess sink pad for probe")

            # Check for pre-existing engine file before creating nvinfer
            self._check_for_engine_file(self.config_file)
            # Create nvinfer
            nvinfer = Gst.ElementFactory.make("nvinfer", "primary-infer")
            if not nvinfer:
                raise RuntimeError("Failed to create nvinfer")
            self.logger.info(f"✅ Created nvinfer: {nvinfer}")
            nvinfer.set_property("config-file-path", self.config_file)
            nvinfer.set_property("input-tensor-meta", True)
            self.logger.info(f"✅ Set nvinfer input-tensor-meta=True")
            
            # Add probe to nvinfer source pad to validate inference output
            nvinfer_src_pad = nvinfer.get_static_pad("src")
            if nvinfer_src_pad:
                nvinfer_src_pad.add_probe(Gst.PadProbeType.BUFFER, self._nvinfer_src_pad_buffer_probe, 0)
                self.logger.info("✅ Added inference validation probe to nvinfer source pad")
            else:
                self.logger.warning("❌ Failed to get nvinfer source pad for validation probe")
                
            # Create nvtracker with configuration file
            nvtracker = Gst.ElementFactory.make("nvtracker", "tracker")
            if not nvtracker:
                raise RuntimeError("Failed to create nvtracker")
            self.logger.info(f"✅ Created nvtracker: {nvtracker}")
                
            # Set tracker properties for batch processing
            nvtracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
            nvtracker.set_property("tracker-width", 640)
            nvtracker.set_property("tracker-height", 384)
            nvtracker.set_property("gpu-id", 0)
            nvtracker.set_property("tracking-id-reset-mode", 0)  # Never reset tracking ID
            
            # Set tracker configuration file for batch processing
            tracker_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_tracker_nvdcf_batch.yml")
            if os.path.exists(tracker_config_path):
                nvtracker.set_property("ll-config-file", tracker_config_path)
                self.logger.info(f"✅ Set batch tracker config file: {tracker_config_path}")
            else:
                # Fallback to basic tracker config
                fallback_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tracker_nvdcf.yml")
                if os.path.exists(fallback_config_path):
                    nvtracker.set_property("ll-config-file", fallback_config_path)
                    self.logger.info(f"✅ Set fallback tracker config file: {fallback_config_path}")
                else:
                    self.logger.warning(f"⚠️ No tracker config file found, using default properties")
                
            # Phase 3.1: Create nvdsanalytics for advanced analytics
            nvanalytics = Gst.ElementFactory.make("nvdsanalytics", "analytics")
            if not nvanalytics:
                raise RuntimeError("Failed to create nvdsanalytics")
            self.logger.info(f"✅ Created nvdsanalytics: {nvanalytics}")
                
            # Set nvdsanalytics configuration
            nvanalytics.set_property("config-file", "config_nvdsanalytics.txt")
                
            # Phase 3.2: Create secondary inference engine (SGIE) for classification
            sgie = None  # Disabled for now to avoid engine file issues
            # sgie = Gst.ElementFactory.make("nvinfer", "secondary-infer")
            # if not sgie:
            #     self.logger.warning("Failed to create secondary inference engine - continuing without SGIE")
            #     sgie = None
            # else:
            #     self.logger.info(f"✅ Created secondary inference engine: {sgie}")
            #     sgie.set_property("config-file-path", "config_infer_secondary_classification.txt")
                
            # Create nvvideoconvert
            nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
            if not nvvidconv:
                raise RuntimeError("Failed to create nvvideoconvert")
            self.logger.info(f"✅ Created nvvideoconvert: {nvvidconv}")
            
            # Create nvdsosd
            nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
            if not nvosd:
                raise RuntimeError("Failed to create nvdsosd")
            self.logger.info(f"✅ Created nvdsosd: {nvosd}")
            
            # Create appsink
            appsink = Gst.ElementFactory.make("appsink", "sink")
            if not appsink:
                raise RuntimeError("Failed to create appsink")
            self.logger.info(f"✅ Created appsink: {appsink}")
            
            # Configure appsink
            appsink.set_property("emit-signals", True)
            appsink.set_property("sync", False)
            appsink.set_property("max-buffers", 1)
            appsink.set_property("drop", True)
            
            # Add all elements to pipeline
            self.pipeline.add(streammux)
            self.pipeline.add(preprocess)
            self.pipeline.add(nvinfer)
            self.pipeline.add(nvtracker)
            self.pipeline.add(nvanalytics) # Added nvdsanalytics
            if sgie:  # Add SGIE if created successfully
                self.pipeline.add(sgie)
            self.pipeline.add(nvvidconv)
            self.pipeline.add(nvosd)
            self.pipeline.add(appsink)
                
            # Store references
            self.streammux = streammux
            self.preprocess = preprocess
            self.nvinfer = nvinfer
            self.nvtracker = nvtracker
            self.nvanalytics = nvanalytics  # Store nvdsanalytics reference
            self.sgie = sgie  # Store secondary inference engine reference
            self.nvvidconv = nvvidconv
            self.nvosd = nvosd
            self.appsink = appsink
                
            # Connect appsink signal
            appsink.connect("new-sample", self._on_new_sample)
                
            # Phase 3.1: Add probe to capture analytics metadata
            analytics_src_pad = nvanalytics.get_static_pad("src")
            if analytics_src_pad:
                analytics_src_pad.add_probe(Gst.PadProbeType.BUFFER, self._analytics_probe, 0)
                self.logger.info("✅ Added analytics probe to nvdsanalytics")
            
            # Create and add source
            source_bin = self._create_source_bin(0, {'url': self.rtsp_url})
            if not source_bin:
                raise RuntimeError("Failed to create source bin")
            self.pipeline.add(source_bin)
            
            # Link source to streammux
            source_pad = source_bin.get_static_pad("src")
            sink_pad = streammux.get_request_pad("sink_0")
            if source_pad.link(sink_pad) != Gst.PadLinkReturn.OK:
                raise RuntimeError("Failed to link source to streammux")
            
            # Link elements based on mode
            if not streammux.link(preprocess):
                raise RuntimeError("Failed to link nvstreammux to nvdspreprocess")
            if not preprocess.link(nvinfer):
                raise RuntimeError("Failed to link nvdspreprocess to nvinfer")
            if not nvinfer.link(nvtracker):
                raise RuntimeError("Failed to link nvinfer to nvtracker")
            if not nvtracker.link(nvanalytics): # Added nvdsanalytics link
                raise RuntimeError("Failed to link nvtracker to nvdsanalytics")
            
            # Phase 3.2: Link through SGIE if available
            if sgie:
                if not nvanalytics.link(sgie):
                    raise RuntimeError("Failed to link nvdsanalytics to secondary inference")
                if not sgie.link(nvvidconv):
                    raise RuntimeError("Failed to link secondary inference to nvvideoconvert")
                self.logger.info("✅ Pipeline linked with secondary inference engine")
            else:
                if not nvanalytics.link(nvvidconv):
                    raise RuntimeError("Failed to link nvdsanalytics to nvvideoconvert")
                self.logger.info("✅ Pipeline linked without secondary inference engine")
            
            if not nvvidconv.link(nvosd):
                raise RuntimeError("Failed to link nvvideoconvert to nvdsosd")
            if not nvosd.link(appsink):
                raise RuntimeError("Failed to link nvdsosd to appsink")
            
            self.logger.info("✅ All elements linked successfully (with simplified tracker)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create pipeline: {e}")
            return False
    
    def _nvinfer_src_pad_buffer_probe(self, pad, info, u_data):
        """Probe to validate inference output and detect black frame causes"""
        try:
            gst_buffer = info.get_buffer()
            if not gst_buffer:
                return Gst.PadProbeReturn.OK
                
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
            if not batch_meta:
                return Gst.PadProbeReturn.OK
                
            # Count detections per frame
            frame_meta_list = batch_meta.frame_meta_list
            while frame_meta_list:
                frame_meta = pyds.NvDsFrameMeta.cast(frame_meta_list.data)
                obj_count = 0
                l_obj = frame_meta.obj_meta_list
                while l_obj:
                    obj_count += 1
                    try:
                        l_obj = l_obj.next
                    except StopIteration:
                        break
                
                if obj_count == 0:
                    self.logger.warning(f"⚠️ Frame {frame_meta.frame_num}: No detections - checking tensor")
                    # Attempt to access tensor meta for debugging
                    user_meta_list = frame_meta.frame_user_meta_list
                    while user_meta_list:
                        user_meta = pyds.NvDsUserMeta.cast(user_meta_list.data)
                        if user_meta.base_meta.meta_type == NVDS_PREPROCESS_BATCH_META:
                            self.logger.debug("Tensor meta present in probe")
                            break
                        user_meta_list = user_meta_list.next
                else:
                    self.logger.debug(f"✅ Frame {frame_meta.frame_num}: {obj_count} detections")
                
                # Add fallback: If zero, add dummy meta (for testing visualization)
                if obj_count == 0 and self.config.visualization.USE_NATIVE_DEEPSTREAM_OSD:
                    dummy_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
                    dummy_meta.class_id = 0
                    dummy_meta.rect_params.left = 0
                    dummy_meta.rect_params.top = 0
                    dummy_meta.rect_params.width = 100
                    dummy_meta.rect_params.height = 100
                    dummy_meta.text_params.display_text = "No Detections - Fallback"
                    pyds.nvds_add_obj_meta_to_frame(frame_meta, dummy_meta, None)
                
                try:
                    frame_meta_list = frame_meta_list.next
                except StopIteration:
                    break
            
            return Gst.PadProbeReturn.OK
        except Exception as e:
            self.logger.error(f"Error in nvinfer probe: {e}")
            return Gst.PadProbeReturn.OK

    def _create_source_bin(self, index: int, source_config: Dict[str, Any]) -> Gst.Element:
        """Create a source bin using DeepStream's canonical nvurisrcbin approach."""
        bin_name = f"source-bin-{index:02d}"
        
        # Create source bin
        source_bin = Gst.Bin.new(bin_name)
        if not source_bin:
            self.logger.error(f"❌ Failed to create source bin {index}")
            return None
        self.logger.info(f"✅ Created source bin {index}: {source_bin}")
        
        # Create nvurisrcbin - DeepStream's canonical source element
        # This automatically handles demuxing, decoding, and format conversion
        uri_decode_bin = Gst.ElementFactory.make("nvurisrcbin", f"uri-decode-bin-{index}")
        if not uri_decode_bin:
            self.logger.error(f"❌ Failed to create nvurisrcbin for source {index}")
            return None
        
        # Set URI property
        source_url = source_config['url']
        uri_decode_bin.set_property("uri", source_url)
        self.logger.info(f"✅ Set URI: {source_url}")
        
        # Configure nvurisrcbin properties
        uri_decode_bin.set_property("gpu-id", self.device_id)
        uri_decode_bin.set_property("cudadec-memtype", 0)  # Device memory for best performance
        uri_decode_bin.set_property("source-id", index)  # Set deterministic source ID
        
        # Enable file-loop for file sources if needed
        if source_url.startswith('file://'):
            uri_decode_bin.set_property("file-loop", 1)
            self.logger.info("✅ Enabled file-loop for file source")
        
        # Set RTSP-specific properties
        if source_url.startswith('rtsp://') or source_url.startswith('rtsps://'):
            # RTSP latency and drop settings are handled by child elements
            # We'll configure them in the child-added callback
            pass
        
        # Add to bin
        source_bin.add(uri_decode_bin)
        
        # Create ghost pad (no target initially - will be set in pad-added callback)
        ghost_pad = Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
        if not ghost_pad:
            self.logger.error(f"❌ Failed to create ghost pad for source bin {index}")
            return None
            
        source_bin.add_pad(ghost_pad)
        
        # Connect pad-added signal to handle dynamic pad creation
        def cb_newpad(decodebin, decoder_src_pad, data):
            """Callback for when nvurisrcbin creates a new pad."""
            self.logger.info(f"🔍 New pad added for source {index}: {decoder_src_pad.get_name()}")
            
            # Get pad capabilities
            caps = decoder_src_pad.get_current_caps()
            if not caps:
                caps = decoder_src_pad.query_caps()
            
            if not caps:
                self.logger.error(f"❌ Failed to get caps for pad {decoder_src_pad.get_name()}")
                return
            
            # Check if this is a video pad
            gststruct = caps.get_structure(0)
            gstname = gststruct.get_name()
            features = caps.get_features(0)
            
            self.logger.info(f"🔍 Pad caps: {gstname}, features: {features}")
            
            if gstname.find("video") != -1:
                # Verify NVIDIA decoder was selected (NVMM memory features)
                if features.contains("memory:NVMM"):
                    # Link to ghost pad
                    if not ghost_pad.set_target(decoder_src_pad):
                        self.logger.error(f"❌ Failed to link decoder src pad to ghost pad for source {index}")
                    else:
                        self.logger.info(f"✅ Linked decoder src pad to ghost pad for source {index}")
                else:
                    self.logger.error(f"❌ Decodebin did not pick NVIDIA decoder plugin for source {index}")
            
        # Connect child-added signal to configure child elements
        def decodebin_child_added(child_proxy, obj, name, user_data):
            """Callback for when nvurisrcbin adds child elements."""
            self.logger.info(f"🔍 Decodebin child added for source {index}: {name}")
            
            # Recursively connect to nested decodebins
            if name.find("decodebin") != -1:
                obj.connect("child-added", decodebin_child_added, user_data)
            
            # Configure nvv4l2decoder specifically to fix ioctl errors
            if "nvv4l2decoder" in name:
                self.logger.info(f"🔍 Found nvv4l2decoder: {name}")
                try:
                    # Set explicit properties to reduce ioctl warnings
                    obj.set_property("gpu-id", self.device_id)
                    obj.set_property("cudadec-memtype", 0)  # Device memory for best performance
                    obj.set_property("skip-frames", 0)  # Don't skip frames
                    obj.set_property("drop-frame-interval", 0)  # Don't drop frames
                    
                    
                    self.logger.info(f"✅ Configured nvv4l2decoder properties for {name}: gpu-id={self.device_id}, device=/dev/nvidia0")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not configure nvv4l2decoder properties: {e}")
            
            # Configure RTSP source properties
            if "source" in name:
                source_element = child_proxy.get_by_name("source")
                if source_element and source_element.find_property('drop-on-latency'):
                    source_element.set_property("drop-on-latency", True)
                    self.logger.info(f"✅ Set drop-on-latency for RTSP source {index}")
        
                # Set RTSP latency
                if source_element and source_element.find_property('latency'):
                    source_element.set_property("latency", self.config.processing.DEEPSTREAM_SOURCE_LATENCY)
                    self.logger.info(f"✅ Set RTSP latency to {self.config.processing.DEEPSTREAM_SOURCE_LATENCY}ms for source {index}")
                
                # Force TCP for RTSP to avoid UDP issues
                if source_element and source_element.find_property('protocols'):
                    source_element.set_property("protocols", 4)  # TCP only
                    self.logger.info(f"✅ Set RTSP protocols to TCP for source {index}")
        
            # Add explicit caps negotiation for H.264 parser
            if "h264parse" in name:
                self.logger.info(f"🔍 Found h264parse: {name}")
                # Rollback: do not override default parser properties
                # obj.set_property("config-interval", -1)  # Removed custom property
                # obj.set_property("disable-passthrough", False)  # Removed custom property
                self.logger.info(f"ℹ️ Using default h264parse properties for {name}")
            
            # Add caps filter to ensure proper format negotiation
            if "capsfilter" in name:
                self.logger.info(f"🔍 Found capsfilter: {name}")
                # Rollback: do not force custom caps, rely on upstream negotiation
                # caps_str = "video/x-h264, stream-format=byte-stream, alignment=au, profile=main, level=4"
                # caps = Gst.Caps.from_string(caps_str)
                # obj.set_property("caps", caps)
                # self.logger.info(f"✅ Set capsfilter caps: {caps_str}")
        
        # Connect callbacks
        uri_decode_bin.connect("pad-added", cb_newpad, source_bin)
        uri_decode_bin.connect("child-added", decodebin_child_added, source_bin)
        
        self.logger.info(f"✅ Created DeepStream source bin {index} with nvurisrcbin")
        return source_bin

    def _on_new_sample(self, appsink: GstApp.AppSink) -> Gst.FlowReturn:
        """Process new sample from appsink with tensor extraction"""
        try:
            sample = appsink.emit("pull-sample")
            if not sample:
                self.logger.warning("No sample received from appsink")
                return Gst.FlowReturn.ERROR
                
            buffer = sample.get_buffer()
            if not buffer:
                self.logger.warning("No buffer in sample")
                return Gst.FlowReturn.ERROR
                
            # Get batch metadata
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))  # type: ignore
            if not batch_meta:
                self.consecutive_no_meta_count += 1
                if self.consecutive_no_meta_count >= self.max_consecutive_no_meta:
                    self.logger.error(f"No batch metadata for {self.consecutive_no_meta_count} consecutive frames")
                return Gst.FlowReturn.OK
                
            # Reset error counter on successful metadata
            self.consecutive_no_meta_count = 0
            batch_user_meta_count = 0  # Add this line
            
            # Phase 1: Enhanced metadata search with detailed logging
            tensor_meta = None
            tensor_meta_found = False
            user_meta_list = batch_meta.batch_user_meta_list
            while user_meta_list:
                user_meta = pyds.NvDsUserMeta.cast(user_meta_list.data)
                meta_type = user_meta.base_meta.meta_type
                
                batch_user_meta_count += 1
                self.logger.debug(f"🔍 Batch meta type: {meta_type} (int: {int(meta_type)})")
                
                # Look for NVDS_PREPROCESS_BATCH_META type (this is what nvdspreprocess attaches)
                if int(meta_type) == NVDS_PREPROCESS_BATCH_META:
                    self.logger.info(f"✅ Found metadata type {NVDS_PREPROCESS_BATCH_META} (NVDS_PREPROCESS_BATCH_META)")
                elif int(meta_type) == NVDSINFER_TENSOR_OUTPUT_META:
                    self.logger.info(f"✅ Found metadata type {NVDSINFER_TENSOR_OUTPUT_META} (NVDSINFER_TENSOR_OUTPUT_META)")
                else:
                    self.logger.info(f"🔍 Found metadata type: {meta_type} (int: {int(meta_type)})")
                
                if int(meta_type) == NVDS_PREPROCESS_BATCH_META:
                    # Validate tensor metadata before extraction
                    if self._validate_tensor_metadata(user_meta.user_meta_data):
                        tensor_meta = user_meta.user_meta_data  # PyCapsule
                        tensor_meta_found = True
                        self.logger.info("✅ Found NVDS_PREPROCESS_BATCH_META PyCapsule")
                        break
                    else:
                        self.logger.warning("❌ Invalid tensor metadata found - skipping")
                try:
                    user_meta_list = user_meta_list.next
                except StopIteration:
                    break
            
            # Scan frame-level user metadata for NVDSINFER_TENSOR_OUTPUT_META (from nvinfer)
            frame_meta_list = batch_meta.frame_meta_list
            frame_user_meta_count = 0
            while frame_meta_list:
                frame_meta = pyds.NvDsFrameMeta.cast(frame_meta_list.data)  # type: ignore
                frame_user_meta_list = frame_meta.frame_user_meta_list
                while frame_user_meta_list:
                    frame_user_meta_count += 1
                    user_meta = pyds.NvDsUserMeta.cast(frame_user_meta_list.data)  # type: ignore
                    meta_type = user_meta.base_meta.meta_type
                    
                    self.logger.debug(f"🔍 Frame meta type: {meta_type}")
                    
                    # Look for NVDSINFER_TENSOR_OUTPUT_META type (this is what nvinfer attaches)
                    if meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                        infer_tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)  # type: ignore
                        if infer_tensor_meta:
                            # Convert NvDsInferTensorMeta to our tensor format if needed
                            self.logger.info("✅ Found inference tensor meta in frame-level NVDSINFER_TENSOR_OUTPUT_META")
                            # For now, we'll use the preprocess tensor meta, but we could also use this
                    try:
                        frame_user_meta_list = frame_user_meta_list.next
                    except StopIteration:
                        break
                if tensor_meta_found:
                    break
                try:
                    frame_meta_list = frame_meta_list.next
                except StopIteration:
                    break
            
            # Phase 1: Log metadata search results
            self.logger.info(f"🔍 Metadata search results: batch_user_meta_count={batch_user_meta_count}, "
                           f"frame_user_meta_count={frame_user_meta_count}, tensor_meta_found={tensor_meta_found}")
            
            # Add detailed debug logging
            self.logger.debug(f"=== METADATA SEARCH DEBUG ===")
            self.logger.debug(f"Expected tensor meta type: {NVDS_PREPROCESS_BATCH_META}")
            self.logger.debug(f"Found batch_user_meta_count: {batch_user_meta_count}")
            self.logger.debug(f"Found frame_user_meta_count: {frame_user_meta_count}")
            self.logger.debug(f"tensor_meta_found: {tensor_meta_found}")
            
            # Process frame metadata
            frame_meta_list = batch_meta.frame_meta_list
            while frame_meta_list:
                fmeta = pyds.NvDsFrameMeta.cast(frame_meta_list.data)  # type: ignore
                detections = self._parse_obj_meta(fmeta)
                
                # Extract tensor data if available
                gpu_tensor = None
                if tensor_meta_found and tensor_meta:
                    gpu_tensor = self._extract_tensor_from_meta(tensor_meta, fmeta.source_id)
                    if gpu_tensor is not None:
                        self.logger.info(f"✅ Extracted valid tensor: shape={gpu_tensor.shape}, dtype={gpu_tensor.dtype}")
                        
                        # Update frame count for statistics
                        self.frame_count += 1
                        
                        # Update FPS monitoring
                        self._update_fps_monitor()
                        
                    else:
                        self.logger.warning("❌ Tensor extraction failed")
                        # Increment error counter
                        self.consecutive_no_meta_count += 1
                else:
                    self.logger.warning("No tensor meta found for frame")
                    # Increment error counter
                    self.consecutive_no_meta_count += 1
                
                frame_dict = {
                    "detections": detections,
                    "frame_num": fmeta.frame_num,
                    "source_id": fmeta.source_id,
                    "timestamp": time.time(),
                    "tensor": gpu_tensor,  # Add tensor to frame data
                }
                
                # Log diagnostic information
                if self.frame_count % 30 == 0:  # Every 30 frames
                    self.logger.info(f"📊 Frame {fmeta.frame_num}: detections={len(detections)}, tensor={'✅' if gpu_tensor is not None else '❌'}, FPS={self.fps_monitor['avg_fps']:.2f}")
                
                # Check pipeline health periodically
                if self.frame_count % 100 == 0:  # Every 100 frames
                    if not self._monitor_pipeline_health():
                        self.logger.warning("⚠️ Pipeline health check failed - considering recovery")
                        if self.error_recovery_enabled:
                            self._attempt_pipeline_recovery()
                
                try:
                    self.tensor_queue.put_nowait(frame_dict)
                    if detections:  # Only log when we have detections
                        self.logger.debug(f"✅ Queued {len(detections)} detections for frame {fmeta.frame_num}")
                except queue.Full:
                    self.logger.warning("tensor_queue full – dropping frame %d", fmeta.frame_num)
                try:
                    frame_meta_list = frame_meta_list.next
                except StopIteration:
                    break
            return Gst.FlowReturn.OK
            
        except Exception as e:
            self.logger.error(f"Error in _on_new_sample: {e}")
            return Gst.FlowReturn.ERROR

    def _preprocess_probe(self, pad, info, u_data):
        """Phase 1: Probe to log buffer metadata from nvdspreprocess sink pad."""
        try:
            gst_buffer = info.get_buffer()
            if not gst_buffer:
                return Gst.PadProbeReturn.OK
                
            # Get batch metadata
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))  # type: ignore
            if not batch_meta:
                self.logger.debug("🔍 Preprocess probe: No batch metadata found")
                return Gst.PadProbeReturn.OK
                
            # Count batch-level user metadata
            batch_user_meta_count = 0
            user_meta_list = batch_meta.batch_user_meta_list
            while user_meta_list:
                batch_user_meta_count += 1
                user_meta = pyds.NvDsUserMeta.cast(user_meta_list.data)  # type: ignore
                meta_type = user_meta.base_meta.meta_type
                self.logger.debug(f"🔍 Preprocess probe: Batch user meta type: {meta_type}")
                
                # Check if this is NVDS_PREPROCESS_BATCH_META
                # NVDS_PREPROCESS_BATCH_META = 27 (based on actual runtime observation)
                if meta_type == 27:
                    self.logger.info("✅ Preprocess probe: Found NVDS_PREPROCESS_BATCH_META in batch-level!")
                    preprocess_batch_meta = pyds.NvDsPreProcessBatchMeta.cast(user_meta.user_meta_data)  # type: ignore
                    if preprocess_batch_meta and preprocess_batch_meta.tensor_meta:
                        tensor_meta = preprocess_batch_meta.tensor_meta
                        self.logger.info(f"✅ Tensor meta details: buffer_size={tensor_meta.buffer_size}, tensor_shape={tensor_meta.tensor_shape}")
                
                try:
                    user_meta_list = user_meta_list.next
                except StopIteration:
                    break
                    
            # Count frame-level user metadata
            frame_user_meta_count = 0
            frame_meta_list = batch_meta.frame_meta_list
            while frame_meta_list:
                frame_meta = pyds.NvDsFrameMeta.cast(frame_meta_list.data)  # type: ignore
                frame_user_meta_list = frame_meta.frame_user_meta_list
                while frame_user_meta_list:
                    frame_user_meta_count += 1
                    user_meta = pyds.NvDsUserMeta.cast(frame_user_meta_list.data)  # type: ignore
                    meta_type = user_meta.base_meta.meta_type
                    self.logger.debug(f"🔍 Preprocess probe: Frame user meta type: {meta_type}")
                    
                    # Check if this is NVDSINFER_TENSOR_OUTPUT_META
                    if meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                        self.logger.info("✅ Preprocess probe: Found NVDSINFER_TENSOR_OUTPUT_META in frame-level!")
                        infer_tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)  # type: ignore
                        self.logger.info(f"✅ Inference tensor meta details: num_output_layers={infer_tensor_meta.num_output_layers}")
                    
                    try:
                        frame_user_meta_list = frame_user_meta_list.next
                    except StopIteration:
                        break
                try:
                    frame_meta_list = frame_meta_list.next
                except StopIteration:
                    break
                    
            # Log metadata counts
            self.logger.debug(f"🔍 Preprocess probe: Batch user meta count: {batch_user_meta_count}, Frame user meta count: {frame_user_meta_count}")
                    
            return Gst.PadProbeReturn.OK
            
        except Exception as e:
            self.logger.error(f"Error in preprocess probe: {e}")
            return Gst.PadProbeReturn.OK
    
    def _analytics_probe(self, pad, info, user_data):
        """Phase 3.1: Probe to capture analytics metadata from nvdsanalytics"""
        try:
            gst_buffer = info.get_buffer()
            if not gst_buffer:
                return Gst.PadProbeReturn.OK
                
            # Get batch metadata
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))  # type: ignore
            if not batch_meta:
                return Gst.PadProbeReturn.OK
                
            # Process frame metadata for analytics
            frame_meta_list = batch_meta.frame_meta_list
            while frame_meta_list:
                frame_meta = pyds.NvDsFrameMeta.cast(frame_meta_list.data)  # type: ignore
                
                # Extract analytics frame metadata
                analytics_frame_data = self._extract_analytics_frame_meta(frame_meta)
                if analytics_frame_data:
                    self.rate_limited_logger.debug(f"📊 Analytics Frame {frame_meta.frame_num}: {analytics_frame_data}")
                
                # Extract analytics object metadata
                obj_meta_list = frame_meta.obj_meta_list
                while obj_meta_list:
                    obj_meta = pyds.NvDsObjectMeta.cast(obj_meta_list.data)  # type: ignore
                    analytics_obj_data = self._extract_analytics_obj_meta(obj_meta)
                    if analytics_obj_data:
                        self.rate_limited_logger.debug(f"📊 Analytics Object {obj_meta.object_id}: {analytics_obj_data}")
                    
                    try:
                        obj_meta_list = obj_meta_list.next
                    except StopIteration:
                        break
                
                try:
                    frame_meta_list = frame_meta_list.next
                except StopIteration:
                    break
                    
            return Gst.PadProbeReturn.OK
            
        except Exception as e:
            self.logger.error(f"Error in analytics probe: {e}")
            return Gst.PadProbeReturn.OK
    
    def _validate_tensor_metadata(self, tensor_meta):
        """Validate tensor metadata before extraction"""
        if not tensor_meta:
            self.logger.debug("❌ Tensor metadata is None")
            return False
        
        # Check required fields
        required_fields = ['buffer_size', 'tensor_shape', 'data_type', 'raw_tensor_buffer']
        for field in required_fields:
            if not hasattr(tensor_meta, field):
                self.logger.debug(f"❌ Missing required field: {field}")
                return False
            
        # Validate tensor shape
        if not tensor_meta.tensor_shape or len(tensor_meta.tensor_shape) == 0:
            self.logger.debug("❌ Invalid tensor shape")
            return False
        
        # Validate buffer size
        if tensor_meta.buffer_size <= 0:
            self.logger.debug(f"❌ Invalid buffer size: {tensor_meta.buffer_size}")
            return False
        
        # Validate GPU pointer
        if not tensor_meta.raw_tensor_buffer:
            self.logger.debug("❌ Invalid GPU pointer")
            return False
        
        # Validate data type
        if tensor_meta.data_type not in [0, 1, 2, 3, 4, 5]:  # Valid DeepStream data types
            self.logger.debug(f"❌ Invalid data type: {tensor_meta.data_type}")
            return False
        
        self.logger.debug("✅ Tensor metadata validation passed")
        return True

    def _validate_gpu_pointer_and_size(self, gpu_ptr, shape, dtype, buffer_size):
        """Validate GPU pointer and size consistency"""
        try:
            # Validate GPU pointer
            if gpu_ptr == 0:
                self.logger.debug("❌ GPU pointer is null")
                return False
            
            # Validate shape
            if not shape or len(shape) == 0:
                self.logger.debug("❌ Invalid tensor shape")
                return False
            
            # Validate dtype
            if dtype is None:
                self.logger.debug("❌ Invalid data type")
                return False
            
            # Validate buffer size matches expected tensor size
            expected_size = np.prod(shape) * dtype.itemsize
            if expected_size != buffer_size:
                self.logger.error(f"❌ Buffer size mismatch: expected {expected_size}, got {buffer_size}")
                return False
            
            # Additional validation for reasonable tensor sizes
            total_elements = np.prod(shape)
            if total_elements <= 0 or total_elements > 100000000:  # 100M elements max
                self.logger.error(f"❌ Unreasonable tensor size: {total_elements} elements")
                return False
            
            self.logger.debug(f"✅ GPU pointer and size validation passed: shape={shape}, size={buffer_size}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error in GPU pointer validation: {e}")
            return False

    def _map_dtype(self, ds_dtype):
        """Map DeepStream data type to PyTorch dtype"""
        dtype_map = {
            0: torch.float32,  # FLOAT
            1: torch.uint8,    # UINT8
            2: torch.int8,     # INT8
            3: torch.uint32,   # UINT32
            4: torch.int32,    # INT32
            5: torch.float16,  # HALF
        }
        return dtype_map.get(ds_dtype, torch.float32)

    def _create_tensor_via_cuda_copy(self, gpu_ptr, shape, dtype):
        """Create PyTorch tensor via CUDA memory copy (safe fallback)"""
        try:
            import torch.cuda as cuda
            
            # Calculate tensor size
            size = np.prod(shape) * dtype.itemsize
            
            # Create PyTorch tensor on GPU
            tensor = torch.empty(shape, dtype=dtype, device='cuda')
            
            # Copy data from GPU pointer to PyTorch tensor
            cuda.memcpy_dtod_async(
                tensor.data_ptr(),
                gpu_ptr,
                size,
                cuda.current_stream()
            )
            
            # Synchronize to ensure copy is complete
            cuda.current_stream().synchronize()
            
            self.logger.debug(f"✅ Created tensor via CUDA copy: shape={shape}, dtype={dtype}")
            return tensor
            
        except Exception as e:
            self.logger.error(f"Failed to create tensor via CUDA copy: {e}")
            return None

    def _create_dlpack_from_gpu_pointer(self, gpu_ptr, shape, dtype):
        """Create proper DLPack capsule from GPU pointer (zero-copy approach)"""
        try:
            # This is a placeholder for proper DLPack integration
            # In a full implementation, you would create a DLManagedTensor capsule
            # For now, fall back to CUDA copy for safety
            self.logger.debug("Using CUDA copy fallback instead of DLPack")
            return self._create_tensor_via_cuda_copy(gpu_ptr, shape, dtype)
            
        except Exception as e:
            self.logger.error(f"Failed to create DLPack from GPU pointer: {e}")
            return None

    def _extract_tensor_from_meta(self, tensor_meta, source_id: int) -> Optional[torch.Tensor]:
        """Extract tensor data from DeepStream tensor metadata (zero-copy GPU)"""
        try:
            self.logger.debug(f"=== TENSOR EXTRACTION DEBUG START ===")
            self.logger.debug(f"Input tensor_meta type: {type(tensor_meta)}")
            
            # Use pyds to get shape as list
            shape = [tensor_meta.tensor_shape[i] for i in range(len(tensor_meta.tensor_shape))]
            dtype = self._map_dtype(tensor_meta.data_type)
            gpu_ptr = tensor_meta.raw_tensor_buffer
            
            self.logger.debug(f"Shape: {shape}")
            self.logger.debug(f"Dtype: {dtype}")
            # Fix ctypes hex() error - safely handle pointer conversion
            ptr_value = gpu_ptr.value if hasattr(gpu_ptr, 'value') else gpu_ptr
            self.logger.debug(f"GPU ptr: {hex(ptr_value) if ptr_value else 'None'}")
            self.logger.debug(f"Buffer size: {tensor_meta.buffer_size}")
            
            # Validate
            if not self._validate_gpu_pointer_and_size(gpu_ptr, shape, dtype, tensor_meta.buffer_size):
                self.logger.debug("=== TENSOR EXTRACTION DEBUG END (VALIDATION FAILED) ===")
                return None
            
            # Create tensor - try DLPack first, fallback to CUDA copy
            tensor = self._create_dlpack_from_gpu_pointer(gpu_ptr, shape, dtype)
            if tensor is None:
                # Fallback to CUDA copy if DLPack fails
                tensor = self._create_tensor_via_cuda_copy(gpu_ptr, shape, dtype)
            
            self.logger.debug(f"Created tensor: shape={tensor.shape}, dtype={tensor.dtype}")
            
            # Reset error counter on successful extraction
            self.consecutive_no_meta_count = 0
            
            # Log success metrics
            self.logger.debug(f"✅ Tensor extraction successful: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
            self.logger.debug("=== TENSOR EXTRACTION DEBUG END (SUCCESS) ===")
            return tensor
            
        except Exception as e:
            self.logger.error(f"❌ Tensor extraction failed: {e}")
            self.logger.debug(f"Exception type: {type(e)}")
            import traceback
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            
            # Increment error counter for health monitoring
            self.consecutive_no_meta_count += 1
            
            # Log detailed error information
            if hasattr(tensor_meta, 'tensor_shape'):
                self.logger.debug(f"Failed tensor shape: {tensor_meta.tensor_shape}")
            if hasattr(tensor_meta, 'data_type'):
                self.logger.debug(f"Failed tensor data type: {tensor_meta.data_type}")
            if hasattr(tensor_meta, 'buffer_size'):
                self.logger.debug(f"Failed tensor buffer size: {tensor_meta.buffer_size}")
            
            self.logger.debug("=== TENSOR EXTRACTION DEBUG END (EXCEPTION) ===")
            return None

    def _extract_analytics_frame_meta(self, frame_meta) -> Optional[Dict[str, Any]]:
        """Extract analytics frame metadata"""
        try:
            user_meta_list = frame_meta.frame_user_meta_list
            while user_meta_list:
                user_meta = pyds.NvDsUserMeta.cast(user_meta_list.data)  # type: ignore
                if user_meta.base_meta.meta_type == pyds.nvds_get_user_meta_type("NVIDIA.DSANALYTICSFRAME.USER_META"):  # type: ignore
                    analytics_frame_meta = pyds.NvDsAnalyticsFrameMeta.cast(user_meta.user_meta_data)  # type: ignore
                    return {
                        'objects_in_roi': analytics_frame_meta.objInROIcnt,
                        'line_crossing_cumulative': analytics_frame_meta.objLCCumCnt,
                        'line_crossing_current': analytics_frame_meta.objLCCurrCnt,
                        'overcrowding_status': analytics_frame_meta.ocStatus
                    }
                try:
                    user_meta_list = user_meta_list.next
                except StopIteration:
                    break
            return None
        except Exception as e:
            self.logger.debug(f"Error extracting analytics frame meta: {e}")
            return None
    
    def _extract_analytics_obj_meta(self, obj_meta) -> Optional[Dict[str, Any]]:
        """Extract analytics object metadata"""
        try:
            user_meta_list = obj_meta.obj_user_meta_list
            while user_meta_list:
                user_meta = pyds.NvDsUserMeta.cast(user_meta_list.data)  # type: ignore
                if user_meta.base_meta.meta_type == pyds.nvds_get_user_meta_type("NVIDIA.DSANALYTICSOBJ.USER_META"):  # type: ignore
                    analytics_obj_meta = pyds.NvDsAnalyticsObjInfo.cast(user_meta.user_meta_data)  # type: ignore
                    return {
                        'direction_status': analytics_obj_meta.dirStatus,
                        'line_crossing_status': analytics_obj_meta.lcStatus,
                        'overcrowding_status': analytics_obj_meta.ocStatus,
                        'roi_status': analytics_obj_meta.roiStatus
                    }
                try:
                    user_meta_list = user_meta_list.next
                except StopIteration:
                    break
            return None
        except Exception as e:
            self.logger.debug(f"Error extracting analytics object meta: {e}")
            return None

    def _monitor_pipeline_health(self) -> bool:
        """Phase 3.4: Monitor pipeline health and performance"""
        try:
            current_time = time.time()
            
            # Check if health check interval has passed
            if current_time - self.last_health_check < self.pipeline_health_check_interval:
                return True
            
            self.last_health_check = current_time
            
            # Check pipeline state
            if self.pipeline:
                state = self.pipeline.get_state(0)  # Non-blocking state check
                if state[1] != Gst.State.PLAYING:
                    self.logger.warning(f"⚠️ Pipeline not in PLAYING state: {state[1]}")
                    return False
            
            # Check FPS performance
            if self.fps_monitor['avg_fps'] < 5.0 and self.frame_count > 100:
                self.fps_monitor['low_fps_count'] += 1
                self.logger.warning(f"⚠️ Low FPS detected: {self.fps_monitor['avg_fps']:.2f} (count: {self.fps_monitor['low_fps_count']})")
                
                if self.fps_monitor['low_fps_count'] >= self.fps_monitor['max_low_fps_count']:
                    self.logger.error("🚨 Persistent low FPS - pipeline may need restart")
                    return False
            else:
                self.fps_monitor['low_fps_count'] = 0
            
            # Check for excessive no-metadata frames
            if self.consecutive_no_meta_count >= self.max_consecutive_no_meta:
                self.logger.error(f"🚨 No metadata for {self.consecutive_no_meta_count} consecutive frames")
                return False
            
            # Log health status
            self.logger.info(f"💚 Pipeline health OK - FPS: {self.fps_monitor['avg_fps']:.2f}, Frames: {self.frame_count}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in pipeline health monitoring: {e}")
            return False
    
    def _attempt_pipeline_recovery(self) -> bool:
        """Phase 3.4: Attempt to recover from pipeline errors"""
        try:
            if not self.error_recovery_enabled:
                return False
            
            if self.pipeline_restart_count >= self.max_pipeline_restarts:
                self.logger.error(f"🚨 Maximum pipeline restarts reached ({self.max_pipeline_restarts})")
                return False
            
            self.logger.info(f"🔄 Attempting pipeline recovery (restart {self.pipeline_restart_count + 1}/{self.max_pipeline_restarts})")
            
            # Stop current pipeline
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
                time.sleep(2)  # Allow cleanup
            
            # Reset counters
            self.consecutive_no_meta_count = 0
            self.fps_monitor['low_fps_count'] = 0
            self.frame_count = 0
            
            # Restart pipeline
            if self._create_pipeline():
                if self.start():
                    self.pipeline_restart_count += 1
                    self.logger.info(f"✅ Pipeline recovery successful")
                    return True
            
            self.logger.error("❌ Pipeline recovery failed")
            return False
            
        except Exception as e:
            self.logger.error(f"Error in pipeline recovery: {e}")
            return False

    def _update_fps_monitor(self) -> None:
        """Update FPS monitoring for health checks"""
        try:
            current_time = time.time()
            if hasattr(self, 'fps_monitor') and self.fps_monitor['last_frame_time']:
                interval = current_time - self.fps_monitor['last_frame_time']
                self.fps_monitor['frame_intervals'].append(interval)
                
                # Keep only last 30 intervals for moving average
                if len(self.fps_monitor['frame_intervals']) > 30:
                    self.fps_monitor['frame_intervals'].pop(0)
                
                # Calculate average FPS
                if len(self.fps_monitor['frame_intervals']) > 0:
                    avg_interval = sum(self.fps_monitor['frame_intervals']) / len(self.fps_monitor['frame_intervals'])
                    self.fps_monitor['avg_fps'] = 1.0 / avg_interval if avg_interval > 0 else 0.0
            
            self.fps_monitor['last_frame_time'] = current_time
            
        except Exception as e:
            self.logger.debug(f"Error updating FPS monitor: {e}")

    def _parse_obj_meta(self, frame_meta) -> List[Dict[str, Any]]:
        """Return list(dict) with keys class_id, confidence, bbox, object_id, and analytics data."""
        detections = []
        l_obj = frame_meta.obj_meta_list
        while l_obj:
            obj = pyds.NvDsObjectMeta.cast(l_obj.data)  # type: ignore
            rect = obj.rect_params
            
            # Basic detection data
            detection = {
                "class_id": obj.class_id,
                "confidence": obj.confidence,
                "bbox": [rect.left, rect.top, rect.width, rect.height],
                "object_id": obj.object_id,
            }
            
            # Phase 3.3: Add analytics metadata if available
            analytics_data = self._extract_analytics_obj_meta(obj)
            if analytics_data:
                detection["analytics"] = analytics_data
            
            # Add secondary inference results if available
            secondary_data = self._extract_secondary_inference_meta(obj)
            if secondary_data:
                detection["secondary_inference"] = secondary_data
            
            detections.append(detection)
            
            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        return detections
    
    def _extract_secondary_inference_meta(self, obj_meta) -> Optional[Dict[str, Any]]:
        """Extract secondary inference metadata from object"""
        try:
            # Look for secondary inference results
            classifier_meta_list = obj_meta.classifier_meta_list
            if classifier_meta_list:
                classifier_meta = pyds.NvDsClassifierMeta.cast(classifier_meta_list.data)  # type: ignore
                if classifier_meta.unique_component_id == 2:  # Our SGIE ID
                    label_info_list = classifier_meta.label_info_list
                    if label_info_list:
                        label_info = pyds.NvDsLabelInfo.cast(label_info_list.data)  # type: ignore
                        return {
                            'classification': label_info.result_label,
                            'confidence': label_info.result_prob,
                            'component_id': classifier_meta.unique_component_id
                        }
            return None
        except Exception as e:
            self.logger.debug(f"Error extracting secondary inference meta: {e}")
            return None

    def _process_detections(self, frame_meta: Any, gpu_tensor: Optional[torch.Tensor] = None) -> None:
        """Process object detections from frame metadata"""
        try:
            detections = []
            
            # Extract object metadata from DeepStream inference
            obj_meta = frame_meta.obj_meta_list
            while obj_meta:
                obj_meta_data = pyds.NvDsObjectMeta.cast(obj_meta.data)  # type: ignore
                
                # Extract bounding box
                rect = obj_meta_data.rect_params
                detection = {
                    'class_id': obj_meta_data.class_id,
                    'confidence': obj_meta_data.confidence,
                    'bbox': [rect.left, rect.top, rect.width, rect.height],
                    'object_id': obj_meta_data.object_id
                }
                detections.append(detection)
                
                try:
                    obj_meta = obj_meta.next
                except StopIteration:
                    break
            
            # Log detection results
            if detections:
                self.logger.info(f"✅ DeepStream detected {len(detections)} objects in frame {frame_meta.frame_num}")
            
            # Send to WebSocket if available (Phase 2: Remove tensor dependencies)
            if self.websocket_server and detections:
                frame_data = {
                    'frame_num': frame_meta.frame_num,
                    'timestamp': time.time(),
                    'detections': detections,
                    'source': 'deepstream_native'  # Phase 2: Mark as native DeepStream inference
                }
                # Use broadcast method that exists in WebSocketServer
                if hasattr(self.websocket_server, 'broadcast_frame'):
                    self.websocket_server.broadcast_frame(frame_data)
                else:
                    self.logger.debug("WebSocket server does not support broadcast_frame method")
                
        except Exception as e:
            self.logger.error(f"Error processing detections: {e}")
    
    def _on_bus_message(self, bus, message):
        """Handle bus messages."""
        msg_type = message.type
        
        if msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            self.logger.error(f"🚨 Pipeline error: {err.message}")
            self.logger.error(f"🚨 Debug info: {debug}")
            self.logger.error(f"🚨 Error source: {message.src.get_name() if message.src else 'unknown'}")
            self.running = False
            if self.mainloop:
                self.mainloop.quit()
        
        elif msg_type == Gst.MessageType.EOS:
            self.logger.info("End of stream")
            self.running = False
            if self.mainloop:
                self.mainloop.quit()
        
        elif msg_type == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            self.logger.warning(f"⚠️  Pipeline warning: {warn.message}")
            self.logger.warning(f"⚠️  Debug info: {debug}")
        
        elif msg_type == Gst.MessageType.INFO:
            info, debug = message.parse_info()
            self.logger.info(f"ℹ️  Pipeline info: {info.message}")
        
        elif msg_type == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old_state, new_state, pending_state = message.parse_state_changed()
                self.logger.info(f"🔄 Pipeline state changed: {old_state.value_nick} → {new_state.value_nick}")
        
        return True
    
    def start(self) -> bool:
        """Start the DeepStream pipeline."""
        try:
            # Create pipeline if not already created
            if not self.pipeline:
                if not self._create_pipeline():
                    return False
            
            # Create main loop
            self.mainloop = GLib.MainLoop()
            
            # Start pipeline - first go to READY to allow pad creation
            self.logger.info("🔍 Setting pipeline to READY state...")
            if self.pipeline is not None:
                ret = self.pipeline.set_state(Gst.State.READY)
                self.logger.info(f"🔍 READY state change result: {ret}")
                
                if ret == Gst.StateChangeReturn.FAILURE:
                    self.logger.error("❌ Failed to set pipeline to READY state")
                    return False
                
                # Wait a moment for pads to be created
                import time
                time.sleep(0.1)
                
                # Now go to PLAYING
                self.logger.info("🔍 Setting pipeline to PLAYING state...")
                ret = self.pipeline.set_state(Gst.State.PLAYING)
                self.logger.info(f"🔍 PLAYING state change result: {ret}")
                
                if ret == Gst.StateChangeReturn.FAILURE:
                    self.logger.error("❌ Failed to start pipeline - immediate failure")
                    return False
                elif ret == Gst.StateChangeReturn.ASYNC:
                    self.logger.info("🔍 Pipeline state change is async, waiting...")
                    # Wait for state change to complete with longer timeout for TensorRT + RTSP
                    ret2 = self.pipeline.get_state(30 * Gst.SECOND)  # 30 second timeout for TensorRT initialization
                    self.logger.info(f"🔍 Final state change result: {ret2}")
                    if ret2[0] == Gst.StateChangeReturn.FAILURE:
                        self.logger.error("❌ Failed to start pipeline - async failure")
                        return False
                    elif ret2[0] == Gst.StateChangeReturn.ASYNC:
                        self.logger.warning("⚠️ Pipeline state change timed out, but continuing...")
                        # For RTSP sources, timeout is often normal - continue anyway
                        pass
            
            self.running = True
            self.start_time = time.time()
            
            # Run main loop in thread
            self.mainloop_thread = threading.Thread(target=self.mainloop.run, daemon=True)
            self.mainloop_thread.start()
            
            self.logger.info("DeepStream pipeline started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start pipeline: {e}")
            return False
    
    def read_gpu_tensor(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Read next GPU tensor from the pipeline.
        
        Returns:
            Tuple of (success, tensor_dict) where tensor_dict contains
            'tensor', 'source_id', 'frame_num', 'timestamp'
        """
        if not self.running:
            return False, None
        
        try:
            # Get tensor with timeout
            tensor_data = self.tensor_queue.get(timeout=0.1)
            return True, tensor_data
        except queue.Empty:
            return False, None
        except Exception as e:
            self.logger.error(f"Error reading tensor: {e}")
            return False, None
    
    def stop(self):
        """Stop the DeepStream pipeline."""
        self.logger.info("Stopping DeepStream pipeline")
        self.running = False
        
        # Stop pipeline
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        
        # Stop main loop
        if self.mainloop:
            self.mainloop.quit()
        
        # Wait for main loop thread
        if hasattr(self, 'mainloop_thread') and self.mainloop_thread.is_alive():
            self.mainloop_thread.join(timeout=2.0)
        
        self.logger.info("DeepStream pipeline stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        runtime = time.time() - self.start_time if self.running else 0
        fps = self.frame_count / runtime if runtime > 0 else 0
        
        return {
            'pipeline_type': 'deepstream',
            'running': self.running,
            'frames_processed': self.frame_count,
            'fps': fps,
            'runtime_seconds': runtime,
            'batch_size': self.batch_size,
            'sources': len(self.sources),
            'queue_size': self.tensor_queue.qsize()
        }


def create_deepstream_video_processor(
    camera_id: str,
    source: Union[str, int, Dict],
    config: AppConfig
) -> DeepStreamVideoPipeline:
    """
    Factory function to create DeepStream video processor.
    
    Args:
        camera_id: Camera identifier
        source: Video source (file path, RTSP URL, or dict config)
        config: Application configuration
        
    Returns:
        Configured DeepStream video pipeline
    """
    # Convert source to URL string
    if isinstance(source, dict):
        source_url = source.get('url', '')
    else:
        source_url = str(source)
    
    return DeepStreamVideoPipeline(
        rtsp_url=source_url,
        config=config,
        websocket_port=8765,
        config_file="config_infer_primary_yolo11.txt",
        preproc_config="config_preproc.txt"
    )


if __name__ == "__main__":
    # Test DeepStream pipeline
    import argparse
    from config import config
    
    parser = argparse.ArgumentParser(description="Test DeepStream Video Pipeline")
    parser.add_argument("--source", required=True, help="Video source (RTSP URL or file)")
    parser.add_argument("--duration", type=int, default=10, help="Test duration in seconds")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Create pipeline
    pipeline = create_deepstream_video_processor("test_cam", args.source, config)
    
    if pipeline.start():
        print("✅ DeepStream pipeline started successfully")
        
        # Read frames for specified duration
        start = time.time()
        frame_count = 0
        last_detection_log_time = 0
        detection_log_interval = 5  # seconds
        
        while time.time() - start < args.duration:
            ret, tensor_data = pipeline.read_gpu_tensor()
            if ret and tensor_data:
                detections = tensor_data['detections']
                current_time = time.time()
                
                # Only log detections when count > 0 and enough time has passed
                if len(detections) > 0 and (current_time - last_detection_log_time) >= detection_log_interval:
                    print(f"🔍 Detected {len(detections)} objects at {current_time - start:.1f}s")
                    last_detection_log_time = current_time
                
                frame_count += 1
            else:
                time.sleep(0.01)
        
        # Print statistics
        stats = pipeline.get_stats()
        print(f"\nPipeline statistics: {stats}")
        
        pipeline.stop()
    else:
        print("❌ Failed to start DeepStream pipeline")