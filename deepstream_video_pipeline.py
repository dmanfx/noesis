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

# Bypass libproxy issues by disabling GIO proxy resolver
# import os  # Removed duplicate import

# Add ctypes imports for PyCapsule handling
import ctypes
from ctypes import c_void_p, c_uint64, c_uint32, c_int, POINTER, Structure, c_char_p, c_float

# ctypes import removed – no longer needed after switching to tensor meta path
from typing import Optional, Tuple, Dict, Any, List, Union

#import numpy as np
import torch
import cupy as cp  # For zero-copy GPU tensor handling
import math
from torch.utils import dlpack  # Re-imported for explicit dlpack module usage

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
# Remove redundant TensorRT imports - DeepStream handles inference natively
# from cuda_context_manager import CudaContextManager  # Commented out - not available
# from tensorrt_inference import GPUOnlyDetectionManager  # Removed - redundant with DeepStream
# from gpu_pipeline import UnifiedGPUPipeline  # Removed - redundant with DeepStream

# Initialize GStreamer
Gst.init(None)

from config import AppConfig  # noqa: E402
# from gpu_memory_pool import get_global_memory_pool  # Unused import

# DeepStream metadata type constants
NVDS_PREPROCESS_BATCH_META = 27
NVDSINFER_TENSOR_OUTPUT_META = 12

# NOTE: DeepStream preprocessing metadata uses C++ STL containers (std::vector, std::string)
# which cannot be directly accessed via ctypes. The custom preprocessing library
# works correctly and passes tensor data to nvinfer, but we cannot extract it from Python.
# 
# Original C++ structures for reference:
# - NvDsPreProcessTensorMeta contains std::vector<int> tensor_shape and std::string tensor_name
# - GstNvDsPreProcessBatchMeta contains std::vector<guint64> target_unique_ids and std::vector<NvDsRoiMeta> roi_vector
# 
# These cannot be mapped with ctypes, so we skip tensor extraction and let nvinfer use the data directly.

# Define ctypes structures matching nvdspreprocess_meta.h
class StdVectorInt(Structure):
    _fields_ = [
        ("begin", c_void_p),
        ("end", c_void_p),
        ("capacity", c_void_p)
    ]

class StdVectorULong(Structure):
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
        ("target_unique_ids", StdVectorULong),  # std::vector<guint64>
        ("tensor_meta", ctypes.POINTER(NvDsPreProcessTensorMeta)),  # NvDsPreProcessTensorMeta*
        ("roi_vector", ctypes.c_void_p),  # std::vector<NvDsRoiMeta> - pointer to vector
        ("private_data", ctypes.c_void_p),  # void*
    ]


# Custom callback functions for preprocessing metadata
def custom_preprocess_copy_func(data, user_data):
    """Custom copy function for preprocessing metadata - simple pass-through"""
    return data

def custom_preprocess_release_func(data, user_data):
    """Custom release function for preprocessing metadata - simple cleanup"""
    # No deep cleanup needed - DeepStream manages the memory
    pass


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
        # Use rate-limited loggers for different types of messages
        self.rate_limited_logger = RateLimitedLogger(self.logger, rate_limit_seconds=5.0)
        self.metadata_logger = RateLimitedLogger(self.logger, rate_limit_seconds=5.0)
        self.tensor_logger = RateLimitedLogger(self.logger, rate_limit_seconds=2.0)
        self.detection_logger = RateLimitedLogger(self.logger, rate_limit_seconds=1.0)
        
        # Initialize pipeline components
        self.pipeline: Optional[Gst.Pipeline] = None
        self.mainloop: Optional[GLib.MainLoop] = None
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
        self.frame_count = 0
        self.frame_count_lock = threading.Lock()
        # The tensor_queue is no longer needed as metadata is extracted via probe
        # self.tensor_queue = queue.Queue(maxsize=30) 
        # JPEG queue for GPU-encoded frames when native OSD is enabled
        self.jpeg_queue: queue.Queue[bytes] = queue.Queue(maxsize=30)
        self.start_time = 0  # Will be set in start()
        
        # Initialize tracking state for telemetry
        self.live_tracking_state: Dict[str, Any] = {
            'active_tracks': [],
            'occupancy': {},
            'transitions': []
        }
        
        # Initialize GStreamer
        Gst.init(None)
        
        # Create pipeline elements
        self._create_pipeline()

    def _is_pycapsule(self, obj) -> bool:
        """Check if object is a PyCapsule"""
        return hasattr(obj, '__class__') and 'PyCapsule' in str(obj.__class__)
    
    def _validate_tensor_meta(self, meta_data, meta_type: int) -> bool:
        """Validate tensor metadata before extraction"""
        if meta_type != NVDS_PREPROCESS_BATCH_META:
                    return False
        # Check if PyCapsule
        if 'PyCapsule' not in str(type(meta_data)):
            self.logger.warning("Not a PyCapsule")
            return False
        return True

    def _log_available_metadata_types(self):
        """Log all available metadata types for debugging"""
        self.logger.info("=== AVAILABLE METADATA TYPES ===")
        try:
            for attr in dir(pyds.NvDsMetaType):
                if not attr.startswith('_') and hasattr(getattr(pyds.NvDsMetaType, attr), '__int__'):
                    value = getattr(pyds.NvDsMetaType, attr)
                    self.logger.info(f"  {attr}: {int(value)}")
        except Exception as e:
            self.logger.error(f"Error logging metadata types: {e}")
        self.logger.info("=== END METADATA TYPES ===")

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

    def update_detection_config(self, config_data: Dict[str, Any]) -> bool:
        """Update DeepStream detection configuration in real-time using GObject properties
        
        Args:
            config_data: Dictionary containing detection configuration updates
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            if not hasattr(self, 'nvinfer') or not self.nvinfer:
                self.logger.warning("nvinfer element not available for dynamic configuration")
                return False
            
            success = True
            
            # Update confidence threshold
            if 'confidence_threshold' in config_data:
                try:
                    new_threshold = float(config_data['confidence_threshold'])
                    self.nvinfer.set_property("confidence-threshold", new_threshold)
                    self.logger.info(f"✅ Updated DeepStream confidence threshold to: {new_threshold}")
                except Exception as e:
                    self.logger.error(f"❌ Failed to update confidence threshold: {e}")
                    success = False
            
            # Update IOU threshold
            if 'iou_threshold' in config_data:
                try:
                    new_iou = float(config_data['iou_threshold'])
                    self.nvinfer.set_property("iou-threshold", new_iou)
                    self.logger.info(f"✅ Updated DeepStream IOU threshold to: {new_iou}")
                except Exception as e:
                    self.logger.error(f"❌ Failed to update IOU threshold: {e}")
                    success = False
            
            # Update detection enable/disable
            if 'detection_enabled' in config_data:
                try:
                    detection_enabled = bool(config_data['detection_enabled'])
                    self.nvinfer.set_property("enable", detection_enabled)
                    self.logger.info(f"✅ Updated DeepStream detection enabled to: {detection_enabled}")
                except Exception as e:
                    self.logger.error(f"❌ Failed to update detection enabled: {e}")
                    success = False
            
            # Update target classes via custom properties
            if 'target_classes' in config_data:
                try:
                    new_classes = config_data['target_classes']
                    if isinstance(new_classes, list):
                        class_string = ','.join(map(str, new_classes))
                        self.nvinfer.set_property("custom-lib-props", f"target-classes:{class_string}")
                        self.logger.info(f"✅ Updated DeepStream target classes to: {new_classes}")
                except Exception as e:
                    self.logger.error(f"❌ Failed to update target classes: {e}")
                    success = False
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Error in update_detection_config: {e}")
            return False

    def update_detection_toggle(self, toggle_name: str, enabled: bool) -> bool:
        """Update specific detection toggles using DeepStream GObject properties
        
        Args:
            toggle_name: Name of the detection toggle to update
            enabled: Whether the detection should be enabled
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            if not hasattr(self, 'nvinfer') or not self.nvinfer:
                self.logger.warning("nvinfer element not available for dynamic configuration")
                return False
            
            # Get current target classes from nvinfer custom properties
            current_classes = []
            try:
                custom_props = self.nvinfer.get_property("custom-lib-props")
                if custom_props and "target-classes:" in custom_props:
                    class_string = custom_props.split("target-classes:")[1]
                    current_classes = [int(x) for x in class_string.split(',') if x.strip()]
            except (AttributeError, TypeError, ValueError):
                # If no custom properties set, assume all classes are enabled
                current_classes = list(range(80))  # COCO has 80 classes
            
            # Update classes based on toggle
            if toggle_name == 'detect_people':
                class_id = 0  # person class
                if enabled and class_id not in current_classes:
                    current_classes.append(class_id)
                elif not enabled and class_id in current_classes:
                    current_classes.remove(class_id)
                    
            elif toggle_name == 'detect_vehicles':
                vehicle_classes = [1, 2, 3, 5, 7, 8]  # bicycle, car, motorcycle, bus, truck, boat
                if enabled:
                    for class_id in vehicle_classes:
                        if class_id not in current_classes:
                            current_classes.append(class_id)
                else:
                    for class_id in vehicle_classes:
                        if class_id in current_classes:
                            current_classes.remove(class_id)
                            
            elif toggle_name == 'detect_furniture':
                furniture_classes = [13, 56, 57, 59, 60, 61]  # bench, chair, couch, bed, dining table, toilet
                if enabled:
                    for class_id in furniture_classes:
                        if class_id not in current_classes:
                            current_classes.append(class_id)
                else:
                    for class_id in furniture_classes:
                        if class_id in current_classes:
                            current_classes.remove(class_id)
            
            # Update DeepStream with new target classes
            if current_classes:
                class_string = ','.join(map(str, current_classes))
                self.nvinfer.set_property("custom-lib-props", f"target-classes:{class_string}")
                self.logger.info(f"✅ Updated DeepStream target classes for {toggle_name}: {current_classes}")
                return True
            else:
                self.logger.warning(f"⚠️ No classes selected for {toggle_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error in update_detection_toggle: {e}")
            return False

    def get_current_detection_config(self) -> Dict[str, Any]:
        """Get current DeepStream detection configuration
        
        Returns:
            Dict containing current detection settings
        """
        try:
            config = {}
            
            if hasattr(self, 'nvinfer') and self.nvinfer:
                # Get confidence threshold
                try:
                    config['confidence_threshold'] = self.nvinfer.get_property("confidence-threshold")
                except (AttributeError, TypeError, ValueError):
                    config['confidence_threshold'] = 0.3
                
                # Get IOU threshold
                try:
                    config['iou_threshold'] = self.nvinfer.get_property("iou-threshold")
                except (AttributeError, TypeError, ValueError):
                    config['iou_threshold'] = 0.45
                
                # Get detection enabled status
                try:
                    config['detection_enabled'] = self.nvinfer.get_property("enable")
                except (AttributeError, TypeError, ValueError):
                    config['detection_enabled'] = True
                
                # Get target classes
                try:
                    custom_props = self.nvinfer.get_property("custom-lib-props")
                    if custom_props and "target-classes:" in custom_props:
                        class_string = custom_props.split("target-classes:")[1]
                        config['target_classes'] = [int(x) for x in class_string.split(',') if x.strip()]
                    else:
                        config['target_classes'] = list(range(80))  # All COCO classes
                except (AttributeError, TypeError, ValueError):
                    config['target_classes'] = list(range(80))
            
            return config
            
        except Exception as e:
            self.logger.error(f"❌ Error getting current detection config: {e}")
            return {}

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
        """Create the DeepStream GStreamer pipeline using a consolidated, linear assembly approach."""
        try:
            # --- Phase A: Create All Elements ---
            self.logger.info("------------- Creating GStreamer Elements-------------")
            self.pipeline = Gst.Pipeline()
            if not self.pipeline:
                raise RuntimeError("Failed to create pipeline")

            element_names = [
                "nvstreammux", "nvdspreprocess", "nvinfer", "nvtracker", "nvdsanalytics", 
                "nvvideoconvert", "nvdsosd", "nvvidconv_post_osd", "caps_post_osd", 
                "nvjpegenc_gpu", "queue_jpeg", "appsink_jpeg"
            ]
            elements = [Gst.ElementFactory.make(name, name) for name in 
                        ["nvstreammux", "nvdspreprocess", "nvinfer", "nvtracker", "nvdsanalytics", 
                         "nvvideoconvert", "nvdsosd", "nvvideoconvert", "capsfilter", 
                         "nvjpegenc", "queue", "appsink"]]
            
            # Assign unique names
            for i, el in enumerate(elements):
                if el:
                    el.set_name(element_names[i])

            streammux, preprocess, nvinfer, nvtracker, nvanalytics, nvvidconv, nvosd, \
            nvvidconv_post, caps_post_osd, jpegenc_gpu, queue_jpeg, appsink_jpeg = elements

            if not all(elements):
                for el, name in zip(elements, element_names):
                    if not el: self.logger.error(f"❌ Failed to create element: {name}")
                raise RuntimeError("Failed to create one or more GStreamer elements.")
            self.logger.info("✅ All GStreamer elements created successfully.")

            # --- Phase B: Configure Elements ---
            self.logger.info("------------- Configuring GStreamer Elements-------------")
            
            # Bus and metadata callbacks
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self._on_bus_message)
            pyds.register_user_copyfunc(custom_preprocess_copy_func)
            pyds.register_user_releasefunc(custom_preprocess_release_func)

            # Streammux configuration
            streammux.set_property("batch-size", self.batch_size)
            streammux.set_property("width", 1280)
            streammux.set_property("height", 720)
            streammux.set_property("batched-push-timeout", 4000000)
            streammux.set_property("live-source", 1)

            # Preprocess configuration
            preprocess.set_property("config-file", self.config.processing.DEEPSTREAM_PREPROCESS_CONFIG)
            
            # Nvinfer configuration
            self._check_for_engine_file(self.config_file)
            nvinfer.set_property("config-file-path", self.config_file)
            nvinfer.set_property("input-tensor-meta", True)

            # Nvtracker configuration
            nvtracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
            tracker_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_tracker_nvdcf_batch.yml")
            nvtracker.set_property("ll-config-file", tracker_config_path)

            # Nvdsanalytics configuration
            nvanalytics.set_property("config-file", "config_nvdsanalytics.txt")
                
            # OSD Sink Pad Probe for metadata
            osd_src_pad = nvosd.get_static_pad("src")
            if not osd_src_pad: raise RuntimeError("Failed to get OSD source pad")
            osd_src_pad.add_probe(Gst.PadProbeType.BUFFER, self.osd_sink_pad_buffer_probe, 0)
            self.logger.info("✅ Added buffer probe to OSD source pad for metadata extraction")

            # JPEG sink configuration
            caps_post_osd.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))
            appsink_jpeg.set_property("emit-signals", True)
            appsink_jpeg.set_property("sync", False)
            appsink_jpeg.set_property("max-buffers", 5)
            appsink_jpeg.set_property("drop", True)
            appsink_jpeg.connect("new-sample", self._on_new_jpeg_sample)

            # --- Phase C: Add and Link Elements ---
            self.logger.info("------------- Adding and Linking All Elements-------------")
            
            # Add all elements to the pipeline
            for el in elements:
                self.pipeline.add(el)
            
            # Create and add the source bin
            source_bin = self._create_source_bin(0, {'url': self.rtsp_url})
            if not source_bin: raise RuntimeError("Failed to create source bin")
            self.pipeline.add(source_bin)
            
            # Link source to streammux
            source_pad = source_bin.get_static_pad("src")
            sink_pad = streammux.get_request_pad("sink_0")
            if source_pad.link(sink_pad) != Gst.PadLinkReturn.OK:
                raise RuntimeError("Failed to link source bin to streammux")
            
            # Link the main processing chain
            if not streammux.link(preprocess): raise RuntimeError("Failed to link streammux to preprocess")
            if not preprocess.link(nvinfer): raise RuntimeError("Failed to link preprocess to nvinfer")
            if not nvinfer.link(nvtracker): raise RuntimeError("Failed to link nvinfer to nvtracker")
            if not nvtracker.link(nvanalytics): raise RuntimeError("Failed to link nvtracker to nvanalytics")
            if not nvanalytics.link(nvvidconv): raise RuntimeError("Failed to link nvanalytics to nvvidconv")
            if not nvvidconv.link(nvosd): raise RuntimeError("Failed to link nvvidconv to nvosd")
            
            # Link the JPEG encoding sink chain
            if not nvosd.link(nvvidconv_post): raise RuntimeError("Failed to link nvosd to nvvidconv_post")
            if not nvvidconv_post.link(caps_post_osd): raise RuntimeError("Failed to link nvvidconv_post to caps_post_osd")
            if not caps_post_osd.link(jpegenc_gpu): raise RuntimeError("Failed to link caps_post_osd to jpegenc_gpu")
            if not jpegenc_gpu.link(queue_jpeg): raise RuntimeError("Failed to link jpegenc_gpu to queue_jpeg")
            if not queue_jpeg.link(appsink_jpeg): raise RuntimeError("Failed to link queue_jpeg to appsink_jpeg")
            
            # --- Finalization ---
            self.logger.info("✅ Pipeline construction complete.")
            
            # Store references
            self.streammux, self.preprocess, self.nvinfer, self.nvtracker, self.nvanalytics, \
            self.nvvidconv, self.nvosd = streammux, preprocess, nvinfer, nvtracker, nvanalytics, nvvidconv, nvosd
            self.jpeg_appsink = appsink_jpeg
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create pipeline: {e}")
            import traceback
            self.logger.error(f"Full Traceback: {traceback.format_exc()}")
            return False
    
    def _create_source_bin(self, index: int, source_config: Dict[str, Any]) -> Gst.Element:
        """Create source bin for RTSP stream."""
        try:
            # Create source bin
            source_bin = Gst.Bin.new(f"source-bin-{index}")
            
            # Create source element based on config
            if isinstance(source_config, str):
                # RTSP URL string
                source = Gst.ElementFactory.make("rtspsrc", f"rtspsrc-{index}")
                source.set_property("location", source_config)
                source.set_property("latency", 0)
                source.set_property("drop-on-latency", True)
                source.set_property("ntp-sync", False)
                source.set_property("ntp-time-source", 3)  # Running time
                source.set_property("buffer-mode", 4)  # Auto
                source.set_property("protocols", "tcp")
                source.set_property("tcp-timeout", 5000000)  # 5 seconds
                source.set_property("retry", 3)
                source.set_property("timeout", 5)
                
                self.logger.info(f"📡 Created RTSP source for: {source_config}")
            else:
                # Dictionary config
                source = Gst.ElementFactory.make("rtspsrc", f"rtspsrc-{index}")
                source.set_property("location", source_config["url"])
                source.set_property("latency", 0)
                source.set_property("drop-on-latency", True)
                source.set_property("ntp-sync", False)
                source.set_property("ntp-time-source", 3)
                source.set_property("buffer-mode", 4)
                source.set_property("protocols", "tcp")
                source.set_property("tcp-timeout", 5000000)
                source.set_property("retry", 3)
                source.set_property("timeout", 5)
                
                self.logger.info(f"📡 Created RTSP source for: {source_config['url']}")
            
            # Create depay element
            depay = Gst.ElementFactory.make("rtph264depay", f"depay-{index}")
            if not depay:
                self.logger.error("Failed to create rtph264depay element")
                return None
            
            # Create decoder
            decoder = Gst.ElementFactory.make("nvv4l2decoder", f"decoder-{index}")
            if not decoder:
                self.logger.error("Failed to create nvv4l2decoder element")
                return None
            
            # Create converter
            converter = Gst.ElementFactory.make("nvvideoconvert", f"converter-{index}")
            if not converter:
                self.logger.error("Failed to create nvvideoconvert element")
                return None
        
            # Create capsfilter
            caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12")
            capsfilter = Gst.ElementFactory.make("capsfilter", f"capsfilter-{index}")
            capsfilter.set_property("caps", caps)
            
            # Add elements to bin
            source_bin.add(source)
            source_bin.add(depay)
            source_bin.add(decoder)
            source_bin.add(converter)
            source_bin.add(capsfilter)
            
            # Link elements
            depay.link(decoder)
            decoder.link(converter)
            converter.link(capsfilter)
            
            # Handle dynamic pad from rtspsrc
            def cb_newpad(decodebin, decoder_src_pad, data):
                self.logger.info(f"📡 New pad from rtspsrc: {decoder_src_pad.get_name()}")
                caps = decoder_src_pad.get_current_caps()
                if not caps:
                    caps = decoder_src_pad.query_caps(None)
                structure = caps.get_structure(0)
                name = structure.get_name()
                
                if name.find("application/x-rtp") != -1:
                    sink_pad = depay.get_static_pad("sink")
                    if not sink_pad.is_linked():
                        ret = decoder_src_pad.link(sink_pad)
                        if ret != Gst.PadLinkReturn.OK:
                            self.logger.error(f"Failed to link rtspsrc to depay: {ret}")
                    else:
                        self.logger.info("✅ Successfully linked rtspsrc to depay")
                else:
                    self.logger.warning(f"Unexpected pad type: {name}")
            
            source.connect("pad-added", cb_newpad, None)
            
            # Set ghost pad
            ghost_pad = Gst.GhostPad.new("src", capsfilter.get_static_pad("src"))
            source_bin.add_pad(ghost_pad)
            
            self.logger.info(f"✅ Created source bin {index} successfully")
            return source_bin
            
        except Exception as e:
            self.logger.error(f"Error creating source bin {index}: {e}")
            return None

    def _on_new_sample(self, appsink: GstApp.AppSink) -> Gst.FlowReturn:
        """DEPRECATED - This function is no longer called. 
        Metadata is handled by the OSD pad probe.
        This function is kept for reference but should not be used.
        """
        self.logger.warning("_on_new_sample is deprecated and should not be called.")
        return Gst.FlowReturn.OK
    
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
                    self.rate_limited_logger.debug(
                        f"📊 Analytics Frame {frame_meta.frame_num}: {analytics_frame_data}"
                    )

                # Extract analytics object metadata
                obj_meta_list = frame_meta.obj_meta_list
                while obj_meta_list:
                    obj_meta = pyds.NvDsObjectMeta.cast(obj_meta_list.data)  # type: ignore
                    analytics_obj_data = self._extract_analytics_obj_meta(obj_meta)
                    if analytics_obj_data:
                        self.rate_limited_logger.debug(
                            f"📊 Analytics Object {obj_meta.object_id}: {analytics_obj_data}"
                        )

                    try:
                        obj_meta_list = obj_meta_list.next
                    except StopIteration:
                        break

                # Update live tracking state using existing parsing logic
                # Only needed when using native OSD mode because _on_new_sample
                # already updates tracking for the Python appsink path
                if self.config.visualization.USE_NATIVE_DEEPSTREAM_OSD:
                    try:
                        self._parse_obj_meta(frame_meta)
                    except Exception as e:
                        self.logger.debug(
                            f"Error updating tracking state from analytics probe: {e}"
                        )

                try:
                    frame_meta_list = frame_meta_list.next
                except StopIteration:
                    break
                    
            return Gst.PadProbeReturn.OK
            
        except Exception as e:
            self.logger.error(f"Error in analytics probe: {e}")
            return Gst.PadProbeReturn.OK
    
    def _extract_tensor_from_meta(self, tensor_meta, source_id: int) -> Optional[torch.Tensor]:
        """Extract tensor from custom preprocessing metadata (type 27)"""
        try:
            if not tensor_meta:
                return None
                
            # Conditional debug logging (every 100th frame only)
            if self.frame_count % 100 == 0:
                self.logger.debug(f"=== CUSTOM PREPROCESS TENSOR EXTRACTION START ===")
                self.logger.debug(f"Input tensor_meta type: {type(tensor_meta)}")
            
            # Check if this is a PyCapsule (raw C pointer)
            if not self._is_pycapsule(tensor_meta):
                self.logger.error("Not a PyCapsule - cannot extract tensor")
                return None
                
            # Extract raw C pointer from PyCapsule
            ctypes.pythonapi.PyCapsule_GetPointer.restype = c_void_p
            ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
            raw_ptr = ctypes.pythonapi.PyCapsule_GetPointer(tensor_meta, None)
            
            if not raw_ptr:
                self.logger.error("Failed to extract pointer from PyCapsule")
                return None
                
            if self.frame_count % 100 == 0:
                self.logger.debug(f"Raw pointer extracted: {hex(raw_ptr)}")
            
            # Cast to GstNvDsPreProcessBatchMeta structure
            batch_meta = ctypes.cast(raw_ptr, ctypes.POINTER(GstNvDsPreProcessBatchMeta)).contents
            
            # Get tensor meta pointer
            if not batch_meta.tensor_meta:
                self.logger.error("tensor_meta pointer is None")
                return None
                
            # Access the first tensor meta (assuming single tensor per batch)
            tensor_meta_struct = batch_meta.tensor_meta.contents
            
            # Extract tensor shape from std::vector<int>
            shape_vector = tensor_meta_struct.tensor_shape
            if not shape_vector.begin or not shape_vector.end:
                self.logger.error("Shape vector begin or end is None")
                return None
                
            # Calculate shape size
            size = (int(shape_vector.end) - int(shape_vector.begin)) // ctypes.sizeof(ctypes.c_int)
            
            if size <= 0 or size > 10:  # Sanity check
                self.logger.error(f"Invalid shape size: {size}")
                return None
                
            # Extract shape array
            shape_array = (ctypes.c_int * size).from_address(int(shape_vector.begin))
            shape = list(shape_array)
            
            # Map data type
            dtype_map = {
                0: torch.float32,   # FLOAT32
                1: torch.uint8,     # UINT8
                2: torch.int8,      # INT8
                3: torch.uint32,    # UINT32
                4: torch.int32,     # INT32
                5: torch.float16,   # FP16
            }
            dtype = dtype_map.get(tensor_meta_struct.data_type, torch.float32)
            
            # Extract tensor data
            if not tensor_meta_struct.raw_tensor_buffer:
                self.logger.error("raw_tensor_buffer is None")
                return None
                
            # Get buffer size and validate
            buffer_size = tensor_meta_struct.buffer_size
            expected_size = math.prod(shape) * dtype.itemsize
            
            if buffer_size != expected_size:
                self.rate_limited_logger.warning(f"Buffer size mismatch: {buffer_size} vs {expected_size}")
            
            # Create zero-copy GPU tensor using CuPy + DLPack
            ptr = int(tensor_meta_struct.raw_tensor_buffer)
            if ptr == 0:
                self.logger.error("Invalid tensor buffer pointer")
                return None
                
            # Map to CuPy dtype
            cp_dtype_map = {
                0: cp.float32,   # FLOAT32
                1: cp.uint8,     # UINT8
                2: cp.int8,      # INT8
                3: cp.uint32,    # UINT32
                4: cp.int32,     # INT32
                5: cp.float16,   # FP16
            }
            cp_dtype = cp_dtype_map.get(tensor_meta_struct.data_type, cp.float32)
            
            # Wrap device memory without copy
            mem = cp.cuda.UnownedMemory(ptr, buffer_size, self)
            memptr = cp.cuda.MemoryPointer(mem, 0)
            cp_array = cp.ndarray(shape, dtype=cp_dtype, memptr=memptr)
            
            # Convert to torch tensor via DLPack (zero-copy)
            torch_tensor = torch.utils.dlpack.from_dlpack(cp_array.toDlpack())
            
            # Remove batch dimension: [1, 3, 640, 640] -> [3, 640, 640]
            torch_tensor = torch_tensor.squeeze(0)
            
            if self.frame_count % 100 == 0:
                self.logger.debug(f"✅ Successfully extracted tensor: {torch_tensor.shape}, {torch_tensor.dtype}")
                self.logger.debug(f"=== CUSTOM PREPROCESS TENSOR EXTRACTION END (SUCCESS) ===")
            return torch_tensor
            
        except Exception as e:
            self.logger.error(f"Error extracting tensor from custom metadata: {e}")
            if self.frame_count % 100 == 0:
                import traceback
                self.logger.debug(f"Traceback: {traceback.format_exc()}")
            self.logger.debug(f"=== CUSTOM PREPROCESS TENSOR EXTRACTION END (FAILED) ===")
            return None

    # ------------------------------------------------------------------
    # Inference tensor (NVDSINFER_TENSOR_OUTPUT_META) → GPU torch tensor
    # ------------------------------------------------------------------
    def _extract_tensor_from_infer_meta(self, infer_tensor_meta) -> Optional[torch.Tensor]:
        """Zero-copy GPU tensor extraction using CuPy + DLPack.

        Only the first output layer is handled (YOLO primary detector).
        No host copies are performed – raw CUdeviceptr is wrapped by
        CuPyʼs UnownedMemory and converted to a PyTorch tensor via DLPack.
        """
        try:
            if infer_tensor_meta is None:
                return None

            if infer_tensor_meta.num_output_layers <= 0:
                return None

            layer = infer_tensor_meta.output_layers_info[0]

            # Validate meta 
            if not self._validate_infer_layer(layer):
                self.logger.warning("Invalid inference layer meta detected – skipping tensor extraction")
                return None

            # Build shape list from NvDsInferDims
            dims = pyds.get_dims(layer.inferDims)
            shape = [dims.d[i] for i in range(dims.numDims)]
            if not shape:
                return None

            # Map DeepStream data types to CuPy dtypes
            dtype_map = {
                0: cp.float32,   # FLOAT32
                1: cp.uint8,     # UINT8
                2: cp.int8,      # INT8
                3: cp.uint32,    # UINT32
                4: cp.int32,     # INT32
                5: cp.float16,   # FP16
            }
            cp_dtype = dtype_map.get(layer.dataType, cp.float32)
            itemsize = cp_dtype().nbytes

            # Raw device pointer
            ptr = int(layer.buffer)
            if ptr == 0:
                return None

            n_elements = math.prod(shape)
            nbytes = n_elements * itemsize

            # Wrap device memory without copy
            mem = cp.cuda.UnownedMemory(ptr, nbytes, self)
            memptr = cp.cuda.MemoryPointer(mem, 0)
            cp_array = cp.ndarray(shape, dtype=cp_dtype, memptr=memptr)

            # Convert to torch tensor via DLPack (zero-copy)
            torch_tensor = torch.utils.dlpack.from_dlpack(cp_array.toDlpack())
            return torch_tensor

        except Exception as e:
            self.logger.error(f"Failed GPU tensor extraction: {e}")
            return None

    def _validate_infer_layer(self, layer) -> bool:
        """Basic sanity checks on layer meta prior to extraction."""
        try:
            if layer.buffer == 0:
                return False
            dims = pyds.get_dims(layer.inferDims)
            if dims.numElements <= 0:
                return False
            # Ensure inferred nbytes fits within 2GB to avoid overflow mistakes
            dtype_sizes = {0:4,1:1,2:1,3:4,4:4,5:2}
            bytes_per_item = dtype_sizes.get(layer.dataType, 4)
            nbytes = dims.numElements * bytes_per_item
            if nbytes <= 0 or nbytes > 2**31:
                return False
            return True
        except Exception as e:
            self.logger.debug(f"Inference layer validation error: {e}")
            return False

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
        """Extract analytics object metadata and normalize key names"""
        try:
            user_meta_list = obj_meta.obj_user_meta_list
            while user_meta_list:
                user_meta = pyds.NvDsUserMeta.cast(user_meta_list.data)  # type: ignore
                if user_meta.base_meta.meta_type == pyds.nvds_get_user_meta_type("NVIDIA.DSANALYTICSOBJ.USER_META"):  # type: ignore
                    analytics_obj_meta = pyds.NvDsAnalyticsObjInfo.cast(user_meta.user_meta_data)  # type: ignore
                    # Normalize to DeepStream SDK key casing so downstream logic works
                    return {
                        'dirStatus': analytics_obj_meta.dirStatus,
                        'lcStatus': analytics_obj_meta.lcStatus,
                        'ocStatus': analytics_obj_meta.ocStatus,
                        'roiStatus': analytics_obj_meta.roiStatus,
                        # Retain legacy snake_case keys for backward compatibility
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
        active_tracks = []
        occupancy = {}
        transitions = []
        
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
            
            # Build tracking data for telemetry
            track_dict = {
                'track_id': obj.object_id,
                'camera_id': f"camera_{frame_meta.source_id}",
                'confidence': obj.confidence,
                'bbox': [rect.left, rect.top, rect.width, rect.height],
                'class_id': obj.class_id
            }
            
            # Compute center point
            center_x = rect.left + rect.width / 2
            center_y = rect.top + rect.height / 2
            track_dict['center'] = [center_x, center_y]
            
            # Add tracker confidence if available
            if hasattr(obj, 'tracker_confidence'):
                track_dict['tracker_confidence'] = obj.tracker_confidence
            
            active_tracks.append(track_dict)
            
            # Phase 3.3: Add analytics metadata if available
            analytics_data = self._extract_analytics_obj_meta(obj)
            if analytics_data:
                detection["analytics"] = analytics_data
                
                # Extract occupancy and transition data from analytics
                if 'roiStatus' in analytics_data:
                    roi_status = analytics_data['roiStatus']
                    self.logger.debug(f"📊 ROI status: {roi_status}")
                    if isinstance(roi_status, dict):
                        for zone_name, status in roi_status.items():
                            if status == 1:  # Object is in this zone
                                occupancy[zone_name] = occupancy.get(zone_name, 0) + 1
                                self.logger.debug(f"📊 Object {obj.object_id} in zone {zone_name}")
                
                if 'lcStatus' in analytics_data:
                    lc_status = analytics_data['lcStatus']
                    self.logger.debug(f"📊 Line crossing status: {lc_status}")
                    if isinstance(lc_status, dict):
                        for line_name, status in lc_status.items():
                            if status == 1:  # Object crossed this line
                                transitions.append({
                                    'track_id': obj.object_id,
                                    'camera_id': f"camera_{frame_meta.source_id}",
                                    'line_name': line_name,
                                    'timestamp': time.time()
                                })
                                self.logger.debug(f"📊 Object {obj.object_id} crossed line {line_name}")
            else:
                self.logger.debug(f"📊 No analytics data for object {obj.object_id}")
            
            # Add secondary inference results if available
            secondary_data = self._extract_secondary_inference_meta(obj)
            if secondary_data:
                detection["secondary_inference"] = secondary_data
            
            detections.append(detection)
            
            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        
        # Update live tracking state
        self.live_tracking_state['active_tracks'] = active_tracks
        self.live_tracking_state['occupancy'] = occupancy
        self.live_tracking_state['transitions'].extend(transitions)
        
        # Keep only recent transitions (last 100)
        if len(self.live_tracking_state['transitions']) > 100:
            self.live_tracking_state['transitions'] = self.live_tracking_state['transitions'][-100:]
        
        # Debug logging for tracking state
        self.logger.debug(f"📊 Final tracking state - Active tracks: {len(active_tracks)}, Occupancy: {occupancy}, Transitions: {len(transitions)}")
        if active_tracks:
            self.logger.debug(f"📊 Active tracks sample: {active_tracks[:2]}")  # Show first 2 tracks
        if occupancy:
            self.logger.debug(f"📊 Occupancy: {occupancy}")
        if transitions:
            self.logger.debug(f"📊 Recent transitions: {transitions[-3:]}")  # Show last 3 transitions
        
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
            self.logger.info("🚀 Starting DeepStream pipeline...")
            
            if not self.pipeline:
                self.logger.error("❌ Pipeline not created")
                return False
            
            # Set pipeline state to PLAYING
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                self.logger.error("❌ Failed to set pipeline to PLAYING state")
                return False
            elif ret == Gst.StateChangeReturn.ASYNC:
                self.logger.info("⏳ Pipeline state change is async, waiting...")
                ret = self.pipeline.get_state(Gst.CLOCK_TIME_NONE)
                if ret[0] != Gst.StateChangeReturn.SUCCESS:
                    self.logger.error(f"❌ Pipeline state change failed: {ret[0]}")
                    return False
            
            self.running = True
            self.start_time = time.time()
            
            # Start main loop in a separate thread
            self.mainloop_thread = threading.Thread(target=self._run_mainloop, daemon=True)
            self.mainloop_thread.start()
            
            self.logger.info("✅ DeepStream pipeline started successfully")
            
            # Log pipeline state after a short delay
            def check_pipeline_state():
                time.sleep(2)
                ret, state, pending = self.pipeline.get_state(0)
                self.logger.info(f"📊 Pipeline state: {state}, pending: {pending}, ret: {ret}")
                
                # Check if we're receiving frames
                if self.frame_count == 0:
                    self.logger.warning("⚠️ No frames received yet - checking pipeline elements...")
                    # Log element states
                    it = self.pipeline.iterate_elements()
                    while True:
                        result, element = it.next()
                        if result != Gst.IteratorResult.OK:
                            break
                        name = element.get_name()
                        ret, state, pending = element.get_state(0)
                        self.logger.info(f"📊 Element {name}: state={state}, pending={pending}, ret={ret}")
            
            threading.Thread(target=check_pipeline_state, daemon=True).start()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error starting pipeline: {e}")
            return False

    def _run_mainloop(self):
        """Creates and runs the GLib MainLoop."""
        self.mainloop = GLib.MainLoop()
        self.mainloop.run()
    
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
            # Since tensor_queue is no longer used (metadata extracted via probe),
            # return empty result for compatibility
            return False, None
        except Exception as e:
            self.logger.error(f"Error reading tensor: {e}")
            return False, None
    
    def stop(self):
        """Stop the DeepStream pipeline."""
        self.logger.info("Stopping DeepStream pipeline")
        self.running = False
        
        # Unregister custom callbacks
        try:
            pyds.unset_callback_funcs()
            self.logger.info("✅ Unregistered custom metadata callbacks")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not unregister custom callbacks: {e}")
        
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
        runtime = time.time() - self.start_time if self.running and self.start_time > 0 else 0
        
        # Use a lock to safely access frame_count
        with self.frame_count_lock:
            frame_count_copy = self.frame_count
            
        fps = frame_count_copy / runtime if runtime > 0 else 0
        
        return {
            'pipeline_type': 'deepstream',
            'running': self.running,
            'frames_processed': frame_count_copy,
            'fps': fps,
            'runtime_seconds': runtime,
            'batch_size': self.batch_size,
            'sources': len(self.sources),
            'queue_size': self.jpeg_queue.qsize() if hasattr(self, 'jpeg_queue') else 0,
            'tracking': self.live_tracking_state
        }

    def _on_new_jpeg_sample(self, appsink: GstApp.AppSink) -> Gst.FlowReturn:
        """Callback for GPU JPEG appsink – push encoded JPEG bytes to queue"""
        try:
            sample = appsink.emit("pull-sample")
            if not sample:
                return Gst.FlowReturn.ERROR
            buffer = sample.get_buffer()
            if not buffer:
                return Gst.FlowReturn.ERROR
            # Extract the full buffer contents
            success, mapinfo = buffer.map(Gst.MapFlags.READ)
            if not success:
                return Gst.FlowReturn.ERROR
            try:
                jpeg_bytes = mapinfo.data  # bytes-like
                if jpeg_bytes:
                    try:
                        self.jpeg_queue.put_nowait(bytes(jpeg_bytes))
                    except queue.Full:
                        # Drop frame if queue full
                        pass
            finally:
                buffer.unmap(mapinfo)
            return Gst.FlowReturn.OK
        except Exception as e:
            self.logger.error(f"Error in _on_new_jpeg_sample: {e}")
            return Gst.FlowReturn.ERROR

    def read_encoded_jpeg(self, timeout: float = 0.1) -> Tuple[bool, Optional[bytes]]:
        """Return next encoded JPEG bytes from the GPU pipeline (native OSD mode)."""
        if not self.running:
            return False, None
        try:
            jpeg_bytes = self.jpeg_queue.get(timeout=timeout)
            return True, jpeg_bytes
        except queue.Empty:
            return False, None

    def update_confidence_threshold(self, confidence_threshold: float) -> bool:
        """Update confidence threshold in real-time using GObject properties
        
        Args:
            confidence_threshold: New confidence threshold (0.0-1.0)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.pipeline or not self.nvinfer:
                self.logger.warning("Pipeline or nvinfer not available for confidence threshold update")
                return False
            
            # Update nvinfer confidence threshold property
            self.nvinfer.set_property("confidence-threshold", confidence_threshold)
            self.logger.info(f"✅ Updated confidence threshold to: {confidence_threshold}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update confidence threshold: {e}")
            return False

    def update_iou_threshold(self, iou_threshold: float) -> bool:
        """Update IOU threshold in real-time using GObject properties
        
        Args:
            iou_threshold: New IOU threshold (0.0-1.0)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.pipeline or not self.nvinfer:
                self.logger.warning("Pipeline or nvinfer not available for IOU threshold update")
                return False
            
            # Update nvinfer IOU threshold property
            self.nvinfer.set_property("iou-threshold", iou_threshold)
            self.logger.info(f"✅ Updated IOU threshold to: {iou_threshold}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update IOU threshold: {e}")
            return False

    def set_detection_enabled(self, enabled: bool) -> bool:
        """Enable or disable detection in real-time using GObject properties
        
        Args:
            enabled: Whether detection should be enabled
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.pipeline or not self.nvinfer:
                self.logger.warning("Pipeline or nvinfer not available for detection enable/disable")
                return False
            
            # Update nvinfer enable property
            self.nvinfer.set_property("enable", enabled)
            self.logger.info(f"✅ Updated detection enabled to: {enabled}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update detection enabled: {e}")
            return False

    def update_target_classes(self, target_classes: List[int]) -> bool:
        """Update target classes in real-time using GObject properties
        
        Args:
            target_classes: List of class IDs to detect
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.pipeline or not self.nvinfer:
                self.logger.warning("Pipeline or nvinfer not available for target classes update")
                return False
            
            # Convert class list to string format for custom library
            class_string = ",".join(map(str, target_classes))
            
            # Update custom library properties for class filtering
            # This requires the custom library to support class filtering via properties
            custom_props = f"target-classes:{class_string}"
            self.nvinfer.set_property("custom-lib-props", custom_props)
            
            self.logger.info(f"✅ Updated target classes to: {target_classes}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update target classes: {e}")
            return False

    def osd_sink_pad_buffer_probe(self, pad, info, u_data):
        """Pad probe function to extract metadata from the buffer."""
        try:
            gst_buffer = info.get_buffer()
            if not gst_buffer:
                return Gst.PadProbeReturn.OK

            # Increment frame count here as this probe sees every frame
            with self.frame_count_lock:
                self.frame_count += 1
            
            if self.frame_count % 30 == 0:
                self.logger.info(f"📹 Probe processing frame {self.frame_count}")

            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
            if not batch_meta:
                return Gst.PadProbeReturn.OK

            l_frame = batch_meta.frame_meta_list
            while l_frame:
                try:
                    frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                    # This function will now populate self.live_tracking_state
                    self._parse_obj_meta(frame_meta)
                except StopIteration:
                    break
                try:
                    l_frame = l_frame.next
                except StopIteration:
                    break
            
            return Gst.PadProbeReturn.OK
        except Exception as e:
            self.logger.error(f"Error in OSD probe: {e}")
            return Gst.PadProbeReturn.OK


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