#!/usr/bin/env python3
"""
DeepStream Video Pipeline

This module provides a high-performance video processing pipeline using NVIDIA DeepStream
for GPU-accelerated video reading, decoding, and preprocessing. It replaces the DALI
pipeline with DeepStream's optimized GStreamer elements.

Key Features:
- GPU-native video decoding with nvurisrcbin
- Batch processing with nvmultiurisrcbin
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
from collections import defaultdict, deque

# Bypass libproxy issues by disabling GIO proxy resolver
from typing import Optional, Tuple, Dict, Any, List, Union

import torch
import math

# GStreamer imports
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GLib, GstApp, GObject  # type: ignore  # noqa: E402

# DeepStream imports
sys.path.append('/opt/nvidia/deepstream/deepstream/lib')
import pyds  # type: ignore  # noqa: E402

# Local imports
from websocket_server import WebSocketServer  # noqa: E402

from utils import RateLimitedLogger  # noqa: E402
import requests  # For REST API calls to nvmultiurisrcbin

# Defer GStreamer initialization to runtime to avoid crashing at import time
# Some environments (or tracer/proxy setups) can abort during init; doing this
# lazily makes failures easier to diagnose and avoids import-time crashes.

from config import AppConfig  # noqa: E402




class DeepStreamVideoPipeline:
    """
    DeepStream-based video pipeline for GPU-accelerated video processing.
    
    This pipeline uses:
    - nvurisrcbin for source handling (RTSP/file/camera)
    - nvmultiurisrcbin for batching
    - nvdspreprocess for GPU preprocessing
    - appsink for tensor output
    """
    
    def __init__(self, sources: List[Dict[str, Any]], config: AppConfig, websocket_port: int = 8765, 
                         config_file: str = "pipelines/config_infer_primary_yolo11.ini",
        preproc_config: str = "pipelines/config_preproc.ini"):
        # Initialize GStreamer as early as possible, but at runtime (not module import)
        try:
            Gst.init(None)
        except Exception as e:
            # Provide clearer guidance if init fails
            raise RuntimeError(f"Failed to initialize GStreamer: {e}.\n"
                               f"Hints: ensure DeepStream is installed and environment is activated.\n"
                               f"Try: source ./activate_deepstream.sh and verify gst-inspect-1.0 works.")

        # Multi-stream configuration
        self.sources = [source for source in sources if source.get('enabled', True)]
        self.websocket_port = websocket_port
        self.config_file = config_file
        self.preproc_config = preproc_config
        self.config = config  # Store config as instance variable
        self.logger = logging.getLogger(__name__)
        
        # Create sensor_id to camera name mapping for telemetry (use stable 1-based IDs)
        self.source_info = {}
        self.sensor_ids: List[int] = []
        self.source_idx_by_sensor_id: Dict[int, int] = {}
        self.sensor_id_by_source_idx: Dict[int, int] = {}
        for index, source in enumerate(self.sources):
            sensor_id = int(source.get('sensor_id', index + 1))  # stable, non-zero
            self.sensor_ids.append(sensor_id)
            self.source_idx_by_sensor_id[sensor_id] = index
            self.sensor_id_by_source_idx[index] = sensor_id

            camera_name = source.get('name', f'Camera_{sensor_id}')
            # Extract clean camera name for UI (e.g., "Living Room Camera" -> "living-room")
            if 'Living Room' in camera_name:
                clean_name = 'living-room'
            elif 'Kitchen' in camera_name:
                clean_name = 'kitchen'
            elif 'Family Room' in camera_name:
                clean_name = 'family-room'
            else:
                clean_name = camera_name.lower().replace(' ', '-').replace('_', '-')

            self.source_info[sensor_id] = {
                'name': camera_name,
                'clean_name': clean_name,
                'url': source.get('url', ''),
                'width': source.get('width', 1920),
                'height': source.get('height', 1080)
            }
        
        self.logger.info(f"🎥 Initializing multi-stream pipeline with {len(self.sources)} sources:")
        for sid in self.sensor_ids:
            info = self.source_info[sid]
            self.logger.info(f"  Sensor {sid}: {info['name']} ({info['clean_name']}) - {info['width']}x{info['height']}")
        
        # Use rate-limited loggers for different types of messages
        self.rate_limited_logger = RateLimitedLogger(self.logger, rate_limit_seconds=5.0)
        self.metadata_logger = RateLimitedLogger(self.logger, rate_limit_seconds=5.0)
        self.tensor_logger = RateLimitedLogger(self.logger, rate_limit_seconds=2.0)
        self.detection_logger = RateLimitedLogger(self.logger, rate_limit_seconds=1.0)
        
        # Initialize pipeline components
        self.pipeline: Optional[Gst.Pipeline] = None
        self.mainloop: Optional[GLib.MainLoop] = None
        self.websocket_server: Optional[WebSocketServer] = None

        
        # Threading and state management
        self.running = False
        self.pipeline_thread: Optional[threading.Thread] = None
        self.websocket_thread: Optional[threading.Thread] = None
        

        
        # Pipeline configuration - dynamic batch size based on number of sources
        self.batch_size = len(self.sensor_ids)
        # Use largest resolution for muxer output to accommodate all streams
        self.max_width = max(source.get('width', 1920) for source in self.sources)
        self.max_height = max(source.get('height', 1080) for source in self.sources)
        self.device_id = 0
        # Default REST API port for nvmultiurisrcbin
        self.multiurisrc_port = 9000
        
        self.logger.info(f"📊 Pipeline config: batch_size={self.batch_size}, resolution={self.max_width}x{self.max_height}")
        
        # Preflight: verify required DeepStream plugins are available with helpful errors
        try:
            registry = Gst.Registry.get()
            required = [
                ("nvmultiurisrcbin", "DeepStream multi-URI source (nvmultiurisrcbin)"),
                ("nvdspreprocess", "DeepStream preprocessor (nvdspreprocess)"),
                ("nvinfer", "DeepStream inference (nvinfer)"),
                ("nvstreamdemux", "DeepStream stream demux (nvstreamdemux)"),
                ("nvdsosd", "DeepStream on-screen display (nvdsosd)"),
                ("nvjpegenc", "NVIDIA JPEG encoder (nvjpegenc)")
            ]
            missing = []
            for name, desc in required:
                if not registry.find_feature(name, Gst.ElementFactory):
                    missing.append(f"{name} – {desc}")
            if missing:
                hint_env = (
                    "Required DeepStream plugins not found:\n  - " + "\n  - ".join(missing) +
                    "\n\nFix: source DeepStream env and set GST paths, e.g.:\n"
                    "  export DEEPSTREAM_DIR=/opt/nvidia/deepstream/deepstream\n"
                    "  export GST_PLUGIN_PATH=$DEEPSTREAM_DIR/lib/gst-plugins:$GST_PLUGIN_PATH\n"
                    "Also verify with: gst-inspect-1.0 nvmultiurisrcbin\n"
                )
                raise RuntimeError(hint_env)
        except Exception as e:
            # Bubble up with context so startup reports a clear error instead of aborting
            raise
        
        # Add missing attributes for compatibility
        self.frame_count = 0
        self.frame_count_lock = threading.Lock()
        # The tensor_queue is no longer needed as metadata is extracted via probe
        # JPEG queues for GPU-encoded frames, keyed by sensor_id
        self.jpeg_queues: Dict[int, queue.Queue[bytes]] = {
            sensor_id: queue.Queue(maxsize=30) for sensor_id in self.sensor_ids
        }
        self.start_time = 0  # Will be set in start()
        
        # Add tracking history for trail visualization
        self.trail_history = defaultdict(lambda: deque(maxlen=self.config.visualization.TRAIL_LENGTH))
        # Respect config default for initial state
        self.trail_visualization_enabled = bool(self.config.visualization.TRAIL_VISUALIZATION_ENABLED)
        
        # --- trail visualisation state ---
        # Maintain histories per sensor_id to avoid cross-stream overlays
        self.trail_last_seen_by_sensor: Dict[int, Dict[int, float]] = defaultdict(dict)
        self.trail_history_by_sensor: Dict[int, defaultdict] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.config.visualization.TRAIL_LENGTH)))
        # seconds to keep a disappeared track's trail (configurable)
        self.trail_timeout_s: float = self.config.visualization.TRAIL_TIMEOUT_S
        # draw only on every Nth frame (≥1)
        self.trail_draw_stride: int = self.config.visualization.TRAIL_DRAW_STRIDE
        # show labels in trail visualization ONLY if enabled in config
        self.trail_show_labels: bool = self.config.visualization.TRAIL_SHOW_LABELS
        
        # Initialize tracking state for telemetry - per sensor_id
        self.live_tracking_state: Dict[int, Dict[str, Any]] = {}
        for sensor_id in self.sensor_ids:
            self.live_tracking_state[sensor_id] = {
                'active_tracks': [],
                'occupancy': {},
                'transitions': []
            }
        self.logger.info(f"📊 Initialized tracking state for {len(self.sensor_ids)} streams")

        # Per-stream per-track zone dwell tracking state
        # { sensor_id: { track_id: { 'current_zone': Optional[str], 'entry_time': Optional[float] } } }
        self.track_zone_state_by_sensor: Dict[int, Dict[int, Dict[str, Any]]] = {
            sensor_id: {} for sensor_id in self.sensor_ids
        }

        # Per-stream per-track motion state for velocity estimation (px/s)
        # { sensor_id: { track_id: { 'last_center': [x, y], 'last_ts': float, 'vel_hist': deque([(vx,vy), ...]) } } }
        self.track_motion_state_by_sensor: Dict[int, Dict[int, Dict[str, Any]]] = {
            sensor_id: {} for sensor_id in self.sensor_ids
        }

        # JPEG branch tracking keyed by sensor_id
        # Per-stream branch elements: queue, nvvideoconvert, caps, nvdsosd, nvjpegenc, appsink
        self._stream_branch_elements: Dict[int, List[Gst.Element]] = {}

        # Precompute and reuse track color cache (track_id -> (r,g,b))
        self._track_color_cache: Dict[int, Tuple[float, float, float]] = {}
        self._demux_requested_pads: Dict[int, Gst.Pad] = {}

        # Demux pad calibration maps
        self.demux_pad_to_source_id: Dict[str, int] = {}
        self.source_id_to_demux_pad: Dict[int, str] = {}
        # Demux pad reuse maps
        self._demux_requested_pads_by_index: Dict[int, Gst.Pad] = {}
        self._demux_requested_pads_by_sensor: Dict[int, Gst.Pad] = {}

        # Local counter for demux probe observations (used to limit early debug logs)
        self._demux_probe_seen: int = 0

        
        # GStreamer already initialized at module import
        
        # Create pipeline elements
        self._create_pipeline()





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

    def _remove_excluded_objects_probe(self, pad, info, udata):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        removed_count = 0
        while l_frame:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            l_obj = frame_meta.obj_meta_list
            while l_obj:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                    l_obj_next = l_obj.next
                except StopIteration:
                    break

                # Inspect analytics meta for ROI status
                remove = False
                l_user = obj_meta.obj_user_meta_list
                while l_user:
                    try:
                        user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                    except StopIteration:
                        break

                    if user_meta.base_meta.meta_type == pyds.nvds_get_user_meta_type("NVIDIA.DSANALYTICSOBJ.USER_META"):
                        ainfo = pyds.NvDsAnalyticsObjInfo.cast(user_meta.user_meta_data)
                        if ainfo and ainfo.roiStatus is not None:
                            for label in ainfo.roiStatus:
                                if label in ("RF1",):  # exclusion labels
                                    remove = True
                                    break
                        if remove:
                            break
                    
                    try:
                        l_user = l_user.next
                    except StopIteration:
                        break

                if remove:
                    removed_count += 1
                    pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj_meta)

                l_obj = l_obj_next
            
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
        
        if removed_count > 4 and self.frame_count <= 300:
            self.logger.info(f"Frame {self.frame_count}: Removed {removed_count} objects from exclusion zone.")

        return Gst.PadProbeReturn.OK

    def _configure_elements(self, elements: Dict[str, Any]) -> bool:
        """Configure properties for all pipeline elements."""
        try:
            self.logger.info("------------- Configuring GStreamer Elements-------------")
            
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self._on_bus_message)

            # Streammux configuration - dynamic batch size for multi-stream
            # nvmultiurisrcbin (manages sources internally)
            uri_list = ",".join(self.source_info[sid]['url'] for sid in self.sensor_ids)
            # Use stable 1-based sensor IDs
            sensor_id_list = ",".join(str(sid) for sid in self.sensor_ids)
            elements['multiurisrc'].set_property("uri-list", uri_list)
            elements['multiurisrc'].set_property("sensor-id-list", sensor_id_list)
            elements['multiurisrc'].set_property("max-batch-size", self.batch_size)
            elements['multiurisrc'].set_property("width", self.max_width)
            elements['multiurisrc'].set_property("height", self.max_height)
            elements['multiurisrc'].set_property("batched-push-timeout", 40000)
            elements['multiurisrc'].set_property("live-source", 1)
            elements['multiurisrc'].set_property("drop-pipeline-eos", 1)
            elements['multiurisrc'].set_property("rtsp-reconnect-interval", 30)
            elements['multiurisrc'].set_property("port", self.multiurisrc_port)
            elements['multiurisrc'].set_property("ip-address", "localhost")
            self.logger.info(f"📊 nvmultiurisrcbin configured: max-batch-size={self.batch_size}, resolution={self.max_width}x{self.max_height}")

            elements['preprocess'].set_property("config-file", self.config.processing.DEEPSTREAM_PREPROCESS_CONFIG)
            
            self._check_for_engine_file(self.config_file)
            elements['nvinfer'].set_property("config-file-path", self.config_file)
            elements['nvinfer'].set_property("input-tensor-meta", True)

            # Exclusion analytics
            elements['nvdsanalytics_exclude'].set_property("unique-id", 101)
            elements['nvdsanalytics_exclude'].set_property("config-file", "pipelines/config_nvdsanalytics_exclude.ini")

            # Tracker configuration
            elements['nvtracker'].set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
            tracker_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipelines/config_tracker_nvdcf_batch.yml")
            elements['nvtracker'].set_property("ll-config-file", tracker_config_path)

            # Post-tracker analytics
            elements['nvdsanalytics_post'].set_property("unique-id", 201)
            elements['nvdsanalytics_post'].set_property("config-file", "pipelines/config_nvdsanalytics_post.ini")
            
            self.logger.info("✅ All elements configured successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error configuring elements: {e}")
            return False

    def _build_main_pipeline_chain(self, elements: Dict[str, Any]) -> bool:
        """Link all main pipeline elements together."""
        try:
            self.logger.info("------------- Linking Main Pipeline Chain-------------")
            
            # Link main processing chain
            if not elements['multiurisrc'].link(elements['preprocess']): 
                raise RuntimeError("Failed to link nvmultiurisrcbin to preprocess")
            else:
                self.logger.info("TRACE linked nvmultiurisrcbin → preprocess")
                
            if not elements['preprocess'].link(elements['nvinfer']): 
                raise RuntimeError("Failed to link preprocess to nvinfer")
            if not elements['nvinfer'].link(elements['q_after_pgie']): 
                raise RuntimeError("Failed to link nvinfer to q_after_pgie")
            if not elements['q_after_pgie'].link(elements['nvdsanalytics_exclude']): 
                raise RuntimeError("Failed to link q_after_pgie to nvdsanalytics_exclude")
            if not elements['nvdsanalytics_exclude'].link(elements['q_before_tracker']): 
                raise RuntimeError("Failed to link nvdsanalytics_exclude to q_before_tracker")
            if not elements['q_before_tracker'].link(elements['nvtracker']): 
                raise RuntimeError("Failed to link q_before_tracker to nvtracker")
            if not elements['nvtracker'].link(elements['q_after_tracker']): 
                raise RuntimeError("Failed to link nvtracker to q_after_tracker")
            if not elements['q_after_tracker'].link(elements['nvdsanalytics_post']): 
                raise RuntimeError("Failed to link q_after_tracker to nvdsanalytics_post")
            if not elements['nvdsanalytics_post'].link(elements['demux']): 
                raise RuntimeError("Failed to link nvdsanalytics_post to demux")

            self.logger.info("✅ Main pipeline chain linked successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error linking pipeline chain: {e}")
            return False

    def _create_pipeline(self) -> bool:
        """Create the DeepStream GStreamer pipeline using refactored helper functions."""
        try:
            # --- Phase A: Create All Elements ---
            self.logger.info("------------- Creating GStreamer Elements-------------")
            self.pipeline = Gst.Pipeline()
            if not self.pipeline:
                raise RuntimeError("Failed to create pipeline")

            # Create primary elements
            multiurisrc = Gst.ElementFactory.make("nvmultiurisrcbin", "nvmultiurisrcbin")
            preprocess = Gst.ElementFactory.make("nvdspreprocess", "nvdspreprocess")
            nvinfer = Gst.ElementFactory.make("nvinfer", "nvinfer")
            
            # Analytics and Tracking
            nvdsanalytics_exclude = Gst.ElementFactory.make("nvdsanalytics", "nvdsanalytics_exclude")
            nvtracker = Gst.ElementFactory.make("nvtracker", "nvtracker")
            nvdsanalytics_post = Gst.ElementFactory.make("nvdsanalytics", "nvdsanalytics_post")
            

            
            # Demuxer (per-branch OSD will be created downstream)
            demux = Gst.ElementFactory.make("nvstreamdemux", "nvstreamdemux")
            if not demux:
                raise RuntimeError("Failed to create nvstreamdemux element.")

            # Queues for pipeline robustness
            q_after_pgie = Gst.ElementFactory.make("queue", "q_after_pgie")
            q_before_tracker = Gst.ElementFactory.make("queue", "q_before_tracker")
            q_after_tracker = Gst.ElementFactory.make("queue", "q_after_tracker")

            # Validate element creation
            element_list = [
                multiurisrc, preprocess, nvinfer, nvdsanalytics_exclude, nvtracker, nvdsanalytics_post,
                demux, q_after_pgie, q_before_tracker, q_after_tracker
            ]
            
            if not all(element_list):
                element_names = [
                    "nvmultiurisrcbin", "nvdspreprocess", "nvinfer", "nvdsanalytics_exclude", "nvtracker", 
                    "nvdsanalytics_post", "nvstreamdemux",
                    "q_after_pgie", "q_before_tracker", "q_after_tracker"
                ]
                for el, name in zip(element_list, element_names):
                    if not el: self.logger.error(f"❌ Failed to create element: {name}")
                raise RuntimeError("Failed to create one or more GStreamer elements.")
            self.logger.info("✅ All GStreamer elements created successfully.")

            # Create elements dictionary for helper functions
            elements = {
                'multiurisrc': multiurisrc, 'preprocess': preprocess, 'nvinfer': nvinfer,
                'nvdsanalytics_exclude': nvdsanalytics_exclude, 'nvtracker': nvtracker, 
                'nvdsanalytics_post': nvdsanalytics_post, 'demux': demux,
                'q_after_pgie': q_after_pgie, 'q_before_tracker': q_before_tracker, 
                'q_after_tracker': q_after_tracker
            }

            # --- Phase B: Configure Elements ---
            if not self._configure_elements(elements):
                raise RuntimeError("Failed to configure pipeline elements")

            # --- Phase C: Add Elements to Pipeline ---
            self.logger.info("------------- Adding Elements to Pipeline-------------")
            for el in element_list:
                self.pipeline.add(el)

            # --- Phase D: Setup Probes ---
            self.logger.info("------------- Setting Up Buffer Probes-------------")
            
            # Telemetry Probe for metadata extraction
            analytics_src_pad = nvdsanalytics_post.get_static_pad("src")
            if not analytics_src_pad: raise RuntimeError("Failed to get nvdsanalytics_post source pad")
            analytics_src_pad.add_probe(Gst.PadProbeType.BUFFER, self._analytics_probe, 0)
            self.logger.info("✅ Added buffer probe to nvdsanalytics_post source pad for telemetry extraction")
            
            # Pad probe to remove excluded objects
            exclude_src_pad = nvdsanalytics_exclude.get_static_pad("src")
            if not exclude_src_pad: raise RuntimeError("Failed to get nvdsanalytics_exclude source pad")
            exclude_src_pad.add_probe(Gst.PadProbeType.BUFFER, self._remove_excluded_objects_probe, None)
            self.logger.info("✅ Added buffer probe to nvdsanalytics_exclude source pad for object removal")

            # Per-branch OSD probe will be attached in per-stream branches

            # (Removed noisy mux src probe)

            # Optional: Add probe to demux sink pad to trace buffer flow for first few frames
            demux_sink = demux.get_static_pad("sink")
            if demux_sink:
                self.logger.debug("Adding buffer probe to demux sink pad (limited logging)")
                demux_sink.add_probe(Gst.PadProbeType.BUFFER, self._demux_debug_probe, None)

            # --- Phase E: Link Main Pipeline Chain ---
            if not self._build_main_pipeline_chain(elements):
                raise RuntimeError("Failed to link main pipeline chain")

            # Store references early for downstream setup that accesses them
            self.multiurisrc, self.preprocess, self.nvinfer, self.nvtracker = multiurisrc, preprocess, nvinfer, nvtracker
            self.nvdsanalytics_exclude, self.nvdsanalytics_post = nvdsanalytics_exclude, nvdsanalytics_post
            self.demux = demux

            # --- Phase F: Calibrate demux pads and build per-stream branches with per-branch OSD ---
            self._calibrate_demux_pad_source_map()
            self.logger.info("🎥 Building per-stream branches with per-branch OSD and JPEG appsinks")
            self._setup_stream_branches()

            # --- Finalization ---
            self.logger.info("✅ Pipeline construction complete.")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create pipeline: {e}")
            import traceback
            self.logger.error(f"Full Traceback: {traceback.format_exc()}")
            return False
    







    
    def _analytics_probe(self, pad, info, user_data):
        """Probe to extract telemetry data after analytics."""
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        while l_frame:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                # This call updates self.live_tracking_state for telemetry
                self._parse_obj_meta(frame_meta)
                l_frame = l_frame.next
            except StopIteration:
                break
        if self.websocket_server and hasattr(self.websocket_server, 'broadcast_frame'):
            for source_id, state in self.live_tracking_state.items():
                if state['active_tracks']:  # Only send if data
                    frame_data = {
                        'source_id': source_id,
                        'timestamp': time.time(),
                        'tracking': state
                    }
                    self.websocket_server.broadcast_frame(frame_data)
        return Gst.PadProbeReturn.OK


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



    def _parse_obj_meta(self, frame_meta) -> List[Dict[str, Any]]:
        """Return list(dict) with keys class_id, confidence, bbox, object_id, and analytics data."""
        detections = []
        active_tracks = []
        occupancy = {}
        transitions = []
        
        # Normalize DeepStream 0-based index to configured 1-based sensor_id
        ds_index = int(frame_meta.source_id)
        sensor_id = self.sensor_id_by_source_idx.get(ds_index, ds_index)
        
        # Keep a copy of previous occupancy to detect vacates
        try:
            prev_occupancy = dict(self.live_tracking_state.get(sensor_id, {}).get('occupancy', {}))
        except Exception:
            prev_occupancy = {}

        l_obj = frame_meta.obj_meta_list
        now_ts = time.time()
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
                'camera_id': f"camera_{sensor_id}",
                'confidence': obj.confidence,
                'bbox': [rect.left, rect.top, rect.width, rect.height],
                'class_id': obj.class_id
            }
            
            # Compute center point
            center_x = rect.left + rect.width / 2
            center_y = rect.top + rect.height / 2
            track_dict['center'] = [center_x, center_y]

            # Estimate velocity (px/s) using per-track motion state with short moving average
            try:
                motion_state = self.track_motion_state_by_sensor.setdefault(sensor_id, {}).setdefault(obj.object_id, {
                    'last_center': None,
                    'last_ts': None,
                    'vel_hist': deque(maxlen=5)
                })
                last_center = motion_state.get('last_center')
                last_ts = motion_state.get('last_ts')
                if last_center is not None and last_ts is not None:
                    dt = max(1e-3, now_ts - float(last_ts))
                    vx = (center_x - float(last_center[0])) / dt
                    vy = (center_y - float(last_center[1])) / dt
                    motion_state['vel_hist'].append((vx, vy))
                    # Compute average velocity over history
                    if motion_state['vel_hist']:
                        hvx = sum(v[0] for v in motion_state['vel_hist']) / len(motion_state['vel_hist'])
                        hvy = sum(v[1] for v in motion_state['vel_hist']) / len(motion_state['vel_hist'])
                        track_dict['velocity'] = [hvx, hvy]
                # Update state
                motion_state['last_center'] = [center_x, center_y]
                motion_state['last_ts'] = now_ts
            except Exception:
                # Never allow velocity estimation errors to break telemetry
                pass
            
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
                    #self.logger.debug(f"📊 ROI status: {roi_status}")

                    def bump(zone):
                        zone = str(zone).strip()
                        if zone:
                            occupancy[zone] = occupancy.get(zone, 0) + 1
                            #self.logger.debug(f"📊 Object {obj.object_id} in zone {zone}")

                    in_zones: List[str] = []
                    if isinstance(roi_status, dict):
                        for z, status in roi_status.items():
                            if status in (1, True, "IN", "inside"):
                                bump(z)
                                in_zones.append(str(z).strip())
                    elif isinstance(roi_status, (list, tuple, set)):
                        for z in roi_status:
                            bump(z)
                            in_zones.append(str(z).strip())
                    elif isinstance(roi_status, str):
                        for z in roi_status.split(','):
                            bump(z)
                            in_zones.append(str(z).strip())

                    # Assign a primary zone (first one if multiple)
                    current_zone = in_zones[0] if in_zones else None
                    if current_zone:
                        track_dict['zone'] = current_zone

                        # Dwell time tracking per (sensor_id, track_id)
                        try:
                            zone_state = self.track_zone_state_by_sensor.setdefault(sensor_id, {}).setdefault(obj.object_id, {
                                'current_zone': None,
                                'entry_time': None
                            })
                            prev_zone = zone_state.get('current_zone')
                            entry_time = zone_state.get('entry_time')
                            if prev_zone == current_zone:
                                # Continue dwell
                                if entry_time is None:
                                    # If we somehow missed entry, initialize now
                                    zone_state['entry_time'] = now_ts
                                    entry_time = now_ts
                                dwell = max(0.0, now_ts - float(entry_time))
                                track_dict['dwell_time'] = dwell
                            else:
                                # Zone changed (or first seen). Record transition if applicable.
                                if prev_zone and prev_zone != current_zone:
                                    transitions.append({
                                        'track_id': obj.object_id,
                                        'camera_id': f"camera_{sensor_id}",
                                        'from_zone': prev_zone,
                                        'to_zone': current_zone,
                                        'timestamp': now_ts
                                    })
                                # Start new dwell timer
                                zone_state['current_zone'] = current_zone
                                zone_state['entry_time'] = now_ts
                                track_dict['dwell_time'] = 0.0
                        except Exception:
                            pass
                
                # Fallback to frame-level counts if no per-object ROI data
                if not occupancy:
                    frame_analytics = self._extract_analytics_frame_meta(frame_meta)
                    if frame_analytics and isinstance(frame_analytics.get('objects_in_roi'), dict):
                        occupancy.update(frame_analytics['objects_in_roi'])
                        #self.logger.debug(f"📊 Using frame-level occupancy: {frame_analytics['objects_in_roi']}")
                
                if 'lcStatus' in analytics_data:
                    lc_status = analytics_data['lcStatus']
                    #self.logger.debug(f"📊 Line crossing status: {lc_status}")
                    if isinstance(lc_status, dict):
                        for line_name, status in lc_status.items():
                            if status == 1:  # Object crossed this line
                                transitions.append({
                                    'track_id': obj.object_id,
                                    'camera_id': f"camera_{sensor_id}",
                                    'line_name': line_name,
                                    'timestamp': time.time()
                                })
                                #self.logger.debug(f"📊 Object {obj.object_id} crossed line {line_name}")
            #else:
                #self.logger.debug(f"📊 No analytics data for object {obj.object_id}")

            # Fallback zone/dwell: if no analytics zone detected, use camera room as zone
            try:
                if 'zone' not in track_dict or not track_dict['zone']:
                    # Map sensor_id to clean camera name
                    cam_info = self.source_info.get(sensor_id, {})
                    fallback_zone = cam_info.get('clean_name') or cam_info.get('name')
                    if fallback_zone:
                        track_dict['zone'] = fallback_zone
                        # Maintain simple dwell timer per (sensor_id, track_id) on this fallback zone
                        zone_state = self.track_zone_state_by_sensor.setdefault(sensor_id, {}).setdefault(obj.object_id, {
                            'current_zone': None,
                            'entry_time': None
                        })
                        prev_zone = zone_state.get('current_zone')
                        entry_time = zone_state.get('entry_time')
                        if prev_zone == fallback_zone:
                            if entry_time is None:
                                zone_state['entry_time'] = now_ts
                                entry_time = now_ts
                            dwell = max(0.0, now_ts - float(entry_time))
                            track_dict['dwell_time'] = dwell
                        else:
                            # Zone changed or first time
                            zone_state['current_zone'] = fallback_zone
                            zone_state['entry_time'] = now_ts
                            track_dict['dwell_time'] = 0.0
            except Exception:
                pass
            
            # Add secondary inference results if available
            secondary_data = self._extract_secondary_inference_meta(obj)
            if secondary_data:
                detection["secondary_inference"] = secondary_data
            
            detections.append(detection)
            
            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        
        # Publish occupancy deltas to integrations (MQTT/Influx)
        try:
            publisher = getattr(self, 'occupancy_publisher', None)
            if publisher is not None and occupancy is not None:
                # Publish current counts (rooms seen this frame)
                first_flag = getattr(self, '_occ_pub_first', True)
                for zone, cnt in occupancy.items():
                    try:
                        room_id = str(zone).strip()
                        publisher.publish_state(room_id=room_id, occupied=(int(cnt) > 0), count=int(cnt), ts_ns=int(time.time_ns()))
                        if first_flag:
                            try:
                                print(f"📡 Occupancy publish: {room_id} -> {int(cnt)}")
                            except Exception:
                                pass
                    except Exception:
                        pass
                # Publish vacates for zones seen previously but not this frame
                for zone in set(prev_occupancy.keys()) - set(occupancy.keys()):
                    try:
                        room_id = str(zone).strip()
                        publisher.publish_state(room_id=room_id, occupied=False, count=0, ts_ns=int(time.time_ns()))
                        if first_flag:
                            try:
                                print(f"📡 Occupancy publish: {room_id} -> 0 (vacate)")
                            except Exception:
                                pass
                    except Exception:
                        pass
                if first_flag:
                    try:
                        self._occ_pub_first = False
                    except Exception:
                        pass
        except Exception:
            # Never allow publishing issues to affect the pipeline
            pass

        # Update live tracking state for this specific stream
        if sensor_id in self.live_tracking_state:
            self.live_tracking_state[sensor_id]['active_tracks'] = active_tracks
            self.live_tracking_state[sensor_id]['occupancy'] = occupancy
            self.live_tracking_state[sensor_id]['transitions'].extend(transitions)
            
            # Keep only recent transitions (last 100) per stream
            if len(self.live_tracking_state[sensor_id]['transitions']) > 100:
                self.live_tracking_state[sensor_id]['transitions'] = self.live_tracking_state[sensor_id]['transitions'][-100:]
        else:
            self.logger.warning(f"⚠️ Unknown source_id {ds_index} (mapped→{sensor_id}) in frame metadata")
        
        # Debug logging for tracking state
        #if len(active_tracks) > 0:
            #self.logger.debug(f"📊 Final tracking state - Active tracks: {len(active_tracks)}, Occupancy: {occupancy}, Transitions: {len(transitions)}")
        #if active_tracks:
            #self.logger.debug(f"📊 Active tracks sample: {active_tracks[:2]}")  # Show first 2 tracks
        #if occupancy:
            #self.logger.debug(f"📊 Occupancy: {occupancy}")
        #if transitions:
            #self.logger.debug(f"📊 Recent transitions: {transitions[-3:]}")  # Show last 3 transitions
        
        return detections

    # --- Color helpers to keep OSD boxes and trails consistent with UI legend ---
    def _hsl_to_rgb(self, h: float, s: float, l: float) -> Tuple[float, float, float]:
        """Convert HSL (0..360, 0..1, 0..1) to RGB floats 0..1."""
        h = h % 360.0
        s = max(0.0, min(1.0, s))
        l = max(0.0, min(1.0, l))
        c = (1.0 - abs(2.0 * l - 1.0)) * s
        x = c * (1.0 - abs(((h / 60.0) % 2.0) - 1.0))
        m = l - c / 2.0
        rp = gp = bp = 0.0
        if 0 <= h < 60:
            rp, gp, bp = c, x, 0
        elif 60 <= h < 120:
            rp, gp, bp = x, c, 0
        elif 120 <= h < 180:
            rp, gp, bp = 0, c, x
        elif 180 <= h < 240:
            rp, gp, bp = 0, x, c
        elif 240 <= h < 300:
            rp, gp, bp = x, 0, c
        else:  # 300..360
            rp, gp, bp = c, 0, x
        r, g, b = rp + m, gp + m, bp + m
        return (max(0.0, min(1.0, r)), max(0.0, min(1.0, g)), max(0.0, min(1.0, b)))

    def _color_for_track(self, track_id: int) -> Tuple[float, float, float]:
        """Match FE colorForTrack: hsl((id*47)%360, 80%, 60%). Return RGB floats 0..1."""
        if track_id in self._track_color_cache:
            return self._track_color_cache[track_id]
        hue = float((int(track_id) * 47) % 360)
        r, g, b = self._hsl_to_rgb(hue, 0.80, 0.60)
        self._track_color_cache[track_id] = (r, g, b)
        return (r, g, b)
    
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
            self.logger.info(f"Pipeline set_state returned: {ret}")
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
        
        queue_sizes = {f"source_{src_id}": q.qsize() for src_id, q in self.jpeg_queues.items()}
        return {
            'pipeline_type': 'deepstream',
            'running': self.running,
            'frames_processed': frame_count_copy,
            'fps': fps,
            'runtime_seconds': runtime,
            'batch_size': self.batch_size,
            'sources': len(self.sensor_ids),
            'queue_sizes': queue_sizes,
            'tracking': self.live_tracking_state
        }

    def _demux_debug_probe(self, pad, info, user_data):
        """Debug probe on nvstreamdemux sink to log frame source IDs"""
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        # Gather source IDs present in this batch
        source_ids: List[int] = []
        l_frame = batch_meta.frame_meta_list
        while l_frame:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                source_ids.append(frame_meta.source_id)
                l_frame = l_frame.next
            except StopIteration:
                break

        expected_source_ids = set(range(len(self.sensor_ids)))  # DeepStream uses 0..N-1 source indices
        actual_source_ids = set(source_ids)
        unexpected_ids = actual_source_ids - expected_source_ids
        missing_ids = expected_source_ids - actual_source_ids

        # Warn only if we see IDs outside expected range; this indicates a real issue
        if unexpected_ids:
            self.rate_limited_logger.warning(
                f"⚠️ Demux sink saw unexpected source_ids {sorted(list(unexpected_ids))}; expected range 0..{len(self.sensor_ids)-1}"
            )
        else:
            # It's normal for some sources to be absent in a given batch. Debug early a few times only.
            if self._demux_probe_seen < 6 and missing_ids:
                self.logger.debug(
                    f"demux sink: some sources absent this batch (normal): missing {sorted(list(missing_ids))}, got {sorted(list(actual_source_ids))}"
                )

        self._demux_probe_seen += 1
        # Continue normal flow without per-frame logging
        return Gst.PadProbeReturn.OK

    # --- Explicit JPEG branch setup using request pads ---
    def _get_or_request_demux_pad(self, index: int) -> Optional[Gst.Pad]:
        """Return existing requested demux pad for src_index or request it once."""
        try:
            if index in self._demux_requested_pads_by_index:
                return self._demux_requested_pads_by_index[index]
            pad_name = f"src_{index}"
            pad = self.demux.get_request_pad(pad_name)
            if pad:
                self._demux_requested_pads_by_index[index] = pad
            return pad
        except Exception as e:
            self.logger.error(f"Failed to get/request demux pad for index {index}: {e}")
            return None
    def _calibrate_demux_pad_source_map(self) -> None:
        """Calibrate mapping between demux src_%u pads and actual frame_meta.source_id.
        Attaches one-shot probes to each src pad and records the first observed source_id."""
        try:
            self.demux_pad_to_source_id.clear()
            self.source_id_to_demux_pad.clear()

            pad_handler_ids: Dict[str, int] = {}

            def _one_shot_probe(pad: Gst.Pad, info: Gst.PadProbeInfo, _user_data):
                gst_buffer = info.get_buffer()
                if not gst_buffer:
                    return Gst.PadProbeReturn.OK
                batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
                if not batch_meta:
                    return Gst.PadProbeReturn.OK
                l_frame = batch_meta.frame_meta_list
                if not l_frame:
                    return Gst.PadProbeReturn.OK
                try:
                    frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                    src_id = int(frame_meta.source_id)
                    pad_name = pad.get_name()
                    # Record mapping if not already recorded
                    if pad_name not in self.demux_pad_to_source_id:
                        self.demux_pad_to_source_id[pad_name] = src_id
                        self.source_id_to_demux_pad[src_id] = pad_name
                        # Remove this probe after first mapping
                        handler_id = pad_handler_ids.get(pad_name)
                        if handler_id is not None:
                            try:
                                pad.remove_probe(handler_id)
                            except Exception:
                                pass
                        # If we've mapped all sources, log once
                        if len(self.source_id_to_demux_pad) >= len(self.sensor_ids):
                            mapping_str = ", ".join([f"{k}→{v}" for k, v in sorted(self.demux_pad_to_source_id.items())])
                            self.logger.info(f"nvstreamdemux mapping: {mapping_str}")
                    # Fill sensor→pad reuse map
                    try:
                        for pad_name, sid in self.demux_pad_to_source_id.items():
                            try:
                                demux_index = int(pad_name.split('_')[1])
                                pad_obj = self._get_or_request_demux_pad(demux_index)
                                if pad_obj:
                                    self._demux_requested_pads_by_sensor[sid] = pad_obj
                            except Exception:
                                continue
                    except Exception:
                        pass
                except StopIteration:
                    pass
                return Gst.PadProbeReturn.OK

            # Attach probes to all current requestable src pads by iterating indices
            num_expected = max(len(self.sensor_ids), 1)
            for idx in range(num_expected):
                pad = self._get_or_request_demux_pad(idx)
                if not pad:
                    continue
                pad_name = pad.get_name()
                handler_id = pad.add_probe(Gst.PadProbeType.BUFFER, _one_shot_probe, None)
                pad_handler_ids[pad_name] = handler_id

            # Wait briefly for first frames to flow and mapping to populate
            # Note: In Python API, we avoid busy waiting; mapping will complete as frames arrive.
            self.logger.info("Calibrating demux pad → source_id mapping (will log once ready)...")
        except Exception as e:
            self.logger.warning(f"Failed to calibrate demux pad mapping: {e}")

    def _setup_stream_branches(self) -> None:
        for sensor_id in self.sensor_ids:
            ok = self._add_jpeg_branch_for_sensor(sensor_id)
            if not ok:
                self.logger.error(f"❌ Failed to setup stream branch for sensor {sensor_id}")
    def _configure_live_queue(self, q: Gst.Element) -> None:
        q.set_property("max-size-buffers", 12)
        q.set_property("max-size-bytes", 0)
        q.set_property("leaky", 2)  # LEAK_DOWNSTREAM

    def _caps_event_probe(self, pad: Gst.Pad, info: Gst.PadProbeInfo, stage: str):
        try:
            event = info.get_event()
            if event and event.type == Gst.EventType.CAPS:
                caps = event.parse_caps()
                # Rate-limit logs per stage
                if not hasattr(self, "_caps_probe_counts"):
                    self._caps_probe_counts = {}
                count = self._caps_probe_counts.get(stage, 0)
                if count < 3:
                    self.logger.debug(f"[caps] {stage}: {caps.to_string() if caps else 'None'}")
                    self._caps_probe_counts[stage] = count + 1
        except Exception as e:
            # Do not disrupt pipeline on probe errors
            self.logger.debug(f"caps probe error at {stage}: {e}")
        return Gst.PadProbeReturn.OK

    def _attach_caps_debug_probes(self, conv: Gst.Element, caps: Gst.Element, jpegenc: Gst.Element, sensor_id: int) -> None:
        try:
            conv_src = conv.get_static_pad("src")
            if conv_src:
                conv_src.add_probe(Gst.PadProbeType.EVENT_DOWNSTREAM, self._caps_event_probe, f"sensor {sensor_id} conv src")
            caps_src = caps.get_static_pad("src")
            if caps_src:
                caps_src.add_probe(Gst.PadProbeType.EVENT_DOWNSTREAM, self._caps_event_probe, f"sensor {sensor_id} caps src")
            enc_sink = jpegenc.get_static_pad("sink")
            if enc_sink:
                # Caps events flow downstream into encoder sink; observe what arrives
                enc_sink.add_probe(Gst.PadProbeType.EVENT_DOWNSTREAM, self._caps_event_probe, f"sensor {sensor_id} nvjpegenc sink")
                # Log template and currently queried caps once
                try:
                    tmpl = enc_sink.get_pad_template_caps()
                    qcaps = enc_sink.query_caps(None)
                    self.logger.debug(
                        f"nvjpegenc sink template caps: {tmpl.to_string() if tmpl else 'Unknown'}; query caps: {qcaps.to_string() if qcaps else 'Unknown'}"
                    )
                except Exception:
                    pass
        except Exception as e:
            self.logger.debug(f"Failed to attach caps probes for sensor {sensor_id}: {e}")

    def _add_jpeg_branch_for_sensor(self, sensor_id: int) -> bool:
        try:
            if sensor_id in self._stream_branch_elements:
                self.logger.debug(f"JPEG branch already exists for sensor {sensor_id}")
                return True

            branch_name_prefix = f"stream_branch_{sensor_id}"
            queue = Gst.ElementFactory.make("queue", f"{branch_name_prefix}_queue")
            conv_pre = Gst.ElementFactory.make("nvvideoconvert", f"{branch_name_prefix}_conv_pre")
            caps_pre = Gst.ElementFactory.make("capsfilter", f"{branch_name_prefix}_caps_pre")
            osd = Gst.ElementFactory.make("nvdsosd", f"{branch_name_prefix}_osd")
            conv_post = Gst.ElementFactory.make("nvvideoconvert", f"{branch_name_prefix}_conv_post")
            caps_post = Gst.ElementFactory.make("capsfilter", f"{branch_name_prefix}_caps_post")
            jpegenc = Gst.ElementFactory.make("nvjpegenc", f"{branch_name_prefix}_enc")
            sink = Gst.ElementFactory.make("appsink", f"{branch_name_prefix}_sink")

            # Optimize nvjpegenc for realtime and respect configured JPEG quality
            try:
                quality = int(getattr(self.config.visualization, 'JPEG_QUALITY', 85))
                jpegenc.set_property("quality", quality)
                jpegenc.set_property("preset-level", 1)  # fast
            except Exception:
                pass

            if not all([queue, conv_pre, caps_pre, osd, conv_post, caps_post, jpegenc, sink]):
                self.logger.error(f"❌ Failed to create elements for JPEG branch sensor {sensor_id}")
                return False

            self._configure_live_queue(queue)
            # Pre-OSD RGBA and post-OSD I420
            caps_pre.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
            caps_post.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))
            sink.set_property("emit-signals", True)
            sink.set_property("sync", False)
            sink.set_property("max-buffers", 5)
            sink.set_property("drop", True)
            sink.connect("new-sample", self._on_new_jpeg_sample, sensor_id)

            for el in (queue, conv_pre, caps_pre, osd, conv_post, caps_post, jpegenc, sink):
                self.pipeline.add(el)

            # Attach detailed caps negotiation probes (non-spammy)
            self._attach_caps_debug_probes(conv_pre, caps_pre, jpegenc, sensor_id)
            self._attach_caps_debug_probes(conv_post, caps_post, jpegenc, sensor_id)

            # Link elements explicitly and validate
            if not queue.link(conv_pre):
                self.logger.error(f"❌ Failed to link queue→conv for sensor {sensor_id}")
                for el in (queue, conv_pre, caps_pre, osd, conv_post, caps_post, jpegenc, sink):
                    try:
                        self.pipeline.remove(el)
                    except Exception:
                        pass
                return False
            if not conv_pre.link(caps_pre):
                self.logger.error(f"❌ Failed to link conv→caps for sensor {sensor_id}")
                for el in (queue, conv_pre, caps_pre, osd, conv_post, caps_post, jpegenc, sink):
                    try:
                        self.pipeline.remove(el)
                    except Exception:
                        pass
                return False
            if not caps_pre.link(osd):
                self.logger.error(f"❌ Failed to link caps→osd for sensor {sensor_id}")
                for el in (queue, conv_pre, caps_pre, osd, conv_post, caps_post, jpegenc, sink):
                    try:
                        self.pipeline.remove(el)
                    except Exception:
                        pass
                return False
            # Configure and attach per-branch OSD probe
            try:
                osd.set_property('process-mode', 0)
                osd.set_property('display-text', 1)
            except Exception:
                pass
            osd_sink_pad = osd.get_static_pad("sink")
            if osd_sink_pad:
                osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, self._per_branch_osd_probe, sensor_id)
            if not osd.link(conv_post):
                self.logger.error(f"❌ Failed to link osd→conv_post for sensor {sensor_id}")
                for el in (queue, conv_pre, caps_pre, osd, conv_post, caps_post, jpegenc, sink):
                    try:
                        self.pipeline.remove(el)
                    except Exception:
                        pass
                return False
            if not conv_post.link(caps_post):
                self.logger.error(f"❌ Failed to link conv_post→caps_post for sensor {sensor_id}")
                for el in (queue, conv_pre, caps_pre, osd, conv_post, caps_post, jpegenc, sink):
                    try:
                        self.pipeline.remove(el)
                    except Exception:
                        pass
                return False
            if not caps_post.link(jpegenc):
                self.logger.error(f"❌ Failed to link caps_post→jpegenc for sensor {sensor_id}")
                for el in (queue, conv_pre, caps_pre, osd, conv_post, caps_post, jpegenc, sink):
                    try:
                        self.pipeline.remove(el)
                    except Exception:
                        pass
                return False
            if not jpegenc.link(sink):
                self.logger.error(f"❌ Failed to link jpegenc→sink for sensor {sensor_id}")
                for el in (queue, conv_pre, caps_pre, osd, conv_post, caps_post, jpegenc, sink):
                    try:
                        self.pipeline.remove(el)
                    except Exception:
                        pass
                return False

            # Request and link demux pad
            # Demux pads are index-based (0..N-1). Map true sensor_id → demux index.
            pad_name = self.source_id_to_demux_pad.get(sensor_id)
            req_pad = None
            if pad_name is None:
                demux_index = self.source_idx_by_sensor_id.get(sensor_id, None)
                if demux_index is None:
                    self.logger.error(f"❌ No demux index for sensor {sensor_id}")
                    for el in (queue, conv_pre, caps_pre, osd, conv_post, caps_post, jpegenc, sink):
                        try:
                            self.pipeline.remove(el)
                        except Exception:
                            pass
                    return False
                req_pad = self._get_or_request_demux_pad(demux_index)
                pad_name = f"src_{demux_index}"
            else:
                # If we already know pad name, map to index and get/reuse pad
                try:
                    demux_index = int(pad_name.split('_')[1])
                except Exception:
                    demux_index = None
                req_pad = self._get_or_request_demux_pad(demux_index) if demux_index is not None else None
            if not req_pad:
                self.logger.error(f"❌ Failed to request demux pad {pad_name} for sensor {sensor_id}")
                for el in (queue, conv_pre, caps_pre, osd, conv_post, caps_post, jpegenc, sink):
                    try:
                        self.pipeline.remove(el)
                    except Exception:
                        pass
                return False

            sink_pad = queue.get_static_pad("sink")
            # Ensure the queue sink pad is active before linking
            if sink_pad is None:
                self.logger.error(f"❌ Missing sink pad on queue for sensor {sensor_id}")
                # Do not release req_pad here since it is managed in the reuse maps
                for el in (queue, conv_pre, caps_pre, osd, conv_post, caps_post, jpegenc, sink):
                    try:
                        self.pipeline.remove(el)
                    except Exception:
                        pass
                return False
            link_ret = req_pad.link(sink_pad)
            if link_ret != Gst.PadLinkReturn.OK:
                self.logger.error(f"❌ Failed to link demux {pad_name} to JPEG queue for sensor {sensor_id} (ret={link_ret})")
                # Do not release req_pad here since it is managed in the reuse maps
                for el in (queue, conv_pre, caps_pre, osd, conv_post, caps_post, jpegenc, sink):
                    try:
                        self.pipeline.remove(el)
                    except Exception:
                        pass
                return False

            self._stream_branch_elements[sensor_id] = [queue, conv_pre, caps_pre, osd, conv_post, caps_post, jpegenc, sink]
            self._demux_requested_pads[sensor_id] = req_pad
            # If calibration already known, also map sensor→pad
            self._demux_requested_pads_by_sensor[sensor_id] = req_pad

            # If pipeline is already playing, sync states for dynamic add
            if self.pipeline.get_state(0)[1] == Gst.State.PLAYING:
                queue.sync_state_with_parent()
                conv_pre.sync_state_with_parent()
                caps_pre.sync_state_with_parent()
                osd.sync_state_with_parent()
                conv_post.sync_state_with_parent()
                caps_post.sync_state_with_parent()
                jpegenc.sync_state_with_parent()
                sink.sync_state_with_parent()

            self.logger.info(f"✅ JPEG branch ready for sensor {sensor_id} via request pad {pad_name}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Exception while adding JPEG branch for sensor {sensor_id}: {e}")
            return False

    def _remove_jpeg_branch_for_sensor(self, sensor_id: int) -> None:
        try:
            # Unlink and release request pad
            req_pad = self._demux_requested_pads.pop(sensor_id, None)
            if req_pad is not None:
                try:
                    self.demux.release_request_pad(req_pad)
                except Exception as e:
                    self.logger.debug(f"Error releasing request pad for sensor {sensor_id}: {e}")

            # Remove elements
            elements = self._stream_branch_elements.pop(sensor_id, None)
            if elements:
                for el in elements:
                    try:
                        el.set_state(Gst.State.NULL)
                        self.pipeline.remove(el)
                    except Exception as e:
                        self.logger.debug(f"Error removing element {el.get_name()} for sensor {sensor_id}: {e}")
            self.logger.info(f"✅ Removed JPEG branch for sensor {sensor_id}")
        except Exception as e:
            self.logger.debug(f"Error tearing down JPEG branch for sensor {sensor_id}: {e}")

    def _setup_jpeg_branches_with_request_pads(self) -> None:
        # Backward-compat alias; now builds full per-stream branches with OSD
        self._setup_stream_branches()

    def _demux_pad_removed_cb(self, demux, pad):
        """Deprecated in this implementation: we use request pads per sensor."""
        pad_name = pad.get_name()
        self.logger.debug(f"Demuxer removed pad notification: {pad_name}")

    def _on_new_jpeg_sample(self, appsink: GstApp.AppSink, sensor_id: int) -> Gst.FlowReturn:
        """Callback for GPU JPEG appsink – push encoded JPEG bytes to the correct queue"""
        try:
            true_id = sensor_id
            sample = appsink.emit("pull-sample")
            if not sample:
                return Gst.FlowReturn.ERROR
            buffer = sample.get_buffer()
            if not buffer:
                return Gst.FlowReturn.ERROR
            
            success, mapinfo = buffer.map(Gst.MapFlags.READ)
            if not success:
                return Gst.FlowReturn.ERROR
            
            try:
                jpeg_bytes = mapinfo.data
                if jpeg_bytes:
                    if true_id in self.jpeg_queues:
                        try:
                            self.jpeg_queues[true_id].put_nowait(bytes(jpeg_bytes))
                            # minimal logging to avoid spam
                        except queue.Full:
                            pass # Drop frame if queue is full
                    else:
                        self.rate_limited_logger.warning(f"No JPEG queue for source_id {true_id}")
            finally:
                buffer.unmap(mapinfo)
            
            return Gst.FlowReturn.OK
        except Exception as e:
            self.logger.error(f"Error in _on_new_jpeg_sample for sensor {sensor_id}: {e}")
            return Gst.FlowReturn.ERROR

    def read_encoded_jpeg(self, source_id: int, timeout: float = 0.1) -> Tuple[bool, Optional[bytes]]:
        """Return next encoded JPEG bytes for a specific sensor_id from the GPU pipeline."""
        if not self.running or source_id not in self.jpeg_queues:
            return False, None
        try:
            jpeg_bytes = self.jpeg_queues[source_id].get(timeout=timeout)
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

    # ------------------------------------------------------------------
    # Dynamic sensor management via nvmultiurisrcbin REST API
    # ------------------------------------------------------------------
    def add_sensor(self, sensor_id: int, uri: str) -> bool:
        """Add a new sensor stream at runtime."""
        try:
            url = f"http://localhost:{self.multiurisrc_port}/stream"
            payload = {"change": "add", "sensorId": str(sensor_id), "uri": uri}
            resp = requests.post(url, json=payload, timeout=2)
            if resp.status_code == 200:
                self.logger.info(f"✅ Added sensor {sensor_id}: {uri}")
                # Update internal state
                if sensor_id not in self.sensor_ids:
                    self.sensor_ids.append(sensor_id)
                # Default metadata if unknown
                self.source_info[sensor_id] = {
                    'name': f'Camera_{sensor_id}',
                    'clean_name': f'camera-{sensor_id}',
                    'url': uri,
                    'width': self.max_width,
                    'height': self.max_height,
                }
                # Ensure queue and tracking state exist
                if sensor_id not in self.jpeg_queues:
                    self.jpeg_queues[sensor_id] = queue.Queue(maxsize=30)
                if sensor_id not in self.live_tracking_state:
                    self.live_tracking_state[sensor_id] = {
                        'active_tracks': [], 'occupancy': {}, 'transitions': []
                    }
                # Request pad for new index and calibrate via one-shot probe
                new_index = self.source_idx_by_sensor_id.get(sensor_id)
                if new_index is not None:
                    pad = self._get_or_request_demux_pad(new_index)
                    if pad:
                        # Attach a one-shot probe by reusing calibration method
                        self._calibrate_demux_pad_source_map()
                # Build stream branch (will use calibrated pad when ready)
                ok = self._add_jpeg_branch_for_sensor(sensor_id)
                if not ok:
                    self.logger.error(f"❌ Failed to create JPEG branch for sensor {sensor_id}")
                    return False
                return True
            else:
                self.logger.error(f"❌ Failed to add sensor {sensor_id}: {resp.status_code} {resp.text}")
                return False
        except Exception as e:
            self.logger.error(f"❌ Exception while adding sensor: {e}")
            return False

    def remove_sensor(self, sensor_id: int) -> bool:
        """Remove an existing sensor stream at runtime."""
        try:
            url = f"http://localhost:{self.multiurisrc_port}/stream"
            payload = {"change": "remove", "sensorId": str(sensor_id), "uri": ""}
            resp = requests.post(url, json=payload, timeout=2)
            if resp.status_code == 200:
                self.logger.info(f"✅ Removed sensor {sensor_id}")
                # Tear down request pad and branch
                self._remove_jpeg_branch_for_sensor(sensor_id)
                # Clean queues and tracking
                self.jpeg_queues.pop(sensor_id, None)
                self.live_tracking_state.pop(sensor_id, None)
                self.source_info.pop(sensor_id, None)
                if sensor_id in self.sensor_ids:
                    try:
                        self.sensor_ids.remove(sensor_id)
                    except ValueError:
                        pass
                # Update demux mapping tables
                pad_to_remove = self.source_id_to_demux_pad.pop(sensor_id, None)
                if pad_to_remove:
                    self.demux_pad_to_source_id.pop(pad_to_remove, None)
                # Release demux pad for this sensor if tracked
                pad_obj = self._demux_requested_pads_by_sensor.pop(sensor_id, None)
                if pad_obj is not None:
                    # Remove from index map too
                    try:
                        idx = int(pad_obj.get_name().split('_')[1])
                        self._demux_requested_pads_by_index.pop(idx, None)
                    except Exception:
                        pass
                    try:
                        self.demux.release_request_pad(pad_obj)
                    except Exception:
                        pass
                return True
            else:
                self.logger.error(f"❌ Failed to remove sensor {sensor_id}: {resp.status_code} {resp.text}")
                return False
        except Exception as e:
            self.logger.error(f"❌ Exception while removing sensor: {e}")
            return False

    def toggle_trail_visualization(self, enabled: bool):
          self.set_trail_visualization(enabled)

    def set_trail_visualization(self, enabled: bool):
        """Enable or disable trail visualization in real-time."""
        self.logger.info(f"Setting trail visualization to: {enabled}")
        self.trail_visualization_enabled = enabled
        if not enabled:
            # Clear history when disabling to prevent stale trails on re-enable
            self.trail_history.clear()
            self.trail_history_by_sensor.clear()
            self.trail_last_seen_by_sensor.clear()

    def _osd_sink_pad_buffer_probe(self, pad, info, _):
        """Probe to draw trails for tracked objects before OSD rendering."""
        now = time.time()
        # Phase 0: early outs
        if not self.trail_visualization_enabled:
            self.trail_history.clear()
            # Clear per-sensor maps
            self.trail_history_by_sensor.clear()
            self.trail_last_seen_by_sensor.clear()
            return Gst.PadProbeReturn.OK

        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK
        
        # Phase 1 ─ Update histories from current detections (global)
        l_frame = batch_meta.frame_meta_list
        while l_frame:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            l_obj = frame_meta.obj_meta_list
            while l_obj:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                tid = obj_meta.object_id
                if tid != -1:
                    cx = obj_meta.rect_params.left + obj_meta.rect_params.width / 2
                    cy = obj_meta.rect_params.top  + obj_meta.rect_params.height
                    self.trail_history[tid].append((cx, cy))
                    # Also maintain per-sensor state
                    sid = int(frame_meta.source_id)
                    self.trail_history_by_sensor[sid][tid].append((cx, cy))
                    self.trail_last_seen_by_sensor[sid][tid] = now
                l_obj = l_obj.next
            l_frame = l_frame.next

        # Phase 2 ─ Prune stale tracks (per-sensor maps)
        for sid, last_seen_map in list(self.trail_last_seen_by_sensor.items()):
            for tid in [t for t, ts in list(last_seen_map.items()) if now - ts > self.trail_timeout_s]:
                self.trail_last_seen_by_sensor[sid].pop(tid, None)
                self.trail_history_by_sensor[sid].pop(tid, None)



        # ────────────────── Phase 3 ─ Draw all trails into one overlay (batched OSD legacy)
        l_frame = batch_meta.frame_meta_list
        while l_frame:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)

            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            if not display_meta:                 # pool exhausted – skip
                l_frame = l_frame.next
                continue
            display_meta.num_lines  = 0
            display_meta.num_labels = 0

            active_ids = [tid for tid, pts in self.trail_history.items()
                          if len(pts) > 1]

            budget_per_track = (250 // len(active_ids)) if active_ids else 250
            budget_per_track = max(1, budget_per_track)

            for tid in active_ids:
                pts = list(self.trail_history[tid])[-(min(self.config.visualization.TRAIL_DRAW_SEGMENTS, budget_per_track) + 1):]

                # draw segments
                for idx in range(len(pts) - 1):  # -1 because we access idx+1
                    if display_meta.num_lines >= 250:
                        break
                    
                    # Check if we have enough line_params available
                    if display_meta.num_lines >= len(display_meta.line_params):
                        break

                    x1, y1 = pts[idx]
                    x2, y2 = pts[idx + 1]

                    lp = display_meta.line_params[display_meta.num_lines]
                    lp.line_width = 3
                    lp.x1, lp.y1, lp.x2, lp.y2 = map(int, (x1, y1, x2, y2))
                    alpha = max((idx + 1) / len(pts), 0.6)
                    try:
                        r, g, b = self._color_for_track(int(tid))
                    except Exception:
                        r, g, b = (1.0, 1.0, 0.0)
                    lp.line_color.set(r, g, b, alpha)   # track-specific color
                    display_meta.num_lines += 1

                # optional label
                if (self.trail_show_labels
                        and display_meta.num_labels < 16
                        and display_meta.num_lines < 250
                        and display_meta.num_labels < len(display_meta.text_params)):
                    tp = display_meta.text_params[display_meta.num_labels]
                    tp.display_text = f"id {tid}"
                    tp.x_offset, tp.y_offset = map(int, pts[-1])
                    tp.font_params.font_name = "Serif"
                    tp.font_params.font_size = 12
                    try:
                        r, g, b = self._color_for_track(int(tid))
                    except Exception:
                        r, g, b = (1.0, 1.0, 0.0)
                    tp.font_params.font_color.set(r, g, b, 1.0)
                    tp.set_bg_clr = 0
                    display_meta.num_labels += 1

                if display_meta.num_lines >= 250:
                    break

            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
            l_frame = l_frame.next

        return Gst.PadProbeReturn.OK

    def _per_branch_osd_probe(self, pad, info, sensor_id: int):
        """Per-branch OSD sink probe to draw only for matching sensor_id using frame_meta.source_id."""
        if not self.trail_visualization_enabled:
            return Gst.PadProbeReturn.OK
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK
        # Update trails for this sensor from current detections
        now = time.time()
        l_frame = batch_meta.frame_meta_list
        while l_frame:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            # Map DS index to configured sensor_id for display logic
            ds_index_local = int(frame_meta.source_id)
            mapped_sensor = self.sensor_id_by_source_idx.get(ds_index_local, ds_index_local)
            if int(mapped_sensor) != int(sensor_id):
                l_frame = l_frame.next
                continue
            l_obj = frame_meta.obj_meta_list
            while l_obj:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                tid = obj_meta.object_id
                # Update per-object bbox label to include track id and detector confidence
                try:
                    if tid != -1:
                        conf_value = float(obj_meta.confidence)
                        obj_meta.text_params.display_text = f"id {tid} ({conf_value:.2f})"
                        # Keep background disabled to avoid covering content
                        obj_meta.text_params.set_bg_clr = 0
                        # Set bbox border color to match track color
                        try:
                            r, g, b = self._color_for_track(int(tid))
                            obj_meta.rect_params.border_width = 3
                            obj_meta.rect_params.border_color.set(r, g, b, 1.0)
                        except Exception:
                            pass
                except Exception:
                    # Never break overlay on label formatting issues
                    pass
                if tid != -1:
                    cx = obj_meta.rect_params.left + obj_meta.rect_params.width / 2
                    cy = obj_meta.rect_params.top  + obj_meta.rect_params.height
                    self.trail_history_by_sensor[int(sensor_id)][tid].append((cx, cy))
                    self.trail_last_seen_by_sensor[int(sensor_id)][tid] = now
                l_obj = l_obj.next

            # Prune stale trails for this sensor
            for tid, ts in list(self.trail_last_seen_by_sensor[int(sensor_id)].items()):
                if now - ts > self.trail_timeout_s:
                    self.trail_last_seen_by_sensor[int(sensor_id)].pop(tid, None)
                    self.trail_history_by_sensor[int(sensor_id)].pop(tid, None)

            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            if not display_meta:
                l_frame = l_frame.next
                continue
            display_meta.num_lines = 0
            display_meta.num_labels = 0
            # Draw trails for this sensor only
            sensor_trails = self.trail_history_by_sensor.get(int(sensor_id), {})
            active_ids = [tid for tid, pts in sensor_trails.items() if len(pts) > 1]
            budget_per_track = (250 // len(active_ids)) if active_ids else 250
            budget_per_track = max(1, budget_per_track)
            for tid in active_ids:
                pts = list(sensor_trails[tid])[-(min(self.config.visualization.TRAIL_DRAW_SEGMENTS, budget_per_track) + 1):]
                for idx in range(len(pts) - 1):
                    if display_meta.num_lines >= 250:
                        break
                    if display_meta.num_lines >= len(display_meta.line_params):
                        break
                    x1, y1 = pts[idx]
                    x2, y2 = pts[idx + 1]
                    lp = display_meta.line_params[display_meta.num_lines]
                    lp.line_width = 3
                    lp.x1, lp.y1, lp.x2, lp.y2 = map(int, (x1, y1, x2, y2))
                    alpha = max((idx + 1) / len(pts), 0.6)
                    try:
                        r, g, b = self._color_for_track(int(tid))
                    except Exception:
                        r, g, b = (1.0, 1.0, 0.0)
                    lp.line_color.set(r, g, b, alpha)
                    display_meta.num_lines += 1
                if (self.trail_show_labels
                        and display_meta.num_labels < 16
                        and display_meta.num_lines < 250
                        and display_meta.num_labels < len(display_meta.text_params)):
                    tp = display_meta.text_params[display_meta.num_labels]
                    tp.display_text = f"id {tid}"
                    tp.x_offset, tp.y_offset = map(int, pts[-1])
                    tp.font_params.font_name = "Serif"
                    tp.font_params.font_size = 12
                    try:
                        r, g, b = self._color_for_track(int(tid))
                    except Exception:
                        r, g, b = (1.0, 1.0, 0.0)
                    tp.font_params.font_color.set(r, g, b, 1.0)
                    tp.set_bg_clr = 0
                    display_meta.num_labels += 1
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
            l_frame = l_frame.next
        return Gst.PadProbeReturn.OK




def create_deepstream_video_processor(
    sources: List[Dict[str, Any]],
    config: AppConfig
) -> DeepStreamVideoPipeline:
    """
    Factory function to create multi-stream DeepStream video processor.
    
    Args:
        sources: List of video source configurations from config.py
        config: Application configuration
        
    Returns:
        Configured multi-stream DeepStream video pipeline
    """
    return DeepStreamVideoPipeline(
        sources=sources,
        config=config,
        websocket_port=config.websocket.PORT,
        config_file="pipelines/config_infer_primary_yolo11.ini",
        preproc_config="pipelines/config_preproc.ini"
    )


if __name__ == "__main__":
    # Test DeepStream pipeline
    import argparse
    from config import config

    parser = argparse.ArgumentParser(description="Test DeepStream Video Pipeline")
    
    # Define the default RTSP URI
    default_rtsp_uri = os.environ.get("NOESIS_RTSP_URI", "rtsps://192.168.3.214:7441/jdr9oLlBkjyl3gDm?enableSrtp")
    
    parser.add_argument("--source-uri", default=default_rtsp_uri, help="Video source (RTSP URL)")
    parser.add_argument("--duration", type=int, default=30, help="Test duration in seconds")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Create test sources from enabled RTSP streams in config
    test_sources = [stream for stream in config.cameras.RTSP_STREAMS if stream.get('enabled', True)]
    if not test_sources:
        print("❌ No enabled RTSP streams found in config")
        exit(1)
    
    print(f"🎥 Testing with {len(test_sources)} streams:")
    for i, source in enumerate(test_sources):
        print(f"  Stream {i}: {source.get('name', 'Unknown')} - {source.get('url', 'No URL')}")
    
    # Create multi-stream pipeline
    pipeline = create_deepstream_video_processor(test_sources, config)

    if pipeline.start():
        print("✅ DeepStream pipeline started successfully")

        # Read frames for specified duration
        start_time = time.time()
        
        while time.time() - start_time < args.duration:
            # The main logic is handled by the pipeline and probes now.
            # We just need to keep the script alive.
            time.sleep(1)

        # Print statistics
        stats = pipeline.get_stats()
        print(f"\nPipeline statistics: {stats}")

        pipeline.stop()
    else:
        print("❌ Failed to start DeepStream pipeline")
