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

os.environ["no_proxy"] = "*"
import logging
import configparser
import re
import threading
import time
import queue
import json
import ctypes
import tempfile
from collections import defaultdict, deque

# Bypass libproxy issues by disabling GIO proxy resolver
from typing import Optional, Tuple, Dict, Any, List, Set, Iterable, TYPE_CHECKING
import numpy as np

import torch
import math

# GStreamer imports
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GLib, GstApp, GObject  # type: ignore  # noqa: E402

# DeepStream imports
sys.path.append("/opt/nvidia/deepstream/deepstream/lib")
import pyds  # type: ignore  # noqa: E402

# Define a stable custom meta type for our MapAnything depth user meta.
# Prefer a named registration via nvds_get_user_meta_type so the value is
# stable across runs and plays nicely with DeepStream bookkeeping. Fall back
# to a slot in the user range if registration is unavailable.
try:
    CUSTOM_MDE_META_TYPE = int(
        pyds.nvds_get_user_meta_type("Noesis.MapAnythingDepthMeta")
    )
except Exception:
    CUSTOM_MDE_META_TYPE = int(pyds.NvDsMetaType.NVDS_START_USER_META) + 10

# GLib helpers for safe text allocations for OSD (DeepStream will g_free)
try:
    _glib = ctypes.CDLL("libglib-2.0.so.0")
    _glib.g_strdup.argtypes = [ctypes.c_char_p]
    _glib.g_strdup.restype = ctypes.c_void_p

    def _alloc_display_text(text: Optional[str]) -> ctypes.c_char_p:
        """Allocate a GLib-managed string for NvOSD display text.

        DeepStream/NvOSD expects ownership of the text pointer and will free it
        with g_free after use. We therefore must allocate using g_strdup rather
        than passing a Python-managed buffer. If GLib is unavailable, return
        a NULL pointer to disable the label safely.
        """
        if not text:
            return ctypes.c_char_p()
        encoded = text.encode("utf-8")
        ptr = _glib.g_strdup(encoded)
        return ctypes.cast(ptr, ctypes.c_char_p)
except Exception:
    _glib = None

    def _alloc_display_text(text: Optional[str]) -> ctypes.c_char_p:
        # Safe fallback when GLib is unavailable: return NULL so DS skips text
        return ctypes.c_char_p()

# Safety cap when reading C strings from user meta to avoid scanning
# unbounded memory if a null terminator is missing due to malformed data.
MAX_META_STRING_BYTES = 65536


def _safe_cstring_from_ptr(ptr_val: int) -> Optional[str]:
    """Best-effort, bounded read of C char* into Python str.

    Attempts pyds.get_string when available, otherwise falls back to a bounded
    ctypes.string_at to reduce risk of runaway reads. Returns None on failure.
    """
    try:
        if not ptr_val:
            return None
        # Prefer pyds.get_string if available in this environment
        try:
            cptr = ctypes.cast(ctypes.c_void_p(ptr_val), ctypes.c_char_p)
            getter = getattr(pyds, "get_string", None)
            if callable(getter):
                s = getter(cptr)  # type: ignore[misc]
                if isinstance(s, str):
                    return s
        except Exception:
            # Fall through to bounded raw read
            pass

        # Bounded raw read with a sane cap to minimize fault surface
        MAX_READ = 1 << 20  # 1 MiB cap
        raw = ctypes.string_at(ptr_val, MAX_READ)
        nul = raw.find(b"\x00")
        if nul != -1:
            raw = raw[:nul]
        return raw.decode("utf-8", "ignore")
    except Exception:
        return None

# Local imports
from websocket_server import WebSocketServer  # noqa: E402

from utils import RateLimitedLogger  # noqa: E402
import requests  # For REST API calls to nvmultiurisrcbin  # noqa: E402

# Defer GStreamer initialization to runtime to avoid crashing at import time
# Some environments (or tracer/proxy setups) can abort during init; doing this
# lazily makes failures easier to diagnose and avoids import-time crashes.

from config import AppConfig  # noqa: E402
from reid.stable_id_manager import StableIDManager  # noqa: E402
from geometry.transform import pixel_to_world, build_align_matrix  # noqa: E402
from pixel_to_world import (
    E_to_world_and_R,
    ray_from_pixel,
    intersect_floor,
    bbox_bottom_center,
)  # noqa: E402
if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from geometry.depth_source import MapAnythingDepthSource


DEPTH_ANNOTATOR_TOLERANCE_US = 350_000

# PyCapsule functions with proper error handling
_CAPSULE_NEW = ctypes.pythonapi.PyCapsule_New
_CAPSULE_NEW.restype = ctypes.py_object
_CAPSULE_NEW.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
_CAPSULE_GET_PTR = ctypes.pythonapi.PyCapsule_GetPointer
_CAPSULE_GET_PTR.restype = ctypes.c_void_p
_CAPSULE_GET_PTR.argtypes = [ctypes.py_object, ctypes.c_char_p]

def safe_pycapsule_new(ptr, name, destructor=None):
    """Safely create a PyCapsule with error handling."""
    try:
        if ptr is None:
            return None
        name_bytes = name.encode('utf-8') if isinstance(name, str) else name
        return _CAPSULE_NEW(ptr, name_bytes, destructor)
    except Exception as e:
        logging.getLogger(__name__).debug(f"Failed to create PyCapsule: {e}")
        return None

def safe_pycapsule_get_ptr(capsule, name):
    """Safely get pointer from PyCapsule with error handling."""
    try:
        if capsule is None:
            return None
        name_bytes = name.encode('utf-8') if isinstance(name, str) else name
        return _CAPSULE_GET_PTR(capsule, name_bytes)
    except Exception as e:
        logging.getLogger(__name__).debug(f"Failed to get PyCapsule pointer: {e}")
        return None

try:
    NVDS_PREPROCESS_BATCH_META = int(
        pyds.nvds_get_user_meta_type("NVIDIA.NvDsPreProcessBatchMeta")
    )
except Exception:
    NVDS_PREPROCESS_BATCH_META = None

class DeepStreamVideoPipeline:
    """
    DeepStream-based video pipeline for GPU-accelerated video processing.

    This pipeline uses:
    - nvurisrcbin for source handling (RTSP/file/camera)
    - nvmultiurisrcbin for batching
    - nvdspreprocess for GPU preprocessing
    - appsink for tensor output
    """

    def __init__(
        self,
        sources: List[Dict[str, Any]],
        config: AppConfig,
        websocket_port: int = 8765,
        config_file: str = "pipelines/config_infer_primary_yolo11.ini",
        preproc_config: str = "pipelines/config_preproc.ini",
    ):
        # Initialize GStreamer as early as possible, but at runtime (not module import)
        try:
            Gst.init(None)
        except Exception as e:
            # Provide clearer guidance if init fails
            raise RuntimeError(
                f"Failed to initialize GStreamer: {e}.\n"
                f"Hints: ensure DeepStream is installed and environment is activated.\n"
                f"Try: source ./activate_deepstream.sh and verify gst-inspect-1.0 works."
            )

        # Multi-stream configuration
        self.sources = [source for source in sources if source.get("enabled", True)]
        self.websocket_port = websocket_port
        self.config_file = config_file
        self.preproc_config = preproc_config
        self.config = config  # Store config as instance variable
        self.logger = logging.getLogger(__name__)
        self.module_dir = os.path.dirname(os.path.abspath(__file__))

        processing_cfg = getattr(self.config, "processing", None)
        self.mapanything_preprocess_config = getattr(
            processing_cfg,
            "DEEPSTREAM_MAPANYTHING_PREPROCESS_CONFIG",
            "pipelines/config_preprocess_mapanything_fused.ini",
        )
        self.mapanything_sgie_config = getattr(
            processing_cfg,
            "DEEPSTREAM_MAPANYTHING_SGIE_CONFIG",
            "pipelines/config_infer_secondary_mapanything_fused.ini",
        )
        self.mapanything_sgie_uid = 22
        intrinsics_default = os.path.join(
            self.module_dir, "models/mapanything_depth/intrinsics_table.txt"
        )
        self.mapanything_intrinsics_table = getattr(
            processing_cfg,
            "MAPANYTHING_INTRINSICS_TABLE",
            intrinsics_default,
        )
        os.environ.setdefault("MA_INTRINSICS_TABLE", self.mapanything_intrinsics_table)

        self.mapanything_preprocess: Optional[Gst.Element] = None
        self.mapanything_sgie: Optional[Gst.Element] = None
        self.mapanything_branch_enabled: bool = True
        default_min_conf = getattr(processing_cfg, "MAPANYTHING_MIN_CONF", None)
        try:
            self._mapanything_min_conf = float(default_min_conf)
        except (TypeError, ValueError):
            self._mapanything_min_conf = 0.5

        self.depth_source: Optional["MapAnythingDepthSource"] = None
        self.calibration_bundle: Optional[Dict[str, Any]] = None
        self.depth_meta_type: int = int(CUSTOM_MDE_META_TYPE)
        self._depth_annotator_stats = {
            "pts_hits": 0,
            "pts_misses": 0,
            "mde": 0,
            "floor": 0,
        }
        self._depth_annotator_last_log = time.time()
        self._mapanything_sgie_stats = {
            "count": 0,
            "last_log": time.time(),
            "cam_ids": set(),
        }

        # Try to load optional C shim for robust ROI→object tensor meta attachment
        self._shim_attach_tensors = None
        self._shim_attach_depth_meta = None
        self._shim_get_last_counts = None
        self._shim_version = None
        try:
            shim_path = os.path.join(self.module_dir, "external/ds_preprocess_shim/libds_preprocess_shim.so")
            if os.path.exists(shim_path):
                lib = ctypes.CDLL(shim_path)
                lib.ds_attach_roi_tensor_to_objects.argtypes = [ctypes.c_void_p, ctypes.c_uint]
                lib.ds_attach_roi_tensor_to_objects.restype = ctypes.c_int
                self._shim_attach_tensors = lib.ds_attach_roi_tensor_to_objects
                lib.ds_preprocess_shim_get_last_counts.argtypes = [
                    ctypes.POINTER(ctypes.c_int),
                    ctypes.POINTER(ctypes.c_int),
                    ctypes.POINTER(ctypes.c_int),
                ]
                lib.ds_preprocess_shim_get_last_counts.restype = None
                self._shim_get_last_counts = lib.ds_preprocess_shim_get_last_counts
                lib.ds_preprocess_shim_version.argtypes = []
                lib.ds_preprocess_shim_version.restype = ctypes.c_int
                try:
                    self._shim_version = int(lib.ds_preprocess_shim_version())
                except Exception:
                    self._shim_version = None
                lib.ds_attach_depth_payload_v2.argtypes = [
                    ctypes.c_void_p,  # batch_meta
                    ctypes.c_void_p,  # obj_meta
                    ctypes.c_uint,    # meta_type
                    ctypes.c_longlong,  # pts_us
                    ctypes.c_char_p,  # method
                    ctypes.c_double,  # depth_m
                    ctypes.c_double,  # conf
                    ctypes.c_int,     # samples
                    ctypes.POINTER(ctypes.c_double),  # world ptr
                    ctypes.c_int,     # world len
                    ctypes.POINTER(ctypes.c_double),  # summary vals
                    ctypes.POINTER(ctypes.c_int),     # summary mask
                    ctypes.c_int,     # summary len
                    ctypes.c_int,     # summary sample count
                    ctypes.c_int,     # has scale
                    ctypes.c_double,  # scale
                    ctypes.POINTER(ctypes.c_double),  # pose ptr
                    ctypes.c_int,     # pose len
                ]
                lib.ds_attach_depth_payload_v2.restype = ctypes.c_int
                self._shim_attach_depth_meta = lib.ds_attach_depth_payload_v2
                if self._shim_version is not None:
                    self.logger.info(
                        "Loaded DS preprocess shim: %s (version=0x%X)",
                        shim_path,
                        self._shim_version,
                    )
                else:
                    self.logger.info("Loaded DS preprocess shim: %s", shim_path)
                self.logger.debug(
                    "DS preprocess shim configured for SGIE UID=%d",
                    self.mapanything_sgie_uid,
                )
            else:
                self.logger.info("DS preprocess shim not found (optional): %s", shim_path)
        except Exception as e:
            self.logger.debug("Failed to load DS preprocess shim: %s", e)

        # Create sensor_id to camera name mapping for telemetry (use stable 1-based IDs)
        self.source_info = {}
        self.sensor_ids: List[int] = []
        self.source_idx_by_sensor_id: Dict[int, int] = {}
        self.sensor_id_by_source_idx: Dict[int, int] = {}
        for index, source in enumerate(self.sources):
            sensor_id = int(source.get("sensor_id", index + 1))  # stable, non-zero
            self.sensor_ids.append(sensor_id)
            self.source_idx_by_sensor_id[sensor_id] = index
            self.sensor_id_by_source_idx[index] = sensor_id

            camera_name = source.get("name", f"Camera_{sensor_id}")
            # Extract clean camera name for UI (e.g., "Living Room Camera" -> "living-room")
            if "Living Room" in camera_name:
                clean_name = "living-room"
            elif "Kitchen" in camera_name:
                clean_name = "kitchen"
            elif "Family Room" in camera_name:
                clean_name = "family-room"
            else:
                clean_name = camera_name.lower().replace(" ", "-").replace("_", "-")

            self.source_info[sensor_id] = {
                "name": camera_name,
                "clean_name": clean_name,
                "url": source.get("url", ""),
                "width": source.get("width", 1920),
                "height": source.get("height", 1080),
            }

        self.logger.info(
            f"🎥 Initializing multi-stream pipeline with {len(self.sources)} sources:"
        )
        for sid in self.sensor_ids:
            info = self.source_info[sid]
            self.logger.info(
                f"  Sensor {sid}: {info['name']} ({info['clean_name']}) - {info['width']}x{info['height']}"
            )

        # Use rate-limited loggers for different types of messages
        self.rate_limited_logger = RateLimitedLogger(
            self.logger, rate_limit_seconds=5.0
        )
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
        self.max_width = max(source.get("width", 1920) for source in self.sources)
        self.max_height = max(source.get("height", 1080) for source in self.sources)
        self.device_id = 0
        # Default REST API port for nvmultiurisrcbin
        self.multiurisrc_port = 9000

        self.logger.info(
            f"📊 Pipeline config: batch_size={self.batch_size}, resolution={self.max_width}x{self.max_height}"
        )

        # Preflight: verify required DeepStream plugins are available with helpful errors
        try:
            registry = Gst.Registry.get()
            required = [
                ("nvmultiurisrcbin", "DeepStream multi-URI source (nvmultiurisrcbin)"),
                ("nvdspreprocess", "DeepStream preprocessor (nvdspreprocess)"),
                ("nvinfer", "DeepStream inference (nvinfer)"),
                ("nvstreamdemux", "DeepStream stream demux (nvstreamdemux)"),
                ("nvdsosd", "DeepStream on-screen display (nvdsosd)"),
                ("nvjpegenc", "NVIDIA JPEG encoder (nvjpegenc)"),
            ]
            missing = []
            for name, desc in required:
                if not registry.find_feature(name, Gst.ElementFactory):
                    missing.append(f"{name} – {desc}")
            if missing:
                hint_env = (
                    "Required DeepStream plugins not found:\n  - "
                    + "\n  - ".join(missing)
                    + "\n\nFix: source DeepStream env and set GST paths, e.g.:\n"
                    "  export DEEPSTREAM_DIR=/opt/nvidia/deepstream/deepstream\n"
                    "  export GST_PLUGIN_PATH=$DEEPSTREAM_DIR/lib/gst-plugins:$GST_PLUGIN_PATH\n"
                    "Also verify with: gst-inspect-1.0 nvmultiurisrcbin\n"
                )
                raise RuntimeError(hint_env)
        except Exception:
            # Bubble up with context so startup reports a clear error instead of aborting
            raise

        # Add missing attributes for compatibility
        self.frame_count = 0
        self.frame_count_lock = threading.Lock()
        # The tensor_queue is no longer needed as metadata is extracted via probe
        # JPEG queues for GPU-encoded frames, keyed by sensor_id
        self.jpeg_queues: Dict[int, queue.Queue[Tuple[bytes, Optional[int]]]] = {
            sensor_id: queue.Queue(maxsize=30) for sensor_id in self.sensor_ids
        }
        self.start_time = 0  # Will be set in start()

        # Add tracking history for trail visualization
        self.trail_history = defaultdict(
            lambda: deque(maxlen=self.config.visualization.TRAIL_LENGTH)
        )
        # Respect config default for initial state
        self.trail_visualization_enabled = bool(
            self.config.visualization.TRAIL_VISUALIZATION_ENABLED
        )

        # --- trail visualisation state ---
        # Maintain histories per sensor_id to avoid cross-stream overlays
        self.trail_last_seen_by_sensor: Dict[int, Dict[int, float]] = defaultdict(dict)
        self.trail_history_by_sensor: Dict[int, defaultdict] = defaultdict(
            lambda: defaultdict(
                lambda: deque(maxlen=self.config.visualization.TRAIL_LENGTH)
            )
        )
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
                "active_tracks": [],
                "occupancy": {},
                "transitions": [],
            }
        self.logger.info(
            f"📊 Initialized tracking state for {len(self.sensor_ids)} streams"
        )

        # --- bbox smoothing state (per sensor_id, per track_id) ---
        try:
            from collections import defaultdict as _dd

            self._bbox_smooth_by_sensor: Dict[int, Dict[int, Dict[str, Any]]] = _dd(
                dict
            )
        except Exception:
            self._bbox_smooth_by_sensor = {}
        self.bbox_smoothing_enabled: bool = bool(
            getattr(self.config.visualization, "BBOX_SMOOTHING_ENABLED", True)
        )
        self.bbox_smoothing_alpha: float = float(
            getattr(self.config.visualization, "BBOX_SMOOTHING_ALPHA", 0.3)
        )
        self.bbox_smoothing_anchor: str = str(
            getattr(self.config.visualization, "BBOX_SMOOTHING_ANCHOR", "bottom")
        ).lower()
        self.bbox_smoothing_max_growth: float = float(
            getattr(self.config.visualization, "BBOX_SMOOTHING_MAX_GROWTH", 1.2)
        )
        self.bbox_smoothing_max_shrink: float = float(
            getattr(self.config.visualization, "BBOX_SMOOTHING_MAX_SHRINK", 0.85)
        )
        self.bbox_max_drop_window_s: float = float(
            getattr(self.config.visualization, "BBOX_MAX_DROP_WINDOW_S", 4.0)
        )
        self.bbox_max_drop_ratio: float = float(
            getattr(self.config.visualization, "BBOX_MAX_DROP_RATIO", 0.85)
        )
        self.trail_max_speed_px_per_s: float = float(
            getattr(self.config.visualization, "TRAIL_MAX_SPEED_PX_PER_S", 600.0)
        )

        # Global Stable ID manager (OSNet/appearance-based global IDs)
        # Respect REID_ENABLED to avoid loading model and consuming GPU if disabled
        try:
            self.reid_enabled: bool = bool(
                getattr(self.config.models, "REID_ENABLED", True)
            )
        except Exception:
            self.reid_enabled = True

        self.stable_id_mgr = None
        if self.reid_enabled:
            try:
                reid_model_path = getattr(self.config.models, "REID_MODEL_PATH", None)
            except Exception:
                reid_model_path = None
            device = getattr(self.config.models, "DEVICE", "cuda:0")
            # Pull model selection and crop size from config
            try:
                reid_model_name = getattr(
                    self.config.models, "REID_MODEL_NAME", "osnet_ibn_x1_0"
                )
            except Exception:
                reid_model_name = "osnet_ibn_x1_0"
            try:
                img_h, img_w = getattr(
                    self.config.models, "REID_IMAGE_SIZE", [256, 128]
                )
                image_size = (int(img_h), int(img_w))
            except Exception:
                image_size = (256, 128)

            self.stable_id_mgr = StableIDManager(
                model_path=reid_model_path,
                device=device,
                model_name=str(reid_model_name),
                image_size=image_size,
                embed_interval_s=float(
                    getattr(self.config.models, "REID_EMBED_INTERVAL_S", 1.0)
                ),
                max_ghost_age_s=float(
                    getattr(self.config.models, "REID_MAX_GHOST_AGE_S", 60.0)
                ),
                cos_sim_threshold=float(
                    getattr(self.config.models, "REID_COS_SIM_THRESHOLD", 0.72)
                ),
                cos_sim_high_threshold=float(
                    getattr(self.config.models, "REID_COS_SIM_HIGH_THRESHOLD", 0.80)
                ),
                allow_multi_zone_active=bool(
                    getattr(self.config.models, "REID_ALLOW_MULTI_ZONE_ACTIVE", True)
                ),
                crop_expand=float(
                    getattr(self.config.models, "REID_CROP_EXPAND", 0.12)
                ),
                tta_flip=bool(getattr(self.config.models, "REID_TTA_FLIP", True)),
                min_crop_h=int(getattr(self.config.models, "REID_MIN_CROP_H", 64)),
                min_laplacian_var=float(
                    getattr(self.config.models, "REID_MIN_LAPLACIAN", 12.0)
                ),
                adaptive_penalty=bool(
                    getattr(self.config.models, "REID_ADAPTIVE_PENALTY", True)
                ),
                size_penalty_alpha=float(
                    getattr(self.config.models, "REID_SIZE_PENALTY_ALPHA", 0.08)
                ),
                brightness_penalty_beta=float(
                    getattr(self.config.models, "REID_BRIGHTNESS_PENALTY_BETA", 0.05)
                ),
                spatial_penalty=bool(
                    getattr(self.config.models, "REID_SPATIAL_PENALTY", True)
                ),
                spatial_penalty_delta=float(
                    getattr(self.config.models, "REID_SPATIAL_PENALTY_DELTA", 0.06)
                ),
                color_penalty_gamma=float(
                    getattr(self.config.models, "REID_COLOR_PENALTY_GAMMA", 0.07)
                ),
                stripe_fusion=bool(
                    getattr(self.config.models, "REID_STRIPE_FUSION", True)
                ),
                stripe_count=int(getattr(self.config.models, "REID_STRIPE_COUNT", 3)),
                multi_scale_crops=bool(
                    getattr(self.config.models, "REID_MULTI_SCALE_CROPS", True)
                ),
                ema_alpha=float(getattr(self.config.models, "REID_EMA_ALPHA", 0.20)),
                active_id_guard_strict=bool(
                    getattr(self.config.models, "REID_ACTIVE_ID_GUARD_STRICT", True)
                ),
                active_id_guard_margin=float(
                    getattr(self.config.models, "REID_ACTIVE_ID_GUARD_MARGIN", 0.03)
                ),
                ghost_strict_age_s=float(
                    getattr(self.config.models, "REID_GHOST_STRICT_AGE_S", 2.0)
                ),
                ghost_extra_margin=float(
                    getattr(self.config.models, "REID_GHOST_EXTRA_MARGIN", 0.03)
                ),
                max_active_ids_per_sensor=int(
                    getattr(self.config.models, "REID_MAX_ACTIVE_IDS_PER_SENSOR", 6)
                ),
                new_id_confirm_frames_at_cap=int(
                    getattr(self.config.models, "REID_NEW_ID_CONFIRM_FRAMES_AT_CAP", 2)
                ),
                new_id_hysteresis_frames=int(
                    getattr(self.config.models, "REID_NEW_ID_HYSTERESIS_FRAMES", 2)
                ),
                active_evict_grace_s=float(
                    getattr(self.config.models, "REID_ACTIVE_EVICT_GRACE_S", 10.0)
                ),
                xcam_handoff_window_s=float(
                    getattr(self.config.models, "REID_XCAM_HANDOFF_WINDOW_S", 4.0)
                ),
                xcam_handoff_margin=float(
                    getattr(self.config.models, "REID_XCAM_HANDOFF_MARGIN", 0.02)
                ),
                max_total_ids=int(
                    getattr(self.config.models, "REID_MAX_TOTAL_IDS", 12)
                ),
                total_id_reuse=bool(
                    getattr(self.config.models, "REID_TOTAL_ID_REUSE", True)
                ),
                total_id_reuse_min_age_s=float(
                    getattr(self.config.models, "REID_TOTAL_ID_REUSE_MIN_AGE_S", 600.0)
                ),
                sid_pool_file=str(
                    getattr(
                        self.config.models,
                        "REID_SID_POOL_FILE",
                        "~/.noesis/sid_pool.json",
                    )
                ),
            )
        # Latest per-sensor JPEG bytes for non-blocking crops
        self._latest_jpeg_bytes_by_sensor: Dict[int, bytes] = {}
        # Decode gating for ReID crops (per sensor)
        from collections import defaultdict as _dd

        self._last_decode_ts_by_sensor: Dict[int, float] = _dd(float)
        try:
            self._reid_decode_min_interval_s: float = float(
                getattr(self.config.models, "REID_DECODE_MIN_INTERVAL_S", 0.2)
            )
        except Exception:
            self._reid_decode_min_interval_s = 0.2

        # Exclusion ROIs parsed from nvdsanalytics exclude config (per DS source index)
        # { stream_index(int): { normalized_label(str): [(x,y), ...] } }
        self._exclusion_rois_by_stream: Dict[
            int, Dict[str, List[Tuple[float, float]]]
        ] = {}

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
        # Negotiated caps per sensor branch for accurate frame dimensions at OSD time
        self._branch_caps_by_sensor: Dict[int, Tuple[int, int]] = {}

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
            if not hasattr(self, "nvinfer") or not self.nvinfer:
                self.logger.warning(
                    "nvinfer element not available for dynamic configuration"
                )
                return False

            success = True

            # Update confidence threshold
            if "confidence_threshold" in config_data:
                try:
                    new_threshold = float(config_data["confidence_threshold"])
                    self.nvinfer.set_property("confidence-threshold", new_threshold)
                    self.logger.info(
                        f"✅ Updated DeepStream confidence threshold to: {new_threshold}"
                    )
                except Exception as e:
                    self.logger.error(f"❌ Failed to update confidence threshold: {e}")
                    success = False

            # Update IOU threshold
            if "iou_threshold" in config_data:
                try:
                    new_iou = float(config_data["iou_threshold"])
                    self.nvinfer.set_property("iou-threshold", new_iou)
                    self.logger.info(
                        f"✅ Updated DeepStream IOU threshold to: {new_iou}"
                    )
                except Exception as e:
                    self.logger.error(f"❌ Failed to update IOU threshold: {e}")
                    success = False

            # Update detection enable/disable
            if "detection_enabled" in config_data:
                try:
                    detection_enabled = bool(config_data["detection_enabled"])
                    self.nvinfer.set_property("enable", detection_enabled)
                    self.logger.info(
                        f"✅ Updated DeepStream detection enabled to: {detection_enabled}"
                    )
                except Exception as e:
                    self.logger.error(f"❌ Failed to update detection enabled: {e}")
                    success = False

            # Update target classes via custom properties
            if "target_classes" in config_data:
                try:
                    new_classes = config_data["target_classes"]
                    if isinstance(new_classes, list):
                        class_string = ",".join(map(str, new_classes))
                        self.nvinfer.set_property(
                            "custom-lib-props", f"target-classes:{class_string}"
                        )
                        self.logger.info(
                            f"✅ Updated DeepStream target classes to: {new_classes}"
                        )
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
            if not hasattr(self, "nvinfer") or not self.nvinfer:
                self.logger.warning(
                    "nvinfer element not available for dynamic configuration"
                )
                return False

            # Get current target classes from nvinfer custom properties
            current_classes = []
            try:
                custom_props = self.nvinfer.get_property("custom-lib-props")
                if custom_props and "target-classes:" in custom_props:
                    class_string = custom_props.split("target-classes:")[1]
                    current_classes = [
                        int(x) for x in class_string.split(",") if x.strip()
                    ]
            except (AttributeError, TypeError, ValueError):
                # If no custom properties set, assume all classes are enabled
                current_classes = list(range(80))  # COCO has 80 classes

            # Update classes based on toggle
            if toggle_name == "detect_people":
                class_id = 0  # person class
                if enabled and class_id not in current_classes:
                    current_classes.append(class_id)
                elif not enabled and class_id in current_classes:
                    current_classes.remove(class_id)

            elif toggle_name == "detect_vehicles":
                vehicle_classes = [
                    1,
                    2,
                    3,
                    5,
                    7,
                    8,
                ]  # bicycle, car, motorcycle, bus, truck, boat
                if enabled:
                    for class_id in vehicle_classes:
                        if class_id not in current_classes:
                            current_classes.append(class_id)
                else:
                    for class_id in vehicle_classes:
                        if class_id in current_classes:
                            current_classes.remove(class_id)

            elif toggle_name == "detect_furniture":
                furniture_classes = [
                    13,
                    56,
                    57,
                    59,
                    60,
                    61,
                ]  # bench, chair, couch, bed, dining table, toilet
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
                class_string = ",".join(map(str, current_classes))
                self.nvinfer.set_property(
                    "custom-lib-props", f"target-classes:{class_string}"
                )
                self.logger.info(
                    f"✅ Updated DeepStream target classes for {toggle_name}: {current_classes}"
                )
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

            if hasattr(self, "nvinfer") and self.nvinfer:
                # Get confidence threshold
                try:
                    config["confidence_threshold"] = self.nvinfer.get_property(
                        "confidence-threshold"
                    )
                except (AttributeError, TypeError, ValueError):
                    config["confidence_threshold"] = 0.3

                # Get IOU threshold
                try:
                    config["iou_threshold"] = self.nvinfer.get_property("iou-threshold")
                except (AttributeError, TypeError, ValueError):
                    config["iou_threshold"] = 0.45

                # Get detection enabled status
                try:
                    config["detection_enabled"] = self.nvinfer.get_property("enable")
                except (AttributeError, TypeError, ValueError):
                    config["detection_enabled"] = True

                # Get target classes
                try:
                    custom_props = self.nvinfer.get_property("custom-lib-props")
                    if custom_props and "target-classes:" in custom_props:
                        class_string = custom_props.split("target-classes:")[1]
                        config["target_classes"] = [
                            int(x) for x in class_string.split(",") if x.strip()
                        ]
                    else:
                        config["target_classes"] = list(range(80))  # All COCO classes
                except (AttributeError, TypeError, ValueError):
                    config["target_classes"] = list(range(80))

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
            workspace_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), ".")
            )

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
                self.logger.warning(
                    f"⚠️  nvinfer config file not found: {config_file_path}"
                )

            if engine_path_from_cfg:
                candidate_paths.append(engine_path_from_cfg)

            # 2) Generate engine path based on actual model file
            if hasattr(self.config, "models") and hasattr(
                self.config.models, "MODEL_PATH"
            ):
                model_path = self.config.models.MODEL_PATH
                if model_path:
                    # Extract model name and generate engine path
                    model_basename = os.path.splitext(os.path.basename(model_path))[0]

                    # Generate multiple possible engine names
                    engine_names = [
                        f"{model_basename}_fp16.engine",  # yolo11m_fp16.engine
                        f"{model_basename}.engine",  # yolo11m.engine
                        f"{model_basename}_b1_gpu0_fp16.engine",  # yolo11m_b1_gpu0_fp16.engine
                        f"{model_basename}.onnx_b1_gpu0_fp16.engine",  # yolo11m.onnx_b1_gpu0_fp16.engine
                        f"detection_fp16.engine",  # fallback
                    ]

                    # Add to search paths in both locations
                    for engine_name in engine_names:
                        candidate_paths.extend(
                            [
                                os.path.join("models", "engines", engine_name),
                                os.path.join("models", engine_name),
                            ]
                        )

            # 3) Config override (if specified)
            if hasattr(self.config, "models") and hasattr(
                self.config.models, "DETECTION_ENGINE_PATH"
            ):
                config_engine_path = self.config.models.DETECTION_ENGINE_PATH
                if config_engine_path:
                    candidate_paths.insert(0, config_engine_path)  # Priority override

            # Resolve & test ----------------------------------------------------------------
            for path in candidate_paths:
                # Skip empty paths
                if not path:
                    continue

                # Resolve relative paths against workspace root
                abs_path = (
                    path if os.path.isabs(path) else os.path.join(workspace_root, path)
                )

                if os.path.exists(abs_path):
                    self.logger.info(f"✅ Found existing TensorRT engine: {abs_path}")
                    self.logger.info(
                        f"✅ Engine file size: {os.path.getsize(abs_path) / (1024 * 1024):.1f} MB"
                    )
                    return  # Engine found – no build required

            # None of the candidates exist --------------------------------------------------
            self.logger.warning(
                "⚠️  No existing TensorRT engine found. nvinfer will build a new one (this may take several minutes)..."
            )
            self.logger.info(f"   Searched locations:")
            for path in candidate_paths:
                if path:
                    abs_path = (
                        path
                        if os.path.isabs(path)
                        else os.path.join(workspace_root, path)
                    )
                    self.logger.info(f"   - {abs_path}")

            # Log the model that will be used for building
            if hasattr(self.config, "models") and hasattr(
                self.config.models, "MODEL_PATH"
            ):
                model_path = self.config.models.MODEL_PATH
                if model_path:
                    abs_model_path = (
                        model_path
                        if os.path.isabs(model_path)
                        else os.path.join(workspace_root, model_path)
                    )
                    if os.path.exists(abs_model_path):
                        self.logger.info(
                            f"✅ Will build engine from model: {abs_model_path}"
                        )
                        self.logger.info(
                            f"   Model file size: {os.path.getsize(abs_model_path) / (1024 * 1024):.1f} MB"
                        )
                    else:
                        self.logger.error(f"❌ Model file not found: {abs_model_path}")

        except Exception as e:
            self.logger.error(f"Error during engine-file check: {e}")


    def _is_object_excluded(self, obj_meta: "pyds.NvDsObjectMeta") -> bool:
        """Determine whether an object was flagged for exclusion earlier."""
        try:
            if float(getattr(obj_meta, "confidence", 0.0)) < 0.0:
                return True
        except Exception:
            pass
        try:
            if int(getattr(obj_meta, "class_id", 0)) < 0:
                return True
        except Exception:
            pass
        return False

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

                try:
                    ds_index = int(frame_meta.source_id)
                except Exception:
                    ds_index = int(getattr(frame_meta, "source_id", 0))

                exclusion_polys = self._exclusion_rois_by_stream.get(ds_index, {})

                rect = obj_meta.rect_params
                left, top, width, height = (
                    float(rect.left),
                    float(rect.top),
                    float(rect.width),
                    float(rect.height),
                )
                x2, y2 = left + width, top + height
                bbox_corners = [(left, top), (x2, top), (x2, y2), (left, y2)]

                candidate_polys: List[List[Tuple[float, float]]] = []
                l_user = obj_meta.obj_user_meta_list
                while l_user:
                    try:
                        user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                    except StopIteration:
                        break

                    if user_meta.base_meta.meta_type == pyds.nvds_get_user_meta_type(
                        "NVIDIA.DSANALYTICSOBJ.USER_META"
                    ):
                        ainfo = pyds.NvDsAnalyticsObjInfo.cast(user_meta.user_meta_data)
                        labels = self._normalize_roi_status_labels(
                            getattr(ainfo, "roiStatus", None)
                        )
                        for lbl in labels:
                            raw_poly = exclusion_polys.get(lbl)
                            sanitized = self._sanitize_polygon(raw_poly)
                            if sanitized is not None:
                                candidate_polys.append(sanitized)
                    try:
                        l_user = l_user.next
                    except StopIteration:
                        break

                if not candidate_polys and exclusion_polys:
                    for raw_poly in exclusion_polys.values():
                        sanitized = self._sanitize_polygon(raw_poly)
                        if sanitized is not None:
                            candidate_polys.append(sanitized)

                remove = False
                for poly in candidate_polys:
                    if self._bbox_fully_inside_polygon(bbox_corners, poly):
                        remove = True
                        break

                if remove:
                    removed_count += 1
                    try:
                        obj_meta.confidence = float("-inf")
                        obj_meta.class_id = -1
                    except Exception:
                        pass
                    try:
                        rect = obj_meta.rect_params
                        rect.width = 0.0
                        rect.height = 0.0
                        rect.border_width = 0
                    except Exception:
                        pass
                    try:
                        tid = int(getattr(obj_meta, "object_id", -1))
                    except Exception:
                        tid = -1
                    if tid != -1:
                        try:
                            self.trail_history.pop(tid, None)
                        except Exception:
                            pass
                        try:
                            self.track_motion_state_by_sensor.get(ds_index, {}).pop(
                                tid, None
                            )
                        except Exception:
                            pass
                        try:
                            sensor_trails = self.trail_history_by_sensor.get(ds_index)
                            if sensor_trails is not None:
                                sensor_trails.pop(tid, None)
                        except Exception:
                            pass
                        try:
                            last_seen_map = self.trail_last_seen_by_sensor.get(ds_index)
                            if last_seen_map is not None:
                                last_seen_map.pop(tid, None)
                        except Exception:
                            pass

                l_obj = l_obj_next

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        if removed_count > 4 and self.frame_count <= 300:
            self.logger.info(
                f"Frame {self.frame_count}: Flagged {removed_count} objects from exclusion zone."
            )

        return Gst.PadProbeReturn.OK

    # ---- Exclusion ROI helpers ----
    def _load_exclusion_rois_from_config(self, path: str) -> None:
        """Parse nvdsanalytics exclude config and cache ROI polygons per stream index.

        Sections are of the form [roi-filtering-stream-<index>], and keys like roi-<LABEL>=x;y;...
        """
        try:
            cfg = configparser.ConfigParser()
            read = cfg.read(path)
            if not read:
                self.logger.warning(
                    f"⚠️ Exclusion ROI config not found or unreadable: {path}"
                )
                return
            pat = re.compile(r"^roi-filtering-stream-(\d+)$", re.IGNORECASE)
            total = 0
            rois_by_stream: Dict[int, Dict[str, List[Tuple[float, float]]]] = {}
            for section in cfg.sections():
                m = pat.match(section.strip())
                if not m:
                    continue
                sidx = int(m.group(1))
                rois_by_stream.setdefault(sidx, {})
                for key, val in cfg.items(section):
                    if not key.lower().startswith("roi-"):
                        continue
                    label_raw = key[len("roi-") :]
                    label_norm = self._normalize_roi_label(label_raw)
                    pts = self._parse_points_list(val)
                    if len(pts) >= 3:
                        # Store under both normalized and original forms for robust lookup
                        rois_by_stream[sidx][label_norm] = pts
                        rois_by_stream[sidx][
                            self._normalize_roi_label("roi-" + label_raw)
                        ] = pts
                        total += 1
            self._exclusion_rois_by_stream = rois_by_stream
            if total:
                self.logger.info(f"✅ Loaded {total} exclusion ROI(s) from {path}")
            else:
                self.logger.info(f"ℹ️ No exclusion ROIs found in {path}")
        except Exception as e:
            self.logger.error(f"Error parsing exclusion ROI config {path}: {e}")

    def _parse_points_list(self, s: str) -> List[Tuple[float, float]]:
        """Parse 'x1;y1; x2;y2; ...' into [(x1,y1), ...]."""
        try:
            # Split by delimiters and filter empties
            tokens = re.split(r"[\s;,]+", s.strip())
            nums = [float(t) for t in tokens if t != ""]
            pts: List[Tuple[float, float]] = []
            for i in range(0, len(nums) - 1, 2):
                pts.append((nums[i], nums[i + 1]))
            return pts
        except Exception:
            return []

    def _normalize_roi_label(self, label: str) -> str:
        """Normalize ROI label for consistent matching (case-insensitive, strip 'roi-' prefix)."""
        lbl = str(label).strip()
        # Remove optional leading 'roi-'
        if lbl.lower().startswith("roi-"):
            lbl = lbl[4:]
        return lbl.strip().lower()

    def _normalize_roi_status_labels(self, roi_status: Any) -> set:
        """Normalize various roiStatus formats into comparable labels."""
        labels = set()
        try:
            if roi_status is None:
                return labels
            if isinstance(roi_status, dict):
                for key, value in roi_status.items():
                    if value in (1, True, "IN", "inside", "INROI", "in"):
                        labels.add(self._normalize_roi_label(key))
            elif isinstance(roi_status, (list, tuple, set)):
                for item in roi_status:
                    labels.add(self._normalize_roi_label(str(item)))
            elif isinstance(roi_status, str):
                for token in re.split(r"[\s,;]+", roi_status.strip()):
                    if token:
                        labels.add(self._normalize_roi_label(token))
        except Exception:
            pass
        return labels

    def _sanitize_polygon(
        self, poly: Optional[Iterable[Tuple[float, float]]]
    ) -> Optional[List[Tuple[float, float]]]:
        if not poly:
            return None
        sanitized: List[Tuple[float, float]] = []
        try:
            for pt in poly:
                if pt is None:
                    return None
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    x = float(pt[0])
                    y = float(pt[1])
                else:
                    return None
                if not math.isfinite(x) or not math.isfinite(y):
                    return None
                sanitized.append((x, y))
        except Exception:
            return None
        if len(sanitized) < 3:
            return None
        return sanitized

    def _point_in_polygon(
        self, x: float, y: float, poly: List[Tuple[float, float]]
    ) -> bool:
        """Ray casting algorithm for point-in-polygon. Includes boundary as inside."""
        inside = False
        n = len(poly)
        if n < 3:
            return False
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            # Check if point is on a horizontal boundary
            if (y == y1 == y2) and min(x1, x2) <= x <= max(x1, x2):
                return True
            # Ray intersects segment?
            intersects = ((y1 > y) != (y2 > y)) and (
                x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1
            )
            if intersects:
                inside = not inside
        return inside

    def _bbox_fully_inside_polygon(
        self, corners: List[Tuple[float, float]], poly: List[Tuple[float, float]]
    ) -> bool:
        """Return True if all bbox corners are inside the polygon."""
        for px, py in corners:
            if not self._point_in_polygon(px, py, poly):
                return False
        return True

    def _configure_elements(self, elements: Dict[str, Any]) -> bool:
        """Configure properties for all pipeline elements."""
        try:
            self.logger.info(
                "------------- Configuring GStreamer Elements-------------"
            )

            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self._on_bus_message)

            # Streammux configuration - dynamic batch size for multi-stream
            # nvmultiurisrcbin (manages sources internally)
            uri_list = ",".join(self.source_info[sid]["url"] for sid in self.sensor_ids)
            # Use stable 1-based sensor IDs
            sensor_id_list = ",".join(str(sid) for sid in self.sensor_ids)
            elements["multiurisrc"].set_property("uri-list", uri_list)
            elements["multiurisrc"].set_property("sensor-id-list", sensor_id_list)
            elements["multiurisrc"].set_property("max-batch-size", self.batch_size)
            elements["multiurisrc"].set_property("width", self.max_width)
            elements["multiurisrc"].set_property("height", self.max_height)
            elements["multiurisrc"].set_property("batched-push-timeout", 40000)
            elements["multiurisrc"].set_property("live-source", 1)
            elements["multiurisrc"].set_property("drop-pipeline-eos", 1)
            elements["multiurisrc"].set_property("rtsp-reconnect-interval", 30)
            elements["multiurisrc"].set_property("port", self.multiurisrc_port)
            elements["multiurisrc"].set_property("ip-address", "localhost")
            self.logger.info(
                f"📊 nvmultiurisrcbin configured: max-batch-size={self.batch_size}, resolution={self.max_width}x{self.max_height}"
            )

            # Resolve all config file paths to absolute so startup is independent of CWD
            _root_dir = os.path.dirname(os.path.abspath(__file__))
            # Preprocess config
            _preproc_cfg = getattr(
                self.config.processing,
                "DEEPSTREAM_PREPROCESS_CONFIG",
                "pipelines/config_preproc.ini",
            )
            if _preproc_cfg and not os.path.isabs(_preproc_cfg):
                _preproc_cfg = os.path.join(_root_dir, _preproc_cfg)
            elements["preprocess"].set_property("config-file", _preproc_cfg)

            # Primary nvinfer config
            _nvinfer_cfg = self.config_file
            if _nvinfer_cfg and not os.path.isabs(_nvinfer_cfg):
                _nvinfer_cfg = os.path.join(_root_dir, _nvinfer_cfg)
            # Use resolved path for engine-file check and element property
            self._check_for_engine_file(_nvinfer_cfg)
            elements["nvinfer"].set_property("config-file-path", _nvinfer_cfg)
            # Ensure raw tensor outputs can be attached if needed
            try:
                elements["nvinfer"].set_property("output-tensor-meta", True)
            except Exception:
                pass

            if self.mapanything_branch_enabled:
                _map_pre_cfg = self.mapanything_preprocess_config
                if _map_pre_cfg and not os.path.isabs(_map_pre_cfg):
                    _map_pre_cfg = os.path.join(_root_dir, _map_pre_cfg)
                elements["mapanything_preprocess"].set_property(
                    "config-file", _map_pre_cfg
                )
                try:
                    self.logger.debug(
                        "MapAnything Preprocess: config-file=%s", _map_pre_cfg
                    )
                except Exception:
                    pass
                _map_sgie_cfg = self.mapanything_sgie_config
                if _map_sgie_cfg and not os.path.isabs(_map_sgie_cfg):
                    _map_sgie_cfg = os.path.join(_root_dir, _map_sgie_cfg)
                elements["mapanything_sgie"].set_property(
                    "config-file-path", _map_sgie_cfg
                )
                
                # Canonical DS 7.1 settings for SGIE consuming nvdspreprocess tensors
                try:
                    elements["mapanything_sgie"].set_property(
                        "output-tensor-meta", True
                    )
                except Exception:
                    pass
                try:
                    elements["mapanything_sgie"].set_property(
                        "input-tensor-from-meta", True
                    )
                except Exception:
                    pass
                try:
                    elements["mapanything_sgie"].set_property(
                        "input-blob-names", "mapanything_fused"
                    )
                except Exception:
                    pass

            # Exclusion analytics
            exclude_cfg_path = "pipelines/config_nvdsanalytics_exclude.ini"
            if exclude_cfg_path and not os.path.isabs(exclude_cfg_path):
                exclude_cfg_path = os.path.join(_root_dir, exclude_cfg_path)
            elements["nvdsanalytics_exclude"].set_property("unique-id", 101)
            elements["nvdsanalytics_exclude"].set_property(
                "config-file", exclude_cfg_path
            )
            # Pre-parse exclusion ROIs for robust containment checks in pad probe
            self._load_exclusion_rois_from_config(exclude_cfg_path)

            # Tracker configuration
            elements["nvtracker"].set_property(
                "ll-lib-file",
                "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
            )
            tracker_config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "pipelines/config_tracker_nvdcf_batch.yml",
            )
            elements["nvtracker"].set_property("ll-config-file", tracker_config_path)

            # Post-tracker analytics
            elements["nvdsanalytics_post"].set_property("unique-id", 201)
            _post_cfg_path = "pipelines/config_nvdsanalytics_post.ini"
            if _post_cfg_path and not os.path.isabs(_post_cfg_path):
                _post_cfg_path = os.path.join(_root_dir, _post_cfg_path)
            elements["nvdsanalytics_post"].set_property("config-file", _post_cfg_path)

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
            if not elements["multiurisrc"].link(elements["preprocess"]):
                raise RuntimeError("Failed to link nvmultiurisrcbin to preprocess")
            else:
                self.logger.info("TRACE linked nvmultiurisrcbin → preprocess")

            if not elements["preprocess"].link(elements["nvinfer"]):
                raise RuntimeError("Failed to link preprocess to nvinfer")
            if not elements["nvinfer"].link(elements["q_after_pgie"]):
                raise RuntimeError("Failed to link nvinfer to q_after_pgie")
            if not elements["q_after_pgie"].link(elements["nvdsanalytics_exclude"]):
                raise RuntimeError(
                    "Failed to link q_after_pgie to nvdsanalytics_exclude"
                )
            if not elements["nvdsanalytics_exclude"].link(elements["q_after_exclude"]):
                raise RuntimeError(
                    "Failed to link nvdsanalytics_exclude to q_after_exclude"
                )
            if not elements["q_after_exclude"].link(elements["mapanything_preprocess"]):
                raise RuntimeError(
                    "Failed to link q_after_exclude to mapanything_preprocess"
                )
            if not elements["mapanything_preprocess"].link(elements["mapanything_sgie"]):
                raise RuntimeError(
                    "Failed to link mapanything_preprocess to mapanything_sgie"
                )
            # Add probe to check objects fed to SGIE input
            map_pre_src = elements["mapanything_preprocess"].get_static_pad("src")
            if map_pre_src:
                map_pre_src.add_probe(
                    Gst.PadProbeType.BUFFER, self._sgie_input_probe, None
                )
                self.logger.info(
                    "✅ Added SGIE input probe to mapanything_preprocess src pad"
                )
            if not elements["mapanything_sgie"].link(elements["q_before_tracker"]):
                raise RuntimeError("Failed to link mapanything_sgie to q_before_tracker")
            if not elements["q_before_tracker"].link(elements["nvtracker"]):
                raise RuntimeError("Failed to link q_before_tracker to nvtracker")
            if not elements["nvtracker"].link(elements["q_after_tracker"]):
                raise RuntimeError("Failed to link nvtracker to q_after_tracker")
            if not elements["q_after_tracker"].link(elements["nvdsanalytics_post"]):
                raise RuntimeError(
                    "Failed to link q_after_tracker to nvdsanalytics_post"
                )
            if not elements["nvdsanalytics_post"].link(elements["demux"]):
                raise RuntimeError("Failed to link nvdsanalytics_post to demux")

            self.logger.info("✅ Main pipeline chain linked successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error linking pipeline chain: {e}")
            return False

    def set_mapanything_intrinsics_table(self, path: str) -> None:
        if not path:
            return
        self.mapanything_intrinsics_table = path
        os.environ["MA_INTRINSICS_TABLE"] = path

    def open_mapanything_branch(self) -> None:
        # Branch remains active; retained for API compatibility
        self.logger.debug("MapAnything branch already active (fused SGIE path).")

    def _create_pipeline(self) -> bool:
        """Create the DeepStream GStreamer pipeline using refactored helper functions."""
        try:
            # --- Phase A: Create All Elements ---
            self.logger.info("------------- Creating GStreamer Elements-------------")
            self.pipeline = Gst.Pipeline()
            if not self.pipeline:
                raise RuntimeError("Failed to create pipeline")

            # Create primary elements
            multiurisrc = Gst.ElementFactory.make(
                "nvmultiurisrcbin", "nvmultiurisrcbin"
            )
            preprocess = Gst.ElementFactory.make("nvdspreprocess", "nvdspreprocess")
            nvinfer = Gst.ElementFactory.make("nvinfer", "nvinfer")
            mapanything_branch_enabled = True

            mapanything_preprocess = Gst.ElementFactory.make(
                "nvdspreprocess", "mapanything_preprocess"
            )
            if not mapanything_preprocess:
                raise RuntimeError(
                    "Required GStreamer element 'nvdspreprocess' not found. "
                    "Ensure DeepStream preprocess plugin is installed."
                )

            mapanything_sgie = Gst.ElementFactory.make(
                "nvinfer", "mapanything_sgie_fused"
            )
            if not mapanything_sgie:
                raise RuntimeError(
                    "Required GStreamer element 'nvinfer' not found. "
                    "Ensure DeepStream inference plugin is installed."
                )

            # Analytics and Tracking
            nvdsanalytics_exclude = Gst.ElementFactory.make(
                "nvdsanalytics", "nvdsanalytics_exclude"
            )
            nvtracker = Gst.ElementFactory.make("nvtracker", "nvtracker")
            nvdsanalytics_post = Gst.ElementFactory.make(
                "nvdsanalytics", "nvdsanalytics_post"
            )

            # Demuxer (per-branch OSD will be created downstream)
            demux = Gst.ElementFactory.make("nvstreamdemux", "nvstreamdemux")
            if not demux:
                raise RuntimeError("Failed to create nvstreamdemux element.")

            # Queues for pipeline robustness
            q_after_pgie = Gst.ElementFactory.make("queue", "q_after_pgie")
            q_after_exclude = Gst.ElementFactory.make("queue", "q_after_exclude")
            q_before_tracker = Gst.ElementFactory.make("queue", "q_before_tracker")
            q_after_tracker = Gst.ElementFactory.make("queue", "q_after_tracker")

            # Validate element creation
            element_list = [
                multiurisrc,
                preprocess,
                nvinfer,
                mapanything_preprocess,
                mapanything_sgie,
                nvdsanalytics_exclude,
                nvtracker,
                nvdsanalytics_post,
                demux,
                q_after_pgie,
                q_after_exclude,
                q_before_tracker,
                q_after_tracker,
            ]

            if not all(element_list):
                element_names = [
                    "nvmultiurisrcbin",
                    "nvdspreprocess",
                    "nvinfer",
                    "mapanything_preprocess",
                    "mapanything_sgie_fused",
                    "nvdsanalytics_exclude",
                    "nvtracker",
                    "nvdsanalytics_post",
                    "nvstreamdemux",
                    "q_after_pgie",
                    "q_after_exclude",
                    "q_before_tracker",
                    "q_after_tracker",
                ]
                for el, name in zip(element_list, element_names):
                    if not el:
                        self.logger.error(f"❌ Failed to create element: {name}")
                raise RuntimeError("Failed to create one or more GStreamer elements.")
            self.logger.info("✅ All GStreamer elements created successfully.")

            self.mapanything_branch_enabled = mapanything_branch_enabled

            # Create elements dictionary for helper functions
            elements = {
                "multiurisrc": multiurisrc,
                "preprocess": preprocess,
                "nvinfer": nvinfer,
                "mapanything_preprocess": mapanything_preprocess,
                "mapanything_sgie": mapanything_sgie,
                "nvdsanalytics_exclude": nvdsanalytics_exclude,
                "nvtracker": nvtracker,
                "nvdsanalytics_post": nvdsanalytics_post,
                "demux": demux,
                "q_after_pgie": q_after_pgie,
                "q_after_exclude": q_after_exclude,
                "q_before_tracker": q_before_tracker,
                "q_after_tracker": q_after_tracker,
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
            if not analytics_src_pad:
                raise RuntimeError("Failed to get nvdsanalytics_post source pad")
            analytics_src_pad.add_probe(
                Gst.PadProbeType.BUFFER, self._analytics_probe, 0
            )
            self.logger.info(
                "✅ Added buffer probe to nvdsanalytics_post source pad for telemetry extraction"
            )

            # Pad probe to remove excluded objects
            exclude_src_pad = nvdsanalytics_exclude.get_static_pad("src")
            if not exclude_src_pad:
                raise RuntimeError("Failed to get nvdsanalytics_exclude source pad")
            exclude_src_pad.add_probe(
                Gst.PadProbeType.BUFFER, self._remove_excluded_objects_probe, None
            )
            self.logger.info(
                "✅ Added buffer probe to nvdsanalytics_exclude source pad for object removal"
            )

            # Per-branch OSD probe will be attached in per-stream branches

            # (Removed noisy mux src probe)

            # Optional: Add probe to demux sink pad to trace buffer flow for first few frames
            demux_sink = demux.get_static_pad("sink")
            if demux_sink:
                self.logger.debug(
                    "Adding buffer probe to demux sink pad (limited logging)"
                )
                demux_sink.add_probe(
                    Gst.PadProbeType.BUFFER, self._demux_debug_probe, None
                )

            # --- Phase E: Link Main Pipeline Chain ---
            if not self._build_main_pipeline_chain(elements):
                raise RuntimeError("Failed to link main pipeline chain")

            # Store references early for downstream setup that accesses them
            self.multiurisrc, self.preprocess, self.nvinfer, self.nvtracker = (
                multiurisrc,
                preprocess,
                nvinfer,
                nvtracker,
            )
            self.mapanything_preprocess = mapanything_preprocess
            self.mapanything_sgie = mapanything_sgie
            self.nvdsanalytics_exclude, self.nvdsanalytics_post = (
                nvdsanalytics_exclude,
                nvdsanalytics_post,
            )
            try:
                self.logger.debug(
                    "SGIE unique-id=%s",
                    mapanything_sgie.get_property("unique-id"),
                )
            except Exception:
                pass
            self.demux = demux

            if self.mapanything_branch_enabled:
                sgie_src_pad = mapanything_sgie.get_static_pad("src")
                if sgie_src_pad:
                    sgie_src_pad.add_probe(
                        Gst.PadProbeType.BUFFER, self._mapanything_depth_probe, None
                    )
                    self.logger.info(
                        "✅ Added MapAnything fused SGIE probe to mapanything_sgie_fused src pad"
                    )
                else:
                    self.logger.warning(
                        "⚠️ Unable to attach MapAnything SGIE probe – src pad unavailable"
                    )
            else:
                self.logger.info(
                    "ℹ️ MapAnything SGIE branch disabled; skipping depth probe attachment"
                )

            # --- Phase F: Calibrate demux pads and build per-stream branches with per-branch OSD ---
            self._calibrate_demux_pad_source_map()
            self.logger.info(
                "🎥 Building per-stream branches with per-branch OSD and JPEG appsinks"
            )
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
        # Note: Per-frame telemetry broadcast disabled to avoid client overload.
        # Periodic stats broadcast (1 Hz) remains enabled via WebSocketServer.
        return Gst.PadProbeReturn.OK

    def _mapanything_depth_probe(self, pad, info, user_data=None):
        """Consume MapAnything depth metadata attached by the native shim."""
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        if self._shim_attach_depth_meta is None and self._shim_attach_tensors is None:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        while l_frame:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break
            except Exception:
                l_frame = l_frame.next
                continue

            ds_index = int(getattr(frame_meta, "source_id", -1))
            sensor_id = self.sensor_id_by_source_idx.get(ds_index)
            if sensor_id is None:
                l_frame = l_frame.next
                continue

            cam_info = self.source_info.get(sensor_id, {})
            cam_id = str(
                cam_info.get("clean_name") or cam_info.get("name") or sensor_id
            )

            self._depth_annotator_stats["pts_hits"] += 1

            if self._shim_attach_tensors is not None:
                try:
                    self._shim_attach_tensors(
                        ctypes.c_void_p(pyds.get_ptr(batch_meta)),
                        ctypes.c_uint(self.mapanything_sgie_uid),
                    )
                except Exception as attach_err:
                    self.logger.debug(
                        "Depth probe: shim tensor attach failed: %s", attach_err
                    )

            l_obj = frame_meta.obj_meta_list
            while l_obj:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break
                except Exception:
                    l_obj = l_obj.next
                    continue

                if self._is_object_excluded(obj_meta):
                    l_obj = l_obj.next
                    continue

                depth_payload = self._extract_depth_meta(obj_meta)
                if depth_payload:
                    self._record_mapanything_sgie_detection(cam_id)
                    method_name = depth_payload.get("method")
                    depth_val = depth_payload.get("depth_m", 0.0)
                    try:
                        depth_val = float(depth_val if depth_val is not None else 0.0)
                    except Exception:
                        depth_val = 0.0
                    if method_name == "mde" and depth_val > 0.0:
                        self._depth_annotator_stats["mde"] += 1
                    else:
                        self._depth_annotator_stats["floor"] += 1

                l_obj = l_obj.next

            l_frame = l_frame.next

        self._maybe_log_depth_annotator_stats()
        self._maybe_log_mapanything_sgie_stats()
        return Gst.PadProbeReturn.OK

    def _sgie_input_probe(self, pad, info, user_data=None):
        """Probe to log objects fed to SGIE input."""
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            self.logger.debug("SGIE input probe: No batch_meta")
            return Gst.PadProbeReturn.OK

        obj_count = 0
        l_frame = batch_meta.frame_meta_list
        while l_frame:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                obj_count += self._count_objects(frame_meta)
            except Exception as e:
                self.logger.debug(f"SGIE input probe: Error processing frame: {e}")
            l_frame = l_frame.next

        self.logger.debug(f"SGIE input probe: {obj_count} objects in batch")
        return Gst.PadProbeReturn.OK

    def _count_objects(self, frame_meta) -> int:
        """Count class=0 objects in a frame."""
        count = 0
        l_obj = frame_meta.obj_meta_list
        while l_obj:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                if getattr(obj_meta, "class_id", -1) == 0:
                    count += 1
            except Exception:
                pass
            l_obj = l_obj.next
        return count

    def _bundle_K(self, cam_id: str) -> Optional[np.ndarray]:
        bundle = self.calibration_bundle
        if not isinstance(bundle, dict):
            return None
        cameras = (
            bundle.get("cameras") if isinstance(bundle.get("cameras"), dict) else None
        )
        if isinstance(cameras, dict):
            k_table = cameras.get("K") if isinstance(cameras.get("K"), dict) else None
            if isinstance(k_table, dict) and cam_id in k_table:
                try:
                    K = np.array(k_table[cam_id], dtype=float)
                    return K.reshape((3, 3)) if K.size == 9 else K
                except Exception:
                    pass
            cam_entry = cameras.get(cam_id)
            if isinstance(cam_entry, dict):
                for key in ("intrinsics", "K", "K3x3"):
                    value = cam_entry.get(key)
                    if value is None:
                        continue
                    try:
                        K = np.array(value, dtype=float)
                        return K.reshape((3, 3)) if K.size == 9 else K
                    except Exception:
                        continue
        return None

    def _bundle_E(self, cam_id: str) -> Optional[List[float]]:
        bundle = self.calibration_bundle
        if not isinstance(bundle, dict):
            return None
        cameras = (
            bundle.get("cameras") if isinstance(bundle.get("cameras"), dict) else None
        )
        if isinstance(cameras, dict):
            e_table = cameras.get("E") if isinstance(cameras.get("E"), dict) else None
            if isinstance(e_table, dict) and cam_id in e_table:
                value = e_table.get(cam_id)
                if isinstance(value, list) and len(value) == 16:
                    return value
            cam_entry = cameras.get(cam_id)
            if isinstance(cam_entry, dict):
                maybe = cam_entry.get("extrinsics")
                if isinstance(maybe, dict):
                    value = maybe.get("E") or maybe.get("matrix")
                    if isinstance(value, list) and len(value) == 16:
                        return value
        return None

    def _bundle_align(self) -> Optional[Dict[str, Any]]:
        bundle = self.calibration_bundle
        if isinstance(bundle, dict):
            align = bundle.get("align")
            if isinstance(align, dict):
                return align
        return None

    def _to_world_from_snap(
        self,
        cam_id: str,
        u: float,
        v: float,
        depth_m: float,
        snap: Dict[str, Any],
    ) -> Optional[List[float]]:
        if depth_m <= 0.0:
            return None
        K = snap.get("K")
        if K is None:
            K = snap.get("intrinsics")
        if K is None:
            K = snap.get("native_intrinsics")
        if K is None:
            K = self._bundle_K(cam_id)
        if K is None:
            return None
        E = snap.get("E")
        if E is None:
            E = self._bundle_E(cam_id)
        if E is None:
            return None
        align = self._bundle_align()
        try:
            world = pixel_to_world(u, v, depth_m, cam_id, K, E, align)
            return [float(world[0]), float(world[1]), float(world[2])]
        except Exception:
            return None

    def _floor_intersect_world(
        self,
        cam_id: str,
        u: float,
        v: float,
        snap: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[float]]:
        K = None
        if snap:
            K = snap.get("K")
            if K is None:
                K = snap.get("intrinsics")
            if K is None:
                K = snap.get("native_intrinsics")
        if K is None:
            K = self._bundle_K(cam_id)
        E = None
        if snap:
            E = snap.get("E")
        if E is None:
            E = self._bundle_E(cam_id)
        if K is None or E is None:
            return None
        try:
            K_mat = np.array(K, dtype=float).reshape((3, 3))
        except Exception:
            return None
        pose = E_to_world_and_R(E)
        if pose is None:
            return None
        C_world, R_wc = pose
        try:
            origin, direction = ray_from_pixel(u, v, K_mat, C_world, R_wc)
        except Exception:
            return None
        align = self._bundle_align()
        floor_y = float((align or {}).get("floor_y") or 0.0)
        hit = intersect_floor(origin, direction, floor_y)
        if hit is None:
            return None
        align_matrix = build_align_matrix(align)
        point = np.array([hit[0], hit[1], hit[2], 1.0], dtype=float)
        aligned = align_matrix @ point
        return [float(aligned[0]), float(aligned[1]), float(aligned[2])]

    def _maybe_log_depth_annotator_stats(self) -> None:
        now = time.time()
        if (now - self._depth_annotator_last_log) < 5.0:
            return
        stats = self._depth_annotator_stats
        self.logger.debug(
            "Depth annotator stats: hits=%s misses=%s mde=%s floor=%s",
            stats["pts_hits"],
            stats["pts_misses"],
            stats["mde"],
            stats["floor"],
        )
        stats.update({"pts_hits": 0, "pts_misses": 0, "mde": 0, "floor": 0})
        self._depth_annotator_last_log = now

    def _record_mapanything_sgie_detection(self, cam_id: str) -> None:
        stats = self._mapanything_sgie_stats
        stats["count"] += 1
        stats["cam_ids"].add(cam_id)

    def _maybe_log_mapanything_sgie_stats(self) -> None:
        stats = self._mapanything_sgie_stats
        now = time.time()
        elapsed = now - stats["last_log"]
        if elapsed < 1.0:
            return
        count = stats["count"]
        sources = ", ".join(sorted(stats["cam_ids"])) if stats["cam_ids"] else "none"
        self.logger.debug(
            "MapAnything SGIE processed %d detections over %.2fs (sources=%s)",
            count,
            elapsed,
            sources,
        )
        stats["count"] = 0
        stats["cam_ids"].clear()
        stats["last_log"] = now

    def _extract_analytics_frame_meta(self, frame_meta) -> Optional[Dict[str, Any]]:
        """Extract analytics frame metadata"""
        try:
            user_meta_list = frame_meta.frame_user_meta_list
            while user_meta_list:
                user_meta = pyds.NvDsUserMeta.cast(user_meta_list.data)  # type: ignore
                if user_meta.base_meta.meta_type == pyds.nvds_get_user_meta_type(
                    "NVIDIA.DSANALYTICSFRAME.USER_META"
                ):  # type: ignore
                    analytics_frame_meta = pyds.NvDsAnalyticsFrameMeta.cast(
                        user_meta.user_meta_data
                    )  # type: ignore
                    return {
                        "objects_in_roi": analytics_frame_meta.objInROIcnt,
                        "line_crossing_cumulative": analytics_frame_meta.objLCCumCnt,
                        "line_crossing_current": analytics_frame_meta.objLCCurrCnt,
                        "overcrowding_status": analytics_frame_meta.ocStatus,
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
                if user_meta.base_meta.meta_type == pyds.nvds_get_user_meta_type(
                    "NVIDIA.DSANALYTICSOBJ.USER_META"
                ):  # type: ignore
                    analytics_obj_meta = pyds.NvDsAnalyticsObjInfo.cast(
                        user_meta.user_meta_data
                    )  # type: ignore
                    # Normalize to DeepStream SDK key casing so downstream logic works
                    return {
                        "dirStatus": analytics_obj_meta.dirStatus,
                        "lcStatus": analytics_obj_meta.lcStatus,
                        "ocStatus": analytics_obj_meta.ocStatus,
                        "roiStatus": analytics_obj_meta.roiStatus,
                        # Retain legacy snake_case keys for backward compatibility
                        "direction_status": analytics_obj_meta.dirStatus,
                        "line_crossing_status": analytics_obj_meta.lcStatus,
                        "overcrowding_status": analytics_obj_meta.ocStatus,
                        "roi_status": analytics_obj_meta.roiStatus,
                    }
                try:
                    user_meta_list = user_meta_list.next
                except StopIteration:
                    break
            return None
        except Exception as e:
            self.logger.debug(f"Error extracting analytics object meta: {e}")
            return None

    def _extract_depth_meta(self, obj_meta) -> Optional[Dict[str, Any]]:
        """Extract MapAnything depth annotation from user meta."""
        try:
            user_meta_list = obj_meta.obj_user_meta_list
            while user_meta_list:
                user_meta = pyds.NvDsUserMeta.cast(user_meta_list.data)  # type: ignore
                meta_type = None
                try:
                    meta_type = (
                        user_meta.base_meta.meta_type
                        if user_meta and user_meta.base_meta
                        else None
                    )
                except Exception:
                    meta_type = None
                try:
                    self.logger.debug(
                        "Inspecting user meta: type=%s(%s) obj_id=%s",
                        meta_type,
                        int(meta_type) if meta_type is not None else None,
                        getattr(obj_meta, "object_id", None),
                    )
                except Exception:
                    pass
                mtype_val = None
                try:
                    mtype_val = int(meta_type)
                except Exception:
                    mtype_val = None
                # Only handle the MapAnything depth meta type we own; skip others.
                if mtype_val is not None and mtype_val == int(CUSTOM_MDE_META_TYPE):
                    data_ptr = ctypes.cast(user_meta.user_meta_data, ctypes.c_void_p).value
                    if data_ptr:
                        try:
                            raw = _safe_cstring_from_ptr(int(data_ptr))
                            if not raw:
                                raise ValueError("empty payload")
                            payload = json.loads(raw)
                            if isinstance(payload, dict) and "mde" in payload:
                                entry = payload["mde"]
                                if isinstance(entry, dict):
                                    try:
                                        payload = json.loads(raw)
                                        if isinstance(payload, dict) and "mde" in payload:
                                            entry = payload["mde"]
                                            if isinstance(entry, dict):
                                                try:
                                                    self.logger.debug(
                                                        "Extracted depth meta for obj_id=%s: %s",
                                                        getattr(obj_meta, "object_id", None),
                                                        entry,
                                                    )
                                                except Exception:
                                                    pass
                                                return entry
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                try:
                    user_meta_list = user_meta_list.next
                except StopIteration:
                    break
        except Exception as exc:
            self.logger.debug(f"Depth meta extraction failed: {exc}")
        finally:
            # Ensure proper cleanup of any allocated resources
            try:
                if 'user_meta_list' in locals() and user_meta_list:
                    # Clean up the user meta list
                    pass
            except:
                pass
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
            prev_occupancy = dict(
                self.live_tracking_state.get(sensor_id, {}).get("occupancy", {})
            )
        except Exception:
            prev_occupancy = {}

        # Lazy-decode JPEG for ReID crops only when needed
        decoded_frame_bgr = None

        l_obj = frame_meta.obj_meta_list
        now_ts = time.time()
        present_ds_ids: List[int] = []
        while l_obj:
            obj = pyds.NvDsObjectMeta.cast(l_obj.data)  # type: ignore
            if self._is_object_excluded(obj):
                l_obj = l_obj.next
                continue
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
                "track_id": obj.object_id,
                "camera_id": f"camera_{sensor_id}",
                "confidence": obj.confidence,
                "bbox": [rect.left, rect.top, rect.width, rect.height],
                "class_id": obj.class_id,
            }

            # Compute center point
            center_x = rect.left + rect.width / 2
            center_y = rect.top + rect.height / 2
            track_dict["center"] = [center_x, center_y]

            # Estimate velocity (px/s) using per-track motion state with short moving average
            try:
                motion_state = self.track_motion_state_by_sensor.setdefault(
                    sensor_id, {}
                ).setdefault(
                    obj.object_id,
                    {"last_center": None, "last_ts": None, "vel_hist": deque(maxlen=5)},
                )
                last_center = motion_state.get("last_center")
                last_ts = motion_state.get("last_ts")
                if last_center is not None and last_ts is not None:
                    dt = max(1e-3, now_ts - float(last_ts))
                    vx = (center_x - float(last_center[0])) / dt
                    vy = (center_y - float(last_center[1])) / dt
                    motion_state["vel_hist"].append((vx, vy))
                    # Compute average velocity over history
                    if motion_state["vel_hist"]:
                        hvx = sum(v[0] for v in motion_state["vel_hist"]) / len(
                            motion_state["vel_hist"]
                        )
                        hvy = sum(v[1] for v in motion_state["vel_hist"]) / len(
                            motion_state["vel_hist"]
                        )
                        track_dict["velocity"] = [hvx, hvy]
                # Update state
                motion_state["last_center"] = [center_x, center_y]
                motion_state["last_ts"] = now_ts
            except Exception:
                # Never allow velocity estimation errors to break telemetry
                pass

            # Add tracker confidence if available
            if hasattr(obj, "tracker_confidence"):
                track_dict["tracker_confidence"] = obj.tracker_confidence

            active_tracks.append(track_dict)

            depth_meta = self._extract_depth_meta(obj)

            # Phase 3.3: Add analytics metadata if available
            analytics_data = self._extract_analytics_obj_meta(obj)
            if analytics_data:
                detection["analytics"] = analytics_data

                # Extract occupancy and transition data from analytics
                if "roiStatus" in analytics_data:
                    roi_status = analytics_data["roiStatus"]

                    def bump(zone):
                        zone_str = str(zone).strip()
                        if zone_str:
                            occupancy[zone_str] = occupancy.get(zone_str, 0) + 1

                    in_zones: List[str] = []
                    if isinstance(roi_status, dict):
                        for zone, status in roi_status.items():
                            if status in (1, True, "IN", "inside"):
                                bump(zone)
                                in_zones.append(str(zone).strip())
                    elif isinstance(roi_status, (list, tuple, set)):
                        for zone in roi_status:
                            bump(zone)
                            in_zones.append(str(zone).strip())
                    elif isinstance(roi_status, str):
                        for zone in roi_status.split(","):
                            bump(zone)
                            in_zones.append(str(zone).strip())

                    current_zone = in_zones[0] if in_zones else None
                    if current_zone:
                        track_dict["zone"] = current_zone
                        try:
                            zone_state = self.track_zone_state_by_sensor.setdefault(
                                sensor_id, {}
                            ).setdefault(
                                obj.object_id,
                                {
                                    "current_zone": None,
                                    "entry_time": None,
                                },
                            )
                            prev_zone = zone_state.get("current_zone")
                            entry_time = zone_state.get("entry_time")
                            if prev_zone == current_zone:
                                if entry_time is None:
                                    zone_state["entry_time"] = now_ts
                                    entry_time = now_ts
                                dwell = max(0.0, now_ts - float(entry_time))
                                track_dict["dwell_time"] = dwell
                            else:
                                if prev_zone and prev_zone != current_zone:
                                    transitions.append(
                                        {
                                            "track_id": obj.object_id,
                                            "camera_id": f"camera_{sensor_id}",
                                            "from_zone": prev_zone,
                                            "to_zone": current_zone,
                                            "timestamp": now_ts,
                                        }
                                    )
                                zone_state["current_zone"] = current_zone
                                zone_state["entry_time"] = now_ts
                                track_dict["dwell_time"] = 0.0
                        except Exception:
                            pass

                if not occupancy:
                    frame_analytics = self._extract_analytics_frame_meta(frame_meta)
                    if frame_analytics and isinstance(
                        frame_analytics.get("objects_in_roi"), dict
                    ):
                        occupancy.update(frame_analytics["objects_in_roi"])

                if "lcStatus" in analytics_data:
                    lc_status = analytics_data["lcStatus"]
                    if isinstance(lc_status, dict):
                        for line_name, status in lc_status.items():
                            if status == 1:
                                transitions.append(
                                    {
                                        "track_id": obj.object_id,
                                        "camera_id": f"camera_{sensor_id}",
                                        "line_name": line_name,
                                        "timestamp": time.time(),
                                    }
                                )

            if depth_meta:
                detection["depth"] = depth_meta
                track_dict["depth"] = depth_meta

            # Fallback zone/dwell: if no analytics zone detected, use camera room as zone
            try:
                if "zone" not in track_dict or not track_dict["zone"]:
                    # Map sensor_id to clean camera name
                    cam_info = self.source_info.get(sensor_id, {})
                    fallback_zone = cam_info.get("clean_name") or cam_info.get("name")
                    if fallback_zone:
                        track_dict["zone"] = fallback_zone
                        # Maintain simple dwell timer per (sensor_id, track_id) on this fallback zone
                        zone_state = self.track_zone_state_by_sensor.setdefault(
                            sensor_id, {}
                        ).setdefault(
                            obj.object_id, {"current_zone": None, "entry_time": None}
                        )
                        prev_zone = zone_state.get("current_zone")
                        entry_time = zone_state.get("entry_time")
                        if prev_zone == fallback_zone:
                            if entry_time is None:
                                zone_state["entry_time"] = now_ts
                                entry_time = now_ts
                            dwell = max(0.0, now_ts - float(entry_time))
                            track_dict["dwell_time"] = dwell
                        else:
                            # Zone changed or first time
                            zone_state["current_zone"] = fallback_zone
                            zone_state["entry_time"] = now_ts
                            track_dict["dwell_time"] = 0.0
            except Exception:
                pass

            # Add secondary inference results if available
            secondary_data = self._extract_secondary_inference_meta(obj)
            if secondary_data:
                detection["secondary_inference"] = secondary_data

            # StableID: update or create global identity (persons only)
            try:
                if (
                    self.reid_enabled
                    and (self.stable_id_mgr is not None)
                    and int(obj.class_id) == 0
                ):  # person
                    bbox_tuple = (
                        float(rect.left),
                        float(rect.top),
                        float(rect.width),
                        float(rect.height),
                    )
                    zone_name = (
                        track_dict.get("zone") if isinstance(track_dict, dict) else None
                    )
                    # Decode at most every N ms per sensor and only on-demand
                    if decoded_frame_bgr is None:
                        last_dec = float(
                            self._last_decode_ts_by_sensor.get(int(sensor_id), 0.0)
                        )
                        if (float(now_ts) - last_dec) >= float(
                            self._reid_decode_min_interval_s
                        ):
                            decoded_frame_bgr = self._decode_latest_jpeg_for_sensor(
                                sensor_id
                            )
                            self._last_decode_ts_by_sensor[int(sensor_id)] = float(
                                now_ts
                            )
                    stable_id = self.stable_id_mgr.update(
                        sensor_id=int(sensor_id),
                        ds_obj_id=int(obj.object_id),
                        bbox_ltrbwh=bbox_tuple,
                        ts=float(now_ts),
                        zone=str(zone_name) if zone_name else None,
                        frame_bgr=decoded_frame_bgr,
                    )
                    track_dict["stable_id"] = int(stable_id)
                else:
                    track_dict["stable_id"] = None
            except Exception:
                try:
                    track_dict["stable_id"] = None
                except Exception:
                    pass

            detections.append(detection)

            # Track presence for removal bookkeeping
            try:
                present_ds_ids.append(int(obj.object_id))
            except Exception:
                pass

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # Publish occupancy deltas to integrations (MQTT/Influx)
        try:
            publisher = getattr(self, "occupancy_publisher", None)
            if publisher is not None and occupancy is not None:
                # Publish current counts (rooms seen this frame)
                first_flag = getattr(self, "_occ_pub_first", True)
                for zone, cnt in occupancy.items():
                    try:
                        room_id = str(zone).strip()
                        publisher.publish_state(
                            room_id=room_id,
                            occupied=(int(cnt) > 0),
                            count=int(cnt),
                            ts_ns=int(time.time_ns()),
                        )
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
                        publisher.publish_state(
                            room_id=room_id,
                            occupied=False,
                            count=0,
                            ts_ns=int(time.time_ns()),
                        )
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

        # End-of-frame: remove tracks not present and prune ghosts
        if self.reid_enabled and (self.stable_id_mgr is not None):
            try:
                self.stable_id_mgr.remove_missing_tracks(
                    int(sensor_id), present_ds_ids, float(now_ts)
                )
            except Exception:
                pass

        # Update live tracking state for this specific stream
        if sensor_id in self.live_tracking_state:
            self.live_tracking_state[sensor_id]["active_tracks"] = active_tracks
            self.live_tracking_state[sensor_id]["occupancy"] = occupancy
            self.live_tracking_state[sensor_id]["transitions"].extend(transitions)

            # Keep only recent transitions (last 100) per stream
            if len(self.live_tracking_state[sensor_id]["transitions"]) > 100:
                self.live_tracking_state[sensor_id]["transitions"] = (
                    self.live_tracking_state[sensor_id]["transitions"][-100:]
                )
        else:
            self.logger.warning(
                f"⚠️ Unknown source_id {ds_index} (mapped→{sensor_id}) in frame metadata"
            )

        # Debug logging for tracking state
        # if len(active_tracks) > 0:
        # self.logger.debug(f"📊 Final tracking state - Active tracks: {len(active_tracks)}, Occupancy: {occupancy}, Transitions: {len(transitions)}")
        # if active_tracks:
        # self.logger.debug(f"📊 Active tracks sample: {active_tracks[:2]}")  # Show first 2 tracks
        # if occupancy:
        # self.logger.debug(f"📊 Occupancy: {occupancy}")
        # if transitions:
        # self.logger.debug(f"📊 Recent transitions: {transitions[-3:]}")  # Show last 3 transitions

        return detections

    def _decode_latest_jpeg_for_sensor(self, sensor_id: int) -> Optional[np.ndarray]:
        """Decode the latest JPEG bytes for a sensor to BGR np.ndarray.

        Uses a side buffer populated by the appsink callback to avoid contention
        with the broadcast loop. Returns None if decode fails or no bytes yet.
        """
        try:
            jpeg_bytes = self._latest_jpeg_bytes_by_sensor.get(int(sensor_id))
            if not jpeg_bytes:
                return None
            import numpy as _np

            npbuf = _np.frombuffer(jpeg_bytes, dtype=_np.uint8)
            import cv2 as _cv2

            frame = _cv2.imdecode(npbuf, _cv2.IMREAD_COLOR)
            return frame
        except Exception:
            return None

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
                classifier_meta = pyds.NvDsClassifierMeta.cast(
                    classifier_meta_list.data
                )  # type: ignore
                if classifier_meta.unique_component_id == 2:  # Our SGIE ID
                    label_info_list = classifier_meta.label_info_list
                    if label_info_list:
                        label_info = pyds.NvDsLabelInfo.cast(label_info_list.data)  # type: ignore
                        return {
                            "classification": label_info.result_label,
                            "confidence": label_info.result_prob,
                            "component_id": classifier_meta.unique_component_id,
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
            self.logger.error(
                f"🚨 Error source: {message.src.get_name() if message.src else 'unknown'}"
            )
            try:
                src_name = message.src.get_name() if message.src else ""
                # Provide actionable hint if source/decoder failed
                if any(
                    k in src_name
                    for k in (
                        "nvmultiurisrcbin",
                        "uridecodebin",
                        "rtspsrc",
                        "decodebin",
                    )
                ):
                    # Summarize configured sources to aid debugging
                    try:
                        sources = []
                        for sid, info in getattr(self, "source_info", {}).items():
                            sources.append(
                                f"[{sid}] {info.get('name', 'unknown')} -> {info.get('url', '')}\n"
                            )
                        if sources:
                            self.logger.error(
                                "🔎 One or more sources failed to start or connect.\n"
                                "    • Verify each stream URL is reachable and producing frames.\n"
                                "    • If using ffmpeg or a local generator, ensure it is running.\n"
                                "Configured sources:\n" + "".join(sources).rstrip()
                            )
                    except Exception:
                        pass
            except Exception:
                pass
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
                self.logger.info(
                    f"🔄 Pipeline state changed: {old_state.value_nick} → {new_state.value_nick}"
                )

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

            # Start GLib mainloop early so bus callbacks can surface errors while we wait
            self.running = True
            self.start_time = time.time()
            self.mainloop_thread = threading.Thread(
                target=self._run_mainloop, daemon=True
            )
            self.mainloop_thread.start()

            # If async, wait with a finite timeout and surface element states on failure
            if ret == Gst.StateChangeReturn.ASYNC:
                self.logger.info(
                    "⏳ Pipeline state change is async, waiting up to 10s..."
                )
                # 10s timeout
                timeout_ns = 10 * Gst.SECOND
                ret_state = self.pipeline.get_state(timeout_ns)
                if ret_state[0] != Gst.StateChangeReturn.SUCCESS:
                    # Collect per-element states for diagnosis
                    states = []
                    try:
                        it = self.pipeline.iterate_elements()
                        while True:
                            result, element = it.next()
                            if result != Gst.IteratorResult.OK:
                                break
                            name = element.get_name()
                            e_ret, e_state, e_pending = element.get_state(0)
                            states.append((name, e_ret, e_state, e_pending))
                    except Exception:
                        pass

                    # Heuristic: identify likely culprit class
                    def _nick(x):
                        try:
                            return x.value_nick  # type: ignore[attr-defined]
                        except Exception:
                            return str(x)

                    src_names = (
                        "nvmultiurisrcbin",
                        "uridecodebin",
                        "rtspsrc",
                        "urisrc",
                        "decodebin",
                    )
                    stuck_src = [
                        s
                        for s in states
                        if any(n in s[0] for n in src_names)
                        and _nick(s[2]) != "playing"
                    ]
                    stuck_inf = [
                        s
                        for s in states
                        if s[0] == "nvinfer" and _nick(s[2]) != "playing"
                    ]
                    stuck_trk = [
                        s
                        for s in states
                        if s[0] == "nvtracker" and _nick(s[2]) != "playing"
                    ]

                    warning_prefix = "⚠️ Pipeline state change timed out"
                    if stuck_src:
                        name, e_ret, e_state, e_pending = stuck_src[0]
                        self.logger.warning(
                            f"{warning_prefix}; source {name} state={_nick(e_state)} pending={_nick(e_pending)}"
                        )
                        self.logger.warning(
                            "Hint: verify stream URLs or generators are active."
                        )
                    elif stuck_inf:
                        name, e_ret, e_state, e_pending = stuck_inf[0]
                        self.logger.warning(
                            f"{warning_prefix}; inference element {name} state={_nick(e_state)} pending={_nick(e_pending)}"
                        )
                    elif stuck_trk:
                        name, e_ret, e_state, e_pending = stuck_trk[0]
                        self.logger.warning(
                            f"{warning_prefix}; tracker element {name} state={_nick(e_state)} pending={_nick(e_pending)}"
                        )
                    else:
                        self.logger.warning(
                            f"{warning_prefix}: {ret_state[0]}"
                        )

                    # Debug-only: full element states
                    if states:
                        self.logger.debug("Element states:")
                        for name, e_ret, e_state, e_pending in states:
                            self.logger.debug(
                                f"  • {name}: state={_nick(e_state)} pending={_nick(e_pending)} ret={e_ret}"
                            )
                    self.logger.warning(
                        "Proceeding despite async pipeline state; monitoring bus messages for further errors."
                    )

            self.logger.info("✅ DeepStream pipeline start requested")

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
            try:
                self.pipeline.set_state(Gst.State.NULL)
                # Wait for state change to complete
                state_ret = self.pipeline.get_state(timeout=5 * Gst.SECOND)
                if state_ret[0] == Gst.StateChangeReturn.ASYNC:
                    state_ret = self.pipeline.get_state(timeout=10 * Gst.SECOND)
            except Exception as e:
                self.logger.warning(f"Error stopping pipeline: {e}")

        # Stop main loop
        if self.mainloop:
            self.mainloop.quit()

        # Wait for main loop thread
        if hasattr(self, "mainloop_thread") and self.mainloop_thread.is_alive():
            self.mainloop_thread.join(timeout=2.0)
        
        # Clear queues to prevent memory leaks
        for queue in self.jpeg_queues.values():
            try:
                while not queue.empty():
                    queue.get_nowait()
            except:
                pass
        self.jpeg_queues.clear()
        
        # Clear trail history to prevent memory leaks
        self.trail_history.clear()
        self.trail_last_seen_by_sensor.clear()
        self.trail_history_by_sensor.clear()
        
        # Clear any other caches to prevent memory leaks
        if hasattr(self, '_bbox_smooth_by_sensor'):
            self._bbox_smooth_by_sensor.clear()
        if hasattr(self, 'track_motion_state_by_sensor'):
            self.track_motion_state_by_sensor.clear()
        if hasattr(self, 'track_zone_state_by_sensor'):
            self.track_zone_state_by_sensor.clear()

        self.logger.info("DeepStream pipeline stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        runtime = (
            time.time() - self.start_time if self.running and self.start_time > 0 else 0
        )

        # Use a lock to safely access frame_count
        with self.frame_count_lock:
            frame_count_copy = self.frame_count

        fps = frame_count_copy / runtime if runtime > 0 else 0

        queue_sizes = {
            f"source_{src_id}": q.qsize() for src_id, q in self.jpeg_queues.items()
        }
        stats = {
            "pipeline_type": "deepstream",
            "running": self.running,
            "frames_processed": frame_count_copy,
            "fps": fps,
            "runtime_seconds": runtime,
            "batch_size": self.batch_size,
            "sources": len(self.sensor_ids),
            "queue_sizes": queue_sizes,
            "tracking": self.live_tracking_state,
        }
        # Attach ReID/SID allocator telemetry if available
        try:
            if hasattr(self, "stable_id_mgr") and self.stable_id_mgr:
                sid_metrics = self.stable_id_mgr.get_sid_metrics()
                if sid_metrics:
                    stats["reid"] = sid_metrics
        except Exception:
            pass
        return stats

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

        expected_source_ids = set(
            range(len(self.sensor_ids))
        )  # DeepStream uses 0..N-1 source indices
        actual_source_ids = set(source_ids)
        unexpected_ids = actual_source_ids - expected_source_ids
        missing_ids = expected_source_ids - actual_source_ids

        # Warn only if we see IDs outside expected range; this indicates a real issue
        if unexpected_ids:
            self.rate_limited_logger.warning(
                f"⚠️ Demux sink saw unexpected source_ids {sorted(list(unexpected_ids))}; expected range 0..{len(self.sensor_ids) - 1}"
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
                            mapping_str = ", ".join(
                                [
                                    f"{k}→{v}"
                                    for k, v in sorted(
                                        self.demux_pad_to_source_id.items()
                                    )
                                ]
                            )
                            self.logger.info(f"nvstreamdemux mapping: {mapping_str}")
                    # Fill sensor→pad reuse map
                    try:
                        for pad_name, sid in self.demux_pad_to_source_id.items():
                            try:
                                demux_index = int(pad_name.split("_")[1])
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
                handler_id = pad.add_probe(
                    Gst.PadProbeType.BUFFER, _one_shot_probe, None
                )
                pad_handler_ids[pad_name] = handler_id

            # Wait briefly for first frames to flow and mapping to populate
            # Note: In Python API, we avoid busy waiting; mapping will complete as frames arrive.
            self.logger.info(
                "Calibrating demux pad → source_id mapping (will log once ready)..."
            )
        except Exception as e:
            self.logger.warning(f"Failed to calibrate demux pad mapping: {e}")

    def _setup_stream_branches(self) -> None:
        for sensor_id in self.sensor_ids:
            ok = self._add_jpeg_branch_for_sensor(sensor_id)
            if not ok:
                self.logger.error(
                    f"❌ Failed to setup stream branch for sensor {sensor_id}"
                )

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
                    self.logger.debug(
                        f"[caps] {stage}: {caps.to_string() if caps else 'None'}"
                    )
                    self._caps_probe_counts[stage] = count + 1

                # Extract negotiated width/height and cache per sensor if stage string encodes it
                try:
                    sensor_id = None
                    # Expected stage patterns include: "sensor {id} conv src", "sensor {id} caps src", "sensor {id} nvjpegenc sink"
                    if isinstance(stage, str) and "sensor" in stage:
                        parts = stage.split()
                        for i, p in enumerate(parts):
                            if p == "sensor" and i + 1 < len(parts):
                                try:
                                    sensor_id = int(parts[i + 1])
                                except Exception:
                                    sensor_id = None
                                break
                    if sensor_id is not None and caps and caps.get_size() > 0:
                        s = caps.get_structure(0)
                        w = None
                        h = None
                        try:
                            ok, wi = s.get_int("width")  # type: ignore
                            if ok:
                                w = int(wi)
                            ok, hi = s.get_int("height")  # type: ignore
                            if ok:
                                h = int(hi)
                        except Exception:
                            # Fallback parse from string
                            cs = caps.to_string()
                            import re as _re

                            mw = _re.search(r"width=(\d+)", cs)
                            mh = _re.search(r"height=(\d+)", cs)
                            if mw:
                                w = int(mw.group(1))
                            if mh:
                                h = int(mh.group(1))
                        if w and h:
                            self._branch_caps_by_sensor[int(sensor_id)] = (
                                int(w),
                                int(h),
                            )
                except Exception:
                    pass
        except Exception as e:
            # Do not disrupt pipeline on probe errors
            self.logger.debug(f"caps probe error at {stage}: {e}")
        return Gst.PadProbeReturn.OK

    def _attach_caps_debug_probes(
        self, conv: Gst.Element, caps: Gst.Element, jpegenc: Gst.Element, sensor_id: int
    ) -> None:
        try:
            conv_src = conv.get_static_pad("src")
            if conv_src:
                conv_src.add_probe(
                    Gst.PadProbeType.EVENT_DOWNSTREAM,
                    self._caps_event_probe,
                    f"sensor {sensor_id} conv src",
                )
            caps_src = caps.get_static_pad("src")
            if caps_src:
                caps_src.add_probe(
                    Gst.PadProbeType.EVENT_DOWNSTREAM,
                    self._caps_event_probe,
                    f"sensor {sensor_id} caps src",
                )
            enc_sink = jpegenc.get_static_pad("sink")
            if enc_sink:
                # Caps events flow downstream into encoder sink; observe what arrives
                enc_sink.add_probe(
                    Gst.PadProbeType.EVENT_DOWNSTREAM,
                    self._caps_event_probe,
                    f"sensor {sensor_id} nvjpegenc sink",
                )
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
            self.logger.debug(
                f"Failed to attach caps probes for sensor {sensor_id}: {e}"
            )

    def _add_jpeg_branch_for_sensor(self, sensor_id: int) -> bool:
        try:
            if sensor_id in self._stream_branch_elements:
                self.logger.debug(f"JPEG branch already exists for sensor {sensor_id}")
                return True

            branch_name_prefix = f"stream_branch_{sensor_id}"
            queue = Gst.ElementFactory.make("queue", f"{branch_name_prefix}_queue")
            conv_pre = Gst.ElementFactory.make(
                "nvvideoconvert", f"{branch_name_prefix}_conv_pre"
            )
            caps_pre = Gst.ElementFactory.make(
                "capsfilter", f"{branch_name_prefix}_caps_pre"
            )
            osd = Gst.ElementFactory.make("nvdsosd", f"{branch_name_prefix}_osd")
            conv_post = Gst.ElementFactory.make(
                "nvvideoconvert", f"{branch_name_prefix}_conv_post"
            )
            caps_post = Gst.ElementFactory.make(
                "capsfilter", f"{branch_name_prefix}_caps_post"
            )
            jpegenc = Gst.ElementFactory.make("nvjpegenc", f"{branch_name_prefix}_enc")
            sink = Gst.ElementFactory.make("appsink", f"{branch_name_prefix}_sink")

            # Optimize nvjpegenc for realtime and respect configured JPEG quality
            try:
                quality = int(getattr(self.config.visualization, "JPEG_QUALITY", 85))
                jpegenc.set_property("quality", quality)
                jpegenc.set_property("preset-level", 1)  # fast
            except Exception:
                pass

            if not all(
                [queue, conv_pre, caps_pre, osd, conv_post, caps_post, jpegenc, sink]
            ):
                self.logger.error(
                    f"❌ Failed to create elements for JPEG branch sensor {sensor_id}"
                )
                return False

            self._configure_live_queue(queue)
            # Pre-OSD RGBA and post-OSD I420
            caps_pre.set_property(
                "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
            )
            caps_post.set_property(
                "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420")
            )
            sink.set_property("emit-signals", True)
            sink.set_property("sync", False)
            sink.set_property("max-buffers", 5)
            sink.set_property("drop", True)
            sink.connect("new-sample", self._on_new_jpeg_sample, sensor_id)

            for el in (
                queue,
                conv_pre,
                caps_pre,
                osd,
                conv_post,
                caps_post,
                jpegenc,
                sink,
            ):
                self.pipeline.add(el)

            # Attach detailed caps negotiation probes (non-spammy)
            self._attach_caps_debug_probes(conv_pre, caps_pre, jpegenc, sensor_id)
            self._attach_caps_debug_probes(conv_post, caps_post, jpegenc, sensor_id)

            # Link elements explicitly and validate
            if not queue.link(conv_pre):
                self.logger.error(
                    f"❌ Failed to link queue→conv for sensor {sensor_id}"
                )
                for el in (
                    queue,
                    conv_pre,
                    caps_pre,
                    osd,
                    conv_post,
                    caps_post,
                    jpegenc,
                    sink,
                ):
                    try:
                        self.pipeline.remove(el)
                    except Exception:
                        pass
                return False
            if not conv_pre.link(caps_pre):
                self.logger.error(f"❌ Failed to link conv→caps for sensor {sensor_id}")
                for el in (
                    queue,
                    conv_pre,
                    caps_pre,
                    osd,
                    conv_post,
                    caps_post,
                    jpegenc,
                    sink,
                ):
                    try:
                        self.pipeline.remove(el)
                    except Exception:
                        pass
                return False
            if not caps_pre.link(osd):
                self.logger.error(f"❌ Failed to link caps→osd for sensor {sensor_id}")
                for el in (
                    queue,
                    conv_pre,
                    caps_pre,
                    osd,
                    conv_post,
                    caps_post,
                    jpegenc,
                    sink,
                ):
                    try:
                        self.pipeline.remove(el)
                    except Exception:
                        pass
                return False
            # Configure and attach per-branch OSD probe
            try:
                osd.set_property("process-mode", 0)
                osd.set_property("display-text", 1)
            except Exception:
                pass
            osd_sink_pad = osd.get_static_pad("sink")
            if osd_sink_pad:
                osd_sink_pad.add_probe(
                    Gst.PadProbeType.BUFFER, self._per_branch_osd_probe, sensor_id
                )
            if not osd.link(conv_post):
                self.logger.error(
                    f"❌ Failed to link osd→conv_post for sensor {sensor_id}"
                )
                for el in (
                    queue,
                    conv_pre,
                    caps_pre,
                    osd,
                    conv_post,
                    caps_post,
                    jpegenc,
                    sink,
                ):
                    try:
                        self.pipeline.remove(el)
                    except Exception:
                        pass
                return False
            if not conv_post.link(caps_post):
                self.logger.error(
                    f"❌ Failed to link conv_post→caps_post for sensor {sensor_id}"
                )
                for el in (
                    queue,
                    conv_pre,
                    caps_pre,
                    osd,
                    conv_post,
                    caps_post,
                    jpegenc,
                    sink,
                ):
                    try:
                        self.pipeline.remove(el)
                    except Exception:
                        pass
                return False
            if not caps_post.link(jpegenc):
                self.logger.error(
                    f"❌ Failed to link caps_post→jpegenc for sensor {sensor_id}"
                )
                for el in (
                    queue,
                    conv_pre,
                    caps_pre,
                    osd,
                    conv_post,
                    caps_post,
                    jpegenc,
                    sink,
                ):
                    try:
                        self.pipeline.remove(el)
                    except Exception:
                        pass
                return False
            if not jpegenc.link(sink):
                self.logger.error(
                    f"❌ Failed to link jpegenc→sink for sensor {sensor_id}"
                )
                for el in (
                    queue,
                    conv_pre,
                    caps_pre,
                    osd,
                    conv_post,
                    caps_post,
                    jpegenc,
                    sink,
                ):
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
                    for el in (
                        queue,
                        conv_pre,
                        caps_pre,
                        osd,
                        conv_post,
                        caps_post,
                        jpegenc,
                        sink,
                    ):
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
                    demux_index = int(pad_name.split("_")[1])
                except Exception:
                    demux_index = None
                req_pad = (
                    self._get_or_request_demux_pad(demux_index)
                    if demux_index is not None
                    else None
                )
            if not req_pad:
                self.logger.error(
                    f"❌ Failed to request demux pad {pad_name} for sensor {sensor_id}"
                )
                for el in (
                    queue,
                    conv_pre,
                    caps_pre,
                    osd,
                    conv_post,
                    caps_post,
                    jpegenc,
                    sink,
                ):
                    try:
                        self.pipeline.remove(el)
                    except Exception:
                        pass
                return False

            sink_pad = queue.get_static_pad("sink")
            # Ensure the queue sink pad is active before linking
            if sink_pad is None:
                self.logger.error(
                    f"❌ Missing sink pad on queue for sensor {sensor_id}"
                )
                # Do not release req_pad here since it is managed in the reuse maps
                for el in (
                    queue,
                    conv_pre,
                    caps_pre,
                    osd,
                    conv_post,
                    caps_post,
                    jpegenc,
                    sink,
                ):
                    try:
                        self.pipeline.remove(el)
                    except Exception:
                        pass
                return False
            link_ret = req_pad.link(sink_pad)
            if link_ret != Gst.PadLinkReturn.OK:
                self.logger.error(
                    f"❌ Failed to link demux {pad_name} to JPEG queue for sensor {sensor_id} (ret={link_ret})"
                )
                # Do not release req_pad here since it is managed in the reuse maps
                for el in (
                    queue,
                    conv_pre,
                    caps_pre,
                    osd,
                    conv_post,
                    caps_post,
                    jpegenc,
                    sink,
                ):
                    try:
                        self.pipeline.remove(el)
                    except Exception:
                        pass
                return False

            self._stream_branch_elements[sensor_id] = [
                queue,
                conv_pre,
                caps_pre,
                osd,
                conv_post,
                caps_post,
                jpegenc,
                sink,
            ]
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

            self.logger.info(
                f"✅ JPEG branch ready for sensor {sensor_id} via request pad {pad_name}"
            )
            return True
        except Exception as e:
            self.logger.error(
                f"❌ Exception while adding JPEG branch for sensor {sensor_id}: {e}"
            )
            return False

    def _remove_jpeg_branch_for_sensor(self, sensor_id: int) -> None:
        try:
            # Unlink and release request pad
            req_pad = self._demux_requested_pads.pop(sensor_id, None)
            if req_pad is not None:
                try:
                    self.demux.release_request_pad(req_pad)
                except Exception as e:
                    self.logger.debug(
                        f"Error releasing request pad for sensor {sensor_id}: {e}"
                    )

            # Remove elements
            elements = self._stream_branch_elements.pop(sensor_id, None)
            if elements:
                for el in elements:
                    try:
                        el.set_state(Gst.State.NULL)
                        self.pipeline.remove(el)
                    except Exception as e:
                        self.logger.debug(
                            f"Error removing element {el.get_name()} for sensor {sensor_id}: {e}"
                        )
            self.logger.info(f"✅ Removed JPEG branch for sensor {sensor_id}")
        except Exception as e:
            self.logger.debug(
                f"Error tearing down JPEG branch for sensor {sensor_id}: {e}"
            )

    def _setup_jpeg_branches_with_request_pads(self) -> None:
        # Backward-compat alias; now builds full per-stream branches with OSD
        self._setup_stream_branches()

    def _demux_pad_removed_cb(self, demux, pad):
        """Deprecated in this implementation: we use request pads per sensor."""
        pad_name = pad.get_name()
        self.logger.debug(f"Demuxer removed pad notification: {pad_name}")

    def _on_new_jpeg_sample(
        self, appsink: GstApp.AppSink, sensor_id: int
    ) -> Gst.FlowReturn:
        """Callback for GPU JPEG appsink – push encoded JPEG bytes to the correct queue"""
        sample: Optional[Gst.Sample] = None  # type: ignore[attr-defined]
        buffer: Optional[Gst.Buffer] = None
        mapinfo = None
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

            jpeg_data = mapinfo.data
            pts_us: Optional[int] = None
            pts_ns = getattr(buffer, "pts", None)
            if pts_ns not in (None, Gst.CLOCK_TIME_NONE):
                pts_us = int(int(pts_ns) // 1_000)
            if jpeg_data:
                payload_bytes = bytes(jpeg_data)
                if true_id in self.jpeg_queues:
                    try:
                        self.jpeg_queues[true_id].put_nowait(
                            (payload_bytes, pts_us)
                        )
                        # Count only frames successfully enqueued (exclude dropped frames)
                        with self.frame_count_lock:
                            self.frame_count += 1
                    except queue.Full:
                        # Drop frame if queue is full; do not increment frame_count
                        pass
                else:
                    self.rate_limited_logger.warning(
                        f"No JPEG queue for source_id {true_id}"
                    )
                # Also store latest bytes for crop decoding
                try:
                    self._latest_jpeg_bytes_by_sensor[int(true_id)] = payload_bytes
                except Exception:
                    pass

            return Gst.FlowReturn.OK
        except Exception as e:
            self.logger.error(
                f"Error in _on_new_jpeg_sample for sensor {sensor_id}: {e}"
            )
            return Gst.FlowReturn.ERROR
        finally:
            if buffer is not None and mapinfo is not None:
                try:
                    buffer.unmap(mapinfo)
                except Exception:
                    pass
            sample = None
            buffer = None

    def read_encoded_jpeg(
        self, source_id: int, timeout: float = 0.1
    ) -> Tuple[bool, Optional[bytes], Optional[int]]:
        """Return next encoded JPEG bytes and presentation timestamp for a sensor."""
        if not self.running or source_id not in self.jpeg_queues:
            return False, None, None
        try:
            item = self.jpeg_queues[source_id].get(timeout=timeout)
        except queue.Empty:
            return False, None, None

        if isinstance(item, tuple):
            jpeg_bytes, pts_us = item
        else:
            jpeg_bytes, pts_us = item, None
        return True, jpeg_bytes, pts_us

    def update_confidence_threshold(self, confidence_threshold: float) -> bool:
        """Update confidence threshold in real-time using GObject properties

        Args:
            confidence_threshold: New confidence threshold (0.0-1.0)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.pipeline or not self.nvinfer:
                self.logger.warning(
                    "Pipeline or nvinfer not available for confidence threshold update"
                )
                return False

            # Update nvinfer confidence threshold property
            self.nvinfer.set_property("confidence-threshold", confidence_threshold)
            self.logger.info(
                f"✅ Updated confidence threshold to: {confidence_threshold}"
            )
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
                self.logger.warning(
                    "Pipeline or nvinfer not available for IOU threshold update"
                )
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
                self.logger.warning(
                    "Pipeline or nvinfer not available for detection enable/disable"
                )
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
                self.logger.warning(
                    "Pipeline or nvinfer not available for target classes update"
                )
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
                    "name": f"Camera_{sensor_id}",
                    "clean_name": f"camera-{sensor_id}",
                    "url": uri,
                    "width": self.max_width,
                    "height": self.max_height,
                }
                # Ensure queue and tracking state exist
                if sensor_id not in self.jpeg_queues:
                    self.jpeg_queues[sensor_id] = queue.Queue(maxsize=30)
                if sensor_id not in self.live_tracking_state:
                    self.live_tracking_state[sensor_id] = {
                        "active_tracks": [],
                        "occupancy": {},
                        "transitions": [],
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
                    self.logger.error(
                        f"❌ Failed to create JPEG branch for sensor {sensor_id}"
                    )
                    return False
                return True
            else:
                self.logger.error(
                    f"❌ Failed to add sensor {sensor_id}: {resp.status_code} {resp.text}"
                )
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
                        idx = int(pad_obj.get_name().split("_")[1])
                        self._demux_requested_pads_by_index.pop(idx, None)
                    except Exception:
                        pass
                    try:
                        self.demux.release_request_pad(pad_obj)
                    except Exception:
                        pass
                return True
            else:
                self.logger.error(
                    f"❌ Failed to remove sensor {sensor_id}: {resp.status_code} {resp.text}"
                )
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
                if self._is_object_excluded(obj_meta):
                    l_obj = l_obj.next
                    continue
                tid = obj_meta.object_id
                if tid != -1:
                    cx = obj_meta.rect_params.left + obj_meta.rect_params.width / 2
                    cy = obj_meta.rect_params.top + obj_meta.rect_params.height
                    self.trail_history[tid].append((cx, cy))
                    # Also maintain per-sensor state
                    sid = int(frame_meta.source_id)
                    self.trail_history_by_sensor[sid][tid].append((cx, cy))
                    self.trail_last_seen_by_sensor[sid][tid] = now
                l_obj = l_obj.next
            l_frame = l_frame.next

        # Phase 2 ─ Prune stale tracks (per-sensor maps)
        for sid, last_seen_map in list(self.trail_last_seen_by_sensor.items()):
            for tid in [
                t
                for t, ts in list(last_seen_map.items())
                if now - ts > self.trail_timeout_s
            ]:
                self.trail_last_seen_by_sensor[sid].pop(tid, None)
                self.trail_history_by_sensor[sid].pop(tid, None)

        # ────────────────── Phase 3 ─ Draw all trails into one overlay (batched OSD legacy)
        l_frame = batch_meta.frame_meta_list
        while l_frame:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)

            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            if not display_meta:  # pool exhausted – skip
                l_frame = l_frame.next
                continue
            display_meta.num_lines = 0
            display_meta.num_labels = 0
            line_capacity = len(display_meta.line_params)
            label_capacity = len(display_meta.text_params)

            active_ids = [
                tid for tid, pts in self.trail_history.items() if len(pts) > 1
            ]

            budget_per_track = (line_capacity // len(active_ids)) if active_ids else line_capacity
            budget_per_track = max(1, budget_per_track)

            for tid in active_ids:
                pts = list(self.trail_history[tid])[
                    -(
                        min(
                            self.config.visualization.TRAIL_DRAW_SEGMENTS,
                            budget_per_track,
                        )
                        + 1
                    ) :
                ]

                # draw segments
                for idx in range(len(pts) - 1):  # -1 because we access idx+1
                    if display_meta.num_lines >= line_capacity:
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
                    lp.line_color.set(r, g, b, alpha)  # track-specific color
                    display_meta.num_lines += 1

                # optional label
                if (
                    self.trail_show_labels
                    and display_meta.num_labels < label_capacity
                    and display_meta.num_lines < line_capacity
                    and display_meta.num_labels < len(display_meta.text_params)
                ):
                    tp = display_meta.text_params[display_meta.num_labels]
                    tp.display_text = _alloc_display_text(f"id {tid}")
                    tp.x_offset, tp.y_offset = map(int, pts[-1])
                    tp.font_params.font_name = _alloc_display_text("Serif")
                    tp.font_params.font_size = 12
                    try:
                        r, g, b = self._color_for_track(int(tid))
                    except Exception:
                        r, g, b = (1.0, 1.0, 0.0)
                    tp.font_params.font_color.set(r, g, b, 1.0)
                    tp.set_bg_clr = 0
                    display_meta.num_labels += 1

                if display_meta.num_lines >= line_capacity:
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
            mapped_sensor = self.sensor_id_by_source_idx.get(
                ds_index_local, ds_index_local
            )
            if int(mapped_sensor) != int(sensor_id):
                l_frame = l_frame.next
                continue
            l_obj = frame_meta.obj_meta_list
            while l_obj:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                if self._is_object_excluded(obj_meta):
                    l_obj = l_obj.next
                    continue
                tid = obj_meta.object_id
                # Update per-object bbox label to include track id and detector confidence
                try:
                    if tid != -1:
                        conf_value = float(obj_meta.confidence)
                        # Resolve stable_id for overlay if available
                        try:
                            sid_map_key = (int(sensor_id), int(tid))
                            stable_id = self.stable_id_mgr.active_tracks.get(
                                sid_map_key, {}
                            ).get("stable_id")
                        except Exception:
                            stable_id = None
                        if stable_id is not None:
                            label = (
                                f"sid {int(stable_id)} ds {int(tid)} ({conf_value:.2f})"
                            )
                        else:
                            label = f"ds {int(tid)} ({conf_value:.2f})"
                        obj_meta.text_params.display_text = _alloc_display_text(label)
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
                # Apply bbox size smoothing (EMA with clamp), anchored to bottom-center by default
                try:
                    if self.bbox_smoothing_enabled and tid != -1:
                        rect = obj_meta.rect_params
                        left = float(rect.left)
                        top = float(rect.top)
                        width = max(1.0, float(rect.width))
                        height = max(1.0, float(rect.height))

                        # Prior state for this (sensor, track)
                        sensor_map = self._bbox_smooth_by_sensor.setdefault(
                            int(sensor_id), {}
                        )
                        prev = sensor_map.get(int(tid))
                        # Init entry with history deque
                        if prev is None:
                            from collections import deque as _deque

                            prev = {"hist_h": _deque(maxlen=240)}  # ~8s at 30 FPS
                            sensor_map[int(tid)] = prev

                        # Maintain height history within time window
                        hist = prev.get("hist_h")
                        if hist is not None:
                            # prune old
                            while hist and (
                                now - hist[0][0] > self.bbox_max_drop_window_s
                            ):
                                hist.popleft()
                            # append current observation
                            hist.append((now, height))
                            # compute recent max height
                            try:
                                max_h_recent = (
                                    max(h for (ts, h) in hist) if hist else height
                                )
                            except Exception:
                                max_h_recent = height
                            # Apply over-window drop limit (height cannot drop below ratio * recent max)
                            min_allowed_h = float(max_h_recent) * float(
                                self.bbox_max_drop_ratio
                            )
                            if height < min_allowed_h:
                                height = min_allowed_h

                        # Clamp change vs previous size before EMA
                        if prev is not None:
                            prev_w = max(1.0, float(prev.get("w", width)))
                            prev_h = max(1.0, float(prev.get("h", height)))
                            max_g = self.bbox_smoothing_max_growth
                            min_s = self.bbox_smoothing_max_shrink
                            # Ensure ratios are sensible
                            if max_g < 1.0:
                                max_g = 1.0
                            if min_s <= 0.0 or min_s > 1.0:
                                min_s = 0.85

                            # Clamp raw observation before smoothing
                            width_clamped = max(
                                prev_w * min_s, min(width, prev_w * max_g)
                            )
                            height_clamped = max(
                                prev_h * min_s, min(height, prev_h * max_g)
                            )

                            a = self.bbox_smoothing_alpha
                            new_w = prev_w + a * (width_clamped - prev_w)
                            new_h = prev_h + a * (height_clamped - prev_h)
                        else:
                            new_w, new_h = width, height

                        # Anchor choice: bottom-center (default) or center
                        if self.bbox_smoothing_anchor == "center":
                            cx = left + width * 0.5
                            cy = top + height * 0.5
                            new_left = cx - new_w * 0.5
                            new_top = cy - new_h * 0.5
                        else:
                            # bottom-center
                            cx = left + width * 0.5
                            by = top + height
                            new_left = cx - new_w * 0.5
                            new_top = by - new_h

                        # Clamp to frame bounds using the ACTUAL surface dims at this stage
                        # Using source_frame_width/height can be wrong for muxed streams.
                        try:
                            # Prefer negotiated caps for this branch (actual frame dims at OSD)
                            fw = fh = 0.0
                            try:
                                dims = self._branch_caps_by_sensor.get(int(sensor_id))
                                if dims:
                                    fw, fh = float(dims[0]), float(dims[1])
                            except Exception:
                                pass
                            if not fw or not fh:
                                try:
                                    # Fallback to NvBufSurface queried from the current GstBuffer
                                    surf = pyds.get_nvds_buf_surface(
                                        hash(gst_buffer), frame_meta.batch_id
                                    )  # type: ignore
                                    sl = surf.surfaceList[frame_meta.batch_id]
                                    fw = float(getattr(sl, "width", 0) or 0)
                                    fh = float(getattr(sl, "height", 0) or 0)
                                except Exception:
                                    fw = float(frame_meta.source_frame_width)
                                    fh = float(frame_meta.source_frame_height)
                            if fw > 0 and fh > 0:
                                new_left = max(0.0, min(new_left, fw - new_w))
                                new_top = max(0.0, min(new_top, fh - new_h))
                        except Exception:
                            pass

                        # Apply smoothed box
                        rect.left = float(new_left)
                        rect.top = float(new_top)
                        rect.width = float(max(1.0, new_w))
                        rect.height = float(max(1.0, new_h))

                        # Persist state with timestamp
                        prev["w"] = float(new_w)
                        prev["h"] = float(new_h)
                        prev["ts"] = float(now)
                except Exception:
                    # Never break rendering on smoothing issues
                    pass

                if tid != -1:
                    # Compute bottom-center with trail speed clamp
                    cx_raw = obj_meta.rect_params.left + obj_meta.rect_params.width / 2
                    cy_raw = obj_meta.rect_params.top + obj_meta.rect_params.height
                    prev_pts = self.trail_history_by_sensor[int(sensor_id)][tid]
                    last_ts = self.trail_last_seen_by_sensor[int(sensor_id)].get(tid)
                    cx, cy = cx_raw, cy_raw
                    if prev_pts and last_ts is not None:
                        px, py = prev_pts[-1]
                        dt = max(0.0, float(now - last_ts))
                        if dt > 0.0:
                            dx = cx_raw - px
                            dy = cy_raw - py
                            dist = (dx * dx + dy * dy) ** 0.5
                            max_step = float(self.trail_max_speed_px_per_s) * dt
                            if dist > max_step > 0.0:
                                scale = max_step / dist
                                dx *= scale
                                dy *= scale
                                cx = px + dx
                                cy = py + dy
                    self.trail_history_by_sensor[int(sensor_id)][tid].append((cx, cy))
                    self.trail_last_seen_by_sensor[int(sensor_id)][tid] = now
                l_obj = l_obj.next

            # Prune stale trails for this sensor
            for tid, ts in list(self.trail_last_seen_by_sensor[int(sensor_id)].items()):
                if now - ts > self.trail_timeout_s:
                    self.trail_last_seen_by_sensor[int(sensor_id)].pop(tid, None)
                    self.trail_history_by_sensor[int(sensor_id)].pop(tid, None)
                    # Also prune bbox smoothing state
                    try:
                        _m = self._bbox_smooth_by_sensor.get(int(sensor_id))
                        if _m and int(tid) in _m:
                            _m.pop(int(tid), None)
                    except Exception:
                        pass

            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            if not display_meta:
                l_frame = l_frame.next
                continue
            display_meta.num_lines = 0
            display_meta.num_labels = 0
            line_capacity = len(display_meta.line_params)
            label_capacity = len(display_meta.text_params)
            # Draw trails for this sensor only
            sensor_trails = self.trail_history_by_sensor.get(int(sensor_id), {})
            active_ids = [tid for tid, pts in sensor_trails.items() if len(pts) > 1]
            budget_per_track = (line_capacity // len(active_ids)) if active_ids else line_capacity
            budget_per_track = max(1, budget_per_track)
            for tid in active_ids:
                pts = list(sensor_trails[tid])[
                    -(
                        min(
                            self.config.visualization.TRAIL_DRAW_SEGMENTS,
                            budget_per_track,
                        )
                        + 1
                    ) :
                ]
                for idx in range(len(pts) - 1):
                    if display_meta.num_lines >= line_capacity:
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
                if (
                    self.trail_show_labels
                    and display_meta.num_labels < label_capacity
                    and display_meta.num_lines < line_capacity
                    and display_meta.num_labels < len(display_meta.text_params)
                ):
                    tp = display_meta.text_params[display_meta.num_labels]
                    tp.display_text = _alloc_display_text(f"id {tid}")
                    tp.x_offset, tp.y_offset = map(int, pts[-1])
                    tp.font_params.font_name = _alloc_display_text("Serif")
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
    sources: List[Dict[str, Any]], config: AppConfig
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
        preproc_config="pipelines/config_preproc.ini",
    )


if __name__ == "__main__":
    # Test DeepStream pipeline
    import argparse
    from config import config

    parser = argparse.ArgumentParser(description="Test DeepStream Video Pipeline")

    # Define the default RTSP URI
    default_rtsp_uri = os.environ.get(
        "NOESIS_RTSP_URI", "rtsps://192.168.3.214:7441/jdr9oLlBkjyl3gDm?enableSrtp"
    )

    parser.add_argument(
        "--source-uri", default=default_rtsp_uri, help="Video source (RTSP URL)"
    )
    parser.add_argument(
        "--duration", type=int, default=30, help="Test duration in seconds"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Create test sources from enabled RTSP streams in config
    test_sources = [
        stream for stream in config.cameras.RTSP_STREAMS if stream.get("enabled", True)
    ]
    if not test_sources:
        print("❌ No enabled RTSP streams found in config")
        exit(1)

    print(f"🎥 Testing with {len(test_sources)} streams:")
    for i, source in enumerate(test_sources):
        print(
            f"  Stream {i}: {source.get('name', 'Unknown')} - {source.get('url', 'No URL')}"
        )

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
