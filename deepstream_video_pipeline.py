#!/usr/bin/env python3
"""
DeepStream Video Pipeline

This module provides a high-performance video processing pipeline using NVIDIA DeepStream
for GPU-accelerated video reading, decoding, and preprocessing. It replaces the DALI
pipeline with DeepStream's optimized GStreamer elements.

Key Features:
- GPU-native video decoding with nvurisrcbin
- Batch processing with nvmultiurisrcbin
- Zero-copy tensor output via appsink
- Support for RTSP, file, and camera sources
"""

# Core imports
import sys
import os
os.environ['no_proxy'] = '*'
import logging
import configparser
import re
import threading
import time
import queue
import socket
import ctypes
from collections import defaultdict, deque

# Bypass libproxy issues by disabling GIO proxy resolver
from typing import Optional, Tuple, Dict, Any, List
import math
import numpy as np

# GStreamer imports
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GLib, GstApp  # type: ignore  # noqa: E402

# DeepStream imports
sys.path.append('/opt/nvidia/deepstream/deepstream/lib')
import pyds  # type: ignore  # noqa: E402
from pipelines import meta_ops  # noqa: E402

# Local imports
from websocket_server import WebSocketServer  # noqa: E402

# Defer GStreamer initialization to runtime to avoid crashing at import time
# Some environments (or tracer/proxy setups) can abort during init; doing this
# lazily makes failures easier to diagnose and avoids import-time crashes.

from config import AppConfig  # noqa: E402
from reid.stable_id_manager import StableIDManager  # noqa: E402




class DeepStreamVideoPipeline:
    """
    DeepStream-based video pipeline for GPU-accelerated video processing.
    
    This pipeline uses:
    - nvurisrcbin for source handling (RTSP/file/camera)
    - nvmultiurisrcbin for batching
    - appsink for tensor output
    """
    
    def __init__(
        self,
        config: AppConfig,
        sources: Optional[List[Dict[str, Any]]] = None,
        websocket_port: int = 8765,
        config_file: str = "pipelines/config_infer_primary_yolo11_seg.ini",
    ):
        # Initialize GStreamer as early as possible, but at runtime (not module import)
        try:
            Gst.init(None)
        except Exception as e:
            # Provide clearer guidance if init fails
            raise RuntimeError(f"Failed to initialize GStreamer: {e}.\n"
                               f"Hints: ensure DeepStream is installed and environment is activated.\n"
                               f"Try: source ./activate_deepstream.sh and verify gst-inspect-1.0 works.")
        # Initialize logger before any diagnostic logging
        self.logger = self._get_logger()
        self._log_required_plugins()

        # Multi-stream configuration (sources parameter handled later)
        if sources:
            raise RuntimeError("Sources parameter deprecated; configure streams via DS_MULTIURISRC_CONFIG")
        self.websocket_port = websocket_port
        self.config_file = config_file
        self.config = config  # Store config as instance variable

        # Resolve DS8 multi-URI source configuration
        try:
            self.ds_multiurisrc_cfg = getattr(self.config.processing, "DS_MULTIURISRC_CONFIG", None)
        except Exception:
            self.ds_multiurisrc_cfg = None
        if not self.ds_multiurisrc_cfg:
            self.logger.error("DS_MULTIURISRC_CONFIG missing or file not found: %s", self.ds_multiurisrc_cfg)
            raise RuntimeError("nvmultiurisrcbin INI required for sources")
        self._multiuri_conf = self._load_multiuri_config(self.ds_multiurisrc_cfg)
        self.multiurisrc_uris = self._parse_rtsp_list(self._multiuri_conf, self.ds_multiurisrc_cfg)
        self._uris = self.multiurisrc_uris

        streammux_cfg = self._multiuri_conf["streammux"] if self._multiuri_conf.has_section("streammux") else {}
        source_attr_cfg = self._multiuri_conf["source-attr-all"] if self._multiuri_conf.has_section("source-attr-all") else {}
        width = int(streammux_cfg.get("width", "1920"))
        height = int(streammux_cfg.get("height", "1080"))
        try:
            ini_max_batch = int(streammux_cfg.get("max-batch-size", str(len(self._uris))))
        except ValueError:
            ini_max_batch = len(self._uris)
        self.max_width = width
        self.max_height = height
        self._batched_push_timeout = int(streammux_cfg.get("batched-push-timeout", "40000"))
        self._select_rtp_protocol = int(source_attr_cfg.get("select-rtp-protocol", "4"))
        self.batch_size = max(ini_max_batch, len(self._uris))

        # Create sensor_id to camera name mapping for telemetry (sensor ids 0..N-1)
        self.sensor_ids = list(range(len(self._uris)))
        self.source_idx_by_sensor_id = {sid: sid for sid in self.sensor_ids}
        self.sensor_id_by_source_idx = {idx: idx for idx in self.sensor_ids}
        self.source_info = {}
        for idx, uri in enumerate(self._uris):
            name = f"Camera {idx + 1}"
            self.source_info[idx] = {
                "name": name,
                "clean_name": self._clean_camera_name(name),
                "url": uri,
                "width": width,
                "height": height,
            }

        self.logger.info("🎥 Initializing multi-stream pipeline with %d sources from %s", len(self.sensor_ids), self.ds_multiurisrc_cfg_resolved)
        for sid, info in self.source_info.items():
            self.logger.info(
                "  Sensor %s: %s (%s) - %sx%s @ %s",
                sid,
                info['name'],
                info['clean_name'],
                info['width'],
                info['height'],
                info['url'],
            )
        
        # Initialize pipeline components
        self.pipeline: Optional[Gst.Pipeline] = None
        self.mainloop: Optional[GLib.MainLoop] = None
        self.websocket_server: Optional[WebSocketServer] = None

        
        # Threading and state management
        self.running = False
        self._prepared = False
        self._activated = False
        self._errors: List[str] = []
        self.pipeline_thread: Optional[threading.Thread] = None
        self.websocket_thread: Optional[threading.Thread] = None
        self._state_watch_names = {
            "src",
            "nvinfer",
            "nvdsroiexclude",
            "nvtracker",
            "nvdsanalytics_post",
            "mosaic_q",
            "mosaic_tiler",
            "mosaic_osd",
            "mosaic_conv_post",
            "mosaic_enc",
            "mosaic_sink",
            "egl_q",
            "egl_sink",
            "post_analytics_bev_q",
            "analytics_fanout_q",
        }

        
        # Pipeline configuration derived from INI
        self.device_id = 0
        self.multiurisrc_host = "localhost"
        self.multiurisrc_port = self._allocate_rest_port(9000, 9010)
        self.logger.info(
            "📊 Pipeline config: batch_size=%s (ini=%s, uri-count=%s), resolution=%sx%s, REST port=%s",
            self.batch_size,
            ini_max_batch,
            len(self._uris),
            self.max_width,
            self.max_height,
            self.multiurisrc_port,
        )

        # Add missing attributes for compatibility
        self.frame_count = 0
        self.frame_count_lock = threading.Lock()
        # GPU mosaic JPEG queue (tiled view)
        self.mosaic_queue: queue.Queue[bytes] = queue.Queue(maxsize=10)
        self.start_time = 0  # Will be set in start()
        self.egl_enabled: bool = False
        self.egl_queue: Optional[Gst.Element] = None
        self.egl_sink: Optional[Gst.Element] = None
        self._egl_tee_pad: Optional[Gst.Pad] = None
        # MapAnything on-demand branch state (pre-PGIE tee → demux → per-stream appsinks)
        self._pre_pgie_tee: Optional[Gst.Element] = None
        self._ma_demux: Optional[Gst.Element] = None
        self._ma_branch_ready: bool = False
        self._ma_demux_requested_src: Dict[int, Gst.Pad] = {}
        self._ma_valves: Dict[int, Gst.Element] = {}
        self._ma_sinks: Dict[int, GstApp.AppSink] = {}
        self._ma_frame_queues: Dict[int, queue.Queue] = {sid: queue.Queue(maxsize=1) for sid in self.sensor_ids}
        self._ma_branch_lock = threading.Lock()
        # Track initial-open window for MA valves; FE will close per-sensor when heatmap ready
        self._ma_initial_open: bool = True
        self._ma_safety_close_scheduled: bool = False

        # Post-analytics BEV branch state (nvdsanalytics_post → tee → demux → per-stream BGR appsinks)
        self._post_analytics_tee: Optional[Gst.Element] = None
        self._bev_demux: Optional[Gst.Element] = None
        self._bev_branch_ready: bool = False
        self._bev_frame_queues: Dict[int, queue.Queue] = {sid: queue.Queue(maxsize=1) for sid in self.sensor_ids}
        self._bev_requested_src: Dict[int, Gst.Pad] = {}
        
        # Add tracking history for trail visualization (per sensor_id, per track_id)
        self.trail_history_by_sensor: Dict[int, Dict[int, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self.config.visualization.TRAIL_LENGTH))
        )
        # Respect config default for initial state
        self.trail_visualization_enabled = bool(self.config.visualization.TRAIL_VISUALIZATION_ENABLED)
        # Last-seen timestamps per sensor/track
        self.trail_last_seen_by_sensor: Dict[int, Dict[int, float]] = defaultdict(dict)
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

        # --- bbox smoothing state (per sensor_id, per track_id) ---
        self._bbox_smooth_by_sensor: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
        self.bbox_smoothing_enabled: bool = bool(getattr(self.config.visualization, 'BBOX_SMOOTHING_ENABLED', True))
        self.bbox_smoothing_alpha: float = float(getattr(self.config.visualization, 'BBOX_SMOOTHING_ALPHA', 0.3))
        self.bbox_smoothing_anchor: str = str(getattr(self.config.visualization, 'BBOX_SMOOTHING_ANCHOR', 'bottom')).lower()
        self.bbox_smoothing_max_growth: float = float(getattr(self.config.visualization, 'BBOX_SMOOTHING_MAX_GROWTH', 1.2))
        self.bbox_smoothing_max_shrink: float = float(getattr(self.config.visualization, 'BBOX_SMOOTHING_MAX_SHRINK', 0.85))
        self.bbox_max_drop_window_s: float = float(getattr(self.config.visualization, 'BBOX_MAX_DROP_WINDOW_S', 4.0))
        self.bbox_max_drop_ratio: float = float(getattr(self.config.visualization, 'BBOX_MAX_DROP_RATIO', 0.85))
        self.trail_max_speed_px_per_s: float = float(getattr(self.config.visualization, 'TRAIL_MAX_SPEED_PX_PER_S', 600.0))

        # Track tiler compositor rect logging (per sensor)
        self._logged_tile_rects: set[int] = set()

        # Global Stable ID manager (OSNet/appearance-based global IDs)
        # Respect REID_ENABLED to avoid loading model and consuming GPU if disabled
        try:
            self.reid_enabled: bool = bool(getattr(self.config.models, 'REID_ENABLED', True))
        except Exception:
            self.reid_enabled = True

        self.stable_id_mgr = None
        if self.reid_enabled:
            try:
                reid_model_path = getattr(self.config.models, 'REID_MODEL_PATH', None)
            except Exception:
                reid_model_path = None
            device = getattr(self.config.models, 'DEVICE', 'cuda:0')
            # Pull model selection and crop size from config
            try:
                reid_model_name = getattr(self.config.models, 'REID_MODEL_NAME', 'osnet_ibn_x1_0')
            except Exception:
                reid_model_name = 'osnet_ibn_x1_0'
            try:
                img_h, img_w = getattr(self.config.models, 'REID_IMAGE_SIZE', [256, 128])
                image_size = (int(img_h), int(img_w))
            except Exception:
                image_size = (256, 128)

            self.stable_id_mgr = StableIDManager(
                model_path=reid_model_path,
                device=device,
                model_name=str(reid_model_name),
                image_size=image_size,
                embed_interval_s=float(getattr(self.config.models, 'REID_EMBED_INTERVAL_S', 1.0)),
                max_ghost_age_s=float(getattr(self.config.models, 'REID_MAX_GHOST_AGE_S', 60.0)),
                cos_sim_threshold=float(getattr(self.config.models, 'REID_COS_SIM_THRESHOLD', 0.72)),
                cos_sim_high_threshold=float(getattr(self.config.models, 'REID_COS_SIM_HIGH_THRESHOLD', 0.80)),
                allow_multi_zone_active=bool(getattr(self.config.models, 'REID_ALLOW_MULTI_ZONE_ACTIVE', True)),
                crop_expand=float(getattr(self.config.models, 'REID_CROP_EXPAND', 0.12)),
                tta_flip=bool(getattr(self.config.models, 'REID_TTA_FLIP', True)),
                min_crop_h=int(getattr(self.config.models, 'REID_MIN_CROP_H', 64)),
                min_laplacian_var=float(getattr(self.config.models, 'REID_MIN_LAPLACIAN', 12.0)),
                adaptive_penalty=bool(getattr(self.config.models, 'REID_ADAPTIVE_PENALTY', True)),
                size_penalty_alpha=float(getattr(self.config.models, 'REID_SIZE_PENALTY_ALPHA', 0.08)),
                brightness_penalty_beta=float(getattr(self.config.models, 'REID_BRIGHTNESS_PENALTY_BETA', 0.05)),
                spatial_penalty=bool(getattr(self.config.models, 'REID_SPATIAL_PENALTY', True)),
                spatial_penalty_delta=float(getattr(self.config.models, 'REID_SPATIAL_PENALTY_DELTA', 0.06)),
                color_penalty_gamma=float(getattr(self.config.models, 'REID_COLOR_PENALTY_GAMMA', 0.07)),
                stripe_fusion=bool(getattr(self.config.models, 'REID_STRIPE_FUSION', True)),
                stripe_count=int(getattr(self.config.models, 'REID_STRIPE_COUNT', 3)),
                multi_scale_crops=bool(getattr(self.config.models, 'REID_MULTI_SCALE_CROPS', True)),
                ema_alpha=float(getattr(self.config.models, 'REID_EMA_ALPHA', 0.20)),
                active_id_guard_strict=bool(getattr(self.config.models, 'REID_ACTIVE_ID_GUARD_STRICT', True)),
                active_id_guard_margin=float(getattr(self.config.models, 'REID_ACTIVE_ID_GUARD_MARGIN', 0.03)),
                ghost_strict_age_s=float(getattr(self.config.models, 'REID_GHOST_STRICT_AGE_S', 2.0)),
                ghost_extra_margin=float(getattr(self.config.models, 'REID_GHOST_EXTRA_MARGIN', 0.03)),
                max_active_ids_per_sensor=int(getattr(self.config.models, 'REID_MAX_ACTIVE_IDS_PER_SENSOR', 6)),
                new_id_confirm_frames_at_cap=int(getattr(self.config.models, 'REID_NEW_ID_CONFIRM_FRAMES_AT_CAP', 2)),
                new_id_hysteresis_frames=int(getattr(self.config.models, 'REID_NEW_ID_HYSTERESIS_FRAMES', 2)),
                active_evict_grace_s=float(getattr(self.config.models, 'REID_ACTIVE_EVICT_GRACE_S', 10.0)),
                xcam_handoff_window_s=float(getattr(self.config.models, 'REID_XCAM_HANDOFF_WINDOW_S', 4.0)),
                xcam_handoff_margin=float(getattr(self.config.models, 'REID_XCAM_HANDOFF_MARGIN', 0.02)),
                max_total_ids=int(getattr(self.config.models, 'REID_MAX_TOTAL_IDS', 12)),
                total_id_reuse=bool(getattr(self.config.models, 'REID_TOTAL_ID_REUSE', True)),
                total_id_reuse_min_age_s=float(getattr(self.config.models, 'REID_TOTAL_ID_REUSE_MIN_AGE_S', 600.0)),
                sid_pool_file=str(getattr(self.config.models, 'REID_SID_POOL_FILE', '~/.noesis/sid_pool.json')),
            )
        # Exclusion ROIs parsed from nvdsanalytics exclude config (per DS source index)
        # { stream_index(int): { normalized_label(str): [(x,y), ...] } }
        self._exclusion_rois_by_stream: Dict[int, Dict[str, List[Tuple[float, float]]]] = {}

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

        # Cache track colors so IDs maintain consistent visualization hues
        self._track_color_cache: Dict[int, Tuple[float, float, float]] = {}

        # GStreamer already initialized at module import
        
        # Create pipeline elements
        prepared_ok = self._create_pipeline()
        self._prepared = bool(prepared_ok)
        if not prepared_ok:
            self._errors.append("pipeline creation failed")

    def _get_logger(self) -> logging.Logger:
        """Return the shared pipeline logger."""
        return logging.getLogger("DeepStreamVideoPipeline")

    def _clean_camera_name(self, name: str) -> str:
        """Normalize camera names to stable identifiers."""
        value = (name or "").strip()
        if not value:
            return "camera"
        if "Living Room" in value:
            return "living-room"
        if "Kitchen" in value:
            return "kitchen"
        if "Family Room" in value:
            return "family-room"
        return value.lower().replace(" ", "-").replace("_", "-")

    def _port_in_use(self, host: str, port: int, timeout: float = 0.25) -> bool:
        """Return True if host:port already accepts connections."""
        try:
            with socket.create_connection((host, int(port)), timeout=timeout):
                return True
        except Exception:
            return False

    def _allocate_rest_port(self, start: int, end: int) -> int:
        """Find an available REST port within [start, end]; raise if none free."""
        for candidate in range(start, end + 1):
            if self._port_in_use("localhost", candidate):
                if candidate < end:
                    self.logger.info("Port %s in use; incremented to %s", candidate, candidate + 1)
                continue
            return candidate
        self.logger.error("No free REST port for nvmultiurisrcbin (%s-%s)", start, end)
        raise RuntimeError("No free REST port for nvmultiurisrcbin (9000-9010)")

    def _log_required_plugins(self) -> None:
        """Log presence/absence of key DeepStream plugins for diagnostics."""
        try:
            registry = Gst.Registry.get()
        except Exception:
            self.logger.debug("Gst.Registry not available for plugin diagnostics")
            return

        required = [
            ("nvmultiurisrcbin", "DeepStream multi-URI source"),
            ("nvdspreprocess", "DeepStream preprocess"),
            ("nvinfer", "DeepStream inference"),
            ("nvtracker", "DeepStream tracker"),
            ("nvdsanalytics", "DeepStream analytics"),
            ("nvmultistreamtiler", "Multistream tiler"),
            ("nvdsosd", "On-screen display"),
            ("nvvideoconvert", "NV MM video convert"),
            ("nvjpegenc", "NV JPEG encoder"),
            ("nveglglessink", "EGL sink"),
        ]

        for name, desc in required:
            try:
                present = bool(registry.find_feature(name, Gst.ElementFactory))
            except Exception:
                present = bool(Gst.ElementFactory.find(name))
            log_fn = self.logger.info if present else self.logger.warning
            log_fn(
                "plugin %s: %s",
                name,
                "available" if present else f"MISSING ({desc})",
            )

    def _load_multiuri_config(self, cfg_path: str) -> configparser.ConfigParser:
        parser = configparser.ConfigParser()
        resolved_path = cfg_path
        if cfg_path and not os.path.isabs(cfg_path):
            resolved_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg_path)
        self.ds_multiurisrc_cfg_resolved = resolved_path

        if not resolved_path or not os.path.exists(resolved_path):
            self.logger.error("DS_MULTIURISRC_CONFIG missing or file not found: %s", resolved_path)
            raise RuntimeError("nvmultiurisrcbin INI required for sources")

        try:
            with open(resolved_path, "r", encoding="utf-8") as cfg_file:
                parser.read_file(cfg_file)
            if not parser.sections():
                self.logger.error("nvmultiurisrcbin config %s is empty or invalid", resolved_path)
                raise RuntimeError("Failed to parse nvmultiurisrcbin INI")
            self.logger.debug("Loaded nvmultiurisrcbin config from %s", resolved_path)
            return parser
        except (OSError, configparser.Error) as exc:
            self.logger.error("Failed to parse nvmultiurisrcbin config %s: %s", resolved_path, exc)
            raise RuntimeError("Failed to parse nvmultiurisrcbin INI") from exc

    def _parse_rtsp_list(self, cfg: configparser.ConfigParser, cfg_path: str) -> List[str]:
        try:
            raw_list = cfg.get("source-list", "list")
        except Exception as exc:
            self.logger.error("Missing [source-list].list in %s: %s", cfg_path, exc)
            raise RuntimeError("nvmultiurisrcbin INI missing [source-list].list") from exc
        uris = [uri.strip() for uri in raw_list.split(";") if uri.strip()]
        if not uris:
            self.logger.error("No URIs defined in %s [source-list].list", cfg_path)
            raise RuntimeError("Empty URI list in nvmultiurisrcbin INI")
        self.logger.info("Parsed %d URIs from %s: %s", len(uris), cfg_path, uris)
        return uris

    def _post_remove_excluded_objects_probe(self, pad, info, udata):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        operator = meta_ops.create_operator(gst_buffer)

        # Ensure ROI definitions are loaded before pruning. Cache persists across buffers.
        if not self._exclusion_rois_by_stream:
            try:
                self._post_load_exclusion_rois_from_config()
            except Exception as exc:
                self.logger.debug(f"Unable to load post-analytics exclusion config: {exc}")

        if not self._exclusion_rois_by_stream:
            return Gst.PadProbeReturn.OK

        removed_total = 0
        for frame_meta in meta_ops.iter_frames(operator, gst_buffer):
            ds_index = meta_ops.get_source_id(operator, frame_meta)
            exclusion_polys = self._exclusion_rois_by_stream.get(ds_index)
            if not exclusion_polys:
                continue

            for obj_meta in list(meta_ops.iter_objects(operator, frame_meta)):
                rect = meta_ops.get_rect_params(operator, obj_meta)
                if rect is None:
                    continue
                bbox_corners = meta_ops.rect_to_corners(rect)

                candidate_polys: List[List[Tuple[float, float]]] = []
                analytics_data = self._extract_analytics_obj_meta(operator, obj_meta)
                if analytics_data:
                    roi_labels = self._normalize_roi_status_labels(analytics_data.get("roiStatus"))
                    for lbl in roi_labels:
                        poly = exclusion_polys.get(lbl)
                        if poly:
                            candidate_polys.append(poly)

                if not candidate_polys:
                    candidate_polys = list(exclusion_polys.values())

                if any(len(poly) < 3 for poly in candidate_polys):
                    candidate_polys = [poly for poly in candidate_polys if len(poly) >= 3]
                if not candidate_polys:
                    continue

                should_remove = any(self._bbox_fully_inside_polygon(bbox_corners, poly) for poly in candidate_polys)
                if should_remove and meta_ops.remove_object(operator, frame_meta, obj_meta):
                    removed_total += 1

        if removed_total:
            self.logger.debug("Pruned %s object(s) via post-analytics exclusion probe", removed_total)

        return Gst.PadProbeReturn.OK

    # ---------------- MapAnything on-demand branch ----------------
    # Rationale:
    # - Insert a thread boundary immediately after nvstreamdemux via a tiny, leaky queue so
    #   demux never back-pressures the tee or main PGIE branch.
    # - Place the valve AFTER that queue; when "drop=True" it still consumes and drops so the
    #   upstream remains unblocked. When toggled on, only the specific per-stream leg flows.
    # - Use SystemMemory BGR caps for the appsink to produce CPU-accessible frames for MA RPC.
    # - Keep pre_pgie_tee.allow-not-linked=True for dynamic attach/detach safety.
    def _setup_mapanything_branch(self, elements: Dict[str, Any]) -> None:
        """Create and link a pre-PGIE tee → nvstreamdemux → per-stream BGR appsinks with valves.

        Frames only flow when valves are opened on-demand. By default, valves drop all buffers.
        """
        pre_tee: Gst.Element = elements['pre_pgie_tee']
        pre_ma_q: Gst.Element = elements.get('pre_pgie_q_ma')
        ma_demux: Gst.Element = elements['ma_demux']

        # Link pre_tee → ma_demux via requested src pad
        tee_src1 = pre_tee.get_request_pad("src_1")
        if not tee_src1:
            raise RuntimeError("Failed to request pre_pgie_tee src_1 pad for MA branch")
        if pre_ma_q is not None:
            pre_ma_sink = pre_ma_q.get_static_pad("sink")
            if not pre_ma_sink:
                raise RuntimeError("Failed to get pre_pgie_q_ma sink pad")
            if tee_src1.link(pre_ma_sink) != Gst.PadLinkReturn.OK:
                raise RuntimeError("Failed to link pre_pgie_tee to pre_pgie_q_ma")
            if not pre_ma_q.link(ma_demux):
                raise RuntimeError("Failed to link pre_pgie_q_ma to ma_demux")
        else:
            demux_sink = ma_demux.get_static_pad("sink")
            if not demux_sink:
                raise RuntimeError("Failed to get ma_demux sink pad")
            if tee_src1.link(demux_sink) != Gst.PadLinkReturn.OK:
                raise RuntimeError("Failed to link pre_pgie_tee to ma_demux")

        # Build per-stream branches
        for sensor_id in self.sensor_ids:
            stream_id = self._stream_id_for_sensor(sensor_id)
            q = Gst.ElementFactory.make("queue", f"ma_q_{stream_id}")
            valve = Gst.ElementFactory.make("valve", f"ma_valve_{stream_id}")
            conv = Gst.ElementFactory.make("nvvideoconvert", f"ma_conv_{stream_id}")
            caps = Gst.ElementFactory.make("capsfilter", f"ma_caps_{stream_id}")
            sink = Gst.ElementFactory.make("appsink", f"ma_sink_{stream_id}")
            if not all([q, valve, conv, caps, sink]):
                raise RuntimeError(f"Failed to create MapAnything branch elements for stream {stream_id}")
            try:
                self.pipeline.add(q); self.pipeline.add(valve); self.pipeline.add(conv); self.pipeline.add(caps); self.pipeline.add(sink)
            except Exception:
                pass
            # Keep MA queue tiny and leaky so it can never backpressure upstream
            try:
                q.set_property("max-size-buffers", 1)
                q.set_property("max-size-bytes", 0)
                q.set_property("max-size-time", 0)
                q.set_property("leaky", 2)  # LEAK_DOWNSTREAM
            except Exception:
                pass
            # Configure caps to request CPU/SystemMemory BGR frames
            caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:SystemMemory), format=BGR"))
            # Configure valve to be open initially (allows caps negotiation), will be closed after negotiation completes
            valve.set_property("drop", False)
            # Configure appsink for on-demand single-buffer
            sink.set_property("emit-signals", True)
            sink.set_property("drop", True)
            sink.set_property("sync", False)
            sink.set_property("max-buffers", 1)
            # Critical for live pipelines: don't make BEV appsink participate in preroll
            try:
                sink.set_property("async", False)
                sink.set_property("enable-last-sample", False)
            except Exception:
                pass
            # Connect callback
            sink.connect("new-sample", self._on_new_ma_frame, sensor_id)
            # Link static elements (queue BEFORE valve)
            if not q.link(valve):
                raise RuntimeError(f"Failed to link ma_q to valve for stream {stream_id}")
            if not valve.link(conv):
                raise RuntimeError(f"Failed to link valve to conv for stream {stream_id}")
            if not conv.link(caps):
                raise RuntimeError(f"Failed to link ma_conv to caps for stream {stream_id}")
            if not caps.link(sink):
                raise RuntimeError(f"Failed to link ma_caps to sink for stream {stream_id}")
            # Link demux src_%u → (optional identity) → q
            pad = ma_demux.get_request_pad(f"src_{stream_id}")
            if not pad:
                raise RuntimeError(f"Failed to request ma_demux src_{stream_id} pad")
            # Optional identity to isolate allocation/caps; safe to fall back if creation fails
            identity_el = None
            try:
                identity_el = Gst.ElementFactory.make("identity", f"ma_ident_{stream_id}")
                if identity_el:
                    identity_el.set_property("single-segment", True)
                    self.pipeline.add(identity_el)
            except Exception:
                identity_el = None

            if identity_el is not None:
                ident_sink = identity_el.get_static_pad("sink")
                if not ident_sink:
                    raise RuntimeError(f"Failed to get identity sink pad for stream {stream_id}")
                if pad.link(ident_sink) != Gst.PadLinkReturn.OK:
                    raise RuntimeError(f"Failed to link ma_demux src_{stream_id} to identity_{stream_id}")
                if not identity_el.link(q):
                    raise RuntimeError(f"Failed to link identity_{stream_id} to ma_q_{stream_id}")
            else:
                q_sink = q.get_static_pad("sink")
                if not q_sink:
                    raise RuntimeError(f"Failed to get ma_q sink pad for stream {stream_id}")
                if pad.link(q_sink) != Gst.PadLinkReturn.OK:
                    raise RuntimeError(f"Failed to link ma_demux src_{stream_id} to ma_q_{stream_id}")
            # Track branch components
            self._ma_demux_requested_src[stream_id] = pad
            self._ma_valves[sensor_id] = valve
            self._ma_sinks[sensor_id] = sink  # type: ignore[assignment]
            # Verify caps memory type for MA caps
            try:
                capstr = caps.get_property("caps").to_string()
                if not capstr.startswith("video/x-raw(memory:SystemMemory)"):
                    self.logger.debug("MA caps for sensor %s (stream %s) negotiated to %s", sensor_id, stream_id, capstr)
            except Exception:
                pass
            # Attach flow probes on MA branch for diagnostics
            try:
                self._attach_flow_probe(q.get_static_pad("src"), f"ma_q_{stream_id}.src")
            except Exception:
                pass
            try:
                self._attach_flow_probe(valve.get_static_pad("src"), f"ma_valve_{stream_id}.src")
            except Exception:
                pass
            try:
                self._attach_flow_probe(caps.get_static_pad("src"), f"ma_caps_{stream_id}.src")
            except Exception:
                pass
        # Save top-level for helpers
        self._pre_pgie_tee = pre_tee
        self._ma_demux = ma_demux
        # Top-level MA demux sink probe
        try:
            self._attach_flow_probe(ma_demux.get_static_pad("sink"), "ma_demux.sink")
        except Exception:
            pass

    def _setup_bev_branch(self, elements: Dict[str, Any]) -> None:
        """Create and link a post-analytics tee → nvstreamdemux → per-stream BGR appsinks.

        Always-on design (no valve):
        - Place a tiny, leaky queue immediately after the tee so the BEV leg can never
          backpressure the tee or the main pipeline. Feed directly into the demux.
        - Keep caps on per-stream branches as SystemMemory BGR for CPU access.
        - Attach lightweight flow probes to aid debugging of buffer flow.
        """
        post_analytics_tee: Gst.Element = elements['post_analytics_tee']
        bev_demux: Gst.Element = elements['bev_demux']
        if post_analytics_tee is None or bev_demux is None:
            raise RuntimeError("post_analytics_tee or bev_demux missing; cannot build BEV branch")

        bev_branch_q = Gst.ElementFactory.make("queue", "post_analytics_bev_q")
        if bev_branch_q is None:
            raise RuntimeError("Failed to create post-analytics BEV queue")
        try:
            bev_branch_q.set_property("leaky", 2)
            bev_branch_q.set_property("max-size-buffers", 1)
            bev_branch_q.set_property("max-size-bytes", 0)
            bev_branch_q.set_property("max-size-time", 0)
        except Exception:
            pass
        self.pipeline.add(bev_branch_q)

        tee_src = post_analytics_tee.get_request_pad("src_1")
        if not tee_src:
            raise RuntimeError("Failed to request post_analytics_tee src_1 pad for BEV branch")
        bev_q_sink = bev_branch_q.get_static_pad("sink")
        if not bev_q_sink:
            raise RuntimeError("Failed to get post-analytics BEV queue sink pad")
        # Queue immediately after tee to avoid backpressure, then feed demux
        if tee_src.link(bev_q_sink) != Gst.PadLinkReturn.OK:
            raise RuntimeError("Failed to link post_analytics_tee to BEV queue")
        # Optional: bypass demux for isolation testing (env NOESIS_BEV_BYPASS=1)
        try:
            import os as _os
            _bypass = str(_os.environ.get("NOESIS_BEV_BYPASS", "0")).strip().lower() in {"1","true","yes","y"}
        except Exception:
            _bypass = False
        if _bypass:
            fakesink = Gst.ElementFactory.make("fakesink", "bev_bypass_fakesink")
            if fakesink:
                try:
                    fakesink.set_property("sync", False)
                except Exception:
                    pass
                try:
                    self.pipeline.add(fakesink)
                    if not bev_branch_q.link(fakesink):
                        raise RuntimeError("Failed to link BEV queue to fakesink (bypass)")
                    self.logger.warning("⚠️ NOESIS_BEV_BYPASS enabled: BEV demux bypassed to fakesink")
                except Exception as exc:
                    self.logger.warning(f"BEV bypass failed, falling back to demux: {exc}")
                    if not bev_branch_q.link(bev_demux):
                        raise RuntimeError("Failed to link BEV queue to bev_demux")
            else:
                if not bev_branch_q.link(bev_demux):
                    raise RuntimeError("Failed to link BEV queue to bev_demux")
        else:
            if not bev_branch_q.link(bev_demux):
                raise RuntimeError("Failed to link BEV queue to bev_demux")

        # Attach flow probes for fast diagnosis of buffer flow
        try:
            self._attach_flow_probe(tee_src, "post_analytics_tee.src_1 (BEV)")
        except Exception:
            pass
        try:
            self._attach_flow_probe(bev_branch_q.get_static_pad("src"), "post_analytics_bev_q.src")
        except Exception:
            pass
        try:
            self._attach_flow_probe(bev_demux.get_static_pad("sink"), "bev_demux.sink")
        except Exception:
            pass

        for sensor_id in self.sensor_ids:
            stream_id = self._stream_id_for_sensor(sensor_id)
            q = Gst.ElementFactory.make("queue", f"bev_q_{stream_id}")
            ident = None
            conv = Gst.ElementFactory.make("nvvideoconvert", f"bev_conv_{stream_id}")
            caps = Gst.ElementFactory.make("capsfilter", f"bev_caps_{stream_id}")
            sink = Gst.ElementFactory.make("appsink", f"bev_sink_{stream_id}")
            # Optional identity to isolate caps/allocation (best-effort)
            try:
                ident = Gst.ElementFactory.make("identity", f"bev_ident_{stream_id}")
                if ident:
                    ident.set_property("single-segment", True)
            except Exception:
                ident = None
            if not all([q, conv, caps, sink]) or (ident is None and False):
                raise RuntimeError(f"Failed to create BEV branch elements for stream {stream_id}")
            try:
                if ident:
                    self.pipeline.add(ident)
                self.pipeline.add(q); self.pipeline.add(conv); self.pipeline.add(caps); self.pipeline.add(sink)
            except Exception:
                pass
            try:
                q.set_property("leaky", 2)
                q.set_property("max-size-buffers", 1)
                q.set_property("max-size-bytes", 0)
                q.set_property("max-size-time", 0)
            except Exception:
                pass
            caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:SystemMemory), format=BGR"))
            sink.set_property("emit-signals", True)
            sink.set_property("drop", True)
            sink.set_property("sync", False)
            sink.set_property("max-buffers", 1)
            sink.connect("new-sample", self._on_new_bev_frame, sensor_id)

            if not q.link(conv):
                raise RuntimeError(f"Failed to link bev_q_{stream_id} to converter")
            if not conv.link(caps):
                raise RuntimeError(f"Failed to link bev_conv_{stream_id} to caps")
            if not caps.link(sink):
                raise RuntimeError(f"Failed to link bev_caps_{stream_id} to sink")

            # Flow probe per-stream queue to observe demux fanout
            try:
                self._attach_flow_probe(q.get_static_pad("src"), f"bev_q_{stream_id}.src")
            except Exception:
                pass

            pad = bev_demux.get_request_pad(f"src_{stream_id}")
            if not pad:
                raise RuntimeError(f"Failed to request bev_demux src_{stream_id} pad")
            # Link demux → optional identity → q
            if ident is not None:
                ident_sink = ident.get_static_pad("sink")
                if not ident_sink:
                    raise RuntimeError(f"Failed to get bev_ident_{stream_id} sink pad")
                if pad.link(ident_sink) != Gst.PadLinkReturn.OK:
                    raise RuntimeError(f"Failed to link bev_demux src_{stream_id} to bev_ident_{stream_id}")
                if not ident.link(q):
                    raise RuntimeError(f"Failed to link bev_ident_{stream_id} to bev_q_{stream_id}")
            else:
                sink_pad = q.get_static_pad("sink")
                if not sink_pad:
                    raise RuntimeError(f"Failed to get bev_q_{stream_id} sink pad")
                if pad.link(sink_pad) != Gst.PadLinkReturn.OK:
                    raise RuntimeError(f"Failed to link bev_demux src_{stream_id} to bev_q_{stream_id}")
            self._bev_requested_src[stream_id] = pad

        self._bev_demux = bev_demux
        self._post_analytics_tee = post_analytics_tee
        

        # No valve used; branch is always-on

    def _stream_id_for_sensor(self, sensor_id: int) -> int:
        """Return the nvstreammux/demux stream index for a given sensor_id.

        Uses the registration order mapping populated at init time.
        """
        try:
            return int(self.source_idx_by_sensor_id.get(int(sensor_id), int(sensor_id)))
        except Exception:
            return int(sensor_id)

    def enable_mapanything_for_sensor(self, sensor_id: int) -> None:
        valve = self._ma_valves.get(sensor_id)
        if not valve:
            self.logger.warning("No MA valve for sensor %s", sensor_id)
            return
        try:
            GLib.idle_add(valve.set_property, "drop", False, priority=GLib.PRIORITY_HIGH)
        except Exception:
            try:
                valve.set_property("drop", False)
            except Exception:
                pass

    def disable_mapanything_for_sensor(self, sensor_id: int) -> None:
        valve = self._ma_valves.get(sensor_id)
        if not valve:
            self.logger.warning("No MA valve for sensor %s", sensor_id)
            return
        try:
            GLib.idle_add(valve.set_property, "drop", True, priority=GLib.PRIORITY_HIGH)
        except Exception:
            try:
                valve.set_property("drop", True)
            except Exception:
                pass

    def enable_mapanything_all(self) -> None:
        for sid in list(self._ma_valves.keys()):
            self.enable_mapanything_for_sensor(sid)

    def disable_mapanything_all(self) -> None:
        for sid in list(self._ma_valves.keys()):
            self.disable_mapanything_for_sensor(sid)

    def is_mapanything_enabled_for_sensor(self, sensor_id: int) -> bool:
        """Check if MapAnything valve is currently open (enabled) for a sensor.
        
        Returns True if valve is open (drop=False), False if closed (drop=True) or not available.
        """
        if not self._ma_branch_ready:
            return False
        valve = self._ma_valves.get(sensor_id)
        if not valve:
            return False
        try:
            # Query the drop property: False means valve is open (enabled), True means closed (disabled)
            drop = valve.get_property("drop")
            return not bool(drop)
        except Exception:
            # If we can't query the property, assume disabled for safety
            return False

    def _teardown_mapanything_branch(self) -> None:
        # Close valves first to stop downstream while consuming/dropping
        for sid, valve in list(self._ma_valves.items()):
            try:
                valve.set_property("drop", True)
            except Exception:
                pass
        # Release requested demux pads
        for stream_id, pad in list(self._ma_demux_requested_src.items()):
            try:
                if self._ma_demux and pad:
                    self._ma_demux.release_request_pad(pad)
            except Exception:
                pass
            self._ma_demux_requested_src.pop(stream_id, None)
        self._ma_initial_open = False
        self._ma_safety_close_scheduled = False

    def _on_new_ma_frame(self, appsink: GstApp.AppSink, sensor_id: int) -> Gst.FlowReturn:
        try:
            sample = appsink.emit("pull-sample")
            if not sample:
                return Gst.FlowReturn.OK
            buf = sample.get_buffer()
            caps = sample.get_caps()
            if not buf or not caps:
                return Gst.FlowReturn.OK
            s = caps.get_structure(0)
            width = s.get_value('width') if s and s.has_field('width') else None
            height = s.get_value('height') if s and s.has_field('height') else None
            if not isinstance(width, int) or not isinstance(height, int):
                return Gst.FlowReturn.OK
            success, map_info = buf.map(Gst.MapFlags.READ)
            if not success:
                return Gst.FlowReturn.OK
            try:
                frame = memoryview(map_info.data)[:]
                # BGR packed
                try:
                    import numpy as _np
                    arr = _np.frombuffer(frame, dtype=_np.uint8)
                    arr = arr.reshape((int(height), int(width), 3))
                except Exception:
                    arr = None
                if arr is not None:
                    try:
                        self.logger.debug(f"MA new-sample sensor={sensor_id} size={width}x{height}")
                    except Exception:
                        pass
                    q = self._ma_frame_queues.get(sensor_id)
                    if q is not None:
                        # Clear stale frame if present to keep latest
                        try:
                            while not q.empty():
                                q.get_nowait()
                        except Exception:
                            pass
                        try:
                            q.put_nowait(arr.copy())
                        except Exception:
                            pass
                    # FE-driven close will manage valves; no per-frame auto-close
            finally:
                buf.unmap(map_info)
        except Exception:
            pass
        return Gst.FlowReturn.OK

    def _on_new_bev_frame(self, appsink: GstApp.AppSink, sensor_id: int) -> Gst.FlowReturn:
        sample = None
        buf = None
        map_info = None
        try:
            sample = appsink.emit("pull-sample")
            if not sample:
                return Gst.FlowReturn.OK
            buf = sample.get_buffer()
            caps = sample.get_caps()
            if not buf or not caps:
                return Gst.FlowReturn.OK
            s = caps.get_structure(0)
            width = s.get_value('width') if s and s.has_field('width') else None
            height = s.get_value('height') if s and s.has_field('height') else None
            if not isinstance(width, int) or not isinstance(height, int):
                return Gst.FlowReturn.OK
            success, map_info = buf.map(Gst.MapFlags.READ)
            if not success:
                return Gst.FlowReturn.OK
            frame = np.frombuffer(map_info.data, dtype=np.uint8)
            frame = frame.reshape((int(height), int(width), 3))
            frame_copy = frame.copy()

            # Log first successful BEV sample per sensor to confirm flow
            try:
                if not hasattr(self, "_bev_first_logged"):
                    self._bev_first_logged = set()
                if sensor_id not in self._bev_first_logged:
                    self.logger.info("🎞️ First BEV sample received for sensor %s: %sx%s", sensor_id, width, height)
                    self._bev_first_logged.add(sensor_id)
            except Exception:
                pass

            operator = meta_ops.create_operator(buf)
            frames = meta_ops.iter_frames(operator, buf)
            frame_meta = frames[0] if frames else None
            if frame_meta is None:
                return Gst.FlowReturn.OK
            frame_num = int(getattr(frame_meta, "frame_num", -1) or -1)
            ntp_ts = int(getattr(frame_meta, "ntp_timestamp", 0) or 0)
            footpoints: List[Dict[str, Any]] = []
            frame_width = float(getattr(frame_meta, "source_frame_width", width) or width)
            frame_height = float(getattr(frame_meta, "source_frame_height", height) or height)
            for obj_meta in meta_ops.iter_objects(operator, frame_meta):
                class_id = meta_ops.get_class_id(operator, obj_meta)
                if class_id not in (0, 1):  # prioritize person class (0) but allow overrides
                    continue
                fp = self._footpoint_from_object(operator, obj_meta, frame_width, frame_height)
                if fp:
                    footpoints.append(fp)

            entry = {
                'frame': frame_copy,
                'frame_num': frame_num,
                'ntp_ts': ntp_ts,
                'pts': int(buf.pts) if buf.pts != Gst.CLOCK_TIME_NONE else None,
                'footpoints': footpoints,
                'sensor_id': sensor_id,
                'width': width,
                'height': height,
            }
            q = self._bev_frame_queues.get(sensor_id)
            if q is not None:
                try:
                    while not q.empty():
                        q.get_nowait()
                except Exception:
                    pass
                try:
                    q.put_nowait(entry)
                except Exception:
                    pass
        except Exception:
            self.logger.debug("BEV appsink callback failed", exc_info=True)
        finally:
            try:
                if buf and map_info:
                    buf.unmap(map_info)
            except Exception:
                pass
            try:
                if sample:
                    sample.unref()
            except Exception:
                pass
        return Gst.FlowReturn.OK

    def _footpoint_from_object(self, operator: Optional[Any], obj_meta: Any, frame_width: float, frame_height: float) -> Optional[Dict[str, Any]]:
        rect = meta_ops.get_rect_params(operator, obj_meta)
        if rect is None:
            return None
        left = float(getattr(rect, "left", 0.0))
        top = float(getattr(rect, "top", 0.0))
        width = float(getattr(rect, "width", 0.0))
        height = float(getattr(rect, "height", 0.0))
        obj_id = meta_ops.get_object_id(operator, obj_meta)

        mask = self._extract_mask_array(obj_meta)
        method = "bbox"
        if mask is not None:
            mask_fp = self._footpoint_from_mask_pixels(mask)
            if mask_fp is not None:
                mx, my, method = mask_fp
                if mask.shape[1] > 0 and mask.shape[0] > 0:
                    u = left + (mx / float(mask.shape[1])) * width
                    v = top + (my / float(mask.shape[0])) * height
                else:
                    u = left + width * 0.5
                    v = top + height
            else:
                u = left + width * 0.5
                v = top + height
        else:
            u = left + width * 0.5
            v = top + height
        u = float(np.clip(u, 0.0, frame_width))
        v = float(np.clip(v, 0.0, frame_height))
        return {'u': u, 'v': v, 'track_id': obj_id, 'method': method}

    @staticmethod
    def _footpoint_from_mask_pixels(mask: np.ndarray) -> Optional[Tuple[float, float, str]]:
        if mask.size == 0:
            return None
        binary = mask > 0
        height, width = binary.shape
        band = max(8, int(height * 0.03))
        start = max(0, height - band)
        best = None
        best_len = 0
        for y in range(start, height):
            row = binary[y]
            if not row.any():
                continue
            run_start = None
            run_len = 0
            best_row = None
            for x, val in enumerate(row):
                if val:
                    if run_start is None:
                        run_start = x
                        run_len = 1
                    else:
                        run_len += 1
                else:
                    if run_start is not None and run_len > 0:
                        if best_row is None or run_len > best_row[2]:
                            best_row = (run_start, x - 1, run_len)
                    run_start = None
                    run_len = 0
            if run_start is not None and run_len > 0:
                if best_row is None or run_len > best_row[2]:
                    best_row = (run_start, width - 1, run_len)
            if best_row and best_row[2] >= best_len:
                best = (y, best_row)
                best_len = best_row[2]
        if best:
            y_row, (x_start, x_end, _) = best
            return ((x_start + x_end) * 0.5, float(y_row), "mask-run")
        coords = np.column_stack(np.nonzero(binary))
        if coords.size > 0:
            cx = float(np.mean(coords[:, 1]))
            cy = float(np.max(coords[:, 0]))
            return (cx, cy, "mask-centroid")
        return None

    @staticmethod
    def _extract_mask_array(obj_meta: Any) -> Optional[np.ndarray]:
        mask_params = getattr(obj_meta, "mask_params", None)
        if mask_params is None:
            return None
        width = int(getattr(mask_params, "width", 0))
        height = int(getattr(mask_params, "height", 0))
        pitch = int(getattr(mask_params, "pitch", width))
        size = int(getattr(mask_params, "size", width * height))
        data_ptr = getattr(mask_params, "data", None)
        if not data_ptr or width <= 0 or height <= 0 or size <= 0:
            return None
        try:
            try:
                ptr = ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_uint8 * size))
            except TypeError:
                ptr = ctypes.cast(int(data_ptr), ctypes.POINTER(ctypes.c_uint8 * size))
            np_array = np.ctypeslib.as_array(ptr.contents)
            mask = np_array.reshape((height, pitch))[:, :width]
            return mask.copy()
        except Exception:
            return None

    def _set_ma_valve(self, sensor_id: int, drop: bool) -> None:
        valve = self._ma_valves.get(sensor_id)
        if not valve:
            return
        try:
            # Prefer scheduling onto GLib loop if available
            if self.mainloop:
                def _apply():
                    try:
                        valve.set_property("drop", bool(drop))
                    except Exception:
                        pass
                    return False
                GLib.idle_add(_apply)
            else:
                valve.set_property("drop", bool(drop))
        except Exception:
            pass

    def _close_ma_valves_after_negotiation(self) -> bool:
        """Close all MA valves after caps negotiation completes.

        This allows the nvstreamdemux to complete caps negotiation with all
        downstream elements before we shut off the flow, preventing the demux
        from blocking the tee and main pipeline.
        
        Returns False to stop the timeout callback from being called again.
        """
        # Close all valves
        for sensor_id in list(self._ma_valves.keys()):
            self.disable_mapanything_for_sensor(sensor_id)
        self.logger.info("✅ Closed all MA valves after caps negotiation")
        self._ma_initial_open = False
        return False  # Stop the timeout callback

    def _close_ma_valves_after_startup(self) -> bool:
        """Safety timeout to close any MA valves still open after startup window."""
        if not self._ma_branch_ready:
            return False
        if not self._ma_initial_open:
            return False
        for sensor_id in list(self._ma_valves.keys()):
            try:
                self.disable_mapanything_for_sensor(sensor_id)
            except Exception:
                pass
        self._ma_initial_open = False
        self.logger.info("✅ Safety timeout elapsed; closed remaining MA valves")
        return False

    def read_ma_bgr(self, sensor_id: int, timeout: float = 1.5):
        """Open the MA valve for a single frame and return BGR ndarray, then close valve.

        Returns None if branch not ready or on timeout/error.
        """
        if not self._ma_branch_ready:
            return None
        if sensor_id not in self._ma_frame_queues:
            return None
        # Check if valve was closed before we attempt capture (for warning suppression)
        valve_was_closed = not self.is_mapanything_enabled_for_sensor(sensor_id)
        q = self._ma_frame_queues[sensor_id]
        # Flush any stale entry
        try:
            while not q.empty():
                q.get_nowait()
        except Exception:
            pass
        # Open valve and wait briefly for frames to start; increase prime for first-sample reliability
        self._set_ma_valve(sensor_id, False)
        try:
            time.sleep(1.0)
        except Exception:
            pass
        # Poll until timeout expires
        end_by = time.time() + max(0.1, float(timeout))
        try:
            while time.time() < end_by:
                try:
                    frame = q.get_nowait()
                    self.logger.debug(f"Successfully captured frame for sensor {sensor_id}")
                    return frame
                except Exception:
                    time.sleep(0.05)
            # If valve was closed before the call, timeout is expected - log at debug level
            if valve_was_closed:
                self.logger.debug(f"Failed to capture frame for sensor {sensor_id} within timeout (valve was closed)")
            else:
                self.logger.warning(f"Failed to capture frame for sensor {sensor_id} within timeout")
            return None
        finally:
            self._set_ma_valve(sensor_id, True)

    def read_bev_frame(self, sensor_id: int, timeout: float = 0.5) -> Optional[Dict[str, Any]]:
        """Retrieve the latest BEV branch frame (BGR) and metadata for a sensor."""
        if not self._bev_branch_ready:
            return None
        q = self._bev_frame_queues.get(sensor_id)
        if q is None:
            return None
        try:
            return q.get(timeout=max(0.01, float(timeout)))
        except Exception:
            return None

    # ---- Exclusion ROI helpers ----
    def _post_load_exclusion_rois_from_config(self, path: Optional[str] = None) -> None:
        """Parse nvdsanalytics post config and cache ROI polygons per stream index.

        Sections are of the form [roi-filtering-stream-<index>], and keys like roi-<LABEL>=x;y;...
        """
        try:
            if not path:
                path = "pipelines/config_nvdsanalytics_post.ini"
            if path and not os.path.isabs(path):
                path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
            cfg = configparser.ConfigParser()
            read = cfg.read(path)
            if not read:
                self.logger.warning(f"⚠️ Exclusion ROI config not found or unreadable: {path}")
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
                    if not key.lower().startswith('roi-'):
                        continue
                    label_raw = key[len('roi-'):]
                    label_norm = self._normalize_roi_label(label_raw)
                    pts = self._parse_points_list(val)
                    if len(pts) >= 3:
                        # Store under both normalized and original forms for robust lookup
                        rois_by_stream[sidx][label_norm] = pts
                        rois_by_stream[sidx][self._normalize_roi_label('roi-' + label_raw)] = pts
                        total += 1
            self._exclusion_rois_by_stream = rois_by_stream
            if total:
                self.logger.debug(f"✅ Loaded {total} exclusion ROI(s) from {path}")
            else:
                self.logger.debug(f"ℹ️ No exclusion ROIs found in {path}")
        except Exception as e:
            self.logger.error(f"Error parsing exclusion ROI config {path}: {e}")

    # ---- Batch meta operator helpers ----
    def _create_batch_meta_operator(self, gst_buffer: Any) -> Optional[Any]:
        # Backward-compat shim (use pipelines.meta_ops instead)
        return meta_ops.create_operator(gst_buffer)

    def _parse_points_list(self, s: str) -> List[Tuple[float, float]]:
        """Parse 'x1;y1; x2;y2; ...' into [(x1,y1), ...]."""
        try:
            tokens = re.split(r"[\s;,]+", s.strip())
            nums = [float(t) for t in tokens if t != '']
            pts: List[Tuple[float, float]] = []
            for i in range(0, len(nums) - 1, 2):
                pts.append((nums[i], nums[i + 1]))
            return pts
        except Exception as e:
            self.logger.debug(f"Failed to parse ROI points list '{s}': {e}")
            return []

    def _normalize_roi_label(self, label: str) -> str:
        """Normalize ROI label for consistent matching (case-insensitive, strip 'roi-' prefix)."""
        lbl = str(label).strip()
        # Remove optional leading 'roi-'
        if lbl.lower().startswith('roi-'):
            lbl = lbl[4:]
        return lbl.strip().lower()

    def _normalize_roi_status_labels(self, roi_status: Any) -> set:
        """Normalize various possible roiStatus formats into a set of comparable labels."""
        labels = set()
        try:
            if roi_status is None:
                return labels
            if isinstance(roi_status, dict):
                for k, v in roi_status.items():
                    if v in (1, True, 'IN', 'inside', 'INROI', 'in'):
                        labels.add(self._normalize_roi_label(k))
            elif isinstance(roi_status, (list, tuple, set)):
                for item in roi_status:
                    labels.add(self._normalize_roi_label(str(item)))
            elif isinstance(roi_status, str):
                for token in re.split(r"[\s,;]+", roi_status.strip()):
                    if token:
                        labels.add(self._normalize_roi_label(token))
        except Exception as e:
            self.logger.debug(f"Failed to normalize roiStatus '{roi_status}': {e}")
        return labels

    def _point_in_polygon(self, x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
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
                x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1)
            if intersects:
                inside = not inside
        return inside

    def _bbox_fully_inside_polygon(self, corners: List[Tuple[float, float]], poly: List[Tuple[float, float]]) -> bool:
        """Return True if all bbox corners are inside the polygon."""
        for (px, py) in corners:
            if not self._point_in_polygon(px, py, poly):
                return False
        return True

    def _configure_elements(self, elements: Dict[str, Any]) -> bool:
        """Configure properties for all pipeline elements."""
        try:
            self.logger.info("------------- Configuring GStreamer Elements-------------")
            
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self._on_bus_message)

            # Streammux configuration - strict DS8-style nvmultiurisrcbin setup
            uri_list = ",".join(self._uris)
            sensor_id_list = ",".join(str(i) for i in self.sensor_ids)
            try:
                multiurisrc = elements['multiurisrc']
                multiurisrc.set_property("uri-list", uri_list)
                multiurisrc.set_property("sensor-id-list", sensor_id_list)
                multiurisrc.set_property("max-batch-size", self.batch_size)
                multiurisrc.set_property("width", self.max_width)
                multiurisrc.set_property("height", self.max_height)
                multiurisrc.set_property("batched-push-timeout", self._batched_push_timeout)
                multiurisrc.set_property("live-source", 1)
                multiurisrc.set_property("select-rtp-protocol", self._select_rtp_protocol)
                multiurisrc.set_property("ip-address", self.multiurisrc_host)
                multiurisrc.set_property("port", self.multiurisrc_port)
            except Exception as exc:
                self.logger.error("Failed to configure nvmultiurisrcbin properties: %s", exc)
                raise
            self.logger.info(
                "Configured nvmultiurisrcbin with %d URIs (uri-list set); sensor-id-list=%s, max-batch-size=%s, port=%s",
                len(self._uris),
                sensor_id_list,
                self.batch_size,
                self.multiurisrc_port,
            )

            # Resolve all config file paths to absolute so startup is independent of CWD
            _root_dir = os.path.dirname(os.path.abspath(__file__))

            # Configure nvdspreprocess
            _preproc_cfg = getattr(self.config.processing, 'DEEPSTREAM_PREPROCESS_CONFIG', 'pipelines/config_preproc.ini')
            if _preproc_cfg and not os.path.isabs(_preproc_cfg):
                _preproc_cfg = os.path.join(_root_dir, _preproc_cfg)
            
            # Check if preprocessing is enabled in the config file
            enabled = 1
            if _preproc_cfg and os.path.exists(_preproc_cfg):
                try:
                    cp = configparser.ConfigParser(interpolation=None, delimiters=("="))
                    cp.read(_preproc_cfg)
                    if cp.has_section("property") and cp.has_option("property", "enable"):
                        enabled = int(cp.get("property", "enable") or "1")
                except Exception as e:
                    self.logger.warning(f"Failed to read enable flag from preprocess config: {e}, defaulting to enabled")
                    enabled = 1
            
            if enabled and _preproc_cfg and os.path.exists(_preproc_cfg):
                elements['nvdspreprocess'].set_property("config-file", _preproc_cfg)
                self.logger.info("✅ Using nvdspreprocess config: %s", _preproc_cfg)
            else:
                self.logger.warning("⚠️  nvdspreprocess config file not found or disabled: %s", _preproc_cfg)

            # Primary nvinfer config — static INI by default (YAML override optional via env)
            use_ds8_yaml = str(os.environ.get("NOESIS_USE_INFER_YAML", "0")).strip().lower() in {"1","true","yes","y"}
            if use_ds8_yaml:
                try:
                    from noesis.config.adapters import nvinfer_props_from_infer_yaml
                    # Default to repo-root/config/infer.yaml (file lives at repo root)
                    yaml_path = os.environ.get(
                        "NOESIS_INFER_YAML",
                        os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "infer.yaml"),
                    )
                    props = nvinfer_props_from_infer_yaml(yaml_path)
                    if props:
                        self.logger.info("Using DS8 infer.yaml for nvinfer properties: %s", yaml_path)
                        self.logger.debug("nvinfer DS8 props: %s", props)
                        for key, val in props.items():
                            try:
                                elements['nvinfer'].set_property(key, val)
                            except Exception as e:
                                self.logger.warning(f"Unable to set nvinfer property {key}={val}: {e}")
                        # Always make tensor meta available to downstream consumers
                        try:
                            elements['nvinfer'].set_property("input-tensor-meta", False)
                        except Exception as e:
                            self.logger.warning(f"Failed to disable nvinfer input-tensor-meta: {e}")
                        # Ensure nvinfer still has a configuration file path (plugin requires it)
                        _nvinfer_cfg = self.config_file
                        if _nvinfer_cfg and not os.path.isabs(_nvinfer_cfg):
                            _nvinfer_cfg = os.path.join(_root_dir, _nvinfer_cfg)
                        if _nvinfer_cfg and os.path.exists(_nvinfer_cfg):
                            try:
                                elements['nvinfer'].set_property("config-file-path", _nvinfer_cfg)
                                self.logger.debug("nvinfer config-file-path set to %s (base INI)", _nvinfer_cfg)
                            except Exception as e:
                                self.logger.warning(f"Failed to set nvinfer config-file-path to {_nvinfer_cfg}: {e}")
                        else:
                            self.logger.warning("No base nvinfer INI found at %s; plugin may require a config file", _nvinfer_cfg)
                    else:
                        self.logger.warning("infer.yaml did not yield nvinfer properties; falling back to INI")
                        use_ds8_yaml = False
                except Exception as exc:
                    self.logger.warning("Failed to apply DS8 infer.yaml override: %s", exc)
                    use_ds8_yaml = False

            if not use_ds8_yaml:
                _nvinfer_cfg = self.config_file
                if _nvinfer_cfg and not os.path.isabs(_nvinfer_cfg):
                    _nvinfer_cfg = os.path.join(_root_dir, _nvinfer_cfg)
                # Static only, no dynamics; rely on canonical DS7 INI configuration
                elements['nvinfer'].set_property("config-file-path", _nvinfer_cfg)
                self.logger.info("✅ Using nvinfer config: %s", _nvinfer_cfg)
                # Enable tensor meta for downstream ReID/analytics checks; remove if proven unnecessary
                elements['nvinfer'].set_property("input-tensor-meta", False)

            nvdsroiexclude = elements.get('nvdsroiexclude')
            if nvdsroiexclude:
                exclude_cfg = getattr(
                    self.config.processing,
                    "DEEPSTREAM_EXCLUDE_CONFIG",
                    "pipelines/config_nvdsanalytics_exclude.ini",
                )
                if exclude_cfg and not os.path.isabs(exclude_cfg):
                    exclude_cfg = os.path.join(_root_dir, exclude_cfg)
                if exclude_cfg and os.path.exists(exclude_cfg):
                    nvdsroiexclude.set_property("config-file", exclude_cfg)
                    # Default to source-id to avoid batch-order/pad-index ambiguity
                    # Use env NOESIS_DS_EXCLUDE_ID_MODE=pad-index only if you explicitly
                    # sort batches or want pad-index semantics.
                    id_mode = os.environ.get("NOESIS_DS_EXCLUDE_ID_MODE", "source-id").strip().lower()
                    try:
                        if id_mode in {"pad-index", "pad_index"}:
                            nvdsroiexclude.set_property("id-mode", "pad-index")
                        else:
                            nvdsroiexclude.set_property("id-mode", "source-id")
                    except Exception:
                        pass
                    self.logger.info("✅ Using nvdsroiexclude config: %s", exclude_cfg)
                else:
                    self.logger.warning("⚠️ nvdsroiexclude config file not found: %s", exclude_cfg)
            else:
                self.logger.warning("nvdsroiexclude element missing; ROI exclusion disabled")

            # Tracker configuration — retain DS7 static config (no YAML overrides)
            elements['nvtracker'].set_property(
                "ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
            )
            tracker_cfg_rel = getattr(
                self.config.processing, "DEEPSTREAM_TRACKER_CONFIG", "pipelines/config_tracker_nvdcf_batch_lowlevel.yml"
            )
            if tracker_cfg_rel and not os.path.isabs(tracker_cfg_rel):
                tracker_cfg_rel = os.path.join(_root_dir, tracker_cfg_rel)
            elements['nvtracker'].set_property("ll-config-file", tracker_cfg_rel)
            self.logger.info("Using nvtracker config: %s", tracker_cfg_rel)

            # Post-tracker analytics
            elements['nvdsanalytics_post'].set_property("unique-id", 201)
            analytics_cfg_path: Optional[str] = None
            use_ds8_analytics_yaml = str(os.environ.get("NOESIS_USE_ANALYTICS_YAML", "0")).strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
            }
            if use_ds8_analytics_yaml:
                try:
                    from noesis.config.adapters import analytics_config_from_yaml

                    analytics_cfg_path = analytics_config_from_yaml()
                except Exception as exc:
                    self.logger.warning("Failed to load DS8 analytics YAML override: %s", exc)
                    analytics_cfg_path = None
            if analytics_cfg_path:
                elements['nvdsanalytics_post'].set_property("config-file", analytics_cfg_path)
                self.logger.info("Using DS8 analytics config: %s", analytics_cfg_path)
            else:
                if use_ds8_analytics_yaml and not analytics_cfg_path:
                    self.logger.warning("Falling back to nvdsanalytics INI because DS8 analytics YAML was unavailable")
                _post_cfg_path = "pipelines/config_nvdsanalytics_post.ini"
                if _post_cfg_path and not os.path.isabs(_post_cfg_path):
                    _post_cfg_path = os.path.join(_root_dir, _post_cfg_path)
                elements['nvdsanalytics_post'].set_property("config-file", _post_cfg_path)
                self.logger.info("Using nvdsanalytics config: %s", _post_cfg_path)

            mosaic_osd = elements.get('mosaic_osd')
            if mosaic_osd:
                # Prefer configuration via INI, similar to other plugins
                _osd_cfg = getattr(self.config.processing, 'DEEPSTREAM_OSD_CONFIG', 'pipelines/config_osd.ini')
                if _osd_cfg and not os.path.isabs(_osd_cfg):
                    _osd_cfg = os.path.join(_root_dir, _osd_cfg)
                applied_osd_cfg = False
                if _osd_cfg and os.path.exists(_osd_cfg):
                    try:
                        cp = configparser.ConfigParser(interpolation=None, delimiters=("="))
                        cp.read(_osd_cfg)
                        if cp.has_section('property'):
                            for key, raw in cp.items('property'):
                                val_s = str(raw).strip()
                                val_l = val_s.lower()
                                if val_l in {"true", "yes", "y"}:
                                    val = True
                                elif val_l in {"false", "no", "n"}:
                                    val = False
                                else:
                                    try:
                                        if val_s.isdigit() or (val_s.startswith('-') and val_s[1:].isdigit()):
                                            val = int(val_s)
                                        else:
                                            val = float(val_s)
                                    except Exception:
                                        val = val_s
                                try:
                                    mosaic_osd.set_property(key, val)
                                except Exception as e:
                                    self.logger.warning(f"Unable to set nvdsosd property {key}={val} from {_osd_cfg}: {e}")
                            applied_osd_cfg = True
                            self.logger.info("✅ Using nvdsosd config: %s", _osd_cfg)
                    except Exception as e:
                        self.logger.warning(f"Failed to apply nvdsosd config {_osd_cfg}: {e}")
                if not applied_osd_cfg:
                    # Fallback defaults: mask on, bbox off, text on, CPU mode
                    mosaic_osd.set_property("process-mode", 1)
                    mosaic_osd.set_property("display-text", 1)
                    # Default to seg-friendly visuals; override below if detection-only model is selected
                    mosaic_osd.set_property("display-bbox", 0)
                    mosaic_osd.set_property("display-mask", 1)

                # If detection-only model is active, prefer bounding boxes and hide masks regardless of INI
                try:
                    use_seg = bool(getattr(self.config.processing, 'USE_SEGMENTATION_MODEL', True))
                    if not use_seg:
                        mosaic_osd.set_property("display-mask", 0)
                        mosaic_osd.set_property("display-bbox", 1)
                        mosaic_osd.set_property("display-text", 1)
                        self.logger.info("OSD adjusted for detection-only model: bbox=1, mask=0")
                except Exception:
                    pass

            # Configure live queues with consistent leaky buffering
            for queue_name in ("q_before_tracker", "mosaic_q", "jpeg_q", "egl_q", "analytics_fanout_q"):
                queue_el = elements.get(queue_name)
                if queue_el:
                    self._configure_live_queue(queue_el)
            
            self.logger.info("✅ All elements configured successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error configuring elements: {e}")
            return False

    def _build_main_pipeline_chain(self, elements: Dict[str, Any]) -> bool:
        """Link all main pipeline elements together."""
        try:
            self.logger.info("------------- Linking Main Pipeline Chain-------------")
            
            # Link main processing chain with pre-PGIE tee
            if not elements['multiurisrc'].link(elements['nvdspreprocess']):
                raise RuntimeError("Failed to link nvmultiurisrcbin to nvdspreprocess")
            # Insert pre-PGIE tee for MapAnything branch
            if not elements['nvdspreprocess'].link(elements['pre_pgie_tee']):
                raise RuntimeError("Failed to link nvdspreprocess to pre_pgie_tee")
            # pre_pgie_tee → pre_pgie_q_main → nvinfer
            pre_tee_src0 = elements['pre_pgie_tee'].get_request_pad("src_0")
            if not pre_tee_src0:
                raise RuntimeError("Failed to request pre_pgie_tee src_0 pad")
            pre_main_sink = elements['pre_pgie_q_main'].get_static_pad("sink")
            if not pre_main_sink:
                raise RuntimeError("Failed to get pre_pgie_q_main sink pad")
            if pre_tee_src0.link(pre_main_sink) != Gst.PadLinkReturn.OK:
                raise RuntimeError("Failed to link pre_pgie_tee to pre_pgie_q_main")
            if not elements['pre_pgie_q_main'].link(elements['nvinfer']):
                raise RuntimeError("Failed to link pre_pgie_q_main to nvinfer")
            # Removed unstable Python probe on nvinfer src (caused occasional segfaults)
            
            if not elements['nvinfer'].link(elements['nvdsroiexclude']):
                raise RuntimeError("Failed to link nvinfer to nvdsroiexclude")
            if not elements['nvdsroiexclude'].link(elements['q_before_tracker']):
                raise RuntimeError("Failed to link nvdsroiexclude to q_before_tracker")
            if not elements['q_before_tracker'].link(elements['nvtracker']): 
                raise RuntimeError("Failed to link q_before_tracker to nvtracker")
            if not elements['nvtracker'].link(elements['nvdsanalytics_post']): 
                raise RuntimeError("Failed to link nvtracker to nvdsanalytics_post")
            # Post-analytics fanout is handled later via main_tee

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
            nvdspreprocess = Gst.ElementFactory.make("nvdspreprocess", "nvdspreprocess")
            # Pre-PGIE tee (for MapAnything branch)
            pre_pgie_tee = Gst.ElementFactory.make("tee", "pre_pgie_tee")
            if pre_pgie_tee:
                try:
                    pre_pgie_tee.set_property("allow-not-linked", True)
                except Exception:
                    pass
            # Queues right after tee to isolate backpressure
            pre_pgie_q_main = Gst.ElementFactory.make("queue", "pre_pgie_q_main")
            pre_pgie_q_ma = Gst.ElementFactory.make("queue", "pre_pgie_q_ma")
            if pre_pgie_q_ma:
                try:
                    pre_pgie_q_ma.set_property("leaky", 2)  # downstream
                    pre_pgie_q_ma.set_property("max-size-buffers", 1)
                    pre_pgie_q_ma.set_property("max-size-bytes", 0)
                    pre_pgie_q_ma.set_property("max-size-time", 0)
                except Exception:
                    pass
            nvinfer = Gst.ElementFactory.make("nvinfer", "nvinfer")
            nvdsroiexclude = Gst.ElementFactory.make("nvdsroiexclude", "nvdsroiexclude")

            # Analytics and Tracking
            nvtracker = Gst.ElementFactory.make("nvtracker", "nvtracker")
            nvdsanalytics_post = Gst.ElementFactory.make("nvdsanalytics", "nvdsanalytics_post")
            # Decoupling queue between analytics and post-analytics tee to avoid fanout backpressure
            analytics_fanout_q = Gst.ElementFactory.make("queue", "analytics_fanout_q")
            post_analytics_tee = Gst.ElementFactory.make("tee", "post_analytics_tee")
            if post_analytics_tee:
                try:
                    post_analytics_tee.set_property("allow-not-linked", True)
                except Exception:
                    pass

            # Split to mosaic sinks (JPEG/EGL) after OSD via a tee
            main_tee = Gst.ElementFactory.make("tee", "main_tee")
            if main_tee:
                try:
                    main_tee.set_property("allow-not-linked", True)
                except Exception:
                    pass
            # MapAnything demux (splits pre-PGIE batched buffers)
            ma_demux = Gst.ElementFactory.make("nvstreamdemux", "ma_demux")
            # Post-analytics BEV demux
            bev_demux = Gst.ElementFactory.make("nvstreamdemux", "bev_demux")

            # Mosaic branch elements (GPU tiler → RGBA → OSD → Tee → JPEG/EGL)
            mosaic_q = Gst.ElementFactory.make("queue", "mosaic_q")
            mosaic_tiler = Gst.ElementFactory.make("nvmultistreamtiler", "mosaic_tiler")
            mosaic_conv_pre = Gst.ElementFactory.make("nvvideoconvert", "mosaic_conv_pre")
            mosaic_caps_rgba = Gst.ElementFactory.make("capsfilter", "mosaic_caps_rgba")
            mosaic_osd = Gst.ElementFactory.make("nvdsosd", "mosaic_osd")
            # Single post-OSD convert to I420 for nvjpegenc
            mosaic_conv_post = Gst.ElementFactory.make("nvvideoconvert", "mosaic_conv_post")
            mosaic_caps = Gst.ElementFactory.make("capsfilter", "mosaic_caps")
            mosaic_enc = Gst.ElementFactory.make("nvjpegenc", "mosaic_enc")
            mosaic_sink = Gst.ElementFactory.make("appsink", "mosaic_sink")

            # Optional EGL visualization branch
            enable_egl = bool(self.config.visualization.ENABLE_EGL)
            egl_q = None
            egl_sink = None
            if enable_egl:
                egl_q = Gst.ElementFactory.make("queue", "egl_q")
                egl_sink = Gst.ElementFactory.make("nveglglessink", "egl_sink")

            # Queues for pipeline robustness
            q_before_tracker = Gst.ElementFactory.make("queue", "q_before_tracker")

            # Add a queue for JPEG branch to decouple tee from encoder
            jpeg_q = Gst.ElementFactory.make("queue", "jpeg_q")

            # Configure queues that sit immediately downstream of tees to be tiny & leaky
            try:
                if mosaic_q:
                    mosaic_q.set_property("leaky", 2)
                    mosaic_q.set_property("max-size-buffers", 1)
                    mosaic_q.set_property("max-size-bytes", 0)
                    mosaic_q.set_property("max-size-time", 0)
            except Exception:
                pass
            try:
                if analytics_fanout_q:
                    analytics_fanout_q.set_property("leaky", 2)
                    analytics_fanout_q.set_property("max-size-buffers", 1)
                    analytics_fanout_q.set_property("max-size-bytes", 0)
                    analytics_fanout_q.set_property("max-size-time", 0)
            except Exception:
                pass
            try:
                if jpeg_q:
                    jpeg_q.set_property("leaky", 2)
                    jpeg_q.set_property("max-size-buffers", 1)
                    jpeg_q.set_property("max-size-bytes", 0)
                    jpeg_q.set_property("max-size-time", 0)
            except Exception:
                pass

            # Validate element creation and assemble elements to add
            element_list = [
                multiurisrc, nvdspreprocess, pre_pgie_tee, pre_pgie_q_main, pre_pgie_q_ma, nvinfer, nvdsroiexclude, nvtracker, nvdsanalytics_post,
                analytics_fanout_q, post_analytics_tee, main_tee, q_before_tracker,
                mosaic_q, mosaic_tiler, mosaic_conv_pre, mosaic_caps_rgba, mosaic_osd,
                jpeg_q, mosaic_conv_post, mosaic_caps, mosaic_enc, mosaic_sink
            ]
            # Include MA demux now; per-stream branches will be added in a helper
            element_list.extend([ma_demux, bev_demux])
            if enable_egl:
                element_list.extend([egl_q, egl_sink])
            
            if not all(element_list):
                # Build the corresponding names list for created elements
                element_names = []
                if multiurisrc:
                    element_names.append("nvmultiurisrcbin")
                if nvdspreprocess:
                    element_names.append("nvdspreprocess")
                if nvinfer:
                    element_names.append("nvinfer")
                element_names.extend([
                    "nvstreamdemux",
                    "nvdsroiexclude",
                    "nvtracker", "nvdsanalytics_post", "tee",
                    "queue",
                    "q_before_tracker",
                    "mosaic_q", "nvmultistreamtiler", "nvvideoconvert", "capsfilter", "nvdsosd", "queue", "nvvideoconvert",
                    "capsfilter", "nvjpegenc", "appsink"
                ])  # type: ignore[list-item]
                if enable_egl:
                    element_names.extend(["egl_q", "nveglglessink"])
                for el, name in zip(element_list, element_names):
                    if not el:
                        self.logger.error(f"❌ Failed to create element: {name}")
                raise RuntimeError("Failed to create one or more GStreamer elements.")
            self.logger.info("✅ All GStreamer elements created successfully.")

            # Create elements dictionary for helper functions
            elements = {
                'multiurisrc': multiurisrc, 'nvdspreprocess': nvdspreprocess, 'pre_pgie_tee': pre_pgie_tee, 'pre_pgie_q_main': pre_pgie_q_main, 'pre_pgie_q_ma': pre_pgie_q_ma, 'nvinfer': nvinfer,
                'nvdsroiexclude': nvdsroiexclude,
                'nvtracker': nvtracker, 
                'nvdsanalytics_post': nvdsanalytics_post, 'analytics_fanout_q': analytics_fanout_q, 'post_analytics_tee': post_analytics_tee, 'main_tee': main_tee,
                'q_before_tracker': q_before_tracker,
                'mosaic_q': mosaic_q, 'mosaic_tiler': mosaic_tiler, 'mosaic_conv_pre': mosaic_conv_pre, 'mosaic_caps_rgba': mosaic_caps_rgba, 'mosaic_osd': mosaic_osd, 'jpeg_q': jpeg_q, 'mosaic_conv_post': mosaic_conv_post,
                'mosaic_caps': mosaic_caps, 'mosaic_enc': mosaic_enc, 'mosaic_sink': mosaic_sink,
                'egl_q': egl_q, 'egl_sink': egl_sink,
                'ma_demux': ma_demux, 'bev_demux': bev_demux,
                }

            # --- Phase B: Configure Elements ---
            if not self._configure_elements(elements):
                raise RuntimeError("Failed to configure pipeline elements")

            # --- Phase C: Add Elements to Pipeline ---
            self.logger.info("------------- Adding Elements to Pipeline-------------")
            for el in element_list:
                self.pipeline.add(el)

            # Configure mosaic branch properties
            try:
                base_width = max(1, int(self.max_width))
                base_height = max(1, int(self.max_height))
                mosaic_tiler.set_property("rows", 2)
                mosaic_tiler.set_property("columns", 2)
                mosaic_tiler.set_property("width", base_width * 2)
                mosaic_tiler.set_property("height", base_height * 2)
                self.logger.info(
                    f"🎛️ Configured mosaic tiler for 2x2 grid at {base_width * 2}x{base_height * 2}"
                )
            except Exception as exc:
                raise RuntimeError(f"Failed to configure tiler grid/resolution: {exc}") from exc
            try:
                mosaic_caps_rgba.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
            except Exception as exc:
                raise RuntimeError(f"Failed to set RGBA caps on mosaic_caps_rgba: {exc}") from exc
            try:
                mosaic_caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))
            except Exception as exc:
                raise RuntimeError(f"Failed to set I420 caps on mosaic_caps: {exc}") from exc
            try:
                mosaic_enc.set_property("quality", int(getattr(self.config.visualization, 'JPEG_QUALITY', 85)))
            except Exception as exc:
                raise RuntimeError(f"Failed to configure nvjpegenc properties: {exc}") from exc
            mosaic_sink.set_property("emit-signals", True)
            mosaic_sink.set_property("drop", True)
            mosaic_sink.set_property("sync", False)
            mosaic_sink.set_property("max-buffers", 1)
            mosaic_sink.connect("new-sample", self._on_new_mosaic_sample)
            if enable_egl and egl_sink:
                egl_sink.set_property("sync", False)
                egl_sink.set_property("qos", False)

            # --- Phase D: Setup Probes ---
            self.logger.info("------------- Setting Up Buffer Probes-------------")
            
            # Telemetry Probe for metadata extraction (attach unconditionally; probe handles DS7/DS8 differences)
            _pyds_ok = hasattr(pyds, 'gst_buffer_get_nvds_batch_meta')
            self._pyds_ok = bool(_pyds_ok)

            analytics_src_pad = nvdsanalytics_post.get_static_pad("src")
            if not analytics_src_pad:
                raise RuntimeError("Failed to get nvdsanalytics_post source pad")
            disable_analytics_probe = str(os.environ.get("NOESIS_DISABLE_ANALYTICS_PROBE", "0")).strip().lower() in {"1","true","yes","y"}
            if disable_analytics_probe:
                self.logger.warning("Skipping analytics telemetry probe due to NOESIS_DISABLE_ANALYTICS_PROBE")
            else:
                analytics_src_pad.add_probe(Gst.PadProbeType.BUFFER, self._analytics_probe, 0)
                self.logger.info("✅ Added buffer probe to nvdsanalytics_post source pad for telemetry extraction")
            
            # Per-branch OSD probe will be attached in per-stream branches

            # (Removed noisy mux src probe)
            # --- Phase E: Link Main Pipeline Chain ---
            if not self._build_main_pipeline_chain(elements):
                raise RuntimeError("Failed to link main pipeline chain")

            # Store references early for downstream setup that accesses them
            self.multiurisrc, self.nvinfer, self.nvtracker = multiurisrc, nvinfer, nvtracker
            self.nvdsanalytics_post = nvdsanalytics_post
            self.main_tee = main_tee
            self._post_analytics_tee = post_analytics_tee
            # Store mosaic elements for diagnostics (caps dumps)
            self.mosaic_osd = mosaic_osd
            self.mosaic_enc = mosaic_enc
            #self.pipeline.add(main_tee)
            self.logger.info("main_tee added to pipeline post-OSD")

            # Build MapAnything on-demand branch (pre-PGIE demux → per-stream BGR appsinks)
            _disable_ma = False
            try:
                import os as _os
                _disable_ma = str(_os.environ.get("NOESIS_DISABLE_MA_BRANCH", "0")).strip().lower() in {"1","true","yes","y"}
            except Exception:
                _disable_ma = False
            if _disable_ma:
                self._ma_branch_ready = False
                self.logger.warning("⏭️ Skipping MapAnything pre-PGIE branch due to NOESIS_DISABLE_MA_BRANCH")
            else:
                try:
                    self._setup_mapanything_branch(elements)
                    self._ma_branch_ready = True
                    self._ma_initial_open = True
                    self._ma_first_frames_seen = {sid: 0 for sid in self.sensor_ids}
                    self._ma_safety_close_scheduled = False
                    self.logger.info("✅ MapAnything pre-PGIE branch constructed (valves opened for negotiation, auto-close after startup)")
                except Exception as exc:
                    self._ma_branch_ready = False
                    self.logger.warning(f"MapAnything branch unavailable: {exc}")

            # Link post-analytics tee to mosaic branch FIRST to ensure upstream caps settle
            # Insert a decoupling queue between analytics and tee to avoid tee fanout stalling upstream
            analytics_fanout_q = elements.get('analytics_fanout_q')
            if analytics_fanout_q is None:
                raise RuntimeError("analytics_fanout_q missing; cannot build analytics fanout")
            if not nvdsanalytics_post.link(analytics_fanout_q):
                raise RuntimeError("Failed to link nvdsanalytics_post to analytics_fanout_q")
            if not analytics_fanout_q.link(post_analytics_tee):
                raise RuntimeError("Failed to link analytics_fanout_q to post_analytics_tee")
            post_analytics_mosaic_pad = post_analytics_tee.get_request_pad("src_0")
            if not post_analytics_mosaic_pad:
                raise RuntimeError("Failed to request post_analytics_tee src_0 pad")
            mosaic_q_sink = mosaic_q.get_static_pad("sink")
            if not mosaic_q_sink:
                raise RuntimeError("Failed to get mosaic_q sink pad")
            if post_analytics_mosaic_pad.link(mosaic_q_sink) != Gst.PadLinkReturn.OK:
                raise RuntimeError("Failed to link post_analytics_tee to mosaic_q")
            # Observe mosaic flow from tee
            try:
                self._attach_flow_probe(post_analytics_mosaic_pad, "post_analytics_tee.src_0 (MOSAIC)")
            except Exception:
                pass
            if not mosaic_q.link(mosaic_tiler):
                raise RuntimeError("Failed to link mosaic_q to mosaic_tiler")
            if not mosaic_tiler.link(mosaic_conv_pre):
                raise RuntimeError("Failed to link mosaic_tiler to mosaic_conv_pre")
            if not mosaic_conv_pre.link(mosaic_caps_rgba):
                raise RuntimeError("Failed to link mosaic_conv_pre to RGBA caps")
            if not mosaic_caps_rgba.link(mosaic_osd):
                raise RuntimeError("Failed to link mosaic_caps_rgba to mosaic_osd")
            # If EGL is enabled, force OSD to GPU mode to ensure NVMM RGBA for nveglglessink
            try:
                if enable_egl and mosaic_osd:
                    mosaic_osd.set_property("process-mode", 0)  # GPU mode
                    self.logger.info("OSD set to GPU mode for EGL compatibility")
            except Exception:
                pass

            # Tee immediately after OSD so EGL can consume RGBA directly; JPEG branch converts to I420
            if not mosaic_osd.link(main_tee):
                raise RuntimeError("Failed to link mosaic_osd to main_tee")

            # Now build BEV post-analytics branch (per-stream CPU BGR appsinks) off src_1
            _disable_bev = False
            try:
                import os as _os
                _disable_bev = str(_os.environ.get("NOESIS_DISABLE_BEV", "0")).strip().lower() in {"1","true","yes","y"}
            except Exception:
                _disable_bev = False
            if _disable_bev:
                self._bev_branch_ready = False
                self.logger.warning("⏭️ Skipping BEV branch due to NOESIS_DISABLE_BEV")
            else:
                try:
                    self._setup_bev_branch(elements)
                    self._bev_branch_ready = True
                    self.logger.info("✅ BEV branch constructed (post-analytics tee → per-stream BGR appsinks)")
                except Exception as exc:
                    self._bev_branch_ready = False
                    self.logger.warning(f"BEV branch unavailable: {exc}")
            self.logger.info("✅ Post-OSD tee configured for shared mosaic outputs")

            # JPEG branch: tee → jpeg_q → nvvideoconvert → I420 caps → nvjpegenc → appsink
            jpeg_pad = main_tee.get_request_pad("src_0")
            if not jpeg_pad:
                raise RuntimeError("Failed to request tee src pad for JPEG branch")
            jpeg_q_sink_pad = jpeg_q.get_static_pad("sink")
            if not jpeg_q_sink_pad:
                raise RuntimeError("Failed to get jpeg_q sink pad")
            if jpeg_pad.link(jpeg_q_sink_pad) != Gst.PadLinkReturn.OK:
                raise RuntimeError("Failed to link post-OSD tee to jpeg_q (JPEG branch)")
            if not jpeg_q.link(mosaic_conv_post):
                raise RuntimeError("Failed to link jpeg_q to nvvideoconvert (JPEG branch)")
            if not mosaic_conv_post.link(mosaic_caps):
                raise RuntimeError("Failed to link mosaic_conv_post to I420 caps (JPEG branch)")
            if not mosaic_caps.link(mosaic_enc):
                raise RuntimeError("Failed to link I420 caps to nvjpegenc (JPEG branch)")
            if not mosaic_enc.link(mosaic_sink):
                raise RuntimeError("Failed to link nvjpegenc to mosaic appsink")
            self.logger.info("✅ JPEG branch: post-OSD tee → jpeg_q → nvvideoconvert → I420 caps → nvjpegenc → appsink")

            # Attach OSD probe unconditionally; probe handles missing API gracefully
            mosaic_osd_sink_pad = mosaic_osd.get_static_pad("sink")
            if not mosaic_osd_sink_pad:
                self.logger.warning("Failed to get mosaic_osd sink pad")
            if mosaic_osd_sink_pad:
                if str(os.environ.get("NOESIS_DISABLE_MOSAIC_OSD_PROBE", "0")).strip().lower() in {"1","true","yes","y"}:
                    self.logger.warning("Skipping mosaic OSD probe due to NOESIS_DISABLE_MOSAIC_OSD_PROBE")
                else:
                    mosaic_osd_sink_pad.add_probe(
                        Gst.PadProbeType.BUFFER, self._mosaic_osd_probe, None
                    )
                    self.logger.info("✅ Attached mosaic OSD probe")
            egl_branch_enabled = bool(enable_egl and egl_q and egl_sink)
            if egl_branch_enabled:
                tee_src_pad = main_tee.get_request_pad("src_1")
                if not tee_src_pad:
                    raise RuntimeError("Failed to request tee src pad for EGL branch")
                egl_q_sink_pad = egl_q.get_static_pad("sink")
                if not egl_q_sink_pad:
                    raise RuntimeError("Failed to get egl_q sink pad")
                if tee_src_pad.link(egl_q_sink_pad) != Gst.PadLinkReturn.OK:
                    raise RuntimeError("Failed to link tee to egl queue")
                if not egl_q.link(egl_sink):
                    raise RuntimeError("Failed to link egl queue to egl sink")
                self.logger.info("✅ EGL branch: post-OSD tee → egl queue → nveglglessink (RGBA)")
                self._egl_tee_pad = tee_src_pad
                self.egl_queue = egl_q
                self.egl_sink = egl_sink
                self.egl_enabled = True
            else:
                self.logger.info("ℹ️ EGL branch disabled via config")
                self._egl_tee_pad = None
                self.egl_queue = None
                self.egl_sink = None
                self.egl_enabled = False

            # Attach flow probes for critical pads to aid debugging
            self._attach_flow_probe(multiurisrc.get_static_pad("src"), "nvmultiurisrcbin.src")
            self._attach_flow_probe(nvinfer.get_static_pad("sink"), "nvinfer.sink")
            self._attach_flow_probe(nvinfer.get_static_pad("src"), "nvinfer.src")
            self._attach_flow_probe(nvdsroiexclude.get_static_pad("sink"), "nvdsroiexclude.sink")
            self._attach_flow_probe(nvdsroiexclude.get_static_pad("src"), "nvdsroiexclude.src")
            self._attach_flow_probe(nvtracker.get_static_pad("sink"), "nvtracker.sink")
            self._attach_flow_probe(nvtracker.get_static_pad("src"), "nvtracker.src")
            self._attach_flow_probe(nvdsanalytics_post.get_static_pad("sink"), "nvdsanalytics_post.sink")
            self._attach_flow_probe(nvdsanalytics_post.get_static_pad("src"), "nvdsanalytics_post.src")
            try:
                _afq = elements.get('analytics_fanout_q')
                if _afq:
                    self._attach_flow_probe(_afq.get_static_pad("src"), "analytics_fanout_q.src")
            except Exception:
                pass
            if post_analytics_tee:
                self._attach_flow_probe(post_analytics_tee.get_static_pad("sink"), "post_analytics_tee.sink")
            self._attach_flow_probe(mosaic_q.get_static_pad("src"), "mosaic_q.src")
            self._attach_flow_probe(mosaic_tiler.get_static_pad("sink"), "mosaic_tiler.sink")
            self._attach_flow_probe(mosaic_tiler.get_static_pad("src"), "mosaic_tiler.src")
            self._attach_flow_probe(main_tee.get_static_pad("sink"), "main_tee.sink")
            self._attach_flow_probe(jpeg_q.get_static_pad("src"), "jpeg_q.src")
            self._attach_flow_probe(mosaic_enc.get_static_pad("sink"), "mosaic_enc.sink")
            if egl_branch_enabled and egl_q and egl_sink:
                self._attach_flow_probe(egl_q.get_static_pad("src"), "egl_q.src")
                self._attach_flow_probe(egl_sink.get_static_pad("sink"), "egl_sink.sink")
            # Add probes for tee branches
            self._attach_flow_probe(jpeg_pad, "main_tee.src_0 (JPEG)")
            if egl_branch_enabled:
                self._attach_flow_probe(tee_src_pad, "main_tee.src_1 (EGL)")

            # --- Finalization ---
            self.logger.info("✅ Pipeline construction complete.")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create pipeline: {e}")
            import traceback
            self.logger.error(f"Full Traceback: {traceback.format_exc()}")
            try:
                self._errors.append(f"pipeline build error: {e}")
            except Exception as e:
                self.logger.debug(f"motion state update failed for tid={obj_id} on sensor={sensor_id}: {e}")
            return False
    







    
    def _analytics_probe(self, pad, info, user_data):
        """Probe to extract telemetry data after analytics."""
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        operator = meta_ops.create_operator(gst_buffer)
        frames = meta_ops.iter_frames(operator, gst_buffer)
        self.logger.debug("analytics_probe: %s frame(s) returned", len(frames))
        for frame_meta in frames:
            try:
                detections = self._parse_obj_meta(operator, gst_buffer, frame_meta)
                try:
                    ds_index = meta_ops.get_source_id(operator, frame_meta)
                except Exception as e:
                    self.logger.debug(f"analytics_probe: get_source_id failed: {e}")
                    ds_index = -1
                self.logger.debug(f"analytics_probe: source={ds_index} objects={len(detections)}")
            except Exception as e:
                self.logger.exception("Failed to parse analytics metadata for frame")
        # Note: Per-frame telemetry broadcast disabled to avoid client overload.
        # Periodic stats broadcast (1 Hz) remains enabled via WebSocketServer.
        return Gst.PadProbeReturn.OK


    def _nvinfer_object_debug_probe(self, _pad, info, _user_data):
        """Temporary instrumentation to inspect PGIE object metadata."""
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        operator = meta_ops.create_operator(gst_buffer)
        frames = meta_ops.iter_frames(operator, gst_buffer)
        total_objects = 0
        for frame_meta in frames:
            objs = meta_ops.iter_objects(operator, frame_meta)
            count = len(objs)
            total_objects += count
            try:
                source_idx = meta_ops.get_source_id(operator, frame_meta)
            except Exception as exc:
                self.logger.debug(f"nvinfer_probe: get_source_id failed: {exc}")
                source_idx = int(getattr(frame_meta, "source_id", -1) or -1)
            self.logger.debug("nvinfer_probe: source=%s objects=%s", source_idx, count)

        if frames:
            self.logger.debug("nvinfer_probe: total_objects=%s frames=%s", total_objects, len(frames))

        return Gst.PadProbeReturn.OK


    def _extract_analytics_frame_meta(self, operator: Optional[Any], frame_meta: Any) -> Optional[Dict[str, Any]]:
        """Extract analytics frame metadata"""
        try:
            target_type = meta_ops.analytics_frame_meta_type()
            for user_meta in meta_ops.iter_user_meta(operator, frame_meta, "frame"):
                meta_type = meta_ops.get_meta_type(operator, user_meta)
                if target_type is not None and meta_type != target_type:
                    continue
                analytics_frame_meta = meta_ops.cast_analytics_frame_meta(operator, user_meta)
                if analytics_frame_meta is None:
                    continue
                return {
                    'objects_in_roi': getattr(analytics_frame_meta, 'objInROIcnt', {}),
                    'line_crossing_cumulative': getattr(analytics_frame_meta, 'objLCCumCnt', {}),
                    'line_crossing_current': getattr(analytics_frame_meta, 'objLCCurrCnt', {}),
                    'overcrowding_status': getattr(analytics_frame_meta, 'ocStatus', None),
                }
        except Exception as e:
            self.logger.debug(f"Error extracting analytics frame meta: {e}")
        return None
    
    def _extract_analytics_obj_meta(self, operator: Optional[Any], obj_meta: Any) -> Optional[Dict[str, Any]]:
        """Extract analytics object metadata and normalize key names"""
        try:
            target_type = meta_ops.analytics_obj_meta_type()
            for user_meta in meta_ops.iter_user_meta(operator, obj_meta, "object"):
                meta_type = meta_ops.get_meta_type(operator, user_meta)
                if target_type is not None and meta_type != target_type:
                    continue
                analytics_obj_meta = meta_ops.cast_analytics_obj_info(operator, user_meta)
                if analytics_obj_meta is None:
                    continue
                dir_status = getattr(analytics_obj_meta, 'dirStatus', None)
                lc_status = getattr(analytics_obj_meta, 'lcStatus', None)
                oc_status = getattr(analytics_obj_meta, 'ocStatus', None)
                roi_status = getattr(analytics_obj_meta, 'roiStatus', None)
                # Normalize to DeepStream SDK key casing so downstream logic works
                return {
                    'dirStatus': dir_status,
                    'lcStatus': lc_status,
                    'ocStatus': oc_status,
                    'roiStatus': roi_status,
                    # Retain legacy snake_case keys for backward compatibility
                    'direction_status': dir_status,
                    'line_crossing_status': lc_status,
                    'overcrowding_status': oc_status,
                    'roi_status': roi_status,
                }
        except Exception as e:
            self.logger.debug(f"Error extracting analytics object meta: {e}")
        return None



    def _parse_obj_meta(
        self,
        operator: Optional[Any],
        _gst_buffer: Any,
        frame_meta: Any,
    ) -> List[Dict[str, Any]]:
        """Return list(dict) with keys class_id, confidence, bbox, object_id, and analytics data."""
        detections: List[Dict[str, Any]] = []
        active_tracks: List[Dict[str, Any]] = []
        occupancy: Dict[str, int] = {}
        transitions: List[Dict[str, Any]] = []

        ds_index = meta_ops.get_source_id(operator, frame_meta)
        sensor_id = self.sensor_id_by_source_idx.get(ds_index, ds_index)

        try:
            prev_occupancy = dict(self.live_tracking_state.get(sensor_id, {}).get('occupancy', {}))
        except Exception:
            prev_occupancy = {}

        decoded_frame_bgr = None
        now_ts = time.time()
        present_ds_ids: List[int] = []
        frame_analytics = self._extract_analytics_frame_meta(operator, frame_meta)

        for obj_meta in meta_ops.iter_objects(operator, frame_meta):
            rect = meta_ops.get_rect_params(operator, obj_meta)
            if rect is None:
                continue

            left = float(getattr(rect, "left", 0.0))
            top = float(getattr(rect, "top", 0.0))
            width = float(getattr(rect, "width", 0.0))
            height = float(getattr(rect, "height", 0.0))
            bbox = [left, top, width, height]


            tid = meta_ops.get_object_id(operator, obj_meta)
            if tid == -1:
                continue
            # Removed verbose confidence logging - confidence values are now properly normalized
                        

            obj_id = meta_ops.get_object_id(operator, obj_meta)
            class_id = meta_ops.get_class_id(operator, obj_meta)
            confidence = meta_ops.get_confidence(operator, obj_meta)
            tracker_conf = meta_ops.get_tracker_confidence(operator, obj_meta)

            detection = {
                "class_id": class_id,
                "confidence": confidence,
                "bbox": bbox,
                "object_id": obj_id,
            }

            track_dict: Dict[str, Any] = {
                'track_id': obj_id,
                'camera_id': f"camera_{sensor_id}",
                'confidence': confidence,
                'bbox': bbox,
                'class_id': class_id,
            }

            center_x = left + width / 2.0
            center_y = top + height / 2.0
            track_dict['center'] = [center_x, center_y]

            try:
                motion_state = self.track_motion_state_by_sensor.setdefault(sensor_id, {}).setdefault(
                    obj_id,
                    {'last_center': None, 'last_ts': None, 'vel_hist': deque(maxlen=5)},
                )
                last_center = motion_state.get('last_center')
                last_ts = motion_state.get('last_ts')
                if last_center is not None and last_ts is not None:
                    dt = max(1e-3, now_ts - float(last_ts))
                    vx = (center_x - float(last_center[0])) / dt
                    vy = (center_y - float(last_center[1])) / dt
                    motion_state['vel_hist'].append((vx, vy))
                    if motion_state['vel_hist']:
                        hvx = sum(v[0] for v in motion_state['vel_hist']) / len(motion_state['vel_hist'])
                        hvy = sum(v[1] for v in motion_state['vel_hist']) / len(motion_state['vel_hist'])
                        track_dict['velocity'] = [hvx, hvy]
                motion_state['last_center'] = [center_x, center_y]
                motion_state['last_ts'] = now_ts
            except Exception:
                pass

            if tracker_conf is not None:
                track_dict['tracker_confidence'] = tracker_conf

            analytics_data = self._extract_analytics_obj_meta(operator, obj_meta)
            if analytics_data:
                detection["analytics"] = analytics_data

                roi_status = analytics_data.get('roiStatus')
                in_zones: List[str] = []

                def bump(zone: Any) -> None:
                    zone_norm = str(zone).strip()
                    if zone_norm:
                        occupancy[zone_norm] = occupancy.get(zone_norm, 0) + 1
                        in_zones.append(zone_norm)

                if isinstance(roi_status, dict):
                    for zone, status in roi_status.items():
                        if status in (1, True, "IN", "inside", "in", "INROI"):
                            bump(zone)
                elif isinstance(roi_status, (list, tuple, set)):
                    for zone in roi_status:
                        bump(zone)
                elif isinstance(roi_status, str):
                    for zone in roi_status.split(','):
                        bump(zone)

                current_zone = in_zones[0] if in_zones else None
                if current_zone:
                    track_dict['zone'] = current_zone
                    try:
                        zone_state = self.track_zone_state_by_sensor.setdefault(sensor_id, {}).setdefault(
                            obj_id,
                            {'current_zone': None, 'entry_time': None},
                        )
                        prev_zone = zone_state.get('current_zone')
                        entry_time = zone_state.get('entry_time')
                        if prev_zone == current_zone:
                            if entry_time is None:
                                zone_state['entry_time'] = now_ts
                                entry_time = now_ts
                            dwell = max(0.0, now_ts - float(entry_time))
                            track_dict['dwell_time'] = dwell
                        else:
                            if prev_zone and prev_zone != current_zone:
                                transitions.append({
                                    'track_id': obj_id,
                                    'camera_id': f"camera_{sensor_id}",
                                    'from_zone': prev_zone,
                                    'to_zone': current_zone,
                                    'timestamp': now_ts,
                                })
                            zone_state['current_zone'] = current_zone
                            zone_state['entry_time'] = now_ts
                            track_dict['dwell_time'] = 0.0
                    except Exception as e:
                        self.logger.debug(f"zone state update failed for tid={obj_id} on sensor={sensor_id}: {e}")

                lc_status = analytics_data.get('lcStatus')
                if isinstance(lc_status, dict):
                    for line_name, status in lc_status.items():
                        if status == 1:
                            transitions.append({
                                'track_id': obj_id,
                                'camera_id': f"camera_{sensor_id}",
                                'line_name': line_name,
                                'timestamp': time.time(),
                            })

            active_tracks.append(track_dict)

            try:
                if 'zone' not in track_dict or not track_dict['zone']:
                    cam_info = self.source_info.get(sensor_id, {})
                    fallback_zone = cam_info.get('clean_name') or cam_info.get('name')
                    if fallback_zone:
                        track_dict['zone'] = fallback_zone
                        zone_state = self.track_zone_state_by_sensor.setdefault(sensor_id, {}).setdefault(
                            obj_id,
                            {'current_zone': None, 'entry_time': None},
                        )
                        prev_zone = zone_state.get('current_zone')
                        entry_time = zone_state.get('entry_time')
                        if prev_zone == fallback_zone:
                            if entry_time is None:
                                zone_state['entry_time'] = now_ts
                                entry_time = now_ts
                            dwell = max(0.0, now_ts - float(entry_time))
                            track_dict['dwell_time'] = dwell
                        else:
                            zone_state['current_zone'] = fallback_zone
                            zone_state['entry_time'] = now_ts
                            track_dict['dwell_time'] = 0.0
            except Exception as e:
                self.logger.debug(f"fallback zone update failed for tid={obj_id} on sensor={sensor_id}: {e}")

            try:
                if self.reid_enabled and (self.stable_id_mgr is not None) and int(class_id) == 0:
                    bbox_tuple = (left, top, width, height)
                    zone_name = track_dict.get('zone') if isinstance(track_dict, dict) else None
                    # No per-stream decode path available yet; pass None so StableIDManager can short-circuit crops.
                    stable_id = self.stable_id_mgr.update(
                        sensor_id=int(sensor_id),
                        ds_obj_id=int(obj_id),
                        bbox_ltrbwh=bbox_tuple,
                        ts=float(now_ts),
                        zone=str(zone_name) if zone_name else None,
                        frame_bgr=decoded_frame_bgr,
                    )
                    track_dict['stable_id'] = int(stable_id)
                else:
                    track_dict['stable_id'] = None
            except Exception as e:
                self.logger.debug(f"stable_id update failed for tid={obj_id} on sensor={sensor_id}: {e}")
                try:
                    track_dict['stable_id'] = None
                except Exception as e2:
                    self.logger.debug(f"failed to set fallback stable_id=None for tid={obj_id}: {e2}")

            detections.append(detection)
            try:
                present_ds_ids.append(int(obj_id))
            except Exception as e:
                self.logger.debug(f"failed to append present_ds_id for tid={obj_id}: {e}")

        if not occupancy and frame_analytics and isinstance(frame_analytics.get('objects_in_roi'), dict):
            try:
                occupancy.update(frame_analytics['objects_in_roi'])
            except Exception as e:
                self.logger.debug(f"failed to merge frame analytics occupancy: {e}")

        try:
            publisher = getattr(self, 'occupancy_publisher', None)
            if publisher is not None and occupancy is not None:
                first_flag = getattr(self, '_occ_pub_first', True)
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
                            except Exception as e:
                                self.logger.debug(f"occupancy first-print failed: {e}")
                    except Exception as e:
                        self.logger.debug(f"occupancy publish failed for zone={zone}: {e}")
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
                            except Exception as e:
                                self.logger.debug(f"occupancy first-print (vacate) failed: {e}")
                    except Exception as e:
                        self.logger.debug(f"occupancy publish (vacate) failed for zone={zone}: {e}")
                if first_flag:
                    try:
                        self._occ_pub_first = False
                    except Exception as e:
                        self.logger.debug(f"failed to set _occ_pub_first flag: {e}")
        except Exception as e:
            self.logger.debug(f"occupancy publishing failed: {e}")

        if self.reid_enabled and (self.stable_id_mgr is not None):
            try:
                self.stable_id_mgr.remove_missing_tracks(int(sensor_id), present_ds_ids, float(now_ts))
                self.stable_id_mgr.prune_ghosts(now_ts)
            except Exception as e:
                self.logger.debug(f"stable_id_mgr maintenance failed for sensor={sensor_id}: {e}")

        if sensor_id in self.live_tracking_state:
            self.live_tracking_state[sensor_id]['active_tracks'] = active_tracks
            self.live_tracking_state[sensor_id]['occupancy'] = occupancy
            self.live_tracking_state[sensor_id]['transitions'].extend(transitions)
            if len(self.live_tracking_state[sensor_id]['transitions']) > 100:
                self.live_tracking_state[sensor_id]['transitions'] = self.live_tracking_state[sensor_id]['transitions'][-100:]
        else:
            self.logger.warning(f"⚠️ Unknown source_id {ds_index} (mapped→{sensor_id}) in frame metadata")

        return detections

    # --- Color helpers to keep OSD boxes and trails consistent with UI legend ---
    def _hsl_to_rgb(self, h: float, s: float, lightness: float) -> Tuple[float, float, float]:
        """Convert HSL (0..360, 0..1, 0..1) to RGB floats 0..1."""
        h = h % 360.0
        s = max(0.0, min(1.0, s))
        lightness = max(0.0, min(1.0, lightness))
        c = (1.0 - abs(2.0 * lightness - 1.0)) * s
        x = c * (1.0 - abs(((h / 60.0) % 2.0) - 1.0))
        m = lightness - c / 2.0
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
    
    def _on_bus_message(self, bus, message):
        """Handle bus messages."""
        msg_type = message.type
        
        if msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            self.logger.error(f"🚨 Pipeline error: {err.message}")
            self.logger.error(f"🚨 Debug info: {debug}")
            self.logger.error(f"🚨 Error source: {message.src.get_name() if message.src else 'unknown'}")
            try:
                src_name = message.src.get_name() if message.src else ''
                # Provide actionable hint if source/decoder failed
                if any(k in src_name for k in ('nvmultiurisrcbin', 'uridecodebin', 'rtspsrc', 'decodebin')):
                    # Summarize configured sources to aid debugging
                    try:
                        sources = []
                        for sid, info in getattr(self, 'source_info', {}).items():
                            sources.append(f"[{sid}] {info.get('name','unknown')} -> {info.get('url','')}\n")
                        if sources:
                            self.logger.error(
                                "🔎 One or more sources failed to start or connect.\n"
                                "    • Verify each stream URL is reachable and producing frames.\n"
                                "    • If using ffmpeg or a local generator, ensure it is running.\n"
                                "Configured sources:\n" + "".join(sources).rstrip()
                            )
                    except Exception as sub_exc:
                        self.logger.debug(f"Failed to enumerate configured sources for error hint: {sub_exc}")
            except Exception as exc:
                self.logger.debug(f"Failed to analyze error source for hints: {exc}")
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
            try:
                old_state, new_state, pending_state = message.parse_state_changed()
            except Exception as exc:
                self.logger.debug(f"Failed to parse state-changed message: {exc}")
                return True

            src = message.src
            name = src.get_name() if hasattr(src, "get_name") else str(src)

            if src == self.pipeline:
                self.logger.info(f"🔄 Pipeline state changed: {old_state.value_nick} → {new_state.value_nick}")

            if name in getattr(self, "_state_watch_names", set()):
                self.logger.debug(
                    "state-changed: %s %s \u2192 %s (pending=%s)",
                    name,
                    getattr(old_state, "value_nick", old_state),
                    getattr(new_state, "value_nick", new_state),
                    getattr(pending_state, "value_nick", pending_state),
                )

        return True
    
    def start(self) -> bool:
        """Start the DeepStream pipeline."""
        try:
            self.logger.info("🚀 Starting DeepStream pipeline...")
            
            if not self.pipeline:
                self.logger.error("❌ Pipeline not created")
                self._errors.append("pipeline not created at start")
                return False
            
            # Set pipeline state to PLAYING
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            self.logger.info(f"Pipeline set_state returned: {ret}")
            if ret == Gst.StateChangeReturn.FAILURE:
                self._errors.append("pipeline failed to enter PLAYING")
                # Try to surface immediate bus error
                try:
                    bus = self.pipeline.get_bus()
                    msg = bus.timed_pop_filtered(2 * Gst.SECOND, Gst.MessageType.ERROR | Gst.MessageType.WARNING)
                    if msg:
                        try:
                            if msg.type == Gst.MessageType.ERROR:
                                err, dbg = msg.parse_error()
                                self.logger.error(f"🚨 Pipeline error (pre-loop): {err.message}")
                                self.logger.error(f"🚨 Debug info: {dbg}")
                                self.logger.error(f"🚨 Error source: {msg.src.get_name() if msg.src else 'unknown'}")
                            elif msg.type == Gst.MessageType.WARNING:
                                warn, dbg = msg.parse_warning()
                                self.logger.warning(f"⚠️ Pipeline warn (pre-loop): {warn.message}")
                                self.logger.warning(f"⚠️ Debug info: {dbg}")
                        except Exception:
                            pass
                except Exception:
                    pass

                # Inspect pad linking to identify not-linked spots
                try:
                    it = self.pipeline.iterate_elements()
                    while True:
                        res, el = it.next()
                        if res != Gst.IteratorResult.OK:
                            break
                        try:
                            pit = el.iterate_pads()
                        except Exception:
                            continue
                        linked = []
                        unlinked = []
                        while True:
                            pres, pad = pit.next()
                            if pres != Gst.IteratorResult.OK:
                                break
                            try:
                                if pad.is_linked():
                                    linked.append(pad.get_name())
                                else:
                                    unlinked.append(pad.get_name())
                            except Exception:
                                pass
                        if unlinked:
                            self.logger.debug(f"element {el.get_name()} has unlinked pads: {unlinked}; linked: {linked}")
                except Exception:
                    pass

                self.logger.error("❌ Failed to set pipeline to PLAYING state")
                return False

            # Start GLib mainloop early so bus callbacks can surface errors while we wait
            self.running = True
            self.mainloop_thread = threading.Thread(target=self._run_mainloop, daemon=True)
            self.mainloop_thread.start()

            # Allow the pipeline to progress toward PLAYING, then query with a bounded timeout
            time.sleep(1.5)
            change_return, state, pending = self.pipeline.get_state(5 * Gst.SECOND)
            state_name = getattr(state, "value_nick", str(state))
            pending_name = getattr(pending, "value_nick", str(pending))
            self.start_time = time.time()
            self.logger.info(
                "Pipeline state after PLAYING: %s (pending=%s, change_return=%s)",
                state_name,
                pending_name,
                change_return,
            )

            if state != Gst.State.PLAYING:
                if pending == Gst.State.PLAYING or change_return in (
                    Gst.StateChangeReturn.ASYNC,
                    Gst.StateChangeReturn.NO_PREROLL,
                ):
                    self.logger.warning(
                        "Pipeline still transitioning to PLAYING (state=%s, pending=%s)",
                        state_name,
                        pending_name,
                    )
                else:
                    self.logger.error("❌ Pipeline did not reach PLAYING state")
                    self._errors.append("pipeline did not reach PLAYING")
                    return False

            self.logger.info("✅ DeepStream pipeline started successfully")
            self._activated = True

            # During startup, keep MA valves open so each stream can emit a few frames,
            # then auto-close either per-sensor (via appsink callback) or via safety timeout.
            if self._ma_branch_ready and not self._ma_safety_close_scheduled:
                self._ma_safety_close_scheduled = True
                try:
                    # Long failsafe close (15s) if no FE connects to signal readiness
                    GLib.timeout_add(15000, self._close_ma_valves_after_startup)
                except Exception:
                    GLib.timeout_add(500, self._close_ma_valves_after_negotiation)

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
                # Dump negotiated caps on critical pads for diagnostics
                try:
                    self._dump_pad_caps(getattr(self, 'mosaic_osd', None), "src", "mosaic_osd.src")
                    self._dump_pad_caps(getattr(self, 'main_tee', None), "sink", "main_tee.sink")
                    self._dump_pad_caps(getattr(self, 'mosaic_enc', None), "sink", "mosaic_enc.sink")
                    if getattr(self, 'egl_enabled', False):
                        self._dump_pad_caps(getattr(self, 'egl_sink', None), "sink", "egl_sink.sink")
                    self._dump_pad_caps(getattr(self, '_post_analytics_tee', None), "sink", "post_analytics_tee.sink")
                    self._dump_pad_caps(getattr(self, '_bev_demux', None), "sink", "bev_demux.sink")
                    
                except Exception as exc:
                    self.logger.warning(f"Caps dump failed: {exc}")
            
            threading.Thread(target=check_pipeline_state, daemon=True).start()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error starting pipeline: {e}")
            return False

    def _run_mainloop(self):
        """Creates and runs the GLib MainLoop.
        
        This thread blocks SIGINT to prevent interruption during GPU operations,
        which can cause illegal instruction errors. Shutdown is handled via
        mainloop.quit() called from the mainloop's own context.
        """
        import signal
        # Block SIGINT in this thread to prevent interrupting GPU operations
        # The main thread will handle shutdown gracefully via mainloop.quit()
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        
        try:
            self.mainloop = GLib.MainLoop()
            self.mainloop.run()
        finally:
            # Restore signal mask when exiting
            signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})
    

    

    
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
        
        # Stop pipeline first - this stops GPU operations gracefully
        if self.pipeline:
            try:
                self.pipeline.set_state(Gst.State.NULL)
                # Wait for state change to complete
                ret = self.pipeline.get_state(timeout=Gst.CLOCK_TIME_NONE)
                if ret[0] == Gst.StateChangeReturn.FAILURE:
                    self.logger.warning("Pipeline state change to NULL failed")
            except Exception as e:
                self.logger.warning(f"Error stopping pipeline: {e}")
        
        # Stop main loop - must be called from mainloop's own thread context
        # Use GLib.idle_add to schedule quit() in the mainloop thread
        if self.mainloop:
            try:
                # Schedule quit() to run in the mainloop's context
                GLib.idle_add(self._quit_mainloop)
            except Exception as e:
                self.logger.warning(f"Error scheduling mainloop quit: {e}")
                # Fallback: try direct quit if idle_add fails
                try:
                    self.mainloop.quit()
                except Exception:
                    pass
        
        # Wait for main loop thread to finish
        if hasattr(self, 'mainloop_thread') and self.mainloop_thread.is_alive():
            self.mainloop_thread.join(timeout=3.0)
            if self.mainloop_thread.is_alive():
                self.logger.warning("Mainloop thread did not terminate within timeout")
        
        self.logger.info("DeepStream pipeline stopped")
    
    def _quit_mainloop(self):
        """Callback to quit mainloop from its own thread context."""
        if self.mainloop and self.mainloop.is_running():
            self.mainloop.quit()
        return False  # Don't call again
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        runtime = time.time() - self.start_time if self.running and self.start_time > 0 else 0
        
        # Use a lock to safely access frame_count
        with self.frame_count_lock:
            frame_count_copy = self.frame_count
            
        fps = frame_count_copy / runtime if runtime > 0 else 0
        
        stats = {
            'pipeline_type': 'deepstream',
            'running': self.running,
            'prepared': self._prepared,
            'activated': self._activated,
            'frames_processed': frame_count_copy,
            'fps': fps,
            'runtime_seconds': runtime,
            'batch_size': self.batch_size,
            'sources': len(self.sensor_ids),
            'mosaic_queue_size': self.mosaic_queue.qsize() if self.mosaic_queue else 0,
            'buffers_flowing': frame_count_copy > 0,
            'errors': list(self._errors),
            'tracking': self.live_tracking_state
        }
        # Attach ReID/SID allocator telemetry if available
        try:
            if hasattr(self, 'stable_id_mgr') and self.stable_id_mgr:
                sid_metrics = self.stable_id_mgr.get_sid_metrics()
                if sid_metrics:
                    stats['reid'] = sid_metrics
        except Exception:
            pass
        return stats

    def _configure_live_queue(self, q: Gst.Element) -> None:
        q.set_property("max-size-buffers", 12)
        q.set_property("max-size-bytes", 0)
        q.set_property("leaky", 2)  # LEAK_DOWNSTREAM

    def _attach_flow_probe(self, pad: Optional[Gst.Pad], label: str) -> None:
        """Attach a lightweight BUFFER probe that logs flow statistics periodically.

        Also emits a one-time attachment log so we can verify probes are present
        even before any buffers arrive.
        """
        if not pad:
            self.logger.debug(f"Skipping flow probe for {label}: pad unavailable")
            return

        try:
            if not hasattr(self, "_flow_probe_state"):
                self._flow_probe_state: Dict[str, Dict[str, Any]] = {}

            state = self._flow_probe_state.setdefault(
                label,
                {"count": 0, "last_log": time.time(), "last_logged_count": 0},
            )

            def _probe(_pad: Gst.Pad, _info: Gst.PadProbeInfo, user_data):
                probe_state = user_data
                probe_state["count"] += 1
                now = time.time()
                if now - probe_state["last_log"] >= 1.0:
                    delta = probe_state["count"] - probe_state["last_logged_count"]
                    self.logger.info(f"flow[{label}]: {delta} buffers in {now - probe_state['last_log']:.1f}s")
                    probe_state["last_logged_count"] = probe_state["count"]
                    probe_state["last_log"] = now
                return Gst.PadProbeReturn.OK

            pad.add_probe(Gst.PadProbeType.BUFFER, _probe, state)
            # One-time attachment confirmation
            try:
                self.logger.info(f"🔎 Flow probe attached: {label}")
            except Exception:
                pass
        except Exception as exc:
            self.logger.warning(f"Unable to attach flow probe for {label}: {exc}")

    def _dump_pad_caps(self, element: Optional[Gst.Element], pad_name: str, label: str) -> None:
        """Log current negotiated caps for a given element pad."""
        try:
            if not element:
                self.logger.info(f"caps[{label}]: element unavailable")
                return
            pad = element.get_static_pad(pad_name)
            if not pad:
                self.logger.info(f"caps[{label}]: pad '{pad_name}' unavailable")
                return
            caps = pad.get_current_caps()
            if not caps:
                self.logger.info(f"caps[{label}]: not negotiated yet")
                return
            try:
                caps_str = caps.to_string()
            except Exception:
                caps_str = str(caps)
            self.logger.info(f"caps[{label}]: {caps_str}")
        except Exception as exc:
            self.logger.info(f"caps[{label}]: error dumping caps: {exc}")

    def _on_new_mosaic_sample(self, appsink: GstApp.AppSink) -> Gst.FlowReturn:
        """Appsink callback for mosaic JPEG branch."""
        try:
            sample = appsink.emit("pull-sample")
            if not sample:
                return Gst.FlowReturn.ERROR
            buffer = sample.get_buffer()
            if not buffer:
                try:
                    sample.unref()
                except Exception as e:
                    self.logger.debug(f"sample.unref() failed on missing buffer: {e}")
                return Gst.FlowReturn.ERROR
            success, mapinfo = buffer.map(Gst.MapFlags.READ)
            if not success:
                try:
                    sample.unref()
                except Exception as e:
                    self.logger.debug(f"sample.unref() failed after map failure: {e}")
                return Gst.FlowReturn.ERROR
            try:
                data = mapinfo.data
                if data:
                    try:
                        self.mosaic_queue.put_nowait(bytes(data))
                    except queue.Full:
                        self.logger.debug("Dropped mosaic frame due to full queue")
                    # Count frames and log the first successful sample
                    try:
                        with self.frame_count_lock:
                            self.frame_count += 1
                            if self.frame_count == 1:
                                self.logger.info(f"🎞️ First mosaic sample received: {len(data)} bytes")
                    except Exception as e:
                        self.logger.warning(f"Failed to increment frame count: {e}")
            finally:
                try:
                    buffer.unmap(mapinfo)
                except Exception as e:
                    self.logger.debug(f"buffer.unmap() failed in mosaic sample: {e}")
                try:
                    sample.unref()
                except Exception as e:
                    self.logger.debug(f"sample.unref() failed in mosaic sample finally: {e}")
            return Gst.FlowReturn.OK
        except Exception as e:
            self.logger.error(f"Error in _on_new_mosaic_sample: {e}")
            return Gst.FlowReturn.ERROR

    def read_mosaic_jpeg(self, timeout: float = 0.1) -> Tuple[bool, Optional[bytes]]:
        """Return the latest mosaic JPEG bytes, if available."""
        try:
            jpeg_bytes = self.mosaic_queue.get(timeout=timeout)
            return True, jpeg_bytes
        except queue.Empty:
            return False, None

    # Dynamic add/remove stream API logic removed per project guidance.

    def toggle_trail_visualization(self, enabled: bool):
        self.set_trail_visualization(enabled)

    def set_trail_visualization(self, enabled: bool):
        """Enable or disable trail visualization in real-time."""
        self.logger.info(f"Setting trail visualization to: {enabled}")
        self.trail_visualization_enabled = enabled
        if not enabled:
            # Clear history when disabling to prevent stale trails on re-enable
            self.trail_history_by_sensor.clear()
            self.trail_last_seen_by_sensor.clear()
            self._bbox_smooth_by_sensor.clear()
            self._logged_tile_rects.clear()

    def _nvinfer_output_probe(self, _pad, info, _user_data):
        """Probe after nvinfer to verify detections are being produced."""
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK
        
        operator = meta_ops.create_operator(gst_buffer)
        frames = meta_ops.iter_frames(operator, gst_buffer)
        
        total_objects = 0
        for frame_meta in frames:
            objects = meta_ops.iter_objects(operator, frame_meta)
            total_objects += len(objects)
            if len(objects) > 0:
                for obj in objects:
                    class_id = meta_ops.get_class_id(operator, obj)
                    confidence = meta_ops.get_confidence(operator, obj)
                    #self.logger.info("🎯 nvinfer output: class_id=%d, confidence=%.3f", class_id, confidence)
        
        #if total_objects > 0:
        #    self.logger.info("✅ nvinfer output probe: %d object(s) detected across %d frame(s)", total_objects, len(frames))
        #elif len(frames) > 0:
        #    self.logger.debug("⚠️  nvinfer output probe: No objects detected in %d frame(s)", len(frames))
        
        return Gst.PadProbeReturn.OK

    def _mosaic_osd_probe(self, _pad, info, _user_data):
        """Single mosaic OSD probe that handles labels, bbox smoothing, and trail rendering."""
        # Provide a breadcrumb so we can confirm probe execution during development.
        self.logger.debug("Mosaic OSD probe fired")

        if not self.trail_visualization_enabled:
            # When disabled, purge any retained state and skip heavy processing.
            self.trail_history_by_sensor.clear()
            self.trail_last_seen_by_sensor.clear()
            self._bbox_smooth_by_sensor.clear()
            return Gst.PadProbeReturn.OK

        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        operator = meta_ops.create_operator(gst_buffer)
        try:
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        except Exception as e:
            self.logger.debug(f"Failed to get NvDsBatchMeta: {e}")
            batch_meta = None

        now = time.time()
        frames = meta_ops.iter_frames(operator, gst_buffer)
        self.logger.debug("mosaic_osd_probe: %s frame(s) available", len(frames))
        if not frames:
            return Gst.PadProbeReturn.OK
        
        # Count total objects across all frames for debugging
        total_objects = 0
        for frame_meta in frames:
            objects = meta_ops.iter_objects(operator, frame_meta)
            total_objects += len(objects)
        if total_objects > 0:
            self.logger.debug("🔍 OSD probe: Found %d object(s) across %d frame(s)", total_objects, len(frames))
        else:
            self.logger.debug("OSD probe: No objects detected in %d frame(s)", len(frames))

        trail_segments = int(getattr(self.config.visualization, "TRAIL_DRAW_SEGMENTS", 64) or 0)
        trail_segments = max(1, trail_segments)
        line_budget_max = 250
        label_budget_max = 16

        # Holdover cache for recent masks/bboxes per (sensor, track)
        # Used to reduce visible flicker when PGIE skips frames (interval>0)
        if not hasattr(self, "_mask_holdover_cache"):
            self._mask_holdover_cache = {}
        hold_ms = int(getattr(self.config.visualization, "MASK_HOLDOVER_MS", 250) or 250)
        hold_s = max(0.0, min(2.0, hold_ms / 1000.0))

        for frame_meta in frames:
            drawn_count = 0
            try:
                source_idx = meta_ops.get_source_id(operator, frame_meta)
            except Exception as e:
                self.logger.debug(f"get_source_id failed, falling back to frame_meta.source_id: {e}")
                source_idx = int(getattr(frame_meta, "source_id", 0) or 0)
            sensor_id = self.sensor_id_by_source_idx.get(source_idx, source_idx)

            frame_width = float(getattr(frame_meta, "source_frame_width", self.max_width) or self.max_width)
            frame_height = float(getattr(frame_meta, "source_frame_height", self.max_height) or self.max_height)
            if operator is not None:
                get_fw = getattr(operator, "get_frame_width", None)
                get_fh = getattr(operator, "get_frame_height", None)
                try:
                    fw = float(get_fw(frame_meta)) if callable(get_fw) else 0.0  # type: ignore[arg-type]
                    if fw > 0:
                        frame_width = fw
                except Exception as e:
                    self.logger.debug(f"Failed to read frame width: {e}")
                try:
                    fh = float(get_fh(frame_meta)) if callable(get_fh) else 0.0  # type: ignore[arg-type]
                    if fh > 0:
                        frame_height = fh
                except Exception as e:
                    self.logger.debug(f"Failed to read frame height: {e}")

            # Full mosaic dimensions (tiler configured 2x2)
            full_width = float(self.max_width * 2)
            full_height = float(self.max_height * 2)

            tile_left = 0.0
            tile_top = 0.0
            tile_width = frame_width
            tile_height = frame_height
            comp_rect = getattr(frame_meta, "compositor_rect", None)
            if comp_rect is not None:
                try:
                    tile_left = float(getattr(comp_rect, "left", tile_left))
                    tile_top = float(getattr(comp_rect, "top", tile_top))
                    tile_width = float(getattr(comp_rect, "width", tile_width))
                    tile_height = float(getattr(comp_rect, "height", tile_height))
                except Exception as e:
                    self.logger.debug(f"Failed to read compositor_rect: {e}")
            if int(sensor_id) not in self._logged_tile_rects:
                self.logger.debug(
                    "tiler rect sensor=%s left=%.1f top=%.1f width=%.1f height=%.1f",
                    sensor_id,
                    tile_left,
                    tile_top,
                    tile_width,
                    tile_height,
                )
                self._logged_tile_rects.add(int(sensor_id))
            tile_right = tile_left + tile_width
            tile_bottom = tile_top + tile_height

            sensor_smooth = self._bbox_smooth_by_sensor.setdefault(int(sensor_id), {})
            sensor_history = self.trail_history_by_sensor[int(sensor_id)]
            sensor_last_seen = self.trail_last_seen_by_sensor[int(sensor_id)]

            for obj_meta in meta_ops.iter_objects(operator, frame_meta):
                tid = meta_ops.get_object_id(operator, obj_meta)
                if tid == -1:
                    continue
                drawn_count += 1

                confidence = meta_ops.get_confidence(operator, obj_meta)
                stable_id = None
                if self.stable_id_mgr is not None:
                    try:
                        stable_id = self.stable_id_mgr.active_tracks.get(
                            (int(sensor_id), int(tid)), {}
                        ).get("stable_id")
                    except Exception as e:
                        self.logger.debug(f"stable_id lookup failed for tid={tid}: {e}")
                        stable_id = None
                # Prefer human-readable class label + confidence; optionally append stable ID
                try:
                    cls_label = getattr(obj_meta, 'obj_label', '') or ''
                except Exception:
                    cls_label = ''
                base_label = cls_label.strip() if cls_label else f"class {meta_ops.get_class_id(operator, obj_meta)}"
                label = f"{base_label} {confidence:.2f}"
                if stable_id not in (None, "", -1):
                    label = f"{label} sid {int(stable_id)}"

                try:
                    text_params = obj_meta.text_params
                    text_params.display_text = label
                    # Make label font larger and readable
                    try:
                        text_params.font_params.font_name = "Sans"
                    except Exception:
                        pass
                    try:
                        text_params.font_params.font_size = 22
                    except Exception:
                        pass
                    try:
                        text_params.set_bg_clr = 0
                    except Exception as e:
                        self.logger.debug(f"Setting text background flag failed for tid={tid}: {e}")
                except Exception as e:
                    self.logger.debug(f"Setting text params failed for tid={tid}: {e}")

                rect = meta_ops.get_rect_params(operator, obj_meta)
                if rect is None:
                    continue

                try:
                    r, g, b = self._color_for_track(int(tid))
                except Exception:
                    r, g, b = (1.0, 1.0, 0.0)
                try:
                    rect.border_width = 3
                    rect.border_color.set(r, g, b, 1.0)
                except Exception as e:
                    self.logger.debug(f"Setting rect border failed for tid={tid}: {e}")

                # ---------------- Holdover mask/bbox to reduce flicker ----------------
                try:
                    key = (int(sensor_id), int(tid))
                    left = float(getattr(rect, "left", 0.0) or 0.0)
                    top = float(getattr(rect, "top", 0.0) or 0.0)
                    width = float(getattr(rect, "width", 0.0) or 0.0)
                    height = float(getattr(rect, "height", 0.0) or 0.0)

                    cache = self._mask_holdover_cache.get(key)
                    has_detection = (confidence is not None) and (float(confidence) >= 0.0)

                    # Update cache on frames with detections
                    if has_detection and width > 1 and height > 1:
                        self._mask_holdover_cache[key] = {
                            "ts": now,
                            "bbox": (left, top, width, height),
                            "color": (r, g, b),
                        }
                    # If this is a tracker-only frame, paint a faint holdover rect if fresh
                    elif cache and (now - float(cache.get("ts", 0.0)) <= hold_s):
                        # Draw a translucent rectangle via DisplayMeta as a visual holdover
                        # This is separate from nvdsosd display-bbox to avoid global toggles
                        try:
                            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                            if display_meta:
                                # Respect budgets
                                if display_meta.num_rects < len(display_meta.rect_params):
                                    rp = display_meta.rect_params[display_meta.num_rects]
                                    cl, ct, cw, ch = cache.get("bbox", (left, top, width, height))
                                    rp.left = float(cl)
                                    rp.top = float(ct)
                                    rp.width = float(cw)
                                    rp.height = float(ch)
                                    cr, cg, cb = cache.get("color", (r, g, b))
                                    # Semi-transparent fill; thin border
                                    rp.has_bg_color = 1
                                    rp.bg_color.set(cr, cg, cb, 0.20)
                                    rp.border_width = 2
                                    rp.border_color.set(cr, cg, cb, 0.8)
                                    display_meta.num_rects += 1
                                    pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
                        except Exception as e:
                            self.logger.debug(f"Holdover draw failed for tid={tid}: {e}")
                except Exception:
                    # Never break probe due to holdover logic
                    pass

                left = float(getattr(rect, "left", 0.0))
                top = float(getattr(rect, "top", 0.0))
                width = max(1.0, float(getattr(rect, "width", 0.0)))
                height = max(1.0, float(getattr(rect, "height", 0.0)))

                prev_state = sensor_smooth.get(int(tid))
                if self.bbox_smoothing_enabled:
                    if prev_state is None:
                        prev_state = {}
                        prev_state["hist_h"] = deque(maxlen=240)
                        sensor_smooth[int(tid)] = prev_state

                    hist = prev_state.get("hist_h")
                    if isinstance(hist, deque):
                        while hist and (now - hist[0][0] > self.bbox_max_drop_window_s):
                            hist.popleft()
                        hist.append((now, height))
                        try:
                            max_h_recent = max(h for (_, h) in hist) if hist else height
                        except Exception:
                            max_h_recent = height
                        min_allowed_h = float(max_h_recent) * float(self.bbox_max_drop_ratio)
                        if height < min_allowed_h:
                            height = min_allowed_h

                    prev_w = max(1.0, float(prev_state.get("w", width)))
                    prev_h = max(1.0, float(prev_state.get("h", height)))
                    max_growth = max(1.0, float(self.bbox_smoothing_max_growth))
                    min_shrink = float(self.bbox_smoothing_max_shrink)
                    if min_shrink <= 0.0 or min_shrink > 1.0:
                        min_shrink = 0.85

                    width_clamped = max(prev_w * min_shrink, min(width, prev_w * max_growth))
                    height_clamped = max(prev_h * min_shrink, min(height, prev_h * max_growth))
                    alpha = float(self.bbox_smoothing_alpha)
                    new_w = prev_w + alpha * (width_clamped - prev_w)
                    new_h = prev_h + alpha * (height_clamped - prev_h)
                else:
                    new_w, new_h = width, height
                    prev_w = prev_h = 0.0  # unused but keep defined for clarity
                    prev_state = None

                if self.bbox_smoothing_anchor == "center":
                    cx_anchor = left + width * 0.5
                    cy_anchor = top + height * 0.5
                    new_left = cx_anchor - new_w * 0.5
                    new_top = cy_anchor - new_h * 0.5
                else:
                    cx_anchor = left + width * 0.5
                    bottom_y = top + height
                    new_left = cx_anchor - new_w * 0.5
                    new_top = bottom_y - new_h

                try:
                    if full_width > 0 and full_height > 0:
                        new_left = max(0.0, min(new_left, full_width - new_w))
                        new_top = max(0.0, min(new_top, full_height - new_h))
                except Exception as e:
                    self.logger.debug(f"Clamping bbox to frame failed for tid={tid}: {e}")

                try:
                    rect.left = float(new_left)
                    rect.top = float(new_top)
                    rect.width = float(max(1.0, new_w))
                    rect.height = float(max(1.0, new_h))
                except Exception as e:
                    self.logger.debug(f"Applying smoothed bbox failed for tid={tid}: {e}")

                if prev_state is not None:
                    prev_state["w"] = float(max(1.0, new_w))
                    prev_state["h"] = float(max(1.0, new_h))
                    prev_state["ts"] = float(now)

                cx = float(new_left + rect.width * 0.5)
                cy = float(new_top + rect.height)
                history = sensor_history[int(tid)]
                last_ts = sensor_last_seen.get(int(tid))
                if last_ts is not None and history:
                    dt = max(0.0, now - last_ts)
                    if dt > 0.0:
                        prev_x, prev_y = history[-1]
                        dx = cx - prev_x
                        dy = cy - prev_y
                        distance = math.hypot(dx, dy)
                        max_step = float(self.trail_max_speed_px_per_s) * dt
                        if max_step > 0.0 and distance > max_step:
                            scale = max_step / distance
                            cx = prev_x + dx * scale
                            cy = prev_y + dy * scale
                cx = max(tile_left, min(cx, tile_right))
                cy = max(tile_top, min(cy, tile_bottom))
                history.append((cx, cy))
                sensor_last_seen[int(tid)] = now

            if batch_meta is None:
                continue

            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            if not display_meta:
                continue

            display_meta.num_lines = 0
            display_meta.num_labels = 0
            remaining_lines = line_budget_max
            remaining_labels = label_budget_max

            sensor_history = self.trail_history_by_sensor[int(sensor_id)]
            for tid, pts_deque in list(sensor_history.items()):
                if remaining_lines <= 0 and remaining_labels <= 0:
                    break
                pts = list(pts_deque)
                if len(pts) <= 1:
                    continue
                limit = min(len(pts) - 1, trail_segments)
                start_index = max(0, len(pts) - (limit + 1))
                segment_points = pts[start_index:]

                for idx in range(len(segment_points) - 1):
                    if remaining_lines <= 0 or display_meta.num_lines >= len(display_meta.line_params):
                        break
                    x1, y1 = segment_points[idx]
                    x2, y2 = segment_points[idx + 1]
                    lp = display_meta.line_params[display_meta.num_lines]
                    lp.x1, lp.y1 = int(x1), int(y1)
                    lp.x2, lp.y2 = int(x2), int(y2)
                    lp.line_width = 3
                    try:
                        r, g, b = self._color_for_track(int(tid))
                    except Exception:
                        r, g, b = (1.0, 1.0, 0.0)
                    alpha = max((idx + 1) / max(1.0, len(segment_points)), 0.6)
                    try:
                        lp.line_color.set(r, g, b, float(alpha))
                    except Exception as e:
                        self.logger.debug(f"Setting trail line color failed for tid={tid}: {e}")
                    display_meta.num_lines += 1
                    remaining_lines -= 1
                    if remaining_lines <= 0:
                        break

                if self.trail_show_labels and remaining_labels > 0 and pts_deque:
                    if display_meta.num_labels < len(display_meta.text_params):
                        last_x, last_y = segment_points[-1]
                        tp = display_meta.text_params[display_meta.num_labels]
                        tp.display_text = f"id {tid}"
                        tp.x_offset = int(last_x)
                        tp.y_offset = int(last_y)
                        tp.font_params.font_name = "Serif"
                        tp.font_params.font_size = 12
                        try:
                            r, g, b = self._color_for_track(int(tid))
                        except Exception:
                            r, g, b = (1.0, 1.0, 0.0)
                        tp.font_params.font_color.set(r, g, b, 1.0)
                        tp.set_bg_clr = 0
                        display_meta.num_labels += 1
                        remaining_labels -= 1

            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
            try:
                ds_index = meta_ops.get_source_id(operator, frame_meta)
            except Exception as e:
                self.logger.debug(f"OSD probe: get_source_id failed: {e}")
                ds_index = -1
            self.logger.debug("mosaic_osd_probe: source=%s objects=%s", sensor_id, drawn_count)

        # Prune stale trails and smoothing state outside the loop to avoid concurrent mutation.
        timeout = float(self.trail_timeout_s)
        for sensor_id, last_seen_map in list(self.trail_last_seen_by_sensor.items()):
            for tid, last_ts in list(last_seen_map.items()):
                if now - last_ts > timeout:
                    last_seen_map.pop(tid, None)
                    history_map = self.trail_history_by_sensor.get(sensor_id)
                    if history_map is not None:
                        history_map.pop(tid, None)
                    smooth_map = self._bbox_smooth_by_sensor.get(sensor_id)
                    if smooth_map is not None:
                        smooth_map.pop(tid, None)
            if not last_seen_map:
                self.trail_last_seen_by_sensor.pop(sensor_id, None)
                if self.trail_history_by_sensor.get(sensor_id) == {}:
                    self.trail_history_by_sensor.pop(sensor_id, None)
                if self._bbox_smooth_by_sensor.get(sensor_id) == {}:
                    self._bbox_smooth_by_sensor.pop(sensor_id, None)

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
        config_file="pipelines/config_infer_primary_yolo11_seg.ini"
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
