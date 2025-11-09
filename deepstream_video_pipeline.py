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
from collections import defaultdict, deque

# Bypass libproxy issues by disabling GIO proxy resolver
from typing import Optional, Tuple, Dict, Any, List
import math

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
                self.logger.info(f"✅ Loaded {total} exclusion ROI(s) from {path}")
            else:
                self.logger.info(f"ℹ️ No exclusion ROIs found in {path}")
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
                mosaic_osd.set_property("process-mode", 1)
                mosaic_osd.set_property("display-text", 1)
                mosaic_osd.set_property("display-bbox", 1)
                # Enable mask rendering for instance segmentation
                mosaic_osd.set_property("display-mask", 1)

            # Configure live queues with consistent leaky buffering
            for queue_name in ("q_after_pgie", "q_before_tracker", "q_after_tracker", "mosaic_q", "jpeg_q", "egl_q"):
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
            
            # Link main processing chain
            if not elements['multiurisrc'].link(elements['nvdspreprocess']):
                raise RuntimeError("Failed to link nvmultiurisrcbin to nvdspreprocess")
            if not elements['nvdspreprocess'].link(elements['nvinfer']):
                raise RuntimeError("Failed to link nvdspreprocess to nvinfer")
            if not elements['nvinfer'].link(elements['q_after_pgie']): 
                raise RuntimeError("Failed to link nvinfer to q_after_pgie")
            # Add probe after nvinfer to verify detections
            q_after_pgie_sink_pad = elements['q_after_pgie'].get_static_pad("sink")
            if q_after_pgie_sink_pad:
                q_after_pgie_sink_pad.add_probe(
                    Gst.PadProbeType.BUFFER, self._nvinfer_output_probe, None
                )
                self.logger.info("✅ Attached nvinfer output probe for debugging")
            
            if not elements['q_after_pgie'].link(elements['q_before_tracker']):
                raise RuntimeError("Failed to link q_after_pgie to q_before_tracker")
            if not elements['q_before_tracker'].link(elements['nvtracker']): 
                raise RuntimeError("Failed to link q_before_tracker to nvtracker")
            if not elements['nvtracker'].link(elements['q_after_tracker']): 
                raise RuntimeError("Failed to link nvtracker to q_after_tracker")
            if not elements['q_after_tracker'].link(elements['nvdsanalytics_post']): 
                raise RuntimeError("Failed to link q_after_tracker to nvdsanalytics_post")
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
            nvinfer = Gst.ElementFactory.make("nvinfer", "nvinfer")
            
            # Analytics and Tracking
            nvtracker = Gst.ElementFactory.make("nvtracker", "nvtracker")
            nvdsanalytics_post = Gst.ElementFactory.make("nvdsanalytics", "nvdsanalytics_post")
            

            
            # Split to mosaic branch (tee duplicates post-analytics stream)
            main_tee = Gst.ElementFactory.make("tee", "main_tee")

            # Mosaic branch elements (GPU tiler → JPEG appsink)
            mosaic_q = Gst.ElementFactory.make("queue", "mosaic_q")
            mosaic_tiler = Gst.ElementFactory.make("nvmultistreamtiler", "mosaic_tiler")
            mosaic_conv_pre = Gst.ElementFactory.make("nvvideoconvert", "mosaic_conv_pre")
            mosaic_caps_rgba = Gst.ElementFactory.make("capsfilter", "mosaic_caps_rgba")
            mosaic_osd = Gst.ElementFactory.make("nvdsosd", "mosaic_osd")
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
            q_after_pgie = Gst.ElementFactory.make("queue", "q_after_pgie")
            q_before_tracker = Gst.ElementFactory.make("queue", "q_before_tracker")
            q_after_tracker = Gst.ElementFactory.make("queue", "q_after_tracker")

            # Add a queue for JPEG branch to decouple tee from encoder
            jpeg_q = Gst.ElementFactory.make("queue", "jpeg_q")

            # Validate element creation and assemble elements to add
            element_list = [
                multiurisrc, nvdspreprocess, nvinfer, nvtracker, nvdsanalytics_post,
                main_tee, q_after_pgie, q_before_tracker, q_after_tracker,
                mosaic_q, mosaic_tiler, mosaic_conv_pre, mosaic_caps_rgba, mosaic_osd,
                jpeg_q, mosaic_conv_post, mosaic_caps, mosaic_enc, mosaic_sink
            ]
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
                    "nvtracker", "nvdsanalytics_post", "tee",
                    "q_after_pgie", "q_before_tracker", "q_after_tracker",
                    "mosaic_q", "nvmultistreamtiler", "nvvideoconvert",
                    "capsfilter", "nvdsosd", "queue", "nvvideoconvert",
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
                'multiurisrc': multiurisrc, 'nvdspreprocess': nvdspreprocess, 'nvinfer': nvinfer,
                'nvtracker': nvtracker, 
                'nvdsanalytics_post': nvdsanalytics_post, 'main_tee': main_tee,
                'q_after_pgie': q_after_pgie, 'q_before_tracker': q_before_tracker, 
                'q_after_tracker': q_after_tracker,
                'mosaic_q': mosaic_q, 'mosaic_tiler': mosaic_tiler, 'mosaic_conv_pre': mosaic_conv_pre,
                'mosaic_caps_rgba': mosaic_caps_rgba, 'mosaic_osd': mosaic_osd, 'jpeg_q': jpeg_q, 'mosaic_conv_post': mosaic_conv_post,
                'mosaic_caps': mosaic_caps, 'mosaic_enc': mosaic_enc, 'mosaic_sink': mosaic_sink,
                'egl_q': egl_q, 'egl_sink': egl_sink,
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
            analytics_src_pad.add_probe(
                Gst.PadProbeType.BUFFER, self._post_remove_excluded_objects_probe, None
            )
            self.logger.info("✅ Added buffer probe to nvdsanalytics_post src pad for exclusion pruning")
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
            # Store mosaic elements for diagnostics (caps dumps)
            self.mosaic_osd = mosaic_osd
            self.mosaic_enc = mosaic_enc
            #self.pipeline.add(main_tee)
            self.logger.info("main_tee added to pipeline post-OSD")

            # Temporary instrumentation: count PGIE objects immediately after nvinfer
            try:
                nvinfer_debug_pad = q_after_pgie.get_static_pad("sink") if q_after_pgie else None
                if nvinfer_debug_pad:
                    nvinfer_debug_pad.add_probe(
                        Gst.PadProbeType.BUFFER,
                        self._nvinfer_object_debug_probe,
                        None,
                    )
                    self.logger.info("✅ Attached nvinfer object debug probe (temporary)")
                else:
                    self.logger.warning("nvinfer object debug probe skipped: q_after_pgie sink pad unavailable")
            except Exception as exc:
                self.logger.warning(f"Failed to attach nvinfer object debug probe: {exc}")

            # Link mosaic branch so both outputs receive the same post-OSD RGBA mosaic
            if not nvdsanalytics_post.link(mosaic_q):
                raise RuntimeError("Failed to link nvdsanalytics_post to mosaic_q")
            if not mosaic_q.link(mosaic_tiler):
                raise RuntimeError("Failed to link mosaic_q to mosaic_tiler")
            if not mosaic_tiler.link(mosaic_conv_pre):
                raise RuntimeError("Failed to link mosaic_tiler to mosaic_conv_pre")
            if not mosaic_conv_pre.link(mosaic_caps_rgba):
                raise RuntimeError("Failed to link mosaic_conv_pre to mosaic_caps_rgba")
            if not mosaic_caps_rgba.link(mosaic_osd):
                raise RuntimeError("Failed to link mosaic_caps_rgba to mosaic_osd")
            # Tee immediately after OSD so EGL can consume RGBA directly; JPEG branch converts to I420
            if not mosaic_osd.link(main_tee):
                raise RuntimeError("Failed to link mosaic_osd to main_tee")
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
            self._attach_flow_probe(nvtracker.get_static_pad("sink"), "nvtracker.sink")
            self._attach_flow_probe(nvtracker.get_static_pad("src"), "nvtracker.src")
            self._attach_flow_probe(nvdsanalytics_post.get_static_pad("sink"), "nvdsanalytics_post.sink")
            self._attach_flow_probe(nvdsanalytics_post.get_static_pad("src"), "nvdsanalytics_post.src")
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
                except Exception as exc:
                    self.logger.warning(f"Caps dump failed: {exc}")
            
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
        """Attach a lightweight BUFFER probe that logs flow statistics periodically."""
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
                if now - probe_state["last_log"] >= 2.0:
                    delta = probe_state["count"] - probe_state["last_logged_count"]
                    self.logger.info(f"flow[{label}]: {delta} buffers in {now - probe_state['last_log']:.1f}s")
                    probe_state["last_logged_count"] = probe_state["count"]
                    probe_state["last_log"] = now
                return Gst.PadProbeReturn.OK

            pad.add_probe(Gst.PadProbeType.BUFFER, _probe, state)
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
                    self.logger.info("🎯 nvinfer output: class_id=%d, confidence=%.3f", class_id, confidence)
        
        if total_objects > 0:
            self.logger.info("✅ nvinfer output probe: %d object(s) detected across %d frame(s)", total_objects, len(frames))
        elif len(frames) > 0:
            self.logger.debug("⚠️  nvinfer output probe: No objects detected in %d frame(s)", len(frames))
        
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
            self.logger.info("🔍 OSD probe: Found %d object(s) across %d frame(s)", total_objects, len(frames))
        else:
            self.logger.debug("OSD probe: No objects detected in %d frame(s)", len(frames))

        trail_segments = int(getattr(self.config.visualization, "TRAIL_DRAW_SEGMENTS", 64) or 0)
        trail_segments = max(1, trail_segments)
        line_budget_max = 250
        label_budget_max = 16

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
                label = f"ds {int(tid)} ({confidence:.2f})"
                if stable_id not in (None, "", -1):
                    label = f"sid {int(stable_id)} ds {int(tid)} ({confidence:.2f})"

                try:
                    text_params = obj_meta.text_params
                    text_params.display_text = label
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
