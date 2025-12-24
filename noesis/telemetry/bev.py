from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Any

import cv2
import numpy as np
import time
import math
import threading

from geometry.homography import Plane, parse_extrinsics, ray_from_pixel, intersect_plane

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalibrationSnapshot:
    camera_id: str
    intrinsics: np.ndarray  # 3x3 matrix
    extrinsics_col_major: Sequence[float]  # 16 values, world→camera
    floor_y: float
    image_size: Tuple[int, int]
    unit_scale: float = 1.0


@dataclass
class BevConfig:
    meters_per_px: float = 0.05
    x_range: Tuple[float, float] = (-4.0, 4.0)
    z_range: Tuple[float, float] = (0.0, 12.0)
    overlay: bool = True
    max_px: int = 768
    auto_fit_extents: bool = True
    max_distance_m: float = 8.0


@dataclass
class Footpoint:
    u: float
    v: float
    method: str = "bbox"
    track_id: Optional[int] = None
    stable_id: Optional[int] = None


@dataclass(frozen=True)
class BevTrailConfig:
    enabled: bool = True
    window_s: float = 8.0
    draw_stride: int = 2
    min_step_px: float = 2.0
    min_dt_s: float = 0.08
    smooth_tau_s: float = 0.25
    max_speed_px_per_s: float = 600.0
    max_points_per_track: int = 129
    max_segments_per_track: int = 64
    max_tracks: int = 8
    line_width: int = 3
    min_alpha: float = 0.15
    color_key: str = "stable_id"

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "window_s", max(0.1, float(self.window_s)))
        object.__setattr__(self, "draw_stride", max(1, int(self.draw_stride)))
        object.__setattr__(self, "min_step_px", max(0.0, float(self.min_step_px)))
        object.__setattr__(self, "min_dt_s", max(0.0, float(self.min_dt_s)))
        object.__setattr__(self, "smooth_tau_s", max(0.0, float(self.smooth_tau_s)))
        object.__setattr__(self, "max_speed_px_per_s", max(0.0, float(self.max_speed_px_per_s)))
        object.__setattr__(self, "max_points_per_track", max(2, int(self.max_points_per_track)))
        object.__setattr__(self, "max_segments_per_track", max(1, int(self.max_segments_per_track)))
        object.__setattr__(self, "max_tracks", max(1, int(self.max_tracks)))
        object.__setattr__(self, "line_width", max(1, int(self.line_width)))
        object.__setattr__(self, "min_alpha", float(min(1.0, max(0.0, float(self.min_alpha)))))
        color_key = str(self.color_key or "stable_id").strip().lower()
        if color_key not in ("stable_id", "track_id"):
            color_key = "stable_id"
        object.__setattr__(self, "color_key", color_key)

    @classmethod
    def from_mapping(cls, cfg: Any) -> "BevTrailConfig":
        if not isinstance(cfg, dict):
            return cls()

        def _bool(value: Any, default: bool) -> bool:
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            if text in ("1", "true", "yes", "y", "on"):
                return True
            if text in ("0", "false", "no", "n", "off"):
                return False
            return default

        def _float(value: Any, default: float) -> float:
            try:
                return float(value)
            except Exception:
                return default

        def _int(value: Any, default: int) -> int:
            try:
                return int(value)
            except Exception:
                return default

        max_points_value = _int(cfg.get("max_points_per_track"), 129)
        max_segments_value = _int(cfg.get("max_segments_per_track"), 64)
        if max_segments_value > (max_points_value - 1):
            max_segments_value = max(1, int(max_points_value) - 1)

        return cls(
            enabled=_bool(cfg.get("enabled"), True),
            window_s=_float(cfg.get("window_s"), 8.0),
            draw_stride=_int(cfg.get("draw_stride"), 2),
            min_step_px=_float(cfg.get("min_step_px"), 2.0),
            min_dt_s=_float(cfg.get("min_dt_s"), 0.08),
            smooth_tau_s=_float(cfg.get("smooth_tau_s"), 0.25),
            max_speed_px_per_s=_float(cfg.get("max_speed_px_per_s"), 600.0),
            max_points_per_track=max_points_value,
            max_segments_per_track=max_segments_value,
            max_tracks=_int(cfg.get("max_tracks"), 8),
            line_width=_int(cfg.get("line_width"), 3),
            min_alpha=_float(cfg.get("min_alpha"), 0.15),
            color_key=str(cfg.get("color_key") or "stable_id"),
        )


@dataclass
class _BevTrailTrackState:
    points: "deque[Tuple[float, float, float]]" = field(default_factory=deque)
    last_seen_ts: float = 0.0
    stable_id: Optional[int] = None
    ema_x: Optional[float] = None
    ema_z: Optional[float] = None
    ema_ts: float = 0.0


@dataclass
class BevResult:
    camera_id: str
    bev_bgr: Optional[np.ndarray]
    bev_points: List[Dict[str, float]]
    config: BevConfig
    timestamp_us: int
    width_px: int
    height_px: int


class HomographyCache:
    def __init__(self) -> None:
        self._cache: Dict[str, np.ndarray] = {}

    @staticmethod
    def _hash_matrix(matrix: np.ndarray) -> int:
        return hash(matrix.tobytes())

    def _key(
        self,
        calib: CalibrationSnapshot,
    ) -> str:
        return "::".join(
            [
                calib.camera_id,
                f"{calib.image_size[0]}x{calib.image_size[1]}",
                f"K={self._hash_matrix(calib.intrinsics)}",
                f"E={hash(tuple(float(x) for x in calib.extrinsics_col_major))}",
                f"floor={calib.floor_y:.4f}",
                f"s={float(getattr(calib, 'unit_scale', 1.0)):.6f}",
            ]
        )

    def get(
        self,
        calib: CalibrationSnapshot,
    ) -> np.ndarray:
        key = self._key(calib)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        
        from geometry.homography import img_to_plane_homography
        H = img_to_plane_homography(
            calib.intrinsics,
            calib.extrinsics_col_major,
            calib.floor_y,
            calib.image_size,
            float(getattr(calib, "unit_scale", 1.0) or 1.0),
        )
        self._cache[key] = H
        return H


class BevRenderer:
    def __init__(
        self,
        ws_server,
        trails_cfg: Optional[Dict[str, Any]] = None,
        *,
        jpeg_enabled: bool = False,
        jpeg_quality: int = 70,
    ) -> None:
        self.ws = ws_server
        self._lock = threading.Lock()
        self.config_per_cam: Dict[str, BevConfig] = {}
        self.h_cache = HomographyCache()
        # Last known good homography per camera for resilience
        self._last_h_by_cam: Dict[str, np.ndarray] = {}
        # Auto-computed extents per camera
        self._auto_extents_by_camera: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
        # Trail state (rendered into BEV images)
        self._trail_cfg = BevTrailConfig.from_mapping(trails_cfg or {})
        self._trails_enabled = bool(self._trail_cfg.enabled)
        self._trail_tracks_by_cam: Dict[str, Dict[int, _BevTrailTrackState]] = {}
        self._trail_frame_counts: Dict[str, int] = {}
        self._trail_color_cache: Dict[int, Tuple[int, int, int]] = {}
        self._jpeg_enabled = bool(jpeg_enabled)
        self._jpeg_quality = max(1, min(100, int(jpeg_quality)))

    def set_trails_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._trails_enabled = bool(enabled)
            if not self._trails_enabled:
                self._trail_tracks_by_cam.clear()
                self._trail_frame_counts.clear()

    def _hsl_to_rgb(self, h: float, s: float, lightness: float) -> Tuple[float, float, float]:
        h = h % 360.0
        s = max(0.0, min(1.0, s))
        lightness = max(0.0, min(1.0, lightness))
        c = (1.0 - abs(2.0 * lightness - 1.0)) * s
        x = c * (1.0 - abs(((h / 60.0) % 2.0) - 1.0))
        m = lightness - c / 2.0
        rp = gp = bp = 0.0
        if 0.0 <= h < 60.0:
            rp, gp, bp = c, x, 0.0
        elif 60.0 <= h < 120.0:
            rp, gp, bp = x, c, 0.0
        elif 120.0 <= h < 180.0:
            rp, gp, bp = 0.0, c, x
        elif 180.0 <= h < 240.0:
            rp, gp, bp = 0.0, x, c
        elif 240.0 <= h < 300.0:
            rp, gp, bp = x, 0.0, c
        else:
            rp, gp, bp = c, 0.0, x
        r, g, b = rp + m, gp + m, bp + m
        return (max(0.0, min(1.0, r)), max(0.0, min(1.0, g)), max(0.0, min(1.0, b)))

    def _color_for_key(self, key: int) -> Tuple[int, int, int]:
        cached = self._trail_color_cache.get(key)
        if cached is not None:
            return cached
        hue = float((int(key) * 47) % 360)
        r, g, b = self._hsl_to_rgb(hue, 0.80, 0.60)
        bgr = (int(round(b * 255.0)), int(round(g * 255.0)), int(round(r * 255.0)))
        self._trail_color_cache[key] = bgr
        return bgr

    @staticmethod
    def _resample_points(
        points: List[Tuple[float, float, float]],
        segments_budget: int,
        *,
        bias: float = 2.0,
    ) -> List[Tuple[float, float, float]]:
        if segments_budget <= 0:
            return []
        segments_available = len(points) - 1
        if segments_available <= segments_budget:
            return points
        selected: List[Tuple[float, float, float]] = []
        last_idx = -1
        for j in range(segments_budget + 1):
            t = 0.0 if segments_budget == 0 else (float(j) / float(segments_budget))
            raw = int(round((t**bias) * float(segments_available)))
            min_idx = j
            max_idx = segments_available - (segments_budget - j)
            idx = max(min_idx, min(max_idx, raw))
            if idx <= last_idx:
                idx = min(max_idx, last_idx + 1)
            selected.append(points[idx])
            last_idx = idx
        return selected

    @staticmethod
    def _local_to_px(
        x_local: float,
        z_local: float,
        cfg: BevConfig,
        bev_shape: Tuple[int, int],
    ) -> Optional[Tuple[int, int]]:
        height_px, width_px = int(bev_shape[0]), int(bev_shape[1])
        if width_px <= 0 or height_px <= 0:
            return None
        span_x = max(1e-3, float(cfg.x_range[1] - cfg.x_range[0]))
        span_z = max(1e-3, float(cfg.z_range[1] - cfg.z_range[0]))
        u_px = int(round((float(x_local) - float(cfg.x_range[0])) / span_x * max(1, width_px - 1)))
        v_px = int(round((float(cfg.z_range[1]) - float(z_local)) / span_z * max(1, height_px - 1)))
        if 0 <= u_px < width_px and 0 <= v_px < height_px:
            return u_px, v_px
        return None

    def _compute_auto_extents_local(
        self, calib: CalibrationSnapshot, cfg: BevConfig
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Compute camera-local BEV extents so footpoints and ranges share the same frame."""
        width, height = calib.image_size
        if width <= 0 or height <= 0:
            return (-4.0, 4.0), (0.0, 12.0)

        R_wc, C_world = parse_extrinsics(calib.extrinsics_col_major)
        scale = float(calib.unit_scale or 1.0)
        C_world = C_world * scale
        plane = Plane.horizontal(float(calib.floor_y) * scale)

        # Align samples with camera forward axis for local coordinates
        dir_world = R_wc @ np.array([0.0, 0.0, 1.0])
        yaw = math.atan2(dir_world[0], dir_world[2])
        cos_yaw = math.cos(-yaw)
        sin_yaw = math.sin(-yaw)

        xs = np.linspace(0, max(0.0, float(width - 1)), 8)
        ys = np.linspace(0, max(0.0, float(height - 1)), 8)
        hits_local: List[Tuple[float, float]] = []
        for u in xs:
            for v in ys:
                origin, direction = ray_from_pixel(float(u), float(v), calib.intrinsics, R_wc, C_world)
                hit = intersect_plane(origin, direction, plane)
                if hit is None:
                    continue
                dx = float(hit[0] - C_world[0])
                dz = float(hit[2] - C_world[2])
                dist = math.hypot(dx, dz)
                if dist > cfg.max_distance_m and dist > 1e-6:
                    scale_d = cfg.max_distance_m / dist
                    dx *= scale_d
                    dz *= scale_d
                lx = dx * cos_yaw - dz * sin_yaw
                lz = dx * sin_yaw + dz * cos_yaw
                hits_local.append((lx, lz))

        if len(hits_local) < 3:
            return (-4.0, 4.0), (0.0, 12.0)

        xs_local = [p[0] for p in hits_local]
        zs_local = [p[1] for p in hits_local]
        padding = 0.25
        x_min = min(xs_local) - padding
        x_max = max(xs_local) + padding
        z_min = max(0.0, min(zs_local) - padding)
        z_max = max(zs_local) + padding

        if (x_max - x_min) < 1.0:
            cx = 0.5 * (x_min + x_max)
            x_min = cx - 0.5
            x_max = cx + 0.5
        if (z_max - z_min) < 1.0:
            cz = 0.5 * (z_min + z_max)
            z_min = max(0.0, cz - 0.5)
            z_max = cz + 0.5

        return (x_min, x_max), (z_min, z_max)

    def publish_status(self, camera_id: str, **fields: Any) -> None:
        """Publish a lightweight BEV status/error message to clients."""
        try:
            payload: Dict[str, Any] = {
                "type": "bev-status",
                "cameraId": str(camera_id),
                "ts": int(time.time() * 1_000_000),
            }
            payload.update({k: v for k, v in fields.items() if k is not None})
            if hasattr(self.ws, "broadcast_sync"):
                self.ws.broadcast_sync(payload)
        except Exception:
            pass

    def update_config(self, camera_id: str, cfg: Dict[str, float]) -> BevConfig:
        current = self.config_per_cam.get(camera_id, BevConfig())
        next_cfg = BevConfig(
            meters_per_px=float(cfg.get("mpp", current.meters_per_px)),
            x_range=(
                float(cfg.get("xMin", current.x_range[0])),
                float(cfg.get("xMax", current.x_range[1])),
            ),
            z_range=(
                float(cfg.get("zMin", current.z_range[0])),
                float(cfg.get("zMax", current.z_range[1])),
            ),
            overlay=bool(cfg.get("overlay", current.overlay)),
            max_px=current.max_px,
            auto_fit_extents=bool(cfg.get("autoFitExtents", current.auto_fit_extents)),
            max_distance_m=float(cfg.get("maxDistanceM", current.max_distance_m)),
        )
        self.config_per_cam[camera_id] = next_cfg
        return next_cfg

    def set_overlay(self, camera_id: str, enabled: bool) -> BevConfig:
        cfg = self.config_per_cam.get(camera_id, BevConfig())
        cfg.overlay = bool(enabled)
        self.config_per_cam[camera_id] = cfg
        return cfg

    def render_and_publish(
        self,
        camera_id: str,
        calib: CalibrationSnapshot,
        frame_bgr: Optional[np.ndarray] = None,
        footpoints: Optional[List[Footpoint]] = None,
        timestamp_us: int = 0,
    ) -> None:
        now_s = float(timestamp_us) / 1_000_000.0
        cfg = self.config_per_cam.get(camera_id, BevConfig())
        footpoints = footpoints or []

        with self._lock:
            # Prune fully expired trails for this camera so we can decide whether it's
            # worth publishing a frame when there are no current footpoints.
            trail_window = float(self._trail_cfg.window_s)
            cam_tracks = self._trail_tracks_by_cam.get(camera_id)
            if cam_tracks:
                expired: List[int] = []
                for track_id, state in cam_tracks.items():
                    while state.points and (now_s - float(state.points[0][0])) > trail_window:
                        state.points.popleft()
                    if not state.points and (now_s - float(state.last_seen_ts)) > trail_window:
                        expired.append(track_id)
                for track_id in expired:
                    cam_tracks.pop(track_id, None)
                if not cam_tracks:
                    self._trail_tracks_by_cam.pop(camera_id, None)

            has_trails = bool(self._trail_tracks_by_cam.get(camera_id))

        if not footpoints and not has_trails:
            return

        # Compute or fetch extents for this camera
        x_range = cfg.x_range
        z_range = cfg.z_range

        if cfg.auto_fit_extents:
            ext_hash = hash(tuple(float(x) for x in calib.extrinsics_col_major))
            intr_hash = HomographyCache._hash_matrix(calib.intrinsics)
            key = "::".join(
                [
                    camera_id,
                    str(ext_hash),
                    str(intr_hash),
                    f"{calib.image_size[0]}x{calib.image_size[1]}",
                    f"{float(calib.floor_y):.4f}",
                    f"{float(getattr(calib, 'unit_scale', 1.0) or 1.0):.6f}",
                ]
            )

            auto_extents = self._auto_extents_by_camera.get(key)
            if auto_extents is None:
                x_range_auto, z_range_auto = self._compute_auto_extents_local(calib, cfg)
                auto_extents = (x_range_auto, z_range_auto)
                self._auto_extents_by_camera[key] = auto_extents
            x_range, z_range = auto_extents

        # Determine canvas size in pixels based on extents and meters-per-pixel.
        # Respect max_px by scaling mpp upward if needed.
        span_x = max(1e-3, float(x_range[1] - x_range[0]))
        span_z = max(1e-3, float(z_range[1] - z_range[0]))
        base_mpp = max(1e-4, float(cfg.meters_per_px))
        width_px = int(math.ceil(span_x / base_mpp))
        height_px = int(math.ceil(span_z / base_mpp))
        effective_mpp = base_mpp
        if cfg.max_px > 0:
            max_dim = max(width_px, height_px)
            if max_dim > cfg.max_px:
                scale = cfg.max_px / float(max_dim)
                width_px = max(1, int(width_px * scale))
                height_px = max(1, int(height_px * scale))
                effective_mpp = base_mpp / max(scale, 1e-6)

        bev: Optional[np.ndarray] = None
        if self._jpeg_enabled:
            bev = np.zeros((max(1, height_px), max(1, width_px), 3), dtype=np.uint8)

        # Compute or reuse homography; fallback to last good if current fails
        H_img2plane = None
        try:
            H_img2plane = self.h_cache.get(calib)
            self._last_h_by_cam[camera_id] = H_img2plane
        except Exception as e:
            logger.warning("BEV: homography computation failed for %s: %s", camera_id, e)
            cached_result = self._last_h_by_cam.get(camera_id)
            if cached_result is None:
                # No homography at all yet; notify clients of failure once per attempt window.
                self.publish_status(camera_id, error="homography_failed", details=str(e))
                return
            H_img2plane = cached_result

        bev_points: List[Dict[str, float]] = []
        current_local_by_track: Dict[int, Tuple[float, float, Optional[int]]] = {}

        # Calculate Camera Yaw and Position for Local Transformation
        # We want points relative to the camera (Local Frame), aligned with the camera view.
        R_wc, C_world = parse_extrinsics(calib.extrinsics_col_major)
        scale = float(calib.unit_scale or 1.0)
        C_world = C_world * scale
        
        # Camera forward vector in camera frame is [0, 0, 1] (assuming standard CV frame)
        # In World frame:
        dir_world = R_wc @ np.array([0.0, 0.0, 1.0])
        # Yaw is angle in XZ plane. atan2(x, z) gives 0 for +Z (North), pi/2 for +X (East)
        yaw = math.atan2(dir_world[0], dir_world[2])
        cos_yaw = math.cos(-yaw)
        sin_yaw = math.sin(-yaw)

        for fp in footpoints:
            vec = np.array([fp.u, fp.v, 1.0], dtype=np.float64)
            # H maps [u, v, 1] -> [x_meters, z_meters, w] (World Coordinates)
            world_pt = H_img2plane @ vec
            w = world_pt[2] if world_pt[2] else 1.0
            wx = float(world_pt[0] / w)
            wz = float(world_pt[1] / w)
            
            if np.isnan(wx) or np.isnan(wz):
                continue
            
            # Transform World -> Local Camera Frame
            # 1. Translate
            dx = wx - C_world[0]
            dz = wz - C_world[2]
            
            # 2. Rotate by -Yaw
            lx = dx * cos_yaw - dz * sin_yaw
            lz = dx * sin_yaw + dz * cos_yaw

            # The frontend expects 'x' and 'y' in the JSON list.
            # We map Local X -> JSON x, Local Z -> JSON y
            bev_points.append({'x': lx, 'y': lz, 'method': fp.method, 'trackId': fp.track_id, 'stableId': fp.stable_id})
            try:
                track_id = int(fp.track_id) if fp.track_id is not None else -1
            except Exception:
                track_id = -1
            if track_id >= 0:
                current_local_by_track[track_id] = (float(lx), float(lz), fp.stable_id)

        # Update config to reflect the actual extents used
        result_config = BevConfig(
            meters_per_px=effective_mpp,
            x_range=x_range,
            z_range=z_range,
            overlay=cfg.overlay,
            max_px=cfg.max_px,
            auto_fit_extents=cfg.auto_fit_extents,
            max_distance_m=cfg.max_distance_m,
        )

        trails_to_draw: List[Tuple[int, Optional[int], List[Tuple[float, float, float]]]] = []
        with self._lock:
            trails_enabled = bool(self._trails_enabled) and bool(self._trail_cfg.enabled)
            cam_tracks = self._trail_tracks_by_cam.setdefault(camera_id, {})

            # Update per-camera frame counter for stride decisions.
            frame_idx = int(self._trail_frame_counts.get(camera_id, 0) + 1)
            self._trail_frame_counts[camera_id] = frame_idx
            do_sample = (frame_idx % int(self._trail_cfg.draw_stride)) == 0

            min_dt_s = float(self._trail_cfg.min_dt_s)
            min_step_m = float(self._trail_cfg.min_step_px) * float(effective_mpp)
            max_points = max(2, int(self._trail_cfg.max_points_per_track))
            window_s = float(self._trail_cfg.window_s)

            if trails_enabled:
                for track_id, (lx, lz, stable_id) in current_local_by_track.items():
                    state = cam_tracks.get(track_id)
                    if state is None:
                        state = _BevTrailTrackState(points=deque(maxlen=max_points))
                        cam_tracks[track_id] = state
                    state.last_seen_ts = float(now_s)
                    try:
                        stable_id_int = int(stable_id) if stable_id not in (None, "", -1) else None
                    except Exception:
                        stable_id_int = None
                    if stable_id_int is not None and stable_id_int > 0:
                        state.stable_id = int(stable_id_int)

                    # Always prune old samples so disappeared tracks naturally fade out.
                    while state.points and (now_s - float(state.points[0][0])) > window_s:
                        state.points.popleft()

                    if not do_sample:
                        continue

                    x = float(lx)
                    z = float(lz)

                    # Clamp spurious jumps based on max speed in BEV pixels/sec scaled by effective mpp.
                    if state.points:
                        prev_ts, prev_x, prev_z = state.points[-1]
                        dt = max(0.0, float(now_s) - float(prev_ts))
                        dist = math.hypot(x - float(prev_x), z - float(prev_z))
                        max_step = float(self._trail_cfg.max_speed_px_per_s) * float(effective_mpp) * dt
                        if max_step > 0.0 and dist > max_step:
                            scale_step = max_step / dist
                            x = float(prev_x) + (x - float(prev_x)) * scale_step
                            z = float(prev_z) + (z - float(prev_z)) * scale_step

                    # Smooth the footpoint directly (time-constant based EMA).
                    if self._trail_cfg.smooth_tau_s > 0.0:
                        if state.ema_x is None or state.ema_z is None:
                            state.ema_x, state.ema_z = x, z
                            state.ema_ts = float(now_s)
                        else:
                            dt_ema = max(0.0, float(now_s) - float(state.ema_ts))
                            tau = float(self._trail_cfg.smooth_tau_s)
                            alpha = 1.0 - math.exp(-dt_ema / tau) if (tau > 0.0 and dt_ema > 0.0) else 1.0
                            state.ema_x = float(state.ema_x + alpha * (x - float(state.ema_x)))
                            state.ema_z = float(state.ema_z + alpha * (z - float(state.ema_z)))
                            state.ema_ts = float(now_s)
                        x, z = float(state.ema_x), float(state.ema_z)

                    # Decimation: enforce a time-window based sampling budget.
                    if state.points:
                        prev_ts, prev_x, prev_z = state.points[-1]
                        dt = max(0.0, float(now_s) - float(prev_ts))
                        dist = math.hypot(x - float(prev_x), z - float(prev_z))
                        if dt < min_dt_s:
                            if dist >= min_step_m:
                                state.points[-1] = (float(prev_ts), float(x), float(z))
                            continue
                        if dist < min_step_m:
                            continue

                    state.points.append((float(now_s), float(x), float(z)))

            # Remove fully expired tracks to keep memory bounded.
            expired: List[int] = []
            for track_id, state in cam_tracks.items():
                while state.points and (now_s - float(state.points[0][0])) > window_s:
                    state.points.popleft()
                if not state.points and (now_s - float(state.last_seen_ts)) > window_s:
                    expired.append(track_id)
            for track_id in expired:
                cam_tracks.pop(track_id, None)

            if trails_enabled and cam_tracks:
                # Select a bounded number of tracks to render.
                track_items = list(cam_tracks.items())
                track_items.sort(key=lambda item: float(getattr(item[1], "last_seen_ts", 0.0)), reverse=True)
                max_tracks = max(1, int(self._trail_cfg.max_tracks))
                for track_id, state in track_items[:max_tracks]:
                    pts = list(state.points)
                    if len(pts) < 2:
                        continue
                    trails_to_draw.append((track_id, state.stable_id, pts))

        # Draw grid + trails + footpoints if overlay is enabled
        if bev is not None and result_config.overlay:
            self._draw_grid(bev, result_config)
            if trails_to_draw:
                inv_window = 1.0 / max(0.1, float(self._trail_cfg.window_s))
                min_alpha = float(self._trail_cfg.min_alpha)
                line_width = int(self._trail_cfg.line_width)
                for track_id, stable_id, pts in trails_to_draw:
                    if len(pts) < 2:
                        continue
                    segments_available = len(pts) - 1
                    segments_budget = min(segments_available, int(self._trail_cfg.max_segments_per_track))
                    if segments_budget <= 0:
                        continue
                    pts = self._resample_points(pts, segments_budget)
                    key_id = int(track_id)
                    if self._trail_cfg.color_key == "stable_id":
                        try:
                            stable_id_int = int(stable_id) if stable_id not in (None, "", -1) else None
                        except Exception:
                            stable_id_int = None
                        if stable_id_int is not None and stable_id_int > 0:
                            key_id = int(stable_id_int)
                    base_bgr = self._color_for_key(key_id)
                    for idx in range(len(pts) - 1):
                        ts0, x1, z1 = pts[idx]
                        _ts1, x2, z2 = pts[idx + 1]
                        p1 = self._local_to_px(x1, z1, result_config, bev.shape[:2])
                        p2 = self._local_to_px(x2, z2, result_config, bev.shape[:2])
                        if p1 is None or p2 is None:
                            continue
                        age = max(0.0, float(now_s) - float(ts0))
                        t = 1.0 - min(1.0, age * inv_window)
                        alpha = min_alpha + (1.0 - min_alpha) * max(0.0, min(1.0, t))
                        color = (
                            int(round(float(base_bgr[0]) * alpha)),
                            int(round(float(base_bgr[1]) * alpha)),
                            int(round(float(base_bgr[2]) * alpha)),
                        )
                        cv2.line(bev, p1, p2, color, line_width, lineType=cv2.LINE_AA)

            for track_id, (lx, lz, stable_id) in current_local_by_track.items():
                p = self._local_to_px(lx, lz, result_config, bev.shape[:2])
                if p is None:
                    continue
                key_id = int(track_id)
                if self._trail_cfg.color_key == "stable_id":
                    try:
                        stable_id_int = int(stable_id) if stable_id not in (None, "", -1) else None
                    except Exception:
                        stable_id_int = None
                    if stable_id_int is not None and stable_id_int > 0:
                        key_id = int(stable_id_int)
                base_bgr = self._color_for_key(key_id)
                cv2.circle(bev, p, 4, base_bgr, -1, lineType=cv2.LINE_AA)

        result = BevResult(
            camera_id=camera_id,
            bev_bgr=bev,
            bev_points=bev_points,
            config=result_config,
            timestamp_us=timestamp_us,
            width_px=width_px,
            height_px=height_px,
        )
        self._publish(result, calib, H_img2plane)

    def _draw_grid(self, bev: np.ndarray, cfg: BevConfig) -> None:
        if cfg.meters_per_px <= 0:
            return
        major_step_px = max(10, int(round(1.0 / cfg.meters_per_px)))
        height, width = bev.shape[:2]
        color = (40, 40, 40)
        for x in range(0, width, major_step_px):
            cv2.line(bev, (x, 0), (x, height - 1), color, 1, lineType=cv2.LINE_AA)
        for y in range(0, height, major_step_px):
            cv2.line(bev, (0, y), (width - 1, y), color, 1, lineType=cv2.LINE_AA)

    def _publish(self, result: BevResult, calib: CalibrationSnapshot, H_to_use: np.ndarray) -> None:
        try:
            # Compute a quick sanity sample: image bottom-center ray intersection in meters (XZ)
            sample_xz: Optional[Tuple[float, float]] = None
            try:
                width_src, height_src = calib.image_size
                u = float(max(0.0, width_src * 0.5))
                v = float(max(0.0, height_src - 1))
                R_wc, C_world = parse_extrinsics(calib.extrinsics_col_major)
                scale = float(calib.unit_scale or 1.0)
                C_world = C_world * scale
                plane = Plane.horizontal(float(calib.floor_y) * scale)
                origin, direction = ray_from_pixel(u, v, calib.intrinsics, R_wc, C_world)
                hit = intersect_plane(origin, direction, plane)
                if hit is not None:
                    sample_xz = (float(hit[0]), float(hit[2]))
            except Exception:
                sample_xz = None
            status = {
                "type": "bev-frame",
                "cameraId": result.camera_id,
                "ts": result.timestamp_us,
                "w": int(result.width_px),
                "h": int(result.height_px),
                "mpp": result.config.meters_per_px,
                "xMin": result.config.x_range[0],
                "xMax": result.config.x_range[1],
                "zMin": result.config.z_range[0],
                "zMax": result.config.z_range[1],
                "overlay": result.config.overlay,
                "footpoints": result.bev_points,
                # Optional: flattened 3x3 homography for client-side debug/overlays
                "H": [float(x) for x in H_to_use.reshape(-1)],
                "sampleXZ": list(sample_xz) if sample_xz is not None else None,
            }
            if hasattr(self.ws, "broadcast_sync"):
                self.ws.broadcast_sync(status)
                if self._jpeg_enabled and result.bev_bgr is not None:
                    ok, jpeg = cv2.imencode(
                        ".jpg",
                        result.bev_bgr,
                        [int(cv2.IMWRITE_JPEG_QUALITY), int(self._jpeg_quality)],
                    )
                    if not ok:
                        self.publish_status(result.camera_id, error="jpeg_encode_failed")
                        return
                    payload = jpeg.tobytes()
                    header = f"bev:{result.camera_id}".encode("utf-8")
                    framed = bytes([len(header)]) + header + payload
                    self.ws.broadcast_sync(framed)
                try:
                    # Lightweight visibility that a BEV frame was queued for broadcast
                    if hasattr(self.ws, "logger") and self.ws.logger:
                        self.ws.logger.debug(
                            "BEV publish %s: %sx%s, points=%d, overlay=%s",
                            result.camera_id,
                            int(result.width_px),
                            int(result.height_px),
                            len(result.bev_points),
                            result.config.overlay,
                        )
                except Exception:
                    pass
        except Exception as e:
            print("BEV publish failed:", e)
            return
