from __future__ import annotations

import logging
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Any, Hashable

import cv2
import numpy as np
import time
import math
import threading

from geometry.homography import Plane, parse_extrinsics, ray_from_pixel, intersect_plane
from noesis.telemetry.motion_smoothing import MotionGatedAlphaBetaSmoother, MotionSmoothingConfig
from noesis.telemetry.person_ground_state import HumanGroundConfig, commit_path_point

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
    max_distance_m: float = 0.0


@dataclass(frozen=True)
class FloorplanSpace:
    x_range: Tuple[float, float]
    z_range: Tuple[float, float]
    grid_shape: Optional[Tuple[int, int]] = None
    grid_res_m: Optional[float] = None
    frame: Optional[str] = None
    units: Optional[str] = None
    snapshot_ts_us: Optional[int] = None
    floorplan_ts_us: Optional[int] = None
    ray_to_floorplan_alignment: Optional[Mapping[str, Any]] = None
    source: str = "active_floorplan"


@dataclass
class Footpoint:
    u: float
    v: float
    method: str = "bbox"
    stable_id: Optional[int] = None
    tracker_id: Optional[int] = None
    world_x: Optional[float] = None
    world_z: Optional[float] = None
    depth_m: Optional[float] = None
    depth_source: Optional[str] = None
    anchor_source: Optional[str] = None
    anchor_quality: Optional[str] = None
    anchor_reason: Optional[str] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    image_size: Optional[Tuple[int, int]] = None
    frame_id: Optional[int] = None
    motion_mode: Optional[str] = None
    posture: Optional[str] = None
    trail_append_allowed: Optional[bool] = None
    idle_jitter_m: Optional[float] = None
    debug: Dict[str, Any] = field(default_factory=dict)


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
    tracker_id: Optional[int] = None
    display_key: Optional[int] = None
    ema_x: Optional[float] = None
    ema_z: Optional[float] = None
    ema_ts: float = 0.0


@dataclass
class BevResult:
    camera_id: str
    bev_bgr: Optional[np.ndarray]
    bev_points: List[Dict[str, Any]]
    backend_trails: List[Dict[str, Any]]
    config: BevConfig
    timestamp_us: int
    width_px: int
    height_px: int
    points_smoothed: bool = False
    bounds_source: str = "config"
    floorplan_space: Optional[FloorplanSpace] = None
    dropped_footpoints: List[Dict[str, Any]] = field(default_factory=list)


class HomographyCache:
    def __init__(self) -> None:
        self._cache: Dict[str, np.ndarray] = {}

    @staticmethod
    def _hash_matrix(matrix: np.ndarray) -> int:
        return hash(matrix.tobytes())

    def _key(
        self,
        calib: CalibrationSnapshot,
        flip_u: bool = False,
        flip_v: bool = False,
    ) -> str:
        return "::".join(
            [
                calib.camera_id,
                f"{calib.image_size[0]}x{calib.image_size[1]}",
                f"K={self._hash_matrix(calib.intrinsics)}",
                f"E={hash(tuple(float(x) for x in calib.extrinsics_col_major))}",
                f"floor={calib.floor_y:.4f}",
                f"flip_u={int(bool(flip_u))}",
                f"flip_v={int(bool(flip_v))}",
            ]
        )

    def get(
        self,
        calib: CalibrationSnapshot,
        *,
        flip_u: bool = False,
        flip_v: bool = False,
    ) -> np.ndarray:
        key = self._key(calib, flip_u=flip_u, flip_v=flip_v)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        from geometry.homography import img_to_plane_homography
        H = img_to_plane_homography(
            calib.intrinsics,
            calib.extrinsics_col_major,
            calib.floor_y,
            calib.image_size,
            1.0,
            flip_u=flip_u,
            flip_v=flip_v,
        )
        self._cache[key] = H
        return H


class BevRenderer:
    def __init__(
        self,
        ws_server,
        trails_cfg: Optional[Dict[str, Any]] = None,
        smoothing_cfg: Optional[Dict[str, Any]] = None,
        frame: str = "menon_scene",
        depth_sampler: Optional[Callable[[str, float, float, int], Optional[Mapping[str, Any]]]] = None,
        floorplan_bounds_provider: Optional[Callable[[str], Optional[Mapping[str, Any]]]] = None,
    ) -> None:
        self.ws = ws_server
        self._lock = threading.Lock()
        self.config_per_cam: Dict[str, BevConfig] = {}
        self.h_cache = HomographyCache()
        self._frame_mode = self._normalize_frame_mode(frame)
        # Last known good homography per camera for resilience
        self._last_h_by_cam: Dict[str, np.ndarray] = {}
        # Auto-computed extents per camera
        self._auto_extents_by_camera: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
        # Image axis flip cache (per camera/extrinsics) for BEV homography alignment.
        self._image_flip_by_key: Dict[str, Tuple[bool, bool]] = {}
        self._image_flip_logged: set[str] = set()
        # Trail state (rendered into BEV images)
        self._trail_cfg = BevTrailConfig.from_mapping(trails_cfg or {})
        self._trails_enabled = bool(self._trail_cfg.enabled)
        self._trail_tracks_by_cam: Dict[str, Dict[Hashable, _BevTrailTrackState]] = {}
        self._trail_frame_counts: Dict[str, int] = {}
        self._trail_color_cache: Dict[int, Tuple[int, int, int]] = {}
        self._smoothing_cfg = MotionSmoothingConfig.from_mapping(smoothing_cfg or {})
        self._smoother = MotionGatedAlphaBetaSmoother(self._smoothing_cfg)
        self._smoother_source_by_key: Dict[Hashable, str] = {}
        self._path_cfg = HumanGroundConfig(
            max_speed_mps=float(self._smoothing_cfg.max_speed_mps),
            max_jump_m=float(self._smoothing_cfg.max_jump_m),
            path_min_step_m=max(0.03, float(self._trail_cfg.min_step_px) * 0.05),
            path_simplify_epsilon_m=0.06,
            path_max_points=max(2, int(self._trail_cfg.max_points_per_track)),
        )
        self._floorplan_space_signature_by_camera: Dict[str, Tuple[Any, ...]] = {}
        self._alignment_debug_enabled = str(
            os.environ.get("NOESIS_BEV_ALIGNMENT_DEBUG", os.environ.get("NOESIS_BEV_DEBUG", "0"))
        ).strip().lower() in ("1", "true", "yes", "y", "on")
        try:
            self._depth_fused_floor_agreement_m = float(
                os.environ.get("NOESIS_BEV_DEPTH_FUSED_MAX_RAY_DELTA_M", "0.75")
            )
        except Exception:
            self._depth_fused_floor_agreement_m = 0.75
        if not math.isfinite(self._depth_fused_floor_agreement_m) or self._depth_fused_floor_agreement_m < 0.0:
            self._depth_fused_floor_agreement_m = 0.75
        # Static floorplan/MapAnything snapshots are not live person-depth inputs.
        # Keep this sampler available only for explicit alignment diagnostics.
        self._depth_sampler = depth_sampler if self._alignment_debug_enabled else None
        self._floorplan_bounds_provider = floorplan_bounds_provider
        # JPEG BEV binary delivery retired (meta-only mode is the supported baseline per design decisions + contracts).
        # The flag/env plumbing remains in the runtime for transition but is ignored here.

    def set_trails_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._trails_enabled = bool(enabled)
            if not self._trails_enabled:
                self._trail_tracks_by_cam.clear()
                self._trail_frame_counts.clear()
                self._smoother_source_by_key.clear()

    @staticmethod
    def _floorplan_space_signature(
        *,
        bounds_source: str,
        x_range: Tuple[float, float],
        z_range: Tuple[float, float],
        floorplan_space: Optional[FloorplanSpace],
    ) -> Tuple[Any, ...]:
        if floorplan_space is None:
            return (
                str(bounds_source or "auto"),
                round(float(x_range[0]), 6),
                round(float(x_range[1]), 6),
                round(float(z_range[0]), 6),
                round(float(z_range[1]), 6),
                None,
                None,
                None,
                None,
            )
        return (
            str(bounds_source or floorplan_space.source or "active_floorplan"),
            round(float(floorplan_space.x_range[0]), 6),
            round(float(floorplan_space.x_range[1]), 6),
            round(float(floorplan_space.z_range[0]), 6),
            round(float(floorplan_space.z_range[1]), 6),
            tuple(int(v) for v in floorplan_space.grid_shape) if floorplan_space.grid_shape else None,
            str(floorplan_space.frame or ""),
            int(floorplan_space.snapshot_ts_us) if floorplan_space.snapshot_ts_us is not None else None,
            int(floorplan_space.floorplan_ts_us) if floorplan_space.floorplan_ts_us is not None else None,
        )

    def _reset_camera_motion_state_locked(self, camera_id: str) -> None:
        self._trail_tracks_by_cam.pop(camera_id, None)
        self._trail_frame_counts.pop(camera_id, None)
        for key in list(self._smoother_source_by_key.keys()):
            if isinstance(key, tuple) and key and key[0] == camera_id:
                self._smoother.reset(key)
                self._smoother_source_by_key.pop(key, None)

    @staticmethod
    def _normalize_frame_mode(value: Any) -> str:
        text = str(value or "").strip().lower()
        if text in ("world", "global", "world_frame", "menon_scene", "backend_world_m"):
            return "world"
        if text in ("camera", "camera_local", "camera_local_ground", "camera_local_ground_m", "local", "cam"):
            return "camera_local"
        if not text:
            return "world"
        return "camera_local"

    def _resolve_max_distance_scene(self, cfg: BevConfig, calib: CalibrationSnapshot) -> float:
        """Resolve distance guardrail in canonical meters."""
        limit = float(cfg.max_distance_m or 0.0)
        if not math.isfinite(limit) or limit <= 0.0:
            return 0.0
        return limit

    def _active_floorplan_ranges(
        self,
        camera_id: str,
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        space = self._active_floorplan_space(camera_id)
        if space is None:
            return None
        return space.x_range, space.z_range

    @staticmethod
    def _parse_grid_shape(payload: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
        raw_shape = payload.get("grid_shape", payload.get("gridShape"))
        if isinstance(raw_shape, Mapping):
            raw_rows = raw_shape.get("rows", raw_shape.get("height"))
            raw_cols = raw_shape.get("cols", raw_shape.get("columns", raw_shape.get("width")))
            raw_shape = [raw_rows, raw_cols]
        if isinstance(raw_shape, (list, tuple)) and len(raw_shape) >= 2:
            try:
                rows = int(raw_shape[0])
                cols = int(raw_shape[1])
                if rows > 0 and cols > 0:
                    return rows, cols
            except Exception:
                return None
        try:
            rows = int(payload.get("grid_rows", payload.get("rows", 0)) or 0)
            cols = int(payload.get("grid_cols", payload.get("cols", 0)) or 0)
            if rows > 0 and cols > 0:
                return rows, cols
        except Exception:
            return None
        return None

    def _active_floorplan_space(
        self,
        camera_id: str,
    ) -> Optional[FloorplanSpace]:
        provider = self._floorplan_bounds_provider
        if provider is None:
            return None
        try:
            bounds_payload = provider(str(camera_id))
        except Exception:
            logger.debug("BEV: active floorplan bounds provider failed for %s", camera_id, exc_info=True)
            return None
        if not isinstance(bounds_payload, Mapping):
            return None
        raw_bounds = bounds_payload.get("bounds") if isinstance(bounds_payload.get("bounds"), Mapping) else bounds_payload
        if not isinstance(raw_bounds, Mapping):
            return None
        try:
            x_min = float(raw_bounds.get("min_x"))
            x_max = float(raw_bounds.get("max_x"))
            z_min = float(raw_bounds.get("min_z"))
            z_max = float(raw_bounds.get("max_z"))
        except Exception:
            return None
        if not all(math.isfinite(v) for v in (x_min, x_max, z_min, z_max)):
            return None
        if x_max <= x_min or z_max <= z_min:
            return None
        grid_shape = self._parse_grid_shape(bounds_payload)
        grid_res_m: Optional[float] = None
        try:
            raw_grid_res = bounds_payload.get("grid_res_m", bounds_payload.get("gridResM"))
            parsed_grid_res = float(raw_grid_res)
            if math.isfinite(parsed_grid_res) and parsed_grid_res > 0.0:
                grid_res_m = float(parsed_grid_res)
        except Exception:
            grid_res_m = None

        def _optional_int(name: str) -> Optional[int]:
            try:
                value = bounds_payload.get(name)
                if value is None:
                    return None
                parsed = int(value)
                return parsed if parsed > 0 else None
            except Exception:
                return None

        return FloorplanSpace(
            x_range=(float(x_min), float(x_max)),
            z_range=(float(z_min), float(z_max)),
            grid_shape=grid_shape,
            grid_res_m=grid_res_m,
            frame=str(bounds_payload.get("frame") or "") or None,
            units=str(bounds_payload.get("units") or "") or None,
            snapshot_ts_us=_optional_int("snapshot_ts_us"),
            floorplan_ts_us=_optional_int("floorplan_ts_us"),
            ray_to_floorplan_alignment=(
                dict(bounds_payload.get("ray_to_floorplan_alignment"))
                if isinstance(bounds_payload.get("ray_to_floorplan_alignment"), Mapping)
                else None
            ),
            source=str(bounds_payload.get("source") or "active_floorplan"),
        )

    @staticmethod
    def _apply_image_flip(
        u: float,
        v: float,
        width: int,
        height: int,
        flip_u: bool,
        flip_v: bool,
    ) -> Tuple[float, float]:
        if flip_u:
            u = float(max(0, width - 1)) - float(u)
        if flip_v:
            v = float(max(0, height - 1)) - float(v)
        return float(u), float(v)

    def _infer_image_flips(self, calib: CalibrationSnapshot) -> Tuple[bool, bool]:
        # Vestigial after menon_world_unification / DS8 design decisions (flips retired from canonical path).
        # If this ever returns non-(False, False) in the future, something has gone wrong with the unification.
        # We keep the machinery for now but force the safe no-flip result.
        return False, False

    @staticmethod
    def _world_to_camera_local_ground(
        world_x: float,
        world_y: float,
        world_z: float,
        R_wc: np.ndarray,
        C_world: np.ndarray,
    ) -> Tuple[float, float]:
        delta = np.array(
            [float(world_x) - float(C_world[0]), float(world_y) - float(C_world[1]), float(world_z) - float(C_world[2])],
            dtype=np.float64,
        )
        local = R_wc.T @ delta
        return float(local[0]), float(local[2])

    @staticmethod
    def _image_to_world_ground(
        H_img2plane: np.ndarray,
        u: float,
        v: float,
    ) -> Optional[Tuple[float, float]]:
        try:
            vec = np.array([float(u), float(v), 1.0], dtype=np.float64)
            world_pt = H_img2plane @ vec
            w = float(world_pt[2]) if float(world_pt[2]) else 1.0
            wx = float(world_pt[0] / w)
            wz = float(world_pt[1] / w)
            if not math.isfinite(wx) or not math.isfinite(wz):
                return None
            return float(wx), float(wz)
        except Exception:
            return None

    @staticmethod
    def _image_depth_to_camera_local(
        calib: CalibrationSnapshot,
        u: float,
        v: float,
        depth_m: float,
    ) -> Optional[Tuple[float, float]]:
        try:
            depth = float(depth_m)
            if not math.isfinite(depth) or depth <= 0.05 or depth > 50.0:
                return None
            k = np.asarray(calib.intrinsics, dtype=np.float64).reshape(3, 3)
            fx = float(k[0, 0])
            cx = float(k[0, 2])
            if not math.isfinite(fx) or abs(fx) <= 1e-9:
                return None
            x_local = (float(u) - cx) * depth / fx
            z_local = depth
            if not math.isfinite(x_local) or not math.isfinite(z_local):
                return None
            return float(x_local), float(z_local)
        except Exception:
            return None

    def _sample_debug_floorplan_depth_to_camera_local(
        self,
        camera_id: str,
        calib: CalibrationSnapshot,
        u: float,
        v: float,
        timestamp_us: int,
    ) -> Optional[Tuple[float, float, Mapping[str, Any]]]:
        sampler = self._depth_sampler
        if sampler is None:
            return None
        try:
            sample = sampler(str(camera_id), float(u), float(v), int(timestamp_us))
        except Exception:
            logger.debug("BEV diagnostic floorplan depth sampler failed for %s", camera_id, exc_info=True)
            return None
        if not isinstance(sample, Mapping):
            return None
        raw_depth = sample.get("depth_m", sample.get("depth"))
        try:
            depth_m = float(raw_depth)
        except Exception:
            return None
        if not math.isfinite(depth_m) or depth_m <= 0.05 or depth_m >= 50.0:
            return None
        local = self._image_depth_to_camera_local(calib, float(u), float(v), depth_m)
        if local is None:
            return None
        return float(local[0]), float(local[1]), sample

    @staticmethod
    def _point_in_metric_bounds(
        x: float,
        z: float,
        x_range: Tuple[float, float],
        z_range: Tuple[float, float],
    ) -> bool:
        return (
            math.isfinite(float(x))
            and math.isfinite(float(z))
            and float(x_range[0]) <= float(x) <= float(x_range[1])
            and float(z_range[0]) <= float(z) <= float(z_range[1])
        )

    @staticmethod
    def _metric_to_grid_cell(
        x: float,
        z: float,
        x_range: Tuple[float, float],
        z_range: Tuple[float, float],
        width_px: int,
        height_px: int,
    ) -> Optional[Tuple[int, int]]:
        if width_px <= 0 or height_px <= 0:
            return None
        span_x = max(1e-6, float(x_range[1]) - float(x_range[0]))
        span_z = max(1e-6, float(z_range[1]) - float(z_range[0]))
        col = int(math.floor(((float(x) - float(x_range[0])) / span_x) * float(width_px)))
        row = int(math.floor((1.0 - ((float(z) - float(z_range[0])) / span_z)) * float(height_px)))
        if col < 0 or col >= width_px or row < 0 or row >= height_px:
            return None
        return int(col), int(row)

    @staticmethod
    def _floorplan_point_fields(
        x: float,
        z: float,
        x_range: Tuple[float, float],
        z_range: Tuple[float, float],
        width_px: int,
        height_px: int,
        floorplan_space: Optional[FloorplanSpace] = None,
    ) -> Dict[str, Any]:
        span_x = max(1e-6, float(x_range[1]) - float(x_range[0]))
        span_z = max(1e-6, float(z_range[1]) - float(z_range[0]))
        norm_x = (float(x) - float(x_range[0])) / span_x
        norm_z = (float(z) - float(z_range[0])) / span_z
        norm_y = 1.0 - norm_z
        inside = (
            math.isfinite(norm_x)
            and math.isfinite(norm_z)
            and 0.0 <= norm_x <= 1.0
            and 0.0 <= norm_z <= 1.0
        )
        rows = int(height_px) if int(height_px) > 0 else 0
        cols = int(width_px) if int(width_px) > 0 else 0
        grid_source = "bev_canvas"
        if floorplan_space is not None and floorplan_space.grid_shape is not None:
            rows = int(floorplan_space.grid_shape[0])
            cols = int(floorplan_space.grid_shape[1])
            grid_source = "active_floorplan"
        grid_row: Optional[int] = None
        grid_col: Optional[int] = None
        if inside and rows > 0 and cols > 0:
            grid_col = int(min(cols - 1, max(0, math.floor(norm_x * float(cols)))))
            grid_row = int(min(rows - 1, max(0, math.floor(norm_y * float(rows)))))
        grid_cell = [int(grid_col), int(grid_row)] if grid_col is not None and grid_row is not None else []
        return {
            "floorplanX": float(x),
            "floorplanZ": float(z),
            "normX": float(norm_x),
            "normY": float(norm_y),
            "normZ": float(norm_z),
            "floorplanInside": bool(inside),
            "gridCol": grid_col,
            "gridRow": grid_row,
            "gridCell": grid_cell,
            "gridRows": int(rows) if rows > 0 else None,
            "gridCols": int(cols) if cols > 0 else None,
            "gridSource": grid_source,
        }

    @staticmethod
    def _floorplan_alignment_matrix(floorplan_space: Optional[FloorplanSpace]) -> Optional[np.ndarray]:
        if floorplan_space is None or not isinstance(floorplan_space.ray_to_floorplan_alignment, Mapping):
            return None
        payload = floorplan_space.ray_to_floorplan_alignment
        if str(payload.get("quality") or "").strip().lower() != "ok":
            return None
        try:
            sample_count = int(payload.get("inlier_count", payload.get("sample_count", 0)) or 0)
        except Exception:
            sample_count = 0
        try:
            min_samples = int(os.environ.get("NOESIS_BEV_FLOORPLAN_ALIGNMENT_MIN_SAMPLES", "64") or "64")
        except Exception:
            min_samples = 64
        if sample_count < max(16, min_samples):
            return None
        residual = payload.get("residual_m")
        if isinstance(residual, Mapping):
            try:
                p50 = float(residual.get("p50", 0.0) or 0.0)
                p95 = float(residual.get("p95", 0.0) or 0.0)
                max_p50 = float(os.environ.get("NOESIS_BEV_FLOORPLAN_ALIGNMENT_MAX_P50_M", "0.35") or "0.35")
                max_p95 = float(os.environ.get("NOESIS_BEV_FLOORPLAN_ALIGNMENT_MAX_P95_M", "1.25") or "1.25")
                if not math.isfinite(p50) or not math.isfinite(p95) or p50 > max_p50 or p95 > max_p95:
                    return None
            except Exception:
                return None
        matrix_raw = payload.get("matrix_2x3")
        try:
            matrix = np.asarray(matrix_raw, dtype=np.float64)
        except Exception:
            return None
        if matrix.shape != (2, 3) or not np.all(np.isfinite(matrix)):
            return None
        try:
            det = float(np.linalg.det(matrix[:, :2]))
            if not math.isfinite(det) or abs(det) < 0.05 or abs(det) > 20.0:
                return None
        except Exception:
            return None
        return matrix

    @staticmethod
    def _apply_floorplan_alignment(
        matrix: Optional[np.ndarray],
        x: float,
        z: float,
    ) -> Tuple[float, float, bool]:
        if matrix is None:
            return float(x), float(z), False
        vec = np.array([float(x), float(z), 1.0], dtype=np.float64)
        out = matrix @ vec
        ax = float(out[0])
        az = float(out[1])
        if not math.isfinite(ax) or not math.isfinite(az):
            return float(x), float(z), False
        return ax, az, True

    @staticmethod
    def _json_number(value: Any) -> Optional[float]:
        try:
            out = float(value)
        except Exception:
            return None
        if not math.isfinite(out):
            return None
        return float(out)

    def _candidate_anchors_for_footpoint(self, fp: Footpoint, calib: CalibrationSnapshot) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        seen: set[Tuple[str, int, int]] = set()

        def _add(name: str, u: Any, v: Any, *, source: str, priority: int) -> None:
            uu = self._json_number(u)
            vv = self._json_number(v)
            if uu is None or vv is None:
                return
            key = (str(name), int(round(uu * 10.0)), int(round(vv * 10.0)))
            if key in seen:
                return
            seen.add(key)
            candidates.append(
                {
                    "name": str(name),
                    "source": str(source),
                    "priority": int(priority),
                    "u": float(uu),
                    "v": float(vv),
                }
            )

        _add(str(fp.method or "active_anchor"), fp.u, fp.v, source="active_footpoint", priority=0)

        debug = fp.debug if isinstance(fp.debug, Mapping) else {}
        raw_candidates = debug.get("image_candidates")
        if isinstance(raw_candidates, list):
            for idx, item in enumerate(raw_candidates):
                if not isinstance(item, Mapping):
                    continue
                _add(
                    str(item.get("name") or item.get("source") or f"candidate_{idx}"),
                    item.get("u"),
                    item.get("v"),
                    source=str(item.get("source") or "track"),
                    priority=int(item.get("priority", 10 + idx) or (10 + idx)),
                )

        if fp.bbox is not None:
            try:
                left, top, width, height = [float(x) for x in fp.bbox[:4]]
                if width > 0.0 and height > 0.0:
                    cx = left + width * 0.5
                    bottom = top + height
                    _add("bbox_bottom_center", cx, bottom, source="bbox", priority=20)
                    _add("bbox_bottom_left_quarter", left + width * 0.30, bottom, source="bbox", priority=21)
                    _add("bbox_bottom_right_quarter", left + width * 0.70, bottom, source="bbox", priority=22)
                    _add("bbox_lower_center_95", cx, top + height * 0.95, source="bbox", priority=23)
            except Exception:
                pass

        image_w, image_h = calib.image_size
        for item in candidates:
            u = float(item["u"])
            v = float(item["v"])
            item["clipped"] = False
            if image_w > 0:
                clipped_u = min(max(u, 0.0), float(max(0, image_w - 1)))
                item["clipped"] = bool(item["clipped"] or abs(clipped_u - u) > 1e-6)
                item["u"] = float(clipped_u)
            if image_h > 0:
                clipped_v = min(max(v, 0.0), float(max(0, image_h - 1)))
                item["clipped"] = bool(item["clipped"] or abs(clipped_v - v) > 1e-6)
                item["v"] = float(clipped_v)

        candidates.sort(key=lambda item: int(item.get("priority", 999)))
        return candidates

    def _build_alignment_debug(
        self,
        *,
        camera_id: str,
        calib: CalibrationSnapshot,
        fp: Footpoint,
        H_img2plane: np.ndarray,
        R_wc: np.ndarray,
        C_world: np.ndarray,
        x_range: Tuple[float, float],
        z_range: Tuple[float, float],
        width_px: int,
        height_px: int,
        floorplan_space: Optional[FloorplanSpace],
        floorplan_alignment: Optional[np.ndarray],
        timestamp_us: int,
        chosen_x: Optional[float],
        chosen_z: Optional[float],
        display_source: str,
        selection_debug: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        candidates_out: List[Dict[str, Any]] = []
        for anchor in self._candidate_anchors_for_footpoint(fp, calib):
            u = float(anchor["u"])
            v = float(anchor["v"])
            item: Dict[str, Any] = {
                "name": str(anchor.get("name") or ""),
                "source": str(anchor.get("source") or ""),
                "u": float(u),
                "v": float(v),
                "clipped": bool(anchor.get("clipped", False)),
            }
            image_world = self._image_to_world_ground(H_img2plane, u, v)
            if image_world is not None:
                iw_x, iw_z = image_world
                ray_x, ray_z = self._world_to_camera_local_ground(
                    float(iw_x),
                    float(calib.floor_y),
                    float(iw_z),
                    R_wc,
                    C_world,
                )
                aligned_x, aligned_z, alignment_applied = self._apply_floorplan_alignment(
                    floorplan_alignment,
                    ray_x,
                    ray_z,
                )
                item["rayFloor"] = {
                    "x": float(aligned_x),
                    "z": float(aligned_z),
                    "rawX": float(ray_x),
                    "rawZ": float(ray_z),
                    "alignmentApplied": bool(alignment_applied),
                    "insideBounds": self._point_in_metric_bounds(aligned_x, aligned_z, x_range, z_range),
                }
                item["rayFloor"].update(
                    self._floorplan_point_fields(
                        aligned_x,
                        aligned_z,
                        x_range,
                        z_range,
                        width_px,
                        height_px,
                        floorplan_space,
                    )
                )
            floorplan_depth = self._sample_debug_floorplan_depth_to_camera_local(
                camera_id,
                calib,
                u,
                v,
                int(timestamp_us),
            )
            if floorplan_depth is not None:
                depth_x, depth_z, sample_meta = floorplan_depth
                item["mapanythingDepth"] = {
                    "x": float(depth_x),
                    "z": float(depth_z),
                    "insideBounds": self._point_in_metric_bounds(depth_x, depth_z, x_range, z_range),
                    "sample": {
                        key: value
                        for key, value in dict(sample_meta).items()
                        if key not in ("depth", "conf", "mask")
                    },
                }
                item["mapanythingDepth"].update(
                    self._floorplan_point_fields(
                        depth_x,
                        depth_z,
                        x_range,
                        z_range,
                        width_px,
                        height_px,
                        floorplan_space,
                    )
                )
                ray = item.get("rayFloor")
                if isinstance(ray, Mapping):
                    try:
                        item["depthVsRayDeltaM"] = float(
                            math.hypot(float(depth_x) - float(ray["x"]), float(depth_z) - float(ray["z"]))
                        )
                    except Exception:
                        pass
            candidates_out.append(item)

        registered_depth_anchor: Optional[Dict[str, Any]] = None
        if fp.depth_m is not None:
            depth_local = self._image_depth_to_camera_local(calib, float(fp.u), float(fp.v), float(fp.depth_m))
            if depth_local is not None:
                depth_x, depth_z = depth_local
                registered_depth_anchor = {
                    "x": float(depth_x),
                    "z": float(depth_z),
                    "depthM": float(fp.depth_m),
                    "depthSource": str(fp.depth_source or ""),
                    "insideBounds": self._point_in_metric_bounds(depth_x, depth_z, x_range, z_range),
                }
                registered_depth_anchor.update(
                    self._floorplan_point_fields(
                        depth_x,
                        depth_z,
                        x_range,
                        z_range,
                        width_px,
                        height_px,
                        floorplan_space,
                    )
                )

        chosen: Dict[str, Any] = {
            "displaySource": str(display_source),
            "x": float(chosen_x) if chosen_x is not None and math.isfinite(float(chosen_x)) else None,
            "z": float(chosen_z) if chosen_z is not None and math.isfinite(float(chosen_z)) else None,
        }
        if chosen["x"] is not None and chosen["z"] is not None:
            chosen["insideBounds"] = self._point_in_metric_bounds(float(chosen["x"]), float(chosen["z"]), x_range, z_range)
            chosen.update(
                self._floorplan_point_fields(
                    float(chosen["x"]),
                    float(chosen["z"]),
                    x_range,
                    z_range,
                    width_px,
                    height_px,
                    floorplan_space,
                )
            )

        track_debug = dict(fp.debug) if isinstance(fp.debug, Mapping) else {}
        track_debug.pop("image_candidates", None)
        payload = {
            "enabled": True,
            "track": track_debug,
            "activeAnchor": {"u": float(fp.u), "v": float(fp.v), "method": str(fp.method or "")},
            "bbox": list(fp.bbox) if fp.bbox is not None else None,
            "imageSize": list(fp.image_size) if fp.image_size is not None else list(calib.image_size),
            "frameId": int(fp.frame_id) if fp.frame_id is not None else None,
            "candidates": candidates_out,
            "chosen": chosen,
        }
        if registered_depth_anchor is not None:
            payload["registeredDepthAnchor"] = registered_depth_anchor
        if isinstance(selection_debug, Mapping):
            payload["displaySelection"] = dict(selection_debug)
        if floorplan_alignment is not None and isinstance(floorplan_space, FloorplanSpace):
            payload["floorplanAlignment"] = {
                "quality": (
                    floorplan_space.ray_to_floorplan_alignment.get("quality")
                    if isinstance(floorplan_space.ray_to_floorplan_alignment, Mapping)
                    else None
                ),
                "reason": (
                    floorplan_space.ray_to_floorplan_alignment.get("reason")
                    if isinstance(floorplan_space.ray_to_floorplan_alignment, Mapping)
                    else None
                ),
                "snapshotTsUs": floorplan_space.snapshot_ts_us,
            }
        return payload

    def _select_floor_contact_ray(
        self,
        *,
        fp: Footpoint,
        calib: CalibrationSnapshot,
        H_img2plane: np.ndarray,
        R_wc: np.ndarray,
        C_world: np.ndarray,
        x_range: Tuple[float, float],
        z_range: Tuple[float, float],
        floorplan_alignment: Optional[np.ndarray],
    ) -> Optional[Tuple[float, float, str, float, float, bool]]:
        for anchor in self._candidate_anchors_for_footpoint(fp, calib):
            try:
                u = float(anchor["u"])
                v = float(anchor["v"])
            except Exception:
                continue
            image_world = self._image_to_world_ground(H_img2plane, u, v)
            if image_world is None:
                continue
            iw_x, iw_z = image_world
            px, pz = self._world_to_camera_local_ground(
                float(iw_x),
                float(calib.floor_y),
                float(iw_z),
                R_wc,
                C_world,
            )
            if not math.isfinite(px) or not math.isfinite(pz):
                continue
            aligned_x, aligned_z, alignment_applied = self._apply_floorplan_alignment(
                floorplan_alignment,
                px,
                pz,
            )
            if not math.isfinite(aligned_x) or not math.isfinite(aligned_z):
                continue
            selected = (
                float(aligned_x),
                float(aligned_z),
                str(anchor.get("name") or fp.method or "image_anchor"),
                float(px),
                float(pz),
                bool(alignment_applied),
            )
            if self._point_in_metric_bounds(float(aligned_x), float(aligned_z), x_range, z_range):
                return selected
        return None

    @staticmethod
    def _world_source_is_live_tracking(anchor_source: Optional[str]) -> bool:
        source = str(anchor_source or "").strip().lower()
        return source in {
            "bbox3d",
            "pose_depth_fused",
            "person_anchor_depth_fused",
            "pose_floor_only",
            "person_anchor_floor_only",
            "gravity_drop",
        }

    @staticmethod
    def _world_source_is_depth_fused(anchor_source: Optional[str]) -> bool:
        source = str(anchor_source or "").strip().lower()
        return source in {
            "bbox3d",
            "pose_depth_fused",
            "person_anchor_depth_fused",
        }

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

    def _history_identity(
        self,
        *,
        stable_id: Optional[int],
        tracker_id: Optional[int],
    ) -> Optional[Tuple[str, int]]:
        if tracker_id is not None and int(tracker_id) >= 0:
            return ("tracker", int(tracker_id))
        if stable_id is not None and int(stable_id) > 0:
            return ("stable", int(stable_id))
        return None

    def _display_color_key(
        self,
        *,
        stable_id: Optional[int],
        tracker_id: Optional[int],
    ) -> Optional[int]:
        if self._trail_cfg.color_key == "stable_id":
            if stable_id is not None and int(stable_id) > 0:
                return int(stable_id)
            if tracker_id is not None and int(tracker_id) >= 0:
                return int(tracker_id)
            return None
        if tracker_id is not None and int(tracker_id) >= 0:
            return int(tracker_id)
        if stable_id is not None and int(stable_id) > 0:
            return int(stable_id)
        return None

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

    def _compute_auto_extents_world(
        self,
        calib: CalibrationSnapshot,
        cfg: BevConfig,
        flip_u: bool,
        flip_v: bool,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Compute world-frame BEV extents so footpoints and ranges share the same frame."""
        width, height = calib.image_size
        if width <= 0 or height <= 0:
            return (-4.0, 4.0), (0.0, 12.0)

        R_wc, C_world = parse_extrinsics(calib.extrinsics_col_major)
        meters_per_scene = 1.0
        plane = Plane.horizontal(float(calib.floor_y))

        xs = np.linspace(0, max(0.0, float(width - 1)), 8)
        ys = np.linspace(0, max(0.0, float(height - 1)), 8)
        hits_world: List[Tuple[float, float]] = []
        for u in xs:
            for v in ys:
                origin, direction = ray_from_pixel(float(u), float(v), calib.intrinsics, R_wc, C_world)
                hit = intersect_plane(origin, direction, plane)
                if hit is None:
                    continue
                dx = float(hit[0] - C_world[0])
                dz = float(hit[2] - C_world[2])
                dist = math.hypot(dx, dz)
                max_distance_scene = float(self._resolve_max_distance_scene(cfg, calib))
                if max_distance_scene > 0.0 and dist > max_distance_scene and dist > 1e-6:
                    scale_d = max_distance_scene / dist
                    dx *= scale_d
                    dz *= scale_d
                wx = float(C_world[0] + dx)
                wz = float(C_world[2] + dz)
                hits_world.append((wx, wz))

        if len(hits_world) < 3:
            return (-4.0, 4.0), (0.0, 12.0)

        xs_world = [p[0] for p in hits_world]
        zs_world = [p[1] for p in hits_world]
        padding = 0.25
        x_min = min(xs_world) - padding
        x_max = max(xs_world) + padding
        z_min = min(zs_world) - padding
        z_max = max(zs_world) + padding

        if (x_max - x_min) < 1.0:
            cx = 0.5 * (x_min + x_max)
            x_min = cx - 0.5
            x_max = cx + 0.5
        if (z_max - z_min) < 1.0:
            cz = 0.5 * (z_min + z_max)
            z_min = max(0.0, cz - 0.5)
            z_max = cz + 0.5

        return (x_min, x_max), (z_min, z_max)

    def _compute_auto_extents_local(
        self,
        calib: CalibrationSnapshot,
        cfg: BevConfig,
        flip_u: bool,
        flip_v: bool,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Compute camera-local BEV extents so footpoints and ranges share the same frame."""
        width, height = calib.image_size
        if width <= 0 or height <= 0:
            return (-4.0, 4.0), (0.0, 12.0)

        R_wc, C_world = parse_extrinsics(calib.extrinsics_col_major)
        meters_per_scene = 1.0
        plane = Plane.horizontal(float(calib.floor_y))

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
                max_distance_scene = float(self._resolve_max_distance_scene(cfg, calib))
                if max_distance_scene > 0.0 and dist > max_distance_scene and dist > 1e-6:
                    scale_d = max_distance_scene / dist
                    dx *= scale_d
                    dz *= scale_d
                wx = float(C_world[0] + dx)
                wz = float(C_world[2] + dz)
                lx, lz = self._world_to_camera_local_ground(wx, float(calib.floor_y), wz, R_wc, C_world)
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
            max_distance_m=float(
                cfg.get(
                    "maxDistanceM",
                    cfg.get("maxDistanceScene", cfg.get("maxDistance", current.max_distance_m)),
                )
            ),
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
        if timestamp_us <= 0:
            timestamp_us = int(time.time() * 1_000_000)
        now_s = float(timestamp_us) / 1_000_000.0
        cfg = self.config_per_cam.get(camera_id, BevConfig())
        footpoints = footpoints or []

        frame_mode = self._frame_mode
        use_world_frame = frame_mode == "world"
        flip_u, flip_v = self._infer_image_flips(calib)

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
        bounds_source = "config"
        active_floorplan_space: Optional[FloorplanSpace] = None

        if cfg.auto_fit_extents:
            bounds_source = "auto_extents"
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
                    str(frame_mode),
                ]
            )

            auto_extents = self._auto_extents_by_camera.get(key)
            if auto_extents is None:
                if use_world_frame:
                    x_range_auto, z_range_auto = self._compute_auto_extents_world(calib, cfg, flip_u, flip_v)
                else:
                    x_range_auto, z_range_auto = self._compute_auto_extents_local(calib, cfg, flip_u, flip_v)
                auto_extents = (x_range_auto, z_range_auto)
                self._auto_extents_by_camera[key] = auto_extents
            x_range, z_range = auto_extents

        if not use_world_frame:
            active_floorplan_space = self._active_floorplan_space(camera_id)
            if active_floorplan_space is not None:
                x_range, z_range = active_floorplan_space.x_range, active_floorplan_space.z_range
                bounds_source = str(active_floorplan_space.source or "active_floorplan")
        floorplan_alignment = self._floorplan_alignment_matrix(active_floorplan_space)
        if not use_world_frame:
            space_signature = self._floorplan_space_signature(
                bounds_source=bounds_source,
                x_range=x_range,
                z_range=z_range,
                floorplan_space=active_floorplan_space,
            )
            with self._lock:
                previous_signature = self._floorplan_space_signature_by_camera.get(camera_id)
                if previous_signature is not None and previous_signature != space_signature:
                    self._reset_camera_motion_state_locked(camera_id)
                self._floorplan_space_signature_by_camera[camera_id] = space_signature

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

        # JPEG BEV image rendering retired (see __init__ comment). We only ever produce the JSON metadata payload now.
        bev: Optional[np.ndarray] = None

        # Compute or reuse homography; fallback to last good if current fails
        H_img2plane = None
        try:
            H_img2plane = self.h_cache.get(calib, flip_u=flip_u, flip_v=flip_v)
            self._last_h_by_cam[camera_id] = H_img2plane
        except Exception as e:
            logger.warning("BEV: homography computation failed for %s: %s", camera_id, e)
            cached_result = self._last_h_by_cam.get(camera_id)
            if cached_result is None:
                # No homography at all yet; notify clients of failure once per attempt window.
                self.publish_status(camera_id, error="homography_failed", details=str(e))
                return
            H_img2plane = cached_result

        bev_points: List[Dict[str, Any]] = []
        raw_points: List[Dict[str, Any]] = []
        current_by_history: Dict[Hashable, Dict[str, Any]] = {}
        dropped_footpoints: List[Dict[str, Any]] = []

        R_wc, C_world = parse_extrinsics(calib.extrinsics_col_major)
        meters_per_scene = 1.0

        max_distance_m = float(self._resolve_max_distance_scene(cfg, calib))
        # In world mode the producer already owns the canonical filtered track.world state.
        # Do not low-pass filter those points again in the BEV renderer.
        apply_backend_smoothing = bool(self._smoother.enabled) and not use_world_frame

        for fp in footpoints:
            anchor_source = str(fp.anchor_source or "").strip().lower()
            if anchor_source == "anchor_hold":
                continue
            selection_debug: Optional[Dict[str, Any]] = None
            wx = wz = None
            if fp.world_x is not None and fp.world_z is not None:
                try:
                    wx_c = float(fp.world_x)
                    wz_c = float(fp.world_z)
                    if math.isfinite(wx_c) and math.isfinite(wz_c):
                        wx = wx_c
                        wz = wz_c
                except Exception:
                    wx = wz = None

            display_source = "world"
            if use_world_frame:
                # In world mode, use producer-owned track world coordinates only.
                # Do not reintroduce a homography fallback path here.
                if wx is None or wz is None:
                    continue
                if not math.isfinite(wx) or not math.isfinite(wz):
                    continue

                dx = wx - C_world[0]
                dz = wz - C_world[2]
                if max_distance_m > 0.0 and math.hypot(dx, dz) > max_distance_m:
                    # Guardrail: discard near-horizon outliers so the UI doesn't draw
                    # teleporting streaks outside the floorplan extents.
                    continue
                px = float(wx)
                pz = float(wz)
            else:
                method_key = str(fp.method or "").strip().lower()
                prefer_image_anchor = method_key in ("image_foot", "image_base", "pose_anchor", "person_anchor")
                prefer_floor_contact = method_key in ("image_foot", "image_base", "pose_anchor", "person_anchor", "bbox")
                px = pz = None
                registered_depth_candidate: Optional[Dict[str, Any]] = None
                if prefer_floor_contact and fp.depth_m is not None:
                    depth_local = self._image_depth_to_camera_local(
                        calib,
                        float(fp.u),
                        float(fp.v),
                        float(fp.depth_m),
                    )
                    if depth_local is not None:
                        depth_x, depth_z = depth_local
                        registered_depth_candidate = {
                            "x": float(depth_x),
                            "z": float(depth_z),
                            "depthM": float(fp.depth_m),
                            "depthSource": str(fp.depth_source or ""),
                            "insideBounds": self._point_in_metric_bounds(depth_x, depth_z, x_range, z_range),
                        }
                floor_contact: Optional[Tuple[float, float, str, float, float, bool]] = None
                floor_contact_candidate: Optional[Dict[str, Any]] = None
                if prefer_floor_contact:
                    floor_contact = self._select_floor_contact_ray(
                        fp=fp,
                        calib=calib,
                        H_img2plane=H_img2plane,
                        R_wc=R_wc,
                        C_world=C_world,
                        x_range=x_range,
                        z_range=z_range,
                        floorplan_alignment=floorplan_alignment,
                    )
                    if floor_contact is not None:
                        floor_contact_candidate = {
                            "x": float(floor_contact[0]),
                            "z": float(floor_contact[1]),
                            "anchor": str(floor_contact[2]),
                            "rawX": float(floor_contact[3]),
                            "rawZ": float(floor_contact[4]),
                            "alignmentApplied": bool(floor_contact[5]),
                            "insideBounds": True,
                        }

                world_candidate: Optional[Dict[str, Any]] = None
                if (
                    wx is not None
                    and wz is not None
                    and self._world_source_is_live_tracking(fp.anchor_source)
                ):
                    candidate_x, candidate_z = self._world_to_camera_local_ground(
                        float(wx),
                        float(calib.floor_y),
                        float(wz),
                        R_wc,
                        C_world,
                    )
                    world_inside = self._point_in_metric_bounds(candidate_x, candidate_z, x_range, z_range)
                    world_candidate = {
                        "x": float(candidate_x),
                        "z": float(candidate_z),
                        "anchorSource": str(fp.anchor_source or ""),
                        "insideBounds": bool(world_inside),
                        "depthFused": bool(self._world_source_is_depth_fused(fp.anchor_source)),
                    }
                    if registered_depth_candidate is not None:
                        world_candidate["deltaToRegisteredDepthM"] = float(
                            math.hypot(
                                float(candidate_x) - float(registered_depth_candidate["x"]),
                                float(candidate_z) - float(registered_depth_candidate["z"]),
                            )
                        )
                    if floor_contact_candidate is not None:
                        world_candidate["deltaToFloorContactM"] = float(
                            math.hypot(
                                float(candidate_x) - float(floor_contact_candidate["x"]),
                                float(candidate_z) - float(floor_contact_candidate["z"]),
                            )
                        )

                registered_depth_usable = False
                if (
                    registered_depth_candidate is not None
                    and bool(registered_depth_candidate.get("insideBounds", False))
                ):
                    registered_depth_usable = True
                    if floor_contact_candidate is not None:
                        try:
                            delta_to_floor = float(
                                math.hypot(
                                    float(registered_depth_candidate["x"]) - float(floor_contact_candidate["x"]),
                                    float(registered_depth_candidate["z"]) - float(floor_contact_candidate["z"]),
                                )
                            )
                            registered_depth_candidate["deltaToFloorContactM"] = delta_to_floor
                            floor_contact_candidate["deltaToRegisteredDepthM"] = delta_to_floor
                        except Exception:
                            registered_depth_usable = False

                if (
                    registered_depth_candidate is not None
                    and bool(registered_depth_candidate.get("insideBounds", False))
                    and not registered_depth_usable
                    and floor_contact is not None
                ):
                    px, pz, _candidate_name = floor_contact[:3]
                    display_source = "floor_contact_ray"
                    selection_debug = {
                        "selected": "floor_contact_ray",
                        "reason": "registered_depth_disagrees_with_floor_contact",
                        "agreementThresholdM": float(self._depth_fused_floor_agreement_m),
                        "registeredDepthCandidate": dict(registered_depth_candidate),
                        "floorContactCandidate": floor_contact_candidate,
                    }

                if (
                    registered_depth_candidate is not None
                    and bool(registered_depth_candidate.get("insideBounds", False))
                    and registered_depth_usable
                ):
                    px = float(registered_depth_candidate["x"])
                    pz = float(registered_depth_candidate["z"])
                    display_source = "registered_depth_anchor"
                    delta_to_floor = registered_depth_candidate.get("deltaToFloorContactM")
                    depth_reason = "registered_depth_matches_floorplan_backprojection"
                    if (
                        isinstance(delta_to_floor, (int, float))
                        and math.isfinite(float(delta_to_floor))
                        and float(self._depth_fused_floor_agreement_m) > 0.0
                        and float(delta_to_floor) > float(self._depth_fused_floor_agreement_m)
                    ):
                        depth_reason = "registered_depth_preferred_over_floor_contact_disagreement"
                    selection_debug = {
                        "selected": "registered_depth_anchor",
                        "reason": depth_reason,
                        "agreementThresholdM": float(self._depth_fused_floor_agreement_m),
                        "registeredDepthCandidate": dict(registered_depth_candidate),
                        "floorContactCandidate": floor_contact_candidate,
                    }

                if (
                    (px is None or pz is None)
                    and world_candidate is not None
                ):
                    candidate_x = float(world_candidate["x"])
                    candidate_z = float(world_candidate["z"])
                    world_inside = bool(world_candidate.get("insideBounds", False))

                    if self._world_source_is_depth_fused(fp.anchor_source):
                        if world_inside:
                            px, pz = candidate_x, candidate_z
                            display_source = "world_to_camera_local"
                            selection_debug = {
                                "selected": "world_to_camera_local",
                                "reason": "depth_fused_world_candidate",
                                "agreementThresholdM": float(self._depth_fused_floor_agreement_m),
                                "worldCandidate": world_candidate,
                                "floorContactCandidate": floor_contact_candidate,
                            }
                        elif floor_contact is not None:
                            px, pz, _candidate_name = floor_contact[:3]
                            display_source = "floor_contact_ray"
                            selection_debug = {
                                "selected": "floor_contact_ray",
                                "reason": "depth_fused_world_outside_bounds",
                                "worldCandidate": world_candidate,
                                "floorContactCandidate": floor_contact_candidate,
                            }
                    elif prefer_floor_contact:
                        if floor_contact is not None:
                            px, pz, _candidate_name = floor_contact[:3]
                            display_source = "floor_contact_ray"
                            selection_debug = {
                                "selected": "floor_contact_ray",
                                "reason": "current_floor_contact_replaces_floor_only_world",
                                "worldCandidate": world_candidate,
                                "floorContactCandidate": floor_contact_candidate,
                            }
                        else:
                            selection_debug = {
                                "selected": None,
                                "reason": "floor_only_world_waiting_for_current_floor_contact",
                                "worldCandidate": world_candidate,
                                "floorContactCandidate": floor_contact_candidate,
                            }
                    elif world_inside:
                        px, pz = candidate_x, candidate_z
                        display_source = "world_to_camera_local"
                        selection_debug = {
                            "selected": "world_to_camera_local",
                            "reason": "non_floor_anchor_world_candidate",
                            "worldCandidate": world_candidate,
                            "floorContactCandidate": floor_contact_candidate,
                        }

                if prefer_floor_contact:
                    if (px is None or pz is None) and floor_contact is not None:
                        px, pz, _candidate_name = floor_contact[:3]
                        display_source = "floor_contact_ray"
                        if selection_debug is None:
                            selection_debug = {
                                "selected": "floor_contact_ray",
                                "reason": "current_floor_contact",
                                "worldCandidate": world_candidate,
                                "floorContactCandidate": floor_contact_candidate,
                            }

                if selection_debug is not None and registered_depth_candidate is not None:
                    selection_debug.setdefault("registeredDepthCandidate", dict(registered_depth_candidate))
                if selection_debug is not None and world_candidate is not None:
                    selection_debug.setdefault("worldCandidate", dict(world_candidate))

                if px is None or pz is None:
                    depth_local = None
                    if not prefer_floor_contact and fp.depth_m is not None:
                        depth_local = self._image_depth_to_camera_local(
                            calib,
                            float(fp.u),
                            float(fp.v),
                            float(fp.depth_m),
                        )
                    if depth_local is not None:
                        px, pz = depth_local
                        display_source = "image_depth_anchor"

                if prefer_image_anchor:
                    if px is None or pz is None:
                        image_world = self._image_to_world_ground(H_img2plane, float(fp.u), float(fp.v))
                        if image_world is not None:
                            iw_x, iw_z = image_world
                            px, pz = self._world_to_camera_local_ground(
                                float(iw_x),
                                float(calib.floor_y),
                                float(iw_z),
                                R_wc,
                                C_world,
                            )
                            display_source = "image_anchor"

                if px is None or pz is None:
                    if wx is not None and wz is not None and not prefer_floor_contact:
                        if math.isfinite(wx) and math.isfinite(wz):
                            px, pz = self._world_to_camera_local_ground(
                                float(wx),
                                float(calib.floor_y),
                                float(wz),
                                R_wc,
                                C_world,
                            )
                            display_source = "world_floor_fallback_to_camera_local"

                if px is None or pz is None:
                    image_world = self._image_to_world_ground(H_img2plane, float(fp.u), float(fp.v))
                    if image_world is None:
                        continue
                    iw_x, iw_z = image_world
                    px, pz = self._world_to_camera_local_ground(
                        float(iw_x),
                        float(calib.floor_y),
                        float(iw_z),
                        R_wc,
                        C_world,
                    )
                    display_source = "image_anchor"

                if max_distance_m > 0.0 and math.hypot(float(px), float(pz)) > max_distance_m:
                    continue
                if prefer_floor_contact and not self._point_in_metric_bounds(float(px), float(pz), x_range, z_range):
                    if self._alignment_debug_enabled:
                        dropped_payload: Dict[str, Any] = {
                            "x": float(px),
                            "y": float(pz),
                            "reason": "floor_contact_outside_floorplan",
                            "displaySource": str(display_source),
                            "method": str(fp.method),
                            "anchorSource": str(fp.anchor_source) if fp.anchor_source not in (None, "") else None,
                        }
                        try:
                            dropped_payload["stableId"] = int(fp.stable_id) if fp.stable_id is not None else None
                        except Exception:
                            dropped_payload["stableId"] = None
                        try:
                            dropped_payload["trackerId"] = int(fp.tracker_id) if fp.tracker_id is not None else None
                        except Exception:
                            dropped_payload["trackerId"] = None
                        dropped_payload.update(
                            self._floorplan_point_fields(
                                float(px),
                                float(pz),
                                x_range,
                                z_range,
                                width_px,
                                height_px,
                                active_floorplan_space,
                            )
                        )
                        dropped_footpoints.append(dropped_payload)
                    continue
            if not math.isfinite(px) or not math.isfinite(pz):
                continue

            alignment_debug: Optional[Dict[str, Any]] = None
            if self._alignment_debug_enabled and not use_world_frame:
                alignment_debug = self._build_alignment_debug(
                    camera_id=camera_id,
                    calib=calib,
                    fp=fp,
                    H_img2plane=H_img2plane,
                    R_wc=R_wc,
                    C_world=C_world,
                    x_range=x_range,
                    z_range=z_range,
                    width_px=width_px,
                    height_px=height_px,
                    floorplan_space=active_floorplan_space,
                    floorplan_alignment=floorplan_alignment,
                    timestamp_us=int(timestamp_us),
                    chosen_x=float(px),
                    chosen_z=float(pz),
                    display_source=str(display_source),
                    selection_debug=selection_debug,
                )

            try:
                stable_id = int(fp.stable_id) if fp.stable_id is not None else None
            except Exception:
                stable_id = None
            if stable_id is not None and stable_id <= 0:
                stable_id = None
            try:
                tracker_id = int(fp.tracker_id) if fp.tracker_id is not None else None
            except Exception:
                tracker_id = None
            if tracker_id is not None and tracker_id < 0:
                tracker_id = None

            history_key = self._history_identity(stable_id=stable_id, tracker_id=tracker_id)
            display_key = self._display_color_key(stable_id=stable_id, tracker_id=tracker_id)
            if history_key is None or display_key is None:
                continue

            raw_points.append(
                {
                    "history_key": history_key,
                    "display_key": int(display_key),
                    "x": float(px),
                    "z": float(pz),
                    "method": str(fp.method),
                    "stable_id": stable_id,
                    "tracker_id": tracker_id,
                    "anchor_source": str(fp.anchor_source) if fp.anchor_source not in (None, "") else None,
                    "anchor_quality": str(fp.anchor_quality) if fp.anchor_quality not in (None, "") else None,
                    "anchor_reason": str(fp.anchor_reason) if fp.anchor_reason not in (None, "") else None,
                    "display_source": display_source,
                    "alignment_debug": alignment_debug,
                    "motion_mode": str(fp.motion_mode) if fp.motion_mode not in (None, "") else None,
                    "posture": str(fp.posture) if fp.posture not in (None, "") else None,
                    "trail_append_allowed": (
                        bool(fp.trail_append_allowed) if fp.trail_append_allowed is not None else True
                    ),
                    "idle_jitter_m": float(fp.idle_jitter_m) if fp.idle_jitter_m is not None else None,
                }
            )

        with self._lock:
            if apply_backend_smoothing:
                self._smoother.prune(now_s)
            for item in raw_points:
                history_key = item["history_key"]
                display_key = int(item["display_key"])
                lx = float(item["x"])
                lz = float(item["z"])
                method = str(item["method"])
                stable_id = item.get("stable_id")
                tracker_id = item.get("tracker_id")
                if apply_backend_smoothing:
                    smooth_x = float(lx)
                    smooth_z = float(lz)
                    smoother_key = (camera_id, *history_key)
                    display_source_key = str(item.get("display_source") or "")
                    # Source labels can alternate between floor ray and registered depth.
                    # Both are already projected into the same floorplan metric space, so
                    # let the motion gate absorb residual deltas instead of teleporting
                    # through a source-change reset.
                    self._smoother_source_by_key[smoother_key] = display_source_key
                    smooth_x, smooth_z = self._smoother.update(smoother_key, now_s, smooth_x, smooth_z)
                    lx = float(smooth_x)
                    lz = float(smooth_z)
                    if (
                        not self._point_in_metric_bounds(lx, lz, x_range, z_range)
                        and self._point_in_metric_bounds(float(item["x"]), float(item["z"]), x_range, z_range)
                    ):
                        # Keep the displayed state on the same visible footprint as the floorplan.
                        self._smoother.reset(smoother_key)
                        self._smoother.update(smoother_key, now_s, float(item["x"]), float(item["z"]))
                        lx = float(item["x"])
                        lz = float(item["z"])
                # The frontend expects 'x' and 'y' in the JSON list.
                # We map X -> JSON x, Z -> JSON y (frame depends on configured mode).
                point_payload: Dict[str, Any] = {
                    'x': float(lx),
                    'y': float(lz),
                    'method': method,
                    'stableId': int(stable_id) if stable_id is not None else None,
                    'trackerId': int(tracker_id) if tracker_id is not None else None,
                    'anchorSource': item.get("anchor_source"),
                    'anchorQuality': item.get("anchor_quality"),
                    'anchorReason': item.get("anchor_reason"),
                    'displaySource': item.get("display_source"),
                    'motionMode': item.get("motion_mode"),
                    'posture': item.get("posture"),
                    'trailAppendAllowed': item.get("trail_append_allowed"),
                    'idleJitterM': item.get("idle_jitter_m"),
                }
                point_payload.update(
                    self._floorplan_point_fields(
                        float(lx),
                        float(lz),
                        x_range,
                        z_range,
                        width_px,
                        height_px,
                        active_floorplan_space,
                    )
                )
                if self._alignment_debug_enabled:
                    point_payload["rawX"] = float(item["x"])
                    point_payload["rawY"] = float(item["z"])
                    point_payload["smoothed"] = bool(apply_backend_smoothing)
                    if isinstance(item.get("alignment_debug"), dict):
                        point_payload["alignmentDebug"] = item["alignment_debug"]
                bev_points.append(point_payload)
                current_by_history[history_key] = {
                    "x": float(lx),
                    "z": float(lz),
                    "stable_id": int(stable_id) if stable_id is not None else None,
                    "tracker_id": int(tracker_id) if tracker_id is not None else None,
                    "display_key": int(display_key),
                    "trail_append_allowed": bool(item.get("trail_append_allowed", True)),
                    "motion_mode": item.get("motion_mode"),
                }

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

        trails_to_draw: List[Tuple[int, List[Tuple[float, float, float]]]] = []
        backend_trails: List[Dict[str, Any]] = []
        with self._lock:
            trails_enabled = bool(self._trails_enabled) and bool(self._trail_cfg.enabled)
            cam_tracks = self._trail_tracks_by_cam.setdefault(camera_id, {})

            # Update per-camera frame counter for stride decisions.
            frame_idx = int(self._trail_frame_counts.get(camera_id, 0) + 1)
            self._trail_frame_counts[camera_id] = frame_idx
            do_sample = (frame_idx % int(self._trail_cfg.draw_stride)) == 0

            min_dt_s = float(self._trail_cfg.min_dt_s)
            min_step_m = float(self._trail_cfg.min_step_px) * float(effective_mpp)
            disable_trail_postprocess = bool(use_world_frame) or bool(apply_backend_smoothing)
            trail_max_speed_px_per_s = 0.0 if disable_trail_postprocess else float(self._trail_cfg.max_speed_px_per_s)
            trail_smooth_tau_s = 0.0 if disable_trail_postprocess else float(self._trail_cfg.smooth_tau_s)
            max_points = max(2, int(self._trail_cfg.max_points_per_track))
            window_s = float(self._trail_cfg.window_s)

            if trails_enabled:
                for history_key, point_meta in current_by_history.items():
                    lx = float(point_meta["x"])
                    lz = float(point_meta["z"])
                    stable_id = point_meta.get("stable_id")
                    tracker_id = point_meta.get("tracker_id")
                    display_key = int(point_meta["display_key"])
                    state = cam_tracks.get(history_key)
                    if state is None:
                        state = _BevTrailTrackState(points=deque(maxlen=max_points))
                        cam_tracks[history_key] = state
                    state.last_seen_ts = float(now_s)
                    state.stable_id = int(stable_id) if stable_id is not None else None
                    state.tracker_id = int(tracker_id) if tracker_id is not None else None
                    state.display_key = int(display_key)

                    # Always prune old samples so disappeared tracks naturally fade out.
                    while state.points and (now_s - float(state.points[0][0])) > window_s:
                        state.points.popleft()

                    if not do_sample:
                        continue

                    x = float(lx)
                    z = float(lz)
                    append_allowed = bool(point_meta.get("trail_append_allowed", True))

                    # Clamp spurious jumps based on max speed in BEV pixels/sec scaled by effective mpp.
                    if state.points:
                        prev_ts, prev_x, prev_z = state.points[-1]
                        dt = max(0.0, float(now_s) - float(prev_ts))
                        dist = math.hypot(x - float(prev_x), z - float(prev_z))
                        max_step = float(trail_max_speed_px_per_s) * float(effective_mpp) * dt
                        if max_step > 0.0 and dist > max_step:
                            scale_step = max_step / dist
                            x = float(prev_x) + (x - float(prev_x)) * scale_step
                            z = float(prev_z) + (z - float(prev_z)) * scale_step

                    # Producer-owned point smoothing is already applied above for world-mode BEV.
                    # Do not add a second trail-stage EMA/speed model on top of that path.
                    if trail_smooth_tau_s > 0.0:
                        if state.ema_x is None or state.ema_z is None:
                            state.ema_x, state.ema_z = x, z
                            state.ema_ts = float(now_s)
                        else:
                            dt_ema = max(0.0, float(now_s) - float(state.ema_ts))
                            tau = float(trail_smooth_tau_s)
                            alpha = 1.0 - math.exp(-dt_ema / tau) if (tau > 0.0 and dt_ema > 0.0) else 1.0
                            state.ema_x = float(state.ema_x + alpha * (x - float(state.ema_x)))
                            state.ema_z = float(state.ema_z + alpha * (z - float(state.ema_z)))
                            state.ema_ts = float(now_s)
                        x, z = float(state.ema_x), float(state.ema_z)
                    else:
                        state.ema_x, state.ema_z = float(x), float(z)
                        state.ema_ts = float(now_s)

                    # Phase 1: stationary tracks do not grow path history.
                    # Phase 6: min-step + RDP simplification on committed history.
                    # Use the trail decimation min-step (already scaled by mpp) as the
                    # primary spatial gate so short walks still form multi-point paths.
                    path_cfg = HumanGroundConfig(
                        path_min_step_m=max(0.0, float(min_step_m)),
                        path_simplify_epsilon_m=max(0.02, float(min_step_m) * 1.25),
                        path_max_points=int(max_points),
                    )
                    if not append_allowed:
                        if state.points:
                            prev_ts, _px, _pz = state.points[-1]
                            state.points[-1] = (float(prev_ts), float(x), float(z))
                        continue
                    if min_dt_s > 0.0 and state.points:
                        prev_ts, prev_x, prev_z = state.points[-1]
                        dt = max(0.0, float(now_s) - float(prev_ts))
                        dist = math.hypot(float(x) - float(prev_x), float(z) - float(prev_z))
                        if dt < float(min_dt_s):
                            if dist >= float(min_step_m):
                                state.points[-1] = (float(prev_ts), float(x), float(z))
                            continue
                        if dist < float(min_step_m):
                            continue
                    # Always seed the first point even when min_step would block.
                    if not state.points:
                        state.points.append((float(now_s), float(x), float(z)))
                    else:
                        commit_path_point(
                            state.points,
                            ts=float(now_s),
                            x=float(x),
                            z=float(z),
                            config=path_cfg,
                            append_allowed=True,
                        )

            # Remove fully expired tracks to keep memory bounded.
            expired: List[Hashable] = []
            for history_key, state in cam_tracks.items():
                while state.points and (now_s - float(state.points[0][0])) > window_s:
                    state.points.popleft()
                if not state.points and (now_s - float(state.last_seen_ts)) > window_s:
                    expired.append(history_key)
            for history_key in expired:
                cam_tracks.pop(history_key, None)

            if trails_enabled and cam_tracks:
                # Select a bounded number of tracks to render.
                track_items = list(cam_tracks.items())
                track_items.sort(key=lambda item: float(getattr(item[1], "last_seen_ts", 0.0)), reverse=True)
                max_tracks = max(1, int(self._trail_cfg.max_tracks))
                for history_key, state in track_items[:max_tracks]:
                    pts = list(state.points)
                    if len(pts) < 2:
                        continue
                    segments_available = len(pts) - 1
                    segments_budget = min(segments_available, int(self._trail_cfg.max_segments_per_track))
                    if segments_budget <= 0:
                        continue
                    pts = self._resample_points(pts, segments_budget)
                    display_key = int(state.display_key) if state.display_key is not None else 0
                    trails_to_draw.append((display_key, pts))
                    trail_points: List[Dict[str, Any]] = []
                    for ts_pt, x_pt, z_pt in pts:
                        point_out: Dict[str, Any] = {
                            "x": float(x_pt),
                            "y": float(z_pt),
                            "t": int(round(float(ts_pt) * 1000.0)),
                        }
                        point_out.update(
                            self._floorplan_point_fields(
                                float(x_pt),
                                float(z_pt),
                                x_range,
                                z_range,
                                width_px,
                                height_px,
                                active_floorplan_space,
                            )
                        )
                        trail_points.append(point_out)
                    backend_trails.append(
                        {
                            "stableId": int(state.stable_id) if state.stable_id is not None else None,
                            "trackerId": int(state.tracker_id) if state.tracker_id is not None else None,
                            "points": trail_points,
                        }
                    )

        # Draw grid + trails + footpoints if overlay is enabled
        if bev is not None and result_config.overlay:
            self._draw_grid(bev, result_config)
            if trails_to_draw:
                inv_window = 1.0 / max(0.1, float(self._trail_cfg.window_s))
                min_alpha = float(self._trail_cfg.min_alpha)
                line_width = int(self._trail_cfg.line_width)
                for stable_id, pts in trails_to_draw:
                    if len(pts) < 2:
                        continue
                    base_bgr = self._color_for_key(int(stable_id))
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

            for point_meta in current_by_history.values():
                lx = float(point_meta["x"])
                lz = float(point_meta["z"])
                p = self._local_to_px(lx, lz, result_config, bev.shape[:2])
                if p is None:
                    continue
                base_bgr = self._color_for_key(int(point_meta["display_key"]))
                cv2.circle(bev, p, 4, base_bgr, -1, lineType=cv2.LINE_AA)

        result = BevResult(
            camera_id=camera_id,
            bev_bgr=bev,
            bev_points=bev_points,
            backend_trails=backend_trails,
            config=result_config,
            timestamp_us=timestamp_us,
            width_px=width_px,
            height_px=height_px,
            points_smoothed=bool(apply_backend_smoothing),
            bounds_source=str(bounds_source),
            floorplan_space=active_floorplan_space,
            dropped_footpoints=dropped_footpoints,
        )
        self._publish(result, calib, H_img2plane, flip_u=flip_u, flip_v=flip_v)

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

    def _publish(
        self,
        result: BevResult,
        calib: CalibrationSnapshot,
        H_to_use: np.ndarray,
        *,
        flip_u: bool = False,
        flip_v: bool = False,
    ) -> None:
        try:
            # Compute a quick sanity sample: image bottom-center ray intersection in scene units (XZ)
            sample_xz: Optional[Tuple[float, float]] = None
            fallback_points = [
                point for point in result.bev_points
                if str(point.get("anchorSource") or "").strip().lower() == "ray_floor_fallback"
            ]
            fallback_reasons: Dict[str, int] = {}
            for point in fallback_points:
                reason = str(point.get("anchorReason") or "legacy_fallback").strip() or "legacy_fallback"
                fallback_reasons[reason] = int(fallback_reasons.get(reason, 0)) + 1
            try:
                width_src, height_src = calib.image_size
                u = float(max(0.0, width_src * 0.5))
                v = float(max(0.0, height_src - 1))
                R_wc, C_world = parse_extrinsics(calib.extrinsics_col_major)
                plane = Plane.horizontal(float(calib.floor_y))
                origin, direction = ray_from_pixel(u, v, calib.intrinsics, R_wc, C_world)
                hit = intersect_plane(origin, direction, plane)
                if hit is not None:
                    sample_xz = (float(hit[0]), float(hit[2]))
            except Exception:
                sample_xz = None
            frame_name = "backend_world_m" if self._frame_mode == "world" else "camera_local_ground_m"
            floorplan_space = result.floorplan_space
            floorplan_bounds = {
                "min_x": float(result.config.x_range[0]),
                "max_x": float(result.config.x_range[1]),
                "min_z": float(result.config.z_range[0]),
                "max_z": float(result.config.z_range[1]),
            }
            floorplan_grid_shape = (
                list(floorplan_space.grid_shape)
                if floorplan_space is not None and floorplan_space.grid_shape is not None
                else None
            )
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
                "boundsSource": str(result.bounds_source or "config"),
                "floorplanCoordinateSpace": "floorplan_normalized_v1",
                "floorplanBounds": floorplan_bounds,
                "floorplanGridShape": floorplan_grid_shape,
                "floorplanGridResM": (
                    float(floorplan_space.grid_res_m)
                    if floorplan_space is not None and floorplan_space.grid_res_m is not None
                    else None
                ),
                "floorplanSnapshotTsUs": (
                    int(floorplan_space.snapshot_ts_us)
                    if floorplan_space is not None and floorplan_space.snapshot_ts_us is not None
                    else None
                ),
                "floorplanTsUs": (
                    int(floorplan_space.floorplan_ts_us)
                    if floorplan_space is not None and floorplan_space.floorplan_ts_us is not None
                    else None
                ),
                "overlay": result.config.overlay,
                "footpoints": result.bev_points,
                "trails": result.backend_trails,
                # Optional: flattened 3x3 homography for client-side debug/overlays
                "H": [float(x) for x in H_to_use.reshape(-1)],
                "sampleXZ": list(sample_xz) if sample_xz is not None else None,
                "frame": frame_name,
                "world_frame": frame_name,
                "frame_mode": self._frame_mode,
                "units": "meters",
                "s_obj_to_m": float(getattr(calib, "unit_scale", 1.0) or 1.0),
                "trail_smoothing_owner": "backend" if (self._frame_mode == "world" or result.points_smoothed) else "none",
                "bev_points_smoothed": bool(result.points_smoothed),
                "bev_world_points_smoothed": bool(result.points_smoothed) if self._frame_mode == "world" else False,
                "fallbackActive": bool(fallback_points),
                "fallbackTrackCount": int(len(fallback_points)),
                "fallbackSources": (["ray_floor_fallback"] if fallback_points else []),
                "fallbackReasonCounts": fallback_reasons,
            }
            if floorplan_space is not None and isinstance(floorplan_space.ray_to_floorplan_alignment, Mapping):
                alignment_payload = floorplan_space.ray_to_floorplan_alignment
                status["floorplanAlignment"] = {
                    "quality": alignment_payload.get("quality"),
                    "reason": alignment_payload.get("reason"),
                    "sampleCount": alignment_payload.get("sample_count"),
                    "inlierCount": alignment_payload.get("inlier_count"),
                    "residualM": alignment_payload.get("residual_m"),
                    "applied": self._floorplan_alignment_matrix(floorplan_space) is not None,
                }
            if self._alignment_debug_enabled:
                source_counts: Dict[str, int] = {}
                snapshots: Dict[str, int] = {}
                chosen_out_of_bounds = 0
                candidate_count = 0
                for point in result.bev_points:
                    src = str(point.get("displaySource") or "unknown")
                    source_counts[src] = int(source_counts.get(src, 0)) + 1
                    debug = point.get("alignmentDebug")
                    if not isinstance(debug, Mapping):
                        continue
                    chosen = debug.get("chosen")
                    if isinstance(chosen, Mapping) and chosen.get("insideBounds") is False:
                        chosen_out_of_bounds += 1
                    for cand in debug.get("candidates") or []:
                        if not isinstance(cand, Mapping):
                            continue
                        candidate_count += 1
                        depth_info = cand.get("mapanythingDepth")
                        if not isinstance(depth_info, Mapping):
                            continue
                        sample = depth_info.get("sample")
                        if not isinstance(sample, Mapping):
                            continue
                        snap = sample.get("snapshot_ts_us")
                        if snap is not None:
                            key = str(snap)
                            snapshots[key] = int(snapshots.get(key, 0)) + 1
                status["alignmentDebug"] = {
                    "enabled": True,
                    "candidateCount": int(candidate_count),
                    "displaySourceCounts": source_counts,
                    "sampledSnapshotCounts": snapshots,
                    "chosenOutOfBounds": int(chosen_out_of_bounds),
                    "droppedFootpointCount": int(len(result.dropped_footpoints)),
                }
                if result.dropped_footpoints:
                    status["droppedFootpoints"] = result.dropped_footpoints
            if hasattr(self.ws, "broadcast_sync"):
                self.ws.broadcast_sync(status)
                # JPEG BEV binary retired (meta-only mode). No cv2 render or framed binary is produced.
                # The general binary coalescer in the WS server remains for future use (e.g., binary depth).
                try:
                    # Lightweight visibility that a BEV frame was queued for broadcast
                    if hasattr(self.ws, "logger") and self.ws.logger:
                        self.ws.logger.debug(
                            "BEV publish %s: %sx%s, points=%d, overlay=%s (meta-only)",
                            result.camera_id,
                            int(result.width_px),
                            int(result.height_px),
                            len(result.bev_points),
                            result.config.overlay,
                        )
                except Exception:
                    pass
        except Exception as e:
            logger.exception("BEV publish failed for %s", getattr(result, "camera_id", "unknown"))
            return
