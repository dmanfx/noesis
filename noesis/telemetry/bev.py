from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Any

import cv2
import numpy as np
import time
import math

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


@dataclass
class BevResult:
    camera_id: str
    bev_bgr: np.ndarray
    bev_points: List[Dict[str, float]]
    config: BevConfig
    timestamp_us: int


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
    def __init__(self, ws_server) -> None:
        self.ws = ws_server
        self.config_per_cam: Dict[str, BevConfig] = {}
        self.h_cache = HomographyCache()
        # Last known good homography per camera for resilience
        self._last_h_by_cam: Dict[str, np.ndarray] = {}
        # Auto-computed extents per camera
        self._auto_extents_by_camera: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {}

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
        frame_bgr: np.ndarray,
        footpoints: List[Footpoint],
        timestamp_us: int,
    ) -> None:
        cfg = self.config_per_cam.get(camera_id, BevConfig())
        
        # Compute or fetch extents for this camera
        x_range = cfg.x_range
        z_range = cfg.z_range
        
        if cfg.auto_fit_extents:
            # Robust key including calibration version to prevent stale frustums
            ext_hash = hash(tuple(float(x) for x in calib.extrinsics_col_major))
            intr_hash = HomographyCache._hash_matrix(calib.intrinsics)
            key = f"{camera_id}::{ext_hash}::{intr_hash}"
            
            auto_extents = self._auto_extents_by_camera.get(key)
            if auto_extents is None:
                from geometry.homography import compute_ground_frustum_aabb
                x_range_auto, z_range_auto = compute_ground_frustum_aabb(
                    calib.intrinsics,
                    calib.extrinsics_col_major,
                    calib.floor_y,
                    calib.image_size,
                    calib.unit_scale,
                    cfg.max_distance_m,
                    padding_m=0.25,
                )
                auto_extents = (x_range_auto, z_range_auto)
                self._auto_extents_by_camera[key] = auto_extents
            x_range, z_range = auto_extents
        
        # Create a temporary config with the actual extents to use for homography computation
        # (Not strictly needed for metric homography, but good for context)
        
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
            
        # Create a tiny placeholder image to satisfy the protocol.
        bev = np.zeros((1, 1, 3), dtype=np.uint8)

        bev_points: List[Dict[str, float]] = []

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

            if camera_id == "kitchen":
                logger.warning(
                    "[BEV kitchen] fp track=%s method=%s pixel=(%.1f, %.1f) world=(%.2f, %.2f) local=(%.2f, %.2f)",
                    fp.track_id,
                    fp.method,
                    fp.u,
                    fp.v,
                    wx,
                    wz,
                    lx,
                    lz,
                )

            # The frontend expects 'x' and 'y' in the JSON list.
            # We map Local X -> JSON x, Local Z -> JSON y
            bev_points.append({'x': lx, 'y': lz, 'method': fp.method, 'trackId': fp.track_id})

        # Update config to reflect the actual extents used
        result_config = BevConfig(
            meters_per_px=cfg.meters_per_px,
            x_range=x_range,
            z_range=z_range,
            overlay=cfg.overlay,
            max_px=cfg.max_px,
            auto_fit_extents=cfg.auto_fit_extents,
            max_distance_m=cfg.max_distance_m,
        )
        
        result = BevResult(
            camera_id=camera_id,
            bev_bgr=bev,
            bev_points=bev_points,
            config=result_config,
            timestamp_us=timestamp_us,
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
        ok, jpeg = cv2.imencode(".jpg", result.bev_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ok:
            # Surface encoding failures so FE can show a message instead of a broken icon
            self.publish_status(result.camera_id, error="jpeg_encode_failed")
            return
        payload = jpeg.tobytes()
        # Use the same header scheme as mosaic: a single camera-id string
        # Frontend distinguishes stream type via the preceding JSON message (type='bev-frame').
        header = f"bev:{result.camera_id}".encode("utf-8")
        framed = bytes([len(header)]) + header + payload
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
            if sample_xz is not None and result.camera_id == "kitchen":
                dx = sample_xz[0] - float(C_world[0])
                dz = sample_xz[1] - float(C_world[2])
                dist = math.hypot(dx, dz)
                logger.warning(
                    "[BEV %s] bottom-center floor hit: xz=(%.2f, %.2f) dist=%.2f m",
                    result.camera_id,
                    sample_xz[0],
                    sample_xz[1],
                    dist,
                )
            status = {
                "type": "bev-frame",
                "cameraId": result.camera_id,
                "ts": result.timestamp_us,
                "w": int(result.bev_bgr.shape[1]),
                "h": int(result.bev_bgr.shape[0]),
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
                self.ws.broadcast_sync(framed)
                try:
                    # Lightweight visibility that a BEV frame was queued for broadcast
                    if hasattr(self.ws, "logger") and self.ws.logger:
                        self.ws.logger.debug(
                            "BEV publish %s: %sx%s, points=%d, overlay=%s",
                            result.camera_id,
                            int(result.bev_bgr.shape[1]),
                            int(result.bev_bgr.shape[0]),
                            len(result.bev_points),
                            result.config.overlay,
                        )
                except Exception:
                    pass
        except Exception as e:
            print("BEV publish failed:", e)
            return
