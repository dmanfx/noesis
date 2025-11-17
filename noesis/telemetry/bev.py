from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from geometry.homography import compose_bev_homography


@dataclass(frozen=True)
class CalibrationSnapshot:
    camera_id: str
    intrinsics: np.ndarray  # 3x3 matrix
    extrinsics_col_major: Sequence[float]  # 16 values, world→camera
    floor_y: float
    image_size: Tuple[int, int]


@dataclass
class BevConfig:
    meters_per_px: float = 0.05
    x_range: Tuple[float, float] = (-4.0, 4.0)
    z_range: Tuple[float, float] = (0.0, 12.0)
    overlay: bool = True
    max_px: int = 768


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
        self._cache: Dict[str, Tuple[np.ndarray, Tuple[int, int]]] = {}

    @staticmethod
    def _hash_matrix(matrix: np.ndarray) -> int:
        return hash(matrix.tobytes())

    def _key(
        self,
        calib: CalibrationSnapshot,
        cfg: BevConfig,
    ) -> str:
        return "::".join(
            [
                calib.camera_id,
                f"{calib.image_size[0]}x{calib.image_size[1]}",
                f"K={self._hash_matrix(calib.intrinsics)}",
                f"E={hash(tuple(float(x) for x in calib.extrinsics_col_major))}",
                f"floor={calib.floor_y:.4f}",
                f"x={cfg.x_range}",
                f"z={cfg.z_range}",
                f"mpp={cfg.meters_per_px:.4f}",
            ]
        )

    def get(
        self,
        calib: CalibrationSnapshot,
        cfg: BevConfig,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        key = self._key(calib, cfg)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        H, bev_size = compose_bev_homography(
            calib.intrinsics,
            calib.extrinsics_col_major,
            calib.floor_y,
            calib.image_size,
            cfg.x_range,
            cfg.z_range,
            cfg.meters_per_px,
        )
        self._cache[key] = (H, bev_size)
        return H, bev_size


class BevRenderer:
    def __init__(self, ws_server) -> None:
        self.ws = ws_server
        self.config_per_cam: Dict[str, BevConfig] = {}
        self.h_cache = HomographyCache()

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
        H_img2bev, (width_px, height_px) = self.h_cache.get(calib, cfg)
        long_edge = max(width_px, height_px)
        H_to_use = H_img2bev
        if long_edge > cfg.max_px:
            scale = cfg.max_px / float(long_edge)
            width_px = max(1, int(width_px * scale))
            height_px = max(1, int(height_px * scale))
            S = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            H_to_use = S @ H_img2bev

        bev = cv2.warpPerspective(frame_bgr, H_to_use, (width_px, height_px), flags=cv2.INTER_LINEAR)
        bev_points: List[Dict[str, float]] = []

        if cfg.overlay:
            self._draw_grid(bev, cfg)

        for fp in footpoints:
            vec = np.array([fp.u, fp.v, 1.0], dtype=np.float64)
            bev_pt = H_to_use @ vec
            w = bev_pt[2] if bev_pt[2] else 1.0
            bx = float(bev_pt[0] / w)
            by = float(bev_pt[1] / w)
            if np.isnan(bx) or np.isnan(by):
                continue
            bev_points.append({'x': bx, 'y': by, 'method': fp.method, 'trackId': fp.track_id})
            if cfg.overlay:
                cv2.circle(
                    bev,
                    (int(np.clip(round(bx), 0, width_px - 1)), int(np.clip(round(by), 0, height_px - 1))),
                    4,
                    (0, 255, 255),
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )

        result = BevResult(
            camera_id=camera_id,
            bev_bgr=bev,
            bev_points=bev_points,
            config=cfg,
            timestamp_us=timestamp_us,
        )
        self._publish(result)

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

    def _publish(self, result: BevResult) -> None:
        ok, jpeg = cv2.imencode(".jpg", result.bev_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ok:
            return
        payload = jpeg.tobytes()
        # Use the same header scheme as mosaic: a single camera-id string
        # Frontend distinguishes stream type via the preceding JSON message (type='bev-frame').
        header = f"bev:{result.camera_id}".encode("utf-8")
        framed = bytes([len(header)]) + header + payload
        try:
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
        except Exception:
            pass
