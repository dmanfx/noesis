"""HTTP client and storage utilities for MapAnything depth inference."""
from __future__ import annotations

import base64
import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import cv2
import numpy as np
import requests
import zarr
from numcodecs import Blosc

from adapters.mapanything_adapter import ViewBuildResult, build_mono_view
from mapanything_config import ServiceConfig, load_service_config
from utils.rate_limited_logger import RateLimitedLogger


@dataclass(frozen=True)
class DepthSummary:
    median: float
    p10: float
    p90: float
    conf_mean: float
    valid_ratio: float
    sample_count: int


@dataclass(frozen=True)
class DepthResult:
    camera_id: str
    ts_us: int
    depth: np.ndarray
    conf: np.ndarray
    mask: np.ndarray
    intrinsics: Optional[np.ndarray]
    native_intrinsics: Optional[np.ndarray]
    summary: DepthSummary
    storage_path: Path


class DepthStorageManager:
    """Persist depth outputs to Zarr for later consumption."""

    def __init__(self, base_path: Path) -> None:
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._locks: Dict[str, threading.Lock] = {}

    def store(
        self,
        camera_id: str,
        ts_us: int,
        depth: np.ndarray,
        conf: np.ndarray,
        mask: np.ndarray,
    ) -> Path:
        lock = self._locks.setdefault(camera_id, threading.Lock())
        with lock:
            timestamp = datetime.utcfromtimestamp(ts_us / 1_000_000.0)
            date_dir = timestamp.strftime("%Y%m%d")
            hour_dir = timestamp.strftime("%H")
            dest_dir = self.base_path / camera_id / date_dir / hour_dir
            dest_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{ts_us}.zarr"
            dest_path = dest_dir / filename

            compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
            root = zarr.open_group(str(dest_path), mode="w")
            chunk_shape = (min(128, depth.shape[0]), min(128, depth.shape[1]))
            root.create_dataset(
                "depth_z",
                data=depth.astype(np.float32),
                compressor=compressor,
                chunks=chunk_shape,
                overwrite=True,
            )
            root.create_dataset(
                "conf",
                data=conf.astype(np.float32),
                compressor=compressor,
                chunks=chunk_shape,
                overwrite=True,
            )
            root.create_dataset(
                "mask",
                data=mask.astype(np.uint8),
                compressor=compressor,
                chunks=chunk_shape,
                overwrite=True,
            )
            root.attrs.update(
                camera_id=camera_id,
                timestamp_us=int(ts_us),
                stored_at=time.time(),
                shape=json.dumps(depth.shape),
            )
        return dest_path

    def latest_entry(self, camera_id: str, ts_max: Optional[int]) -> Optional[Path]:
        camera_dir = self.base_path / camera_id
        if not camera_dir.exists():
            return None
        candidates = []
        for path in camera_dir.rglob('*.zarr'):
            try:
                ts = int(path.stem)
            except ValueError:
                continue
            if ts_max is not None and ts > ts_max:
                continue
            candidates.append((ts, path))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def load_datasets(self, path: Path) -> Optional[Dict[str, np.ndarray]]:
        try:
            group = zarr.open_group(str(path), mode='r')
            depth = np.array(group['depth_z'])
            conf = np.array(group['conf'])
            mask = np.array(group['mask'])
            return {'depth': depth, 'conf': conf, 'mask': mask}
        except Exception:
            return None


class MapAnythingDepthSource:
    """Client for invoking MapAnything inference service and persisting results."""

    def __init__(self, config: Optional[ServiceConfig] = None) -> None:
        self.config = config or load_service_config()
        self.session = requests.Session()
        self.logger = RateLimitedLogger(logging.getLogger(__name__), rate_limit_seconds=2.0)
        self.storage = DepthStorageManager(Path(self.config.storage.depth_base))
        self.min_conf = float(self.config.performance.min_conf)
        self.mono_interval = 1.0 / max(self.config.performance.mono_freq_hz, 1e-6)
        self.last_request_per_camera: Dict[str, float] = {}
        self.timeout = 5.0

    def should_infer(self, camera_id: str, timestamp_s: float) -> bool:
        last = self.last_request_per_camera.get(camera_id)
        if last is None:
            return True
        return (timestamp_s - last) >= self.mono_interval

    def maybe_infer_mono(
        self,
        camera_id: str,
        frame_bgr: np.ndarray,
        calib_bundle: Optional[Mapping[str, object]],
        timestamp_s: Optional[float] = None,
    ) -> Optional[DepthResult]:
        ts = timestamp_s or time.time()
        if not self.should_infer(camera_id, ts):
            return None
        try:
            result = self._infer_mono(camera_id, frame_bgr, calib_bundle, ts)
            self.last_request_per_camera[camera_id] = ts
            return result
        except Exception as exc:
            self.logger.error(f"Mono depth inference failed for {camera_id}: {exc}")
            return None

    def _infer_mono(
        self,
        camera_id: str,
        frame_bgr: np.ndarray,
        calib_bundle: Optional[Mapping[str, object]],
        timestamp_s: float,
    ) -> DepthResult:
        view_result = build_mono_view(frame_bgr, camera_id, calib_bundle)
        payload = {"view": view_result.payload}
        response_json = self._post_json("/infer_mono", payload)

        depth, conf, mask = self._parse_response(response_json)
        depth, conf, mask = self._align_to_original_shape(depth, conf, mask, view_result)

        ts_us = int(timestamp_s * 1_000_000)
        storage_path = self.storage.store(camera_id, ts_us, depth, conf, mask)
        summary = self._compute_summary(depth, conf, mask)

        return DepthResult(
            camera_id=camera_id,
            ts_us=ts_us,
            depth=depth,
            conf=conf,
            mask=mask,
            intrinsics=view_result.native_intrinsics.copy() if view_result.native_intrinsics is not None else None,
            native_intrinsics=view_result.native_intrinsics.copy() if view_result.native_intrinsics is not None else None,
            summary=summary,
            storage_path=storage_path,
        )

    def _post_json(self, endpoint: str, payload: Dict[str, object]) -> Dict[str, object]:
        url = f"{self.config.service.base_url}{endpoint}"
        headers = {"X-API-Key": self.config.service.api_key}
        backoff = 0.5
        for attempt in range(1, 4):
            try:
                response = self.session.post(url, json=payload, headers=headers, timeout=self.timeout)
                if response.status_code == 200:
                    return response.json()
                self.logger.warning(f"MapAnything service returned {response.status_code}: {response.text}")
            except requests.RequestException as exc:
                self.logger.warning(f"MapAnything request failed (attempt {attempt}): {exc}")
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 4.0)
        raise RuntimeError("MapAnything request failed after retries")

    def _parse_response(self, response: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        depth = np.array(response.get("depth_z"), dtype=np.float32)
        conf = np.array(response.get("conf"), dtype=np.float32)
        mask = np.array(response.get("mask"), dtype=bool)
        if depth.ndim != 2 or conf.ndim != 2:
            raise ValueError("Depth/conf arrays must be 2D")
        if mask.shape != depth.shape:
            mask = mask.reshape(depth.shape)
        return depth, conf, mask

    def _align_to_original_shape(
        self,
        depth: np.ndarray,
        conf: np.ndarray,
        mask: np.ndarray,
        view_result: ViewBuildResult,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        target_h, target_w = view_result.original_shape[:2]
        current_h, current_w = depth.shape[:2]
        if (target_h, target_w) == (current_h, current_w):
            return depth, conf, mask
        depth_resized = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        conf_resized = cv2.resize(conf, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        mask_uint8 = cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        mask_resized = mask_uint8.astype(bool)
        return depth_resized, conf_resized, mask_resized

    def _compute_summary(self, depth: np.ndarray, conf: np.ndarray, mask: np.ndarray) -> DepthSummary:
        valid = mask & np.isfinite(depth) & (conf >= self.min_conf) & (depth > 0.0)
        total = depth.size
        if total == 0:
            return DepthSummary(median=0.0, p10=0.0, p90=0.0, conf_mean=0.0, valid_ratio=0.0, sample_count=0)
        if not np.any(valid):
            return DepthSummary(median=0.0, p10=0.0, p90=0.0, conf_mean=float(conf.mean()), valid_ratio=0.0, sample_count=0)
        valid_depth = depth[valid]
        return DepthSummary(
            median=float(np.median(valid_depth)),
            p10=float(np.percentile(valid_depth, 10)),
            p90=float(np.percentile(valid_depth, 90)),
            conf_mean=float(conf[valid].mean()),
            valid_ratio=float(valid.sum() / total),
            sample_count=int(valid.sum()),
        )

    def load_latest_depth(self, camera_id: str, ts_max: Optional[int] = None) -> Optional[Dict[str, Any]]:
        path = self.storage.latest_entry(camera_id, ts_max)
        if not path:
            return None
        datasets = self.storage.load_datasets(path)
        if not datasets:
            return None
        depth = datasets['depth'].astype(np.float32, copy=False)
        conf = datasets['conf'].astype(np.float32, copy=False)
        mask = datasets['mask'].astype(np.uint8, copy=False)
        height, width = depth.shape[:2]
        ts_us = int(path.stem)
        return {
            'ts': ts_us,
            'depth_b64': base64.b64encode(depth.tobytes()).decode('ascii'),
            'conf_b64': base64.b64encode(conf.tobytes()).decode('ascii'),
            'mask_b64': base64.b64encode(mask.tobytes()).decode('ascii'),
            'shape': [int(height), int(width)],
        }


__all__ = [
    "MapAnythingDepthSource",
    "DepthResult",
    "DepthSummary",
]
