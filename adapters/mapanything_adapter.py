"""Adapters for preparing DeepStream frames for MapAnything inference."""
from __future__ import annotations

import base64
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional, Tuple

import cv2
import numpy as np
import yaml

from mapanything_config import load_service_config
from utils.rate_limited_logger import RateLimitedLogger


@dataclass(frozen=True)
class ViewBuildResult:
    """Container describing an encoded view ready for RPC."""

    payload: Dict[str, object]
    original_shape: Tuple[int, int, int]
    resized_shape: Tuple[int, int, int]
    intrinsics: Optional[np.ndarray]
    native_intrinsics: Optional[np.ndarray]
    scale_factors: Tuple[float, float]


_logger = RateLimitedLogger(logging.getLogger(__name__), rate_limit_seconds=5.0)
_config = load_service_config()
_INTRINSICS_CACHE: MutableMapping[str, np.ndarray] = {}
_FALLBACK_CAM_YAML_CACHE: Optional[Dict[str, object]] = None
_FALLBACK_CAM_YAML_MTIME: Optional[float] = None
_FALLBACK_LOCK = threading.Lock()
_CAMERAS_YAML_PATH = Path("config/cameras.yaml")


def build_mono_view(
    frame_bgr: np.ndarray,
    cam_id: str,
    calib_bundle: Optional[Mapping[str, object]],
    *,
    max_resolution: Optional[int] = None,
) -> ViewBuildResult:
    """Convert a DeepStream BGR frame into a MapAnything-ready payload.

    Args:
        frame_bgr: Input image in BGR channel order (uint8)
        cam_id: Camera identifier
        calib_bundle: Calibration bundle containing intrinsics
        max_resolution: Optional cap on max image dimension (defaults to config)

    Returns:
        ViewBuildResult with payload ready to send to `/infer_mono`
    """
    if frame_bgr is None:
        raise ValueError("frame_bgr must not be None")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] not in (3, 4):
        raise ValueError(f"Expected image with 3 channels, got shape {frame_bgr.shape}")

    frame_rgb = _convert_to_rgb(frame_bgr)
    original_shape = frame_rgb.shape

    cap = max_resolution if max_resolution is not None else _config.performance.max_res
    resized_rgb, scale_x, scale_y = _resize_frame(frame_rgb, cap)
    resized_shape = resized_rgb.shape

    intrinsics = _resolve_intrinsics(cam_id, calib_bundle)
    intrinsics_scaled = _scale_intrinsics(intrinsics, scale_x, scale_y) if intrinsics is not None else None

    payload = _encode_view_payload(cam_id, resized_rgb, intrinsics_scaled)
    return ViewBuildResult(
        payload=payload,
        original_shape=original_shape,
        resized_shape=resized_shape,
        intrinsics=intrinsics_scaled,
        native_intrinsics=intrinsics.copy() if intrinsics is not None else None,
        scale_factors=(scale_x, scale_y),
    )


def build_multi_views(
    frames_bgr: Mapping[str, np.ndarray],
    calib_bundle: Optional[Mapping[str, object]],
    *,
    max_resolution: Optional[int] = None,
    max_views: int = 6,
) -> Tuple[ViewBuildResult, ...]:
    """Prepare up to max_views frames for multi-view inference."""
    results = []
    for idx, (cam_id, frame) in enumerate(frames_bgr.items()):
        if idx >= max_views:
            break
        results.append(
            build_mono_view(
                frame,
                cam_id,
                calib_bundle,
                max_resolution=max_resolution,
            )
        )
    return tuple(results)


def _convert_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    if frame_bgr.shape[2] == 4:
        frame_bgr = frame_bgr[:, :, :3]
    if frame_bgr.dtype != np.uint8:
        frame_bgr = np.clip(frame_bgr, 0, 255).astype(np.uint8)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _resize_frame(frame_rgb: np.ndarray, max_resolution: int) -> Tuple[np.ndarray, float, float]:
    height, width = frame_rgb.shape[:2]
    if max_resolution <= 0:
        return frame_rgb, 1.0, 1.0
    max_dim = max(height, width)
    if max_dim <= max_resolution:
        return frame_rgb, 1.0, 1.0
    scale = max_resolution / float(max_dim)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(frame_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
    scale_x = new_width / width
    scale_y = new_height / height
    return resized, scale_x, scale_y


def _resolve_intrinsics(cam_id: str, calib_bundle: Optional[Mapping[str, object]]) -> Optional[np.ndarray]:
    if cam_id in _INTRINSICS_CACHE:
        return _INTRINSICS_CACHE[cam_id].copy()
    matrix = _extract_intrinsics_from_bundle(cam_id, calib_bundle)
    if matrix is None:
        return None
    _INTRINSICS_CACHE[cam_id] = matrix
    return matrix.copy()


def _extract_intrinsics_from_bundle(cam_id: str, calib_bundle: Optional[Mapping[str, object]]) -> Optional[np.ndarray]:
    if not calib_bundle:
        _logger.warning(f"No calibration bundle available for {cam_id}")
        return None
    cameras = calib_bundle.get('cameras') if isinstance(calib_bundle, Mapping) else None
    matrix = None
    if isinstance(cameras, Mapping):
        k_table = cameras.get('K') if isinstance(cameras.get('K'), Mapping) else None
        if isinstance(k_table, Mapping):
            matrix = _coerce_intrinsics(k_table.get(cam_id))
        if matrix is None and cam_id in cameras:
            matrix = _coerce_intrinsics(cameras[cam_id])
        if matrix is None:
            e_table = cameras.get(cam_id)
            if isinstance(e_table, Mapping):
                matrix = _coerce_intrinsics(e_table.get('K') or e_table.get('intrinsics'))
    if matrix is None:
        matrix = _fallback_intrinsics_from_cameras_yaml(cam_id)
    if matrix is None:
        _logger.warning(f"Unable to find intrinsics for camera {cam_id}")
    elif cam_id not in _INTRINSICS_CACHE:
        _logger.info(f"Using fallback intrinsics for camera {cam_id}")
    return matrix


def _fallback_intrinsics_from_cameras_yaml(cam_id: str) -> Optional[np.ndarray]:
    try:
        with _FALLBACK_LOCK:
            if not _CAMERAS_YAML_PATH.exists():
                return None
            global _FALLBACK_CAM_YAML_CACHE, _FALLBACK_CAM_YAML_MTIME
            mtime = _CAMERAS_YAML_PATH.stat().st_mtime
            if _FALLBACK_CAM_YAML_CACHE is None or _FALLBACK_CAM_YAML_MTIME != mtime:
                data = yaml.safe_load(_CAMERAS_YAML_PATH.read_text(encoding='utf-8')) or {}
                _FALLBACK_CAM_YAML_CACHE = data
                _FALLBACK_CAM_YAML_MTIME = mtime
            else:
                data = _FALLBACK_CAM_YAML_CACHE or {}
    except Exception as exc:
        _logger.debug(f"Failed to load fallback cameras.yaml: {exc}")
        return None

    if not isinstance(data, Mapping):
        return None
    cameras = data.get('cameras') or data.get('sources') or {}
    models = data.get('intrinsics_models') or data.get('models') or {}
    entry = None
    target = cam_id.strip().lower()
    for key, value in cameras.items():
        if not isinstance(value, Mapping):
            continue
        name = str(value.get('name') or key).strip().lower()
        if name == target:
            entry = value
            break
    if entry is None:
        # Try to parse numeric index from IDs like 'camera-1', 'camera_1', 'rtsp_0', or plain digits
        idx: Optional[int] = None
        try:
            lower = target
            if lower.startswith('camera-') or lower.startswith('camera_'):
                num = lower.split('-', 1)[-1] if '-' in lower else lower.split('_', 1)[-1]
                if num.isdigit():
                    # camera-1 refers to DS index 0
                    idx = max(0, int(num) - 1)
            elif lower.startswith('rtsp_') or lower.startswith('rtsp-'):
                num = lower.split('_', 1)[-1] if '_' in lower else lower.split('-', 1)[-1]
                if num.isdigit():
                    idx = int(num)
            elif lower.isdigit():
                # Ambiguous: treat as DS index
                idx = int(lower)
        except Exception:
            idx = None
        if idx is not None:
            for key, value in cameras.items():
                try:
                    key_int = int(str(key))
                except Exception:
                    continue
                if key_int == idx and isinstance(value, Mapping):
                    entry = value
                    break
        if entry is None:
            return None
    intr_data = entry.get('intrinsics')
    if intr_data is None:
        model_key = entry.get('model') or entry.get('intrinsics_model')
        if model_key and model_key in models and isinstance(models[model_key], Mapping):
            intr_data = models[model_key].get('intrinsics') or models[model_key]
    return _coerce_intrinsics(intr_data)


def _coerce_intrinsics(value: object) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, np.ndarray) and value.shape == (3, 3):
        return value.astype(np.float32)
    if isinstance(value, (list, tuple)):
        if len(value) == 4:
            fx, fy, cx, cy = map(float, value)
            return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        if len(value) == 9:
            arr = np.array(value, dtype=np.float32).reshape((3, 3))
            return arr
    if isinstance(value, Mapping):
        for key in ('K', 'K3x3', 'matrix'):
            matrix = _coerce_intrinsics(value.get(key))  # type: ignore[index]
            if matrix is not None:
                return matrix
        if all(k in value for k in ('fx', 'fy', 'cx', 'cy')):
            fx = float(value['fx'])
            fy = float(value['fy'])
            cx = float(value['cx'])
            cy = float(value['cy'])
            return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return None


def _scale_intrinsics(intrinsics: Optional[np.ndarray], scale_x: float, scale_y: float) -> Optional[np.ndarray]:
    if intrinsics is None:
        return None
    scaled = intrinsics.astype(np.float32).copy()
    scaled[0, 0] *= scale_x
    scaled[1, 1] *= scale_y
    scaled[0, 2] *= scale_x
    scaled[1, 2] *= scale_y
    return scaled


def _encode_view_payload(cam_id: str, frame_rgb: np.ndarray, intrinsics: Optional[np.ndarray]) -> Dict[str, object]:
    frame_contiguous = np.ascontiguousarray(frame_rgb)
    img_b64 = base64.b64encode(frame_contiguous.tobytes()).decode('ascii')
    payload: Dict[str, object] = {
        'cam_id': cam_id,
        'img_b64': img_b64,
        'shape': (frame_rgb.shape[0], frame_rgb.shape[1], frame_rgb.shape[2]),
    }
    if intrinsics is not None:
        payload['intrinsics'] = intrinsics.tolist()
    return payload


__all__ = [
    'build_mono_view',
    'build_multi_views',
    'ViewBuildResult',
]
