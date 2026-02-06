"""Utilities for supplying DS8 calibration poses to MapAnything.

MapAnything expects per-view `camera_poses` in **OpenCV cam→world** convention:
  - +X right, +Y down, +Z forward (camera frame)
  - 4x4 `T_wc` (camera-to-world)

Noesis DS8 stores per-camera extrinsics as `E` in `config/camera_calibration.json`:
  - `E` is **world→camera**
  - flat 16 floats, **column-major** (Fortran order)

This module converts `E` into MapAnything-compatible `camera_poses` and (optionally)
applies `config/ply_alignment.json`:
  - First apply the alignment rigid transform `M` (row-major) as `T' = M @ T_wc`
  - Then apply unit scaling to translation only via `units.s_obj_to_m`
    (pose rotations remain orthonormal; scaling is a world-unit conversion)

Conventions Reference:
  - See `docs/DS8_api_contracts_ws.md` §8 for unified calibration data conventions.
  - See `plans/DS8/ds8_design_decisions.md` for MapAnything pose conditioning rationale.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from calibration_bundle import load_alignment, load_extrinsics


def _alignment_components(align: Optional[Dict[str, Any]]) -> Tuple[np.ndarray, float]:
    matrix = np.eye(4, dtype=np.float64)
    scale = 1.0
    if isinstance(align, dict):
        mat = align.get("matrix")
        if isinstance(mat, list) and len(mat) == 16:
            try:
                matrix = np.array(mat, dtype=np.float64).reshape((4, 4), order="C")
            except Exception:
                matrix = np.eye(4, dtype=np.float64)
        units = align.get("units")
        if isinstance(units, dict):
            s = units.get("s_obj_to_m")
            if isinstance(s, (int, float)):
                try:
                    scale = float(s)
                except Exception:
                    scale = 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return matrix, scale


def mapanything_camera_pose_cam2world_opencv(
    E_world_to_camera_col_major_16: Any,
    *,
    alignment: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Convert Noesis `E` (world→camera) into MapAnything `camera_poses` (cam→world).

    Returns a 4x4 float64 camera-to-world matrix in OpenCV camera convention.
    """
    if not (isinstance(E_world_to_camera_col_major_16, list) and len(E_world_to_camera_col_major_16) == 16):
        raise ValueError("Expected E as a list[16] (column-major world→camera)")

    E = np.array(E_world_to_camera_col_major_16, dtype=np.float64).reshape((4, 4), order="F")
    try:
        T_wc = np.linalg.inv(E)
    except Exception as exc:
        raise ValueError("Invalid extrinsics E (non-invertible)") from exc

    M_align, s_obj_to_m = _alignment_components(alignment)
    try:
        T_wc = M_align @ T_wc
    except Exception:
        # Defensive: if alignment is malformed, fall back to unaligned pose.
        pass

    # Apply unit scaling to translation only (world units conversion).
    T_wc = np.array(T_wc, dtype=np.float64, copy=True)
    T_wc[:3, 3] *= float(s_obj_to_m)
    T_wc[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return T_wc


def load_mapanything_camera_poses(
    *,
    extrinsics_path: Path,
    alignment_path: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """Load all camera poses from config files as MapAnything-ready cam→world matrices."""
    extr = load_extrinsics(str(extrinsics_path))
    align = load_alignment(str(alignment_path)) if alignment_path else None
    out: Dict[str, np.ndarray] = {}
    cameras = extr.get("cameras") if isinstance(extr, dict) else None
    if not isinstance(cameras, dict):
        return out
    for cam_id, entry in cameras.items():
        if not isinstance(cam_id, str) or not isinstance(entry, dict):
            continue
        E = entry.get("E")
        if not (isinstance(E, list) and len(E) == 16):
            continue
        try:
            out[cam_id] = mapanything_camera_pose_cam2world_opencv(E, alignment=align)
        except Exception:
            continue
    return out


@dataclass
class MapAnythingPoseProvider:
    """File-backed pose provider that auto-reloads when calibration changes."""

    extrinsics_path: Path
    alignment_path: Optional[Path] = None

    _cache: Dict[str, np.ndarray] = None  # type: ignore[assignment]
    _mtimes: Tuple[Optional[float], Optional[float]] = (None, None)

    def __post_init__(self) -> None:
        self.extrinsics_path = Path(self.extrinsics_path)
        self.alignment_path = Path(self.alignment_path) if self.alignment_path else None
        self._cache = {}
        self._mtimes = (None, None)

    def _current_mtimes(self) -> Tuple[Optional[float], Optional[float]]:
        def _mtime(path: Optional[Path]) -> Optional[float]:
            if not path:
                return None
            try:
                return float(os.path.getmtime(path))
            except Exception:
                return None

        return _mtime(self.extrinsics_path), _mtime(self.alignment_path)

    def _reload_if_needed(self) -> None:
        mtimes = self._current_mtimes()
        if mtimes == self._mtimes and self._cache:
            return
        self._cache = load_mapanything_camera_poses(
            extrinsics_path=self.extrinsics_path,
            alignment_path=self.alignment_path,
        )
        self._mtimes = mtimes

    def get(self, camera_id: str) -> Optional[np.ndarray]:
        self._reload_if_needed()
        return self._cache.get(str(camera_id))

    def all(self) -> Dict[str, np.ndarray]:
        self._reload_if_needed()
        return dict(self._cache)

