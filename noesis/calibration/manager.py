"""CalibrationManager — single owner for DS8 calibration data.

Consolidates loading, validation, and broadcasting of:
- K (intrinsics) from config/cameras.yaml ONLY
- E (extrinsics) from config/camera_calibration.json
- align from config/ply_alignment.json

See plans/DS8/ds8_calibration_workflow_unification_work_order.md for conventions.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from noesis.metadata.intrinsics import CameraConfigLoader, CameraIntrinsics

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CalibrationValidationError(Exception):
    """Raised when calibration data fails validation."""

    pass


# ---------------------------------------------------------------------------
# CalibrationSnapshot (immutable per-camera calibration)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationSnapshot:
    """Immutable calibration snapshot for a single camera."""

    camera_id: str
    intrinsics: np.ndarray  # 3x3 K matrix
    extrinsics_col_major: List[float]  # 16 floats, column-major E (world->camera)
    floor_y: float  # meters
    image_size: Tuple[int, int]  # (width, height)
    unit_scale: float = 1.0  # s_obj_to_m


# ---------------------------------------------------------------------------
# File I/O helpers (reused from calibration_bundle.py patterns)
# ---------------------------------------------------------------------------


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        if not path or not os.path.exists(path):
            return None
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _write_json(path: str, data: Dict[str, Any]) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_E(E: List[float], camera_id: str) -> None:
    """Validate extrinsics matrix E (16 floats, column-major, world->camera)."""
    if not isinstance(E, list) or len(E) != 16:
        raise CalibrationValidationError(f"{camera_id}: E must be 16 floats, got {len(E) if isinstance(E, list) else type(E)}")

    try:
        E_arr = np.array(E, dtype=np.float64).reshape((4, 4), order="F")
    except Exception as exc:
        raise CalibrationValidationError(f"{camera_id}: E reshape failed: {exc}") from exc

    # Check for NaN/Inf
    if not np.all(np.isfinite(E_arr)):
        raise CalibrationValidationError(f"{camera_id}: E contains NaN or Inf")

    # Check invertibility
    det = np.linalg.det(E_arr)
    if abs(det) < 1e-9:
        raise CalibrationValidationError(f"{camera_id}: E is singular (det={det:.2e})")

    # Check rotation orthonormality (top-left 3x3)
    R = E_arr[:3, :3]
    RtR = R.T @ R
    I = np.eye(3)
    ortho_err = np.linalg.norm(RtR - I, "fro")
    if ortho_err > 1e-3:
        raise CalibrationValidationError(f"{camera_id}: E rotation not orthonormal (err={ortho_err:.4f})")

    # Check rotation det ≈ 1 (not reflection)
    rot_det = np.linalg.det(R)
    if abs(rot_det - 1.0) > 1e-3:
        raise CalibrationValidationError(f"{camera_id}: E rotation det={rot_det:.4f}, expected ~1")


def _format_mat4_col_major(values: object) -> str:
    try:
        if not (isinstance(values, list) and len(values) == 16):
            return str(values)
        mat = np.array([float(x) for x in values], dtype=np.float64).reshape((4, 4), order="F")
        return np.array2string(mat, precision=6, suppress_small=True)
    except Exception:
        return str(values)


def _validate_K(K: np.ndarray, camera_id: str) -> None:
    """Validate intrinsics matrix K (3x3)."""
    if K.shape != (3, 3):
        raise CalibrationValidationError(f"{camera_id}: K must be 3x3, got {K.shape}")

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    if fx <= 0 or fy <= 0:
        raise CalibrationValidationError(f"{camera_id}: K focal lengths must be positive (fx={fx}, fy={fy})")

    # Principal point reasonable bounds (should be within typical image dimensions)
    if cx < 0 or cy < 0 or cx > 10000 or cy > 10000:
        raise CalibrationValidationError(f"{camera_id}: K principal point out of bounds (cx={cx}, cy={cy})")


def _validate_align_matrix(matrix: List[float]) -> None:
    """Validate alignment matrix (16 floats, row-major)."""
    if not isinstance(matrix, list) or len(matrix) != 16:
        raise CalibrationValidationError(f"align.matrix must be 16 floats, got {len(matrix) if isinstance(matrix, list) else type(matrix)}")

    try:
        M = np.array(matrix, dtype=np.float64).reshape((4, 4))
    except Exception as exc:
        raise CalibrationValidationError(f"align.matrix reshape failed: {exc}") from exc

    if not np.all(np.isfinite(M)):
        raise CalibrationValidationError("align.matrix contains NaN or Inf")

    det = np.linalg.det(M)
    if abs(det) < 1e-9:
        raise CalibrationValidationError(f"align.matrix is singular (det={det:.2e})")


# ---------------------------------------------------------------------------
# CalibrationManager
# ---------------------------------------------------------------------------


class CalibrationManager:
    """Single owner for DS8 calibration loading, validation, and broadcasting.

    Responsibilities:
    - Load K from config/cameras.yaml ONLY (deprecate intrinsics.json, config.py)
    - Apply streammux resolution scaling to K
    - Load E from config/camera_calibration.json
    - Load align from config/ply_alignment.json
    - Validate all inputs
    - Build CalibrationSnapshot and WS calibration-bundle
    - Provide reload entrypoints
    """

    def __init__(
        self,
        cameras_yaml_path: Path,
        camera_calibration_json_path: Path,
        ply_alignment_json_path: Path,
        streammux_size: Tuple[int, int] = (1920, 1080),
    ) -> None:
        self._cameras_yaml_path = Path(cameras_yaml_path)
        self._camera_calibration_path = Path(camera_calibration_json_path)
        self._ply_alignment_path = Path(ply_alignment_json_path)
        self._streammux_size = streammux_size

        self._lock = threading.RLock()
        self._intrinsics_loader = CameraConfigLoader(self._cameras_yaml_path)

        # Cached data
        self._extrinsics: Dict[str, Any] = {}
        self._align: Dict[str, Any] = {}
        self._camera_labels: Dict[int, str] = {}  # source_id -> camera_name
        self._bundle_cache: Optional[Dict[str, Any]] = None

        # Load initial data
        self._load_extrinsics()
        self._load_alignment()

        # Optional callback for derived artifact regeneration (V3DT camInfo)
        self._on_extrinsics_changed: Optional[Callable[[str], None]] = None

    # -----------------------------------------------------------------------
    # Public API: Configuration
    # -----------------------------------------------------------------------

    def set_camera_labels(self, labels: Dict[int, str]) -> None:
        """Set the mapping from source_id to camera name."""
        with self._lock:
            self._camera_labels = dict(labels or {})
            self._bundle_cache = None

    def set_on_extrinsics_changed(self, callback: Optional[Callable[[str], None]]) -> None:
        """Set callback to invoke when extrinsics change (for V3DT camInfo regeneration)."""
        self._on_extrinsics_changed = callback

    # -----------------------------------------------------------------------
    # Public API: Reload
    # -----------------------------------------------------------------------

    def reload_extrinsics(self) -> None:
        """Reload extrinsics from camera_calibration.json."""
        with self._lock:
            self._load_extrinsics()
            self._bundle_cache = None

    def reload_alignment(self) -> None:
        """Reload alignment from ply_alignment.json."""
        with self._lock:
            self._load_alignment()
            self._bundle_cache = None

    def reload_intrinsics(self) -> None:
        """Invalidate and reload intrinsics from cameras.yaml."""
        with self._lock:
            self._intrinsics_loader.invalidate()
            self._bundle_cache = None

    def reload_all(self) -> None:
        """Reload all calibration data."""
        self.reload_intrinsics()
        self.reload_extrinsics()
        self.reload_alignment()

    # -----------------------------------------------------------------------
    # Public API: Snapshot
    # -----------------------------------------------------------------------

    def snapshot(self, source_id: int, camera_id: str) -> Optional[CalibrationSnapshot]:
        """Get calibration snapshot for a camera, with K scaled to streammux resolution."""
        with self._lock:
            intr = self._intrinsics_loader.get(source_id)
            if intr is None:
                _LOGGER.warning("snapshot: no intrinsics for source_id=%d camera_id=%s", source_id, camera_id)
                return None

            K = self._build_scaled_K(intr, camera_id)
            if K is None:
                return None

            # Validate K
            try:
                _validate_K(K, camera_id)
            except CalibrationValidationError as exc:
                _LOGGER.error("snapshot validation failed: %s", exc)
                return None

            # Get extrinsics
            E = self._get_E(camera_id)
            if E is None:
                _LOGGER.warning("snapshot: no extrinsics for camera_id=%s", camera_id)
                return None

            # Validate E
            try:
                _validate_E(E, camera_id)
            except CalibrationValidationError as exc:
                _LOGGER.error("snapshot validation failed: %s", exc)
                return None

            floor_y = float(self._align.get("floor_y", 0.0) or 0.0)
            try:
                unit_scale = float((self._align.get("units") or {}).get("s_obj_to_m", 1.0))
            except Exception:
                unit_scale = 1.0

            return CalibrationSnapshot(
                camera_id=camera_id,
                intrinsics=K,
                extrinsics_col_major=list(E),
                floor_y=floor_y,
                image_size=self._streammux_size,
                unit_scale=unit_scale,
            )

    # -----------------------------------------------------------------------
    # Public API: Bundle
    # -----------------------------------------------------------------------

    def calibration_bundle(self) -> Dict[str, Any]:
        """Build the WS calibration-bundle payload."""
        with self._lock:
            if self._bundle_cache is not None:
                return dict(self._bundle_cache)

            camera_ids = sorted({name for name in self._camera_labels.values() if isinstance(name, str)})
            if not camera_ids:
                return {}

            k_table: Dict[str, List[float]] = {}
            e_table: Dict[str, List[float]] = {}

            for src_id, cam_name in self._camera_labels.items():
                # Intrinsics
                intr = self._intrinsics_loader.get(src_id)
                if intr is not None:
                    K = self._build_scaled_K(intr, cam_name)
                    if K is not None:
                        k_table[cam_name] = [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]

                # Extrinsics
                E = self._get_E(cam_name)
                if E is not None:
                    e_table[cam_name] = list(E)

            # Alignment
            align_matrix = self._align.get("matrix")
            if not isinstance(align_matrix, list) or len(align_matrix) != 16:
                align_matrix = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

            floor_y = float(self._align.get("floor_y", 0.0) or 0.0)
            units = self._align.get("units") or {}
            s_obj_to_m = float(units.get("s_obj_to_m", 1.0) or 1.0)

            bundle: Dict[str, Any] = {
                "align": {
                    "matrix": [float(x) for x in align_matrix],
                    "floor_y": floor_y,
                    "units": {"s_obj_to_m": s_obj_to_m},
                },
                "cameras": {
                    "K": k_table,
                    "E": e_table,
                    "pose_confidence": {},
                },
                "meta": {
                    "version": 2,
                    "conventions": {"E": "world→camera", "handedness": "RH", "up": "Y"},
                },
                "metric_scale": 1.0,
            }

            self._bundle_cache = bundle
            return dict(bundle)

    # -----------------------------------------------------------------------
    # Public API: Persistence (set_extrinsics, set_align)
    # -----------------------------------------------------------------------

    def _coerce_extrinsics_translation_to_meters(self, camera_id: str, E_col_major: List[float]) -> Tuple[List[float], Optional[str]]:
        """Best-effort normalize incoming extrinsics translation units to meters.

        DS8 conventions require `config/camera_calibration.json` to store E (world→camera)
        in meters. Some upstream calibration tools/UI flows may emit translations in centimeters.

        Control via env:
          - NOESIS_EXTRINSICS_INPUT_UNITS=auto|m|cm (default: auto)
        """
        mode = str(os.environ.get("NOESIS_EXTRINSICS_INPUT_UNITS", "auto") or "").strip().lower()
        if mode in ("m", "meter", "meters"):
            return E_col_major, None

        def _scale_translation(scale: float, reason: str) -> Tuple[List[float], str]:
            try:
                E_arr = np.array(E_col_major, dtype=np.float64).reshape((4, 4), order="F")
                E_arr = E_arr.copy()
                E_arr[:3, 3] *= float(scale)
                return list(E_arr.flatten(order="F")), reason
            except Exception:
                # Fall back to list slicing if reshape fails (shouldn't happen after validation).
                out = list(E_col_major)
                for idx in (12, 13, 14):
                    try:
                        out[idx] = float(out[idx]) * float(scale)
                    except Exception:
                        pass
                return out, reason

        if mode in ("cm", "centimeter", "centimeters"):
            return _scale_translation(0.01, "cm→m (NOESIS_EXTRINSICS_INPUT_UNITS=cm)")

        # Auto-detect cm-like inputs using camera height prior (best) or magnitude fallback.
        try:
            E_arr = np.array(E_col_major, dtype=np.float64).reshape((4, 4), order="F")
            R = E_arr[:3, :3]
            t = E_arr[:3, 3]
            C_world = -R.T @ t
            C_y = float(C_world[1])
        except Exception:
            return E_col_major, None

        expected_y_m: Optional[float] = None
        try:
            floor_y = float(self._align.get("floor_y", 0.0) or 0.0)
        except Exception:
            floor_y = 0.0
        try:
            # Resolve camera height from cameras.yaml using the source-id → camera-name mapping.
            src_id = None
            for sid, name in (self._camera_labels or {}).items():
                if name == camera_id:
                    src_id = int(sid)
                    break
            if src_id is not None:
                intr = self._intrinsics_loader.get(src_id)
                if intr is not None:
                    height_m = float(getattr(intr, "height_m", 0.0) or 0.0)
                    if math.isfinite(height_m) and height_m > 0.0:
                        expected_y_m = float(floor_y) + float(height_m)
        except Exception:
            expected_y_m = None

        if expected_y_m is not None and math.isfinite(expected_y_m):
            err_m = abs(C_y - expected_y_m)
            err_cm = abs((C_y / 100.0) - expected_y_m)
            if err_cm <= 1.0 and err_m >= 10.0:
                return _scale_translation(0.01, f"cm→m (auto; C_y={C_y:.3f} looked like cm, expected~{expected_y_m:.3f}m)")
            return E_col_major, None

        # Conservative fallback: camera height should not be tens/hundreds of meters in home scenes.
        if abs(C_y) > 20.0 and abs(C_y / 100.0) < 20.0:
            return _scale_translation(0.01, f"cm→m (auto; |C_y|={abs(C_y):.3f} too large for meters)")

        return E_col_major, None

    def _log_raw_extrinsics_payload(
        self,
        camera_id: str,
        raw_kind: str,
        raw_payload: Dict[str, Any],
        stored_E: List[float],
        units_note: Optional[str],
    ) -> None:
        if not camera_id:
            return
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        repo_root = self._camera_calibration_path.parent.parent
        out_dir = repo_root / "logs" / "calibration_raw"
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(camera_id))
        out_path = out_dir / f"{ts}_{safe_name}.json"
        payload = {
            "timestamp": ts,
            "camera_id": camera_id,
            "raw_kind": raw_kind,
            "raw_payload": raw_payload,
            "stored_E": stored_E,
            "units_note": units_note,
            "env": {
                "NOESIS_EXTRINSICS_INPUT_UNITS": os.environ.get("NOESIS_EXTRINSICS_INPUT_UNITS"),
            },
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def set_extrinsics(
        self,
        camera_id: str,
        E: Optional[List[float]] = None,
        Twc: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Persist extrinsics for a camera. Accepts E (world->camera) or Twc (camera->world).

        Returns: {"ok": bool, "error": str|None}
        """
        if not camera_id:
            return {"ok": False, "error": "cameraId_required"}

        # Convert Twc to E if provided
        E_to_save: Optional[List[float]] = None
        raw_payload: Dict[str, Any] = {}
        raw_kind = "unknown"
        if E is not None and isinstance(E, list) and len(E) == 16:
            E_to_save = [float(x) for x in E]
            raw_payload = {"E": list(E)}
            raw_kind = "E"
        elif Twc is not None and isinstance(Twc, list) and len(Twc) == 16:
            try:
                Twc_arr = np.array(Twc, dtype=np.float64).reshape((4, 4), order="F")
                E_arr = np.linalg.inv(Twc_arr)
                E_to_save = list(E_arr.flatten(order="F"))
                raw_payload = {"Twc": list(Twc)}
                raw_kind = "Twc"
            except Exception as exc:
                return {"ok": False, "error": f"Twc_inversion_failed: {exc}"}
        else:
            return {"ok": False, "error": "E_or_Twc_required"}

        E_to_save, units_note = self._coerce_extrinsics_translation_to_meters(camera_id, E_to_save)
        if units_note:
            _LOGGER.warning("set_extrinsics: coerced translation units for %s: %s", camera_id, units_note)
        try:
            self._log_raw_extrinsics_payload(camera_id, raw_kind, raw_payload, E_to_save, units_note)
        except Exception:
            _LOGGER.debug("Failed to log raw extrinsics payload", exc_info=True)

        # Validate before persisting
        try:
            _validate_E(E_to_save, camera_id)
        except CalibrationValidationError as exc:
            return {"ok": False, "error": str(exc)}

        # Persist
        if not self._save_extrinsics(camera_id, E_to_save):
            return {"ok": False, "error": "persist_failed"}

        # Reload and invalidate cache
        self.reload_extrinsics()

        # Notify callback for derived artifacts
        if self._on_extrinsics_changed:
            try:
                self._on_extrinsics_changed(camera_id)
            except Exception:
                _LOGGER.debug("on_extrinsics_changed callback failed", exc_info=True)

        return {"ok": True}

    def set_align(self, align_update: Dict[str, Any]) -> Dict[str, Any]:
        """Persist alignment data. Supports partial updates (matrix, floor_y, units).

        Returns: {"ok": bool, "error": str|None}
        """
        if not isinstance(align_update, dict):
            return {"ok": False, "error": "align_required"}

        # Validate matrix if provided
        matrix = align_update.get("matrix")
        if matrix is not None:
            try:
                _validate_align_matrix(matrix)
            except CalibrationValidationError as exc:
                return {"ok": False, "error": str(exc)}

        # Validate floor_y if provided
        floor_y = align_update.get("floor_y")
        if floor_y is not None and not isinstance(floor_y, (int, float)):
            return {"ok": False, "error": "invalid_floor_y"}

        # Validate s_obj_to_m if provided
        units = align_update.get("units")
        if isinstance(units, dict):
            s_obj_to_m = units.get("s_obj_to_m")
            if s_obj_to_m is not None:
                try:
                    sval = float(s_obj_to_m)
                    if sval <= 0:
                        return {"ok": False, "error": "invalid_s_obj_to_m"}
                except Exception:
                    return {"ok": False, "error": "invalid_s_obj_to_m"}

        # Persist
        if not self._save_alignment(align_update):
            return {"ok": False, "error": "persist_failed"}

        # Reload and invalidate cache
        self.reload_alignment()

        return {"ok": True}

    # -----------------------------------------------------------------------
    # Internal: Loading
    # -----------------------------------------------------------------------

    def _load_extrinsics(self) -> None:
        """Load extrinsics from camera_calibration.json."""
        data = _read_json(str(self._camera_calibration_path))
        if not isinstance(data, dict):
            self._extrinsics = {"cameras": {}}
            return

        cams = data.get("cameras") or {}
        parsed: Dict[str, Dict[str, Any]] = {}
        for cam_id, entry in cams.items():
            if not isinstance(entry, dict):
                continue
            E = entry.get("E")
            if isinstance(E, list) and len(E) == 16:
                # Log validation warnings but don't reject at load time
                try:
                    _validate_E(E, cam_id)
                except CalibrationValidationError as exc:
                    _LOGGER.warning("Invalid extrinsics at load time: %s", exc)
                parsed[cam_id] = {"E": [float(x) for x in E]}

        self._extrinsics = {"cameras": parsed}

    def _load_alignment(self) -> None:
        """Load alignment from ply_alignment.json."""
        data = _read_json(str(self._ply_alignment_path))
        default = {
            "matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            "floor_y": 0.0,
            "units": {"s_obj_to_m": 1.0},
        }
        if not isinstance(data, dict):
            self._align = default
            return

        # Parse matrix
        mat = data.get("matrix") or data.get("align", {}).get("matrix")
        if isinstance(mat, list) and len(mat) == 16:
            try:
                _validate_align_matrix(mat)
                default["matrix"] = [float(x) for x in mat]
            except CalibrationValidationError as exc:
                _LOGGER.warning("Invalid alignment matrix at load time: %s", exc)

        # Parse floor_y
        fy = data.get("floor_y") or data.get("align", {}).get("floor_y")
        if isinstance(fy, (int, float)):
            default["floor_y"] = float(fy)

        # Parse units
        units = data.get("units") or data.get("align", {}).get("units")
        if isinstance(units, dict):
            s = units.get("s_obj_to_m")
            if isinstance(s, (int, float)) and float(s) > 0:
                default["units"] = {"s_obj_to_m": float(s)}

        self._align = default

    # -----------------------------------------------------------------------
    # Internal: Helpers
    # -----------------------------------------------------------------------

    def _get_E(self, camera_id: str) -> Optional[List[float]]:
        """Get extrinsics for a camera."""
        cams = self._extrinsics.get("cameras") or {}
        entry = cams.get(camera_id)
        if not isinstance(entry, dict):
            return None
        E = entry.get("E")
        if isinstance(E, list) and len(E) == 16:
            return E
        return None

    def _build_scaled_K(self, intr: CameraIntrinsics, camera_id: str) -> Optional[np.ndarray]:
        """Build K matrix from intrinsics, scaled to streammux resolution."""
        try:
            K = np.array(
                [
                    [float(intr.fx), 0.0, float(intr.cx)],
                    [0.0, float(intr.fy), float(intr.cy)],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
        except Exception:
            return None

        frame_w, frame_h = self._streammux_size
        if frame_w <= 0 or frame_h <= 0:
            frame_w, frame_h = 1920, 1080

        # Determine base resolution from intrinsics principal point
        # (cx, cy assumed to be at image center)
        base_w = int(round(float(K[0, 2]) * 2.0))
        base_h = int(round(float(K[1, 2]) * 2.0))

        if base_w > 0 and base_h > 0 and (base_w != frame_w or base_h != frame_h):
            sx = float(frame_w) / float(base_w)
            sy = float(frame_h) / float(base_h)
            K = K.copy()
            K[0, 0] *= sx
            K[0, 2] *= sx
            K[1, 1] *= sy
            K[1, 2] *= sy

        return K

    def _save_extrinsics(self, camera_id: str, E: List[float]) -> bool:
        """Persist extrinsics to camera_calibration.json."""
        try:
            current = _read_json(str(self._camera_calibration_path)) or {}
            if "cameras" not in current or not isinstance(current["cameras"], dict):
                current["cameras"] = {}
            current["cameras"][camera_id] = {"E": [float(x) for x in E]}
            ok = _write_json(str(self._camera_calibration_path), current)
            if ok:
                _LOGGER.warning(
                    "Saved extrinsics to %s camera=%s E_col_major=%s",
                    str(self._camera_calibration_path),
                    camera_id,
                    [float(x) for x in E],
                )
                _LOGGER.warning("Saved extrinsics camera=%s E_matrix=%s", camera_id, _format_mat4_col_major(E))
            return ok
        except Exception:
            _LOGGER.exception("Failed to save extrinsics for %s", camera_id)
            return False

    def _save_alignment(self, align_update: Dict[str, Any]) -> bool:
        """Persist alignment to ply_alignment.json."""
        try:
            current = _read_json(str(self._ply_alignment_path)) or {
                "matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                "floor_y": 0.0,
                "units": {"s_obj_to_m": 1.0},
            }

            if "matrix" in align_update and isinstance(align_update["matrix"], list) and len(align_update["matrix"]) == 16:
                current["matrix"] = [float(x) for x in align_update["matrix"]]

            if "floor_y" in align_update and isinstance(align_update["floor_y"], (int, float)):
                current["floor_y"] = float(align_update["floor_y"])

            if "units" in align_update and isinstance(align_update["units"], dict):
                s = align_update["units"].get("s_obj_to_m")
                if isinstance(s, (int, float)) and float(s) > 0:
                    current["units"] = {"s_obj_to_m": float(s)}

            return _write_json(str(self._ply_alignment_path), current)
        except Exception:
            _LOGGER.exception("Failed to save alignment")
            return False
