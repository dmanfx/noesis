from __future__ import annotations

import logging
import numbers
import re
import threading
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

try:  # pragma: no cover - DeepStream bindings not available during unit tests
    import pyds  # type: ignore
except ImportError:  # pragma: no cover - handled via graceful degradation
    pyds = None  # type: ignore[assignment]

_LOGGER = logging.getLogger(__name__)

_INTRINSICS_META_TYPE_NAME = "NOESIS.INTRINSICS"
_META_TYPE_ID: Optional[int] = None
_META_TYPE_LOCK = threading.Lock()
_LOG_ONCE_LOCK = threading.Lock()
_LOGGED_SOURCES: set[int] = set()
_MISSING_SOURCES: set[tuple[int, Path]] = set()


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    height_m: float = 0.0

    def as_payload(self) -> Dict[str, float]:
        """Return a dict payload ready for user-meta serialization."""
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "k1": self.k1,
            "k2": self.k2,
            "k3": self.k3,
            "height_m": self.height_m,
        }


@dataclass
class _IntrinsicsMetaPayload:
    payload: Dict[str, float]

    def clone(self) -> "_IntrinsicsMetaPayload":
        return _IntrinsicsMetaPayload(dict(self.payload))


class CameraConfigLoader:
    """Hot-reloadable loader for config/cameras.yaml."""

    def __init__(self, config_path: Path) -> None:
        self._config_path = Path(config_path)
        self._lock = threading.RLock()
        self._cached: Dict[int, CameraIntrinsics] = {}
        self._mtime: Optional[int] = None

    @property
    def config_path(self) -> Path:
        return self._config_path

    def get(self, source_id: int) -> Optional[CameraIntrinsics]:
        return self._ensure_loaded().get(source_id)

    def all(self) -> Dict[int, CameraIntrinsics]:
        return dict(self._ensure_loaded())

    def invalidate(self) -> None:
        with self._lock:
            self._mtime = None

    def _ensure_loaded(self) -> Dict[int, CameraIntrinsics]:
        with self._lock:
            path = self._config_path
            if not path.exists():
                if self._cached:
                    _LOGGER.warning(
                        "Camera intrinsics config %s missing; clearing cache", path
                    )
                    self._cached = {}
                    self._mtime = None
                return self._cached

            mtime_ns = path.stat().st_mtime_ns
            if self._mtime == mtime_ns:
                return self._cached

            try:
                self._cached = load_cameras_yaml(path)
            except Exception:  # pragma: no cover - surfaces via logs
                _LOGGER.exception("Failed to load camera intrinsics from %s", path)
                self._cached = {}
            else:
                _LOGGER.debug(
                    "Loaded %d camera intrinsics entries from %s",
                    len(self._cached),
                    path,
                )

            self._mtime = mtime_ns
            return self._cached


_DEFAULT_CONFIG_PATH = Path("config/cameras.yaml")
_GLOBAL_LOADER: Optional[CameraConfigLoader] = None
_GLOBAL_LOADER_LOCK = threading.Lock()


def load_cameras_yaml(path: str | Path) -> Dict[int, CameraIntrinsics]:
    """Parse a cameras.yaml configuration file."""
    cfg_path = Path(path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"Expected mapping root in {cfg_path}")

    cameras_section = raw.get("cameras") or raw.get("sources")
    if not isinstance(cameras_section, Mapping):
        raise ValueError(
            f"{cfg_path} must contain a mapping under 'cameras' (or 'sources')"
        )

    models_section = raw.get("intrinsics_models") or raw.get("models")
    if models_section is None:
        models = {}
    elif isinstance(models_section, Mapping):
        models = models_section
    else:
        raise TypeError("intrinsics_models must be a mapping when present")

    parsed: Dict[int, CameraIntrinsics] = {}
    for key, entry in cameras_section.items():
        if entry is None:
            continue
        if not isinstance(entry, Mapping):
            raise TypeError(f"Camera entry {key!r} must be a mapping")
        source_id = _coerce_source_id(key, entry)
        try:
            parsed[source_id] = _build_intrinsics(entry, models, cfg_path, source_id)
        except Exception as exc:
            _LOGGER.warning(
                "Skipping camera %s in %s due to invalid intrinsics: %s",
                key,
                cfg_path,
                exc,
            )
    return parsed


def attach_intrinsics(
    frame_meta: Any,
    source_id: int | str,
    *,
    loader: CameraConfigLoader | None = None,
    config_path: Path | None = None,
) -> None:
    """Attach intrinsics for the given source onto an NvDsFrameMeta or dict."""
    if loader and config_path:
        raise ValueError("Specify either loader or config_path, not both")

    normalized_id: int
    try:
        normalized_id = _parse_source_identifier(source_id)
    except ValueError:
        _LOGGER.debug("attach_intrinsics: unable to coerce source id %r", source_id)
        return

    active_loader = loader
    if active_loader is None:
        if config_path is not None:
            active_loader = CameraConfigLoader(Path(config_path))
        else:
            active_loader = _get_default_loader()

    intrinsics = active_loader.get(normalized_id)
    if intrinsics is None:
        _log_missing_once(normalized_id, active_loader.config_path)
        return

    payload = intrinsics.as_payload()
    if _attach_to_pyds_frame(frame_meta, payload, normalized_id):
        _log_attach_once(normalized_id, payload, "nvds")
        return

    if isinstance(frame_meta, MutableMapping):
        user_meta = frame_meta.setdefault("user_meta", {})
        if isinstance(user_meta, MutableMapping):
            user_meta["intrinsics"] = dict(payload)
            _log_attach_once(normalized_id, payload, "mapping")
            return

    if hasattr(frame_meta, "__dict__"):
        setattr(frame_meta, "noesis_intrinsics", dict(payload))
        _log_attach_once(normalized_id, payload, "attr")
        return

    _LOGGER.debug(
        "attach_intrinsics: unsupported frame_meta type %s for source %s",
        type(frame_meta).__name__,
        normalized_id,
    )


def _get_default_loader() -> CameraConfigLoader:
    global _GLOBAL_LOADER
    with _GLOBAL_LOADER_LOCK:
        if _GLOBAL_LOADER is None:
            _GLOBAL_LOADER = CameraConfigLoader(_DEFAULT_CONFIG_PATH)
        return _GLOBAL_LOADER


def _attach_to_pyds_frame(
    frame_meta: Any, payload: Dict[str, float], source_id: int
) -> bool:
    if pyds is None:
        return False
    if not hasattr(frame_meta, "frame_user_meta_list") or not hasattr(
        frame_meta, "batch_meta"
    ):
        return False

    batch_meta = getattr(frame_meta, "batch_meta", None)
    if batch_meta is None:
        return False

    try:
        user_meta = pyds.nvds_acquire_user_meta_from_pool(batch_meta)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - defensive guard
        _LOGGER.exception(
            "Failed to acquire NvDsUserMeta for source %s intrinsics", source_id
        )
        return False

    if not user_meta:
        return False

    meta_type = _ensure_meta_type()
    meta_payload = _IntrinsicsMetaPayload(dict(payload))
    user_meta.user_meta_data = meta_payload
    if meta_type is not None:
        user_meta.base_meta.meta_type = meta_type  # type: ignore[attr-defined]

    user_meta.base_meta.release_func = _intrinsics_release_func  # type: ignore[attr-defined]
    user_meta.base_meta.copy_func = _intrinsics_copy_func  # type: ignore[attr-defined]

    try:
        pyds.nvds_add_user_meta_to_frame(frame_meta, user_meta)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - defensive guard
        _LOGGER.exception(
            "Failed to attach intrinsics user meta for source %s", source_id
        )
        return False
    return True


def _intrinsics_copy_func(data: Any, user_data: Any) -> Any:  # pragma: no cover - DS runtime only
    if pyds is None:
        return data
    user_meta = pyds.NvDsUserMeta.cast(data)  # type: ignore[attr-defined]
    payload = user_meta.user_meta_data
    if isinstance(payload, _IntrinsicsMetaPayload):
        return payload.clone()
    if isinstance(payload, MutableMapping):
        return dict(payload)
    if isinstance(payload, Mapping):
        return dict(payload)
    return payload


def _intrinsics_release_func(data: Any, user_data: Any) -> None:  # pragma: no cover - DS runtime only
    if pyds is None:
        return
    user_meta = pyds.NvDsUserMeta.cast(data)  # type: ignore[attr-defined]
    payload = user_meta.user_meta_data
    if isinstance(payload, _IntrinsicsMetaPayload):
        payload.payload.clear()
    user_meta.user_meta_data = None


def _ensure_meta_type() -> Optional[int]:
    if pyds is None:
        return None
    global _META_TYPE_ID
    if _META_TYPE_ID is not None:
        return _META_TYPE_ID
    with _META_TYPE_LOCK:
        if _META_TYPE_ID is None:
            _META_TYPE_ID = pyds.nvds_get_user_meta_type(_INTRINSICS_META_TYPE_NAME)  # type: ignore[attr-defined]
    return _META_TYPE_ID


def _build_intrinsics(
    entry: Mapping[str, Any],
    models: Mapping[str, Any],
    cfg_path: Path,
    source_id: int,
) -> CameraIntrinsics:
    intr_section = _resolve_intrinsics_section(entry, models, cfg_path, source_id)

    fx = _lookup_float(intr_section, ("fx", "f_x"))
    fy = _lookup_float(intr_section, ("fy", "f_y"))
    cx = _lookup_float(intr_section, ("cx", "c_x"))
    cy = _lookup_float(intr_section, ("cy", "c_y"))

    if None in (fx, fy, cx, cy):
        matrix = intr_section.get("K_matrix") or intr_section.get("K") or intr_section.get(
            "camera_matrix"
        )
        if matrix is not None:
            fx = fx or _matrix_component(matrix, 0, 0)
            fy = fy or _matrix_component(matrix, 1, 1)
            cx = cx or _matrix_component(matrix, 0, 2)
            cy = cy or _matrix_component(matrix, 1, 2)

    if None in (fx, fy, cx, cy):
        raise ValueError("Missing fx/fy/cx/cy values")

    k1, k2, k3 = _extract_k_values(
        intr_section.get("distortion_coeffs"),
        entry.get("distortion_coeffs"),
        intr_section.get("distortion"),
        entry.get("distortion"),
        intr_section.get("distortion_params"),
        entry.get("distortion_params"),
        intr_section.get("coefficients"),
        entry.get("coefficients"),
        (
            intr_section.get("k1"),
            intr_section.get("k2"),
            intr_section.get("k3"),
        ),
        (
            entry.get("k1"),
            entry.get("k2"),
            entry.get("k3"),
        ),
    )

    height_m = _first_float(
        entry.get("height_m"),
        entry.get("mount_height_m"),
        _nested_get(entry, ("mount", "height_m")),
        intr_section.get("height_m"),
    )

    return CameraIntrinsics(
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
        k1=k1,
        k2=k2,
        k3=k3,
        height_m=height_m,
    )


def _resolve_intrinsics_section(
    entry: Mapping[str, Any],
    models: Mapping[str, Any],
    cfg_path: Path,
    source_id: int,
) -> Dict[str, Any]:
    direct_intr = entry.get("intrinsics")
    if isinstance(direct_intr, Mapping):
        return dict(direct_intr)
    if isinstance(direct_intr, str):
        return _load_model_intrinsics(direct_intr, models, cfg_path, source_id)

    model_ref = entry.get("intrinsics_model") or entry.get("model") or entry.get(
        "intrinsics_ref"
    )
    if isinstance(model_ref, str):
        return _load_model_intrinsics(model_ref, models, cfg_path, source_id)

    calibration = entry.get("calibration")
    if isinstance(calibration, Mapping):
        cal_intr = calibration.get("intrinsics")
        if isinstance(cal_intr, Mapping):
            return dict(cal_intr)

    # As a last resort treat the entry itself as intrinsics payload.
    return dict(entry)


def _load_model_intrinsics(
    model_key: str,
    models: Mapping[str, Any],
    cfg_path: Path,
    source_id: int,
) -> Dict[str, Any]:
    candidate = models.get(model_key)
    if candidate is None:
        raise ValueError(
            f"Unknown intrinsics model '{model_key}' for source {source_id} in {cfg_path}"
        )
    if not isinstance(candidate, Mapping):
        raise TypeError(f"Intrinsics model '{model_key}' must be a mapping")
    if "intrinsics" in candidate and isinstance(candidate["intrinsics"], Mapping):
        return dict(candidate["intrinsics"])
    return dict(candidate)


def _lookup_float(
    mapping: Mapping[str, Any],
    keys: Sequence[str],
) -> Optional[float]:
    for key in keys:
        if key in mapping:
            return _to_float(mapping[key])
    return None


def _matrix_component(matrix: Any, row: int, col: int) -> float:
    if isinstance(matrix, Mapping):
        matrix = matrix.get("data") or matrix.get("values") or matrix
    if not isinstance(matrix, Iterable):
        raise TypeError("camera matrix must be iterable")
    rows = list(matrix)
    try:
        target_row = rows[row]
        target_col = target_row[col]
    except (IndexError, TypeError):
        raise ValueError("camera matrix must be 3x3") from None
    return _to_float(target_col)


def _extract_k_values(*sources: Any) -> tuple[float, float, float]:
    for candidate in sources:
        if candidate is None:
            continue
        if isinstance(candidate, Mapping):
            values = [
                candidate.get("k1"),
                candidate.get("k2"),
                candidate.get("k3"),
            ]
        elif isinstance(candidate, Iterable) and not isinstance(candidate, (str, bytes)):
            seq = list(candidate)
            if not seq:
                continue
            values = [
                seq[0] if len(seq) > 0 else 0.0,
                seq[1] if len(seq) > 1 else 0.0,
                seq[2] if len(seq) > 2 else 0.0,
            ]
        else:
            continue
        return (
            _safe_float(values[0]),
            _safe_float(values[1]),
            _safe_float(values[2]),
        )
    return 0.0, 0.0, 0.0


def _first_float(*values: Any) -> float:
    for value in values:
        result = _safe_float(value, default=None)
        if result is not None:
            return result
    return 0.0


def _nested_get(entry: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = entry
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def _safe_float(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
    if value is None:
        return default
    try:
        return _to_float(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any) -> float:
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Empty string cannot be converted to float")
        return float(text)
    raise TypeError(f"Cannot convert {value!r} to float")


def _coerce_source_id(key: Any, entry: Mapping[str, Any]) -> int:
    for candidate in ("source_id", "source", "sensor_id"):
        if candidate in entry:
            return _parse_source_identifier(entry[candidate])
    return _parse_source_identifier(key)


def _parse_source_identifier(value: Any) -> int:
    if isinstance(value, numbers.Integral):
        return int(value)
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        match = re.search(r"-?\d+", text)
        if match:
            return int(match.group(0))
    raise ValueError(f"Unable to interpret {value!r} as source identifier")


def _log_attach_once(source_id: int, payload: Mapping[str, float], via: str) -> None:
    with _LOG_ONCE_LOCK:
        if source_id in _LOGGED_SOURCES:
            return
        _LOGGED_SOURCES.add(source_id)
    _LOGGER.debug(
        "Attached intrinsics for source %s via %s (fx=%.3f, fy=%.3f, cx=%.3f, cy=%.3f, h=%.3f)",
        source_id,
        via,
        payload.get("fx", 0.0),
        payload.get("fy", 0.0),
        payload.get("cx", 0.0),
        payload.get("cy", 0.0),
        payload.get("height_m", 0.0),
    )


def _log_missing_once(source_id: int, config_path: Path) -> None:
    key = (source_id, config_path.resolve())
    with _LOG_ONCE_LOCK:
        if key in _MISSING_SOURCES:
            return
        _MISSING_SOURCES.add(key)
    _LOGGER.warning(
        "No intrinsics entry for source %s in %s", source_id, config_path
    )
