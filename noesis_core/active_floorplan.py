"""Bounded runtime registry for the exact floorplan currently driving BEV."""

from __future__ import annotations

import json
import math
import re
import threading
from copy import deepcopy
from numbers import Integral
from types import MappingProxyType
from typing import Any, Mapping

_FRAME = "camera_local_ground_m"
_ORIENTATION = "camera_ground_right_forward"
_FLOORPLAN_CONTRACT_VERSION = 10
_UNITS = "meters"
_MAX_CAMERAS = 64
_MAX_ALIASES = 256
_MAX_IDENTIFIER_BYTES = 256
_MAX_GRID_DIMENSION = 16_384
_MAX_GRID_CELLS = 16 * 1024 * 1024
_MAX_GRID_VALUE_ABS = 1_000_000.0
_MAX_BOUND_ABS_M = 10_000.0
_MAX_GRID_RES_M = 10_000.0
_MAX_EXTENT_M = 10_000.0
_MAX_ALIGNMENT_BYTES = 64 * 1024
_MAX_ALIGNMENT_TEXT_BYTES = 512
_MAX_ALIGNMENT_COUNT = 100_000_000
_MAX_ALIGNMENT_VALUE_ABS = 1_000_000.0
_MAX_SNAPSHOT_REF_BYTES = 1024
_MAX_ERROR_TEXT_BYTES = 1024
_LAYER_NAMES = (
    "walkable",
    "height",
    "height_agl",
    "obstacle_height",
    "distance",
    "density",
    "gradient",
)
_ALIGNMENT_KEYS = frozenset(
    {
        "version",
        "quality",
        "reason",
        "source",
        "from",
        "to",
        "matrix_2x3",
        "sample_count",
        "inlier_count",
        "sample_mode",
        "residual_m",
        "determinant",
    }
)
_ALIGNMENT_QUALITY = frozenset({"ok", "high_residual", "unavailable"})
_ALIGNMENT_SOURCE = "floorplan_depth_snapshot"
_ALIGNMENT_FROM = "calibrated_floor_contact_ray_camera_local_xz"
_ALIGNMENT_TO = "floorplan_depth_camera_heading_ground"
_PORTABLE_REF_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ActiveFloorplanError(ValueError):
    """Raised when successful floorplan output violates the BEV contract."""


def _required_text(
    value: object,
    name: str,
    *,
    max_bytes: int = _MAX_IDENTIFIER_BYTES,
) -> str:
    text = str(value or "").strip()
    if not text:
        raise ActiveFloorplanError(f"{name} is required")
    if len(text.encode("utf-8")) > max_bytes:
        raise ActiveFloorplanError(f"{name} exceeds the bounded contract")
    return text


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ActiveFloorplanError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ActiveFloorplanError(f"{name} must be a positive integer")
    return result


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ActiveFloorplanError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0 or result > _MAX_ALIGNMENT_COUNT:
        raise ActiveFloorplanError(f"{name} exceeds the bounded contract")
    return result


def _finite_bounded(
    value: object,
    name: str,
    *,
    max_abs: float,
    positive: bool = False,
) -> float:
    if isinstance(value, bool):
        qualifier = "finite and positive" if positive else "finite"
        raise ActiveFloorplanError(f"{name} must be {qualifier}")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        qualifier = "finite and positive" if positive else "finite"
        raise ActiveFloorplanError(f"{name} must be {qualifier}") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ActiveFloorplanError(f"{name} must be {qualifier}")
    if abs(result) > max_abs:
        raise ActiveFloorplanError(f"{name} exceeds the bounded contract")
    return result


def _bounds(payload: Mapping[str, Any]) -> dict[str, float]:
    raw = payload.get("bounds")
    if not isinstance(raw, Mapping):
        raise ActiveFloorplanError("bounds are required")
    try:
        values = {
            name: _finite_bounded(
                raw[name], f"bounds.{name}", max_abs=_MAX_BOUND_ABS_M
            )
            for name in ("min_x", "max_x", "min_z", "max_z")
        }
    except KeyError as exc:
        raise ActiveFloorplanError("bounds must contain finite X/Z extents") from exc
    if values["max_x"] <= values["min_x"] or values["max_z"] <= values["min_z"]:
        raise ActiveFloorplanError("bounds must have positive X/Z extents")
    return values


def _grid_shape(payload: Mapping[str, Any]) -> tuple[list[int], str]:
    shapes: dict[str, tuple[int, int]] = {}
    for layer_name in _LAYER_NAMES:
        layer = payload.get(layer_name)
        if layer is None:
            continue
        if not isinstance(layer, Mapping):
            raise ActiveFloorplanError(f"{layer_name} must be an object")
        raw = layer.get("grid_shape", layer.get("shape"))
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ActiveFloorplanError(
                f"{layer_name}.grid_shape must contain exactly two dimensions"
            )
        rows = _positive_int(raw[0], f"{layer_name}.grid_shape[0]")
        columns = _positive_int(raw[1], f"{layer_name}.grid_shape[1]")
        if (
            rows > _MAX_GRID_DIMENSION
            or columns > _MAX_GRID_DIMENSION
            or rows * columns > _MAX_GRID_CELLS
        ):
            raise ActiveFloorplanError("floorplan grid shape exceeds the bounded contract")
        shapes[layer_name] = (rows, columns)

        value_min = None
        value_max = None
        if "value_min" in layer:
            value_min = _finite_bounded(
                layer["value_min"],
                f"{layer_name}.value_min",
                max_abs=_MAX_GRID_VALUE_ABS,
            )
        if "value_max" in layer:
            value_max = _finite_bounded(
                layer["value_max"],
                f"{layer_name}.value_max",
                max_abs=_MAX_GRID_VALUE_ABS,
            )
        if value_min is not None and value_max is not None and value_max < value_min:
            raise ActiveFloorplanError(
                f"{layer_name}.value_max must be greater than or equal to value_min"
            )

    if not shapes:
        raise ActiveFloorplanError("a positive floorplan grid shape is required")
    if len(set(shapes.values())) != 1:
        raise ActiveFloorplanError("all present floorplan layers must use one grid shape")
    source = next(layer for layer in _LAYER_NAMES if layer in shapes)
    rows, columns = shapes[source]
    return [rows, columns], source


def _alignment_text(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str):
        raise ActiveFloorplanError(f"ray_to_floorplan_alignment.{key} must be text")
    return _required_text(
        value,
        f"ray_to_floorplan_alignment.{key}",
        max_bytes=_MAX_ALIGNMENT_TEXT_BYTES,
    )


def _alignment_matrix(raw: object) -> list[list[float]]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ActiveFloorplanError(
            "ray_to_floorplan_alignment.matrix_2x3 must have shape 2x3"
        )
    matrix: list[list[float]] = []
    for row_index, raw_row in enumerate(raw):
        if not isinstance(raw_row, (list, tuple)) or len(raw_row) != 3:
            raise ActiveFloorplanError(
                "ray_to_floorplan_alignment.matrix_2x3 must have shape 2x3"
            )
        matrix.append(
            [
                _finite_bounded(
                    value,
                    f"ray_to_floorplan_alignment.matrix_2x3[{row_index}][{column_index}]",
                    max_abs=_MAX_ALIGNMENT_VALUE_ABS,
                )
                for column_index, value in enumerate(raw_row)
            ]
        )
    return matrix


def _alignment(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = payload.get("ray_to_floorplan_alignment")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ActiveFloorplanError("ray_to_floorplan_alignment must be an object")
    unknown = set(raw).difference(_ALIGNMENT_KEYS)
    if unknown:
        raise ActiveFloorplanError(
            "ray_to_floorplan_alignment contains unsupported fields: "
            + ", ".join(sorted(str(key) for key in unknown))
        )
    version = _positive_int(raw.get("version"), "ray_to_floorplan_alignment.version")
    if version != 1:
        raise ActiveFloorplanError("ray_to_floorplan_alignment.version must be 1")
    quality = _alignment_text(raw, "quality").lower()
    if quality not in _ALIGNMENT_QUALITY:
        raise ActiveFloorplanError("ray_to_floorplan_alignment.quality is unsupported")
    reason = _alignment_text(raw, "reason")
    source = _alignment_text(raw, "source")
    source_frame = _alignment_text(raw, "from")
    target_frame = _alignment_text(raw, "to")
    if (
        source != _ALIGNMENT_SOURCE
        or source_frame != _ALIGNMENT_FROM
        or target_frame != _ALIGNMENT_TO
    ):
        raise ActiveFloorplanError(
            "ray_to_floorplan_alignment frame identity does not match the floorplan contract"
        )

    decoded: dict[str, Any] = {
        "version": version,
        "quality": quality,
        "reason": reason,
        "source": source,
        "from": source_frame,
        "to": target_frame,
    }
    for key in ("sample_count", "inlier_count"):
        if key in raw:
            decoded[key] = _nonnegative_int(
                raw[key], f"ray_to_floorplan_alignment.{key}"
            )
    if "sample_mode" in raw:
        decoded["sample_mode"] = _alignment_text(raw, "sample_mode")

    matrix_raw = raw.get("matrix_2x3")
    residual_raw = raw.get("residual_m")
    determinant_raw = raw.get("determinant")
    has_fit = quality in {"ok", "high_residual"}
    if has_fit:
        matrix = _alignment_matrix(matrix_raw)
        sample_count = decoded.get("sample_count")
        inlier_count = decoded.get("inlier_count")
        if (
            not isinstance(sample_count, int)
            or not isinstance(inlier_count, int)
            or inlier_count <= 0
            or inlier_count > sample_count
        ):
            raise ActiveFloorplanError(
                "ray_to_floorplan_alignment fit counts are inconsistent"
            )
        if not isinstance(residual_raw, Mapping):
            raise ActiveFloorplanError(
                "ray_to_floorplan_alignment.residual_m must be an object"
            )
        expected_residual_keys = {"p50", "p90", "p95", "max"}
        if set(residual_raw) != expected_residual_keys:
            raise ActiveFloorplanError(
                "ray_to_floorplan_alignment.residual_m has an invalid shape"
            )
        residual = {
            key: _finite_bounded(
                residual_raw[key],
                f"ray_to_floorplan_alignment.residual_m.{key}",
                max_abs=_MAX_ALIGNMENT_VALUE_ABS,
            )
            for key in ("p50", "p90", "p95", "max")
        }
        if any(value < 0.0 for value in residual.values()) or not (
            residual["p50"]
            <= residual["p90"]
            <= residual["p95"]
            <= residual["max"]
        ):
            raise ActiveFloorplanError(
                "ray_to_floorplan_alignment residuals are inconsistent"
            )
        determinant = _finite_bounded(
            determinant_raw,
            "ray_to_floorplan_alignment.determinant",
            max_abs=_MAX_ALIGNMENT_VALUE_ABS,
        )
        computed_determinant = (
            matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        )
        tolerance = max(1e-6, abs(computed_determinant) * 1e-5)
        if abs(determinant - computed_determinant) > tolerance:
            raise ActiveFloorplanError(
                "ray_to_floorplan_alignment determinant does not match matrix_2x3"
            )
        decoded.update(
            {
                "matrix_2x3": matrix,
                "residual_m": residual,
                "determinant": determinant,
            }
        )
    elif any(key in raw for key in ("matrix_2x3", "residual_m", "determinant")):
        raise ActiveFloorplanError(
            "unavailable ray_to_floorplan_alignment must not contain fit values"
        )

    encoded = json.dumps(
        decoded,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    if len(encoded) > _MAX_ALIGNMENT_BYTES:
        raise ActiveFloorplanError("ray_to_floorplan_alignment is too large")
    return decoded


def _sha256_field(payload: Mapping[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ActiveFloorplanError(f"{name} must be a lowercase SHA-256")
    return value


def _calibration_fingerprint(payload: Mapping[str, Any]) -> str:
    return _sha256_field(payload, "calibration_fingerprint")


def _snapshot_timestamp(payload: Mapping[str, Any]) -> int:
    values = []
    for name in ("snapshot_ts", "snapshot_ts_us"):
        if name in payload:
            values.append(_positive_int(payload[name], name))
    if not values:
        raise ActiveFloorplanError("snapshot_ts_us is required")
    if len(set(values)) != 1:
        raise ActiveFloorplanError("snapshot timestamp fields conflict")
    return values[0]


def _snapshot_ref(value: object, snapshot_ts_us: int) -> str:
    if not isinstance(value, str) or value != value.strip() or not value:
        raise ActiveFloorplanError("snapshot_ref must be a portable relative path")
    if len(value.encode("utf-8")) > _MAX_SNAPSHOT_REF_BYTES:
        raise ActiveFloorplanError("snapshot_ref exceeds the bounded contract")
    if value.startswith("/") or "\\" in value or "//" in value:
        raise ActiveFloorplanError("snapshot_ref must be a portable relative path")
    parts = value.split("/")
    if any(_PORTABLE_REF_COMPONENT_RE.fullmatch(part) is None for part in parts):
        raise ActiveFloorplanError("snapshot_ref must be a portable relative path")
    if parts[-1] != f"{snapshot_ts_us}.zarr":
        raise ActiveFloorplanError(
            "snapshot_ref must identify the exact snapshot timestamp"
        )
    return value


def _portable_reference(value: object, name: str) -> str:
    if not isinstance(value, str) or value != value.strip() or not value:
        raise ActiveFloorplanError(f"{name} must be a portable relative path")
    if len(value.encode("utf-8")) > _MAX_SNAPSHOT_REF_BYTES:
        raise ActiveFloorplanError(f"{name} exceeds the bounded contract")
    if value.startswith("/") or "\\" in value or "//" in value:
        raise ActiveFloorplanError(f"{name} must be a portable relative path")
    parts = value.split("/")
    if any(_PORTABLE_REF_COMPONENT_RE.fullmatch(part) is None for part in parts):
        raise ActiveFloorplanError(f"{name} must be a portable relative path")
    return value


class _SameVersionConflict(ActiveFloorplanError):
    """Private signal used to classify conflicting record attempts."""


class ActiveFloorplanRegistry:
    """Keep one monotonic, contract-valid floorplan per configured camera."""

    def __init__(self, aliases: Mapping[str, str]) -> None:
        if len(aliases) > _MAX_ALIASES:
            raise ActiveFloorplanError(
                f"configured aliases must not exceed {_MAX_ALIASES} entries"
            )
        normalized_aliases: dict[str, str] = {}
        cameras: set[str] = set()
        for raw_alias, raw_camera in aliases.items():
            alias = _required_text(raw_alias, "camera alias")
            camera = _required_text(raw_camera, "canonical camera")
            previous = normalized_aliases.get(alias)
            if previous is not None and previous != camera:
                raise ActiveFloorplanError(f"camera alias is ambiguous: {alias}")
            normalized_aliases[alias] = camera
            cameras.add(camera)
            if len(normalized_aliases) > _MAX_ALIASES:
                raise ActiveFloorplanError(
                    f"configured aliases must not exceed {_MAX_ALIASES} entries"
                )
        if not cameras or len(cameras) > _MAX_CAMERAS:
            raise ActiveFloorplanError(
                f"configured cameras must contain between 1 and {_MAX_CAMERAS} entries"
            )

        # A canonical camera name must always resolve to itself.  Validate this
        # after collecting every target so input ordering cannot turn one
        # camera's canonical name into an alias for another camera.
        for camera in cameras:
            alias_target = normalized_aliases.get(camera)
            if alias_target is not None and alias_target != camera:
                raise ActiveFloorplanError(
                    "camera alias collides with canonical camera: "
                    f"{camera} resolves to {alias_target}"
                )

        normalized = dict(normalized_aliases)
        for camera in cameras:
            normalized[camera] = camera
        if len(normalized) > _MAX_ALIASES:
            raise ActiveFloorplanError(
                f"configured aliases must not exceed {_MAX_ALIASES} entries"
            )
        self._aliases = MappingProxyType(normalized)
        self._cameras = frozenset(cameras)
        self._records: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._rejection_count = 0
        self._stale_count = 0
        self._conflict_count = 0
        self._last_error: dict[str, Any] | None = None
        self._reset_count = 0
        self._last_reset: dict[str, Any] | None = None

    def canonical_camera(self, camera_id: object) -> str:
        key = _required_text(camera_id, "camera_id")
        try:
            return self._aliases[key]
        except KeyError as exc:
            raise ActiveFloorplanError(f"camera is not configured: {key}") from exc

    def record(
        self,
        requested_camera: object,
        payload: Mapping[str, Any],
        *,
        snapshot_ref: str | None = None,
    ) -> bool:
        try:
            return self._record_validated(
                requested_camera,
                payload,
                snapshot_ref=snapshot_ref,
            )
        except _SameVersionConflict as exc:
            self._note_rejection(
                "conflict",
                requested_camera,
                str(exc),
            )
            raise
        except ActiveFloorplanError as exc:
            self._note_rejection(
                "invalid",
                requested_camera,
                str(exc),
            )
            raise

    def record_scene_prior(
        self,
        requested_camera: object,
        payload: Mapping[str, Any],
        *,
        calibration_fingerprint: str,
    ) -> bool:
        """Admit an explicitly requested canonical PCF presentation for BEV."""

        try:
            return self._record_scene_prior_validated(
                requested_camera,
                payload,
                calibration_fingerprint=calibration_fingerprint,
            )
        except _SameVersionConflict as exc:
            self._note_rejection("conflict", requested_camera, str(exc))
            raise
        except ActiveFloorplanError as exc:
            self._note_rejection("invalid", requested_camera, str(exc))
            raise

    def _record_scene_prior_validated(
        self,
        requested_camera: object,
        payload: Mapping[str, Any],
        *,
        calibration_fingerprint: str,
    ) -> bool:
        if not isinstance(payload, Mapping):
            raise ActiveFloorplanError("scene-prior floorplan payload must be an object")
        if payload.get("error"):
            return False
        canonical = self.canonical_camera(requested_camera)
        payload_camera = self.canonical_camera(payload.get("camera_id") or canonical)
        if payload_camera != canonical:
            raise ActiveFloorplanError("floorplan camera does not match the request")
        if payload.get("scene_prior_only") is not True or payload.get("display_source") != "pcf":
            raise ActiveFloorplanError("scene-prior BEV authority must be the explicit PCF presentation")
        if payload.get("served_from_cache") is not True:
            raise ActiveFloorplanError("scene-prior BEV authority must be immutable cached evidence")
        frame = _required_text(payload.get("frame"), "frame")
        units = _required_text(payload.get("units"), "units")
        if frame != _FRAME or units != _UNITS:
            raise ActiveFloorplanError(
                f"scene-prior BEV authority must use frame={_FRAME} and units={_UNITS}"
            )
        meta = payload.get("scene_prior_meta")
        if not isinstance(meta, Mapping):
            raise ActiveFloorplanError("scene_prior_meta is required")
        if (
            meta.get("contract") != "noesis.scene_prior.floorplan_composite"
            or meta.get("status") != "pcf"
            or meta.get("display_source") != "pcf"
        ):
            raise ActiveFloorplanError("scene-prior metadata does not identify canonical PCF")
        prior_id = _required_text(meta.get("prior_id"), "scene_prior_meta.prior_id")
        manifest_path = _portable_reference(
            meta.get("revision_manifest_path"),
            "scene_prior_meta.revision_manifest_path",
        )
        manifest_sha256 = _sha256_field(
            meta,
            "revision_manifest_sha256",
        )
        calibration_sha256 = _sha256_field(
            {"calibration_fingerprint": calibration_fingerprint},
            "calibration_fingerprint",
        )
        diagnostic_meta = payload.get("scene_prior_diagnostic_meta")
        if not isinstance(diagnostic_meta, Mapping):
            raise ActiveFloorplanError("scene_prior_diagnostic_meta is required")
        raw_shape = diagnostic_meta.get("grid_shape")
        if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != 2:
            raise ActiveFloorplanError("scene-prior diagnostic grid shape is invalid")
        rows = _positive_int(raw_shape[0], "scene-prior grid rows")
        columns = _positive_int(raw_shape[1], "scene-prior grid columns")
        if (
            rows > _MAX_GRID_DIMENSION
            or columns > _MAX_GRID_DIMENSION
            or rows * columns > _MAX_GRID_CELLS
        ):
            raise ActiveFloorplanError("scene-prior grid shape exceeds the bounded contract")
        grid_res_m = _finite_bounded(
            payload.get("scale_m_per_px"),
            "scale_m_per_px",
            max_abs=_MAX_GRID_RES_M,
            positive=True,
        )
        bounds = _bounds(payload)
        snapshot_ts_us = _positive_int(payload.get("ts"), "scene-prior timestamp")
        snapshot_ref = f"scene_priors/{manifest_path}"
        max_extent_m = max(
            bounds["max_x"] - bounds["min_x"],
            bounds["max_z"] - bounds["min_z"],
        )
        snapshot_identity = {
            "camera_id": canonical,
            "snapshot_ts_us": snapshot_ts_us,
            "snapshot_ref": snapshot_ref,
            "snapshot_id": prior_id,
            "snapshot_content_sha256": manifest_sha256,
            "calibration_fingerprint": calibration_sha256,
        }
        record: dict[str, Any] = {
            "camera_id": canonical,
            "snapshot_ts_us": snapshot_ts_us,
            "floorplan_ts_us": snapshot_ts_us,
            "served_from_cache": True,
            "grid_res_m": grid_res_m,
            "max_extent_m": max_extent_m,
            "grid_shape": [rows, columns],
            "grid_shape_source": "scene_prior_diagnostic_meta",
            "bounds": bounds,
            "frame": frame,
            "orientation": _ORIENTATION,
            "floorplan_contract_version": _FLOORPLAN_CONTRACT_VERSION,
            "units": units,
            "calibration_fingerprint": calibration_sha256,
            "snapshot_ref": snapshot_ref,
            "snapshot_id": prior_id,
            "snapshot_content_sha256": manifest_sha256,
            "snapshot_identity": snapshot_identity,
            "source": "active_floorplan",
            "authority_kind": "scene_prior_pcf",
        }
        return self._store_record(canonical, record)

    def _record_validated(
        self,
        requested_camera: object,
        payload: Mapping[str, Any],
        *,
        snapshot_ref: str | None,
    ) -> bool:
        if not isinstance(payload, Mapping):
            raise ActiveFloorplanError("floorplan payload must be an object")
        error_value = payload.get("error")
        if error_value:
            self._note_rejection(
                "payload_error",
                requested_camera,
                f"floorplan producer error: {error_value}",
            )
            return False
        canonical = self.canonical_camera(requested_camera)
        payload_camera = self.canonical_camera(payload.get("camera_id") or canonical)
        if payload_camera != canonical:
            raise ActiveFloorplanError("floorplan camera does not match the request")
        frame = _required_text(payload.get("frame"), "frame")
        orientation = _required_text(
            payload.get("orientation"),
            "orientation",
        )
        contract_version = _positive_int(
            payload.get("floorplan_contract_version"),
            "floorplan_contract_version",
        )
        units = _required_text(payload.get("units"), "units")
        if (
            frame != _FRAME
            or orientation != _ORIENTATION
            or contract_version != _FLOORPLAN_CONTRACT_VERSION
            or units != _UNITS
        ):
            raise ActiveFloorplanError(
                "floorplan must use "
                f"frame={_FRAME}, orientation={_ORIENTATION}, "
                f"contract_version={_FLOORPLAN_CONTRACT_VERSION}, "
                f"and units={_UNITS}"
            )
        snapshot_ts_us = _snapshot_timestamp(payload)
        floorplan_ts_us = _positive_int(payload.get("ts"), "floorplan_ts_us")
        served_from_cache = payload.get("served_from_cache")
        if not isinstance(served_from_cache, bool):
            raise ActiveFloorplanError("served_from_cache must be boolean")
        calibration_fingerprint = _calibration_fingerprint(payload)
        snapshot_id = _required_text(payload.get("snapshot_id"), "snapshot_id")
        snapshot_content_sha256 = _sha256_field(
            payload,
            "snapshot_content_sha256",
        )
        portable_snapshot_ref = _snapshot_ref(snapshot_ref, snapshot_ts_us)
        grid_shape, grid_shape_source = _grid_shape(payload)
        alignment = _alignment(payload)
        snapshot_identity = {
            "camera_id": canonical,
            "snapshot_ts_us": snapshot_ts_us,
            "snapshot_ref": portable_snapshot_ref,
            "snapshot_id": snapshot_id,
            "snapshot_content_sha256": snapshot_content_sha256,
            "calibration_fingerprint": calibration_fingerprint,
        }
        record: dict[str, Any] = {
            "camera_id": canonical,
            "snapshot_ts_us": snapshot_ts_us,
            "floorplan_ts_us": floorplan_ts_us,
            "served_from_cache": served_from_cache,
            "grid_res_m": _finite_bounded(
                payload.get("grid_res_m"),
                "grid_res_m",
                max_abs=_MAX_GRID_RES_M,
                positive=True,
            ),
            "max_extent_m": _finite_bounded(
                payload.get("max_extent_m"),
                "max_extent_m",
                max_abs=_MAX_EXTENT_M,
                positive=True,
            ),
            "grid_shape": grid_shape,
            "grid_shape_source": grid_shape_source,
            "bounds": _bounds(payload),
            "frame": frame,
            "orientation": orientation,
            "floorplan_contract_version": contract_version,
            "units": units,
            "calibration_fingerprint": calibration_fingerprint,
            "snapshot_ref": portable_snapshot_ref,
            "snapshot_id": snapshot_id,
            "snapshot_content_sha256": snapshot_content_sha256,
            "snapshot_identity": snapshot_identity,
            "source": "active_floorplan",
        }
        if alignment is not None:
            record["ray_to_floorplan_alignment"] = alignment
        return self._store_record(canonical, record)

    def _store_record(self, canonical: str, record: dict[str, Any]) -> bool:
        snapshot_ts_us = int(record["snapshot_ts_us"])
        floorplan_ts_us = int(record["floorplan_ts_us"])
        with self._lock:
            previous = self._records.get(canonical)
            if previous is not None:
                version = (snapshot_ts_us, floorplan_ts_us)
                previous_version = (
                    int(previous["snapshot_ts_us"]),
                    int(previous["floorplan_ts_us"]),
                )
                if version < previous_version:
                    self._note_rejection_locked(
                        "stale",
                        canonical,
                        "floorplan version is older than the active snapshot",
                    )
                    return False
                if version == previous_version:
                    # Cache delivery is request-local metadata, not part of the
                    # immutable floorplan version.  Every other canonical field,
                    # including the durable snapshot reference, must agree.
                    comparable = dict(record)
                    comparable_previous = dict(previous)
                    comparable.pop("served_from_cache", None)
                    comparable_previous.pop("served_from_cache", None)
                    if comparable != comparable_previous:
                        raise _SameVersionConflict(
                            "same floorplan version has conflicting payload"
                        )
                    return True
            self._records[canonical] = record
        return True

    def bounds_for(self, camera_id: object) -> dict[str, Any] | None:
        canonical = self.canonical_camera(camera_id)
        with self._lock:
            record = self._records.get(canonical)
            return deepcopy(record) if record is not None else None

    def clear(self, camera_id: object | None = None) -> int:
        """Clear one canonical camera, or every camera when ``camera_id`` is None."""
        canonical = None if camera_id is None else self.canonical_camera(camera_id)
        with self._lock:
            if canonical is None:
                removed_count = len(self._records)
                self._records.clear()
                reset: dict[str, Any] = {
                    "scope": "all",
                    "removed_count": removed_count,
                }
            else:
                removed_count = int(self._records.pop(canonical, None) is not None)
                reset = {
                    "scope": "camera",
                    "camera_id": canonical,
                    "removed_count": removed_count,
                }
            self._reset_count += 1
            reset["sequence"] = self._reset_count
            self._last_reset = reset
            return removed_count

    @staticmethod
    def _diagnostic_text(value: object, *, limit: int = _MAX_ERROR_TEXT_BYTES) -> str:
        try:
            text = str(value or "").strip()
        except Exception:
            text = "unprintable"
        encoded = text.encode("utf-8", errors="replace")
        if len(encoded) <= limit:
            return text
        return encoded[:limit].decode("utf-8", errors="ignore")

    def _note_rejection(
        self,
        kind: str,
        camera_id: object,
        message: object,
    ) -> None:
        with self._lock:
            self._note_rejection_locked(kind, camera_id, message)

    def _note_rejection_locked(
        self,
        kind: str,
        camera_id: object,
        message: object,
    ) -> None:
        self._rejection_count += 1
        if kind == "stale":
            self._stale_count += 1
        elif kind == "conflict":
            self._conflict_count += 1
        self._last_error = {
            "kind": str(kind),
            "camera_id": self._diagnostic_text(
                camera_id,
                limit=_MAX_IDENTIFIER_BYTES,
            )
            or None,
            "message": self._diagnostic_text(message),
            "rejection_sequence": self._rejection_count,
        }

    def health_snapshot(self) -> dict[str, Any]:
        with self._lock:
            records = {
                camera: {
                    "snapshot_ts_us": int(record["snapshot_ts_us"]),
                    "floorplan_ts_us": int(record["floorplan_ts_us"]),
                    "snapshot_ref": record["snapshot_ref"],
                    "snapshot_id": record["snapshot_id"],
                    "snapshot_content_sha256": record[
                        "snapshot_content_sha256"
                    ],
                    "calibration_fingerprint": record["calibration_fingerprint"],
                    "frame": record["frame"],
                    "units": record["units"],
                    "authority_kind": record.get("authority_kind", "depth_snapshot"),
                }
                for camera, record in sorted(self._records.items())
            }
            missing_cameras = sorted(self._cameras.difference(records))
            return {
                "contract": "noesis.active_floorplan.health",
                "contract_version": 1,
                "healthy": not missing_cameras,
                "configured_camera_count": len(self._cameras),
                "active_camera_count": len(records),
                "missing_cameras": missing_cameras,
                "rejection_count": self._rejection_count,
                "stale_count": self._stale_count,
                "conflict_count": self._conflict_count,
                "last_error": deepcopy(self._last_error),
                "reset_count": self._reset_count,
                "last_reset": deepcopy(self._last_reset),
                "cameras": records,
            }


__all__ = [
    "ActiveFloorplanError",
    "ActiveFloorplanRegistry",
]
