from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
from noesis_core.strict_json import strict_json_loads

CONTRACT_VERSION = 1
HARDENED_CONTRACT_VERSION = 2
SUPPORTED_CONTRACT_VERSIONS = (CONTRACT_VERSION, HARDENED_CONTRACT_VERSION)
MODEL_CONTENT_BINDING_CONTRACT = "noesis.depth_registration.model_content.v1"
OCCUPIED_ANCHOR_VALIDATION_CONTRACT = (
    "noesis.depth_registration.occupied_person_holdout.v2"
)
RUNTIME_OCCUPIED_ANCHOR_EVIDENCE_CONTRACT = (
    "noesis.depth_registration.runtime_occupied_anchor_evidence.v4"
)
MAPANYTHING_FRAME_SCALE_STATISTIC_CONTRACT = (
    "noesis.depth_registration.mapanything_frame_scale.v1"
)
MAPANYTHING_FRAME_SCALE_NORMALIZATION_CONTRACT = (
    "noesis.depth_registration.mapanything_frame_scale_normalization.v1"
)
MAPANYTHING_FRAME_SCALE_STATISTIC = {
    "contract": MAPANYTHING_FRAME_SCALE_STATISTIC_CONTRACT,
    "statistic": "median_valid_nonperson_depth_m",
    "person_exclusion": "expanded_person_bbox_union",
    "bbox_margin_ratio": 0.15,
    "bbox_margin_min_px": 16,
    "min_support_count": 4096,
}
MIN_MAPANYTHING_FRAME_SCALE_SUPPORT_COUNT = 4096
MOVING_TRACK_ELIGIBILITY_CONTRACT = (
    "noesis.depth_registration.moving_track_eligibility.v1"
)
MIN_OCCUPIED_FIT_OBSERVATIONS = 32
MIN_OCCUPIED_HOLDOUT_OBSERVATIONS = 32
MIN_OCCUPIED_CAPTURES_PER_PARTITION = 2
MIN_OCCUPIED_ANCHOR_SUPPORT = 8
MAX_OCCUPIED_ANCHOR_TIME_DELTA_MS = 250.0
MAX_OCCUPIED_MEDIAN_ABS_ERROR_M = 0.75
MAX_OCCUPIED_P95_ABS_ERROR_M = 1.50
MAX_OCCUPIED_ABS_ERROR_M = 2.50
TRANSFORM_TYPE_PIECEWISE = "piecewise_linear_1d"
SOURCE_SPACE = "dav2_anchor_range_m_raw"
TARGET_SPACE = "mapanything_room_range_m"
REGISTRATION_SCOPE = "people_tracking_depth_registration"
CALIBRATION_FINGERPRINT_BASIS = "image_space_intrinsics_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_DEPTH_REGISTRATION_BYTES = 32 * 1024 * 1024


class DepthRegistrationError(Exception):
    """Raised when a depth-registration artifact is invalid or mismatched."""


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _require_sha256(value: Any, *, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if _SHA256_RE.fullmatch(normalized) is None:
        raise DepthRegistrationError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _is_exact_json_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def validate_mapanything_frame_scale_statistic(payload: Any) -> None:
    """Validate the exact typed evidence-v4 frame-scale policy."""

    expected_keys = {
        "contract",
        "statistic",
        "person_exclusion",
        "bbox_margin_ratio",
        "bbox_margin_min_px",
        "min_support_count",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        raise DepthRegistrationError(
            "MapAnything frame-scale statistic fields do not match the exact contract"
        )
    if (
        payload.get("contract") != MAPANYTHING_FRAME_SCALE_STATISTIC_CONTRACT
        or payload.get("statistic") != "median_valid_nonperson_depth_m"
        or payload.get("person_exclusion") != "expanded_person_bbox_union"
    ):
        raise DepthRegistrationError(
            "MapAnything frame-scale statistic identity is invalid"
        )
    margin_ratio = payload.get("bbox_margin_ratio")
    if (
        not _is_exact_json_number(margin_ratio)
        or not math.isfinite(float(margin_ratio))
        or float(margin_ratio) != 0.15
    ):
        raise DepthRegistrationError(
            "MapAnything frame-scale bbox_margin_ratio is invalid"
        )
    if (
        isinstance(payload.get("bbox_margin_min_px"), bool)
        or not isinstance(payload.get("bbox_margin_min_px"), int)
        or payload.get("bbox_margin_min_px") != 16
        or isinstance(payload.get("min_support_count"), bool)
        or not isinstance(payload.get("min_support_count"), int)
        or payload.get("min_support_count") != MIN_MAPANYTHING_FRAME_SCALE_SUPPORT_COUNT
    ):
        raise DepthRegistrationError(
            "MapAnything frame-scale integer policy is invalid"
        )


def _portable_logical_path(value: str, *, name: str) -> str:
    logical = str(value or "").strip()
    if not logical or Path(logical).is_absolute() or ".." in Path(logical).parts:
        raise DepthRegistrationError(f"{name} must be portable and relative")
    return logical


def _hash_content_file(
    path: Path,
    *,
    logical_path: str,
    stream_digest: Any | None = None,
) -> dict[str, Any]:
    logical = _portable_logical_path(
        logical_path,
        name="content fingerprint logical_path",
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(Path(path), flags)
    except OSError as exc:
        raise DepthRegistrationError(
            f"unable to open content fingerprint input {path}: {exc}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or int(before.st_size) <= 0:
            raise DepthRegistrationError(
                f"content fingerprint input must be a non-empty regular file: {path}"
            )
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            if stream_digest is not None:
                stream_digest.update(block)
        after = os.fstat(descriptor)
        identity_before = (
            int(before.st_dev),
            int(before.st_ino),
            int(before.st_size),
            int(before.st_mtime_ns),
            int(before.st_ctime_ns),
        )
        identity_after = (
            int(after.st_dev),
            int(after.st_ino),
            int(after.st_size),
            int(after.st_mtime_ns),
            int(after.st_ctime_ns),
        )
        if identity_after != identity_before:
            raise DepthRegistrationError(
                f"content fingerprint input changed while hashing: {path}"
            )
        return {
            "logical_path": logical,
            "size_bytes": int(before.st_size),
            "sha256": digest.hexdigest(),
        }
    finally:
        os.close(descriptor)


def content_file_fingerprint(
    path: Path,
    *,
    logical_path: str,
) -> dict[str, Any]:
    """Hash one stable regular file without following its final path component."""

    return _hash_content_file(path, logical_path=logical_path)


def content_bundle_fingerprint(
    members: Sequence[tuple[str, Path, str]],
) -> dict[str, Any]:
    """Hash an ordered file bundle using the DS9 engine-source bundle contract."""

    if not members:
        raise DepthRegistrationError("content bundle must contain at least one file")
    labels: set[str] = set()
    digest = hashlib.sha256()
    files: list[dict[str, Any]] = []
    for label_raw, path, logical_path in members:
        label = str(label_raw or "").strip()
        if not label or "\0" in label or label in labels:
            raise DepthRegistrationError(
                "content bundle labels must be non-empty and unique"
            )
        labels.add(label)
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        record = _hash_content_file(
            path,
            logical_path=logical_path,
            stream_digest=digest,
        )
        digest.update(b"\0")
        files.append({"label": label, **record})
    return {
        "bundle_sha256": digest.hexdigest(),
        "files": files,
    }


def _normalize_path_like(value: Any, *, repo_root: Path) -> Any:
    if not isinstance(value, str):
        return value
    raw = value.strip()
    if not raw:
        return raw
    candidate = Path(raw)
    if candidate.is_absolute():
        # Preserve repo-relative intent before resolving symlinks such as models -> second-drive storage.
        try:
            return candidate.absolute().relative_to(repo_root.absolute()).as_posix()
        except Exception:
            pass
        try:
            return candidate.resolve().relative_to(repo_root.resolve()).as_posix()
        except Exception:
            return candidate.name
    return candidate.as_posix()


def _float_tuple(values: Sequence[Any], *, name: str) -> tuple[float, ...]:
    out: list[float] = []
    for value in values:
        try:
            parsed = float(value)
        except Exception as exc:
            raise DepthRegistrationError(
                f"{name} contains a non-numeric value: {value!r}"
            ) from exc
        if not math.isfinite(parsed):
            raise DepthRegistrationError(f"{name} contains a non-finite value")
        out.append(parsed)
    return tuple(out)


def _ensure_monotonic_non_decreasing(values: Sequence[float], *, name: str) -> None:
    for index in range(1, len(values)):
        if float(values[index]) < float(values[index - 1]):
            raise DepthRegistrationError(f"{name} must be monotonically non-decreasing")


def _ensure_strictly_increasing(values: Sequence[float], *, name: str) -> None:
    for index in range(1, len(values)):
        if float(values[index]) <= float(values[index - 1]):
            raise DepthRegistrationError(f"{name} must be strictly increasing")


def calibration_fingerprint_from_snapshot(snapshot: Any) -> dict[str, Any]:
    intrinsics = np.asarray(getattr(snapshot, "intrinsics"), dtype=np.float64)
    if intrinsics.shape != (3, 3):
        raise DepthRegistrationError("Calibration snapshot intrinsics must be 3x3")
    image_size = getattr(snapshot, "image_size")
    payload: dict[str, Any] = {
        "basis": CALIBRATION_FINGERPRINT_BASIS,
        "camera_id": str(getattr(snapshot, "camera_id", "") or ""),
        "intrinsics": [[float(intrinsics[r, c]) for c in range(3)] for r in range(3)],
        "image_size": [int(image_size[0]), int(image_size[1])],
    }
    payload["fingerprint_sha256"] = _sha256(payload)
    return payload


def calibration_fingerprint(snapshot: Any) -> dict[str, Any]:
    return calibration_fingerprint_from_snapshot(snapshot)


def _normalize_calibration_fingerprint_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    intrinsics = np.asarray(payload.get("intrinsics"), dtype=np.float64)
    if intrinsics.shape != (3, 3):
        raise DepthRegistrationError(
            "Depth registration calibration fingerprint intrinsics must be 3x3"
        )
    image_size = payload.get("image_size")
    if not isinstance(image_size, Sequence) or len(image_size) != 2:
        raise DepthRegistrationError(
            "Depth registration calibration fingerprint image_size must contain two values"
        )
    normalized: dict[str, Any] = {
        "basis": CALIBRATION_FINGERPRINT_BASIS,
        "camera_id": str(payload.get("camera_id") or ""),
        "intrinsics": [[float(intrinsics[r, c]) for c in range(3)] for r in range(3)],
        "image_size": [int(image_size[0]), int(image_size[1])],
    }
    normalized["fingerprint_sha256"] = _sha256(normalized)
    return normalized


def model_profile_fingerprint(
    model_cfg: Mapping[str, Any],
    *,
    repo_root: Path,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key in sorted(model_cfg.keys()):
        value = model_cfg[key]
        if isinstance(value, Mapping):
            normalized[str(key)] = {
                str(nested_key): _normalize_path_like(nested_value, repo_root=repo_root)
                for nested_key, nested_value in sorted(
                    value.items(), key=lambda item: str(item[0])
                )
            }
        elif isinstance(value, (list, tuple)):
            normalized[str(key)] = [
                _normalize_path_like(v, repo_root=repo_root) for v in value
            ]
        else:
            normalized[str(key)] = _normalize_path_like(value, repo_root=repo_root)
    if extra:
        for key in sorted(extra.keys()):
            normalized[str(key)] = extra[key]
    normalized["fingerprint_sha256"] = _sha256(normalized)
    return normalized


def profile_fingerprint(
    model_cfg: Mapping[str, Any],
    *,
    repo_root: Path,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return model_profile_fingerprint(model_cfg, repo_root=repo_root, extra=extra)


def _profile_payload_without_fingerprint(profile: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(k): v for k, v in dict(profile).items() if str(k) != "fingerprint_sha256"
    }


def _validate_profile_fingerprint(profile: Mapping[str, Any], *, name: str) -> None:
    expected = _require_sha256(
        profile.get("fingerprint_sha256"), name=f"{name}.fingerprint_sha256"
    )
    actual = _sha256(_profile_payload_without_fingerprint(profile))
    if actual != expected:
        raise DepthRegistrationError(
            f"{name}.fingerprint_sha256 does not match its canonical payload"
        )


def _validate_content_file_record(record: Any, *, name: str) -> None:
    if not isinstance(record, Mapping):
        raise DepthRegistrationError(f"{name} must be a mapping")
    if set(record) != {"logical_path", "size_bytes", "sha256"}:
        raise DepthRegistrationError(f"{name} fields do not match the exact contract")
    logical_path = str(record.get("logical_path") or "").strip()
    if (
        not logical_path
        or Path(logical_path).is_absolute()
        or ".." in Path(logical_path).parts
    ):
        raise DepthRegistrationError(
            f"{name}.logical_path must be a portable relative path"
        )
    try:
        size_bytes = int(record.get("size_bytes"))
    except Exception as exc:
        raise DepthRegistrationError(
            f"{name}.size_bytes must be a positive integer"
        ) from exc
    if size_bytes <= 0:
        raise DepthRegistrationError(f"{name}.size_bytes must be a positive integer")
    _require_sha256(record.get("sha256"), name=f"{name}.sha256")


def _validate_model_content_binding(profile: Mapping[str, Any], *, name: str) -> None:
    binding = profile.get("content_binding")
    if not isinstance(binding, Mapping):
        raise DepthRegistrationError(f"{name}.content_binding must be a mapping")
    if set(binding) != {
        "contract",
        "engine",
        "reviewed_config",
        "runtime_config",
        "onnx_authority",
    }:
        raise DepthRegistrationError(
            f"{name}.content_binding fields do not match the exact contract"
        )
    if str(binding.get("contract") or "") != MODEL_CONTENT_BINDING_CONTRACT:
        raise DepthRegistrationError(f"{name}.content_binding.contract is unsupported")
    _validate_content_file_record(
        binding.get("engine"), name=f"{name}.content_binding.engine"
    )
    _validate_content_file_record(
        binding.get("reviewed_config"),
        name=f"{name}.content_binding.reviewed_config",
    )
    _validate_content_file_record(
        binding.get("runtime_config"),
        name=f"{name}.content_binding.runtime_config",
    )
    onnx = binding.get("onnx_authority")
    if not isinstance(onnx, Mapping):
        raise DepthRegistrationError(
            f"{name}.content_binding.onnx_authority must be a mapping"
        )
    if set(onnx) != {
        "logical_path",
        "raw_sha256",
        "bundle_sha256",
        "bundle_files",
        "source_contract_selector",
        "source_contract_sha256",
    }:
        raise DepthRegistrationError(
            f"{name}.content_binding.onnx_authority fields do not match the exact contract"
        )
    logical_path = str(onnx.get("logical_path") or "").strip()
    if (
        not logical_path
        or Path(logical_path).is_absolute()
        or ".." in Path(logical_path).parts
    ):
        raise DepthRegistrationError(
            f"{name}.content_binding.onnx_authority.logical_path must be a portable relative path"
        )
    selector = str(onnx.get("source_contract_selector") or "").strip()
    if not selector:
        raise DepthRegistrationError(
            f"{name}.content_binding.onnx_authority.source_contract_selector is required"
        )
    raw_sha256 = _require_sha256(
        onnx.get("raw_sha256"),
        name=f"{name}.content_binding.onnx_authority.raw_sha256",
    )
    _require_sha256(
        onnx.get("bundle_sha256"),
        name=f"{name}.content_binding.onnx_authority.bundle_sha256",
    )
    _require_sha256(
        onnx.get("source_contract_sha256"),
        name=f"{name}.content_binding.onnx_authority.source_contract_sha256",
    )
    bundle_files = onnx.get("bundle_files")
    if not isinstance(bundle_files, list) or not bundle_files:
        raise DepthRegistrationError(
            f"{name}.content_binding.onnx_authority.bundle_files must be non-empty"
        )
    labels: set[str] = set()
    for index, file_record in enumerate(bundle_files):
        if not isinstance(file_record, Mapping) or set(file_record) != {
            "label",
            "logical_path",
            "size_bytes",
            "sha256",
        }:
            raise DepthRegistrationError(
                f"{name}.content_binding.onnx_authority.bundle_files[{index}] is invalid"
            )
        label = str(file_record.get("label") or "").strip()
        if not label or label in labels:
            raise DepthRegistrationError(
                f"{name}.content_binding.onnx_authority.bundle_files labels must be unique"
            )
        labels.add(label)
        _validate_content_file_record(
            {key: value for key, value in file_record.items() if key != "label"},
            name=f"{name}.content_binding.onnx_authority.bundle_files[{index}]",
        )
    main = bundle_files[0]
    if (
        str(main.get("label") or "") != "main"
        or str(main.get("logical_path") or "") != logical_path
        or str(main.get("sha256") or "") != raw_sha256
    ):
        raise DepthRegistrationError(
            f"{name}.content_binding.onnx_authority main file does not match its authority"
        )
    _validate_profile_fingerprint(profile, name=name)


def _validate_occupied_anchor_validation(payload: Any) -> None:
    if not isinstance(payload, Mapping):
        raise DepthRegistrationError("occupied_anchor_validation must be a mapping")
    expected_keys = {
        "contract",
        "evidence_sha256",
        "fit_observations",
        "holdout_observations",
        "fit_capture_count",
        "holdout_capture_count",
        "fit_raw_range_m",
        "holdout_raw_range_m",
        "holdout_target_range_m",
        "domain_coverage_fraction",
        "median_abs_error_m",
        "p95_abs_error_m",
        "max_abs_error_m",
        "thresholds",
        "min_anchor_support",
        "max_timestamp_delta_ms",
        "moving_track_eligibility",
        "reference_scale_normalization",
    }
    if set(payload) != expected_keys:
        raise DepthRegistrationError(
            "occupied_anchor_validation fields do not match the exact contract"
        )
    if str(payload.get("contract") or "") != OCCUPIED_ANCHOR_VALIDATION_CONTRACT:
        raise DepthRegistrationError(
            "occupied_anchor_validation.contract is unsupported"
        )
    _require_sha256(
        payload.get("evidence_sha256"),
        name="occupied_anchor_validation.evidence_sha256",
    )
    integer_minima = {
        "fit_observations": MIN_OCCUPIED_FIT_OBSERVATIONS,
        "holdout_observations": MIN_OCCUPIED_HOLDOUT_OBSERVATIONS,
        "fit_capture_count": MIN_OCCUPIED_CAPTURES_PER_PARTITION,
        "holdout_capture_count": MIN_OCCUPIED_CAPTURES_PER_PARTITION,
        "min_anchor_support": MIN_OCCUPIED_ANCHOR_SUPPORT,
    }
    for key, minimum in integer_minima.items():
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise DepthRegistrationError(
                f"occupied_anchor_validation.{key} must be an integer >= {minimum}"
            )
    if int(payload["fit_capture_count"]) > int(payload["fit_observations"]):
        raise DepthRegistrationError(
            "occupied_anchor_validation fit_capture_count exceeds fit_observations"
        )
    if int(payload["holdout_capture_count"]) > int(payload["holdout_observations"]):
        raise DepthRegistrationError(
            "occupied_anchor_validation holdout_capture_count exceeds holdout_observations"
        )

    ranges: dict[str, tuple[float, float]] = {}
    for key in ("fit_raw_range_m", "holdout_raw_range_m", "holdout_target_range_m"):
        values = payload.get(key)
        if not isinstance(values, list) or len(values) != 2:
            raise DepthRegistrationError(
                f"occupied_anchor_validation.{key} must contain exactly two values"
            )
        try:
            parsed = (float(values[0]), float(values[1]))
        except Exception as exc:
            raise DepthRegistrationError(
                f"occupied_anchor_validation.{key} must contain numeric values"
            ) from exc
        if (
            not all(math.isfinite(value) and value > 0.0 for value in parsed)
            or parsed[0] > parsed[1]
        ):
            raise DepthRegistrationError(
                f"occupied_anchor_validation.{key} must be finite, positive, and ordered"
            )
        ranges[key] = parsed
    fit_range = ranges["fit_raw_range_m"]
    holdout_range = ranges["holdout_raw_range_m"]
    if holdout_range[0] < fit_range[0] or holdout_range[1] > fit_range[1]:
        raise DepthRegistrationError(
            "occupied_anchor_validation holdout raw range requires extrapolation"
        )
    for key in (
        "domain_coverage_fraction",
        "median_abs_error_m",
        "p95_abs_error_m",
        "max_abs_error_m",
    ):
        try:
            value = float(payload.get(key))
        except Exception as exc:
            raise DepthRegistrationError(
                f"occupied_anchor_validation.{key} must be finite"
            ) from exc
        if not math.isfinite(value) or value < 0.0:
            raise DepthRegistrationError(
                f"occupied_anchor_validation.{key} must be finite and non-negative"
            )
    if abs(float(payload.get("domain_coverage_fraction")) - 1.0) > 1e-12:
        raise DepthRegistrationError(
            "occupied_anchor_validation.domain_coverage_fraction must be exactly 1.0"
        )
    thresholds = payload.get("thresholds")
    policy = {
        "max_median_abs_error_m": MAX_OCCUPIED_MEDIAN_ABS_ERROR_M,
        "max_p95_abs_error_m": MAX_OCCUPIED_P95_ABS_ERROR_M,
        "max_abs_error_m": MAX_OCCUPIED_ABS_ERROR_M,
    }
    if not isinstance(thresholds, Mapping) or set(thresholds) != set(policy):
        raise DepthRegistrationError(
            "occupied_anchor_validation.thresholds fields do not match the exact contract"
        )
    for metric_key, threshold_key in (
        ("median_abs_error_m", "max_median_abs_error_m"),
        ("p95_abs_error_m", "max_p95_abs_error_m"),
        ("max_abs_error_m", "max_abs_error_m"),
    ):
        try:
            observed = float(payload.get(metric_key))
            limit = float(thresholds.get(threshold_key))
        except Exception as exc:
            raise DepthRegistrationError(
                f"occupied_anchor_validation threshold {threshold_key} must be finite"
            ) from exc
        if (
            not math.isfinite(limit)
            or limit <= 0.0
            or limit > policy[threshold_key]
            or observed > limit
        ):
            raise DepthRegistrationError(
                f"occupied_anchor_validation {metric_key} exceeds {threshold_key}"
            )
    try:
        max_timestamp_delta_ms = float(payload.get("max_timestamp_delta_ms"))
    except Exception as exc:
        raise DepthRegistrationError(
            "occupied_anchor_validation.max_timestamp_delta_ms must be numeric"
        ) from exc
    if (
        not math.isfinite(max_timestamp_delta_ms)
        or max_timestamp_delta_ms < 0.0
        or max_timestamp_delta_ms > MAX_OCCUPIED_ANCHOR_TIME_DELTA_MS
    ):
        raise DepthRegistrationError(
            "occupied_anchor_validation.max_timestamp_delta_ms exceeds policy"
        )
    moving = payload.get("moving_track_eligibility")
    _validate_moving_track_eligibility(moving)
    normalization = payload.get("reference_scale_normalization")
    _validate_reference_scale_normalization(normalization)
    if int(moving["eligible_observation_count"]) != int(
        payload["fit_observations"]
    ) + int(payload["holdout_observations"]):
        raise DepthRegistrationError(
            "moving-track eligible observation count disagrees with occupied validation"
        )
    if int(normalization["normalized_observation_count"]) != int(
        moving["eligible_observation_count"]
    ):
        raise DepthRegistrationError(
            "reference normalization observation count disagrees with moving-track eligibility"
        )
    if tuple(
        float(value) for value in normalization["holdout_normalized_reference_range_m"]
    ) != tuple(float(value) for value in payload["holdout_target_range_m"]):
        raise DepthRegistrationError(
            "normalized holdout reference range disagrees with occupied validation"
        )


def _validate_exact_positive_range(payload: Any, *, name: str) -> tuple[float, float]:
    if not isinstance(payload, list) or len(payload) != 2:
        raise DepthRegistrationError(f"{name} must contain exactly two values")
    if not all(_is_exact_json_number(value) for value in payload):
        raise DepthRegistrationError(f"{name} must contain JSON numbers")
    try:
        parsed = (float(payload[0]), float(payload[1]))
    except Exception as exc:
        raise DepthRegistrationError(f"{name} must contain numeric values") from exc
    if (
        not all(math.isfinite(value) and value > 0.0 for value in parsed)
        or parsed[0] > parsed[1]
    ):
        raise DepthRegistrationError(f"{name} must be finite, positive, and ordered")
    return parsed


def _validate_reference_scale_normalization(payload: Any) -> None:
    expected_keys = {
        "contract",
        "source_statistic",
        "algorithm",
        "baseline_partition",
        "baseline_estimator",
        "fit_baseline_m",
        "fit_distinct_frame_count",
        "holdout_distinct_frame_count",
        "fit_frame_scale_range_m",
        "holdout_frame_scale_range_m",
        "fit_normalization_factor_range",
        "holdout_normalization_factor_range",
        "fit_raw_reference_range_m",
        "holdout_raw_reference_range_m",
        "fit_normalized_reference_range_m",
        "holdout_normalized_reference_range_m",
        "normalized_observation_count",
        "normalized_observations_sha256",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        raise DepthRegistrationError(
            "reference_scale_normalization fields do not match the exact contract"
        )
    if (
        payload.get("contract") != MAPANYTHING_FRAME_SCALE_NORMALIZATION_CONTRACT
        or payload.get("algorithm")
        != "raw_reference_times_fit_median_over_frame_median"
        or payload.get("baseline_partition") != "fit"
        or payload.get("baseline_estimator") != "median_distinct_source_frames"
    ):
        raise DepthRegistrationError(
            "reference_scale_normalization identity is invalid"
        )
    validate_mapanything_frame_scale_statistic(payload.get("source_statistic"))
    if not _is_exact_json_number(payload.get("fit_baseline_m")):
        raise DepthRegistrationError(
            "reference_scale_normalization.fit_baseline_m must be a JSON number"
        )
    baseline = float(payload["fit_baseline_m"])
    if not math.isfinite(baseline) or baseline <= 0.0:
        raise DepthRegistrationError(
            "reference_scale_normalization.fit_baseline_m must be finite and positive"
        )
    for key in (
        "fit_distinct_frame_count",
        "holdout_distinct_frame_count",
        "normalized_observation_count",
    ):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise DepthRegistrationError(
                f"reference_scale_normalization.{key} must be a positive integer"
            )
    ranges = {
        key: _validate_exact_positive_range(
            payload.get(key), name=f"reference_scale_normalization.{key}"
        )
        for key in (
            "fit_frame_scale_range_m",
            "holdout_frame_scale_range_m",
            "fit_normalization_factor_range",
            "holdout_normalization_factor_range",
            "fit_raw_reference_range_m",
            "holdout_raw_reference_range_m",
            "fit_normalized_reference_range_m",
            "holdout_normalized_reference_range_m",
        )
    }
    fit_scale_range = ranges["fit_frame_scale_range_m"]
    if baseline < fit_scale_range[0] or baseline > fit_scale_range[1]:
        raise DepthRegistrationError(
            "reference_scale_normalization fit baseline lies outside fit scale range"
        )
    for prefix in ("fit", "holdout"):
        scale_range = ranges[f"{prefix}_frame_scale_range_m"]
        expected_factor_range = (
            baseline / scale_range[1],
            baseline / scale_range[0],
        )
        observed_factor_range = ranges[f"{prefix}_normalization_factor_range"]
        if any(
            not math.isclose(
                observed_factor_range[index],
                expected_factor_range[index],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for index in (0, 1)
        ):
            raise DepthRegistrationError(
                f"reference_scale_normalization {prefix} factor range is inconsistent"
            )
    normalized_observations_sha256 = payload.get("normalized_observations_sha256")
    if not isinstance(normalized_observations_sha256, str):
        raise DepthRegistrationError(
            "reference_scale_normalization.normalized_observations_sha256 "
            "must be a string"
        )
    _require_sha256(
        normalized_observations_sha256,
        name="reference_scale_normalization.normalized_observations_sha256",
    )


def _validate_moving_track_eligibility(payload: Any) -> None:
    expected_keys = {
        "contract",
        "grouping",
        "selection_inputs",
        "percentile_rank_method",
        "min_track_observations",
        "min_distinct_frames_per_track",
        "min_track_timestamp_span_us",
        "min_robust_anchor_motion_frame_diagonal_fraction",
        "anchor_frame_size",
        "min_robust_anchor_motion_px",
        "input_track_count",
        "eligible_track_count",
        "rejected_track_count",
        "input_observation_count",
        "eligible_observation_count",
        "rejected_observation_count",
        "eligible_observations_sha256",
        "phases",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        raise DepthRegistrationError(
            "moving_track_eligibility fields do not match the exact contract"
        )
    if (
        payload.get("contract") != MOVING_TRACK_ELIGIBILITY_CONTRACT
        or payload.get("grouping") != "camera_capture_id_tracker_id"
        or payload.get("selection_inputs")
        != [
            "capture_id",
            "source_frame_id",
            "tracker_id",
            "anchor_uv",
            "raw_timestamp_us",
        ]
        or payload.get("percentile_rank_method")
        != "sorted_floor_p10_ceil_p90_integer_rank"
    ):
        raise DepthRegistrationError("moving_track_eligibility policy is invalid")
    integer_policy = {
        "min_track_observations": 8,
        "min_distinct_frames_per_track": 8,
        "min_track_timestamp_span_us": 1_000_000,
    }
    if any(
        isinstance(payload.get(key), bool)
        or not isinstance(payload.get(key), int)
        or payload.get(key) != expected
        for key, expected in integer_policy.items()
    ):
        raise DepthRegistrationError("moving_track_eligibility policy is invalid")
    motion_fraction = payload.get("min_robust_anchor_motion_frame_diagonal_fraction")
    if (
        not _is_exact_json_number(motion_fraction)
        or not math.isfinite(float(motion_fraction))
        or float(motion_fraction) != 0.005
    ):
        raise DepthRegistrationError("moving_track_eligibility policy is invalid")
    frame_size = payload.get("anchor_frame_size")
    if (
        not isinstance(frame_size, list)
        or len(frame_size) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in frame_size
        )
    ):
        raise DepthRegistrationError(
            "moving_track_eligibility.anchor_frame_size is invalid"
        )
    expected_motion_px = 0.005 * math.hypot(float(frame_size[0]), float(frame_size[1]))
    if not _is_exact_json_number(payload.get("min_robust_anchor_motion_px")):
        raise DepthRegistrationError(
            "moving_track_eligibility.min_robust_anchor_motion_px must be a JSON number"
        )
    observed_motion_px = float(payload["min_robust_anchor_motion_px"])
    if not math.isclose(
        observed_motion_px,
        expected_motion_px,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise DepthRegistrationError(
            "moving_track_eligibility minimum motion disagrees with frame diagonal"
        )
    count_keys = (
        "input_track_count",
        "eligible_track_count",
        "rejected_track_count",
        "input_observation_count",
        "eligible_observation_count",
        "rejected_observation_count",
    )
    for key in count_keys:
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DepthRegistrationError(
                f"moving_track_eligibility.{key} must be a non-negative integer"
            )
    if (
        int(payload["input_track_count"])
        != int(payload["eligible_track_count"]) + int(payload["rejected_track_count"])
        or int(payload["input_observation_count"])
        != int(payload["eligible_observation_count"])
        + int(payload["rejected_observation_count"])
        or int(payload["eligible_track_count"]) <= 0
        or int(payload["eligible_observation_count"]) <= 0
    ):
        raise DepthRegistrationError(
            "moving_track_eligibility aggregate counts are inconsistent"
        )
    phases = payload.get("phases")
    phase_count_keys = {
        "input_track_count",
        "eligible_track_count",
        "rejected_track_count",
        "input_observation_count",
        "eligible_observation_count",
        "rejected_observation_count",
    }
    if not isinstance(phases, Mapping) or set(phases) != {
        "fit-a",
        "fit-b",
        "holdout-a",
        "holdout-b",
    }:
        raise DepthRegistrationError(
            "moving_track_eligibility phases do not match the exact capture contract"
        )
    phase_totals = {key: 0 for key in count_keys}
    for capture_id, counts in phases.items():
        if not isinstance(counts, Mapping) or set(counts) != phase_count_keys:
            raise DepthRegistrationError(
                f"moving_track_eligibility phase {capture_id} fields are invalid"
            )
        for key in count_keys:
            value = counts.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DepthRegistrationError(
                    f"moving_track_eligibility phase {capture_id}.{key} is invalid"
                )
            phase_totals[key] += int(value)
        if int(counts["input_track_count"]) != int(
            counts["eligible_track_count"]
        ) + int(counts["rejected_track_count"]) or int(
            counts["input_observation_count"]
        ) != int(counts["eligible_observation_count"]) + int(
            counts["rejected_observation_count"]
        ):
            raise DepthRegistrationError(
                f"moving_track_eligibility phase {capture_id} counts are inconsistent"
            )
    if any(int(payload[key]) != total for key, total in phase_totals.items()):
        raise DepthRegistrationError(
            "moving_track_eligibility phase counts disagree with aggregate counts"
        )
    eligible_observations_sha256 = payload.get("eligible_observations_sha256")
    if not isinstance(eligible_observations_sha256, str):
        raise DepthRegistrationError(
            "moving_track_eligibility.eligible_observations_sha256 must be a string"
        )
    _require_sha256(
        eligible_observations_sha256,
        name="moving_track_eligibility.eligible_observations_sha256",
    )


def _dav2_identity_payload(profile: Mapping[str, Any]) -> dict[str, Any]:
    payload = _profile_payload_without_fingerprint(profile)
    identity: dict[str, Any] = {
        "model_name": str(payload.get("model_name") or ""),
    }
    try:
        identity["input_size"] = [int(v) for v in (payload.get("input_size") or [])]
    except Exception:
        identity["input_size"] = payload.get("input_size")
    for key in ("batch_size", "gie_id"):
        try:
            identity[key] = int(payload.get(key))
        except Exception:
            identity[key] = payload.get(key)
    engine = payload.get("model-engine-file", payload.get("engine"))
    if engine is not None:
        identity["engine"] = str(engine)
    onnx = payload.get("onnx-file", payload.get("onnx"))
    if onnx is not None:
        identity["onnx"] = str(onnx)
    return identity


def _dav2_profile_matches(
    artifact_profile: Mapping[str, Any],
    runtime_profile: Mapping[str, Any],
) -> bool:
    if dict(artifact_profile).get("fingerprint_sha256") == dict(runtime_profile).get(
        "fingerprint_sha256"
    ):
        return True

    # The DAv2 registration maps raw model depth to MapAnything room depth. It
    # depends on model/input identity, not on how often the same model runs.
    # Accept cadence-only changes so runtime interval tuning does not force an
    # unnecessary room-registration rebuild. Older artifacts were generated from
    # the YAML model stanza (`engine`), while runtime fingerprints are generated
    # from the materialized nvinfer stanza (`model-engine-file`/`onnx-file`).
    artifact_core = _dav2_identity_payload(artifact_profile)
    runtime_core = _dav2_identity_payload(runtime_profile)
    for key in ("model_name", "input_size", "batch_size", "gie_id", "engine"):
        if artifact_core.get(key) != runtime_core.get(key):
            return False
    if (
        "onnx" in artifact_core
        and "onnx" in runtime_core
        and artifact_core.get("onnx") != runtime_core.get("onnx")
    ):
        return False
    return True


def _mapanything_identity_payload(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the registration-relevant MapAnything model identity.

    Runtime graph materialization may replace the reviewed nvinfer config path
    with a content-addressed engine-only copy. That path and inference cadence
    are plumbing; all other recorded fields remain part of the legacy semantic
    identity and therefore compare strictly.
    """

    payload = _profile_payload_without_fingerprint(profile)
    for key in (
        "config-file",
        "config-file-path",
        "interval",
        "secondary-reinfer-interval",
    ):
        payload.pop(key, None)

    aliases = (
        ("engine", ("engine", "model-engine-file")),
        ("batch_size", ("batch_size", "batch-size")),
        ("gie_id", ("gie_id", "gie-id", "unique-id")),
        ("attach_tensor_meta", ("attach_tensor_meta", "output-tensor-meta")),
    )
    for canonical, keys in aliases:
        values = [payload.pop(key) for key in keys if key in payload]
        if not values:
            continue
        normalized_values: list[Any] = []
        for value in values:
            if canonical in {"batch_size", "gie_id"}:
                try:
                    value = int(value)
                except Exception:
                    pass
            elif canonical == "attach_tensor_meta":
                if isinstance(value, str):
                    value = value.strip().lower() in {"1", "true", "yes", "on"}
                else:
                    value = bool(value)
            else:
                value = str(value)
            normalized_values.append(value)
        payload[canonical] = (
            normalized_values[0]
            if all(value == normalized_values[0] for value in normalized_values)
            else {"alias_conflict": normalized_values}
        )

    # Source files are explicit maintenance inputs, not runtime model identity
    # once the selected engine is bound.
    payload.pop("onnx", None)
    payload.pop("onnx-file", None)
    return payload


def _mapanything_profile_matches(
    artifact_profile: Mapping[str, Any],
    runtime_profile: Mapping[str, Any],
) -> bool:
    if dict(artifact_profile).get("fingerprint_sha256") == dict(runtime_profile).get(
        "fingerprint_sha256"
    ):
        return True
    return _mapanything_identity_payload(
        artifact_profile
    ) == _mapanything_identity_payload(runtime_profile)


@dataclass(frozen=True, slots=True)
class DepthRegistrationEntry:
    camera_id: str
    created_ts_us: int
    transform_type: str
    source_space: str
    target_space: str
    scope: str
    raw_range_domain_m: tuple[float, float]
    knots_raw_m: tuple[float, ...]
    knots_registered_m: tuple[float, ...]
    calibration_fingerprint: Mapping[str, Any]
    dav2_profile: Mapping[str, Any]
    mapanything_profile: Mapping[str, Any]
    fit_metrics: Mapping[str, Any]
    sample_counts: Mapping[str, Any]
    generation_tool_version: str
    provenance: Mapping[str, Any]
    occupied_anchor_validation: Mapping[str, Any] = field(default_factory=dict)

    @property
    def registration_id(self) -> str:
        if self.occupied_anchor_validation:
            payload = {
                "camera_id": self.camera_id,
                "created_ts_us": int(self.created_ts_us),
                "transform_type": self.transform_type,
                "source_space": self.source_space,
                "target_space": self.target_space,
                "scope": self.scope,
                "raw_range_domain_m": list(self.raw_range_domain_m),
                "knots_raw_m": list(self.knots_raw_m),
                "knots_registered_m": list(self.knots_registered_m),
                "calibration_fingerprint": dict(self.calibration_fingerprint),
                "dav2_profile": dict(self.dav2_profile),
                "mapanything_profile": dict(self.mapanything_profile),
                "fit_metrics": dict(self.fit_metrics),
                "sample_counts": dict(self.sample_counts),
                "generation_tool_version": self.generation_tool_version,
                "provenance": dict(self.provenance),
                "occupied_anchor_validation": dict(self.occupied_anchor_validation),
            }
        else:
            payload = {
                "camera_id": self.camera_id,
                "created_ts_us": int(self.created_ts_us),
                "raw_range_domain_m": list(self.raw_range_domain_m),
                "knots_raw_m": list(self.knots_raw_m),
                "knots_registered_m": list(self.knots_registered_m),
            }
        digest = _sha256(payload)
        return f"{self.camera_id}:{digest if self.occupied_anchor_validation else digest[:12]}"

    def apply(self, raw_depth_m: float) -> float | None:
        if not math.isfinite(float(raw_depth_m)) or float(raw_depth_m) <= 0.0:
            return None
        lo, hi = self.raw_range_domain_m
        value = float(raw_depth_m)
        if value < float(lo) or value > float(hi):
            return None
        return float(np.interp(value, self.knots_raw_m, self.knots_registered_m))

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "camera_id": self.camera_id,
            "created_ts_us": int(self.created_ts_us),
            "transform_type": self.transform_type,
            "source_space": self.source_space,
            "target_space": self.target_space,
            "scope": self.scope,
            "raw_range_domain_m": [
                float(self.raw_range_domain_m[0]),
                float(self.raw_range_domain_m[1]),
            ],
            "knots_raw_m": [float(v) for v in self.knots_raw_m],
            "knots_registered_m": [float(v) for v in self.knots_registered_m],
            "calibration_fingerprint": dict(self.calibration_fingerprint),
            "dav2_profile": dict(self.dav2_profile),
            "mapanything_profile": dict(self.mapanything_profile),
            "fit_metrics": dict(self.fit_metrics),
            "sample_counts": dict(self.sample_counts),
            "generation_tool_version": self.generation_tool_version,
            "provenance": dict(self.provenance),
            "registration_id": self.registration_id,
        }
        if self.occupied_anchor_validation:
            payload["occupied_anchor_validation"] = dict(
                self.occupied_anchor_validation
            )
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DepthRegistrationEntry":
        if str(payload.get("transform_type")) != TRANSFORM_TYPE_PIECEWISE:
            raise DepthRegistrationError(
                "Unsupported depth registration transform_type"
            )
        if str(payload.get("source_space")) != SOURCE_SPACE:
            raise DepthRegistrationError("Unsupported depth registration source_space")
        if str(payload.get("target_space")) != TARGET_SPACE:
            raise DepthRegistrationError("Unsupported depth registration target_space")
        if str(payload.get("scope")) != REGISTRATION_SCOPE:
            raise DepthRegistrationError("Unsupported depth registration scope")
        raw_knots = _float_tuple(payload.get("knots_raw_m") or [], name="knots_raw_m")
        registered_knots = _float_tuple(
            payload.get("knots_registered_m") or [], name="knots_registered_m"
        )
        if len(raw_knots) < 2 or len(raw_knots) != len(registered_knots):
            raise DepthRegistrationError(
                "knots_raw_m and knots_registered_m must have the same length >= 2"
            )
        _ensure_strictly_increasing(raw_knots, name="knots_raw_m")
        _ensure_monotonic_non_decreasing(registered_knots, name="knots_registered_m")
        raw_domain_seq = payload.get("raw_range_domain_m") or []
        if not isinstance(raw_domain_seq, Sequence) or len(raw_domain_seq) != 2:
            raise DepthRegistrationError(
                "raw_range_domain_m must contain exactly two values"
            )
        raw_domain = (float(raw_domain_seq[0]), float(raw_domain_seq[1]))
        if raw_domain[0] > raw_domain[1]:
            raise DepthRegistrationError("raw_range_domain_m must be ordered low->high")
        if (
            abs(raw_domain[0] - raw_knots[0]) > 1e-6
            or abs(raw_domain[1] - raw_knots[-1]) > 1e-6
        ):
            raise DepthRegistrationError(
                "raw_range_domain_m must match the knot endpoints"
            )
        entry = cls(
            camera_id=str(payload.get("camera_id") or ""),
            created_ts_us=int(payload.get("created_ts_us") or 0),
            transform_type=TRANSFORM_TYPE_PIECEWISE,
            source_space=SOURCE_SPACE,
            target_space=TARGET_SPACE,
            scope=REGISTRATION_SCOPE,
            raw_range_domain_m=raw_domain,
            knots_raw_m=raw_knots,
            knots_registered_m=registered_knots,
            calibration_fingerprint=dict(payload.get("calibration_fingerprint") or {}),
            dav2_profile=dict(payload.get("dav2_profile") or {}),
            mapanything_profile=dict(payload.get("mapanything_profile") or {}),
            fit_metrics=dict(payload.get("fit_metrics") or {}),
            sample_counts=dict(payload.get("sample_counts") or {}),
            generation_tool_version=str(payload.get("generation_tool_version") or ""),
            provenance=dict(payload.get("provenance") or {}),
            occupied_anchor_validation=dict(
                payload.get("occupied_anchor_validation") or {}
            ),
        )
        recorded_registration_id = str(payload.get("registration_id") or "").strip()
        if (
            entry.occupied_anchor_validation
            and recorded_registration_id != entry.registration_id
        ):
            raise DepthRegistrationError(
                "registration_id does not match the hardened entry payload"
            )
        return entry


@dataclass(frozen=True, slots=True)
class DepthRegistrationBundle:
    entries: Mapping[str, DepthRegistrationEntry]
    contract_version: int = CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth_registration_contract_version": int(self.contract_version),
            "cameras": {
                camera_id: entry.to_dict()
                for camera_id, entry in sorted(self.entries.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DepthRegistrationBundle":
        version = int(
            payload.get("depth_registration_contract_version")
            or payload.get("contract_version")
            or 0
        )
        if version not in SUPPORTED_CONTRACT_VERSIONS:
            raise DepthRegistrationError(
                "Unsupported depth registration contract version "
                f"{version}; expected one of {SUPPORTED_CONTRACT_VERSIONS}"
            )
        if version == HARDENED_CONTRACT_VERSION and set(payload) != {
            "depth_registration_contract_version",
            "cameras",
        }:
            raise DepthRegistrationError(
                "hardened depth registration root fields do not match the exact contract"
            )
        cameras = payload.get("cameras")
        if not isinstance(cameras, Mapping):
            raise DepthRegistrationError(
                "depth registration bundle missing cameras mapping"
            )
        if version == HARDENED_CONTRACT_VERSION and not cameras:
            raise DepthRegistrationError(
                "hardened depth registration bundle requires at least one camera"
            )
        entries: dict[str, DepthRegistrationEntry] = {}
        for camera_id, entry_payload in cameras.items():
            if not isinstance(entry_payload, Mapping):
                raise DepthRegistrationError(
                    f"camera {camera_id!r} registration entry must be a mapping"
                )
            if version == HARDENED_CONTRACT_VERSION and set(entry_payload) != {
                "camera_id",
                "created_ts_us",
                "transform_type",
                "source_space",
                "target_space",
                "scope",
                "raw_range_domain_m",
                "knots_raw_m",
                "knots_registered_m",
                "calibration_fingerprint",
                "dav2_profile",
                "mapanything_profile",
                "fit_metrics",
                "sample_counts",
                "generation_tool_version",
                "provenance",
                "registration_id",
                "occupied_anchor_validation",
            }:
                raise DepthRegistrationError(
                    f"camera {camera_id!r} hardened entry fields do not match the exact contract"
                )
            entry = DepthRegistrationEntry.from_dict(entry_payload)
            if entry.camera_id and entry.camera_id != str(camera_id):
                raise DepthRegistrationError(
                    f"camera id mismatch in entry {camera_id!r}"
                )
            entries[str(camera_id)] = entry
            if version == HARDENED_CONTRACT_VERSION:
                if not entry.camera_id or int(entry.created_ts_us) <= 0:
                    raise DepthRegistrationError(
                        f"camera {camera_id!r} hardened registration identity is incomplete"
                    )
                normalized_calibration = _normalize_calibration_fingerprint_payload(
                    entry.calibration_fingerprint
                )
                if dict(entry.calibration_fingerprint) != normalized_calibration:
                    raise DepthRegistrationError(
                        f"camera {camera_id!r} calibration fingerprint is not canonical"
                    )
                _validate_model_content_binding(entry.dav2_profile, name="dav2_profile")
                _validate_model_content_binding(
                    entry.mapanything_profile,
                    name="mapanything_profile",
                )
                _validate_occupied_anchor_validation(entry.occupied_anchor_validation)
                occupied = entry.occupied_anchor_validation
                fit_range = tuple(float(value) for value in occupied["fit_raw_range_m"])
                if any(
                    abs(
                        float(fit_range[index]) - float(entry.raw_range_domain_m[index])
                    )
                    > 1e-9
                    for index in (0, 1)
                ):
                    raise DepthRegistrationError(
                        "occupied fit range does not match the registration domain"
                    )
                count_fields = {
                    "occupied_fit_observations": "fit_observations",
                    "occupied_holdout_observations": "holdout_observations",
                    "occupied_fit_captures": "fit_capture_count",
                    "occupied_holdout_captures": "holdout_capture_count",
                }
                for sample_key, validation_key in count_fields.items():
                    if entry.sample_counts.get(sample_key) != occupied.get(
                        validation_key
                    ):
                        raise DepthRegistrationError(
                            f"sample_counts.{sample_key} disagrees with occupied validation"
                        )
                holdout_metrics = entry.fit_metrics.get("occupied_person_holdout")
                if not isinstance(holdout_metrics, Mapping):
                    raise DepthRegistrationError(
                        "fit_metrics.occupied_person_holdout is required"
                    )
                for metric in (
                    "median_abs_error_m",
                    "p95_abs_error_m",
                    "max_abs_error_m",
                ):
                    if float(holdout_metrics.get(metric, math.nan)) != float(
                        occupied.get(metric, math.nan)
                    ):
                        raise DepthRegistrationError(
                            f"fit_metrics.occupied_person_holdout.{metric} disagrees with occupied validation"
                        )
                if entry.provenance.get(
                    "runtime_anchor_evidence_sha256"
                ) != occupied.get("evidence_sha256"):
                    raise DepthRegistrationError(
                        "provenance runtime anchor evidence disagrees with occupied validation"
                    )
                if entry.provenance.get("extrapolation_allowed") is not False:
                    raise DepthRegistrationError(
                        "hardened depth registration must explicitly forbid extrapolation"
                    )
                expected_provenance_keys = {
                    "source_uri",
                    "source_sha256",
                    "source_size_bytes",
                    "source_media",
                    "source_derivative_provenance",
                    "source_bundle_sha256",
                    "runtime_anchor_evidence_contract",
                    "runtime_anchor_evidence_sha256",
                    "runtime_instance_id",
                    "runtime_run_id",
                    "observation_timestamp_range_us",
                    "fit_source",
                    "extrapolation_allowed",
                    "calibration_admission",
                    "moving_track_eligibility",
                    "reference_scale_normalization",
                }
                if set(entry.provenance) != expected_provenance_keys:
                    raise DepthRegistrationError(
                        "hardened depth registration provenance fields do not match the exact contract"
                    )
                if (
                    not str(entry.provenance.get("source_uri") or "").strip()
                    or _SHA256_RE.fullmatch(
                        str(entry.provenance.get("source_sha256") or "")
                    )
                    is None
                    or isinstance(entry.provenance.get("source_size_bytes"), bool)
                    or not isinstance(entry.provenance.get("source_size_bytes"), int)
                    or int(entry.provenance.get("source_size_bytes")) <= 0
                    or _SHA256_RE.fullmatch(
                        str(entry.provenance.get("source_bundle_sha256") or "")
                    )
                    is None
                    or not str(
                        entry.provenance.get("runtime_instance_id") or ""
                    ).strip()
                    or not str(entry.provenance.get("runtime_run_id") or "").strip()
                    or entry.provenance.get("runtime_anchor_evidence_contract")
                    != RUNTIME_OCCUPIED_ANCHOR_EVIDENCE_CONTRACT
                    or entry.provenance.get("fit_source")
                    != "occupied_runtime_anchor_pairs"
                    or entry.generation_tool_version != "depth_registration_builder_v3"
                ):
                    raise DepthRegistrationError(
                        "hardened depth registration provenance identity is invalid"
                    )
                if entry.provenance.get("moving_track_eligibility") != occupied.get(
                    "moving_track_eligibility"
                ):
                    raise DepthRegistrationError(
                        "provenance moving-track eligibility disagrees with occupied validation"
                    )
                if entry.provenance.get(
                    "reference_scale_normalization"
                ) != occupied.get("reference_scale_normalization"):
                    raise DepthRegistrationError(
                        "provenance reference-scale normalization disagrees with occupied validation"
                    )
                source_media = entry.provenance.get("source_media")
                if (
                    not isinstance(source_media, Mapping)
                    or set(source_media)
                    != {
                        "codec_name",
                        "width",
                        "height",
                        "nominal_frame_rate",
                        "average_frame_rate",
                        "duration_seconds",
                        "start_time_seconds",
                        "frame_count",
                        "first_pts_seconds",
                        "last_pts_seconds",
                        "pts_monotonic_verified",
                        "file_loop",
                    }
                    or source_media.get("codec_name") != "h264"
                    or source_media.get("nominal_frame_rate") != "30/1"
                    or source_media.get("average_frame_rate") != "30/1"
                    or source_media.get("file_loop") is not False
                    or source_media.get("pts_monotonic_verified") is not True
                    or isinstance(source_media.get("width"), bool)
                    or not isinstance(source_media.get("width"), int)
                    or int(source_media.get("width")) <= 0
                    or isinstance(source_media.get("height"), bool)
                    or not isinstance(source_media.get("height"), int)
                    or int(source_media.get("height")) <= 0
                    or isinstance(source_media.get("duration_seconds"), bool)
                    or not isinstance(
                        source_media.get("duration_seconds"), (int, float)
                    )
                    or not math.isfinite(float(source_media.get("duration_seconds")))
                    or abs(float(source_media.get("duration_seconds")) - 68.0) > 0.002
                    or isinstance(source_media.get("start_time_seconds"), bool)
                    or not isinstance(
                        source_media.get("start_time_seconds"), (int, float)
                    )
                    or abs(float(source_media.get("start_time_seconds"))) > 0.000001
                    or isinstance(source_media.get("frame_count"), bool)
                    or source_media.get("frame_count") != 2040
                    or isinstance(source_media.get("first_pts_seconds"), bool)
                    or not isinstance(
                        source_media.get("first_pts_seconds"), (int, float)
                    )
                    or abs(float(source_media.get("first_pts_seconds"))) > 0.000001
                    or isinstance(source_media.get("last_pts_seconds"), bool)
                    or not isinstance(
                        source_media.get("last_pts_seconds"), (int, float)
                    )
                    or abs(float(source_media.get("last_pts_seconds")) - 67.966667)
                    > 0.000002
                ):
                    raise DepthRegistrationError(
                        "hardened depth registration source media provenance is invalid"
                    )
                derivative_provenance = entry.provenance.get(
                    "source_derivative_provenance"
                )
                if (
                    not isinstance(derivative_provenance, Mapping)
                    or set(derivative_provenance)
                    != {"contract", "size_bytes", "sha256"}
                    or derivative_provenance.get("contract")
                    != "noesis.ds9.calibration.source_derivative.v3"
                    or isinstance(derivative_provenance.get("size_bytes"), bool)
                    or not isinstance(derivative_provenance.get("size_bytes"), int)
                    or int(derivative_provenance.get("size_bytes")) <= 0
                    or _SHA256_RE.fullmatch(
                        str(derivative_provenance.get("sha256") or "")
                    )
                    is None
                ):
                    raise DepthRegistrationError(
                        "hardened depth registration source derivative provenance is invalid"
                    )
                admission = entry.provenance.get("calibration_admission")
                if (
                    not isinstance(admission, Mapping)
                    or set(admission)
                    != {
                        "contract",
                        "session_id",
                        "launcher_manifest_sha256",
                        "summary_sha256",
                        "mounted_authority_sha256",
                        "checkout_authority_file_sha256",
                        "gpu_owner_ledger_sha256",
                        "plan_authorization_sha256",
                        "runtime_anchor_evidence_sha256",
                    }
                    or admission.get("contract")
                    != "noesis.ds9.depth_registration.calibration_admission.v2"
                    or not str(admission.get("session_id") or "").strip()
                    or any(
                        _SHA256_RE.fullmatch(str(admission.get(key) or "")) is None
                        for key in (
                            "launcher_manifest_sha256",
                            "summary_sha256",
                            "mounted_authority_sha256",
                            "checkout_authority_file_sha256",
                            "gpu_owner_ledger_sha256",
                            "plan_authorization_sha256",
                            "runtime_anchor_evidence_sha256",
                        )
                    )
                    or admission.get("runtime_anchor_evidence_sha256")
                    != entry.provenance.get("runtime_anchor_evidence_sha256")
                ):
                    raise DepthRegistrationError(
                        "hardened depth registration calibration admission is invalid"
                    )
                timestamp_range = entry.provenance.get("observation_timestamp_range_us")
                if (
                    not isinstance(timestamp_range, list)
                    or len(timestamp_range) != 2
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value <= 0
                        for value in timestamp_range
                    )
                    or timestamp_range[0] > timestamp_range[1]
                ):
                    raise DepthRegistrationError(
                        "hardened depth registration observation timestamp range is invalid"
                    )
        if version == HARDENED_CONTRACT_VERSION and entries:
            reference = next(iter(entries.values()))
            reference_identity = {
                "calibration_admission": dict(
                    reference.provenance["calibration_admission"]
                ),
                "source_bundle_sha256": reference.provenance["source_bundle_sha256"],
                "runtime_anchor_evidence_sha256": reference.provenance[
                    "runtime_anchor_evidence_sha256"
                ],
                "runtime_instance_id": reference.provenance["runtime_instance_id"],
                "runtime_run_id": reference.provenance["runtime_run_id"],
                "dav2_profile": dict(reference.dav2_profile),
                "mapanything_profile": dict(reference.mapanything_profile),
            }
            for camera_id, entry in entries.items():
                observed_identity = {
                    "calibration_admission": dict(
                        entry.provenance["calibration_admission"]
                    ),
                    "source_bundle_sha256": entry.provenance["source_bundle_sha256"],
                    "runtime_anchor_evidence_sha256": entry.provenance[
                        "runtime_anchor_evidence_sha256"
                    ],
                    "runtime_instance_id": entry.provenance["runtime_instance_id"],
                    "runtime_run_id": entry.provenance["runtime_run_id"],
                    "dav2_profile": dict(entry.dav2_profile),
                    "mapanything_profile": dict(entry.mapanything_profile),
                }
                if observed_identity != reference_identity:
                    raise DepthRegistrationError(
                        "hardened depth registration cameras must share one admitted "
                        f"capture/profile identity; camera {camera_id!r} differs"
                    )
        return cls(entries=entries, contract_version=version)


def write_depth_registration_bundle(
    *,
    path: Path,
    entries: Mapping[str, DepthRegistrationEntry],
    contract_version: int = CONTRACT_VERSION,
    expected_existing_sha256: str | None = None,
) -> Path:
    bundle = DepthRegistrationBundle(
        entries=dict(entries),
        contract_version=int(contract_version),
    )
    # Round-trip validation is part of publication; malformed v2 evidence must
    # never be written as an apparently usable artifact.
    DepthRegistrationBundle.from_dict(bundle.to_dict())
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    parent = destination.parent.resolve(strict=True)
    if destination.parent != parent or destination.is_symlink():
        raise DepthRegistrationError(
            "depth registration publication requires a canonical non-symlink destination"
        )
    replacing_existing = destination.exists()
    if replacing_existing:
        expected_digest = str(expected_existing_sha256 or "").strip().lower()
        if _SHA256_RE.fullmatch(expected_digest) is None:
            raise DepthRegistrationError(
                "existing depth registration replacement requires its admitted SHA-256"
            )
        existing = content_file_fingerprint(
            destination,
            logical_path=destination.name,
        )
        if existing["sha256"] != expected_digest:
            raise DepthRegistrationError(
                "existing depth registration bytes differ from the admitted checkout"
            )
    elif expected_existing_sha256 is not None:
        raise DepthRegistrationError("expected existing depth registration is missing")
    raw = (
        json.dumps(bundle.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    lock_path = parent / f".{destination.name}.publication.lock"
    temp_path = parent / f".{destination.name}.tmp-{os.getpid()}-{os.urandom(8).hex()}"
    lock_descriptor: int | None = None
    temp_descriptor: int | None = None
    directory_descriptor: int | None = None
    try:
        lock_descriptor = os.open(
            lock_path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.write(lock_descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(lock_descriptor)
        temp_descriptor = os.open(
            temp_path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        offset = 0
        while offset < len(raw):
            written = os.write(temp_descriptor, raw[offset:])
            if written <= 0:
                raise DepthRegistrationError(
                    "depth registration publication write made no progress"
                )
            offset += written
        os.fsync(temp_descriptor)
        metadata = os.fstat(temp_descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or int(metadata.st_nlink) != 1
            or int(metadata.st_uid) != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or int(metadata.st_size) != len(raw)
        ):
            raise DepthRegistrationError(
                "depth registration publication temporary-file contract failed"
            )
        os.close(temp_descriptor)
        temp_descriptor = None
        if replacing_existing:
            current = content_file_fingerprint(
                destination,
                logical_path=destination.name,
            )
            if current["sha256"] != str(expected_existing_sha256).strip().lower():
                raise DepthRegistrationError(
                    "existing depth registration changed during guarded publication"
                )
            os.replace(temp_path, destination)
        else:
            try:
                os.link(temp_path, destination, follow_symlinks=False)
            except FileExistsError as exc:
                raise DepthRegistrationError(
                    "depth registration publication destination appeared concurrently"
                ) from exc
            temp_path.unlink()
        directory_descriptor = os.open(
            parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        os.fsync(directory_descriptor)
    except OSError as exc:
        raise DepthRegistrationError(
            f"atomic depth registration publication failed: {exc}"
        ) from exc
    finally:
        if temp_descriptor is not None:
            os.close(temp_descriptor)
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        if lock_descriptor is not None:
            os.close(lock_descriptor)
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
    return destination


def load_depth_registration(path: Path) -> DepthRegistrationBundle:
    candidate = Path(path)
    lock_path = candidate.parent / f".{candidate.name}.publication.lock"

    def require_no_publication_lock(*, phase: str) -> None:
        try:
            lock_path.lstat()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise DepthRegistrationError(
                f"unable to inspect depth registration publication lock during {phase}: {exc}"
            ) from exc
        raise DepthRegistrationError(
            f"depth registration publication lock residue is present during {phase}: {lock_path}"
        )

    require_no_publication_lock(phase="pre-read")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except FileNotFoundError as exc:
        raise DepthRegistrationError(
            f"depth registration artifact not found: {candidate}"
        ) from exc
    except OSError as exc:
        raise DepthRegistrationError(
            f"unable to open depth registration artifact {candidate}: {exc}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or int(before.st_nlink) != 1
            or int(before.st_uid) != os.geteuid()
            or int(before.st_size) <= 0
            or int(before.st_size) > MAX_DEPTH_REGISTRATION_BYTES
        ):
            raise DepthRegistrationError(
                "depth registration artifact must be a bounded, current-user, "
                "single-link regular file"
            )
        chunks: list[bytes] = []
        remaining = int(before.st_size)
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                raise DepthRegistrationError(
                    "depth registration artifact ended while reading"
                )
            chunks.append(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise DepthRegistrationError(
                "depth registration artifact grew while reading"
            )
        after = os.fstat(descriptor)
        try:
            named = candidate.lstat()
        except OSError as exc:
            raise DepthRegistrationError(
                "depth registration named destination disappeared while reading"
            ) from exc

        def identity(value: os.stat_result) -> tuple[int, ...]:
            return (
                int(value.st_dev),
                int(value.st_ino),
                int(value.st_mode),
                int(value.st_nlink),
                int(value.st_uid),
                int(value.st_size),
                int(value.st_mtime_ns),
                int(value.st_ctime_ns),
            )

        if identity(before) != identity(after) or identity(before) != identity(named):
            raise DepthRegistrationError(
                "depth registration artifact changed while reading"
            )
        require_no_publication_lock(phase="post-read")
        raw = b"".join(chunks)
    finally:
        os.close(descriptor)
    try:
        payload = strict_json_loads(
            raw,
            label="depth registration artifact",
        )
    except Exception as exc:
        raise DepthRegistrationError(
            f"Unable to parse depth registration artifact {candidate}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise DepthRegistrationError(
            "depth registration artifact root must be a mapping"
        )
    return DepthRegistrationBundle.from_dict(payload)


class DepthRegistrationManager:
    def __init__(
        self, bundle: DepthRegistrationBundle, *, path: Path | None = None
    ) -> None:
        self.bundle = bundle
        self.path = path

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        required_contract_version: int | None = None,
    ) -> "DepthRegistrationManager":
        bundle = load_depth_registration(path)
        if required_contract_version is not None and int(
            bundle.contract_version
        ) != int(required_contract_version):
            raise DepthRegistrationError(
                "depth_registration_contract_version_mismatch:"
                f"required={int(required_contract_version)} observed={int(bundle.contract_version)}"
            )
        return cls(bundle, path=Path(path))

    def validate_runtime(
        self,
        *,
        camera_id: str,
        snapshot: Any,
        dav2_profile: Mapping[str, Any],
        mapanything_profile: Mapping[str, Any],
    ) -> DepthRegistrationEntry:
        entry = self.bundle.entries.get(str(camera_id))
        if entry is None:
            raise DepthRegistrationError(f"registration_missing:{camera_id}")
        expected_cal = calibration_fingerprint_from_snapshot(snapshot)
        actual_cal = _normalize_calibration_fingerprint_payload(
            dict(entry.calibration_fingerprint)
        )
        if actual_cal.get("fingerprint_sha256") != expected_cal.get(
            "fingerprint_sha256"
        ):
            raise DepthRegistrationError("calibration_fingerprint_mismatch")
        if self.bundle.contract_version == HARDENED_CONTRACT_VERSION:
            _validate_model_content_binding(dav2_profile, name="runtime.dav2_profile")
            _validate_model_content_binding(
                mapanything_profile,
                name="runtime.mapanything_profile",
            )
            dav2_matches = entry.dav2_profile.get(
                "fingerprint_sha256"
            ) == dav2_profile.get("fingerprint_sha256")
            mapanything_matches = entry.mapanything_profile.get(
                "fingerprint_sha256"
            ) == mapanything_profile.get("fingerprint_sha256")
        else:
            dav2_matches = _dav2_profile_matches(entry.dav2_profile, dav2_profile)
            mapanything_matches = _mapanything_profile_matches(
                entry.mapanything_profile,
                mapanything_profile,
            )
        if not dav2_matches:
            raise DepthRegistrationError("dav2_profile_fingerprint_mismatch")
        if not mapanything_matches:
            raise DepthRegistrationError("mapanything_profile_fingerprint_mismatch")
        return entry

    def apply(
        self, *, camera_id: str, raw_depth_m: float
    ) -> tuple[float | None, str, str | None]:
        entry = self.bundle.entries.get(str(camera_id))
        if entry is None:
            return None, "registration_missing", None
        corrected = entry.apply(float(raw_depth_m))
        if corrected is None:
            return None, "out_of_domain_or_invalid", entry.registration_id
        return corrected, "ok", entry.registration_id


def fit_piecewise_registration(
    raw_depth_m: Sequence[float],
    registered_depth_m: Sequence[float],
    *,
    num_knots: int = 8,
) -> MutableMapping[str, Any]:
    raw = np.asarray(raw_depth_m, dtype=np.float64).reshape(-1)
    registered = np.asarray(registered_depth_m, dtype=np.float64).reshape(-1)
    finite = (
        np.isfinite(raw) & np.isfinite(registered) & (raw > 0.0) & (registered > 0.0)
    )
    raw = raw[finite]
    registered = registered[finite]
    if raw.size < max(4, int(num_knots)):
        raise DepthRegistrationError("insufficient_samples_for_piecewise_fit")
    quantiles = np.linspace(0.0, 1.0, max(4, int(num_knots)))
    raw_knots = np.quantile(raw, quantiles)
    reg_knots = np.quantile(registered, quantiles)
    reg_knots = np.maximum.accumulate(reg_knots)
    return {
        "transform_type": TRANSFORM_TYPE_PIECEWISE,
        "raw_range_domain_m": [float(raw_knots[0]), float(raw_knots[-1])],
        "knots_raw_m": [float(v) for v in raw_knots.tolist()],
        "knots_registered_m": [float(v) for v in reg_knots.tolist()],
    }


__all__ = [
    "CONTRACT_VERSION",
    "HARDENED_CONTRACT_VERSION",
    "MAX_OCCUPIED_ABS_ERROR_M",
    "MAX_OCCUPIED_ANCHOR_TIME_DELTA_MS",
    "MAX_OCCUPIED_MEDIAN_ABS_ERROR_M",
    "MAX_OCCUPIED_P95_ABS_ERROR_M",
    "MAPANYTHING_FRAME_SCALE_NORMALIZATION_CONTRACT",
    "MAPANYTHING_FRAME_SCALE_STATISTIC",
    "MAPANYTHING_FRAME_SCALE_STATISTIC_CONTRACT",
    "MIN_MAPANYTHING_FRAME_SCALE_SUPPORT_COUNT",
    "MIN_OCCUPIED_ANCHOR_SUPPORT",
    "MIN_OCCUPIED_CAPTURES_PER_PARTITION",
    "MIN_OCCUPIED_FIT_OBSERVATIONS",
    "MIN_OCCUPIED_HOLDOUT_OBSERVATIONS",
    "MODEL_CONTENT_BINDING_CONTRACT",
    "MOVING_TRACK_ELIGIBILITY_CONTRACT",
    "OCCUPIED_ANCHOR_VALIDATION_CONTRACT",
    "RUNTIME_OCCUPIED_ANCHOR_EVIDENCE_CONTRACT",
    "SUPPORTED_CONTRACT_VERSIONS",
    "DepthRegistrationBundle",
    "DepthRegistrationEntry",
    "DepthRegistrationError",
    "DepthRegistrationManager",
    "REGISTRATION_SCOPE",
    "SOURCE_SPACE",
    "TARGET_SPACE",
    "TRANSFORM_TYPE_PIECEWISE",
    "calibration_fingerprint",
    "calibration_fingerprint_from_snapshot",
    "content_bundle_fingerprint",
    "content_file_fingerprint",
    "fit_piecewise_registration",
    "load_depth_registration",
    "model_profile_fingerprint",
    "profile_fingerprint",
    "validate_mapanything_frame_scale_statistic",
    "write_depth_registration_bundle",
]
