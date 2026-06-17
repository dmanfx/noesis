from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np

CONTRACT_VERSION = 1
TRANSFORM_TYPE_PIECEWISE = "piecewise_linear_1d"
SOURCE_SPACE = "dav2_anchor_range_m_raw"
TARGET_SPACE = "mapanything_room_range_m"
REGISTRATION_SCOPE = "people_tracking_depth_registration"
CALIBRATION_FINGERPRINT_BASIS = "image_space_intrinsics_v1"


class DepthRegistrationError(Exception):
    """Raised when the DS8 depth-registration artifact is invalid or mismatched."""


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _normalize_path_like(value: Any, *, repo_root: Path) -> Any:
    if not isinstance(value, str):
        return value
    raw = value.strip()
    if not raw:
        return raw
    candidate = Path(raw)
    if candidate.is_absolute():
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
            raise DepthRegistrationError(f"{name} contains a non-numeric value: {value!r}") from exc
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


def _normalize_calibration_fingerprint_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    intrinsics = np.asarray(payload.get("intrinsics"), dtype=np.float64)
    if intrinsics.shape != (3, 3):
        raise DepthRegistrationError("Depth registration calibration fingerprint intrinsics must be 3x3")
    image_size = payload.get("image_size")
    if not isinstance(image_size, Sequence) or len(image_size) != 2:
        raise DepthRegistrationError("Depth registration calibration fingerprint image_size must contain two values")
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
                for nested_key, nested_value in sorted(value.items(), key=lambda item: str(item[0]))
            }
        elif isinstance(value, (list, tuple)):
            normalized[str(key)] = [_normalize_path_like(v, repo_root=repo_root) for v in value]
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

    @property
    def registration_id(self) -> str:
        payload = {
            "camera_id": self.camera_id,
            "created_ts_us": int(self.created_ts_us),
            "raw_range_domain_m": list(self.raw_range_domain_m),
            "knots_raw_m": list(self.knots_raw_m),
            "knots_registered_m": list(self.knots_registered_m),
        }
        return f"{self.camera_id}:{_sha256(payload)[:12]}"

    def apply(self, raw_depth_m: float) -> float | None:
        if not math.isfinite(float(raw_depth_m)) or float(raw_depth_m) <= 0.0:
            return None
        lo, hi = self.raw_range_domain_m
        value = float(raw_depth_m)
        if value < float(lo) or value > float(hi):
            return None
        return float(np.interp(value, self.knots_raw_m, self.knots_registered_m))

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "created_ts_us": int(self.created_ts_us),
            "transform_type": self.transform_type,
            "source_space": self.source_space,
            "target_space": self.target_space,
            "scope": self.scope,
            "raw_range_domain_m": [float(self.raw_range_domain_m[0]), float(self.raw_range_domain_m[1])],
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

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DepthRegistrationEntry":
        if str(payload.get("transform_type")) != TRANSFORM_TYPE_PIECEWISE:
            raise DepthRegistrationError("Unsupported depth registration transform_type")
        if str(payload.get("source_space")) != SOURCE_SPACE:
            raise DepthRegistrationError("Unsupported depth registration source_space")
        if str(payload.get("target_space")) != TARGET_SPACE:
            raise DepthRegistrationError("Unsupported depth registration target_space")
        if str(payload.get("scope")) != REGISTRATION_SCOPE:
            raise DepthRegistrationError("Unsupported depth registration scope")
        raw_knots = _float_tuple(payload.get("knots_raw_m") or [], name="knots_raw_m")
        registered_knots = _float_tuple(payload.get("knots_registered_m") or [], name="knots_registered_m")
        if len(raw_knots) < 2 or len(raw_knots) != len(registered_knots):
            raise DepthRegistrationError("knots_raw_m and knots_registered_m must have the same length >= 2")
        _ensure_strictly_increasing(raw_knots, name="knots_raw_m")
        _ensure_monotonic_non_decreasing(registered_knots, name="knots_registered_m")
        raw_domain_seq = payload.get("raw_range_domain_m") or []
        if not isinstance(raw_domain_seq, Sequence) or len(raw_domain_seq) != 2:
            raise DepthRegistrationError("raw_range_domain_m must contain exactly two values")
        raw_domain = (float(raw_domain_seq[0]), float(raw_domain_seq[1]))
        if raw_domain[0] > raw_domain[1]:
            raise DepthRegistrationError("raw_range_domain_m must be ordered low->high")
        if abs(raw_domain[0] - raw_knots[0]) > 1e-6 or abs(raw_domain[1] - raw_knots[-1]) > 1e-6:
            raise DepthRegistrationError("raw_range_domain_m must match the knot endpoints")
        return cls(
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
        )


@dataclass(frozen=True, slots=True)
class DepthRegistrationBundle:
    entries: Mapping[str, DepthRegistrationEntry]

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth_registration_contract_version": CONTRACT_VERSION,
            "cameras": {camera_id: entry.to_dict() for camera_id, entry in sorted(self.entries.items())},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DepthRegistrationBundle":
        version = int(payload.get("depth_registration_contract_version") or payload.get("contract_version") or 0)
        if version != CONTRACT_VERSION:
            raise DepthRegistrationError(
                f"Unsupported depth registration contract version {version}; expected {CONTRACT_VERSION}"
            )
        cameras = payload.get("cameras")
        if not isinstance(cameras, Mapping):
            raise DepthRegistrationError("depth registration bundle missing cameras mapping")
        entries: dict[str, DepthRegistrationEntry] = {}
        for camera_id, entry_payload in cameras.items():
            if not isinstance(entry_payload, Mapping):
                raise DepthRegistrationError(f"camera {camera_id!r} registration entry must be a mapping")
            entry = DepthRegistrationEntry.from_dict(entry_payload)
            if entry.camera_id and entry.camera_id != str(camera_id):
                raise DepthRegistrationError(f"camera id mismatch in entry {camera_id!r}")
            entries[str(camera_id)] = entry
        return cls(entries=entries)


def write_depth_registration_bundle(
    *,
    path: Path,
    entries: Mapping[str, DepthRegistrationEntry],
) -> Path:
    bundle = DepthRegistrationBundle(entries=dict(entries))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_depth_registration(path: Path) -> DepthRegistrationBundle:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DepthRegistrationError(f"depth registration artifact not found: {path}") from exc
    except Exception as exc:
        raise DepthRegistrationError(f"Unable to parse depth registration artifact {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise DepthRegistrationError("depth registration artifact root must be a mapping")
    return DepthRegistrationBundle.from_dict(payload)


class DepthRegistrationManager:
    def __init__(self, bundle: DepthRegistrationBundle, *, path: Path | None = None) -> None:
        self.bundle = bundle
        self.path = path

    @classmethod
    def load(cls, path: Path) -> "DepthRegistrationManager":
        return cls(load_depth_registration(path), path=Path(path))

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
        actual_cal = _normalize_calibration_fingerprint_payload(dict(entry.calibration_fingerprint))
        if actual_cal.get("fingerprint_sha256") != expected_cal.get("fingerprint_sha256"):
            raise DepthRegistrationError("calibration_fingerprint_mismatch")
        if dict(entry.dav2_profile).get("fingerprint_sha256") != dict(dav2_profile).get("fingerprint_sha256"):
            raise DepthRegistrationError("dav2_profile_fingerprint_mismatch")
        if dict(entry.mapanything_profile).get("fingerprint_sha256") != dict(mapanything_profile).get("fingerprint_sha256"):
            raise DepthRegistrationError("mapanything_profile_fingerprint_mismatch")
        return entry

    def apply(self, *, camera_id: str, raw_depth_m: float) -> tuple[float | None, str, str | None]:
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
    finite = np.isfinite(raw) & np.isfinite(registered) & (raw > 0.0) & (registered > 0.0)
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
    "fit_piecewise_registration",
    "load_depth_registration",
    "model_profile_fingerprint",
    "profile_fingerprint",
    "write_depth_registration_bundle",
]
