from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


CONTRACT = "noesis.world_measurement_fusion_policy"
CONTRACT_VERSION = 2
MAX_POLICY_BYTES = 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class WorldFusionPolicyError(ValueError):
    """Raised when the calibrated world-measurement policy is invalid."""


def _reject_constant(value: str) -> None:
    raise WorldFusionPolicyError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WorldFusionPolicyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _require_keys(payload: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise WorldFusionPolicyError(
            f"{label} fields do not match the exact contract; missing={missing} extra={extra}"
        )


def _require_sha256(value: Any, *, label: str) -> str:
    normalized = str(value or "").strip().lower()
    if _SHA256_RE.fullmatch(normalized) is None:
        raise WorldFusionPolicyError(f"{label} must be a lowercase SHA-256 digest")
    return normalized


def _require_scale(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorldFusionPolicyError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise WorldFusionPolicyError(f"{label} must be finite and within [0, 1]")
    return result


def _require_positive_range(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorldFusionPolicyError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0 or result > 100.0:
        raise WorldFusionPolicyError(f"{label} must be finite and within (0, 100]")
    return result


@dataclass(frozen=True, slots=True)
class CameraWorldFusionProfile:
    camera_id: str
    calibration_fingerprint_sha256: str
    registration_id: str
    floor_weight_scale: float
    depth_weight_scale: float
    floor_only_allowed: bool
    floor_ray_max_range_m: float


@dataclass(frozen=True, slots=True)
class WorldFusionPolicy:
    policy_id: str
    runtime_lane: str
    evidence: Mapping[str, Any]
    cameras: Mapping[str, CameraWorldFusionProfile]

    def profile(self, camera_id: str) -> CameraWorldFusionProfile:
        try:
            return self.cameras[str(camera_id)]
        except KeyError as exc:
            raise WorldFusionPolicyError(
                f"world fusion policy has no profile for active camera {camera_id!r}"
            ) from exc


def load_world_fusion_policy(
    path: str | Path,
    *,
    runtime_lane: str,
    active_camera_ids: Sequence[str],
    depth_registration: Any,
) -> WorldFusionPolicy:
    lane = str(runtime_lane).strip().lower()
    if lane not in {"ds8", "ds9"}:
        raise WorldFusionPolicyError("runtime_lane must be ds8 or ds9")
    target = Path(path)
    try:
        size = target.stat().st_size
    except OSError as exc:
        raise WorldFusionPolicyError(f"unable to stat world fusion policy {target}: {exc}") from exc
    if size <= 0 or size > MAX_POLICY_BYTES:
        raise WorldFusionPolicyError("world fusion policy size is invalid")
    try:
        payload = json.loads(
            target.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except WorldFusionPolicyError:
        raise
    except Exception as exc:
        raise WorldFusionPolicyError(f"unable to parse world fusion policy {target}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise WorldFusionPolicyError("world fusion policy root must be an object")
    _require_keys(
        payload,
        {"contract", "contract_version", "evidence", "cameras"},
        label="world fusion policy",
    )
    if payload.get("contract") != CONTRACT or payload.get("contract_version") != CONTRACT_VERSION:
        raise WorldFusionPolicyError("world fusion policy contract identity is unsupported")

    evidence = payload.get("evidence")
    if not isinstance(evidence, Mapping):
        raise WorldFusionPolicyError("world fusion policy evidence must be an object")
    _require_keys(
        evidence,
        {
            "capture_id",
            "telemetry_sha256",
            "range_capture_id",
            "range_capture_sha256",
            "room_zones_sha256",
        },
        label="world fusion policy evidence",
    )
    if not str(evidence.get("capture_id") or "").strip():
        raise WorldFusionPolicyError("world fusion policy evidence capture_id is required")
    if not str(evidence.get("range_capture_id") or "").strip():
        raise WorldFusionPolicyError("world fusion policy evidence range_capture_id is required")
    _require_sha256(evidence.get("telemetry_sha256"), label="evidence.telemetry_sha256")
    _require_sha256(evidence.get("range_capture_sha256"), label="evidence.range_capture_sha256")
    _require_sha256(evidence.get("room_zones_sha256"), label="evidence.room_zones_sha256")

    raw_cameras = payload.get("cameras")
    if not isinstance(raw_cameras, Mapping):
        raise WorldFusionPolicyError("world fusion policy cameras must be an object")
    active = {str(camera_id) for camera_id in active_camera_ids}
    if set(raw_cameras) != active:
        raise WorldFusionPolicyError(
            "world fusion policy camera set does not match active cameras; "
            f"expected={sorted(active)} actual={sorted(raw_cameras)}"
        )
    entries = getattr(getattr(depth_registration, "bundle", None), "entries", None)
    if not isinstance(entries, Mapping):
        raise WorldFusionPolicyError("validated depth registration manager is required")

    profiles: dict[str, CameraWorldFusionProfile] = {}
    for camera_id in sorted(active):
        raw = raw_cameras.get(camera_id)
        if not isinstance(raw, Mapping):
            raise WorldFusionPolicyError(f"camera profile {camera_id!r} must be an object")
        _require_keys(
            raw,
            {
                "calibration_fingerprint_sha256",
                "registration_ids",
                "floor_weight_scale",
                "depth_weight_scale",
                "floor_only_allowed",
                "floor_ray_max_range_m",
                "evidence",
            },
            label=f"camera profile {camera_id}",
        )
        registration_ids = raw.get("registration_ids")
        if not isinstance(registration_ids, Mapping):
            raise WorldFusionPolicyError(f"camera profile {camera_id} registration_ids must be an object")
        _require_keys(registration_ids, {"ds8", "ds9"}, label=f"camera profile {camera_id} registration_ids")
        registration_id = str(registration_ids.get(lane) or "").strip()
        if not registration_id.startswith(f"{camera_id}:"):
            raise WorldFusionPolicyError(f"camera profile {camera_id} has invalid {lane} registration id")
        calibration_sha = _require_sha256(
            raw.get("calibration_fingerprint_sha256"),
            label=f"camera profile {camera_id} calibration fingerprint",
        )
        floor_scale = _require_scale(raw.get("floor_weight_scale"), label=f"{camera_id}.floor_weight_scale")
        depth_scale = _require_scale(raw.get("depth_weight_scale"), label=f"{camera_id}.depth_weight_scale")
        if floor_scale == 0.0 and depth_scale == 0.0:
            raise WorldFusionPolicyError(f"camera profile {camera_id} disables both measurements")
        floor_only_allowed = raw.get("floor_only_allowed")
        if not isinstance(floor_only_allowed, bool):
            raise WorldFusionPolicyError(f"{camera_id}.floor_only_allowed must be boolean")
        floor_ray_max_range_m = _require_positive_range(
            raw.get("floor_ray_max_range_m"),
            label=f"{camera_id}.floor_ray_max_range_m",
        )
        profile_evidence = raw.get("evidence")
        if not isinstance(profile_evidence, Mapping):
            raise WorldFusionPolicyError(f"camera profile {camera_id} evidence must be an object")
        _require_keys(
            profile_evidence,
            {
                "sample_count",
                "floor_room_containment",
                "depth_room_containment",
                "fused_room_containment",
                "final_room_containment",
            },
            label=f"camera profile {camera_id} evidence",
        )
        for name, value in profile_evidence.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise WorldFusionPolicyError(f"{camera_id}.evidence.{name} must be numeric")
            number = float(value)
            if not math.isfinite(number) or number < 0.0:
                raise WorldFusionPolicyError(f"{camera_id}.evidence.{name} is invalid")
            if name != "sample_count" and number > 1.0:
                raise WorldFusionPolicyError(f"{camera_id}.evidence.{name} must be within [0, 1]")

        entry = entries.get(camera_id)
        if entry is None:
            raise WorldFusionPolicyError(f"depth registration is missing camera {camera_id}")
        if str(getattr(entry, "registration_id", "")) != registration_id:
            raise WorldFusionPolicyError(f"world fusion registration binding mismatch for {camera_id}")
        entry_calibration = getattr(entry, "calibration_fingerprint", {})
        entry_calibration_sha = (
            str(entry_calibration.get("fingerprint_sha256") or "").strip().lower()
            if isinstance(entry_calibration, Mapping)
            else ""
        )
        if entry_calibration_sha != calibration_sha:
            raise WorldFusionPolicyError(f"world fusion calibration binding mismatch for {camera_id}")
        profiles[camera_id] = CameraWorldFusionProfile(
            camera_id=camera_id,
            calibration_fingerprint_sha256=calibration_sha,
            registration_id=registration_id,
            floor_weight_scale=floor_scale,
            depth_weight_scale=depth_scale,
            floor_only_allowed=floor_only_allowed,
            floor_ray_max_range_m=floor_ray_max_range_m,
        )

    return WorldFusionPolicy(
        policy_id=hashlib.sha256(_canonical_json(payload)).hexdigest(),
        runtime_lane=lane,
        evidence=dict(evidence),
        cameras=profiles,
    )


__all__ = [
    "CONTRACT",
    "CONTRACT_VERSION",
    "CameraWorldFusionProfile",
    "WorldFusionPolicy",
    "WorldFusionPolicyError",
    "load_world_fusion_policy",
]
