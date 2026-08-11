from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import stat
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from noesis.calibration.scene_registration import solve_scene_similarity
from noesis.calibration.depth_registration_builder import (
    DepthRegistrationBuildError,
    fit_monotonic_piecewise_mapping,
)
from noesis.calibration.depth_registration import (
    MAX_OCCUPIED_ABS_ERROR_M,
    MAX_OCCUPIED_MEDIAN_ABS_ERROR_M,
    MAX_OCCUPIED_P95_ABS_ERROR_M,
)
from noesis.validation.core import (
    CheckStatus,
    FailureType,
    SourceMetadata,
    ValidationCheck,
    ValidationReport,
)
from noesis.validation.reports import render_markdown
from noesis.validation.transforms import matrix_from_col_major, transform_point
from noesis_core.private_paths import (
    PRIVATE_FILE_MODE,
    PrivatePathError,
    atomic_write_private_file,
    ensure_private_directory,
    prepare_private_writable_file,
    validate_private_file,
)


CAPTURE_CONTRACT = "noesis.alignment.walk_capture"
SAMPLE_CONTRACT = "noesis.alignment.walk_sample"
WORLD_CONTRACT = "noesis.alignment.walk_world"
MARKER_CONTRACT = "noesis.alignment.walk_marker"
WAYPOINT_CONTRACT = "noesis.alignment.walk_waypoints"
WAYPOINT_EVIDENCE_CONTRACT = "noesis.alignment.waypoint_evidence"
WAYPOINT_METRICS_CONTRACT = "noesis.alignment.waypoint_metrics"
CONTRACT_VERSION = 1

_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}\Z")
_RELEASE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}\Z")
_TRACKLET_KEY_RE = re.compile(r"tracklet-[1-9][0-9]{0,9}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_FORBIDDEN_PERSISTED_KEYS = {
    "appearance_id",
    "embedding",
    "embedding_dimension",
    "embedding_model_sha256",
    "embedding_present",
    "embedding_sequence",
    "entity_id",
    "identity",
    "identity_kind",
    "identity_observation_key",
    "identity_state",
    "identity_v2",
    "image_bytes",
    "image_url",
    "keypoints",
    "mask",
    "observation_id",
    "reid_confidence",
    "reid_identity",
    "sid_candidate",
    "stable_id",
    "subject",
}


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    return result if math.isfinite(result) else None


def _finite_vector(value: Any, size: int) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    if len(value) < size:
        return None
    result = [_finite_float(item) for item in value[:size]]
    if any(item is None for item in result):
        return None
    return [float(item) for item in result if item is not None]


def _finite_point3(value: Any) -> list[float] | None:
    if isinstance(value, Mapping):
        return _finite_vector([value.get("x"), value.get("y"), value.get("z")], 3)
    return _finite_vector(value, 3)


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _text(value: Any, *, maximum: int = 200) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    if not result:
        return None
    return result[:maximum]


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def validate_scene_binding(payload: Mapping[str, Any] | None) -> dict[str, str] | None:
    """Normalize the immutable authored-scene binding supplied by the operator UI."""

    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError("scene_binding must be an object")
    expected_keys = {
        "release_id",
        "authored_scene_sha256",
        "world_to_scene_sha256",
    }
    extra = set(payload) - expected_keys
    missing = expected_keys - set(payload)
    if extra or missing:
        detail = []
        if missing:
            detail.append(f"missing {', '.join(sorted(missing))}")
        if extra:
            detail.append(f"unknown {', '.join(sorted(extra))}")
        raise ValueError(f"scene_binding is invalid: {'; '.join(detail)}")
    release_id = _text(payload.get("release_id"), maximum=200)
    if release_id is None or _RELEASE_ID_RE.fullmatch(release_id) is None:
        raise ValueError("scene_binding.release_id contains unsafe characters")
    result = {"release_id": release_id}
    for key in ("authored_scene_sha256", "world_to_scene_sha256"):
        value = str(payload.get(key) or "").strip().lower()
        if _SHA256_RE.fullmatch(value) is None:
            raise ValueError(f"scene_binding.{key} must be a lowercase SHA-256 digest")
        result[key] = value
    return result


def _fd_is_private_file(descriptor: int) -> bool:
    info = os.fstat(descriptor)
    return bool(
        stat.S_ISREG(info.st_mode)
        and info.st_uid == os.geteuid()
        and stat.S_IMODE(info.st_mode) == PRIVATE_FILE_MODE
        and info.st_nlink == 1
    )


class PrivateNdjsonWriter:
    """Append-only owner-private NDJSON writer with bounded durability delay."""

    def __init__(self, path: str | Path, *, fsync_interval_s: float = 1.0) -> None:
        self.path = prepare_private_writable_file(path, label="alignment evidence stream")
        flags = os.O_WRONLY | os.O_APPEND | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        self._descriptor = os.open(self.path, flags)
        if not _fd_is_private_file(self._descriptor):
            os.close(self._descriptor)
            raise PrivatePathError("alignment evidence stream changed or became unsafe")
        self._fsync_interval_s = max(0.0, float(fsync_interval_s))
        self._last_fsync = time.monotonic()
        self._closed = False

    def append(self, payload: Mapping[str, Any], *, force_fsync: bool = False) -> None:
        if self._closed:
            raise RuntimeError("alignment evidence stream is closed")
        encoded = canonical_json_bytes(payload) + b"\n"
        fcntl.flock(self._descriptor, fcntl.LOCK_EX)
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(self._descriptor, view)
                if written <= 0:
                    raise RuntimeError("alignment evidence append made no progress")
                view = view[written:]
            now = time.monotonic()
            if force_fsync or now - self._last_fsync >= self._fsync_interval_s:
                os.fsync(self._descriptor)
                self._last_fsync = now
        finally:
            fcntl.flock(self._descriptor, fcntl.LOCK_UN)

    def close(self) -> None:
        if self._closed:
            return
        try:
            os.fsync(self._descriptor)
        finally:
            os.close(self._descriptor)
            self._closed = True


def read_ndjson_private(path: str | Path) -> list[dict[str, Any]]:
    target = validate_private_file(path, label="alignment evidence stream")
    rows: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            for line_number, raw in enumerate(handle, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"invalid alignment NDJSON at {target}:{line_number}"
                    ) from exc
                if not isinstance(payload, dict):
                    raise RuntimeError(
                        f"alignment NDJSON row is not an object at {target}:{line_number}"
                    )
                rows.append(payload)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return rows


@dataclass
class EphemeralIds:
    _tracks: dict[tuple[str, int, int], str] = field(default_factory=dict)
    _entities: dict[str, str] = field(default_factory=dict)
    _observations: dict[str, str] = field(default_factory=dict)

    def track(self, camera_id: str, tracker_id: int, generation: int) -> str:
        key = (str(camera_id), int(tracker_id), int(generation))
        if key not in self._tracks:
            self._tracks[key] = f"tracklet-{len(self._tracks) + 1}"
        return self._tracks[key]

    def entity(self, entity_id: Any) -> str:
        key = str(entity_id)
        if key not in self._entities:
            self._entities[key] = f"entity-{len(self._entities) + 1}"
        return self._entities[key]

    def observation(self, observation_id: Any) -> str:
        key = str(observation_id)
        if key not in self._observations:
            self._observations[key] = (
                f"observation-{len(self._observations) + 1}"
            )
        return self._observations[key]


def sanitize_calibration_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    align = bundle.get("align") if isinstance(bundle.get("align"), Mapping) else {}
    cameras = bundle.get("cameras") if isinstance(bundle.get("cameras"), Mapping) else {}
    meta = bundle.get("meta") if isinstance(bundle.get("meta"), Mapping) else {}
    result = {
        "contract": "noesis.alignment.calibration_bundle",
        "contract_version": CONTRACT_VERSION,
        "align": _json_copy(align),
        "cameras": {
            key: _json_copy(cameras.get(key))
            for key in ("K", "E", "pose", "pose_confidence")
            if key in cameras
        },
        "meta": _json_copy(meta),
        "metric_scale": _finite_float(bundle.get("metric_scale")),
    }
    return result


def _active_similarity_from_calibration(
    calibration: Mapping[str, Any],
) -> tuple[np.ndarray, float, str]:
    """Return one internally consistent active world-to-scene transform.

    ``align.scene_similarity`` is a complete authority: its transform and
    scale must travel together.  The legacy ``align.matrix`` and
    ``align.units`` pair is used only when no scene-similarity transform is
    present.
    """

    align = (
        calibration.get("align")
        if isinstance(calibration.get("align"), Mapping)
        else {}
    )
    similarity = (
        align.get("scene_similarity")
        if isinstance(align.get("scene_similarity"), Mapping)
        else {}
    )
    similarity_matrix = similarity.get("world_to_scene_col_major")
    if isinstance(similarity_matrix, list):
        matrix = matrix_from_col_major(
            similarity_matrix, name="scene_similarity.world_to_scene_col_major"
        )
        scene_to_m = _finite_float(similarity.get("s_obj_to_m"))
        if scene_to_m is None or scene_to_m <= 0.0:
            raise RuntimeError(
                "scene_similarity requires a positive s_obj_to_m beside its transform"
            )
        singular_values = np.linalg.svd(matrix[:3, :3], compute_uv=False)
        if (
            singular_values.shape != (3,)
            or not np.all(np.isfinite(singular_values))
            or float(np.min(singular_values)) <= 1e-12
        ):
            raise RuntimeError("scene_similarity transform has an invalid scale")
        matrix_scale = float(np.mean(singular_values))
        uniform_error = float(np.max(np.abs(singular_values - matrix_scale)))
        if uniform_error > max(1e-9, matrix_scale * 1e-5):
            raise RuntimeError(
                "scene_similarity transform is not a uniform similarity"
            )
        expected_scene_to_m = 1.0 / matrix_scale
        if not math.isclose(
            scene_to_m,
            expected_scene_to_m,
            rel_tol=1e-5,
            abs_tol=1e-9,
        ):
            raise RuntimeError(
                "scene_similarity s_obj_to_m does not match its transform scale"
            )
        computed_digest = payload_sha256(
            {
                "world_to_scene_col_major": [
                    float(value) for value in matrix.flatten(order="F")
                ],
                "s_obj_to_m": float(scene_to_m),
            }
        )
        declared_digest = str(
            similarity.get("world_to_scene_sha256") or ""
        ).strip().lower()
        if declared_digest and (
            _SHA256_RE.fullmatch(declared_digest) is None
            or declared_digest != computed_digest
        ):
            raise RuntimeError(
                "scene_similarity world_to_scene_sha256 does not match its transform and scale"
            )
        return matrix, float(scene_to_m), "align.scene_similarity"

    matrix = matrix_from_col_major(
        align.get("matrix"), name="align.matrix"
    )
    units = align.get("units") if isinstance(align.get("units"), Mapping) else {}
    scene_to_m = _finite_float(units.get("s_obj_to_m"))
    if scene_to_m is None:
        scene_to_m = 1.0
    if scene_to_m <= 0.0:
        raise RuntimeError("calibration scene-to-meter scale must be positive")
    return matrix, float(scene_to_m), "align.matrix"


def _artifact_fingerprints(message: Mapping[str, Any]) -> dict[tuple[str, int, int], dict[str, str]]:
    result: dict[tuple[str, int, int], dict[str, str]] = {}
    observations = message.get("observations")
    if not isinstance(observations, list):
        return result
    for observation in observations:
        if not isinstance(observation, Mapping):
            continue
        payload = observation.get("payload") if isinstance(observation.get("payload"), Mapping) else {}
        tracklet = payload.get("tracklet") if isinstance(payload.get("tracklet"), Mapping) else {}
        camera = _text(tracklet.get("camera_id"))
        tracker = _integer(tracklet.get("tracker_id"))
        frame = _integer(tracklet.get("frame_id"))
        if camera is None or tracker is None or frame is None:
            continue
        fingerprints: dict[str, str] = {}
        observation_id = _text(observation.get("observation_id"))
        if observation_id is not None:
            fingerprints["_observation_id"] = observation_id
        for key in ("calibration", "model", "config"):
            artifact = observation.get(key)
            sha = _text(artifact.get("sha256"), maximum=64) if isinstance(artifact, Mapping) else None
            if sha and len(sha) == 64:
                fingerprints[f"{key}_sha256"] = sha
        result[(camera, tracker, frame)] = fingerprints
    return result


_POINT_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "floor_world_raw_m": (
        "world_floor_candidate",
        "floor_world_raw_m",
        "floor_world_raw",
        "world_floor_raw",
    ),
    "depth_world_raw_m": (
        "world_depth_candidate",
        "depth_world_raw_m",
        "depth_world_raw",
        "world_depth_raw",
    ),
    "world_measurement_raw_m": (
        "world_prefilter_measurement",
        "world_measurement_raw_m",
        "world_measurement_raw",
    ),
    "world_prediction_m": (
        "world_filter_prediction",
        "world_prediction_m",
        "world_prediction",
    ),
}


def sanitize_tracking_message(
    message: Mapping[str, Any],
    *,
    ids: EphemeralIds,
    calibration_bundle_sha256: str | None,
    received_at_us: int,
    received_monotonic_ns: int,
    first_sequence: int,
) -> list[dict[str, Any]]:
    tracks = message.get("tracks")
    if not isinstance(tracks, list):
        return []
    top_camera = _text(message.get("camera_id"))
    source_id = _integer(message.get("source_id"))
    frame_id_top = _integer(message.get("frame_id"))
    observation_fingerprints = _artifact_fingerprints(message)
    rows: list[dict[str, Any]] = []
    for track_index, track in enumerate(tracks):
        if not isinstance(track, Mapping):
            continue
        camera_id = _text(track.get("camera_id")) or top_camera
        tracker_id = _integer(track.get("tracker_id", track.get("track_id")))
        frame_id = _integer(track.get("frame_id"))
        if frame_id is None:
            frame_id = frame_id_top
        generation = _integer(track.get("tracker_lifecycle_generation")) or 0
        if camera_id is None or tracker_id is None or frame_id is None:
            continue
        row: dict[str, Any] = {
            "contract": SAMPLE_CONTRACT,
            "contract_version": CONTRACT_VERSION,
            "sequence": int(first_sequence + len(rows)),
            "received_at_us": int(received_at_us),
            "received_monotonic_ns": int(received_monotonic_ns),
            "source_id": source_id,
            "camera_id": camera_id,
            "frame_id": frame_id,
            "tracking_publication_sequence": _integer(message.get("tracking_publication_sequence")),
            "tracklet_key": ids.track(camera_id, tracker_id, generation),
            "captured_at_us": _integer(track.get("captured_at_us", message.get("captured_at_us"))),
            "observed_at_us": _integer(track.get("observed_at_us", message.get("observed_at_us"))),
            "media_pts_ns": _integer(track.get("media_pts_ns", message.get("media_pts_ns"))),
            "capture_time_status": _text(track.get("capture_time_status", message.get("capture_time_status"))),
            "bbox_xywh": _finite_vector(track.get("bbox"), 4),
            "image_size": _finite_vector(track.get("image_size", message.get("image_size")), 2),
            "image_foot": _finite_vector(track.get("image_foot"), 2),
            "image_base": _finite_vector(track.get("image_base"), 2),
            "confidence": _finite_float(track.get("confidence")),
            "tracker_confidence": _finite_float(track.get("tracker_confidence")),
            "zone": _text(track.get("zone")),
            "zone_source": _text(track.get("zone_source")),
            "zone_authoritative": (
                bool(track.get("zone_authoritative"))
                if isinstance(track.get("zone_authoritative"), bool)
                else None
            ),
            "depth_status": _text(track.get("depth_status")),
            "depth_anchor_source": _text(track.get("depth_anchor_source")),
            "depth_anchor_m": _finite_float(track.get("depth_anchor_m")),
            "depth_registered_m": _finite_float(track.get("depth_registered_m")),
            "depth_used_m": _finite_float(track.get("depth_used_m")),
            "depth_registration_status": _text(track.get("depth_registration_status")),
            "depth_registration_id": _text(track.get("depth_registration_id")),
            "depth_sample_count": _integer(track.get("depth_sample_count")),
            "depth_valid_fraction": _finite_float(track.get("depth_valid_fraction")),
            "world": _finite_vector(track.get("world"), 3),
            "world_valid": track.get("world_valid") is True,
            "world_frame": _text(track.get("world_frame")),
            "world_source": _text(track.get("world_source")),
            "world_quality": _text(track.get("world_quality")),
            "world_quality_reason": _text(track.get("world_quality_reason")),
            "world_floor_range_m": _finite_float(track.get("world_floor_range_m")),
            "world_floor_range_limit_m": _finite_float(track.get("world_floor_range_limit_m")),
            "world_floor_incidence_sin": _finite_float(track.get("world_floor_incidence_sin")),
            "world_floor_admitted": (
                bool(track.get("world_floor_admitted"))
                if isinstance(track.get("world_floor_admitted"), bool)
                else None
            ),
            "world_floor_rejection_reason": _text(track.get("world_floor_rejection_reason")),
            "motion_mode": _text(track.get("motion_mode")),
            "posture": _text(track.get("posture")),
            "world_innovation_m": _finite_float(track.get("world_innovation_m")),
            "world_measurement_accepted": (
                bool(track.get("world_measurement_accepted"))
                if isinstance(track.get("world_measurement_accepted"), bool)
                else None
            ),
            "world_rejection_reason": _text(track.get("world_rejection_reason")),
            "world_innovation_limit_m": _finite_float(track.get("world_innovation_limit_m")),
            "world_reacquire_count": _integer(track.get("world_reacquire_count")),
            "world_reacquired": (
                bool(track.get("world_reacquired"))
                if isinstance(track.get("world_reacquired"), bool)
                else None
            ),
            "trail_break_required": (
                bool(track.get("trail_break_required"))
                if isinstance(track.get("trail_break_required"), bool)
                else None
            ),
            "trail_append_allowed": (
                bool(track.get("trail_append_allowed"))
                if isinstance(track.get("trail_append_allowed"), bool)
                else None
            ),
            "idle_jitter_m": _finite_float(track.get("idle_jitter_m")),
            "source_switch_count": _integer(track.get("source_switch_count")),
            "sticky_world_source": _text(track.get("sticky_world_source")),
            "calibration_bundle_sha256": calibration_bundle_sha256,
        }
        for output_key, aliases in _POINT_FIELD_ALIASES.items():
            for input_key in aliases:
                point = _finite_vector(track.get(input_key), 3)
                if point is not None:
                    row[output_key] = point
                    break
        observation_metadata = dict(
            observation_fingerprints.get((camera_id, tracker_id, frame_id), {})
        )
        observation_id = observation_metadata.pop("_observation_id", None)
        if observation_id is not None:
            row["observation_key"] = ids.observation(observation_id)
        row.update(observation_metadata)
        rows.append({key: value for key, value in row.items() if value is not None})
    return rows


def sanitize_world_snapshot(
    snapshot: Mapping[str, Any],
    *,
    ids: EphemeralIds,
    received_at_us: int,
    received_monotonic_ns: int,
    sequence: int,
) -> dict[str, Any] | None:
    if snapshot.get("contract") != "noesis.world.snapshot":
        return None
    entities_out: list[dict[str, Any]] = []
    entities = snapshot.get("entities")
    if not isinstance(entities, list):
        entities = []
    for entity in entities:
        if not isinstance(entity, Mapping):
            continue
        raw_entity = entity.get("entity_id")
        if raw_entity is None:
            continue
        sources_out: list[dict[str, Any]] = []
        sources = entity.get("sources")
        if isinstance(sources, list):
            for source in sources:
                if not isinstance(source, Mapping):
                    continue
                sources_out.append(
                    {
                        key: value
                        for key, value in {
                            "observation_key": (
                                ids.observation(source.get("observation_id"))
                                if _text(source.get("observation_id")) is not None
                                else None
                            ),
                            "camera_id": _text(source.get("camera_id")),
                            "zone": _text(source.get("zone")),
                            "zone_source": _text(source.get("zone_source")),
                            "zone_authoritative": (
                                bool(source.get("zone_authoritative"))
                                if isinstance(source.get("zone_authoritative"), bool)
                                else None
                            ),
                            "observed_at_us": _integer(source.get("observed_at_us")),
                            "position": _finite_point3(source.get("position")),
                            "accepted": (
                                bool(source.get("accepted"))
                                if isinstance(source.get("accepted"), bool)
                                else None
                            ),
                            "rejection_reason": _text(source.get("rejection_reason")),
                        }.items()
                        if value is not None
                    }
                )
        entities_out.append(
            {
                key: value
                for key, value in {
                    "entity_key": ids.entity(raw_entity),
                    "lifecycle": _text(entity.get("lifecycle")),
                    "position": _finite_point3(entity.get("position")),
                    "covariance": _finite_vector(entity.get("covariance", {}).get("values") if isinstance(entity.get("covariance"), Mapping) else entity.get("covariance"), 9),
                    "velocity_mps": _finite_point3(entity.get("velocity_mps")),
                    "room_id": _text(entity.get("room_id")),
                    "observed_at_us": _integer(entity.get("observed_at_us")),
                    "stale_after_us": _integer(entity.get("stale_after_us")),
                    "conflict": bool(entity.get("conflict")),
                    "conflict_reason": _text(entity.get("conflict_reason")),
                    "sources": sources_out,
                }.items()
                if value is not None
            }
        )
    return {
        "contract": WORLD_CONTRACT,
        "contract_version": CONTRACT_VERSION,
        "sequence": int(sequence),
        "received_at_us": int(received_at_us),
        "received_monotonic_ns": int(received_monotonic_ns),
        "world_sequence": _integer(snapshot.get("sequence")),
        "observed_start_us": _integer(snapshot.get("observed_start_us")),
        "observed_end_us": _integer(snapshot.get("observed_end_us")),
        "published_at_us": _integer(snapshot.get("published_at_us")),
        "frame": _text(snapshot.get("frame")),
        "units": _text(snapshot.get("units")),
        "entities": entities_out,
    }


def validate_waypoint_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("contract") not in {None, WAYPOINT_CONTRACT}:
        raise ValueError("waypoint manifest contract is invalid")
    if payload.get("contract_version") not in {None, CONTRACT_VERSION}:
        raise ValueError("waypoint manifest version is invalid")
    raw_waypoints = payload.get("waypoints")
    if not isinstance(raw_waypoints, list) or not raw_waypoints:
        raise ValueError("waypoint manifest requires a nonempty waypoints list")
    seen: set[str] = set()
    waypoints: list[dict[str, Any]] = []
    for raw in raw_waypoints:
        if not isinstance(raw, Mapping):
            raise ValueError("waypoint entries must be objects")
        waypoint_id = _text(raw.get("id"), maximum=120)
        camera_id = _text(raw.get("camera_id"), maximum=160)
        if not waypoint_id or not camera_id or waypoint_id in seen:
            raise ValueError("waypoint ids must be unique and camera_id is required")
        seen.add(waypoint_id)
        expected = _finite_vector(raw.get("expected_scene_xz"), 2)
        expected_xyz = _finite_vector(raw.get("expected_scene_xyz"), 3)
        split = _text(raw.get("split"), maximum=20)
        if split is not None:
            split = split.lower()
        if (expected_xyz is None) != (split is None):
            raise ValueError(
                "guided calibration waypoints require both expected_scene_xyz and split"
            )
        if split is not None and split not in {"fit", "holdout"}:
            raise ValueError("guided calibration waypoint split must be fit or holdout")
        rooms_raw = raw.get("rooms")
        rooms = [str(value).strip()[:160] for value in rooms_raw] if isinstance(rooms_raw, list) else []
        rooms = [value for value in rooms if value]
        waypoints.append(
            {
                key: value
                for key, value in {
                    "id": waypoint_id,
                    "camera_id": camera_id,
                    "label": _text(raw.get("label"), maximum=240),
                    "rooms": rooms,
                    "expected_scene_xz": expected,
                    "expected_scene_xyz": expected_xyz,
                    "split": split,
                    "pause_s": _finite_float(raw.get("pause_s")),
                }.items()
                if value not in (None, [])
            }
        )
    return {
        "contract": WAYPOINT_CONTRACT,
        "contract_version": CONTRACT_VERSION,
        "waypoints": waypoints,
    }


def default_output_root() -> Path:
    explicit = str(os.environ.get("NOESIS_ALIGNMENT_WALK_ROOT") or "").strip()
    if explicit:
        return Path(explicit)
    audit_dir = str(os.environ.get("NOESIS_CALIBRATION_AUDIT_DIR") or "").strip()
    if audit_dir:
        return Path(audit_dir).parent / "alignment-walk"
    state_root = Path(os.environ.get("XDG_STATE_HOME") or (Path.home() / ".local" / "state"))
    return state_root / "noesis" / "alignment-walk"


def _safe_endpoint_uri(value: str) -> str:
    parsed = urlsplit(str(value))
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    port = f":{parsed.port}" if parsed.port is not None else ""
    return urlunsplit((parsed.scheme, f"{hostname}{port}", parsed.path, "", ""))


def create_run_directory(output_root: str | Path, run_id: str) -> Path:
    if _RUN_ID_RE.fullmatch(str(run_id)) is None:
        raise ValueError("run_id contains unsafe characters")
    root = ensure_private_directory(output_root, label="alignment evidence root")
    run_dir = root / str(run_id)
    if run_dir.exists() or run_dir.is_symlink():
        raise FileExistsError(f"alignment evidence run already exists: {run_dir}")
    return ensure_private_directory(run_dir, label="alignment evidence run")


@dataclass
class AlignmentWalkCapture:
    run_dir: Path
    run_id: str
    ws_uri: str
    scene_binding: Mapping[str, Any] | None = None
    ids: EphemeralIds = field(default_factory=EphemeralIds)
    sample_sequence: int = 0
    world_sequence: int = 0
    calibration_bundle_sha256: str | None = None
    calibration_digests: list[str] = field(default_factory=list)
    runtime_identity: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    scene_binding_verification: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.scene_binding = validate_scene_binding(self.scene_binding)
        if self.scene_binding is not None:
            self.scene_binding_verification = {"status": "pending"}
        self.run_dir = ensure_private_directory(self.run_dir, label="alignment evidence run")
        self.calibration_dir = ensure_private_directory(self.run_dir / "calibration", label="alignment calibration evidence")
        self.samples = PrivateNdjsonWriter(self.run_dir / "samples.ndjson")
        self.world = PrivateNdjsonWriter(self.run_dir / "world.ndjson")
        prepare_private_writable_file(self.run_dir / "markers.ndjson", label="alignment waypoint markers")
        self.started_at_us = int(time.time_ns() // 1_000)
        self.started_monotonic_ns = int(time.monotonic_ns())
        self._write_session("recording")

    def _write_session(self, status: str, *, error: str | None = None) -> None:
        payload = {
            "contract": CAPTURE_CONTRACT,
            "contract_version": CONTRACT_VERSION,
            "run_id": self.run_id,
            "status": str(status),
            "ws_uri": _safe_endpoint_uri(self.ws_uri),
            "started_at_us": self.started_at_us,
            "started_monotonic_ns": self.started_monotonic_ns,
            "updated_at_us": int(time.time_ns() // 1_000),
            "runtime": dict(self.runtime_identity),
            "scene_binding": _json_copy(self.scene_binding),
            "scene_binding_verification": _json_copy(
                self.scene_binding_verification
            ),
            "calibration_bundle_sha256": self.calibration_bundle_sha256,
            "calibration_bundle_digests": list(self.calibration_digests),
            "counts": dict(sorted(self.counts.items())),
            "privacy": {
                "images_persisted": False,
                "embeddings_persisted": False,
                "identity_labels_persisted": False,
                "track_and_entity_ids": "run_local_ephemeral",
            },
        }
        if status != "recording":
            payload["finished_at_us"] = int(time.time_ns() // 1_000)
            payload["finished_monotonic_ns"] = int(time.monotonic_ns())
        if error:
            payload["error"] = str(error)[:500]
        atomic_write_private_file(
            self.run_dir / "session.json",
            json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n",
            label="alignment capture session",
        )

    def copy_waypoints(self, payload: Mapping[str, Any]) -> None:
        normalized = validate_waypoint_manifest(payload)
        atomic_write_private_file(
            self.run_dir / "waypoints.json",
            json.dumps(normalized, indent=2, sort_keys=True).encode("utf-8") + b"\n",
            label="alignment waypoint manifest",
        )

    def process_message(
        self,
        message: Mapping[str, Any],
        *,
        received_at_us: int | None = None,
        received_monotonic_ns: int | None = None,
    ) -> list[dict[str, Any]]:
        wall_us = int(received_at_us or (time.time_ns() // 1_000))
        monotonic_ns = int(received_monotonic_ns or time.monotonic_ns())
        message_type = str(message.get("type") or "")
        if message_type == "calibration-bundle":
            bundle = message.get("data")
            if not isinstance(bundle, Mapping):
                return []
            sanitized = sanitize_calibration_bundle(bundle)
            matrix, scene_to_m, similarity_source = (
                _active_similarity_from_calibration(sanitized)
            )
            active_similarity = _similarity_evidence(
                matrix,
                scene_to_m,
                source="captured_active_similarity",
            )
            if self.scene_binding is not None:
                requested_world_digest = self.scene_binding[
                    "world_to_scene_sha256"
                ]
                captured_world_digest = str(active_similarity["sha256"])
                verification: dict[str, Any] = {
                    "status": "partial",
                    "similarity_source": similarity_source,
                    "captured_world_to_scene_sha256": captured_world_digest,
                    "verified_fields": ["world_to_scene_sha256"],
                    "operator_asserted_fields": [
                        "release_id",
                        "authored_scene_sha256",
                    ],
                }
                meta = (
                    sanitized.get("meta")
                    if isinstance(sanitized.get("meta"), Mapping)
                    else {}
                )
                runtime_release_id = _text(
                    meta.get("release_id", meta.get("scene_release_id")),
                    maximum=200,
                )
                runtime_authored_digest = str(
                    meta.get("authored_scene_sha256") or ""
                ).strip().lower()
                if runtime_release_id is not None:
                    verification["captured_release_id"] = runtime_release_id
                    verification["verified_fields"].append("release_id")
                    verification["operator_asserted_fields"].remove(
                        "release_id"
                    )
                if _SHA256_RE.fullmatch(runtime_authored_digest):
                    verification[
                        "captured_authored_scene_sha256"
                    ] = runtime_authored_digest
                    verification["verified_fields"].append(
                        "authored_scene_sha256"
                    )
                    verification["operator_asserted_fields"].remove(
                        "authored_scene_sha256"
                    )
                mismatches: list[str] = []
                if requested_world_digest != captured_world_digest:
                    mismatches.append("world_to_scene_sha256")
                if (
                    runtime_release_id is not None
                    and runtime_release_id != self.scene_binding["release_id"]
                ):
                    mismatches.append("release_id")
                if (
                    _SHA256_RE.fullmatch(runtime_authored_digest)
                    and runtime_authored_digest
                    != self.scene_binding["authored_scene_sha256"]
                ):
                    mismatches.append("authored_scene_sha256")
                if mismatches:
                    verification["status"] = "mismatch"
                    verification["mismatched_fields"] = mismatches
                    self.counts["scene_binding_mismatches"] += 1
                    self.scene_binding_verification = verification
                    raise RuntimeError(
                        "scene binding does not match the active calibration: "
                        + ", ".join(mismatches)
                    )
                if not verification["operator_asserted_fields"]:
                    verification["status"] = "matched"
                self.scene_binding_verification = verification
            digest = payload_sha256(sanitized)
            if digest not in self.calibration_digests:
                atomic_write_private_file(
                    self.calibration_dir / f"{digest}.json",
                    json.dumps(sanitized, indent=2, sort_keys=True).encode("utf-8") + b"\n",
                    label="alignment calibration bundle",
                )
                self.calibration_digests.append(digest)
            self.calibration_bundle_sha256 = digest
            self.counts["calibration_bundles"] += 1
            return []
        if message_type != "tracking":
            return []
        if not self.calibration_bundle_sha256:
            self.counts["tracking_messages_before_calibration"] += 1
            return []
        rows = sanitize_tracking_message(
            message,
            ids=self.ids,
            calibration_bundle_sha256=self.calibration_bundle_sha256,
            received_at_us=wall_us,
            received_monotonic_ns=monotonic_ns,
            first_sequence=self.sample_sequence,
        )
        for row in rows:
            self.samples.append(row)
            self.sample_sequence += 1
        self.counts["tracking_messages"] += 1
        self.counts["track_samples"] += len(rows)
        snapshot = message.get("world_snapshot")
        if isinstance(snapshot, Mapping):
            sanitized_world = sanitize_world_snapshot(
                snapshot,
                ids=self.ids,
                received_at_us=wall_us,
                received_monotonic_ns=monotonic_ns,
                sequence=self.world_sequence,
            )
            if sanitized_world is not None:
                self.world.append(sanitized_world)
                self.world_sequence += 1
                self.counts["world_snapshots"] += 1
                producer = snapshot.get("producer")
                if isinstance(producer, Mapping) and not self.runtime_identity:
                    self.runtime_identity = {
                        key: value
                        for key, value in {
                            "runtime": _text(producer.get("runtime")),
                            "instance_id": _text(producer.get("instance_id")),
                            "run_id": _text(producer.get("run_id")),
                        }.items()
                        if value is not None
                    }
        return rows

    def finish(self, *, status: str = "complete", error: str | None = None) -> None:
        self.samples.close()
        self.world.close()
        final_status = str(status)
        if final_status == "complete" and not self.calibration_bundle_sha256:
            final_status = "failed"
            error = error or "calibration bundle was not observed"
        if final_status == "complete" and len(self.calibration_digests) != 1:
            final_status = "failed"
            error = error or "calibration bundle changed during capture"
        if final_status == "complete" and int(self.counts.get("track_samples", 0)) <= 0:
            final_status = "failed"
            error = error or "no calibrated tracking samples were observed"
        self._write_session(final_status, error=error)
        write_artifact_index(self.run_dir)


def _load_json_file(path: Path, *, private: bool) -> dict[str, Any]:
    if private:
        validate_private_file(path, label="alignment JSON evidence")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def append_waypoint_marker(
    run_dir: str | Path,
    *,
    waypoint_id: str,
    phase: str = "arrived",
    actor: str | None = None,
    tracklet_key: str | None = None,
    recorded_at_us: int | None = None,
    monotonic_ns: int | None = None,
) -> dict[str, Any]:
    root = ensure_private_directory(run_dir, label="alignment evidence run")
    session = _load_json_file(root / "session.json", private=True)
    if session.get("status") != "recording":
        raise RuntimeError("waypoint markers may only be added while capture is recording")
    waypoint_value = _text(waypoint_id, maximum=120)
    if not waypoint_value:
        raise ValueError("waypoint_id is required")
    phase_value = str(phase).strip().lower()
    if phase_value not in {"arrived", "leave"}:
        raise ValueError("marker phase must be arrived or leave")
    waypoint_path = root / "waypoints.json"
    waypoint_camera: str | None = None
    if waypoint_path.exists():
        manifest = _load_json_file(waypoint_path, private=True)
        declared = {
            str(item.get("id")): item
            for item in manifest.get("waypoints", [])
            if isinstance(item, Mapping)
        }
        if waypoint_value not in declared:
            raise ValueError(f"unknown waypoint: {waypoint_value}")
        waypoint_camera = _text(declared[waypoint_value].get("camera_id"), maximum=160)
    selected_tracklet = _text(tracklet_key, maximum=120)
    if selected_tracklet is not None:
        if _TRACKLET_KEY_RE.fullmatch(selected_tracklet) is None:
            raise ValueError("tracklet_key must be a run-local ephemeral tracklet")
        matching_samples = [
            row
            for row in read_ndjson_private(root / "samples.ndjson")
            if row.get("tracklet_key") == selected_tracklet
            and (
                waypoint_camera is None
                or row.get("camera_id") == waypoint_camera
            )
        ]
        if not matching_samples:
            raise ValueError(
                "tracklet_key is not present for the waypoint camera in this run"
            )
    marker_path = prepare_private_writable_file(root / "markers.ndjson", label="alignment waypoint markers")
    flags = os.O_WRONLY | os.O_APPEND | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(marker_path, flags)
    try:
        if not _fd_is_private_file(descriptor):
            raise PrivatePathError("alignment waypoint marker file changed or became unsafe")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            marker_monotonic_ns = int(monotonic_ns or time.monotonic_ns())
            marker = {
                "contract": MARKER_CONTRACT,
                "contract_version": CONTRACT_VERSION,
                "sequence": marker_monotonic_ns,
                "recorded_at_us": int(recorded_at_us or (time.time_ns() // 1_000)),
                "monotonic_ns": marker_monotonic_ns,
                "waypoint_id": waypoint_value,
                "phase": phase_value,
            }
            actor_value = _text(actor, maximum=80)
            if actor_value:
                marker["actor"] = actor_value
            if selected_tracklet is not None:
                marker["tracklet_key"] = selected_tracklet
            encoded = canonical_json_bytes(marker) + b"\n"
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise RuntimeError("waypoint marker append made no progress")
                view = view[written:]
            os.fsync(descriptor)
            return marker
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _artifact_paths(run_dir: Path) -> list[Path]:
    paths = [run_dir / "session.json", run_dir / "samples.ndjson", run_dir / "world.ndjson", run_dir / "markers.ndjson"]
    if (run_dir / "waypoints.json").is_file():
        paths.append(run_dir / "waypoints.json")
    calibration_dir = run_dir / "calibration"
    if calibration_dir.is_dir():
        paths.extend(sorted(calibration_dir.glob("*.json")))
    return paths


def write_artifact_index(run_dir: str | Path) -> Path:
    root = ensure_private_directory(run_dir, label="alignment evidence run")
    artifacts = {}
    for path in _artifact_paths(root):
        validate_private_file(path, label="alignment evidence artifact")
        artifacts[str(path.relative_to(root))] = {
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    payload = {
        "contract": "noesis.alignment.walk_artifact_index",
        "contract_version": CONTRACT_VERSION,
        "artifacts": artifacts,
    }
    return atomic_write_private_file(
        root / "artifact_index.json",
        json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        label="alignment artifact index",
    )


def _contains_forbidden_key(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key) in _FORBIDDEN_PERSISTED_KEYS:
                return str(key)
            found = _contains_forbidden_key(nested)
            if found:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _contains_forbidden_key(nested)
            if found:
                return found
    return None


def verify_run(run_dir: str | Path, *, require_complete: bool = True) -> dict[str, Any]:
    root = ensure_private_directory(run_dir, label="alignment evidence run")
    errors: list[str] = []
    try:
        for path in root.rglob("*"):
            info = path.lstat()
            if stat.S_ISDIR(info.st_mode):
                if stat.S_IMODE(info.st_mode) != 0o700 or info.st_uid != os.geteuid():
                    errors.append(f"unsafe directory: {path.relative_to(root)}")
            elif stat.S_ISREG(info.st_mode):
                try:
                    validate_private_file(path, label="alignment evidence artifact")
                except Exception as exc:
                    errors.append(str(exc))
            else:
                errors.append(f"unsupported artifact type: {path.relative_to(root)}")
    except Exception as exc:
        errors.append(str(exc))

    try:
        session = _load_json_file(root / "session.json", private=True)
        if session.get("contract") != CAPTURE_CONTRACT:
            errors.append("session contract is invalid")
        if require_complete and session.get("status") != "complete":
            errors.append(f"capture status is {session.get('status')!r}, not complete")
    except Exception as exc:
        session = {}
        errors.append(str(exc))

    streams: dict[str, list[dict[str, Any]]] = {}
    expected_contracts = {
        "samples.ndjson": SAMPLE_CONTRACT,
        "world.ndjson": WORLD_CONTRACT,
        "markers.ndjson": MARKER_CONTRACT,
    }
    for name, contract in expected_contracts.items():
        try:
            rows = read_ndjson_private(root / name)
            streams[name] = rows
            sequences = []
            for row in rows:
                if row.get("contract") != contract:
                    errors.append(f"{name} contains an invalid contract")
                forbidden = _contains_forbidden_key(row)
                if forbidden:
                    errors.append(f"{name} persists forbidden key {forbidden}")
                sequence = _integer(row.get("sequence"))
                if sequence is not None:
                    sequences.append(sequence)
            if sequences != sorted(sequences) or len(sequences) != len(set(sequences)):
                errors.append(f"{name} sequences are not strictly ordered")
            if name != "markers.ndjson" and sequences and sequences != list(range(len(sequences))):
                errors.append(f"{name} sequences are not contiguous from zero")
        except Exception as exc:
            errors.append(str(exc))

    samples = streams.get("samples.ndjson", [])
    if not samples:
        errors.append("capture contains no calibrated tracking samples")
    elif not any(
        _finite_vector(row.get("image_foot"), 2) is not None
        and (
            (image_size := _finite_vector(row.get("image_size"), 2)) is not None
            and image_size[0] > 0.0
            and image_size[1] > 0.0
        )
        for row in samples
    ):
        errors.append("capture contains no usable image-foot tracking samples")
    session_digest = _text(session.get("calibration_bundle_sha256"), maximum=64)
    if session_digest and any(
        row.get("calibration_bundle_sha256") != session_digest for row in samples
    ):
        errors.append("tracking samples are not bound to the sole session calibration bundle")

    try:
        index = _load_json_file(root / "artifact_index.json", private=True)
        artifacts = index.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise RuntimeError("artifact index is invalid")
        for relative, metadata in artifacts.items():
            path = root / str(relative)
            expected = metadata.get("sha256") if isinstance(metadata, Mapping) else None
            if not isinstance(expected, str) or file_sha256(path) != expected:
                errors.append(f"artifact digest mismatch: {relative}")
    except Exception as exc:
        errors.append(str(exc))

    return {
        "ok": not errors,
        "run_dir": str(root),
        "status": session.get("status"),
        "errors": errors,
        "counts": {name: len(rows) for name, rows in streams.items()},
    }


def _normalize_room_name(value: str) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _load_room_zones(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError("room zones must be a JSON list")
    zones: list[dict[str, Any]] = []
    for raw in payload:
        if not isinstance(raw, Mapping):
            continue
        bounds = raw.get("bounds")
        name = _text(raw.get("name"))
        if not isinstance(bounds, Mapping) or not name:
            continue
        values = {key: _finite_float(bounds.get(key)) for key in ("minX", "maxX", "minZ", "maxZ")}
        if any(value is None for value in values.values()):
            continue
        zones.append({"name": name, "key": _normalize_room_name(name), "bounds": values})
    return zones


def _target_zones(camera_id: str, declared_rooms: Sequence[str], zones: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    keys = {_normalize_room_name(room) for room in declared_rooms if str(room).strip()}
    camera_key = _normalize_room_name(camera_id)
    if not keys:
        keys.add(camera_key)
    matches = []
    for zone in zones:
        zone_key = str(zone.get("key") or "")
        if zone_key in keys or zone_key == camera_key:
            matches.append(zone)
        elif camera_key == "kitchen" and zone_key.startswith("kitchen"):
            matches.append(zone)
    return matches


def _point_zone_distance_scene(point_xz: Sequence[float], zones: Sequence[Mapping[str, Any]]) -> float:
    x, z = float(point_xz[0]), float(point_xz[1])
    distances = []
    for zone in zones:
        bounds = zone.get("bounds") if isinstance(zone.get("bounds"), Mapping) else {}
        min_x, max_x = float(bounds["minX"]), float(bounds["maxX"])
        min_z, max_z = float(bounds["minZ"]), float(bounds["maxZ"])
        dx = min_x - x if x < min_x else x - max_x if x > max_x else 0.0
        dz = min_z - z if z < min_z else z - max_z if z > max_z else 0.0
        distances.append(math.hypot(dx, dz))
    return min(distances) if distances else math.inf


def _calibration_for_session(run_dir: Path, session: Mapping[str, Any]) -> tuple[dict[str, Any], np.ndarray, float]:
    digest = _text(session.get("calibration_bundle_sha256"), maximum=64)
    if not digest:
        raise RuntimeError("session has no calibration bundle digest")
    calibration = _load_json_file(run_dir / "calibration" / f"{digest}.json", private=True)
    matrix, scene_to_m, _source = _active_similarity_from_calibration(calibration)
    return calibration, matrix, scene_to_m


def _track_audit(samples: Sequence[Mapping[str, Any]], matrix: np.ndarray) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in samples:
        key = _text(row.get("tracklet_key"))
        if key:
            grouped[key].append(row)
    audit: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row.get("received_monotonic_ns") or 0))
        scene_points = []
        for row in rows:
            point = _finite_vector(row.get("world"), 3)
            if point is not None and row.get("world_valid") is True:
                scene_points.append(transform_point(point, matrix))
        if scene_points:
            array = np.asarray(scene_points, dtype=np.float64)
            scene_bounds = {
                "min_xz": [float(np.min(array[:, 0])), float(np.min(array[:, 2]))],
                "max_xz": [float(np.max(array[:, 0])), float(np.max(array[:, 2]))],
            }
        else:
            scene_bounds = None
        audit.append(
            {
                "tracklet_key": key,
                "camera_id": rows[0].get("camera_id"),
                "sample_count": len(rows),
                "first_received_monotonic_ns": rows[0].get("received_monotonic_ns"),
                "last_received_monotonic_ns": rows[-1].get("received_monotonic_ns"),
                "world_valid_count": len(scene_points),
                "depth_registered_count": sum(row.get("depth_registered_m") is not None for row in rows),
                "scene_bounds": scene_bounds,
            }
        )
    return audit


def assign_waypoints(
    samples: Sequence[Mapping[str, Any]],
    markers: Sequence[Mapping[str, Any]],
    waypoints: Mapping[str, Any],
    *,
    world_to_scene: np.ndarray,
    scene_to_m: float,
    window_before_s: float = 2.0,
    window_after_s: float = 3.0,
    ambiguity_margin: float = 0.12,
) -> list[dict[str, Any]]:
    """Bind arrivals to run-local tracklets without trusting current position.

    An explicit ``tracklet_key`` on the marker is authoritative for the
    assignment. Automatic assignment remains available for older captures, but
    it ranks arrival/hold behavior only; the current world position and its
    error to the waypoint are retained strictly as diagnostics. This avoids
    using the calibration under test as ground truth for actor selection.
    """

    waypoint_by_id = {
        str(item.get("id")): item
        for item in waypoints.get("waypoints", [])
        if isinstance(item, Mapping)
    }
    result: list[dict[str, Any]] = []
    actor_previous: dict[tuple[str, str], str] = {}
    arrived_waypoints: set[str] = set()
    for marker in sorted(markers, key=lambda item: int(item.get("monotonic_ns") or 0)):
        if marker.get("phase") != "arrived":
            continue
        waypoint_id = str(marker.get("waypoint_id") or "")
        waypoint = waypoint_by_id.get(waypoint_id)
        if waypoint is None:
            result.append({"waypoint_id": waypoint_id, "status": "unknown_waypoint", "candidates": []})
            continue
        arrived_waypoints.add(waypoint_id)
        camera_id = str(waypoint.get("camera_id") or "")
        marker_ns = int(marker.get("monotonic_ns") or 0)
        before_ns = marker_ns - int(float(window_before_s) * 1_000_000_000)
        after_ns = marker_ns + int(float(window_after_s) * 1_000_000_000)
        grouped_all: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in samples:
            if row.get("camera_id") != camera_id:
                continue
            received = _integer(row.get("received_monotonic_ns"))
            key = _text(row.get("tracklet_key"))
            if received is None or key is None or not (before_ns <= received <= after_ns):
                continue
            grouped_all[key].append(row)
        expected = _finite_vector(waypoint.get("expected_scene_xz"), 2)
        actor = str(marker.get("actor") or "default")
        previous_key = actor_previous.get((actor, camera_id))
        explicit_tracklet = _text(marker.get("tracklet_key"), maximum=120)
        candidates: list[dict[str, Any]] = []
        for key, rows in grouped_all.items():
            if explicit_tracklet is not None and key != explicit_tracklet:
                continue
            rows.sort(key=lambda row: int(row.get("received_monotonic_ns") or 0))
            if not rows or (explicit_tracklet is None and len(rows) < 2):
                continue
            before_all = [
                row
                for row in rows
                if int(row.get("received_monotonic_ns") or 0) <= marker_ns
            ]
            after_all = [
                row
                for row in rows
                if int(row.get("received_monotonic_ns") or 0) >= marker_ns
            ]
            hold_all = after_all or rows

            def image_motion(
                motion_rows: Sequence[Mapping[str, Any]],
                *,
                spread: bool,
            ) -> float | None:
                points: list[np.ndarray] = []
                diagonals: list[float] = []
                for sample in motion_rows:
                    point = _finite_vector(sample.get("image_foot"), 2)
                    size = _finite_vector(sample.get("image_size"), 2)
                    if (
                        point is None
                        or size is None
                        or size[0] <= 0.0
                        or size[1] <= 0.0
                    ):
                        continue
                    points.append(np.asarray(point, dtype=np.float64))
                    diagonals.append(math.hypot(size[0], size[1]))
                if len(points) < 2 or not diagonals:
                    return None
                scale = float(np.median(np.asarray(diagonals, dtype=np.float64)))
                if scale <= 0.0:
                    return None
                if spread:
                    array = np.stack(points, axis=0)
                    median = np.median(array, axis=0)
                    return float(
                        np.max(np.linalg.norm(array - median, axis=1)) / scale
                    )
                return float(np.linalg.norm(points[-1] - points[0]) / scale)

            world_before = [
                row
                for row in before_all
                if row.get("world_valid") is True
                and _finite_vector(row.get("world"), 3) is not None
            ]
            world_hold = [
                row
                for row in hold_all
                if row.get("world_valid") is True
                and _finite_vector(row.get("world"), 3) is not None
            ]
            pre_motion_m: float | None = None
            if len(world_before) >= 2:
                start = transform_point(
                    _finite_vector(world_before[0].get("world"), 3),
                    world_to_scene,
                )
                end = transform_point(
                    _finite_vector(world_before[-1].get("world"), 3),
                    world_to_scene,
                )
                pre_motion_m = float(
                    np.linalg.norm(end[[0, 2]] - start[[0, 2]]) * scene_to_m
                )
            post_spread_m: float | None = None
            median_scene: np.ndarray | None = None
            if world_hold:
                hold_scene = np.asarray(
                    [
                        transform_point(
                            _finite_vector(row.get("world"), 3),
                            world_to_scene,
                        )
                        for row in world_hold
                    ],
                    dtype=np.float64,
                )
                median_scene = np.median(hold_scene, axis=0)
                post_spread_m = float(
                    np.max(
                        np.linalg.norm(
                            hold_scene[:, [0, 2]] - median_scene[[0, 2]],
                            axis=1,
                        )
                    )
                    * scene_to_m
                )
            image_pre_motion = image_motion(before_all, spread=False)
            image_post_spread = image_motion(hold_all, spread=True)
            error_m = None
            observed_scene_xz = None
            if expected is not None and median_scene is not None:
                observed_scene_xz = [
                    float(median_scene[0]),
                    float(median_scene[2]),
                ]
                error_m = float(
                    np.linalg.norm(
                        median_scene[[0, 2]] - np.asarray(expected)
                    )
                    * scene_to_m
                )
            support_score = min(1.0, len(rows) / 8.0)
            # Current-world motion is diagnostic only: it is produced by the
            # calibration under test and therefore cannot rank the actor.
            arrival_score = min(
                1.0, float(image_pre_motion or 0.0) / 0.02
            )
            hold_score = (
                max(0.0, 1.0 - float(image_post_spread) / 0.03)
                if image_post_spread is not None
                else 0.0
            )
            continuity_bonus = 0.15 if previous_key == key else 0.0
            score = (
                0.25 * support_score
                + 0.40 * arrival_score
                + 0.35 * hold_score
                + continuity_bonus
            )
            exact_row = min(
                rows,
                key=lambda row: (
                    abs(int(row.get("received_monotonic_ns") or 0) - marker_ns),
                    int(row.get("received_monotonic_ns") or 0),
                    int(row.get("sequence") or 0),
                ),
            )
            exact_received_ns = int(exact_row.get("received_monotonic_ns") or 0)
            solver_rows = sorted(
                (
                    row
                    for row in rows
                    if int(row.get("received_monotonic_ns") or 0) >= marker_ns
                ),
                key=lambda row: (
                    int(row.get("received_monotonic_ns") or 0),
                    int(row.get("sequence") or 0),
                ),
            )[:256]
            candidates.append(
                {
                    "tracklet_key": key,
                    "score": float(score),
                    "sample_count": len(rows),
                    "pre_marker_motion_m": pre_motion_m,
                    "post_marker_spread_m": post_spread_m,
                    "pre_marker_image_motion_fraction": image_pre_motion,
                    "post_marker_image_spread_fraction": image_post_spread,
                    "world_valid_sample_count": len(world_hold),
                    "observed_scene_xz": observed_scene_xz,
                    "error_m": error_m,
                    "continuity_bonus": continuity_bonus,
                    "exact_sample": {
                        "sample_sequence": _integer(exact_row.get("sequence")),
                        "camera_id": _text(exact_row.get("camera_id")),
                        "tracklet_key": _text(exact_row.get("tracklet_key")),
                        "frame_id": _integer(exact_row.get("frame_id")),
                        "media_pts_ns": _integer(exact_row.get("media_pts_ns")),
                        "captured_at_us": _integer(exact_row.get("captured_at_us")),
                        "observed_at_us": _integer(exact_row.get("observed_at_us")),
                        "capture_time_status": _text(
                            exact_row.get("capture_time_status")
                        ),
                        "received_at_us": _integer(exact_row.get("received_at_us")),
                        "received_monotonic_ns": exact_received_ns,
                        "marker_delta_ms": float(
                            (exact_received_ns - marker_ns) / 1_000_000.0
                        ),
                        "image_foot": _finite_vector(exact_row.get("image_foot"), 2),
                        "image_size": _finite_vector(exact_row.get("image_size"), 2),
                    },
                    "post_marker_sample_sequences": [
                        int(row["sequence"])
                        for row in solver_rows
                        if _integer(row.get("sequence")) is not None
                    ],
                }
            )
        candidates.sort(key=lambda item: float(item["score"]), reverse=True)
        status = "no_candidates"
        selected = None
        margin = None
        selection_source = (
            "explicit_tracklet_marker"
            if explicit_tracklet is not None
            else "automatic_arrival_hold"
        )
        if explicit_tracklet is not None and not candidates:
            status = "selected_tracklet_missing"
        elif candidates:
            selected = candidates[0]
            margin = (
                None
                if explicit_tracklet is not None
                else float(selected["score"] - candidates[1]["score"])
                if len(candidates) > 1
                else 1.0
            )
            status = (
                "assigned"
                if explicit_tracklet is not None
                else "ambiguous"
                if len(candidates) > 1 and float(margin or 0.0) < float(ambiguity_margin)
                else "assigned"
            )
            if status == "assigned":
                actor_previous[(actor, camera_id)] = str(selected["tracklet_key"])
        result.append(
            {
                "waypoint_id": waypoint_id,
                "camera_id": camera_id,
                "actor": actor,
                "marker_monotonic_ns": marker_ns,
                "marker_recorded_at_us": _integer(marker.get("recorded_at_us")),
                "status": status,
                "selection_source": selection_source,
                "requested_tracklet_key": explicit_tracklet,
                "score_margin": margin,
                "expected_scene_xz": expected,
                "selected": selected,
                "candidates": candidates[:5],
            }
        )
    for waypoint_id, waypoint in sorted(waypoint_by_id.items()):
        if waypoint_id in arrived_waypoints:
            continue
        result.append(
            {
                "waypoint_id": waypoint_id,
                "camera_id": str(waypoint.get("camera_id") or ""),
                "actor": None,
                "marker_monotonic_ns": None,
                "status": "missing_marker",
                "score_margin": None,
                "expected_scene_xz": _finite_vector(waypoint.get("expected_scene_xz"), 2),
                "selected": None,
                "candidates": [],
            }
        )
    return result


_WAYPOINT_STAGE_FIELDS: dict[str, str] = {
    "floor_candidate": "floor_world_raw_m",
    "depth_candidate": "depth_world_raw_m",
    "prefilter_measurement": "world_measurement_raw_m",
    "filter_prediction": "world_prediction_m",
    "final_world": "world",
}
_WAYPOINT_ROTATION_MIN_DIRECTIONS = 3
_WAYPOINT_ROTATION_GOOD_ANGULAR_DEG = 2.0
_WAYPOINT_ROTATION_FAIL_ANGULAR_DEG = 5.0
_WAYPOINT_ROTATION_FAIL_IMAGE_PX = 50.0
_WAYPOINT_DEPTH_MIN_FIT_PAIRS = 32


def _camera_intrinsics(
    calibration: Mapping[str, Any], camera_id: str
) -> tuple[Any, np.ndarray | None]:
    cameras = (
        calibration.get("cameras")
        if isinstance(calibration.get("cameras"), Mapping)
        else {}
    )
    table = cameras.get("K") if isinstance(cameras.get("K"), Mapping) else {}
    raw = table.get(camera_id)
    try:
        values = np.asarray(raw, dtype=np.float64)
        if values.shape == (4,):
            fx, fy, cx, cy = [float(value) for value in values]
            matrix = np.asarray(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
        elif values.size == 9:
            matrix = values.reshape((3, 3))
        else:
            return _json_copy(raw), None
        if not np.all(np.isfinite(matrix)) or matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
            return _json_copy(raw), None
        return _json_copy(raw), matrix
    except Exception:
        return _json_copy(raw), None


def _camera_extrinsics(
    calibration: Mapping[str, Any], camera_id: str
) -> tuple[Any, np.ndarray | None]:
    cameras = (
        calibration.get("cameras")
        if isinstance(calibration.get("cameras"), Mapping)
        else {}
    )
    table = cameras.get("E") if isinstance(cameras.get("E"), Mapping) else {}
    raw = table.get(camera_id)
    try:
        matrix = matrix_from_col_major(raw, name=f"{camera_id}.E")
        if not np.all(np.isfinite(matrix)):
            return _json_copy(raw), None
        return _json_copy(raw), matrix
    except Exception:
        return _json_copy(raw), None


def _similarity_evidence(
    matrix: np.ndarray, scene_to_m: float, *, source: str
) -> dict[str, Any]:
    col_major = [float(value) for value in matrix.flatten(order="F")]
    digest_payload = {
        "world_to_scene_col_major": col_major,
        "s_obj_to_m": float(scene_to_m),
    }
    return {
        "source": str(source),
        **digest_payload,
        "sha256": payload_sha256(digest_payload),
    }


def _point_error(
    point_world_m: Sequence[float],
    expected_scene_xyz: Sequence[float],
    matrix: np.ndarray,
    scene_to_m: float,
) -> dict[str, Any]:
    scene = transform_point(point_world_m, matrix)
    expected = np.asarray(expected_scene_xyz, dtype=np.float64)
    delta_scene = scene - expected
    delta_m = delta_scene * float(scene_to_m)
    return {
        "backend_world_m": [float(value) for value in point_world_m],
        "menon_scene_xyz": [float(value) for value in scene],
        "delta_scene_xyz": [float(value) for value in delta_scene],
        "delta_m_xyz": [float(value) for value in delta_m],
        "error_m": float(np.linalg.norm(delta_m)),
        "error_xz_m": float(np.linalg.norm(delta_m[[0, 2]])),
        "vertical_error_m": float(abs(delta_m[1])),
    }


def _expected_camera_geometry(
    expected_scene_xyz: Sequence[float],
    *,
    world_to_scene: np.ndarray,
    K: np.ndarray | None,
    E: np.ndarray | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        scene_to_world = np.linalg.inv(world_to_scene)
        world = transform_point(expected_scene_xyz, scene_to_world)
    except Exception:
        return result
    result["backend_world_m"] = [float(value) for value in world]
    if E is None:
        return result
    homogeneous = np.asarray([world[0], world[1], world[2], 1.0], dtype=np.float64)
    camera_h = E @ homogeneous
    if not np.all(np.isfinite(camera_h[:3])):
        return result
    camera = camera_h[:3]
    result["camera_xyz_m"] = [float(value) for value in camera]
    result["optical_depth_m"] = float(camera[2])
    if K is None or float(camera[2]) <= 1e-9:
        return result
    pixel_h = K @ camera
    pixel = pixel_h[:2] / pixel_h[2]
    if np.all(np.isfinite(pixel)):
        result["image_uv"] = [float(value) for value in pixel]
    return result


def _observed_camera_ray_geometry(
    image_foot: Sequence[float],
    *,
    K: np.ndarray | None,
    E: np.ndarray | None,
) -> dict[str, Any]:
    """Return the exact fixed-center ray needed by a later rotation-only solve."""

    if K is None or E is None:
        return {}
    try:
        uv1 = np.asarray(
            [float(image_foot[0]), float(image_foot[1]), 1.0],
            dtype=np.float64,
        )
        ray_camera = np.linalg.inv(K) @ uv1
        camera_norm = float(np.linalg.norm(ray_camera))
        if not math.isfinite(camera_norm) or camera_norm <= 1e-12:
            return {}
        ray_camera /= camera_norm
        camera_to_world = np.linalg.inv(E)
        camera_center = camera_to_world[:3, 3]
        ray_world = camera_to_world[:3, :3] @ ray_camera
        world_norm = float(np.linalg.norm(ray_world))
        if not math.isfinite(world_norm) or world_norm <= 1e-12:
            return {}
        ray_world /= world_norm
        if not (
            np.all(np.isfinite(camera_center))
            and np.all(np.isfinite(ray_camera))
            and np.all(np.isfinite(ray_world))
        ):
            return {}
        return {
            "camera_center_backend_world_m": [
                float(value) for value in camera_center
            ],
            "observed_unit_ray_camera": [float(value) for value in ray_camera],
            "observed_unit_ray_backend_world": [
                float(value) for value in ray_world
            ],
        }
    except Exception:
        return {}


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    finite = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "rmse": float(np.sqrt(np.mean(finite * finite))),
        "median": float(np.median(finite)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def _guided_waypoints(waypoints: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        waypoint
        for waypoint in waypoints.get("waypoints", [])
        if isinstance(waypoint, Mapping)
        and _finite_vector(waypoint.get("expected_scene_xyz"), 3) is not None
        and waypoint.get("split") in {"fit", "holdout"}
    ]


def _candidate_similarity(
    evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    fit_rows = [
        row
        for row in evidence
        if row.get("split") == "fit"
        and row.get("status") == "complete"
        and isinstance(row.get("stages"), Mapping)
        and isinstance(row.get("stages", {}).get("active"), Mapping)
        and isinstance(row.get("stages", {}).get("active", {}).get("final_world"), Mapping)
    ]
    if len(fit_rows) < 3:
        return {
            "status": "blocked",
            "reason": "at least three complete fit waypoints are required",
            "fit_waypoint_count": len(fit_rows),
            "advisory_only": True,
        }
    source = np.asarray(
        [row["stages"]["active"]["final_world"]["backend_world_m"] for row in fit_rows],
        dtype=np.float64,
    )
    target = np.asarray(
        [row["expected_scene_xyz"] for row in fit_rows], dtype=np.float64
    )
    if np.linalg.matrix_rank(source - np.mean(source, axis=0)) < 2:
        return {
            "status": "blocked",
            "reason": "fit waypoint backend points are collinear or degenerate",
            "fit_waypoint_count": len(fit_rows),
            "advisory_only": True,
        }
    if np.linalg.matrix_rank(target - np.mean(target, axis=0)) < 2:
        return {
            "status": "blocked",
            "reason": "fit waypoint Menon points are collinear or degenerate",
            "fit_waypoint_count": len(fit_rows),
            "advisory_only": True,
        }
    correspondences = [
        {
            "anchor_id": str(row.get("waypoint_id")),
            "camera_id": str(row.get("camera_id")),
            "world_position_m": row["stages"]["active"]["final_world"][
                "backend_world_m"
            ],
            "scene_position": row["expected_scene_xyz"],
        }
        for row in fit_rows
    ]
    try:
        solved = solve_scene_similarity(
            correspondences,
            source="guided_waypoint_fit_advisory",
            residual_units="menon_scene_units",
        )
        matrix = matrix_from_col_major(
            solved["world_to_scene_col_major"],
            name="guided_waypoint_fit.world_to_scene_col_major",
        )
        scene_to_m = float(solved["s_obj_to_m"])
        similarity = _similarity_evidence(
            matrix, scene_to_m, source="guided_waypoint_fit_advisory"
        )
    except Exception as exc:
        return {
            "status": "blocked",
            "reason": f"candidate similarity solve failed: {exc}",
            "fit_waypoint_count": len(fit_rows),
            "advisory_only": True,
        }
    return {
        "status": "complete",
        "advisory_only": True,
        "active_config_modified": False,
        "fit_waypoint_count": len(fit_rows),
        "similarity": similarity,
        "solver": solved,
    }


def _unit_vector(values: Sequence[float]) -> np.ndarray | None:
    try:
        vector = np.asarray(values, dtype=np.float64).reshape(3)
    except Exception:
        return None
    norm = float(np.linalg.norm(vector))
    if not np.all(np.isfinite(vector)) or not math.isfinite(norm) or norm <= 1e-12:
        return None
    return vector / norm


def _angle_deg(first: Sequence[float], second: Sequence[float]) -> float | None:
    first_unit = _unit_vector(first)
    second_unit = _unit_vector(second)
    if first_unit is None or second_unit is None:
        return None
    cosine = float(np.clip(np.dot(first_unit, second_unit), -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def _camera_matrices_from_evidence(
    row: Mapping[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None, Any, Any]:
    calibration_row = (
        row.get("calibration")
        if isinstance(row.get("calibration"), Mapping)
        else {}
    )
    camera_id = str(row.get("camera_id") or "")
    K_raw, K = _camera_intrinsics(
        {"cameras": {"K": {camera_id: calibration_row.get("K")}}},
        camera_id,
    )
    E_raw, E = _camera_extrinsics(
        {
            "cameras": {
                "E": {
                    camera_id: calibration_row.get("E_world_to_camera_col_major")
                }
            }
        },
        camera_id,
    )
    return K, E, K_raw, E_raw


def _expected_backend_point(row: Mapping[str, Any]) -> list[float] | None:
    geometry = (
        row.get("expected_camera_geometry")
        if isinstance(row.get("expected_camera_geometry"), Mapping)
        else {}
    )
    return _finite_vector(geometry.get("backend_world_m"), 3)


def _project_backend_point(
    point_world_m: Sequence[float], K: np.ndarray, E: np.ndarray
) -> tuple[list[float] | None, list[float] | None]:
    homogeneous = np.asarray(
        [float(point_world_m[0]), float(point_world_m[1]), float(point_world_m[2]), 1.0],
        dtype=np.float64,
    )
    camera = E @ homogeneous
    if not np.all(np.isfinite(camera[:3])):
        return None, None
    camera_xyz = [float(value) for value in camera[:3]]
    if float(camera[2]) <= 1e-9:
        return camera_xyz, None
    pixel_h = K @ camera[:3]
    pixel = pixel_h[:2] / pixel_h[2]
    if not np.all(np.isfinite(pixel)):
        return camera_xyz, None
    return camera_xyz, [float(value) for value in pixel]


def _fit_fixed_center_rotation(
    camera_id: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    source_capture_artifact_index_sha256: str,
    physical_target_similarity_sha256: str,
) -> dict[str, Any]:
    fit_rows = [
        row
        for row in rows
        if row.get("split") == "fit" and row.get("status") == "complete"
    ]
    if len(fit_rows) < _WAYPOINT_ROTATION_MIN_DIRECTIONS:
        return {
            "status": "blocked",
            "admission_status": "blocked",
            "reason": (
                f"at least {_WAYPOINT_ROTATION_MIN_DIRECTIONS} complete fit "
                "waypoint directions are required"
            ),
            "fit_waypoint_count": len(fit_rows),
            "advisory_only": True,
            "active_config_modified": False,
        }
    first = fit_rows[0]
    K, source_E, K_raw, E_raw = _camera_matrices_from_evidence(first)
    center = _finite_vector(
        first.get("observed_camera_ray_geometry", {}).get(
            "camera_center_backend_world_m"
        ),
        3,
    )
    if K is None or source_E is None or center is None:
        return {
            "status": "blocked",
            "admission_status": "blocked",
            "reason": "camera K, E, or fixed optical center is unavailable",
            "fit_waypoint_count": len(fit_rows),
            "advisory_only": True,
            "active_config_modified": False,
        }
    camera_rays: list[np.ndarray] = []
    target_rays: list[np.ndarray] = []
    fit_waypoint_ids: list[str] = []
    fit_inputs: list[dict[str, Any]] = []
    for row in fit_rows:
        row_K, row_E, row_K_raw, row_E_raw = _camera_matrices_from_evidence(row)
        row_center = _finite_vector(
            row.get("observed_camera_ray_geometry", {}).get(
                "camera_center_backend_world_m"
            ),
            3,
        )
        observed_camera_ray = _finite_vector(
            row.get("observed_camera_ray_geometry", {}).get(
                "observed_unit_ray_camera"
            ),
            3,
        )
        expected_world = _expected_backend_point(row)
        if (
            row_K is None
            or row_E is None
            or row_center is None
            or observed_camera_ray is None
            or expected_world is None
            or canonical_json_bytes({"K": row_K_raw})
            != canonical_json_bytes({"K": K_raw})
            or canonical_json_bytes({"E": row_E_raw})
            != canonical_json_bytes({"E": E_raw})
            or float(np.linalg.norm(np.asarray(row_center) - np.asarray(center)))
            > 1e-9
        ):
            continue
        observed_unit = _unit_vector(observed_camera_ray)
        target_unit = _unit_vector(np.asarray(expected_world) - np.asarray(center))
        if observed_unit is None or target_unit is None:
            continue
        waypoint_id = str(row.get("waypoint_id"))
        camera_rays.append(observed_unit)
        target_rays.append(target_unit)
        fit_waypoint_ids.append(waypoint_id)
        fit_inputs.append(
            {
                "waypoint_id": waypoint_id,
                "sample_sequence": row.get("binding", {}).get("sample_sequence"),
                "observed_unit_ray_camera": [float(value) for value in observed_unit],
                "target_unit_ray_backend_world": [
                    float(value) for value in target_unit
                ],
                "expected_backend_world_m": expected_world,
            }
        )
    if len(camera_rays) < _WAYPOINT_ROTATION_MIN_DIRECTIONS:
        return {
            "status": "blocked",
            "admission_status": "blocked",
            "reason": "too few coherent fit direction pairs remain after provenance checks",
            "fit_waypoint_count": len(camera_rays),
            "advisory_only": True,
            "active_config_modified": False,
        }
    observed = np.stack(camera_rays, axis=0)
    target = np.stack(target_rays, axis=0)
    if np.linalg.matrix_rank(observed, tol=1e-8) < 2 or np.linalg.matrix_rank(
        target, tol=1e-8
    ) < 2:
        return {
            "status": "blocked",
            "admission_status": "blocked",
            "reason": "fit directions are parallel or geometrically degenerate",
            "fit_waypoint_count": len(camera_rays),
            "advisory_only": True,
            "active_config_modified": False,
        }
    covariance = target.T @ observed
    try:
        u, singular_values, vt = np.linalg.svd(covariance)
    except np.linalg.LinAlgError as exc:
        return {
            "status": "blocked",
            "admission_status": "blocked",
            "reason": f"proper-rotation SVD failed: {exc}",
            "fit_waypoint_count": len(camera_rays),
            "advisory_only": True,
            "active_config_modified": False,
        }
    unconstrained = u @ vt
    reflection_correction_applied = float(np.linalg.det(unconstrained)) < 0.0
    correction = np.eye(3, dtype=np.float64)
    if reflection_correction_applied:
        correction[-1, -1] = -1.0
    camera_to_world_rotation = u @ correction @ vt
    determinant = float(np.linalg.det(camera_to_world_rotation))
    orthogonality_error = float(
        np.max(
            np.abs(
                camera_to_world_rotation.T @ camera_to_world_rotation
                - np.eye(3, dtype=np.float64)
            )
        )
    )
    if (
        not np.all(np.isfinite(camera_to_world_rotation))
        or abs(determinant - 1.0) > 1e-8
        or orthogonality_error > 1e-8
    ):
        return {
            "status": "blocked",
            "admission_status": "blocked",
            "reason": "solver did not produce a finite proper rotation",
            "fit_waypoint_count": len(camera_rays),
            "determinant": determinant,
            "orthogonality_error": orthogonality_error,
            "advisory_only": True,
            "active_config_modified": False,
        }
    world_to_camera_rotation = camera_to_world_rotation.T
    center_array = np.asarray(center, dtype=np.float64)
    candidate_E = np.eye(4, dtype=np.float64)
    candidate_E[:3, :3] = world_to_camera_rotation
    candidate_E[:3, 3] = -(world_to_camera_rotation @ center_array)
    recovered_center = -(candidate_E[:3, :3].T @ candidate_E[:3, 3])
    center_error_m = float(np.linalg.norm(recovered_center - center_array))
    angular_errors = [
        float(_angle_deg(camera_to_world_rotation @ ray, target_ray) or 0.0)
        for ray, target_ray in zip(camera_rays, target_rays, strict=True)
    ]
    image_errors: list[float] = []
    for row in fit_rows:
        expected_world = _expected_backend_point(row)
        image_foot = _finite_vector(row.get("binding", {}).get("image_foot"), 2)
        if expected_world is None or image_foot is None:
            continue
        _camera_xyz, pixel = _project_backend_point(expected_world, K, candidate_E)
        if pixel is not None:
            image_errors.append(
                float(np.linalg.norm(np.asarray(pixel) - np.asarray(image_foot)))
            )
    angular_summary = _distribution(angular_errors)
    image_summary = _distribution(image_errors)
    p95_angle = _finite_float(angular_summary.get("p95"))
    max_angle = _finite_float(angular_summary.get("max"))
    p95_image = _finite_float(image_summary.get("p95"))
    admission_rejection_reasons: list[str] = []
    if reflection_correction_applied:
        admission_rejection_reasons.append(
            "unconstrained direction fit is reflective; mirrored evidence is not admissible"
        )
    if center_error_m > 1e-9:
        admission_rejection_reasons.append(
            "candidate camera center is not preserved within 1e-9 m"
        )
    if p95_angle is None or max_angle is None:
        admission_rejection_reasons.append("fit angular error metrics are unavailable")
    elif (
        p95_angle > _WAYPOINT_ROTATION_FAIL_ANGULAR_DEG
        or max_angle > 2.0 * _WAYPOINT_ROTATION_FAIL_ANGULAR_DEG
    ):
        admission_rejection_reasons.append(
            "fit angular error exceeds the admission threshold"
        )
    if p95_image is None:
        admission_rejection_reasons.append(
            "fit image reprojection error metrics are unavailable"
        )
    elif p95_image > _WAYPOINT_ROTATION_FAIL_IMAGE_PX:
        admission_rejection_reasons.append(
            "fit image reprojection error exceeds the admission threshold"
        )
    admissible = bool(
        not admission_rejection_reasons
    )
    source_bindings = {
        "source_capture_artifact_index_sha256": source_capture_artifact_index_sha256,
        "calibration_bundle_sha256": first.get("calibration", {}).get(
            "bundle_sha256"
        ),
        "active_similarity_sha256": first.get("calibration", {})
        .get("similarity", {})
        .get("sha256"),
        "physical_target_similarity_sha256": physical_target_similarity_sha256,
        "camera_K_sha256": payload_sha256({"K": K_raw}),
        "source_camera_E_sha256": payload_sha256({"E": E_raw}),
        "fit_input_sha256": payload_sha256(
            {"camera_id": camera_id, "fit_inputs": fit_inputs}
        ),
    }
    candidate_payload = {
        "camera_id": camera_id,
        "fixed_camera_center_backend_world_m": center,
        "candidate_E_world_to_camera_col_major": [
            float(value) for value in candidate_E.flatten(order="F")
        ],
        "fit_waypoint_ids": fit_waypoint_ids,
        "source_bindings": source_bindings,
    }
    return {
        "status": "complete",
        "admission_status": "admissible" if admissible else "rejected",
        "advisory_only": True,
        "active_config_modified": False,
        "solver": "proper_wahba_kabsch_fixed_center_v1",
        "camera_id": camera_id,
        "fit_waypoint_count": len(fit_waypoint_ids),
        "fit_waypoint_ids": fit_waypoint_ids,
        "holdout_used_by_solver": False,
        "source_E_world_to_camera_col_major": E_raw,
        "fixed_camera_center_backend_world_m": center,
        "candidate_camera_center_backend_world_m": [
            float(value) for value in recovered_center
        ],
        "camera_center_preservation_error_m": center_error_m,
        "camera_to_world_rotation_row_major": [
            float(value) for value in camera_to_world_rotation.reshape(-1)
        ],
        "candidate_E_world_to_camera_col_major": candidate_payload[
            "candidate_E_world_to_camera_col_major"
        ],
        "determinant": determinant,
        "orthogonality_error": orthogonality_error,
        "reflection_correction_applied": reflection_correction_applied,
        "admission_rejection_reasons": admission_rejection_reasons,
        "singular_values": [float(value) for value in singular_values],
        "fit_metrics": {
            "angular_error_deg": angular_summary,
            "image_reprojection_error_px": image_summary,
        },
        "source_bindings": source_bindings,
        "candidate_sha256": payload_sha256(candidate_payload),
    }


def _rotation_row_metrics(
    row: Mapping[str, Any], candidate_E: np.ndarray
) -> dict[str, Any]:
    K, _source_E, _K_raw, _E_raw = _camera_matrices_from_evidence(row)
    expected_world = _expected_backend_point(row)
    image_foot = _finite_vector(row.get("binding", {}).get("image_foot"), 2)
    observed_camera_ray = _finite_vector(
        row.get("observed_camera_ray_geometry", {}).get(
            "observed_unit_ray_camera"
        ),
        3,
    )
    center = _finite_vector(
        row.get("observed_camera_ray_geometry", {}).get(
            "camera_center_backend_world_m"
        ),
        3,
    )
    if (
        K is None
        or expected_world is None
        or image_foot is None
        or observed_camera_ray is None
        or center is None
    ):
        return {"status": "blocked", "reason": "rotation evaluation inputs missing"}
    camera_to_world_rotation = candidate_E[:3, :3].T
    predicted_world_ray = camera_to_world_rotation @ np.asarray(
        observed_camera_ray, dtype=np.float64
    )
    target_world_ray = np.asarray(expected_world) - np.asarray(center)
    angular_error = _angle_deg(predicted_world_ray, target_world_ray)
    camera_xyz, pixel = _project_backend_point(expected_world, K, candidate_E)
    image_error = (
        float(np.linalg.norm(np.asarray(pixel) - np.asarray(image_foot)))
        if pixel is not None
        else None
    )
    optical_depth = (
        _finite_float(camera_xyz[2]) if camera_xyz is not None else None
    )
    registered_depth = _finite_float(row.get("depth", {}).get("registered_m"))
    registered_error = (
        abs(registered_depth - optical_depth)
        if registered_depth is not None and optical_depth is not None
        else None
    )
    predicted_unit = _unit_vector(predicted_world_ray)
    target_unit = _unit_vector(target_world_ray)
    return {
        "status": "complete",
        "angular_ray_error_deg": angular_error,
        "image_reprojection_error_px": image_error,
        "expected_camera_xyz_m": camera_xyz,
        "expected_optical_depth_m": optical_depth,
        "registered_optical_depth_error_m": registered_error,
        "predicted_unit_ray_backend_world": [
            float(value) for value in (predicted_unit if predicted_unit is not None else [])
        ],
        "target_unit_ray_backend_world": [
            float(value) for value in (target_unit if target_unit is not None else [])
        ],
    }


def _fit_physical_depth_registration(
    camera_id: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    rotation_candidate: Mapping[str, Any],
    source_capture_artifact_index_sha256: str,
    physical_target_similarity_sha256: str,
) -> dict[str, Any]:
    if (
        rotation_candidate.get("status") != "complete"
        or rotation_candidate.get("admission_status") != "admissible"
    ):
        return {
            "status": "blocked",
            "admission_status": "blocked",
            "reason": "an admissible fixed-center rotation is required first",
            "advisory_only": True,
            "active_config_modified": False,
        }
    candidate_E = matrix_from_col_major(
        rotation_candidate.get("candidate_E_world_to_camera_col_major"),
        name=f"{camera_id}.waypoint_candidate_E",
    )
    fit_pairs: list[dict[str, Any]] = []
    for row in rows:
        if row.get("split") != "fit" or row.get("status") != "complete":
            continue
        expected_world = _expected_backend_point(row)
        if expected_world is None:
            continue
        K, _source_E, _K_raw, _E_raw = _camera_matrices_from_evidence(row)
        if K is None:
            continue
        camera_xyz, _pixel = _project_backend_point(expected_world, K, candidate_E)
        target_depth = _finite_float(camera_xyz[2]) if camera_xyz is not None else None
        if target_depth is None or target_depth <= 0.0:
            continue
        for sample in row.get("post_marker_samples", []):
            if not isinstance(sample, Mapping):
                continue
            raw_depth = _finite_float(sample.get("raw_depth_m"))
            if (
                raw_depth is None
                or raw_depth <= 0.0
                or sample.get("camera_id") != camera_id
                or sample.get("tracklet_key")
                != row.get("binding", {}).get("tracklet_key")
                or sample.get("calibration_bundle_sha256")
                != row.get("calibration", {}).get("bundle_sha256")
            ):
                continue
            fit_pairs.append(
                {
                    "waypoint_id": str(row.get("waypoint_id")),
                    "sample_sequence": _integer(sample.get("sample_sequence")),
                    "raw_depth_m": raw_depth,
                    "target_optical_depth_m": target_depth,
                }
            )
    if len(fit_pairs) < _WAYPOINT_DEPTH_MIN_FIT_PAIRS:
        return {
            "status": "blocked",
            "admission_status": "blocked",
            "reason": (
                f"at least {_WAYPOINT_DEPTH_MIN_FIT_PAIRS} coherent fit depth "
                f"pairs are required; got {len(fit_pairs)}"
            ),
            "fit_pair_count": len(fit_pairs),
            "fit_waypoint_ids": sorted(
                {str(pair["waypoint_id"]) for pair in fit_pairs}
            ),
            "holdout_used_by_solver": False,
            "advisory_only": True,
            "active_config_modified": False,
        }
    raw_values = [float(pair["raw_depth_m"]) for pair in fit_pairs]
    target_values = [float(pair["target_optical_depth_m"]) for pair in fit_pairs]
    try:
        fit = fit_monotonic_piecewise_mapping(
            raw_values,
            target_values,
            num_knots=8,
            min_samples=_WAYPOINT_DEPTH_MIN_FIT_PAIRS,
        )
    except DepthRegistrationBuildError as exc:
        return {
            "status": "blocked",
            "admission_status": "blocked",
            "reason": f"verified monotonic depth fitter rejected evidence: {exc}",
            "fit_pair_count": len(fit_pairs),
            "fit_waypoint_ids": sorted(
                {str(pair["waypoint_id"]) for pair in fit_pairs}
            ),
            "holdout_used_by_solver": False,
            "advisory_only": True,
            "active_config_modified": False,
        }
    raw_knots = np.asarray(fit.knots_raw_m, dtype=np.float64)
    target_knots = np.asarray(fit.knots_registered_m, dtype=np.float64)
    domain_min = float(raw_knots[0])
    domain_max = float(raw_knots[-1])
    admitted_fit_errors: list[float] = []
    admitted_count = 0
    for pair in fit_pairs:
        raw_depth = float(pair["raw_depth_m"])
        if not (domain_min <= raw_depth <= domain_max):
            continue
        mapped = float(np.interp(raw_depth, raw_knots, target_knots))
        admitted_fit_errors.append(
            abs(mapped - float(pair["target_optical_depth_m"]))
        )
        admitted_count += 1
    error_summary = _distribution(admitted_fit_errors)
    fit_coverage = float(admitted_count / len(fit_pairs)) if fit_pairs else 0.0
    median_error = _finite_float(error_summary.get("median"))
    p95_error = _finite_float(error_summary.get("p95"))
    max_error = _finite_float(error_summary.get("max"))
    admissible = bool(
        fit_coverage >= 0.90
        and median_error is not None
        and p95_error is not None
        and max_error is not None
        and median_error <= MAX_OCCUPIED_MEDIAN_ABS_ERROR_M
        and p95_error <= MAX_OCCUPIED_P95_ABS_ERROR_M
        and max_error <= MAX_OCCUPIED_ABS_ERROR_M
    )
    first = next(row for row in rows if row.get("camera_id") == camera_id)
    source_bindings = {
        "source_capture_artifact_index_sha256": source_capture_artifact_index_sha256,
        "calibration_bundle_sha256": first.get("calibration", {}).get(
            "bundle_sha256"
        ),
        "active_similarity_sha256": first.get("calibration", {})
        .get("similarity", {})
        .get("sha256"),
        "physical_target_similarity_sha256": physical_target_similarity_sha256,
        "rotation_candidate_sha256": rotation_candidate.get("candidate_sha256"),
        "fit_input_sha256": payload_sha256(
            {"camera_id": camera_id, "fit_pairs": fit_pairs}
        ),
    }
    candidate_payload = {
        "camera_id": camera_id,
        "knots_raw_m": [float(value) for value in raw_knots],
        "knots_physical_optical_depth_m": [
            float(value) for value in target_knots
        ],
        "source_bindings": source_bindings,
        "fit_waypoint_ids": sorted(
            {str(pair["waypoint_id"]) for pair in fit_pairs}
        ),
        "fit_sample_sequences": [pair["sample_sequence"] for pair in fit_pairs],
    }
    return {
        "status": "complete",
        "admission_status": "admissible" if admissible else "rejected",
        "advisory_only": True,
        "active_config_modified": False,
        "solver": (
            "noesis.calibration.depth_registration_builder."
            "fit_monotonic_piecewise_mapping"
        ),
        "camera_id": camera_id,
        "transform_type": "piecewise_linear_1d",
        "source_space": "dav2_anchor_range_m_raw",
        "target_space": "camera_optical_z_m_physical",
        "fit_pair_count": len(fit_pairs),
        "fit_waypoint_ids": candidate_payload["fit_waypoint_ids"],
        "fit_sample_sequences": candidate_payload["fit_sample_sequences"],
        "holdout_used_by_solver": False,
        "raw_domain_m": [domain_min, domain_max],
        "knots_raw_m": candidate_payload["knots_raw_m"],
        "knots_physical_optical_depth_m": candidate_payload[
            "knots_physical_optical_depth_m"
        ],
        "fit_coverage": fit_coverage,
        "fit_error_m": error_summary,
        "verified_fitter_metrics": dict(fit.fit_metrics),
        "verified_fitter_sample_counts": dict(fit.sample_counts),
        "source_bindings": source_bindings,
        "candidate_sha256": payload_sha256(candidate_payload),
    }


def _mapped_depth(
    raw_depth_m: float | None, candidate: Mapping[str, Any]
) -> float | None:
    if raw_depth_m is None or candidate.get("status") != "complete":
        return None
    raw_knots = _finite_vector(
        candidate.get("knots_raw_m"), len(candidate.get("knots_raw_m") or [])
    )
    target_knots = _finite_vector(
        candidate.get("knots_physical_optical_depth_m"),
        len(candidate.get("knots_physical_optical_depth_m") or []),
    )
    if (
        raw_knots is None
        or target_knots is None
        or len(raw_knots) < 2
        or len(raw_knots) != len(target_knots)
        or not (raw_knots[0] <= raw_depth_m <= raw_knots[-1])
    ):
        return None
    return float(np.interp(float(raw_depth_m), raw_knots, target_knots))


def _reconstruct_candidate_points(
    *,
    K: np.ndarray,
    candidate_E: np.ndarray,
    image_foot: Sequence[float],
    floor_y_backend_m: float | None,
    optical_depth_m: float | None,
) -> tuple[list[float] | None, list[float] | None]:
    try:
        uv1 = np.asarray(
            [float(image_foot[0]), float(image_foot[1]), 1.0], dtype=np.float64
        )
        ray_camera_z1 = np.linalg.inv(K) @ uv1
        camera_to_world = np.linalg.inv(candidate_E)
        center = camera_to_world[:3, 3]
        ray_world = camera_to_world[:3, :3] @ ray_camera_z1
    except Exception:
        return None, None
    floor_point = None
    if floor_y_backend_m is not None and abs(float(ray_world[1])) > 1e-9:
        distance = (float(floor_y_backend_m) - float(center[1])) / float(
            ray_world[1]
        )
        if math.isfinite(distance) and distance >= 0.0:
            point = center + distance * ray_world
            if np.all(np.isfinite(point)):
                floor_point = [float(value) for value in point]
    depth_point = None
    if optical_depth_m is not None and optical_depth_m > 0.0:
        camera_point = ray_camera_z1 * float(optical_depth_m)
        point = camera_to_world[:3, :3] @ camera_point + center
        if np.all(np.isfinite(point)):
            depth_point = [float(value) for value in point]
    return floor_point, depth_point


def _evaluate_camera_candidates(
    camera_id: str,
    rows: Sequence[dict[str, Any]],
    *,
    rotation_candidate: Mapping[str, Any],
    depth_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    if rotation_candidate.get("status") != "complete":
        for row in rows:
            row["camera_solver_evaluation"] = {
                "status": "blocked",
                "reason": "fixed-center rotation candidate unavailable",
            }
        return {"status": "blocked", "reason": "rotation candidate unavailable"}
    candidate_E = matrix_from_col_major(
        rotation_candidate.get("candidate_E_world_to_camera_col_major"),
        name=f"{camera_id}.candidate_E",
    )
    for row in rows:
        rotation_metrics = _rotation_row_metrics(row, candidate_E)
        expected_world = _expected_backend_point(row)
        K, _source_E, _K_raw, _E_raw = _camera_matrices_from_evidence(row)
        floor_y = _finite_float(
            row.get("calibration", {}).get("floor_y_backend_m")
        )
        sample_metrics: list[dict[str, Any]] = []
        for sample in row.get("post_marker_samples", []):
            if not isinstance(sample, Mapping):
                continue
            raw_depth = _finite_float(sample.get("raw_depth_m"))
            mapped = _mapped_depth(raw_depth, depth_candidate)
            image_foot = _finite_vector(sample.get("image_foot"), 2)
            target_depth = _finite_float(rotation_metrics.get("expected_optical_depth_m"))
            floor_point = None
            depth_point = None
            if K is not None and image_foot is not None:
                floor_point, depth_point = _reconstruct_candidate_points(
                    K=K,
                    candidate_E=candidate_E,
                    image_foot=image_foot,
                    floor_y_backend_m=floor_y,
                    optical_depth_m=mapped,
                )
            registered_depth = _finite_float(sample.get("registered_depth_m"))
            metrics_row = {
                "sample_sequence": _integer(sample.get("sample_sequence")),
                "frame_id": _integer(sample.get("frame_id")),
                "media_pts_ns": _integer(sample.get("media_pts_ns")),
                "raw_depth_m": raw_depth,
                "mapped_physical_optical_depth_m": mapped,
                "target_optical_depth_m": target_depth,
                "mapped_optical_depth_error_m": (
                    abs(mapped - target_depth)
                    if mapped is not None and target_depth is not None
                    else None
                ),
                "registered_optical_depth_error_m": (
                    abs(registered_depth - target_depth)
                    if registered_depth is not None and target_depth is not None
                    else None
                ),
                "reconstructed_floor_backend_world_m": floor_point,
                "reconstructed_depth_backend_world_m": depth_point,
                "reconstructed_floor_position_error_m": (
                    float(
                        np.linalg.norm(
                            np.asarray(floor_point) - np.asarray(expected_world)
                        )
                    )
                    if floor_point is not None and expected_world is not None
                    else None
                ),
                "reconstructed_depth_position_error_m": (
                    float(
                        np.linalg.norm(
                            np.asarray(depth_point) - np.asarray(expected_world)
                        )
                    )
                    if depth_point is not None and expected_world is not None
                    else None
                ),
                "producer_floor_candidate_position_error_m": (
                    float(
                        np.linalg.norm(
                            np.asarray(sample.get("floor_world_raw_m"))
                            - np.asarray(expected_world)
                        )
                    )
                    if _finite_vector(sample.get("floor_world_raw_m"), 3) is not None
                    and expected_world is not None
                    else None
                ),
                "producer_depth_candidate_position_error_m": (
                    float(
                        np.linalg.norm(
                            np.asarray(sample.get("depth_world_raw_m"))
                            - np.asarray(expected_world)
                        )
                    )
                    if _finite_vector(sample.get("depth_world_raw_m"), 3) is not None
                    and expected_world is not None
                    else None
                ),
                "producer_final_world_position_error_m": (
                    float(
                        np.linalg.norm(
                            np.asarray(sample.get("final_world_m"))
                            - np.asarray(expected_world)
                        )
                    )
                    if _finite_vector(sample.get("final_world_m"), 3) is not None
                    and expected_world is not None
                    and sample.get("world_valid") is True
                    else None
                ),
            }
            sample_metrics.append(metrics_row)
        row["camera_solver_evaluation"] = {
            "status": "complete",
            "rotation": rotation_metrics,
            "depth_samples": sample_metrics,
        }
    summaries: dict[str, Any] = {}
    for split in ("fit", "holdout"):
        split_rows = [row for row in rows if row.get("split") == split]
        rotation_metric_names = (
            "angular_ray_error_deg",
            "image_reprojection_error_px",
            "registered_optical_depth_error_m",
        )
        depth_metric_names = (
            "mapped_optical_depth_error_m",
            "registered_optical_depth_error_m",
            "reconstructed_floor_position_error_m",
            "reconstructed_depth_position_error_m",
            "producer_floor_candidate_position_error_m",
            "producer_depth_candidate_position_error_m",
            "producer_final_world_position_error_m",
        )
        split_summary: dict[str, Any] = {
            "waypoint_count": len(split_rows),
            "rotation": {},
            "depth": {},
        }
        for metric_name in rotation_metric_names:
            values = [
                float(value)
                for row in split_rows
                if isinstance(row.get("camera_solver_evaluation"), Mapping)
                for value in [
                    row.get("camera_solver_evaluation", {})
                    .get("rotation", {})
                    .get(metric_name)
                ]
                if _finite_float(value) is not None
            ]
            split_summary["rotation"][metric_name] = {
                "coverage": float(len(values) / len(split_rows)) if split_rows else 0.0,
                "distribution": _distribution(values),
            }
        all_samples = [
            sample
            for row in split_rows
            for sample in row.get("camera_solver_evaluation", {}).get(
                "depth_samples", []
            )
            if isinstance(sample, Mapping)
        ]
        split_summary["depth_sample_count"] = len(all_samples)
        for metric_name in depth_metric_names:
            values = [
                float(sample[metric_name])
                for sample in all_samples
                if _finite_float(sample.get(metric_name)) is not None
            ]
            split_summary["depth"][metric_name] = {
                "coverage": float(len(values) / len(all_samples))
                if all_samples
                else 0.0,
                "distribution": _distribution(values),
            }
        summaries[split] = split_summary
    return {"status": "complete", "splits": summaries}


def _build_camera_calibration_candidates(
    evidence: Sequence[dict[str, Any]],
    *,
    similarity_candidate: Mapping[str, Any],
    source_capture_artifact_index_sha256: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "contract": "noesis.alignment.waypoint_camera_calibration_candidates",
        "contract_version": CONTRACT_VERSION,
        "advisory_only": True,
        "active_config_modified": False,
        "source_capture_artifact_index_sha256": source_capture_artifact_index_sha256,
        "cameras": {},
    }
    similarity = (
        similarity_candidate.get("similarity")
        if isinstance(similarity_candidate.get("similarity"), Mapping)
        else {}
    )
    similarity_sha = _text(similarity.get("sha256"), maximum=64)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evidence:
        camera_id = _text(row.get("camera_id"))
        if camera_id:
            grouped[camera_id].append(row)
    for camera_id, rows in sorted(grouped.items()):
        active_similarity_sha = next(
            (
                _text(
                    row.get("calibration", {})
                    .get("similarity", {})
                    .get("sha256"),
                    maximum=64,
                )
                for row in rows
                if isinstance(row.get("calibration"), Mapping)
                and isinstance(
                    row.get("calibration", {}).get("similarity"), Mapping
                )
            ),
            None,
        )
        if active_similarity_sha is None:
            rotation = {
                "status": "blocked",
                "admission_status": "blocked",
                "reason": "captured active scene similarity is unavailable",
                "advisory_only": True,
                "active_config_modified": False,
            }
        else:
            rotation = _fit_fixed_center_rotation(
                camera_id,
                rows,
                source_capture_artifact_index_sha256=(
                    source_capture_artifact_index_sha256
                ),
                physical_target_similarity_sha256=active_similarity_sha,
            )
        depth = _fit_physical_depth_registration(
            camera_id,
            rows,
            rotation_candidate=rotation,
            source_capture_artifact_index_sha256=(
                source_capture_artifact_index_sha256
            ),
            physical_target_similarity_sha256=active_similarity_sha or "",
        )
        metrics = _evaluate_camera_candidates(
            camera_id,
            rows,
            rotation_candidate=rotation,
            depth_candidate=depth,
        )
        result["cameras"][camera_id] = {
            "rotation": rotation,
            "depth_registration": depth,
            "metrics": metrics,
        }
    digest_payload = {
        "source_capture_artifact_index_sha256": source_capture_artifact_index_sha256,
        "advisory_candidate_similarity_sha256": similarity_sha,
        "cameras": result["cameras"],
    }
    result["sha256"] = payload_sha256(digest_payload)
    return result


def _waypoint_metric_summary(
    evidence: Sequence[Mapping[str, Any]],
    *,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("fit", "holdout"):
        rows = [row for row in evidence if row.get("split") == split]
        denominator = len(rows)
        split_result: dict[str, Any] = {
            "declared_waypoint_count": denominator,
            "complete_binding_count": sum(row.get("status") == "complete" for row in rows),
            "binding_coverage": (
                float(sum(row.get("status") == "complete" for row in rows) / denominator)
                if denominator
                else 0.0
            ),
            "camera_count": len(
                {str(row.get("camera_id")) for row in rows if row.get("camera_id")}
            ),
        }
        for transform_name in ("active", "candidate"):
            stage_result: dict[str, Any] = {}
            for stage_name in _WAYPOINT_STAGE_FIELDS:
                errors = []
                for row in rows:
                    stages = row.get("stages")
                    transform_stages = (
                        stages.get(transform_name)
                        if isinstance(stages, Mapping)
                        and isinstance(stages.get(transform_name), Mapping)
                        else {}
                    )
                    stage = transform_stages.get(stage_name)
                    if isinstance(stage, Mapping) and _finite_float(stage.get("error_m")) is not None:
                        errors.append(float(stage["error_m"]))
                stage_result[stage_name] = {
                    "coverage": float(len(errors) / denominator) if denominator else 0.0,
                    "error_m": _distribution(errors),
                }
            split_result[f"{transform_name}_stages"] = stage_result
        for metric_name in (
            "image_foot_error_px",
            "raw_depth_error_m",
            "registered_depth_error_m",
            "candidate_image_foot_error_px",
            "candidate_raw_depth_error_m",
            "candidate_registered_depth_error_m",
        ):
            values = [
                float(row["camera_metrics"][metric_name])
                for row in rows
                if isinstance(row.get("camera_metrics"), Mapping)
                and _finite_float(row["camera_metrics"].get(metric_name)) is not None
            ]
            split_result[metric_name] = {
                "coverage": float(len(values) / denominator) if denominator else 0.0,
                "distribution": _distribution(values),
            }
        result[split] = split_result
    return {
        "contract": WAYPOINT_METRICS_CONTRACT,
        "contract_version": CONTRACT_VERSION,
        "candidate_status": candidate.get("status"),
        "splits": result,
    }


def build_waypoint_calibration_evidence(
    samples: Sequence[Mapping[str, Any]],
    assignments: Sequence[Mapping[str, Any]],
    waypoints: Mapping[str, Any],
    *,
    calibration: Mapping[str, Any],
    calibration_bundle_sha256: str,
    source_capture_artifact_index_sha256: str,
    world_to_scene: np.ndarray,
    scene_to_m: float,
    scene_binding: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Build exact, privacy-safe waypoint correspondences and split metrics.

    The advisory fit is derived only from ``split=fit`` final-world points.
    Holdout points are never used by the solver and remain an independent
    physical validation set. No result from this function mutates calibration.
    """

    assignment_by_id = {
        str(item.get("waypoint_id")): item for item in assignments
    }
    sample_by_sequence = {
        int(row["sequence"]): row
        for row in samples
        if _integer(row.get("sequence")) is not None
    }
    active_similarity = _similarity_evidence(
        world_to_scene, scene_to_m, source="captured_active_similarity"
    )
    evidence: list[dict[str, Any]] = []
    for waypoint in _guided_waypoints(waypoints):
        waypoint_id = str(waypoint.get("id"))
        camera_id = str(waypoint.get("camera_id"))
        expected_scene = _finite_vector(waypoint.get("expected_scene_xyz"), 3)
        split = str(waypoint.get("split"))
        assignment = assignment_by_id.get(waypoint_id, {})
        selected = (
            assignment.get("selected")
            if isinstance(assignment.get("selected"), Mapping)
            else {}
        )
        exact = (
            selected.get("exact_sample")
            if isinstance(selected.get("exact_sample"), Mapping)
            else {}
        )
        sample_sequence = _integer(exact.get("sample_sequence"))
        sample = sample_by_sequence.get(sample_sequence) if sample_sequence is not None else None
        raw_solver_sequences = selected.get("post_marker_sample_sequences")
        solver_sequences = (
            [
                int(value)
                for value in raw_solver_sequences
                if _integer(value) is not None
            ]
            if isinstance(raw_solver_sequences, list)
            else ([sample_sequence] if sample_sequence is not None else [])
        )
        K_raw, K = _camera_intrinsics(calibration, camera_id)
        E_raw, E = _camera_extrinsics(calibration, camera_id)
        missing: list[str] = []
        diagnostics: list[str] = []
        if assignment.get("status") != "assigned":
            missing.append("unambiguous_assigned_tracklet")
        if sample is None:
            missing.append("exact_sample")
        if expected_scene is None:
            missing.append("expected_scene_xyz")
        if K is None:
            missing.append("camera_K")
        if E is None:
            missing.append("camera_E")
        binding: dict[str, Any] = {
            "camera_id": camera_id,
            "tracklet_key": _text(exact.get("tracklet_key")),
            "frame_id": _integer(exact.get("frame_id")),
            "media_pts_ns": _integer(exact.get("media_pts_ns")),
            "captured_at_us": _integer(exact.get("captured_at_us")),
            "observed_at_us": _integer(exact.get("observed_at_us")),
            "capture_time_status": _text(exact.get("capture_time_status")),
            "sample_sequence": sample_sequence,
            "received_at_us": _integer(exact.get("received_at_us")),
            "received_monotonic_ns": _integer(exact.get("received_monotonic_ns")),
            "marker_monotonic_ns": _integer(assignment.get("marker_monotonic_ns")),
            "marker_recorded_at_us": _integer(assignment.get("marker_recorded_at_us")),
            "marker_delta_ms": _finite_float(exact.get("marker_delta_ms")),
            "image_foot": _finite_vector(exact.get("image_foot"), 2),
            "image_size": _finite_vector(exact.get("image_size"), 2),
            "assignment_score_margin": _finite_float(assignment.get("score_margin")),
        }
        for required in (
            "tracklet_key",
            "frame_id",
            "media_pts_ns",
            "captured_at_us",
            "observed_at_us",
            "sample_sequence",
            "received_monotonic_ns",
            "image_foot",
            "image_size",
        ):
            if binding.get(required) is None:
                missing.append(required)
        media_pts_ns = _integer(binding.get("media_pts_ns"))
        frame_id = _integer(binding.get("frame_id"))
        if media_pts_ns is not None and media_pts_ns < 0:
            missing.append("nonnegative_media_pts_ns")
        if frame_id is not None and frame_id < 0:
            missing.append("nonnegative_frame_id")
        image_size = _finite_vector(binding.get("image_size"), 2)
        image_foot = _finite_vector(binding.get("image_foot"), 2)
        if image_size is not None and (
            image_size[0] <= 0.0 or image_size[1] <= 0.0
        ):
            missing.append("positive_image_size")
        if image_size is not None and image_foot is not None and not (
            0.0 <= image_foot[0] < image_size[0]
            and 0.0 <= image_foot[1] < image_size[1]
        ):
            missing.append("image_foot_inside_image")
        if sample is not None:
            if sample.get("camera_id") != camera_id:
                missing.append("sample_camera_match")
            if sample.get("tracklet_key") != binding.get("tracklet_key"):
                missing.append("sample_tracklet_match")
            if sample.get("frame_id") != binding.get("frame_id"):
                missing.append("sample_frame_match")
            if sample.get("media_pts_ns") != binding.get("media_pts_ns"):
                missing.append("sample_media_pts_match")
            if sample.get("calibration_bundle_sha256") != calibration_bundle_sha256:
                missing.append("sample_calibration_bundle_match")
        expected_geometry = (
            _expected_camera_geometry(
                expected_scene,
                world_to_scene=world_to_scene,
                K=K,
                E=E,
            )
            if expected_scene is not None
            else {}
        )
        if "backend_world_m" not in expected_geometry:
            missing.append("invertible_similarity")
        if "image_uv" not in expected_geometry:
            diagnostics.append("active_expected_projection_not_visible")
        observed_ray_geometry = (
            _observed_camera_ray_geometry(image_foot, K=K, E=E)
            if image_foot is not None
            else {}
        )
        if not observed_ray_geometry:
            missing.append("observed_camera_ray")
        align_payload = (
            calibration.get("align")
            if isinstance(calibration.get("align"), Mapping)
            else {}
        )
        floor_y_backend_m = _finite_float(align_payload.get("floor_y"))
        stages: dict[str, Any] = {}
        if sample is not None and expected_scene is not None:
            for stage_name, field_name in _WAYPOINT_STAGE_FIELDS.items():
                if (
                    stage_name == "final_world"
                    and sample.get("world_valid") is not True
                ):
                    continue
                point = _finite_vector(sample.get(field_name), 3)
                if point is not None:
                    stages[stage_name] = _point_error(
                        point, expected_scene, world_to_scene, scene_to_m
                    )
        if (
            "final_world" not in stages
            or sample is None
            or sample.get("world_valid") is not True
        ):
            diagnostics.append("producer_final_world_unavailable")
        camera_metrics: dict[str, Any] = {}
        expected_uv = _finite_vector(expected_geometry.get("image_uv"), 2)
        if image_foot is not None and expected_uv is not None:
            camera_metrics["image_foot_error_px"] = float(
                np.linalg.norm(np.asarray(image_foot) - np.asarray(expected_uv))
            )
        expected_depth = _finite_float(expected_geometry.get("optical_depth_m"))
        raw_depth = _finite_float(sample.get("depth_anchor_m")) if sample is not None else None
        registered_depth = (
            _finite_float(sample.get("depth_registered_m")) if sample is not None else None
        )
        used_depth = _finite_float(sample.get("depth_used_m")) if sample is not None else None
        if expected_depth is not None and raw_depth is not None:
            camera_metrics["raw_depth_error_m"] = abs(raw_depth - expected_depth)
        if expected_depth is not None and registered_depth is not None:
            camera_metrics["registered_depth_error_m"] = abs(
                registered_depth - expected_depth
            )
        row: dict[str, Any] = {
            "contract": WAYPOINT_EVIDENCE_CONTRACT,
            "contract_version": CONTRACT_VERSION,
            "waypoint_id": waypoint_id,
            "label": waypoint.get("label"),
            "camera_id": camera_id,
            "split": split,
            "scene_binding": _json_copy(scene_binding),
            "expected_scene_xyz": expected_scene,
            "assignment_status": assignment.get("status"),
            "status": "complete" if not missing else "blocked",
            "missing_required_fields": sorted(set(missing)),
            "diagnostic_flags": sorted(set(diagnostics)),
            "binding": binding,
            "calibration": {
                "bundle_sha256": calibration_bundle_sha256,
                "sample_bundle_sha256": (
                    sample.get("calibration_bundle_sha256") if sample is not None else None
                ),
                "K": K_raw,
                "image_size": binding.get("image_size"),
                "E_world_to_camera_col_major": E_raw,
                "floor_y_backend_m": floor_y_backend_m,
                "similarity": active_similarity,
            },
            "depth": {
                "raw_anchor_m": raw_depth,
                "registered_m": registered_depth,
                "used_m": used_depth,
                "registration_status": (
                    sample.get("depth_registration_status") if sample is not None else None
                ),
                "registration_id": (
                    sample.get("depth_registration_id") if sample is not None else None
                ),
                "physical_registration_pair": {
                    "raw_depth_m": raw_depth,
                    "registered_depth_m": registered_depth,
                    "target_optical_depth_m": expected_depth,
                },
            },
            "expected_camera_geometry": expected_geometry,
            "observed_camera_ray_geometry": observed_ray_geometry,
            "camera_metrics": camera_metrics,
            "stages": {"active": stages},
            "producer_provenance": {
                key: sample.get(key) if sample is not None else None
                for key in ("calibration_sha256", "model_sha256", "config_sha256")
            },
        }
        solver_samples: list[dict[str, Any]] = []
        for solver_sequence in solver_sequences[:256]:
            solver_sample = sample_by_sequence.get(int(solver_sequence))
            if solver_sample is None:
                continue
            solver_samples.append(
                {
                    "sample_sequence": int(solver_sequence),
                    "camera_id": _text(solver_sample.get("camera_id")),
                    "tracklet_key": _text(solver_sample.get("tracklet_key")),
                    "frame_id": _integer(solver_sample.get("frame_id")),
                    "media_pts_ns": _integer(solver_sample.get("media_pts_ns")),
                    "captured_at_us": _integer(solver_sample.get("captured_at_us")),
                    "observed_at_us": _integer(solver_sample.get("observed_at_us")),
                    "received_monotonic_ns": _integer(
                        solver_sample.get("received_monotonic_ns")
                    ),
                    "image_foot": _finite_vector(
                        solver_sample.get("image_foot"), 2
                    ),
                    "image_size": _finite_vector(
                        solver_sample.get("image_size"), 2
                    ),
                    "raw_depth_m": _finite_float(
                        solver_sample.get("depth_anchor_m")
                    ),
                    "registered_depth_m": _finite_float(
                        solver_sample.get("depth_registered_m")
                    ),
                    "used_depth_m": _finite_float(
                        solver_sample.get("depth_used_m")
                    ),
                    "calibration_bundle_sha256": solver_sample.get(
                        "calibration_bundle_sha256"
                    ),
                    "floor_world_raw_m": _finite_vector(
                        solver_sample.get("floor_world_raw_m"), 3
                    ),
                    "depth_world_raw_m": _finite_vector(
                        solver_sample.get("depth_world_raw_m"), 3
                    ),
                    "final_world_m": _finite_vector(solver_sample.get("world"), 3),
                    "world_valid": solver_sample.get("world_valid") is True,
                }
            )
        row["post_marker_samples"] = solver_samples
        evidence.append(row)
    candidate = _candidate_similarity(evidence)
    if candidate.get("status") == "complete":
        similarity = candidate.get("similarity")
        candidate_matrix = matrix_from_col_major(
            similarity["world_to_scene_col_major"],
            name="guided_waypoint_candidate_similarity",
        )
        candidate_scene_to_m = float(similarity["s_obj_to_m"])
        for row in evidence:
            expected_scene = _finite_vector(row.get("expected_scene_xyz"), 3)
            active_stages = row.get("stages", {}).get("active", {})
            candidate_stages: dict[str, Any] = {}
            if expected_scene is not None and isinstance(active_stages, Mapping):
                for stage_name, stage in active_stages.items():
                    point = (
                        _finite_vector(stage.get("backend_world_m"), 3)
                        if isinstance(stage, Mapping)
                        else None
                    )
                    if point is not None:
                        candidate_stages[str(stage_name)] = _point_error(
                            point,
                            expected_scene,
                            candidate_matrix,
                            candidate_scene_to_m,
                        )
            row["stages"]["candidate"] = candidate_stages
            calibration_row = (
                row.get("calibration")
                if isinstance(row.get("calibration"), Mapping)
                else {}
            )
            _K_raw, K = _camera_intrinsics(
                {"cameras": {"K": {str(row.get("camera_id")): calibration_row.get("K")}}},
                str(row.get("camera_id")),
            )
            _E_raw, E = _camera_extrinsics(
                {
                    "cameras": {
                        "E": {
                            str(row.get("camera_id")): calibration_row.get(
                                "E_world_to_camera_col_major"
                            )
                        }
                    }
                },
                str(row.get("camera_id")),
            )
            candidate_geometry = (
                _expected_camera_geometry(
                    expected_scene,
                    world_to_scene=candidate_matrix,
                    K=K,
                    E=E,
                )
                if expected_scene is not None
                else {}
            )
            row["candidate_expected_camera_geometry"] = candidate_geometry
            candidate_depth = _finite_float(candidate_geometry.get("optical_depth_m"))
            depth = row.get("depth") if isinstance(row.get("depth"), dict) else {}
            raw_depth = _finite_float(depth.get("raw_anchor_m"))
            registered_depth = _finite_float(depth.get("registered_m"))
            depth["candidate_physical_registration_pair"] = {
                "raw_depth_m": raw_depth,
                "registered_depth_m": registered_depth,
                "target_optical_depth_m": candidate_depth,
            }
            camera_metrics = (
                row.get("camera_metrics")
                if isinstance(row.get("camera_metrics"), dict)
                else {}
            )
            candidate_uv = _finite_vector(candidate_geometry.get("image_uv"), 2)
            image_foot = _finite_vector(row.get("binding", {}).get("image_foot"), 2)
            if candidate_uv is not None and image_foot is not None:
                camera_metrics["candidate_image_foot_error_px"] = float(
                    np.linalg.norm(np.asarray(image_foot) - np.asarray(candidate_uv))
                )
            if candidate_depth is not None and raw_depth is not None:
                camera_metrics["candidate_raw_depth_error_m"] = abs(
                    raw_depth - candidate_depth
                )
            if candidate_depth is not None and registered_depth is not None:
                camera_metrics["candidate_registered_depth_error_m"] = abs(
                    registered_depth - candidate_depth
                )
    camera_candidates = _build_camera_calibration_candidates(
        evidence,
        similarity_candidate=candidate,
        source_capture_artifact_index_sha256=(
            source_capture_artifact_index_sha256
        ),
    )
    candidate["camera_calibration_candidates"] = camera_candidates
    metrics = _waypoint_metric_summary(evidence, candidate=candidate)
    metrics["camera_calibration_candidates"] = {
        "sha256": camera_candidates.get("sha256"),
        "cameras": {
            camera_id: payload.get("metrics")
            for camera_id, payload in camera_candidates.get("cameras", {}).items()
            if isinstance(payload, Mapping)
        },
    }
    return evidence, metrics, candidate


def build_waypoint_calibration_report(
    run_dir: str | Path,
    *,
    window_before_s: float = 2.0,
    window_after_s: float = 3.0,
    ambiguity_margin: float = 0.12,
    good_error_m: float = 0.5,
    fail_error_m: float = 1.0,
) -> tuple[ValidationReport, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if not (0.0 < float(good_error_m) < float(fail_error_m)):
        raise ValueError("waypoint error thresholds must satisfy 0 < good < fail")
    root = ensure_private_directory(run_dir, label="alignment evidence run")
    verification = verify_run(root)
    session = _load_json_file(root / "session.json", private=True)
    samples = read_ndjson_private(root / "samples.ndjson")
    markers = read_ndjson_private(root / "markers.ndjson")
    waypoint_path = root / "waypoints.json"
    if not waypoint_path.exists():
        raise RuntimeError("guided waypoint calibration requires waypoints.json")
    waypoints = _load_json_file(waypoint_path, private=True)
    calibration, matrix, scene_to_m = _calibration_for_session(root, session)
    assignments = assign_waypoints(
        samples,
        markers,
        waypoints,
        world_to_scene=matrix,
        scene_to_m=scene_to_m,
        window_before_s=window_before_s,
        window_after_s=window_after_s,
        ambiguity_margin=ambiguity_margin,
    )
    calibration_digest = _text(session.get("calibration_bundle_sha256"), maximum=64)
    if calibration_digest is None:
        raise RuntimeError("guided waypoint calibration has no calibration digest")
    source_index_sha256 = file_sha256(root / "artifact_index.json")
    scene_binding = validate_scene_binding(
        session.get("scene_binding")
        if isinstance(session.get("scene_binding"), Mapping)
        else None
    )
    scene_binding_verification = (
        _json_copy(session.get("scene_binding_verification"))
        if isinstance(session.get("scene_binding_verification"), Mapping)
        else {}
    )
    evidence, metrics, candidate = build_waypoint_calibration_evidence(
        samples,
        assignments,
        waypoints,
        calibration=calibration,
        calibration_bundle_sha256=calibration_digest,
        source_capture_artifact_index_sha256=source_index_sha256,
        world_to_scene=matrix,
        scene_to_m=scene_to_m,
        scene_binding=scene_binding,
    )
    metrics["scene_binding"] = _json_copy(scene_binding)
    metrics["scene_binding_verification"] = scene_binding_verification
    candidate["scene_binding"] = _json_copy(scene_binding)
    candidate["scene_binding_verification"] = scene_binding_verification
    metrics["source_capture_artifact_index_sha256"] = source_index_sha256
    candidate["source_capture_artifact_index_sha256"] = source_index_sha256
    for row in evidence:
        row["source_capture_artifact_index_sha256"] = source_index_sha256
    report = ValidationReport(
        run_id=str(session.get("run_id") or root.name),
        source=SourceMetadata(repo="Noesis_Devel"),
        scope={
            "tiers": ["runtime", "menon_scene", "guided_waypoint"],
            "cameras": sorted(
                {str(row.get("camera_id")) for row in evidence if row.get("camera_id")}
            ),
            "privacy": session.get("privacy"),
            "fit_waypoints_drive_candidate": True,
            "holdout_waypoints_drive_candidate": False,
            "candidate_is_advisory_only": True,
            "scene_binding": _json_copy(scene_binding),
            "scene_binding_verification": scene_binding_verification,
        },
    )
    report.add_check(
        ValidationCheck(
            id="ALIGN.waypoint_capture.integrity",
            domain="infrastructure",
            name="guided_waypoint_capture_integrity",
            status=CheckStatus.PASS if verification["ok"] else CheckStatus.FAIL,
            failure_type=None if verification["ok"] else FailureType.INFRASTRUCTURE,
            metric={"errors": verification["errors"], "counts": verification["counts"]},
            detail=(
                "Waypoint source evidence is complete, private, and digest-valid."
                if verification["ok"]
                else "Waypoint source evidence failed verification."
            ),
        )
    )
    binding_state = str(scene_binding_verification.get("status") or "missing")
    if binding_state == "matched":
        binding_check_status = CheckStatus.PASS
        binding_failure_type = None
    elif binding_state == "partial":
        binding_check_status = CheckStatus.WARNING
        binding_failure_type = FailureType.DATA_QUALITY
    elif binding_state == "mismatch":
        binding_check_status = CheckStatus.FAIL
        binding_failure_type = FailureType.CALIBRATION
    else:
        binding_check_status = CheckStatus.BLOCKED
        binding_failure_type = FailureType.DATA_QUALITY
    report.add_check(
        ValidationCheck(
            id="ALIGN.scene_binding",
            domain="calibration",
            name="guided_waypoint_authored_scene_binding",
            status=binding_check_status,
            failure_type=binding_failure_type,
            metric={
                "status": binding_state,
                "scene_binding": scene_binding,
                "verification": scene_binding_verification,
            },
            threshold={
                "full_authoritative_match_required_for_release_claim": True,
                "partial_binding_is_tentative": True,
            },
            detail=(
                "The active transform, authored scene, and scene release are authoritatively bound."
                if binding_state == "matched"
                else "The active transform is verified, while release or authored-scene fields remain operator asserted."
                if binding_state == "partial"
                else "The guided capture is not bound to a verified authored-scene release."
            ),
        )
    )
    if not evidence:
        report.add_check(
            ValidationCheck(
                id="ALIGN.waypoint_contract",
                domain="calibration",
                name="guided_waypoint_contract",
                status=CheckStatus.BLOCKED,
                failure_type=FailureType.DATA_QUALITY,
                detail=(
                    "No waypoint declares both expected_scene_xyz and a fit/holdout split."
                ),
            )
        )
    for row in evidence:
        final_stage = row.get("stages", {}).get("active", {}).get("final_world", {})
        error_m = (
            _finite_float(final_stage.get("error_m"))
            if isinstance(final_stage, Mapping)
            else None
        )
        status = (
            CheckStatus.PASS
            if row.get("status") == "complete"
            else CheckStatus.BLOCKED
        )
        report.add_check(
            ValidationCheck(
                id=f"ALIGN.waypoint_exact.{row.get('waypoint_id')}",
                domain="calibration",
                name="exact_guided_waypoint_correspondence",
                status=status,
                failure_type=(
                    None
                    if status == CheckStatus.PASS
                    else FailureType.DATA_QUALITY
                ),
                camera=_text(row.get("camera_id")),
                metric={
                    "split": row.get("split"),
                    "active_final_error_m": error_m,
                    "binding": row.get("binding"),
                    "stage_errors": {
                        name: stage.get("error_m")
                        for name, stage in row.get("stages", {}).get("active", {}).items()
                        if isinstance(stage, Mapping)
                    },
                    "camera_metrics": row.get("camera_metrics"),
                    "missing_required_fields": row.get("missing_required_fields"),
                },
                threshold={
                    "exact_binding_required": True,
                    "active_final_world_error_is_diagnostic_only": True,
                },
                detail=(
                    "The marked waypoint is bound to one exact calibrated media frame; current-world error is diagnostic only."
                    if row.get("status") == "complete"
                    else "The marked waypoint lacks required exact-frame calibration evidence."
                ),
            )
        )
    fit_count = int(metrics.get("splits", {}).get("fit", {}).get("declared_waypoint_count", 0))
    holdout_count = int(
        metrics.get("splits", {}).get("holdout", {}).get("declared_waypoint_count", 0)
    )
    split_status = (
        CheckStatus.PASS
        if (
            fit_count >= 3
            and holdout_count >= 1
            and all(row.get("status") == "complete" for row in evidence)
        )
        else CheckStatus.BLOCKED
    )
    report.add_check(
        ValidationCheck(
            id="ALIGN.waypoint_split",
            domain="calibration",
            name="guided_waypoint_fit_holdout_split",
            status=split_status,
            failure_type=None if split_status == CheckStatus.PASS else FailureType.DATA_QUALITY,
            metric={
                "fit_waypoint_count": fit_count,
                "holdout_waypoint_count": holdout_count,
                "exact_binding_count": sum(
                    row.get("status") == "complete" for row in evidence
                ),
                "declared_waypoint_count": len(evidence),
                "advisory_similarity_status": candidate.get("status"),
                "advisory_similarity_reason": candidate.get("reason"),
            },
            threshold={"min_fit_waypoints": 3, "min_holdout_waypoints": 1},
            detail=(
                "All exact bindings are complete and the camera solvers retain an untouched holdout."
                if split_status == CheckStatus.PASS
                else "Complete exact bindings, a fit set, and at least one untouched holdout are required."
            ),
        )
    )
    for split in ("fit", "holdout"):
        split_metrics = metrics.get("splits", {}).get(split, {})
        final_distribution = (
            split_metrics.get("candidate_stages", {})
            .get("final_world", {})
            .get("error_m", {})
        )
        coverage = float(
            split_metrics.get("candidate_stages", {})
            .get("final_world", {})
            .get("coverage", 0.0)
        )
        maximum = _finite_float(final_distribution.get("max"))
        report.add_check(
            ValidationCheck(
                id=f"ALIGN.waypoint_candidate.{split}",
                domain="calibration",
                name=f"guided_waypoint_advisory_similarity_diagnostic_{split}",
                status=CheckStatus.PASS,
                failure_type=None,
                metric={
                    **split_metrics,
                    "advisory_similarity_status": candidate.get("status"),
                    "advisory_similarity_reason": candidate.get("reason"),
                    "final_world_coverage": coverage,
                    "final_world_max_error_m": maximum,
                },
                threshold={
                    "admission_gating": False,
                },
                detail=(
                    "Current-world advisory similarity metrics are retained for diagnosis and do not gate camera calibration."
                ),
            )
        )
    camera_candidates = candidate.get("camera_calibration_candidates")
    cameras_payload = (
        camera_candidates.get("cameras")
        if isinstance(camera_candidates, Mapping)
        and isinstance(camera_candidates.get("cameras"), Mapping)
        else {}
    )
    expected_cameras = {
        str(row.get("camera_id"))
        for row in evidence
        if str(row.get("camera_id") or "").strip()
    }
    usable_cameras = {
        str(camera_id)
        for camera_id, payload in cameras_payload.items()
        if isinstance(payload, Mapping)
        and isinstance(payload.get("rotation"), Mapping)
        and payload["rotation"].get("status") == "complete"
        and payload["rotation"].get("admission_status") == "admissible"
        and isinstance(payload.get("depth_registration"), Mapping)
        and payload["depth_registration"].get("status") == "complete"
        and payload["depth_registration"].get("admission_status") == "admissible"
        and isinstance(payload.get("metrics"), Mapping)
        and int(
            payload["metrics"]
            .get("splits", {})
            .get("holdout", {})
            .get("waypoint_count", 0)
        )
        > 0
    }
    camera_set_status = (
        CheckStatus.PASS
        if expected_cameras and usable_cameras == expected_cameras
        else CheckStatus.BLOCKED
    )
    report.add_check(
        ValidationCheck(
            id="ALIGN.camera_candidates.complete",
            domain="calibration",
            name="guided_waypoint_camera_candidate_coverage",
            status=camera_set_status,
            failure_type=(
                None
                if camera_set_status == CheckStatus.PASS
                else FailureType.DATA_QUALITY
            ),
            metric={
                "expected_cameras": sorted(expected_cameras),
                "usable_cameras": sorted(usable_cameras),
                "missing_or_blocked_cameras": sorted(
                    expected_cameras - usable_cameras
                ),
            },
            threshold={
                "all_cameras_require_admissible_rotation_and_depth": True,
                "all_cameras_require_holdout_evidence": True,
            },
            detail=(
                "Every declared camera has admissible fixed-center rotation and physical-depth candidates with holdout evidence."
                if camera_set_status == CheckStatus.PASS
                else "One or more declared cameras lack complete admissible rotation/depth candidates or holdout evidence."
            ),
        )
    )
    for camera_id, camera_payload in sorted(cameras_payload.items()):
        if not isinstance(camera_payload, Mapping):
            continue
        rotation = (
            camera_payload.get("rotation")
            if isinstance(camera_payload.get("rotation"), Mapping)
            else {}
        )
        camera_metrics = (
            camera_payload.get("metrics")
            if isinstance(camera_payload.get("metrics"), Mapping)
            else {}
        )
        holdout_metrics = (
            camera_metrics.get("splits", {}).get("holdout", {})
            if isinstance(camera_metrics.get("splits"), Mapping)
            else {}
        )
        holdout_rotation = (
            holdout_metrics.get("rotation")
            if isinstance(holdout_metrics, Mapping)
            and isinstance(holdout_metrics.get("rotation"), Mapping)
            else {}
        )
        angular_metric = (
            holdout_rotation.get("angular_ray_error_deg")
            if isinstance(holdout_rotation.get("angular_ray_error_deg"), Mapping)
            else {}
        )
        image_metric = (
            holdout_rotation.get("image_reprojection_error_px")
            if isinstance(
                holdout_rotation.get("image_reprojection_error_px"), Mapping
            )
            else {}
        )
        angular_p95 = _finite_float(
            angular_metric.get("distribution", {}).get("p95")
        )
        image_p95 = _finite_float(
            image_metric.get("distribution", {}).get("p95")
        )
        if rotation.get("status") != "complete":
            rotation_status = CheckStatus.BLOCKED
        elif rotation.get("admission_status") != "admissible":
            rotation_status = CheckStatus.FAIL
        elif (
            int(holdout_metrics.get("waypoint_count") or 0) <= 0
            or float(angular_metric.get("coverage") or 0.0) < 1.0
            or float(image_metric.get("coverage") or 0.0) < 1.0
            or angular_p95 is None
            or image_p95 is None
        ):
            rotation_status = CheckStatus.BLOCKED
        elif (
            angular_p95 > _WAYPOINT_ROTATION_FAIL_ANGULAR_DEG
            or image_p95 > _WAYPOINT_ROTATION_FAIL_IMAGE_PX
        ):
            rotation_status = CheckStatus.FAIL
        elif (
            angular_p95 > _WAYPOINT_ROTATION_GOOD_ANGULAR_DEG
            or image_p95 > 12.0
        ):
            rotation_status = CheckStatus.WARNING
        else:
            rotation_status = CheckStatus.PASS
        report.add_check(
            ValidationCheck(
                id=f"ALIGN.camera_rotation.{camera_id}",
                domain="calibration",
                name="guided_waypoint_fixed_center_rotation",
                status=rotation_status,
                failure_type=(
                    None
                    if rotation_status == CheckStatus.PASS
                    else FailureType.CALIBRATION
                    if rotation_status in {CheckStatus.WARNING, CheckStatus.FAIL}
                    else FailureType.DATA_QUALITY
                ),
                camera=str(camera_id),
                metric={
                    "solver": rotation,
                    "fit": camera_metrics.get("splits", {}).get("fit"),
                    "holdout": holdout_metrics,
                },
                threshold={
                    "min_fit_directions": _WAYPOINT_ROTATION_MIN_DIRECTIONS,
                    "good_holdout_p95_angular_deg": (
                        _WAYPOINT_ROTATION_GOOD_ANGULAR_DEG
                    ),
                    "fail_holdout_p95_angular_deg": (
                        _WAYPOINT_ROTATION_FAIL_ANGULAR_DEG
                    ),
                    "good_holdout_p95_image_px": 12.0,
                    "fail_holdout_p95_image_px": (
                        _WAYPOINT_ROTATION_FAIL_IMAGE_PX
                    ),
                    "required_holdout_coverage": 1.0,
                    "required_determinant": 1.0,
                },
                detail=(
                    "A proper rotation preserves the source camera center and is independently evaluated on holdouts."
                ),
            )
        )
        depth = (
            camera_payload.get("depth_registration")
            if isinstance(camera_payload.get("depth_registration"), Mapping)
            else {}
        )
        holdout_depth = (
            holdout_metrics.get("depth")
            if isinstance(holdout_metrics, Mapping)
            and isinstance(holdout_metrics.get("depth"), Mapping)
            else {}
        )
        mapped_metric = (
            holdout_depth.get("mapped_optical_depth_error_m")
            if isinstance(
                holdout_depth.get("mapped_optical_depth_error_m"), Mapping
            )
            else {}
        )
        mapped_distribution = (
            mapped_metric.get("distribution")
            if isinstance(mapped_metric.get("distribution"), Mapping)
            else {}
        )
        mapped_coverage = float(mapped_metric.get("coverage") or 0.0)
        mapped_median = _finite_float(mapped_distribution.get("median"))
        mapped_p95 = _finite_float(mapped_distribution.get("p95"))
        mapped_max = _finite_float(mapped_distribution.get("max"))
        if depth.get("status") != "complete":
            depth_status = CheckStatus.BLOCKED
        elif depth.get("admission_status") != "admissible":
            depth_status = CheckStatus.FAIL
        elif (
            int(holdout_metrics.get("depth_sample_count") or 0) <= 0
            or mapped_coverage < 0.80
            or mapped_median is None
            or mapped_p95 is None
            or mapped_max is None
        ):
            depth_status = CheckStatus.FAIL
        elif (
            mapped_median > MAX_OCCUPIED_MEDIAN_ABS_ERROR_M
            or mapped_p95 > MAX_OCCUPIED_P95_ABS_ERROR_M
            or mapped_max > MAX_OCCUPIED_ABS_ERROR_M
        ):
            depth_status = CheckStatus.FAIL
        elif mapped_median > 0.25 or mapped_p95 > 0.50 or mapped_max > 1.0:
            depth_status = CheckStatus.WARNING
        else:
            depth_status = CheckStatus.PASS
        report.add_check(
            ValidationCheck(
                id=f"ALIGN.camera_depth_registration.{camera_id}",
                domain="calibration",
                name="guided_waypoint_physical_depth_registration",
                status=depth_status,
                failure_type=(
                    None
                    if depth_status == CheckStatus.PASS
                    else FailureType.CALIBRATION
                    if depth_status in {CheckStatus.WARNING, CheckStatus.FAIL}
                    else FailureType.DATA_QUALITY
                ),
                camera=str(camera_id),
                metric={
                    "solver": depth,
                    "fit": camera_metrics.get("splits", {}).get("fit"),
                    "holdout": holdout_metrics,
                },
                threshold={
                    "min_fit_pairs": _WAYPOINT_DEPTH_MIN_FIT_PAIRS,
                    "min_holdout_mapping_coverage": 0.80,
                    "good_holdout_median_error_m": 0.25,
                    "good_holdout_p95_error_m": 0.50,
                    "good_holdout_max_error_m": 1.0,
                    "fail_holdout_median_error_m": (
                        MAX_OCCUPIED_MEDIAN_ABS_ERROR_M
                    ),
                    "fail_holdout_p95_error_m": MAX_OCCUPIED_P95_ABS_ERROR_M,
                    "fail_holdout_max_error_m": MAX_OCCUPIED_ABS_ERROR_M,
                },
                detail=(
                    "A monotonic raw DAv2-to-physical-optical-depth candidate is fit only from fit samples and evaluated on holdouts."
                ),
            )
        )
    return report, evidence, metrics, candidate


def _holdout_gated_candidate_status(
    *,
    fit_solver_status: str,
    holdout_check_statuses: Sequence[CheckStatus],
    expected_check_count: int,
) -> str:
    if (
        fit_solver_status != "admissible"
        or len(holdout_check_statuses) != int(expected_check_count)
    ):
        return "blocked"
    if all(status == CheckStatus.PASS for status in holdout_check_statuses):
        return "admissible"
    if any(status == CheckStatus.FAIL for status in holdout_check_statuses):
        return "rejected"
    if any(status == CheckStatus.BLOCKED for status in holdout_check_statuses):
        return "blocked"
    return "warning"


def write_waypoint_calibration_report(
    run_dir: str | Path,
    *,
    window_before_s: float = 2.0,
    window_after_s: float = 3.0,
    ambiguity_margin: float = 0.12,
    good_error_m: float = 0.5,
    fail_error_m: float = 1.0,
) -> dict[str, Any]:
    root = ensure_private_directory(run_dir, label="alignment evidence run")
    report, evidence, metrics, candidate = build_waypoint_calibration_report(
        root,
        window_before_s=window_before_s,
        window_after_s=window_after_s,
        ambiguity_margin=ambiguity_margin,
        good_error_m=good_error_m,
        fail_error_m=fail_error_m,
    )
    report.artifacts = {
        "waypoint_evidence": "waypoint_calibration_evidence.json",
        "waypoint_metrics": "waypoint_calibration_metrics.json",
        "candidate_similarity": "waypoint_candidate_similarity.json",
        "camera_calibration_candidates": (
            "waypoint_camera_calibration_candidates.json"
        ),
    }
    camera_payloads = (
        candidate.get("camera_calibration_candidates", {}).get("cameras", {})
        if isinstance(candidate.get("camera_calibration_candidates"), Mapping)
        else {}
    )
    camera_candidate_status = (
        "complete"
        if camera_payloads
        and all(
            isinstance(payload, Mapping)
            and isinstance(payload.get("rotation"), Mapping)
            and payload["rotation"].get("status") == "complete"
            and isinstance(payload.get("depth_registration"), Mapping)
            and payload["depth_registration"].get("status") == "complete"
            for payload in camera_payloads.values()
        )
        else "blocked"
    )
    fit_solver_status = (
        "admissible"
        if camera_candidate_status == "complete"
        and all(
            payload["rotation"].get("admission_status") == "admissible"
            and payload["depth_registration"].get("admission_status")
            == "admissible"
            for payload in camera_payloads.values()
            if isinstance(payload, Mapping)
        )
        else "blocked"
    )
    holdout_check_statuses = [
        check.status
        for check in report.checks
        if check.id.startswith("ALIGN.camera_rotation.")
        or check.id.startswith("ALIGN.camera_depth_registration.")
    ]
    calibration_candidate_status = _holdout_gated_candidate_status(
        fit_solver_status=fit_solver_status,
        holdout_check_statuses=holdout_check_statuses,
        expected_check_count=2 * len(camera_payloads),
    )
    candidate_summary = {
        "fit_solver_status": fit_solver_status,
        "calibration_candidate_status": calibration_candidate_status,
        "advisory_only": True,
        "active_config_modified": False,
        "camera_candidate_status": camera_candidate_status,
        "advisory_similarity_status": candidate.get("status"),
    }
    candidate["candidate_summary"] = candidate_summary
    outputs = {
        "waypoint_calibration_evidence.json": evidence,
        "waypoint_calibration_metrics.json": metrics,
        "waypoint_candidate_similarity.json": candidate,
        "waypoint_camera_calibration_candidates.json": candidate.get(
            "camera_calibration_candidates", {}
        ),
        "waypoint_calibration_report.json": report.to_dict(),
    }
    for name, payload in outputs.items():
        atomic_write_private_file(
            root / name,
            json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n",
            label="guided waypoint calibration artifact",
        )
    atomic_write_private_file(
        root / "waypoint_calibration_report.md",
        render_markdown(report).encode("utf-8"),
        label="guided waypoint calibration markdown",
    )
    derived_artifacts = {
        name: {
            "sha256": file_sha256(root / name),
            "size_bytes": (root / name).stat().st_size,
        }
        for name in (
            "waypoint_calibration_evidence.json",
            "waypoint_calibration_metrics.json",
            "waypoint_candidate_similarity.json",
            "waypoint_camera_calibration_candidates.json",
            "waypoint_calibration_report.json",
            "waypoint_calibration_report.md",
        )
    }
    atomic_write_private_file(
        root / "waypoint_calibration_artifact_index.json",
        json.dumps(
            {
                "contract": "noesis.alignment.waypoint_artifact_index",
                "contract_version": CONTRACT_VERSION,
                "source_capture_artifact_index_sha256": file_sha256(
                    root / "artifact_index.json"
                ),
                "artifacts": derived_artifacts,
            },
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n",
        label="guided waypoint calibration artifact index",
    )
    return {
        "status": report.status.value,
        "level": report.level.value,
        "report": str(root / "waypoint_calibration_report.json"),
        "markdown": str(root / "waypoint_calibration_report.md"),
        "waypoint_count": len(evidence),
        "fit_waypoint_count": metrics.get("splits", {}).get("fit", {}).get(
            "declared_waypoint_count", 0
        ),
        "holdout_waypoint_count": metrics.get("splits", {})
        .get("holdout", {})
        .get("declared_waypoint_count", 0),
        "advisory_similarity_status": candidate.get("status"),
        "camera_candidate_status": camera_candidate_status,
        "fit_solver_status": fit_solver_status,
        "calibration_candidate_status": calibration_candidate_status,
        "candidate_summary": candidate_summary,
    }


def build_alignment_report(
    run_dir: str | Path,
    *,
    room_zones_path: str | Path,
    window_before_s: float = 2.0,
    window_after_s: float = 3.0,
    ambiguity_margin: float = 0.12,
) -> tuple[ValidationReport, list[dict[str, Any]], list[dict[str, Any]]]:
    root = ensure_private_directory(run_dir, label="alignment evidence run")
    verification = verify_run(root)
    session = _load_json_file(root / "session.json", private=True)
    samples = read_ndjson_private(root / "samples.ndjson")
    markers = read_ndjson_private(root / "markers.ndjson")
    waypoints_path = root / "waypoints.json"
    waypoints = _load_json_file(waypoints_path, private=True) if waypoints_path.exists() else {
        "contract": WAYPOINT_CONTRACT,
        "contract_version": CONTRACT_VERSION,
        "waypoints": [],
    }
    _calibration, matrix, scene_to_m = _calibration_for_session(root, session)
    zones = _load_room_zones(room_zones_path)
    assignments = assign_waypoints(
        samples,
        markers,
        waypoints,
        world_to_scene=matrix,
        scene_to_m=scene_to_m,
        window_before_s=window_before_s,
        window_after_s=window_after_s,
        ambiguity_margin=ambiguity_margin,
    )
    report = ValidationReport(
        run_id=str(session.get("run_id") or root.name),
        source=SourceMetadata(repo="Noesis_Devel"),
        scope={
            "tiers": ["runtime", "menon_scene"],
            "cameras": sorted({str(row.get("camera_id")) for row in samples if row.get("camera_id")}),
            "privacy": session.get("privacy"),
        },
    )
    report.add_check(
        ValidationCheck(
            id="ALIGN.capture.integrity",
            domain="infrastructure",
            name="alignment_capture_integrity",
            status=CheckStatus.PASS if verification["ok"] else CheckStatus.FAIL,
            failure_type=None if verification["ok"] else FailureType.INFRASTRUCTURE,
            metric={"errors": verification["errors"], "counts": verification["counts"]},
            detail="Capture evidence is complete, private, and digest-valid." if verification["ok"] else "Capture evidence failed verification.",
        )
    )
    declared_rooms_by_camera: dict[str, list[str]] = defaultdict(list)
    for waypoint in waypoints.get("waypoints", []):
        if isinstance(waypoint, Mapping):
            declared_rooms_by_camera[str(waypoint.get("camera_id") or "")].extend(
                str(room) for room in waypoint.get("rooms", [])
            )
    samples_by_camera: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in samples:
        camera = _text(row.get("camera_id"))
        if camera:
            samples_by_camera[camera].append(row)
    for camera, rows in sorted(samples_by_camera.items()):
        targets = _target_zones(camera, declared_rooms_by_camera.get(camera, []), zones)
        if not targets:
            report.add_check(
                ValidationCheck(
                    id=f"ALIGN.room.{camera}",
                    domain="geometry",
                    name="camera_track_room_containment",
                    status=CheckStatus.BLOCKED,
                    failure_type=FailureType.INFRASTRUCTURE,
                    camera=camera,
                    metric={"sample_count": len(rows), "matched_room_zone_count": 0},
                    detail="No Menon room zone matches this camera or its declared waypoint rooms.",
                )
            )
            continue
        distances_m: list[float] = []
        valid_count = 0
        for row in rows:
            point = _finite_vector(row.get("world"), 3)
            if point is None or row.get("world_valid") is not True:
                continue
            scene = transform_point(point, matrix)
            distances_m.append(_point_zone_distance_scene([scene[0], scene[2]], targets) * scene_to_m)
            valid_count += 1
        inside_count = sum(distance <= 1e-9 for distance in distances_m)
        ratio = float(inside_count / valid_count) if valid_count else 0.0
        status = CheckStatus.PASS if valid_count and ratio >= 0.95 else CheckStatus.WARNING if valid_count and ratio > 0 else CheckStatus.FAIL if valid_count else CheckStatus.BLOCKED
        report.add_check(
            ValidationCheck(
                id=f"ALIGN.room.{camera}",
                domain="geometry",
                name="camera_track_room_containment",
                status=status,
                failure_type=None if status == CheckStatus.PASS else FailureType.PROJECTION,
                camera=camera,
                metric={
                    "sample_count": len(rows),
                    "world_valid_count": valid_count,
                    "inside_count": inside_count,
                    "inside_ratio": ratio,
                    "max_outside_distance_m": max(distances_m) if distances_m else None,
                },
                threshold={"min_inside_ratio": 0.95},
                detail="World-valid samples land in the camera's declared room zones." if status == CheckStatus.PASS else "Some or all world-valid samples land outside the camera's declared room zones.",
            )
        )
    for assignment in assignments:
        status_text = str(assignment.get("status"))
        if status_text == "assigned":
            selected = assignment.get("selected") if isinstance(assignment.get("selected"), Mapping) else {}
            error_m = _finite_float(selected.get("error_m"))
            if error_m is None or error_m <= 0.5:
                status = CheckStatus.PASS
            elif error_m <= 1.0:
                status = CheckStatus.WARNING
            else:
                status = CheckStatus.FAIL
        elif status_text == "ambiguous":
            status = CheckStatus.WARNING
        else:
            status = CheckStatus.BLOCKED
        report.add_check(
            ValidationCheck(
                id=f"ALIGN.waypoint.{assignment.get('waypoint_id')}",
                domain="geometry",
                name="guided_waypoint_assignment",
                status=status,
                failure_type=None if status == CheckStatus.PASS else FailureType.PROJECTION if status == CheckStatus.FAIL else FailureType.DATA_QUALITY,
                camera=_text(assignment.get("camera_id")),
                metric={
                    "assignment_status": status_text,
                    "score_margin": assignment.get("score_margin"),
                    "selected": assignment.get("selected"),
                    "candidate_count": len(assignment.get("candidates") or []),
                },
                threshold={"good_error_m": 0.5, "fail_error_m": 1.0, "ambiguity_margin": ambiguity_margin},
                detail="Waypoint was assigned to one run-local tracklet." if status_text == "assigned" else "Waypoint assignment is ambiguous or lacks usable candidates.",
            )
        )
    audit = _track_audit(samples, matrix)
    return report, assignments, audit


def write_alignment_report(
    run_dir: str | Path,
    *,
    room_zones_path: str | Path,
    window_before_s: float = 2.0,
    window_after_s: float = 3.0,
    ambiguity_margin: float = 0.12,
) -> dict[str, Any]:
    root = ensure_private_directory(run_dir, label="alignment evidence run")
    report, assignments, audit = build_alignment_report(
        root,
        room_zones_path=room_zones_path,
        window_before_s=window_before_s,
        window_after_s=window_after_s,
        ambiguity_margin=ambiguity_margin,
    )
    report.artifacts = {
        "waypoint_assignments": "waypoint_assignments.json",
        "track_audit": "track_audit.json",
    }
    atomic_write_private_file(
        root / "waypoint_assignments.json",
        json.dumps(assignments, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        label="alignment waypoint assignments",
    )
    atomic_write_private_file(
        root / "track_audit.json",
        json.dumps(audit, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        label="alignment track audit",
    )
    atomic_write_private_file(
        root / "report.json",
        json.dumps(report.to_dict(), indent=2, sort_keys=True).encode("utf-8") + b"\n",
        label="alignment validation report",
    )
    atomic_write_private_file(
        root / "report.md",
        render_markdown(report).encode("utf-8"),
        label="alignment validation report markdown",
    )
    return {
        "status": report.status.value,
        "level": report.level.value,
        "report": str(root / "report.json"),
        "markdown": str(root / "report.md"),
        "assignment_count": len(assignments),
        "tracklet_count": len(audit),
    }


__all__ = [
    "AlignmentWalkCapture",
    "CAPTURE_CONTRACT",
    "CONTRACT_VERSION",
    "EphemeralIds",
    "MARKER_CONTRACT",
    "SAMPLE_CONTRACT",
    "WAYPOINT_CONTRACT",
    "WAYPOINT_EVIDENCE_CONTRACT",
    "WAYPOINT_METRICS_CONTRACT",
    "WORLD_CONTRACT",
    "append_waypoint_marker",
    "assign_waypoints",
    "build_alignment_report",
    "build_waypoint_calibration_evidence",
    "build_waypoint_calibration_report",
    "create_run_directory",
    "default_output_root",
    "read_ndjson_private",
    "sanitize_tracking_message",
    "validate_waypoint_manifest",
    "verify_run",
    "write_alignment_report",
    "write_waypoint_calibration_report",
]
