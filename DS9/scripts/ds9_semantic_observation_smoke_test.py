#!/usr/bin/env python3
"""Fail-closed occupied-scene semantic observation gate for DS9.

The pass unit is one bounded identity-continuous cohort.  It starts from an
exact public-track/canonical-observation/private-evidence embedding anchor and
may join pose, usable registered depth, and backend-world components from
other frames only when every frame remains on the same runtime, source,
camera, tracker, immutable identity, and artifact fingerprints for at most
1.5 seconds.  The report contains bounded scalar provenance and never copies
embeddings or bearer bytes.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis_core.contracts.identity_calibration import (  # noqa: E402
    ShadowIdentityEvidenceRecord,
)
from noesis_core.contracts.observation import ObservationEnvelope  # noqa: E402
from noesis_core.private_paths import (  # noqa: E402
    PrivatePathError,
    atomic_create_private_file,
    read_private_file,
    require_fresh_private_file_bundle,
)
from noesis_core.strict_json import (  # noqa: E402
    StrictJSONError,
    strict_json_loads,
)
from noesis_core.tracking_continuity import (  # noqa: E402
    TRACKER_LIFECYCLE_GENERATION_FIELD,
    TRACKING_CONTINUITY_CONTRACT,
    TRACKING_CONTINUITY_CONTRACT_VERSION,
)
from scripts.internal_auth_client import (  # noqa: E402
    RequiredInternalAuth,
    add_auth_token_file_argument,
    connect_required_websocket,
    load_required_internal_auth,
)

CONTRACT = "noesis.ds9.semantic-observation-live-gate"
SCHEMA_VERSION = 3
CONTRACT_VERSION = 3
CANONICAL_REPORT_FILENAME = "semantic-observation.json"
CANONICAL_IDENTITY_SNAPSHOT_FILENAME = "semantic-identity-evidence.jsonl"
CANONICAL_SOURCE_TRANSCRIPT_FILENAME = "semantic-observation-source.json"
SOURCE_TRANSCRIPT_CONTRACT = "noesis.ds9.semantic-observation-source-transcript"
SOURCE_TRANSCRIPT_SCHEMA_VERSION = 3
SOURCE_TRANSCRIPT_CONTRACT_VERSION = 3
SESSION_RE = re.compile(r"^[a-z0-9][a-z0-9-]{5,47}$")
RUNTIME_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
EXPECTED_EMBEDDING_DIMENSION = 256
SEMANTIC_COHORT_MAX_SPAN_US = 1_500_000
SEMANTIC_CAPTURE_PRE_WINDOW_LEEWAY_US = 2_000_000
MAX_EVIDENCE_BYTES = 256 * 1024 * 1024
MAX_REPORT_BYTES = 256 * 1024
MAX_SOURCE_TRANSCRIPT_BYTES = 64 * 1024 * 1024
MAX_SOURCE_MESSAGES = 4096
MAX_WS_MESSAGE_BYTES = 8 * 1024 * 1024
MAX_LINKS = 4096
MAX_ERRORS = 64
MAX_ERROR_CHARS = 240
RAW_VECTOR_KEYS = frozenset(
    {
        "embedding",
        "embedding_vector",
        "embedding_values",
        "feature_vector",
        "reid_embedding",
        "vector",
    }
)
RAW_VECTOR_KEY_TOKENS = frozenset(
    {
        "appearance",
        "descriptor",
        "embedding",
        "featurevector",
        "latent",
        "reid",
        "representation",
    }
)
AUTHENTICATION_KEY_TOKENS = frozenset(
    {
        "authorization",
        "bearer",
        "cookie",
        "credential",
        "password",
        "secret",
        "token",
    }
)
ALLOWED_EMBEDDING_SCALAR_FIELDS = frozenset(
    {
        "embeddingdimension",
        "embeddingdim",
        "embeddingmodellayer",
        "embeddingmodelsemanticprofilesha256",
        "embeddingmodelsha256",
        "embeddingpresent",
        "embeddingsequence",
        "freshembedding",
    }
)
AUTHENTICATION_VALUE_RE = re.compile(
    r"(?i)(?:^|[\s;,])(?:authorization|bearer|basic|cookie|credential|"
    r"password|secret|token)\s*(?::|=|\s)\s*\S+"
)
SOURCE_REDACTION_MARKER = "_semantic_source_redaction"
PROVENANCE_FIELDS = (
    "embedding_sequence",
    "embedding_model_sha256",
    "embedding_dimension",
)
DEPTH_FIELDS = (
    "depth_used_m",
    "depth_registered_m",
    "depth_anchor_m",
    "depth_median_m",
    "depth_center_m",
)
SOURCE_PRIVACY_POLICY = {
    "raw_embedding_vectors": "forbidden",
    "authentication_material": "excluded",
    "transcript_shape": "strict_semantic_projection_v1",
    "embedding_sized_numeric_vectors": "forbidden",
    "maximum_messages": MAX_SOURCE_MESSAGES,
}


def _acquisition_window_document(
    *,
    started_at_us: int,
    finished_at_us: int,
) -> dict[str, object]:
    if not _is_int(started_at_us) or int(started_at_us) <= 0:
        raise ValueError("acquisition started_at_us must be a positive integer")
    if not _is_int(finished_at_us) or int(finished_at_us) < int(started_at_us):
        raise ValueError("acquisition finished_at_us must not precede its start")
    return {
        "started_at_us": int(started_at_us),
        "finished_at_us": int(finished_at_us),
        "capture_pre_window_leeway_us": SEMANTIC_CAPTURE_PRE_WINDOW_LEEWAY_US,
        "observed_time_bounds": "closed_acquisition_window_v1",
        "capture_time_bounds": "closed_window_with_pre_receive_leeway_v1",
    }


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_finite(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0.0 else None


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _normalized_key(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def _numeric_leaf_count(value: object, *, limit: int) -> int | None:
    """Count a pure numeric tensor, including JSON numeric-string encodings."""

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return 1
    if isinstance(value, str):
        try:
            float(value)
        except (TypeError, ValueError):
            return None
        return 1
    if isinstance(value, Mapping):
        children = tuple(value.values())
    elif isinstance(value, (list, tuple)):
        children = tuple(value)
    else:
        return None
    if not children:
        return None
    count = 0
    for child in children:
        child_count = _numeric_leaf_count(child, limit=limit - count)
        if child_count is None:
            return None
        count += child_count
        if count > limit:
            return count
    return count


def _numeric_vector(value: object) -> bool:
    if not isinstance(value, (Mapping, list, tuple)):
        return False
    return (
        _numeric_leaf_count(value, limit=EXPECTED_EMBEDDING_DIMENSION)
        == EXPECTED_EMBEDDING_DIMENSION
    )


def _allowed_embedding_scalar(key: object, value: object) -> bool:
    normalized = _normalized_key(key)
    if normalized not in ALLOWED_EMBEDDING_SCALAR_FIELDS:
        return False
    if normalized in {"embeddingpresent", "freshembedding"}:
        return type(value) is bool  # noqa: E721 - exact public wire type
    if normalized in {"embeddingsequence", "embeddingdimension", "embeddingdim"}:
        return _is_int(value) and int(value) >= 0
    if normalized == "embeddingmodellayer":
        return isinstance(value, str) and bool(value.strip()) and len(value) <= 200
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _authentication_key(key: object) -> bool:
    normalized = _normalized_key(key)
    return any(token in normalized for token in AUTHENTICATION_KEY_TOKENS)


def _authentication_value(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and AUTHENTICATION_VALUE_RE.search(value.strip()) is not None
    )


def _allowed_privacy_declaration(key: object, value: object) -> bool:
    if not isinstance(value, str):
        return False
    return (_normalized_key(key), value) in {
        ("authenticationmaterial", "excluded"),
        ("embeddingsizednumericvectors", "forbidden"),
        ("rawembeddingvectors", "forbidden"),
    }


def _sensitive_named_vector(key: object, value: object) -> bool:
    normalized = _normalized_key(key)
    if _allowed_embedding_scalar(key, value):
        return False
    sensitive_name = (
        str(key).casefold() in RAW_VECTOR_KEYS
        or any(token in normalized for token in RAW_VECTOR_KEY_TOKENS)
    )
    if not sensitive_name:
        return False
    if isinstance(value, (str, bytes, bytearray, list, tuple)):
        return bool(value)
    if isinstance(value, Mapping):
        return bool(value)
    return False


def _raw_vector_paths(value: object, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            child = f"{path}.{key}"
            if (
                (
                    _authentication_key(key)
                    and not _allowed_privacy_declaration(key, item)
                )
                or _authentication_value(item)
                or (
                    _sensitive_named_vector(key, item)
                    and not _allowed_privacy_declaration(key, item)
                )
                or _numeric_vector(item)
            ):
                found.append(child)
            if len(found) < 8:
                found.extend(_raw_vector_paths(item, child))
            if len(found) >= 8:
                break
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            if _authentication_value(item):
                found.append(f"{path}[{index}]")
            found.extend(_raw_vector_paths(item, f"{path}[{index}]"))
            if len(found) >= 8:
                break
    return found[:8]


def _provenance(value: Mapping[str, Any], label: str) -> tuple[int, str, int] | None:
    present = tuple(value.get(field) is not None for field in PROVENANCE_FIELDS)
    if any(present) and not all(present):
        raise ValueError(f"{label} contains a partial embedding provenance triad")
    if not any(present):
        return None
    sequence = value.get("embedding_sequence")
    model = value.get("embedding_model_sha256")
    dimension = value.get("embedding_dimension")
    if not _is_int(sequence) or int(sequence) < 0:
        raise ValueError(f"{label} embedding_sequence must be nonnegative")
    if (
        not isinstance(model, str)
        or len(model) != 64
        or any(character not in "0123456789abcdef" for character in model)
    ):
        raise ValueError(f"{label} embedding_model_sha256 must be lowercase SHA-256")
    if dimension != EXPECTED_EMBEDDING_DIMENSION:
        raise ValueError(
            f"{label} embedding_dimension must be {EXPECTED_EMBEDDING_DIMENSION}"
        )
    return int(sequence), model, int(dimension)


def _usable_depth(track: Mapping[str, Any]) -> tuple[str, float] | None:
    if str(track.get("depth_status") or "").strip().lower() != "ok":
        return None
    registration = str(track.get("depth_registration_status") or "").strip().lower()
    if registration != "ok":
        return None
    registered = _positive_finite(track.get("depth_registered_m"))
    if registered is None:
        return None
    used_raw = track.get("depth_used_m")
    if used_raw is not None:
        used = _positive_finite(used_raw)
        if used is None or not math.isclose(
            used,
            registered,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            return None
    return "depth_registered_m", registered


def _world_xyz(track: Mapping[str, Any]) -> tuple[float, float, float] | None:
    if track.get("world_valid") is not True:
        return None
    if track.get("world_frame") != "backend_world_m":
        return None
    raw = track.get("world")
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        return None
    try:
        values = tuple(float(item) for item in raw)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in values):
        return None
    return values  # type: ignore[return-value]


def _load_reviewed_reid_contract(path: Path) -> tuple[str, int]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise ValueError(f"unable to load reviewed pipeline config: {exc}") from exc
    models = payload.get("models") if isinstance(payload, Mapping) else None
    reid = models.get("reid") if isinstance(models, Mapping) else None
    if not isinstance(reid, Mapping) or reid.get("enable") is not True:
        raise ValueError("reviewed pipeline config must enable models.reid")
    layer = str(reid.get("layer") or "").strip()
    dimension = reid.get("embedding_dim")
    if not layer:
        raise ValueError("reviewed pipeline config must declare models.reid.layer")
    if dimension != EXPECTED_EMBEDDING_DIMENSION:
        raise ValueError(
            "reviewed pipeline config must declare models.reid.embedding_dim=256"
        )
    return layer, int(dimension)


@dataclass(frozen=True)
class IdentityContinuity:
    state: str
    subject_id: str
    compatibility_sid: int
    resident_uuid: str | None
    visitor_generation: int | None

    def as_report(self) -> dict[str, object]:
        return {
            "state": self.state,
            "subject_id": self.subject_id,
            "compatibility_sid": self.compatibility_sid,
            "resident_uuid": self.resident_uuid,
            "visitor_generation": self.visitor_generation,
        }


@dataclass(frozen=True)
class SemanticFingerprints:
    calibration_sha256: str
    model_sha256: str
    config_sha256: str

    def as_report(self) -> dict[str, str]:
        return {
            "calibration_sha256": self.calibration_sha256,
            "model_sha256": self.model_sha256,
            "config_sha256": self.config_sha256,
        }


@dataclass(frozen=True)
class SemanticObservationSample:
    source_id: int
    camera_id: str
    tracker_id: int
    tracker_lifecycle_generation: int
    frame_id: int
    run_id: str
    producer_runtime: str
    producer_instance_id: str
    canonical_observation_id: str
    observation_sequence: int
    captured_at_us: int
    observed_at_us: int
    published_at_us: int
    capture_time_status: str
    media_pts_ns: int
    fingerprints: SemanticFingerprints
    identity: IdentityContinuity | None
    identity_fresh_embedding: bool
    identity_observation_id: str | None
    embedding_sequence: int | None
    embedding_model_sha256: str | None
    embedding_dimension: int | None
    pose_present: bool
    depth_present: bool
    depth_metric_name: str | None
    depth_metric_value: float | None
    world_present: bool

    @property
    def tracker_key(self) -> tuple[str, int, str, int, int]:
        return (
            self.run_id,
            self.source_id,
            self.camera_id,
            self.tracker_id,
            self.tracker_lifecycle_generation,
        )

    @property
    def association_key(self) -> tuple[int, str, int, int]:
        return self.source_id, self.camera_id, self.tracker_id, self.frame_id

    @property
    def has_embedding_provenance(self) -> bool:
        return all(
            value is not None
            for value in (
                self.identity_observation_id,
                self.embedding_sequence,
                self.embedding_model_sha256,
                self.embedding_dimension,
            )
        )


@dataclass(frozen=True)
class TrackingFramePresence:
    source_id: int
    frame_id: int
    publication_sequence: int
    captured_at_us: int
    observed_at_us: int
    media_pts_ns: int
    trackers: frozenset[tuple[str, int, int]]


@dataclass(frozen=True)
class PersistedAnchor:
    sample: SemanticObservationSample
    evidence: ShadowIdentityEvidenceRecord


@dataclass(frozen=True)
class SemanticCohort:
    anchor: PersistedAnchor
    pose: SemanticObservationSample
    depth: SemanticObservationSample
    world: SemanticObservationSample

    @property
    def started_at_us(self) -> int:
        return min(
            self.anchor.sample.observed_at_us,
            self.pose.observed_at_us,
            self.depth.observed_at_us,
            self.world.observed_at_us,
        )

    @property
    def finished_at_us(self) -> int:
        return max(
            self.anchor.sample.observed_at_us,
            self.pose.observed_at_us,
            self.depth.observed_at_us,
            self.world.observed_at_us,
        )

    @property
    def span_us(self) -> int:
        return self.finished_at_us - self.started_at_us

    @property
    def capture_started_at_us(self) -> int:
        return min(
            self.anchor.sample.captured_at_us,
            self.pose.captured_at_us,
            self.depth.captured_at_us,
            self.world.captured_at_us,
        )

    @property
    def capture_finished_at_us(self) -> int:
        return max(
            self.anchor.sample.captured_at_us,
            self.pose.captured_at_us,
            self.depth.captured_at_us,
            self.world.captured_at_us,
        )

    @property
    def capture_span_us(self) -> int:
        return self.capture_finished_at_us - self.capture_started_at_us

    @property
    def media_pts_span_ns(self) -> int:
        values = (
            self.anchor.sample.media_pts_ns,
            self.pose.media_pts_ns,
            self.depth.media_pts_ns,
            self.world.media_pts_ns,
        )
        return max(values) - min(values)


def _identity_continuity(track: Mapping[str, Any]) -> tuple[IdentityContinuity | None, bool]:
    raw = track.get("identity_v2")
    if not isinstance(raw, Mapping):
        raise ValueError("identity_v2 is missing")
    if raw.get("mode") != "shadow":
        raise ValueError("identity_v2.mode must be shadow")
    fresh = raw.get("fresh_embedding")
    if type(fresh) is not bool:  # noqa: E721 - exact wire type
        raise ValueError("identity_v2.fresh_embedding must be boolean")
    state = str(raw.get("state") or "").strip().lower()
    subject_id = raw.get("subject_id")
    compatibility_sid = raw.get("compatibility_sid")
    resident_uuid = raw.get("resident_uuid")
    visitor_generation = raw.get("visitor_generation")
    if state in {"unknown", "provisional"}:
        if any(
            value is not None
            for value in (
                subject_id,
                compatibility_sid,
                resident_uuid,
                visitor_generation,
            )
        ):
            raise ValueError("unknown/provisional identity_v2 must not carry identity")
        return None, fresh
    if state not in {"resident", "visitor"}:
        raise ValueError("identity_v2.state is invalid")
    subject = str(subject_id or "").strip()
    if not subject or len(subject) > 240:
        raise ValueError("identity_v2.subject_id is invalid")
    if not _is_int(compatibility_sid) or int(compatibility_sid) <= 0:
        raise ValueError("identity_v2.compatibility_sid must be positive")
    if state == "resident":
        resident = str(resident_uuid or "").strip()
        if not resident or subject != f"resident:{resident}":
            raise ValueError("resident identity_v2 subject/UUID is incoherent")
        if visitor_generation is not None:
            raise ValueError("resident identity_v2 must not carry visitor generation")
        generation = None
    else:
        if resident_uuid is not None:
            raise ValueError("visitor identity_v2 must not carry resident UUID")
        if not _is_int(visitor_generation) or int(visitor_generation) < 0:
            raise ValueError("visitor identity_v2 generation is invalid")
        generation = int(visitor_generation)
        prefix = "visitor:"
        suffix = f":generation:{generation}"
        if (
            not subject.startswith(prefix)
            or not subject.endswith(suffix)
            or len(subject) <= len(prefix) + len(suffix)
        ):
            raise ValueError("visitor identity_v2 subject/generation is incoherent")
        resident = None
    return (
        IdentityContinuity(
            state=state,
            subject_id=subject,
            compatibility_sid=int(compatibility_sid),
            resident_uuid=resident,
            visitor_generation=generation,
        ),
        fresh,
    )


def _semantic_fingerprints(observation: ObservationEnvelope) -> SemanticFingerprints:
    if (
        observation.calibration.role != "camera_calibration"
        or observation.model.role != "tracking_model_manifest"
        or observation.config.role != "pipeline_config"
    ):
        raise ValueError("canonical observation artifact fingerprint roles drifted")
    return SemanticFingerprints(
        calibration_sha256=observation.calibration.sha256,
        model_sha256=observation.model.sha256,
        config_sha256=observation.config.sha256,
    )


TRACK_PROJECTION_FIELDS = (
    "class_id",
    "camera_id",
    "tracker_id",
    TRACKER_LIFECYCLE_GENERATION_FIELD,
    "frame_id",
    "observed_at_us",
    "embedding_present",
    "identity_observation_key",
    *PROVENANCE_FIELDS,
    "identity_v2",
    "pose_present",
    "depth_status",
    "depth_registration_status",
    "depth_registered_m",
    "depth_used_m",
    "world_valid",
    "world_frame",
    "world",
)
IDENTITY_V2_PROJECTION_FIELDS = (
    "mode",
    "state",
    "subject_id",
    "compatibility_sid",
    "resident_uuid",
    "visitor_generation",
    "fresh_embedding",
)
IDENTITY_OBSERVATION_KEY_PROJECTION_FIELDS = (
    "run_id",
    "observation_id",
    "camera_id",
    "tracker_id",
    "frame_id",
)
TOMBSTONE_PROJECTION_FIELDS = (
    "camera_id",
    "tracker_id",
    TRACKER_LIFECYCLE_GENERATION_FIELD,
    "last_seen_frame_id",
    "last_seen_observed_at_us",
    "disappeared_at_frame_id",
    "disappeared_at_observed_at_us",
)


def _mapping_projection(
    value: object,
    fields: tuple[str, ...],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {field: value[field] for field in fields if field in value}


def _canonical_projection_value(value: object) -> object:
    """Keep retained source evidence inside the strict JSON value domain."""

    if isinstance(value, float) and not math.isfinite(value):
        return "__noesis_nonfinite_number__"
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_projection_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_projection_value(item) for item in value]
    return value


def _semantic_source_projection(payload: Mapping[str, Any]) -> dict[str, object]:
    message_type = str(payload.get("type") or "")
    marker = payload.get(SOURCE_REDACTION_MARKER)
    if marker is not None:
        marker_projection = _mapping_projection(
            marker,
            ("raw_vector_field_count",),
        )
        return {
            "type": message_type,
            SOURCE_REDACTION_MARKER: marker_projection,
        }
    if message_type == "semantic_source_rejection":
        return {
            "type": "semantic_source_rejection",
            "reason": payload.get("reason"),
        }
    if message_type == "stats":
        raw_payload = payload.get("payload")
        raw_pipeline = (
            raw_payload.get("pipeline") if isinstance(raw_payload, Mapping) else None
        )
        return {
            "type": "stats",
            "payload": {
                "pipeline": {
                    "errors": (
                        raw_pipeline.get("errors")
                        if isinstance(raw_pipeline, Mapping)
                        else None
                    )
                }
            },
        }
    if message_type != "tracking":
        return {"type": message_type}

    tracks_raw = payload.get("tracks")
    tracks: object
    if isinstance(tracks_raw, list):
        tracks = []
        for raw_track in tracks_raw:
            track = _mapping_projection(raw_track, TRACK_PROJECTION_FIELDS)
            if "identity_v2" in track:
                track["identity_v2"] = _mapping_projection(
                    track["identity_v2"],
                    IDENTITY_V2_PROJECTION_FIELDS,
                )
            if "identity_observation_key" in track:
                track["identity_observation_key"] = _mapping_projection(
                    track["identity_observation_key"],
                    IDENTITY_OBSERVATION_KEY_PROJECTION_FIELDS,
                )
            tracks.append(track)
    else:
        tracks = tracks_raw
    observations_raw = payload.get("observations")
    observations: object
    if isinstance(observations_raw, list):
        observations = []
        for raw in observations_raw:
            try:
                if not isinstance(raw, Mapping):
                    raise ValueError("observation must be an object")
                observations.append(
                    ObservationEnvelope.model_validate(raw).model_dump(mode="json")
                )
            except Exception:
                observations.append({})
    else:
        observations = observations_raw
    tombstones_raw = payload.get("tracker_lifecycle_tombstones")
    tombstones = (
        [
            _mapping_projection(item, TOMBSTONE_PROJECTION_FIELDS)
            for item in tombstones_raw
        ]
        if isinstance(tombstones_raw, list)
        else tombstones_raw
    )
    projected: dict[str, object] = {
        "type": "tracking",
        "source_id": payload.get("source_id"),
        "frame_id": payload.get("frame_id"),
        "captured_at_us": payload.get("captured_at_us"),
        "observed_at_us": payload.get("observed_at_us"),
        "capture_time_status": payload.get("capture_time_status"),
        "media_pts_ns": payload.get("media_pts_ns"),
        "tracking_continuity_contract": payload.get(
            "tracking_continuity_contract"
        ),
        "tracking_continuity_contract_version": payload.get(
            "tracking_continuity_contract_version"
        ),
        "tracking_publication_sequence": payload.get(
            "tracking_publication_sequence"
        ),
        "tracker_lifecycle_tombstones": tombstones,
        "track_count": payload.get("track_count"),
        "tracks": tracks,
        "observation_contract": payload.get("observation_contract"),
        "observation_contract_version": payload.get(
            "observation_contract_version"
        ),
        "observations": observations,
    }
    return projected


@dataclass
class SemanticObservationCollector:
    tracking_messages: int = 0
    stats_messages: int = 0
    person_tracks: int = 0
    canonical_observations: int = 0
    raw_vector_fields: int = 0
    unanchored_origin_tombstones: int = 0
    pipeline_errors: set[str] = field(default_factory=set)
    samples: list[SemanticObservationSample] = field(default_factory=list)
    frames: list[TrackingFramePresence] = field(default_factory=list)
    canonical_publication_clocks: list[tuple[int, int]] = field(
        default_factory=list
    )
    errors: list[str] = field(default_factory=list)
    source_messages: list[dict[str, object]] = field(default_factory=list)
    _seen_frames: set[tuple[int, int]] = field(default_factory=set, repr=False)
    _last_frame_by_source: dict[int, tuple[int, int, int, int, int]] = field(
        default_factory=dict,
        repr=False,
    )
    _seen_canonical_observation_ids: set[str] = field(
        default_factory=set,
        repr=False,
    )
    _seen_observation_sequences: set[tuple[str, int, int]] = field(
        default_factory=set,
        repr=False,
    )
    _last_presence_by_tracker: dict[
        tuple[int, str, int, int], tuple[int, int]
    ] = field(default_factory=dict, repr=False)
    _retired_tracker_lifecycles: set[tuple[int, str, int, int]] = field(
        default_factory=set,
        repr=False,
    )

    def add_error(self, message: object) -> None:
        if len(self.errors) >= MAX_ERRORS:
            return
        clean = " ".join(str(message).split())[:MAX_ERROR_CHARS]
        if clean and clean not in self.errors:
            self.errors.append(clean)

    def observe_payload(self, payload: object) -> None:
        if not isinstance(payload, Mapping):
            return
        message_type = str(payload.get("type") or "")
        if message_type in {"stats", "tracking", "semantic_source_rejection"}:
            if len(self.source_messages) >= MAX_SOURCE_MESSAGES:
                self.add_error("semantic source transcript message bound exceeded")
                return
            raw_paths = _raw_vector_paths(payload)
            if raw_paths:
                projected = {
                    "type": message_type,
                    SOURCE_REDACTION_MARKER: {
                        "raw_vector_field_count": min(len(raw_paths), 8),
                    },
                }
            else:
                projected = _semantic_source_projection(payload)
            projected = _canonical_projection_value(projected)
            if not isinstance(projected, dict):
                raise ValueError("semantic source projection must be an object")
            self.source_messages.append(projected)
            self._observe_privacy_marker(projected)
        if message_type == "stats":
            self._observe_stats(self.source_messages[-1])
        elif message_type == "tracking":
            self._observe_tracking(self.source_messages[-1])
        elif message_type == "semantic_source_rejection":
            reason = self.source_messages[-1].get("reason")
            if reason not in {"duplicate_json_key", "invalid_json"}:
                self.add_error("semantic source rejection reason is invalid")
            else:
                self.add_error(f"semantic source rejected {reason}")

    def observe_source_rejection(self, reason: str) -> None:
        self.observe_payload(
            {
                "type": "semantic_source_rejection",
                "reason": str(reason),
            }
        )

    def _observe_privacy_marker(self, payload: Mapping[str, Any]) -> None:
        marker = payload.get(SOURCE_REDACTION_MARKER)
        if marker is None:
            return
        raw_count = marker.get("raw_vector_field_count") if isinstance(marker, Mapping) else None
        if not _is_int(raw_count) or not 0 < int(raw_count) <= 8:
            self.add_error("semantic source redaction marker is invalid")
            return
        self.raw_vector_fields += int(raw_count)
        self.add_error("public telemetry exposed forbidden privacy material")

    def _observe_stats(self, payload: Mapping[str, Any]) -> None:
        raw_stats = payload.get("payload")
        pipeline = raw_stats.get("pipeline") if isinstance(raw_stats, Mapping) else None
        if not isinstance(pipeline, Mapping):
            return
        self.stats_messages += 1
        raw_errors = pipeline.get("errors")
        values = raw_errors if isinstance(raw_errors, (list, tuple)) else (raw_errors,)
        for raw_error in values:
            clean = " ".join(str(raw_error or "").split())[:MAX_ERROR_CHARS]
            if clean:
                self.pipeline_errors.add(clean)

    def _observe_tracking(self, payload: Mapping[str, Any]) -> None:
        self.tracking_messages += 1
        source_id = payload.get("source_id")
        frame_id = payload.get("frame_id")
        captured_at_us = payload.get("captured_at_us")
        observed_at_us = payload.get("observed_at_us")
        capture_time_status = payload.get("capture_time_status")
        media_pts_ns = payload.get("media_pts_ns")
        publication_sequence = payload.get("tracking_publication_sequence")
        tombstones = payload.get("tracker_lifecycle_tombstones")
        tracks = payload.get("tracks")
        observations = payload.get("observations")
        if not _is_int(source_id) or int(source_id) < 0:
            self.add_error("tracking payload source_id is invalid")
            return
        if not _is_int(frame_id) or int(frame_id) < 0:
            self.add_error("tracking payload frame_id is invalid")
            return
        if not _is_int(observed_at_us) or int(observed_at_us) <= 0:
            self.add_error("tracking payload observed_at_us is invalid")
            return
        if not _is_int(captured_at_us) or not 0 < int(captured_at_us) <= int(observed_at_us):
            self.add_error("tracking payload captured_at_us is invalid")
            return
        if capture_time_status not in {"synced", "estimated"}:
            self.add_error("tracking payload capture_time_status is not usable")
            return
        if not _is_int(media_pts_ns) or int(media_pts_ns) < 0:
            self.add_error("tracking payload media_pts_ns is invalid")
            return
        if (
            payload.get("tracking_continuity_contract")
            != TRACKING_CONTINUITY_CONTRACT
            or payload.get("tracking_continuity_contract_version")
            != TRACKING_CONTINUITY_CONTRACT_VERSION
            or not _is_int(publication_sequence)
            or int(publication_sequence) < 0
        ):
            self.add_error("tracking publication continuity contract is invalid")
            return
        if not isinstance(tombstones, list):
            self.add_error("tracking lifecycle tombstones must be a list")
            return
        source_int = int(source_id)
        frame_int = int(frame_id)
        captured_int = int(captured_at_us)
        observed_int = int(observed_at_us)
        media_pts_int = int(media_pts_ns)
        publication_int = int(publication_sequence)
        frame_key = (source_int, frame_int)
        if frame_key in self._seen_frames:
            self.add_error("tracking payload replayed a source frame")
        self._seen_frames.add(frame_key)
        previous_frame = self._last_frame_by_source.get(source_int)
        first_source_frame = previous_frame is None
        if previous_frame is not None and (
            frame_int <= previous_frame[0]
            or captured_int < previous_frame[1]
            or observed_int < previous_frame[2]
            or media_pts_int < previous_frame[3]
            or publication_int != previous_frame[4] + 1
        ):
            self.add_error("tracking payload source sequence is discontinuous")
        self._last_frame_by_source[source_int] = (
            frame_int,
            captured_int,
            observed_int,
            media_pts_int,
            publication_int,
        )
        if not isinstance(tracks, list):
            self.add_error("tracking payload tracks must be a list")
            return
        if payload.get("track_count") != len(tracks):
            self.add_error("tracking payload track_count does not match tracks")
        if payload.get("observation_contract") != "noesis.observation.person":
            self.add_error("tracking payload observation contract is missing")
            return
        if payload.get("observation_contract_version") != 1:
            self.add_error("tracking payload observation contract version is invalid")
            return
        if not isinstance(observations, list):
            self.add_error("tracking payload observations must be a list")
            return

        tombstone_keys: set[tuple[str, int, int]] = set()
        unanchored_origin_keys: set[tuple[str, int, int]] = set()
        for index, tombstone in enumerate(tombstones):
            try:
                if not isinstance(tombstone, Mapping) or set(tombstone) != set(
                    TOMBSTONE_PROJECTION_FIELDS
                ):
                    raise ValueError("shape is invalid")
                camera = str(tombstone.get("camera_id") or "").strip()
                tracker = tombstone.get("tracker_id")
                generation = tombstone.get(TRACKER_LIFECYCLE_GENERATION_FIELD)
                last_frame = tombstone.get("last_seen_frame_id")
                last_observed = tombstone.get("last_seen_observed_at_us")
                disappeared_frame = tombstone.get("disappeared_at_frame_id")
                disappeared_observed = tombstone.get(
                    "disappeared_at_observed_at_us"
                )
                if not camera:
                    raise ValueError("camera_id is missing")
                for label, value, positive in (
                    ("tracker_id", tracker, False),
                    ("generation", generation, True),
                    ("last_seen_frame_id", last_frame, False),
                    ("last_seen_observed_at_us", last_observed, True),
                ):
                    if (
                        not _is_int(value)
                        or int(value) < 0
                        or (positive and int(value) <= 0)
                    ):
                        raise ValueError(f"{label} is invalid")
                if disappeared_frame != frame_int or disappeared_observed != observed_int:
                    raise ValueError("disappearance does not belong to this frame")
                if int(last_frame) >= frame_int or int(last_observed) > observed_int:
                    raise ValueError("last-seen evidence does not precede disappearance")
                key = (camera, int(tracker), int(generation))
                if key in tombstone_keys:
                    raise ValueError("duplicate lifecycle tombstone")
                lifecycle_key = (source_int, *key)
                actual_last_presence = self._last_presence_by_tracker.get(
                    lifecycle_key
                )
                if actual_last_presence is None:
                    if not first_source_frame:
                        raise ValueError(
                            "last-seen lifecycle was not previously published in-window"
                        )
                    unanchored_origin_keys.add(key)
                elif actual_last_presence != (int(last_frame), int(last_observed)):
                    raise ValueError(
                        "last-seen frame/time does not match actual published presence"
                    )
                tombstone_keys.add(key)
            except Exception as exc:
                self.add_error(f"tracking lifecycle tombstone {index} invalid: {exc}")

        observation_by_key: dict[tuple[str, int, int, int], ObservationEnvelope] = {}
        for index, raw_observation in enumerate(observations):
            try:
                if not isinstance(raw_observation, Mapping):
                    raise ValueError("observation must be an object")
                provenance = _provenance(
                    raw_observation.get("payload")
                    if isinstance(raw_observation.get("payload"), Mapping)
                    else {},
                    "canonical observation",
                )
                observation = ObservationEnvelope.model_validate(raw_observation)
                tracklet = observation.payload.tracklet
                key = (
                    tracklet.camera_id,
                    int(tracklet.source_id),
                    int(tracklet.tracker_id),
                    int(tracklet.frame_id),
                )
                if key in observation_by_key:
                    raise ValueError("duplicate canonical observation association")
                if int(tracklet.source_id) != int(source_id):
                    raise ValueError("canonical observation source_id mismatch")
                if int(tracklet.frame_id) != int(frame_id):
                    raise ValueError("canonical observation frame_id mismatch")
                if int(observation.observed_at_us) != observed_int:
                    raise ValueError("canonical observation timestamp mismatch")
                if (
                    observation.captured_at_us != captured_int
                    or observation.capture_time_status != capture_time_status
                    or observation.media_pts_ns != media_pts_int
                ):
                    raise ValueError("canonical observation capture/PTS mismatch")
                if observation.observation_id in self._seen_canonical_observation_ids:
                    raise ValueError("canonical observation_id was replayed")
                sequence_key = (
                    observation.producer.run_id,
                    source_int,
                    int(observation.sequence),
                )
                if sequence_key in self._seen_observation_sequences:
                    raise ValueError("canonical observation sequence was replayed")
                self._seen_canonical_observation_ids.add(observation.observation_id)
                self._seen_observation_sequences.add(sequence_key)
                if provenance is None and observation.payload.embedding_sequence is not None:
                    raise ValueError("canonical observation provenance serialization drifted")
                observation_by_key[key] = observation
                self.canonical_publication_clocks.append(
                    (
                        int(observation.observed_at_us),
                        int(observation.published_at_us),
                    )
                )
                self.canonical_observations += 1
            except Exception as exc:
                self.add_error(f"canonical observation {index} invalid: {exc}")

        matched_keys: set[tuple[str, int, int, int]] = set()
        present_trackers: set[tuple[str, int, int]] = set()
        for index, raw_track in enumerate(tracks):
            if not isinstance(raw_track, Mapping):
                self.add_error(f"public track {index} must be an object")
                continue
            if raw_track.get("class_id") != 0:
                continue
            self.person_tracks += 1
            try:
                camera_id = str(raw_track.get("camera_id") or "").strip()
                tracker_id = raw_track.get("tracker_id")
                lifecycle_generation = raw_track.get(
                    TRACKER_LIFECYCLE_GENERATION_FIELD
                )
                track_frame_id = raw_track.get("frame_id")
                if not camera_id:
                    raise ValueError("camera_id is missing")
                if not _is_int(tracker_id) or int(tracker_id) < 0:
                    raise ValueError("tracker_id is invalid")
                if (
                    not _is_int(lifecycle_generation)
                    or int(lifecycle_generation) <= 0
                ):
                    raise ValueError("tracker lifecycle generation is invalid")
                if track_frame_id != frame_id:
                    raise ValueError("frame_id disagrees with tracking payload")
                if raw_track.get("observed_at_us") != observed_int:
                    raise ValueError("observed_at_us disagrees with tracking payload")
                key = (camera_id, int(source_id), int(tracker_id), int(frame_id))
                if key in matched_keys:
                    raise ValueError("duplicate public track association")
                present_trackers.add(
                    (camera_id, int(tracker_id), int(lifecycle_generation))
                )
                lifecycle_key = (
                    source_int,
                    camera_id,
                    int(tracker_id),
                    int(lifecycle_generation),
                )
                if lifecycle_key in self._retired_tracker_lifecycles:
                    raise ValueError("retired tracker lifecycle generation reappeared")
                observation = observation_by_key.get(key)
                if observation is None:
                    raise ValueError("matching canonical observation is missing")
                matched_keys.add(key)
                track_provenance = _provenance(raw_track, "public track")
                observation_provenance = _provenance(
                    observation.payload.model_dump(mode="json"),
                    "canonical observation",
                )
                if track_provenance != observation_provenance:
                    raise ValueError("track and observation embedding provenance disagree")
                embedding_present = raw_track.get("embedding_present")
                if type(embedding_present) is not bool:  # noqa: E721
                    raise ValueError("embedding_present must be boolean")
                identity, identity_fresh = _identity_continuity(raw_track)
                identity_key = raw_track.get("identity_observation_key")
                identity_observation_id: str | None = None
                if track_provenance is None:
                    if embedding_present is True:
                        raise ValueError("embedding_present=true lacks persisted provenance")
                    if identity_key is not None:
                        raise ValueError(
                            "identity_observation_key requires persisted provenance"
                        )
                    if identity_fresh:
                        raise ValueError(
                            "identity_v2 fresh embedding lacks persisted provenance"
                        )
                else:
                    if embedding_present is not True:
                        raise ValueError(
                            "persisted provenance requires embedding_present=true"
                        )
                    if identity_fresh is not True:
                        raise ValueError(
                            "persisted provenance requires a fresh identity_v2 decision"
                        )
                    if not isinstance(identity_key, Mapping):
                        raise ValueError("identity_observation_key is missing")
                    identity_run_id = str(identity_key.get("run_id") or "").strip()
                    identity_observation_id = str(
                        identity_key.get("observation_id") or ""
                    ).strip()
                    if not identity_run_id or not identity_observation_id:
                        raise ValueError("identity observation linkage is incomplete")
                    if identity_run_id != observation.producer.run_id:
                        raise ValueError("identity observation run_id mismatch")
                    if identity_key.get("camera_id") != camera_id:
                        raise ValueError("identity observation camera_id mismatch")
                    if str(identity_key.get("tracker_id")) != str(int(tracker_id)):
                        raise ValueError("identity observation tracker_id mismatch")
                    if identity_key.get("frame_id") != int(frame_id):
                        raise ValueError("identity observation frame_id mismatch")

                pose_present = bool(
                    raw_track.get("pose_present") is True
                    and observation.payload.pose_present is True
                )
                depth = _usable_depth(raw_track)
                depth_present = bool(
                    depth is not None and observation.payload.depth_present is True
                )
                track_world = _world_xyz(raw_track)
                observation_world = observation.payload.world
                world_present = False
                if track_world is not None and observation_world is not None:
                    observed_world = (
                        float(observation_world.position.x),
                        float(observation_world.position.y),
                        float(observation_world.position.z),
                    )
                    world_present = bool(
                        observation.coordinate_frame == "backend_world_m"
                        and observation.units == "meters"
                        and observation_world.frame == "backend_world_m"
                        and observation_world.units == "meters"
                        and all(math.isfinite(item) for item in observed_world)
                        and all(
                            math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)
                            for left, right in zip(track_world, observed_world)
                        )
                    )
                if len(self.samples) >= MAX_LINKS:
                    raise ValueError("semantic observation sample bound exceeded")
                self.samples.append(
                    SemanticObservationSample(
                        source_id=int(source_id),
                        camera_id=camera_id,
                        tracker_id=int(tracker_id),
                        tracker_lifecycle_generation=int(lifecycle_generation),
                        frame_id=int(frame_id),
                        run_id=observation.producer.run_id,
                        producer_runtime=observation.producer.runtime,
                        producer_instance_id=observation.producer.instance_id,
                        canonical_observation_id=observation.observation_id,
                        observation_sequence=int(observation.sequence),
                        captured_at_us=captured_int,
                        observed_at_us=observed_int,
                        published_at_us=int(observation.published_at_us),
                        capture_time_status=str(capture_time_status),
                        media_pts_ns=media_pts_int,
                        fingerprints=_semantic_fingerprints(observation),
                        identity=identity,
                        identity_fresh_embedding=identity_fresh,
                        identity_observation_id=identity_observation_id,
                        embedding_sequence=(
                            track_provenance[0]
                            if track_provenance is not None
                            else None
                        ),
                        embedding_model_sha256=(
                            track_provenance[1]
                            if track_provenance is not None
                            else None
                        ),
                        embedding_dimension=(
                            track_provenance[2]
                            if track_provenance is not None
                            else None
                        ),
                        pose_present=pose_present,
                        depth_present=depth_present,
                        depth_metric_name=depth[0] if depth is not None else None,
                        depth_metric_value=depth[1] if depth is not None else None,
                        world_present=world_present,
                    )
                )
            except Exception as exc:
                self.add_error(f"public track {index} invalid: {exc}")

        unmatched = set(observation_by_key) - matched_keys
        if unmatched:
            self.add_error("canonical observation has no associated public person track")
        simultaneous_lifecycles = tombstone_keys.intersection(present_trackers)
        if simultaneous_lifecycles:
            self.add_error("tracking lifecycle is simultaneously present and tombstoned")
        self.unanchored_origin_tombstones += len(
            unanchored_origin_keys - simultaneous_lifecycles
        )
        previous_presence = next(
            (
                frame
                for frame in reversed(self.frames)
                if frame.source_id == source_int
            ),
            None,
        )
        if previous_presence is not None:
            expected_tombstones = set(previous_presence.trackers) - present_trackers
            if tombstone_keys != expected_tombstones:
                self.add_error(
                    "tracking lifecycle tombstones do not exactly cover disappearances"
                )
        for camera, tracker, generation in tombstone_keys:
            lifecycle_key = (source_int, camera, tracker, generation)
            self._last_presence_by_tracker.pop(lifecycle_key, None)
            self._retired_tracker_lifecycles.add(lifecycle_key)
        for camera, tracker, generation in present_trackers:
            self._last_presence_by_tracker[
                (source_int, camera, tracker, generation)
            ] = (frame_int, observed_int)
        self.frames.append(
            TrackingFramePresence(
                source_id=source_int,
                frame_id=frame_int,
                publication_sequence=publication_int,
                captured_at_us=captured_int,
                observed_at_us=observed_int,
                media_pts_ns=media_pts_int,
                trackers=frozenset(present_trackers),
            )
        )


@dataclass(frozen=True)
class EvidenceSnapshot:
    rows_by_sequence: Mapping[int, ShadowIdentityEvidenceRecord]
    byte_count: int
    row_count: int
    sha256: str
    source_path: str
    payload: bytes


def _parse_evidence_snapshot_bytes(
    payload: bytes,
    *,
    source_path: str,
) -> EvidenceSnapshot:
    if not payload:
        raise ValueError("identity evidence JSONL is empty")
    if not payload.endswith(b"\n"):
        raise ValueError("identity evidence JSONL has an incomplete trailing record")
    rows: dict[int, ShadowIdentityEvidenceRecord] = {}
    previous: ShadowIdentityEvidenceRecord | None = None
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line or len(line) > 1024 * 1024:
            raise ValueError(f"identity evidence line {line_number} is invalid")
        try:
            raw = strict_json_loads(
                line,
                label=f"identity evidence line {line_number}",
            )
            if not isinstance(raw, Mapping):
                raise ValueError("row must be an object")
            if _raw_vector_paths(raw):
                raise ValueError("row contains raw embedding vector fields")
            row = ShadowIdentityEvidenceRecord.model_validate(raw)
            if line != _canonical_bytes(row.model_dump(mode="json")):
                raise ValueError("row is not canonical JSON")
            body = row.model_dump(mode="json")
            body.pop("event_id", None)
            expected = hashlib.sha256(
                b"noesis-identity-shadow-evidence-v2\0" + _canonical_bytes(body)
            ).hexdigest()
            if row.event_id != expected:
                raise ValueError("event digest mismatch")
            if previous is not None:
                if row.sequence != previous.sequence + 1:
                    raise ValueError("sequence is not contiguous")
                if row.previous_event_id != previous.event_id:
                    raise ValueError("hash chain is not contiguous")
                if row.observed_at_us < previous.observed_at_us:
                    raise ValueError("timestamps move backwards")
            if row.sequence in rows:
                raise ValueError("duplicate sequence")
            rows[row.sequence] = row
            previous = row
        except Exception as exc:
            raise ValueError(
                f"identity evidence line {line_number} is invalid: {exc}"
            ) from exc
    return EvidenceSnapshot(
        rows_by_sequence=rows,
        byte_count=len(payload),
        row_count=len(rows),
        sha256=hashlib.sha256(payload).hexdigest(),
        source_path=str(source_path),
        payload=bytes(payload),
    )


def _load_evidence_snapshot(path: Path) -> EvidenceSnapshot:
    payload = b""
    absolute = path.expanduser().absolute()
    for attempt in range(3):
        payload = read_private_file(
            absolute,
            label="DS9 identity evidence JSONL",
            max_bytes=MAX_EVIDENCE_BYTES,
        )
        if payload and payload.endswith(b"\n"):
            break
        if attempt < 2:
            time.sleep(0.05)
    return _parse_evidence_snapshot_bytes(payload, source_path=str(absolute))


def _source_transcript_document(
    *,
    session_id: str,
    runtime_lane: str,
    runtime_instance_id: str,
    runtime_run_id: str,
    acquisition_started_at_us: int,
    acquisition_finished_at_us: int,
    collector: SemanticObservationCollector,
) -> dict[str, object]:
    return {
        "schema_version": SOURCE_TRANSCRIPT_SCHEMA_VERSION,
        "contract": SOURCE_TRANSCRIPT_CONTRACT,
        "contract_version": SOURCE_TRANSCRIPT_CONTRACT_VERSION,
        "session_id": str(session_id).strip().lower(),
        "runtime_lane": str(runtime_lane).strip().lower(),
        "runtime_instance_id": str(runtime_instance_id),
        "runtime_run_id": str(runtime_run_id),
        "acquisition_window": _acquisition_window_document(
            started_at_us=acquisition_started_at_us,
            finished_at_us=acquisition_finished_at_us,
        ),
        "privacy": dict(SOURCE_PRIVACY_POLICY),
        "policy": {
            "maximum_cohort_span_us": SEMANTIC_COHORT_MAX_SPAN_US,
            "identity_continuity": "same_non_null_identity_v2_v1",
            "artifact_continuity": "exact_calibration_model_config_sha256_v1",
            "tracker_continuity": (
                "contiguous_within_acquisition_window_unanchored_origin_v1"
            ),
            "publication_sequence_origin": (
                "externally_unanchored_live_partial_capture_v1"
            ),
            "lifecycle_generation_origin": (
                "externally_unanchored_live_partial_capture_v1"
            ),
            "tombstone_last_presence": (
                "exact_within_window_after_unanchored_origin_v1"
            ),
            "acquisition_time": (
                "observed_closed_capture_pre_receive_leeway_v1"
            ),
            "temporal_continuity": "captured_observed_and_media_pts_v1",
            "association": "one_public_track_to_one_canonical_observation_v1",
        },
        "message_count": len(collector.source_messages),
        "messages": list(collector.source_messages),
    }


def _encoded_source_transcript(document: Mapping[str, object]) -> bytes:
    return (
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _source_evidence_metadata(
    *,
    filename: str,
    encoded: bytes,
    document: Mapping[str, object],
) -> dict[str, object]:
    acquisition = document.get("acquisition_window")
    if not isinstance(acquisition, Mapping):
        raise ValueError("semantic source acquisition window is missing")
    canonical_acquisition = _acquisition_window_document(
        started_at_us=acquisition.get("started_at_us"),
        finished_at_us=acquisition.get("finished_at_us"),
    )
    if dict(acquisition) != canonical_acquisition:
        raise ValueError("semantic source acquisition window drifted")
    tracking_messages = [
        message
        for message in document.get("messages", [])
        if isinstance(message, Mapping) and message.get("type") == "tracking"
    ]
    observed = [
        int(message["observed_at_us"])
        for message in tracking_messages
        if _is_int(message.get("observed_at_us"))
        and int(message["observed_at_us"]) > 0
    ]
    captured = [
        int(message["captured_at_us"])
        for message in tracking_messages
        if _is_int(message.get("captured_at_us"))
        and int(message["captured_at_us"]) > 0
    ]
    published = [
        int(observation["published_at_us"])
        for message in tracking_messages
        for observations in (message.get("observations"),)
        if isinstance(observations, list)
        for observation in observations
        if isinstance(observation, Mapping)
        and _is_int(observation.get("published_at_us"))
        and int(observation["published_at_us"]) > 0
    ]
    return {
        "filename": filename,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "message_count": document.get("message_count"),
        "first_observed_at_us": min(observed) if observed else None,
        "last_observed_at_us": max(observed) if observed else None,
        "first_captured_at_us": min(captured) if captured else None,
        "last_captured_at_us": max(captured) if captured else None,
        "first_published_at_us": min(published) if published else None,
        "last_published_at_us": max(published) if published else None,
        "acquisition_started_at_us": canonical_acquisition["started_at_us"],
        "acquisition_finished_at_us": canonical_acquisition["finished_at_us"],
        "capture_pre_window_leeway_us": canonical_acquisition[
            "capture_pre_window_leeway_us"
        ],
    }


def _matches_private_anchor(
    sample: SemanticObservationSample,
    row: ShadowIdentityEvidenceRecord,
    *,
    session_id: str,
    runtime_run_id: str,
    expected_model_layer: str,
    expected_embedding_dimension: int,
) -> bool:
    identity = sample.identity
    if identity is None or not sample.has_embedding_provenance:
        return False
    return bool(
        row.observation_id == sample.identity_observation_id
        and row.model_sha256 == sample.embedding_model_sha256
        and row.embedding_dim == sample.embedding_dimension
        and row.model_layer == expected_model_layer
        and row.run_id == sample.run_id == runtime_run_id
        and row.camera_id == sample.camera_id
        and row.tracker_id == str(sample.tracker_id)
        and row.frame_id == sample.frame_id
        and row.observed_at_us == sample.observed_at_us
        and row.session_id == session_id
        and row.runtime == "ds9"
        and row.runtime_mode == "shadow"
        and row.source.value == "shadow"
        and row.final_outcome == identity.state
        and row.final_winner_id == identity.subject_id
        and sample.embedding_dimension == expected_embedding_dimension
        and sample.identity_fresh_embedding is True
    )


def _tracker_was_absent_between(
    collector: SemanticObservationCollector,
    earlier: SemanticObservationSample,
    later: SemanticObservationSample,
) -> bool:
    expected = (
        earlier.camera_id,
        earlier.tracker_id,
        earlier.tracker_lifecycle_generation,
    )
    for frame in collector.frames:
        if frame.source_id != earlier.source_id:
            continue
        if not earlier.observed_at_us <= frame.observed_at_us <= later.observed_at_us:
            continue
        if not earlier.frame_id <= frame.frame_id <= later.frame_id:
            continue
        if expected not in frame.trackers:
            return True
    return False


def _continuity_segments(
    collector: SemanticObservationCollector,
) -> tuple[list[list[SemanticObservationSample]], dict[str, int]]:
    by_key: dict[
        tuple[str, int, str, int, int], list[SemanticObservationSample]
    ] = {}
    for sample in collector.samples:
        by_key.setdefault(sample.tracker_key, []).append(sample)
    segments: list[list[SemanticObservationSample]] = []
    identity_breaks = 0
    fingerprint_breaks = 0
    generations_by_tracker: dict[tuple[str, int, str, int], set[int]] = {}
    for sample in collector.samples:
        generations_by_tracker.setdefault(
            (sample.run_id, sample.source_id, sample.camera_id, sample.tracker_id),
            set(),
        ).add(sample.tracker_lifecycle_generation)
    tracker_reuse_breaks = sum(
        max(0, len(generations) - 1)
        for generations in generations_by_tracker.values()
    )
    temporal_breaks = 0
    for samples in by_key.values():
        ordered = sorted(
            samples,
            key=lambda item: (
                item.observed_at_us,
                item.frame_id,
                item.observation_sequence,
            ),
        )
        current: list[SemanticObservationSample] = []
        previous: SemanticObservationSample | None = None
        for sample in ordered:
            if sample.identity is None:
                if previous is not None and previous.identity is not None:
                    identity_breaks += 1
                if current:
                    segments.append(current)
                    current = []
                previous = sample
                continue
            broken = False
            if previous is not None:
                if previous.identity != sample.identity:
                    identity_breaks += 1
                    broken = True
                if previous.fingerprints != sample.fingerprints:
                    fingerprint_breaks += 1
                    broken = True
                if (
                    previous.capture_time_status != sample.capture_time_status
                    or previous.captured_at_us > sample.captured_at_us
                    or previous.published_at_us > sample.published_at_us
                    or previous.media_pts_ns > sample.media_pts_ns
                ):
                    temporal_breaks += 1
                    broken = True
                if _tracker_was_absent_between(collector, previous, sample):
                    tracker_reuse_breaks += 1
                    broken = True
            if broken and current:
                segments.append(current)
                current = []
            current.append(sample)
            previous = sample
        if current:
            segments.append(current)
    return segments, {
        "identity_discontinuities": identity_breaks,
        "fingerprint_discontinuities": fingerprint_breaks,
        "tracker_reuse_discontinuities": tracker_reuse_breaks,
        "temporal_discontinuities": temporal_breaks,
    }


def _sample_inside_acquisition_window(
    sample: SemanticObservationSample,
    *,
    acquisition_started_at_us: int,
    acquisition_finished_at_us: int,
) -> bool:
    capture_lower_bound = max(
        1,
        acquisition_started_at_us - SEMANTIC_CAPTURE_PRE_WINDOW_LEEWAY_US,
    )
    return bool(
        acquisition_started_at_us
        <= sample.observed_at_us
        <= acquisition_finished_at_us
        and capture_lower_bound
        <= sample.captured_at_us
        <= acquisition_finished_at_us
    )


def _frame_inside_acquisition_window(
    frame: TrackingFramePresence,
    *,
    acquisition_started_at_us: int,
    acquisition_finished_at_us: int,
) -> bool:
    capture_lower_bound = max(
        1,
        acquisition_started_at_us - SEMANTIC_CAPTURE_PRE_WINDOW_LEEWAY_US,
    )
    return bool(
        acquisition_started_at_us
        <= frame.observed_at_us
        <= acquisition_finished_at_us
        and capture_lower_bound
        <= frame.captured_at_us
        <= acquisition_finished_at_us
    )


def _publication_clock_inside_acquisition_window(
    sample: SemanticObservationSample,
    *,
    acquisition_finished_at_us: int,
) -> bool:
    return bool(
        sample.observed_at_us
        <= sample.published_at_us
        <= acquisition_finished_at_us
    )


def _accepted_cohorts(
    collector: SemanticObservationCollector,
    anchors: list[PersistedAnchor],
    *,
    acquisition_started_at_us: int,
    acquisition_finished_at_us: int,
) -> tuple[list[SemanticCohort], dict[str, int]]:
    anchor_by_sample = {anchor.sample: anchor for anchor in anchors}
    segments, discontinuities = _continuity_segments(collector)
    accepted: list[SemanticCohort] = []
    accepted_anchor_keys: set[tuple[int, str, int, int, int]] = set()
    for segment in segments:
        left = 0
        for right, newest in enumerate(segment):
            while (
                newest.observed_at_us - segment[left].observed_at_us
                > SEMANTIC_COHORT_MAX_SPAN_US
            ):
                left += 1
            window = segment[left : right + 1]
            poses = [sample for sample in window if sample.pose_present]
            depths = [sample for sample in window if sample.depth_present]
            worlds = [sample for sample in window if sample.world_present]
            if not poses or not depths or not worlds:
                continue
            for sample in window:
                anchor = anchor_by_sample.get(sample)
                if anchor is None:
                    continue
                anchor_key = (
                    sample.source_id,
                    sample.camera_id,
                    sample.tracker_id,
                    sample.frame_id,
                    int(sample.embedding_sequence or 0),
                )
                if anchor_key in accepted_anchor_keys:
                    continue
                cohort = SemanticCohort(
                    anchor=anchor,
                    pose=min(
                        poses,
                        key=lambda item: abs(
                            item.observed_at_us - sample.observed_at_us
                        ),
                    ),
                    depth=min(
                        depths,
                        key=lambda item: abs(
                            item.observed_at_us - sample.observed_at_us
                        ),
                    ),
                    world=min(
                        worlds,
                        key=lambda item: abs(
                            item.observed_at_us - sample.observed_at_us
                        ),
                    ),
                )
                if (
                    cohort.span_us <= SEMANTIC_COHORT_MAX_SPAN_US
                    and cohort.capture_span_us <= SEMANTIC_COHORT_MAX_SPAN_US
                    and cohort.media_pts_span_ns
                    <= SEMANTIC_COHORT_MAX_SPAN_US * 1000
                    and all(
                        _sample_inside_acquisition_window(
                            component,
                            acquisition_started_at_us=acquisition_started_at_us,
                            acquisition_finished_at_us=acquisition_finished_at_us,
                        )
                        and _publication_clock_inside_acquisition_window(
                            component,
                            acquisition_finished_at_us=acquisition_finished_at_us,
                        )
                        for component in (
                            cohort.anchor.sample,
                            cohort.pose,
                            cohort.depth,
                            cohort.world,
                        )
                    )
                ):
                    accepted.append(cohort)
                    accepted_anchor_keys.add(anchor_key)
    return accepted, discontinuities


def _component_report(
    sample: SemanticObservationSample,
    *,
    include_depth_metric: bool = False,
) -> dict[str, object]:
    result: dict[str, object] = {
        "frame_id": sample.frame_id,
        "canonical_observation_id": sample.canonical_observation_id,
        "captured_at_us": sample.captured_at_us,
        "observed_at_us": sample.observed_at_us,
        "published_at_us": sample.published_at_us,
        "media_pts_ns": sample.media_pts_ns,
    }
    if include_depth_metric:
        result["depth_metric"] = sample.depth_metric_name
    return result


def _sample_cohort_report(cohort: SemanticCohort | None) -> dict[str, object] | None:
    if cohort is None:
        return None
    sample = cohort.anchor.sample
    row = cohort.anchor.evidence
    assert sample.identity is not None
    return {
        "source_id": sample.source_id,
        "camera_id": sample.camera_id,
        "tracker_id": sample.tracker_id,
        "tracker_lifecycle_generation": sample.tracker_lifecycle_generation,
        "run_id": sample.run_id,
        "started_at_us": cohort.started_at_us,
        "finished_at_us": cohort.finished_at_us,
        "span_us": cohort.span_us,
        "capture_started_at_us": cohort.capture_started_at_us,
        "capture_finished_at_us": cohort.capture_finished_at_us,
        "capture_span_us": cohort.capture_span_us,
        "media_pts_span_ns": cohort.media_pts_span_ns,
        "maximum_span_us": SEMANTIC_COHORT_MAX_SPAN_US,
        "identity": sample.identity.as_report(),
        "fingerprints": sample.fingerprints.as_report(),
        "anchor": {
            "frame_id": sample.frame_id,
            "canonical_observation_id": sample.canonical_observation_id,
            "identity_observation_id": sample.identity_observation_id,
            "captured_at_us": sample.captured_at_us,
            "observed_at_us": sample.observed_at_us,
            "published_at_us": sample.published_at_us,
            "media_pts_ns": sample.media_pts_ns,
            "event_id": row.event_id,
            "embedding_sequence": sample.embedding_sequence,
            "embedding_model_sha256": sample.embedding_model_sha256,
            "embedding_model_semantic_profile_sha256": (
                row.model_semantic_profile_sha256
            ),
            "embedding_model_layer": row.model_layer,
            "embedding_dimension": sample.embedding_dimension,
        },
        "components": {
            "pose": _component_report(cohort.pose),
            "usable_depth": _component_report(
                cohort.depth,
                include_depth_metric=True,
            ),
            "backend_world_m": _component_report(cohort.world),
        },
    }


def _evaluate(
    collector: SemanticObservationCollector,
    snapshot: EvidenceSnapshot | None,
    *,
    session_id: str,
    runtime_lane: str,
    runtime_instance_id: str,
    runtime_run_id: str,
    acquisition_started_at_us: int,
    acquisition_finished_at_us: int,
    expected_model_layer: str,
    expected_embedding_dimension: int,
    sealed_snapshot_filename: str,
    source_evidence: Mapping[str, object],
    extra_errors: list[str] | None = None,
) -> dict[str, object]:
    session_id = str(session_id).strip().lower()
    runtime_lane = str(runtime_lane).strip().lower()
    if SESSION_RE.fullmatch(session_id) is None:
        raise ValueError("session_id must match [a-z0-9][a-z0-9-]{5,47}")
    if runtime_lane not in {"baseline", "v3dt"}:
        raise ValueError("semantic observation evidence requires baseline or v3dt lane")
    if (
        RUNTIME_ID_RE.fullmatch(str(runtime_instance_id)) is None
        or RUNTIME_ID_RE.fullmatch(str(runtime_run_id)) is None
    ):
        raise ValueError("runtime instance/run identity is invalid")
    acquisition_window = _acquisition_window_document(
        started_at_us=acquisition_started_at_us,
        finished_at_us=acquisition_finished_at_us,
    )
    errors = list(collector.errors)
    errors.extend(extra_errors or [])
    if collector.stats_messages < 1:
        errors.append("pipeline stats were not observed")
    if collector.pipeline_errors:
        errors.append(
            f"pipeline reported {len(collector.pipeline_errors)} error(s)"
        )
    if collector.person_tracks < 1:
        errors.append("occupied scene was not observed")
    one_to_one_association = bool(
        collector.person_tracks
        == collector.canonical_observations
        == len(collector.samples)
    )
    if not one_to_one_association:
        errors.append("public tracks and canonical observations are not one-to-one")
    provenance_samples = [
        sample for sample in collector.samples if sample.has_embedding_provenance
    ]
    pose_samples = [sample for sample in collector.samples if sample.pose_present]
    depth_samples = [sample for sample in collector.samples if sample.depth_present]
    world_samples = [sample for sample in collector.samples if sample.world_present]
    if not provenance_samples:
        errors.append("no complete public embedding provenance link was observed")
    if not pose_samples:
        errors.append("pose component was not observed")
    if not depth_samples:
        errors.append("usable depth component was not observed")
    if not world_samples:
        errors.append("backend_world_m component was not observed")
    if any(
        sample.producer_runtime != "ds9"
        or sample.producer_instance_id != runtime_instance_id
        or sample.run_id != runtime_run_id
        for sample in collector.samples
    ):
        errors.append("canonical observation runtime identity drifted")
    acquisition_window_violations = sum(
        not _frame_inside_acquisition_window(
            frame,
            acquisition_started_at_us=acquisition_started_at_us,
            acquisition_finished_at_us=acquisition_finished_at_us,
        )
        for frame in collector.frames
    )
    if acquisition_window_violations:
        errors.append(
            "tracking frame capture/observed time escapes acquisition window"
        )
    publication_clock_violations = sum(
        not (
            observed_at_us
            <= published_at_us
            <= acquisition_finished_at_us
        )
        for observed_at_us, published_at_us in (
            collector.canonical_publication_clocks
        )
    )
    if publication_clock_violations:
        errors.append("canonical observation publication clock escapes acquisition window")
    expected_acquisition_metadata = {
        "acquisition_started_at_us": acquisition_window["started_at_us"],
        "acquisition_finished_at_us": acquisition_window["finished_at_us"],
        "capture_pre_window_leeway_us": acquisition_window[
            "capture_pre_window_leeway_us"
        ],
    }
    if any(
        source_evidence.get(key) != value
        for key, value in expected_acquisition_metadata.items()
    ):
        errors.append("semantic source evidence acquisition window drifted")

    snapshot_rows = list(snapshot.rows_by_sequence.values()) if snapshot is not None else []
    snapshot_model_sha256: str | None = None
    snapshot_semantic_sha256: str | None = None
    if snapshot_rows:
        model_hashes = {row.model_sha256 for row in snapshot_rows}
        snapshot_model_sha256 = next(iter(model_hashes)) if len(model_hashes) == 1 else None
        semantic_hashes = {
            row.model_semantic_profile_sha256 for row in snapshot_rows
        }
        snapshot_semantic_sha256 = (
            next(iter(semantic_hashes)) if len(semantic_hashes) == 1 else None
        )
        if any(
            row.session_id != session_id
            or row.runtime != "ds9"
            or row.runtime_mode != "shadow"
            or row.run_id != runtime_run_id
            or row.model_layer != expected_model_layer
            or row.embedding_dim != expected_embedding_dimension
            for row in snapshot_rows
        ):
            errors.append("identity evidence snapshot session/run/model contract drifted")
        if snapshot_model_sha256 is None:
            errors.append("identity evidence snapshot contains multiple model hashes")
        if snapshot_semantic_sha256 is None:
            errors.append(
                "identity evidence snapshot contains multiple semantic profile hashes"
            )

    matched_anchors: list[PersistedAnchor] = []
    if snapshot is not None:
        for sample in provenance_samples:
            assert sample.embedding_sequence is not None
            row = snapshot.rows_by_sequence.get(sample.embedding_sequence)
            if row is None:
                errors.append(
                    f"identity evidence sequence {sample.embedding_sequence} is missing"
                )
                continue
            if not _matches_private_anchor(
                sample,
                row,
                session_id=session_id,
                runtime_run_id=runtime_run_id,
                expected_model_layer=expected_model_layer,
                expected_embedding_dimension=expected_embedding_dimension,
            ):
                errors.append(
                    f"identity evidence sequence {sample.embedding_sequence} does not match its public linkage"
                )
                continue
            matched_anchors.append(PersistedAnchor(sample=sample, evidence=row))
    cohorts, discontinuities = _accepted_cohorts(
        collector,
        matched_anchors,
        acquisition_started_at_us=acquisition_started_at_us,
        acquisition_finished_at_us=acquisition_finished_at_us,
    )
    if not cohorts:
        errors.append(
            "no bounded identity-continuous cohort joined persisted identity, pose, usable depth, and backend world"
        )
    errors = list(
        dict.fromkeys(
            " ".join(str(item).split())[:MAX_ERROR_CHARS]
            for item in errors
            if str(item).strip()
        )
    )[:MAX_ERRORS]
    occupied = collector.person_tracks > 0
    ok = not errors
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT,
        "contract_version": CONTRACT_VERSION,
        "session_id": session_id,
        "runtime_lane": runtime_lane,
        "runtime_instance_id": str(runtime_instance_id),
        "runtime_run_id": str(runtime_run_id),
        "ok": ok,
        "status": "pass" if ok else ("blocked" if not occupied else "failed"),
        "checks": {
            "occupied_scene_observed": occupied,
            "pipeline_stats_observed": collector.stats_messages > 0,
            "pipeline_errors_absent": not collector.pipeline_errors,
            "raw_embedding_vectors_absent": collector.raw_vector_fields == 0,
            "public_track_observation_association": not collector.errors,
            "one_to_one_track_observation_association": one_to_one_association,
            "persisted_embedding_anchor": bool(matched_anchors),
            "pose_component_observed": bool(pose_samples),
            "usable_depth_component_observed": bool(depth_samples),
            "backend_world_m_component_observed": bool(world_samples),
            "bounded_identity_cohort_accepted": bool(cohorts),
            "capture_and_observation_bounds_enforced": bool(cohorts),
            "acquisition_window_enforced": bool(
                cohorts and acquisition_window_violations == 0
            ),
            "publication_clock_within_acquisition_window": bool(
                cohorts and publication_clock_violations == 0
            ),
            "artifact_fingerprint_continuity_enforced": bool(cohorts),
            "tracker_continuity_within_acquisition_window": bool(cohorts),
            "tracking_publication_contiguous_within_window_unanchored_origin": bool(
                cohorts
            ),
            "tracker_lifecycle_contiguous_within_window_unanchored_origin": bool(
                cohorts
            ),
            "tombstone_last_published_presence_exact_after_unanchored_origin": bool(
                cohorts and not collector.errors
            ),
        },
        "counts": {
            "tracking_messages": collector.tracking_messages,
            "stats_messages": collector.stats_messages,
            "person_tracks": collector.person_tracks,
            "canonical_observations": collector.canonical_observations,
            "associated_samples": len(collector.samples),
            "public_provenance_links": len(provenance_samples),
            "persisted_anchors": len(matched_anchors),
            "pose_component_samples": len(pose_samples),
            "usable_depth_component_samples": len(depth_samples),
            "backend_world_m_component_samples": len(world_samples),
            "accepted_semantic_cohorts": len(cohorts),
            **discontinuities,
            "pipeline_error_count": len(collector.pipeline_errors),
            "raw_vector_field_count": collector.raw_vector_fields,
            "acquisition_window_violations": acquisition_window_violations,
            "publication_clock_violations": publication_clock_violations,
            "unanchored_origin_tombstones": (
                collector.unanchored_origin_tombstones
            ),
        },
        "evidence_snapshot": (
            {
                "bytes": snapshot.byte_count,
                "rows": snapshot.row_count,
                "sha256": snapshot.sha256,
                "source_path": snapshot.source_path,
                "sealed_filename": sealed_snapshot_filename,
                "session_id": session_id,
                "runtime_run_id": runtime_run_id,
                "model_sha256": snapshot_model_sha256,
                "model_semantic_profile_sha256": snapshot_semantic_sha256,
                "model_layer": expected_model_layer,
                "embedding_dimension": expected_embedding_dimension,
                "first_sequence": min(snapshot.rows_by_sequence, default=None),
                "last_sequence": max(snapshot.rows_by_sequence, default=None),
            }
            if snapshot is not None
            else None
        ),
        "component_availability": {
            "pose": bool(pose_samples),
            "usable_depth": bool(depth_samples),
            "backend_world_m": bool(world_samples),
        },
        "sample_cohort": _sample_cohort_report(cohorts[0] if cohorts else None),
        "source_evidence": dict(source_evidence),
        "errors": errors,
    }


def _strict_json_object(payload: bytes, *, label: str) -> Mapping[str, object]:
    document = strict_json_loads(payload, label=label)
    if not isinstance(document, Mapping):
        raise ValueError(f"{label} must be an object")
    return document


def validate_sealed_semantic_report(
    report_path: Path,
    snapshot_path: Path,
    source_path: Path,
    *,
    session_id: str,
    runtime_lane: str,
    runtime_instance_id: str,
    runtime_run_id: str,
    expected_identity_evidence_path: Path,
    expected_model_layer: str,
    expected_embedding_dimension: int,
) -> Mapping[str, object]:
    """Validate and exactly replay one complete semantic-v3 evidence bundle."""

    report_path = report_path.expanduser().absolute()
    snapshot_path = snapshot_path.expanduser().absolute()
    source_path = source_path.expanduser().absolute()
    if len({report_path.parent, snapshot_path.parent, source_path.parent}) != 1:
        raise ValueError("semantic report, snapshot, and source must share one evidence directory")
    if report_path.name != CANONICAL_REPORT_FILENAME:
        raise ValueError("semantic report filename is not canonical")
    if snapshot_path.name != CANONICAL_IDENTITY_SNAPSHOT_FILENAME:
        raise ValueError("semantic identity snapshot filename is not canonical")
    if source_path.name != CANONICAL_SOURCE_TRANSCRIPT_FILENAME:
        raise ValueError("semantic source transcript filename is not canonical")
    report_raw = read_private_file(
        report_path,
        label="sealed semantic observation report",
        max_bytes=MAX_REPORT_BYTES,
    )
    source_raw = read_private_file(
        source_path,
        label="sealed semantic observation source transcript",
        max_bytes=MAX_SOURCE_TRANSCRIPT_BYTES,
    )
    snapshot_raw = read_private_file(
        snapshot_path,
        label="sealed semantic identity evidence snapshot",
        max_bytes=MAX_EVIDENCE_BYTES,
    )
    report = _strict_json_object(report_raw, label="semantic observation report")
    if report_raw != _encoded_report(report):
        raise ValueError("semantic observation report is not canonical JSON")
    expected_report_keys = {
        "schema_version",
        "contract",
        "contract_version",
        "session_id",
        "runtime_lane",
        "runtime_instance_id",
        "runtime_run_id",
        "ok",
        "status",
        "checks",
        "counts",
        "evidence_snapshot",
        "component_availability",
        "sample_cohort",
        "source_evidence",
        "errors",
    }
    if set(report) != expected_report_keys:
        raise ValueError("semantic observation report shape is not exact")
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or type(report.get("schema_version")) is not int  # noqa: E721
        or report.get("contract") != CONTRACT
        or report.get("contract_version") != CONTRACT_VERSION
        or type(report.get("contract_version")) is not int  # noqa: E721
        or report.get("session_id") != session_id
        or report.get("runtime_lane") != runtime_lane
        or report.get("runtime_instance_id") != runtime_instance_id
        or report.get("runtime_run_id") != runtime_run_id
        or report.get("ok") is not True
        or report.get("status") != "pass"
        or report.get("errors") != []
    ):
        raise ValueError("semantic observation report envelope is not a pass")

    source = _strict_json_object(
        source_raw,
        label="semantic observation source transcript",
    )
    if source_raw != _encoded_source_transcript(source):
        raise ValueError("semantic source transcript is not canonical JSON")
    acquisition_raw = source.get("acquisition_window")
    if not isinstance(acquisition_raw, Mapping):
        raise ValueError("semantic source acquisition window is missing")
    acquisition_window = _acquisition_window_document(
        started_at_us=acquisition_raw.get("started_at_us"),
        finished_at_us=acquisition_raw.get("finished_at_us"),
    )
    if dict(acquisition_raw) != acquisition_window:
        raise ValueError("semantic source acquisition window is not canonical")
    collector = SemanticObservationCollector()
    messages = source.get("messages")
    if not isinstance(messages, list) or len(messages) > MAX_SOURCE_MESSAGES:
        raise ValueError("semantic source transcript messages are invalid")
    for message in messages:
        collector.observe_payload(message)
    canonical_source = _source_transcript_document(
        session_id=session_id,
        runtime_lane=runtime_lane,
        runtime_instance_id=runtime_instance_id,
        runtime_run_id=runtime_run_id,
        acquisition_started_at_us=int(acquisition_window["started_at_us"]),
        acquisition_finished_at_us=int(acquisition_window["finished_at_us"]),
        collector=collector,
    )
    if source != canonical_source:
        raise ValueError("semantic source transcript is not canonical")
    source_metadata = _source_evidence_metadata(
        filename=CANONICAL_SOURCE_TRANSCRIPT_FILENAME,
        encoded=source_raw,
        document=source,
    )
    if report.get("source_evidence") != source_metadata:
        raise ValueError("semantic source evidence binding drifted")

    expected_identity_path = expected_identity_evidence_path.expanduser().absolute()
    snapshot = _parse_evidence_snapshot_bytes(
        snapshot_raw,
        source_path=str(expected_identity_path),
    )
    recomputed = _evaluate(
        collector,
        snapshot,
        session_id=session_id,
        runtime_lane=runtime_lane,
        runtime_instance_id=runtime_instance_id,
        runtime_run_id=runtime_run_id,
        acquisition_started_at_us=int(acquisition_window["started_at_us"]),
        acquisition_finished_at_us=int(acquisition_window["finished_at_us"]),
        expected_model_layer=expected_model_layer,
        expected_embedding_dimension=expected_embedding_dimension,
        sealed_snapshot_filename=CANONICAL_IDENTITY_SNAPSHOT_FILENAME,
        source_evidence=source_metadata,
    )
    if report != recomputed:
        raise ValueError("semantic report does not exactly replay from sealed evidence")
    return report


async def _collect(
    ws_url: str,
    auth: RequiredInternalAuth,
    *,
    duration_s: float,
    collector: SemanticObservationCollector,
) -> None:
    async with connect_required_websocket(
        ws_url,
        auth,
        max_size=MAX_WS_MESSAGE_BYTES,
        compression=None,
    ) as websocket:
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            timeout_s = min(2.0, max(0.05, deadline - time.monotonic()))
            try:
                raw = await asyncio.wait_for(websocket.recv(), timeout=timeout_s)
            except asyncio.TimeoutError:
                continue
            if not isinstance(raw, str):
                continue
            try:
                payload = strict_json_loads(raw, label="semantic websocket message")
            except StrictJSONError as exc:
                collector.observe_source_rejection(
                    "duplicate_json_key"
                    if exc.reason == "duplicate_key"
                    else "invalid_json"
                )
                continue
            collector.observe_payload(payload)


def _encoded_report(report: Mapping[str, object]) -> bytes:
    return (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _create_evidence_file(
    path: Path,
    payload: bytes,
    *,
    expected_filename: str,
    max_bytes: int,
    label: str,
) -> None:
    destination = path.expanduser().absolute()
    if destination.name != expected_filename:
        raise ValueError(f"{label} filename is not canonical")
    try:
        atomic_create_private_file(
            destination,
            payload,
            label=label,
            max_bytes=max_bytes,
        )
    except PrivatePathError as exc:
        raise ValueError(str(exc)) from exc


def _write_report(path: Path, report: Mapping[str, object]) -> None:
    _create_evidence_file(
        path,
        _encoded_report(report),
        expected_filename=CANONICAL_REPORT_FILENAME,
        max_bytes=MAX_REPORT_BYTES,
        label="immutable semantic observation report",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate DS9 occupied-scene semantic observations."
    )
    parser.add_argument("--ws", default="ws://127.0.0.1:6008")
    parser.add_argument(
        "--pipeline-config", type=Path, default=Path("DS9/config/infer.yaml")
    )
    parser.add_argument("--duration", type=float, default=45.0)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--runtime-lane", choices=("baseline", "v3dt"), required=True)
    parser.add_argument("--runtime-instance-id", required=True)
    parser.add_argument("--runtime-run-id", required=True)
    parser.add_argument(
        "--identity-evidence",
        type=Path,
        required=True,
        help="Exact owner-only DS9 identity-v2 JSONL written by this runtime.",
    )
    parser.add_argument("--snapshot-out", type=Path, required=True)
    parser.add_argument("--source-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    add_auth_token_file_argument(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require_fresh_private_file_bundle(
            (args.snapshot_out, args.source_out, args.out),
            label="semantic behavior evidence bundle",
        )
    except (PrivatePathError, ValueError) as exc:
        print(f"[FAIL] semantic evidence session is not fresh: {exc}", file=sys.stderr)
        return 1
    collector = SemanticObservationCollector()
    snapshot: EvidenceSnapshot | None = None
    errors: list[str] = []
    source_published = False
    expected_layer = ""
    expected_dimension = 0
    acquisition_started_at_us = time.time_ns() // 1000
    acquisition_finished_at_us = acquisition_started_at_us
    try:
        duration = float(args.duration)
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("duration must be finite and positive")
        expected_layer, expected_dimension = _load_reviewed_reid_contract(
            args.pipeline_config
        )
        auth = load_required_internal_auth(args.auth_token_file)
        acquisition_started_at_us = time.time_ns() // 1000
        try:
            asyncio.run(
                _collect(
                    args.ws,
                    auth,
                    duration_s=duration,
                    collector=collector,
                )
            )
        finally:
            acquisition_finished_at_us = max(
                acquisition_started_at_us,
                time.time_ns() // 1000,
            )
        snapshot = _load_evidence_snapshot(args.identity_evidence)
        _create_evidence_file(
            args.snapshot_out,
            snapshot.payload,
            expected_filename=CANONICAL_IDENTITY_SNAPSHOT_FILENAME,
            max_bytes=MAX_EVIDENCE_BYTES,
            label="sealed semantic identity evidence snapshot",
        )
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    source_document = _source_transcript_document(
        session_id=args.session_id,
        runtime_lane=args.runtime_lane,
        runtime_instance_id=args.runtime_instance_id,
        runtime_run_id=args.runtime_run_id,
        acquisition_started_at_us=acquisition_started_at_us,
        acquisition_finished_at_us=acquisition_finished_at_us,
        collector=collector,
    )
    source_encoded = _encoded_source_transcript(source_document)
    source_evidence = _source_evidence_metadata(
        filename=args.source_out.name,
        encoded=source_encoded,
        document=source_document,
    )
    try:
        if len(source_encoded) > MAX_SOURCE_TRANSCRIPT_BYTES:
            raise ValueError("semantic source transcript exceeded size bound")
        _create_evidence_file(
            args.source_out,
            source_encoded,
            expected_filename=CANONICAL_SOURCE_TRANSCRIPT_FILENAME,
            max_bytes=MAX_SOURCE_TRANSCRIPT_BYTES,
            label="sealed semantic observation source transcript",
        )
        source_published = True
    except Exception as exc:
        errors.append(f"unable to seal semantic source transcript: {exc}")
    if not source_published:
        print(
            "[FAIL] semantic source evidence was not published; use a fresh session",
            file=sys.stderr,
        )
        return 1
    report = _evaluate(
        collector,
        snapshot,
        session_id=args.session_id,
        runtime_lane=args.runtime_lane,
        runtime_instance_id=args.runtime_instance_id,
        runtime_run_id=args.runtime_run_id,
        acquisition_started_at_us=acquisition_started_at_us,
        acquisition_finished_at_us=acquisition_finished_at_us,
        expected_model_layer=expected_layer,
        expected_embedding_dimension=expected_dimension,
        sealed_snapshot_filename=args.snapshot_out.name,
        source_evidence=source_evidence,
        extra_errors=errors,
    )
    try:
        _write_report(args.out, report)
    except Exception as exc:
        print(f"[FAIL] unable to write semantic observation report: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
