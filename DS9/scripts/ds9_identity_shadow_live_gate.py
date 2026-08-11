#!/usr/bin/env python3
"""Fail-closed live evidence gate for the DS9 identity-v2 shadow runtime.

The default gate proves runtime wiring, fresh server-produced ReID evidence,
tracker-local subject continuity, and that public authority remains blocked.
Cross-camera assignment continuity and an open-set non-force observation are
reported separately and can be made mandatory for a staged/natural scene.  No
truth labels are inferred from telemetry, so identity accuracy is never claimed
by this gate.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import http.client
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis_core.private_paths import (  # noqa: E402
    PrivatePathError,
    atomic_create_private_file,
    read_private_file,
    require_fresh_private_file_bundle,
)
from noesis_core.strict_json import strict_json_loads  # noqa: E402
from scripts.internal_auth_client import (  # noqa: E402
    RequiredInternalAuth,
    add_auth_token_file_argument,
    connect_required_websocket,
    load_required_internal_auth,
)

CONTRACT = "noesis.ds9.identity-shadow-live-gate"
CONTRACT_VERSION = 1
BASELINE_CANONICAL_REPORT_FILENAME = "identity-open-set-occupied.json"
V3DT_CANONICAL_REPORT_FILENAME = "v3dt-identity-open-set-occupied.json"
AUXILIARY_CANONICAL_REPORT_FILENAME = "identity-shadow.json"
SOURCE_TRANSCRIPT_CONTRACT = "noesis.ds9.identity-shadow-source-transcript"
SOURCE_TRANSCRIPT_VERSION = 2
BASELINE_SOURCE_TRANSCRIPT_FILENAME = "identity-open-set-occupied-source.json"
V3DT_SOURCE_TRANSCRIPT_FILENAME = "v3dt-identity-open-set-occupied-source.json"
AUXILIARY_SOURCE_TRANSCRIPT_FILENAME = "identity-shadow-source.json"
SESSION_RE = re.compile(r"^[a-z0-9][a-z0-9-]{5,47}$")
RUNTIME_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
RUNTIME_LANES = frozenset(
    {"baseline", "v3dt", "wholebody49-s", "wholebody49-x"}
)
MAX_HEALTH_RESPONSE_BYTES = 64 * 1024
MAX_REPORT_BYTES = 256 * 1024
MAX_SOURCE_TRANSCRIPT_BYTES = 8 * 1024 * 1024
MAX_SOURCE_MESSAGES = 4096
MAX_SOURCE_ROWS = 16384
OBSERVATION_ID_RE = re.compile(r"^obs1:[0-9a-f]{64}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_STATES = frozenset({"unknown", "provisional", "visitor", "resident"})
HEALTH_SOURCE_FIELDS = frozenset(
    {
        "schema_version",
        "runtime_mode",
        "public_authority_cutover_status",
        "authority_cutover_artifact_id",
        "runtime_model_fingerprint",
        "runtime_model_layer",
        "runtime_embedding_dim",
        "scoring_calibration_status",
        "scoring_authority_scope",
        "resident_count",
        "observation_cache_entries",
        "observation_cache_max_entries",
        "observation_cache_ttl_s",
    }
)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(value: object, label: str) -> int:
    if not _is_int(value) or int(value) <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _nonnegative_int(value: object, label: str) -> int:
    if not _is_int(value) or int(value) < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)


def _nonempty_text(value: object, label: str) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(f"{label} must be non-empty")
    return text


def _load_expected_reid_contract(path: Path) -> tuple[str, int]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise ValueError(f"unable to load pipeline config {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("pipeline config root must be a mapping")
    models = payload.get("models")
    reid = models.get("reid") if isinstance(models, Mapping) else None
    if not isinstance(reid, Mapping) or reid.get("enable") is not True:
        raise ValueError("pipeline config must enable models.reid")
    layer = _nonempty_text(reid.get("layer"), "models.reid.layer")
    dimension = _positive_int(
        reid.get("embedding_dim"), "models.reid.embedding_dim"
    )
    return layer, dimension


def _http_json_get(
    rest_url: str,
    path: str,
    auth: RequiredInternalAuth,
    *,
    timeout_s: float,
) -> Mapping[str, Any]:
    parsed = urlparse(rest_url)
    if parsed.scheme != "http":
        raise ValueError("identity health endpoint must use http://")
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    base = parsed.path.rstrip("/")
    request_path = f"{base}{path}"
    connection = http.client.HTTPConnection(host, port, timeout=timeout_s)
    try:
        headers = {
            "Accept": "application/json",
            "Connection": "close",
            **auth.authorization_headers(),
        }
        connection.request("GET", request_path, headers=headers)
        response = connection.getresponse()
        raw = response.read(MAX_HEALTH_RESPONSE_BYTES + 1)
        if response.status != 200:
            raise RuntimeError(
                f"identity health endpoint returned HTTP {response.status}"
            )
        if len(raw) > MAX_HEALTH_RESPONSE_BYTES:
            raise RuntimeError("identity health response exceeded size bound")
        try:
            payload = strict_json_loads(raw, label="identity health response")
        except Exception as exc:
            raise RuntimeError("identity health endpoint returned invalid JSON") from exc
        if not isinstance(payload, Mapping):
            raise RuntimeError("identity health response must be an object")
        return payload
    finally:
        connection.close()


def _validate_health_snapshot(
    payload: Mapping[str, Any],
    *,
    expected_layer: str,
    expected_dimension: int,
) -> dict[str, object]:
    if payload.get("schema_version") != 2:
        raise ValueError("identity-v2 health schema_version must be 2")
    if payload.get("runtime_mode") != "shadow":
        raise ValueError("identity-v2 runtime_mode must remain shadow")
    if payload.get("public_authority_cutover_status") != "blocked":
        raise ValueError("identity-v2 public authority must remain blocked")
    if payload.get("authority_cutover_artifact_id") is not None:
        raise ValueError("shadow runtime must not expose an authority cutover artifact")
    fingerprint = _nonempty_text(
        payload.get("runtime_model_fingerprint"), "runtime_model_fingerprint"
    )
    if SHA256_RE.fullmatch(fingerprint) is None:
        raise ValueError("runtime_model_fingerprint must be a lowercase SHA-256")
    if payload.get("runtime_model_layer") != expected_layer:
        raise ValueError(
            "runtime ReID layer does not match the reviewed pipeline config"
        )
    if payload.get("runtime_embedding_dim") != expected_dimension:
        raise ValueError(
            "runtime ReID dimension does not match the reviewed pipeline config"
        )
    cache_entries = _nonnegative_int(
        payload.get("observation_cache_entries"), "observation_cache_entries"
    )
    cache_max = _positive_int(
        payload.get("observation_cache_max_entries"),
        "observation_cache_max_entries",
    )
    if cache_entries > cache_max:
        raise ValueError("observation_cache_entries exceeds observation_cache_max_entries")
    cache_ttl = payload.get("observation_cache_ttl_s")
    if isinstance(cache_ttl, bool) or not isinstance(cache_ttl, (int, float)):
        raise ValueError("observation_cache_ttl_s must be numeric")
    cache_ttl_f = float(cache_ttl)
    if not math.isfinite(cache_ttl_f) or cache_ttl_f <= 0.0:
        raise ValueError("observation_cache_ttl_s must be finite and positive")
    scoring_status = _nonempty_text(
        payload.get("scoring_calibration_status"), "scoring_calibration_status"
    )
    scoring_scope = payload.get("scoring_authority_scope")
    if scoring_scope is not None and not isinstance(scoring_scope, str):
        raise ValueError("scoring_authority_scope must be a string or null")
    return {
        "runtime_mode": "shadow",
        "public_authority_cutover_status": "blocked",
        "runtime_model_fingerprint": fingerprint,
        "runtime_model_layer": expected_layer,
        "runtime_embedding_dim": expected_dimension,
        "scoring_calibration_status": scoring_status,
        "scoring_authority_scope": scoring_scope,
        "resident_count": _nonnegative_int(
            payload.get("resident_count"), "resident_count"
        ),
        "observation_cache_entries": cache_entries,
        "observation_cache_max_entries": cache_max,
        "observation_cache_ttl_s": cache_ttl_f,
    }


def _validate_health_pair(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    expected_layer: str,
    expected_dimension: int,
) -> tuple[dict[str, object], dict[str, object]]:
    first = _validate_health_snapshot(
        before,
        expected_layer=expected_layer,
        expected_dimension=expected_dimension,
    )
    second = _validate_health_snapshot(
        after,
        expected_layer=expected_layer,
        expected_dimension=expected_dimension,
    )
    for key in (
        "runtime_mode",
        "public_authority_cutover_status",
        "runtime_model_fingerprint",
        "runtime_model_layer",
        "runtime_embedding_dim",
        "scoring_calibration_status",
        "scoring_authority_scope",
        "resident_count",
        "observation_cache_max_entries",
        "observation_cache_ttl_s",
    ):
        if first[key] != second[key]:
            raise ValueError(f"identity-v2 health changed during gate: {key}")
    return first, second


@dataclass
class _TrackSegment:
    last_observed_at_us: int
    last_frame_id: int
    subject_id: str | None = None
    subject_frames: set[int] = field(default_factory=set)


@dataclass
class IdentityEvidenceCollector:
    continuity_gap_s: float = 2.0
    tracking_messages: int = 0
    person_rows: int = 0
    fresh_embedding_rows: int = 0
    held_rows: int = 0
    open_set_non_force_rows: int = 0
    overlap_permit_rows: int = 0
    continuity_subjects: int = 0
    errors: list[str] = field(default_factory=list)
    _segments: dict[tuple[str, str], _TrackSegment] = field(default_factory=dict)
    _continuous_subjects: set[str] = field(default_factory=set)
    _subject_cameras: dict[str, set[str]] = field(default_factory=dict)
    _run_ids: set[str] = field(default_factory=set)
    source_messages: list[dict[str, object]] = field(default_factory=list)
    source_row_count: int = 0

    def observe_payload(self, payload: object) -> None:
        if not isinstance(payload, Mapping) or payload.get("type") != "tracking":
            return
        self.tracking_messages += 1
        tracks = payload.get("tracks")
        if not isinstance(tracks, list):
            self.errors.append("tracking payload tracks must be a list")
            return
        if len(self.source_messages) >= MAX_SOURCE_MESSAGES:
            self.errors.append("identity source transcript message bound exceeded")
            return
        if self.source_row_count + len(tracks) > MAX_SOURCE_ROWS:
            self.errors.append("identity source transcript row bound exceeded")
            return
        self.source_messages.append(
            {
                "type": "tracking",
                "tracks": [self._source_track(row) for row in tracks],
            }
        )
        self.source_row_count += len(tracks)
        for index, raw_track in enumerate(tracks):
            try:
                self._observe_track(raw_track)
            except Exception as exc:
                self.errors.append(
                    f"tracking row {self.tracking_messages}:{index} invalid: {exc}"
                )

    @staticmethod
    def _source_track(raw_track: object) -> object:
        if not isinstance(raw_track, Mapping):
            return raw_track
        identity = raw_track.get("identity_v2")
        key = raw_track.get("identity_observation_key")
        return {
            name: raw_track.get(name)
            for name in (
                "class_id",
                "camera_id",
                "tracker_id",
                "frame_id",
                "observed_at_us",
                "embedding_present",
            )
        } | {
            "identity_observation_key": (
                {
                    name: key.get(name)
                    for name in (
                        "run_id",
                        "camera_id",
                        "tracker_id",
                        "frame_id",
                        "observation_id",
                    )
                }
                if isinstance(key, Mapping)
                else key
            ),
            "identity_v2": (
                {
                    name: identity.get(name)
                    for name in (
                        "mode",
                        "state",
                        "reason",
                        "subject_id",
                        "compatibility_sid",
                        "display_name",
                        "resident_uuid",
                        "visitor_generation",
                        "calibrated_confidence",
                        "overlap_permit",
                        "fresh_embedding",
                    )
                }
                if isinstance(identity, Mapping)
                else identity
            ),
        }

    def _observe_track(self, raw_track: object) -> None:
        if not isinstance(raw_track, Mapping):
            raise ValueError("track must be an object")
        class_id = raw_track.get("class_id")
        if class_id != 0:
            raise ValueError("public tracking rows must be person class 0")
        self.person_rows += 1
        camera_id = _nonempty_text(raw_track.get("camera_id"), "camera_id")
        tracker_id = _nonempty_text(raw_track.get("tracker_id"), "tracker_id")
        frame_id = _nonnegative_int(raw_track.get("frame_id"), "frame_id")
        observed_at_us = _positive_int(
            raw_track.get("observed_at_us"), "observed_at_us"
        )
        identity = raw_track.get("identity_v2")
        if not isinstance(identity, Mapping):
            raise ValueError("identity_v2 object is required")
        if identity.get("mode") != "shadow":
            raise ValueError("identity_v2 mode must remain shadow")
        state = _nonempty_text(identity.get("state"), "identity_v2.state")
        if state not in ALLOWED_STATES:
            raise ValueError(f"unsupported identity_v2 state {state!r}")
        raw_fresh = identity.get("fresh_embedding", False)
        if not isinstance(raw_fresh, bool):
            raise ValueError("identity_v2.fresh_embedding must be boolean")
        fresh = bool(raw_fresh)
        key = raw_track.get("identity_observation_key")
        if fresh:
            self._validate_observation_key(
                key,
                camera_id=camera_id,
                tracker_id=tracker_id,
                frame_id=frame_id,
            )
            if raw_track.get("embedding_present") is not True:
                raise ValueError(
                    "fresh identity observation must declare embedding_present=true"
                )
            self.fresh_embedding_rows += 1
        else:
            if key is not None:
                raise ValueError(
                    "non-fresh identity row must not expose identity_observation_key"
                )
            self.held_rows += 1

        subject_raw = identity.get("subject_id")
        subject_id = None if subject_raw is None else _nonempty_text(
            subject_raw, "identity_v2.subject_id"
        )
        compatibility_sid = identity.get("compatibility_sid")
        calibrated_confidence = identity.get("calibrated_confidence")
        reason = _nonempty_text(identity.get("reason"), "identity_v2.reason")
        if calibrated_confidence is not None:
            if isinstance(calibrated_confidence, bool) or not isinstance(
                calibrated_confidence, (int, float)
            ):
                raise ValueError("calibrated_confidence must be numeric or null")
            confidence = float(calibrated_confidence)
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("calibrated_confidence must be within [0,1]")

        if state == "unknown":
            if subject_id is not None or compatibility_sid is not None:
                raise ValueError("unknown shadow decision must not expose a subject or SID")
            for field_name in ("display_name", "resident_uuid", "visitor_generation"):
                if identity.get(field_name) is not None:
                    raise ValueError(
                        f"unknown shadow decision must clear {field_name}"
                    )
            if fresh:
                if calibrated_confidence is None:
                    raise ValueError(
                        "fresh unknown decision must expose calibrated_confidence"
                    )
                if not reason:
                    raise ValueError("fresh unknown decision requires a reason")
                self.open_set_non_force_rows += 1
        elif state == "provisional":
            if subject_id is not None or compatibility_sid is not None:
                raise ValueError("provisional decision must not expose a subject or SID")
        elif state == "resident":
            if subject_id is None or not subject_id.startswith("resident:"):
                raise ValueError("resident decision requires a resident subject")
            _positive_int(compatibility_sid, "resident compatibility_sid")
            resident_uuid = _nonempty_text(
                identity.get("resident_uuid"), "resident_uuid"
            )
            if subject_id != f"resident:{resident_uuid}":
                raise ValueError("resident subject and resident_uuid disagree")
            _nonempty_text(identity.get("display_name"), "resident display_name")
            if identity.get("visitor_generation") is not None:
                raise ValueError("resident decision must clear visitor_generation")
        elif state == "visitor":
            if subject_id is None or not subject_id.startswith("visitor:"):
                raise ValueError("visitor decision requires a visitor subject")
            _positive_int(compatibility_sid, "visitor compatibility_sid")
            _nonnegative_int(
                identity.get("visitor_generation"), "visitor_generation"
            )
            if identity.get("resident_uuid") is not None:
                raise ValueError("visitor decision must clear resident_uuid")
            if identity.get("display_name") is not None:
                raise ValueError("visitor decision must clear display_name")

        overlap = identity.get("overlap_permit", False)
        if not isinstance(overlap, bool):
            raise ValueError("identity_v2.overlap_permit must be boolean")
        if overlap:
            if subject_id is None:
                raise ValueError("overlap permit requires an accepted subject")
            self.overlap_permit_rows += 1

        if fresh:
            self._record_continuity(
                camera_id=camera_id,
                tracker_id=tracker_id,
                frame_id=frame_id,
                observed_at_us=observed_at_us,
                subject_id=subject_id,
            )

    def _validate_observation_key(
        self,
        key: object,
        *,
        camera_id: str,
        tracker_id: str,
        frame_id: int,
    ) -> None:
        if not isinstance(key, Mapping):
            raise ValueError("fresh identity row requires identity_observation_key")
        run_id = _nonempty_text(key.get("run_id"), "observation run_id")
        if key.get("camera_id") != camera_id:
            raise ValueError("identity observation camera_id mismatch")
        if str(key.get("tracker_id")) != tracker_id:
            raise ValueError("identity observation tracker_id mismatch")
        if key.get("frame_id") != frame_id:
            raise ValueError("identity observation frame_id mismatch")
        observation_id = _nonempty_text(
            key.get("observation_id"), "observation_id"
        )
        if OBSERVATION_ID_RE.fullmatch(observation_id) is None:
            raise ValueError("identity observation_id must be obs1:<sha256>")
        self._run_ids.add(run_id)

    def _record_continuity(
        self,
        *,
        camera_id: str,
        tracker_id: str,
        frame_id: int,
        observed_at_us: int,
        subject_id: str | None,
    ) -> None:
        key = (camera_id, tracker_id)
        segment = self._segments.get(key)
        gap_us = int(max(0.0, self.continuity_gap_s) * 1_000_000)
        if segment is not None:
            if observed_at_us < segment.last_observed_at_us:
                raise ValueError("tracker observation time moved backwards")
            if (
                observed_at_us - segment.last_observed_at_us <= gap_us
                and frame_id <= segment.last_frame_id
            ):
                raise ValueError("tracker frame_id replayed within one continuity segment")
            if observed_at_us - segment.last_observed_at_us > gap_us:
                segment = None
        if segment is None:
            segment = _TrackSegment(
                last_observed_at_us=observed_at_us,
                last_frame_id=frame_id,
            )
            self._segments[key] = segment
        segment.last_observed_at_us = observed_at_us
        segment.last_frame_id = frame_id
        if subject_id is None:
            return
        if segment.subject_id is not None and segment.subject_id != subject_id:
            raise ValueError("identity subject flipped within tracker continuity segment")
        segment.subject_id = subject_id
        segment.subject_frames.add(frame_id)
        self._subject_cameras.setdefault(subject_id, set()).add(camera_id)
        if len(segment.subject_frames) >= 2:
            self._continuous_subjects.add(subject_id)
            self.continuity_subjects = len(self._continuous_subjects)

    @property
    def run_id_count(self) -> int:
        return len(self._run_ids)

    @property
    def observed_run_id(self) -> str | None:
        return next(iter(self._run_ids)) if len(self._run_ids) == 1 else None

    @property
    def cross_camera_subject_count(self) -> int:
        return sum(1 for cameras in self._subject_cameras.values() if len(cameras) >= 2)

    @property
    def cross_camera_pair_count(self) -> int:
        pairs: set[tuple[str, str]] = set()
        for cameras in self._subject_cameras.values():
            ordered = sorted(cameras)
            for index, camera_a in enumerate(ordered):
                for camera_b in ordered[index + 1 :]:
                    pairs.add((camera_a, camera_b))
        return len(pairs)


async def _collect_tracking(
    uri: str,
    auth: RequiredInternalAuth,
    *,
    duration_s: float,
    collector: IdentityEvidenceCollector,
) -> None:
    deadline = time.monotonic() + duration_s
    async with connect_required_websocket(
        uri,
        auth,
        compression=None,
        max_size=4 * 1024 * 1024,
        open_timeout=10.0,
        close_timeout=5.0,
    ) as websocket:
        while time.monotonic() < deadline:
            timeout = min(2.0, max(0.05, deadline - time.monotonic()))
            try:
                raw = await asyncio.wait_for(websocket.recv(), timeout=timeout)
            except asyncio.TimeoutError:
                continue
            if not isinstance(raw, str):
                collector.errors.append("WebSocket tracking message must be text JSON")
                continue
            try:
                payload = strict_json_loads(raw, label="identity tracking message")
            except Exception:
                collector.errors.append("WebSocket emitted invalid JSON")
                continue
            collector.observe_payload(payload)


def _claim(status: str, *, required: bool, evidence_count: int) -> dict[str, object]:
    return {
        "status": status,
        "required": bool(required),
        "evidence_count": int(evidence_count),
    }


def _source_health_snapshot(payload: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError("identity source health snapshot must be an object")
    return {key: payload.get(key) for key in sorted(HEALTH_SOURCE_FIELDS)}


def _source_transcript_document(
    *,
    session_id: str,
    runtime_lane: str,
    runtime_instance_id: str,
    runtime_run_id: str,
    collector: IdentityEvidenceCollector,
    health_before: Mapping[str, object],
    health_after: Mapping[str, object],
    require_cross_camera: bool,
    require_open_set: bool,
    min_fresh_embeddings: int,
) -> dict[str, object]:
    if type(require_cross_camera) is not bool or type(require_open_set) is not bool:  # noqa: E721
        raise ValueError("identity source requirement flags must be boolean")
    return {
        "schema_version": SOURCE_TRANSCRIPT_VERSION,
        "contract": SOURCE_TRANSCRIPT_CONTRACT,
        "contract_version": SOURCE_TRANSCRIPT_VERSION,
        "session_id": str(session_id).strip().lower(),
        "runtime_lane": str(runtime_lane).strip().lower(),
        "runtime_instance_id": str(runtime_instance_id),
        "runtime_run_id": str(runtime_run_id),
        "policy": {
            "continuity_gap_s": float(collector.continuity_gap_s),
            "require_cross_camera": require_cross_camera,
            "require_open_set": require_open_set,
            "min_fresh_embeddings": _positive_int(
                min_fresh_embeddings, "min_fresh_embeddings"
            ),
        },
        "health_before": _source_health_snapshot(health_before),
        "health_after": _source_health_snapshot(health_after),
        "message_count": len(collector.source_messages),
        "row_count": collector.source_row_count,
        "messages": list(collector.source_messages),
    }


def _source_evidence_metadata(
    *,
    filename: str,
    encoded: bytes,
    document: Mapping[str, object],
    collector: IdentityEvidenceCollector,
) -> dict[str, object]:
    observed = [
        int(track["observed_at_us"])
        for message in collector.source_messages
        for track in message.get("tracks", [])
        if isinstance(track, Mapping)
        and _is_int(track.get("observed_at_us"))
        and int(track["observed_at_us"]) > 0
    ]
    return {
        "filename": filename,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "message_count": document["message_count"],
        "row_count": document["row_count"],
        "first_observed_at_us": min(observed) if observed else None,
        "last_observed_at_us": max(observed) if observed else None,
        "observed_run_id": collector.observed_run_id,
    }


def _build_report(
    *,
    session_id: str,
    runtime_lane: str,
    runtime_instance_id: str,
    runtime_run_id: str,
    collector: IdentityEvidenceCollector,
    health_before: Mapping[str, object] | None,
    health_after: Mapping[str, object] | None,
    require_cross_camera: bool,
    require_open_set: bool,
    min_fresh_embeddings: int,
    source_evidence: Mapping[str, object],
    errors: list[str],
) -> dict[str, object]:
    session_id = str(session_id).strip().lower()
    runtime_lane = str(runtime_lane).strip().lower()
    if SESSION_RE.fullmatch(session_id) is None:
        raise ValueError("session_id must match [a-z0-9][a-z0-9-]{5,47}")
    if runtime_lane not in RUNTIME_LANES:
        raise ValueError(f"unsupported runtime lane: {runtime_lane!r}")
    if (
        RUNTIME_ID_RE.fullmatch(str(runtime_instance_id)) is None
        or RUNTIME_ID_RE.fullmatch(str(runtime_run_id)) is None
    ):
        raise ValueError("runtime instance/run identity is invalid")
    combined_errors = [*errors, *collector.errors]
    if collector.tracking_messages < 1:
        combined_errors.append("no tracking messages observed")
    if collector.person_rows < 1:
        combined_errors.append("no occupied person rows observed")
    if collector.fresh_embedding_rows < min_fresh_embeddings:
        combined_errors.append(
            "insufficient fresh server-produced ReID observations: "
            f"{collector.fresh_embedding_rows} < {min_fresh_embeddings}"
        )
    if collector.run_id_count != 1:
        combined_errors.append(
            f"fresh identity observations must bind exactly one run_id, got {collector.run_id_count}"
        )
    elif collector.observed_run_id != str(runtime_run_id):
        combined_errors.append(
            "identity observation run_id differs from the supervisor runtime run_id"
        )
    if collector.continuity_subjects < 1:
        combined_errors.append(
            "no accepted shadow subject persisted across fresh frames on one tracker"
        )
    cross_observed = collector.cross_camera_subject_count > 0
    open_observed = collector.open_set_non_force_rows > 0
    if require_cross_camera and not cross_observed:
        combined_errors.append(
            "required cross-camera shadow assignment continuity was not observed"
        )
    if require_open_set and not open_observed:
        combined_errors.append(
            "required fresh open-set non-force shadow decision was not observed"
        )
    runtime_ok = health_before is not None and health_after is not None
    continuity_ok = collector.continuity_subjects >= 1
    claims = {
        "runtime_shadow_health": _claim(
            "pass" if runtime_ok else "fail", required=True, evidence_count=2 if runtime_ok else 0
        ),
        "tracker_subject_continuity": _claim(
            "pass" if continuity_ok else "fail",
            required=True,
            evidence_count=collector.continuity_subjects,
        ),
        "cross_camera_assignment_continuity": _claim(
            "observed" if cross_observed else "not_observed",
            required=require_cross_camera,
            evidence_count=collector.cross_camera_subject_count,
        ),
        "open_set_non_force": _claim(
            "observed" if open_observed else "not_observed",
            required=require_open_set,
            evidence_count=collector.open_set_non_force_rows,
        ),
        "semantic_accuracy": {
            "status": "not_evaluated",
            "required": False,
            "reason": "live telemetry has no licensed resident/unknown truth labels",
        },
        "public_authority": {
            "status": "blocked",
            "required": True,
            "reason": "identity-v2 runtime remained shadow for the full gate",
        },
    }
    return {
        "schema_version": 1,
        "contract": CONTRACT,
        "contract_version": CONTRACT_VERSION,
        "session_id": session_id,
        "runtime_lane": runtime_lane,
        "runtime_instance_id": str(runtime_instance_id),
        "runtime_run_id": str(runtime_run_id),
        "ok": not combined_errors,
        "claims": claims,
        "counts": {
            "tracking_messages": collector.tracking_messages,
            "person_rows": collector.person_rows,
            "fresh_embedding_rows": collector.fresh_embedding_rows,
            "held_rows": collector.held_rows,
            "continuity_subjects": collector.continuity_subjects,
            "cross_camera_subjects": collector.cross_camera_subject_count,
            "cross_camera_pairs": collector.cross_camera_pair_count,
            "open_set_non_force_rows": collector.open_set_non_force_rows,
            "overlap_permit_rows": collector.overlap_permit_rows,
            "run_id_count": collector.run_id_count,
        },
        "health_before": dict(health_before or {}),
        "health_after": dict(health_after or {}),
        "source_evidence": dict(source_evidence),
        "errors": combined_errors,
    }


def _encoded_private_json(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _canonical_evidence_filenames(runtime_lane: str) -> tuple[str, str]:
    lane = str(runtime_lane).strip().lower()
    if lane == "baseline":
        return (
            BASELINE_CANONICAL_REPORT_FILENAME,
            BASELINE_SOURCE_TRANSCRIPT_FILENAME,
        )
    if lane == "v3dt":
        return V3DT_CANONICAL_REPORT_FILENAME, V3DT_SOURCE_TRANSCRIPT_FILENAME
    if lane in {"wholebody49-s", "wholebody49-x"}:
        return (
            AUXILIARY_CANONICAL_REPORT_FILENAME,
            AUXILIARY_SOURCE_TRANSCRIPT_FILENAME,
        )
    raise ValueError("identity replay received an unsupported runtime lane")


def _read_private_json_object(
    path: Path,
    *,
    expected_filename: str,
    max_bytes: int,
    label: str,
) -> tuple[bytes, Mapping[str, object]]:
    candidate = Path(path).expanduser().absolute()
    if candidate.name != expected_filename:
        raise ValueError(f"{label} filename is not canonical")
    raw = read_private_file(candidate, label=label, max_bytes=max_bytes)
    if not raw:
        raise ValueError(f"{label} is empty")
    document = strict_json_loads(raw, label=label)
    if not isinstance(document, Mapping):
        raise ValueError(f"{label} must be an object")
    return raw, document


def validate_sealed_identity_shadow_report(
    report_path: Path,
    source_path: Path,
    *,
    pipeline_config: Path,
    session_id: str,
    runtime_lane: str,
    runtime_instance_id: str,
    runtime_run_id: str,
    require_cross_camera: bool,
    require_open_set: bool,
    min_fresh_embeddings: int = 2,
) -> Mapping[str, object]:
    """Strictly replay one owner-private identity report from its sealed source."""

    if Path(report_path).expanduser().absolute().parent != Path(
        source_path
    ).expanduser().absolute().parent:
        raise ValueError("identity report and source must share one evidence directory")
    report_filename, source_filename = _canonical_evidence_filenames(runtime_lane)
    report_raw, report = _read_private_json_object(
        report_path,
        expected_filename=report_filename,
        max_bytes=MAX_REPORT_BYTES,
        label="identity shadow report",
    )
    source_raw, source = _read_private_json_object(
        source_path,
        expected_filename=source_filename,
        max_bytes=MAX_SOURCE_TRANSCRIPT_BYTES,
        label="identity shadow source transcript",
    )
    expected_source_fields = {
        "schema_version",
        "contract",
        "contract_version",
        "session_id",
        "runtime_lane",
        "runtime_instance_id",
        "runtime_run_id",
        "policy",
        "health_before",
        "health_after",
        "message_count",
        "row_count",
        "messages",
    }
    if set(source) != expected_source_fields:
        raise ValueError("identity source transcript schema drifted")
    if (
        type(source.get("schema_version")) is not int  # noqa: E721
        or source.get("schema_version") != SOURCE_TRANSCRIPT_VERSION
        or source.get("contract") != SOURCE_TRANSCRIPT_CONTRACT
        or type(source.get("contract_version")) is not int  # noqa: E721
        or source.get("contract_version") != SOURCE_TRANSCRIPT_VERSION
        or source.get("session_id") != str(session_id).strip().lower()
        or source.get("runtime_lane") != str(runtime_lane).strip().lower()
        or source.get("runtime_instance_id") != str(runtime_instance_id)
        or source.get("runtime_run_id") != str(runtime_run_id)
    ):
        raise ValueError("identity source transcript identity binding drifted")
    policy = source.get("policy")
    if not isinstance(policy, Mapping) or set(policy) != {
        "continuity_gap_s",
        "require_cross_camera",
        "require_open_set",
        "min_fresh_embeddings",
    }:
        raise ValueError("identity source transcript policy drifted")
    gap = policy.get("continuity_gap_s")
    if (
        isinstance(gap, bool)
        or not isinstance(gap, (int, float))
        or not math.isfinite(float(gap))
        or float(gap) <= 0.0
        or policy.get("require_cross_camera") is not require_cross_camera
        or policy.get("require_open_set") is not require_open_set
        or policy.get("min_fresh_embeddings") != _positive_int(
            min_fresh_embeddings, "min_fresh_embeddings"
        )
    ):
        raise ValueError("identity source transcript policy binding drifted")
    messages = source.get("messages")
    if not isinstance(messages, list):
        raise ValueError("identity source transcript messages must be a list")
    collector = IdentityEvidenceCollector(continuity_gap_s=float(gap))
    for message in messages:
        collector.observe_payload(message)
    expected_layer, expected_dimension = _load_expected_reid_contract(
        Path(pipeline_config)
    )
    health_before_raw = source.get("health_before")
    health_after_raw = source.get("health_after")
    if (
        not isinstance(health_before_raw, Mapping)
        or set(health_before_raw) != HEALTH_SOURCE_FIELDS
        or not isinstance(health_after_raw, Mapping)
        or set(health_after_raw) != HEALTH_SOURCE_FIELDS
    ):
        raise ValueError("identity source health evidence schema drifted")
    health_before, health_after = _validate_health_pair(
        health_before_raw,
        health_after_raw,
        expected_layer=expected_layer,
        expected_dimension=expected_dimension,
    )
    replay_errors: list[str] = []
    if int(health_after["observation_cache_entries"]) < 1:
        replay_errors.append(
            "identity observation cache remained empty after live evidence"
        )
    canonical_source = _source_transcript_document(
        session_id=session_id,
        runtime_lane=runtime_lane,
        runtime_instance_id=runtime_instance_id,
        runtime_run_id=runtime_run_id,
        collector=collector,
        health_before=health_before_raw,
        health_after=health_after_raw,
        require_cross_camera=require_cross_camera,
        require_open_set=require_open_set,
        min_fresh_embeddings=min_fresh_embeddings,
    )
    if source != canonical_source or source_raw != _encoded_private_json(canonical_source):
        raise ValueError("identity source transcript is not canonical")
    source_evidence = _source_evidence_metadata(
        filename=source_filename,
        encoded=source_raw,
        document=source,
        collector=collector,
    )
    recomputed = _build_report(
        session_id=session_id,
        runtime_lane=runtime_lane,
        runtime_instance_id=runtime_instance_id,
        runtime_run_id=runtime_run_id,
        collector=collector,
        health_before=health_before,
        health_after=health_after,
        require_cross_camera=require_cross_camera,
        require_open_set=require_open_set,
        min_fresh_embeddings=min_fresh_embeddings,
        source_evidence=source_evidence,
        errors=replay_errors,
    )
    if report != recomputed or report_raw != _encoded_private_json(recomputed):
        raise ValueError("identity report does not exactly replay from sealed source")
    if report.get("ok") is not True or report.get("errors") != []:
        raise ValueError("identity report is not a passing exact replay")
    return report


def _write_private_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    max_bytes: int = MAX_REPORT_BYTES,
) -> bytes:
    path = path.expanduser().absolute()
    allowed = {
        BASELINE_CANONICAL_REPORT_FILENAME,
        V3DT_CANONICAL_REPORT_FILENAME,
        AUXILIARY_CANONICAL_REPORT_FILENAME,
        BASELINE_SOURCE_TRANSCRIPT_FILENAME,
        V3DT_SOURCE_TRANSCRIPT_FILENAME,
        AUXILIARY_SOURCE_TRANSCRIPT_FILENAME,
    }
    if path.name not in allowed:
        raise ValueError("identity evidence output filename is not canonical")
    encoded = _encoded_private_json(payload)
    try:
        atomic_create_private_file(
            path,
            encoded,
            label=f"immutable identity evidence {path.name}",
            max_bytes=max_bytes,
        )
    except PrivatePathError as exc:
        raise RuntimeError(str(exc)) from exc
    return encoded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate DS9 identity-v2 shadow wiring and live evidence."
    )
    parser.add_argument("--ws", default="ws://127.0.0.1:6008")
    parser.add_argument("--rest", default="http://127.0.0.1:8080")
    parser.add_argument(
        "--pipeline-config", type=Path, default=Path("DS9/config/infer.yaml")
    )
    parser.add_argument("--duration", type=float, default=35.0)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--runtime-lane", choices=tuple(sorted(RUNTIME_LANES)), required=True)
    parser.add_argument("--runtime-instance-id", required=True)
    parser.add_argument("--runtime-run-id", required=True)
    parser.add_argument("--continuity-gap-s", type=float, default=2.0)
    parser.add_argument("--min-fresh-embeddings", type=int, default=2)
    parser.add_argument("--require-cross-camera", action="store_true")
    parser.add_argument("--require-open-set", action="store_true")
    parser.add_argument("--source-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    add_auth_token_file_argument(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require_fresh_private_file_bundle(
            (args.source_out, args.out),
            label="identity behavior evidence bundle",
        )
    except (PrivatePathError, ValueError) as exc:
        print(f"[FAIL] identity evidence session is not fresh: {exc}", file=sys.stderr)
        return 1
    errors: list[str] = []
    before_summary: Mapping[str, object] | None = None
    after_summary: Mapping[str, object] | None = None
    before_raw: Mapping[str, object] | None = None
    after_raw: Mapping[str, object] | None = None
    collector = IdentityEvidenceCollector(
        continuity_gap_s=float(args.continuity_gap_s)
    )
    try:
        if not math.isfinite(float(args.duration)) or float(args.duration) <= 0.0:
            raise ValueError("duration must be finite and positive")
        if (
            not math.isfinite(float(args.continuity_gap_s))
            or float(args.continuity_gap_s) <= 0.0
        ):
            raise ValueError("continuity gap must be finite and positive")
        min_fresh = _positive_int(
            args.min_fresh_embeddings, "min-fresh-embeddings"
        )
        expected_layer, expected_dimension = _load_expected_reid_contract(
            args.pipeline_config
        )
        auth = load_required_internal_auth(args.auth_token_file)
        before_raw = _http_json_get(
            args.rest,
            "/api/v2/reid/health",
            auth,
            timeout_s=5.0,
        )
        before_summary = _validate_health_snapshot(
            before_raw,
            expected_layer=expected_layer,
            expected_dimension=expected_dimension,
        )
        asyncio.run(
            _collect_tracking(
                args.ws,
                auth,
                duration_s=float(args.duration),
                collector=collector,
            )
        )
        after_raw = _http_json_get(
            args.rest,
            "/api/v2/reid/health",
            auth,
            timeout_s=5.0,
        )
        before_summary, after_summary = _validate_health_pair(
            before_raw,
            after_raw,
            expected_layer=expected_layer,
            expected_dimension=expected_dimension,
        )
        if int(after_summary["observation_cache_entries"]) < 1:
            errors.append("identity observation cache remained empty after live evidence")
    except Exception as exc:
        min_fresh = max(1, int(getattr(args, "min_fresh_embeddings", 2) or 2))
        errors.append(f"{type(exc).__name__}: {exc}")

    source_health_before = before_raw if before_raw is not None else {}
    source_health_after = after_raw if after_raw is not None else {}
    source_document = _source_transcript_document(
        session_id=args.session_id,
        runtime_lane=args.runtime_lane,
        runtime_instance_id=args.runtime_instance_id,
        runtime_run_id=args.runtime_run_id,
        collector=collector,
        health_before=source_health_before,
        health_after=source_health_after,
        require_cross_camera=bool(args.require_cross_camera),
        require_open_set=bool(args.require_open_set),
        min_fresh_embeddings=min_fresh,
    )
    source_encoded = _encoded_private_json(source_document)
    source_evidence = _source_evidence_metadata(
        filename=args.source_out.name,
        encoded=source_encoded,
        document=source_document,
        collector=collector,
    )
    try:
        _write_private_json(
            args.source_out,
            source_document,
            max_bytes=MAX_SOURCE_TRANSCRIPT_BYTES,
        )
    except Exception as exc:
        print(
            f"[FAIL] identity source evidence was not published; use a fresh session: {exc}",
            file=sys.stderr,
        )
        return 1
    report = _build_report(
        session_id=args.session_id,
        runtime_lane=args.runtime_lane,
        runtime_instance_id=args.runtime_instance_id,
        runtime_run_id=args.runtime_run_id,
        collector=collector,
        health_before=before_summary,
        health_after=after_summary,
        require_cross_camera=bool(args.require_cross_camera),
        require_open_set=bool(args.require_open_set),
        min_fresh_embeddings=min_fresh,
        source_evidence=source_evidence,
        errors=errors,
    )
    try:
        _write_private_json(args.out, report)
    except Exception as exc:
        print(f"[FAIL] unable to write identity gate report: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
