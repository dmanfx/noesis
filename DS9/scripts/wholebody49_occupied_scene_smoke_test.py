#!/usr/bin/env python3
"""Validate occupied-scene Wholebody49 parser/mask-or-box runtime behavior."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence


DS9_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DS9_ROOT.parent
for _path in (str(REPO_ROOT), str(DS9_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

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


CONTRACT = "noesis.ds9.wholebody49_occupied_scene_gate"
CONTRACT_VERSION = 2
CANONICAL_REPORT_FILENAME = "wholebody49-occupied-scene.json"
CANONICAL_SOURCE_TRANSCRIPT_FILENAME = "wholebody49-occupied-scene-source.json"
SOURCE_TRANSCRIPT_CONTRACT = "noesis.ds9.wholebody49-source-transcript"
SOURCE_PRIVACY_POLICY = {
    "payload_policy": "minimal_canonical_fields",
    "raw_embedding_vectors": "absent",
    "image_frames": "absent",
    "secrets": "absent",
}
MAX_SOURCE_MESSAGES = 4096
MAX_SOURCE_TRANSCRIPT_BYTES = 8 * 1024 * 1024
MAX_REPORT_BYTES = 256 * 1024
MIN_TRACKING_WINDOW_SECONDS = 30.0
MIN_SOURCE_FRAME_RATE = 2.0
MAX_SOURCE_OBSERVATION_GAP_SECONDS = 2.5
SESSION_RE = re.compile(r"^[a-z0-9][a-z0-9-]{5,47}$")
RUNTIME_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
MASK_COUNTER = "detection_wake.object_depth_native_mask_stats"
BBOX_COUNTER = "detection_wake.object_depth_native_stats"
CPU_VIOLATION_COUNTER = "core_path.cpu_copy_violation.total"


def _thresholds() -> dict[str, float]:
    return {
        "minimum_tracking_window_seconds": MIN_TRACKING_WINDOW_SECONDS,
        "minimum_source_frame_rate": MIN_SOURCE_FRAME_RATE,
        "maximum_source_observation_gap_seconds": (
            MAX_SOURCE_OBSERVATION_GAP_SECONDS
        ),
    }


def _expected_source_ids(values: Sequence[object]) -> tuple[int, ...]:
    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("expected source IDs must be exact non-negative integers")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "expected source IDs must be exact non-negative integers"
            ) from exc
        if parsed < 0 or str(value).strip() != str(parsed):
            raise ValueError("expected source IDs must be canonical non-negative integers")
        normalized.append(parsed)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("expected source IDs must be non-empty and unique")
    return tuple(sorted(normalized))


def analyze_evidence(
    *,
    session_id: str,
    runtime_lane: str,
    runtime_instance_id: str,
    runtime_run_id: str,
    mode: str,
    expected_source_ids: Sequence[object],
    events: Sequence[object],
    source_evidence: Mapping[str, object] | None = None,
) -> dict[str, object]:
    mode = str(mode).strip().lower()
    if mode not in {"masks", "boxes"}:
        raise ValueError(f"unsupported Wholebody49 mode: {mode!r}")
    session_id = str(session_id).strip().lower()
    runtime_lane = str(runtime_lane).strip().lower()
    expected_lane = "wholebody49-s" if mode == "masks" else "wholebody49-x"
    if SESSION_RE.fullmatch(session_id) is None:
        raise ValueError("session_id must match [a-z0-9][a-z0-9-]{5,47}")
    if runtime_lane != expected_lane:
        raise ValueError(
            f"Wholebody49 {mode} evidence requires runtime lane {expected_lane!r}"
        )
    if (
        RUNTIME_ID_RE.fullmatch(str(runtime_instance_id)) is None
        or RUNTIME_ID_RE.fullmatch(str(runtime_run_id)) is None
    ):
        raise ValueError("runtime instance/run identity is invalid")
    normalized_sources = _expected_source_ids(expected_source_ids)
    replay = _replay_source_events(events, expected_source_ids=normalized_sources)
    counters = replay["counters"]
    mask_samples = int(counters[MASK_COUNTER])
    bbox_samples = int(counters[BBOX_COUNTER])
    cpu_violations = int(counters[CPU_VIOLATION_COUNTER])
    source_metrics = replay["source_metrics"]
    observed_sources = tuple(int(row["source_id"]) for row in source_metrics)
    checks = {
        "tracking_messages": int(replay["tracking_messages"]) > 0,
        "occupied_person_tracks": int(replay["tracks_seen"]) > 0,
        "stats_samples": int(replay["stats_samples"]) > 0,
        "pipeline_ready": replay["pipeline_ready"] is True,
        "pipeline_errors_absent": not replay["pipeline_errors"],
        "core_cpu_copy_violations_absent": cpu_violations == 0,
        "counters_monotonic": replay["counters_monotonic"] is True,
        "source_inventory_exact": observed_sources == normalized_sources,
        "source_frames_advancing": bool(source_metrics)
        and all(row["frames_strictly_advance"] is True for row in source_metrics),
        "tracking_window_sufficient": float(replay["tracking_window_seconds"])
        >= MIN_TRACKING_WINDOW_SECONDS,
        "source_frame_rate_sufficient": bool(source_metrics)
        and all(
            float(row["frame_rate"]) >= MIN_SOURCE_FRAME_RATE
            for row in source_metrics
        ),
        "tracking_cadence_within_limit": bool(source_metrics)
        and all(
            float(row["maximum_observation_gap_seconds"])
            <= MAX_SOURCE_OBSERVATION_GAP_SECONDS
            for row in source_metrics
        ),
        "mask_path_active": mask_samples > 0 if mode == "masks" else mask_samples == 0,
        "bbox_path_active": bbox_samples > 0 if mode == "boxes" else True,
    }
    occupied = int(replay["tracks_seen"]) > 0
    return {
        "schema_version": 1,
        "contract": CONTRACT,
        "contract_version": CONTRACT_VERSION,
        "session_id": session_id,
        "runtime_lane": runtime_lane,
        "runtime_instance_id": str(runtime_instance_id),
        "runtime_run_id": str(runtime_run_id),
        "ok": all(checks.values()),
        "status": "pass" if all(checks.values()) else "fail",
        "scene_status": "occupied" if occupied else "empty",
        "mode": mode,
        "expected_source_ids": list(normalized_sources),
        "observed_source_ids": list(observed_sources),
        "thresholds": _thresholds(),
        "checks": checks,
        "tracking_messages": int(replay["tracking_messages"]),
        "tracks_seen": int(replay["tracks_seen"]),
        "stats_samples": int(replay["stats_samples"]),
        "tracking_window_seconds": float(replay["tracking_window_seconds"]),
        "source_metrics": source_metrics,
        "counters": {
            MASK_COUNTER: mask_samples,
            BBOX_COUNTER: bbox_samples,
            CPU_VIOLATION_COUNTER: cpu_violations,
        },
        "pipeline_errors": list(replay["pipeline_errors"]),
        "source_evidence": dict(source_evidence or {}),
        "render_evidence_policy": (
            "pair_with_the_same_lane_rtsp_decode_and_supervisor_runtime_log; "
            "this gate proves parser output consumption, not subjective overlay aesthetics"
        ),
    }


def _replay_source_events(
    events: Sequence[object],
    *,
    expected_source_ids: Sequence[object],
) -> dict[str, object]:
    if len(events) > MAX_SOURCE_MESSAGES:
        raise ValueError("Wholebody49 source transcript message bound exceeded")
    expected_sources = _expected_source_ids(expected_source_ids)
    tracking_messages = 0
    tracks_seen = 0
    stats_samples = 0
    counters: dict[str, int] = {}
    pipeline_errors: list[str] = []
    counters_monotonic = True
    pipeline_ready = True
    frames_by_source: dict[int, list[tuple[int, int]]] = {}
    previous_observed_at_us = 0
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise ValueError(f"source event {index} must be an object")
        observed_at_us = event.get("observed_at_us")
        if (
            isinstance(observed_at_us, bool)
            or not isinstance(observed_at_us, int)
            or observed_at_us <= 0
        ):
            raise ValueError(f"source event {index} timestamp is invalid")
        if observed_at_us <= previous_observed_at_us:
            raise ValueError("source event timestamps must strictly advance")
        previous_observed_at_us = observed_at_us
        event_type = event.get("type")
        if event_type == "tracking":
            if set(event) != {
                "type",
                "observed_at_us",
                "source_id",
                "frame_id",
                "person_track_count",
            }:
                raise ValueError(f"source tracking event {index} schema drifted")
            source_id = event.get("source_id")
            frame_id = event.get("frame_id")
            count = event.get("person_track_count")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (source_id, frame_id, count)
            ):
                raise ValueError(f"source tracking event {index} count is invalid")
            tracking_messages += 1
            tracks_seen += count
            frames_by_source.setdefault(source_id, []).append(
                (observed_at_us, frame_id)
            )
        elif event_type == "stats":
            if set(event) != {
                "type",
                "observed_at_us",
                "counters",
                "pipeline_errors",
                "pipeline_prepared",
                "pipeline_activated",
                "application_running",
            }:
                raise ValueError(f"source stats event {index} schema drifted")
            raw_counters = event.get("counters")
            raw_errors = event.get("pipeline_errors")
            if not isinstance(raw_counters, Mapping) or set(raw_counters) != {
                MASK_COUNTER,
                BBOX_COUNTER,
                CPU_VIOLATION_COUNTER,
            }:
                raise ValueError(f"source stats event {index} counters drifted")
            if not isinstance(raw_errors, list) or any(
                item != "redacted_pipeline_error" for item in raw_errors
            ):
                raise ValueError(f"source stats event {index} errors are invalid")
            stats_samples += 1
            pipeline_errors = list(raw_errors)
            readiness_values = (
                event.get("pipeline_prepared"),
                event.get("pipeline_activated"),
                event.get("application_running"),
            )
            if any(type(value) is not bool for value in readiness_values):  # noqa: E721
                raise ValueError(f"source stats event {index} readiness drifted")
            pipeline_ready = pipeline_ready and all(readiness_values)
            for key, value in raw_counters.items():
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(
                        f"source stats event {index} counter {key} is invalid"
                    )
                previous = counters.get(str(key))
                if previous is not None and value < previous:
                    counters_monotonic = False
                counters[str(key)] = value
        else:
            raise ValueError(f"source event {index} type is invalid")
    for key in (MASK_COUNTER, BBOX_COUNTER, CPU_VIOLATION_COUNTER):
        counters.setdefault(key, 0)
    source_metrics: list[dict[str, object]] = []
    for source_id in sorted(frames_by_source):
        rows = frames_by_source[source_id]
        timestamps = [row[0] for row in rows]
        frame_ids = [row[1] for row in rows]
        span_seconds = (
            (timestamps[-1] - timestamps[0]) / 1_000_000.0
            if len(timestamps) >= 2
            else 0.0
        )
        frame_delta = frame_ids[-1] - frame_ids[0] if len(frame_ids) >= 2 else 0
        gaps = [
            (right - left) / 1_000_000.0
            for left, right in zip(timestamps, timestamps[1:])
        ]
        source_metrics.append(
            {
                "source_id": source_id,
                "message_count": len(rows),
                "first_frame_id": frame_ids[0],
                "last_frame_id": frame_ids[-1],
                "frame_delta": frame_delta,
                "observation_span_seconds": span_seconds,
                "frame_rate": frame_delta / span_seconds if span_seconds > 0 else 0.0,
                "maximum_observation_gap_seconds": max(gaps, default=0.0),
                "frames_strictly_advance": len(frame_ids) >= 2
                and all(left < right for left, right in zip(frame_ids, frame_ids[1:])),
            }
        )
    tracking_timestamps = [
        int(event["observed_at_us"])
        for event in events
        if isinstance(event, Mapping) and event.get("type") == "tracking"
    ]
    tracking_window_seconds = (
        (tracking_timestamps[-1] - tracking_timestamps[0]) / 1_000_000.0
        if len(tracking_timestamps) >= 2
        else 0.0
    )
    return {
        "expected_source_ids": list(expected_sources),
        "tracking_messages": tracking_messages,
        "tracks_seen": tracks_seen,
        "stats_samples": stats_samples,
        "counters": counters,
        "pipeline_errors": pipeline_errors,
        "pipeline_ready": pipeline_ready and stats_samples > 0,
        "counters_monotonic": counters_monotonic and stats_samples > 0,
        "tracking_window_seconds": tracking_window_seconds,
        "source_metrics": source_metrics,
    }


def _source_transcript_document(
    *,
    session_id: str,
    runtime_lane: str,
    runtime_instance_id: str,
    runtime_run_id: str,
    mode: str,
    expected_source_ids: Sequence[object],
    events: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "contract": SOURCE_TRANSCRIPT_CONTRACT,
        "contract_version": CONTRACT_VERSION,
        "session_id": str(session_id).strip().lower(),
        "runtime_lane": str(runtime_lane).strip().lower(),
        "runtime_instance_id": str(runtime_instance_id),
        "runtime_run_id": str(runtime_run_id),
        "mode": str(mode).strip().lower(),
        "expected_source_ids": list(_expected_source_ids(expected_source_ids)),
        "thresholds": _thresholds(),
        "privacy": dict(SOURCE_PRIVACY_POLICY),
        "message_count": len(events),
        "messages": [dict(event) for event in events],
    }


def _encoded_private_json(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _source_evidence_metadata(
    *,
    encoded: bytes,
    document: Mapping[str, object],
) -> dict[str, object]:
    raw_messages = document.get("messages")
    messages = raw_messages if isinstance(raw_messages, list) else []
    timestamps = [
        int(event["observed_at_us"])
        for event in messages
        if isinstance(event, Mapping)
        and isinstance(event.get("observed_at_us"), int)
        and not isinstance(event.get("observed_at_us"), bool)
    ]
    return {
        "filename": CANONICAL_SOURCE_TRANSCRIPT_FILENAME,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "message_count": document.get("message_count"),
        "first_observed_at_us": min(timestamps) if timestamps else None,
        "last_observed_at_us": max(timestamps) if timestamps else None,
    }


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


def validate_sealed_wholebody49_report(
    report_path: Path,
    source_path: Path,
    *,
    session_id: str,
    runtime_lane: str,
    runtime_instance_id: str,
    runtime_run_id: str,
    mode: str,
    expected_source_ids: Sequence[object],
) -> Mapping[str, object]:
    """Strictly replay a Wholebody49 occupied report from its sealed source."""

    if Path(report_path).expanduser().absolute().parent != Path(
        source_path
    ).expanduser().absolute().parent:
        raise ValueError(
            "Wholebody49 occupied report and source must share one evidence directory"
        )
    report_raw, report = _read_private_json_object(
        report_path,
        expected_filename=CANONICAL_REPORT_FILENAME,
        max_bytes=MAX_REPORT_BYTES,
        label="Wholebody49 occupied report",
    )
    source_raw, source = _read_private_json_object(
        source_path,
        expected_filename=CANONICAL_SOURCE_TRANSCRIPT_FILENAME,
        max_bytes=MAX_SOURCE_TRANSCRIPT_BYTES,
        label="Wholebody49 occupied source transcript",
    )
    expected_source_fields = {
        "schema_version",
        "contract",
        "contract_version",
        "session_id",
        "runtime_lane",
        "runtime_instance_id",
        "runtime_run_id",
        "mode",
        "expected_source_ids",
        "thresholds",
        "privacy",
        "message_count",
        "messages",
    }
    if set(source) != expected_source_fields:
        raise ValueError("Wholebody49 occupied source schema drifted")
    normalized_sources = _expected_source_ids(expected_source_ids)
    normalized_mode = str(mode).strip().lower()
    if (
        type(source.get("schema_version")) is not int  # noqa: E721
        or source.get("schema_version") != 1
        or source.get("contract") != SOURCE_TRANSCRIPT_CONTRACT
        or type(source.get("contract_version")) is not int  # noqa: E721
        or source.get("contract_version") != CONTRACT_VERSION
        or source.get("session_id") != str(session_id).strip().lower()
        or source.get("runtime_lane") != str(runtime_lane).strip().lower()
        or source.get("runtime_instance_id") != str(runtime_instance_id)
        or source.get("runtime_run_id") != str(runtime_run_id)
        or source.get("mode") != normalized_mode
        or source.get("expected_source_ids") != list(normalized_sources)
        or source.get("thresholds") != _thresholds()
        or source.get("privacy") != SOURCE_PRIVACY_POLICY
    ):
        raise ValueError("Wholebody49 occupied source binding drifted")
    messages = source.get("messages")
    if not isinstance(messages, list) or source.get("message_count") != len(messages):
        raise ValueError("Wholebody49 occupied source message count drifted")
    canonical_source = _source_transcript_document(
        session_id=session_id,
        runtime_lane=runtime_lane,
        runtime_instance_id=runtime_instance_id,
        runtime_run_id=runtime_run_id,
        mode=normalized_mode,
        expected_source_ids=normalized_sources,
        events=messages,
    )
    if source != canonical_source or source_raw != _encoded_private_json(canonical_source):
        raise ValueError("Wholebody49 occupied source is not canonical")
    source_evidence = _source_evidence_metadata(
        encoded=source_raw,
        document=source,
    )
    recomputed = analyze_evidence(
        session_id=session_id,
        runtime_lane=runtime_lane,
        runtime_instance_id=runtime_instance_id,
        runtime_run_id=runtime_run_id,
        mode=normalized_mode,
        expected_source_ids=normalized_sources,
        events=messages,
        source_evidence=source_evidence,
    )
    if report != recomputed or report_raw != _encoded_private_json(recomputed):
        raise ValueError(
            "Wholebody49 occupied report does not exactly replay from sealed source"
        )
    if report.get("ok") is not True or report.get("status") != "pass":
        raise ValueError("Wholebody49 occupied report is not a passing exact replay")
    return report


def _write_private_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    max_bytes: int = MAX_REPORT_BYTES,
) -> None:
    destination = path.expanduser().absolute()
    if destination.name not in {
        CANONICAL_REPORT_FILENAME,
        CANONICAL_SOURCE_TRANSCRIPT_FILENAME,
    }:
        raise ValueError("Wholebody49 evidence output filename is not canonical")
    encoded = _encoded_private_json(payload)
    try:
        atomic_create_private_file(
            destination,
            encoded,
            label=f"immutable Wholebody49 evidence {destination.name}",
            max_bytes=max_bytes,
        )
    except PrivatePathError as exc:
        raise ValueError(str(exc)) from exc


async def collect_evidence(
    uri: str,
    auth: RequiredInternalAuth,
    *,
    duration_s: float,
) -> list[dict[str, object]]:
    counters: dict[str, int] = {}
    pipeline_errors: list[str] = []
    source_events: list[dict[str, object]] = []
    last_observed_at_us = 0

    def observed_now_us() -> int:
        nonlocal last_observed_at_us
        observed = max(time.time_ns() // 1_000, last_observed_at_us + 1)
        last_observed_at_us = observed
        return observed

    async with connect_required_websocket(uri, auth, max_size=None) as websocket:
        await websocket.send(json.dumps({"type": "clear_stats"}, separators=(",", ":")))
        deadline = time.monotonic() + max(1.0, float(duration_s))
        while time.monotonic() < deadline:
            timeout_s = min(2.0, max(0.05, deadline - time.monotonic()))
            try:
                raw = await asyncio.wait_for(websocket.recv(), timeout=timeout_s)
            except asyncio.TimeoutError:
                continue
            if not isinstance(raw, str):
                continue
            try:
                payload = strict_json_loads(raw, label="Wholebody49 live message")
            except ValueError as exc:
                raise ValueError(
                    "Wholebody49 WebSocket emitted invalid strict JSON"
                ) from exc
            if not isinstance(payload, Mapping):
                continue
            if payload.get("type") == "tracking":
                if len(source_events) >= MAX_SOURCE_MESSAGES:
                    raise ValueError("Wholebody49 source transcript message bound exceeded")
                source_id = payload.get("source_id")
                frame_id = payload.get("frame_id")
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in (source_id, frame_id)
                ):
                    raise ValueError(
                        "Wholebody49 tracking payload lacks exact source/frame identity"
                    )
                rows = payload.get("tracks")
                person_track_count = (
                    len([row for row in rows if isinstance(row, Mapping)])
                    if isinstance(rows, list)
                    else 0
                )
                source_events.append(
                    {
                        "type": "tracking",
                        "observed_at_us": observed_now_us(),
                        "source_id": source_id,
                        "frame_id": frame_id,
                        "person_track_count": person_track_count,
                    }
                )
                continue
            if payload.get("type") != "stats":
                continue
            if len(source_events) >= MAX_SOURCE_MESSAGES:
                raise ValueError("Wholebody49 source transcript message bound exceeded")
            stats = payload.get("payload")
            pipeline = stats.get("pipeline") if isinstance(stats, Mapping) else None
            if not isinstance(pipeline, Mapping):
                continue
            errors = pipeline.get("errors")
            if isinstance(errors, list):
                observed_errors = [
                    "redacted_pipeline_error"
                    for item in errors
                    if str(item).strip()
                ][-16:]
                if observed_errors:
                    pipeline_errors = (pipeline_errors + observed_errors)[-16:]
            zero_copy = pipeline.get("zero_copy_core")
            application = stats.get("application") if isinstance(stats, Mapping) else None
            if not isinstance(application, Mapping):
                raise ValueError("Wholebody49 stats payload lacks application state")
            current = zero_copy.get("counters") if isinstance(zero_copy, Mapping) else None
            if isinstance(current, Mapping):
                for key, value in current.items():
                    try:
                        normalized = int(value or 0)
                    except (TypeError, ValueError):
                        continue
                    counters[str(key)] = max(counters.get(str(key), 0), normalized)
            source_events.append(
                {
                    "type": "stats",
                    "observed_at_us": observed_now_us(),
                    "counters": {
                        MASK_COUNTER: int(counters.get(MASK_COUNTER, 0)),
                        BBOX_COUNTER: int(counters.get(BBOX_COUNTER, 0)),
                        CPU_VIOLATION_COUNTER: int(
                            counters.get(CPU_VIOLATION_COUNTER, 0)
                        ),
                    },
                    "pipeline_errors": list(pipeline_errors),
                    "pipeline_prepared": pipeline.get("prepared") is True,
                    "pipeline_activated": pipeline.get("activated") is True,
                    "application_running": application.get("running") is True,
                }
            )
    return source_events


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("masks", "boxes"), required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument(
        "--runtime-lane", choices=("wholebody49-s", "wholebody49-x"), required=True
    )
    parser.add_argument("--runtime-instance-id", required=True)
    parser.add_argument("--runtime-run-id", required=True)
    parser.add_argument("--ws", default="ws://127.0.0.1:6008")
    parser.add_argument("--duration", type=float, default=45.0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--source-out", type=Path, required=True)
    parser.add_argument(
        "--expected-source-id",
        action="append",
        required=True,
        help="Canonical non-negative source ID; repeat for the complete lane inventory.",
    )
    add_auth_token_file_argument(parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_paths = (args.source_out,) if args.out is None else (args.source_out, args.out)
    try:
        require_fresh_private_file_bundle(
            output_paths,
            label="Wholebody49 behavior evidence bundle",
        )
    except (PrivatePathError, ValueError) as exc:
        print(f"[FAIL] Wholebody49 evidence session is not fresh: {exc}")
        return 1
    source_published = False
    try:
        auth = load_required_internal_auth(args.auth_token_file)
        source_events = asyncio.run(
            collect_evidence(args.ws, auth, duration_s=float(args.duration))
        )
        expected_source_ids = _expected_source_ids(args.expected_source_id)
        source_document = _source_transcript_document(
            session_id=args.session_id,
            runtime_lane=args.runtime_lane,
            runtime_instance_id=args.runtime_instance_id,
            runtime_run_id=args.runtime_run_id,
            mode=args.mode,
            expected_source_ids=expected_source_ids,
            events=source_events,
        )
        source_encoded = _encoded_private_json(source_document)
        if len(source_encoded) > MAX_SOURCE_TRANSCRIPT_BYTES:
            raise ValueError("Wholebody49 source transcript byte bound exceeded")
        _write_private_json(
            args.source_out,
            source_document,
            max_bytes=MAX_SOURCE_TRANSCRIPT_BYTES,
        )
        source_published = True
        result = analyze_evidence(
            session_id=args.session_id,
            runtime_lane=args.runtime_lane,
            runtime_instance_id=args.runtime_instance_id,
            runtime_run_id=args.runtime_run_id,
            mode=args.mode,
            expected_source_ids=expected_source_ids,
            events=source_events,
            source_evidence=_source_evidence_metadata(
                encoded=source_encoded,
                document=source_document,
            ),
        )
    except Exception as exc:
        result = {
            "schema_version": 1,
            "contract": CONTRACT,
            "contract_version": CONTRACT_VERSION,
            "session_id": str(args.session_id).strip().lower(),
            "runtime_lane": str(args.runtime_lane).strip().lower(),
            "runtime_instance_id": str(args.runtime_instance_id),
            "runtime_run_id": str(args.runtime_run_id),
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    if not source_published:
        print(
            "[FAIL] Wholebody49 source evidence was not published; use a fresh session"
        )
        return 1
    if args.out is not None:
        _write_private_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if bool(result.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
