#!/usr/bin/env python3
"""Render recorded DS9 cohorts beside their source video and captured PCF BEV.

This is a deterministic, offline review utility. It does not run inference,
modify recorded world positions, or treat the PCF raster as occupied-person
ground truth. Tracking/BEV joins are exact cohort joins; unmatched records are
reported instead of being paired by last-seen state.
"""
from __future__ import annotations

import argparse
import base64
import configparser
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Iterator, Mapping

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CAMERA_ORDER = ("living-room", "kitchen", "family-room")
DEFAULT_DEWARPER = {
    "living-room": ROOT / "DS9/config/dewarper_g3_instant_charuco_1080.txt",
    "kitchen": ROOT / "DS9/config/dewarper_g3_instant_charuco_1080.txt",
    "family-room": ROOT / "DS9/config/dewarper_g4_instant_charuco_720_to_1080.txt",
}
MAX_CAPTURE_ROWS = 100_000
MAX_NDJSON_BYTES = 1024 * 1024 * 1024
MAX_NDJSON_LINE_BYTES = 32 * 1024 * 1024
MAX_GRID_CELLS = 16 * 1024 * 1024
MAX_TRACKS_PER_COHORT = 512
MAX_TRACK_RECORDS_PER_FILE = 1_000_000
MAX_BEV_ITEMS_PER_COHORT = 1_024
MAX_TRAILS_PER_COHORT = 512
MAX_TRAIL_POINTS_PER_COHORT = 16_384
CANVAS_SIZE = (1920, 1000)
VIDEO_RECT = (24, 100, 920, 518)
BEV_RECT = (982, 104, 910, 820)
PANEL_BG = (24, 27, 32)
TEXT_COLOR = (231, 235, 240)


class ReviewBoundExceeded(RuntimeError):
    """An input exceeded a hard offline review work or allocation bound."""


def _get(mapping: Mapping[str, Any], snake: str, camel: str | None = None) -> Any:
    if snake in mapping:
        return mapping[snake]
    if camel and camel in mapping:
        return mapping[camel]
    return None


def _unwrap_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Accept raw WS rows and the retained companion envelope without joining loosely."""

    if isinstance(row.get("type"), str):
        return dict(row)
    message = row.get("message")
    if isinstance(message, Mapping) and isinstance(message.get("type"), str):
        return dict(message)
    return None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return converted


def _finite(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return converted if math.isfinite(converted) else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cohort_key(message: Mapping[str, Any]) -> tuple[Any, ...] | None:
    """Return the exact shared tracking/BEV cohort identity (no temporal fallback)."""

    camera = _get(message, "camera_id", "cameraId")
    source = _get(message, "source_id", "sourceId")
    frame = _get(message, "frame_id", "frameId")
    observed = _get(message, "observed_at_us", "observedAtUs")
    sequence = _get(message, "tracking_publication_sequence", "trackingPublicationSequence")
    cohort = message.get("cohort")
    if not isinstance(cohort, Mapping):
        cohort = {}
    camera = camera if isinstance(camera, str) and camera else None
    source = _integer(source if source is not None else cohort.get("source_id"))
    frame = _integer(frame if frame is not None else cohort.get("frame_id"))
    observed = _integer(observed if observed is not None else cohort.get("observed_at_us"))
    sequence = _integer(
        sequence
        if sequence is not None
        else cohort.get("tracking_publication_sequence")
    )
    epoch = _integer(
        _get(message, "source_epoch", "sourceEpoch")
        if _get(message, "source_epoch", "sourceEpoch") is not None
        else cohort.get("source_epoch")
    )
    # The replay capture starts one media epoch when the message schema omits it.
    if epoch is None:
        epoch = 0
    if camera is None or None in (source, frame, observed, sequence):
        return None
    return camera, source, epoch, frame, observed, sequence


def _cohort_fields_agree(message: Mapping[str, Any]) -> bool:
    cohort = message.get("cohort")
    if not isinstance(cohort, Mapping):
        return True
    for snake, camel in (
        ("source_id", "sourceId"), ("source_epoch", "sourceEpoch"),
        ("frame_id", "frameId"), ("observed_at_us", "observedAtUs"),
        ("tracking_publication_sequence", "trackingPublicationSequence"),
    ):
        nested = cohort.get(snake)
        top = _get(message, snake, camel)
        if nested is not None and top is not None and _integer(nested) != _integer(top):
            return False
    submission = cohort.get("tracking_outbound_submission_id")
    top_submission = _get(message, "tracking_outbound_submission_id", "trackingOutboundSubmissionId")
    return not (
        submission is not None
        and top_submission is not None
        and _integer(submission) != _integer(top_submission)
    )


def _track_key(track: Mapping[str, Any]) -> tuple[int, int] | None:
    tracker = _integer(_get(track, "tracker_id", "trackerId"))
    generation = _integer(
        _get(track, "tracker_lifecycle_generation", "trackerLifecycleGeneration")
    )
    if tracker is None:
        return None
    return tracker, generation if generation is not None else -1


def _world(track: Mapping[str, Any]) -> tuple[float, float, float] | None:
    if track.get("world_valid") is not True:
        return None
    value = track.get("world")
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    xyz = tuple(_finite(value[index]) for index in range(3))
    if any(component is None for component in xyz):
        return None
    return float(xyz[0]), float(xyz[1]), float(xyz[2])


def _measurement_state(track: Mapping[str, Any], point: Mapping[str, Any] | None) -> str:
    source = str(track.get("world_source") or (point or {}).get("anchorSource") or "").lower()
    admission = str((point or {}).get("worldAdmission") or "").lower()
    quality = str(track.get("world_quality") or "").lower()
    # Source semantics are the strongest distinction for diagnostic state:
    # projective/CV continuation is prediction even when quality is labelled
    # held. Keep world_quality and BEV worldAdmission separately in metrics.
    if source in {"cv_prediction", "image_motion_prediction"}:
        return "predicted"
    if admission == "held" or source == "anchor_hold" or quality == "held":
        return "held"
    if admission == "predicted":
        return "predicted"
    if track.get("world_measurement_accepted") is True:
        return "measured"
    if admission == "accepted":
        return "accepted-unclassified"
    return "unknown"


def _bounded_ndjson_lines(path: Path) -> Iterator[tuple[int, str]]:
    total_bytes = 0
    with path.open("rb") as handle:
        line_number = 0
        while True:
            raw_line = handle.readline(MAX_NDJSON_LINE_BYTES + 1)
            if not raw_line:
                break
            line_number += 1
            total_bytes += len(raw_line)
            if len(raw_line) > MAX_NDJSON_LINE_BYTES:
                raise ValueError(
                    f"NDJSON line exceeds {MAX_NDJSON_LINE_BYTES} byte bound: {path}:{line_number}"
                )
            if total_bytes > MAX_NDJSON_BYTES:
                raise ValueError(f"NDJSON input exceeds {MAX_NDJSON_BYTES} byte bound: {path}")
            if line_number > MAX_CAPTURE_ROWS:
                raise ValueError(f"NDJSON input exceeds {MAX_CAPTURE_ROWS} row bound: {path}")
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"invalid UTF-8 at {path}:{line_number}: {exc}") from exc
            yield line_number, line


def _bounded_json_file(path: Path) -> Mapping[str, Any]:
    with path.open("rb") as handle:
        raw = handle.read(MAX_NDJSON_LINE_BYTES + 1)
    if len(raw) > MAX_NDJSON_LINE_BYTES:
        raise ValueError(f"JSON input exceeds {MAX_NDJSON_LINE_BYTES} byte bound: {path}")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON input must be an object: {path}")
    return value


def _expected_base64_chars(decoded_bytes: int) -> int:
    return 4 * ((decoded_bytes + 2) // 3)


def _decode_bounded_base64(encoded: Any, decoded_bytes: int, label: str) -> bytes | None:
    if not isinstance(encoded, str):
        return None
    expected_chars = _expected_base64_chars(decoded_bytes)
    if len(encoded) > expected_chars:
        raise ReviewBoundExceeded(f"{label} exceeds its expected encoded-size bound")
    if len(encoded) != expected_chars:
        return None
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (TypeError, ValueError):
        return None
    return decoded if len(decoded) == decoded_bytes else None


def _load_ndjson(path: Path) -> tuple[list[dict[str, Any]], Counter[str]]:
    messages: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    track_records = 0
    for line_number, line in _bounded_ndjson_lines(path):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(raw, Mapping):
            counts["non_object_rows"] += 1
            continue
        message = _unwrap_row(raw)
        if message is None:
            counts["unrecognized_rows"] += 1
            continue
        kind = str(message.get("type"))
        counts[kind] += 1
        if kind == "tracking":
            tracks = message.get("tracks")
            count = len(tracks) if isinstance(tracks, list) else 0
            if count > MAX_TRACKS_PER_COHORT:
                raise ValueError(
                    f"tracking cohort exceeds {MAX_TRACKS_PER_COHORT} track bound at {path}:{line_number}"
                )
            track_records += count
            if track_records > MAX_TRACK_RECORDS_PER_FILE:
                raise ValueError(
                    f"NDJSON input exceeds {MAX_TRACK_RECORDS_PER_FILE} track-record bound: {path}"
                )
        elif kind == "bev-frame":
            points = message.get("footpoints")
            dropped = message.get("droppedFootpoints")
            trails = message.get("trails")
            point_count = len(points) if isinstance(points, list) else 0
            dropped_count = len(dropped) if isinstance(dropped, list) else 0
            trail_count = len(trails) if isinstance(trails, list) else 0
            if point_count + dropped_count > MAX_BEV_ITEMS_PER_COHORT:
                raise ValueError(
                    f"BEV cohort exceeds {MAX_BEV_ITEMS_PER_COHORT} footpoint/drop bound at {path}:{line_number}"
                )
            if trail_count > MAX_TRAILS_PER_COHORT:
                raise ValueError(
                    f"BEV cohort exceeds {MAX_TRAILS_PER_COHORT} trail bound at {path}:{line_number}"
                )
            trail_points = sum(
                len(trail.get("points", []))
                for trail in trails if isinstance(trail, Mapping)
                and isinstance(trail.get("points"), list)
            ) if isinstance(trails, list) else 0
            if trail_points > MAX_TRAIL_POINTS_PER_COHORT:
                raise ValueError(
                    f"BEV cohort exceeds {MAX_TRAIL_POINTS_PER_COHORT} trail-point bound at {path}:{line_number}"
                )
        if kind in {"tracking", "bev-frame", "floorplan_response", "floorplan-response"}:
            messages.append(message)
    return messages, counts


def _load_kv(items: Iterable[str], label: str, *, path_value: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        camera, sep, raw_value = item.partition("=")
        camera = camera.strip()
        raw_value = raw_value.strip()
        if not sep or camera not in CAMERA_ORDER or not raw_value:
            raise ValueError(f"{label} must use CAMERA=value for a canonical camera")
        if camera in result:
            raise ValueError(f"duplicate {label} for {camera}")
        result[camera] = Path(raw_value).expanduser().resolve() if path_value else float(raw_value)
    return result


def _pipeline_paths(capture: Path, explicit: Path | None) -> tuple[dict[str, Path], dict[str, Path]]:
    pipeline_path = explicit or capture.parent / "infer-replay.yaml"
    if not pipeline_path.is_file():
        pipeline_path = ROOT / "DS9/config/infer.yaml"
    payload = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("sources"), list):
        return {}, {}
    videos: dict[str, Path] = {}
    dewarpers: dict[str, Path] = {}
    for camera, source in zip(CAMERA_ORDER, payload["sources"]):
        if not isinstance(source, Mapping):
            continue
        uri = source.get("uri")
        if isinstance(uri, str) and uri.startswith("file://"):
            from urllib.parse import unquote, urlparse

            parsed = urlparse(uri)
            videos[camera] = Path(unquote(parsed.path)).resolve()
        dewarper = source.get("dewarper")
        if isinstance(dewarper, Mapping) and isinstance(dewarper.get("config-file"), str):
            dewarper_path = Path(dewarper["config-file"])
            if not dewarper_path.is_absolute():
                dewarper_path = pipeline_path.parent / dewarper_path
                if not dewarper_path.exists():
                    dewarper_path = ROOT / dewarper["config-file"]
            dewarpers[camera] = dewarper_path.resolve()
    return videos, dewarpers


def _read_dewarper(path: Path, raw_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    if not parser.read(path):
        raise ValueError(f"cannot read dewarper config: {path}")
    surface = parser["surface0"]
    destination = parser["property"]
    width = int(destination.get("output-width", surface["width"]))
    height = int(destination.get("output-height", surface["height"]))
    # These installed configs are authored for native sensor sizes. Scale only
    # source intrinsics if the retained video was resized; destination K stays
    # in the configured rectified frame.
    source_width, source_height = raw_shape
    nominal_width, nominal_height = (1280, 720) if "720_to_1080" in path.name else (1920, 1080)
    sx, sy = source_width / nominal_width, source_height / nominal_height
    fx, fy = (float(v) for v in surface["focal-length"].split(";"))
    cx, cy = float(surface["src-x0"]), float(surface["src-y0"])
    source_k = np.asarray([[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy], [0.0, 0.0, 1.0]])
    distortion = np.asarray([float(v) for v in surface["distortion"].split(";")], dtype=np.float64)
    dx, dy = (float(v) for v in surface["dst-focal-length"].split(";"))
    px, py = (float(v) for v in surface["dst-principal-point"].split(";") )
    destination_k = np.asarray([[dx, 0.0, px], [0.0, dy, py], [0.0, 0.0, 1.0]])
    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
        source_k, distortion, np.eye(3), destination_k, (width, height), cv2.CV_32FC1
    )
    return map_x, map_y, (width, height)


def _decode_mask(encoded: Any, shape: tuple[int, int]) -> np.ndarray | None:
    if not isinstance(encoded, str) or not encoded.startswith("bit:"):
        return None
    try:
        _, count_text, data = encoded.split(":", 2)
        count = int(count_text)
        cell_count = shape[0] * shape[1]
        if count != cell_count or cell_count > MAX_GRID_CELLS:
            return None
        byte_count = (cell_count + 7) // 8
        raw = _decode_bounded_base64(data, byte_count, "encoded mask")
        if raw is None:
            return None
        return np.unpackbits(np.frombuffer(raw, dtype=np.uint8), count=cell_count, bitorder="big").reshape(shape).astype(bool)
    except (TypeError, ValueError):
        return None


def _pcf_rgb(floorplan: Mapping[str, Any] | None) -> tuple[np.ndarray | None, dict[str, float] | None, str | None]:
    if floorplan is None:
        return None, None, None
    layer = floorplan.get("scene_prior_diagnostic_surface_rgb")
    bounds = floorplan.get("bounds")
    if not isinstance(layer, Mapping) or not isinstance(bounds, Mapping):
        return None, None, None
    shape_raw = layer.get("rgb_shape")
    encoded = layer.get("rgb_b64")
    if not isinstance(shape_raw, (list, tuple)) or len(shape_raw) != 3 or not isinstance(encoded, str):
        return None, None, None
    try:
        height, width, channels = (int(value) for value in shape_raw)
        count = height * width
        if channels != 3 or height <= 0 or width <= 0 or count > MAX_GRID_CELLS:
            return None, None, None
        raw = _decode_bounded_base64(encoded, count * 3, "PCF RGB payload")
        if raw is None:
            return None, None, None
        rgb = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()
        observed = _decode_mask(layer.get("observed_b64"), (height, width))
        if observed is not None:
            rgb[~observed] = (15, 17, 20)
        normalized_bounds = {
            "min_x": _finite(bounds.get("min_x")), "max_x": _finite(bounds.get("max_x")),
            "min_z": _finite(bounds.get("min_z")), "max_z": _finite(bounds.get("max_z")),
        }
        if any(value is None for value in normalized_bounds.values()):
            return None, None, None
        snapshot = floorplan.get("scene_prior_meta")
        prior_id = snapshot.get("prior_id") if isinstance(snapshot, Mapping) else None
        return rgb, {key: float(value) for key, value in normalized_bounds.items()}, str(prior_id) if prior_id else None
    except (TypeError, ValueError):
        return None, None, None


def _color(identity: Any) -> tuple[int, int, int]:
    try:
        value = int(identity or 0)
    except (TypeError, ValueError):
        value = 0
    hue = (value * 0.6180339887498949) % 1.0
    rgb = cv2.cvtColor(np.uint8([[[int(hue * 179), 210, 245]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def _exact_join(
    tracking: list[dict[str, Any]], bev_rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]]
) -> tuple[dict[tuple[Any, ...], dict[str, Any]], dict[tuple[Any, ...], dict[str, Any]], Counter[str]]:
    counts: Counter[str] = Counter()
    track_map: dict[tuple[Any, ...], dict[str, Any]] = {}
    bev_map: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in tracking:
        if not _cohort_fields_agree(row):
            counts["tracking_contradictory_cohort_fields"] += 1
            continue
        key = _cohort_key(row)
        if key is None:
            counts["tracking_missing_exact_cohort_key"] += 1
            continue
        if key in track_map:
            counts["duplicate_tracking_cohort"] += 1
            continue
        track_map[key] = row
    for row in bev_rows:
        if not _cohort_fields_agree(row):
            counts["bev_contradictory_cohort_fields"] += 1
            continue
        key = _cohort_key(row)
        if key is None:
            counts["bev_missing_exact_cohort_key"] += 1
            continue
        if key in bev_map:
            counts["duplicate_bev_cohort"] += 1
            continue
        bev_map[key] = row
    for row in reference_rows:
        if not _cohort_fields_agree(row):
            counts["reference_bev_contradictory_cohort_fields"] += 1
            continue
        key = _cohort_key(row)
        if key is None or key in bev_map:
            if key is None:
                counts["reference_bev_missing_exact_cohort_key"] += 1
            continue
        bev_map[key] = row
        counts["reference_bev_frames_used"] += 1
    counts["tracking_cohorts"] = len(track_map)
    counts["captured_bev_cohorts"] = len({key for row in bev_rows if (key := _cohort_key(row)) is not None and _cohort_fields_agree(row)})
    counts["exact_tracking_bev_cohort_joins"] = len(set(track_map) & set(bev_map))
    counts["tracking_without_bev_cohort"] = len(set(track_map) - set(bev_map))
    counts["bev_without_tracking_cohort"] = len(set(bev_map) - set(track_map))
    return track_map, bev_map, counts


def _build_metrics(track_map: Mapping[tuple[Any, ...], dict[str, Any]], bev_map: Mapping[tuple[Any, ...], dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    total = Counter()
    camera_counts: dict[str, Counter[str]] = defaultdict(Counter)
    state_counts: Counter[str] = Counter()
    depth_status: Counter[str] = Counter()
    registration_status: Counter[str] = Counter()
    depth_anchor_source: Counter[str] = Counter()
    selected_candidates: Counter[str] = Counter()
    candidate_kinds: Counter[str] = Counter()
    world_sources: Counter[str] = Counter()
    depth_values: dict[str, list[float]] = defaultdict(list)
    previous_source: dict[tuple[str, int, int, int, int], str] = {}
    source_switches = 0
    details: list[dict[str, Any]] = []

    for key, message in sorted(track_map.items(), key=lambda item: (item[0][0], item[0][4], item[0][5])):
        camera, source, epoch, frame_id, observed_us, sequence = key
        bev = bev_map.get(key)
        tracks = message.get("tracks") if isinstance(message.get("tracks"), list) else []
        points = bev.get("footpoints") if isinstance(bev, Mapping) and isinstance(bev.get("footpoints"), list) else []
        drops = bev.get("droppedFootpoints") if isinstance(bev, Mapping) and isinstance(bev.get("droppedFootpoints"), list) else []
        point_by_key = {_track_key(point): point for point in points if isinstance(point, Mapping) and _track_key(point) is not None}
        dropped_by_key = {_track_key(point): point for point in drops if isinstance(point, Mapping) and _track_key(point) is not None}
        row_counts = Counter()
        row_depth, row_candidates = Counter(), Counter()
        max_candidates_this_cohort = 0
        for track in tracks:
            if not isinstance(track, Mapping):
                continue
            total["person_observations"] += 1
            row_counts["person_observations"] += 1
            ident = _track_key(track)
            world = _world(track)
            if world is not None:
                total["valid_world_observations"] += 1
                row_counts["valid_world_observations"] += 1
            point = point_by_key.get(ident)
            if point is not None:
                total["valid_world_to_bev_heads"] += int(world is not None)
                row_counts["emitted_heads"] += 1
            elif world is not None and bev is None:
                total["valid_world_positions_in_unmatched_tracking_cohort"] += 1
                row_counts["valid_world_in_unmatched_cohort"] += 1
            elif world is not None:
                total["valid_world_positions_without_bev_head"] += 1
                row_counts["valid_world_without_head"] += 1
                reason = dropped_by_key.get(ident, {}).get("reason") if ident is not None else None
                row_counts[f"missing_head:{reason or 'point_not_emitted'}"] += 1
            state = _measurement_state(track, point)
            state_counts[state] += 1
            row_counts[f"state:{state}"] += 1
            source_label = str(track.get("world_source") or "unknown")
            world_sources[source_label] += 1
            row_counts[f"world_source:{source_label}"] += 1
            lifecycle = ident or (-1, -1)
            source_identity = (camera, source, epoch, lifecycle[0], lifecycle[1])
            if source_identity in previous_source and previous_source[source_identity] != source_label:
                source_switches += 1
            previous_source[source_identity] = source_label
            row_depth[str(track.get("depth_status") or "missing")] += 1
            depth_status[str(track.get("depth_status") or "missing")] += 1
            registration_status[str(track.get("depth_registration_status") or "missing")] += 1
            depth_anchor_source[str(track.get("depth_anchor_source") or "missing")] += 1
            for field in ("depth_anchor_m", "depth_used_m", "depth_registered_m"):
                value = _finite(track.get(field))
                if value is not None:
                    depth_values[field].append(value)
            selected = track.get("world_resolver_selected_id") or track.get("world_resolver_selected_kind")
            if selected:
                selected_candidates[str(selected)] += 1
                row_candidates[str(selected)] += 1
            diagnostic = point.get("resolverDiagnostics") if isinstance(point, Mapping) else None
            if isinstance(diagnostic, Mapping):
                selected = diagnostic.get("selectedKind")
                if selected:
                    selected_candidates[str(selected)] += 1
                    row_candidates[str(selected)] += 1
                for candidate in diagnostic.get("candidates", []):
                    if isinstance(candidate, Mapping):
                        max_candidates_this_cohort += 1
                        kind = candidate.get("kind") or candidate.get("id") or "unknown"
                        candidate_kinds[str(kind)] += 1
                        row_candidates[str(kind)] += 1
        camera_counts[camera].update(row_counts)
        detail = {
            "camera_id": camera, "source_id": source, "source_epoch": epoch,
            "frame_id": frame_id, "observed_at_us": observed_us,
            "tracking_publication_sequence": sequence,
            "bev_present": bev is not None,
            "bev_origin": (
                "offline_reference"
                if isinstance(bev, Mapping) and bev.get("_review_origin") == "offline_reference"
                else ("captured" if bev is not None else None)
            ),
            "person_observations": len(tracks),
            "emitted_bev_heads": len(points),
            "dropped_bev_footpoints": len(drops),
            "valid_world_to_bev_missing": row_counts["valid_world_without_head"],
            "valid_world_in_unmatched_cohort": row_counts["valid_world_in_unmatched_cohort"],
            "states": {name.removeprefix("state:"): value for name, value in row_counts.items() if name.startswith("state:")},
            "missing_head_reasons": {name.removeprefix("missing_head:"): value for name, value in row_counts.items() if name.startswith("missing_head:")},
            "depth_status": dict(row_depth), "resolver_candidates": dict(row_candidates),
            "resolver_candidate_count": max_candidates_this_cohort,
        }
        details.append(detail)

    step_stats = _measurement_steps(track_map)
    summary = {
        "cohort_count": len(track_map),
        "occupied_cohort_count": sum(1 for row in track_map.values() if row.get("tracks")),
        "person_observations": total["person_observations"],
        "valid_world_observations": total["valid_world_observations"],
        "valid_world_to_bev_heads": total["valid_world_to_bev_heads"],
        "valid_world_positions_without_bev_head": total["valid_world_positions_without_bev_head"],
        "valid_world_positions_in_unmatched_tracking_cohort": total["valid_world_positions_in_unmatched_tracking_cohort"],
        "source_switches": source_switches,
        "measurement_state_counts": dict(state_counts),
        "depth_status_counts": dict(depth_status),
        "depth_registration_status_counts": dict(registration_status),
        "depth_anchor_source_counts": dict(depth_anchor_source),
        "world_source_counts": dict(world_sources),
        "depth_range_summary_m": {
            field: {
                "count": len(values),
                "min": min(values) if values else None,
                "median": float(np.median(values)) if values else None,
                "max": max(values) if values else None,
            }
            for field, values in depth_values.items()
        },
        "resolver_selected_candidate_counts": dict(selected_candidates),
        "resolver_candidate_kind_counts": dict(candidate_kinds),
        "measured_step_metrics": step_stats,
        "per_camera": {camera: dict(values) for camera, values in camera_counts.items()},
        "valid_world_missing_head_drop_analysis": _valid_world_missing_head_drop_analysis(
            track_map, bev_map
        ),
    }
    return summary, details


def _valid_world_missing_head_drop_analysis(
    track_map: Mapping[tuple[Any, ...], dict[str, Any]],
    bev_map: Mapping[tuple[Any, ...], dict[str, Any]],
) -> dict[str, Any]:
    """Classify valid-world heads not emitted by captured BEV, on exact joins only."""

    per_camera: dict[str, dict[str, Any]] = {}
    for key, tracking in track_map.items():
        bev = bev_map.get(key)
        if not isinstance(bev, Mapping):
            continue
        camera = str(key[0])
        value = per_camera.setdefault(camera, {
            "valid_world_missing_heads": 0,
            "drop_reason_counts": Counter(),
            "world_source_counts": Counter(),
            "world_measurement_accepted_counts": Counter(),
            "outside_display_bounds_axes": Counter(),
            "outside_display_bounds_distance_m": [],
            "display_bounds_xz": None,
            "examples": [],
        })
        tracks = tracking.get("tracks") if isinstance(tracking.get("tracks"), list) else []
        points = bev.get("footpoints") if isinstance(bev.get("footpoints"), list) else []
        drops = bev.get("droppedFootpoints") if isinstance(bev.get("droppedFootpoints"), list) else []
        point_by_key = {
            _track_key(point): point
            for point in points
            if isinstance(point, Mapping) and _track_key(point) is not None
        }
        dropped_by_key = {
            _track_key(drop): drop
            for drop in drops
            if isinstance(drop, Mapping) and _track_key(drop) is not None
        }
        display_bounds = bev.get("displayBounds")
        if isinstance(display_bounds, Mapping):
            bounds = {
                "min_x": _finite(display_bounds.get("min_x")),
                "max_x": _finite(display_bounds.get("max_x")),
                "min_z": _finite(display_bounds.get("min_z")),
                "max_z": _finite(display_bounds.get("max_z")),
            }
            if all(bound is not None for bound in bounds.values()):
                value["display_bounds_xz"] = bounds
            else:
                bounds = None
        else:
            bounds = None
        for track in tracks:
            if not isinstance(track, Mapping):
                continue
            world = _world(track)
            ident = _track_key(track)
            if world is None or ident is None or ident in point_by_key:
                continue
            value["valid_world_missing_heads"] += 1
            dropped = dropped_by_key.get(ident, {})
            reason = str(dropped.get("reason") or "point_not_emitted")
            value["drop_reason_counts"][reason] += 1
            value["world_source_counts"][str(track.get("world_source") or "unknown")] += 1
            accepted = track.get("world_measurement_accepted") is True
            value["world_measurement_accepted_counts"]["accepted" if accepted else "not_accepted"] += 1

            if reason == "canonical_world_outside_display_guard" and bounds is not None:
                x = _finite(dropped.get("floorplanX", dropped.get("x")))
                z = _finite(dropped.get("floorplanZ", dropped.get("y")))
                if x is not None and z is not None:
                    dx = max(float(bounds["min_x"]) - x, 0.0, x - float(bounds["max_x"]))
                    dz = max(float(bounds["min_z"]) - z, 0.0, z - float(bounds["max_z"]))
                    axes = "both" if dx > 0.0 and dz > 0.0 else ("x_only" if dx > 0.0 else "z_only" if dz > 0.0 else "inside")
                    value["outside_display_bounds_axes"][axes] += 1
                    value["outside_display_bounds_distance_m"].append(math.hypot(dx, dz))
                    if len(value["examples"]) < 8:
                        value["examples"].append({
                            "source_id": key[1],
                            "source_epoch": key[2],
                            "frame_id": key[3],
                            "observed_at_us": key[4],
                            "tracking_publication_sequence": key[5],
                            "media_pts_ns": _integer(track.get("media_pts_ns")),
                            "stable_id": track.get("stable_id"),
                            "tracker_id": track.get("tracker_id"),
                            "tracker_lifecycle_generation": track.get("tracker_lifecycle_generation"),
                            "world_xyz_m": list(world),
                            "camera_local_ground_xz_m": [x, z],
                            "display_bounds_xz_m": bounds,
                            "outside_bounds_distance_m": math.hypot(dx, dz),
                            "floorplan_inside": dropped.get("floorplanInside"),
                            "coverage_inside": dropped.get("coverageInside"),
                            "drop_reason": reason,
                        })

    for value in per_camera.values():
        distances = value["outside_display_bounds_distance_m"]
        value["drop_reason_counts"] = dict(value["drop_reason_counts"])
        value["world_source_counts"] = dict(value["world_source_counts"])
        value["world_measurement_accepted_counts"] = dict(value["world_measurement_accepted_counts"])
        value["outside_display_bounds_axes"] = dict(value["outside_display_bounds_axes"])
        value["outside_display_bounds_distance_m"] = {
            "count": len(distances),
            "median": float(np.median(distances)) if distances else None,
            "max": max(distances) if distances else None,
        }
    return {
        "scope": "valid track.world values without an emitted footpoint in exact tracking/BEV cohort joins; unmatched tracking cohorts excluded",
        "display_guard_coordinates": "camera_local_ground_m X/Z; finite canonical world XYZ is separately retained in examples",
        "per_camera": per_camera,
    }


def _measurement_steps(track_map: Mapping[tuple[Any, ...], dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int, int, int, int], list[tuple[int, tuple[float, float, float], str]]] = defaultdict(list)
    for cohort, message in track_map.items():
        camera, _, _, _, observed_us, _ = cohort
        for track in message.get("tracks", []) if isinstance(message.get("tracks"), list) else []:
            if not isinstance(track, Mapping) or track.get("world_measurement_accepted") is not True:
                continue
            world = _world(track)
            key = _track_key(track)
            if world is None or key is None:
                continue
            grouped[(camera, cohort[1], cohort[2], key[0], key[1])].append((observed_us, world, str(track.get("world_source") or "unknown")))
    distances: list[float] = []
    speeds: list[float] = []
    lifecycle_counts = 0
    lifecycle_max: list[dict[str, Any]] = []
    for (camera, source, epoch, tracker, generation), observations in grouped.items():
        observations.sort(key=lambda item: item[0])
        lifecycle_counts += 1
        life_distances, life_speeds = [], []
        for left, right in zip(observations, observations[1:]):
            delta_t = (right[0] - left[0]) / 1e6
            if delta_t <= 0:
                continue
            distance = math.dist(left[1], right[1])
            life_distances.append(distance)
            life_speeds.append(distance / delta_t)
        distances.extend(life_distances)
        speeds.extend(life_speeds)
        lifecycle_max.append({
            "camera_id": camera, "source_id": source, "source_epoch": epoch,
            "tracker_id": tracker, "generation": generation,
            "measured_observation_count": len(observations),
            "max_consecutive_measured_step_m": max(life_distances, default=None),
            "max_consecutive_measured_speed_m_s": max(life_speeds, default=None),
        })
    return {
        "metric": "3D Euclidean displacement between consecutive explicitly accepted world measurements; descriptive, not truth error",
        "lifecycle_count": lifecycle_counts,
        "step_count": len(distances),
        "median_step_m": float(np.median(distances)) if distances else None,
        "p95_step_m": float(np.percentile(distances, 95)) if distances else None,
        "max_step_m": max(distances, default=None),
        "max_speed_m_s": max(speeds, default=None),
        "largest_lifecycles": sorted(lifecycle_max, key=lambda row: row["max_consecutive_measured_step_m"] or -1, reverse=True)[:24],
    }


def _coordinate_parity(
    track_map: Mapping[tuple[Any, ...], dict[str, Any]],
    bev_map: Mapping[tuple[Any, ...], dict[str, Any]],
    floorplans: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    per_camera: dict[str, list[float]] = defaultdict(list)
    for key, tracking in track_map.items():
        bev = bev_map.get(key)
        floorplan = floorplans.get(key[0])
        if not isinstance(bev, Mapping) or bev.get("frame_mode") != "camera_local":
            continue
        diagnostic_meta = floorplan.get("scene_prior_diagnostic_meta") if isinstance(floorplan, Mapping) else None
        if not isinstance(diagnostic_meta, Mapping):
            continue
        camera = diagnostic_meta.get("camera_position_world_m")
        right = diagnostic_meta.get("camera_right_world_xz")
        forward = diagnostic_meta.get("camera_forward_world_xz")
        if (
            not isinstance(camera, (list, tuple)) or len(camera) != 3
            or not isinstance(right, (list, tuple)) or len(right) != 2
            or not isinstance(forward, (list, tuple)) or len(forward) != 2
        ):
            continue
        camera_x, camera_z = _finite(camera[0]), _finite(camera[2])
        right_x, right_z = _finite(right[0]), _finite(right[1])
        forward_x, forward_z = _finite(forward[0]), _finite(forward[1])
        if None in (camera_x, camera_z, right_x, right_z, forward_x, forward_z):
            continue
        point_by_key = {
            _track_key(point): point
            for point in bev.get("footpoints", [])
            if isinstance(point, Mapping) and _track_key(point) is not None
        }
        for track in tracking.get("tracks", []) if isinstance(tracking.get("tracks"), list) else []:
            if not isinstance(track, Mapping):
                continue
            xyz = _world(track)
            point = point_by_key.get(_track_key(track))
            if xyz is None or point is None:
                continue
            dx, dz = xyz[0] - camera_x, xyz[2] - camera_z
            expected_x = float(right_x) * dx + float(right_z) * dz
            expected_z = float(forward_x) * dx + float(forward_z) * dz
            actual_x, actual_z = _finite(point.get("x")), _finite(point.get("y"))
            if actual_x is None or actual_z is None:
                continue
            per_camera[key[0]].append(math.hypot(actual_x - expected_x, actual_z - expected_z))
    return {
        "metric": "canonical world X/Z projected into the captured PCF camera_local_ground_m basis; transform parity only, not accuracy",
        "tolerance_m": 1e-6,
        "per_camera": {
            camera: {
                "matched_valid_world_heads": len(errors),
                "max_xy_discrepancy_m": max(errors) if errors else None,
                "p95_xy_discrepancy_m": float(np.percentile(errors, 95)) if errors else None,
                "within_tolerance_count": sum(value <= 1e-6 for value in errors),
            }
            for camera, errors in per_camera.items()
        },
    }


def _video_metadata(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open source video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if width <= 0 or height <= 0 or not math.isfinite(frames) or not math.isfinite(fps) or frames <= 0 or fps <= 0:
        raise RuntimeError(f"cannot determine source video metadata: {path}")
    return {"width": width, "height": height, "fps": float(fps), "frame_count": int(frames), "duration_s": float(frames / fps)}


def _dewarper_native_size(path: Path) -> tuple[int, int]:
    return (1280, 720) if "720_to_1080" in path.name else (1920, 1080)


def _read_reference_bev(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    rows, _ = _load_ndjson(path)
    return [row for row in rows if row.get("type") == "bev-frame"]


def _floorplans(messages: list[dict[str, Any]], paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for message in messages:
        if message.get("type") not in {"floorplan_response", "floorplan-response"}:
            continue
        camera = _get(message, "camera_id", "cameraId") or message.get("camera")
        if (
            isinstance(camera, str)
            and message.get("display_source") == "pcf"
            and message.get("error") is None
            and message.get("scene_prior_only") is True
        ):
            found[camera] = message
    for camera, path in paths.items():
        found[camera] = dict(_bounded_json_file(path))
    return found


def _bev_color(stable_id: Any, state: str) -> tuple[int, int, int]:
    if state == "held":
        return (160, 165, 170)
    if state == "predicted":
        return (70, 180, 245)
    return _color(stable_id)


def _pixel(point: Mapping[str, Any], bounds: Mapping[str, Any], rect: tuple[int, int, int, int]) -> tuple[int, int] | None:
    x, z = _finite(point.get("x")), _finite(point.get("y"))
    min_x, max_x = _finite(bounds.get("min_x")), _finite(bounds.get("max_x"))
    min_z, max_z = _finite(bounds.get("min_z")), _finite(bounds.get("max_z"))
    if None in (x, z, min_x, max_x, min_z, max_z) or max_x <= min_x or max_z <= min_z:
        return None
    left, top, width, height = rect
    return int(round(left + (x - min_x) / (max_x - min_x) * width)), int(round(top + (max_z - z) / (max_z - min_z) * height))


def _draw_frame(
    camera: str,
    tracking: Mapping[str, Any],
    bev: Mapping[str, Any] | None,
    floorplan: Mapping[str, Any] | None,
    video_frame: np.ndarray,
    output: Path,
    sample_seconds: float,
    rectification: str,
) -> dict[str, Any]:
    canvas = np.full((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), PANEL_BG, dtype=np.uint8)
    tracks = tracking.get("tracks") if isinstance(tracking.get("tracks"), list) else []
    points = bev.get("footpoints") if isinstance(bev, Mapping) and isinstance(bev.get("footpoints"), list) else []
    drops = bev.get("droppedFootpoints") if isinstance(bev, Mapping) and isinstance(bev.get("droppedFootpoints"), list) else []
    point_by_key = {_track_key(item): item for item in points if isinstance(item, Mapping) and _track_key(item) is not None}
    x0, y0, w0, h0 = VIDEO_RECT
    h, w = video_frame.shape[:2]
    display = cv2.resize(video_frame, (w0, h0), interpolation=cv2.INTER_AREA)
    scale_x, scale_y = w0 / w, h0 / h
    for track in tracks:
        if not isinstance(track, Mapping):
            continue
        color = _color(track.get("stable_id"))
        bbox = track.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            values = [_finite(value) for value in bbox]
            if all(value is not None for value in values):
                bx, by, bw, bh = values
                p0 = (int(bx * scale_x), int(by * scale_y))
                p1 = (int((bx + bw) * scale_x), int((by + bh) * scale_y))
                cv2.rectangle(display, p0, p1, color, 2)
                cv2.putText(display, f"SID {track.get('stable_id', '?')} / T{track.get('tracker_id', '?')}", (p0[0], max(17, p0[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
        anchor = track.get("image_foot") or track.get("image_base")
        if isinstance(anchor, (list, tuple)) and len(anchor) >= 2:
            ax, ay = _finite(anchor[0]), _finite(anchor[1])
            if ax is not None and ay is not None:
                cv2.drawMarker(display, (int(ax * scale_x), int(ay * scale_y)), (0, 240, 255), cv2.MARKER_CROSS, 14, 2)
    canvas[y0:y0 + h0, x0:x0 + w0] = display
    cv2.putText(canvas, f"{camera} | requested {sample_seconds:.2f}s | source video + exact cohort", (x0, 66), cv2.FONT_HERSHEY_SIMPLEX, .75, TEXT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"frame {tracking.get('frame_id', tracking.get('frameId'))} | seq {tracking.get('tracking_publication_sequence', tracking.get('trackingPublicationSequence'))} | tracks {len(tracks)} | {rectification}", (x0, 92), cv2.FONT_HERSHEY_SIMPLEX, .48, (175, 186, 197), 1, cv2.LINE_AA)

    bx, by, bw, bh = BEV_RECT
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (12, 14, 17), -1)
    display_bounds = bev.get("displayBounds") if isinstance(bev, Mapping) else None
    bev_rect = (bx, by, bw, bh)
    rgb, floor_bounds, prior_id = _pcf_rgb(floorplan)
    if rgb is not None and floor_bounds is not None and isinstance(display_bounds, Mapping) and bev.get("frame") == "camera_local_ground_m":
        fp_left = _pixel({"x": floor_bounds["min_x"], "y": floor_bounds["max_z"]}, display_bounds, bev_rect)
        fp_right = _pixel({"x": floor_bounds["max_x"], "y": floor_bounds["min_z"]}, display_bounds, bev_rect)
        if fp_left is not None and fp_right is not None:
            left, top = fp_left
            right, bottom = fp_right
            left, right = max(bx, left), min(bx + bw, right)
            top, bottom = max(by, top), min(by + bh, bottom)
            if right > left and bottom > top:
                patch = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (right - left, bottom - top), interpolation=cv2.INTER_NEAREST)
                canvas[top:bottom, left:right] = patch
    elif bev is None:
        cv2.putText(canvas, "No captured or reference BEV for this exact cohort", (bx + 48, by + bh // 2), cv2.FONT_HERSHEY_SIMPLEX, .65, (150, 158, 166), 1, cv2.LINE_AA)
    elif rgb is None:
        cv2.putText(canvas, "PCF raster unavailable in this capture", (bx + 70, by + bh // 2), cv2.FONT_HERSHEY_SIMPLEX, .62, (150, 158, 166), 1, cv2.LINE_AA)

    if isinstance(display_bounds, Mapping):
        # One-metre grid in the producer's declared display frame.
        min_x, max_x = _finite(display_bounds.get("min_x")), _finite(display_bounds.get("max_x"))
        min_z, max_z = _finite(display_bounds.get("min_z")), _finite(display_bounds.get("max_z"))
        if None not in (min_x, max_x, min_z, max_z) and max_x > min_x and max_z > min_z:
            for x in np.arange(math.ceil(min_x), math.floor(max_x) + 1, 1.0):
                pt0 = _pixel({"x": x, "y": min_z}, display_bounds, bev_rect)
                pt1 = _pixel({"x": x, "y": max_z}, display_bounds, bev_rect)
                if pt0 and pt1:
                    cv2.line(canvas, pt0, pt1, (57, 60, 64), 1)
            for z in np.arange(math.ceil(min_z), math.floor(max_z) + 1, 1.0):
                pt0 = _pixel({"x": min_x, "y": z}, display_bounds, bev_rect)
                pt1 = _pixel({"x": max_x, "y": z}, display_bounds, bev_rect)
                if pt0 and pt1:
                    cv2.line(canvas, pt0, pt1, (57, 60, 64), 1)
        for trail in bev.get("trails", []):
            if not isinstance(trail, Mapping):
                continue
            track_key = (trail.get("trackerId"), trail.get("trackerLifecycleGeneration"))
            source_track = next((item for item in tracks if _track_key(item) == track_key), {})
            color = _bev_color(source_track.get("stable_id"), "unknown") if isinstance(source_track, Mapping) else (210, 210, 210)
            coords: list[tuple[int, int]] = []
            for trail_point in trail.get("points", []):
                if not isinstance(trail_point, Mapping):
                    continue
                if trail_point.get("breakBefore"):
                    if len(coords) > 1:
                        cv2.polylines(canvas, [np.asarray(coords, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
                    coords = []
                    continue
                pixel = _pixel(trail_point, display_bounds, bev_rect)
                if pixel is not None:
                    coords.append(pixel)
            if len(coords) > 1:
                cv2.polylines(canvas, [np.asarray(coords, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
        for point in points:
            if not isinstance(point, Mapping):
                continue
            pixel = _pixel(point, display_bounds, bev_rect)
            if pixel is None:
                continue
            track = next((item for item in tracks if _track_key(item) == _track_key(point)), {})
            state = _measurement_state(track, point) if isinstance(track, Mapping) else str(point.get("worldAdmission") or "unknown")
            point_color = _bev_color(point.get("stableId"), state)
            cv2.circle(canvas, pixel, 7, point_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, pixel, 9, (8, 10, 12), 1, cv2.LINE_AA)
            cv2.putText(canvas, str(point.get("stableId") or "?"), (pixel[0] + 10, pixel[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, .48, point_color, 1, cv2.LINE_AA)
    origin = "captured producer BEV" if bev is not None and bev.get("_review_origin") != "offline_reference" else "offline reference BEV"
    cv2.putText(canvas, f"PCF + {origin} | {bev.get('frame') if isinstance(bev, Mapping) else 'no frame'}", (bx, 70), cv2.FONT_HERSHEY_SIMPLEX, .69, TEXT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"PCF prior {prior_id or 'unavailable'} | admission: accepted / predicted / held", (bx, 94), cv2.FONT_HERSHEY_SIMPLEX, .46, (175, 186, 197), 1, cv2.LINE_AA)

    # Compact state panel: depth is telemetry, not an independent range label.
    legend_y = 778
    cv2.putText(canvas, "Review diagnostics (not independent position truth)", (x0, 680), cv2.FONT_HERSHEY_SIMPLEX, .58, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Yellow cross = producer image foot/base anchor; colors identify StableID", (x0, 708), cv2.FONT_HERSHEY_SIMPLEX, .48, (177, 186, 195), 1, cv2.LINE_AA)
    cv2.putText(canvas, "PCF is static scene context; no person torso/foot labels are inferred from it", (x0, 734), cv2.FONT_HERSHEY_SIMPLEX, .48, (177, 186, 195), 1, cv2.LINE_AA)
    for track in tracks[:5]:
        if not isinstance(track, Mapping):
            continue
        point = point_by_key.get(_track_key(track))
        state = _measurement_state(track, point)
        world = _world(track)
        world_text = f"XYZ {world[0]:.2f}, {world[1]:.2f}, {world[2]:.2f} m" if world else "no valid world point"
        text = (f"SID {track.get('stable_id', '?')} / T{track.get('tracker_id', '?')}:{track.get('tracker_lifecycle_generation', '?')} | {state} | "
                f"{track.get('world_source') or 'no source'} | depth {track.get('depth_status') or '-'} / {track.get('depth_registration_status') or '-'} | {world_text}")
        cv2.putText(canvas, text[:136], (x0, legend_y), cv2.FONT_HERSHEY_SIMPLEX, .42, _color(track.get("stable_id")), 1, cv2.LINE_AA)
        legend_y += 27
    cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 93])
    return {
        "camera_id": camera,
        "requested_replay_second": sample_seconds,
        "tracking_frame_id": tracking.get("frame_id", tracking.get("frameId")),
        "tracking_observed_at_us": tracking.get("observed_at_us", tracking.get("observedAtUs")),
        "tracking_publication_sequence": tracking.get("tracking_publication_sequence", tracking.get("trackingPublicationSequence")),
        "bev_present": bev is not None,
        "bev_origin": "offline_reference" if isinstance(bev, Mapping) and bev.get("_review_origin") == "offline_reference" else ("captured" if bev is not None else None),
        "pcf_prior_id": prior_id,
        "person_count": len(tracks),
        "bev_head_count": len(points),
        "dropped_footpoint_count": len(drops),
    }


def _video_frame(path: Path, seek_seconds: float) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open source video: {path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, seek_seconds) * 1000.0)
    ok, frame = cap.read()
    decoded = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot decode {path} at {seek_seconds:.3f}s")
    return frame, decoded


def _video_time_seconds(message: Mapping[str, Any], origin_us: int) -> float:
    media_pts_ns = _finite(_get(message, "media_pts_ns", "mediaPtsNs"))
    if media_pts_ns is not None and media_pts_ns >= 0:
        # Captured recorded-replay PTS is per-source and relative to that
        # source file. It is the correct clock for selecting a decoded frame.
        return media_pts_ns / 1e9
    observed = _integer(_get(message, "observed_at_us", "observedAtUs"))
    if observed is None:
        return 0.0
    return (observed - origin_us) / 1e6


def _select_samples(
    track_map: Mapping[tuple[Any, ...], dict[str, Any]],
    requested: list[float],
    sample_count: int,
    video_durations: Mapping[str, float],
) -> dict[str, list[tuple[float, tuple[Any, ...], dict[str, Any]]]]:
    all_by_camera: dict[str, list[tuple[float, tuple[Any, ...], dict[str, Any]]]] = defaultdict(list)
    for key, message in track_map.items():
        all_by_camera[key[0]].append((key[4] / 1e6, key, message))
    by_camera: dict[str, list[tuple[float, tuple[Any, ...], dict[str, Any]]]] = defaultdict(list)
    for camera, rows in all_by_camera.items():
        rows.sort(key=lambda value: value[0])
        origin = rows[0][0]
        for stamp, key, message in rows:
            media_s = _video_time_seconds(message, int(origin * 1e6))
            duration = video_durations.get(camera)
            if message.get("tracks") and media_s >= 0 and (duration is None or media_s <= duration + 0.05):
                by_camera[camera].append((media_s, key, message))
    result: dict[str, list[tuple[float, tuple[Any, ...], dict[str, Any]]]] = {}
    for camera, rows in by_camera.items():
        rows.sort(key=lambda value: value[0])
        if not rows:
            result[camera] = []
            continue
        relative = rows
        seconds = requested or [relative[round(i * (len(relative) - 1) / max(1, sample_count - 1))][0] for i in range(min(sample_count, len(relative)))]
        used_keys: set[tuple[Any, ...]] = set()
        selected = []
        for target in seconds:
            actual, key, row = min(relative, key=lambda value: abs(value[0] - target))
            if key in used_keys:
                continue
            selected.append((target, key, row))
            used_keys.add(key)
        result[camera] = selected
    return result


def _reference_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    rows, _ = _load_ndjson(path)
    for row in rows:
        if row.get("type") == "bev-frame":
            row["_review_origin"] = "offline_reference"
    return [row for row in rows if row.get("type") == "bev-frame"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True, help="raw websocket cohorts.ndjson")
    parser.add_argument("--output", type=Path, required=True, help="new, non-existing review output directory")
    parser.add_argument("--video", action="append", default=[], help="CAMERA=/absolute/source-video; repeat per camera")
    parser.add_argument("--floorplan", action="append", default=[], help="CAMERA=/captured-floorplan-response.json; optional when included in capture")
    parser.add_argument("--reference-bev", type=Path, help="optional existing offline reconstructed_bev.ndjson used only when captured BEV is absent")
    parser.add_argument("--pipeline-config", type=Path, help="exact replay config; defaults to capture sibling infer-replay.yaml then canonical infer.yaml")
    parser.add_argument("--dewarper", action="append", default=[], help="CAMERA=/dewarper-config.txt; otherwise use recorded replay config or current canonical config")
    parser.add_argument("--video-offset", action="append", default=[], help="CAMERA=seconds offset from selected replay-relative second to source-video PTS")
    parser.add_argument("--already-rectified", action="store_true", help="source video is already in the canonical tracker pixel frame")
    parser.add_argument("--samples", default="", help="comma-separated replay-relative seconds; nearest actually occupied cohorts are selected independently per camera")
    parser.add_argument("--sample-count", type=int, default=6, help="auto-select this many occupied cohorts per camera when --samples is omitted")
    args = parser.parse_args(argv)
    capture = args.capture.expanduser().resolve(strict=True)
    out = args.output.expanduser().resolve(strict=False)
    if out.exists():
        parser.error(f"output directory already exists; choose a new path: {out}")
    if not 1 <= args.sample_count <= 24:
        parser.error("--sample-count must be 1..24")
    try:
        sample_seconds = [float(value.strip()) for value in args.samples.split(",") if value.strip()]
        if any(not math.isfinite(value) or value < 0 or value > 600 for value in sample_seconds):
            raise ValueError("sample seconds must be finite and between 0 and 600")
        videos = _load_kv(args.video, "--video")
        floorplan_paths = _load_kv(args.floorplan, "--floorplan")
        dewarper_paths = _load_kv(args.dewarper, "--dewarper")
        video_offsets = _load_kv(args.video_offset, "--video-offset", path_value=False)
        configured_videos, configured_dewarpers = _pipeline_paths(capture, args.pipeline_config)
        for camera, path in configured_videos.items():
            videos.setdefault(camera, path)
        for camera, path in configured_dewarpers.items():
            dewarper_paths.setdefault(camera, path)
        for camera, path in DEFAULT_DEWARPER.items():
            dewarper_paths.setdefault(camera, path)
        if not videos:
            raise ValueError("supply --video sources or keep infer-replay.yaml beside the capture")
        for camera, path in videos.items():
            if not path.is_file():
                raise ValueError(f"source video is missing for {camera}: {path}")
        video_metadata = {camera: _video_metadata(path) for camera, path in videos.items()}
        video_durations = {camera: metadata["duration_s"] for camera, metadata in video_metadata.items()}
        out.mkdir(parents=True, exist_ok=False)
        messages, input_counts = _load_ndjson(capture)
        tracking = [row for row in messages if row.get("type") == "tracking"]
        captured_bev = [row for row in messages if row.get("type") == "bev-frame"]
        reference_bev = _reference_rows(args.reference_bev)
        track_map, bev_map, join_counts = _exact_join(tracking, captured_bev, reference_bev)
        floorplans = _floorplans(messages, floorplan_paths)
        summary, details = _build_metrics(track_map, bev_map)
        summary.update(dict(join_counts))
        summary["captured_message_counts"] = dict(input_counts)
        summary["bounded_input_limits"] = {
            "ndjson_bytes_per_file_max": MAX_NDJSON_BYTES,
            "ndjson_line_bytes_max": MAX_NDJSON_LINE_BYTES,
            "ndjson_rows_per_file_max": MAX_CAPTURE_ROWS,
            "tracks_per_cohort_max": MAX_TRACKS_PER_COHORT,
            "track_records_per_file_max": MAX_TRACK_RECORDS_PER_FILE,
            "bev_footpoints_and_drops_per_cohort_max": MAX_BEV_ITEMS_PER_COHORT,
            "trails_per_cohort_max": MAX_TRAILS_PER_COHORT,
            "trail_points_per_cohort_max": MAX_TRAIL_POINTS_PER_COHORT,
            "grid_cells_max": MAX_GRID_CELLS,
        }
        summary["capture_path"] = str(capture)
        summary["capture_sha256"] = _sha256_file(capture)
        summary["reference_bev_path"] = str(args.reference_bev.resolve()) if args.reference_bev else None
        summary["independent_occupied_person_labels_available"] = False
        summary["truth_caveat"] = "PCF and Noesis world tracks are not independent occupied-person truth; no phone optical-center point is used as truth."
        summary["video_clock_caveat"] = "Captured media_pts_ns selects the corresponding per-source video PTS when present; no cross-camera synchronization or exposure-level truth is claimed."
        summary["review_mode"] = "offline; captured BEV preferred; optional exact-cohort offline reference BEV tagged separately"
        summary["camera_video_sources"] = {camera: str(path) for camera, path in videos.items()}
        summary["camera_video_metadata"] = video_metadata
        summary["dewarper_source_size_assessment"] = {
            camera: {
                "config_path": str(dewarper_paths[camera]) if camera in dewarper_paths else None,
                "config_native_source_size": list(_dewarper_native_size(dewarper_paths[camera])) if camera in dewarper_paths else None,
                "video_source_size": [metadata["width"], metadata["height"]],
                "dimensions_match_dewarper_source": (
                    [metadata["width"], metadata["height"]] == list(_dewarper_native_size(dewarper_paths[camera]))
                    if camera in dewarper_paths else None
                ),
                "overlay_warning": (
                    "source dimensions differ from the native dewarper calibration; offline pixel-scaled rectification is not runtime parity"
                    if camera in dewarper_paths and [metadata["width"], metadata["height"]] != list(_dewarper_native_size(dewarper_paths[camera]))
                    else None
                ),
            }
            for camera, metadata in video_metadata.items()
        }
        summary["floorplan_response_cameras"] = sorted(floorplans)
        summary["coordinate_transform_parity"] = _coordinate_parity(track_map, bev_map, floorplans)
        summary["generated_at_note"] = "no inference, runtime process, calibration fit, track correction, or database write"
        (out / "analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        with (out / "cohort_metrics.ndjson").open("x", encoding="utf-8") as handle:
            for detail in details:
                handle.write(json.dumps(detail, separators=(",", ":"), allow_nan=False) + "\n")
        selections = _select_samples(track_map, sample_seconds, args.sample_count, video_durations)
        sample_metadata: list[dict[str, Any]] = []
        for camera, selected in selections.items():
            video_path = videos.get(camera)
            if video_path is None:
                continue
            for requested, key, row in selected:
                first_observed_us = min(k[4] for k in track_map if k[0] == camera)
                actual_relative_s = _video_time_seconds(row, first_observed_us)
                seek_s = actual_relative_s + float(video_offsets.get(camera, 0.0))
                video_frame, decoded_pts_s = _video_frame(video_path, seek_s)
                rectification = "already-rectified" if args.already_rectified else "raw-source"
                dewarper = None if args.already_rectified else dewarper_paths.get(camera)
                native_size_match = None
                if dewarper is not None:
                    raw_h, raw_w = video_frame.shape[:2]
                    native_size_match = (raw_w, raw_h) == _dewarper_native_size(dewarper)
                    map_x, map_y, out_size = _read_dewarper(dewarper, (raw_w, raw_h))
                    video_frame = cv2.remap(video_frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    if video_frame.shape[1] != out_size[0] or video_frame.shape[0] != out_size[1]:
                        raise RuntimeError("dewarper produced unexpected output shape")
                    rectification = (
                        f"rectified:{dewarper.name}"
                        if native_size_match
                        else f"offline-scaled-K:{dewarper.name}; NOT runtime parity"
                    )
                bev = bev_map.get(key)
                floorplan = floorplans.get(camera)
                stem = f"{camera}-{key[3]}-{key[5]}"
                details_path = out / f"{stem}.jpg"
                sample = _draw_frame(camera, row, bev, floorplan, video_frame, details_path, requested, rectification)
                sample["video_path"] = str(video_path)
                sample["video_requested_pts_s"] = seek_s
                sample["video_decoded_next_pts_s"] = decoded_pts_s
                sample["video_media_pts_s"] = actual_relative_s
                sample["video_source_size"] = [video_metadata[camera]["width"], video_metadata[camera]["height"]]
                sample["native_dewarper_source_size_match"] = native_size_match
                sample["runtime_source_alignment_valid"] = native_size_match is not False
                sample["image_file"] = details_path.name
                sample_metadata.append(sample)
        report = {
            "samples": sample_metadata,
            "sample_count": len(sample_metadata),
            "sample_selection": "nearest occupied tracking cohort independently per camera; no cross-camera timestamp join",
            "output_dir": str(out),
            "source_alignment_warning": summary["video_clock_caveat"],
        }
        (out / "samples.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(out), "summary": summary, "sample_count": len(sample_metadata)}, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"review failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
