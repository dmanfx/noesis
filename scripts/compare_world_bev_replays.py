#!/usr/bin/env python3
"""Compare matched world outputs between two recorded replay captures.

This is a before/after output-difference report, not an accuracy test. It
matches same-camera cohorts by source media PTS within a finite tolerance,
without reusing either cohort, then matches person boxes one-to-one by IoU.
Only matched boxes with finite valid world positions in both captures contribute
to the XZ delta distribution. No phone path, PCF location, or static-room point
is used as position truth.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml


CANONICAL_CAMERAS = {"living-room", "kitchen", "family-room"}
MAX_CAPTURE_ROWS = 100_000
MAX_CAPTURE_BYTES = 1024 * 1024 * 1024
MAX_CAPTURE_LINE_BYTES = 32 * 1024 * 1024
MAX_TRACKS_PER_COHORT = 512
MAX_PTS_PAIR_EDGES = 1_000_000
MAX_BBOX_PAIR_COMPARISONS = 5_000_000


def _get(row: Mapping[str, Any], snake: str, camel: str) -> Any:
    return row.get(snake, row.get(camel))


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_uris(capture: Path) -> tuple[dict[int, str], Path]:
    config_path = capture.parent / "infer-replay.yaml"
    if not config_path.is_file():
        raise ValueError(f"capture replay config is missing: {config_path}")
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or not isinstance(value.get("sources"), list):
        raise ValueError(f"invalid replay source list: {config_path}")
    uris: dict[int, str] = {}
    for source_id, source in enumerate(value["sources"]):
        if not isinstance(source, Mapping) or not isinstance(source.get("uri"), str):
            raise ValueError(f"source {source_id} has no URI in {config_path}")
        uris[source_id] = source["uri"]
    return uris, config_path


def _load_tracking(path: Path) -> tuple[dict[tuple[str, int, int], list[dict[str, Any]]], Counter[str]]:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    exact_keys: set[tuple[Any, ...]] = set()
    total_bytes = 0
    with path.open("rb") as handle:
        line_number = 0
        while True:
            raw_line = handle.readline(MAX_CAPTURE_LINE_BYTES + 1)
            if not raw_line:
                break
            line_number += 1
            total_bytes += len(raw_line)
            if len(raw_line) > MAX_CAPTURE_LINE_BYTES:
                raise ValueError(
                    f"capture line exceeds {MAX_CAPTURE_LINE_BYTES} byte bound: {path}:{line_number}"
                )
            if total_bytes > MAX_CAPTURE_BYTES:
                raise ValueError(f"capture exceeds {MAX_CAPTURE_BYTES} byte bound: {path}")
            if line_number > MAX_CAPTURE_ROWS:
                raise ValueError(f"capture exceeds {MAX_CAPTURE_ROWS} rows: {path}")
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"invalid UTF-8 at {path}:{line_number}: {exc}") from exc
            if not line.strip():
                continue
            try:
                outer = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(outer, Mapping):
                counts["non_object_rows"] += 1
                continue
            row = outer
            if not isinstance(row.get("type"), str):
                nested = row.get("message")
                if isinstance(nested, Mapping):
                    row = nested
            kind = row.get("type")
            counts[str(kind or "untyped")] += 1
            if kind != "tracking":
                continue
            tracks = row.get("tracks")
            if isinstance(tracks, list) and len(tracks) > MAX_TRACKS_PER_COHORT:
                raise ValueError(
                    f"tracking cohort exceeds {MAX_TRACKS_PER_COHORT} track bound at {path}:{line_number}"
                )
            camera = _get(row, "camera_id", "cameraId")
            if camera not in CANONICAL_CAMERAS:
                counts["unsupported_camera"] += 1
                continue
            source_id = _integer(_get(row, "source_id", "sourceId"))
            epoch = _integer(_get(row, "source_epoch", "sourceEpoch"))
            frame = _integer(_get(row, "frame_id", "frameId"))
            observed = _integer(_get(row, "observed_at_us", "observedAtUs"))
            sequence = _integer(_get(row, "tracking_publication_sequence", "trackingPublicationSequence"))
            pts = _integer(_get(row, "media_pts_ns", "mediaPtsNs"))
            if None in (source_id, frame, observed, sequence, pts) or pts < 0:
                counts["tracking_missing_key_or_pts"] += 1
                continue
            cohort = row.get("cohort")
            if epoch is None and isinstance(cohort, Mapping):
                epoch = _integer(cohort.get("source_epoch"))
            epoch = 0 if epoch is None else epoch
            exact_key = (camera, source_id, epoch, frame, observed, sequence)
            if exact_key in exact_keys:
                raise ValueError(f"duplicate tracking cohort key at {path}:{line_number}")
            exact_keys.add(exact_key)
            row_copy = dict(row)
            row_copy["_media_pts_ns"] = pts
            row_copy["_source_id_int"] = source_id
            row_copy["_source_epoch_int"] = epoch
            groups[(str(camera), source_id, epoch)].append(row_copy)
    for rows in groups.values():
        rows.sort(key=lambda row: (row["_media_pts_ns"], row.get("frame_id", row.get("frameId", -1))))
    return groups, counts


def _bbox(track: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    raw = track.get("bbox")
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    values = tuple(_finite(value) for value in raw)
    if any(value is None for value in values):
        return None
    x, y, width, height = (float(value) for value in values)
    if width <= 0.0 or height <= 0.0:
        return None
    return x, y, width, height


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0.0, min(ay + ah, by + bh) - max(ay, by)
    )
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0.0 else 0.0


def _world_xz(track: Mapping[str, Any]) -> tuple[float, float] | None:
    world = track.get("world")
    if track.get("world_valid") is not True or not isinstance(world, (list, tuple)) or len(world) < 3:
        return None
    x, z = _finite(world[0]), _finite(world[2])
    return (x, z) if x is not None and z is not None else None


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _temporal_pairs(
    before: list[dict[str, Any]], after: list[dict[str, Any]], threshold_ns: int,
    *, max_candidate_edges: int,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any], int]], int, int, int]:
    """Greedily choose shortest-PTS-delta cohort edges, with no cohort reuse."""

    before_pts = [row["_media_pts_ns"] for row in before]
    edges: list[tuple[int, int, int, int, int]] = []
    for after_index, after_row in enumerate(after):
        pts = after_row["_media_pts_ns"]
        first = bisect_left(before_pts, pts - threshold_ns)
        last = bisect_right(before_pts, pts + threshold_ns)
        for before_index in range(first, last):
            if len(edges) >= max_candidate_edges:
                raise ValueError(
                    f"PTS cohort matching exceeds {max_candidate_edges} candidate-edge work bound"
                )
            delta = abs(pts - before_pts[before_index])
            edges.append((delta, pts, before_pts[before_index], after_index, before_index))
    edges.sort()
    used_after: set[int] = set()
    used_before: set[int] = set()
    pairs = []
    for delta, _, _, after_index, before_index in edges:
        if after_index in used_after or before_index in used_before:
            continue
        used_after.add(after_index)
        used_before.add(before_index)
        pairs.append((before[before_index], after[after_index], delta))
    pairs.sort(key=lambda pair: (pair[1]["_media_pts_ns"], pair[0]["_media_pts_ns"]))
    return pairs, len(after) - len(used_after), len(before) - len(used_before), len(edges)


def _match_tracks(
    before_row: Mapping[str, Any], after_row: Mapping[str, Any], min_iou: float,
    *, max_pair_comparisons: int,
) -> tuple[list[tuple[Mapping[str, Any], Mapping[str, Any], float]], int]:
    before_tracks = [t for t in before_row.get("tracks", []) if isinstance(t, Mapping) and _bbox(t)]
    after_tracks = [t for t in after_row.get("tracks", []) if isinstance(t, Mapping) and _bbox(t)]
    pair_comparisons = len(before_tracks) * len(after_tracks)
    if pair_comparisons > max_pair_comparisons:
        raise ValueError(
            f"bbox track matching exceeds {max_pair_comparisons} pair-comparison work bound"
        )
    edges = []
    for after_index, after_track in enumerate(after_tracks):
        after_box = _bbox(after_track)
        assert after_box is not None
        for before_index, before_track in enumerate(before_tracks):
            before_box = _bbox(before_track)
            assert before_box is not None
            overlap = _iou(after_box, before_box)
            if overlap >= min_iou:
                edges.append((overlap, after_index, before_index))
    edges.sort(key=lambda edge: (-edge[0], edge[1], edge[2]))
    used_after: set[int] = set()
    used_before: set[int] = set()
    matches = []
    for overlap, after_index, before_index in edges:
        if after_index in used_after or before_index in used_before:
            continue
        used_after.add(after_index)
        used_before.add(before_index)
        matches.append((before_tracks[before_index], after_tracks[after_index], overlap))
    return matches, pair_comparisons


def _track_lifecycle(track: Mapping[str, Any]) -> tuple[Any, Any]:
    return (
        track.get("tracker_id", track.get("trackerId")),
        track.get("tracker_lifecycle_generation", track.get("trackerLifecycleGeneration")),
    )


def compare(
    before_path: Path,
    after_path: Path,
    *,
    max_pts_delta_ms: float,
    min_bbox_iou: float,
) -> dict[str, Any]:
    before_uris, before_config = _source_uris(before_path)
    after_uris, after_config = _source_uris(after_path)
    before_groups, before_counts = _load_tracking(before_path)
    after_groups, after_counts = _load_tracking(after_path)
    threshold_ns_value = max_pts_delta_ms * 1_000_000
    if not math.isfinite(threshold_ns_value):
        raise ValueError("PTS tolerance overflows nanosecond conversion")
    threshold_ns = int(round(threshold_ns_value))
    keys = sorted(set(before_groups) | set(after_groups))
    per_camera: dict[str, dict[str, Any]] = {}
    temporal_details: list[dict[str, Any]] = []
    track_details: list[dict[str, Any]] = []
    pair_edge_work = 0
    bbox_pair_comparisons = 0

    for group in keys:
        camera, source_id, epoch = group
        before = before_groups.get(group, [])
        after = after_groups.get(group, [])
        before_uri = before_uris.get(source_id)
        after_uri = after_uris.get(source_id)
        if before and after and (before_uri is None or before_uri != after_uri):
            raise ValueError(
                f"source URI differs for {camera} source {source_id}: {before_uri!r} vs {after_uri!r}"
            )
        pairs, after_unmatched, before_unmatched, edge_count = _temporal_pairs(
            before, after, threshold_ns,
            max_candidate_edges=MAX_PTS_PAIR_EDGES - pair_edge_work,
        )
        pair_edge_work += edge_count
        bbox_count = Counter()
        delta_x: list[float] = []
        delta_z: list[float] = []
        delta_norm: list[float] = []
        pts_delta_ms = []
        for before_row, after_row, delta_ns in pairs:
            signed_delta_ns = after_row["_media_pts_ns"] - before_row["_media_pts_ns"]
            pts_delta_ms.append(abs(signed_delta_ns) / 1_000_000.0)
            matched_tracks, comparison_count = _match_tracks(
                before_row, after_row, min_bbox_iou,
                max_pair_comparisons=MAX_BBOX_PAIR_COMPARISONS - bbox_pair_comparisons,
            )
            bbox_pair_comparisons += comparison_count
            pair_track_counts = Counter()
            temporal_details.append({
                "camera_id": camera,
                "source_id": source_id,
                "source_epoch": epoch,
                "before_frame_id": _get(before_row, "frame_id", "frameId"),
                "after_frame_id": _get(after_row, "frame_id", "frameId"),
                "before_media_pts_ns": before_row["_media_pts_ns"],
                "after_media_pts_ns": after_row["_media_pts_ns"],
                "after_minus_before_pts_ms": signed_delta_ns / 1_000_000.0,
                "bbox_iou_matches": len(matched_tracks),
            })
            for before_track, after_track, overlap in matched_tracks:
                before_xz = _world_xz(before_track)
                after_xz = _world_xz(after_track)
                validity = (
                    "both_valid" if before_xz is not None and after_xz is not None
                    else "before_only_valid" if before_xz is not None
                    else "after_only_valid" if after_xz is not None
                    else "neither_valid"
                )
                bbox_count["one_to_one_bbox_matches"] += 1
                bbox_count[validity] += 1
                delta = None
                if before_xz is not None and after_xz is not None:
                    dx = after_xz[0] - before_xz[0]
                    dz = after_xz[1] - before_xz[1]
                    delta = {"x_m": dx, "z_m": dz, "norm_xz_m": math.hypot(dx, dz)}
                    delta_x.append(dx)
                    delta_z.append(dz)
                    delta_norm.append(delta["norm_xz_m"])
                track_details.append({
                    "camera_id": camera,
                    "source_id": source_id,
                    "source_epoch": epoch,
                    "before_frame_id": _get(before_row, "frame_id", "frameId"),
                    "after_frame_id": _get(after_row, "frame_id", "frameId"),
                    "before_media_pts_ns": before_row["_media_pts_ns"],
                    "after_media_pts_ns": after_row["_media_pts_ns"],
                    "after_minus_before_pts_ms": signed_delta_ns / 1_000_000.0,
                    "bbox_iou": overlap,
                    "before_lifecycle": _track_lifecycle(before_track),
                    "after_lifecycle": _track_lifecycle(after_track),
                    "before_stable_id": before_track.get("stable_id", before_track.get("stableId")),
                    "after_stable_id": after_track.get("stable_id", after_track.get("stableId")),
                    "before_world_valid": before_xz is not None,
                    "after_world_valid": after_xz is not None,
                    "before_world_xz_m": list(before_xz) if before_xz is not None else None,
                    "after_world_xz_m": list(after_xz) if after_xz is not None else None,
                    "delta_after_minus_before_xz_m": delta,
                })

        existing = per_camera.setdefault(camera, {
            "before_tracking_cohorts": 0,
            "after_tracking_cohorts": 0,
            "temporal_cohort_pairs_one_to_one": 0,
            "before_cohorts_without_pair_within_threshold": 0,
            "after_cohorts_without_pair_within_threshold": 0,
            "temporal_reuse": "prohibited: each before and after cohort is used at most once per camera/source/epoch",
            "source_ids_and_epochs": [],
            "pts_abs_delta_ms": [],
            "bbox_match_validity_counts": Counter(),
            "delta_x_m": [],
            "delta_z_m": [],
            "delta_norm_xz_m": [],
        })
        existing["before_tracking_cohorts"] += len(before)
        existing["after_tracking_cohorts"] += len(after)
        existing["temporal_cohort_pairs_one_to_one"] += len(pairs)
        existing["before_cohorts_without_pair_within_threshold"] += before_unmatched
        existing["after_cohorts_without_pair_within_threshold"] += after_unmatched
        existing["source_ids_and_epochs"].append({
            "source_id": source_id,
            "source_epoch": epoch,
            "before_video_uri": before_uri,
            "after_video_uri": after_uri,
        })
        existing["pts_abs_delta_ms"].extend(pts_delta_ms)
        existing["bbox_match_validity_counts"].update(bbox_count)
        existing["delta_x_m"].extend(delta_x)
        existing["delta_z_m"].extend(delta_z)
        existing["delta_norm_xz_m"].extend(delta_norm)

    for camera, values in per_camera.items():
        pts = values.pop("pts_abs_delta_ms")
        dx = values.pop("delta_x_m")
        dz = values.pop("delta_z_m")
        norm = values.pop("delta_norm_xz_m")
        counts = values.pop("bbox_match_validity_counts")
        values["bbox_match_validity_counts"] = dict(counts)
        values["pts_abs_delta_ms_distribution"] = {
            "p50": _percentile(pts, 0.50),
            "p95": _percentile(pts, 0.95),
            "max": max(pts) if pts else None,
        }
        values["both_world_valid_pairs"] = len(norm)
        values["delta_x_after_minus_before_m"] = {
            "median_signed": _percentile(dx, 0.50),
            "abs_p95": _percentile([abs(v) for v in dx], 0.95),
            "abs_p99": _percentile([abs(v) for v in dx], 0.99),
            "abs_max": max((abs(v) for v in dx), default=None),
        }
        values["delta_z_after_minus_before_m"] = {
            "median_signed": _percentile(dz, 0.50),
            "abs_p95": _percentile([abs(v) for v in dz], 0.95),
            "abs_p99": _percentile([abs(v) for v in dz], 0.99),
            "abs_max": max((abs(v) for v in dz), default=None),
        }
        values["delta_norm_xz_m"] = {
            "p50": _percentile(norm, 0.50),
            "p95": _percentile(norm, 0.95),
            "p99": _percentile(norm, 0.99),
            "max": max(norm) if norm else None,
        }

    return {
        "report_kind": "matched before/after world output difference; not accuracy",
        "method": {
            "cohort_match": "same camera, source_id, and source_epoch; global greedy minimum absolute media PTS delta, threshold inclusive",
            "bbox_match": "one-to-one greedy descending IoU, threshold inclusive",
            "world_delta": "after minus before, canonical world X and Z in meters; only both-valid matched detections",
            "max_pts_delta_ms": max_pts_delta_ms,
            "min_bbox_iou": min_bbox_iou,
            "temporal_reuse": "none; each before and after cohort is used at most once within camera/source/epoch",
            "not_truth": "Neither Noesis outputs, PCF, nor phone optical-center/phone path are independent occupied-person position truth.",
        },
        "bounded_work": {
            "capture_rows_per_input_max": MAX_CAPTURE_ROWS,
            "capture_bytes_per_input_max": MAX_CAPTURE_BYTES,
            "capture_line_bytes_max": MAX_CAPTURE_LINE_BYTES,
            "tracks_per_cohort_max": MAX_TRACKS_PER_COHORT,
            "pts_candidate_edges_max": MAX_PTS_PAIR_EDGES,
            "pts_candidate_edges_used": pair_edge_work,
            "bbox_pair_comparisons_max": MAX_BBOX_PAIR_COMPARISONS,
            "bbox_pair_comparisons_used": bbox_pair_comparisons,
        },
        "inputs": {
            "before_capture": str(before_path.resolve()),
            "before_capture_sha256": _sha256(before_path),
            "before_replay_config": str(before_config.resolve()),
            "before_replay_config_sha256": _sha256(before_config),
            "after_capture": str(after_path.resolve()),
            "after_capture_sha256": _sha256(after_path),
            "after_replay_config": str(after_config.resolve()),
            "after_replay_config_sha256": _sha256(after_config),
            "input_row_counts": {"before": dict(before_counts), "after": dict(after_counts)},
        },
        "summary_by_camera": per_camera,
        "temporal_cohort_pairs": temporal_details,
        "bbox_matched_tracks": track_details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=Path, help="before capture cohorts.ndjson")
    parser.add_argument("--after", required=True, type=Path, help="after capture cohorts.ndjson")
    parser.add_argument("--output", required=True, type=Path, help="new output directory under review artifacts")
    parser.add_argument("--max-pts-delta-ms", type=float, default=100.0)
    parser.add_argument("--min-bbox-iou", type=float, default=0.5)
    args = parser.parse_args()
    before = args.before.expanduser().resolve()
    after = args.after.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not before.is_file() or not after.is_file():
        raise SystemExit("both capture inputs must exist")
    if not math.isfinite(args.max_pts_delta_ms) or args.max_pts_delta_ms < 0.0:
        raise SystemExit("--max-pts-delta-ms must be finite and nonnegative")
    if not math.isfinite(args.min_bbox_iou) or not 0.0 <= args.min_bbox_iou <= 1.0:
        raise SystemExit("--min-bbox-iou must be between 0 and 1")
    report = compare(
        before,
        after,
        max_pts_delta_ms=args.max_pts_delta_ms,
        min_bbox_iou=args.min_bbox_iou,
    )
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(report_path), "summary_by_camera": report["summary_by_camera"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
