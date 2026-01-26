#!/usr/bin/env python3
"""Summarize SV3DT terminated track dumps + diagnostics logs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


TRACKER_STATE = {
    0: "EMPTY",
    1: "ACTIVE",
    2: "INACTIVE",
    3: "TENTATIVE",
    4: "PROJECTED",
    5: "QUASIACTIVE",
}


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _iter_ndjson(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            if isinstance(record, dict):
                yield record


def _parse_list(token: str) -> List[float]:
    if not token or token == "-1":
        return []
    out: List[float] = []
    for piece in token.split("|"):
        if not piece:
            continue
        try:
            out.append(float(piece))
        except Exception:
            continue
    return out


@dataclass
class DumpEntry:
    frame: int
    track_id: int
    bbox: Tuple[float, float, float, float]
    conf: float
    col8: float
    col9: float
    col10: float
    col11: float
    col12: float
    col13: float
    foot_u: float
    foot_v: float
    poly_len: int
    col17: float
    list18_len: int


def _parse_track_dump(path: Path) -> List[DumpEntry]:
    entries: List[DumpEntry] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("Target:"):
                continue
            fields = line.split(",")
            if len(fields) < 15:
                continue
            frame = _safe_int(fields[0])
            track_id = _safe_int(fields[1])
            left = _safe_float(fields[2])
            top = _safe_float(fields[3])
            width = _safe_float(fields[4])
            height = _safe_float(fields[5])
            conf = _safe_float(fields[6])
            col8 = _safe_float(fields[7])
            col9 = _safe_float(fields[8])
            col10 = _safe_float(fields[9])
            col11 = _safe_float(fields[10])
            col12 = _safe_float(fields[11])
            col13 = _safe_float(fields[12])
            foot_u = _safe_float(fields[13])
            foot_v = _safe_float(fields[14])
            poly = _parse_list(fields[15]) if len(fields) > 15 else []
            col17 = _safe_float(fields[16]) if len(fields) > 16 else float("nan")
            list18 = _parse_list(fields[17]) if len(fields) > 17 else []
            entries.append(
                DumpEntry(
                    frame=frame,
                    track_id=track_id,
                    bbox=(left, top, width, height),
                    conf=conf,
                    col8=col8,
                    col9=col9,
                    col10=col10,
                    col11=col11,
                    col12=col12,
                    col13=col13,
                    foot_u=foot_u,
                    foot_v=foot_v,
                    poly_len=len(poly),
                    col17=col17,
                    list18_len=len(list18),
                )
            )
    return entries


def _load_cameras(path: Optional[Path]) -> Dict[int, str]:
    if path is None or not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cameras = data.get("cameras") or {}
    mapping: Dict[int, str] = {}
    for cam_id, entry in cameras.items():
        try:
            idx = int(cam_id)
        except Exception:
            continue
        if isinstance(entry, dict):
            name = entry.get("name")
            if isinstance(name, str) and name:
                mapping[idx] = name
    return mapping


def _source_from_dump(path: Path) -> Optional[int]:
    match = re.search(r"noesis_track_dump_(\d+)", path.name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


@dataclass
class TrackStats:
    first_frame: int
    last_frame: int
    last_entry: Dict[str, Any]
    bbox3d_seen: bool = False
    bbox3d_last: bool = False
    world_valid_seen: bool = False
    world_valid_last: bool = False
    visibility_last: Optional[float] = None
    history: deque = None


def _collect_tracking_stats(diag_path: Path, gap: int) -> Dict[str, Any]:
    by_camera: Dict[str, Dict[int, TrackStats]] = defaultdict(dict)
    frame_counts: Dict[str, Dict[int, int]] = defaultdict(dict)
    ts_range: Dict[str, List[float]] = defaultdict(lambda: [float("inf"), float("-inf")])

    for record in _iter_ndjson(diag_path):
        if record.get("type") not in ("v3dt_tracking_frame", "tracking"):
            continue
        camera_id = str(record.get("camera_id") or record.get("camera") or "")
        if not camera_id:
            continue
        frame_id = _safe_int(record.get("frame_id", -1), -1)
        if frame_id < 0:
            continue
        tracks = record.get("tracks") or []
        if not isinstance(tracks, list):
            continue
        frame_counts[camera_id][frame_id] = len(tracks)
        ts = _safe_float(record.get("ts"), float("nan"))
        if math.isfinite(ts):
            ts_range[camera_id][0] = min(ts_range[camera_id][0], ts)
            ts_range[camera_id][1] = max(ts_range[camera_id][1], ts)
        for track in tracks:
            if not isinstance(track, dict):
                continue
            track_id = _safe_int(track.get("track_id", -1), -1)
            if track_id < 0:
                continue
            entry = track
            stats = by_camera[camera_id].get(track_id)
            if stats is None:
                stats = TrackStats(
                    first_frame=frame_id,
                    last_frame=frame_id,
                    last_entry=entry,
                    history=deque(maxlen=5),
                )
                by_camera[camera_id][track_id] = stats
            stats.last_frame = frame_id
            stats.last_entry = entry
            stats.history.append(entry)
            bbox3d_present = isinstance(entry.get("bbox3d"), dict)
            stats.bbox3d_seen = stats.bbox3d_seen or bbox3d_present
            stats.bbox3d_last = bbox3d_present
            world_valid = entry.get("world_valid")
            if isinstance(world_valid, bool):
                stats.world_valid_seen = stats.world_valid_seen or world_valid
                stats.world_valid_last = world_valid
            vis = entry.get("visibility")
            if isinstance(vis, (int, float)):
                stats.visibility_last = float(vis)

    summary: Dict[str, Any] = {}
    for camera_id, tracks in by_camera.items():
        max_frame = max(frame_counts.get(camera_id, {0: 0}).keys() or [0])
        terminations = []
        lengths = []
        active = 0
        for track_id, stats in tracks.items():
            length = stats.last_frame - stats.first_frame + 1
            lengths.append(length)
            terminated = stats.last_frame <= (max_frame - gap)
            if not terminated:
                active += 1
                continue
            terminations.append(
                {
                    "camera_id": camera_id,
                    "track_id": track_id,
                    "start_frame": stats.first_frame,
                    "end_frame": stats.last_frame,
                    "last_entry": stats.last_entry,
                    "bbox3d_seen": stats.bbox3d_seen,
                    "bbox3d_last": stats.bbox3d_last,
                    "world_valid_seen": stats.world_valid_seen,
                    "world_valid_last": stats.world_valid_last,
                    "visibility_last": stats.visibility_last,
                    "multi_person_end": frame_counts[camera_id].get(stats.last_frame, 0) > 1,
                }
            )
        duration_s = None
        ts_min, ts_max = ts_range[camera_id]
        if math.isfinite(ts_min) and math.isfinite(ts_max) and ts_max > ts_min:
            duration_s = ts_max - ts_min
        summary[camera_id] = {
            "max_frame": max_frame,
            "track_lengths": lengths,
            "active_tracks": active,
            "terminations": terminations,
            "duration_s": duration_s,
        }
    return summary


def _mean_abs_error(pairs: Sequence[Tuple[float, float]]) -> Optional[float]:
    diffs = []
    for a, b in pairs:
        if math.isfinite(a) and math.isfinite(b):
            diffs.append(abs(a - b))
    if not diffs:
        return None
    return sum(diffs) / len(diffs)


def _infer_dump_schema(
    entries: Sequence[DumpEntry],
    ndjson_tracks: Dict[Tuple[str, int, int], Dict[str, Any]],
    camera_id: str,
) -> Dict[str, Any]:
    matches: List[Tuple[DumpEntry, Dict[str, Any]]] = []
    for entry in entries:
        key = (camera_id, entry.track_id, entry.frame)
        track = ndjson_tracks.get(key)
        if track:
            matches.append((entry, track))

    def collect_pairs(get_a, get_b) -> List[Tuple[float, float]]:
        pairs = []
        for entry, track in matches:
            pairs.append((get_a(entry), get_b(track)))
        return pairs

    conf_vs_det = _mean_abs_error(collect_pairs(lambda e: e.conf, lambda t: _safe_float(t.get("confidence"))))
    conf_vs_track = _mean_abs_error(
        collect_pairs(lambda e: e.conf, lambda t: _safe_float(t.get("tracker_confidence")))
    )
    col11_vs_class = _mean_abs_error(
        collect_pairs(lambda e: e.col11, lambda t: float(_safe_int(t.get("class_id", -1))))
    )
    col13_vs_vis = _mean_abs_error(
        collect_pairs(lambda e: e.col13, lambda t: _safe_float(t.get("visibility")))
    )
    foot_err = _mean_abs_error(
        collect_pairs(lambda e: e.foot_u, lambda t: _safe_float((t.get("image_foot") or [float("nan")])[0]))
    )
    foot_err_y = _mean_abs_error(
        collect_pairs(lambda e: e.foot_v, lambda t: _safe_float((t.get("image_foot") or [float("nan"), float("nan")])[1]))
    )
    col8_vs_bbox3d_x = _mean_abs_error(
        collect_pairs(lambda e: e.col8, lambda t: _safe_float((t.get("bbox3d") or {}).get("xCentre")))
    )
    col9_vs_bbox3d_y = _mean_abs_error(
        collect_pairs(lambda e: e.col9, lambda t: _safe_float((t.get("bbox3d") or {}).get("yCentre")))
    )
    col8_vs_world_x = _mean_abs_error(
        collect_pairs(lambda e: e.col8, lambda t: _safe_float((t.get("world") or [float("nan")])[0]))
    )
    col9_vs_world_y = _mean_abs_error(
        collect_pairs(lambda e: e.col9, lambda t: _safe_float((t.get("world") or [float("nan"), float("nan")])[1]))
    )

    return {
        "matches": len(matches),
        "conf_vs_det": conf_vs_det,
        "conf_vs_tracker": conf_vs_track,
        "col11_vs_class": col11_vs_class,
        "col13_vs_visibility": col13_vs_vis,
        "foot_u_vs_image_foot_u": foot_err,
        "foot_v_vs_image_foot_v": foot_err_y,
        "col8_vs_bbox3d_x": col8_vs_bbox3d_x,
        "col9_vs_bbox3d_y": col9_vs_bbox3d_y,
        "col8_vs_world_x": col8_vs_world_x,
        "col9_vs_world_y": col9_vs_world_y,
    }


def _flatten_track(track: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    bbox = track.get("bbox") or []
    out["bbox_h"] = _safe_float(bbox[3]) if isinstance(bbox, Sequence) and len(bbox) >= 4 else float("nan")
    out["det_conf"] = _safe_float(track.get("confidence"))
    out["tracker_conf"] = _safe_float(track.get("tracker_confidence"))
    out["bbox3d_present"] = isinstance(track.get("bbox3d"), dict)
    out["world_valid"] = track.get("world_valid") if isinstance(track.get("world_valid"), bool) else None
    vis = track.get("visibility")
    out["visibility"] = float(vis) if isinstance(vis, (int, float)) else None
    return out


def _summarize_terminations(
    diag_summary: Dict[str, Any],
    dump_summary: Dict[Tuple[str, int], Dict[str, Any]],
    tentative_conf: Optional[float],
    min_conf: Optional[float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for camera_id, data in diag_summary.items():
        for term in data.get("terminations", []):
            track_id = term["track_id"]
            entry = term["last_entry"] if isinstance(term.get("last_entry"), dict) else {}
            flattened = _flatten_track(entry)
            dump = dump_summary.get((camera_id, track_id), {})
            det_conf = flattened.get("det_conf")
            bbox_h = flattened.get("bbox_h")
            row = {
                "camera": camera_id,
                "track_id": track_id,
                "start_frame": term.get("start_frame"),
                "end_frame": term.get("end_frame"),
                "bbox_h_end": bbox_h,
                "det_conf_end": det_conf,
                "tracker_conf_end": flattened.get("tracker_conf"),
                "bbox3d_present_end": flattened.get("bbox3d_present"),
                "world_valid_end": flattened.get("world_valid"),
                "visibility_end": flattened.get("visibility"),
                "multi_person_end": term.get("multi_person_end"),
                "bbox3d_flipped": term.get("bbox3d_seen") and not term.get("bbox3d_last"),
                "world_valid_flipped": term.get("world_valid_seen") and not term.get("world_valid_last"),
                "dump_col13_end": dump.get("col13_last"),
                "dump_col13_flipped": dump.get("col13_flipped"),
                "dump_tracker_state": dump.get("tracker_state"),
            }
            if tentative_conf is not None and isinstance(det_conf, (int, float)):
                row["below_tentative"] = det_conf < tentative_conf
            if min_conf is not None and isinstance(det_conf, (int, float)):
                row["below_min_conf"] = det_conf < min_conf
            rows.append(row)
    return rows


def _format_ratio(count: int, total: int) -> str:
    if total <= 0:
        return "n/a"
    return f"{count}/{total} ({(100.0 * count / total):.1f}%)"


def _write_markdown(
    out_path: Path,
    diag_summary: Dict[str, Any],
    term_rows: List[Dict[str, Any]],
    dump_schema: Dict[str, Any],
    dump_files: List[Path],
    tentative_conf: Optional[float],
    min_conf: Optional[float],
) -> None:
    lines: List[str] = []
    lines.append("# Track Termination Report")
    lines.append("")
    if dump_files:
        lines.append("## Track dump inputs")
        for path in dump_files:
            lines.append(f"- {path}")
        lines.append("")
    if dump_schema:
        lines.append("## Track dump schema inference")
        lines.append(f"- matched rows: {dump_schema.get('matches', 0)}")
        lines.append(f"- conf vs det_conf MAE: {dump_schema.get('conf_vs_det')}")
        lines.append(f"- conf vs tracker_conf MAE: {dump_schema.get('conf_vs_tracker')}")
        lines.append(f"- col11 vs class_id MAE: {dump_schema.get('col11_vs_class')}")
        lines.append(f"- col13 vs visibility MAE: {dump_schema.get('col13_vs_visibility')}")
        lines.append(f"- foot_u vs image_foot.u MAE: {dump_schema.get('foot_u_vs_image_foot_u')}")
        lines.append(f"- foot_v vs image_foot.v MAE: {dump_schema.get('foot_v_vs_image_foot_v')}")
        lines.append(f"- col8 vs bbox3d.xCentre MAE: {dump_schema.get('col8_vs_bbox3d_x')}")
        lines.append(f"- col9 vs bbox3d.yCentre MAE: {dump_schema.get('col9_vs_bbox3d_y')}")
        lines.append(f"- col8 vs world.x MAE: {dump_schema.get('col8_vs_world_x')}")
        lines.append(f"- col9 vs world.y MAE: {dump_schema.get('col9_vs_world_y')}")
        lines.append("")

    lines.append("## Per-camera summary")
    for camera_id, data in diag_summary.items():
        lengths = data.get("track_lengths") or []
        terminations = data.get("terminations") or []
        active = data.get("active_tracks", 0)
        duration_s = data.get("duration_s")
        lines.append(f"- {camera_id}: tracks={len(lengths)}, terminated={len(terminations)}, active_end={active}")
        if lengths:
            lines.append(
                f"  - length_frames median={statistics.median(lengths):.1f} "
                f"mean={statistics.mean(lengths):.1f} min={min(lengths)} max={max(lengths)}"
            )
        if duration_s:
            rate = len(terminations) / duration_s * 60.0 if duration_s > 0 else 0.0
            lines.append(f"  - duration={duration_s:.1f}s terminations_per_min={rate:.2f}")
    lines.append("")

    if term_rows:
        total = len(term_rows)
        below_tentative = sum(1 for row in term_rows if row.get("below_tentative"))
        below_min = sum(1 for row in term_rows if row.get("below_min_conf"))
        small_bbox = []
        for row in term_rows:
            bbox_h = row.get("bbox_h_end")
            if isinstance(bbox_h, (int, float)) and math.isfinite(bbox_h):
                small_bbox.append(bbox_h)
        bbox_threshold = None
        if small_bbox:
            bbox_threshold = statistics.quantiles(small_bbox, n=4)[0]
        small_bbox_hits = 0
        for row in term_rows:
            bbox_h = row.get("bbox_h_end")
            if bbox_threshold is not None and isinstance(bbox_h, (int, float)):
                if bbox_h <= bbox_threshold:
                    small_bbox_hits += 1

        lines.append("## Termination correlations")
        if tentative_conf is not None:
            lines.append(f"- det_conf < tentative ({tentative_conf}): {_format_ratio(below_tentative, total)}")
        if min_conf is not None:
            lines.append(f"- det_conf < min_detector ({min_conf}): {_format_ratio(below_min, total)}")
        if bbox_threshold is not None:
            lines.append(f"- bbox_h in bottom quartile (<= {bbox_threshold:.1f}px): {_format_ratio(small_bbox_hits, total)}")
        bbox3d_drop = sum(1 for row in term_rows if row.get("bbox3d_flipped"))
        lines.append(f"- bbox3d present then missing at end: {_format_ratio(bbox3d_drop, total)}")
        world_drop = sum(1 for row in term_rows if row.get("world_valid_flipped"))
        lines.append(f"- world_valid true then false at end: {_format_ratio(world_drop, total)}")
        multi = sum(1 for row in term_rows if row.get("multi_person_end"))
        lines.append(f"- multi-person at end frame: {_format_ratio(multi, total)}")
        lines.append("")

    if term_rows:
        lines.append("## Termination table (CSV fields)")
        lines.append(
            "Columns: camera, track_id, start_frame, end_frame, bbox_h_end, det_conf_end, "
            "tracker_conf_end, reason, dump_col13_end, dump_col13_flipped, bbox3d_present_end, world_valid_end"
        )
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SV3DT track dump + diagnostics logs.")
    parser.add_argument("--dump-dir", type=str, default="/tmp", help="Directory with noesis_track_dump_*.txt")
    parser.add_argument("--diag-log", type=str, required=True, help="V3DT NDJSON log path")
    parser.add_argument("--cameras-config", type=str, default="", help="cameras.yaml path for id->name")
    parser.add_argument("--out-dir", type=str, default="diagnostics", help="Output directory")
    parser.add_argument("--gap", type=int, default=3, help="Gap frames to treat as terminated")
    parser.add_argument("--tentative-conf", type=float, default=None, help="tentativeDetectorConfidence")
    parser.add_argument("--min-conf", type=float, default=None, help="minDetectorConfidence")
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    diag_log = Path(args.diag_log)
    if not diag_log.exists():
        raise SystemExit(f"Diagnostics log not found: {diag_log}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    camera_map = _load_cameras(Path(args.cameras_config)) if args.cameras_config else {}

    dump_entries_by_camera: Dict[str, List[DumpEntry]] = defaultdict(list)
    dump_summary: Dict[Tuple[str, int], Dict[str, Any]] = {}
    dump_files: List[Path] = []
    for path in sorted(dump_dir.glob("noesis_track_dump_*.txt")):
        dump_files.append(path)
        entries = _parse_track_dump(path)
        source_id = _source_from_dump(path)
        camera_id = camera_map.get(source_id, str(source_id if source_id is not None else "unknown"))
        dump_entries_by_camera[camera_id].extend(entries)
        grouped: Dict[int, List[DumpEntry]] = defaultdict(list)
        for entry in entries:
            grouped[entry.track_id].append(entry)
        for track_id, items in grouped.items():
            items.sort(key=lambda item: item.frame)
            col13_values = [item.col13 for item in items if math.isfinite(item.col13)]
            col13_last = col13_values[-1] if col13_values else None
            col13_flipped = False
            if col13_values and col13_values[-1] == 0.0 and any(v > 0.0 for v in col13_values[:-1]):
                col13_flipped = True
            state_raw = items[-1].col12
            state = None
            if math.isfinite(state_raw):
                state_int = int(state_raw)
                if state_int in TRACKER_STATE:
                    state = TRACKER_STATE[state_int]
            dump_summary[(camera_id, track_id)] = {
                "col13_last": col13_last,
                "col13_flipped": col13_flipped,
                "tracker_state": state,
            }

    diag_summary = _collect_tracking_stats(diag_log, gap=args.gap)

    ndjson_tracks: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    for record in _iter_ndjson(diag_log):
        if record.get("type") not in ("v3dt_tracking_frame", "tracking"):
            continue
        camera_id = str(record.get("camera_id") or record.get("camera") or "")
        frame_id = _safe_int(record.get("frame_id", -1), -1)
        if not camera_id or frame_id < 0:
            continue
        tracks = record.get("tracks") or []
        if not isinstance(tracks, list):
            continue
        for track in tracks:
            if not isinstance(track, dict):
                continue
            track_id = _safe_int(track.get("track_id", -1), -1)
            if track_id < 0:
                continue
            ndjson_tracks[(camera_id, track_id, frame_id)] = track

    dump_schema: Dict[str, Any] = {}
    if dump_entries_by_camera:
        camera_id = next(iter(dump_entries_by_camera.keys()))
        dump_schema = _infer_dump_schema(dump_entries_by_camera[camera_id], ndjson_tracks, camera_id)

    term_rows = _summarize_terminations(
        diag_summary,
        dump_summary,
        tentative_conf=args.tentative_conf,
        min_conf=args.min_conf,
    )

    csv_path = out_dir / "track_terminations.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(term_rows[0].keys()) if term_rows else [])
        if term_rows:
            writer.writeheader()
            writer.writerows(term_rows)

    report_path = out_dir / "trackdump_report.md"
    _write_markdown(
        report_path,
        diag_summary,
        term_rows,
        dump_schema,
        dump_files,
        tentative_conf=args.tentative_conf,
        min_conf=args.min_conf,
    )

    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
