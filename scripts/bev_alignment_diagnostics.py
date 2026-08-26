#!/usr/bin/env python3
"""Collect and summarize DS9.1 BEV alignment diagnostics from MP4/file sources."""

from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse, unquote

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _now_run_id() -> str:
    return time.strftime("bev_alignment_%Y%m%d_%H%M%S")


def _file_path_from_uri(uri: str) -> Path | None:
    parsed = urlparse(str(uri))
    if parsed.scheme != "file":
        return None
    return Path(unquote(parsed.path)).expanduser()


def _sha1_head(path: Path, max_bytes: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        h.update(fh.read(max_bytes))
    return h.hexdigest()


def _source_blocks(lines: list[str]) -> list[tuple[int, int]]:
    starts: list[int] = []
    for idx, line in enumerate(lines):
        if re.match(r"^\s*-\s+element\s*:", line):
            starts.append(idx)
    blocks: list[tuple[int, int]] = []
    for pos, start in enumerate(starts):
        end = starts[pos + 1] if pos + 1 < len(starts) else len(lines)
        blocks.append((start, end))
    return blocks


def materialize_file_source_config(
    base_config: Path,
    output_config: Path,
    *,
    rtsp_port: int | None = None,
) -> dict[str, Any]:
    text = base_config.read_text(encoding="utf-8")
    lines = text.splitlines()
    replacements: list[dict[str, Any]] = []

    for start, end in _source_blocks(lines):
        active_idx = None
        active_uri = None
        commented_file_uris: list[str] = []
        for idx in range(start, end):
            active_match = re.match(r"^(\s*)uri\s*:\s*(\S.*)$", lines[idx])
            if active_match and not lines[idx].lstrip().startswith("#"):
                active_idx = idx
                active_uri = active_match.group(2).strip()
            commented_match = re.match(r"^\s*#\s*uri\s*:\s*(file://\S.*)$", lines[idx])
            if commented_match:
                commented_file_uris.append(commented_match.group(1).strip())
        if active_idx is None or active_uri is None:
            continue
        chosen_uri = active_uri
        if str(active_uri).strip().lower().startswith("rtsp://"):
            if not commented_file_uris:
                raise RuntimeError(f"No commented file:// source found near source block starting line {start + 1}")
            chosen_uri = commented_file_uris[0]
            indent = re.match(r"^(\s*)", lines[active_idx]).group(1)  # type: ignore[union-attr]
            lines[active_idx] = f"{indent}uri: {chosen_uri}"
        replacements.append({"line": active_idx + 1, "from": active_uri, "to": chosen_uri})

    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rtsp_override: int | None = None
    if rtsp_port is not None:
        cfg = _load_config(output_config)
        mosaic_cfg = cfg.setdefault("mosaic_output", {})
        if not isinstance(mosaic_cfg, dict):
            raise RuntimeError("mosaic_output must be a mapping when --rtsp-port is used")
        mosaic_cfg["rtsp_port"] = int(rtsp_port)
        output_config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        rtsp_override = int(rtsp_port)
    return {
        "base_config": str(base_config),
        "output_config": str(output_config),
        "replacements": replacements,
        "rtsp_port_override": rtsp_override,
    }


def _load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Config is not a YAML mapping: {path}")
    return data


def preflight_file_sources(config_path: Path, *, require_file: bool = True) -> dict[str, Any]:
    cfg = _load_config(config_path)
    sources = cfg.get("sources")
    if not isinstance(sources, list) or not sources:
        raise RuntimeError(f"No sources[] in {config_path}")
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for idx, src in enumerate(sources):
        if not isinstance(src, Mapping):
            errors.append(f"source {idx}: not a mapping")
            continue
        uri = str(src.get("uri") or "")
        row: dict[str, Any] = {"source_id": idx, "uri": uri}
        parsed = urlparse(uri)
        row["scheme"] = parsed.scheme
        if require_file and parsed.scheme != "file":
            errors.append(f"source {idx}: active uri is not file:// ({uri})")
        path = _file_path_from_uri(uri)
        if path is not None:
            row["path"] = str(path)
            row["exists"] = path.is_file()
            if path.is_file():
                stat = path.stat()
                row["size_bytes"] = int(stat.st_size)
                row["mtime_ns"] = int(stat.st_mtime_ns)
                row["sha1_head"] = _sha1_head(path)
                try:
                    import cv2  # type: ignore

                    cap = cv2.VideoCapture(str(path))
                    ok, frame = cap.read()
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    cap.release()
                    row["decode_ok"] = bool(ok)
                    row["fps"] = float(fps) if math.isfinite(float(fps)) else None
                    row["frame_count"] = int(frame_count) if math.isfinite(float(frame_count)) else None
                    if ok and frame is not None:
                        row["decoded_shape"] = [int(frame.shape[1]), int(frame.shape[0])]
                    else:
                        errors.append(f"source {idx}: unable to decode first frame ({path})")
                except Exception as exc:
                    row["decode_error"] = str(exc)
                    errors.append(f"source {idx}: decode preflight failed ({exc})")
            else:
                errors.append(f"source {idx}: file does not exist ({path})")
        rows.append(row)
    return {"config": str(config_path), "sources": rows, "errors": errors, "ok": not errors}


async def _recv_json(ws: Any, timeout_s: float) -> dict[str, Any] | None:
    try:
        raw = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
    except asyncio.TimeoutError:
        return None
    if isinstance(raw, (bytes, bytearray)):
        return None
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


async def _capture_ws(
    ws_url: str,
    *,
    cameras: list[str],
    duration_s: float,
    grid_res_m: float,
    max_extent_m: float,
    floorplan_cache_only: bool = False,
) -> list[dict[str, Any]]:
    try:
        import websockets  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"websockets package required: {exc}") from exc

    messages: list[dict[str, Any]] = []
    async with websockets.connect(ws_url, max_size=None) as ws:
        for camera in cameras:
            request_id = f"bev-align-{camera}-{int(time.time() * 1000)}"
            await ws.send(
                json.dumps(
                    {
                        "type": "get_floorplan",
                        "request_id": request_id,
                        "camera": camera,
                        "max_age_sec": 0,
                        "grid_res_m": grid_res_m,
                        "max_extent_m": max_extent_m,
                        "cache_only": bool(floorplan_cache_only),
                    }
                )
            )
            deadline = time.time() + 20.0
            while time.time() < deadline:
                payload = await _recv_json(ws, 2.0)
                if payload is None:
                    continue
                messages.append(payload)
                if payload.get("type") == "floorplan_response" and payload.get("request_id") == request_id:
                    break
        end_at = time.time() + max(1.0, float(duration_s))
        while time.time() < end_at:
            payload = await _recv_json(ws, min(2.0, max(0.1, end_at - time.time())))
            if payload is None:
                continue
            messages.append(payload)
    return messages


def _percentile(values: list[float], pct: float) -> float | None:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return None
    idx = int(round((len(vals) - 1) * max(0.0, min(1.0, pct))))
    return float(vals[idx])


def _stats(values: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return {
        "count": len(vals),
        "p05": _percentile(vals, 0.05),
        "p50": _percentile(vals, 0.50),
        "p95": _percentile(vals, 0.95),
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
    }


def _counter_dict(counter: collections.Counter[str]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def _summarize_message_window(messages: list[Mapping[str, Any]]) -> dict[str, Any]:
    floorplans: dict[str, dict[str, Any]] = {}
    tracking_counts: dict[str, int] = {}
    bev_counts: dict[str, int] = {}
    footpoint_counts: dict[str, int] = {}
    display_sources: dict[str, int] = {}
    display_sources_by_camera: dict[str, collections.Counter[str]] = {}
    depth_status_by_camera: dict[str, collections.Counter[str]] = {}
    depth_registration_status_by_camera: dict[str, collections.Counter[str]] = {}
    world_source_by_camera: dict[str, collections.Counter[str]] = {}
    selected_reason_by_camera: dict[str, collections.Counter[str]] = {}
    floorplan_alignment_by_camera: dict[str, collections.Counter[str]] = {}
    point_norm_y_by_camera: dict[str, list[float]] = {}
    candidate_ray_norm_y_by_camera: dict[str, dict[str, list[float]]] = {}
    candidate_depth_norm_y_by_camera: dict[str, dict[str, list[float]]] = {}
    snapshot_mismatch = 0
    snapshot_samples = 0
    chosen_out_of_bounds = 0
    footpoint_out_of_bounds = 0
    dropped_footpoints = 0
    registered_depth_anchor_count = 0
    registered_depth_anchor_inside = 0
    candidate_deltas: list[float] = []
    speeds: list[float] = []
    raw_speeds: list[float] = []
    speed_by_source_transition: dict[str, list[float]] = {}
    raw_speed_by_source_transition: dict[str, list[float]] = {}
    trail_segment_speeds: list[float] = []
    top_trail_segments: list[dict[str, Any]] = []
    top_jumps: list[dict[str, Any]] = []
    last_point: dict[tuple[str, str], dict[str, Any]] = {}

    for msg in messages:
        msg_type = msg.get("type")
        if msg_type == "floorplan_response":
            cam = str(msg.get("camera_id") or msg.get("camera") or "")
            if cam:
                floorplans[cam] = {
                    "snapshot_ts": msg.get("snapshot_ts"),
                    "ts": msg.get("ts"),
                    "served_from_cache": msg.get("served_from_cache"),
                    "bounds": msg.get("bounds"),
                    "grid_res_m": msg.get("grid_res_m"),
                    "error": msg.get("error"),
                }
        elif msg_type == "tracking":
            cam = str(msg.get("camera_id") or "")
            tracks = msg.get("tracks")
            if cam and isinstance(tracks, list):
                tracking_counts[cam] = tracking_counts.get(cam, 0) + len(tracks)
                depth_counter = depth_status_by_camera.setdefault(cam, collections.Counter())
                reg_counter = depth_registration_status_by_camera.setdefault(cam, collections.Counter())
                world_counter = world_source_by_camera.setdefault(cam, collections.Counter())
                for track in tracks:
                    if not isinstance(track, Mapping):
                        continue
                    depth_counter[str(track.get("depth_status"))] += 1
                    reg_counter[str(track.get("depth_registration_status"))] += 1
                    world_counter[str(track.get("world_source"))] += 1
        elif msg_type == "bev-frame":
            cam = str(msg.get("cameraId") or "")
            if not cam:
                continue
            bev_counts[cam] = bev_counts.get(cam, 0) + 1
            alignment = msg.get("floorplanAlignment")
            if isinstance(alignment, Mapping):
                quality = str(alignment.get("quality") or "unknown")
                applied = "applied" if alignment.get("applied") is True else "not_applied"
                reason = str(alignment.get("reason") or "")
                key = f"{quality}:{applied}" + (f":{reason}" if reason else "")
                floorplan_alignment_by_camera.setdefault(cam, collections.Counter())[key] += 1
            dropped = msg.get("droppedFootpoints")
            if isinstance(dropped, list):
                dropped_footpoints += len(dropped)
            trails = msg.get("trails")
            if isinstance(trails, list):
                for trail in trails:
                    if not isinstance(trail, Mapping):
                        continue
                    points = trail.get("points")
                    if not isinstance(points, list):
                        continue
                    for prev_point, next_point in zip(points, points[1:]):
                        if not isinstance(prev_point, Mapping) or not isinstance(next_point, Mapping):
                            continue
                        try:
                            t0 = float(prev_point.get("t"))
                            t1 = float(next_point.get("t"))
                            dt = (t1 - t0) / 1000.0
                            if dt <= 1e-3:
                                continue
                            x0 = float(prev_point.get("x"))
                            z0 = float(prev_point.get("y"))
                            x1 = float(next_point.get("x"))
                            z1 = float(next_point.get("y"))
                            speed = math.hypot(x1 - x0, z1 - z0) / dt
                        except Exception:
                            continue
                        trail_segment_speeds.append(float(speed))
                        top_trail_segments.append(
                            {
                                "camera": cam,
                                "stableId": trail.get("stableId"),
                                "trackerId": trail.get("trackerId"),
                                "from_t_ms": int(t0),
                                "to_t_ms": int(t1),
                                "dt_s": float(dt),
                                "speed_mps": float(speed),
                                "from": {"x": float(x0), "y": float(z0)},
                                "to": {"x": float(x1), "y": float(z1)},
                            }
                        )
            x_min = float(msg.get("xMin", float("nan")))
            x_max = float(msg.get("xMax", float("nan")))
            z_min = float(msg.get("zMin", float("nan")))
            z_max = float(msg.get("zMax", float("nan")))
            ts = int(msg.get("ts", 0) or 0)
            for fp in msg.get("footpoints") or []:
                if not isinstance(fp, Mapping):
                    continue
                footpoint_counts[cam] = footpoint_counts.get(cam, 0) + 1
                src = str(fp.get("displaySource") or "unknown")
                display_sources[src] = display_sources.get(src, 0) + 1
                display_sources_by_camera.setdefault(cam, collections.Counter())[src] += 1
                norm_y = fp.get("normY")
                if isinstance(norm_y, (int, float)) and math.isfinite(float(norm_y)):
                    point_norm_y_by_camera.setdefault(cam, []).append(float(norm_y))
                try:
                    x = float(fp.get("x"))
                    z = float(fp.get("y"))
                except Exception:
                    continue
                try:
                    raw_x = float(fp.get("rawX", x))
                    raw_z = float(fp.get("rawY", z))
                except Exception:
                    raw_x, raw_z = x, z
                selection_reason = None
                debug = fp.get("alignmentDebug")
                selection = None
                if isinstance(debug, Mapping):
                    selection = debug.get("displaySelection")
                    if isinstance(selection, Mapping):
                        selection_reason = str(selection.get("reason") or selection.get("selected") or "unknown")
                if not (x_min <= x <= x_max and z_min <= z <= z_max):
                    footpoint_out_of_bounds += 1
                key_id = fp.get("trackerId", fp.get("stableId"))
                if key_id is not None:
                    key = (cam, str(key_id))
                    prev = last_point.get(key)
                    if prev is not None and ts > int(prev["ts"]):
                        dt = (ts - int(prev["ts"])) / 1_000_000.0
                        if dt > 1e-3:
                            speed = math.hypot(x - float(prev["x"]), z - float(prev["z"])) / dt
                            raw_speed = math.hypot(raw_x - float(prev["raw_x"]), raw_z - float(prev["raw_z"])) / dt
                            speeds.append(speed)
                            raw_speeds.append(raw_speed)
                            transition = f"{prev.get('source', 'unknown')}->{src}"
                            speed_by_source_transition.setdefault(transition, []).append(speed)
                            raw_speed_by_source_transition.setdefault(transition, []).append(raw_speed)
                            top_jumps.append(
                                {
                                    "camera": cam,
                                    "track": str(key_id),
                                    "dt_s": float(dt),
                                    "speed_mps": float(speed),
                                    "raw_speed_mps": float(raw_speed),
                                    "from_source": str(prev.get("source", "unknown")),
                                    "to_source": str(src),
                                    "from_reason": prev.get("reason"),
                                    "to_reason": selection_reason,
                                    "from_ts": int(prev["ts"]),
                                    "to_ts": int(ts),
                                }
                            )
                    last_point[key] = {
                        "ts": int(ts),
                        "x": float(x),
                        "z": float(z),
                        "raw_x": float(raw_x),
                        "raw_z": float(raw_z),
                        "source": str(src),
                        "reason": selection_reason,
                    }
                if not isinstance(debug, Mapping):
                    continue
                chosen = debug.get("chosen")
                if isinstance(chosen, Mapping) and chosen.get("insideBounds") is False:
                    chosen_out_of_bounds += 1
                if isinstance(selection, Mapping):
                    reason = selection_reason or "unknown"
                    selected_reason_by_camera.setdefault(cam, collections.Counter())[reason] += 1
                registered_depth = debug.get("registeredDepthAnchor")
                if isinstance(registered_depth, Mapping):
                    registered_depth_anchor_count += 1
                    if registered_depth.get("insideBounds") is not False and registered_depth.get("floorplanInside") is not False:
                        registered_depth_anchor_inside += 1
                for cand in debug.get("candidates") or []:
                    if not isinstance(cand, Mapping):
                        continue
                    cand_name = str(cand.get("name") or "unknown")
                    ray_info = cand.get("rayFloor")
                    if isinstance(ray_info, Mapping):
                        ray_ny = ray_info.get("normY")
                        if isinstance(ray_ny, (int, float)) and math.isfinite(float(ray_ny)):
                            candidate_ray_norm_y_by_camera.setdefault(cam, {}).setdefault(cand_name, []).append(float(ray_ny))
                    delta = cand.get("depthVsRayDeltaM")
                    if isinstance(delta, (int, float)) and math.isfinite(float(delta)):
                        candidate_deltas.append(float(delta))
                    depth_info = cand.get("mapanythingDepth")
                    if isinstance(depth_info, Mapping):
                        depth_ny = depth_info.get("normY")
                        if isinstance(depth_ny, (int, float)) and math.isfinite(float(depth_ny)):
                            candidate_depth_norm_y_by_camera.setdefault(cam, {}).setdefault(cand_name, []).append(float(depth_ny))
                        sample = depth_info.get("sample")
                        if isinstance(sample, Mapping):
                            snapshot_samples += 1
                            if sample.get("snapshot_matches_active_floorplan") is False:
                                snapshot_mismatch += 1

    return {
        "message_counts": {
            "total": len(messages),
            "tracking_tracks_by_camera": tracking_counts,
            "bev_frames_by_camera": bev_counts,
            "bev_footpoints_by_camera": footpoint_counts,
        },
        "floorplans": floorplans,
        "display_sources": display_sources,
        "display_sources_by_camera": {
            cam: _counter_dict(counter) for cam, counter in sorted(display_sources_by_camera.items())
        },
        "tracking_depth_status_by_camera": {
            cam: _counter_dict(counter) for cam, counter in sorted(depth_status_by_camera.items())
        },
        "tracking_depth_registration_status_by_camera": {
            cam: _counter_dict(counter) for cam, counter in sorted(depth_registration_status_by_camera.items())
        },
        "tracking_world_source_by_camera": {
            cam: _counter_dict(counter) for cam, counter in sorted(world_source_by_camera.items())
        },
        "display_selection_reason_by_camera": {
            cam: _counter_dict(counter) for cam, counter in sorted(selected_reason_by_camera.items())
        },
        "floorplan_alignment_by_camera": {
            cam: _counter_dict(counter) for cam, counter in sorted(floorplan_alignment_by_camera.items())
        },
        "point_norm_y_by_camera": {
            cam: _stats(values) for cam, values in sorted(point_norm_y_by_camera.items())
        },
        "candidate_ray_norm_y_by_camera": {
            cam: {name: _stats(values) for name, values in sorted(candidates.items())}
            for cam, candidates in sorted(candidate_ray_norm_y_by_camera.items())
        },
        "candidate_depth_norm_y_by_camera": {
            cam: {name: _stats(values) for name, values in sorted(candidates.items())}
            for cam, candidates in sorted(candidate_depth_norm_y_by_camera.items())
        },
        "snapshot_samples": snapshot_samples,
        "snapshot_mismatch": snapshot_mismatch,
        "snapshot_mismatch_rate": (float(snapshot_mismatch) / snapshot_samples) if snapshot_samples else None,
        "footpoint_out_of_bounds": footpoint_out_of_bounds,
        "chosen_debug_out_of_bounds": chosen_out_of_bounds,
        "dropped_footpoints": int(dropped_footpoints),
        "registered_depth_anchor": {
            "count": int(registered_depth_anchor_count),
            "inside": int(registered_depth_anchor_inside),
            "inside_rate": (
                float(registered_depth_anchor_inside) / registered_depth_anchor_count
                if registered_depth_anchor_count
                else None
            ),
        },
        "candidate_depth_vs_ray_delta_m": {
            "count": len(candidate_deltas),
            "p50": _percentile(candidate_deltas, 0.50),
            "p95": _percentile(candidate_deltas, 0.95),
            "max": max(candidate_deltas) if candidate_deltas else None,
        },
        "speed_mps": {
            "count": len(speeds),
            "p50": _percentile(speeds, 0.50),
            "p95": _percentile(speeds, 0.95),
            "max": max(speeds) if speeds else None,
        },
        "raw_speed_mps": {
            "count": len(raw_speeds),
            "p50": _percentile(raw_speeds, 0.50),
            "p95": _percentile(raw_speeds, 0.95),
            "max": max(raw_speeds) if raw_speeds else None,
        },
        "speed_by_source_transition_mps": {
            key: _stats(values) for key, values in sorted(speed_by_source_transition.items())
        },
        "raw_speed_by_source_transition_mps": {
            key: _stats(values) for key, values in sorted(raw_speed_by_source_transition.items())
        },
        "trail_segment_speed_mps": {
            "count": len(trail_segment_speeds),
            "p50": _percentile(trail_segment_speeds, 0.50),
            "p95": _percentile(trail_segment_speeds, 0.95),
            "max": max(trail_segment_speeds) if trail_segment_speeds else None,
        },
        "top_trail_segments": sorted(
            top_trail_segments,
            key=lambda item: float(item.get("speed_mps") or 0.0),
            reverse=True,
        )[:12],
        "top_jumps": sorted(
            top_jumps,
            key=lambda item: float(item.get("raw_speed_mps") or item.get("speed_mps") or 0.0),
            reverse=True,
        )[:12],
    }


def _floorplan_ready_messages(messages: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    ready_cameras: set[str] = set()
    ready: list[Mapping[str, Any]] = []
    for msg in messages:
        msg_type = msg.get("type")
        if msg_type == "floorplan_response":
            cam = str(msg.get("camera_id") or msg.get("camera") or "")
            if cam and not msg.get("error"):
                ready_cameras.add(cam)
                ready.append(msg)
            continue
        if msg_type == "tracking":
            cam = str(msg.get("camera_id") or "")
            if cam in ready_cameras:
                ready.append(msg)
            continue
        if msg_type == "bev-frame":
            cam = str(msg.get("cameraId") or "")
            if cam in ready_cameras:
                ready.append(msg)
            continue
    return ready


def summarize_messages(messages: list[Mapping[str, Any]]) -> dict[str, Any]:
    summary = _summarize_message_window(messages)
    ready_messages = _floorplan_ready_messages(messages)
    ready_summary = _summarize_message_window(ready_messages)
    ready_summary["analysis_window"] = {
        "description": "messages for a camera after its floorplan_response succeeded",
        "message_count": len(ready_messages),
    }
    summary["floorplan_ready_window"] = ready_summary
    return summary


def _wait_for_ws(ws_url: str, timeout_s: float) -> None:
    async def _probe() -> bool:
        try:
            import websockets  # type: ignore

            async with websockets.connect(ws_url, max_size=None, open_timeout=2.0):
                return True
        except Exception:
            return False

    deadline = time.time() + max(1.0, timeout_s)
    while time.time() < deadline:
        if asyncio.run(_probe()):
            return
        time.sleep(1.0)
    raise RuntimeError(f"WebSocket did not become reachable: {ws_url}")


def _spawn_runtime(args: argparse.Namespace, config_path: Path) -> subprocess.Popen:
    parsed = urlparse(str(args.ws))
    ws_port = parsed.port or 6008
    cmd = [
        sys.executable,
        "DS9/noesis/ds9_runtime.py",
        "--pipeline-config",
        str(config_path),
        "--cameras-config",
        str(args.cameras_config),
        "--ws-port",
        str(ws_port),
    ]
    if args.disable_rest:
        cmd.append("--disable-rest")
    env = os.environ.copy()
    env["NOESIS_BEV_ALIGNMENT_DEBUG"] = "1"
    if args.disable_mosaic:
        env["NOESIS_MOSAIC_RTSP_ENABLED"] = "0"
        env["NOESIS_MOSAIC_WEBRTC_ENABLED"] = "0"
    return subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-config", default="DS9/config/infer.yaml")
    parser.add_argument("--cameras-config", default="config/cameras.yaml")
    parser.add_argument("--output-root", default="diagnostics/bev_alignment")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--ws", default="ws://127.0.0.1:6008")
    parser.add_argument("--duration", type=float, default=45.0)
    parser.add_argument("--grid-res-m", type=float, default=0.15)
    parser.add_argument("--max-extent-m", type=float, default=20.0)
    parser.add_argument("--cameras", nargs="+", default=["living-room", "kitchen", "family-room"])
    parser.add_argument("--rtsp-port", type=int, default=None)
    parser.add_argument("--floorplan-cache-only", action="store_true")
    parser.add_argument("--no-spawn", action="store_true")
    parser.add_argument("--disable-rest", action="store_true")
    parser.add_argument("--disable-mosaic", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    run_id = str(args.run_id or _now_run_id())
    out_dir = (REPO_ROOT / args.output_root / run_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_config = out_dir / "infer_file_sources.yaml"

    materialized = materialize_file_source_config(
        (REPO_ROOT / args.pipeline_config).resolve(),
        generated_config,
        rtsp_port=args.rtsp_port,
    )
    preflight = preflight_file_sources(generated_config, require_file=True)
    (out_dir / "config_materialization.json").write_text(json.dumps(materialized, indent=2) + "\n", encoding="utf-8")
    (out_dir / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")

    if not preflight.get("ok"):
        print(json.dumps({"status": "preflight_failed", "output_dir": str(out_dir), "errors": preflight.get("errors")}, indent=2))
        return 2
    if args.prepare_only:
        print(json.dumps({"status": "prepared", "output_dir": str(out_dir), "config": str(generated_config)}, indent=2))
        return 0

    proc: subprocess.Popen | None = None
    if not args.no_spawn:
        proc = _spawn_runtime(args, generated_config)
        try:
            _wait_for_ws(str(args.ws), timeout_s=45.0)
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError(f"Runtime exited early with code {proc.returncode}")
            raise

    try:
        messages = asyncio.run(
            _capture_ws(
                str(args.ws),
                cameras=[str(c) for c in args.cameras],
                duration_s=float(args.duration),
                grid_res_m=float(args.grid_res_m),
                max_extent_m=float(args.max_extent_m),
                floorplan_cache_only=bool(args.floorplan_cache_only),
            )
        )
        messages_path = out_dir / "messages.ndjson"
        with messages_path.open("w", encoding="utf-8") as fh:
            for msg in messages:
                fh.write(json.dumps(msg, sort_keys=True) + "\n")
        summary = summarize_messages(messages)
        summary.update({"status": "ok", "output_dir": str(out_dir), "messages": str(messages_path)})
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    finally:
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=12.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
