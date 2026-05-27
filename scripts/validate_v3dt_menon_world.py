#!/usr/bin/env python3
"""Validate V3DT public world tracks against Menon's existing scene path."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import websockets  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"[FAIL] websockets package required: {exc}") from exc

from geometry.homography import project_world_to_image
from noesis.calibration.manager import CalibrationManager
from noesis.virtual_twin.store import VirtualTwinStore


REQUIRED_TRACK_FIELDS = {
    "world",
    "world_valid",
    "world_frame",
    "world_source",
    "stable_id",
    "tracker_id",
    "camera_id",
}
FORBIDDEN_PUBLIC_V3DT_FIELDS = {"bbox3d", "velocity3d", "visibility", "cuboid", "cuboid_vertices"}
EXPECTED_WORLD_FRAME = "backend_world_m"
EXPECTED_WORLD_SOURCE = "v3dt_bbox3d_foot"
CAMERA_ROOM_ALIASES = {
    "living-room": ("living room", "foyer"),
    "kitchen": ("kitchen", "kitchen2", "foyer"),
    "family-room": ("family room",),
}


def _load_json(path: Path, fallback: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def _load_yaml(path: Path, fallback: Any = None) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or fallback
    except Exception:
        return fallback


def _percentile(values: Iterable[float], q: float) -> float | None:
    vals = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if vals.size == 0:
        return None
    return float(np.percentile(vals, q))


def _stats(values: Iterable[float]) -> dict[str, Any]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return {
        "count": len(vals),
        "median": _percentile(vals, 50),
        "p90": _percentile(vals, 90),
        "p95": _percentile(vals, 95),
        "max": _percentile(vals, 100),
    }


def _matrix_col_major(values: Any) -> np.ndarray | None:
    if not isinstance(values, list) or len(values) != 16:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return None
    return arr.reshape((4, 4), order="F")


def _transform_point(matrix: np.ndarray, point: Iterable[float]) -> list[float] | None:
    try:
        vec = np.asarray([*list(point)[:3], 1.0], dtype=np.float64)
    except Exception:
        return None
    out = matrix @ vec
    if not np.all(np.isfinite(out[:3])):
        return None
    return [float(out[0]), float(out[1]), float(out[2])]


def _normalize_zone(name: Any) -> str:
    return str(name or "").strip().lower().replace("_", " ").replace("-", " ")


def _room_union_bounds(room_zones: list[dict[str, Any]], camera_id: str) -> dict[str, float] | None:
    aliases = set(CAMERA_ROOM_ALIASES.get(camera_id, (_normalize_zone(camera_id),)))
    matches = []
    for zone in room_zones:
        if _normalize_zone(zone.get("name")) not in aliases:
            continue
        bounds = zone.get("bounds")
        if not isinstance(bounds, dict):
            continue
        try:
            matches.append(
                {
                    "minX": float(bounds["minX"]),
                    "maxX": float(bounds["maxX"]),
                    "minZ": float(bounds["minZ"]),
                    "maxZ": float(bounds["maxZ"]),
                    "floorY": float(zone.get("floorY", 0.0)),
                }
            )
        except Exception:
            continue
    if not matches:
        return None
    return {
        "minX": min(m["minX"] for m in matches),
        "maxX": max(m["maxX"] for m in matches),
        "minZ": min(m["minZ"] for m in matches),
        "maxZ": max(m["maxZ"] for m in matches),
        "floorY": float(np.median([m["floorY"] for m in matches])),
    }


def _inside_room(point: list[float] | None, bounds: dict[str, float] | None, margin: float = 0.0) -> bool:
    if point is None or bounds is None:
        return False
    x, _y, z = point
    return (
        float(bounds["minX"]) - margin <= x <= float(bounds["maxX"]) + margin
        and float(bounds["minZ"]) - margin <= z <= float(bounds["maxZ"]) + margin
    )


def _load_scene_matrix(ply_alignment_path: Path) -> np.ndarray | None:
    align = _load_json(ply_alignment_path, {}) or {}
    similarity = align.get("scene_similarity") if isinstance(align, dict) else None
    matrix = _matrix_col_major((similarity or {}).get("world_to_scene_col_major"))
    if matrix is not None:
        return matrix
    raw = align.get("matrix") if isinstance(align, dict) else None
    return _matrix_col_major(raw)


def _latest_vt_alignments(root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    try:
        store = VirtualTwinStore(root=root)
        rows = store.list_revisions()
    except Exception:
        return out
    required_gates = (
        "mapanything_min_coverage_pass",
        "zeroplane_min_planes_pass",
        "registration_min_correspondences_pass",
        "registration_normal_median_pass",
        "registration_normal_p90_pass",
        "room_model_leakage_pass",
    )
    for row in rows:
        camera_id = str(row.get("camera") or "").strip()
        revision_id = str(row.get("revision_id") or "").strip()
        if not camera_id or not revision_id or camera_id in out:
            continue
        try:
            metrics = store.read_metrics(revision_id)
            tracking = store.read_tracking_alignment(revision_id)
        except Exception:
            continue
        gates = metrics.get("gates") if isinstance(metrics, dict) else {}
        if not all((gates or {}).get(key) is True for key in required_gates):
            continue
        out[camera_id] = {
            "revision_id": revision_id,
            "metrics": metrics,
            "tracking_alignment": tracking,
        }
    return out


def _menon_scene_point(
    world: list[float],
    camera_id: str,
    *,
    scene_matrix: np.ndarray | None,
    room_bounds: dict[str, float] | None,
    vt_alignment: dict[str, Any] | None,
) -> tuple[list[float] | None, str, dict[str, Any] | None]:
    base_scene = _transform_point(scene_matrix, world) if scene_matrix is not None else None
    if vt_alignment and room_bounds:
        tracking = vt_alignment.get("tracking_alignment") or {}
        footprint = tracking.get("valid_tracking_footprint") or {}
        mn = footprint.get("min")
        mx = footprint.get("max")
        if isinstance(mn, list) and isinstance(mx, list) and len(mn) >= 2 and len(mx) >= 2:
            try:
                source_min_x, source_max_x = sorted((float(mn[0]), float(mx[0])))
                source_min_z, source_max_z = sorted((float(mn[1]), float(mx[1])))
                source_w = source_max_x - source_min_x
                source_d = source_max_z - source_min_z
                target_w = float(room_bounds["maxX"]) - float(room_bounds["minX"])
                target_d = float(room_bounds["maxZ"]) - float(room_bounds["minZ"])
                if source_w > 1e-6 and source_d > 1e-6 and target_w > 1e-6 and target_d > 1e-6:
                    sx = target_w / source_w
                    sz = target_d / source_d
                    tx = float(room_bounds["minX"]) - (source_min_x * sx)
                    tz = float(room_bounds["minZ"]) - (source_min_z * sz)
                    point = [float(world[0]) * sx + tx, float(room_bounds["floorY"]), float(world[2]) * sz + tz]
                    return point, "virtual_twin_footprint_to_room_bounds", {
                        "revision_id": vt_alignment.get("revision_id"),
                        "scale": {"x": sx, "z": sz},
                        "translation": {"x": tx, "z": tz},
                        "base_scene_similarity": base_scene,
                    }
            except Exception:
                pass
    return base_scene, "noesis_backend_world_scene_similarity", None


async def _collect(uri: str, duration_s: float) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    async with websockets.connect(uri, max_size=None) as ws:
        end = time.time() + max(1.0, float(duration_s))
        while time.time() < end:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
            if isinstance(raw, (bytes, bytearray)):
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            if not isinstance(payload, dict) or payload.get("type") != "tracking":
                continue
            now = time.time()
            tracks = payload.get("tracks") if isinstance(payload.get("tracks"), list) else []
            for track in tracks:
                if not isinstance(track, dict):
                    continue
                row = dict(track)
                row["_message_ts"] = now
                row["_source_id"] = payload.get("source_id")
                row["_top_camera_id"] = payload.get("camera_id")
                row["_top_world_source"] = payload.get("world_source")
                row["_top_world_frame"] = payload.get("world_frame")
                samples.append(row)
    return samples


def _track_identity(sample: dict[str, Any]) -> str:
    stable = sample.get("stable_id")
    tracker = sample.get("tracker_id", sample.get("track_id"))
    camera = sample.get("camera_id", sample.get("_top_camera_id", "unknown"))
    if stable not in (None, "", -1):
        return f"stable:{stable}"
    return f"{camera}:tracker:{tracker}"


def _analyze(samples: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    pipeline_cfg = _load_yaml(Path(args.pipeline_config), {}) or {}
    streammux = pipeline_cfg.get("streammux") if isinstance(pipeline_cfg, dict) else {}
    streammux_size = (
        int((streammux or {}).get("width", 1920) or 1920),
        int((streammux or {}).get("height", 1080) or 1080),
    )
    calibration = CalibrationManager(
        cameras_yaml_path=Path(args.cameras_config),
        camera_calibration_json_path=Path(args.camera_calibration),
        ply_alignment_json_path=Path(args.ply_alignment),
        streammux_size=streammux_size,
    )
    scene_matrix = _load_scene_matrix(Path(args.ply_alignment))
    room_zones = _load_json(Path(args.room_zones), []) or []
    if not isinstance(room_zones, list):
        room_zones = []
    vt_alignments = _latest_vt_alignments(Path(args.virtual_twin_root))

    required_missing = 0
    forbidden_present = 0
    invalid_world = 0
    wrong_frame = 0
    wrong_source = 0
    valid_rows: list[dict[str, Any]] = []
    by_camera: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for sample in samples:
        if REQUIRED_TRACK_FIELDS.difference(sample.keys()):
            required_missing += 1
        if FORBIDDEN_PUBLIC_V3DT_FIELDS.intersection(sample.keys()):
            forbidden_present += 1
        if sample.get("world_frame") != EXPECTED_WORLD_FRAME:
            wrong_frame += 1
        if sample.get("world_source") != EXPECTED_WORLD_SOURCE:
            wrong_source += 1
        world = sample.get("world")
        if not (
            sample.get("world_valid") is True
            and isinstance(world, list)
            and len(world) >= 3
            and all(math.isfinite(float(v)) for v in world[:3])
        ):
            invalid_world += 1
            continue
        row = dict(sample)
        row["world"] = [float(world[0]), float(world[1]), float(world[2])]
        camera_id = str(row.get("camera_id") or row.get("_top_camera_id") or "unknown")
        row["camera_id"] = camera_id
        valid_rows.append(row)
        by_camera[camera_id].append(row)

    summary: dict[str, Any] = {
        "status": "ok",
        "sample_count": len(samples),
        "valid_world_sample_count": len(valid_rows),
        "payload_validation": {
            "required_missing": required_missing,
            "forbidden_public_v3dt_fields_present": forbidden_present,
            "invalid_world": invalid_world,
            "wrong_world_frame": wrong_frame,
            "wrong_world_source": wrong_source,
        },
        "cameras": {},
        "virtual_twin_alignment_cameras": sorted(vt_alignments.keys()),
    }

    for camera_id, rows in sorted(by_camera.items()):
        floor_errors = [abs(float(row["world"][1]) - float(args.floor_y_m)) for row in rows]
        reproj_norm_errors = []
        reproj_px_errors = []
        direction_cosines = []
        scene_hits = []
        scene_floor_errors = []
        menon_transform_sources = defaultdict(int)
        reprojection_target_sources = defaultdict(int)
        sample_points = []
        room_bounds = _room_union_bounds(room_zones, camera_id)
        vt_alignment = vt_alignments.get(camera_id)
        source_id = rows[0].get("_source_id")
        try:
            source_id_int = int(source_id)
        except Exception:
            source_id_int = {"living-room": 0, "kitchen": 1, "family-room": 2}.get(camera_id, 0)
        try:
            snapshot = calibration.snapshot(source_id_int, camera_id)
        except Exception:
            snapshot = None

        previous_by_id: dict[str, tuple[list[float], list[float], float]] = {}
        step_dists = []
        for row in rows:
            world = row["world"]
            bbox = row.get("bbox")
            image_base = row.get("image_base")
            target = None
            target_source = None
            bbox_floor_proxy = None
            reproj = None
            if snapshot is not None:
                reproj_tuple = project_world_to_image(
                    np.asarray(world, dtype=np.float64),
                    snapshot.intrinsics,
                    snapshot.extrinsics_col_major,
                    tuple(int(v) for v in snapshot.image_size),
                    unit_scale=1.0,
                    flip_u=False,
                    flip_v=False,
                )
                if reproj_tuple is not None:
                    reproj = [float(reproj_tuple[0]), float(reproj_tuple[1])]
            if isinstance(image_base, list) and len(image_base) >= 2:
                try:
                    target = [float(image_base[0]), float(image_base[1])]
                    if all(math.isfinite(value) for value in target):
                        target_source = "image_base"
                    else:
                        target = None
                except Exception:
                    target = None
            if isinstance(bbox, list) and len(bbox) >= 4:
                try:
                    left, top, width, height = [float(v) for v in bbox[:4]]
                    if width > 0.0 and height > 0.0:
                        bbox_floor_proxy = [left + width * 0.5, top + height]
                        if target is None:
                            target = list(bbox_floor_proxy)
                            target_source = "bbox_bottom_proxy"
                        if reproj is not None and target is not None:
                            if target_source:
                                reprojection_target_sources[str(target_source)] += 1
                            err = math.hypot(reproj[0] - target[0], reproj[1] - target[1])
                            reproj_px_errors.append(err)
                            reproj_norm_errors.append(err / max(1.0, height))
                except Exception:
                    bbox_floor_proxy = None
            menon_point, transform_source, transform_detail = _menon_scene_point(
                world,
                camera_id,
                scene_matrix=scene_matrix,
                room_bounds=room_bounds,
                vt_alignment=vt_alignment,
            )
            menon_transform_sources[transform_source] += 1
            scene_hits.append(_inside_room(menon_point, room_bounds, margin=float(args.scene_room_margin)))
            if menon_point is not None and room_bounds is not None:
                scene_floor_errors.append(abs(float(menon_point[1]) - float(room_bounds["floorY"])))
            identity = _track_identity(row)
            prev = previous_by_id.get(identity)
            if prev is not None:
                prev_world, prev_target, prev_ts = prev
                dt = max(1e-3, float(row.get("_message_ts", 0.0)) - float(prev_ts))
                step_dists.append(float(np.linalg.norm(np.asarray(world) - np.asarray(prev_world))))
                if target is not None and reproj is not None and prev_target:
                    video_delta = np.asarray(target) - np.asarray(prev_target)
                    reproj_prev = project_world_to_image(
                        np.asarray(prev_world, dtype=np.float64),
                        snapshot.intrinsics,
                        snapshot.extrinsics_col_major,
                        tuple(int(v) for v in snapshot.image_size),
                        unit_scale=1.0,
                        flip_u=False,
                        flip_v=False,
                    ) if snapshot is not None else None
                    if reproj_prev is not None:
                        world_delta = np.asarray(reproj) - np.asarray(reproj_prev)
                        nv = float(np.linalg.norm(video_delta))
                        nw = float(np.linalg.norm(world_delta))
                        if nv >= 2.0 and nw >= 2.0:
                            direction_cosines.append(float(np.dot(video_delta, world_delta) / (nv * nw)))
                _ = dt
            previous_by_id[identity] = (world, target or [], float(row.get("_message_ts", 0.0)))
            if len(sample_points) < 5:
                sample_points.append(
                    {
                        "stable_id": row.get("stable_id"),
                        "tracker_id": row.get("tracker_id"),
                        "world": world,
                        "menon_scene": menon_point,
                        "menon_transform_source": transform_source,
                        "menon_transform_detail": transform_detail,
                        "reprojected_pixel": reproj,
                        "reprojection_target_pixel": target,
                        "reprojection_target_source": target_source,
                        "bbox_floor_proxy_pixel": bbox_floor_proxy,
                        "image_base": image_base,
                        "world_source": row.get("world_source"),
                    }
                )

        summary["cameras"][camera_id] = {
            "sample_count": len(rows),
            "floor_error_m": _stats(floor_errors),
            "step_distance_m": _stats(step_dists),
            "reprojection_error_px_to_target": _stats(reproj_px_errors),
            "reprojection_error_px_to_bbox_base_proxy": _stats(reproj_px_errors),
            "reprojection_error_norm_bbox_height": _stats(reproj_norm_errors),
            "reprojection_target_sources": dict(sorted(reprojection_target_sources.items())),
            "reprojection_good_ratio_lt_0_10": float(np.mean(np.asarray(reproj_norm_errors) < 0.10)) if reproj_norm_errors else None,
            "reprojection_acceptable_ratio_lt_0_20": float(np.mean(np.asarray(reproj_norm_errors) < 0.20)) if reproj_norm_errors else None,
            "movement_direction_cosine": _stats(direction_cosines),
            "menon_scene_room_hit_ratio": float(np.mean(scene_hits)) if scene_hits else None,
            "menon_scene_floor_error_units": _stats(scene_floor_errors),
            "menon_transform_sources": dict(sorted(menon_transform_sources.items())),
            "room_bounds": room_bounds,
            "sample_points": sample_points,
        }

    cross_rows = []
    by_stable: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid_rows:
        stable = row.get("stable_id")
        if stable not in (None, "", -1):
            by_stable[str(stable)].append(row)
    for stable, rows in by_stable.items():
        rows = sorted(rows, key=lambda item: float(item.get("_message_ts", 0.0)))
        for a in rows:
            for b in rows:
                if a is b or a.get("camera_id") >= b.get("camera_id"):
                    continue
                if abs(float(a.get("_message_ts", 0.0)) - float(b.get("_message_ts", 0.0))) > float(args.cross_camera_window_s):
                    continue
                dist = float(np.linalg.norm(np.asarray(a["world"]) - np.asarray(b["world"])))
                cross_rows.append({"stable_id": stable, "a": a.get("camera_id"), "b": b.get("camera_id"), "distance_m": dist})
                break
    summary["cross_camera"] = {
        "pair_count": len(cross_rows),
        "distance_m": _stats(row["distance_m"] for row in cross_rows),
        "samples": cross_rows[:20],
    }

    failed = []
    payload = summary["payload_validation"]
    if any(int(payload[key]) for key in payload):
        failed.append("payload_shape")
    for camera_id, cam in summary["cameras"].items():
        floor_p95 = cam["floor_error_m"]["p95"]
        if floor_p95 is not None and floor_p95 > float(args.max_floor_error_m):
            failed.append(f"{camera_id}:floor")
        hit_ratio = cam["menon_scene_room_hit_ratio"]
        if hit_ratio is not None and hit_ratio < float(args.min_room_hit_ratio):
            failed.append(f"{camera_id}:room")
        direction_p50 = cam["movement_direction_cosine"]["median"]
        if direction_p50 is not None and direction_p50 < float(args.min_direction_cosine):
            failed.append(f"{camera_id}:direction")
    if failed:
        summary["status"] = "fail"
        summary["failed_gates"] = failed
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ws", default="ws://127.0.0.1:6008")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--pipeline-config", default="config/infer_v3dt_reimpl_fast1056_mp4.yaml")
    parser.add_argument("--cameras-config", default="config/cameras.yaml")
    parser.add_argument("--camera-calibration", default="config/camera_calibration.json")
    parser.add_argument("--ply-alignment", default="config/ply_alignment.json")
    parser.add_argument("--room-zones", default="../Menon/public/config/room-zones.json")
    parser.add_argument("--virtual-twin-root", default="data/virtual_twin")
    parser.add_argument("--out-dir", default="diagnostics/v3dt_menon_world")
    parser.add_argument("--floor-y-m", type=float, default=0.0)
    parser.add_argument("--max-floor-error-m", type=float, default=0.15)
    parser.add_argument("--min-room-hit-ratio", type=float, default=0.80)
    parser.add_argument("--scene-room-margin", type=float, default=75.0)
    parser.add_argument("--min-direction-cosine", type=float, default=-0.10)
    parser.add_argument("--cross-camera-window-s", type=float, default=0.25)
    args = parser.parse_args()

    samples = asyncio.run(_collect(str(args.ws), float(args.duration)))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    samples_path = out_dir / f"samples_{stamp}.jsonl"
    with samples_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, sort_keys=True) + "\n")
    summary = _analyze(samples, args)
    summary["samples_path"] = str(samples_path)
    summary_path = out_dir / f"summary_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[INFO] wrote {samples_path}")
    print(f"[INFO] wrote {summary_path}")
    return 0 if summary.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
