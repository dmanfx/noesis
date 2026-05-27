#!/usr/bin/env python3
"""Validate live tracking samples against a virtual-twin tracking alignment.

This intentionally fails closed when no samples for the target camera are seen:
the virtual-twin tracking correction should not be promoted from shadow/readback
mode without live or replay evidence for that camera.
"""

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

try:
    import websockets  # type: ignore
except Exception as exc:  # pragma: no cover
    print(f"[FAIL] websockets package required: {exc}")
    sys.exit(1)


def _resolve_revision(root: Path, revision: str) -> tuple[str, Path]:
    if revision == "latest":
        latest = (root / "latest").read_text(encoding="utf-8").strip()
        if not latest:
            raise RuntimeError(f"empty latest virtual-twin pointer: {root / 'latest'}")
        revision = latest
    revision_dir = root / "revisions" / revision
    if not revision_dir.is_dir():
        raise RuntimeError(f"virtual-twin revision not found: {revision_dir}")
    return revision, revision_dir


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object at {path}")
    return payload


def _matrix_col_major(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size != 16:
        raise RuntimeError(f"{name} must contain 16 values")
    matrix = arr.reshape((4, 4), order="F")
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError(f"{name} contains non-finite values")
    return matrix


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    return (matrix[:3, :3] @ pts.T).T + matrix[:3, 3]


def _percentile(values: Iterable[float], q: float) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return None
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q))


async def _recv_json(ws: Any, *, timeout_s: float) -> dict[str, Any] | None:
    try:
        msg = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
    except asyncio.TimeoutError:
        return None
    if isinstance(msg, (bytes, bytearray)):
        return None
    try:
        payload = json.loads(msg)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


async def _sample_tracks(uri: str, camera: str, duration_s: float) -> tuple[list[dict[str, Any]], dict[str, int]]:
    samples: list[dict[str, Any]] = []
    seen_by_camera: dict[str, int] = defaultdict(int)
    async with websockets.connect(uri, max_size=None) as ws:
        end_at = time.time() + max(1.0, float(duration_s))
        while time.time() < end_at:
            payload = await _recv_json(ws, timeout_s=2.0)
            if not payload or payload.get("type") != "tracking":
                continue
            tracks = payload.get("tracks")
            if not isinstance(tracks, list):
                continue
            for track in tracks:
                if not isinstance(track, dict):
                    continue
                camera_id = str(track.get("camera_id") or "")
                if camera_id:
                    seen_by_camera[camera_id] += 1
                if camera_id != camera or track.get("world_valid") is not True:
                    continue
                world = track.get("world")
                if isinstance(world, list) and len(world) == 3:
                    try:
                        point = [float(world[0]), float(world[1]), float(world[2])]
                    except Exception:
                        continue
                    if all(math.isfinite(v) for v in point):
                        samples.append(
                            {
                                "world": point,
                                "stable_id": track.get("stable_id"),
                                "tracker_id": track.get("tracker_id", track.get("track_id")),
                                "frame_id": track.get("frame_id"),
                            }
                        )
    return samples, dict(sorted(seen_by_camera.items()))


def _step_stats(samples: list[dict[str, Any]], points: np.ndarray, *, scale_to_m: float = 1.0) -> dict[str, Any]:
    by_id: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        identity = sample.get("stable_id")
        if identity in (None, "", -1):
            identity = sample.get("tracker_id")
        if identity in (None, "", -1):
            continue
        frame = sample.get("frame_id")
        try:
            frame_i = int(frame)
        except Exception:
            frame_i = idx
        by_id[str(identity)].append((frame_i, points[idx]))
    steps: list[float] = []
    for rows in by_id.values():
        rows.sort(key=lambda item: item[0])
        for (_, a), (_, b) in zip(rows, rows[1:]):
            steps.append(float(np.linalg.norm((b - a) * float(scale_to_m))))
    return {
        "track_id_count": len(by_id),
        "step_count": len(steps),
        "median_step_m": _percentile(steps, 50.0),
        "p95_step_m": _percentile(steps, 95.0),
    }


def _bbox_xz(points: np.ndarray) -> dict[str, Any]:
    pts = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    if pts.size == 0:
        return {"min": None, "max": None, "area": 0.0}
    xz = pts[:, [0, 2]]
    mn = np.min(xz, axis=0)
    mx = np.max(xz, axis=0)
    return {
        "min": [float(mn[0]), float(mn[1])],
        "max": [float(mx[0]), float(mx[1])],
        "area": float(max(0.0, mx[0] - mn[0]) * max(0.0, mx[1] - mn[1])),
    }


def _inside_xz_footprint(points: np.ndarray, footprint: dict[str, Any]) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    mn = footprint.get("min")
    mx = footprint.get("max")
    if not (isinstance(mn, list) and isinstance(mx, list) and len(mn) == 2 and len(mx) == 2):
        return np.zeros((pts.shape[0],), dtype=bool)
    min_x, min_z = float(mn[0]), float(mn[1])
    max_x, max_z = float(mx[0]), float(mx[1])
    return (pts[:, 0] >= min_x) & (pts[:, 0] <= max_x) & (pts[:, 2] >= min_z) & (pts[:, 2] <= max_z)


def _inside_scene_bounds(points_scene: np.ndarray, leakage: dict[str, Any]) -> np.ndarray:
    pts = np.asarray(points_scene, dtype=np.float64).reshape((-1, 3))
    mn = leakage.get("bounds_min")
    mx = leakage.get("bounds_max")
    margin = float(leakage.get("scene_margin_units") or 0.0)
    if not (isinstance(mn, list) and isinstance(mx, list) and len(mn) == 3 and len(mx) == 3):
        return np.zeros((pts.shape[0],), dtype=bool)
    lo = np.asarray(mn, dtype=np.float64) - margin
    hi = np.asarray(mx, dtype=np.float64) + margin
    return np.all((pts >= lo) & (pts <= hi), axis=1)


def _validate(args: argparse.Namespace) -> int:
    root = Path(args.virtual_twin_root)
    revision, revision_dir = _resolve_revision(root, str(args.revision))
    tracking_alignment = _load_json(revision_dir / "tracking_alignment.json")
    metrics = _load_json(revision_dir / "metrics.json")
    camera = str(args.camera or tracking_alignment.get("camera") or metrics.get("camera") or "")
    if not camera:
        raise RuntimeError("target camera was not provided and revision does not declare one")

    pose = tracking_alignment.get("pose_correction") or {}
    final_matrix = _matrix_col_major(pose.get("world_to_menon_scene_col_major"), name="world_to_menon_scene_col_major")
    initial_matrix = _matrix_col_major(
        pose.get("initial_world_to_menon_scene_col_major"),
        name="initial_world_to_menon_scene_col_major",
    )
    scene_to_m = float(pose.get("s_obj_to_m") or 1.0)

    samples, seen_by_camera = asyncio.run(_sample_tracks(str(args.ws), camera, float(args.duration)))
    summary: dict[str, Any] = {
        "revision_id": revision,
        "camera": camera,
        "ws": str(args.ws),
        "duration_s": float(args.duration),
        "seen_by_camera": seen_by_camera,
        "sample_count": len(samples),
    }
    if not samples:
        summary["status"] = "no_samples"
        print(json.dumps(summary, indent=2))
        print(f"[FAIL] no world-valid tracking samples observed for camera={camera!r}")
        return 1

    world_points = np.asarray([row["world"] for row in samples], dtype=np.float64)
    initial_scene = _transform_points(world_points, initial_matrix)
    corrected_scene = _transform_points(world_points, final_matrix)
    correction_delta_scene = np.linalg.norm(corrected_scene - initial_scene, axis=1)
    correction_delta_m = correction_delta_scene * scene_to_m

    floor = tracking_alignment.get("floor_plane") or {}
    floor_y = float(floor.get("floor_y") or 0.0)
    floor_abs_err = np.abs(world_points[:, 1] - floor_y)
    floor_hits = floor_abs_err <= float(args.floor_tolerance_m)

    footprint = tracking_alignment.get("valid_tracking_footprint") or {}
    footprint_hits = _inside_xz_footprint(world_points, footprint)
    scene_hits = _inside_scene_bounds(corrected_scene, metrics.get("room_model_leakage") or {})

    summary.update(
        {
            "status": "ok",
            "floor_tolerance_m": float(args.floor_tolerance_m),
            "valid_floor_hit_ratio": float(np.count_nonzero(floor_hits) / len(floor_hits)),
            "valid_tracking_footprint_hit_ratio": float(np.count_nonzero(footprint_hits) / len(footprint_hits)),
            "wall_leakage_ratio_scene_bounds": float(1.0 - (np.count_nonzero(scene_hits) / len(scene_hits))),
            "correction_delta_m": {
                "median": _percentile(correction_delta_m, 50.0),
                "p90": _percentile(correction_delta_m, 90.0),
                "max": _percentile(correction_delta_m, 100.0),
            },
            "correction_delta_scene_units": {
                "median": _percentile(correction_delta_scene, 50.0),
                "p90": _percentile(correction_delta_scene, 90.0),
                "max": _percentile(correction_delta_scene, 100.0),
            },
            "footprint_before_backend_world_m": _bbox_xz(world_points),
            "footprint_after_corrected_scene_units": _bbox_xz(corrected_scene),
            "stability_before_backend_world_m": _step_stats(samples, world_points, scale_to_m=1.0),
            "stability_after_corrected_scene_m": _step_stats(samples, corrected_scene, scale_to_m=scene_to_m),
            "tracking_alignment_mode": (tracking_alignment.get("image_to_floor_lookup") or {}).get("mode"),
        }
    )
    print(json.dumps(summary, indent=2))

    failed = []
    if summary["valid_floor_hit_ratio"] < float(args.min_floor_hit_ratio):
        failed.append("floor_hit_ratio")
    if summary["valid_tracking_footprint_hit_ratio"] < float(args.min_footprint_hit_ratio):
        failed.append("footprint_hit_ratio")
    if summary["wall_leakage_ratio_scene_bounds"] > float(args.max_wall_leakage_ratio):
        failed.append("wall_leakage_ratio")
    if failed:
        print(f"[FAIL] virtual-twin tracking validation gates failed: {', '.join(failed)}")
        return 1
    print("[PASS] virtual-twin tracking alignment validated in shadow/readback mode")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ws", default="ws://127.0.0.1:6008", help="Noesis tracking WebSocket URL")
    parser.add_argument("--virtual-twin-root", default="data/virtual_twin")
    parser.add_argument("--revision", default="latest")
    parser.add_argument("--camera", default="", help="Target camera id; defaults to the revision camera")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--floor-tolerance-m", type=float, default=0.35)
    parser.add_argument("--min-floor-hit-ratio", type=float, default=0.95)
    parser.add_argument("--min-footprint-hit-ratio", type=float, default=0.90)
    parser.add_argument("--max-wall-leakage-ratio", type=float, default=0.03)
    args = parser.parse_args()
    try:
        return _validate(args)
    except Exception as exc:
        print(f"[FAIL] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
