#!/usr/bin/env python3
"""Report and optionally sync Menon-exported camera poses into Noesis calibration.

Reads Menon's `/api/export` payload, compares `camera_poses` against a target
calibration JSON, and optionally updates per-camera `pose` + derived `E`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis.calibration.bundle import pose_to_E_col_major


def _read_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return data


def _fetch_export(url: str, timeout_s: float) -> Dict[str, Any]:
    try:
        with urlopen(url, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except URLError as exc:
        raise RuntimeError(f"failed to fetch Menon export from {url}: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"invalid JSON from {url}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Menon export at {url} is not a JSON object")
    return payload


def _norm_angle_deg(delta: float) -> float:
    out = float(delta)
    while out > 180.0:
        out -= 360.0
    while out < -180.0:
        out += 360.0
    return out


def _normalize_pose(raw: Any, *, default_source: str) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    position = raw.get("position")
    ypr = raw.get("yaw_pitch_roll_deg")
    rotation_order = str(raw.get("rotation_order") or "").strip().upper()
    frame = str(raw.get("frame") or "").strip()
    if not (isinstance(position, list) and len(position) == 3):
        return None
    if not (isinstance(ypr, list) and len(ypr) == 3):
        return None
    try:
        p = [float(position[0]), float(position[1]), float(position[2])]
        r = [float(ypr[0]), float(ypr[1]), float(ypr[2])]
    except Exception:
        return None
    if not all(math.isfinite(v) for v in (p + r)):
        return None
    if rotation_order != "YXZ":
        return None
    if frame != "menon_scene":
        return None
    source = str(raw.get("source") or "").strip() or default_source
    return {
        "position": p,
        "yaw_pitch_roll_deg": r,
        "rotation_order": "YXZ",
        "frame": "menon_scene",
        "source": source,
    }


@dataclass
class CameraDelta:
    camera_id: str
    pos_delta: List[float]
    pos_dist: float
    ypr_delta: List[float]


def _camera_deltas(
    menon_poses: Dict[str, Dict[str, Any]],
    calib_cameras: Dict[str, Dict[str, Any]],
) -> List[CameraDelta]:
    out: List[CameraDelta] = []
    for camera_id in sorted(set(menon_poses.keys()) & set(calib_cameras.keys())):
        pose_new = menon_poses[camera_id]
        pose_old = calib_cameras[camera_id].get("pose")
        pose_old_norm = _normalize_pose(pose_old, default_source="existing_pose")
        if pose_old_norm is None:
            out.append(
                CameraDelta(
                    camera_id=camera_id,
                    pos_delta=[float("nan"), float("nan"), float("nan")],
                    pos_dist=float("nan"),
                    ypr_delta=[float("nan"), float("nan"), float("nan")],
                )
            )
            continue
        dp = [pose_new["position"][i] - pose_old_norm["position"][i] for i in range(3)]
        dypr = [_norm_angle_deg(pose_new["yaw_pitch_roll_deg"][i] - pose_old_norm["yaw_pitch_roll_deg"][i]) for i in range(3)]
        out.append(
            CameraDelta(
                camera_id=camera_id,
                pos_delta=dp,
                pos_dist=math.sqrt(dp[0] ** 2 + dp[1] ** 2 + dp[2] ** 2),
                ypr_delta=dypr,
            )
        )
    return out


def _format_d(v: float) -> str:
    if not math.isfinite(v):
        return "nan"
    return f"{v:.3f}"


def _print_report(
    deltas: List[CameraDelta],
    *,
    menon_only: List[str],
    calib_only: List[str],
) -> None:
    print("[report] Menon pose vs calibration pose deltas")
    if not deltas:
        print("  (no overlapping camera IDs)")
    for item in deltas:
        print(
            f"  - {item.camera_id}: "
            f"pos_dist={_format_d(item.pos_dist)} "
            f"dp=[{_format_d(item.pos_delta[0])}, {_format_d(item.pos_delta[1])}, {_format_d(item.pos_delta[2])}] "
            f"dypr=[{_format_d(item.ypr_delta[0])}, {_format_d(item.ypr_delta[1])}, {_format_d(item.ypr_delta[2])}]"
        )
    if menon_only:
        print(f"  - Menon-only cameras: {', '.join(sorted(menon_only))}")
    if calib_only:
        print(f"  - Calibration-only cameras: {', '.join(sorted(calib_only))}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Report/sync Menon-exported poses to Noesis calibration.")
    parser.add_argument(
        "--export-url",
        default="http://127.0.0.1:3001/api/export",
        help="Menon export endpoint.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=3.0,
        help="HTTP timeout for export fetch.",
    )
    parser.add_argument(
        "--calibration-json",
        default="config/camera_calibration_menon_obj.json",
        help="Target calibration JSON to compare/sync.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write Menon poses + derived E back to the target calibration JSON.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional output path when --write is set (default: overwrite --calibration-json).",
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Return non-zero when any overlapping camera exceeds thresholds.",
    )
    parser.add_argument(
        "--max-pos-delta",
        type=float,
        default=1.0,
        help="Mismatch threshold (position distance in scene units) for --fail-on-mismatch.",
    )
    parser.add_argument(
        "--max-angle-delta",
        type=float,
        default=3.0,
        help="Mismatch threshold (absolute yaw/pitch/roll delta in degrees) for --fail-on-mismatch.",
    )
    args = parser.parse_args()

    calibration_path = Path(args.calibration_json)
    if not calibration_path.is_absolute():
        calibration_path = REPO_ROOT / calibration_path
    if not calibration_path.exists():
        print(f"[FAIL] calibration JSON not found: {calibration_path}")
        return 1

    export = _fetch_export(args.export_url, timeout_s=float(args.timeout_s))
    raw_camera_poses = export.get("camera_poses")
    if not isinstance(raw_camera_poses, dict):
        print(f"[FAIL] export payload missing camera_poses at {args.export_url}")
        return 1

    menon_poses: Dict[str, Dict[str, Any]] = {}
    invalid_ids: List[str] = []
    for camera_id, raw_pose in raw_camera_poses.items():
        normalized = _normalize_pose(raw_pose, default_source="menon_export")
        if normalized is None:
            invalid_ids.append(str(camera_id))
            continue
        menon_poses[str(camera_id)] = normalized
    if invalid_ids:
        print(f"[WARN] ignored invalid Menon poses for camera IDs: {', '.join(sorted(invalid_ids))}")

    calibration = _read_json(calibration_path)
    cameras = calibration.get("cameras")
    if not isinstance(cameras, dict):
        print(f"[FAIL] calibration JSON missing cameras object: {calibration_path}")
        return 1

    calib_cameras: Dict[str, Dict[str, Any]] = {}
    for camera_id, entry in cameras.items():
        if isinstance(entry, dict):
            calib_cameras[str(camera_id)] = entry

    deltas = _camera_deltas(menon_poses, calib_cameras)
    menon_only = [camera_id for camera_id in menon_poses.keys() if camera_id not in calib_cameras]
    calib_only = [camera_id for camera_id in calib_cameras.keys() if camera_id not in menon_poses]
    _print_report(deltas, menon_only=menon_only, calib_only=calib_only)

    mismatches = 0
    if args.fail_on_mismatch:
        pos_th = float(args.max_pos_delta)
        ang_th = float(args.max_angle_delta)
        for item in deltas:
            if not math.isfinite(item.pos_dist):
                mismatches += 1
                continue
            if item.pos_dist > pos_th or any(abs(v) > ang_th for v in item.ypr_delta if math.isfinite(v)):
                mismatches += 1

    if args.write:
        target_path = Path(args.output_json).expanduser() if str(args.output_json).strip() else calibration_path
        if not target_path.is_absolute():
            target_path = REPO_ROOT / target_path
        updated = 0
        for camera_id, pose in menon_poses.items():
            entry = calib_cameras.get(camera_id)
            if entry is None:
                entry = {}
                cameras[camera_id] = entry
                calib_cameras[camera_id] = entry
            entry["pose"] = dict(pose)
            E = pose_to_E_col_major(entry["pose"])
            if not (isinstance(E, list) and len(E) == 16):
                print(f"[FAIL] pose_to_E failed for camera={camera_id}")
                return 1
            entry["E"] = [float(x) for x in E]
            updated += 1
        target_path.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
        print(f"[PASS] wrote {updated} camera pose/E entries -> {target_path}")

    if args.fail_on_mismatch and mismatches > 0:
        print(f"[FAIL] mismatch threshold exceeded for {mismatches} camera(s)")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
