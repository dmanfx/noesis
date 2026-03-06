#!/usr/bin/env python3
"""Smoke checks for strict Menon pose calibration validity.

Checks:
- every configured camera has PoseV1 and derived E
- stored E matches pose-derived E
- camera center is above floor_y
- camera forward ray intersects the floor in front of the camera
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calibration_bundle import load_alignment, load_extrinsics, pose_to_E_col_major


def _camera_names_from_yaml(path: Path) -> List[str]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    cameras = data.get("cameras") if isinstance(data, dict) else None
    if not isinstance(cameras, dict):
        return []
    names: List[str] = []
    for entry in cameras.values():
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return sorted(set(names))


def _compare_e(a: List[float], b: List[float], tol: float = 1e-6) -> bool:
    if not (isinstance(a, list) and isinstance(b, list) and len(a) == 16 and len(b) == 16):
        return False
    for left, right in zip(a, b):
        try:
            if abs(float(left) - float(right)) > tol:
                return False
        except Exception:
            return False
    return True


def _check_floor_geometry(E_col_major: List[float], floor_y: float) -> str:
    try:
        Emat = np.array(E_col_major, dtype=np.float64).reshape((4, 4), order="F")
        Twc = np.linalg.inv(Emat)
        C_world = Twc[:3, 3].copy()
        if not np.all(np.isfinite(C_world)):
            return "camera_center_non_finite"
        if float(C_world[1]) <= float(floor_y) + 1e-3:
            return f"camera_not_above_floor(Cy={float(C_world[1]):.6f},floor={float(floor_y):.6f})"

        R_wc = Twc[:3, :3].copy()
        forward = R_wc @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        denom = float(forward[1])
        if abs(denom) < 1e-6:
            return "forward_parallel_to_floor"
        t_hit = (float(floor_y) - float(C_world[1])) / denom
        if not math.isfinite(t_hit):
            return "forward_floor_intersection_non_finite"
        if t_hit <= 0.0:
            return f"forward_misses_floor(t={float(t_hit):.6f})"
    except Exception as exc:
        return f"floor_geometry_exception({exc})"
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate pose-only Menon calibration contract.")
    parser.add_argument(
        "--cameras-config",
        default="config/cameras.yaml",
        help="Path to cameras.yaml (used to resolve expected camera names).",
    )
    parser.add_argument(
        "--calibration-json",
        default="config/camera_calibration.json",
        help="Path to camera_calibration.json.",
    )
    parser.add_argument(
        "--alignment-json",
        default="config/ply_alignment.json",
        help="Path to ply_alignment.json.",
    )
    args = parser.parse_args()

    cameras_path = Path(args.cameras_config)
    calibration_path = Path(args.calibration_json)
    alignment_path = Path(args.alignment_json)

    camera_names = _camera_names_from_yaml(cameras_path)
    if not camera_names:
        print(f"[FAIL] no camera names found in {cameras_path}")
        return 1

    extrinsics = load_extrinsics(str(calibration_path))
    cameras_node: Dict[str, object] = extrinsics.get("cameras", {}) if isinstance(extrinsics, dict) else {}
    align = load_alignment(str(alignment_path))
    try:
        floor_y = float((align or {}).get("floor_y", 0.0) or 0.0)
    except Exception:
        floor_y = 0.0

    failures: List[str] = []
    for camera_id in camera_names:
        entry = cameras_node.get(camera_id) if isinstance(cameras_node, dict) else None
        if not isinstance(entry, dict):
            failures.append(f"{camera_id}:missing_camera_entry")
            continue

        pose = entry.get("pose")
        if not isinstance(pose, dict):
            failures.append(f"{camera_id}:missing_pose")
            continue

        E_stored = entry.get("E")
        if not (isinstance(E_stored, list) and len(E_stored) == 16):
            failures.append(f"{camera_id}:missing_or_invalid_E")
            continue

        E_from_pose = pose_to_E_col_major(pose)
        if not (isinstance(E_from_pose, list) and len(E_from_pose) == 16):
            failures.append(f"{camera_id}:pose_to_E_failed")
            continue
        if not _compare_e(E_stored, E_from_pose, tol=1e-6):
            failures.append(f"{camera_id}:stored_E_mismatch_pose")
            continue

        floor_error = _check_floor_geometry(E_stored, floor_y)
        if floor_error:
            failures.append(f"{camera_id}:{floor_error}")
            continue

        print(f"[PASS] {camera_id}: pose/E/floor geometry valid")

    if failures:
        print("[FAIL] strict pose calibration validation failed:")
        for item in failures:
            print(f"- {item}")
        return 1

    print(
        f"[PASS] validated {len(camera_names)} configured cameras "
        f"against pose-only contract (floor_y={floor_y:.6f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
