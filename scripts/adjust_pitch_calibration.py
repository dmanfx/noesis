#!/usr/bin/env python3
"""Adjust camera pitch in a calibration JSON without changing camera position.

This rotates the camera around its local X axis by a delta (degrees) or
adjusts to a target pitch. The output is written to a preview file so
production calibration can remain untouched.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from noesis.calibration.pose_v1 import E_col_major_to_pose_v1
from noesis.calibration.tilt_preview import _camera_pose_from_E, _yaw_pitch_roll


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _rotation_x(deg: float) -> np.ndarray:
    rad = math.radians(float(deg))
    c = math.cos(rad)
    s = math.sin(rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _adjust_pitch(E_col_major: Sequence[float], delta_deg: float) -> Tuple[List[float], float, float]:
    R_wc, C_world = _camera_pose_from_E(E_col_major)
    yaw_before, pitch_before, roll_before = _yaw_pitch_roll(R_wc)

    R_wc_new = R_wc @ _rotation_x(delta_deg)
    yaw_after, pitch_after, roll_after = _yaw_pitch_roll(R_wc_new)

    R_cw_new = R_wc_new.T
    t_cw_new = -R_cw_new @ C_world
    E_new = np.eye(4, dtype=np.float64)
    E_new[:3, :3] = R_cw_new
    E_new[:3, 3] = t_cw_new
    return E_new.flatten(order="F").tolist(), pitch_before, pitch_after


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Adjust calibration pitch for selected cameras.")
    parser.add_argument("--input", default="config/camera_calibration.json", help="Input calibration JSON")
    parser.add_argument("--output", default="config/camera_calibration_preview_pitch.json", help="Output JSON path")
    parser.add_argument("--camera", action="append", help="Camera ID(s) to adjust (repeatable)")
    parser.add_argument("--pitch-delta-deg", type=float, default=None, help="Pitch delta in degrees (local X axis)")
    parser.add_argument("--target-pitch-deg", type=float, default=None, help="Target pitch in degrees")
    args = parser.parse_args()

    if args.pitch_delta_deg is None and args.target_pitch_deg is None:
        print("Provide --pitch-delta-deg or --target-pitch-deg", file=sys.stderr)
        return 2

    input_path = Path(args.input)
    output_path = Path(args.output)
    data = _load_json(input_path)
    cameras = data.get("cameras")
    if not isinstance(cameras, dict):
        print("Invalid calibration format: missing cameras map", file=sys.stderr)
        return 2

    cam_ids = args.camera or list(cameras.keys())
    meta_entries = []

    for cam_id in cam_ids:
        cam_entry = cameras.get(cam_id)
        if not isinstance(cam_entry, dict):
            print(f"Skipping camera '{cam_id}': not found", file=sys.stderr)
            continue
        E = cam_entry.get("E")
        if not isinstance(E, list) or len(E) != 16:
            print(f"Skipping camera '{cam_id}': invalid E matrix", file=sys.stderr)
            continue
        delta = args.pitch_delta_deg
        if delta is None:
            # Pitch adjustment is inverted relative to the target delta.
            # New pitch approx = pitch_before - delta.
            _, pitch_before, _ = _adjust_pitch(E, 0.0)
            delta = float(pitch_before) - float(args.target_pitch_deg)
        E_new, pitch_before, pitch_after = _adjust_pitch(E, float(delta))
        cam_entry["E"] = E_new
        existing_source = (cam_entry.get("pose") or {}).get("source") if isinstance(cam_entry.get("pose"), dict) else None
        pose = E_col_major_to_pose_v1(E_new, source=existing_source or "derived_from_E")
        if pose is not None:
            cam_entry["pose"] = pose
        meta_entries.append({
            "camera": cam_id,
            "pitch_before": float(pitch_before),
            "pitch_after": float(pitch_after),
            "pitch_delta_deg": float(delta),
            "target_pitch_deg": float(args.target_pitch_deg) if args.target_pitch_deg is not None else None,
        })

    data["preview_meta"] = {
        "method": "pitch_adjust",
        "generated_at": _now_stamp(),
        "source_path": str(input_path),
        "cameras": meta_entries,
    }

    _write_json(output_path, data)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
