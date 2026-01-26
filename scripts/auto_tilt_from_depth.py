#!/usr/bin/env python3
"""Preview tilt-only calibration updates from the latest MapAnything depth cache.

This fits a ground plane per camera, adjusts pitch/roll (preserves yaw and camera
center), and writes a separate preview calibration file so production calibration
is not overwritten.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from noesis.calibration.tilt_preview import tilt_preview_from_latest_depth


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate tilt-only calibration preview from depth.")
    parser.add_argument("--camera", action="append", help="Camera ID(s) to process (defaults to all)")
    parser.add_argument(
        "--input",
        default="config/camera_calibration.json",
        help="Base calibration file (world->camera, column-major)",
    )
    parser.add_argument(
        "--output",
        default="config/camera_calibration_preview.json",
        help="Preview calibration output (will not overwrite input)",
    )
    parser.add_argument(
        "--cameras-config",
        default=None,
        help="Optional cameras.yaml to read intrinsics from (defaults to config/cameras.yaml)",
    )
    parser.add_argument("--min-conf", type=float, default=0.3, help="Minimum depth confidence")
    parser.add_argument("--max-points", type=int, default=200_000, help="Max points for plane fit")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for plane sampling")
    parser.add_argument(
        "--normal-sign",
        action="append",
        default=[],
        help="Override plane-normal sign per camera, e.g. living-room=-1 (repeatable)",
    )
    parser.add_argument(
        "--allow-yaw",
        action="store_true",
        help="Allow yaw correction (default: preserve yaw)",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow output to overwrite the input calibration file",
    )
    parser.add_argument(
        "--no-write",
        dest="persist",
        action="store_false",
        help="Do not write the preview file (print summary only)",
    )
    parser.add_argument(
        "--flip-image-y",
        action="store_true",
        help="Flip image-space Y when fitting the ground plane (aligns depth with Y-up world)",
    )
    parser.set_defaults(persist=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if args.persist and not args.allow_overwrite:
        try:
            if input_path.resolve() == output_path.resolve():
                print("Refusing to overwrite input calibration file without --allow-overwrite", file=sys.stderr)
                return 2
        except Exception:
            pass

    normal_signs = {}
    for item in args.normal_sign or []:
        if not item:
            continue
        if "=" in item:
            cam, raw = item.split("=", 1)
        elif ":" in item:
            cam, raw = item.split(":", 1)
        else:
            print(f"Invalid --normal-sign '{item}', expected camera=+1 or camera=-1", file=sys.stderr)
            return 2
        cam = cam.strip()
        raw = raw.strip()
        try:
            val = float(raw)
        except Exception:
            print(f"Invalid --normal-sign value '{item}', expected numeric sign", file=sys.stderr)
            return 2
        if not cam:
            print(f"Invalid --normal-sign '{item}', missing camera id", file=sys.stderr)
            return 2
        normal_signs[cam] = val

    summary = tilt_preview_from_latest_depth(
        args.camera,
        input_path=input_path,
        output_path=output_path,
        cameras_path=Path(args.cameras_config) if args.cameras_config else None,
        persist=args.persist,
        min_conf=args.min_conf,
        max_points=args.max_points,
        seed=args.seed,
        preserve_yaw=not args.allow_yaw,
        normal_signs=normal_signs or None,
        flip_image_y=args.flip_image_y,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
