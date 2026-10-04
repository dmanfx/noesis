#!/usr/bin/env python3
"""Build a scene-only, measured-floor registration for current Noesis frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis.calibration.presentation_registration import (  # noqa: E402
    build_registration,
    load_registration,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--authored-obj", type=Path, required=True)
    parser.add_argument("--parent-authored-obj", type=Path, required=True)
    parser.add_argument("--room-group-map", type=Path, required=True)
    parser.add_argument("--support-measurements", type=Path, required=True)
    parser.add_argument("--support-surfaces", type=Path, required=True)
    parser.add_argument("--upper-room", required=True)
    parser.add_argument("--lower-room", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = build_registration(
        bundle_path=args.bundle,
        authored_obj_path=args.authored_obj,
        parent_authored_obj_path=args.parent_authored_obj,
        room_group_map_path=args.room_group_map,
        support_measurements_path=args.support_measurements,
        support_surfaces_path=args.support_surfaces,
        upper_room=args.upper_room,
        lower_rooms=tuple(args.lower_room),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    load_registration(args.output, authored_obj_path=args.authored_obj)
    print(json.dumps({
        "registration_sha256": artifact["registration_sha256"],
        "authored_scene_sha256": artifact["authored_scene_sha256"],
        "target_revisions": sorted(artifact["presentations"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
