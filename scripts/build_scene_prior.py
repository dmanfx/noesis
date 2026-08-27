#!/usr/bin/env python3
"""Build and catalog one generic Noesis scene-prior revision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis.scene_prior_builder import (  # noqa: E402
    ScenePriorBuildConfig,
    ScenePriorBuildError,
    build_scene_prior,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Derive an immutable, backend-world scene prior from one verified "
            "room-walk bundle and bind it to zero or more cameras in shadow mode."
        )
    )
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--space-id", required=True)
    parser.add_argument(
        "--semantic-room",
        action="append",
        required=True,
        dest="semantic_rooms",
        help="Exact room label from the authored room-group map; repeat for a multi-room space.",
    )
    parser.add_argument("--authored-scene", type=Path, required=True)
    parser.add_argument("--room-group-map", type=Path, required=True)
    parser.add_argument("--world-to-scene", type=Path, required=True)
    parser.add_argument(
        "--camera-map-lock",
        type=Path,
        help=(
            "Optional revision-bound static-camera yaw residual to compose at "
            "the camera-to-PCF boundary."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data" / "scene_priors",
    )
    parser.add_argument(
        "--bind-camera",
        action="append",
        default=[],
        dest="camera_ids",
        help="Camera ID that occupies this physical space; repeat as needed.",
    )
    parser.add_argument("--grid-resolution-m", type=float, default=0.05)
    parser.add_argument("--floor-support-band-m", type=float, default=0.12)
    parser.add_argument("--obstacle-min-height-m", type=float, default=0.18)
    parser.add_argument("--obstacle-max-height-m", type=float, default=2.20)
    parser.add_argument("--obstacle-min-support", type=int, default=3)
    parser.add_argument("--max-source-height-m", type=float, default=3.20)
    parser.add_argument(
        "--no-floorplan-layers",
        action="store_true",
        help="Keep track diagnostics but do not add static/composite floorplan layers.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = ScenePriorBuildConfig(
        source_bundle=args.source_bundle,
        site_id=args.site_id,
        space_id=args.space_id,
        semantic_rooms=tuple(args.semantic_rooms),
        authored_scene=args.authored_scene,
        room_group_map=args.room_group_map,
        world_to_scene=args.world_to_scene,
        output_root=args.output_root,
        camera_map_lock=args.camera_map_lock,
        camera_ids=tuple(args.camera_ids),
        grid_resolution_m=args.grid_resolution_m,
        floor_support_band_m=args.floor_support_band_m,
        obstacle_min_height_m=args.obstacle_min_height_m,
        obstacle_max_height_m=args.obstacle_max_height_m,
        obstacle_min_support=args.obstacle_min_support,
        max_source_height_m=args.max_source_height_m,
        include_floorplan_layers=not args.no_floorplan_layers,
    )
    try:
        result = build_scene_prior(config)
    except (ScenePriorBuildError, OSError, ValueError) as exc:
        print(f"scene-prior build failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
