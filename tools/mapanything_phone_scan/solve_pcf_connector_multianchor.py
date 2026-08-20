#!/usr/bin/env python3
"""Solve one connector RoomWalk against two already joined PCF room anchors.

The input registration reports contain depth-backed PnP observations from the
same connector carrier into each accepted room.  This command expresses both
observation sets in the existing fixed-room world, then solves one floor-locked
planar pose graph with a smooth per-view correction field.  It does not use ICP
or nearest-neighbour point-cloud fitting.

Rejected solutions remain useful only as explicitly review-only candidates;
the report never promotes a failed graph to canonical geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.mapanything_phone_scan import pcf_multiroom_pose_graph as graph  # noqa: E402
from tools.mapanything_phone_scan.register_pcf_rooms import (  # noqa: E402
    _manifest_transform,
)


class ConnectorMultiAnchorError(RuntimeError):
    """Raised when the two anchor reports cannot form one connector graph."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConnectorMultiAnchorError(f"{path} does not contain a JSON object")
    return value


def _candidate_transform(report: dict[str, Any], label: str) -> np.ndarray:
    pose_graph = report.get("pose_graph")
    candidate = pose_graph.get("candidate") if isinstance(pose_graph, dict) else None
    value = (
        candidate.get("global_transform_moving_to_fixed_row_major")
        if isinstance(candidate, dict)
        else report.get("global_transform_moving_to_fixed_row_major")
    )
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ConnectorMultiAnchorError(f"{label} has no finite candidate transform")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ConnectorMultiAnchorError(f"{label} candidate has an invalid last row")
    if not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=2e-3):
        raise ConnectorMultiAnchorError(f"{label} candidate is not rigid")
    if not np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=2e-3):
        raise ConnectorMultiAnchorError(f"{label} candidate is reflected")
    return transform


def _room_poses(room: dict[str, Any], output_from_room: np.ndarray) -> dict[int, np.ndarray]:
    raw_root = Path(str(room["pcf_raw_root"]))
    world_manifest = Path(str(room["world_manifest"]))
    raw_paths = sorted(raw_root.glob("view_*.npz"))
    if not raw_paths:
        raise ConnectorMultiAnchorError(f"no PCF views in {raw_root}")
    world_from_local = _manifest_transform(world_manifest)
    poses: dict[int, np.ndarray] = {}
    for index, path in enumerate(raw_paths):
        with np.load(path, allow_pickle=False) as row:
            if "camera_pose" not in row.files:
                raise ConnectorMultiAnchorError(f"{path} has no camera_pose")
            local_pose = np.asarray(row["camera_pose"], dtype=np.float64)
        if local_pose.shape != (4, 4) or not np.isfinite(local_pose).all():
            raise ConnectorMultiAnchorError(f"malformed camera pose in {path}")
        poses[index] = output_from_room @ world_from_local @ local_pose
    return poses


def _observations(
    report: dict[str, Any],
    *,
    output_from_anchor: np.ndarray,
    fixed_view_offset: int,
    anchor_name: str,
) -> list[graph.PnPTransformObservation]:
    rows = report.get("admitted_pnp")
    if not isinstance(rows, list) or not rows:
        raise ConnectorMultiAnchorError(f"{anchor_name} report has no admitted PnP")
    result: list[graph.PnPTransformObservation] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ConnectorMultiAnchorError(
                f"{anchor_name} PnP row {index} is not an object"
            )
        transform = np.asarray(
            row.get("transform_moving_to_fixed_row_major"), dtype=np.float64
        )
        if transform.shape != (4, 4):
            raise ConnectorMultiAnchorError(
                f"{anchor_name} PnP row {index} has no transform"
            )
        result.append(
            graph.PnPTransformObservation(
                moving_view=int(row["moving_view"]),
                fixed_view=fixed_view_offset + int(row["fixed_view"]),
                direction=f"{anchor_name}:{row['direction']}",
                transform_moving_to_fixed=output_from_anchor @ transform,
                inlier_count=int(row["inlier_count"]),
                reprojection_median_px=float(row["reprojection_median_px"]),
                reprojection_p80_px=float(row["reprojection_p80_px"]),
            )
        )
    return result


def solve(
    *,
    kitchen_report_path: Path,
    living_report_path: Path,
    kitchen_family_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    kitchen_report = _load_json(kitchen_report_path)
    living_report = _load_json(living_report_path)
    kitchen_family = _load_json(kitchen_family_manifest_path)
    moving_kitchen = kitchen_report.get("moving_room")
    moving_living = living_report.get("moving_room")
    if not isinstance(moving_kitchen, dict) or not isinstance(moving_living, dict):
        raise ConnectorMultiAnchorError("registration reports lack moving-room records")
    identity_fields = ("pcf_raw_root", "world_manifest", "scan_id", "prior_id")
    mismatches = [
        field
        for field in identity_fields
        if moving_kitchen.get(field) != moving_living.get(field)
    ]
    if mismatches:
        raise ConnectorMultiAnchorError(
            f"anchor reports do not use the same connector carrier: {mismatches}"
        )
    moving_room = moving_kitchen
    kitchen_room = kitchen_report.get("fixed_room")
    living_room = living_report.get("fixed_room")
    if not isinstance(kitchen_room, dict) or not isinstance(living_room, dict):
        raise ConnectorMultiAnchorError("registration reports lack fixed-room records")

    moving_manifest = kitchen_family.get("moving_room")
    if not isinstance(moving_manifest, dict):
        raise ConnectorMultiAnchorError("Kitchen/Family manifest lacks moving_room")
    kitchen_to_family = np.asarray(
        moving_manifest.get("global_world_correction_row_major"), dtype=np.float64
    )
    if kitchen_to_family.shape != (4, 4):
        raise ConnectorMultiAnchorError("Kitchen/Family manifest lacks its transform")

    connector_to_kitchen = _candidate_transform(kitchen_report, "Kitchen anchor")
    connector_to_living = _candidate_transform(living_report, "Living anchor")
    living_to_family = (
        kitchen_to_family @ connector_to_kitchen @ np.linalg.inv(connector_to_living)
    )

    moving_poses = _room_poses(moving_room, np.eye(4, dtype=np.float64))
    kitchen_poses = _room_poses(kitchen_room, kitchen_to_family)
    living_offset = 1_000
    living_poses = {
        living_offset + index: pose
        for index, pose in _room_poses(living_room, living_to_family).items()
    }
    fixed_poses = {**kitchen_poses, **living_poses}
    observations = [
        *_observations(
            kitchen_report,
            output_from_anchor=kitchen_to_family,
            fixed_view_offset=0,
            anchor_name="kitchen",
        ),
        *_observations(
            living_report,
            output_from_anchor=living_to_family,
            fixed_view_offset=living_offset,
            anchor_name="living",
        ),
    ]

    result = graph.solve_cross_session_pose_graph(
        observations,
        moving_poses,
        fixed_poses,
        vertical_translation_m=0.0,
    )
    candidate = result.report.get("candidate")
    if not isinstance(candidate, dict):
        raise ConnectorMultiAnchorError(
            "combined graph had insufficient support and produced no candidate"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema": "noesis.pcf.connector_multianchor_pose_graph.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "passed" if result.accepted else "rejected",
        "accepted_for_canonical_use": bool(result.accepted),
        "disposition": (
            "validated_cross_session_registration"
            if result.accepted
            else "review_only_rejected_registration_candidate"
        ),
        "method": (
            "two_endpoint_depth_pnp_floor_locked_planar_smooth_per_view_pose_graph"
        ),
        "whole_cloud_icp_used": False,
        "nearest_neighbor_surface_fitting_used": False,
        "fixed_world": "family_accepted_backend_world_m",
        "reason_codes": list(result.reason_codes),
        "pose_graph": result.report,
        "candidate": candidate,
        "derived_endpoint_transforms": {
            "connector_to_kitchen_row_major": connector_to_kitchen.tolist(),
            "connector_to_living_row_major": connector_to_living.tolist(),
            "kitchen_to_family_row_major": kitchen_to_family.tolist(),
            "living_to_family_row_major": living_to_family.tolist(),
        },
        "support_input": {
            "connector_view_count": len(moving_poses),
            "kitchen_fixed_view_count": len(kitchen_poses),
            "living_fixed_view_count": len(living_poses),
            "kitchen_pnp_direction_count": len(kitchen_report["admitted_pnp"]),
            "living_pnp_direction_count": len(living_report["admitted_pnp"]),
            "combined_pnp_direction_count": len(observations),
            "living_fixed_view_id_offset": living_offset,
        },
        "sources": {
            "kitchen_registration_report": {
                "path": str(kitchen_report_path),
                "sha256": _sha256(kitchen_report_path),
                "status": kitchen_report.get("status"),
            },
            "living_registration_report": {
                "path": str(living_report_path),
                "sha256": _sha256(living_report_path),
                "status": living_report.get("status"),
            },
            "kitchen_family_manifest": {
                "path": str(kitchen_family_manifest_path),
                "sha256": _sha256(kitchen_family_manifest_path),
                "status": kitchen_family.get("status"),
            },
            "connector": moving_room,
            "kitchen": kitchen_room,
            "living": living_room,
        },
    }
    output_path = output_dir / "registration_report.json"
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kitchen-report", type=Path, required=True)
    parser.add_argument("--living-report", type=Path, required=True)
    parser.add_argument("--kitchen-family-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = solve(
        kitchen_report_path=args.kitchen_report,
        living_report_path=args.living_report,
        kitchen_family_manifest_path=args.kitchen_family_manifest,
        output_dir=args.output_dir,
    )
    metrics = report["candidate"].get("training_observation_error", {})
    print(
        json.dumps(
            {
                "status": report["status"],
                "reason_codes": report["reason_codes"],
                "training_translation_p80_m": metrics.get("translation_p80_m"),
                "training_yaw_p80_deg": metrics.get("yaw_p80_deg"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
