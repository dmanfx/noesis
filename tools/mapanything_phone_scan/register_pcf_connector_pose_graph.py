#!/usr/bin/env python3
"""Register a drifted PCF connector walk to an accepted room PCF.

The command retrieves cross-walk RGB overlap with mutual SIFT, verifies each
direction with PCF-depth-backed PnP, and solves a floor-locked planar pose graph
with a smoothly varying correction for every connector view.  It deliberately
does not run whole-cloud ICP or nearest-neighbour surface alignment.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import pcf_multiroom_pose_graph as pose_graph
import register_pcf_rooms as core


@dataclass
class MatchRow:
    moving_view: int
    fixed_view: int
    retrieval_score: float
    mutual_count: int
    geometric_count: int
    moving_pixels: np.ndarray
    fixed_pixels: np.ndarray


@dataclass
class PnPRow:
    moving_view: int
    fixed_view: int
    direction: str
    transform_moving_to_fixed: np.ndarray
    evidence: core.PnPCorrection
    moving_pixels: np.ndarray
    fixed_pixels: np.ndarray


def _pnp_report(row: PnPRow) -> dict[str, Any]:
    parameters = core._parameters_from_transform(row.transform_moving_to_fixed)
    return {
        "moving_view": row.moving_view,
        "fixed_view": row.fixed_view,
        "direction": row.direction,
        "inlier_count": row.evidence.inlier_count,
        "inlier_fraction": row.evidence.inlier_fraction,
        "reprojection_median_px": row.evidence.reprojection_median_px,
        "reprojection_p80_px": row.evidence.reprojection_p80_px,
        "source_pnp_non_yaw_rotation_deg": row.evidence.non_yaw_rotation_deg,
        "source_3d_support_span_m": row.evidence.support_span_m,
        "yaw_deg": math.degrees(float(parameters[0])),
        "translation_m": parameters[1:].tolist(),
        "transform_moving_to_fixed_row_major": (
            row.transform_moving_to_fixed.tolist()
        ),
    }


def _room_report(room: core.Room) -> dict[str, Any]:
    prepared_manifest = room.scan_dir / "prepared_frames_manifest.json"
    return {
        "name": room.name,
        "prior_id": room.prior_id,
        "scan_id": room.scan_id,
        "view_count": len(room.views),
        "pcf_raw_root": str(room.raw_root),
        "world_manifest": str(room.world_manifest),
        "world_manifest_sha256": core._sha256(room.world_manifest),
        "prepared_manifest": str(prepared_manifest),
        "prepared_manifest_sha256": core._sha256(prepared_manifest),
    }


def _write_match_sheet(
    moving: core.Room,
    fixed: core.Room,
    rows: list[PnPRow],
    path: Path,
) -> None:
    selected = sorted(
        rows,
        key=lambda row: (
            -row.evidence.inlier_count,
            row.evidence.reprojection_median_px,
        ),
    )[:10]
    if not selected:
        return
    tile_width = (
        moving.views[0].feature_rgb.shape[1]
        + fixed.views[0].feature_rgb.shape[1]
    )
    tile_height = max(
        moving.views[0].feature_rgb.shape[0],
        fixed.views[0].feature_rgb.shape[0],
    )
    canvas = np.full(
        (tile_height * len(selected), tile_width, 3), 24, dtype=np.uint8
    )
    palette = [(255, 96, 96), (96, 220, 255), (160, 255, 128), (255, 210, 96)]
    for row_index, row in enumerate(selected):
        moving_rgb = moving.views[row.moving_view].feature_rgb
        fixed_rgb = fixed.views[row.fixed_view].feature_rgb
        top = row_index * tile_height
        canvas[top : top + moving_rgb.shape[0], : moving_rgb.shape[1]] = moving_rgb
        canvas[
            top : top + fixed_rgb.shape[0],
            moving_rgb.shape[1] : moving_rgb.shape[1] + fixed_rgb.shape[1],
        ] = fixed_rgb
        for point_index, (moving_xy, fixed_xy) in enumerate(
            zip(row.moving_pixels[:80], row.fixed_pixels[:80], strict=False)
        ):
            color = palette[point_index % len(palette)]
            start = tuple(int(round(value)) for value in moving_xy)
            end = (
                int(round(float(fixed_xy[0]))) + moving_rgb.shape[1],
                int(round(float(fixed_xy[1]))),
            )
            cv2.line(
                canvas,
                (start[0], top + start[1]),
                (end[0], top + end[1]),
                color,
                1,
                cv2.LINE_AA,
            )
        label = (
            f"{moving.name} {row.moving_view:02d} -> {fixed.name} "
            f"{row.fixed_view:02d} | {row.direction} | "
            f"{row.evidence.inlier_count} PnP inliers | "
            f"{row.evidence.reprojection_median_px:.2f}px median"
        )
        cv2.putText(
            canvas,
            label,
            (8, top + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    if not cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)):
        raise core.PCFRoomRegistrationError(f"could not write {path}")


def run(
    moving: core.Room,
    fixed: core.Room,
    output_dir: Path,
    *,
    vertical_translation_m: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    settings = core.Settings(
        retrieval_pairs=240,
        detailed_features=6_000,
        detailed_ratio=0.78,
    )
    print("[1/4] Retrieving cross-session RGB candidates", flush=True)
    retrieval = core._retrieve_pairs(moving, fixed, settings)
    moving_indices = sorted({row[0] for row in retrieval})
    fixed_indices = sorted({row[1] for row in retrieval})
    moving_features = {
        index: core._sift_features(moving.views[index], settings)
        for index in moving_indices
    }
    fixed_features = {
        index: core._sift_features(fixed.views[index], settings)
        for index in fixed_indices
    }
    matches: list[MatchRow] = []
    pnp_rows: list[PnPRow] = []
    print(
        f"[2/4] Verifying {len(retrieval)} candidates with mutual SIFT and bidirectional PnP",
        flush=True,
    )
    for pair_number, (moving_index, fixed_index, score) in enumerate(
        retrieval, start=1
    ):
        moving_feature = moving_features[moving_index]
        fixed_feature = fixed_features[fixed_index]
        mutual, geometric_mask = core._mutual_sift_matches(
            moving_feature,
            fixed_feature,
            settings.detailed_ratio,
        )
        if len(mutual) < 10 or int(np.count_nonzero(geometric_mask)) < 8:
            continue
        moving_pixels = np.float32(
            [moving_feature[0][left].pt for left, _ in mutual]
        )[geometric_mask]
        fixed_pixels = np.float32(
            [fixed_feature[0][right].pt for _, right in mutual]
        )[geometric_mask]
        matches.append(
            MatchRow(
                moving_view=moving_index,
                fixed_view=fixed_index,
                retrieval_score=score,
                mutual_count=len(mutual),
                geometric_count=len(moving_pixels),
                moving_pixels=moving_pixels,
                fixed_pixels=fixed_pixels,
            )
        )
        forward = core._pnp_correction(
            moving,
            fixed,
            moving_index,
            fixed_index,
            moving_pixels,
            fixed_pixels,
            minimum_inliers=8,
        )
        if forward is not None:
            pnp_rows.append(
                PnPRow(
                    moving_view=moving_index,
                    fixed_view=fixed_index,
                    direction="moving_depth_to_fixed_image",
                    transform_moving_to_fixed=forward.transform,
                    evidence=forward,
                    moving_pixels=moving_pixels,
                    fixed_pixels=fixed_pixels,
                )
            )
        reverse = core._pnp_correction(
            fixed,
            moving,
            fixed_index,
            moving_index,
            fixed_pixels,
            moving_pixels,
            minimum_inliers=8,
        )
        if reverse is not None:
            pnp_rows.append(
                PnPRow(
                    moving_view=moving_index,
                    fixed_view=fixed_index,
                    direction="fixed_depth_to_moving_image",
                    transform_moving_to_fixed=np.linalg.inv(reverse.transform),
                    evidence=reverse,
                    moving_pixels=moving_pixels,
                    fixed_pixels=fixed_pixels,
                )
            )
        if pair_number % 40 == 0:
            print(
                f"      checked {pair_number}/{len(retrieval)}; "
                f"PnP directions={len(pnp_rows)}",
                flush=True,
            )

    # The unconstrained PnP rotation is retained only as a diagnostic by the
    # core helper.  Large pitch/roll disagreement indicates an ill-conditioned
    # one-way solution even when its yaw-only reprojection happens to be low.
    admitted_rows = [
        row for row in pnp_rows if row.evidence.non_yaw_rotation_deg <= 3.5
    ]
    observations = [
        pose_graph.PnPTransformObservation(
            moving_view=row.moving_view,
            fixed_view=row.fixed_view,
            direction=row.direction,
            transform_moving_to_fixed=row.transform_moving_to_fixed,
            inlier_count=row.evidence.inlier_count,
            reprojection_median_px=row.evidence.reprojection_median_px,
            reprojection_p80_px=row.evidence.reprojection_p80_px,
        )
        for row in admitted_rows
    ]
    moving_poses = {
        view.index: moving.world_from_local @ view.camera_pose_local
        for view in moving.views
    }
    fixed_poses = {
        view.index: fixed.world_from_local @ view.camera_pose_local
        for view in fixed.views
    }
    print(
        f"[3/4] Solving floor-locked drift-aware graph from {len(observations)} PnP directions",
        flush=True,
    )
    result = pose_graph.solve_cross_session_pose_graph(
        observations,
        moving_poses,
        fixed_poses,
        vertical_translation_m=vertical_translation_m,
    )
    _write_match_sheet(moving, fixed, admitted_rows, output_dir / "pnp_matches.png")
    report = {
        "schema": "noesis.pcf.connector_pose_graph_registration.v1",
        "generated_at": core._utc_now(),
        "status": "passed" if result.accepted else "rejected",
        "accepted_for_canonical_use": False,
        "disposition": (
            "validated_review_candidate"
            if result.accepted
            else "rejected_registration_candidate"
        ),
        "canonical_limitation": (
            "the connector carrier is a review-only re-projection even when "
            "the cross-session pose graph passes"
        ),
        "method": (
            "mutual_sift_bidirectional_depth_pnp_floor_locked_planar_"
            "smooth_per_view_pose_graph"
        ),
        "whole_cloud_icp_used": False,
        "nearest_neighbor_surface_fitting_used": False,
        "moving_room": _room_report(moving),
        "fixed_room": _room_report(fixed),
        "settings": {
            **asdict(settings),
            "maximum_source_pnp_non_yaw_rotation_deg": 3.5,
            "locked_vertical_translation_m": vertical_translation_m,
        },
        "retrieval_candidate_count": len(retrieval),
        "geometrically_matched_pair_count": len(matches),
        "raw_pnp_direction_count": len(pnp_rows),
        "admitted_pnp_direction_count": len(admitted_rows),
        "admitted_pnp": [_pnp_report(row) for row in admitted_rows],
        "rejected_non_yaw_pnp": [
            _pnp_report(row)
            for row in pnp_rows
            if row.evidence.non_yaw_rotation_deg > 3.5
        ],
        "pose_graph": result.report,
        "reason_codes": list(result.reason_codes),
        "global_transform_moving_to_fixed_row_major": (
            result.global_transform_moving_to_fixed.tolist()
            if result.global_transform_moving_to_fixed is not None
            else None
        ),
        "per_view_transforms_moving_to_fixed_row_major": {
            str(index): transform.tolist()
            for index, transform in result.per_view_transforms_moving_to_fixed.items()
        },
        "artifacts": {"pnp_matches": "pnp_matches.png"},
    }
    (output_dir / "registration_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("[4/4] Wrote fail-closed registration report", flush=True)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moving-name", required=True)
    parser.add_argument("--moving-prior-id", required=True)
    parser.add_argument("--moving-scan-id", required=True)
    parser.add_argument("--moving-scan-dir", type=Path, required=True)
    parser.add_argument("--moving-pcf-root", type=Path, required=True)
    parser.add_argument("--moving-world-manifest", type=Path, required=True)
    parser.add_argument("--fixed-name", required=True)
    parser.add_argument("--fixed-prior-id", required=True)
    parser.add_argument("--fixed-scan-id", required=True)
    parser.add_argument("--fixed-scan-dir", type=Path, required=True)
    parser.add_argument("--fixed-pcf-root", type=Path, required=True)
    parser.add_argument("--fixed-world-manifest", type=Path, required=True)
    parser.add_argument("--vertical-translation-m", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    moving = core._load_room(
        name=arguments.moving_name,
        prior_id=arguments.moving_prior_id,
        scan_id=arguments.moving_scan_id,
        scan_dir=arguments.moving_scan_dir,
        pcf_root=arguments.moving_pcf_root,
        world_manifest=arguments.moving_world_manifest,
    )
    fixed = core._load_room(
        name=arguments.fixed_name,
        prior_id=arguments.fixed_prior_id,
        scan_id=arguments.fixed_scan_id,
        scan_dir=arguments.fixed_scan_dir,
        pcf_root=arguments.fixed_pcf_root,
        world_manifest=arguments.fixed_world_manifest,
    )
    report = run(
        moving,
        fixed,
        arguments.output_dir,
        vertical_translation_m=arguments.vertical_translation_m,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "reason_codes": report["reason_codes"],
                "admitted_pnp_direction_count": report[
                    "admitted_pnp_direction_count"
                ],
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
