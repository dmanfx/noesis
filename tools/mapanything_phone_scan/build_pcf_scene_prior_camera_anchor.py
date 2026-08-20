#!/usr/bin/env python3
"""Build a floor-locked Family camera anchor from an admitted Scene Prior."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


class ScenePriorCameraAnchorError(RuntimeError):
    """Raised when the admitted camera pose cannot be bound safely."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ScenePriorCameraAnchorError(f"{path} is not a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _camera_to_world(values: Any) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.size != 16 or not np.isfinite(matrix).all():
        raise ScenePriorCameraAnchorError("camera extrinsics are malformed")
    result = np.linalg.inv(matrix.reshape((4, 4), order="F"))
    if not math.isclose(float(np.linalg.det(result[:3, :3])), 1.0, abs_tol=1e-6):
        raise ScenePriorCameraAnchorError("camera pose is not proper rigid")
    return result


def _heading_deg(forward: np.ndarray) -> float:
    return float(math.degrees(math.atan2(float(forward[0]), float(forward[2]))))


def _angle_delta_deg(left: float, right: float) -> float:
    return float((left - right + 180.0) % 360.0 - 180.0)


def _yaw_rotation(delta_deg: float) -> np.ndarray:
    angle = math.radians(delta_deg)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def build_report(
    *,
    scene_prior_manifest: Path,
    alignment_report: Path,
    pnp_diagnostic_report: Path,
    camera_calibration: Path,
    assembly_npz: Path,
    camera_id: str,
) -> dict[str, Any]:
    prior = _json(scene_prior_manifest)
    preview = prior.get("preview")
    quality = prior.get("quality")
    if (
        not isinstance(preview, dict)
        or preview.get("reference_camera_id") != camera_id
        or preview.get("coordinate_frame") != "camera_local_ground_m"
        or not isinstance(quality, dict)
        or quality.get("passed") is not True
        or quality.get("alignment_status") != "passed"
    ):
        raise ScenePriorCameraAnchorError("Scene Prior camera binding is not admitted")
    position = preview.get("camera_position_world_m")
    forward_xz = preview.get("camera_forward_world_xz")
    if (
        not isinstance(position, list)
        or len(position) != 3
        or not isinstance(forward_xz, list)
        or len(forward_xz) != 2
        or not all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in [*position, *forward_xz]
        )
    ):
        raise ScenePriorCameraAnchorError("Scene Prior camera pose is malformed")
    floor_y = float(prior.get("derivation", {}).get("floor_y_m"))
    if not math.isfinite(floor_y) or abs(floor_y) > 0.05:
        raise ScenePriorCameraAnchorError("Scene Prior floor authority is unavailable")

    calibration = _json(camera_calibration)
    camera = calibration.get("cameras", {}).get(camera_id)
    if not isinstance(camera, dict):
        raise ScenePriorCameraAnchorError("calibrated reference camera is missing")
    calibration_sha = _sha256(camera_calibration)
    if preview.get("camera_calibration", {}).get("sha256") != calibration_sha:
        raise ScenePriorCameraAnchorError("Scene Prior calibration digest disagrees")
    device_reference = _camera_to_world(camera.get("E"))
    calibrated_heading = _heading_deg(device_reference[:3, 2])
    prior_heading = _heading_deg(
        np.asarray([forward_xz[0], 0.0, forward_xz[1]], dtype=np.float64)
    )
    yaw_delta = _angle_delta_deg(prior_heading, calibrated_heading)
    camera_to_assembly = np.eye(4, dtype=np.float64)
    camera_to_assembly[:3, :3] = (
        _yaw_rotation(yaw_delta) @ device_reference[:3, :3]
    )
    camera_to_assembly[:3, 3] = np.asarray(position, dtype=np.float64)

    alignment = _json(alignment_report)
    fixed_reprojection = alignment.get("fixed_camera_reprojection")
    if (
        alignment.get("status") != "passed"
        or alignment.get("quality_gate", {}).get("passed") is not True
        or alignment.get("target", {}).get("camera_id") != camera_id
        or not isinstance(fixed_reprojection, dict)
    ):
        raise ScenePriorCameraAnchorError("conditioned-fusion alignment is not admitted")
    pnp = _json(pnp_diagnostic_report)
    pnp_estimate = pnp.get("estimate")
    if (
        pnp.get("schema") != "noesis.pcf.static_camera_anchor.v1"
        or not isinstance(pnp_estimate, dict)
    ):
        raise ScenePriorCameraAnchorError("PnP diagnostic report is unavailable")
    translation_uncertainty = max(
        float(fixed_reprojection.get("overlap_depth_delta_p80_m")),
        float(pnp_estimate.get("translation_uncertainty_p80_m")),
    )
    yaw_uncertainty = float(pnp_estimate.get("yaw_uncertainty_p80_deg"))
    if not all(math.isfinite(value) and value >= 0.0 for value in (
        translation_uncertainty,
        yaw_uncertainty,
    )):
        raise ScenePriorCameraAnchorError("camera uncertainty is not finite")

    return {
        "schema": "noesis.pcf.static_camera_anchor.v2",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "passed_review_anchor",
        "accepted_for_canonical_use": False,
        "camera_id": camera_id,
        "coordinate_frame": "family_accepted_backend_world_m",
        "pose_convention": "opencv_cam2world_x_right_y_down_z_forward",
        "anchor_mode": "floor_locked_planar",
        "method": "admitted_scene_prior_reference_camera_floor_locked_planar_anchor",
        "constraints": {
            "metric_scale_fixed": True,
            "gravity_fixed": True,
            "floor_y_m": floor_y,
            "camera_height_source": "admitted_scene_prior_reference_camera",
            "vertical_anchor_translation_m": 0.0,
            "calibrated_pitch_roll_preserved": True,
            "pnp_translation_used": False,
            "whole_cloud_icp_used": False,
            "bounding_box_anchor_used": False,
            "manual_scene_nudge_used": False,
        },
        "camera_to_assembly_row_major": camera_to_assembly.tolist(),
        "camera_to_assembly_col_major": camera_to_assembly.reshape(
            -1, order="F"
        ).tolist(),
        "assembly_to_camera_row_major": np.linalg.inv(camera_to_assembly).tolist(),
        "device_reference_camera_to_assembly_col_major": device_reference.reshape(
            -1, order="F"
        ).tolist(),
        "estimate": {
            "camera_center_assembly_m": [float(value) for value in position],
            "camera_heading_deg": prior_heading,
            "calibrated_heading_deg": calibrated_heading,
            "yaw_correction_from_device_reference_deg": yaw_delta,
            "device_reference_camera_center_displacement_m": float(
                np.linalg.norm(device_reference[:3, 3] - camera_to_assembly[:3, 3])
            ),
            "translation_uncertainty_p80_m": translation_uncertainty,
            "yaw_uncertainty_p80_deg": yaw_uncertainty,
        },
        "evidence": {
            "scene_prior_quality_passed": True,
            "conditioned_fusion_alignment_passed": True,
            "fixed_camera_reprojection_admitted": alignment.get("quality_gate", {})
            .get("checks", {})
            .get("fixed_camera_reprojection_admitted")
            is True,
            "pnp_report_role": "uncertainty_diagnostic_only",
            "pnp_translation_used": False,
        },
        "inputs": {
            "scene_prior_manifest_sha256": _sha256(scene_prior_manifest),
            "alignment_report_sha256": _sha256(alignment_report),
            "pnp_diagnostic_report_sha256": _sha256(pnp_diagnostic_report),
            "camera_calibration_sha256": calibration_sha,
            "assembly_npz_sha256": _sha256(assembly_npz),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-prior-manifest", type=Path, required=True)
    parser.add_argument("--alignment-report", type=Path, required=True)
    parser.add_argument("--pnp-diagnostic-report", type=Path, required=True)
    parser.add_argument("--camera-calibration", type=Path, required=True)
    parser.add_argument("--assembly-npz", type=Path, required=True)
    parser.add_argument("--camera-id", default="family-room")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = build_report(
        scene_prior_manifest=args.scene_prior_manifest,
        alignment_report=args.alignment_report,
        pnp_diagnostic_report=args.pnp_diagnostic_report,
        camera_calibration=args.camera_calibration,
        assembly_npz=args.assembly_npz,
        camera_id=args.camera_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "estimate": report["estimate"]}))


if __name__ == "__main__":
    main()
