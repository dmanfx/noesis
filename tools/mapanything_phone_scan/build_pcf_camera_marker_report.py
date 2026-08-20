#!/usr/bin/env python3
"""Build review-only static-camera markers in a multi-room PCF assembly frame."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


class CameraMarkerError(RuntimeError):
    """Raised when camera markers cannot be derived without ambiguity."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CameraMarkerError(f"{path} is not a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _proper_pose(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise CameraMarkerError(f"{label} is not a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise CameraMarkerError(f"{label} is not affine")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise CameraMarkerError(f"{label} is not rigid")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
        raise CameraMarkerError(f"{label} is reflected")
    return matrix


def _scene_camera(
    manifest_path: Path,
    expected_camera_id: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    manifest = _json(manifest_path)
    preview = manifest.get("preview")
    position = preview.get("camera_position_world_m") if isinstance(preview, dict) else None
    camera_id = preview.get("reference_camera_id") if isinstance(preview, dict) else None
    if (
        camera_id != expected_camera_id
        or not isinstance(position, list)
        or len(position) != 3
        or not all(isinstance(item, (int, float)) and math.isfinite(item) for item in position)
    ):
        raise CameraMarkerError(
            f"{manifest_path} has no valid {expected_camera_id} reference camera"
        )
    return np.asarray(position, dtype=np.float64), manifest


def build_report(
    *,
    camera_anchor_report: Path,
    multiroom_registration_report: Path,
    kitchen_scene_prior_manifest: Path,
    living_scene_prior_manifest: Path,
) -> dict[str, Any]:
    anchor = _json(camera_anchor_report)
    if (
        anchor.get("schema") != "noesis.pcf.static_camera_anchor.v2"
        or anchor.get("status") != "passed_review_anchor"
        or anchor.get("accepted_for_canonical_use") is not False
        or anchor.get("coordinate_frame") != "family_accepted_backend_world_m"
    ):
        raise CameraMarkerError("Family Room camera anchor is not admissible")
    estimate = anchor.get("estimate")
    family_position = (
        estimate.get("camera_center_assembly_m") if isinstance(estimate, dict) else None
    )
    if (
        not isinstance(family_position, list)
        or len(family_position) != 3
        or not all(isinstance(item, (int, float)) and math.isfinite(item) for item in family_position)
    ):
        raise CameraMarkerError("Family Room camera anchor has no finite center")

    registration = _json(multiroom_registration_report)
    transforms = registration.get("derived_endpoint_transforms")
    if not isinstance(transforms, dict):
        raise CameraMarkerError("multi-room registration has no endpoint transforms")
    kitchen_to_family = _proper_pose(
        transforms.get("kitchen_to_family_row_major"),
        "Kitchen-to-Family transform",
    )
    living_to_family = _proper_pose(
        transforms.get("living_to_family_row_major"),
        "Living-to-Family transform",
    )
    kitchen_position, _ = _scene_camera(kitchen_scene_prior_manifest, "kitchen")
    living_position, _ = _scene_camera(living_scene_prior_manifest, "living-room")

    def transformed(matrix: np.ndarray, position: np.ndarray) -> list[float]:
        result = matrix @ np.append(position, 1.0)
        return [float(item) for item in result[:3]]

    markers = [
        {
            "camera_id": "family-room",
            "position_assembly_m": [float(item) for item in family_position],
            "source_role": "admitted_scene_prior_reference_camera",
            "room_transform_status": "fixed_assembly_gauge",
        },
        {
            "camera_id": "kitchen",
            "position_assembly_m": transformed(kitchen_to_family, kitchen_position),
            "source_role": "scene_prior_reference_camera_through_room_transform",
            "room_transform_status": "review_only",
        },
        {
            "camera_id": "living-room",
            "position_assembly_m": transformed(living_to_family, living_position),
            "source_role": "scene_prior_reference_camera_through_room_transform",
            "room_transform_status": str(registration.get("status") or "unknown"),
        },
    ]
    inputs = {
        str(path): {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
        for path in (
            camera_anchor_report,
            multiroom_registration_report,
            kitchen_scene_prior_manifest,
            living_scene_prior_manifest,
        )
    }
    return {
        "schema": "noesis.pcf.static_camera_markers.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "review_only",
        "accepted_for_canonical_use": False,
        "coordinate_frame": "family_accepted_backend_world_m",
        "method": "static_camera_centers_composed_through_recorded_room_transforms",
        "sphere_radius_m": 0.18,
        "color_hex": "#ffd400",
        "markers": markers,
        "constraints": {
            "same_assembly_transform_as_geometry": True,
            "bounding_box_anchor_used": False,
            "manual_scene_nudge_used": False,
            "reflection_used": False,
        },
        "inputs": inputs,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-anchor-report", type=Path, required=True)
    parser.add_argument("--multiroom-registration-report", type=Path, required=True)
    parser.add_argument("--kitchen-scene-prior-manifest", type=Path, required=True)
    parser.add_argument("--living-scene-prior-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = build_report(
        camera_anchor_report=args.camera_anchor_report,
        multiroom_registration_report=args.multiroom_registration_report,
        kitchen_scene_prior_manifest=args.kitchen_scene_prior_manifest,
        living_scene_prior_manifest=args.living_scene_prior_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "markers": report["markers"]}))


if __name__ == "__main__":
    main()
