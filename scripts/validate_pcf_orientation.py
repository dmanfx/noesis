#!/usr/bin/env python3
"""Validate the canonical PCF/Scene Prior presentation coordinate chain."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis_core.coordinate_frames import (  # noqa: E402
    CAMERA_LOCAL_RASTER_ORIENTATION,
    CameraGroundFrame,
    camera_local_raster_indices,
    transform_positions,
)
from noesis_core.scene_prior import (  # noqa: E402
    ScenePriorSet,
    _calibrated_raster_geometry,
)


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _proper_linear(matrix: np.ndarray, *, label: str) -> dict[str, float]:
    linear = np.asarray(matrix, dtype=np.float64)[:3, :3]
    orthonormal_error = float(np.max(np.abs(linear.T @ linear - np.eye(3))))
    determinant = float(np.linalg.det(linear))
    if orthonormal_error > 2e-5 or not math.isclose(
        determinant,
        1.0,
        abs_tol=2e-5,
    ):
        raise ValueError(
            f"{label} must be a proper metric transform: "
            f"det={determinant}, orthonormal_error={orthonormal_error}"
        )
    return {
        "linear_determinant": determinant,
        "orthonormal_max_abs_error": orthonormal_error,
    }


def _display_probes(
    matrix: np.ndarray,
    camera: np.ndarray,
    right: np.ndarray,
    forward: np.ndarray,
) -> dict[str, list[float]]:
    probes = np.stack((camera, camera + right, camera + forward))
    local = transform_positions(probes, matrix)
    if not np.allclose(local[0, [0, 2]], [0.0, 0.0], atol=2e-6):
        raise ValueError("display transform does not place the camera at X/Z origin")
    if not np.allclose(local[1, [0, 2]], [1.0, 0.0], atol=2e-6):
        raise ValueError("display transform does not map camera-right to +X")
    if not np.allclose(local[2, [0, 2]], [0.0, 1.0], atol=2e-6):
        raise ValueError("display transform does not map camera-forward to +Z")
    return {
        "camera_xz": local[0, [0, 2]].tolist(),
        "camera_right_landmark_xz": local[1, [0, 2]].tolist(),
        "camera_forward_landmark_xz": local[2, [0, 2]].tolist(),
    }


def _validate_rooms(catalog_path: Path, calibration_path: Path) -> dict[str, Any]:
    priors = ScenePriorSet.load(catalog_path)
    calibration = _load_object(calibration_path).get("cameras")
    if not isinstance(calibration, dict):
        raise ValueError("camera calibration has no cameras object")
    expected = {"living-room", "family-room", "kitchen"}
    if set(priors.camera_ids) != expected:
        raise ValueError(f"expected camera bindings {sorted(expected)}")

    rooms: dict[str, Any] = {}
    for camera_id in sorted(expected):
        row = calibration.get(camera_id)
        if not isinstance(row, dict) or not isinstance(row.get("E"), list):
            raise ValueError(f"camera {camera_id} has no calibrated E")
        world_to_camera = np.asarray(row["E"], dtype=np.float64).reshape(
            (4, 4), order="F"
        )
        metric = _proper_linear(world_to_camera, label=f"{camera_id} E")
        revision = priors.revision_for_camera(camera_id)
        if revision is None:
            raise ValueError(f"camera {camera_id} has no Scene Prior")
        metadata = priors.camera_view_metadata(
            camera_id,
            extrinsics_col_major=row["E"],
        )
        display = np.asarray(
            metadata["world_to_camera_local_row_major"], dtype=np.float64
        )
        determinant = float(np.linalg.det(display[:3, :3]))
        if not math.isclose(determinant, -1.0, abs_tol=2e-5):
            raise ValueError(
                f"{camera_id} camera-display transform must have determinant -1"
            )
        probes = _display_probes(
            display,
            np.asarray(metadata["camera_position_world_m"], dtype=np.float64),
            np.asarray(metadata["camera_right_world"], dtype=np.float64),
            np.asarray(metadata["camera_forward_world"], dtype=np.float64),
        )
        geometry = _calibrated_raster_geometry(revision, row["E"])
        camera_row_fraction = float(geometry.max_z / (geometry.max_z - geometry.min_z))
        if camera_row_fraction < 0.85:
            raise ValueError(f"{camera_id} camera is not at the bottom of the PCF raster")
        rooms[camera_id] = {
            "prior_id": revision.manifest.prior_id,
            "metric_camera_extrinsics": metric,
            "display_linear_determinant": determinant,
            "display_probes": probes,
            "raster_orientation": CAMERA_LOCAL_RASTER_ORIENTATION,
            "raster_shape": [geometry.rows, geometry.columns],
            "raster_bounds": geometry.bounds_payload(),
            "camera_row_fraction_from_top": camera_row_fraction,
        }

    raster_rows, raster_columns, raster_valid = camera_local_raster_indices(
        np.asarray([0.5, 0.5, 3.5]),
        np.asarray([0.5, 2.5, 0.5]),
        min_x_m=0.0,
        min_z_m=0.0,
        resolution_m=1.0,
        rows=3,
        columns=4,
    )
    if not (
        np.array_equal(raster_rows, [2, 0, 2])
        and np.array_equal(raster_columns, [0, 0, 3])
        and np.all(raster_valid)
    ):
        raise ValueError("asymmetric raster landmarks failed")
    return rooms


def _validate_reference(
    presentation_manifest_path: Path,
    priors: ScenePriorSet,
) -> dict[str, Any]:
    manifest = _load_object(presentation_manifest_path)
    if manifest.get("presentation_only") is not True:
        raise ValueError("reference display matrix is not presentation-only")
    backend = manifest.get("backend_geometry")
    if not isinstance(backend, dict) or not (
        backend.get("mutated") is False
        and backend.get("source_npz_unchanged") is True
        and backend.get("transformed_npz_written") is False
    ):
        raise ValueError("reference presentation did not preserve backend geometry")
    presentation = manifest.get("presentation_frame")
    if not isinstance(presentation, dict):
        raise ValueError("reference has no presentation frame")
    matrix = np.asarray(
        presentation.get("world_to_presentation_row_major"), dtype=np.float64
    )
    determinant = float(np.linalg.det(matrix[:3, :3]))
    if not math.isclose(determinant, -1.0, abs_tol=2e-6):
        raise ValueError("reference display determinant is not -1")
    camera = np.asarray(presentation["camera_position_world_m"], dtype=np.float64)
    right_xz = presentation["camera_right_world_xz"]
    forward_xz = presentation["camera_forward_world_xz"]
    right = np.asarray(
        [right_xz[0], 0.0, right_xz[1]],
        dtype=np.float64,
    )
    forward = np.asarray(
        [forward_xz[0], 0.0, forward_xz[1]],
        dtype=np.float64,
    )
    probes = _display_probes(matrix, camera, right, forward)

    family = priors.revision_for_camera("family-room")
    if family is None or family.manifest.preview is None:
        raise ValueError("Family Room Scene Prior preview is unavailable")
    preview = family.manifest.preview
    expected = CameraGroundFrame(
        camera_world_m=np.asarray(preview.camera_position_world_m, dtype=np.float64),
        camera_right_world=right,
        camera_forward_world=forward,
    ).world_to_camera_local_display_matrix(
        float(family.manifest.derivation.floor_y_m)
    )
    if not np.allclose(matrix, expected, atol=2e-8):
        raise ValueError("reference display matrix differs from Family Scene Prior")

    reintegration_manifest_path = Path(
        str(manifest["source_reintegration_manifest"]["path"])
    )
    reintegration = _load_object(reintegration_manifest_path)
    proper_transforms = {}
    for role in ("fixed_room", "moving_room"):
        room = reintegration.get(role)
        if not isinstance(room, dict):
            raise ValueError(f"reintegration has no {role}")
        correction = np.asarray(
            room.get("global_world_correction_row_major"), dtype=np.float64
        )
        proper_transforms[str(room.get("name") or role)] = _proper_linear(
            correction,
            label=f"{role} global world correction",
        )

    return {
        "status": manifest.get("status"),
        "accepted_for_canonical_use": manifest.get("accepted_for_canonical_use"),
        "presentation_only": True,
        "backend_geometry_unchanged": True,
        "display_linear_determinant": determinant,
        "display_probes": probes,
        "matches_family_scene_prior_display": True,
        "reintegration_metric_transforms": proper_transforms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=REPO_ROOT / "data/scene_priors/catalog.json",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=REPO_ROOT / "config/camera_calibration.json",
    )
    parser.add_argument("--presentation-manifest", type=Path)
    args = parser.parse_args()

    catalog_path = args.catalog.resolve()
    calibration_path = args.calibration.resolve()
    rooms = _validate_rooms(catalog_path, calibration_path)
    report: dict[str, Any] = {
        "status": "passed",
        "canonical_contract": {
            "raw_camera": "opencv_camera_x_right_y_down_z_forward",
            "metric_registration": "proper_sim3_only",
            "backend_geometry": "backend_world_m",
            "display_frame": "camera_local_ground_m_x_right_y_up_z_forward",
            "display_transform_use": "presentation_only",
            "raster_orientation": CAMERA_LOCAL_RASTER_ORIENTATION,
            "three_presentation": "camera_only_no_negative_model_scale",
        },
        "rooms": rooms,
    }
    if args.presentation_manifest is not None:
        priors = ScenePriorSet.load(catalog_path)
        report["reference_evidence"] = _validate_reference(
            args.presentation_manifest.resolve(),
            priors,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
