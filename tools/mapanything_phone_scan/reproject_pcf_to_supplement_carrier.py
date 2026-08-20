#!/usr/bin/env python3
"""Reproject PCF evidence through an accepted supplement camera trajectory.

The PCF depth and agreement fields remain authoritative for surfaces.  Camera
poses come from a separately accepted MapAnything supplement revision.  Exact
duplicate RGB-D bridge views estimate the one global metric-depth adjustment;
no nearest-neighbour cloud fitting is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class CarrierReprojectionError(RuntimeError):
    """Raised when the evidence cannot support a safe carrier transfer."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_paths(root: Path) -> list[Path]:
    paths = sorted((root / "raw").glob("view_*.npz"))
    if len(paths) < 2:
        raise CarrierReprojectionError(f"too few raw PCF views in {root / 'raw'}")
    return paths


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as row:
        return {name: np.asarray(row[name]) for name in row.files}


def _parse_bridge(value: str) -> tuple[int, int]:
    try:
        source, reference = (int(item) for item in value.split(":", 1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("bridge must be SOURCE:REFERENCE") from exc
    if source < 0 or reference < 0:
        raise argparse.ArgumentTypeError("bridge indices must be non-negative")
    return source, reference


def _carrier_transform(path: Path) -> tuple[float, np.ndarray, np.ndarray, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    matrix = np.asarray(payload.get("base_from_append_row_major"), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise CarrierReprojectionError(f"malformed carrier transform in {path}")
    linear = matrix[:3, :3]
    scale = float(np.cbrt(np.linalg.det(linear)))
    if not math.isfinite(scale) or scale <= 0.0:
        raise CarrierReprojectionError("carrier transform has invalid metric scale")
    rotation = linear / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise CarrierReprojectionError("carrier transform contains shear")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=2e-3):
        raise CarrierReprojectionError("carrier transform contains a reflection")
    declared = float(payload.get("scale", scale))
    if not math.isclose(scale, declared, rel_tol=2e-3, abs_tol=2e-3):
        raise CarrierReprojectionError("declared and matrix carrier scales disagree")
    return scale, rotation, matrix[:3, 3], payload


def _depth_scale(
    source_paths: list[Path],
    reference_paths: list[Path],
    bridges: list[tuple[int, int]],
) -> tuple[float, dict[str, Any]]:
    if len(bridges) < 4:
        raise CarrierReprojectionError("at least four duplicate RGB-D bridges are required")
    rows: list[dict[str, Any]] = []
    log_medians: list[float] = []
    for source_index, reference_index in bridges:
        if source_index >= len(source_paths) or reference_index >= len(reference_paths):
            raise CarrierReprojectionError("bridge index is outside the raw view set")
        source = _load(source_paths[source_index])
        reference = _load(reference_paths[reference_index])
        source_depth = np.asarray(source["depth_z"], dtype=np.float64)
        reference_depth = np.asarray(reference["depth_z"], dtype=np.float64)
        if source_depth.shape != reference_depth.shape:
            raise CarrierReprojectionError("duplicate bridge depth grids differ in shape")
        valid = np.asarray(source["mask"], dtype=bool) & np.asarray(
            reference["mask"], dtype=bool
        )
        for row in (source, reference):
            if "cross_model_uncertain" in row:
                valid &= ~np.asarray(row["cross_model_uncertain"], dtype=bool)
        valid = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        valid &= (
            np.isfinite(source_depth)
            & np.isfinite(reference_depth)
            & (source_depth > 0.2)
            & (reference_depth > 0.2)
        )
        ratio = reference_depth[valid] / source_depth[valid]
        if ratio.size < 2_000:
            raise CarrierReprojectionError(
                f"bridge {source_index}:{reference_index} has too little common depth"
            )
        low, high = np.percentile(ratio, (10.0, 90.0))
        ratio = ratio[(ratio >= low) & (ratio <= high)]
        log_median = float(np.median(np.log(ratio)))
        log_medians.append(log_median)
        rows.append(
            {
                "source_view": source_index,
                "reference_view": reference_index,
                "common_pixel_count": int(np.count_nonzero(valid)),
                "depth_scale_median": float(np.exp(log_median)),
                "depth_scale_p20": float(np.percentile(ratio, 20.0)),
                "depth_scale_p80": float(np.percentile(ratio, 80.0)),
            }
        )
    median = float(np.median(log_medians))
    scale = float(np.exp(median))
    mad = float(np.median(np.abs(np.asarray(log_medians) - median)))
    if not 0.80 <= scale <= 1.25 or mad > 0.08:
        raise CarrierReprojectionError(
            f"duplicate bridge metric depth is unstable: scale={scale:.3f}, log_mad={mad:.3f}"
        )
    return scale, {
        "method": "view_balanced_median_log_depth_ratio_on_exact_duplicate_bridges",
        "scale": scale,
        "per_view_log_mad": mad,
        "bridges": rows,
    }


def _project_rotation(matrix: np.ndarray) -> np.ndarray:
    left, _, right = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    return rotation


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _world_frame_similarity(
    carrier_reference_paths: list[Path],
    pcf_reference_paths: list[Path],
) -> tuple[float, np.ndarray, np.ndarray, dict[str, Any]]:
    if len(carrier_reference_paths) != len(pcf_reference_paths):
        raise CarrierReprojectionError(
            "world-frame reference view counts do not match"
        )
    if len(carrier_reference_paths) < 12:
        raise CarrierReprojectionError(
            "world-frame alignment requires at least twelve views"
        )
    carrier_centers = np.stack(
        [_load(path)["camera_pose"][:3, 3] for path in carrier_reference_paths]
    ).astype(np.float64)
    pcf_centers = np.stack(
        [_load(path)["camera_pose"][:3, 3] for path in pcf_reference_paths]
    ).astype(np.float64)
    carrier_rotations = np.stack(
        [_load(path)["camera_pose"][:3, :3] for path in carrier_reference_paths]
    ).astype(np.float64)
    pcf_rotations = np.stack(
        [_load(path)["camera_pose"][:3, :3] for path in pcf_reference_paths]
    ).astype(np.float64)
    relative_rotations = np.einsum(
        "nij,nkj->nik", pcf_rotations, carrier_rotations
    )
    pair_angles = np.zeros(
        (len(relative_rotations), len(relative_rotations)), dtype=np.float64
    )
    for left in range(len(relative_rotations)):
        for right in range(left + 1, len(relative_rotations)):
            angle = _rotation_angle_deg(
                relative_rotations[left] @ relative_rotations[right].T
            )
            pair_angles[left, right] = angle
            pair_angles[right, left] = angle
    medoid = int(np.argmin(np.median(pair_angles, axis=1)))
    medoid_errors = pair_angles[medoid]
    median_error = float(np.median(medoid_errors))
    mad_error = float(np.median(np.abs(medoid_errors - median_error)))
    rotation_inliers = medoid_errors <= max(
        8.0, median_error + 2.5 * mad_error
    )
    if int(np.count_nonzero(rotation_inliers)) < 12:
        raise CarrierReprojectionError(
            "world-frame alignment has too few orientation-consistent views"
        )
    rotation = _project_rotation(
        np.mean(relative_rotations[rotation_inliers], axis=0)
    )
    ratios: list[float] = []
    for left in range(len(carrier_centers)):
        for right in range(left + 1, len(carrier_centers)):
            carrier_distance = float(
                np.linalg.norm(carrier_centers[left] - carrier_centers[right])
            )
            pcf_distance = float(np.linalg.norm(pcf_centers[left] - pcf_centers[right]))
            if carrier_distance >= 0.10 and pcf_distance >= 0.10:
                ratios.append(pcf_distance / carrier_distance)
    if len(ratios) < 24:
        raise CarrierReprojectionError(
            "world-frame alignment has too few pose-distance pairs"
        )
    values = np.asarray(ratios, dtype=np.float64)
    scale = float(np.median(values))
    p20 = float(np.percentile(values, 20.0))
    p80 = float(np.percentile(values, 80.0))
    if not 0.75 <= scale <= 1.50 or p80 / max(p20, 1e-8) > 1.65:
        raise CarrierReprojectionError(
            "world-frame alignment scale is unstable: "
            f"scale={scale:.3f}, p20={p20:.3f}, p80={p80:.3f}"
        )
    translations = pcf_centers - scale * (rotation @ carrier_centers.T).T
    translation = np.median(translations[rotation_inliers], axis=0)
    predicted_centers = scale * (rotation @ carrier_centers.T).T + translation
    position_errors = np.linalg.norm(predicted_centers - pcf_centers, axis=1)
    orientation_errors = np.asarray(
        [
            _rotation_angle_deg(
                (rotation @ carrier_rotations[index])
                @ pcf_rotations[index].T
            )
            for index in range(len(carrier_rotations))
        ],
        dtype=np.float64,
    )
    position_p80 = float(np.percentile(position_errors, 80.0))
    orientation_p80 = float(np.percentile(orientation_errors, 80.0))
    if position_p80 > 1.0 or orientation_p80 > 10.0:
        raise CarrierReprojectionError(
            "world-frame camera alignment is unstable: "
            f"position_p80={position_p80:.3f}m, "
            f"orientation_p80={orientation_p80:.3f}deg"
        )
    return scale, rotation, translation, {
        "method": (
            "robust_corresponding_camera_similarity_from_original_mapanything_"
            "trajectory_to_accepted_pcf_trajectory"
        ),
        "scale": scale,
        "pair_count": len(values),
        "ratio_p20": p20,
        "ratio_p80": p80,
        "rotation_row_major": rotation.tolist(),
        "translation_m": translation.tolist(),
        "orientation_inlier_count": int(np.count_nonzero(rotation_inliers)),
        "position_error_median_m": float(np.median(position_errors)),
        "position_error_p80_m": position_p80,
        "position_error_max_m": float(np.max(position_errors)),
        "orientation_error_median_deg": float(np.median(orientation_errors)),
        "orientation_error_p80_deg": orientation_p80,
        "orientation_error_max_deg": float(np.max(orientation_errors)),
    }


def _reproject(
    source: dict[str, np.ndarray],
    carrier: dict[str, np.ndarray],
    *,
    depth_scale: float,
    carrier_scale: float,
    carrier_rotation: np.ndarray,
    carrier_translation: np.ndarray,
    world_scale: float,
    world_rotation: np.ndarray,
    world_translation: np.ndarray,
) -> dict[str, np.ndarray]:
    source_pose = np.asarray(source["camera_pose"], dtype=np.float64)
    carrier_pose = np.asarray(carrier["camera_pose"], dtype=np.float64)
    points = np.asarray(source["world_points"], dtype=np.float64)
    if source_pose.shape != (4, 4) or carrier_pose.shape != (4, 4):
        raise CarrierReprojectionError("camera pose is malformed")
    if points.ndim != 3 or points.shape[2] != 3:
        raise CarrierReprojectionError("PCF world point grid is malformed")
    carrier_base_rotation = carrier_rotation @ carrier_pose[:3, :3]
    carrier_base_translation = (
        carrier_scale * (carrier_rotation @ carrier_pose[:3, 3])
        + carrier_translation
    )
    base_rotation = world_rotation @ carrier_base_rotation
    base_translation = (
        world_scale * (world_rotation @ carrier_base_translation)
        + world_translation
    )
    local = (
        source_pose[:3, :3].T
        @ (points.reshape(-1, 3) - source_pose[:3, 3]).T
    ).T
    reprojected = (
        base_rotation @ (depth_scale * local).T
    ).T + base_translation
    output = {name: np.asarray(value) for name, value in source.items()}
    output["world_points"] = reprojected.reshape(points.shape).astype(np.float32)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = base_rotation.astype(np.float32)
    pose[:3, 3] = base_translation.astype(np.float32)
    output["camera_pose"] = pose
    output["depth_z"] = (
        np.asarray(source["depth_z"], dtype=np.float32) * np.float32(depth_scale)
    )
    if "metric_scaling_factor" in output:
        output["metric_scaling_factor"] = (
            np.asarray(output["metric_scaling_factor"], dtype=np.float32)
            * np.float32(depth_scale)
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pcf-root", type=Path, required=True)
    parser.add_argument("--carrier-raw", type=Path, required=True)
    parser.add_argument("--carrier-transform", type=Path, required=True)
    parser.add_argument("--metric-reference-pcf-root", type=Path, required=True)
    parser.add_argument("--world-reference-carrier-raw", type=Path, required=True)
    parser.add_argument("--world-reference-pcf-root", type=Path, required=True)
    parser.add_argument("--bridge", action="append", type=_parse_bridge, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_pcf_root.resolve()
    source_paths = _raw_paths(source_root)
    carrier_paths = sorted(args.carrier_raw.resolve().glob("view_*.npz"))
    if len(carrier_paths) != len(source_paths):
        raise CarrierReprojectionError(
            f"source/carrier view mismatch: {len(source_paths)} != {len(carrier_paths)}"
        )
    reference_root = args.metric_reference_pcf_root.resolve()
    reference_paths = _raw_paths(reference_root)
    carrier_transform_path = args.carrier_transform.resolve()
    carrier_scale, carrier_rotation, carrier_translation, carrier_payload = (
        _carrier_transform(carrier_transform_path)
    )
    depth_scale, depth_report = _depth_scale(
        source_paths, reference_paths, list(args.bridge)
    )
    world_reference_carrier = sorted(
        args.world_reference_carrier_raw.resolve().glob("view_*.npz")
    )
    world_reference_pcf = _raw_paths(
        args.world_reference_pcf_root.resolve()
    )
    world_scale, world_rotation, world_translation, world_report = (
        _world_frame_similarity(
            world_reference_carrier,
            world_reference_pcf,
        )
    )
    output_dir = args.output_dir.resolve()
    output_raw = output_dir / "raw"
    output_raw.mkdir(parents=True, exist_ok=False)
    for index, (source_path, carrier_path) in enumerate(
        zip(source_paths, carrier_paths, strict=True)
    ):
        output = _reproject(
            _load(source_path),
            _load(carrier_path),
            depth_scale=depth_scale,
            carrier_scale=carrier_scale,
            carrier_rotation=carrier_rotation,
            carrier_translation=carrier_translation,
            world_scale=world_scale,
            world_rotation=world_rotation,
            world_translation=world_translation,
        )
        np.savez_compressed(output_raw / f"view_{index:04d}.npz", **output)
    manifest = {
        "schema": "noesis.pcf.supplement_carrier_reprojection.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "review_only",
        "accepted_for_canonical_use": False,
        "view_count": len(source_paths),
        "output_coordinate_frame": "accepted_reference_pcf_local",
        "surface_authority": "prior_conditioned_consensus_depth_and_agreement",
        "trajectory_authority": "accepted_mapanything_supplement_revision",
        "registration_method": "exact_view_carrier_transfer_without_cloud_icp",
        "metric_depth_normalization": depth_report,
        "world_frame_alignment": world_report,
        "carrier_transform": {
            "path": str(carrier_transform_path),
            "sha256": _sha256(carrier_transform_path),
            "scale": carrier_scale,
            "rotation_row_major": carrier_rotation.tolist(),
            "translation_m": carrier_translation.tolist(),
            "source_schema": carrier_payload.get("schema"),
        },
        "inputs": {
            "source_pcf_root": str(source_root),
            "metric_reference_pcf_root": str(reference_root),
            "carrier_raw": str(args.carrier_raw.resolve()),
            "world_reference_carrier_raw": str(
                args.world_reference_carrier_raw.resolve()
            ),
            "world_reference_pcf_root": str(
                args.world_reference_pcf_root.resolve()
            ),
        },
        "artifacts": {"raw": "raw"},
    }
    manifest_path = output_dir / "carrier_reprojection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
