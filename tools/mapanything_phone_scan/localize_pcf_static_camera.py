#!/usr/bin/env python3
"""Localize a fixed camera inside a PCF room without whole-cloud fitting.

The estimator uses mutual static/phone RGB features, a per-view fundamental
matrix gate, PCF depth-backed 3D points, and independent per-phone-view PnP.
It then takes a view-balanced consensus of the admitted camera centers and
headings.  Metric scale, gravity, floor height, calibrated pitch, and calibrated
roll remain fixed.  The output is a review anchor, never canonical geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class StaticCameraLocalizationError(RuntimeError):
    """Raised when the preserved evidence cannot support a camera anchor."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StaticCameraLocalizationError(f"{path} is not a JSON object")
    return value


def _heading_deg(forward: np.ndarray) -> float:
    return float(math.degrees(math.atan2(float(forward[0]), float(forward[2]))))


def _angle_delta_deg(left: float, right: float) -> float:
    return float((left - right + 180.0) % 360.0 - 180.0)


def _circular_median_deg(values: list[float]) -> float:
    if not values:
        raise StaticCameraLocalizationError("camera-heading consensus is empty")
    candidates = np.asarray(values, dtype=np.float64)
    costs = [
        float(np.sum(np.abs([_angle_delta_deg(value, item) for item in values])))
        for value in candidates
    ]
    return float(candidates[int(np.argmin(costs))])


def _camera_to_world_from_extrinsics(values: Any) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.size != 16 or not np.isfinite(matrix).all():
        raise StaticCameraLocalizationError("calibrated camera extrinsics are malformed")
    return np.linalg.inv(matrix.reshape((4, 4), order="F"))


def _yaw_rotation(delta_deg: float) -> np.ndarray:
    angle = math.radians(delta_deg)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def _mutual_matches(
    matcher: cv2.BFMatcher,
    fixed_descriptors: np.ndarray,
    phone_descriptors: np.ndarray,
    ratio: float,
) -> list[tuple[int, int]]:
    forward_pairs = matcher.knnMatch(fixed_descriptors, phone_descriptors, k=2)
    reverse_pairs = matcher.knnMatch(phone_descriptors, fixed_descriptors, k=2)
    forward = {
        int(pair[0].queryIdx): int(pair[0].trainIdx)
        for pair in forward_pairs
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
    }
    reverse = {
        int(pair[0].queryIdx): int(pair[0].trainIdx)
        for pair in reverse_pairs
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
    }
    return [
        (fixed_index, phone_index)
        for fixed_index, phone_index in forward.items()
        if reverse.get(phone_index) == fixed_index
    ]


def _fundamental_inliers(
    pairs: list[tuple[int, int]],
    fixed_keypoints: list[cv2.KeyPoint],
    phone_keypoints: list[cv2.KeyPoint],
) -> list[tuple[int, int]]:
    if len(pairs) < 12:
        return []
    fixed_pixels = np.asarray(
        [fixed_keypoints[left].pt for left, _ in pairs], dtype=np.float64
    )
    phone_pixels = np.asarray(
        [phone_keypoints[right].pt for _, right in pairs], dtype=np.float64
    )
    _, mask = cv2.findFundamentalMat(
        fixed_pixels,
        phone_pixels,
        cv2.FM_RANSAC,
        2.0,
        0.999,
    )
    if mask is None:
        return []
    return [pair for pair, keep in zip(pairs, mask.reshape(-1)) if bool(keep)]


def _nearest_pcf_point(
    *,
    phone_pixel: tuple[float, float],
    phone_size: tuple[int, int],
    world_points: np.ndarray,
    mask: np.ndarray,
    confidence: np.ndarray,
    agreement: np.ndarray,
) -> np.ndarray | None:
    grid_height, grid_width = mask.shape
    phone_width, phone_height = phone_size
    x = int(round(phone_pixel[0] * (grid_width - 1) / (phone_width - 1)))
    y = int(round(phone_pixel[1] * (grid_height - 1) / (phone_height - 1)))
    best: tuple[float, int, int] | None = None
    for radius in range(4):
        for row in range(max(0, y - radius), min(grid_height, y + radius + 1)):
            for column in range(
                max(0, x - radius), min(grid_width, x + radius + 1)
            ):
                if (
                    not bool(mask[row, column])
                    or float(confidence[row, column]) <= 0.12
                    or not np.isfinite(world_points[row, column]).all()
                ):
                    continue
                score = float((column - x) ** 2 + (row - y) ** 2)
                if not bool(agreement[row, column]):
                    score += 0.10
                if best is None or score < best[0]:
                    best = (score, row, column)
        if best is not None:
            break
    if best is None:
        return None
    return np.asarray(world_points[best[1], best[2]], dtype=np.float64)


def _render_reprojection(
    *,
    output_path: Path,
    static_image: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_assembly: np.ndarray,
    assembly_npz: Path,
    owner_room_id: int,
) -> dict[str, int]:
    with np.load(assembly_npz) as payload:
        points = np.asarray(payload["points"], dtype=np.float64)
        colors = np.asarray(payload["colors"], dtype=np.uint8)
        owners = np.asarray(payload["owner_room_id"], dtype=np.uint8)
    selected = owners == int(owner_room_id)
    points = points[selected][::2]
    colors = colors[selected][::2, ::-1]
    camera_from_assembly = np.linalg.inv(camera_to_assembly)
    homogeneous = np.column_stack((points, np.ones(points.shape[0])))
    camera_points = (camera_from_assembly @ homogeneous.T).T[:, :3]
    valid = (
        np.isfinite(camera_points).all(axis=1)
        & (camera_points[:, 2] > 0.10)
        & (camera_points[:, 2] < 12.0)
    )
    camera_points = camera_points[valid]
    colors = colors[valid]
    projected = (intrinsics @ camera_points.T).T
    pixels = projected[:, :2] / projected[:, 2, None]
    width = 960
    height = 540
    pixels *= np.asarray(
        [width / static_image.shape[1], height / static_image.shape[0]],
        dtype=np.float64,
    )
    columns = np.rint(pixels[:, 0]).astype(np.int32)
    rows = np.rint(pixels[:, 1]).astype(np.int32)
    inside = (
        (columns >= 0)
        & (columns < width)
        & (rows >= 0)
        & (rows < height)
    )
    columns = columns[inside]
    rows = rows[inside]
    depths = camera_points[inside, 2]
    colors = colors[inside]
    order = np.argsort(depths)[::-1]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[rows[order], columns[order]] = colors[order]
    measured = np.any(canvas != 0, axis=2)
    canvas = cv2.dilate(canvas, np.ones((3, 3), dtype=np.uint8), iterations=1)
    reference = cv2.resize(static_image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.putText(
        reference,
        "STATIC CAMERA",
        (15, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "PCF THROUGH SOLVED STATIC POSE",
        (15, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    combined = np.vstack((reference, canvas))
    if not cv2.imwrite(str(output_path), combined):
        raise StaticCameraLocalizationError(f"could not write {output_path}")
    return {
        "projected_point_count": int(columns.size),
        "measured_pixel_count": int(np.count_nonzero(measured)),
    }


def localize(
    *,
    scan_dir: Path,
    raw_root: Path,
    static_keyframe: Path,
    static_metadata: Path,
    source_world_manifest: Path,
    camera_calibration: Path,
    camera_id: str,
    output_dir: Path,
    assembly_npz: Path | None,
    owner_room_id: int,
) -> dict[str, Any]:
    required = [
        static_keyframe,
        static_metadata,
        source_world_manifest,
        camera_calibration,
    ]
    if assembly_npz is not None:
        required.append(assembly_npz)
    for path in required:
        if not path.is_file():
            raise StaticCameraLocalizationError(f"required input is missing: {path}")
    frame_paths = sorted((scan_dir / "frames").glob("frame_*.jpg"))
    raw_paths = sorted(raw_root.glob("view_*.npz"))
    if len(frame_paths) < 2 or len(frame_paths) != len(raw_paths):
        raise StaticCameraLocalizationError(
            "prepared phone frames and PCF raw views are incomplete or mismatched"
        )
    if output_dir.exists():
        raise StaticCameraLocalizationError(f"output already exists: {output_dir}")

    metadata = _json(static_metadata)
    intrinsics = np.asarray(metadata.get("intrinsics"), dtype=np.float64)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise StaticCameraLocalizationError("static-camera intrinsics are malformed")
    world_manifest = _json(source_world_manifest)
    alignment = world_manifest.get("alignment")
    if not isinstance(alignment, dict):
        raise StaticCameraLocalizationError("source world manifest has no alignment")
    rotation = np.asarray(alignment.get("rotation_row_major"), dtype=np.float64)
    translation = np.asarray(alignment.get("translation"), dtype=np.float64)
    scale = float(alignment.get("scale"))
    if (
        rotation.shape != (3, 3)
        or translation.shape != (3,)
        or not np.isfinite(rotation).all()
        or not np.isfinite(translation).all()
        or not math.isfinite(scale)
        or scale <= 0.0
    ):
        raise StaticCameraLocalizationError("source local-to-assembly Sim3 is malformed")

    calibration = _json(camera_calibration).get("cameras", {})
    camera = calibration.get(camera_id) if isinstance(calibration, dict) else None
    if not isinstance(camera, dict):
        raise StaticCameraLocalizationError(f"camera calibration is missing {camera_id}")
    calibrated_camera_to_world = _camera_to_world_from_extrinsics(camera.get("E"))

    static_bgr = cv2.imread(str(static_keyframe), cv2.IMREAD_COLOR)
    if static_bgr is None:
        raise StaticCameraLocalizationError("static keyframe is unreadable")
    static_gray = cv2.cvtColor(static_bgr, cv2.COLOR_BGR2GRAY)
    cv2.setRNGSeed(7)
    sift = cv2.SIFT_create(nfeatures=10_000, contrastThreshold=0.01)
    fixed_keypoints, fixed_descriptors = sift.detectAndCompute(static_gray, None)
    if fixed_descriptors is None or len(fixed_keypoints) < 100:
        raise StaticCameraLocalizationError("static keyframe has too few SIFT features")
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    candidates: list[dict[str, Any]] = []
    ratio = 0.82
    for view_index, (frame_path, raw_path) in enumerate(zip(frame_paths, raw_paths)):
        phone_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if phone_bgr is None:
            continue
        phone_gray = cv2.cvtColor(phone_bgr, cv2.COLOR_BGR2GRAY)
        phone_keypoints, phone_descriptors = sift.detectAndCompute(phone_gray, None)
        if phone_descriptors is None:
            continue
        mutual = _mutual_matches(
            matcher, fixed_descriptors, phone_descriptors, ratio
        )
        geometric = _fundamental_inliers(
            mutual, fixed_keypoints, phone_keypoints
        )
        if len(geometric) < 8:
            continue
        with np.load(raw_path) as payload:
            world_points = np.asarray(payload["world_points"], dtype=np.float64)
            valid_mask = np.asarray(payload["mask"], dtype=bool)
            confidence = np.asarray(payload["confidence"], dtype=np.float64)
            agreement = np.asarray(payload["cross_model_agreement"], dtype=bool)
        object_points: list[np.ndarray] = []
        image_points: list[tuple[float, float]] = []
        phone_size = (phone_bgr.shape[1], phone_bgr.shape[0])
        for fixed_index, phone_index in geometric:
            local_point = _nearest_pcf_point(
                phone_pixel=phone_keypoints[phone_index].pt,
                phone_size=phone_size,
                world_points=world_points,
                mask=valid_mask,
                confidence=confidence,
                agreement=agreement,
            )
            if local_point is None:
                continue
            object_points.append(scale * (rotation @ local_point) + translation)
            image_points.append(fixed_keypoints[fixed_index].pt)
        if len(object_points) < 8:
            continue
        objects = np.asarray(object_points, dtype=np.float64)
        images = np.asarray(image_points, dtype=np.float64)
        solved, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
            objects,
            images,
            intrinsics,
            None,
            iterationsCount=20_000,
            reprojectionError=6.0,
            confidence=0.99999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not solved or inliers is None or len(inliers) < 8:
            continue
        admitted = np.asarray(inliers, dtype=np.int64).reshape(-1)
        rotation_vector, translation_vector = cv2.solvePnPRefineLM(
            objects[admitted],
            images[admitted],
            intrinsics,
            None,
            rotation_vector,
            translation_vector,
        )
        projected, _ = cv2.projectPoints(
            objects[admitted],
            rotation_vector,
            translation_vector,
            intrinsics,
            None,
        )
        errors = np.linalg.norm(
            projected.reshape((-1, 2)) - images[admitted], axis=1
        )
        camera_from_world = np.eye(4, dtype=np.float64)
        camera_from_world[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
        camera_from_world[:3, 3] = np.asarray(
            translation_vector, dtype=np.float64
        ).reshape(3)
        camera_to_world = np.linalg.inv(camera_from_world)
        center = camera_to_world[:3, 3]
        heading = _heading_deg(camera_to_world[:3, 2])
        median_error = float(np.median(errors))
        p80_error = float(np.percentile(errors, 80.0))
        accepted = bool(
            len(admitted) >= 8
            and median_error <= 5.0
            and p80_error <= 6.0
            and 0.40 <= float(center[1]) <= 3.50
        )
        if accepted:
            candidates.append(
                {
                    "view_index": view_index,
                    "mutual_match_count": len(mutual),
                    "fundamental_inlier_count": len(geometric),
                    "depth_supported_count": len(objects),
                    "pnp_inlier_count": int(len(admitted)),
                    "reprojection_median_px": median_error,
                    "reprojection_p80_px": p80_error,
                    "camera_center_assembly_m": center.tolist(),
                    "camera_heading_deg": heading,
                }
            )

    if len(candidates) < 4:
        raise StaticCameraLocalizationError(
            f"only {len(candidates)} independent phone views localized the static camera"
        )
    centers = np.asarray(
        [row["camera_center_assembly_m"] for row in candidates], dtype=np.float64
    )
    headings = [float(row["camera_heading_deg"]) for row in candidates]
    center_median = np.median(centers, axis=0)
    heading_median = _circular_median_deg(headings)
    horizontal_deviation = np.linalg.norm(
        centers[:, (0, 2)] - center_median[[0, 2]], axis=1
    )
    yaw_deviation = np.abs(
        np.asarray(
            [_angle_delta_deg(value, heading_median) for value in headings],
            dtype=np.float64,
        )
    )
    admitted_indices = [
        index
        for index, (horizontal, yaw) in enumerate(
            zip(horizontal_deviation, yaw_deviation)
        )
        if float(horizontal) <= 1.0 and float(yaw) <= 12.0
    ]
    admitted_candidates = [candidates[index] for index in admitted_indices]
    if len(admitted_candidates) < 4:
        raise StaticCameraLocalizationError(
            "static-camera pose candidates do not form a repeatable consensus"
        )
    admitted_centers = centers[admitted_indices]
    admitted_headings = [headings[index] for index in admitted_indices]
    center = np.median(admitted_centers, axis=0)
    heading = _circular_median_deg(admitted_headings)
    view_count = len(frame_paths)
    view_indices = [int(row["view_index"]) for row in admitted_candidates]
    if min(view_indices) > view_count // 4 or max(view_indices) < 3 * view_count // 4:
        raise StaticCameraLocalizationError(
            "static-camera localization lacks independent early/late walk support"
        )
    center[1] = calibrated_camera_to_world[1, 3]
    calibrated_heading = _heading_deg(calibrated_camera_to_world[:3, 2])
    yaw_delta = _angle_delta_deg(heading, calibrated_heading)
    camera_to_assembly = np.eye(4, dtype=np.float64)
    camera_to_assembly[:3, :3] = (
        _yaw_rotation(yaw_delta) @ calibrated_camera_to_world[:3, :3]
    )
    camera_to_assembly[:3, 3] = center
    horizontal_deviation = np.linalg.norm(
        admitted_centers[:, (0, 2)] - center[[0, 2]], axis=1
    )
    yaw_deviation = np.abs(
        np.asarray(
            [_angle_delta_deg(value, heading) for value in admitted_headings],
            dtype=np.float64,
        )
    )
    old_displacement = float(
        np.linalg.norm(calibrated_camera_to_world[:3, 3] - center)
    )
    output_dir.mkdir(parents=True)
    reprojection: dict[str, Any] | None = None
    if assembly_npz is not None:
        reprojection_path = output_dir / "static_camera_reprojection_review.jpg"
        reprojection = {
            **_render_reprojection(
                output_path=reprojection_path,
                static_image=static_bgr,
                intrinsics=intrinsics,
                camera_to_assembly=camera_to_assembly,
                assembly_npz=assembly_npz,
                owner_room_id=owner_room_id,
            ),
            "relative_path": reprojection_path.name,
            "sha256": _sha256(reprojection_path),
        }
    report = {
        "schema": "noesis.pcf.static_camera_anchor.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_review_anchor",
        "accepted_for_canonical_use": False,
        "camera_id": camera_id,
        "coordinate_frame": "family_accepted_backend_world_m",
        "pose_convention": "opencv_cam2world_x_right_y_down_z_forward",
        "method": (
            "mutual_sift_fundamental_gate_pcf_depth_pnp_per_view_"
            "view_balanced_ground_pose_consensus"
        ),
        "constraints": {
            "metric_scale_fixed": True,
            "gravity_fixed": True,
            "floor_camera_height_fixed_m": float(center[1]),
            "calibrated_pitch_roll_preserved": True,
            "whole_cloud_icp_used": False,
            "bounding_box_anchor_used": False,
            "manual_scene_nudge_used": False,
        },
        "camera_to_assembly_row_major": camera_to_assembly.tolist(),
        "camera_to_assembly_col_major": camera_to_assembly.reshape(
            -1, order="F"
        ).tolist(),
        "assembly_to_camera_row_major": np.linalg.inv(camera_to_assembly).tolist(),
        "device_reference_camera_to_assembly_col_major": (
            calibrated_camera_to_world.reshape(-1, order="F").tolist()
        ),
        "estimate": {
            "camera_center_assembly_m": center.tolist(),
            "camera_heading_deg": heading,
            "calibrated_heading_deg": calibrated_heading,
            "yaw_correction_from_legacy_backend_pose_deg": yaw_delta,
            "legacy_camera_center_displacement_m": old_displacement,
            "translation_uncertainty_p80_m": float(
                np.percentile(horizontal_deviation, 80.0)
            ),
            "yaw_uncertainty_p80_deg": float(np.percentile(yaw_deviation, 80.0)),
        },
        "evidence": {
            "phone_view_count": view_count,
            "admitted_view_count": len(admitted_candidates),
            "admitted_view_indices": view_indices,
            "early_view_supported": min(view_indices) <= view_count // 4,
            "late_view_supported": max(view_indices) >= 3 * view_count // 4,
            "total_pnp_inlier_count": int(
                sum(int(row["pnp_inlier_count"]) for row in admitted_candidates)
            ),
            "per_view": admitted_candidates,
        },
        "inputs": {
            "static_keyframe_sha256": _sha256(static_keyframe),
            "static_metadata_sha256": _sha256(static_metadata),
            "source_world_manifest_sha256": _sha256(source_world_manifest),
            "camera_calibration_sha256": _sha256(camera_calibration),
            "assembly_npz_sha256": (
                _sha256(assembly_npz) if assembly_npz is not None else None
            ),
        },
        "reprojection_review": reprojection,
    }
    report_path = output_dir / "static_camera_anchor_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--static-keyframe", type=Path, required=True)
    parser.add_argument("--static-metadata", type=Path, required=True)
    parser.add_argument("--source-world-manifest", type=Path, required=True)
    parser.add_argument("--camera-calibration", type=Path, required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--assembly-npz", type=Path)
    parser.add_argument("--owner-room-id", type=int, default=1)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = localize(
        scan_dir=args.scan_dir,
        raw_root=args.raw_root,
        static_keyframe=args.static_keyframe,
        static_metadata=args.static_metadata,
        source_world_manifest=args.source_world_manifest,
        camera_calibration=args.camera_calibration,
        camera_id=args.camera_id,
        output_dir=args.output_dir,
        assembly_npz=args.assembly_npz,
        owner_room_id=args.owner_room_id,
    )
    print(
        json.dumps(
            {
                "camera_id": report["camera_id"],
                "status": report["status"],
                "admitted_view_count": report["evidence"]["admitted_view_count"],
                "camera_center_assembly_m": report["estimate"][
                    "camera_center_assembly_m"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
