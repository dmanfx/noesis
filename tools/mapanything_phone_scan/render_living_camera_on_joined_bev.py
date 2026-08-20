#!/usr/bin/env python3
"""Place the Living Room reference camera on a joined Kitchen/Family BEV.

The placement composes two independently estimated connector endpoint
transforms through the already-reviewed Kitchen-to-Family transform.  It never
changes the source room geometry and always emits a review-only artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class LivingCameraOverlayError(RuntimeError):
    """Raised when transform provenance is incomplete or inconsistent."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LivingCameraOverlayError(f"{path} is not a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _transform(value: Any, label: str, *, allow_reflection: bool = False) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise LivingCameraOverlayError(f"{label} is not a finite 4x4 transform")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise LivingCameraOverlayError(f"{label} has an invalid homogeneous row")
    linear = matrix[:3, :3]
    if not np.allclose(linear.T @ linear, np.eye(3), atol=2e-3):
        raise LivingCameraOverlayError(f"{label} contains scale or shear")
    determinant = float(np.linalg.det(linear))
    expected = {-1.0, 1.0} if allow_reflection else {1.0}
    if not any(math.isclose(determinant, item, abs_tol=2e-3) for item in expected):
        raise LivingCameraOverlayError(f"{label} has invalid determinant {determinant}")
    return matrix


def _candidate(report: dict[str, Any], label: str) -> tuple[np.ndarray, dict[str, Any]]:
    pose_graph = report.get("pose_graph")
    candidate = pose_graph.get("candidate") if isinstance(pose_graph, dict) else None
    if not isinstance(candidate, dict):
        raise LivingCameraOverlayError(f"{label} has no evaluated pose-graph candidate")
    transform = _transform(
        candidate.get("global_transform_moving_to_fixed_row_major"),
        f"{label} candidate",
    )
    return transform, candidate


def _heldout_uncertainty(candidate: dict[str, Any]) -> tuple[float, float]:
    leaveout = candidate.get("leave_one_whole_moving_view_out", {})
    translation = float(leaveout.get("translation_p80_m", float("nan")))
    yaw = float(leaveout.get("yaw_p80_deg", float("nan")))
    if not math.isfinite(translation) or not math.isfinite(yaw):
        raise LivingCameraOverlayError("held-out uncertainty is unavailable")
    return translation, yaw


def render(
    *,
    source_png: Path,
    source_bev_manifest: Path,
    source_multiroom_manifest: Path,
    source_presentation_manifest: Path,
    connector_to_kitchen_report: Path,
    connector_to_living_report: Path,
    living_scene_prior_manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    paths = [
        source_png,
        source_bev_manifest,
        source_multiroom_manifest,
        source_presentation_manifest,
        connector_to_kitchen_report,
        connector_to_living_report,
        living_scene_prior_manifest,
    ]
    paths = [path.resolve() for path in paths]
    (
        source_png,
        source_bev_manifest,
        source_multiroom_manifest,
        source_presentation_manifest,
        connector_to_kitchen_report,
        connector_to_living_report,
        living_scene_prior_manifest,
    ) = paths
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise LivingCameraOverlayError(f"output directory already exists: {output_dir}")

    bev = _json(source_bev_manifest)
    multiroom = _json(source_multiroom_manifest)
    presentation = _json(source_presentation_manifest)
    kitchen_report = _json(connector_to_kitchen_report)
    living_report = _json(connector_to_living_report)
    living_prior = _json(living_scene_prior_manifest)
    connector_to_kitchen, kitchen_candidate = _candidate(
        kitchen_report, "connector-to-Kitchen report"
    )
    connector_to_living, living_candidate = _candidate(
        living_report, "connector-to-Living report"
    )
    living_to_kitchen = connector_to_kitchen @ np.linalg.inv(connector_to_living)
    kitchen_to_family = _transform(
        multiroom.get("moving_room", {}).get("global_world_correction_row_major"),
        "Kitchen-to-Family review transform",
    )
    world_to_presentation = _transform(
        presentation.get("presentation_frame", {}).get(
            "world_to_presentation_row_major"
        ),
        "Family presentation transform",
        allow_reflection=True,
    )
    living_to_presentation = (
        world_to_presentation @ kitchen_to_family @ living_to_kitchen
    )

    preview = living_prior.get("preview", {})
    camera_position = np.asarray(
        preview.get("camera_position_world_m"), dtype=np.float64
    )
    camera_forward_xz = np.asarray(
        preview.get("camera_forward_world_xz"), dtype=np.float64
    )
    if camera_position.shape != (3,) or camera_forward_xz.shape != (2,):
        raise LivingCameraOverlayError("Living Room Scene Prior camera pose is malformed")
    camera_h = np.r_[camera_position, 1.0]
    forward_world = camera_position + np.asarray(
        [camera_forward_xz[0], 0.0, camera_forward_xz[1]], dtype=np.float64
    )
    camera_presentation = (living_to_presentation @ camera_h)[:3]
    forward_presentation = (
        living_to_presentation @ np.r_[forward_world, 1.0]
    )[:3]
    direction_xz = forward_presentation[[0, 2]] - camera_presentation[[0, 2]]
    direction_norm = float(np.linalg.norm(direction_xz))
    if direction_norm < 1e-6:
        raise LivingCameraOverlayError("transformed Living camera forward is degenerate")
    direction_xz /= direction_norm

    bounds = bev.get("raster_derivation", {}).get(
        "bounds_camera_local_ground_m", {}
    )
    min_x = float(bounds.get("min_x", float("nan")))
    max_x = float(bounds.get("max_x", float("nan")))
    min_z = float(bounds.get("min_z", float("nan")))
    max_z = float(bounds.get("max_z", float("nan")))
    if not all(math.isfinite(value) for value in (min_x, max_x, min_z, max_z)):
        raise LivingCameraOverlayError("source BEV bounds are missing")
    x_m, z_m = camera_presentation[[0, 2]]
    if not (min_x <= x_m <= max_x and min_z <= z_m <= max_z):
        raise LivingCameraOverlayError(
            "Living camera falls outside the existing joined BEV; regenerate a larger grid"
        )

    source_hash_before = _sha256(source_png)
    image = Image.open(source_png).convert("RGBA")
    width, height = image.size
    x_px = (x_m - min_x) / (max_x - min_x) * width
    y_px = (max_z - z_m) / (max_z - min_z) * height
    pixels_per_meter_x = width / (max_x - min_x)
    pixels_per_meter_y = height / (max_z - min_z)
    kitchen_translation_p80, kitchen_yaw_p80 = _heldout_uncertainty(
        kitchen_candidate
    )
    living_translation_p80, living_yaw_p80 = _heldout_uncertainty(
        living_candidate
    )
    uncertainty_m = math.sqrt(
        kitchen_translation_p80**2 + living_translation_p80**2
    )
    uncertainty_yaw_deg = kitchen_yaw_p80 + living_yaw_p80

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    radius_x = uncertainty_m * pixels_per_meter_x
    radius_y = uncertainty_m * pixels_per_meter_y
    draw.ellipse(
        [x_px - radius_x, y_px - radius_y, x_px + radius_x, y_px + radius_y],
        fill=(72, 205, 220, 34),
        outline=(92, 226, 238, 210),
        width=2,
    )
    marker_size = max(11.0, min(width, height) * 0.014)
    # Image Y is opposite presentation +Z.
    forward_px = np.asarray([direction_xz[0], -direction_xz[1]], dtype=np.float64)
    right_px = np.asarray([-forward_px[1], forward_px[0]], dtype=np.float64)
    center = np.asarray([x_px, y_px], dtype=np.float64)
    triangle = [
        tuple(center + 1.55 * marker_size * forward_px),
        tuple(center - 0.65 * marker_size * forward_px + marker_size * right_px),
        tuple(center - 0.65 * marker_size * forward_px - marker_size * right_px),
    ]
    draw.polygon(triangle, fill=(79, 226, 236, 255), outline=(0, 0, 0, 255))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(13, int(marker_size)))
        small_font = ImageFont.truetype(
            "DejaVuSans.ttf", max(11, int(marker_size * 0.72))
        )
    except OSError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    label_x = x_px + 1.3 * marker_size
    label_y = y_px - 1.15 * marker_size
    draw.text(
        (label_x, label_y),
        "LIVING CAM",
        font=font,
        fill=(99, 235, 242, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0, 255),
    )
    draw.text(
        (label_x, label_y + 1.15 * marker_size),
        f"review ±{uncertainty_m:.2f} m",
        font=small_font,
        fill=(180, 241, 245, 240),
        stroke_width=1,
        stroke_fill=(0, 0, 0, 255),
    )
    output_dir.mkdir(parents=True)
    output_png = output_dir / "kitchen_family_with_living_camera_review.png"
    Image.alpha_composite(image, overlay).convert("RGB").save(
        output_png, format="PNG", optimize=True
    )
    source_hash_after = _sha256(source_png)
    if source_hash_after != source_hash_before:
        raise LivingCameraOverlayError("source BEV changed during overlay")

    manifest = {
        "schema": "noesis.pcf.multiroom_camera_overlay.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "review_only",
        "accepted_for_canonical_use": False,
        "source_geometry_mutated": False,
        "method": (
            "connector_endpoint_transform_composition_without_cloud_icp_or_"
            "manual_canvas_nudging"
        ),
        "camera": {
            "id": preview.get("reference_camera_id"),
            "position_living_backend_world_m": camera_position.tolist(),
            "position_family_presentation_m": camera_presentation.tolist(),
            "forward_family_presentation_xz": direction_xz.tolist(),
            "pixel_xy": [float(x_px), float(y_px)],
            "review_translation_uncertainty_p80_m": uncertainty_m,
            "review_yaw_uncertainty_bound_deg": uncertainty_yaw_deg,
        },
        "transform_chain": {
            "connector_to_kitchen_row_major": connector_to_kitchen.tolist(),
            "connector_to_living_row_major": connector_to_living.tolist(),
            "living_to_kitchen_row_major": living_to_kitchen.tolist(),
            "kitchen_to_family_row_major": kitchen_to_family.tolist(),
            "family_world_to_presentation_row_major": (
                world_to_presentation.tolist()
            ),
            "living_world_to_family_presentation_row_major": (
                living_to_presentation.tolist()
            ),
        },
        "registration_gate_status": {
            "connector_to_kitchen": kitchen_report.get("status"),
            "connector_to_kitchen_reason_codes": kitchen_report.get(
                "reason_codes"
            ),
            "connector_to_living": living_report.get("status"),
            "connector_to_living_reason_codes": living_report.get("reason_codes"),
        },
        "inputs": {
            str(path): {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for path in paths
        },
        "source_bev_sha256_before": source_hash_before,
        "source_bev_sha256_after": source_hash_after,
        "source_bev_unchanged": source_hash_before == source_hash_after,
        "output": {
            "path": output_png.name,
            "sha256": _sha256(output_png),
            "size_bytes": output_png.stat().st_size,
            "width_px": width,
            "height_px": height,
        },
    }
    (output_dir / "living_camera_overlay_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-png", type=Path, required=True)
    parser.add_argument("--source-bev-manifest", type=Path, required=True)
    parser.add_argument("--source-multiroom-manifest", type=Path, required=True)
    parser.add_argument("--source-presentation-manifest", type=Path, required=True)
    parser.add_argument("--connector-to-kitchen-report", type=Path, required=True)
    parser.add_argument("--connector-to-living-report", type=Path, required=True)
    parser.add_argument("--living-scene-prior-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    result = render(
        source_png=arguments.source_png,
        source_bev_manifest=arguments.source_bev_manifest,
        source_multiroom_manifest=arguments.source_multiroom_manifest,
        source_presentation_manifest=arguments.source_presentation_manifest,
        connector_to_kitchen_report=arguments.connector_to_kitchen_report,
        connector_to_living_report=arguments.connector_to_living_report,
        living_scene_prior_manifest=arguments.living_scene_prior_manifest,
        output_dir=arguments.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
