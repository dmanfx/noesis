#!/usr/bin/env python3
"""Layer a complete floor plane beneath trusted dashboard foreground geometry."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry.floorplan_shadow_fill import (  # noqa: E402
    ShadowFillSettings,
    apply_floor_underlay,
    build_floor_underlay_masks,
    build_phone_floor_visibility_from_views,
    compute_height_occlusion_shadow,
    compute_radial_occlusion_artifact,
    compute_static_floor_occlusion_shadow,
    floorplan_grid_centers,
    phone_view_points_to_static_floorplan,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON input: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must contain an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _decode_layer(payload: Mapping[str, Any], name: str) -> np.ndarray:
    node = payload.get(name)
    if not isinstance(node, Mapping):
        raise ValueError(f"baseline floorplan is missing required layer {name!r}")
    shape = node.get("grid_shape")
    encoded = node.get("grid_b64")
    if (
        not isinstance(shape, Sequence)
        or isinstance(shape, (str, bytes))
        or len(shape) != 2
        or not isinstance(encoded, str)
    ):
        raise ValueError(f"baseline layer {name!r} has an invalid grid contract")
    rows, cols = int(shape[0]), int(shape[1])
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"baseline layer {name!r} is not valid base64") from exc
    array = np.frombuffer(raw, dtype="<f4")
    if array.size != rows * cols:
        raise ValueError(f"baseline layer {name!r} has the wrong byte length")
    return array.copy().reshape(rows, cols)


def _layer(array: np.ndarray, value_min: float, value_max: float) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array, dtype="<f4")
    return {
        "grid_b64": base64.b64encode(contiguous.tobytes()).decode("ascii"),
        "grid_shape": [int(contiguous.shape[0]), int(contiguous.shape[1])],
        "value_min": float(value_min),
        "value_max": float(value_max),
    }


def _finite_range(array: np.ndarray, default_max: float = 1.0) -> tuple[float, float]:
    finite = np.asarray(array)[np.isfinite(array)]
    if finite.size <= 0:
        return 0.0, float(default_max)
    low = float(np.min(finite))
    high = float(np.max(finite))
    if high <= low:
        high = low + float(default_max)
    return low, high


def _inferno_rgb(values: np.ndarray, low: float, high: float) -> np.ndarray:
    stops = np.asarray(
        [
            [0.0, 0.0, 4.0],
            [35.0, 6.0, 59.0],
            [99.0, 23.0, 94.0],
            [159.0, 43.0, 73.0],
            [218.0, 83.0, 32.0],
            [252.0, 255.0, 164.0],
        ],
        dtype=np.float32,
    )
    normalized = np.nan_to_num(
        np.clip(
            (np.asarray(values, dtype=np.float32) - float(low))
            / max(float(high) - float(low), 1e-6),
            0.0,
            1.0,
        ),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    position = normalized * 5.0
    left = np.clip(np.floor(position).astype(np.int32), 0, 4)
    fraction = (position - left).astype(np.float32)
    return np.clip(
        np.rint(stops[left] + ((stops[left + 1] - stops[left]) * fraction[..., None])),
        0,
        255,
    ).astype(np.uint8)


def _height_image(
    height: np.ndarray,
    density: np.ndarray,
    *,
    locked_range: tuple[float, float],
    width: int = 1000,
) -> Image.Image:
    rgb = _inferno_rgb(height, locked_range[0], locked_range[1])
    rows, cols = height.shape
    checker = np.zeros((rows, cols, 3), dtype=np.uint8)
    rr, cc = np.indices((rows, cols))
    alternate = ((rr // 4) + (cc // 4)) % 2 == 1
    checker[:] = [16, 22, 32]
    checker[alternate] = [26, 34, 47]
    observed = np.isfinite(height) & np.isfinite(density) & (density > 0.0)
    rgb[~observed] = checker[~observed]
    image = Image.fromarray(rgb, mode="RGB")
    height_px = max(1, int(round(width * rows / cols)))
    return image.resize((width, height_px), Image.Resampling.BICUBIC)


def _authority_image(
    authority: np.ndarray,
    occluder: np.ndarray,
    *,
    width: int = 1000,
) -> Image.Image:
    colors = np.asarray(
        [
            [8, 12, 18],
            [95, 105, 118],
            [244, 164, 61],
            [50, 215, 176],
            [66, 126, 190],
        ],
        dtype=np.uint8,
    )
    rgb = colors[np.clip(authority.astype(np.int32), 0, 4)]
    edge = np.asarray(occluder, dtype=bool) & ~ndi.binary_erosion(occluder)
    rgb[edge] = [245, 72, 90]
    image = Image.fromarray(rgb, mode="RGB")
    rows, cols = authority.shape
    height_px = max(1, int(round(width * rows / cols)))
    return image.resize((width, height_px), Image.Resampling.NEAREST)


def _artifact_evidence_image(
    *,
    phone_floor: np.ndarray,
    floor_confirmed_artifact: np.ndarray,
    expanded_artifact: np.ndarray,
    removed_artifact: np.ndarray,
    trusted_foreground: np.ndarray,
    width: int = 1000,
) -> Image.Image:
    shape = np.asarray(phone_floor, dtype=bool).shape
    rgb = np.zeros((*shape, 3), dtype=np.uint8)
    rgb[:] = [8, 12, 18]
    rgb[np.asarray(phone_floor, dtype=bool)] = [42, 85, 115]
    rgb[np.asarray(expanded_artifact, dtype=bool)] = [69, 76, 109]
    rgb[np.asarray(floor_confirmed_artifact, dtype=bool)] = [244, 164, 61]
    rgb[np.asarray(removed_artifact, dtype=bool)] = [50, 215, 176]
    foreground = np.asarray(trusted_foreground, dtype=bool)
    edge = foreground & ~ndi.binary_erosion(foreground)
    rgb[edge] = [245, 72, 90]
    image = Image.fromarray(rgb, mode="RGB")
    rows, cols = shape
    height_px = max(1, int(round(width * rows / cols)))
    return image.resize((width, height_px), Image.Resampling.NEAREST)


def _label(image: Image.Image, title: str, subtitle: str = "") -> Image.Image:
    header = 58 if subtitle else 38
    output = Image.new("RGB", (image.width, image.height + header), (8, 12, 18))
    output.paste(image, (0, header))
    draw = ImageDraw.Draw(output)
    draw.text((12, 8), title, fill=(238, 243, 248))
    if subtitle:
        draw.text((12, 28), subtitle, fill=(157, 171, 190))
    return output


def _combine_horizontal(images: Sequence[Image.Image], gap: int = 12) -> Image.Image:
    width = sum(image.width for image in images) + gap * max(0, len(images) - 1)
    height = max(image.height for image in images)
    output = Image.new("RGB", (width, height), (7, 11, 18))
    x = 0
    for image in images:
        output.paste(image, (x, 0))
        x += image.width + gap
    return output


def _parse_roi(value: str | None) -> tuple[float, float, float, float] | None:
    if not value:
        return None
    try:
        items = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError("focus ROI must be min_x,max_x,min_z,max_z") from exc
    if len(items) != 4 or items[1] <= items[0] or items[3] <= items[2]:
        raise ValueError("focus ROI must be min_x,max_x,min_z,max_z")
    return items


def _focus_image(
    before: np.ndarray,
    after: np.ndarray,
    density_before: np.ndarray,
    density_after: np.ndarray,
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    roi: tuple[float, float, float, float],
    locked_range: tuple[float, float],
) -> Image.Image:
    selection = (
        (x_grid >= roi[0])
        & (x_grid <= roi[1])
        & (z_grid >= roi[2])
        & (z_grid <= roi[3])
    )
    rows, cols = np.where(selection)
    if rows.size <= 0:
        raise ValueError("focus ROI does not intersect the floorplan")
    row_slice = slice(int(rows.min()), int(rows.max()) + 1)
    col_slice = slice(int(cols.min()), int(cols.max()) + 1)
    left = _label(
        _height_image(
            before[row_slice, col_slice],
            density_before[row_slice, col_slice],
            locked_range=locked_range,
            width=700,
        ),
        "Focus before",
    )
    right = _label(
        _height_image(
            after[row_slice, col_slice],
            density_after[row_slice, col_slice],
            locked_range=locked_range,
            width=700,
        ),
        "Focus after",
    )
    return _combine_horizontal([left, right])


def _review_html() -> str:
    return """<!doctype html>
<html lang="en"><head><meta charset="UTF-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Noesis complete floor underlay review</title>
<style>
:root{color-scheme:dark;font-family:Inter,system-ui,sans-serif;background:#070b12;color:#eef3f8}
*{box-sizing:border-box}body{margin:0;background:#070b12}.page{padding:18px}
.heading{display:flex;align-items:end;justify-content:space-between;margin:0 4px 14px}.heading h1{font-size:23px;margin:0}.heading p{color:#91a0b2;margin:0;font-size:13px}
.grid{display:grid;grid-template-columns:repeat(3,720px);gap:14px}.panel{background:#0d1420;border:1px solid #253247;border-radius:10px;padding:10px}
.title{display:flex;justify-content:space-between;gap:12px;align-items:baseline;margin:0 2px 7px}.title strong{font-size:16px}.title span{font-size:12px;color:#58dce4}
.desc{height:32px;color:#9eabbb;font-size:12px;line-height:1.35;margin:0 2px 8px}.map{width:720px;height:600px;display:block;background:#020509}
.stats{display:flex;gap:14px;color:#b8c3d0;font-size:12px;padding:8px 2px 1px}.single .grid{display:block}.single .panel{width:1220px;padding:10px}
.single .map{width:1200px;height:1000px}.single .desc{height:auto}.single .heading{width:1220px}
</style></head><body><div id="root"></div><script src="review_data.js"></script><script src="renderer.bundle.js"></script></body></html>
"""


def _run_chrome_screenshot(
    chrome: Path,
    page: Path,
    output: Path,
    *,
    candidate: str,
    scale: str,
) -> None:
    url = (
        page.resolve().as_uri()
        + f"?layout=single&scale={scale}&candidate={candidate}"
    )
    with tempfile.TemporaryDirectory(prefix="noesis-shadow-review-") as profile:
        command = [
            str(chrome),
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--allow-file-access-from-files",
            "--force-device-scale-factor=1",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=1500",
            "--window-size=1260,1160",
            f"--user-data-dir={profile}",
            f"--screenshot={output.resolve()}",
            url,
        ]
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
    if completed.returncode != 0 or not output.is_file():
        raise RuntimeError(
            "headless Chrome did not produce the exact-render screenshot: "
            + completed.stdout[-2000:]
        )


def _load_original_phone_observations(
    *,
    outputs_dir: Path,
    bootstrap_path: Path,
    scene_fusion_manifest: Mapping[str, Any],
    reference_pose: np.ndarray,
    floor_y_m: float,
    scene_to_static: np.ndarray,
    sample_stride_px: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load preserved phone-only pixels through the admitted joint pose bridge."""

    stride = int(sample_stride_px)
    if stride < 1:
        raise ValueError("phone sample stride must be at least one pixel")
    outputs_manifest_path = outputs_dir / "scan_outputs_manifest.json"
    source_solution_path = outputs_dir / "camera_solution.npz"
    if not outputs_manifest_path.is_file() or not source_solution_path.is_file():
        raise ValueError("phone outputs directory is missing its manifest or camera solution")
    outputs_manifest = _load_json(outputs_manifest_path)
    if outputs_manifest.get("schema") != "noesis.mapanything.phone_scan.outputs.v1":
        raise ValueError("phone outputs manifest has an unsupported schema")
    if outputs_manifest.get("coordinate_frame") != "mapanything_metric_world_unaligned_to_noesis":
        raise ValueError("phone outputs are not in the expected original MA world frame")
    frame_rows = outputs_manifest.get("frames")
    if not isinstance(frame_rows, Sequence) or isinstance(frame_rows, (str, bytes)):
        raise ValueError("phone outputs manifest has no frame collection")

    with np.load(source_solution_path, allow_pickle=False) as loaded:
        source_poses = np.asarray(loaded["camera_poses"], dtype=np.float64)
    with np.load(bootstrap_path, allow_pickle=False) as loaded:
        if not {"source_phone_poses", "camera_poses"}.issubset(loaded.files):
            raise ValueError("phone pose bootstrap lacks source or admitted poses")
        bootstrap_source_poses = np.asarray(
            loaded["source_phone_poses"],
            dtype=np.float64,
        )
        admitted_poses = np.asarray(loaded["camera_poses"], dtype=np.float64)
    frame_count = len(frame_rows)
    expected_shape = (frame_count, 4, 4)
    if (
        source_poses.shape != expected_shape
        or bootstrap_source_poses.shape != expected_shape
        or admitted_poses.shape != expected_shape
    ):
        raise ValueError("phone pose collections do not match the raw view count")
    if not np.allclose(source_poses, bootstrap_source_poses, atol=1e-5, rtol=1e-5):
        raise ValueError("pose bootstrap does not belong to these phone-only outputs")

    method = scene_fusion_manifest.get("method")
    refinement_values = (
        method.get("phone_to_fixed_refinement_row_major")
        if isinstance(method, Mapping)
        else None
    )
    if not isinstance(refinement_values, Sequence) or len(refinement_values) != 16:
        raise ValueError("scene fusion has no phone-to-fixed refinement")
    refinement = np.asarray(refinement_values, dtype=np.float64).reshape(4, 4)

    point_rows: list[np.ndarray] = []
    confidence_rows: list[np.ndarray] = []
    view_rows: list[np.ndarray] = []
    raw_bytes = 0
    for index, frame in enumerate(frame_rows):
        if not isinstance(frame, Mapping) or int(frame.get("index", -1)) != index:
            raise ValueError("phone output frame indices are not contiguous")
        raw_name = frame.get("raw_npz")
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError(f"phone frame {index} has no raw NPZ")
        raw_path = outputs_dir.parent / raw_name
        if not raw_path.is_file():
            raise ValueError(f"phone raw view is missing: {raw_path}")
        raw_bytes += raw_path.stat().st_size
        with np.load(raw_path, allow_pickle=False) as loaded:
            world_points = np.asarray(loaded["world_points"], dtype=np.float32)
            mask = np.asarray(loaded["mask"], dtype=np.uint8)
            confidence = np.asarray(loaded["confidence"], dtype=np.float32)
            raw_pose = np.asarray(loaded["camera_pose"], dtype=np.float64)
        if (
            world_points.ndim != 3
            or world_points.shape[2] != 3
            or mask.shape != world_points.shape[:2]
            or confidence.shape != world_points.shape[:2]
            or raw_pose.shape != (4, 4)
        ):
            raise ValueError(f"phone raw view {index} has incompatible arrays")
        if not np.allclose(raw_pose, source_poses[index], atol=1e-5, rtol=1e-5):
            raise ValueError(f"phone raw view {index} pose does not match its solution")
        sampled_points = world_points[::stride, ::stride].reshape(-1, 3)
        sampled_mask = mask[::stride, ::stride].reshape(-1) > 0
        sampled_confidence = confidence[::stride, ::stride].reshape(-1)
        valid = (
            sampled_mask
            & np.isfinite(sampled_points).all(axis=1)
            & np.isfinite(sampled_confidence)
        )
        sampled_points = sampled_points[valid]
        sampled_confidence = sampled_confidence[valid]
        static_points = phone_view_points_to_static_floorplan(
            points_phone_world_m=sampled_points,
            source_camera_to_world=source_poses[index],
            admitted_camera_to_backend_world=admitted_poses[index],
            phone_to_fixed_refinement=refinement,
            reference_camera_to_world=reference_pose,
            floor_y_m=floor_y_m,
            scene_to_static_transform=scene_to_static,
        )
        point_rows.append(static_points.astype(np.float32, copy=False))
        confidence_rows.append(sampled_confidence.astype(np.float32, copy=False))
        view_rows.append(np.full(static_points.shape[0], index, dtype=np.uint16))

    if not point_rows:
        raise ValueError("phone outputs contain no usable observations")
    return (
        np.concatenate(point_rows, axis=0),
        np.concatenate(confidence_rows),
        np.concatenate(view_rows),
        {
            "outputs_manifest": str(outputs_manifest_path),
            "outputs_manifest_sha256": _sha256(outputs_manifest_path),
            "source_camera_solution": str(source_solution_path),
            "source_camera_solution_sha256": _sha256(source_solution_path),
            "bootstrap_solution": str(bootstrap_path),
            "bootstrap_solution_sha256": _sha256(bootstrap_path),
            "raw_view_count": frame_count,
            "raw_view_bytes": raw_bytes,
            "sample_stride_px": stride,
            "sampled_valid_points": int(sum(row.shape[0] for row in point_rows)),
        },
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-floorplan", type=Path, required=True)
    parser.add_argument("--scene-fusion-manifest", type=Path, required=True)
    parser.add_argument("--registration-npz", type=Path, required=True)
    parser.add_argument("--phone-scan-outputs-dir", type=Path, required=True)
    parser.add_argument("--phone-pose-bootstrap-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--renderer-bundle", type=Path)
    parser.add_argument("--chrome", type=Path)
    parser.add_argument(
        "--focus-roi",
        help="optional min_x,max_x,min_z,max_z metric review crop",
    )
    parser.add_argument("--phone-sample-stride-px", type=int, default=3)
    parser.add_argument("--angular-step-deg", type=float, default=0.30)
    parser.add_argument("--artifact-expansion-m", type=float, default=0.12)
    parser.add_argument("--artifact-min-phone-cells", type=int, default=12)
    parser.add_argument("--artifact-min-phone-fraction", type=float, default=0.70)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    baseline_path = args.baseline_floorplan.resolve()
    manifest_path = args.scene_fusion_manifest.resolve()
    registration_path = args.registration_npz.resolve()
    phone_outputs_dir = args.phone_scan_outputs_dir.resolve()
    phone_bootstrap_path = args.phone_pose_bootstrap_npz.resolve()
    output_dir = args.output_dir.resolve()
    for path in (
        baseline_path,
        manifest_path,
        registration_path,
        phone_bootstrap_path,
    ):
        if not path.is_file():
            raise ValueError(f"required input does not exist: {path}")
    if not phone_outputs_dir.is_dir():
        raise ValueError(f"required phone outputs directory does not exist: {phone_outputs_dir}")
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")
    if (args.renderer_bundle is None) != (args.chrome is None):
        raise ValueError("--renderer-bundle and --chrome must be supplied together")

    baseline_sha256_before = _sha256(baseline_path)
    baseline = _load_json(baseline_path)
    manifest = _load_json(manifest_path)
    if manifest.get("status") != "passed":
        raise ValueError("scene-fusion manifest is not passed")
    quality_gates = manifest.get("quality_gates")
    if not isinstance(quality_gates, Mapping) or not quality_gates:
        raise ValueError("scene-fusion manifest has no quality gates")
    failed_gates = [key for key, value in quality_gates.items() if value is not True]
    if failed_gates:
        raise ValueError(f"scene-fusion quality gates failed: {failed_gates}")
    if baseline.get("camera_id") != manifest.get("camera_id"):
        raise ValueError("baseline and scene fusion name different cameras")

    height = _decode_layer(baseline, "height")
    height_agl = _decode_layer(baseline, "height_agl")
    density = _decode_layer(baseline, "density")
    observed = _decode_layer(baseline, "observed") > 0.5
    structural = _decode_layer(baseline, "structural_height")
    surface_observed = _decode_layer(baseline, "surface_observed") > 0.5
    room_footprint = _decode_layer(baseline, "room_footprint") > 0.5
    wall_support = _decode_layer(baseline, "wall_support")
    x_grid, z_grid = floorplan_grid_centers(baseline["bounds"], height.shape)

    with np.load(registration_path, allow_pickle=False) as loaded:
        if "scene_to_static_transform" not in loaded:
            raise ValueError("registration NPZ lacks scene_to_static_transform")
        scene_to_static = np.asarray(
            loaded["scene_to_static_transform"],
            dtype=np.float64,
        )

    pose_values = manifest.get("reference_camera_to_world_row_major")
    if not isinstance(pose_values, Sequence) or len(pose_values) != 16:
        raise ValueError("scene-fusion manifest has no reference camera pose")
    reference_pose = np.asarray(pose_values, dtype=np.float64).reshape(4, 4)
    floor_y_m = float(manifest.get("floor_y_m"))
    camera_height_m = float(reference_pose[1, 3] - floor_y_m)
    settings = ShadowFillSettings(
        angular_step_deg=float(args.angular_step_deg),
        floor_artifact_expansion_m=float(args.artifact_expansion_m),
        floor_artifact_min_phone_cells=int(args.artifact_min_phone_cells),
        floor_artifact_min_phone_fraction=float(
            args.artifact_min_phone_fraction
        ),
    )

    shadow = compute_static_floor_occlusion_shadow(
        structural_height_m=structural,
        surface_observed=surface_observed,
        raw_height_agl_m=height_agl,
        wall_support=wall_support,
        room_footprint=room_footprint,
        x_grid_m=x_grid,
        z_grid_m=z_grid,
        camera_height_m=camera_height_m,
        settings=settings,
    )
    (
        raw_phone_points,
        raw_phone_confidence,
        raw_phone_views,
        raw_phone_meta,
    ) = _load_original_phone_observations(
        outputs_dir=phone_outputs_dir,
        bootstrap_path=phone_bootstrap_path,
        scene_fusion_manifest=manifest,
        reference_pose=reference_pose,
        floor_y_m=floor_y_m,
        scene_to_static=scene_to_static,
        sample_stride_px=int(args.phone_sample_stride_px),
    )
    floor_visibility = build_phone_floor_visibility_from_views(
        points_static_m=raw_phone_points,
        view_indices=raw_phone_views,
        confidence=raw_phone_confidence,
        room_footprint=room_footprint,
        x_grid_m=x_grid,
        z_grid_m=z_grid,
        settings=settings,
    )
    target_height = np.where(
        observed & np.isfinite(height_agl),
        height_agl,
        0.0,
    ).astype(np.float32, copy=False)
    static_surface_shadow = compute_height_occlusion_shadow(
        occluder_height_m=shadow.occluder_height_m,
        target_height_m=target_height,
        room_footprint=room_footprint,
        x_grid_m=x_grid,
        z_grid_m=z_grid,
        camera_height_m=camera_height_m,
        settings=settings,
    )
    structural_agreement = (
        observed
        & surface_observed
        & np.isfinite(height_agl)
        & np.isfinite(structural)
        & (
            np.abs(height_agl - structural)
            <= float(settings.static_surface_agreement_m)
        )
    )
    reliable_wall_surface = (
        observed
        & np.isfinite(wall_support)
        & (wall_support >= float(settings.wall_support_min))
    )
    reliable_static_surface = structural_agreement | reliable_wall_surface
    radial_artifact = compute_radial_occlusion_artifact(
        raw_height_agl_m=height_agl,
        structural_height_m=structural,
        density=density,
        observed=observed,
        wall_support=wall_support,
        room_footprint=room_footprint,
        floor_shadow=shadow.shadow,
        x_grid_m=x_grid,
        z_grid_m=z_grid,
        settings=settings,
    )
    underlay_masks = build_floor_underlay_masks(
        room_footprint=room_footprint,
        foreground_candidate=shadow.occluder,
        floor_shadow=shadow.shadow,
        radial_artifact_region=radial_artifact.region,
        phone_floor_visibility=floor_visibility.visible,
        density=density,
        wall_support=wall_support,
        x_grid_m=x_grid,
        z_grid_m=z_grid,
        settings=settings,
    )
    result = apply_floor_underlay(
        static_height=height,
        static_height_agl=height_agl,
        static_density=density,
        static_observed=observed,
        room_footprint=room_footprint,
        masks=underlay_masks,
    )
    fusion_authority = result.authority

    output_dir.mkdir(parents=True, mode=0o755)
    arrays_path = output_dir / "floor_underlay_arrays.npz"
    np.savez_compressed(
        arrays_path,
        raw_height=height,
        raw_height_agl=height_agl,
        raw_density=density,
        raw_observed=observed,
        room_footprint=room_footprint,
        occluder=shadow.occluder,
        occluder_height_m=shadow.occluder_height_m,
        static_occlusion_shadow=shadow.shadow,
        static_surface_shadow=static_surface_shadow,
        reliable_static_surface=reliable_static_surface,
        radial_artifact_seed=radial_artifact.seed,
        radial_artifact_region=radial_artifact.region,
        radial_curvature_energy=radial_artifact.radial_curvature_energy,
        radial_alignment=radial_artifact.radial_alignment,
        radial_oscillation_fraction=radial_artifact.oscillation_fraction,
        aligned_floor_visibility=floor_visibility.visible,
        aligned_floor_view_support=floor_visibility.view_support,
        aligned_floor_quality=floor_visibility.quality,
        floor_confirmed_artifact=underlay_masks.floor_confirmed_artifact,
        expanded_floor_artifact=underlay_masks.expanded_artifact,
        removed_floor_artifact=underlay_masks.removed_artifact,
        trusted_foreground=underlay_masks.foreground,
        complete_floor_underlay=underlay_masks.floor,
        artifact_component_id=underlay_masks.artifact_component_id,
        artifact_phone_support=underlay_masks.artifact_phone_support,
        artifact_phone_fraction=underlay_masks.artifact_phone_fraction,
        changed=result.changed,
        authority=fusion_authority,
        completed_height=result.height,
        completed_height_agl=result.height_agl,
        completed_density=result.density,
        completed_observed=result.observed,
        scene_to_static_transform=scene_to_static,
    )

    candidate = copy.deepcopy(baseline)
    height_min, height_max = _finite_range(result.height)
    agl_min, agl_max = _finite_range(result.height_agl)
    candidate["height"] = _layer(result.height, height_min, height_max)
    candidate["height_agl"] = _layer(result.height_agl, max(0.0, agl_min), agl_max)
    candidate["density"] = _layer(result.density, 0.0, 1.0)
    candidate["observed"] = _layer(result.observed.astype(np.float32), 0.0, 1.0)
    candidate["unknown"] = _layer(result.unknown.astype(np.float32), 0.0, 1.0)
    candidate["static_occlusion_shadow"] = _layer(
        shadow.shadow.astype(np.float32), 0.0, 1.0
    )
    candidate["static_surface_shadow"] = _layer(
        static_surface_shadow.astype(np.float32), 0.0, 1.0
    )
    candidate["radial_occlusion_artifact"] = _layer(
        radial_artifact.region.astype(np.float32), 0.0, 1.0
    )
    candidate["aligned_floor_visibility"] = _layer(
        floor_visibility.visible.astype(np.float32), 0.0, 1.0
    )
    candidate["aligned_floor_view_support"] = _layer(
        floor_visibility.view_support.astype(np.float32),
        0.0,
        float(max(1, int(np.max(floor_visibility.view_support)))),
    )
    candidate["aligned_floor_quality"] = _layer(
        floor_visibility.quality, 0.0, 1.0
    )
    candidate["floor_confirmed_artifact"] = _layer(
        underlay_masks.floor_confirmed_artifact.astype(np.float32), 0.0, 1.0
    )
    candidate["removed_floor_artifact"] = _layer(
        underlay_masks.removed_artifact.astype(np.float32), 0.0, 1.0
    )
    candidate["trusted_foreground"] = _layer(
        underlay_masks.foreground.astype(np.float32), 0.0, 1.0
    )
    candidate["complete_floor_underlay"] = _layer(
        underlay_masks.floor.astype(np.float32), 0.0, 1.0
    )
    candidate["source_authority"] = _layer(
        fusion_authority.astype(np.float32), 0.0, 3.0
    )
    candidate["floor_underlay_meta"] = {
        "contract": "noesis.floorplan.complete_floor_underlay.v1",
        "mode": "offline_derived_review_only",
        "raw_layers_modified": False,
        "floor_source": "zero_agl_plane_on_existing_static_room_footprint",
        "alignment": "native_baseline_4cm_grid_without_resampling",
        "layering": "complete_floor_under_trusted_static_foreground",
        "artifact_policy": "radial_components_require_direct_multiview_floor_support",
        "artifact_expansion_m": settings.floor_artifact_expansion_m,
        "floor_height_gate_m": settings.phone_floor_visibility_max_height_m,
        "floor_view_gate": settings.phone_floor_visibility_min_views,
        "phone_coordinate_authority": "passed_joint_inference_pose_bridge_and_refinement",
        "camera_height_m": camera_height_m,
        "settings": settings.__dict__,
    }
    candidate_path = output_dir / "complete_floor_underlay_response.json"
    _write_json(candidate_path, candidate)

    static_values = height[observed & np.isfinite(height)]
    baseline_range = (
        float(np.percentile(static_values, 5.0)),
        float(np.percentile(static_values, 95.0)),
    )
    observed_replaced = int(np.count_nonzero(underlay_masks.floor & observed))
    unknown_filled = int(np.count_nonzero(underlay_masks.floor & ~observed))
    changed_values = np.abs(result.height - height)[result.changed]
    stats = {
        "room_footprint_cells": int(np.count_nonzero(room_footprint)),
        "static_occluder_cells": int(np.count_nonzero(shadow.occluder)),
        "static_shadow_cells": int(np.count_nonzero(shadow.shadow)),
        "static_surface_shadow_cells": int(np.count_nonzero(static_surface_shadow)),
        "radial_artifact_seed_cells": int(np.count_nonzero(radial_artifact.seed)),
        "radial_artifact_region_cells": int(np.count_nonzero(radial_artifact.region)),
        "radial_artifact_energy_threshold": radial_artifact.energy_threshold,
        "aligned_floor_point_count": floor_visibility.floor_point_count,
        "aligned_floor_unique_view_cell_count": (
            floor_visibility.unique_view_cell_count
        ),
        "aligned_floor_visibility_cells": int(
            np.count_nonzero(floor_visibility.visible)
        ),
        "aligned_floor_visibility_in_static_shadow_cells": int(
            np.count_nonzero(floor_visibility.visible & shadow.shadow)
        ),
        "aligned_floor_max_view_support": int(
            np.max(floor_visibility.view_support)
        ),
        "accepted_floor_artifact_components": underlay_masks.accepted_component_count,
        "floor_confirmed_artifact_cells": int(
            np.count_nonzero(underlay_masks.floor_confirmed_artifact)
        ),
        "expanded_floor_artifact_cells": int(
            np.count_nonzero(underlay_masks.expanded_artifact)
        ),
        "removed_artifact_foreground_cells": int(
            np.count_nonzero(underlay_masks.removed_artifact)
        ),
        "removed_artifact_with_direct_phone_floor_cells": int(
            np.count_nonzero(
                underlay_masks.removed_artifact & floor_visibility.visible
            )
        ),
        "trusted_foreground_cells": int(
            np.count_nonzero(underlay_masks.foreground)
        ),
        "complete_floor_underlay_cells": int(
            np.count_nonzero(underlay_masks.floor)
        ),
        "floor_authority_cells": int(np.count_nonzero(underlay_masks.floor)),
        "changed_cells": int(np.count_nonzero(result.changed)),
        "observed_cells_corrected": observed_replaced,
        "unknown_cells_filled": unknown_filled,
        "observed_change_percent": (
            100.0 * observed_replaced / max(1, int(np.count_nonzero(observed)))
        ),
        "room_floor_underlay_coverage_percent": (
            100.0
            * int(np.count_nonzero(underlay_masks.floor))
            / max(1, int(np.count_nonzero(room_footprint)))
        ),
        "median_abs_height_change_m": (
            float(np.median(changed_values)) if changed_values.size else 0.0
        ),
        "p90_abs_height_change_m": (
            float(np.percentile(changed_values, 90.0)) if changed_values.size else 0.0
        ),
    }
    component_rows: list[dict[str, Any]] = []
    component_ids = underlay_masks.artifact_component_id
    for component_id in range(1, int(np.max(component_ids)) + 1):
        component = component_ids == component_id
        if not np.any(component):
            continue
        component_row, component_column = np.where(component)
        component_cells = int(np.count_nonzero(component))
        phone_cells = int(np.count_nonzero(component & floor_visibility.visible))
        component_rows.append(
            {
                "component_id": component_id,
                "accepted": bool(
                    np.any(component & underlay_masks.floor_confirmed_artifact)
                ),
                "cells": component_cells,
                "foreground_cells": int(
                    np.count_nonzero(component & shadow.occluder)
                ),
                "phone_floor_cells": phone_cells,
                "phone_floor_fraction": phone_cells / max(1, component_cells),
                "removed_cells_in_component": int(
                    np.count_nonzero(component & underlay_masks.removed_artifact)
                ),
                "bounds_xz_m": {
                    "min_x": float(np.min(x_grid[component_row, component_column])),
                    "max_x": float(np.max(x_grid[component_row, component_column])),
                    "min_z": float(np.min(z_grid[component_row, component_column])),
                    "max_z": float(np.max(z_grid[component_row, component_column])),
                },
            }
        )
    connectivity = np.ones((3, 3), dtype=bool)
    _, original_foreground_components = ndi.label(
        shadow.occluder,
        structure=connectivity,
    )
    _, retained_foreground_components = ndi.label(
        underlay_masks.foreground,
        structure=connectivity,
    )
    original_foreground_edge = shadow.occluder & ~ndi.binary_erosion(
        shadow.occluder
    )
    edge_outside_cleanup = (
        original_foreground_edge & ~underlay_masks.expanded_artifact
    )
    protected_dense_foreground = (
        shadow.occluder
        & np.isfinite(density)
        & (density > settings.floor_artifact_max_density)
    )
    protected_wall_foreground = (
        shadow.occluder
        & np.isfinite(wall_support)
        & (wall_support >= settings.wall_support_min)
    )
    phone_floor_foreground_conflict = (
        floor_visibility.visible & shadow.occluder
    )
    preserved_phone_floor_conflict = (
        phone_floor_foreground_conflict & underlay_masks.foreground
    )
    removed_phone_floor_conflict = (
        phone_floor_foreground_conflict & underlay_masks.removed_artifact
    )
    stats.update(
        {
            "original_foreground_components": original_foreground_components,
            "retained_foreground_components": retained_foreground_components,
            "protected_dense_foreground_cells": int(
                np.count_nonzero(protected_dense_foreground)
            ),
            "protected_wall_foreground_cells": int(
                np.count_nonzero(protected_wall_foreground)
            ),
            "foreground_edge_cells_outside_cleanup": int(
                np.count_nonzero(edge_outside_cleanup)
            ),
            "phone_floor_foreground_conflict_cells": int(
                np.count_nonzero(phone_floor_foreground_conflict)
            ),
            "phone_floor_conflicts_preserved_as_foreground": int(
                np.count_nonzero(preserved_phone_floor_conflict)
            ),
            "phone_floor_conflicts_removed_as_artifact": int(
                np.count_nonzero(removed_phone_floor_conflict)
            ),
        }
    )
    focus_roi = _parse_roi(args.focus_roi)
    focus_stats: dict[str, Any] | None = None
    if focus_roi is not None:
        focus = (
            (x_grid >= focus_roi[0])
            & (x_grid <= focus_roi[1])
            & (z_grid >= focus_roi[2])
            & (z_grid <= focus_roi[3])
            & room_footprint
        )
        focus_floor = focus & underlay_masks.floor
        _, focus_floor_components = ndi.label(
            focus_floor,
            structure=np.ones((3, 3), dtype=bool),
        )
        focus_stats = {
            "bounds_xz_m": list(focus_roi),
            "cells": int(np.count_nonzero(focus)),
            "shadow_cells": int(np.count_nonzero(focus & shadow.shadow)),
            "aligned_floor_visibility_cells": int(
                np.count_nonzero(focus & floor_visibility.visible)
            ),
            "floor_confirmed_artifact_cells": int(
                np.count_nonzero(focus & underlay_masks.floor_confirmed_artifact)
            ),
            "removed_artifact_foreground_cells": int(
                np.count_nonzero(focus & underlay_masks.removed_artifact)
            ),
            "trusted_foreground_cells": int(
                np.count_nonzero(focus & underlay_masks.foreground)
            ),
            "complete_floor_underlay_cells": int(
                np.count_nonzero(focus_floor)
            ),
            "floor_connected_components": focus_floor_components,
            "floor_height_agl_std_m": float(
                np.std(result.height_agl[focus_floor])
            ),
            "raised_cells_before_gt_0_25m": int(
                np.count_nonzero(focus & (height_agl > 0.25))
            ),
            "raised_cells_after_gt_0_25m": int(
                np.count_nonzero(focus & (result.height_agl > 0.25))
            ),
        }

    report: dict[str, Any] = {
        "contract": "noesis.floorplan.complete_floor_underlay",
        "contract_version": 1,
        "mode": "offline_review_only",
        "runtime_or_dashboard_modified": False,
        "raw_inputs_modified": False,
        "camera_id": baseline.get("camera_id"),
        "policy": {
            "floor_layer": "complete_zero_agl_plane_on_native_static_grid",
            "foreground_layer": "exact_static_occluder_geometry",
            "phone_role": "classify_radial_artifacts_not_paint_floor_pixels",
            "artifact_admission": "component_floor_support_then_bounded_sparse_expansion",
            "layer_order": "floor_below_foreground",
        },
        "inputs": {
            "baseline_floorplan": {
                "path": str(baseline_path),
                "sha256": baseline_sha256_before,
            },
            "scene_fusion_manifest": {
                "path": str(manifest_path),
                "sha256": _sha256(manifest_path),
                "fusion_id": manifest.get("fusion_id"),
            },
            "original_phone_mapanything": raw_phone_meta,
            "registration": {
                "path": str(registration_path),
                "sha256": _sha256(registration_path),
            },
        },
        "settings": settings.__dict__,
        "camera_height_m": camera_height_m,
        "statistics": stats,
        "artifact_components": component_rows,
        "focus_roi": focus_stats,
        "validation": {
            "scene_fusion_quality_gates_all_passed": True,
            "floor_and_foreground_are_disjoint": bool(
                not np.any(underlay_masks.floor & underlay_masks.foreground)
            ),
            "floor_and_foreground_cover_room": bool(
                np.array_equal(
                    underlay_masks.floor | underlay_masks.foreground,
                    room_footprint,
                )
            ),
            "trusted_foreground_height_byte_identical": bool(
                np.array_equal(
                    result.height[underlay_masks.foreground],
                    height[underlay_masks.foreground],
                    equal_nan=True,
                )
            ),
            "trusted_foreground_agl_byte_identical": bool(
                np.array_equal(
                    result.height_agl[underlay_masks.foreground],
                    height_agl[underlay_masks.foreground],
                    equal_nan=True,
                )
            ),
            "trusted_foreground_density_byte_identical": bool(
                np.array_equal(
                    result.density[underlay_masks.foreground],
                    density[underlay_masks.foreground],
                    equal_nan=True,
                )
            ),
            "outside_room_byte_identical": bool(
                np.array_equal(
                    result.height[~room_footprint],
                    height[~room_footprint],
                    equal_nan=True,
                )
            ),
            "removed_artifact_subset_of_original_foreground": bool(
                np.all(
                    ~underlay_masks.removed_artifact | shadow.occluder
                )
            ),
            "removed_artifact_inside_confirmed_expansion": bool(
                np.all(
                    ~underlay_masks.removed_artifact
                    | underlay_masks.expanded_artifact
                )
            ),
            "removed_artifact_has_no_wall_support": bool(
                not np.any(
                    underlay_masks.removed_artifact
                    & (wall_support >= settings.wall_support_min)
                )
            ),
            "all_dense_foreground_is_retained": bool(
                np.all(~protected_dense_foreground | underlay_masks.foreground)
            ),
            "all_wall_foreground_is_retained": bool(
                np.all(~protected_wall_foreground | underlay_masks.foreground)
            ),
            "foreground_edge_outside_cleanup_is_retained": bool(
                np.all(~edge_outside_cleanup | underlay_masks.foreground)
            ),
            "foreground_component_count_is_preserved": bool(
                original_foreground_components == retained_foreground_components
            ),
            "phone_floor_foreground_conflicts_are_fully_partitioned": bool(
                np.array_equal(
                    preserved_phone_floor_conflict | removed_phone_floor_conflict,
                    phone_floor_foreground_conflict,
                )
                and not np.any(
                    preserved_phone_floor_conflict & removed_phone_floor_conflict
                )
            ),
            "accepted_artifacts_meet_phone_fraction_gate": bool(
                np.all(
                    ~underlay_masks.floor_confirmed_artifact
                    | (
                        underlay_masks.artifact_phone_fraction
                        >= settings.floor_artifact_min_phone_fraction
                    )
                )
            ),
            "floor_underlay_zero_agl": bool(
                np.all(result.height_agl[underlay_masks.floor] == 0.0)
            ),
            "floor_underlay_is_complete_and_observed": bool(
                np.all(result.observed[underlay_masks.floor])
            ),
            "raw_payload_unchanged": _sha256(baseline_path)
            == baseline_sha256_before,
        },
        "artifacts": {
            "candidate_floorplan": candidate_path.name,
            "arrays": arrays_path.name,
        },
    }
    if focus_stats is not None:
        report["validation"].update(
            {
                "focus_floor_is_one_connected_surface": (
                    focus_stats["floor_connected_components"] == 1
                ),
                "focus_floor_has_zero_height_variance": (
                    focus_stats["floor_height_agl_std_m"] == 0.0
                ),
                "focus_raised_artifact_cells_are_reduced": (
                    focus_stats["raised_cells_after_gt_0_25m"]
                    < focus_stats["raised_cells_before_gt_0_25m"]
                ),
            }
        )
    report_path = output_dir / "report.json"
    _write_json(report_path, report)

    before = _label(
        _height_image(height, density, locked_range=baseline_range),
        "Original static dashboard height",
        "Stored 4 cm grid; locked original 5th/95th percentile range",
    )
    after = _label(
        _height_image(result.height, result.density, locked_range=baseline_range),
        "Complete floor beneath trusted static foreground",
        f"Native 4 cm grid; {stats['removed_artifact_foreground_cells']:,} artifact cells removed",
    )
    before.save(output_dir / "01_original_static_floorplan.png")
    after.save(output_dir / "02_complete_floor_underlay.png")
    _label(
        _authority_image(fusion_authority, underlay_masks.foreground),
        "Disjoint floor and foreground authority",
        "green complete floor; gray exact static foreground; red retained foreground boundary",
    ).save(output_dir / "03_visibility_authority.png")
    _label(
        _artifact_evidence_image(
            phone_floor=floor_visibility.visible,
            floor_confirmed_artifact=underlay_masks.floor_confirmed_artifact,
            expanded_artifact=underlay_masks.expanded_artifact,
            removed_artifact=underlay_masks.removed_artifact,
            trusted_foreground=underlay_masks.foreground,
        ),
        "Artifact cleanup evidence",
        "blue direct phone floor; orange accepted radial component; green removed comb; red retained boundary",
    ).save(output_dir / "04_artifact_cleanup_evidence.png")
    _combine_horizontal([before, after]).save(output_dir / "05_before_after_locked.png")
    if focus_roi is not None:
        _focus_image(
            height,
            result.height,
            density,
            result.density,
            x_grid,
            z_grid,
            focus_roi,
            baseline_range,
        ).save(output_dir / "06_focus_before_after.png")

    if args.renderer_bundle is not None and args.chrome is not None:
        renderer_source = args.renderer_bundle.resolve()
        chrome = args.chrome.resolve()
        if not renderer_source.is_file() or not chrome.is_file():
            raise ValueError("renderer bundle and Chrome executable must exist")
        shutil.copyfile(renderer_source, output_dir / "renderer.bundle.js")
        review_stats = {
            "changed_cells": stats["changed_cells"],
            "observed_cells_corrected": observed_replaced,
            "unknown_cells_filled": unknown_filled,
            "observed_change_percent": stats["observed_change_percent"],
        }
        review_data = {
            "boundsAspect": (
                (float(baseline["bounds"]["max_x"]) - float(baseline["bounds"]["min_x"]))
                / (float(baseline["bounds"]["max_z"]) - float(baseline["bounds"]["min_z"]))
            ),
            "gridShape": [int(height.shape[0]), int(height.shape[1])],
            "baselineRange": list(baseline_range),
            "candidates": [
                {
                    "id": "baseline",
                    "label": "Original dashboard BEV",
                    "description": "Unmodified stored static-camera floorplan response.",
                    "height": baseline["height"],
                    "density": baseline["density"],
                    "stats": {
                        "changed_cells": 0,
                        "observed_cells_corrected": 0,
                        "unknown_cells_filled": 0,
                        "observed_change_percent": 0.0,
                    },
                },
                {
                    "id": "complete_floor_underlay",
                    "label": "Complete floor under trusted foreground",
                    "description": "A native-grid floor plane is layered beneath byte-preserved static furniture geometry.",
                    "height": candidate["height"],
                    "density": candidate["density"],
                    "stats": review_stats,
                },
            ],
        }
        (output_dir / "review_data.js").write_text(
            "window.REVIEW_DATA = "
            + json.dumps(review_data, separators=(",", ":"), allow_nan=False)
            + ";\n",
            encoding="utf-8",
        )
        index_path = output_dir / "index.html"
        index_path.write_text(_review_html(), encoding="utf-8")
        exact_before = output_dir / "07_exact_renderer_original_locked.png"
        exact_after = output_dir / "08_exact_renderer_completed_locked.png"
        exact_dynamic = output_dir / "09_exact_renderer_completed_dashboard_dynamic.png"
        _run_chrome_screenshot(
            chrome,
            index_path,
            exact_before,
            candidate="baseline",
            scale="locked",
        )
        _run_chrome_screenshot(
            chrome,
            index_path,
            exact_after,
            candidate="complete_floor_underlay",
            scale="locked",
        )
        _run_chrome_screenshot(
            chrome,
            index_path,
            exact_dynamic,
            candidate="complete_floor_underlay",
            scale="dynamic",
        )
        _combine_horizontal(
            [Image.open(exact_before).convert("RGB"), Image.open(exact_after).convert("RGB")],
            gap=8,
        ).save(output_dir / "00_exact_before_after_locked.png")
        report["artifacts"].update(
            {
                "exact_before_after": "00_exact_before_after_locked.png",
                "exact_original_locked": exact_before.name,
                "exact_completed_locked": exact_after.name,
                "exact_completed_dashboard_dynamic": exact_dynamic.name,
                "browser_review": "index.html",
            }
        )
        _write_json(report_path, report)

    readme_lines = [
        "Noesis complete floor underlay and trusted foreground composite",
        "",
        "This directory is an offline review artifact. It did not modify the live runtime, dashboard, or raw inputs.",
        "",
        "Authority policy:",
        "- the floor is one complete 0 m AGL plane on the original dashboard's native 4 cm grid",
        "- trusted static furniture is layered above it and remains byte-identical",
        "- the floor and furniture masks are disjoint, so floor can never paint through a couch",
        "- aligned three-view phone floor is used only to classify radial comb artifacts",
        f"- accepted artifact components expand at most {settings.floor_artifact_expansion_m:.2f} m through sparse non-wall shadow cells",
        "- the passed joint inference supplies the admitted phone-to-static coordinate bridge",
        "",
        f"Aligned floor-visibility cells: {stats['aligned_floor_visibility_cells']}",
        f"Complete floor cells: {stats['complete_floor_underlay_cells']}",
        f"Trusted foreground cells: {stats['trusted_foreground_cells']}",
        f"Removed radial-artifact foreground cells: {stats['removed_artifact_foreground_cells']}",
        "",
        "Open index.html for the exact dashboard-renderer comparison when present.",
    ]
    (output_dir / "README.txt").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "statistics": stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
