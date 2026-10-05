#!/usr/bin/env python3
"""Build a revision-bound parallel floor patch from measured rise and imagery.

The rise is user/survey evidence, never a tracking constant. Image annotations
locate the finite patch; the admitted PCF bounds its reconstruction coverage.
Neither person tracks nor phone optical-center positions enter this calculation.
The existing calibrated lower plane remains authoritative outside the patch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from noesis_core.contracts.support_surface import SupportSurface, SupportSurfaceSet


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_polygon(camera: dict, binding: dict, pixels, offset: float) -> np.ndarray:
    world_from_camera = (
        np.asarray(binding["target_from_calibration_col_major"]).reshape(4, 4, order="F")
        @ np.linalg.inv(np.asarray(camera["E"]).reshape(4, 4, order="F"))
    )
    normal = np.asarray(binding["world_floor_plane"]["normal"], dtype=float)
    fx, fy, cx, cy = camera["K"]
    pixels = np.asarray(pixels, dtype=float)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or not 3 <= len(pixels) <= 64:
        raise ValueError("finite floor polygon needs 3 to 64 image vertices")
    rays = np.c_[(pixels[:, 0] - cx) / fx, (pixels[:, 1] - cy) / fy, np.ones(len(pixels))]
    rays = rays @ world_from_camera[:3, :3].T
    origin = world_from_camera[:3, 3]
    denominator = rays @ normal
    if np.any(abs(denominator) < 1e-6):
        raise ValueError("floor boundary intersects the camera horizon")
    distance = -(normal @ origin + offset) / denominator
    points = origin + distance[:, None] * rays
    if not np.isfinite(points).all() or np.any(distance <= 0):
        raise ValueError("floor boundary has invalid or behind-camera intersections")
    return points


def convex_polygon(points: np.ndarray) -> np.ndarray:
    # This only orders an already-convex annotation. Never silently fill a
    # concave room/obstacle cutout using its convex hull.
    hull = ConvexHull(points)
    if len(hull.vertices) != len(points):
        raise ValueError("annotated floor polygon must be convex")
    return points[hull.vertices]


def inside(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    admitted = np.ones(len(points), dtype=bool)
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge, delta = b - a, points - a
        admitted &= edge[0] * delta[:, 1] - edge[1] * delta[:, 0] >= -1e-8
    return admitted


def clip_convex(subject: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    result = list(subject)
    for a, b in zip(bounds, np.roll(bounds, -1, axis=0)):
        edge = b - a
        def side(p):
            delta = p - a
            return edge[0] * delta[1] - edge[1] * delta[0]
        previous, result_next = result[-1], []
        for current in result:
            sp, sc = side(previous), side(current)
            if (sp >= 0) != (sc >= 0):
                result_next.append(previous + (current - previous) * sp / (sp - sc))
            if sc >= 0:
                result_next.append(current)
            previous = current
        if len(result_next) < 3:
            raise ValueError("floor annotation has no bounded PCF overlap")
        result = result_next
    return np.asarray(result)


def build(measurement_path: Path) -> tuple[SupportSurfaceSet, dict]:
    measurement = json.loads(measurement_path.read_text())
    paths = {name: ROOT / measurement[name] for name in (
        "calibration_path", "source_image_path", "pcf_manifest_path", "pcf_grid_path")}
    calibration = json.loads(paths["calibration_path"].read_text())
    manifest = json.loads(paths["pcf_manifest_path"].read_text())
    binding = calibration["frame_binding"]
    camera_id = measurement["camera_id"]
    if (measurement["floor_model"] != "upper_parallel_to_calibrated_lower"
            or measurement["pixel_space"] != "rectified_source_image"):
        raise ValueError("explicit parallel-plane and rectified-raster evidence required")
    if manifest["prior_id"] != binding["world_frame"]["revision"]:
        raise ValueError("PCF and image calibration world revisions differ")
    if manifest["preview"]["camera_calibration"]["sha256"] != binding["camera_calibration_sha256"]:
        raise ValueError("PCF and image calibration fingerprints differ")
    if manifest["preview"]["reference_camera_id"] != camera_id:
        raise ValueError("PCF and source camera differ")
    grid_artifact = next(row for row in manifest["artifacts"] if row["role"] == "grid_npz")
    if digest(paths["pcf_grid_path"]) != grid_artifact["sha256"]:
        raise ValueError("PCF grid fingerprint differs from its admitted manifest")
    from PIL import Image
    if list(Image.open(paths["source_image_path"]).size) != measurement["image_size"]:
        raise ValueError("annotation raster differs from source image")
    rises = np.asarray([row["floor_to_floor_rise_m"] for row in measurement["rise_observations"]])
    if len(rises) < 2 or not np.isfinite(rises).all() or np.any(rises <= 0):
        raise ValueError("positive finite floor-to-floor measurements required")
    if np.ptp(rises) > 1e-9:
        raise ValueError("unequal endpoint rises need explicit measurement review, not a fitted slope")
    rise = float(np.mean(rises))
    normal = np.asarray(binding["world_floor_plane"]["normal"])
    lower_offset = float(binding["world_floor_plane"]["offset_m"])
    # Rise is vertical floor-to-floor separation. Parallel planes keep exactly
    # the calibrated normal; this is not a learned tilt or a changed map.
    upper_offset = lower_offset - normal[1] * rise
    camera = calibration["cameras"][camera_id]
    upper_world = project_polygon(camera, binding, measurement["upper_floor_extent_pixels"], upper_offset)
    annotation_polygon = convex_polygon(upper_world[:, [0, 2]])
    grid_spec = manifest["grid"]
    with np.load(paths["pcf_grid_path"], allow_pickle=False) as grid:
        rows, columns = np.where(grid["observed"] > 0)
        resolution = float(grid_spec["resolution_m"])
        points = np.c_[grid_spec["bounds"]["min_x"] + (columns + .5) * resolution,
                       grid_spec["bounds"]["min_z"] + (rows + .5) * resolution]
        coverage = points[ConvexHull(points).vertices]
        polygon = clip_convex(annotation_polygon, coverage)
        lower_world = project_polygon(camera, binding, measurement["lower_reference_patch_pixels"], lower_offset)
        local = inside(points, convex_polygon(lower_world[:, [0, 2]]))
        local &= grid["floor_support_count"][rows, columns] > 0
        heights = grid["floor_height_m"][rows[local], columns[local]]
        lower_expected = -(normal[0] * points[local, 0] + normal[2] * points[local, 1] + lower_offset) / normal[1]
        residuals = heights - lower_expected
        residuals = residuals[np.isfinite(residuals)]
        if len(residuals) < 3:
            raise ValueError("lower reference patch lacks retained floor-height evidence")
        sigma = float(np.sqrt(np.mean(residuals ** 2)))
        if not sigma > 0:
            raise ValueError("lower plane residual cannot be established from quantized zero data")
        support_count = int(np.count_nonzero(inside(points, polygon)))
    hashes = {key: digest(path) for key, path in paths.items()}
    hashes["measurement"] = digest(measurement_path)
    evidence_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    surface = SupportSurface(
        surface_id=measurement["surface_id"], revision=evidence_hash,
        normal_world=tuple(normal), offset_m=upper_offset,
        extent_xz=tuple(map(tuple, polygon)), residual_sigma_m=sigma,
        source_observation_count=support_count,
        source_kind="measured_rise_parallel_plane_visible_pcf_extent",
        source_artifact_sha256=evidence_hash,
    )
    result = SupportSurfaceSet(
        camera_id=camera_id, world_frame_revision=binding["world_frame"]["revision"],
        calibration_revision=binding["camera_calibration_sha256"],
        world_transform_sha256=binding["transform_sha256"], evidence_sha256=evidence_hash,
        authority="measured_calibration", surfaces=(surface,),
    )
    review = {"inputs": measurement, "input_sha256": hashes, "measured_rise_m": rise,
              "unclipped_upper_extent_world_m": upper_world.tolist(),
              "lower_residual_sample_count": len(residuals), "lower_local_rms_m": sigma,
              "pcf_covered_cells": support_count, "support_surfaces": result.model_dump(),
              "limits": ["Local PCF floor scatter is propagated, not an absolute calibration error bound.",
                         "Two equal tape readings specify rise and parallelism, not survey precision.",
                         "Visible floor perimeter is bounded by PCF coverage; no whole-map shift or slope fit.",
                         "Lower calibrated floor remains unchanged outside this observed upper patch."]}
    return result, review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    result, review = build(args.measurements)
    for path, payload in ((args.output, result.model_dump()), (args.review, review)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
        measurement = review["inputs"]
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        axes[0].imshow(Image.open(ROOT / measurement["source_image_path"]))
        for key, color in (("upper_floor_extent_pixels", "cyan"), ("lower_reference_patch_pixels", "yellow")):
            poly = np.asarray(measurement[key])
            closed = np.r_[poly, poly[:1]]
            axes[0].plot(closed[:, 0], closed[:, 1], color=color, linewidth=2)
        axes[0].set_title("Observed foyer perimeter; yellow: lower reference patch")
        axes[0].axis("off")
        manifest = json.loads((ROOT / measurement["pcf_manifest_path"]).read_text())
        spec = manifest["grid"]
        bounds = spec["bounds"]
        with np.load(ROOT / measurement["pcf_grid_path"], allow_pickle=False) as grid:
            axes[1].imshow(grid["observed"], origin="lower", cmap="gray_r", interpolation="nearest",
                           extent=(bounds["min_x"], bounds["max_x"], bounds["min_z"], bounds["max_z"]))
        poly = np.asarray(result.surfaces[0].extent_xz)
        axes[1].fill(poly[:, 0], poly[:, 1], color="cyan", alpha=.35)
        closed = np.r_[poly, poly[:1]]
        axes[1].plot(closed[:, 0], closed[:, 1], color="teal")
        axes[1].set_title(f"PCF metric frame: measured parallel rise {review['measured_rise_m']:.4f} m")
        axes[1].set(xlabel="World X (m)", ylabel="World Z (m)", aspect="equal")
        fig.tight_layout()
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=140, bbox_inches="tight")
        plt.close(fig)
    print(json.dumps({"surface": result.surfaces[0].surface_id,
                      "rise_m": review["measured_rise_m"], "pcf_cells": review["pcf_covered_cells"]}))


if __name__ == "__main__":
    main()
