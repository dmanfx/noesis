#!/usr/bin/env python3
"""Offline image-identified wall review using existing admitted PCF points.

Never changes calibration, runtime selection or structural admission. Recipe
rectangles identify surfaces in exact retained rasters, not authored geometry.
Outputs are review evidence, including rejected patches and measured tilt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from noesis.calibration.structural_carrier_evidence import _alignment, _consensus
from noesis.calibration.structural_registration import StructuralRegistrationError, read_pcf_points_glb
from noesis.calibration.structural_roi_planes import (
    estimate_admitted_normals, fit_supported_region_plane, grow_connected_plane_support, line_at_observed_height,
    select_admitted_region_support,
)


def reference(path: Path) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size}


def read_json(path: Path) -> dict:
    if path.stat().st_size > 2 * 1024 * 1024:
        raise StructuralRegistrationError("review JSON exceeds its bound")
    return json.loads(path.read_text())


def load_view(path: Path) -> dict:
    if path.stat().st_size > 16 * 1024 * 1024:
        raise StructuralRegistrationError("carrier NPZ exceeds bound")
    with ZipFile(path) as archive:
        if len(archive.infolist()) > 32 or sum(v.file_size for v in archive.infolist()) > 32 * 1024 * 1024:
            raise StructuralRegistrationError("carrier arrays exceed bound")
    with np.load(path, allow_pickle=False) as raw:
        result = {key: raw[key] for key in ("model_rgb", "world_points", "confidence", "mask",
                                          "source_selection", "cross_model_uncertain")}
    rgb = result["model_rgb"]
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or max(rgb.shape[:2]) > 8192:
        raise StructuralRegistrationError("invalid bounded carrier raster")
    shape = rgb.shape[:2]
    if (result["world_points"].shape != (*shape, 3)
            or any(result[key].shape != shape for key in result if key not in ("world_points", "model_rgb"))):
        raise StructuralRegistrationError("carrier geometry/quality differs from raster")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-manifest", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grow-connected", action="store_true",
                        help="review connected PCF support around the fixed observed seed planes")
    parser.add_argument("--reference-image", type=Path,
                        help="exact retained fixed-camera keyframe, for projection review only")
    parser.add_argument("--reference-alignment-report", type=Path,
                        help="retained camera-alignment report bound by the source bundle")
    args = parser.parse_args()
    if args.output.exists():
        raise StructuralRegistrationError("refusing to replace existing review evidence")
    prior = read_json(args.prior_manifest)
    bundle_path = args.source_bundle / "bundle_manifest.json"
    if reference(bundle_path)["sha256"] != prior["source"]["bundle_manifest_sha256"]:
        raise StructuralRegistrationError("source bundle differs from admitted prior")
    bundle = read_json(bundle_path)
    # Verify the small, explicit source inventory, not a recursive media tree.
    rows = bundle.get("files", bundle.get("artifacts"))
    if not isinstance(rows, list) or len(rows) > 16:
        raise StructuralRegistrationError("unsupported bounded bundle inventory")
    for row in rows:
        relative = row.get("relative_path", row.get("path"))
        if (not isinstance(relative, str) or Path(relative).is_absolute()
                or ".." in Path(relative).parts):
            raise StructuralRegistrationError("invalid bundle inventory path")
        ref = reference(args.source_bundle / relative)
        if ref["sha256"] != row["sha256"] or ref["size_bytes"] != row["size_bytes"]:
            raise StructuralRegistrationError("source bundle inventory bytes differ")
    source_path = args.source_bundle / "evaluation/static_world_point_cloud_manifest.json"
    source = read_json(source_path)
    alignment = _alignment(source)
    surfel_path = Path(source["source"])
    carrier = surfel_path.parent
    consensus_path = carrier / "consensus_manifest.json"
    consensus = read_json(consensus_path)
    recipe = read_json(args.recipe)
    if recipe.get("schema") != "noesis.review.pcf_image_wall_regions.v1":
        raise StructuralRegistrationError("unsupported review recipe")
    regions = recipe["regions"]
    if not isinstance(regions, list) or not 1 <= len(regions) <= 12:
        raise StructuralRegistrationError("wall region count exceeds bound")
    _consensus(consensus, {f'{r["view_index"]:04d}': None for r in regions})
    artifact = next(v for v in prior["artifacts"] if v["role"] == "points_glb")
    glb_path = args.prior_manifest.parent / artifact["relative_path"]
    glb_ref = reference(glb_path)
    if glb_ref["sha256"] != artifact["sha256"] or glb_ref["size_bytes"] != artifact["size_bytes"]:
        raise StructuralRegistrationError("admitted PCF differs from its manifest")
    admitted = read_pcf_points_glb(glb_path)
    local_normals = estimate_admitted_normals(admitted) if args.grow_connected else None
    with np.load(surfel_path, allow_pickle=False) as raw:
        surfels = raw["points"]
    if surfels.shape != (source["source_point_count"], 3):
        raise StructuralRegistrationError("carrier surfel count/shape differs")
    aligned_surfels = surfels @ alignment[:3, :3].T + alignment[:3, 3]
    distance, _ = cKDTree(aligned_surfels).query(admitted, workers=1)
    if float(distance.max()) > 2e-5:
        raise StructuralRegistrationError("admitted PCF does not replay from retained carrier transform")
    args.output.mkdir(parents=True)
    report = {"schema": "noesis.review.pcf_image_wall_planes.v1", "status": "review_only",
              "runtime_changed": False, "structural_mapping_admitted": False,
              "prior": reference(args.prior_manifest), "pcf": glb_ref,
              "source_bundle": reference(bundle_path), "source_manifest": reference(source_path),
              "carrier_consensus": reference(consensus_path), "carrier_surfels": reference(surfel_path),
              "recipe": reference(args.recipe), "alignment": source["alignment"],
              "carrier_to_admitted_max_distance_m": float(distance.max()), "regions": [],
              "limitations": ["Retained views share one fused reconstruction; they are not independent physical ground truth.",
                              "Extracting a plane does not validate an authored counterpart or admit a structural mapping."]}
    support_by_surface = {}
    plot_points = {}
    connected_points = {}
    for index, region in enumerate(regions):
        path = carrier / "raw" / f'view_{region["view_index"]:04d}.npz'
        raw = load_view(path)
        box = region["box_xyxy"]
        if len(box) != 4 or any(type(v) is not int for v in box):
            raise StructuralRegistrationError("region must use integer raster bounds")
        x0, y0, x1, y1 = box
        h, w = raw["model_rgb"].shape[:2]
        if not 0 <= x0 < x1 <= w or not 0 <= y0 < y1 <= h:
            raise StructuralRegistrationError("region lies outside exact raster")
        roi = np.s_[y0:y1, x0:x1]
        xyz = raw["world_points"][roi].reshape(-1, 3)
        good = ((raw["confidence"][roi].reshape(-1) >= .70)
                & raw["mask"][roi].reshape(-1).astype(bool)
                & (raw["source_selection"][roi].reshape(-1) == 1)
                & ~raw["cross_model_uncertain"][roi].reshape(-1).astype(bool)
                & np.isfinite(xyz).all(axis=1))
        observed = xyz[good] @ alignment[:3, :3].T + alignment[:3, 3]
        indices = (select_admitted_region_support(admitted, observed,
                                                  maximum_distance_m=recipe["association_distance_m"])
                   if len(observed) >= 3 else np.empty(0, dtype=int))
        measured = admitted[indices]
        result = {**region, "view": reference(path), "quality_point_count": len(observed),
                  "selected_admitted_count": len(indices)}
        if len(indices) >= 3:
            result["plane"] = fit_supported_region_plane(measured)
            support_by_surface.setdefault(region["surface"], []).append(indices)
            plot_points[f'{region["surface"]}-{region["view_index"]}'] = measured
        else:
            result["plane"] = {"extraction_support_passed": False,
                               "reason_codes": ["quality_support" if len(observed) < 3 else "support_count"]}
        report["regions"].append(result)
        image = Image.fromarray(raw["model_rgb"])
        draw = ImageDraw.Draw(image)
        draw.rectangle((x0, y0, x1-1, y1-1), outline="red", width=2)
        image.save(args.output / f"region-{index:02d}-view-{region['view_index']:04d}.png")
    pooled = np.unique(np.concatenate([v for rows in support_by_surface.values() for v in rows])) if support_by_surface else []
    reference_y = float(np.median(admitted[pooled, 1])) if len(pooled) else None
    report["common_observed_reference_y_m"] = reference_y
    report["surfaces"] = []
    for name, rows in support_by_surface.items():
        measured = admitted[np.unique(np.concatenate(rows))]
        plane = fit_supported_region_plane(measured)
        row = {"surface": name, "plane": plane}
        if args.grow_connected:
            indices = grow_connected_plane_support(admitted, np.unique(np.concatenate(rows)), plane,
                                                    local_normals=local_normals)
            row["connected_support_count"] = len(indices)
            if len(indices) >= 3:
                grown = fit_supported_region_plane(admitted[indices])
                row["connected_plane"] = grown
                if grown["review_candidate_passed"]:
                    row["connected_segment"] = line_at_observed_height(grown, segment_id=name,
                                                                          reference_y_m=reference_y)
                plot_points[name+"-connected"] = admitted[indices]
                connected_points[name] = admitted[indices]
        if plane["review_candidate_passed"]:
            row["measured_segment"] = line_at_observed_height(plane, segment_id=name,
                                                              reference_y_m=reference_y)
        report["surfaces"].append(row)
    # Corresponding RGB regions and actual admitted support, without any target fit.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(admitted[::8, 0], admitted[::8, 2], s=.3, color="lightgray", label="admitted PCF")
    for name, points in plot_points.items():
        ax.scatter(points[::3, 0], points[::3, 2], s=2, label=name)
    ax.set(xlabel="backend world X (m)", ylabel="backend world Z (m)",
           title="Image-identified support in exact admitted Kitchen PCF (not a model fit)")
    ax.set_aspect("equal"); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(args.output / "admitted-region-support-topdown.png", dpi=160)
    plt.close(fig)
    report["plane_pair_angles_deg"] = []
    usable = [row for row in report["surfaces"]
              if row.get("connected_plane", {}).get("review_candidate_passed")]
    for i, first in enumerate(usable):
        for second in usable[i+1:]:
            n1 = np.asarray(first["connected_plane"]["normal_xyz"])[[0, 2]]
            n2 = np.asarray(second["connected_plane"]["normal_xyz"])[[0, 2]]
            angle = float(np.degrees(np.arccos(np.clip(abs(n1@n2)/(np.linalg.norm(n1)*np.linalg.norm(n2)), 0, 1))))
            report["plane_pair_angles_deg"].append({"surfaces": [first["surface"], second["surface"]],
                                                     "acute_angle_deg": angle})
    if args.reference_image:
        source_reference = read_json(args.source_bundle / "reference.json")["noesis_reference"]
        metadata_path = args.source_bundle / source_reference["revision"] / "room_points_meta.json"
        metadata = read_json(metadata_path)
        alignment_report = read_json(args.source_bundle / "alignment/alignment_report.json")
        if not args.reference_alignment_report:
            raise StructuralRegistrationError("projection needs its exact retained source alignment report")
        alignment_ref = reference(args.reference_alignment_report)
        if alignment_ref["sha256"] != alignment_report["source_alignment"]["sha256"]:
            raise StructuralRegistrationError("projection alignment report differs from source binding")
        source_alignment_report = read_json(args.reference_alignment_report)
        image_ref = reference(args.reference_image)
        if image_ref["sha256"] != source_alignment_report["inputs"]["fixed_camera_keyframe_sha256"]:
            raise StructuralRegistrationError("projection image is not the bound static keyframe")
        with Image.open(args.reference_image) as original:
            image = np.asarray(original.convert("RGB"))
        if max(image.shape[:2]) > 8192:
            raise StructuralRegistrationError("projection image exceeds raster bound")
        E = np.asarray(metadata["extrinsics_col_major"]).reshape(4, 4, order="F")
        K = np.asarray(metadata["intrinsics"])
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.imshow(image)
        colors = {"stone_facade": "lime", "portrait_wall": "magenta", "tv_wall": "cyan"}
        projection_counts = {}
        for name, points in connected_points.items():
            # Rejected planes stay in JSON, not on a successful-evidence overlay.
            if not any(row["surface"] == name for row in usable):
                continue
            camera = points@E[:3, :3].T+E[:3, 3]
            valid = camera[:, 2] > .01
            projected = camera[valid]@K.T
            uv = projected[:, :2]/projected[:, 2:3]
            visible = ((uv[:, 0] >= 0) & (uv[:, 0] < image.shape[1])
                       & (uv[:, 1] >= 0) & (uv[:, 1] < image.shape[0]))
            uv = uv[visible]
            projection_counts[name] = len(uv)
            ax.scatter(uv[:, 0], uv[:, 1], s=1, alpha=.35,
                       color=colors.get(name), label=name+" admitted PCF support")
        ax.set_title("Exact admitted PCF plane support projected through retained fixed-camera K/E\n"
                     "Review evidence, not independent physical accuracy or an active model fit")
        ax.legend(loc="lower left"); ax.set_xlim(0, image.shape[1]); ax.set_ylim(image.shape[0], 0)
        ax.axis("off"); fig.tight_layout()
        fig.savefig(args.output / "fixed-camera-plane-support.png", dpi=160); plt.close(fig)
        report["fixed_camera_projection"] = {"image": image_ref, "metadata": reference(metadata_path),
                                              "source_alignment_report": alignment_ref,
                                              "visible_point_counts": projection_counts,
                                              "occlusion_tested": False}
    (args.output / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "carrier_replay_max_m": float(distance.max()),
                      "surfaces": report["surfaces"]}, indent=2))


if __name__ == "__main__":
    main()
