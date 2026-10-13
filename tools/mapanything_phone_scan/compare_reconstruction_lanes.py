#!/usr/bin/env python3
"""CPU-only, equal-budget static/withheld comparison of two saved PCF lanes.

All paths are explicit. Reuses the supplied compare_cross_walk.py helpers;
does not fit geometry, infer depth, admit a Scene Prior, or control runtime.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "2")

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.mapanything_phone_scan.alignment import _project_depth_grid
from tools.mapanything_phone_scan.evaluate_mapanything_prior_variants import _load_target_points
from tools.mapanything_phone_scan.run_mapanything_prior_variants import _load_static_reference

POINT_BUDGET = 600_000
STATIC_CELL_PX = 8
HELDOUT_CELL_PX = 4
LANES = ("baseline", "candidate")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _identity(path: Path) -> dict[str, Any]:
    """Hash an explicitly named artifact, never inventory a directory tree."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def _artifact(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError(f"Artifact escapes its declared input directory: {value}")
    return path


def _load_helpers(path: Path):
    spec = importlib.util.spec_from_file_location("pcf_comparison_reference_helpers", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load reference helpers: {path}")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    for name in ("exact_cloud", "depth_comparison", "error_tails",
                 "masked_depth_cells", "load_heldouts"):
        if not callable(getattr(helpers, name, None)):
            raise ValueError(f"Reference helpers lack {name}")
    return helpers


class _BoundRawViews:
    """Give the unchanged sampler only the alignment's exact named views."""

    def __init__(self, paths: list[Path]):
        self.paths = paths

    def glob(self, pattern: str):
        if pattern != "view_*.npz":
            raise ValueError("Only the declared raw view set may be sampled")
        return iter(self.paths)


def _load_lane(consensus: Path, alignment: Path) -> dict[str, Any]:
    fusion_path = consensus / "consensus_manifest.json"
    source_path = consensus / "scan_outputs_manifest.json"
    transform_path = alignment / "phone_ma_to_noesis_world.json"
    report_path = alignment / "alignment_report.json"
    fusion, source = _json(fusion_path), _json(source_path)
    transform, report = _json(transform_path), _json(report_path)
    if (report.get("status") != "passed"
            or report.get("quality_gate", {}).get("passed") is not True
            or any(value is not True for value in report["quality_gate"].get("checks", {}).values())):
        raise ValueError(f"Comparison requires a passed alignment: {alignment}")
    if report.get("transform") != transform:
        raise ValueError(f"Alignment report and transform disagree: {alignment}")
    target_binding = transform.get("target_binding")
    if not isinstance(target_binding, dict) or not target_binding:
        raise ValueError(f"Alignment has no target binding: {alignment}")
    if (transform.get("source_coordinate_frame") != source.get("coordinate_frame")
            or not source.get("coordinate_frame")
            or transform.get("target_coordinate_frame") != "backend_world_m_stream_points"):
        raise ValueError(f"Alignment coordinate frames differ from source/target: {alignment}")
    matrix = np.asarray(transform["world_from_mapanything_row_major"], dtype=np.float64)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=2e-4, rtol=0)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1.0, atol=2e-4, rtol=0)
            or transform.get("scale") != 1.0):
        raise ValueError(f"Alignment is not the fixed-scale rigid world transform: {alignment}")
    provenance = report["inputs"]["phone_source"]
    count = fusion.get("view_count")
    if (isinstance(count, bool) or not isinstance(count, int) or not 2 <= count <= 4096
            or count != source.get("view_count") or count != provenance.get("view_count")):
        raise ValueError(f"Source view counts disagree or are unsupported: {consensus}")
    manifest_identity = _identity(source_path)
    if (provenance["output_manifest"] != transform.get("source_output_manifest")
            or provenance["output_manifest"]["sha256"] != manifest_identity["sha256"]
            or provenance["raw_root"] != transform.get("source_raw_root")
            or report["inputs"].get("phone_output_manifest_sha256") != manifest_identity["sha256"]):
        raise ValueError(f"Alignment is bound to a different source manifest: {alignment}")
    views = provenance["views"]
    expected_names = [f"view_{index:04d}.npz" for index in range(count)]
    if len(views) != count or [row["name"] for row in views] != expected_names:
        raise ValueError(f"Alignment does not declare the exact sequential raw views: {alignment}")
    records = [{key: row[key] for key in ("name", "sha256", "size_bytes")} for row in views]
    set_hash = hashlib.sha256(json.dumps(
        records, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    if (set_hash != provenance["raw_view_set_sha256"]
            or set_hash != transform.get("source_raw_view_set_sha256")):
        raise ValueError(f"Alignment raw-view-set binding is inconsistent: {alignment}")
    paths = [consensus / "raw" / name for name in expected_names]
    # Verify only files already enumerated by the passed alignment. No rglob,
    # recursive hashing, extra-file discovery, or scan of model/artifact trees.
    for path, record in zip(paths, records, strict=True):
        identity = _identity(path)
        if any(identity[key] != record[key] for key in ("sha256", "size_bytes")):
            raise ValueError(f"Raw source no longer matches its passed alignment: {path}")
    return {"fusion": fusion, "matrix": matrix, "transform": transform,
            "report": report, "paths": paths, "consensus": _identity(fusion_path),
            "alignment_report": _identity(report_path), "world_transform": _identity(transform_path),
            "source_binding": {"output_manifest": manifest_identity,
                               "raw_view_set_sha256": set_hash, "views": records,
                               "verification": "exact_declared_files_sha256; no_recursive_inventory"}}


def _score(grids: dict[str, np.ndarray], target: np.ndarray, helpers):
    if any(grid.shape != target.shape for grid in grids.values()):
        raise ValueError("Candidate and target depth grids have different shapes")
    common = np.isfinite(target)
    for grid in grids.values():
        common = common & np.isfinite(grid)
    scores, errors = {}, {}
    for name, grid in grids.items():
        own = np.isfinite(target) & np.isfinite(grid)
        own_errors = np.abs(grid[own] - target[own])
        common_errors = np.abs(grid[common] - target[common])
        scores[name] = {
            "own_support": {**helpers.depth_comparison(grid, target), **helpers.error_tails(own_errors)},
            "common_support": {**helpers.depth_comparison(grid, target, common), **helpers.error_tails(common_errors)},
        }
        errors[name] = {"own": own_errors, "common": common_errors}
    return common, scores, errors


def _heldout_report(path, manifest, heldouts, helpers):
    aggregate = {name: {"own": [], "common": [], "valid": 0} for name in LANES}
    report = {"manifest": _identity(path), "manifest_provenance": manifest,
              "evaluation_type": "temporally_withheld_static_model_depth",
              "independent_of_phone_geometry": manifest["independent_of_phone_geometry"],
              "used_for_alignment": False, "used_for_reference_inference": False,
              "fitting_performed": False, "metric_ground_truth": False,
              "cell_px": HELDOUT_CELL_PX, "occlusion_error_clipping": False,
              "mask_rule": "entire4x4_model_pixel_cell_valid_outside_exclusions; no_hole_fill",
              "limitations": manifest.get("heldout_limitations"), "frames": []}
    for row in heldouts:
        target = row["target_grid"]
        common, scores, errors = _score(row["grids"], target, helpers)
        item = {key: row[key] for key in (
            "source_frame_index", "encoded_pts_ns", "artifact", "valid_model_pixels",
            "image_width", "image_height",
        )}
        item.update(target_valid_cell_count=int(np.isfinite(target).sum()),
                    common_cell_count=int(common.sum()), candidates=scores)
        for name in LANES:
            scores[name]["coverage"] = scores[name]["own_support"]["target_cell_coverage_fraction"]
            aggregate[name]["own"].append(errors[name]["own"])
            aggregate[name]["common"].append(errors[name]["common"])
            aggregate[name]["valid"] += item["target_valid_cell_count"]
        report["frames"].append(item)
    report["pooled"] = {}
    for name, row in aggregate.items():
        own, common = np.concatenate(row["own"]), np.concatenate(row["common"])
        report["pooled"][name] = {
            "weighting": "pooled_valid_frame_cells_not_frame_means",
            "target_valid_cell_count": row["valid"],
            "own_support": helpers.error_tails(own), "common_support": helpers.error_tails(common),
            "coverage": len(own) / max(1, row["valid"]),
            "common_coverage": len(common) / max(1, row["valid"]),
        }
    return report


def _markdown(result):
    def fmt(value):
        return "n/a" if value is None else f"{value:.4f}"

    candidates, pooled = result["candidates"], result["heldout"]["pooled"]
    rows = [("Sampled points", *[str(candidates[name]["sampled_points"]) for name in LANES]),
            ("Source views", *[str(candidates[name]["view_count"]) for name in LANES])]
    for title, support, key in (
        ("Static own coverage", "static_own_support", "target_cell_coverage_fraction"),
        ("Static own median (m)", "static_own_support", "median_m"),
        ("Static own p80 (m)", "static_own_support", "p80_m"),
        ("Static common median (m)", "static_common_support", "median_m"),
        ("Static common p80 (m)", "static_common_support", "p80_m"),
        ("Static common p95 (m)", "static_common_support", "p95_m"),
    ):
        rows.append((title, *[fmt(candidates[name][support][key]) for name in LANES]))
    rows.append(("Withheld own coverage", *[fmt(pooled[name]["coverage"]) for name in LANES]))
    for title, key in (("Withheld common median (m)", "median_m"),
                       ("Withheld common p80 (m)", "p80_m"),
                       ("Withheld common p95 (m)", "p95_m"),
                       ("Withheld common fraction >0.50 m", "fraction_gt_0_50m"),
                       ("Withheld common fraction >1 m", "fraction_gt_1m")):
        rows.append((title, *[fmt(pooled[name]["common_support"][key]) for name in LANES]))
    text = ["# Reconstruction lane comparison", "",
            "Offline review only. Metric ground truth: **false**. No geometry fit or runtime admission.", "",
            "| Metric | Baseline | Candidate |", "| --- | ---: | ---: |"]
    text.extend(f"| {title} | {baseline} | {candidate} |" for title, baseline, candidate in rows)
    text.extend(["", "Both lanes use exactly 600,000 points, per-view 45th-percentile confidence eligibility, "
                 "and deterministic sampling without replacement. Each uses its own passed, source-bound "
                 "world transform exactly once; no presentation transform is substituted.", "",
                 f"Static common support: {result['common_static_cells']:,} identical cells. "
                 "Static cells are 8px, withheld cells 4px. The entire declared mask cell must be valid; "
                 "no hole filling or error clipping. Own-support errors and coverage retain support differences.", "",
                 "The static keyframe was used for alignment, so its scores are in-domain model consistency. "
                 "Withheld observations were not used for alignment/reference inference, but share the camera, "
                 "calibration and model family; they are not surveyed metric ground truth. "
                 "Full withheld timing, selection, model and exclusion provenance remains in metrics.json.", "",
                 "This comparison does not isolate resolution from any other changed inference settings. "
                 "Inputs and sampling audits are recorded in metrics.json; static grids are retained in "
                 "static_depth_grids.npz."])
    return "\n".join(text) + "\n"


def compare(args: argparse.Namespace) -> dict[str, Any]:
    cv2.setNumThreads(2)
    path_names = ("baseline_consensus", "baseline_alignment", "candidate_consensus",
                  "candidate_alignment", "target_revision", "calibration",
                  "heldout_manifest", "output_dir", "reference_helpers")
    paths = {name: getattr(args, name).resolve() for name in path_names}
    output = paths["output_dir"]
    if output.exists():
        raise FileExistsError(f"Use a fresh comparison output directory: {output}")
    protected = [paths[name] for name in path_names[:5]] + [paths["heldout_manifest"].parent]
    if any(output.is_relative_to(root) for root in protected):
        raise ValueError("Comparison output must be outside the supplied input directories")
    helpers = _load_helpers(paths["reference_helpers"])
    lanes = {name: _load_lane(paths[f"{name}_consensus"], paths[f"{name}_alignment"])
             for name in LANES}
    binding = lanes["baseline"]["transform"]["target_binding"]
    if lanes["candidate"]["transform"]["target_binding"] != binding:
        raise ValueError("Baseline/candidate have different target bindings")
    target_root, calibration = paths["target_revision"], paths["calibration"]
    meta = _json(target_root / "room_points_meta.json")
    expected_binding = {
        "world_frame": meta.get("world_frame", "backend_world_m"),
        "world_frame_revision": meta.get("world_frame_revision"),
        "camera_frame_binding": meta.get("camera_frame_binding"),
        "reference_kind": meta.get("reference_kind", "saved_static"),
        "companion_session_id": meta.get("companion_session_id"),
    }
    if binding != expected_binding:
        raise ValueError("Supplied target revision differs from the alignment target binding")
    static = _load_static_reference(target_root, calibration, args.camera)
    target_points, _ = _load_target_points(target_root)
    mask_value = meta.get("rgb_keyframe_valid_masks", {}).get(args.camera)
    if not mask_value:
        raise ValueError("Comparison requires the declared static keyframe-valid mask")
    mask_path = _artifact(target_root, mask_value)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    height, width = static.image.shape[:2]
    if mask is None or mask.shape != (height, width):
        raise ValueError("Static valid mask is missing or shape-incompatible")
    target_inputs = {
        "target_points_sha256": _identity(target_root / "room_points.npz"),
        "target_meta_sha256": _identity(target_root / "room_points_meta.json"),
        "fixed_camera_keyframe_sha256": _identity(static.source_image),
        "camera_calibration_sha256": _identity(calibration),
        "fixed_camera_rgb_valid_mask_sha256": _identity(mask_path),
    }
    for name, lane in lanes.items():
        if lane["report"]["target"].get("camera_id") != args.camera:
            raise ValueError(f"{name} alignment uses another camera")
        for key, identity in target_inputs.items():
            if lane["report"]["inputs"].get(key) != identity["sha256"]:
                raise ValueError(f"{name} alignment target input differs: {key}")
    target_grid = _project_depth_grid(target_points, static.camera_from_world,
                                     static.intrinsics, width, height, STATIC_CELL_PX)
    mask_cells = helpers.masked_depth_cells(np.ones(mask.shape), mask > 0, STATIC_CELL_PX)
    target_grid[~np.isfinite(mask_cells)] = np.inf
    heldout_path = paths["heldout_manifest"]
    declared = _json(heldout_path)
    frames = declared.get("source", {}).get("frames", [])
    if (not 1 <= len(frames) <= 256
            or declared.get("independent_of_phone_geometry") is not True
            or len({row["source_frame_index"] for row in frames}) != len(frames)):
        raise ValueError("Require a bounded, nonempty, independently withheld frame set")
    for row in frames:
        _artifact(heldout_path.parent, row["artifact"])
    heldout_manifest, heldouts = helpers.load_heldouts(heldout_path, static, binding, args.camera)
    if len(heldouts) != len(frames):
        raise ValueError("Heldout helper did not retain the entire declared frame set")
    result = {"schema": "noesis.review.reconstruction_lane_comparison.v1",
              "generated_at": datetime.now(timezone.utc).isoformat(),
              "status": "offline_review_only", "metric_ground_truth": False,
              "metric_accuracy_proven": False, "accepted_for_canonical_use": False,
              "point_budget_each": POINT_BUDGET, "target_binding": binding,
              "camera": args.camera, "target": str(target_root), "target_inputs": target_inputs,
              "target_mask": _identity(mask_path), "script": _identity(Path(__file__)),
              "reference_helpers": _identity(paths["reference_helpers"]),
              "rules": {"static_cell_px": STATIC_CELL_PX, "heldout_cell_px": HELDOUT_CELL_PX,
                        "sampling": "equal_view_strata_exact_total; confidence_percentile_45; deterministic_linspace_without_replacement",
                        "support": "own=target_valid_and_lane_finite; common=target_valid_and_both_lanes_finite",
                        "hole_filling": False, "error_clipping": False}, "candidates": {}}
    grids = {}
    for name, lane in lanes.items():
        print(f"Sampling {name}: exactly {POINT_BUDGET:,} points", flush=True)
        cloud = helpers.exact_cloud(_BoundRawViews(lane["paths"]), POINT_BUDGET, "consensus")
        if len(cloud.points) != POINT_BUDGET:
            raise ValueError("Reference sampler did not retain the exact equal point budget")
        audit = helpers.exact_cloud.audit
        if ([row["view"] for row in audit] != [path.name for path in lane["paths"]]
                or sum(row["sampled_points"] for row in audit) != POINT_BUDGET):
            raise ValueError("Sampler audit differs from the source-bound view set/budget")
        matrix = lane["matrix"]
        points = cloud.points @ matrix[:3, :3].T + matrix[:3, 3]
        grids[name] = _project_depth_grid(points, static.camera_from_world,
                                         static.intrinsics, width, height, STATIC_CELL_PX)
        for row in heldouts:
            row["grids"][name] = _project_depth_grid(
                points, row["camera_from_world"], row["intrinsics"],
                row["image_width"], row["image_height"], HELDOUT_CELL_PX,
            )
        fusion = lane["fusion"]
        result["candidates"][name] = {
            "consensus": lane["consensus"], "alignment_report": lane["alignment_report"],
            "world_transform": {**lane["world_transform"],
                                "world_from_source_row_major": matrix.tolist(),
                                "source_coordinate_frame": lane["transform"]["source_coordinate_frame"],
                                "target_coordinate_frame": lane["transform"]["target_coordinate_frame"]},
            "source_binding": lane["source_binding"], "sampled_points": len(points),
            "view_count": fusion["view_count"], "processed_depth_shape_hw": list(cloud.frame_zero_depth.shape),
            "per_view_sampling": audit,
            "internal_multiview": fusion.get("multiview_consistency", {}).get("consensus"),
            "surfel_fusion": fusion.get("surfel_fusion"),
        }
        del cloud, points
        gc.collect()
    common, scores, _ = _score(grids, target_grid, helpers)
    result["common_static_cells"] = int(common.sum())
    for name in LANES:
        result["candidates"][name].update(
            static_own_support=scores[name]["own_support"],
            static_common_support=scores[name]["common_support"],
        )
    result["heldout"] = _heldout_report(heldout_path, heldout_manifest, heldouts, helpers)
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    output.mkdir(parents=True, exist_ok=False)
    with (output / "static_depth_grids.npz").open("xb") as handle:
        np.savez_compressed(handle, target=target_grid, common=common, **grids)
    (output / "metrics.json").write_text(serialized, encoding="utf-8")
    (output / "COMPARISON.md").write_text(_markdown(result), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-consensus", "baseline-alignment", "candidate-consensus",
                 "candidate-alignment", "target-revision", "calibration",
                 "heldout-manifest", "output-dir", "reference-helpers"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--camera", default="living-room")
    result = compare(parser.parse_args(argv))
    print(json.dumps({"static_common": {name: row["static_common_support"]
                                       for name, row in result["candidates"].items()},
                      "heldout": result["heldout"]["pooled"]}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
