#!/usr/bin/env python3
"""Derive the existing offline review artifact families from a completed lane.

All input paths are explicit and all changed outputs use a new root. Completed
stages can be resumed only with the same command and bound output manifests.
This never publishes calibration, Scene Priors, scan state or live geometry.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools.mapanything_phone_scan.run_highres_reconstruction_lane import binding, run_stage, save, utc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("lane-root", "output-root", "texture-photo-dir", "renderer", "rasterizer", "reference-visibility"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    lane, root = args.lane_root.resolve(), args.output_root.resolve()
    source = json.loads((lane / "lane_manifest.json").read_text())
    lease = source["runtime_lease"]
    restoration_required = lease["appliance_was_active"] and lease["pause_requested"]
    if source["status"] != "complete" or (restoration_required and not lease["restore_passed"]):
        raise ValueError("source lane must be complete with successful runtime restoration")
    scan = Path(source["scan"])
    if root.is_relative_to(scan) or root.is_relative_to(lane):
        raise ValueError("presentation outputs must be separate from scan and inference roots")
    inputs = {"lane": binding(lane / "lane_manifest.json"),
        "photos": binding(args.texture_photo_dir / "bound_texture_photos_manifest.json"),
        "renderer": binding(args.renderer), "rasterizer": binding(args.rasterizer),
        "reference_visibility": binding(args.reference_visibility)}
    state_path = root / "postprocessing_manifest.json"
    if args.resume:
        state = json.loads(state_path.read_text())
        if state["inputs"] != inputs:
            raise ValueError("resume source bindings changed")
        state.update(status="running", error=None, resumed_at=utc())
    else:
        if root.exists():
            raise ValueError("use a fresh presentation output root")
        root.mkdir(parents=True, exist_ok=False)
        state = {"schema": "noesis.highres_review_artifacts.v1", "created_at": utc(),
                 "review_only": True, "accepted_for_canonical_use": False,
                 "inputs": inputs, "stages": [], "status": "running"}
    (root / "tmp").mkdir(exist_ok=True)
    save(state_path, state)
    os.environ["TMPDIR"] = str(root / "tmp")
    # CPU-only derivatives must not initialize another inference GPU context.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    py = sys.executable
    def script(name):
        return str(REPO / "tools/mapanything_phone_scan" / name)
    consensus = lane / "pcf/prior_conditioned_consensus_da3_carrier"
    alignment = lane / "alignment_consensus"
    clean = root / "surface_closed_clean"
    photos = ["--mesh-dir", str(clean), "--scan-dir", str(scan), "--consensus-dir", str(consensus),
        "--alignment-dir", str(alignment), "--texture-photo-dir", str(args.texture_photo_dir.resolve()),
        "--texture-edge-px", "2560", "--max-faces", "5000000", "--max-vertices", "3000000",
        "--max-input-pixels", "64000000", "--max-texture-texels", "512000000"]
    render = ["--renderer", str(args.renderer.resolve()), "--rasterizer", str(args.rasterizer.resolve()),
              "--reference-visibility", str(args.reference_visibility.resolve()), "--width", "960", "--height", "540"]
    for index in (0, 32, 64, 96, 129):
        render += ["--phone-view", str(index)]
    stages = [
        ("evaluation", [py, script("evaluate_mapanything_prior_variants.py"), str(scan),
            "--suite-root", str(lane / "pcf"), "--da3-raw", str(lane / "da3/raw"),
            "--prior-consensus-raw", str(consensus / "raw"), "--variants", "da3_pose_sparse_depth",
            "--world-from-da3", str(lane / "alignment_da3/phone_ma_to_noesis_world.json"),
            "--target-revision", source["static_target_revision"], "--calibration", source["calibration"]["path"],
            "--camera", "living-room", "--output-dir", str(root / "evaluation_static_world")],
            "evaluation_static_world/evaluation_metrics.json"),
        ("point_preserving", [py, str(REPO / "testpipelines/roomform/build_point_preserving_fusion.py"),
            str(consensus / "raw"), str(root / "point_preserving_fusion_2cm")],
            "point_preserving_fusion_2cm/point_preserving_fusion_report.json"),
        ("dense_surfels", [py, script("build_dense_review_surfels.py"), "--consensus-dir", str(consensus),
            "--alignment-dir", str(alignment), "--camera-id", "living-room", "--voxel-m", ".02",
            "--max-total-pixels", "48000000", "--output-dir", str(root / "dense_2cm")],
            "dense_2cm/dense_surfel_manifest.json"),
        ("measured_surface", [py, script("build_single_room_review_surface.py"),
            "--consensus-dir", str(consensus), "--alignment-dir", str(alignment),
            "--dense-surfel-dir", str(root / "dense_2cm"), "--room-name", "living", "--owner", "3",
            "--ceiling-mode", "full_height", "--voxel-size-m", ".02", "--normal-radius-m", ".06",
            "--support-distance-m", ".075", "--poisson-depth", "10", "--density-percentile", "2",
            "--minimum-component-triangles", "64", "--poisson-threads", "1",
            "--visibility-policy", "qualified_consensus", "--output-dir", str(root / "surface_measured")],
            "surface_measured/surface_mesh_manifest.json"),
        ("closed_surface", [py, script("build_closed_room_review_surface.py"),
            "--measured-surface-dir", str(root / "surface_measured"), "--poisson-depth", "10",
            "--poisson-threads", "1", "--maximum-hole-radius-m", "10", "--output-dir", str(root / "surface_closed")],
            "surface_closed/surface_mesh_manifest.json"),
        ("clean_surface", [py, script("clean_completed_surface_mesh.py"), "--source-dir", str(root / "surface_closed"),
            "--minimum-component-faces", "64", "--max-faces", "5000000", "--max-vertices", "3000000",
            "--output-dir", str(clean)],
            "surface_closed_clean/surface_mesh_manifest.json"),
        ("mesh_photo_visibility", [py, script("build_mesh_texture_visibility.py"), *photos,
            "--renderer", str(args.renderer.resolve()), "--rasterizer", str(args.rasterizer.resolve()),
            "--visibility-edge-px", "960", "--output-dir", str(root / "mesh_photo_visibility")],
            "mesh_photo_visibility/mesh_texture_visibility_manifest.json"),
        ("photo_texture", [py, script("texture_single_room_review_surface.py"), *photos,
            "--surface-visibility-dir", str(root / "mesh_photo_visibility"), "--top-k", "12",
            "--smoothness-m", ".2", "--label-iterations", "12", "--normalize-measured-exposure",
            "--crop-selected-photo-regions", "--candidate-cache-dir", str(root / "photo_candidate_cache"),
            "--output-dir", str(root / "textured_closed_2560")], "textured_closed_2560/texture_manifest.json"),
        ("continuous_colour", [py, script("bake_continuous_video_color.py"), *photos,
            "--photometric-manifest", str(root / "textured_closed_2560/texture_manifest.json"),
            "--color-top-k", "4", "--output-dir", str(root / "continuous_video_colour")],
            "continuous_video_colour/surface_mesh_manifest.json"),
        ("final_colour", [py, script("bake_continuous_video_color.py"), "finalize",
            "--source-dir", str(root / "continuous_video_colour"), "--output-dir", str(root / "final_closed_video_colour")],
            "final_closed_video_colour/surface_mesh_manifest.json"),
    ]
    for name, mesh_dir, texture_manifest in (("render_measured", root / "surface_measured", None),
        ("render_final", root / "final_closed_video_colour", None),
        ("render_photo", clean, root / "textured_closed_2560/texture_manifest.json")):
        glb = mesh_dir / "surface_mesh.glb"
        extra = []
        if texture_manifest:
            glb = root / "textured_closed_2560/textured_surface_mesh.glb"
            extra = ["--texture-manifest", str(texture_manifest)]
        stages.append((name, [py, script("render_cpu_review_mesh.py"), "--glb", str(glb),
            "--mesh-manifest", str(mesh_dir / "surface_mesh_manifest.json"), *render, *extra,
            "--output-dir", str(root / name)], f"{name}/render_manifest.json"))
    try:
        for name, command, output_manifest in stages:
            prior = next((row for row in reversed(state["stages"]) if row["name"] == name), None)
            if prior and prior["status"] == "complete":
                if prior["command"] != command or prior.get("output_manifest") != binding(root / output_manifest):
                    raise ValueError(f"{name}: completed checkpoint changed")
                print(f"Reusing bound completed stage: {name}", flush=True)
                continue
            run_stage(root, name, command, state, state_path=state_path)
            state["stages"][-1]["output_manifest"] = binding(root / output_manifest)
            save(state_path, state)
        state.update(status="complete", error=None, finished_at=utc())
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_at=utc())
        raise
    finally:
        save(state_path, state)


if __name__ == "__main__":
    main()
