#!/usr/bin/env python3
"""CPU-only browser derivatives from one completed, source-bound review mesh."""
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
    for name in ("source-dir", "camera-manifest", "output-root", "gltf-cli"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--scan-id", required=True)
    parser.add_argument("--max-source-faces", type=int, default=3_000_000)
    parser.add_argument("--max-source-vertices", type=int, default=2_000_000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root, source = args.output_root.resolve(), args.source_dir.resolve()
    source_glb = source / "surface_mesh.glb"
    source_manifest = source / "surface_mesh_manifest.json"
    mesh = json.loads(source_manifest.read_text())
    camera = json.loads(args.camera_manifest.read_text())
    inputs = {"source_mesh": binding(source_glb), "source_manifest": binding(source_manifest),
              "camera_manifest": binding(args.camera_manifest), "gltf_cli": binding(args.gltf_cli.resolve())}
    if (mesh.get("status") != "review_only" or mesh.get("accepted_for_canonical_use") is not False
            or mesh["output"]["sha256"] != inputs["source_mesh"]["sha256"]
            or camera["source"]["glb_sha256"] != inputs["source_mesh"]["sha256"]):
        raise ValueError("browser source/cameras must bind the same completed review mesh")
    if root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("browser derivatives must be outside the source directory")
    state_path = root / "browser_build_manifest.json"
    if args.resume:
        state = json.loads(state_path.read_text())
        if state["inputs"] != inputs:
            raise ValueError("browser resume bindings changed")
    else:
        if root.exists():
            raise ValueError("use a fresh browser output root")
        root.mkdir(parents=True, exist_ok=False)
        state = {"schema": "noesis.highres_browser_derivatives.v1", "inputs": inputs,
                 "created_at": utc(), "stages": [], "review_only": True,
                 "accepted_for_canonical_use": False}
    state.update(status="running", error=None)
    save(state_path, state)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    (root / "tmp").mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(root / "tmp")
    py = sys.executable
    tool = REPO / "tools/room_mesh_export"
    limits = ["--max-source-faces", str(args.max_source_faces),
              "--max-source-vertices", str(args.max_source_vertices)]
    stages = []
    for name, triangles, edge in (("preview", 40_000, 2048), ("balanced", 150_000, 4096), ("detail", 400_000, 4096)):
        command = [py, str(tool / "build_browser_mesh.py"), "--source-glb", str(source_glb),
            "--source-manifest", str(source_manifest), "--scan-id", args.scan_id,
            "--triangles", str(triangles), "--texture-edge", str(edge), "--threads", "2",
            "--atlas-padding", "2", "--bake-padding", "8", "--chart-iterations", "2",
            "--chart-area", ".25", *limits, "--output-dir", str(root / name)]
        stages.append((name, command, root / name / "mesh_manifest.json"))
    stages.append(("photo_blend", [py, str(tool / "refine_texture.py"), "--base-dir", str(root / "detail"),
        "--output-dir", str(root / "photo-blend"), "--mode", "photo-blend", "--texture-edge", "4096",
        "--atlas-padding", "2", "--bake-padding", "8", "--threads", "2",
        "--chart-iterations", "2", "--chart-area", ".25"], root / "photo-blend/mesh_manifest.json"))
    stages.append(("package", ["node", str(tool / "package_trial.mjs"), "--trial-root", str(root),
        "--gltf-cli", str(args.gltf_cli.resolve()), "--camera-manifest", str(args.camera_manifest.resolve()),
        "--include-photo", "true", "--default-asset", "photo-blend",
        "--title", "Living Room · higher-resolution experiment",
        "--dataset-label", "130-view 756×420 experiment · retained for comparison, not selected over baseline"], root / "trial.json"))
    try:
        for name, command, output in stages:
            previous = next((row for row in reversed(state["stages"]) if row["name"] == name), None)
            if previous and previous["status"] == "complete":
                if previous["command"] != command or previous["output_manifest"] != binding(output):
                    raise ValueError(f"{name}: completed browser checkpoint changed")
                continue
            run_stage(root, name, command, state, state_path=state_path)
            state["stages"][-1]["output_manifest"] = binding(output)
            save(state_path, state)
        state.update(status="complete", finished_at=utc())
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_at=utc())
        raise
    finally:
        save(state_path, state)


if __name__ == "__main__":
    main()
