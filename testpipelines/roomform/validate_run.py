#!/usr/bin/env python3
"""Validate Roomform artifacts and optionally compare two precision runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


REQUIRED_GLTF_NODES = {
    "evidence",
    "shell-wall",
    "shell-floor",
    "shell-ceiling",
}


def _validate_glb(path: Path, *, expect_objects: bool) -> dict[str, Any]:
    scene = trimesh.load(path, force="scene")
    nodes = set(scene.graph.nodes_geometry)
    missing = sorted(REQUIRED_GLTF_NODES - nodes)
    if missing:
        raise RuntimeError(f"{path} is missing expected GLTF nodes: {missing}")
    if expect_objects and not ({"objects", "objects-flagged"} & nodes):
        raise RuntimeError(f"{path} has no exported object-box geometry")
    bounds = np.asarray(scene.bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
        raise RuntimeError(f"{path} has invalid bounds: {bounds}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "geometry_count": len(scene.geometry),
        "nodes": sorted(nodes),
        "bounds": bounds.tolist(),
    }


def _compare(reference: Path, candidate: Path) -> dict[str, Any]:
    ref = np.load(reference / "patchgraph.npz")
    cand = np.load(candidate / "patchgraph.npz")
    result: dict[str, Any] = {
        "reference": str(reference),
        "candidate": str(candidate),
        "arrays": {},
        "node_masks": {},
    }
    for key in ("node_probs", "edge_probs", "offsets", "openings"):
        left = np.asarray(ref[key], dtype=np.float32)
        right = np.asarray(cand[key], dtype=np.float32)
        if left.shape != right.shape:
            raise RuntimeError(f"precision comparison shape mismatch for {key}")
        result["arrays"][key] = {
            "mean_absolute_error": float(np.mean(np.abs(left - right))),
            "max_absolute_error": float(np.max(np.abs(left - right))),
        }
    for index, name in enumerate(("wall", "floor", "ceiling")):
        left = ref["node_probs"][index] >= 0.5
        right = cand["node_probs"][index] >= 0.5
        union = int(np.count_nonzero(left | right))
        intersection = int(np.count_nonzero(left & right))
        result["node_masks"][name] = {
            "iou": float(intersection / union) if union else 1.0,
            "agreement": float(np.mean(left == right)),
            "reference_cells": int(np.count_nonzero(left)),
            "candidate_cells": int(np.count_nonzero(right)),
        }
    return result


def validate(run_dir: Path, comparison_run: Path | None) -> dict[str, Any]:
    report = json.loads((run_dir / "run_report.json").read_text(encoding="utf-8"))
    input_manifest = json.loads(
        (run_dir / "input" / "input_manifest.json").read_text(encoding="utf-8")
    )
    if report.get("camera") != "living-room" or input_manifest.get("camera") != "living-room":
        raise RuntimeError("run is not living-room-only")
    patchgraph = np.load(run_dir / "patchgraph.npz")
    scene_doc = json.loads((run_dir / "scene.json").read_text(encoding="utf-8"))
    object_count = len(scene_doc.get("objects", []))
    expected_arrays = {"node_probs", "edge_probs", "offsets", "openings"}
    if set(patchgraph.files) != expected_arrays:
        raise RuntimeError(
            f"unexpected patchgraph arrays: expected {sorted(expected_arrays)}, "
            f"received {sorted(patchgraph.files)}"
        )
    result: dict[str, Any] = {
        "schema": "noesis.roomform.validation.v1",
        "run_dir": str(run_dir),
        "camera": "living-room",
        "checkpoint_sha256": report["checkpoint_sha256"],
        "source_revision": input_manifest["source_revision"],
        "point_count": input_manifest["point_count"],
        "patchgraph_arrays": {
            key: {
                "shape": list(patchgraph[key].shape),
                "dtype": str(patchgraph[key].dtype),
            }
            for key in patchgraph.files
        },
        "scene_glb": _validate_glb(
            run_dir / "scene.glb", expect_objects=bool(object_count)
        ),
        "scene_planar_glb": _validate_glb(
            run_dir / "scene-planar.glb", expect_objects=bool(object_count)
        ),
        "object_count": object_count,
        "object_mesh_count": len(list((run_dir / "objects").glob("*.glb"))),
        "render_count": len(list((run_dir / "renders").glob("*.png"))),
    }
    labels_path = run_dir / "labels.npz"
    if report.get("lifting_runtime") is not None:
        if not labels_path.is_file():
            raise RuntimeError("local PTv3 run has no labels.npz")
        labels = np.load(labels_path)
        if set(labels.files) != {"pts", "label", "classes"}:
            raise RuntimeError(f"unexpected PTv3 label arrays: {labels.files}")
        if len(labels["pts"]) != len(labels["label"]):
            raise RuntimeError("PTv3 point/label count mismatch")
        result["ptv3_labels"] = {
            "point_count": int(len(labels["pts"])),
            "class_count": int(len(labels["classes"])),
        }
    if comparison_run is not None:
        result["precision_comparison"] = _compare(comparison_run, run_dir)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--comparison-run", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    comparison = (
        args.comparison_run.expanduser().resolve()
        if args.comparison_run is not None
        else None
    )
    result = validate(run_dir, comparison)
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else run_dir / "validation.json"
    )
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
