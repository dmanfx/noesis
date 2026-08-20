#!/usr/bin/env python3
"""Publish a derived PCF surface mesh as an immutable review assembly revision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish(args: argparse.Namespace) -> dict[str, Any]:
    storage_root = args.pcf_storage_root.resolve()
    current_path = args.current_descriptor.resolve()
    current = _json(current_path)
    source_artifact = current.get("artifact", {})
    if (
        current.get("contract") != "noesis.scene.review_assembly"
        or current.get("contract_version") != 3
        or current.get("status") != "review_only"
        or current.get("accepted_for_canonical_use") is not False
        or source_artifact.get("role") != "multiroom_points_glb"
    ):
        raise ValueError("current descriptor is not a v3 point review assembly")
    source_glb = (storage_root / source_artifact["relative_path"]).resolve()
    if storage_root not in source_glb.parents:
        raise ValueError("source point GLB escapes PCF storage")
    if _sha256(source_glb) != source_artifact.get("sha256"):
        raise ValueError("source point GLB digest mismatch")

    mesh_manifest = _json(args.mesh_manifest)
    mesh_sha = _sha256(args.mesh_glb)
    mesh_output = mesh_manifest.get("output", {})
    if (
        mesh_manifest.get("contract") != "noesis.pcf.review_surface_mesh"
        or mesh_manifest.get("status") != "review_only"
        or mesh_manifest.get("accepted_for_canonical_use") is not False
        or mesh_output.get("sha256") != mesh_sha
        or int(mesh_output.get("size_bytes") or -1) != args.mesh_glb.stat().st_size
        or mesh_manifest.get("source", {}).get("surfels_manifest_sha256")
        != current.get("provenance", {}).get(
            "source_reintegration_manifest_sha256"
        )
    ):
        raise ValueError("surface mesh does not bind to the active review source")
    if not args.assembly_id or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for character in args.assembly_id
    ):
        raise ValueError("assembly id contains unsupported characters")
    structural = _json(args.source_structural_alignment)
    if (
        structural.get("contract") != "noesis.pcf.menon_structural_alignment"
        or structural.get("assembly_id") != current.get("assembly_id")
        or structural.get("source", {}).get("review_artifact_sha256")
        != source_artifact.get("sha256")
    ):
        raise ValueError("structural alignment does not bind to the source assembly")

    assemblies_root = storage_root / "review-assemblies"
    destination = assemblies_root / args.assembly_id
    if destination.exists():
        raise ValueError(f"assembly already exists: {destination}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{args.assembly_id}.", dir=assemblies_root)
    )
    try:
        artifact_name = "multiroom_surface_mesh.glb"
        copied_mesh = temporary / artifact_name
        shutil.copyfile(args.mesh_glb, copied_mesh)
        copied_sha = _sha256(copied_mesh)
        if copied_sha != mesh_sha:
            raise ValueError("copied surface mesh digest mismatch")
        owner_triangle_counts = {
            str(owner): int(metrics["output_triangle_count"])
            for owner, metrics in mesh_manifest.get("owners", {}).items()
        }
        descriptor = deepcopy(current)
        descriptor.update(
            {
                "contract_version": 4,
                "assembly_id": args.assembly_id,
                "created_at_us": time.time_ns() // 1_000,
                "artifact": {
                    "role": "multiroom_surface_mesh_glb",
                    "relative_path": (
                        Path("review-assemblies")
                        / args.assembly_id
                        / artifact_name
                    ).as_posix(),
                    "sha256": copied_sha,
                    "size_bytes": copied_mesh.stat().st_size,
                    "media_type": "model/gltf-binary",
                    "vertex_count": int(mesh_output["vertex_count"]),
                    "triangle_count": int(mesh_output["triangle_count"]),
                    "owner_triangle_counts": owner_triangle_counts,
                },
            }
        )
        descriptor["provenance"] = {
            **current["provenance"],
            "source_glb_sha256": copied_sha,
            "source_point_glb_sha256": source_artifact["sha256"],
            "source_surface_mesh_manifest_sha256": _sha256(args.mesh_manifest),
        }
        descriptor.pop("artifact_url", None)
        _atomic_json(temporary / "manifest.json", descriptor)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    structural["assembly_id"] = args.assembly_id
    structural["source"]["review_artifact_sha256"] = mesh_sha
    structural["source"]["source_point_assembly_id"] = current["assembly_id"]
    structural["source"]["source_point_artifact_sha256"] = source_artifact["sha256"]
    structural["source"]["surface_mesh_manifest_sha256"] = _sha256(
        args.mesh_manifest
    )
    for output in args.structural_output:
        _atomic_json(output.resolve(), structural)

    published_descriptor = _json(destination / "manifest.json")
    if args.activate:
        _atomic_json(current_path, published_descriptor)
    return {
        "assembly_id": args.assembly_id,
        "descriptor": str((destination / "manifest.json").resolve()),
        "artifact": str((destination / "multiroom_surface_mesh.glb").resolve()),
        "artifact_sha256": mesh_sha,
        "activated": bool(args.activate),
        "structural_outputs": [str(path.resolve()) for path in args.structural_output],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcf-storage-root", type=Path, required=True)
    parser.add_argument("--current-descriptor", type=Path, required=True)
    parser.add_argument("--mesh-glb", type=Path, required=True)
    parser.add_argument("--mesh-manifest", type=Path, required=True)
    parser.add_argument("--source-structural-alignment", type=Path, required=True)
    parser.add_argument(
        "--structural-output",
        action="append",
        type=Path,
        required=True,
    )
    parser.add_argument("--assembly-id", required=True)
    parser.add_argument("--activate", action="store_true")
    return parser


def main() -> None:
    result = publish(_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
