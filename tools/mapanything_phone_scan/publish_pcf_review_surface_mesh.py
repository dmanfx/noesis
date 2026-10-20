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


def replacement_descriptor(
    *, current: dict[str, Any], current_path: Path, storage_root: Path,
    source_mesh_manifest: Path, mesh_glb: Path, mesh_manifest: Path,
    assembly_id: str,
) -> dict[str, Any]:
    """Prepare a v4-to-v4 replacement without writing any active selector.

    A replacement must preserve the exact untrimmed source and explicitly bind
    the retained current mesh. It is not a route around current source checks.
    """
    artifact, provenance = current.get("artifact", {}), current.get("provenance", {})
    if (current.get("contract") != "noesis.scene.review_assembly"
            or current.get("contract_version") != 4
            or current.get("status") != "review_only"
            or current.get("accepted_for_canonical_use") is not False
            or artifact.get("role") != "multiroom_surface_mesh_glb"):
        raise ValueError("replacement source is not a v4 review mesh")
    source_glb = (storage_root.resolve() / artifact["relative_path"]).resolve()
    if storage_root.resolve() not in source_glb.parents:
        raise ValueError("source mesh escapes PCF storage")
    if (_sha256(source_glb) != artifact.get("sha256")
            or source_glb.stat().st_size != artifact.get("size_bytes")
            or _sha256(source_mesh_manifest) != provenance.get("source_surface_mesh_manifest_sha256")):
        raise ValueError("replacement source mesh digest mismatch")
    old, new = _json(source_mesh_manifest), _json(mesh_manifest)
    source, output = new.get("source", {}), new.get("output", {})
    if (new.get("contract") != "noesis.pcf.review_surface_mesh"
            or new.get("status") != "review_only" or new.get("accepted_for_canonical_use") is not False
            or new.get("coordinate_frame") != current.get("assembly_gauge")
            or source.get("surfels_manifest_sha256") != provenance.get("source_reintegration_manifest_sha256")
            or source.get("surfels_npz_sha256") != old.get("source", {}).get("surfels_npz_sha256")
            or source.get("retained_low_glb_sha256") != artifact.get("sha256")
            or source.get("retained_low_mesh_manifest_sha256") != _sha256(source_mesh_manifest)
            or source.get("source_descriptor_sha256") != _sha256(current_path)
            or output.get("sha256") != _sha256(mesh_glb)
            or output.get("size_bytes") != mesh_glb.stat().st_size):
        raise ValueError("replacement mesh does not preserve the exact current source chain")
    if not assembly_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in assembly_id):
        raise ValueError("assembly id contains unsupported characters")
    counts = {str(owner):int(row["output_triangle_count"]) for owner,row in new.get("owners", {}).items()}
    if (set(counts) != set(artifact.get("owner_triangle_counts", {}))
            or any(value <= 0 for value in counts.values())
            or sum(counts.values()) != output.get("triangle_count")
            or not 0 < output.get("size_bytes", 0) <= 64 * 1024 * 1024):
        raise ValueError("replacement owner/count/size inventory is invalid")
    descriptor = deepcopy(current)
    descriptor.update(assembly_id=assembly_id, created_at_us=time.time_ns() // 1000)
    descriptor["artifact"] = {**artifact, "relative_path":f"review-assemblies/{assembly_id}/multiroom_surface_mesh.glb",
        **{key:output[key] for key in ("sha256", "size_bytes", "vertex_count", "triangle_count")},
        "owner_triangle_counts":counts}
    # No claim that the unchanged legacy triangles passed newer ray/class gates.
    descriptor["artifact"].pop("support", None)
    descriptor["provenance"].update(source_glb_sha256=output["sha256"],
        source_surface_mesh_manifest_sha256=_sha256(mesh_manifest))
    descriptor.pop("artifact_url", None)
    return descriptor


def publish(args: argparse.Namespace) -> dict[str, Any]:
    storage_root = args.pcf_storage_root.resolve()
    current_path = args.current_descriptor.resolve()
    current = _json(current_path)
    source_artifact = current.get("artifact", {})
    if (
        current.get("contract") != "noesis.scene.review_assembly"
        or current.get("contract_version") not in (3, 4)
        or current.get("status") != "review_only"
        or current.get("accepted_for_canonical_use") is not False
        or source_artifact.get("role") != ({3:"multiroom_points_glb",4:"multiroom_surface_mesh_glb"}.get(current.get("contract_version")))
    ):
        raise ValueError("current descriptor is not a supported point/mesh review assembly")
    source_glb = (storage_root / source_artifact["relative_path"]).resolve()
    if storage_root not in source_glb.parents:
        raise ValueError("source point GLB escapes PCF storage")
    if _sha256(source_glb) != source_artifact.get("sha256"):
        raise ValueError("source point GLB digest mismatch")

    mesh_manifest = _json(args.mesh_manifest)
    mesh_sha = _sha256(args.mesh_glb)
    mesh_output = mesh_manifest.get("output", {})
    replacement = None
    if current["contract_version"] == 4:
        if getattr(args, "source_mesh_manifest", None) is None:
            raise ValueError("mesh replacement requires the current source mesh manifest")
        replacement = replacement_descriptor(current=current,current_path=current_path,storage_root=storage_root,
            source_mesh_manifest=args.source_mesh_manifest,mesh_glb=args.mesh_glb,
            mesh_manifest=args.mesh_manifest,assembly_id=args.assembly_id)
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
        support_class_counts = {"observed": 0, "uncertain": 0, "unknown": 0}
        for metrics in mesh_manifest.get("owners", {}).values():
            for key in support_class_counts:
                support_class_counts[key] += int(
                    metrics.get("support_class_counts", {}).get(key, 0)
                )
        frame_identity = mesh_manifest.get("frame_identity")
        if not isinstance(frame_identity, dict):
            raise ValueError("surface mesh has no source frame identity")
        source_coordinate_frame = str(
            frame_identity.get("coordinate_frame") or ""
        ).strip()
        if not source_coordinate_frame:
            raise ValueError("surface mesh source frame identity is empty")
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
                    "support": {
                        "coordinate_frame": source_coordinate_frame,
                        "class_counts": support_class_counts,
                        "triangle_support_checks": True,
                        "ray_visibility_checks": True,
                        "source_frame_identity": frame_identity,
                    },
                },
            }
        )
        descriptor["provenance"] = {
            **current["provenance"],
            "source_glb_sha256": copied_sha,
            "source_point_glb_sha256": source_artifact["sha256"],
            "source_surface_mesh_manifest_sha256": _sha256(args.mesh_manifest),
        }
        if replacement is not None:
            descriptor = replacement
        descriptor.pop("artifact_url", None)
        _atomic_json(temporary / "manifest.json", descriptor)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    structural["assembly_id"] = args.assembly_id
    structural["source"]["review_artifact_sha256"] = mesh_sha
    if replacement is None:
        structural["source"]["source_point_assembly_id"] = current["assembly_id"]
        structural["source"]["source_point_artifact_sha256"] = source_artifact["sha256"]
    else:
        structural["source"]["retained_mesh_assembly_id"] = current["assembly_id"]
        structural["source"]["retained_mesh_artifact_sha256"] = source_artifact["sha256"]
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
    parser.add_argument("--source-mesh-manifest", type=Path,
                        help="required for v4 mesh-to-mesh replacement; preserves the original source chain")
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
