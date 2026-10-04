#!/usr/bin/env python3
"""Review and solve room-independent, measured PCF-to-authored registration.

This is an offline geometry tool, not a deployment command. ``inspect`` emits
identifiable candidates; ``solve`` uses explicitly reviewed feature identities
and writes both accepted and rejected evidence. Nothing changes live state.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from noesis.calibration.authored_structures import extract_authored_structures
from noesis.calibration.presentation_registration import _read_json, sha256_file
from noesis.calibration.structural_presentation import _relative_file, build_structural_presentation
from noesis.calibration.structural_registration import (
    StructuralRegistrationError, extract_pcf_wall_planes, read_pcf_points_glb,
    solve_structural_registration,
)
from noesis.validation.authored_scene import AuthoredSceneGeometry
from noesis_core.contracts.scene_prior import ScenePriorRevision
from noesis_core.contracts.support_surface import SupportSurfaceSet


def _path(value: str, root: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _room_inputs(bundle: Mapping[str, Any], camera: str, recipe: Mapping[str, Any], authored: Path):
    manifest_path = _path(recipe["manifest_path"])
    manifest = ScenePriorRevision.model_validate(_read_json(manifest_path))
    binding = bundle["cameras"]["frame_bindings"][camera]
    if (manifest.prior_id != binding["world_frame"]["revision"]
            or manifest.preview.reference_camera_id != camera
            or manifest.preview.camera_calibration.sha256 != binding["camera_calibration_sha256"]):
        raise StructuralRegistrationError("recipe PCF is not the current admitted camera/revision")
    points = next(a for a in manifest.artifacts if a.role == "points_glb")
    source = manifest_path.parent / points.relative_path
    if source.stat().st_size != points.size_bytes or sha256_file(source) != points.sha256:
        raise StructuralRegistrationError("PCF points changed since their immutable manifest")
    hint = float(recipe.get("scene_per_m_hint", bundle["align"]["scene_similarity"]["scene_per_m"]))
    bounds = recipe.get("bounds_xz_m")
    source_segments = extract_pcf_wall_planes(read_pcf_points_glb(source), bounds_xz_m=bounds)
    geometry = AuthoredSceneGeometry.from_obj(authored)
    structures = extract_authored_structures(geometry, manifest.semantic_binding.authored_groups,
                                            initial_scene_units_per_meter=hint)
    return manifest_path, manifest, source_segments, geometry, structures, hint


def inspect_room(bundle: Mapping[str, Any], camera: str, recipe: Mapping[str, Any], authored: Path) -> dict:
    manifest_path, manifest, source, geometry, structures, hint = _room_inputs(bundle, camera, recipe, authored)
    return {
        "status": "unmatched_structural_review_only", "camera_id": camera,
        "world_frame": bundle["cameras"]["frame_bindings"][camera]["world_frame"],
        "pcf_manifest_path": manifest_path.relative_to(ROOT).as_posix(),
        "pcf_manifest_sha256": sha256_file(manifest_path),
        "authored_scene_sha256": geometry.source_sha256,
        "scene_per_m_hint_for_tolerances_only": hint,
        "source_segments": [asdict(s) for s in source],
        "authored_structures": asdict(structures),
        "room_labels": list(manifest.semantic_binding.room_labels),
        "limitation": "A plane or floor edge is not automatically a wall identity; review images and source geometry before matching.",
    }


def solve_room(bundle: Mapping[str, Any], camera: str, recipe: Mapping[str, Any], authored: Path) -> dict:
    from noesis.calibration.structural_image_evidence import project_structural_image_point

    manifest_path, _, source, geometry, structures, hint = _room_inputs(bundle, camera, recipe, authored)
    source_by_id = {s.segment_id: asdict(s) for s in source}
    authored_by_id = {s.segment_id: s for s in structures.wall_segments}
    source_ids = recipe["source_segment_ids"]
    authored_ids = recipe["authored_segment_ids"]
    if len(set(source_ids)) != len(source_ids) or len(set(authored_ids)) != len(authored_ids):
        raise StructuralRegistrationError("selected feature IDs are duplicated")
    selected_source = [source_by_id[i] for i in source_ids]
    selected_authored = [authored_by_id[i] for i in authored_ids]
    points, image_evidence = [], []
    for row in recipe.get("image_points", []):
        if set(row) != {"point_id", "kind", "image_path", "capture_manifest_path", "image_size", "pixel", "pixel_bounds", "plane_segment_id", "surface_y_m", "identity_review"}:
            raise StructuralRegistrationError("image point recipe has unknown/missing fields")
        if not isinstance(row["identity_review"], str) or not row["identity_review"].strip():
            raise StructuralRegistrationError("an image feature requires a reviewed architectural identity")
        result = project_structural_image_point(
            bundle=bundle, camera_id=camera, image_path=_path(row["image_path"]),
            image_size=row["image_size"], pixel=row["pixel"], kind=row["kind"],
            point_id=row["point_id"],
            plane_segment=source_by_id[row["plane_segment_id"]] if row["plane_segment_id"] else None,
            surface_y_m=row["surface_y_m"],
        )
        points.append(result["point"])
        capture_manifest = _path(row["capture_manifest_path"])
        capture = {"manifest": _relative_file(capture_manifest, ROOT),
                   "calibration": _relative_file(capture_manifest.parent / "calibration.json", ROOT),
                   "metadata": _relative_file(capture_manifest.parent / "room_points_meta.json", ROOT)}
        image_evidence.append({**result, "image_file": _relative_file(_path(row["image_path"]), ROOT), "capture": capture,
                               "identity_review": row["identity_review"], "pixel_bounds": row["pixel_bounds"]})
    authored_points = []
    openings = {o.opening_id: o for o in structures.openings}
    for row in recipe.get("authored_points", []):
        if row.get("kind") == "opening" and set(row) == {"point_id", "kind", "opening_id", "endpoint", "identity_review"}:
            opening = openings[row["opening_id"]]
            if row["endpoint"] not in ("start", "end"):
                raise StructuralRegistrationError("opening endpoint must be start or end")
            xz = opening.start_xz if row["endpoint"] == "start" else opening.end_xz
        elif row.get("kind") == "corner" and set(row) == {"point_id", "kind", "vertex_index", "identity_review"}:
            index = row["vertex_index"]
            if type(index) is not int or not 1 <= index <= len(geometry.vertices):
                raise StructuralRegistrationError("invalid authored vertex index")
            vertex = geometry.vertices[index - 1]
            xz = (vertex[0], vertex[2])
        else:
            raise StructuralRegistrationError("authored point requires an exact opening endpoint or vertex selector")
        if not isinstance(row["identity_review"], str) or not row["identity_review"].strip():
            raise StructuralRegistrationError("authored feature identity is not reviewed")
        authored_points.append({"point_id": row["point_id"], "kind": row["kind"], "xz": list(xz)})
    candidate = solve_structural_registration(
        selected_source, selected_authored, correspondences=recipe["correspondences"],
        source_points=points, authored_points=authored_points,
        point_correspondences=recipe.get("point_correspondences", []), scene_per_m_hint=hint,
    )
    return {"manifest_path": str(manifest_path), "candidate": candidate,
            "source_image_evidence": image_evidence,
            "authored_point_selectors": recipe.get("authored_points", [])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inspect", "solve", "build"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--authored-scene", type=Path, required=True)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--review", type=Path, help="Accepted solve review used by build")
    parser.add_argument("--room-group-map", type=Path)
    parser.add_argument("--scene-prior-catalog", type=Path)
    parser.add_argument("--parent-authored-scene", type=Path)
    parser.add_argument("--retained-registration", type=Path, help="Explicit unchanged-room registration, never a fit fallback")
    parser.add_argument("--retained-authored-scene", type=Path, help="Exact original geometry of retained room mappings")
    parser.add_argument("--support-surfaces", type=Path, action="append", default=[])
    parser.add_argument("--measured-floor-groups", type=Path, help="Camera -> measured surface ID -> authored group list")
    parser.add_argument("--owner-selection", type=Path, help="Explicit exact-matrix selection receipt; never structural qualification")
    parser.add_argument("--authored-room-rebinding", type=Path, help="Reviewed new-model room identities, preserving the original PCF semantic binding")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new evidence output path; retained results are not overwritten")
    if args.mode == "build":
        if args.review is None or args.room_group_map is None or args.scene_prior_catalog is None:
            parser.error("build requires --review, --room-group-map and --scene-prior-catalog")
        review = _read_json(args.review)
        if (set(review) != {"contract", "contract_version", "mode", "rooms", "authority"}
                or review["contract"] != "noesis.structural_registration.review"
                or review["contract_version"] != 1 or review["mode"] != "solve"):
            parser.error("build requires a structural solve review, not an inspection or guessed matrix")
        sets = {}
        for path in args.support_surfaces:
            measured = SupportSurfaceSet.model_validate(_read_json(path))
            if measured.camera_id in sets:
                parser.error("duplicate measured support camera")
            sets[measured.camera_id] = measured
        payload = build_structural_presentation(
            bundle_path=args.bundle, authored_obj_path=args.authored_scene,
            room_group_map_path=args.room_group_map, room_solutions=review["rooms"],
            scene_prior_catalog_path=args.scene_prior_catalog,
            support_surfaces=sets,
            measured_floor_groups=_read_json(args.measured_floor_groups) if args.measured_floor_groups else {},
            retained_registration_path=args.retained_registration,
            retained_authored_obj_path=args.retained_authored_scene,
            parent_authored_obj_path=args.parent_authored_scene,
            owner_selection=_read_json(args.owner_selection) if args.owner_selection else None,
            authored_room_rebinding=_read_json(args.authored_room_rebinding) if args.authored_room_rebinding else None,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"registration_sha256": payload["registration_sha256"],
                          "targets": {key: value["kind"] for key,value in payload["presentations"].items()}}))
        return 0
    if args.recipe is None:
        parser.error("inspect/solve requires --recipe")
    bundle, recipe = _read_json(args.bundle), _read_json(args.recipe)
    if set(recipe) != {"contract", "contract_version", "rooms"} or recipe["contract"] != "noesis.structural_registration.recipe" or recipe["contract_version"] != 1:
        parser.error("unsupported structural recipe contract")
    rooms = recipe["rooms"]
    if not isinstance(rooms, dict) or not 1 <= len(rooms) <= 8:
        parser.error("a bounded set of current room/camera recipes is required")
    method = inspect_room if args.mode == "inspect" else solve_room
    results = {camera: method(bundle, camera, row, args.authored_scene) for camera, row in rooms.items()}
    payload = {"contract": "noesis.structural_registration.review", "contract_version": 1,
               "mode": args.mode, "rooms": results,
               "authority": "offline_geometry_review_only_not_live_registration"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({camera: (row["candidate"]["solver_status"] if args.mode == "solve" else row["status"])
                      for camera, row in results.items()}))
    return 2 if args.mode == "solve" and any(row["candidate"]["solver_status"] != "accepted" for row in results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
