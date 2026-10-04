"""Verified canonical-target to RGB PCF review-mesh presentation edges.

These edges replay the transforms that built a review artifact. They do not
accept a rejected room join, change calibration, or create people positions.
The independent view deliberately has no authored-scene/camera-marker anchor.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from noesis_core.contracts.scene_prior import ScenePriorCatalog, ScenePriorRevision

MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_RECIPE_BYTES = 256 * 1024
CONTRACT = "noesis.pcf.tracking_review"


class PcfTrackingReviewError(ValueError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, maximum: int = MAX_JSON_BYTES) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > maximum:
        raise PcfTrackingReviewError("review input is missing or exceeds its bound")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PcfTrackingReviewError("review input must be an object")
    return value


def _keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PcfTrackingReviewError(f"{label} has unknown or missing fields")


def _input(ref: Any, base: Path) -> tuple[Path, dict[str, Any]]:
    _keys(ref, {"path", "sha256", "size_bytes"}, "review file reference")
    path = Path(ref["path"])
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file() or path.stat().st_size != ref["size_bytes"] or _sha(path) != ref["sha256"]:
        raise PcfTrackingReviewError("review input digest or length changed")
    return path, _json(path)


def _pose(values: Any, label: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise PcfTrackingReviewError(f"{label} is not a finite pose")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-10):
        raise PcfTrackingReviewError(f"{label} is not affine")
    if not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-8) or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-8):
        raise PcfTrackingReviewError(f"{label} must preserve meters, handedness and gravity")
    if not np.allclose(matrix[1, :3], [0, 1, 0], atol=1e-8):
        raise PcfTrackingReviewError(f"{label} does not preserve gravity")
    return matrix


def _points(prior: ScenePriorRevision) -> Any:
    rows = [row for row in prior.artifacts if row.role == "points_glb"]
    if len(rows) != 1:
        raise PcfTrackingReviewError("prior must have one measured point artifact")
    return rows[0]


def _frame_payload(binding: Any, prior_id: str) -> dict[str, Any]:
    # v2 geometry may be materialized in a different metric frame. It requires
    # an explicitly composed mesh edge, not reuse of a v1 prior-local matrix.
    if binding is None or binding.contract_version != 1 or binding.target_frame.revision != prior_id:
        raise PcfTrackingReviewError("review edge requires an exact prior-local target; rebuild the edge for a new metric frame")
    return {
        "contract": "noesis.calibration.frame_binding", "contract_version": 1,
        "calibration_frame": binding.source_frame.model_dump(mode="json"),
        "world_frame": binding.target_frame.model_dump(mode="json"),
        "target_from_calibration_col_major": list(binding.target_from_source_col_major),
        "transform_sha256": binding.target_from_source_sha256,
        "calibration_floor_plane": {"normal": list(binding.source_floor_plane.normal), "offset_m": binding.source_floor_plane.offset_m},
        "world_floor_plane": {"normal": list(binding.target_floor_plane.normal), "offset_m": binding.target_floor_plane.offset_m},
        "camera_calibration_sha256": binding.source_camera_calibration_sha256,
        "world_alignment_sha256": binding.source_world_alignment_sha256,
        "camera_calibration_physical_sha256": None,
        "world_alignment_physical_sha256": None,
        "scene_prior_id": prior_id, "artifact_revision_id": None,
        "target_revision_id": binding.target_revision_id,
    }


def build_tracking_review(
    recipe_path: Path, *, catalog_path: Path, current_descriptor: Mapping[str, Any],
    current_descriptor_path: Path, repo_root: Path,
) -> dict[str, Any]:
    """Revalidate bounded frozen source inputs and the active prior bindings."""
    recipe_path, catalog_path = recipe_path.resolve(), catalog_path.resolve()
    recipe = _json(recipe_path, MAX_RECIPE_BYTES)
    _keys(recipe, {"contract", "contract_version", "catalog", "source_descriptor", "mesh_manifest", "reintegration", "family_kitchen", "endpoint_registration", "rooms"}, "tracking review recipe")
    if recipe["contract"] != "noesis.pcf.tracking_review_binding" or recipe["contract_version"] != 1:
        raise PcfTrackingReviewError("unsupported tracking review recipe")
    base = repo_root.resolve()
    bound_catalog_path, catalog_json = _input(recipe["catalog"], base)
    if bound_catalog_path != catalog_path:
        raise PcfTrackingReviewError("recipe is not bound to the configured current catalog")
    catalog = ScenePriorCatalog.model_validate(catalog_json)
    bindings = {row.camera_id: row for row in catalog.camera_bindings}
    descriptor_path, descriptor_input = _input(recipe["source_descriptor"], base)
    if descriptor_path != current_descriptor_path.resolve() or descriptor_input != dict(current_descriptor):
        raise PcfTrackingReviewError("selected review descriptor differs from the frozen mesh source")
    _, mesh = _input(recipe["mesh_manifest"], base)
    _, reintegration = _input(recipe["reintegration"], base)
    _, family_kitchen = _input(recipe["family_kitchen"], base)
    _, registration = _input(recipe["endpoint_registration"], base)
    descriptor = dict(current_descriptor)
    artifact = descriptor.get("artifact", {})
    provenance = descriptor.get("provenance", {})
    if (descriptor.get("contract") != "noesis.scene.review_assembly"
            or descriptor.get("contract_version") != 4
            or descriptor.get("accepted_for_canonical_use") is not False
            or descriptor.get("assembly_gauge") != "family_accepted_backend_world_m"
            or artifact.get("role") != "multiroom_surface_mesh_glb"
            or mesh.get("contract") != "noesis.pcf.review_surface_mesh"
            or mesh.get("coordinate_frame") != descriptor["assembly_gauge"]
            or mesh.get("accepted_for_canonical_use") is not False
            or mesh.get("output", {}).get("sha256") != artifact.get("sha256")
            or mesh.get("output", {}).get("size_bytes") != artifact.get("size_bytes")
            or provenance.get("source_surface_mesh_manifest_sha256") != recipe["mesh_manifest"]["sha256"]
            or provenance.get("source_reintegration_manifest_sha256") != recipe["reintegration"]["sha256"]
            or mesh.get("source", {}).get("surfels_manifest_sha256") != recipe["reintegration"]["sha256"]
            or reintegration.get("existing_kitchen_family_authority", {}).get("manifest_sha256") != recipe["family_kitchen"]["sha256"]
            or reintegration.get("registration", {}).get("sha256") != recipe["endpoint_registration"]["sha256"]):
        raise PcfTrackingReviewError("RGB mesh source chain does not match the selected artifact")
    _keys(recipe["rooms"], {row.camera_id for row in catalog.camera_bindings}, "review room set")
    room_frames = {}
    for camera, row in recipe["rooms"].items():
        _keys(row, {"source_prior", "source_alignment", "source_world_manifest", "mesh_owner", "mesh_name", "source_role"}, "review room source")
        source_prior_path, source_prior_json = _input(row["source_prior"], base)
        source_prior = ScenePriorRevision.model_validate(source_prior_json)
        _, alignment = _input(row["source_alignment"], base)
        _, world_manifest = _input(row["source_world_manifest"], base)
        current = bindings[camera]
        entries = [entry for entry in catalog.revisions if entry.prior_id == current.prior_id]
        if len(entries) != 1:
            raise PcfTrackingReviewError("active prior manifest is ambiguous")
        entry = entries[0]
        current_prior_path = (catalog_path.parent / entry.manifest_path).resolve()
        if _sha(current_prior_path) != entry.manifest_sha256 or current_prior_path.stat().st_size != entry.manifest_size_bytes:
            raise PcfTrackingReviewError("current prior manifest changed")
        current_prior = ScenePriorRevision.model_validate(_json(current_prior_path))
        source_points, current_points = _points(source_prior), _points(current_prior)
        if (source_prior.preview.reference_camera_id != camera
                or current_prior.preview.reference_camera_id != camera
                or source_points.sha256 != current_points.sha256
                or source_points.size_bytes != current_points.size_bytes
                or source_prior.source != current_prior.source
                or source_prior.alignment != current_prior.alignment
                or source_prior.alignment.sha256 != row["source_alignment"]["sha256"]
                or alignment.get("conditioned_cloud_manifest", {}).get("sha256") != row["source_world_manifest"]["sha256"]
                or world_manifest.get("output_coordinate_frame") != "backend_world_m_stream_points"):
            raise PcfTrackingReviewError("current prior geometry is not the exact RGB mesh source frame")
        for prior_path, point in ((source_prior_path, source_points), (current_prior_path, current_points)):
            point_path = (prior_path.parent / point.relative_path).resolve()
            if _sha(point_path) != point.sha256 or point_path.stat().st_size != point.size_bytes:
                raise PcfTrackingReviewError("bound point geometry changed")
        role = row["source_role"]
        if role in {"fixed_room", "moving_room"}:
            realized = family_kitchen.get(role, {})
            if (realized.get("prior_id") != source_prior.prior_id
                    or realized.get("accepted_world_manifest_sha256") != row["source_world_manifest"]["sha256"]
                    or realized.get("per_view_world_corrections_row_major") != {}):
                raise PcfTrackingReviewError("mesh room does not have one replayable rigid edge")
            matrix = _pose(realized.get("global_world_correction_row_major"), "realized room transform")
            join_status = "same_source_geometry" if role == "fixed_room" else "review_only"
        elif role == "living":
            realized = reintegration.get("sources", {}).get(role, {})
            if (realized.get("world_manifest_sha256") != row["source_world_manifest"]["sha256"]
                    or registration.get("sources", {}).get(role, {}).get("prior_id") != source_prior.prior_id):
                raise PcfTrackingReviewError("mesh endpoint transform has a different source")
            matrix = _pose(registration.get("derived_endpoint_transforms", {}).get("living_to_family_row_major"), "realized endpoint transform")
            join_status = str(registration.get("status"))
        else:
            raise PcfTrackingReviewError("unsupported recorded source role")
        owner = str(row["mesh_owner"])
        if owner not in artifact.get("owner_triangle_counts", {}) or int(artifact["owner_triangle_counts"][owner]) <= 0:
            raise PcfTrackingReviewError("RGB mesh lacks the declared room owner")
        values = [float(value) for value in matrix.flatten(order="F")]
        canonical = [format(value, ".17g") for value in values]
        encoded = json.dumps(canonical, separators=(",", ":")).encode("utf-8")
        room_frames[camera] = {
            "owner_mesh_name": row["mesh_name"],
            "frame_binding": _frame_payload(current.frame_binding, current.prior_id),
            "assembly_from_world_col_major": values,
            "assembly_from_world_canonical_values": canonical,
            "assembly_from_world_sha256": hashlib.sha256(encoded).hexdigest(),
            "registration_status": join_status,
            "source_prior_id": source_prior.prior_id,
            "source_prior_manifest_sha256": row["source_prior"]["sha256"],
            "current_prior_manifest_sha256": entry.manifest_sha256,
            "points_sha256": current_points.sha256,
        }
    return {
        "contract": CONTRACT, "contract_version": 1,
        "status": "review_only", "accepted_for_canonical_use": False,
        "assembly_id": descriptor["assembly_id"], "coordinate_frame": "backend_world_m",
        "assembly_gauge": descriptor["assembly_gauge"], "units": "meters",
        "artifact": {key: artifact[key] for key in ("role", "sha256", "size_bytes", "vertex_count", "triangle_count")} | {
            "url": "/api/v1/scenes/current/review-assemblies/whole-home/tracking/artifacts/multiroom_surface_mesh_glb"},
        "provenance": {
            "source_descriptor_sha256": _sha(current_descriptor_path),
            "source_surface_mesh_manifest_sha256": recipe["mesh_manifest"]["sha256"],
            "source_reintegration_manifest_sha256": recipe["reintegration"]["sha256"],
            "source_catalog_sha256": recipe["catalog"]["sha256"],
        },
        "room_frames": room_frames,
    }


def build_high_detail_tracking_review(
    manifest_path: Path, *, recipe_path: Path, low_descriptor: Mapping[str, Any], repo_root: Path,
) -> tuple[dict[str, Any], Path]:
    """Admit one separately selected detail artifact, with exactly the low mesh's metric edges.

    Callers must first revalidate the low descriptor against the current PCF.
    This changes surface appearance only; it cannot accept a room registration.
    """
    manifest_path = manifest_path.resolve()
    value = _json(manifest_path, MAX_RECIPE_BYTES)
    _keys(value, {"contract", "contract_version", "status", "accepted_for_canonical_use",
        "assembly_id", "assembly_gauge", "coordinate_frame", "units", "source", "output", "owners"}, "high-detail manifest")
    if (value["contract"] != "noesis.pcf.tracking_review_high_detail" or value["contract_version"] != 1
            or value["status"] != "review_only" or value["accepted_for_canonical_use"] is not False
            or any(value[key] != low_descriptor[key] for key in ("assembly_id", "assembly_gauge", "coordinate_frame", "units"))):
        raise PcfTrackingReviewError("high-detail mesh has a different assembly or authority")
    recipe = _json(recipe_path.resolve(), MAX_RECIPE_BYTES)
    _keys(value["source"], {"binding_sha256", "low_glb_sha256", "low_mesh_manifest_sha256",
        "reintegration_manifest_sha256", "surfels_npz_sha256", "detail_manifest_sha256"}, "high-detail sources")
    source = value["source"]
    _, reintegration = _input(recipe["reintegration"], repo_root.resolve())
    if (source["binding_sha256"] != _sha(recipe_path)
            or source["low_glb_sha256"] != low_descriptor["artifact"]["sha256"]
            or source["low_mesh_manifest_sha256"] != recipe["mesh_manifest"]["sha256"]
            or source["reintegration_manifest_sha256"] != recipe["reintegration"]["sha256"]
            or source["surfels_npz_sha256"] != reintegration.get("artifacts", {}).get("surfels_npz", {}).get("sha256")):
        raise PcfTrackingReviewError("high-detail source chain differs from the current RGB mesh")
    detail_path = manifest_path.parent / "mesh_manifest.json"
    detail = _json(detail_path)
    if (_sha(detail_path) != source["detail_manifest_sha256"]
            or detail.get("contract") != "noesis.pcf.high_detail_review_surface_mesh"
            or detail.get("contract_version") != 1 or detail.get("accepted_for_canonical_use") is not False
            or detail.get("coordinate_frame") != value["assembly_gauge"]):
        raise PcfTrackingReviewError("high-detail photo/geometry provenance changed")
    output = value["output"]
    _keys(output, {"path", "sha256", "size_bytes", "vertex_count", "triangle_count", "owner_triangle_counts"}, "high-detail output")
    if output["path"] != "multiroom_surface_mesh_high_detail.glb":
        raise PcfTrackingReviewError("high-detail artifact path is not the declared embedded GLB")
    for key, maximum in (("size_bytes", 64*1024*1024), ("vertex_count", 20_000_000), ("triangle_count", 20_000_000)):
        if type(output[key]) is not int or not 0 < output[key] <= maximum:
            raise PcfTrackingReviewError("high-detail artifact exceeds the decode bound")
    artifact_path = manifest_path.parent / output["path"]
    if (artifact_path.is_symlink() or not artifact_path.is_file()
            or artifact_path.stat().st_size != output["size_bytes"] or _sha(artifact_path) != output["sha256"]
            or any(detail.get("output", {}).get(key) != output[key] for key in ("sha256", "size_bytes", "vertex_count", "triangle_count"))):
        raise PcfTrackingReviewError("high-detail artifact digest or geometry counts changed")
    owners = {str(row["mesh_owner"]): (camera, row) for camera, row in recipe["rooms"].items()}
    _keys(value["owners"], set(owners), "high-detail owner set")
    _keys(output["owner_triangle_counts"], set(owners), "high-detail owner counts")
    for owner, (camera, row) in owners.items():
        entry = value["owners"][owner]
        _keys(entry, {"room", "mesh_name", "texture_path", "texture_sha256", "texture_size_bytes",
            "direct_photo_triangle_count", "fallback_triangle_count", "accepted_image_count"}, "high-detail owner")
        if entry["room"] != camera or entry["mesh_name"] != row["mesh_name"]:
            raise PcfTrackingReviewError("high-detail room ownership differs from the bound mesh")
        if entry["texture_path"] != row["mesh_name"] + "_atlas.jpg":
            raise PcfTrackingReviewError("high-detail texture path is invalid")
        for key in ("direct_photo_triangle_count", "fallback_triangle_count", "accepted_image_count", "texture_size_bytes"):
            if type(entry[key]) is not int or not 0 <= entry[key] <= 64*1024*1024:
                raise PcfTrackingReviewError("high-detail owner counts are invalid")
        count = output["owner_triangle_counts"][owner]
        if type(count) is not int or count <= 0 or count != entry["direct_photo_triangle_count"] + entry["fallback_triangle_count"]:
            raise PcfTrackingReviewError("high-detail owner triangles disagree")
        texture = manifest_path.parent / entry["texture_path"]
        if (texture.is_symlink() or not texture.is_file() or texture.stat().st_size != entry["texture_size_bytes"]
                or _sha(texture) != entry["texture_sha256"]):
            raise PcfTrackingReviewError("high-detail source atlas changed")
        recorded = detail.get("owners", {}).get(owner, {}).get("atlas", {})
        if recorded.get("sha256") != entry["texture_sha256"] or recorded.get("size_bytes") != entry["texture_size_bytes"]:
            raise PcfTrackingReviewError("high-detail texture provenance disagrees")
    if sum(output["owner_triangle_counts"].values()) != output["triangle_count"]:
        raise PcfTrackingReviewError("high-detail triangle inventory disagrees")
    result = dict(low_descriptor)
    result.update(contract_version=2, mesh_detail="high")
    result["artifact"] = {"role":"multiroom_surface_mesh_glb",
        "url":"/api/v1/scenes/current/review-assemblies/whole-home/tracking/high/artifacts/multiroom_surface_mesh_glb",
        **{key:output[key] for key in ("sha256", "size_bytes", "vertex_count", "triangle_count")}}
    result["provenance"] = {**low_descriptor["provenance"], "source_high_detail_manifest_sha256":_sha(manifest_path)}
    return result, artifact_path
