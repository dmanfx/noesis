"""Measured, scene-only registration for revisioned world presentation.

This module never changes camera extrinsics, target frame bindings, or person
localization. It builds a gravity-locked renderer similarity from the accepted
lower floor, the measured foyer step, authored floor geometry, and the existing
camera XZ anchors. The inherited yaw/XZ anchors are explicitly recorded as
limited presentation evidence, not a new metric calibration.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from noesis.validation.authored_scene import AuthoredSceneGeometry
from noesis_core.contracts.scene_prior import (
    ScenePriorFrameRef,
    scene_revision_sha256,
    world_to_scene_presentation_sha256,
)
from noesis_core.contracts.support_surface import SupportSurfaceSet


CONTRACT = "noesis.scene_presentation.registration"
CONTRACT_VERSION = 1
MAX_ARTIFACT_BYTES = 256 * 1024


class PresentationRegistrationError(ValueError):
    """The declared scene presentation cannot be admitted."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PresentationRegistrationError(f"expected JSON object: {path}")
    return payload


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _exact_keys(payload: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(payload) != expected:
        raise PresentationRegistrationError(f"{label} has missing or unknown fields")


def _sha(value: Any, *, label: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise PresentationRegistrationError(f"{label} is not a SHA-256 digest")


def _matrix(values: Any, *, label: str) -> np.ndarray:
    if not isinstance(values, (tuple, list)) or len(values) != 16:
        raise PresentationRegistrationError(f"{label} requires 16 column-major values")
    result = np.asarray(values, dtype=np.float64).reshape((4, 4), order="F")
    if not np.isfinite(result).all() or not np.allclose(result[3], (0, 0, 0, 1), atol=1e-9):
        raise PresentationRegistrationError(f"{label} is not a finite affine matrix")
    return result


def _point3(values: Any, *, label: str) -> np.ndarray:
    if not isinstance(values, (tuple, list)) or len(values) != 3:
        raise PresentationRegistrationError(f"{label} requires three coordinates")
    point = np.asarray(values, dtype=np.float64)
    if not np.isfinite(point).all():
        raise PresentationRegistrationError(f"{label} must be finite")
    return point


def _horizontal_yaw(prior: np.ndarray, scale: float) -> tuple[float, np.ndarray]:
    # Project the existing proper orientation onto the target floor and take
    # its nearest proper planar rotation. This preserves its yaw; it does not
    # infer yaw from people, PCF texture, or a visually tuned marker.
    planar = prior[np.ix_((0, 2), (0, 2))] / scale
    left, _, right = np.linalg.svd(planar)
    correction = np.diag((1.0, float(np.linalg.det(left @ right))))
    planar_rotation = left @ correction @ right
    if float(np.linalg.det(planar_rotation)) < 0.999999:
        raise PresentationRegistrationError("projected orientation is not proper")
    yaw = math.atan2(float(planar_rotation[0, 1]), float(planar_rotation[0, 0]))
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)), dtype=np.float64)
    return yaw, rotation


def _lower_floor(binding: Mapping[str, Any], *, camera_id: str) -> None:
    floor = binding.get("world_floor_plane")
    if not isinstance(floor, Mapping):
        raise PresentationRegistrationError(f"{camera_id}: target lower floor is missing")
    normal = _point3(floor.get("normal"), label=f"{camera_id} lower floor normal")
    offset = float(floor.get("offset_m"))
    if not np.allclose(normal, (0.0, 1.0, 0.0), atol=1e-6, rtol=0.0) or abs(offset) > 1e-6:
        raise PresentationRegistrationError(f"{camera_id}: target lower floor is not admitted Y=0")


def derive_entry(
    *, camera_id: str, binding: Mapping[str, Any], similarity: Mapping[str, Any],
    correspondence: Mapping[str, Any], calibration_revision: str,
    support_surface_revision: str | None = None,
) -> dict[str, Any]:
    """Derive one floor-locked presentation from exact existing evidence."""

    _lower_floor(binding, camera_id=camera_id)
    target = binding.get("world_frame")
    source = binding.get("calibration_frame")
    if not isinstance(target, Mapping) or not isinstance(source, Mapping):
        raise PresentationRegistrationError(f"{camera_id}: frame identities are missing")
    frame = ScenePriorFrameRef.model_validate(target)
    if frame.frame_id != "backend_world_m" or source.get("frame_id") != "backend_world_m":
        raise PresentationRegistrationError(f"{camera_id}: unsupported frame identity")
    edge = _matrix(binding.get("target_from_calibration_col_major"), label=f"{camera_id} edge")
    base = _matrix(similarity.get("world_to_scene_col_major"), label="source scene similarity")
    scale = float(similarity.get("scene_per_m"))
    if not math.isfinite(scale) or scale <= 0:
        raise PresentationRegistrationError("existing scene scale is invalid")
    prior = base @ np.linalg.inv(edge)
    singular = np.linalg.svd(prior[:3, :3], compute_uv=False)
    if not np.allclose(singular, scale, atol=scale * 1e-5, rtol=0.0):
        raise PresentationRegistrationError(f"{camera_id}: existing mapping scale is not uniform")
    center_source = _point3(correspondence.get("world_position_m"), label=f"{camera_id} camera center")
    marker_scene = _point3(correspondence.get("scene_position"), label=f"{camera_id} scene marker")
    center_target = (edge @ np.append(center_source, 1.0))[:3]
    yaw, rotation = _horizontal_yaw(prior, scale)
    corrected = np.eye(4, dtype=np.float64)
    corrected[:3, :3] = scale * rotation
    corrected[0, 3] = marker_scene[0] - scale * float(rotation[0] @ center_target)
    corrected[2, 3] = marker_scene[2] - scale * float(rotation[2] @ center_target)
    # Both admitted target lower floors and authored lower room slabs are Y=0.
    # No camera-marker height or person coordinate participates in this row.
    corrected[1, 3] = 0.0
    corrected_center = (corrected @ np.append(center_target, 1.0))[:3]
    marker_residual_m = float((marker_scene[1] - corrected_center[1]) / scale)
    matrix = tuple(float(value) for value in corrected.flatten(order="F"))
    prior_matrix = tuple(float(value) for value in prior.flatten(order="F"))
    source_digest = str(binding.get("transform_sha256") or "")
    if len(source_digest) != 64:
        raise PresentationRegistrationError(f"{camera_id}: source transform digest is missing")
    return {
        "camera_id": camera_id,
        "world_frame": frame.model_dump(mode="json"),
        "calibration_frame_revision": str(source.get("revision") or ""),
        "calibration_revision": calibration_revision,
        "source_transform_sha256": source_digest,
        "prior_mapping_sha256": world_to_scene_presentation_sha256(frame, "menon_scene", prior_matrix),
        "world_to_scene_col_major": list(matrix),
        "world_to_scene_sha256": world_to_scene_presentation_sha256(frame, "menon_scene", matrix),
        "scene_revision_id": scene_revision_sha256(frame, "menon_scene", matrix),
        "support_surface_revision": support_surface_revision,
        "constraints": {
            "source": "admitted_pcf_lower_floor_and_authored_floor",
            "lower_floor_world_normal": [0.0, 1.0, 0.0],
            "lower_floor_world_offset_m": 0.0,
            "lower_floor_scene_y": 0.0,
            "scene_per_m": scale,
            "yaw_source": "existing_proper_orientation_projected_to_floor",
            "inherited_yaw_rad": yaw,
            "xz_anchor_source": "independent_existing_camera_marker",
            "camera_center_target_m": [float(value) for value in center_target],
            "camera_marker_scene_xz": [float(marker_scene[0]), float(marker_scene[2])],
            "camera_marker_scene_y_diagnostic": float(marker_scene[1]),
            "corrected_camera_scene_y": float(corrected_center[1]),
            "camera_marker_height_residual_m": marker_residual_m,
            "limitation": "yaw and XZ inherit existing camera marker registration; no independent authored room corner fit",
        },
    }


def _top_y(geometry: AuthoredSceneGeometry, group: str) -> float:
    heights = [
        float(geometry.vertices[index][1])
        for surface in geometry.walkable_triangles
        if surface.authority_group == group
        for index in surface.triangle.vertex_indices
    ]
    if not heights or max(heights) - min(heights) > 1e-3:
        raise PresentationRegistrationError(f"authored floor {group} has no uniform top plane")
    return float(sum(heights) / len(heights))


def build_registration(
    *, bundle_path: str | Path, authored_obj_path: str | Path,
    parent_authored_obj_path: str | Path, room_group_map_path: str | Path,
    support_measurements_path: str | Path, support_surfaces_path: str | Path,
    upper_room: str, lower_rooms: tuple[str, ...],
) -> dict[str, Any]:
    """Build a deterministic scene-only artifact from explicit current files."""

    bundle = _read_json(bundle_path)
    rooms = _read_json(room_group_map_path)
    measurements = _read_json(support_measurements_path)
    surface_set = SupportSurfaceSet.model_validate(_read_json(support_surfaces_path))
    derived = AuthoredSceneGeometry.from_obj(authored_obj_path)
    parent = AuthoredSceneGeometry.from_obj(parent_authored_obj_path)
    if rooms.get("authored_scene_sha256") != parent.source_sha256:
        raise PresentationRegistrationError("room group map is not bound to the parent OBJ")
    room_groups = rooms.get("rooms")
    if not isinstance(room_groups, Mapping):
        raise PresentationRegistrationError("authored room group map is missing")
    if not upper_room or not lower_rooms or len(set(lower_rooms)) != len(lower_rooms) or upper_room in lower_rooms:
        raise PresentationRegistrationError("one upper room and distinct lower rooms are required")
    foyer_groups = room_groups.get(upper_room)
    if not isinstance(foyer_groups, list) or len(foyer_groups) != 1:
        raise PresentationRegistrationError("exactly one authored upper-room floor group is required")
    foyer_group = str(foyer_groups[0])
    for label in lower_rooms:
        groups = room_groups.get(label)
        if not isinstance(groups, list) or not groups:
            raise PresentationRegistrationError(f"authored lower floor group is missing: {label}")
        for group in groups:
            if abs(_top_y(derived, str(group))) > 1e-3:
                raise PresentationRegistrationError(f"authored lower floor is not Y=0: {group}")
    if abs(_top_y(parent, foyer_group)) > 1e-3:
        raise PresentationRegistrationError("parent authored upper-room floor is not Y=0")
    rises = measurements.get("rise_observations")
    if not isinstance(rises, list) or len(rises) < 2:
        raise PresentationRegistrationError("two independent measured step observations are required")
    rise_values = [float(item["floor_to_floor_rise_m"]) for item in rises]
    if not all(math.isfinite(value) and value > 0 for value in rise_values) or max(rise_values) - min(rise_values) > 1e-6:
        raise PresentationRegistrationError("measured step rise observations disagree")
    rise_m = sum(rise_values) / len(rise_values)
    support = next((item for item in surface_set.surfaces if item.surface_id == measurements.get("surface_id")), None)
    if surface_set.authority != "measured_calibration" or support is None:
        raise PresentationRegistrationError("measured foyer support surface is unavailable")
    if not np.allclose(support.normal_world, (0, 1, 0), atol=1e-6, rtol=0) or abs(support.offset_m + rise_m) > 1e-6:
        raise PresentationRegistrationError("measured foyer plane disagrees with step rise")
    align = bundle.get("align")
    similarity = align.get("scene_similarity") if isinstance(align, Mapping) else None
    cameras = bundle.get("cameras")
    bindings = cameras.get("frame_bindings") if isinstance(cameras, Mapping) else None
    if not isinstance(similarity, Mapping) or not isinstance(bindings, Mapping):
        raise PresentationRegistrationError("current calibration bundle lacks scene and frame bindings")
    scale = float(similarity.get("scene_per_m"))
    expected_foyer_y = rise_m * scale
    if abs(_top_y(derived, foyer_group) - expected_foyer_y) > 1e-3:
        raise PresentationRegistrationError("derived Foyer top disagrees with measured metric rise")
    correspondences = similarity.get("correspondences")
    if not isinstance(correspondences, list):
        raise PresentationRegistrationError("independent camera XZ anchors are missing")
    by_camera = {str(item.get("camera_id")): item for item in correspondences if isinstance(item, Mapping)}
    if set(bindings) != set(by_camera):
        raise PresentationRegistrationError("camera XZ anchors and target bindings differ")
    support_binding = bindings.get(surface_set.camera_id)
    if not isinstance(support_binding, Mapping) or (
        support_binding.get("world_frame", {}).get("revision") != surface_set.world_frame_revision
        or support_binding.get("transform_sha256") != surface_set.world_transform_sha256
        or support_binding.get("camera_calibration_sha256") != surface_set.calibration_revision
    ):
        raise PresentationRegistrationError("measured surface does not match active calibration and target frame")
    if measurements.get("camera_id") != surface_set.camera_id or measurements.get("pcf_manifest_path", "").find(surface_set.world_frame_revision) < 0:
        raise PresentationRegistrationError("measured rise does not name active PCF revision")
    entries: dict[str, dict[str, Any]] = {}
    for camera_id, binding in sorted(bindings.items()):
        if not isinstance(binding, Mapping):
            raise PresentationRegistrationError(f"invalid frame binding: {camera_id}")
        revision = str(binding.get("world_frame", {}).get("revision") or "")
        entry = derive_entry(
            camera_id=str(camera_id), binding=binding, similarity=similarity,
            correspondence=by_camera[camera_id],
            calibration_revision=str(binding.get("camera_calibration_sha256") or ""),
            support_surface_revision=(support.revision if camera_id == surface_set.camera_id else None),
        )
        if revision in entries:
            raise PresentationRegistrationError("multiple cameras share one target revision without a common registration")
        entries[revision] = entry
    artifact: dict[str, Any] = {
        "contract": CONTRACT,
        "contract_version": CONTRACT_VERSION,
        "authored_scene_sha256": derived.source_sha256,
        "parent_authored_scene_sha256": parent.source_sha256,
        "room_group_map_sha256": sha256_file(room_group_map_path),
        "support_measurements_sha256": sha256_file(support_measurements_path),
        "support_surfaces_sha256": sha256_file(support_surfaces_path),
        "support_surface_evidence_sha256": surface_set.evidence_sha256,
        "support_camera_id": surface_set.camera_id,
        "support_calibration_revision": surface_set.calibration_revision,
        "support_world_transform_sha256": surface_set.world_transform_sha256,
        "support_surface_id": support.surface_id,
        "support_surface_revision": support.revision,
        "support_surface_normal_world": list(support.normal_world),
        "support_surface_offset_m": support.offset_m,
        "upper_room": upper_room,
        "lower_rooms": list(lower_rooms),
        "foyer_group": foyer_group,
        "foyer_rise_m": rise_m,
        "foyer_top_scene_y": expected_foyer_y,
        "presentations": entries,
    }
    artifact["registration_sha256"] = _canonical_sha256(artifact)
    return artifact


def load_registration(
    path: str | Path, *, authored_obj_path: str | Path,
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file() or not 0 < source.stat().st_size <= 512 * 1024:
        raise PresentationRegistrationError("scene presentation artifact is missing or oversized")
    artifact = _read_json(source)
    if artifact.get("contract") in ("noesis.scene_presentation.structural_registration",
                                   "noesis.scene_presentation.owner_selected_structural_registration"):
        from noesis.calibration.structural_presentation import load_structural_presentation
        return load_structural_presentation(source, authored_obj_path=authored_obj_path)
    if source.stat().st_size > MAX_ARTIFACT_BYTES:
        raise PresentationRegistrationError("scene presentation artifact is oversized")
    _exact_keys(artifact, {
        "contract", "contract_version", "authored_scene_sha256",
        "parent_authored_scene_sha256", "room_group_map_sha256",
        "support_measurements_sha256", "support_surfaces_sha256",
        "support_surface_evidence_sha256", "support_camera_id",
        "support_calibration_revision", "support_world_transform_sha256",
        "support_surface_id", "support_surface_revision",
        "support_surface_normal_world", "support_surface_offset_m",
        "upper_room", "lower_rooms", "foyer_group", "foyer_rise_m",
        "foyer_top_scene_y", "presentations", "registration_sha256",
    }, label="scene presentation artifact")
    if artifact.get("contract") != CONTRACT or artifact.get("contract_version") != CONTRACT_VERSION:
        raise PresentationRegistrationError("unsupported scene presentation contract")
    for key in (
        "authored_scene_sha256", "parent_authored_scene_sha256",
        "room_group_map_sha256", "support_measurements_sha256",
        "support_surfaces_sha256", "support_surface_evidence_sha256",
        "support_world_transform_sha256", "registration_sha256",
    ):
        _sha(artifact[key], label=key)
    for key in ("support_camera_id", "support_calibration_revision", "support_surface_id",
                "support_surface_revision", "upper_room", "foyer_group"):
        if not isinstance(artifact[key], str) or not artifact[key]:
            raise PresentationRegistrationError(f"{key} must be nonempty")
    if not isinstance(artifact["lower_rooms"], list) or not artifact["lower_rooms"] or not all(
        isinstance(room, str) and room for room in artifact["lower_rooms"]
    ) or len(set(artifact["lower_rooms"])) != len(artifact["lower_rooms"]):
        raise PresentationRegistrationError("lower room names are invalid")
    _point3(artifact["support_surface_normal_world"], label="support normal")
    for key in ("support_surface_offset_m", "foyer_rise_m", "foyer_top_scene_y"):
        if not isinstance(artifact[key], (int, float)) or not math.isfinite(artifact[key]):
            raise PresentationRegistrationError(f"{key} must be finite")
    digest = artifact.get("registration_sha256")
    content = {key: value for key, value in artifact.items() if key != "registration_sha256"}
    if digest != _canonical_sha256(content):
        raise PresentationRegistrationError("scene presentation registration digest mismatch")
    if artifact.get("authored_scene_sha256") != sha256_file(authored_obj_path):
        raise PresentationRegistrationError("loaded authored OBJ does not match presentation registration")
    entries = artifact.get("presentations")
    if not isinstance(entries, dict) or not entries or len(entries) > 8:
        raise PresentationRegistrationError("scene presentation entries are missing or unbounded")
    cameras: set[str] = set()
    for revision, entry in entries.items():
        if not isinstance(entry, Mapping):
            raise PresentationRegistrationError("scene presentation entry is not an object")
        _exact_keys(entry, {
            "camera_id", "world_frame", "calibration_frame_revision",
            "calibration_revision", "source_transform_sha256",
            "prior_mapping_sha256", "world_to_scene_col_major",
            "world_to_scene_sha256", "scene_revision_id",
            "support_surface_revision", "constraints",
        }, label="scene presentation entry")
        frame = ScenePriorFrameRef.model_validate(entry["world_frame"])
        if frame.revision != revision or frame.frame_id != "backend_world_m":
            raise PresentationRegistrationError("scene presentation revision key is invalid")
        camera_id = entry["camera_id"]
        if not isinstance(camera_id, str) or not camera_id or camera_id in cameras:
            raise PresentationRegistrationError("scene presentation camera identity is invalid")
        cameras.add(camera_id)
        for key in ("calibration_frame_revision", "calibration_revision"):
            if not isinstance(entry[key], str) or not entry[key]:
                raise PresentationRegistrationError(f"{key} must be nonempty")
        for key in ("source_transform_sha256", "prior_mapping_sha256",
                    "world_to_scene_sha256", "scene_revision_id"):
            _sha(entry[key], label=key)
        _matrix(entry["world_to_scene_col_major"], label="scene presentation matrix")
        if entry["world_to_scene_sha256"] != world_to_scene_presentation_sha256(
            frame, "menon_scene", tuple(entry["world_to_scene_col_major"])
        ) or entry["scene_revision_id"] != scene_revision_sha256(
            frame, "menon_scene", tuple(entry["world_to_scene_col_major"])
        ):
            raise PresentationRegistrationError("scene presentation matrix digest is invalid")
        if entry["support_surface_revision"] is not None and not isinstance(entry["support_surface_revision"], str):
            raise PresentationRegistrationError("support surface revision is invalid")
        constraints = entry["constraints"]
        if not isinstance(constraints, Mapping):
            raise PresentationRegistrationError("scene presentation constraints are missing")
        _exact_keys(constraints, {
            "source", "lower_floor_world_normal", "lower_floor_world_offset_m",
            "lower_floor_scene_y", "scene_per_m", "yaw_source",
            "inherited_yaw_rad", "xz_anchor_source", "camera_center_target_m",
            "camera_marker_scene_xz", "camera_marker_scene_y_diagnostic",
            "corrected_camera_scene_y", "camera_marker_height_residual_m",
            "limitation",
        }, label="scene presentation constraints")
        _point3(constraints["lower_floor_world_normal"], label="lower floor normal")
        _point3(constraints["camera_center_target_m"], label="camera center")
        if not isinstance(constraints["camera_marker_scene_xz"], list) or len(constraints["camera_marker_scene_xz"]) != 2 or not all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in constraints["camera_marker_scene_xz"]
        ):
            raise PresentationRegistrationError("camera marker XZ is invalid")
        for key in ("lower_floor_world_offset_m", "lower_floor_scene_y", "scene_per_m",
                    "inherited_yaw_rad", "camera_marker_scene_y_diagnostic",
                    "corrected_camera_scene_y", "camera_marker_height_residual_m"):
            if not isinstance(constraints[key], (int, float)) or not math.isfinite(constraints[key]):
                raise PresentationRegistrationError(f"{key} must be finite")
        for key in ("source", "yaw_source", "xz_anchor_source", "limitation"):
            if not isinstance(constraints[key], str) or not constraints[key]:
                raise PresentationRegistrationError(f"{key} must be nonempty")
    return artifact
