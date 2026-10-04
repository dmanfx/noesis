"""Revision-bound image rays for review-only structural registration points.

An opening jamb is first intersected with a verified measured PCF wall plane.
Its horizontal tangent coordinate is then transferred to that plane's Y=0
intersection line. The glass pixel is never treated as a floor contact. An
actual floor contact instead intersects an explicitly supplied bound support
height. Neither operation changes tracking or canonical world authority.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from noesis.calibration.structural_registration import StructuralRegistrationError


MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_IMAGE_SIDE = 8192
MAX_RAY_RANGE_M = 30.0
MIN_RAY_INCIDENCE = 0.10


def _point2(values: Any, label: str) -> tuple[float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 2 or any(
        isinstance(value, (bool, str, bytes)) for value in values
    ):
        raise StructuralRegistrationError(f"{label} requires two numeric values")
    result = (float(values[0]), float(values[1]))
    if not all(math.isfinite(value) for value in result):
        raise StructuralRegistrationError(f"{label} must be finite")
    return result


def _matrix4(values: Any, label: str) -> np.ndarray:
    if not isinstance(values, (list, tuple)) or len(values) != 16 or any(
        isinstance(value, (bool, str, bytes)) for value in values
    ):
        raise StructuralRegistrationError(f"{label} requires 16 column-major numeric values")
    result = np.asarray(values, dtype=np.float64).reshape((4, 4), order="F")
    if not np.isfinite(result).all() or not np.allclose(result[3], (0, 0, 0, 1), atol=1e-9, rtol=0):
        raise StructuralRegistrationError(f"{label} is not finite affine geometry")
    rotation = result[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5,
    ):
        raise StructuralRegistrationError(f"{label} is not a proper rigid transform")
    return result


def _image_ref(path: Path, expected_size: tuple[int, int]) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise StructuralRegistrationError("structural reference image must be a regular file")
    size = path.stat().st_size
    if not 1 <= size <= MAX_IMAGE_BYTES:
        raise StructuralRegistrationError("structural reference image exceeds bound")
    with Image.open(path) as image:
        if image.format not in ("PNG", "JPEG") or image.size != expected_size:
            raise StructuralRegistrationError("structural image format or raster size disagrees")
        if image.getexif().get(274, 1) != 1:
            raise StructuralRegistrationError("rotated EXIF structural image is unsupported")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "size_bytes": size,
            "width": expected_size[0], "height": expected_size[1]}


def _binding(bundle: Mapping[str, Any], camera_id: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[float]]:
    cameras = bundle.get("cameras")
    if not isinstance(cameras, Mapping):
        raise StructuralRegistrationError("calibration bundle lacks cameras")
    e_table, k_table, frame_bindings = (cameras.get(key) for key in ("E", "K", "frame_bindings"))
    if not all(isinstance(table, Mapping) for table in (e_table, k_table, frame_bindings)):
        raise StructuralRegistrationError("calibration bundle lacks E/K/frame bindings")
    binding = frame_bindings.get(camera_id)
    if not isinstance(binding, Mapping):
        raise StructuralRegistrationError("camera has no target frame binding")
    world_frame = binding.get("world_frame")
    calibration_frame = binding.get("calibration_frame")
    if not isinstance(world_frame, Mapping) or not isinstance(calibration_frame, Mapping) or (
        world_frame.get("frame_id") != "backend_world_m" or calibration_frame.get("frame_id") != "backend_world_m"
    ):
        raise StructuralRegistrationError("structural ray has unsupported frame identity")
    if world_frame.get("revision") != binding.get("scene_prior_id"):
        raise StructuralRegistrationError("structural ray target revision is not its admitted scene prior")
    E = _matrix4(e_table.get(camera_id), "camera_from_calibration E")
    target_from_calibration = _matrix4(binding.get("target_from_calibration_col_major"),
                                       "target_from_calibration")
    K = k_table.get(camera_id)
    if not isinstance(K, (list, tuple)) or len(K) != 4 or any(isinstance(v, (bool, str, bytes)) for v in K):
        raise StructuralRegistrationError("camera K requires fx,fy,cx,cy")
    intrinsics = [float(v) for v in K]
    if not all(math.isfinite(value) for value in intrinsics) or min(intrinsics[:2]) <= 0:
        raise StructuralRegistrationError("camera K is invalid")
    camera_to_target = target_from_calibration @ np.linalg.inv(E)
    return camera_to_target, E, dict(binding), intrinsics


def _plane_from_verified_segment(segment: Any) -> tuple[np.ndarray, float, str, str]:
    if not isinstance(segment, Mapping):
        segment = {key: getattr(segment, key, None) for key in
                   ("segment_id", "start_xz", "end_xz", "source_refs", "support_count", "vertical_span")}
    segment_id = segment.get("segment_id")
    if not isinstance(segment_id, str) or not segment_id.startswith("pcf_plane_"):
        raise StructuralRegistrationError("opening requires a measured PCF plane segment")
    refs = segment.get("source_refs")
    if not isinstance(refs, (tuple, list)) or len(refs) > 16:
        raise StructuralRegistrationError("measured PCF plane lacks bounded source refs")
    try:
        metadata = dict(ref.split(":", 1) for ref in refs)
        if metadata.get("method") != "vertical_plane_ransac" or metadata.get("pcf") != "points.glb":
            raise ValueError("wrong source method")
        normal = np.asarray([float(v) for v in metadata["normal_xyz"].split(",")], dtype=np.float64)
        d = float(metadata["plane_d_m"])
        support_sha = metadata["support_sha256"]
    except (ValueError, KeyError, AttributeError) as exc:
        raise StructuralRegistrationError("measured PCF plane refs are incomplete") from exc
    if normal.shape != (3,) or not np.isfinite(normal).all() or not math.isfinite(d) or (
        abs(float(np.linalg.norm(normal)) - 1.0) > 1e-4 or np.linalg.norm(normal[[0, 2]]) < 0.95
    ):
        raise StructuralRegistrationError("PCF wall plane normal/offset are invalid")
    if not isinstance(support_sha, str) or len(support_sha) != 64 or any(ch not in "0123456789abcdef" for ch in support_sha):
        raise StructuralRegistrationError("PCF wall plane support digest is invalid")
    if type(segment.get("support_count")) is not int or segment["support_count"] < 200 or (
        not math.isfinite(float(segment.get("vertical_span", 0))) or float(segment["vertical_span"]) < 1.2
    ):
        raise StructuralRegistrationError("PCF wall plane has inadequate measured support")
    start, end = (_point2(segment.get(key), f"PCF wall {key}") for key in ("start_xz", "end_xz"))
    horizontal = normal[[0, 2]]
    line_offset = -d / float(np.linalg.norm(horizontal))
    horizontal /= np.linalg.norm(horizontal)
    if any(abs(float(horizontal @ np.asarray(point)) - line_offset) > 0.02 for point in (start, end)):
        raise StructuralRegistrationError("PCF wall endpoints disagree with measured floor intersection")
    return normal, d, segment_id, support_sha


def _ray(camera_to_target: np.ndarray, intrinsics: Sequence[float], pixel: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    fx, fy, cx, cy = intrinsics
    camera_direction = np.asarray(((pixel[0] - cx) / fx, (pixel[1] - cy) / fy, 1.0), dtype=np.float64)
    camera_direction /= np.linalg.norm(camera_direction)
    direction = camera_to_target[:3, :3] @ camera_direction
    return camera_to_target[:3, 3], direction


def project_structural_image_point(
    *, bundle: Mapping[str, Any], camera_id: str, image_path: Path,
    image_size: Sequence[int], pixel: Sequence[float], kind: str, point_id: str,
    plane_segment: Any | None, surface_y_m: float | None,
) -> dict[str, Any]:
    """Project a reviewed image feature to a structural target-world XZ point.

    The image/plane/calibration exact-byte bindings are recorded for the
    caller's artifact builder to compare against current admitted sources.
    Pixels must already be in the rectified K raster, with no further resize.
    """
    if not isinstance(camera_id, str) or not camera_id or not isinstance(point_id, str) or not point_id or len(point_id) > 160:
        raise StructuralRegistrationError("structural image point has invalid identity")
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2 or any(type(v) is not int for v in image_size):
        raise StructuralRegistrationError("structural image_size requires exact integer width/height")
    width, height = image_size
    if not 1 <= width <= MAX_IMAGE_SIDE or not 1 <= height <= MAX_IMAGE_SIDE:
        raise StructuralRegistrationError("structural image dimensions exceed bound")
    uv = _point2(pixel, "structural pixel")
    if not (0 <= uv[0] < width and 0 <= uv[1] < height):
        raise StructuralRegistrationError("structural pixel lies outside its exact raster")
    image = _image_ref(Path(image_path), (width, height))
    camera_to_target, E, binding, intrinsics = _binding(bundle, camera_id)
    origin, direction = _ray(camera_to_target, intrinsics, uv)
    if kind == "opening":
        if plane_segment is None or surface_y_m is not None:
            raise StructuralRegistrationError("opening requires wall plane, never floor-ray height")
        normal, offset, segment_id, support_sha = _plane_from_verified_segment(plane_segment)
        denominator = float(normal @ direction)
        incidence = abs(denominator)
        if incidence < MIN_RAY_INCIDENCE:
            raise StructuralRegistrationError("opening ray is too parallel to measured wall")
        distance = -float(normal @ origin + offset) / denominator
        if not 0 < distance <= MAX_RAY_RANGE_M:
            raise StructuralRegistrationError("opening ray does not hit measured wall in front")
        hit = origin + distance * direction
        if not -0.5 <= float(hit[1]) <= 4.0:
            raise StructuralRegistrationError("opening wall hit has implausible height")
        horizontal = normal[[0, 2]].copy()
        horizontal_norm = float(np.linalg.norm(horizontal))
        horizontal /= horizontal_norm
        tangent = np.asarray((-horizontal[1], horizontal[0]))
        floor_line_offset = -offset / horizontal_norm
        tangent_coordinate = float(tangent @ hit[[0, 2]])
        point_xz = floor_line_offset * horizontal + tangent_coordinate * tangent
        method = "ray_measured_pcf_wall_then_floor_line_tangent_v1"
        source_wall_ids = [segment_id]
        surface = None
        plane_evidence = {"segment_id": segment_id, "plane_normal_xyz": normal.tolist(),
                          "plane_d_m": offset, "support_sha256": support_sha,
                          "tangent_coordinate_m": tangent_coordinate,
                          "floor_line_offset_m": floor_line_offset}
    elif kind == "corner":
        if plane_segment is not None or surface_y_m is None or isinstance(surface_y_m, (bool, str, bytes)):
            raise StructuralRegistrationError("floor contact requires explicit surface height, not wall plane")
        surface = float(surface_y_m)
        if not math.isfinite(surface) or not -20.0 <= surface <= 20.0:
            raise StructuralRegistrationError("floor contact surface height is invalid")
        floor = binding.get("world_floor_plane")
        if not isinstance(floor, Mapping) or not np.allclose(floor.get("normal"), (0, 1, 0), atol=1e-6, rtol=0) or abs(float(floor.get("offset_m", math.nan))) > 1e-6:
            raise StructuralRegistrationError("floor contact target lacks admitted Y=0 lower floor")
        denominator = float(direction[1])
        incidence = abs(denominator)
        if incidence < MIN_RAY_INCIDENCE:
            raise StructuralRegistrationError("floor-contact ray is too parallel to support plane")
        distance = (surface - float(origin[1])) / denominator
        if not 0 < distance <= MAX_RAY_RANGE_M:
            raise StructuralRegistrationError("floor-contact ray does not hit support in front")
        hit = origin + distance * direction
        point_xz = hit[[0, 2]].copy()
        method = "ray_bound_horizontal_support_plane_v1"
        source_wall_ids = []
        plane_evidence = None
    else:
        raise StructuralRegistrationError("structural image point kind must be opening or corner")
    calibration_sha = binding.get("camera_calibration_sha256")
    transform_sha = binding.get("transform_sha256")
    target_revision = binding["world_frame"].get("revision")
    if any(not isinstance(value, str) or len(value) != 64 for value in (calibration_sha, transform_sha)):
        raise StructuralRegistrationError("structural ray calibration identities are incomplete")
    uv_ref = f"pixel_uv:{uv[0]:.6f},{uv[1]:.6f}"
    evidence_refs = [f"image_sha256:{image['sha256']}", uv_ref, f"method:{method}"]
    if plane_evidence is not None:
        evidence_refs.append(f"pcf_wall_support_sha256:{plane_evidence['support_sha256']}")
    else:
        evidence_refs.append(f"support_y_m:{surface:.6f}")
    point = {"point_id": point_id, "kind": kind, "xz": point_xz.tolist(),
             "source_wall_ids": source_wall_ids, "evidence_refs": evidence_refs}
    image_evidence = {
        "method": method, "camera_id": camera_id,
        "image": image, "pixel_uv": list(uv), "pixel_space": "rectified_source_image",
        "K4": intrinsics, "E_col_major": E.flatten(order="F").tolist(),
        "target_from_calibration_col_major": binding["target_from_calibration_col_major"],
        "target_world_revision": target_revision,
        "camera_calibration_sha256": calibration_sha,
        "target_transform_sha256": transform_sha,
        "camera_origin_target_m": origin.tolist(),
        "ray_direction_target": direction.tolist(),
        "ray_hit_target_m": hit.tolist(),
        "ray_range_m": float(distance), "incidence_sin": incidence,
        "surface_y_m": surface, "wall_plane": plane_evidence,
        "projected_xz_target_m": point_xz.tolist(),
    }
    return {"point": point, "image_evidence": image_evidence}


def verify_structural_image_point(
    frozen: Mapping[str, Any], **projection_kwargs: Any,
) -> dict[str, Any]:
    """Reproject exact bound inputs and compare every frozen geometry field."""
    replay = project_structural_image_point(**projection_kwargs)
    if not isinstance(frozen, Mapping) or set(frozen) != set(replay):
        raise StructuralRegistrationError("structural image evidence fields differ from replay")

    def compare(left: Any, right: Any, label: str) -> None:
        if type(left) is not type(right):
            raise StructuralRegistrationError(f"{label} differs from image ray replay")
        if isinstance(left, dict):
            if set(left) != set(right):
                raise StructuralRegistrationError(f"{label} fields differ from image ray replay")
            for key in left:
                compare(left[key], right[key], f"{label}.{key}")
        elif isinstance(left, list):
            if len(left) != len(right):
                raise StructuralRegistrationError(f"{label} length differs from image ray replay")
            for index, (a, b) in enumerate(zip(left, right)):
                compare(a, b, f"{label}[{index}]")
        elif isinstance(left, float):
            if not math.isfinite(left) or not math.isclose(left, right, abs_tol=1e-7, rel_tol=1e-9):
                raise StructuralRegistrationError(f"{label} differs from image ray replay")
        elif left != right:
            raise StructuralRegistrationError(f"{label} differs from image ray replay")

    compare(dict(frozen), replay, "image_evidence")
    return replay
