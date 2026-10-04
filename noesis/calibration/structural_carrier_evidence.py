"""Bounded proof for structural features measured in a retained phone carrier.

The expensive admitted-PCF point selection and robust plane fit run only while
building an artifact. Startup verifies exact source bytes, the frozen plane and
line math, and each reviewed image ray. No phone reconstruction is promoted to
live calibration or person-world authority by this module.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from zipfile import ZipFile

import numpy as np

from noesis.calibration.presentation_registration import sha256_file
from noesis.calibration.structural_registration import (
    StructuralRegistrationError, _compare_replay, read_pcf_points_glb,
    verify_pcf_source_segments,
)


KIND = "retained_carrier_pcf_roi_plane"
MAX_VIEWS = 16
MAX_WALLS = 8
MAX_ROIS_PER_WALL = 12
MAX_POINTS = 4
MAX_NPZ_BYTES = 16 * 1024 * 1024
MAX_UNCOMPRESSED_NPZ_BYTES = 32 * 1024 * 1024
MAX_IMAGE_SIDE = 8192


def _fail(message: str) -> None:
    raise StructuralRegistrationError(message)


def _keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        _fail(f"{label} has missing or unknown fields")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, (bool, str, bytes)):
        _fail(f"{label} must be finite numeric evidence")
    try:
        number = float(value)
    except (TypeError, ValueError):
        _fail(f"{label} must be finite numeric evidence")
    if not math.isfinite(number):
        _fail(f"{label} must be finite numeric evidence")
    return number


def _vector(value: Any, length: int, label: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        _fail(f"{label} has wrong shape")
    return np.asarray([_finite(v, label) for v in value], dtype=np.float64)


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        _fail(f"{label} is not a SHA-256 digest")
    return value


def _file(ref: Any, root: Path, *, maximum: int) -> Path:
    ref = _keys(ref, {"path", "sha256", "size_bytes"}, "carrier file reference")
    relative = ref["path"]
    if (not isinstance(relative, str) or not relative or relative.startswith("/")
            or "\\" in relative or "\x00" in relative
            or any(part in ("", ".", "..") for part in relative.split("/"))):
        _fail("carrier evidence path must be portable and root relative")
    _sha(ref["sha256"], "carrier file")
    size = ref["size_bytes"]
    if type(size) is not int or not 0 < size <= maximum:
        _fail("carrier evidence file size exceeds its bound")
    path = root / PurePosixPath(relative)
    cursor = root
    for part in PurePosixPath(relative).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            _fail("carrier evidence path contains a symlink")
    if not path.is_file() or path.stat().st_size != size or sha256_file(path) != ref["sha256"]:
        _fail("carrier evidence bytes differ from their frozen reference")
    return path


def _json(ref: Any, root: Path) -> dict[str, Any]:
    import json

    path = _file(ref, root, maximum=1024 * 1024)
    try:
        result = json.loads(path.read_text())
    except (UnicodeError, ValueError) as exc:
        raise StructuralRegistrationError("carrier manifest is not valid JSON") from exc
    if not isinstance(result, dict):
        _fail("carrier manifest is not an object")
    return result


def _npz(ref: Any, root: Path, *, image_only: bool = False) -> dict[str, np.ndarray]:
    path = _file(ref, root, maximum=MAX_NPZ_BYTES)
    try:
        with ZipFile(path) as archive:
            if len(archive.infolist()) > 32 or sum(i.file_size for i in archive.infolist()) > MAX_UNCOMPRESSED_NPZ_BYTES:
                _fail("carrier NPZ has unbounded array contents")
        required = {"model_rgb", "camera_pose", "intrinsics"}
        if not image_only:
            required |= {"world_points", "confidence", "source_selection", "cross_model_uncertain"}
        with np.load(path, allow_pickle=False) as arrays:
            if not required.issubset(arrays.files):
                _fail("carrier view lacks required measured arrays")
            result = {key: np.asarray(arrays[key]) for key in required}
    except (OSError, ValueError, KeyError) as exc:
        raise StructuralRegistrationError("carrier NPZ cannot be decoded") from exc
    rgb = result["model_rgb"]
    if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
            or max(rgb.shape[:2]) > MAX_IMAGE_SIDE):
        _fail("carrier RGB is not a bounded exact raster")
    if result["intrinsics"].shape != (3, 3) or result["camera_pose"].shape != (4, 4):
        _fail("carrier K/pose has invalid shape")
    if not image_only:
        size = rgb.shape[:2]
        if (result["world_points"].shape != (*size, 3)
                or any(result[key].shape != size for key in
                       ("confidence", "source_selection", "cross_model_uncertain"))):
            _fail("carrier point/quality arrays disagree with raster")
    return result


def _alignment(source_manifest: Mapping[str, Any]) -> np.ndarray:
    if (source_manifest.get("schema") != "noesis.phone_walk.static_world_point_cloud_review.v1"
            or source_manifest.get("source_coordinate_frame") != "da3_relative_metric_world"
            or source_manifest.get("output_coordinate_frame") != "backend_world_m_stream_points"
            or source_manifest.get("static_reference_points_included") is not False):
        _fail("carrier source manifest has unsupported frame or provenance")
    fit = source_manifest.get("alignment")
    if not isinstance(fit, Mapping) or fit.get("method") != "all_phone_pose_correspondence_sim3_to_validated_da3_backend_path":
        _fail("carrier source alignment is not the retained validated path")
    try:
        rotation = np.asarray(fit["rotation_row_major"], dtype=np.float64)
        scale = _finite(fit["scale"], "carrier scale")
        translation = _vector(fit["translation"], 3, "carrier translation")
    except (KeyError, TypeError, ValueError) as exc:
        raise StructuralRegistrationError("carrier source alignment is malformed") from exc
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all() or scale <= 0
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0)
            or not math.isclose(float(np.linalg.det(rotation)), 1, abs_tol=1e-5)):
        _fail("carrier source alignment is not a proper Sim(3)")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    return matrix


def _consensus(manifest: Mapping[str, Any], views: Mapping[str, Any]) -> None:
    grid = manifest.get("common_ray_grid")
    if (manifest.get("schema") != "noesis.phone_walk.consensus_fusion.v1"
            or manifest.get("phone_walk_only") is not True
            or manifest.get("static_camera_data_used") is not False
            or manifest.get("pose_carrier") != "da3"
            or not isinstance(grid, Mapping)
            or type(grid.get("width")) is not int or type(grid.get("height")) is not int
            or not 1 <= grid["width"] <= MAX_IMAGE_SIDE
            or not 1 <= grid["height"] <= MAX_IMAGE_SIDE
            or type(manifest.get("view_count")) is not int):
        _fail("carrier consensus provenance is not the admitted phone-only method")
    if any(type(key) is not str or len(key) != 4 or not key.isdecimal()
           or int(key) >= manifest["view_count"] for key in views):
        _fail("carrier view index is not bound to consensus")


def _base_plane(segment: Mapping[str, Any]) -> tuple[np.ndarray, float]:
    refs = segment.get("source_refs")
    if not isinstance(refs, list) or len(refs) > 16:
        _fail("base PCF plane refs are missing")
    try:
        fields = dict(ref.split(":", 1) for ref in refs)
        if fields.get("method") != "vertical_plane_ransac" or fields.get("pcf") != "points.glb":
            _fail("base segment is not an admitted RANSAC plane")
        normal = _vector([float(v) for v in fields["normal_xyz"].split(",")], 3, "base plane normal")
        offset = _finite(float(fields["plane_d_m"]), "base plane offset")
    except (KeyError, ValueError) as exc:
        raise StructuralRegistrationError("base plane refs are incomplete") from exc
    if not math.isclose(float(np.linalg.norm(normal)), 1, abs_tol=1e-4) or np.linalg.norm(normal[[0, 2]]) < 0.95:
        _fail("base PCF plane is not near vertical")
    return normal, offset


def _line_at_height(base: Mapping[str, Any], normal: np.ndarray, offset: float,
                    reference_y: float, segment_id: str, source_refs: list[str]) -> dict[str, Any]:
    horizontal = normal[[0, 2]] / np.linalg.norm(normal[[0, 2]])
    line_offset = -(offset + normal[1] * reference_y) / np.linalg.norm(normal[[0, 2]])
    ends = []
    for key in ("start_xz", "end_xz"):
        end = _vector(base[key], 2, "base segment endpoint")
        ends.append((end + (line_offset - horizontal @ end) * horizontal).tolist())
    return {"segment_id": segment_id, "start_xz": ends[0], "end_xz": ends[1],
            "support_count": base["support_count"], "vertical_span": base["vertical_span"],
            "source_refs": source_refs}


def _roi_points(raw: Mapping[str, np.ndarray], box: Any, alignment: np.ndarray,
                height_bounds: np.ndarray) -> np.ndarray:
    if (not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box)):
        _fail("carrier ROI must be exact integer raster bounds")
    x0, y0, x1, y1 = box
    height, width = raw["model_rgb"].shape[:2]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        _fail("carrier ROI lies outside its exact raster")
    points = raw["world_points"][y0:y1, x0:x1].reshape(-1, 3)
    points = points @ alignment[:3, :3].T + alignment[:3, 3]
    quality = raw["confidence"][y0:y1, x0:x1].reshape(-1)
    selection = raw["source_selection"][y0:y1, x0:x1].reshape(-1)
    uncertain = raw["cross_model_uncertain"][y0:y1, x0:x1].reshape(-1)
    good = ((quality >= 0.70) & (selection == 1) & (~uncertain)
            & (points[:, 1] >= height_bounds[0]) & (points[:, 1] <= height_bounds[1])
            & np.isfinite(points).all(axis=1))
    selected = points[good]
    if len(selected) < 100:
        _fail("carrier opaque ROI lacks measured quality support")
    return selected


def _fit_admitted_plane(base: Mapping[str, Any], rois: list[Mapping[str, Any]],
                        views: Mapping[str, Any], alignment: np.ndarray,
                        admitted: np.ndarray, reference_y: float, height_bounds: np.ndarray,
                        root: Path) -> tuple[np.ndarray, float, int]:
    from scipy.optimize import least_squares

    base_normal, base_offset = _base_plane(base)
    horizontal = base_normal[[0, 2]] / np.linalg.norm(base_normal[[0, 2]])
    tangent = np.asarray((-horizontal[1], horizontal[0]))
    origin_t = float(tangent @ ((_vector(base["start_xz"], 2, "base start")
                                  + _vector(base["end_xz"], 2, "base end")) / 2))
    selected_sets = []
    for roi in rois:
        _keys(roi, {"name", "view_index", "box_xyxy", "selected_glb_sha256"}, "carrier ROI")
        view = roi["view_index"]
        if type(view) is not int or f"{view:04d}" not in views:
            _fail("carrier ROI view is absent from bounded evidence")
        refs = _roi_points(_npz(views[f"{view:04d}"], root), roi["box_xyxy"], alignment,
                           height_bounds)
        ref_t = refs[:, [0, 2]] @ tangent
        low_t, high_t = np.quantile(ref_t, [0.02, 0.98]) + np.asarray((-0.05, 0.05))
        low_y, high_y = np.quantile(refs[:, 1], [0.02, 0.98]) + np.asarray((-0.05, 0.05))
        selected = admitted[(admitted[:, [0, 2]] @ tangent >= low_t)
                            & (admitted[:, [0, 2]] @ tangent <= high_t)
                            & (admitted[:, 1] >= low_y) & (admitted[:, 1] <= high_y)
                            & (np.abs(admitted @ base_normal + base_offset) <= 0.18)]
        digest = hashlib.sha256(np.asarray(selected, dtype="<f8").tobytes()).hexdigest()
        if digest != _sha(roi["selected_glb_sha256"], "selected admitted PCF support"):
            _fail("selected admitted PCF support changed")
        selected_sets.append(selected)
    merged = np.concatenate(selected_sets)
    if len(merged) < 200:
        _fail("admitted PCF plane lacks measured support")
    design = np.stack((merged[:, [0, 2]] @ tangent - origin_t,
                       merged[:, 1] - reference_y, np.ones(len(merged))), axis=1)
    solved = least_squares(lambda params: merged @ base_normal + base_offset + design @ params,
                           np.zeros(3), loss="soft_l1", f_scale=0.03)
    a, b, c = solved.x
    normal = base_normal + a * np.asarray((tangent[0], 0, tangent[1])) + b * np.asarray((0, 1, 0))
    offset = base_offset - a * origin_t - b * reference_y + c
    factor = np.linalg.norm(normal)
    return normal / factor, float(offset / factor), len(merged)


def _binding(evidence: Mapping[str, Any], binding: Mapping[str, Any]) -> None:
    if (evidence.get("target_world_revision") != binding.get("world_frame", {}).get("revision")
            or evidence.get("target_transform_sha256") != binding.get("transform_sha256")
            or evidence.get("target_calibration_sha256") != binding.get("camera_calibration_sha256")):
        _fail("carrier proof does not target this admitted camera/PCF revision")


def verify_source_feature_evidence(candidate: Mapping[str, Any], points_path: str | Path,
                                   evidence: Mapping[str, Any], root: Path, *, offline: bool,
                                   binding: Mapping[str, Any]) -> None:
    """Verify the exact admitted-PCF planes and their bound carrier point rays."""
    expected = {"kind", "capture_id", "source_manifest", "source_bundle_manifest",
                "diagnostic_report", "diagnostic_candidate_key", "consensus_manifest",
                "selection_policy", "views", "reference_y_m",
                "target_world_revision", "target_transform_sha256", "target_calibration_sha256",
                "walls", "point_evidence"}
    _keys(evidence, expected, "carrier source feature evidence")
    if evidence["kind"] != KIND:
        _fail("unsupported carrier feature evidence")
    _binding(evidence, binding)
    views = evidence["views"]
    if not isinstance(views, Mapping) or not 1 <= len(views) <= MAX_VIEWS:
        _fail("carrier proof has unbounded view set")
    source_manifest = _json(evidence["source_manifest"], root)
    bundle_manifest = _json(evidence["source_bundle_manifest"], root)
    diagnostic = _json(evidence["diagnostic_report"], root)
    consensus = _json(evidence["consensus_manifest"], root)
    _consensus(consensus, views)
    alignment = _alignment(source_manifest)
    capture_id = evidence["capture_id"]
    if (not isinstance(capture_id, str) or not 1 <= len(capture_id) <= 160
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in capture_id)):
        _fail("carrier capture identity is invalid")
    source_path = source_manifest.get("source")
    if (not isinstance(source_path, str) or capture_id not in source_path
            or Path(source_path).name != "surfel_points.npz"):
        _fail("carrier source manifest does not name the selected phone capture")
    import json

    admitted_manifest_path = Path(points_path).parent / "manifest.json"
    if not admitted_manifest_path.is_file() or admitted_manifest_path.stat().st_size > 1024 * 1024:
        _fail("admitted PCF manifest is unavailable to carrier proof")
    try:
        admitted_manifest = json.loads(admitted_manifest_path.read_text())
    except (UnicodeError, ValueError) as exc:
        raise StructuralRegistrationError("admitted PCF manifest is malformed") from exc
    point_rows = [a for a in admitted_manifest.get("artifacts", []) if a.get("role") == "points_glb"]
    bundle_rows = [row for row in bundle_manifest.get("files", [])
                   if row.get("path") == "evaluation/static_world_point_cloud_manifest.json"]
    diagnostic_key = evidence["diagnostic_candidate_key"]
    diagnostic_hashes = diagnostic.get("input_sha256", {})
    selected_trial = diagnostic.get(diagnostic_key) if isinstance(diagnostic_key, str) else None
    selected_inputs = selected_trial.get("inputs", {}) if isinstance(selected_trial, Mapping) else {}
    selected_solution = selected_trial.get("solution", {}) if isinstance(selected_trial, Mapping) else {}
    candidate_inputs = candidate.get("inputs", {})
    if (not isinstance(diagnostic_key, str) or not diagnostic_key
            or selected_inputs.get("source_segments") != candidate_inputs.get("source_segments")
            or selected_inputs.get("source_points") != candidate_inputs.get("source_points")
            or selected_solution.get("matrix_col_major") != candidate.get("solution", {}).get("matrix_col_major")
            or diagnostic.get("source_manifest_alignment") != source_manifest.get("alignment")
            or diagnostic_hashes.get("source_manifest") != evidence["source_manifest"]["sha256"]
            or diagnostic_hashes.get("admitted_pcf_manifest") != sha256_file(admitted_manifest_path)
            or len(point_rows) != 1
            or diagnostic_hashes.get("admitted_pcf_glb") != point_rows[0].get("sha256")):
        _fail("carrier proof does not reproduce the selected diagnostic trial")
    if (bundle_manifest.get("schema") != "noesis.reference.room_scan_bundle_manifest.v1"
            or bundle_manifest.get("bundle_id") != admitted_manifest.get("source", {}).get("bundle_id")
            or len(bundle_rows) != 1
            or bundle_rows[0].get("sha256") != evidence["source_manifest"]["sha256"]
            or bundle_rows[0].get("size_bytes") != evidence["source_manifest"]["size_bytes"]
            or evidence["source_bundle_manifest"]["sha256"] != admitted_manifest.get("source", {}).get("bundle_manifest_sha256")):
        _fail("carrier source manifest is not in the admitted immutable bundle")
    if (admitted_manifest.get("prior_id") != evidence["target_world_revision"]
            or admitted_manifest.get("source", {}).get("capture_id") != capture_id
            or len(point_rows) != 1
            or point_rows[0].get("relative_path") != Path(points_path).name
            or point_rows[0].get("sha256") != sha256_file(Path(points_path))):
        _fail("carrier evidence is not bound to admitted PCF source/revision")
    if {key for key in diagnostic_hashes if key.startswith("carrier_raw_view_")} != {
        f"carrier_raw_view_{view}" for view in views
    }:
        _fail("carrier proof omits a view from the selected diagnostic")
    for view, ref in views.items():
        if diagnostic_hashes.get(f"carrier_raw_view_{view}") != ref.get("sha256"):
            _fail("carrier view bytes differ from the exact selected diagnostic")
    for ref in views.values():
        _file(ref, root, maximum=MAX_NPZ_BYTES)
    policy = _keys(evidence["selection_policy"], {"world_y_bounds_m"}, "carrier selection policy")
    height_bounds = _vector(policy["world_y_bounds_m"], 2, "carrier height interval")
    if not -20 <= height_bounds[0] < height_bounds[1] <= 20:
        _fail("carrier height interval is invalid")
    reference_y = _finite(evidence["reference_y_m"], "carrier reference height")
    walls = evidence["walls"]
    if not isinstance(walls, list) or not 1 <= len(walls) <= MAX_WALLS:
        _fail("carrier wall evidence is missing or unbounded")
    candidate_walls = {wall["segment_id"]: wall for wall in candidate["inputs"]["source_segments"]}
    if set(candidate_walls) != {row["segment_id"] for row in walls}:
        _fail("carrier proof does not cover exactly the candidate walls")
    admitted = read_pcf_points_glb(points_path) if offline else None
    base_segments = [row["base_segment"] for row in walls]
    if offline:
        verify_pcf_source_segments(points_path, base_segments)
    fit_by_id = {}
    for wall in walls:
        _keys(wall, {"segment_id", "base_segment", "rois", "fit", "diagnostic_support_key"},
              "carrier wall proof")
        segment_id = wall["segment_id"]
        if not isinstance(segment_id, str) or not segment_id:
            _fail("carrier wall ID is invalid")
        base = wall["base_segment"]
        _base_plane(base)
        rois = wall["rois"]
        if not isinstance(rois, list) or not 1 <= len(rois) <= MAX_ROIS_PER_WALL:
            _fail("carrier wall ROI set is missing or unbounded")
        if len({r["name"] for r in rois}) != len(rois):
            _fail("carrier wall ROI names are duplicated")
        fit = _keys(wall["fit"], {"normal_xyz", "offset_m", "point_count"}, "carrier plane fit")
        support = diagnostic.get("admitted_glb_roi_support", {}).get(wall["diagnostic_support_key"])
        if (not isinstance(support, Mapping)
                or fit != {"normal_xyz": support.get("normal_xyz"),
                              "offset_m": support.get("offset_m"),
                              "point_count": support.get("point_count")}
                or {r["name"]: r["selected_glb_sha256"] for r in rois} != {
                    name: data["admitted_glb_selected_xyz_float64_le_sha256"]
                    for name, data in support.get("roi_results", {}).items()
                }):
            _fail("carrier wall proof differs from exact selected GLB diagnostic")
        normal = _vector(fit["normal_xyz"], 3, "carrier plane normal")
        offset = _finite(fit["offset_m"], "carrier plane offset")
        count = fit["point_count"]
        if (not math.isclose(float(np.linalg.norm(normal)), 1, abs_tol=1e-6)
                or np.linalg.norm(normal[[0, 2]]) < 0.95
                or type(count) is not int or count < 200):
            _fail("carrier fit has invalid geometry or support")
        if offline:
            replay_normal, replay_offset, replay_count = _fit_admitted_plane(
                base, rois, views, alignment, admitted, reference_y, height_bounds, root)
            if (not np.allclose(replay_normal, normal, atol=1e-7, rtol=0)
                    or not math.isclose(replay_offset, offset, abs_tol=1e-7)
                    or replay_count != count):
                _fail("carrier wall plane does not replay from admitted PCF")
        candidate_wall = candidate_walls[segment_id]
        line = _line_at_height(base, normal, offset, reference_y, segment_id,
                               candidate_wall["source_refs"])
        _compare_replay(candidate_wall, line, path=f"carrier wall {segment_id}")
        fit_by_id[segment_id] = (normal, offset)
    point_evidence = evidence["point_evidence"]
    points = candidate["inputs"]["source_points"]
    if not isinstance(point_evidence, list) or len(point_evidence) != len(points) or len(points) > MAX_POINTS:
        _fail("carrier proof must cover every candidate source point")
    for frozen, point in zip(point_evidence, points):
        if frozen.get("point") != point or frozen.get("segment_id") not in fit_by_id:
            _fail("carrier point proof does not match candidate or wall")
        normal, offset = fit_by_id[frozen["segment_id"]]
        projection = frozen.get("projection")
        if (not isinstance(projection, Mapping)
                or projection.get("normal_xyz") != normal.tolist()
                or projection.get("offset_m") != offset
                or projection.get("reference_y_m") != reference_y
                or projection.get("source_manifest") != evidence["source_manifest"]
                or projection.get("view") != views.get(f"{frozen.get('view_index', -1):04d}")
                or point["source_wall_ids"] != [frozen["segment_id"]]):
            _fail("carrier point ray is not bound to its verified PCF wall")
        band = diagnostic.get("admitted_glb_opening_pixel_bands", {}).get(point["point_id"])
        if (not isinstance(band, Mapping) or band.get("source_view") != frozen["view_index"]
                or band.get("center_pixel_uv") != frozen["pixel_uv"]):
            _fail("carrier point is not the selected reviewed diagnostic pixel")
        verify_carrier_image_point(frozen, point, binding, root)


def _project(evidence: Mapping[str, Any], pixel: Any, root: Path) -> np.ndarray:
    projection = _keys(evidence.get("projection"),
                       {"view", "source_manifest", "normal_xyz", "offset_m", "reference_y_m"},
                       "carrier point projection")
    raw = _npz(projection["view"], root, image_only=True)
    alignment = _alignment(_json(projection["source_manifest"], root))
    rgb = raw["model_rgb"]
    from PIL import Image

    image_path = _file(evidence["image_file"], root, maximum=32 * 1024 * 1024)
    with Image.open(image_path) as image:
        if image.format != "PNG" or image.size != (rgb.shape[1], rgb.shape[0]):
            _fail("carrier review PNG differs from retained model raster")
        if not np.array_equal(np.asarray(image.convert("RGB")), rgb):
            _fail("carrier review PNG pixels differ from retained model RGB")
    uv = _vector(pixel, 2, "carrier pixel")
    if not (0 <= uv[0] < rgb.shape[1] and 0 <= uv[1] < rgb.shape[0]):
        _fail("carrier pixel lies outside bound model RGB")
    # Keep the retained float32 K inverse: the diagnostic ray used that exact
    # provider matrix before its source-world Sim(3) promoted math to float64.
    K = raw["intrinsics"]
    pose = raw["camera_pose"]
    if (not np.isfinite(K).all() or not np.isfinite(pose).all()
            or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], (0, 0, 1), atol=1e-5, rtol=0)
            or not np.allclose(pose[3], (0, 0, 0, 1), atol=1e-5, rtol=0)):
        _fail("carrier point has invalid retained K or pose")
    origin = (alignment @ pose)[:3, 3]
    ray = alignment[:3, :3] @ pose[:3, :3] @ np.linalg.inv(K) @ np.asarray((uv[0], uv[1], 1.0))
    normal = _vector(projection["normal_xyz"], 3, "carrier wall normal")
    offset = _finite(projection["offset_m"], "carrier wall offset")
    reference_y = _finite(projection["reference_y_m"], "carrier reference height")
    denominator = float(normal @ ray)
    if abs(denominator) < 0.10:
        _fail("carrier ray is too parallel to measured wall")
    parameter = -float(normal @ origin + offset) / denominator
    hit = origin + parameter * ray
    if parameter <= 0 or np.linalg.norm(hit - origin) > 30 or not np.isfinite(hit).all():
        _fail("carrier ray does not meet measured wall in front")
    hnorm2 = float(normal[[0, 2]] @ normal[[0, 2]])
    if hnorm2 < 0.95 ** 2:
        _fail("carrier source wall is not near vertical")
    delta_y = reference_y - hit[1]
    hit[[0, 2]] -= delta_y * normal[1] * normal[[0, 2]] / hnorm2
    return hit[[0, 2]]


def project_carrier_image_point(evidence: Mapping[str, Any], pixel: Any,
                                point: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Reproject a raster interval choice while retaining point identity/refs."""
    if evidence.get("kind") != KIND or evidence.get("point") != point:
        _fail("carrier point projection identity differs")
    return {**point, "xz": _project(evidence, pixel, root).tolist()}


def verify_carrier_image_point(evidence: Mapping[str, Any], point: Mapping[str, Any],
                               binding: Mapping[str, Any], root: Path) -> None:
    """Check a reviewed carrier pixel against its hashed K, pose and plane."""
    _keys(evidence, {"kind", "point", "view_index", "pixel_uv", "pixel_bounds",
                     "segment_id", "identity_review", "projection", "image_file", "image_evidence",
                     "target_world_revision",
                     "target_transform_sha256", "target_calibration_sha256"}, "carrier image point")
    if evidence["kind"] != KIND or evidence["point"] != point:
        _fail("carrier image identity differs from candidate")
    _binding(evidence, binding)
    if (not isinstance(evidence["identity_review"], str)
            or not evidence["identity_review"].strip()
            or type(evidence["view_index"]) is not int
            or evidence["view_index"] < 0):
        _fail("carrier point review identity is incomplete")
    centre = _vector(evidence["pixel_uv"], 2, "carrier review pixel")
    measured = _keys(evidence["image_evidence"],
                     {"pixel_uv", "image", "wall_plane", "surface_y_m"}, "carrier image evidence")
    image = _keys(measured["image"], {"width", "height"}, "carrier image size")
    plane = _keys(measured["wall_plane"], {"segment_id"}, "carrier image plane")
    if (measured["pixel_uv"] != evidence["pixel_uv"] or plane["segment_id"] != evidence["segment_id"]
            or measured["surface_y_m"] is not None
            or type(image["width"]) is not int or type(image["height"]) is not int):
        _fail("carrier image metadata differs from reviewed point")
    raw = _npz(evidence["projection"]["view"], root, image_only=True)
    if [image["width"], image["height"]] != [raw["model_rgb"].shape[1], raw["model_rgb"].shape[0]]:
        _fail("carrier review dimensions differ from retained raster")
    bounds = evidence["pixel_bounds"]
    if (not isinstance(bounds, list) or len(bounds) != 2
            or any(not isinstance(row, list) or len(row) != 2 for row in bounds)):
        _fail("carrier pixel bounds are malformed")
    low, high = (_vector(row, 2, "carrier pixel bounds") for row in bounds)
    if np.any(centre - low < 0.5) or np.any(high - centre < 0.5) or np.any(high - low > 64):
        _fail("carrier reviewed pixel interval is invalid")
    expected = _project(evidence, centre.tolist(), root)
    if not np.allclose(expected, _vector(point["xz"], 2, "candidate carrier point"), atol=1e-7, rtol=0):
        _fail("carrier point does not replay from reviewed raster ray")
    for corner in (low, high):
        _project(evidence, corner.tolist(), root)
