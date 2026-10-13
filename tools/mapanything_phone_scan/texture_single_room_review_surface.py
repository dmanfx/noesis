#!/usr/bin/env python3
"""Project exact prepared phone photographs onto one supported review mesh.

This is a CPU-only presentation export. Geometry, masks, depth, camera poses,
alignment, and runtime state are never changed. Photos are not blended or
inpainted. Invalid photo support becomes explicitly neutral material rather
than disappearing from the mesh. Shared-edge label regularization is applied
only between valid photo candidates on the same supported surface.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import inspect
import importlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy import sparse
from scipy.optimize import least_squares
from scipy.sparse.csgraph import connected_components
import trimesh
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from tools.mapanything_phone_scan.build_high_detail_pcf_mesh import image_match
from tools.mapanything_phone_scan.build_single_room_review_surface import (
    _rigid_transform,
    _validate_alignment,
)
from tools.mapanything_phone_scan.prepared_frame_identity import prepared_frame_identity


class TextureSurfaceError(ValueError):
    """A required source binding or a finite export bound did not hold."""


@dataclass
class PhotoView:
    index: int
    depth: np.ndarray
    mask: np.ndarray
    confidence: np.ndarray
    intrinsics: np.ndarray
    from_world: np.ndarray
    model_intrinsics: np.ndarray
    model_to_photo: np.ndarray | None
    photo_path: Path
    photo_size: tuple[int, int]
    quality: float
    provenance: dict[str, Any]
    surface_depth: np.ndarray | None = None


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha(path), "size_bytes": path.stat().st_size}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TextureSurfaceError(f"expected a JSON object: {path}")
    return value


def _bound_file(record: dict[str, Any], path: Path, label: str) -> None:
    if (not isinstance(record, dict) or Path(str(record.get("path", ""))).resolve() != path.resolve()
            or record.get("sha256") != _sha(path)):
        raise TextureSurfaceError(f"{label} path/digest binding mismatch")


def _rgb(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim != 3 or value.shape[2] != 3 or not np.isfinite(value).all():
        raise TextureSurfaceError("provider RGB is not a finite HxWx3 array")
    if np.issubdtype(value.dtype, np.floating) and value.max() <= 1.5:
        value = value * 255.0
    return np.rint(value).clip(0, 255).astype(np.uint8)


def _size_at_edge(size: tuple[int, int], edge: int) -> tuple[int, int]:
    scale = min(1.0, edge / max(size))
    return max(1, round(size[0] * scale)), max(1, round(size[1] * scale))


def _official_rgb_replayers():
    """Load only the CPU preprocessing functions, never model inference."""
    source_root = Path(__file__).resolve().parents[2] / "external" / "Depth-Anything-3" / "src"
    inserted = str(source_root) not in sys.path
    if inserted:
        sys.path.insert(0, str(source_root))
    try:
        da_module = importlib.import_module("depth_anything_3.utils.io.input_processor")
        ma_module = importlib.import_module("mapanything.utils.cropping")
    finally:
        if inserted:
            sys.path.remove(str(source_root))
    return da_module.InputProcessor(), ma_module.crop_resize_if_necessary, {
        "da3_input_processor": _record(Path(inspect.getfile(da_module))),
        "da3_processed_rgb_conversion_source": _record(source_root / "depth_anything_3" / "api.py"),
        "mapanything_crop_processor": _record(Path(inspect.getfile(ma_module))),
    }


def _compose_photo_map(
    source_size: tuple[int, int], da3_size: tuple[int, int], photo_size: tuple[int, int],
    da3_intrinsics: np.ndarray, processed_input_intrinsics: np.ndarray,
) -> np.ndarray:
    # The DA3 upper-bound *resize* rounds width and height independently to
    # patch multiples; it scales K rows directly. MA's official crop accounts
    # for pixel-centre offsets in its returned input K. These are image-space
    # transforms only: MA's predicted K remains the geometry authority.
    da3_to_model = processed_input_intrinsics @ np.linalg.inv(da3_intrinsics)
    da3_to_photo = np.diag([photo_size[0] / da3_size[0], photo_size[1] / da3_size[1], 1.0])
    model_to_photo = da3_to_photo @ np.linalg.inv(da3_to_model)
    if (not np.isfinite(model_to_photo).all()
            or not np.allclose(model_to_photo[2], [0, 0, 1], atol=1e-8)
            or min(source_size + da3_size + photo_size) <= 0):
        raise TextureSurfaceError("non-affine official image preprocessing transform")
    return model_to_photo[:2]


def verified_official_photo_maps(
    args: argparse.Namespace, frames: list[dict[str, Any]], consensus_manifest: dict[str, Any],
    model_raw: Path,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Use deterministic maps only after byte-exact replay of EVERY bound view.

    Unsupported preprocessing, missing provenance, or any non-identical saved
    RGB falls back to the existing measured SIFT mapping for the entire cohort.
    The diagnostic records the failed proof; no geometry K is substituted.
    """
    report: dict[str, Any] = {
        "status": "fallback_to_measured_image_match", "verified_view_count": 0,
        "required_view_count": len(frames), "geometry_intrinsics_changed": False,
        "source_pixel_pipeline": "prepared_photo_da3_upper_bound_resize_then_official_mapanything_crop",
        "verification_scope": "saved_model_rgb_identity_not_physical_camera_accuracy", "views": [],
    }
    maps: dict[int, np.ndarray] = {}
    try:
        da3_raw = Path(consensus_manifest["inputs"]["da3_raw"]).resolve()
        run_path, output_path = da3_raw.parent / "run_inputs.json", da3_raw.parent / "scan_outputs_manifest.json"
        run, output = _json(run_path), _json(output_path)
        prepared_path = args.scan_dir.resolve() / "prepared_frames_manifest.json"
        if (Path(str(run["scan"])).resolve() != args.scan_dir.resolve()
                or Path(str(run["prepared_manifest"])).resolve() != prepared_path
                or run["prepared_sha256"] != _sha(prepared_path)
                or run["view_count"] != len(frames) or output["view_count"] != len(frames)
                or len(output["frames"]) != len(frames)):
            raise TextureSurfaceError("DA3 replay source manifest binding mismatch")
        process_res = int(run["settings"]["process_res"])
        if not 14 <= process_res <= 4096:
            raise TextureSurfaceError("DA3 replay resolution exceeds CPU work bound")
        processor, crop, code = _official_rgb_replayers()
        report.update(official_sources=code, da3_run_inputs=_record(run_path), da3_output_manifest=_record(output_path),
                      process_res=process_res, process_res_method="upper_bound_resize")
        for index, frame in enumerate(frames):
            source = (args.scan_dir.resolve() / str(frame["frame"])).resolve()
            da_frame = output["frames"][index]
            if (not source.is_relative_to(args.scan_dir.resolve()) or _sha(source) != frame["sha256"]
                    or frame["index"] != index or frame["frame_id"] != prepared_frame_identity(index, frame["sha256"])
                    or da_frame["index"] != index or da_frame["source_frame"] != frame["frame"]
                    or da_frame["timestamp_s"] != frame["timestamp_s"]
                    or da_frame["fixed_camera_anchor"] is not False):
                raise TextureSurfaceError(f"DA3 replay view {index} source identity mismatch")
            source_size = (int(frame["width"]), int(frame["height"]))
            if math.prod(source_size) > args.max_input_pixels:
                raise TextureSurfaceError(f"DA3 replay view {index} exceeds source pixel bound")
            with Image.open(source) as source_image:
                if source_image.size != source_size:
                    raise TextureSurfaceError(f"DA3 replay view {index} source dimensions mismatch")
            da_path, ma_path = da3_raw / f"view_{index:04d}.npz", model_raw / f"view_{index:04d}.npz"
            with np.load(da_path, allow_pickle=False) as raw:
                da_rgb, da_k = _rgb(raw["model_rgb"]), np.asarray(raw["intrinsics"], dtype=np.float64)
            with np.load(ma_path, allow_pickle=False) as raw:
                ma_rgb = _rgb(raw["model_rgb"])
            tensor, shape, _, _ = processor._process_one(
                str(source), process_res=process_res, process_res_method="upper_bound_resize",
            )
            # Exact official api._add_processed_images conversion: numpy
            # float64 ImageNet denormalization, then truncation, not rounding.
            normalized = tensor.permute(1, 2, 0).cpu().numpy()
            replay_da = (np.clip(normalized * np.array([.229, .224, .225])
                                 + np.array([.485, .456, .406]), 0, 1) * 255).astype(np.uint8)
            ma_size = (ma_rgb.shape[1], ma_rgb.shape[0])
            cropped, processed_k = crop(da_rgb, ma_size, intrinsics=da_k)
            replay_ma = np.asarray(cropped)
            if (shape != da_rgb.shape[:2] or not np.array_equal(replay_da, da_rgb)
                    or not np.array_equal(replay_ma, ma_rgb)):
                raise TextureSurfaceError(f"official RGB replay is not byte-exact at view {index}")
            photo_size = _size_at_edge(source_size, args.texture_edge_px)
            affine = _compose_photo_map(source_size, (da_rgb.shape[1], da_rgb.shape[0]), photo_size,
                                        da_k, np.asarray(processed_k, dtype=np.float64))
            maps[index] = affine
            report["views"].append({"index": index, "da3_raw": _record(da_path),
                                    "source_size": list(source_size), "da3_size": [da_rgb.shape[1], da_rgb.shape[0]],
                                    "model_size": list(ma_size), "photo_size": list(photo_size),
                                    "prepared_to_da3_maximum_byte_error": 0, "da3_to_model_maximum_byte_error": 0,
                                    "model_to_photo_row_major": affine.tolist()})
            report["verified_view_count"] += 1
        report["status"] = "verified_byte_exact_official_preprocessing_all_views"
        return maps, report
    except (ImportError, OSError, KeyError, ValueError, TypeError) as exc:
        report["reason"] = f"{type(exc).__name__}: {exc}"
        return {}, report


def bound_texture_photo_rows(args, prepared):
    directory = getattr(args, "texture_photo_dir", None)
    if directory is None:
        return {}, None
    directory = directory.resolve()
    path = directory / "bound_texture_photos_manifest.json"
    manifest = _json(path)
    prepared_path = args.scan_dir.resolve() / "prepared_frames_manifest.json"
    _bound_file(manifest["prepared_manifest"], prepared_path, "native texture prepared manifest")
    source_path = (args.scan_dir.resolve() / prepared["video"]["path"]).resolve()
    source = manifest["source_video"]
    if (manifest.get("schema") != "noesis.mapanything.phone_scan.bound_texture_photos.v1"
            or manifest.get("status") != "verified_all_bound_views"
            or source.get("sha256") != prepared["video"]["sha256"]
            or source.get("sha256") != _sha(source_path)
            or source.get("size_bytes") != source_path.stat().st_size
            or source.get("rotation_degrees") != 0
            or manifest["extraction"].get("selection") != "single_decode_pass_exact_decoded_frame_index"
            or manifest["extraction"].get("prepared_pixels_used_as_output") is not False
            or manifest["geometry"] != {"source_pixel_geometry": "native_noautorotate_uniform_scale_only",
                "intrinsics_modified": False, "poses_modified": False, "prepared_manifest_modified": False}
            or len(manifest["views"]) != len(prepared["frames"])):
        raise TextureSurfaceError("native texture video, frame count, or pixel geometry binding mismatch")
    rows = {}
    for index, (row, frame) in enumerate(zip(manifest["views"], prepared["frames"], strict=True)):
        photo = row["output_photo"]
        photo_path = (directory / photo["path"]).resolve()
        original = row["prepared_photo"]
        if (row.get("index") != index or row.get("frame_id") != frame["frame_id"]
                or any(row.get(field) != frame.get(field) for field in
                       ("source_frame_index", "capture_time_ns", "timestamp_s", "timestamp_source"))
                or original != {"path": frame["frame"], "sha256": frame["sha256"],
                                "width": frame["width"], "height": frame["height"]}
                or row["image_mapping_verification"].get("status") != "verified_unique_same_decoded_frame_geometry"
                or not photo_path.is_relative_to(directory)
                or photo.get("sha256") != _sha(photo_path)
                or photo.get("encoding") != "png_lossless_rgb24"
                or not 1 <= int(photo["width"]) <= 4096 or not 1 <= int(photo["height"]) <= 4096
                or int(photo["width"]) * int(photo["height"]) > args.max_input_pixels
                or int(photo["width"]) * frame["height"] != int(photo["height"]) * frame["width"]):
            raise TextureSurfaceError(f"native texture view {index} exact-frame/photo binding mismatch")
        with Image.open(photo_path) as image:
            if image.size != (photo["width"], photo["height"]):
                raise TextureSurfaceError(f"native texture view {index} dimensions mismatch")
        rows[index] = {**row, "resolved_photo": photo_path}
    return rows, {"manifest": _record(path), "source_video": _record(source_path),
                  "verified_view_count": len(rows), "prepared_images_upsampled": False,
                  "camera_intrinsics_and_poses_changed": False}


def scaled_photo_map(affine, original_size, photo_size):
    """Compose only uniform, same-FOV image scaling in pixel-centre coordinates."""
    scale = np.asarray(photo_size, dtype=np.float64) / np.asarray(original_size, dtype=np.float64)
    if not np.isfinite(scale).all() or np.any(scale <= 0) or not np.isclose(scale[0], scale[1], atol=1e-8):
        raise TextureSurfaceError("native texture mapping is not one uniform same-FOV scale")
    result = np.asarray(affine, dtype=np.float64).copy()
    result *= scale[:, None]
    result[:, 2] += .5 * (scale - 1)
    return result


def load_bound_views(args: argparse.Namespace) -> tuple[list[PhotoView], dict[str, Any]]:
    scan, consensus, alignment = args.scan_dir.resolve(), args.consensus_dir.resolve(), args.alignment_dir.resolve()
    prepared_path = scan / "prepared_frames_manifest.json"
    prepared = _json(prepared_path)
    frames = prepared["frames"]
    texture_rows, texture_photo_report = bound_texture_photo_rows(args, prepared)
    manifest_path = consensus / "consensus_manifest.json"
    consensus_manifest = _json(manifest_path)
    if (consensus_manifest.get("phone_walk_only") is not True
            or consensus_manifest.get("static_camera_data_used") is not False
            or consensus_manifest.get("pose_carrier") != "da3"):
        raise TextureSurfaceError("texture input must preserve the explicit phone-only DA3-carried consensus")
    target_camera = _json(alignment / "alignment_report.json")["target"]["camera_id"]
    report, _, transform, raw_rows = _validate_alignment(
        consensus, alignment, expected_camera_id=target_camera,
    )
    if len(frames) != len(raw_rows) or not 2 <= len(frames) <= args.max_views:
        raise TextureSurfaceError("prepared, consensus, or configured view count mismatch")
    model_raw = Path(consensus_manifest["inputs"]["mapanything_raw"]).resolve()
    variant_path = model_raw.parent / "variant_manifest.json"
    suite_path = model_raw.parent.parent / "variant_suite_manifest.json"
    variant, suite = _json(variant_path), _json(suite_path)
    if suite.get("prepared_frames_manifest_sha256") != _sha(prepared_path):
        raise TextureSurfaceError("conditioned provider used a different prepared manifest")
    if (Path(str(suite.get("scan_dir"))).resolve() != scan
            or variant.get("inputs", {}).get("phone_view_count") != len(frames)
            or variant.get("inputs", {}).get("uses_static_reference") is not False
            or variant.get("world_output_policy") != "mapanything_depth_and_intrinsics_backprojected_through_da3_pose_carrier"):
        raise TextureSurfaceError("conditioned provider source/carrier identity mismatch")
    model_frames = variant["frames"]
    if len(model_frames) != len(frames):
        raise TextureSurfaceError("provider/prepared count mismatch")
    official_maps, official_report = verified_official_photo_maps(args, frames, consensus_manifest, model_raw)
    relative_model_origin = None
    relative_consensus_origin = None
    views: list[PhotoView] = []
    for index, (frame, model_frame, raw_row) in enumerate(zip(frames, model_frames, raw_rows, strict=True)):
        photo_path = (scan / str(frame["frame"])).resolve()
        if (frame.get("index") != index
                or frame.get("frame_id") != prepared_frame_identity(index, frame["sha256"])
                or not photo_path.is_relative_to(scan)
                or _sha(photo_path) != frame["sha256"]
                or model_frame.get("index") != index
                or model_frame.get("phone_frame_index") != index
                or model_frame.get("source_frame") != frame["frame"]
                or model_frame.get("timestamp_s") != frame.get("timestamp_s")
                or model_frame.get("fixed_camera_anchor") is not False):
            raise TextureSurfaceError(f"prepared/provider view {index} source identity mismatch")
        raw_path = Path(raw_row["path"])
        model_path = model_raw / f"view_{index:04d}.npz"
        with np.load(raw_path, allow_pickle=False) as raw:
            depth = np.asarray(raw["depth_z"], dtype=np.float32).copy()
            mask = np.asarray(raw["mask"], dtype=bool).copy()
            confidence = np.asarray(raw["confidence"], dtype=np.float32).copy()
            intrinsics = np.asarray(raw["intrinsics"], dtype=np.float64).copy()
            pose = _rigid_transform(raw["camera_pose"], f"consensus pose {index}")
        with np.load(model_path, allow_pickle=False) as model:
            model_rgb = _rgb(model["model_rgb"])
            model_k = np.asarray(model["intrinsics"], dtype=np.float64).copy()
            model_pose = _rigid_transform(model["camera_pose"], f"supplied provider pose {index}")
        if (depth.ndim != 2 or mask.shape != depth.shape or confidence.shape != depth.shape
                or intrinsics.shape != (3, 3) or model_k.shape != (3, 3)
                or pose.shape != (4, 4) or model_pose.shape != (4, 4)
                or not np.isfinite(intrinsics).all() or not np.isfinite(model_k).all()
                or not np.isfinite(pose).all() or not np.isfinite(model_pose).all()):
            raise TextureSurfaceError(f"malformed geometry for source view {index}")
        if relative_model_origin is None:
            relative_model_origin = np.linalg.inv(model_pose)
            relative_consensus_origin = np.linalg.inv(pose)
        if not np.allclose(relative_model_origin @ model_pose, relative_consensus_origin @ pose, atol=3e-4, rtol=3e-4):
            raise TextureSurfaceError(f"conditioned RGB source view {index} has a different supplied pose carrier")
        prepared_photo_path = photo_path
        with Image.open(prepared_photo_path) as source:
            if source.size != (frame["width"], frame["height"]) or math.prod(source.size) > args.max_input_pixels:
                raise TextureSurfaceError(f"photo size/bound mismatch for view {index}")
            original_photo_size = _size_at_edge(source.size, args.texture_edge_px)
        if texture_rows:
            photo_path = texture_rows[index]["resolved_photo"]
        with Image.open(photo_path) as source:
            photo_size = _size_at_edge(source.size, args.texture_edge_px)
            photo = np.asarray(source.convert("RGB").resize(photo_size, Image.Resampling.LANCZOS))
        quality = frame.get("quality") or {}
        blur = float(quality.get("blur_score", 0.0))
        luminance = float(quality.get("luminance_p50", 0.0))
        excluded = blur < args.minimum_texture_blur or luminance < args.minimum_luminance
        affine, image_report, reason = None, None, "eligible"
        if excluded:
            reason = "texture_only_sharpness_or_luminance_exclusion"
        elif official_maps:
            affine = official_maps[index]
            if texture_rows:
                affine = scaled_photo_map(affine, original_photo_size, photo_size)
            image_report = {"status": "verified_byte_exact_official_preprocessing", "feature_fit_used": False,
                            "model_to_photo_row_major": affine.tolist()}
        else:
            try:
                affine, image_report = image_match(model_rgb, photo)
                angle = math.degrees(math.atan2(affine[1, 0], affine[0, 0]))
                if abs(angle) > 0.5:
                    raise ValueError("same-frame image map requires an unexplained rotation")
            except ValueError as exc:
                affine = None
                reason = f"texture_only_image_pair_rejection: {exc}"
        provenance = {
            "view_index": index, "frame_id": frame["frame_id"],
            "source_frame_index": frame.get("source_frame_index"),
            "capture_time_ns": frame.get("capture_time_ns"), "timestamp_s": frame.get("timestamp_s"),
            "prepared_photo": _record(prepared_photo_path), "consensus_raw": raw_row,
            "conditioned_provider_raw": _record(model_path), "photo_size": list(photo_size),
            "provider_intrinsics": model_k.tolist(), "common_intrinsics": intrinsics.tolist(),
            "camera_to_world": (transform @ pose).tolist(),
            "model_to_photo_row_major": affine.tolist() if affine is not None else None,
            "image_match": image_report, "texture_status": reason,
            "blur_score": blur, "luminance_p50": luminance,
            "geometry_preserved_when_texture_excluded": True,
        }
        if texture_rows:
            provenance["texture_photo"] = _record(photo_path)
            provenance["native_photo_binding"] = {key: value for key, value in texture_rows[index].items()
                                                  if key != "resolved_photo"}
        views.append(PhotoView(
            index, depth, mask, confidence, intrinsics, np.linalg.inv(transform @ pose),
            model_k, affine, photo_path, photo_size,
            math.sqrt(min(1.0, max(0.01, blur / 400.0))), provenance,
        ))
        print(f"verified photo view {index + 1}/{len(frames)}: {reason}", flush=True)
    return views, {
        "prepared_manifest": _record(prepared_path), "consensus_manifest": _record(manifest_path),
        "conditioned_variant_manifest": _record(variant_path), "conditioned_suite_manifest": _record(suite_path),
        "alignment_report": _record(alignment / "alignment_report.json"),
        "alignment_transform": _record(alignment / "phone_ma_to_noesis_world.json"),
        "target": report["target"], "world_from_consensus": transform.tolist(),
        "photo_mapping_verification": official_report,
        **({"bound_native_texture_photos": texture_photo_report} if texture_photo_report else {}),
    }


def project_photo(points: np.ndarray, view: PhotoView) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera = np.asarray(points) @ view.from_world[:3, :3].T + view.from_world[:3, 3]
    z = camera[:, 2]
    safe = np.where(z > 1e-6, z, 1.0)
    model_uv = camera @ view.model_intrinsics.T
    model_uv = model_uv[:, :2] / safe[:, None]
    uv = (model_uv @ view.model_to_photo[:, :2].T + view.model_to_photo[:, 2]
          if view.model_to_photo is not None else np.full((len(points), 2), np.nan))
    return camera, z, uv


def depth_visible(points: np.ndarray, view: PhotoView, *, minimum_quality: float = 0.20) -> tuple[np.ndarray, np.ndarray]:
    camera, z, photo_uv = project_photo(points, view)
    if view.surface_depth is not None:
        # An explicit presentation-only alternative: test the exact camera's
        # z-buffer of this same reconstructed mesh, not raw neural depth. This
        # does not upgrade inferred geometry to measured support.
        height, width = view.surface_depth.shape
        scaled = photo_uv * [width / view.photo_size[0], height / view.photo_size[1]]
        inside = (np.isfinite(scaled).all(1) & (z > .05) & (z < 12.)
                  & (photo_uv[:, 0] >= .5) & (photo_uv[:, 0] <= view.photo_size[0] - 1.5)
                  & (photo_uv[:, 1] >= .5) & (photo_uv[:, 1] <= view.photo_size[1] - 1.5)
                  & (scaled[:, 0] >= 0) & (scaled[:, 0] < width)
                  & (scaled[:, 1] >= 0) & (scaled[:, 1] < height))
        rows = np.flatnonzero(inside)
        visible = np.zeros(len(points), dtype=bool)
        pixels = np.floor(scaled[rows]).astype(np.int32)
        depth = view.surface_depth[pixels[:, 1], pixels[:, 0]]
        visible[rows] = np.isfinite(depth) & (np.abs(z[rows] - depth) <= np.maximum(.03, .015 * depth))
        return visible, visible.astype(np.float32)
    h, w = view.depth.shape
    common = camera @ view.intrinsics.T
    common = common[:, :2] / np.where(z > 1e-6, z, 1.0)[:, None]
    finite = np.isfinite(common).all(1) & np.isfinite(photo_uv).all(1)
    inside = finite & (z > 0.05) & (z < 12.0)
    inside &= ((common[:, 0] >= 1) & (common[:, 0] < w - 1)
               & (common[:, 1] >= 1) & (common[:, 1] < h - 1))
    inside &= ((photo_uv[:, 0] >= 0.5) & (photo_uv[:, 0] <= view.photo_size[0] - 1.5)
               & (photo_uv[:, 1] >= 0.5) & (photo_uv[:, 1] <= view.photo_size[1] - 1.5))
    rows = np.flatnonzero(inside)
    visible = np.zeros(len(points), dtype=bool)
    score = np.zeros(len(points), dtype=np.float32)
    if len(rows):
        pix = np.rint(common[rows]).astype(np.int32)
        depth = view.depth[pix[:, 1], pix[:, 0]]
        confidence = view.confidence[pix[:, 1], pix[:, 0]]
        tolerance = np.maximum(0.05, 0.03 * depth)
        valid = (view.mask[pix[:, 1], pix[:, 0]] & np.isfinite(depth) & np.isfinite(confidence)
                 & (confidence >= minimum_quality) & (depth > 0.05))
        # Both front and behind-depth disagreement are rejected, so an occluded
        # photo cannot colour the mesh through a foreground surface.
        visible[rows] = valid & (np.abs(z[rows] - depth) <= tolerance)
        score[rows] = np.where(visible[rows], np.clip(confidence, 0, 1), 0)
    return visible, score


def face_candidates(vertices: np.ndarray, faces: np.ndarray, views: list[PhotoView], *, top_k: int = 6) -> tuple[np.ndarray, np.ndarray]:
    labels = np.full((len(faces), top_k), -1, dtype=np.int16)
    scores = np.zeros(labels.shape, dtype=np.float32)
    corners = vertices[faces]
    centers = corners.mean(1)
    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    length = np.linalg.norm(normal, axis=1)
    normal /= np.maximum(length[:, None], 1e-12)
    for view in views:
        if view.model_to_photo is None:
            continue
        vertex_valid, vertex_quality = depth_visible(vertices, view)
        center_valid, center_quality = depth_visible(centers, view)
        valid = center_valid & vertex_valid[faces].all(1) & (length > 1e-12)
        camera_center = np.linalg.inv(view.from_world)[:3, 3]
        toward = camera_center - centers
        distance = np.linalg.norm(toward, axis=1)
        cosine = np.abs(np.einsum("ij,ij->i", normal, toward)) / np.maximum(distance, 1e-6)
        valid &= cosine >= 0.17
        candidate = (np.minimum(center_quality, vertex_quality[faces].min(1))
                     * view.quality * np.sqrt(cosine) / np.maximum(distance, 0.4))
        candidate[~valid] = 0
        worst = scores.argmin(1)
        rows = np.flatnonzero(candidate > scores[np.arange(len(faces)), worst])
        labels[rows, worst[rows]] = view.index
        scores[rows, worst[rows]] = candidate[rows]
    order = np.argsort(-scores, axis=1, kind="stable")
    return np.take_along_axis(labels, order, axis=1), np.take_along_axis(scores, order, axis=1)


def face_adjacency(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    neighbors = np.full((len(faces), 3), -1, dtype=np.int32)
    weights = np.zeros((len(faces), 3), dtype=np.float32)
    if not len(faces):
        return neighbors, weights
    edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    ordered = edges[order]
    starts = np.flatnonzero(np.r_[True, np.any(ordered[1:] != ordered[:-1], axis=1)])
    sizes = np.diff(np.r_[starts, len(order)])
    shared = starts[sizes == 2]
    left, right = order[shared] // 3, order[shared + 1] // 3
    left_slot, right_slot = order[shared] % 3, order[shared + 1] % 3
    normal = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-12)
    cosine = np.abs(np.einsum("ij,ij->i", normal[left], normal[right]))
    smooth = np.clip((cosine - 0.70) / 0.30, 0, 1)
    neighbors[left, left_slot], neighbors[right, right_slot] = right, left
    weights[left, left_slot], weights[right, right_slot] = smooth, smooth
    return neighbors, weights


def _component_labels(labels: np.ndarray, neighbors: np.ndarray) -> tuple[int, np.ndarray]:
    rows, slots = np.nonzero(neighbors >= 0)
    cols = neighbors[rows, slots]
    same = (labels[rows] >= 0) & (labels[rows] == labels[cols])
    graph = sparse.csr_matrix((np.ones(np.count_nonzero(same)), (rows[same], cols[same])), shape=(len(labels), len(labels)))
    return connected_components(graph, directed=False)


def coherent_labels(candidates: np.ndarray, scores: np.ndarray, neighbors: np.ndarray, weights: np.ndarray, *, smoothness: float = 0.8, iterations: int = 8, minimum_patch_faces: int = 24) -> tuple[np.ndarray, dict[str, Any]]:
    """Bounded valid-candidate Potts label optimization, never RGB averaging."""
    labels = candidates[:, 0].copy()
    before = labels.copy()
    best = scores.max(1)
    unary = -np.log(np.maximum(scores / np.maximum(best[:, None], 1e-12), 1e-8))
    unary[candidates < 0] = np.inf
    # A degree-three triangle-dual graph has a greedy four-colour bound.
    # Each update group is independent, so the actual objective cannot increase
    # through a synchronous checkerboard flip of neighboring labels.
    colours = np.zeros(len(labels), dtype=np.uint8)
    for face in range(len(labels)):
        previous = neighbors[face][(neighbors[face] >= 0) & (neighbors[face] < face)]
        used = set(colours[previous].tolist())
        colours[face] = next(colour for colour in range(4) if colour not in used)
    passes = 0
    for _ in range(iterations):
        changed = 0
        for colour in range(4):
            rows = np.flatnonzero((colours == colour) & (best > 0))
            energy = unary[rows].copy()
            for slot in range(3):
                neighbor = neighbors[rows, slot]
                valid = neighbor >= 0
                neighbor_labels = labels[np.maximum(neighbor, 0)]
                valid &= neighbor_labels >= 0
                energy += (smoothness * weights[rows, slot, None] * valid[:, None]
                           * (candidates[rows] != neighbor_labels[:, None]))
            chosen = candidates[rows, energy.argmin(1)]
            changed += np.count_nonzero(chosen != labels[rows])
            labels[rows] = chosen
        passes += 1
        if not changed:
            break
    # Coalesce tiny islands only into a neighboring label that is a directly
    # validated photo candidate on EVERY face in that island.
    component_count, component = _component_labels(labels, neighbors)
    sizes = np.bincount(component, minlength=component_count)
    relabeled = 0
    component_order = np.argsort(component, kind="stable")
    component_end = np.cumsum(sizes)
    component_start = component_end - sizes
    for region in np.flatnonzero(sizes < minimum_patch_faces):
        rows = component_order[component_start[region]:component_end[region]]
        if not len(rows) or labels[rows[0]] < 0:
            continue
        border = neighbors[rows][weights[rows] >= 0.5]
        border = border[border >= 0]
        alternatives = np.unique(labels[border])
        alternatives = alternatives[(alternatives >= 0) & (alternatives != labels[rows[0]])]
        options = []
        for alternative in alternatives:
            valid = candidates[rows] == alternative
            if valid.any(1).all():
                cost = np.mean(unary[rows, valid.argmax(1)])
                options.append((float(cost), int(alternative)))
        if options:
            cost, alternative = min(options)
            current = np.mean(unary[rows, (candidates[rows] == labels[rows, None]).argmax(1)])
            if cost <= current + smoothness:
                labels[rows] = alternative
                relabeled += len(rows)
    if not np.all((labels < 0) | (candidates == labels[:, None]).any(1)):
        raise TextureSurfaceError("adjacency refinement selected an invalid photograph")
    def stats(values):
        count, component = _component_labels(values, neighbors)
        counts = np.bincount(component, minlength=count)
        ids = np.unique(component[values >= 0])
        return {"photo_patch_count": len(ids), "tiny_photo_patch_count": int(np.count_nonzero(counts[ids] < minimum_patch_faces))}
    return labels, {"before": stats(before), "after": stats(labels), "iterations": passes,
                    "changed_face_count": int(np.count_nonzero(before != labels)), "tiny_patch_relabeled_faces": relabeled,
                    "minimum_patch_faces": minimum_patch_faces, "smoothness": smoothness,
                    "candidate_validity_preserved": True, "colour_blending": False}


def _unlit(tree: dict[str, Any]) -> None:
    for material in tree.get("materials", []):
        material.setdefault("extensions", {})["KHR_materials_unlit"] = {}
    tree["extensionsUsed"] = sorted(set(tree.get("extensionsUsed", [])) | {"KHR_materials_unlit"})


def _linear_rgb(rgb: np.ndarray) -> np.ndarray:
    value = np.asarray(rgb, dtype=np.float64) / 255.0
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def _encoded_rgb(linear: np.ndarray) -> np.ndarray:
    value = np.asarray(linear)
    encoded = np.where(value <= 0.0031308, 12.92 * value, 1.055 * np.maximum(value, 0) ** (1 / 2.4) - 0.055)
    return np.rint(encoded * 255).clip(0, 255).astype(np.uint8)


def training_component_anchors(
    indices: list[int], training_edges: list[tuple[int, int, np.ndarray]],
    preferred_anchor: int, face_counts: dict[int, int],
) -> tuple[list[int], dict[str, Any]]:
    """Preselect at most three measured references without validation data.

    Only training pair medians and assigned phone-surface support enter this
    function. It has no reserved RGB/masks, static-camera data, or geometry
    access. The unconstrained training gauge is used for reference selection,
    never directly applied as a gain. All actual fits retain their bounds.
    """
    indices = sorted(indices)
    if preferred_anchor not in indices or not training_edges:
        raise TextureSurfaceError("component anchor preselection requires a measured connected graph")
    active = [index for index in indices if index != preferred_anchor]
    slot = {index: row for row, index in enumerate(active)}
    # Equivalent to the incidence least-squares training gauge, without an
    # edge_count × view_count dense matrix. Fix the preferred reference to zero.
    normal = np.zeros((len(active), len(active)), dtype=np.float64)
    rhs = np.zeros((len(active), 3), dtype=np.float64)
    ordered = sorted(training_edges, key=lambda edge: (edge[0], edge[1]))
    for first, second, target in ordered:
        target = np.asarray(target, dtype=np.float64)
        if first not in indices or second not in indices or target.shape != (3,) or not np.isfinite(target).all():
            raise TextureSurfaceError("malformed component training edge")
        if first != preferred_anchor:
            row = slot[first]
            normal[row, row] += 1
            rhs[row] += target
        if second != preferred_anchor:
            row = slot[second]
            normal[row, row] += 1
            rhs[row] -= target
        if first != preferred_anchor and second != preferred_anchor:
            normal[slot[first], slot[second]] -= 1
            normal[slot[second], slot[first]] -= 1
    if not len(active) or np.linalg.matrix_rank(normal) != len(active):
        raise TextureSurfaceError("component training graph is not connected")
    solution = np.linalg.solve(normal, rhs)
    relative = {preferred_anchor: np.zeros(3), **{index: solution[slot[index]] for index in active}}
    gauges = np.asarray([relative[index] for index in indices])
    centre = (gauges.max(0) + gauges.min(0)) / 2 - (math.log(.67) + math.log(1.5)) / 2
    central = min(indices, key=lambda index: (float(np.max(np.abs(relative[index] - centre))),
                                              -face_counts.get(index, 0), index))
    support_threshold = float(np.percentile([face_counts.get(index, 0) for index in indices], 75))
    high_support = [index for index in indices if face_counts.get(index, 0) >= support_threshold]
    dimmest = max(high_support, key=lambda index: (float(np.mean(relative[index])), face_counts.get(index, 0), -index))
    selected = [preferred_anchor]
    for candidate in (central, dimmest):
        if candidate not in selected:
            selected.append(candidate)
    cycle = np.asarray([relative[first] - relative[second] - target for first, second, target in ordered])
    return selected, {
        "policy": "preferred_then_central_training_gauge_then_dimmest_training_gauge_among_high_support_views",
        "anchor_preselection_uses_reserved_samples": False, "maximum_anchor_count": 3,
        "preselected_anchor_views": selected, "central_training_gauge_anchor": central,
        "dimmest_high_support_training_gauge_anchor": dimmest,
        "high_support_minimum_assigned_faces": support_threshold,
        "training_gain_dynamic_range": np.exp(gauges.max(0) - gauges.min(0)).tolist(),
        "training_cycle_residual_p80_log_rgb": float(np.percentile(np.abs(cycle), 80)),
        "unbounded_training_gains_applied": False,
    }


def _fit_component_gains(
    indices: list[int], pairs: list[tuple[int, int, np.ndarray, np.ndarray]], reference: int,
    sampled: dict[int, np.ndarray], photos: dict[int, Image.Image],
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    active = [index for index in sorted(indices) if index != reference]
    slot = {index: row for row, index in enumerate(active)}
    def residual(parameters):
        values = parameters.reshape(-1, 3)
        return np.concatenate([(values[slot[first]] if first != reference else np.zeros(3))
                               - (values[slot[second]] if second != reference else np.zeros(3)) - target
                               for first, second, target, _ in pairs])
    solved = least_squares(residual, np.zeros(3 * len(active)), bounds=(math.log(.67), math.log(1.5)),
                           loss="soft_l1", f_scale=.05, max_nfev=100)
    fitted = {index: np.ones(3) for index in indices}
    for index in active:
        fitted[index] = np.exp(solved.x.reshape(-1, 3)[slot[index]])
    before, after = [], []
    for first, second, _, holdout in pairs:
        before.extend(np.mean(np.abs(sampled[first][holdout] - sampled[second][holdout]), axis=1))
        after.extend(np.mean(np.abs(sampled[first][holdout] * fitted[first] - sampled[second][holdout] * fitted[second]), axis=1))
    before, after = np.asarray(before), np.asarray(after)
    before_median, after_median = float(np.median(before)), float(np.median(after))
    new_clipping = {}
    for index in active:
        linear = _linear_rgb(np.asarray(photos[index]))
        new_clipping[index] = float(np.mean(np.any((linear < .98) & (linear * fitted[index] > 1), axis=-1)))
    residual_pass = (solved.success and after_median <= 0.06 and after_median <= 0.90 * before_median
                     and float(np.percentile(after, 80)) <= float(np.percentile(before, 80)))
    clipping_pass = all(value <= .01 for value in new_clipping.values())
    passed = bool(residual_pass and clipping_pass)
    return fitted, {
        "anchor_view": reference, "status": "accepted_measured_gains" if passed else "rejected_holdout_or_clipping_original_photos_retained",
        "solver_success": bool(solved.success), "residual_checks_passed": bool(residual_pass), "clipping_checks_passed": clipping_pass,
        "heldout_linear_rgb_error_before_p50": before_median, "heldout_linear_rgb_error_after_p50": after_median,
        "heldout_linear_rgb_error_before_p80": float(np.percentile(before, 80)),
        "heldout_linear_rgb_error_after_p80": float(np.percentile(after, 80)),
        "candidate_gains": {str(index): fitted[index].tolist() for index in sorted(indices)},
        "newly_clipped_pixel_fraction": {str(index): value for index, value in new_clipping.items()},
    }


def measured_photo_gains(
    points: np.ndarray, candidates: np.ndarray, views: dict[int, PhotoView],
    photos: dict[int, Image.Image], *, anchor: int, minimum_pair_samples: int = 64,
    face_counts: dict[int, int] | None = None,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Fit bounded gains from flat same-surface overlap; reserve 20% to reject.

    There are no hand-authored offsets, RGB averaging, image warps, or additional
    geometry constraints. Each measured component has its own reference and
    independent acceptance test; disconnected components are NOT globally
    calibrated. Weak, singleton, or rejected components stay unchanged.
    """
    report: dict[str, Any] = {"status": "insufficient_measured_overlap", "anchor_view": anchor,
                              "gain_limits": [0.67, 1.50], "reserved_sample_fraction": 0.20,
                              "reserved_sample_scope": "same_scene_same_bound_images_not_independent_accuracy",
                              "maximum_linear_rgb_holdout_error": 0.06,
                              "minimum_holdout_improvement_fraction": 0.10, "pairs": [],
                              "components": [], "component_count": len(photos),
                              "accepted_component_count": 0, "fitted_view_count": 0,
                              "attempted_fitted_view_count": 0, "accepted_adjusted_view_count": 0,
                              "disconnected_components_globally_calibrated": False,
                              "normalization_scope": "independently_anchored_measured_components",
                              "maximum_newly_clipped_pixel_fraction": 0.01,
                              "maximum_training_log_ratio_mad": 0.15,
                              "minimum_training_samples_per_pair": minimum_pair_samples,
                              "minimum_reserved_samples_per_pair": 16,
                              "maximum_anchors_per_component": 3, "attempted_anchor_fit_count": 0,
                              "attempted_view_anchor_fit_count": 0,
                              "anchor_preselection_uses_reserved_samples": False,
                              "anchor_acceptance_uses_reserved_samples": True,
                              "absolute_colour_calibration_established": False}
    if anchor not in photos:
        raise TextureSurfaceError("photometric anchor must be an exact bound selected photograph")
    sampled, valid = {}, {}
    for index, photo in photos.items():
        _, _, uv = project_photo(points, views[index])
        rgb = np.asarray(photo)
        visible, _ = depth_visible(points, views[index])
        inside = visible & (candidates == index).any(1)
        rows = np.flatnonzero(inside)
        if len(rows):
            pixel = np.rint(uv[rows]).astype(np.int32)
            neighborhood_inside = ((pixel[:, 0] >= 1) & (pixel[:, 0] < rgb.shape[1] - 1)
                                   & (pixel[:, 1] >= 1) & (pixel[:, 1] < rgb.shape[0] - 1))
            rows = rows[neighborhood_inside]
        values = np.zeros((len(points), 3), dtype=np.float64)
        good = np.zeros(len(points), dtype=bool)
        if len(rows):
            x, y = uv[rows, 0].astype(np.float32), uv[rows, 1].astype(np.float32)
            colour = cv2.remap(rgb, x[:, None], y[:, None], cv2.INTER_LINEAR)[:, 0]
            pixel = np.rint(uv[rows]).astype(np.int32)
            nearby = np.stack([rgb[pixel[:, 1] + dy, pixel[:, 0] + dx]
                               for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))])
            gradient = nearby.astype(np.int16).max(0) - nearby.astype(np.int16).min(0)
            linear = _linear_rgb(colour)
            accepted = (np.max(gradient, axis=1) <= 16) & np.all((linear >= 0.025) & (linear <= 0.80), axis=1)
            values[rows] = linear
            good[rows] = accepted
        sampled[index], valid[index] = values, good
    pairs = []
    indices = sorted(photos)
    reserved = np.arange(len(points)) % 5 == 0
    for position, first in enumerate(indices):
        for second in indices[position + 1:]:
            common = valid[first] & valid[second]
            training, holdout = common & ~reserved, common & reserved
            if np.count_nonzero(training) < minimum_pair_samples or np.count_nonzero(holdout) < 16:
                continue
            ratio = np.log(sampled[second][training] / sampled[first][training])
            target = np.median(ratio, axis=0)
            mad = np.median(np.abs(ratio - target), axis=0)
            if np.max(mad) > 0.15:
                continue
            pairs.append((first, second, target, holdout))
            report["pairs"].append({"first_view": first, "second_view": second,
                                    "training_samples": int(np.count_nonzero(training)),
                                    "holdout_samples": int(np.count_nonzero(holdout)),
                                    "training_log_ratio_median": target.tolist(), "training_log_ratio_mad": mad.tolist()})
    neighbors = {index: set() for index in indices}
    for first, second, _, _ in pairs:
        neighbors[first].add(second)
        neighbors[second].add(first)
    remaining = set(indices)
    components = []
    while remaining:
        component, frontier = set(), {min(remaining)}
        while frontier:
            index = frontier.pop()
            if index not in component:
                component.add(index)
                frontier.update(neighbors[index] - component)
        components.append(component)
        remaining -= component
    report["component_count"] = len(components)
    report["measured_component_count"] = sum(len(component) > 1 for component in components)
    support = face_counts or {}
    gains = {index: np.ones(3) for index in indices}
    attempted = {}
    for component_id, component in enumerate(components):
        reference = anchor if anchor in component else max(component, key=lambda index: (support.get(index, 0), -index))
        active = sorted(component - {reference})
        component_pairs = [pair for pair in pairs if pair[0] in component and pair[1] in component]
        item: dict[str, Any] = {
            "component_id": component_id, "view_indices": sorted(component), "anchor_view": reference,
            "anchor_policy": "supplied_anchor_preserved" if reference == anchor else "largest_assigned_face_support_then_lowest_index",
            "assigned_face_support": {str(index): int(support.get(index, 0)) for index in sorted(component)},
            "pair_count": len(component_pairs), "attempted_fitted_view_count": len(active),
            "training_pair_sample_count": sum(pair["training_samples"] for pair in report["pairs"] if pair["first_view"] in component),
            "heldout_pair_sample_count": sum(pair["holdout_samples"] for pair in report["pairs"] if pair["first_view"] in component),
            "fitted_view_count": 0, "accepted_adjusted_view_count": 0,
            "preferred_anchor_view": reference, "selected_anchor_view": None, "anchor_trials": [],
            "status": "singleton_or_weak_measured_overlap_original_photos_retained",
        }
        report["components"].append(item)
        if anchor in component:
            # Retained for older readers; this describes graph connectivity,
            # not whether other independently anchored components were fitted.
            report["disconnected_views"] = sorted(set(indices) - component)
            report["disconnected_views_definition"] = "not_connected_to_supplied_anchor_component"
        if not active or not component_pairs:
            continue
        fitted, trial = _fit_component_gains(sorted(component), component_pairs, reference, sampled, photos)
        item["anchor_trials"].append(trial)
        item.update(trial)
        passed = trial["status"] == "accepted_measured_gains"
        if not passed and anchor not in component:
            # Strip reserved masks completely before preselection: only the
            # qualified training edge medians enter the reference-choice rule.
            choices, preselection = training_component_anchors(
                sorted(component), [(first, second, target) for first, second, target, _ in component_pairs],
                reference, support,
            )
            item["anchor_preselection"] = preselection
            for candidate in choices[1:]:
                candidate_fitted, candidate_trial = _fit_component_gains(sorted(component), component_pairs, candidate, sampled, photos)
                item["anchor_trials"].append(candidate_trial)
                if candidate_trial["status"] == "accepted_measured_gains":
                    fitted, passed = candidate_fitted, True
                    item.update(candidate_trial)
                    item["anchor_policy"] = "first_accepted_training_only_bounded_preselection"
                    break
        item["attempted_anchor_fit_count"] = len(item["anchor_trials"])
        report["attempted_anchor_fit_count"] += len(item["anchor_trials"])
        report["attempted_view_anchor_fit_count"] += len(active) * len(item["anchor_trials"])
        attempted.update(fitted)
        report["attempted_fitted_view_count"] += len(active)
        if passed:
            gains.update(fitted)
            adjusted_count = sum(not np.array_equal(fitted[index], np.ones(3)) for index in component)
            item.update(fitted_view_count=len(active), accepted_adjusted_view_count=adjusted_count,
                        selected_anchor_view=item["anchor_view"])
            report["accepted_component_count"] += 1
            report["fitted_view_count"] += len(active)
            report["accepted_adjusted_view_count"] += adjusted_count
        if anchor in component:
            # These compatibility summary residuals are explicitly restricted
            # to the supplied anchor's component, never pooled across gauges.
            report.update({key: item[key] for key in (
                "solver_success", "heldout_linear_rgb_error_before_p50", "heldout_linear_rgb_error_after_p50",
                "heldout_linear_rgb_error_before_p80", "heldout_linear_rgb_error_after_p80")})
            report["summary_residual_scope"] = "supplied_anchor_component_only"
    report["candidate_gains"] = {str(index): gain.tolist() for index, gain in attempted.items()}
    report["applied_gains"] = {str(index): gain.tolist() for index, gain in gains.items()}
    report["status"] = ("accepted_measured_gains" if report["accepted_component_count"] else
                        "rejected_holdout_or_clipping_original_photos_retained" if report["attempted_fitted_view_count"] else
                        "insufficient_measured_overlap")
    return gains, report


def export_partition(mesh: trimesh.Trimesh, labels: np.ndarray, views: dict[int, PhotoView], scene: trimesh.Scene, name: str, photos: dict[int, Image.Image], *, geometry_support: np.ndarray | None = None, crop_photos: bool = False) -> dict[str, Any]:
    areas = mesh.area_faces
    neutral_count = int(np.count_nonzero(labels < 0))
    embedded_texels, crops = 0, {}
    for label in np.unique(labels):
        rows = np.flatnonzero(labels == label)
        vertices, inverse = np.unique(mesh.faces[rows].reshape(-1), return_inverse=True)
        part = trimesh.Trimesh(vertices=mesh.vertices[vertices], faces=inverse.reshape(-1, 3),
                               vertex_normals=mesh.vertex_normals[vertices], process=False)
        if label >= 0:
            view = views[int(label)]
            _, _, photo_uv = project_photo(part.vertices, view)
            if (not np.isfinite(photo_uv).all() or np.any(photo_uv < 0.5)
                    or np.any(photo_uv[:, 0] > view.photo_size[0] - 1.5)
                    or np.any(photo_uv[:, 1] > view.photo_size[1] - 1.5)):
                raise TextureSurfaceError("texture projection escaped the accepted photo; UV clipping is prohibited")
            photo = photos[int(label)]
            crop = [0, 0, *view.photo_size]
            if crop_photos:
                minimum = np.floor(photo_uv.min(0)).astype(int) - 8
                maximum = np.ceil(photo_uv.max(0)).astype(int) + 9
                crop = [max(0, int(minimum[0])), max(0, int(minimum[1])),
                        min(view.photo_size[0], int(maximum[0])), min(view.photo_size[1], int(maximum[1]))]
                photo = photo.crop(crop)
            uv = (photo_uv - crop[:2]) / np.asarray(photo.size)
            uv[:, 1] = 1.0 - uv[:, 1]
            embedded_texels += math.prod(photo.size)
            crops[str(int(label))] = {"source_photo_size": list(view.photo_size), "crop_xyxy": crop,
                                      "embedded_size": list(photo.size), "resampling": False}
            material = PBRMaterial(name=f"exact_phone_view_{label}", baseColorTexture=photo,
                                   baseColorFactor=[255, 255, 255, 255],
                                   metallicFactor=0.0, roughnessFactor=1.0, doubleSided=True)
            part.visual = TextureVisuals(uv=uv, material=material)
            support_class = ("photo_visible_surface_with_explicit_geometry_support" if geometry_support is not None
                             else "photo_visible_observed_surface")
        else:
            part.visual.vertex_colors = np.tile([136, 142, 146, 255], (len(vertices), 1))
            support_class = ("geometry_photo_unavailable_neutral_with_explicit_support" if geometry_support is not None
                             else "supported_geometry_photo_unavailable_neutral")
        part.metadata.update(review_only=True, source_geometry=name, source_view_index=int(label), support_class=support_class)
        if geometry_support is not None:
            part.metadata["geometry_support_class_counts"] = {
                str(index): int(np.count_nonzero(geometry_support[rows] == index)) for index in range(3)}
        node = f"{name}_photo_{label}" if label >= 0 else f"{name}_neutral_photo_unavailable"
        scene.add_geometry(part, geom_name=node, node_name=node)
    return {"triangle_count": len(labels), "photo_triangle_count": len(labels) - neutral_count,
            "neutral_triangle_count": neutral_count, "neutral_area_m2": float(areas[labels < 0].sum()),
            "total_area_m2": float(areas.sum()), "source_view_count": len(np.unique(labels[labels >= 0])),
            "embedded_photo_texel_count": embedded_texels, "photo_region_crops": crops}


def completed_geometry_support(manifest: dict[str, Any], name: str, count: int) -> np.ndarray | None:
    completion = manifest.get("completion")
    if completion is None:
        return None
    record = completion.get("geometry_support", {})
    path = Path(str(record.get("path", ""))).resolve()
    _bound_file(record, path, "completed geometry support")
    if completion.get("geometry_name") != name:
        raise TextureSurfaceError("completed geometry name differs from its support sidecar")
    with np.load(path, allow_pickle=False) as data:
        classes = data["face_support_class"].copy()
        if (str(data["geometry_name"]) != name or classes.shape != (count,)
                or not np.issubdtype(classes.dtype, np.integer) or np.any(classes > 2) or np.any(classes < 0)
                or not np.array_equal(data["face_local_index"], np.arange(count))
                or data["class_names"].tolist() != ["multi_view_supported", "poisson_inferred", "boundary_cap_inferred"]):
            raise TextureSurfaceError("completed geometry support is not exact face provenance")
    return classes.astype(np.uint8)


def reusable_photo_assignments(args, mesh_manifest, views):
    parent_dir = getattr(args, "reuse_assignments_from", None)
    if parent_dir is None:
        return None, None
    parent_path = parent_dir / "texture_manifest.json"
    parent = _json(parent_path)
    _bound_file(parent["source"]["mesh"], args.mesh_dir / "surface_mesh.glb", "reused mesh")
    _bound_file(parent["source"]["mesh_manifest"], args.mesh_dir / "surface_mesh_manifest.json", "reused mesh manifest")
    if (parent.get("status") != "review_only" or parent.get("accepted_for_canonical_use") is not False
            or parent["source"]["mesh"]["sha256"] != mesh_manifest["output"]["sha256"]
            or parent["source"]["mesh_manifest"]["sha256"] != _sha(args.mesh_dir / "surface_mesh_manifest.json")):
        raise TextureSurfaceError("reused photo assignment has a different geometry source")
    for field, path in (("prepared_manifest", args.scan_dir / "prepared_frames_manifest.json"),
                        ("consensus_manifest", args.consensus_dir / "consensus_manifest.json"),
                        ("alignment_report", args.alignment_dir / "alignment_report.json"),
                        ("alignment_transform", args.alignment_dir / "phone_ma_to_noesis_world.json")):
        _bound_file(parent["source"][field], path, "reused " + field)
    if len(parent["views"]) != len(views):
        raise TextureSurfaceError("reused assignment source view count changed")
    for old, new in zip(parent["views"], views, strict=True):
        current = new.provenance
        for field in ("frame_id", "source_frame_index", "prepared_photo", "conditioned_provider_raw", "consensus_raw",
                      "provider_intrinsics", "common_intrinsics", "camera_to_world"):
            if old[field] != current[field]:
                raise TextureSurfaceError("reused assignment camera/frame/model evidence changed")
    path = parent_dir / "face_photo_assignment.npz"
    _bound_file(parent["output"]["assignments"], path, "reused assignments")
    records = {}
    with np.load(path, allow_pickle=False) as data:
        names = data["geometry_names"].tolist()
        if len(set(names)) != len(names):
            raise TextureSurfaceError("reused assignment geometry names are not unique")
        for index, name in enumerate(names):
            rows = np.flatnonzero(data["face_geometry_index"] == index)
            if not np.array_equal(data["face_local_index"][rows], np.arange(len(rows))):
                raise TextureSurfaceError("reused face assignment ordering is not exact")
            records[str(name)] = (data["face_source_view"][rows].copy(), parent["geometries"][str(name)]["coherence"])
    return records, {"manifest": _record(parent_path), "assignments": _record(path),
                     "new_photo_corner_and_centroid_visibility_rechecked": True}


def validate_assigned_photos(mesh, labels, views):
    if (labels.shape != (len(mesh.faces),) or not np.issubdtype(labels.dtype, np.integer)
            or np.any(labels < -1)):
        raise TextureSurfaceError("reused assignment triangle count changed")
    view_map = {view.index: view for view in views}
    for label in np.unique(labels[labels >= 0]):
        if int(label) not in view_map or view_map[int(label)].model_to_photo is None:
            raise TextureSurfaceError("reused photograph is no longer eligible")
        view = view_map[int(label)]
        rows = np.flatnonzero(labels == label)
        camera_center = np.linalg.inv(view.from_world)[:3, 3]
        for start in range(0, len(rows), 8192):
            corners = mesh.vertices[mesh.faces[rows[start:start + 8192]]]
            centers = corners.mean(1)
            valid, _ = depth_visible(corners.reshape(-1, 3), view)
            center_valid, _ = depth_visible(centers, view)
            normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
            toward = camera_center - centers
            cosine = np.abs(np.einsum("ij,ij->i", normal, toward)) / np.maximum(
                np.linalg.norm(normal, axis=1) * np.linalg.norm(toward, axis=1), 1e-12)
            if not np.all(valid.reshape(-1, 3).all(1) & center_valid & (cosine >= .17)):
                raise TextureSurfaceError("reused source does not pass new photo visibility; no stale assignment is exported")


def cached_face_candidates(mesh, views, *, top_k, cache_dir, geometry_index, mesh_sha256):
    if cache_dir is None:
        return face_candidates(mesh.vertices, mesh.faces, views, top_k=top_k)
    algorithm = "\n".join(inspect.getsource(function) for function in (face_candidates, depth_visible, project_photo))
    signature = hashlib.sha256(json.dumps({"mesh_sha256": mesh_sha256, "top_k": top_k,
        "views": [view.provenance for view in views], "algorithm_sha256": hashlib.sha256(algorithm.encode()).hexdigest()},
        sort_keys=True, allow_nan=False).encode()).hexdigest()
    array_path = cache_dir / f"geometry_{geometry_index:04d}.npz"
    record_path = array_path.with_suffix(".json")
    if array_path.exists() != record_path.exists():
        raise TextureSurfaceError("incomplete candidate checkpoint; use a new cache directory")
    if array_path.exists():
        record = _json(record_path)
        if (record.get("schema") != "noesis.pcf.photo_candidate_checkpoint.v1"
                or record.get("signature") != signature or record.get("array_sha256") != _sha(array_path)):
            raise TextureSurfaceError("candidate checkpoint source/algorithm/hash mismatch")
        with np.load(array_path, allow_pickle=False) as data:
            candidates, scores = data["candidates"].copy(), data["scores"].copy()
        if (candidates.shape != (len(mesh.faces), top_k) or scores.shape != candidates.shape
                or not np.issubdtype(candidates.dtype, np.integer) or not np.isfinite(scores).all()
                or np.any(candidates < -1) or np.any(scores < 0)
                or not np.isin(candidates[candidates >= 0], [view.index for view in views]).all()
                or np.any((candidates >= 0) != (scores > 0))):
            raise TextureSurfaceError("candidate checkpoint contains invalid bounded arrays")
        print(f"reused exact candidate checkpoint {geometry_index}", flush=True)
        return candidates, scores
    candidates, scores = face_candidates(mesh.vertices, mesh.faces, views, top_k=top_k)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(array_path, candidates=candidates, scores=scores)
    record_path.write_text(json.dumps({"schema": "noesis.pcf.photo_candidate_checkpoint.v1", "signature": signature,
                                      "array_sha256": _sha(array_path), "face_count": len(mesh.faces)}, indent=2) + "\n")
    print(f"saved exact candidate checkpoint {geometry_index}", flush=True)
    return candidates, scores


def mesh_work_bounds(args) -> tuple[int, int]:
    """Opt-in larger offline meshes, without changing default work budgets."""
    faces = getattr(args, "max_faces", 3_000_000)
    vertices = getattr(args, "max_vertices", 2_000_000)
    if (isinstance(faces, bool) or not isinstance(faces, int) or not 1 <= faces <= 5_000_000
            or isinstance(vertices, bool) or not isinstance(vertices, int) or not 1 <= vertices <= 3_000_000):
        raise TextureSurfaceError("mesh work bounds exceed finite supported limits")
    return faces, vertices


def build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    if args.output_dir.exists():
        raise TextureSurfaceError("use a new texture output directory")
    mesh_work_bounds(args)
    if not (2 <= args.top_k <= 12 and 1 <= args.label_iterations <= 20
            and 256 <= args.texture_edge_px <= 4096 and 0 <= args.smoothness <= 10
            and 1 <= args.minimum_patch_faces <= 10_000
            and 0 <= args.minimum_texture_blur <= 1_000_000 and 0 <= args.minimum_luminance <= 255
            and 2 <= args.max_views <= 512
            and 1 <= args.max_input_pixels <= 64_000_000
            and 1 <= args.max_texture_texels <= 512_000_000 and 128 <= args.photometric_sample_limit <= 100_000):
        raise TextureSurfaceError("texture parameters exceed finite supported bounds")
    physical_smoothness = getattr(args, "smoothness_m", None)
    if physical_smoothness is not None and (not math.isfinite(physical_smoothness) or not 0 <= physical_smoothness <= 10):
        raise TextureSurfaceError("physical texture smoothness is outside finite bounds")
    mesh_dir = args.mesh_dir.resolve()
    mesh_path, mesh_manifest_path = mesh_dir / "surface_mesh.glb", mesh_dir / "surface_mesh_manifest.json"
    mesh_manifest = _json(mesh_manifest_path)
    if mesh_manifest.get("status") != "review_only" or mesh_manifest.get("accepted_for_canonical_use") is not False:
        raise TextureSurfaceError("mesh is not explicitly review-only")
    if mesh_manifest.get("coordinate_frame") != "backend_world_m_stream_points":
        raise TextureSurfaceError("mesh coordinate frame is not explicit backend meters")
    if mesh_manifest["output"].get("sha256") != _sha(mesh_path):
        raise TextureSurfaceError("mesh digest mismatch")
    input_manifest_path = Path(mesh_manifest["source"]["surfels_manifest"]).resolve()
    if mesh_manifest["source"].get("surfels_manifest_sha256") != _sha(input_manifest_path):
        raise TextureSurfaceError("mesh input manifest digest mismatch")
    input_manifest = _json(input_manifest_path)
    _bound_file(input_manifest["source"]["consensus_manifest"], args.consensus_dir / "consensus_manifest.json", "mesh consensus")
    _bound_file(input_manifest["source"]["alignment_report"], args.alignment_dir / "alignment_report.json", "mesh alignment")
    _bound_file(input_manifest["source"]["alignment_transform"], args.alignment_dir / "phone_ma_to_noesis_world.json", "mesh alignment transform")
    views, provenance = load_bound_views(args)
    surface_visibility_dir = getattr(args, "surface_visibility_dir", None)
    if surface_visibility_dir is not None:
        from tools.mapanything_phone_scan.build_mesh_texture_visibility import read_bound_visibility
        provenance["mesh_surface_texture_visibility"] = read_bound_visibility(
            surface_visibility_dir, views, mesh_manifest["output"]["sha256"])
    reused, reuse_report = reusable_photo_assignments(args, mesh_manifest, views)
    if reuse_report is not None:
        provenance["photo_assignment_reuse"] = reuse_report
    scene_in = trimesh.load(mesh_path, force="scene", process=False)
    parts = []
    support_parts = []
    total_faces = 0
    total_vertices = 0
    for geometry_index, node in enumerate(scene_in.graph.nodes_geometry):
        transform, geometry_name = scene_in.graph[node]
        if not np.allclose(transform, np.eye(4), atol=1e-9):
            raise TextureSurfaceError("source mesh has an unbound node transform")
        mesh = scene_in.geometry[geometry_name]
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            raise TextureSurfaceError("source artifact includes unsupported non-surface geometry")
        total_faces += len(mesh.faces)
        total_vertices += len(mesh.vertices)
        if total_faces > args.max_faces or total_vertices > args.max_vertices:
            raise TextureSurfaceError("surface exceeds bounded face/vertex work")
        if reused is None:
            candidates, scores = cached_face_candidates(mesh, views, top_k=args.top_k,
                cache_dir=getattr(args, "candidate_cache_dir", None), geometry_index=geometry_index,
                mesh_sha256=mesh_manifest["output"]["sha256"])
            neighbors, weights = face_adjacency(mesh.vertices, mesh.faces)
            if physical_smoothness is None:
                labels, coherence = coherent_labels(candidates, scores, neighbors, weights,
                                                    smoothness=args.smoothness, iterations=args.label_iterations,
                                                    minimum_patch_faces=args.minimum_patch_faces)
            else:
                from tools.mapanything_phone_scan.coherent_surface_photo_labels import coherent_labels as physical_labels
                labels, coherence = physical_labels(mesh.vertices, mesh.faces, candidates, scores, neighbors, weights,
                    smoothness_m=physical_smoothness, iterations=args.label_iterations,
                    minimum_patch_area_m2=getattr(args, "minimum_patch_area_m2", .01),
                    max_faces=args.max_faces, max_vertices=args.max_vertices)
        else:
            if str(geometry_name) not in reused:
                raise TextureSurfaceError("reused assignment omits an input geometry")
            labels, original_coherence = reused.pop(str(geometry_name))
            validate_assigned_photos(mesh, labels, views)
            coherence = {**original_coherence, "reused_exact_labels_new_photos_revalidated": True}
        support_parts.append(completed_geometry_support(mesh_manifest, str(geometry_name), len(mesh.faces)))
        stride = max(1, math.ceil(len(mesh.faces) / args.photometric_sample_limit))
        sampled_faces = mesh.faces[::stride]
        if reused is None:
            sampled_candidates = candidates[::stride]
        else:
            ids, inverse = np.unique(sampled_faces.reshape(-1), return_inverse=True)
            sampled_candidates, _ = face_candidates(mesh.vertices[ids], inverse.reshape(-1, 3), views, top_k=args.top_k)
        parts.append((str(geometry_name), mesh, labels, coherence, mesh.vertices[sampled_faces].mean(1), sampled_candidates))
    if reused:
        raise TextureSurfaceError("reused assignment contains non-source geometries")
    if total_faces != mesh_manifest["output"]["triangle_count"]:
        raise TextureSurfaceError("loaded mesh triangle count differs from its manifest")
    used = sorted({int(index) for _, _, labels, _, _, _ in parts for index in np.unique(labels[labels >= 0])})
    view_map = {view.index: view for view in views}
    texels = sum(math.prod(view_map[index].photo_size) for index in used)
    if texels > args.max_texture_texels:
        raise TextureSurfaceError(f"selected photos exceed texel budget: {texels} > {args.max_texture_texels}")
    photos = {}
    for index in used:
        with Image.open(view_map[index].photo_path) as image:
            photos[index] = image.convert("RGB").resize(view_map[index].photo_size, Image.Resampling.LANCZOS)
    photometric_report = {"status": "not_requested_original_photos_retained"}
    if args.normalize_measured_exposure and used:
        face_counts = {index: sum(int(np.count_nonzero(labels == index)) for _, _, labels, _, _, _ in parts) for index in used}
        anchor = max(used, key=lambda index: face_counts[index])
        points = np.concatenate([part[4] for part in parts])
        candidates = np.concatenate([part[5] for part in parts])
        if len(points) > args.photometric_sample_limit:
            stride = math.ceil(len(points) / args.photometric_sample_limit)
            points, candidates = points[::stride], candidates[::stride]
        gains, photometric_report = measured_photo_gains(points, candidates, view_map, photos, anchor=anchor,
                                                       face_counts=face_counts)
        if photometric_report["status"] == "accepted_measured_gains":
            photos = {index: (Image.fromarray(_encoded_rgb(_linear_rgb(np.asarray(photo)) * gains[index]))
                              if not np.array_equal(gains[index], np.ones(3)) else photo)
                      for index, photo in photos.items()}
    scene = trimesh.Scene(metadata={"review_only": True, "coordinate_frame": "backend_world_m_stream_points"})
    metrics = {}
    labels_all, geometry_ids, local_faces = [], [], []
    for index, (name, mesh, labels, coherence, _, _) in enumerate(parts):
        metrics[name] = {**export_partition(mesh, labels, view_map, scene, name, photos,
                                          geometry_support=support_parts[index], crop_photos=getattr(args, "crop_selected_photo_regions", False)), "coherence": coherence}
        labels_all.append(labels)
        geometry_ids.append(np.full(len(labels), index, dtype=np.uint16))
        local_faces.append(np.arange(len(labels), dtype=np.uint32))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    glb_path = args.output_dir / "textured_surface_mesh.glb"
    glb_path.write_bytes(trimesh.exchange.gltf.export_glb(scene, include_normals=True, tree_postprocessor=_unlit))
    assignment_path = args.output_dir / "face_photo_assignment.npz"
    support_arrays = ({"face_geometry_support_class": np.concatenate(support_parts),
                       "geometry_support_class_names": np.asarray(["multi_view_supported", "poisson_inferred", "boundary_cap_inferred"])}
                      if any(item is not None for item in support_parts) else {})
    np.savez_compressed(assignment_path, face_source_view=np.concatenate(labels_all),
                        face_geometry_index=np.concatenate(geometry_ids), face_local_index=np.concatenate(local_faces),
                        geometry_names=np.asarray([part[0] for part in parts]), **support_arrays)
    # The actual exported direct consumer must retain every source triangle.
    reloaded = trimesh.load(glb_path, force="scene", process=False)
    exported_faces = sum(len(item.faces) for item in reloaded.geometry.values() if isinstance(item, trimesh.Trimesh))
    if exported_faces != total_faces:
        raise TextureSurfaceError("textured GLB consumer lost source geometry")
    report = {
        "schema": "noesis.pcf.single_room_photo_texture.v1", "status": "review_only",
        "accepted_for_canonical_use": False, "coordinate_frame": "backend_world_m_stream_points",
        "source": {"mesh": _record(mesh_path), "mesh_manifest": _record(mesh_manifest_path),
                   "mesh_input_manifest": _record(input_manifest_path), **provenance},
        "method": "exact_prepared_photo_projection_with_depth_visible_valid_candidate_adjacency_labels",
        "policy": {"geometry_changed": False, "uv_clipping": False, "inpainting": False,
                   "cross_view_colour_blending": False, "photo_colour_policy": "measured_bounded_linear_rgb_gain_if_holdout_passes_else_original_rgb",
                   "illumination": "KHR_materials_unlit", "triangle_visibility": "all_corners_and_centroid",
                   "texture_visibility_evidence": ("reconstructed_surface_zbuffer" if surface_visibility_dir else "provider_depth_agreement"),
                   "depth_tolerance_m": ("max(0.03,0.015*mesh_zbuffer_depth)" if surface_visibility_dir else "max(0.05,0.03*source_depth)"),
                   "geometry_support_gates_changed": False, "occluded_photo_rejected": True,
                   "unassigned_photo_region_crop": getattr(args, "crop_selected_photo_regions", False),
                   "unsupported_texture": "neutral_vertex_colour_existing_geometry_preserved"},
        "geometry_completion": mesh_manifest.get("completion"),
        "parameters": {key: value for key, value in vars(args).items() if not isinstance(value, Path)},
        "views": [view.provenance for view in views], "geometries": metrics, "photometric_normalization": photometric_report,
        "output": {"glb": _record(glb_path), "assignments": _record(assignment_path),
                   "source_triangle_count": total_faces, "exported_triangle_count": exported_faces,
                   "photo_triangle_count": sum(row["photo_triangle_count"] for row in metrics.values()),
                   "neutral_triangle_count": sum(row["neutral_triangle_count"] for row in metrics.values()),
                   "texture_material_count": len(used), "photo_texel_count": texels,
                   "embedded_photo_texel_count": sum(row["embedded_photo_texel_count"] for row in metrics.values())},
        "elapsed_s": time.monotonic() - started,
        "limitations": ["model_predicted_geometry_and_phone_pose_uncertainty_remain",
                        "no_independent_physical_accuracy_claim",
                        ("bounded_measured_gains_do_not_establish_absolute_colour_accuracy"
                         if photometric_report["status"] == "accepted_measured_gains"
                         else "original_exposure_and_white_balance_are_retained"),
                        "photo_exclusions_affect_presentation_only_not_supported_geometry"],
    }
    (args.output_dir / "texture_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["output"], indent=2), flush=True)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("mesh-dir", "scan-dir", "consensus-dir", "alignment-dir", "output-dir"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--texture-edge-px", type=int, default=1920)
    parser.add_argument("--texture-photo-dir", type=Path,
                        help="Use independently verified higher-resolution exact native-video frames, never upsample prepared photos.")
    parser.add_argument("--surface-visibility-dir", type=Path,
                        help="Opt in to bound same-mesh z-buffer photo visibility; presentation only, not provider depth support.")
    parser.add_argument("--minimum-texture-blur", type=float, default=100.0)
    parser.add_argument("--minimum-luminance", type=float, default=20.0)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--smoothness", type=float, default=0.8)
    parser.add_argument("--smoothness-m", type=float,
                        help="Opt in to physical area/shared-edge region optimization; legacy face-count policy otherwise.")
    parser.add_argument("--minimum-patch-area-m2", type=float, default=.01)
    parser.add_argument("--reuse-assignments-from", type=Path,
                        help="Reuse exact source/geometry-bound labels, revalidating new photo corner/centroid visibility.")
    parser.add_argument("--candidate-cache-dir", type=Path,
                        help="Optional source/algorithm/hash-bound candidate checkpoints before label refinement.")
    parser.add_argument("--label-iterations", type=int, default=8)
    parser.add_argument("--minimum-patch-faces", type=int, default=24)
    parser.add_argument("--normalize-measured-exposure", action="store_true")
    parser.add_argument("--crop-selected-photo-regions", action="store_true",
                        help="Embed only exact selected-photo UV bounding rectangles plus8-pixel border, without resampling.")
    parser.add_argument("--photometric-sample-limit", type=int, default=20_000)
    parser.add_argument("--max-views", type=int, default=256)
    parser.add_argument("--max-faces", type=int, default=750_000)
    parser.add_argument("--max-vertices", type=int, default=800_000)
    parser.add_argument("--max-input-pixels", type=int, default=16_000_000)
    parser.add_argument("--max-texture-texels", type=int, default=256_000_000)
    return parser


def main() -> int:
    cv2.setNumThreads(2)
    build(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
