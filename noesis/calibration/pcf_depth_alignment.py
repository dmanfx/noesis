"""Offline, review-only MapAnything optical-Z alignment to an admitted PCF room.

Static PCF points supply scale at visible, stationary background pixels. They
never supply the depth of a person occupying those pixels. An occupied sample
can use the resulting scalar only when its MapAnything value was measured on
the person's own current-frame support at the DAv2 anchor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

CONTRACT = "noesis.pcf_depth_alignment.review.v1"
EXCLUSIONS_CONTRACT = "noesis.pcf_depth_alignment.person_exclusions.v1"
OCCUPIED_CONTRACT = "noesis.pcf_depth_alignment.occupied_person_samples.v1"
OPTICAL_Z = "camera_optical_z_m"
PERSON_ANCHORS = {"lower_body_band", "pose_ankle_support", "torso_core", "pose_torso_support"}
DEPTH_BANDS_M = ((0.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 6.0), (6.0, 8.0), (8.0, 12.0))


class AlignmentError(ValueError):
    """A static or occupied source is unbound or unsafe for this review."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AlignmentError(f"expected object: {path}")
    return value


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _verified_file(root: Path, record: Mapping[str, Any]) -> Path:
    relative = Path(str(record.get("path", record.get("relative_path", record.get("manifest_path", "")))))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise AlignmentError("artifact path is not relative")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file() or path.is_symlink():
        raise AlignmentError(f"artifact missing or symlinked: {path}")
    expected_size = record.get("size_bytes", record.get("manifest_size_bytes"))
    expected_digest = record.get("sha256", record.get("manifest_sha256"))
    if path.stat().st_size != expected_size or _digest(path) != expected_digest:
        raise AlignmentError(f"artifact digest/size mismatch: {path}")
    return path


def _artifact(records: Sequence[Mapping[str, Any]], name: str) -> Mapping[str, Any]:
    matches = [r for r in records if r.get("path", r.get("relative_path")) == name]
    if len(matches) != 1:
        raise AlignmentError(f"expected exactly one artifact record for {name}")
    return matches[0]


def _binding(catalog: Mapping[str, Any], static: Mapping[str, Any], camera: str) -> tuple[dict[str, Any], str]:
    matches = [b for b in catalog.get("camera_bindings", []) if b.get("camera_id") == camera]
    if len(matches) != 1 or matches[0].get("mode") != "shadow":
        raise AlignmentError("camera has no unique admitted shadow PCF binding")
    selected = matches[0]
    bound = selected.get("frame_binding")
    static_bound = static.get("camera_frame_binding")
    if not isinstance(bound, dict) or not isinstance(static_bound, dict):
        raise AlignmentError("static camera/world/transform binding disagrees with PCF")
    same_binding = (
        bound.get("source_camera_calibration_sha256") == static_bound.get("camera_calibration_sha256")
        and bound.get("source_world_alignment_sha256") == static_bound.get("world_alignment_sha256")
        and bound.get("source_frame") == static_bound.get("calibration_frame")
        and bound.get("target_frame") == static_bound.get("world_frame")
        and bound.get("target_from_source_col_major") == static_bound.get("target_from_calibration_col_major")
        and bound.get("target_from_source_sha256") == static_bound.get("transform_sha256")
        and bound.get("target_revision_id") == static_bound.get("target_revision_id")
        and bound.get("source_floor_plane", {}).get("normal") == static_bound.get("calibration_floor_plane", {}).get("normal")
        and bound.get("source_floor_plane", {}).get("offset_m") == static_bound.get("calibration_floor_plane", {}).get("offset_m")
        and bound.get("target_floor_plane", {}).get("normal") == static_bound.get("world_floor_plane", {}).get("normal")
        and bound.get("target_floor_plane", {}).get("offset_m") == static_bound.get("world_floor_plane", {}).get("offset_m")
        and static_bound.get("scene_prior_id") == selected.get("prior_id")
    )
    if not same_binding:
        raise AlignmentError("static camera/world/transform binding disagrees with PCF")
    for key in ("source_camera_calibration_sha256", "source_world_alignment_sha256",
                "target_from_source_sha256"):
        value = bound.get(key)
        if not (isinstance(value, str) and len(value) == 64
                and all(c in "0123456789abcdef" for c in value)):
            raise AlignmentError(f"PCF {key} digest missing")
    if static.get("camera_id") != camera or static.get("world_frame") != "backend_world_m":
        raise AlignmentError("static camera or world frame mismatch")
    if static.get("world_frame_revision") != selected.get("prior_id"):
        raise AlignmentError("static world revision does not match selected PCF")
    if static.get("status") != "complete" or static.get("reference_kind") != "paired_static":
        raise AlignmentError("static reference is incomplete or wrong kind")
    source = static.get("source", {})
    if not isinstance(source, dict) or len(source.get("frames", [])) < 4:
        raise AlignmentError("static reference needs at least four source frames")
    for frame in source["frames"]:
        if frame.get("depth_semantics") != "model_metric_range_on_captured_calibrated_rays_converted_to_z":
            raise AlignmentError("static MapAnything depth is not camera optical Z")
    return bound, str(selected["prior_id"])


def _exclusion_masks(
    exclusions: Mapping[str, Any], static: Mapping[str, Any],
    static_manifest_sha256: str, shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    if (exclusions.get("contract") != EXCLUSIONS_CONTRACT
            or exclusions.get("camera_id") != static.get("camera_id")
            or exclusions.get("static_manifest_sha256") != static_manifest_sha256):
        raise AlignmentError("person exclusions are not bound to this static reference")
    rows = exclusions.get("frames")
    expected = {f.get("raw_artifact"): f for f in static["source"]["frames"]}
    if (not isinstance(rows, list) or len(rows) != len(expected)
            or not all(isinstance(r, dict) for r in rows)
            or {r.get("raw_artifact") for r in rows} != set(expected)):
        raise AlignmentError("person exclusions must cover every exact static frame")
    h, w = shape
    masks: dict[str, np.ndarray] = {}
    for row in rows:
        name = row["raw_artifact"]
        source = expected[name]
        if (row.get("source_frame_index") != source.get("source_frame_index")
                or row.get("raw_frame_sha256") != source.get("raw_frame_sha256")):
            raise AlignmentError(f"person exclusion frame identity mismatch: {name}")
        boxes = row.get("rectangles_xyxy")
        if not isinstance(boxes, list):
            raise AlignmentError("person exclusion rectangles missing")
        mask = np.zeros(shape, dtype=bool)
        for box in boxes:
            if (not isinstance(box, list) or len(box) != 4
                    or any(type(v) is not int for v in box)):
                raise AlignmentError("person exclusion rectangle must be integer xyxy")
            x0, y0, x1, y1 = box
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                raise AlignmentError("person exclusion rectangle is out of bounds")
            mask[y0:y1, x0:x1] = True
        masks[name] = mask
    return masks


def project_pcf_optical_z(points_world: np.ndarray, camera_to_world: np.ndarray, K: np.ndarray,
                          shape: tuple[int, int]) -> np.ndarray:
    """Nearest visible point per model pixel, in calibrated optical-Z meters."""
    points = np.asarray(points_world, dtype=np.float64)
    transform = np.asarray(camera_to_world, dtype=np.float64)
    intrinsics = np.asarray(K, dtype=np.float64)
    h, w = shape
    if (points.ndim != 2 or points.shape[1] != 3 or transform.shape != (4, 4)
            or intrinsics.shape != (3, 3) or not np.all(np.isfinite(points))
            or not np.all(np.isfinite(transform)) or not np.all(np.isfinite(intrinsics))
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-5)
            or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0
            or h <= 0 or w <= 0):
        raise AlignmentError("invalid PCF camera projection inputs")
    camera = (points - transform[:3, 3]) @ transform[:3, :3]
    z = camera[:, 2]
    valid = np.isfinite(z) & (z > 0.3)
    u = np.rint(intrinsics[0, 0] * camera[valid, 0] / z[valid] + intrinsics[0, 2]).astype(int)
    v = np.rint(intrinsics[1, 1] * camera[valid, 1] / z[valid] + intrinsics[1, 2]).astype(int)
    depth = z[valid]
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (depth < 30.0)
    flat = np.full(h * w, np.inf, dtype=np.float32)
    np.minimum.at(flat, v[inside] * w + u[inside], depth[inside])
    return flat.reshape(shape)


def _residuals(reference: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    error = np.abs(np.asarray(reference, dtype=float) - np.asarray(predicted, dtype=float))
    return {"count": int(error.size),
            "median_abs_error_m": float(np.median(error)) if error.size else None,
            "p90_abs_error_m": float(np.percentile(error, 90)) if error.size else None}


def _heldout_row(reference: np.ndarray, model: np.ndarray, scale: float) -> dict[str, Any]:
    return {"count": int(reference.size),
            "pcf_optical_z_range_m": [float(np.min(reference)), float(np.max(reference))] if reference.size else None,
            "median_pcf_to_model_ratio": float(np.median(reference / model)) if reference.size else None,
            "before": _residuals(reference, model), "after": _residuals(reference, model * scale)}


def review_static_scale(catalog_path: Path, static_manifest_path: Path,
                        exclusions_path: Path, camera_id: str) -> dict[str, Any]:
    """Fit only on person-excluded static pixels; evaluate other frames/tiles."""
    catalog = _json(catalog_path)
    static = _json(static_manifest_path)
    bound, prior_id = _binding(catalog, static, camera_id)
    revision_records = [r for r in catalog.get("revisions", []) if r.get("prior_id") == prior_id]
    if len(revision_records) != 1:
        raise AlignmentError("selected PCF revision missing from catalog")
    scene_manifest_path = _verified_file(catalog_path.parent, revision_records[0])
    scene = _json(scene_manifest_path)
    if scene.get("prior_id") != prior_id or scene.get("quality", {}).get("passed") is not True:
        raise AlignmentError("selected PCF manifest is not passed")
    scene_files = scene.get("artifacts", [])
    pcf_path = _verified_file(scene_manifest_path.parent, _artifact(scene_files, "points.glb"))
    files = static.get("files", [])
    static_root = static_manifest_path.parent
    fused_path = _verified_file(static_root, _artifact(files, "fused_depth.npz"))
    with np.load(fused_path, allow_pickle=False) as fused:
        K = np.asarray(fused["intrinsics"], dtype=float)
        c2w = np.asarray(fused["camera_to_world"], dtype=float)
        shape = tuple(int(v) for v in fused["depth_z"].shape)
    if len(shape) != 2:
        raise AlignmentError("fused depth must be 2D")
    exclusion_sha = _digest(exclusions_path)
    masks = _exclusion_masks(_json(exclusions_path), static, _digest(static_manifest_path), shape)
    import trimesh  # Offline review dependency; never loaded by the runtime.
    loaded = trimesh.load(pcf_path, force="scene")
    pointclouds = loaded.dump()
    if len(pointclouds) != 1 or not isinstance(pointclouds[0], trimesh.points.PointCloud):
        raise AlignmentError("PCF display artifact is not one point cloud")
    points = np.asarray(pointclouds[0].vertices)
    if len(points) != scene.get("quality", {}).get("selected_point_count"):
        raise AlignmentError("PCF point count disagrees with manifest")
    pcf = project_pcf_optical_z(points, c2w, K, shape)
    filled = np.isfinite(pcf)
    local_min = cv2.erode(pcf, np.ones((3, 3), dtype=np.uint8))
    local_max = cv2.dilate(np.where(filled, pcf, 0), np.ones((3, 3), dtype=np.uint8))
    yy, xx = np.indices(shape)
    spaced = (yy % 6 == 0) & (xx % 6 == 0)
    fit_tile = ((yy // 24 + xx // 24) % 2 == 0)
    rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for frame in static["source"]["frames"]:
        name = str(frame["raw_artifact"])
        path = _verified_file(static_root, _artifact(files, name))
        with np.load(path, allow_pickle=False) as source:
            model = np.asarray(source["depth_z"], dtype=np.float32)
            mask = np.asarray(source["mask"], dtype=bool)
            conf = np.asarray(source["confidence"], dtype=np.float32)
            frame_K = np.asarray(source["intrinsics"], dtype=float)
            frame_idx = int(np.asarray(source["source_frame_index"]).reshape(-1)[0])
            pts_ns = int(np.asarray(source["encoded_pts_ns"]).reshape(-1)[0])
        if (model.shape != shape or mask.shape != shape or conf.shape != shape
                or not np.allclose(frame_K, K, atol=1e-6)
                or frame_idx != frame.get("source_frame_index")
                or pts_ns != int(frame.get("encoded_pts_ns", -1))):
            raise AlignmentError(f"static optical-Z raster identity mismatch: {name}")
        valid = (filled & mask & ~masks[name] & np.isfinite(model) & (model > 0.3)
                 & (model < 12.0) & (pcf < 12.0) & (conf >= 0.5)
                 & (local_max - local_min < 0.2) & spaced)
        rows.append((name, pcf[valid & fit_tile], model[valid & fit_tile], valid))
    train_count = min(3, len(rows) // 2)
    if train_count < 2:
        raise AlignmentError("not enough static frames for disjoint fit and holdout")
    training = np.concatenate([rows[i][1] / rows[i][2] for i in range(train_count)])
    if len(training) < 64 or not np.all(np.isfinite(training)):
        raise AlignmentError("person-excluded PCF scale support is insufficient")
    scale = float(np.median(training))
    holdout = []
    for name, _, _, valid in rows[train_count:]:
        path = static_root / name
        with np.load(path, allow_pickle=False) as source:
            model = np.asarray(source["depth_z"], dtype=np.float32)
        select = valid & ~fit_tile
        reference = pcf[select]
        estimate = model[select]
        if len(reference) < 32:
            raise AlignmentError(f"held-out PCF support insufficient: {name}")
        bands = []
        for lo, hi in DEPTH_BANDS_M:
            band = (reference >= lo) & (reference < hi)
            bands.append({"range_m": [lo, hi], **_heldout_row(reference[band], estimate[band], scale)})
        holdout.append({"static_artifact": name, **_heldout_row(reference, estimate, scale),
                        "bands": bands})
    return {
        "contract": CONTRACT,
        "camera_id": camera_id,
        "world_frame": "backend_world_m",
        "world_frame_revision": prior_id,
        "camera_calibration_sha256": bound["source_camera_calibration_sha256"],
        "world_transform_sha256": bound["target_from_source_sha256"],
        "optical_depth_space": OPTICAL_Z,
        "source_prior_id": prior_id,
        "static_reference_id": static["revision_id"],
        "catalog_sha256": _digest(catalog_path),
        "static_manifest_sha256": _digest(static_manifest_path),
        "person_exclusions_sha256": exclusion_sha,
        "fit": {"scale": scale, "count": int(len(training)),
                "partition": f"first_{train_count}_frames_even_24px_tiles_stride6"},
        "holdout": holdout,
        "runtime_eligible": False,
        "limitation": "Static PCF scale consistency is not occupied-person depth qualification.",
        "occupied_application": "Multiply current-frame MapAnything person-support optical Z at the DAv2 anchor; never substitute PCF depth under a person.",
    }


def scale_occupied_person_samples(report: Mapping[str, Any], document: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Create review pairs only from exact, current person-support optical Z."""
    if document.get("contract") != OCCUPIED_CONTRACT:
        raise AlignmentError("occupied sample contract mismatch")
    for key in ("camera_id", "world_frame", "world_frame_revision", "camera_calibration_sha256", "world_transform_sha256"):
        if document.get(key) != report.get(key):
            raise AlignmentError(f"occupied sample {key} binding mismatch")
    if document.get("optical_depth_space") != OPTICAL_Z:
        raise AlignmentError("occupied MapAnything depth is not optical Z")
    if report.get("contract") != CONTRACT or report.get("runtime_eligible") is not False:
        raise AlignmentError("static report is not a review-only PCF alignment")
    scale = float(report["fit"]["scale"])
    if not math.isfinite(scale) or scale <= 0:
        raise AlignmentError("invalid static scale")
    result = []
    rows = document.get("samples")
    if not isinstance(rows, list) or not rows:
        raise AlignmentError("occupied review has no samples")
    seen_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise AlignmentError("occupied sample must be an object")
        if (row.get("anchor_source") not in PERSON_ANCHORS
                or row.get("support_kind") != "current_person_silhouette_or_pose_capsule"
                or not isinstance(row.get("source_frame_id"), str)
                or not row["source_frame_id"]
                or row.get("source_frame_id") != row.get("mapanything_frame_id")
                or row.get("source_frame_id") != row.get("dav2_frame_id")
                or row.get("person_support_valid") is not True
                or type(row.get("person_support_count")) is not int
                or row["person_support_count"] < 8):
            raise AlignmentError("occupied row lacks exact current person support")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen_ids:
            raise AlignmentError("occupied sample id is absent or duplicated")
        seen_ids.add(sample_id)
        frame_ids = (row.get("source_frame_index"), row.get("mapanything_frame_index"),
                     row.get("dav2_frame_index"))
        pts = (row.get("source_encoded_pts_us"), row.get("mapanything_encoded_pts_us"),
               row.get("dav2_encoded_pts_us"))
        if (any(type(value) is not int for value in (*frame_ids, *pts))
                or len(set(frame_ids)) != 1 or len(set(pts)) != 1):
            raise AlignmentError("occupied source/model frame index or encoded PTS mismatch")
        uv = row.get("anchor_uv_normalized")
        model_uv = row.get("mapanything_anchor_uv_normalized")
        dav2_uv = row.get("dav2_anchor_uv_normalized")
        if (not isinstance(uv, list) or len(uv) != 2
                or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 or v > 1 for v in uv)
                or model_uv != uv or dav2_uv != uv):
            raise AlignmentError("occupied anchor pixel is not identical across paired maps")
        if not (isinstance(row.get("person_support_sha256"), str)
                and len(row["person_support_sha256"]) == 64
                and all(c in "0123456789abcdef" for c in row["person_support_sha256"])):
            raise AlignmentError("occupied person support digest missing")
        raw = float(row["raw_dav2_optical_z_m"])
        model = float(row["mapanything_person_optical_z_m"])
        if not all(math.isfinite(v) and v > 0 for v in (raw, model)):
            raise AlignmentError("occupied optical Z is invalid")
        result.append({"sample_id": sample_id, "source_frame_id": row["source_frame_id"],
                       "source_frame_index": frame_ids[0], "encoded_pts_us": pts[0],
                       "anchor_uv_normalized": uv,
                       "anchor_source": row["anchor_source"], "raw_dav2_optical_z_m": raw,
                       "target_mapanything_pcf_scaled_optical_z_m": scale * model,
                       "person_support_sha256": row["person_support_sha256"]})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--static-manifest", required=True, type=Path)
    parser.add_argument("--person-exclusions", required=True, type=Path)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--occupied-samples", type=Path)
    args = parser.parse_args()
    report = review_static_scale(args.catalog, args.static_manifest, args.person_exclusions, args.camera)
    if args.occupied_samples is not None:
        report["occupied_review_pairs"] = scale_occupied_person_samples(report, _json(args.occupied_samples))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise AlignmentError("refusing to overwrite an existing review")
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
