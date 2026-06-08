#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis.virtual_twin.artifacts import write_json
from noesis.virtual_twin.geometry import (
    PlaneCandidate,
    compute_depth_normals_camera,
    decode_mask_rle,
    fuse_mapanything_with_planes,
    transform_plane,
    transform_points,
)
from noesis.virtual_twin.store import VirtualTwinStore, validate_revision_id


class PlaneAnnotationError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PlaneAnnotationError(f"expected JSON object at {path}")
    return payload


def _replace_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        path.unlink()
    write_json(path, payload)


def _camera_to_world(calibration: Mapping[str, Any]) -> np.ndarray:
    extr = calibration.get("extrinsics_col_major")
    if not isinstance(extr, Sequence) or len(extr) != 16:
        raise PlaneAnnotationError("frame calibration missing extrinsics_col_major")
    world_to_camera = np.asarray([float(x) for x in extr], dtype=np.float64).reshape((4, 4), order="F")
    return np.linalg.inv(world_to_camera)


def _intrinsics(calibration: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(calibration.get("intrinsics"), dtype=np.float64).reshape(3, 3)


def _frame_rows(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = ((manifest.get("source_file_refs") or {}).get("frames") or [])
    if not isinstance(rows, list) or not rows:
        raise PlaneAnnotationError("manifest has no source_file_refs.frames entries")
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        frame_id = str(row.get("frame_id") or "").strip()
        artifacts = row.get("revision_artifacts") or {}
        if frame_id and isinstance(artifacts, dict):
            out[frame_id] = row
    if not out:
        raise PlaneAnnotationError("manifest has no usable frame rows")
    return out


def _load_depth_payload(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise PlaneAnnotationError(f"MapAnything frame evidence is missing: {path}")
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _persist_normals(path: Path, payload: Mapping[str, np.ndarray], normals: np.ndarray) -> None:
    updated = {key: np.asarray(value) for key, value in payload.items()}
    mags = np.linalg.norm(normals[:, :, :3], axis=-1)
    updated["normals_camera"] = normals[:, :, :3].astype(np.float16)
    updated["normals_valid"] = (
        np.isfinite(normals[:, :, :3]).all(axis=-1) & np.isfinite(mags) & (mags > 0.20)
    ).astype(np.uint8)
    if path.exists():
        path.unlink()
    np.savez_compressed(path, **updated)


def _normal_support_with_world(support: Mapping[str, Any], camera_to_world: np.ndarray) -> dict[str, Any]:
    out = dict(support or {})
    normal_camera = out.get("mapanything_normal_camera")
    if isinstance(normal_camera, Sequence) and len(normal_camera) >= 3:
        try:
            n_cam = np.asarray([float(x) for x in normal_camera[:3]], dtype=np.float64).reshape(3)
            n_world = np.asarray(camera_to_world[:3, :3], dtype=np.float64) @ n_cam
            norm = float(np.linalg.norm(n_world))
            if np.isfinite(norm) and norm > 1e-9:
                out["mapanything_normal_world"] = [float(x) for x in (n_world / norm)]
        except Exception:
            out["mapanything_normal_world"] = None
    return out


def _candidate_from_plane(plane: Mapping[str, Any]) -> PlaneCandidate | None:
    try:
        mask = decode_mask_rle(plane.get("mask_rle") or {})
        normal = np.asarray(plane.get("camera_normal"), dtype=np.float32).reshape(3)
        offset = float(plane.get("camera_offset"))
    except Exception:
        return None
    return PlaneCandidate(
        frame_id=str(plane.get("frame_id") or ""),
        plane_id=str(plane.get("plane_id") or ""),
        mask=mask,
        normal=normal,
        offset=offset,
        confidence=float(plane.get("confidence", 1.0) or 1.0),
        semantic_label=str(plane.get("semantic_label") or "").strip() or None,
    )


def annotate_revision(
    *,
    store: VirtualTwinStore,
    source_revision: str,
    output_revision: str,
    min_confidence: float,
    min_plane_support: int,
) -> dict[str, Any]:
    source_revision = validate_revision_id(source_revision)
    output_revision = validate_revision_id(output_revision)
    source_dir = store.revision_dir(source_revision)
    output_dir = store.revision_dir(output_revision)
    if not source_dir.exists():
        raise PlaneAnnotationError(f"source revision does not exist: {source_revision}")
    if output_dir.exists():
        raise PlaneAnnotationError(f"output revision already exists: {output_revision}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, output_dir)

    manifest_path = output_dir / "manifest.json"
    planes_path = output_dir / "planes.json"
    metrics_path = output_dir / "metrics.json"
    manifest = _read_json(manifest_path)
    planes_payload = _read_json(planes_path)
    metrics = _read_json(metrics_path) if metrics_path.exists() else {}
    frames = _frame_rows(manifest)
    source_planes = [p for p in (planes_payload.get("planes") or []) if isinstance(p, dict)]
    planes_by_frame: dict[str, list[dict[str, Any]]] = {}
    for plane in source_planes:
        planes_by_frame.setdefault(str(plane.get("frame_id") or ""), []).append(plane)

    annotated_planes: list[dict[str, Any]] = []
    per_frame_metrics: list[dict[str, Any]] = []
    normal_error_values: list[float] = []
    rejected_count = 0
    supported_count = 0

    for frame_id, frame_row in frames.items():
        artifacts = frame_row.get("revision_artifacts") or {}
        depth_rel = artifacts.get("mapanything_npz")
        calibration = artifacts.get("calibration")
        if not isinstance(depth_rel, str):
            candidate_rel = Path("mapanything") / f"{frame_id}.npz"
            if (output_dir / candidate_rel).exists():
                depth_rel = str(candidate_rel)
        if not isinstance(depth_rel, str) or not isinstance(calibration, dict):
            continue
        depth_path = output_dir / depth_rel
        depth_payload = _load_depth_payload(depth_path)
        depth = np.asarray(depth_payload["depth"], dtype=np.float32)
        confidence = np.asarray(depth_payload["confidence"], dtype=np.float32)
        mask = np.asarray(depth_payload["mask"], dtype=bool)
        valid = mask & np.isfinite(depth) & (depth > 0.0) & np.isfinite(confidence) & (confidence >= float(min_confidence))
        intrinsics = _intrinsics(calibration)
        normals = compute_depth_normals_camera(depth, valid, intrinsics)
        normal_valid = np.linalg.norm(normals, axis=-1) > 0.20
        _persist_normals(depth_path, depth_payload, normals)

        candidates = [
            candidate
            for candidate in (_candidate_from_plane(plane) for plane in planes_by_frame.get(frame_id, []))
            if candidate is not None
        ]
        if not candidates:
            continue
        fusion = fuse_mapanything_with_planes(
            frame_id=frame_id,
            map_depth=depth,
            map_confidence=confidence,
            map_mask=mask,
            map_normals_camera=normals,
            map_normals_valid=normal_valid,
            intrinsics=intrinsics,
            plane_candidates=candidates,
            min_confidence=float(min_confidence),
            min_plane_support=int(min_plane_support),
        )
        camera_to_world = _camera_to_world(calibration)
        per_frame_metrics.append({"frame_id": frame_id, **fusion.metrics})
        rejected_count += int(fusion.metrics.get("normal_rejected_plane_count") or 0)
        supported_count += int(fusion.metrics.get("normal_supported_plane_count") or 0)
        for plane in fusion.planes:
            world_normal, world_offset = transform_plane(plane.normal, plane.offset, camera_to_world)
            world_centroid = transform_points(np.asarray([plane.centroid], dtype=np.float32), camera_to_world)[0]
            support = _normal_support_with_world(plane.normal_support, camera_to_world)
            if support.get("normal_angular_error_deg_median") is not None:
                normal_error_values.append(float(support["normal_angular_error_deg_median"]))
            annotated_planes.append(
                {
                    "frame_id": plane.frame_id,
                    "plane_id": plane.plane_id,
                    "semantic_label": plane.semantic_label,
                    "camera_normal": [float(x) for x in plane.normal],
                    "camera_offset": float(plane.offset),
                    "world_normal": [float(x) for x in world_normal],
                    "world_offset": float(world_offset),
                    "world_centroid_m": [float(x) for x in world_centroid],
                    "polygon": plane.polygon,
                    "mask_rle": plane.mask_rle,
                    "support_pixels": int(plane.support_pixels),
                    "confidence": float(plane.confidence),
                    "fusion_score": float(plane.fusion_score),
                    "median_residual_m": float(plane.median_residual_m),
                    "p90_residual_m": float(plane.p90_residual_m),
                    "raw_median_residual_m": float(plane.raw_median_residual_m),
                    "depth_support": dict(plane.depth_support or {}),
                    "normal_support": support,
                }
            )

    created_ts_us = int(time.time() * 1_000_000)
    manifest["revision_id"] = output_revision
    manifest["created_ts_us"] = created_ts_us
    manifest["source_revision_id"] = source_revision
    model_fingerprints = manifest.setdefault("model_fingerprints", {})
    model_fingerprints["mapanything_normals_fusion"] = {
        "source_revision_id": source_revision,
        "method": "revision_depth_recomputed_camera_normals",
        "min_confidence": float(min_confidence),
    }
    planes_payload.update(
        {
            "schema": "noesis.virtual_twin.planes.v2",
            "revision_id": output_revision,
            "normal_fusion": {
                "status": "computed_from_mapanything_depth",
                "source_revision_id": source_revision,
                "support_fields": ["normal_support", "depth_support", "fusion_score"],
            },
            "planes": annotated_planes,
        }
    )
    metrics["revision_id"] = output_revision
    metrics["source_revision_id"] = source_revision
    metrics["accepted_plane_count"] = len(annotated_planes)
    metrics["mapanything_normals_fusion"] = {
        "status": "computed_from_depth",
        "supported_plane_count": supported_count,
        "rejected_plane_count": rejected_count,
        "median_angular_error_deg": float(np.median(normal_error_values)) if normal_error_values else None,
        "p90_angular_error_deg": float(np.percentile(normal_error_values, 90.0)) if normal_error_values else None,
        "per_frame": per_frame_metrics,
    }
    _replace_json(manifest_path, manifest)
    _replace_json(planes_path, planes_payload)
    _replace_json(metrics_path, metrics)
    for name in ("tracking_alignment.json",):
        path = output_dir / name
        if path.exists():
            payload = _read_json(path)
            payload["revision_id"] = output_revision
            payload["source_revision_id"] = source_revision
            _replace_json(path, payload)
    return {
        "source_revision_id": source_revision,
        "revision_id": output_revision,
        "revision_dir": str(output_dir),
        "accepted_plane_count": len(annotated_planes),
        "normal_supported_plane_count": supported_count,
        "normal_rejected_plane_count": rejected_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clone a virtual-twin revision and annotate planes with MapAnything normal support")
    parser.add_argument("revision", help="Source revision id")
    parser.add_argument("--output-revision", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--min-plane-support", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = validate_revision_id(args.revision)
    output = validate_revision_id(args.output_revision or f"{source}_normals_{time.strftime('%Y%m%dT%H%M%S')}")
    result = annotate_revision(
        store=VirtualTwinStore(Path(args.output_root) if args.output_root else None),
        source_revision=source,
        output_revision=output,
        min_confidence=float(args.min_confidence),
        min_plane_support=int(args.min_plane_support),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
