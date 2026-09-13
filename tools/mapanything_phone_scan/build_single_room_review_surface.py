#!/usr/bin/env python3
"""Build one room's review-only surface from a passed consensus reconstruction.

The command adapts a single-room consensus to the existing per-owner surface
mesh builder. It applies only the already-passed rigid alignment, preserves
single-view-withheld points as a separate point artifact, and never publishes
or mutates runtime/Menon state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import trimesh

from tools.mapanything_phone_scan.build_pcf_review_surface_mesh import build as build_surface_mesh


OWNER_CAMERA_FIELDS = {
    1: "fixed_camera_positions",
    2: "moving_camera_positions",
    3: "living_camera_positions",
    4: "connector_camera_positions",
}
OWNER_NAMES = {1: "family", 2: "kitchen", 3: "living", 4: "connector"}
OWNER_CAMERA_IDS = {
    1: "family-room",
    2: "kitchen",
    3: "living-room",
    4: "connector",
}
ROOM_ALIASES = {
    "family-room": "family",
    "family_room": "family",
    "kitchen-room": "kitchen",
    "living-room": "living",
    "living_room": "living",
}
HASH_CHUNK = 1024 * 1024


class SingleRoomSurfaceError(ValueError):
    """Raised when a single-room surface cannot be bound without ambiguity."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SingleRoomSurfaceError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise SingleRoomSurfaceError(f"JSON artifact is not an object: {path}")
    return value


def normalize_room_name(value: str, owner: int) -> str:
    room = str(value).strip().lower().replace(" ", "-")
    room = ROOM_ALIASES.get(room, room)
    expected = OWNER_NAMES.get(int(owner))
    if expected is None:
        raise SingleRoomSurfaceError(f"owner must be one of {sorted(OWNER_NAMES)}")
    if room != expected:
        raise SingleRoomSurfaceError(
            f"room-name {value!r} does not match owner {owner} ({expected!r})"
        )
    return room


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    return (transform[:3, :3] @ values.T).T + transform[:3, 3]


def _rigid_transform(value: Any, label: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise SingleRoomSurfaceError(f"{label} is not a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise SingleRoomSurfaceError(f"{label} is not affine")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise SingleRoomSurfaceError(f"{label} contains scale or shear")
    if abs(float(np.linalg.det(rotation)) - 1.0) > 2e-3:
        raise SingleRoomSurfaceError(f"{label} is reflected or otherwise invalid")
    return transform


def _required_path_binding(
    value: Any,
    *,
    label: str,
    expected_path: Path,
) -> dict[str, Any]:
    """Require and verify a manifest-named path plus its content digest."""

    if not isinstance(value, Mapping):
        raise SingleRoomSurfaceError(f"{label} binding is missing")
    declared_path = str(value.get("path") or "").strip()
    declared_sha = str(value.get("sha256") or "").strip()
    if not declared_path or not declared_sha:
        raise SingleRoomSurfaceError(f"{label} binding requires path and sha256")
    expected_path = expected_path.resolve()
    if Path(declared_path).resolve() != expected_path:
        raise SingleRoomSurfaceError(f"{label} path does not match expected source")
    if not expected_path.is_file():
        raise SingleRoomSurfaceError(f"{label} source file is missing: {expected_path}")
    actual_sha = _sha256(expected_path)
    if declared_sha != actual_sha:
        raise SingleRoomSurfaceError(f"{label} digest mismatch")
    return {
        "path": str(expected_path),
        "sha256": actual_sha,
    }


def _validate_alignment(
    consensus_dir: Path,
    alignment_dir: Path,
    *,
    expected_camera_id: str,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    alignment_path = alignment_dir / "alignment_report.json"
    transform_path = alignment_dir / "phone_ma_to_noesis_world.json"
    alignment = _json(alignment_path)
    transform_payload = _json(transform_path)
    if alignment.get("status") != "passed":
        raise SingleRoomSurfaceError("alignment report status is not passed")
    if alignment.get("quality_gate", {}).get("passed") is not True:
        raise SingleRoomSurfaceError("alignment quality gate is not passed")
    target = alignment.get("target")
    if not isinstance(target, Mapping):
        raise SingleRoomSurfaceError("alignment report has no target binding")
    if target.get("coordinate_frame") != "backend_world_m_stream_points":
        raise SingleRoomSurfaceError("alignment target coordinate frame is not backend world meters")
    if target.get("world_frame") != "backend_world_m":
        raise SingleRoomSurfaceError("alignment target world frame is not backend_world_m")
    target_camera_id = str(target.get("camera_id") or "").strip()
    if not target_camera_id:
        raise SingleRoomSurfaceError("alignment target has no camera id")
    if target_camera_id != expected_camera_id:
        raise SingleRoomSurfaceError(
            f"alignment target camera id {target_camera_id!r} does not match requested room camera {expected_camera_id!r}"
        )
    if not str(target.get("revision_id") or "").strip():
        raise SingleRoomSurfaceError("alignment target has no revision id")
    target_world_frame_revision = str(target.get("world_frame_revision") or "").strip()
    if not target_world_frame_revision:
        raise SingleRoomSurfaceError("alignment target has no world frame revision")
    source_raw = (consensus_dir / "raw").resolve()
    declared_raw = alignment.get("transform", {}).get("source_raw_root")
    if not str(declared_raw or "").strip():
        raise SingleRoomSurfaceError("alignment source raw root binding is missing")
    if Path(str(declared_raw)).resolve() != source_raw:
        raise SingleRoomSurfaceError("alignment source raw root does not match consensus directory")
    transform = _rigid_transform(
        alignment.get("transform", {}).get("world_from_mapanything_row_major"),
        "alignment world_from_mapanything",
    )
    standalone_transform = _rigid_transform(
        transform_payload.get("world_from_mapanything_row_major"),
        "world-alignment artifact",
    )
    if not np.allclose(transform, standalone_transform, atol=1e-9):
        raise SingleRoomSurfaceError("alignment report and world-alignment transform disagree")
    if transform_payload.get("target_coordinate_frame") != "backend_world_m_stream_points":
        raise SingleRoomSurfaceError("world-alignment target frame is not backend world meters")
    target_binding = transform_payload.get("target_binding")
    if not isinstance(target_binding, Mapping):
        raise SingleRoomSurfaceError("world-alignment target binding is missing")
    if target_binding.get("world_frame") != target.get("world_frame"):
        raise SingleRoomSurfaceError("world-alignment target world frame disagrees with alignment report")
    if target_binding.get("world_frame_revision") != target_world_frame_revision:
        raise SingleRoomSurfaceError(
            "world-alignment target world frame revision disagrees with alignment report"
        )
    target_camera_binding = target.get("camera_frame_binding")
    payload_camera_binding = target_binding.get("camera_frame_binding")
    if not isinstance(target_camera_binding, Mapping):
        raise SingleRoomSurfaceError("alignment target camera frame binding is missing")
    if not isinstance(payload_camera_binding, Mapping):
        raise SingleRoomSurfaceError("world-alignment camera frame binding is missing")
    if payload_camera_binding != target_camera_binding:
        raise SingleRoomSurfaceError(
            "world-alignment camera frame binding disagrees with alignment report"
        )
    if not str(transform_payload.get("source_coordinate_frame") or "").strip():
        raise SingleRoomSurfaceError("world-alignment source coordinate frame is missing")
    if not str(transform_payload.get("source_raw_root") or "").strip():
        raise SingleRoomSurfaceError("world-alignment source raw root binding is missing")
    if Path(str(transform_payload["source_raw_root"])).resolve() != source_raw:
        raise SingleRoomSurfaceError("world-alignment source raw root does not match consensus directory")
    manifest_path = (consensus_dir / "scan_outputs_manifest.json").resolve()
    _required_path_binding(
        alignment.get("transform", {}).get("source_output_manifest"),
        label="alignment source output manifest",
        expected_path=manifest_path,
    )
    _required_path_binding(
        transform_payload.get("source_output_manifest"),
        label="world-alignment source output manifest",
        expected_path=manifest_path,
    )
    output_manifest = _json(manifest_path)
    view_count = int(output_manifest.get("view_count") or 0)
    if view_count < 2 or view_count > 4096:
        raise SingleRoomSurfaceError(f"unsupported consensus view count: {view_count}")
    phone_source = alignment.get("inputs", {}).get("phone_source", {})
    if not isinstance(phone_source, Mapping):
        raise SingleRoomSurfaceError("alignment report has no phone source binding")
    if not str(phone_source.get("raw_root") or "").strip():
        raise SingleRoomSurfaceError("alignment phone source raw root binding is missing")
    if Path(str(phone_source["raw_root"])).resolve() != source_raw:
        raise SingleRoomSurfaceError("alignment phone source raw root does not match consensus directory")
    _required_path_binding(
        phone_source.get("output_manifest"),
        label="alignment phone source output manifest",
        expected_path=manifest_path,
    )
    views = phone_source.get("views")
    if not isinstance(views, list) or len(views) != view_count:
        raise SingleRoomSurfaceError("alignment report does not name every consensus raw view")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(views):
        if not isinstance(item, Mapping):
            raise SingleRoomSurfaceError(f"alignment source view {index} is malformed")
        path = (source_raw / f"view_{index:04d}.npz").resolve()
        declared_path = Path(str(item.get("path") or "")).resolve()
        if declared_path != path or not path.is_file():
            raise SingleRoomSurfaceError(f"alignment source view {index} is not the expected raw row")
        actual_sha = _sha256(path)
        declared_sha = str(item.get("sha256") or "")
        if not declared_sha:
            raise SingleRoomSurfaceError(f"alignment source view {index} digest is missing")
        if declared_sha != actual_sha:
            raise SingleRoomSurfaceError(f"alignment source view {index} digest mismatch")
        declared_size = item.get("size_bytes")
        if declared_size is not None and int(declared_size) != path.stat().st_size:
            raise SingleRoomSurfaceError(f"alignment source view {index} size mismatch")
        rows.append(
            {
                "path": str(path),
                "view_index": index,
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": actual_sha,
                "effective_world_from_local_row_major": transform.tolist(),
            }
        )
    return alignment, transform_payload, transform, rows


def _load_points(path: Path, *, label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as arrays:
            points = np.asarray(arrays["points"], dtype=np.float64)
            colors = np.asarray(arrays["colors"], dtype=np.uint8)
            support = np.asarray(arrays["support_view_count"], dtype=np.uint16)
    except (OSError, KeyError, ValueError) as exc:
        raise SingleRoomSurfaceError(f"{label} surfel artifact is malformed: {path}") from exc
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise SingleRoomSurfaceError(f"{label} points/colors shape mismatch: {path}")
    if support.ndim != 1 or len(support) != len(points):
        raise SingleRoomSurfaceError(f"{label} support count shape mismatch: {path}")
    if not np.isfinite(points).all():
        raise SingleRoomSurfaceError(f"{label} points contain non-finite values: {path}")
    return points, colors, support


def _write_withheld_points(points: np.ndarray, colors: np.ndarray, path: Path) -> dict[str, Any]:
    tinted = np.rint(
        0.55 * colors.astype(np.float64) + 0.45 * np.asarray([255.0, 180.0, 0.0])
    ).clip(0, 255).astype(np.uint8)
    rgba = np.column_stack((tinted, np.full(len(tinted), 220, dtype=np.uint8)))
    cloud = trimesh.points.PointCloud(points, colors=rgba)
    cloud.metadata.update({
        "review_only": True,
        "support_class": "single_view_withheld",
        "coordinate_frame": "backend_world_m_stream_points",
    })
    scene = trimesh.Scene(metadata={"review_only": True, "support_class": "single_view_withheld"})
    scene.add_geometry(
        cloud,
        geom_name="pcf_surface_single_room_uncertain_points",
        node_name="pcf_surface_single_room_uncertain_points",
    )
    path.write_bytes(trimesh.exchange.gltf.export_glb(scene))
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "point_count": int(len(points)),
    }


def prepare_input(
    *,
    consensus_dir: Path,
    alignment_dir: Path,
    room_name: str,
    owner: int,
    output_dir: Path,
    ceiling_mode: str = "full_height",
    ceiling_cutaway_m: float = 1.85,
    full_height_margin_m: float = 0.03,
) -> dict[str, Any]:
    consensus_dir = consensus_dir.resolve()
    alignment_dir = alignment_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise SingleRoomSurfaceError(f"refusing to overwrite existing output directory: {output_dir}")
    if ceiling_mode not in {"full_height", "cutaway"}:
        raise SingleRoomSurfaceError("ceiling-mode must be full_height or cutaway")
    if not math.isfinite(float(ceiling_cutaway_m)) or ceiling_cutaway_m <= 0.0:
        raise SingleRoomSurfaceError("ceiling-cutaway-m must be finite and positive")
    if not math.isfinite(float(full_height_margin_m)) or full_height_margin_m < 0.0:
        raise SingleRoomSurfaceError("full-height-margin-m must be finite and non-negative")
    canonical_room = normalize_room_name(room_name, owner)
    expected_camera_id = OWNER_CAMERA_IDS[int(owner)]
    alignment, transform_payload, transform, raw_rows = _validate_alignment(
        consensus_dir,
        alignment_dir,
        expected_camera_id=expected_camera_id,
    )
    accepted_path = consensus_dir / "surfel_points.npz"
    withheld_path = consensus_dir / "surfel_points_single_view.npz"
    consensus_manifest_path = consensus_dir / "consensus_manifest.json"
    accepted_points, accepted_colors, accepted_support = _load_points(accepted_path, label="accepted")
    withheld_points, withheld_colors, withheld_support = _load_points(withheld_path, label="withheld")
    if len(accepted_points) < 1 or np.min(accepted_support) < 2:
        raise SingleRoomSurfaceError("accepted surfels do not have distinct multi-view support")
    if len(withheld_points) and np.any(withheld_support != 1):
        raise SingleRoomSurfaceError("withheld surfels are not all single-view support")
    transformed_accepted = _transform_points(accepted_points, transform).astype(np.float32)
    transformed_withheld = _transform_points(withheld_points, transform).astype(np.float32)
    if not np.isfinite(transformed_accepted).all() or not np.isfinite(transformed_withheld).all():
        raise SingleRoomSurfaceError("transformed surfels contain non-finite points")
    measured_upper_y = float(np.max(transformed_accepted[:, 1]))
    if ceiling_mode == "full_height":
        effective_ceiling = measured_upper_y + float(full_height_margin_m)
    else:
        effective_ceiling = float(ceiling_cutaway_m)
    output_dir.mkdir(parents=True, exist_ok=False)
    output_npz = output_dir / "surface_input_surfels.npz"
    output_manifest = output_dir / "surface_input_manifest.json"
    withheld_output = output_dir / "withheld_single_view_points.glb"
    camera_field = OWNER_CAMERA_FIELDS[int(owner)]
    camera_positions: list[np.ndarray] = []
    for row in raw_rows:
        with np.load(row["path"], allow_pickle=False) as arrays:
            pose = np.asarray(arrays["camera_pose"], dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise SingleRoomSurfaceError(f"camera pose is malformed: {row['path']}")
        camera_positions.append((transform @ pose)[:3, 3])
    np.savez_compressed(
        output_npz,
        points=transformed_accepted,
        colors=accepted_colors,
        owner_room_id=np.full(len(transformed_accepted), int(owner), dtype=np.uint8),
        view_count=accepted_support,
        geometry_status=np.zeros(len(transformed_accepted), dtype=np.uint8),
        **{camera_field: np.asarray(camera_positions, dtype=np.float32)},
    )
    withheld_artifact = _write_withheld_points(transformed_withheld, withheld_colors, withheld_output)
    target = alignment["target"]
    registration = {
        "status": "passed",
        "disposition": "validated_review_candidate",
        "accepted_for_canonical_use": False,
        "source_coordinate_frame": transform_payload.get("source_coordinate_frame"),
        "target_coordinate_frame": transform_payload.get("target_coordinate_frame"),
        "target_camera_id": target.get("camera_id"),
        "target_revision_id": target.get("revision_id"),
        "world_frame": target.get("world_frame"),
        "world_frame_revision": target.get("world_frame_revision"),
        "alignment_report_sha256": _sha256(alignment_dir / "alignment_report.json"),
        "world_alignment_file_sha256": _sha256(alignment_dir / "phone_ma_to_noesis_world.json"),
        "world_from_mapanything_row_major": transform.tolist(),
    }
    manifest = {
        "schema": "noesis.pcf.single_room_surface_input.v2",
        "status": "review_only",
        "accepted_for_canonical_use": False,
        "phone_walk_only": True,
        "static_camera_points_included": False,
        "whole_cloud_icp_used": False,
        "output_coordinate_frame": "backend_world_m_stream_points",
        "room_name": canonical_room,
        "owner_room_id": int(owner),
        "registration": registration,
        "sources": {
            canonical_room: {
                "name": canonical_room,
                "room": canonical_room,
                "camera_id": alignment["target"]["camera_id"],
                "view_count": len(raw_rows),
                "raw_root": str((consensus_dir / "raw").resolve()),
                "raw_views": raw_rows,
                "source_output_manifest": _record(consensus_dir / "scan_outputs_manifest.json"),
            }
        },
        "artifacts": {
            "surfels_npz": {
                "path": output_npz.name,
                "sha256": _sha256(output_npz),
                "size_bytes": output_npz.stat().st_size,
            },
            "withheld_single_view_points": withheld_artifact,
        },
        "source": {
            "consensus_manifest": _record(consensus_manifest_path),
            "source_output_manifest": _record(consensus_dir / "scan_outputs_manifest.json"),
            "alignment_report": _record(alignment_dir / "alignment_report.json"),
            "alignment_transform": _record(alignment_dir / "phone_ma_to_noesis_world.json"),
            "source_raw_view_set_sha256": transform_payload.get("source_raw_view_set_sha256"),
            "accepted_surfels": _record(accepted_path),
            "withheld_surfels": _record(withheld_path),
        },
        "support_metrics": {
            "accepted_multi_view_count": int(len(transformed_accepted)),
            "accepted_multi_view_support_count_min": int(np.min(accepted_support)),
            "accepted_multi_view_support_count_median": float(np.median(accepted_support)),
            "accepted_multi_view_support_count_max": int(np.max(accepted_support)),
            "withheld_single_view_count": int(len(transformed_withheld)),
            "withheld_single_view_class": "uncertain_points_separate_artifact",
            "unknown_count": None,
            "unknown_count_status": "not_assessed",
            "contradiction_count": None,
            "contradiction_count_status": "not_assessed",
            "accepted_mesh_input_excludes_withheld_single_view": True,
        },
        "ceiling": {
            "mode": ceiling_mode,
            "effective_ceiling_cutaway_m": effective_ceiling,
            "finite_measured_upper_y_m": measured_upper_y,
            "full_height_margin_m": float(full_height_margin_m),
            "standard_cutaway_m": float(ceiling_cutaway_m),
        },
        "frame_identity": {
            "coordinate_frame": "backend_world_m_stream_points",
            "source_coordinate_frame": registration["source_coordinate_frame"],
            "world_frame": target.get("world_frame"),
            "world_frame_revision": target.get("world_frame_revision"),
            "target_revision_id": target.get("revision_id"),
            "target_camera_id": target.get("camera_id"),
            "room_name": canonical_room,
            "owner_room_id": int(owner),
            "source_alignment_report_sha256": _sha256(alignment_dir / "alignment_report.json"),
            "source_alignment_transform_file_sha256": _sha256(alignment_dir / "phone_ma_to_noesis_world.json"),
        },
        "limitations": [
            "surface_mesh_is_review_presentation_only",
            "single_view_withheld_points_are_not_mesh_input",
            "no_cross_room_join_or_intrinsic_change_was_performed",
            "no_runtime_or_menon_selector_was_mutated",
        ],
    }
    output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "manifest": manifest,
        "output_dir": output_dir,
        "output_npz": output_npz,
        "output_manifest": output_manifest,
        "withheld_output": withheld_output,
        "effective_ceiling": effective_ceiling,
    }


def build_single_room_surface(args: argparse.Namespace) -> dict[str, Any]:
    prepared = prepare_input(
        consensus_dir=args.consensus_dir,
        alignment_dir=args.alignment_dir,
        room_name=args.room_name,
        owner=args.owner,
        output_dir=args.output_dir,
        ceiling_mode=args.ceiling_mode,
        ceiling_cutaway_m=args.ceiling_cutaway_m,
        full_height_margin_m=args.full_height_margin_m,
    )
    output_dir = prepared["output_dir"]
    mesh_args = argparse.Namespace(
        surfels_npz=prepared["output_npz"],
        surfels_manifest=prepared["output_manifest"],
        output_glb=output_dir / "surface_mesh.glb",
        output_manifest=output_dir / "surface_mesh_manifest.json",
        voxel_size_m=args.voxel_size_m,
        normal_radius_m=args.normal_radius_m,
        support_distance_m=args.support_distance_m,
        ceiling_cutaway_m=prepared["effective_ceiling"],
        poisson_depth=args.poisson_depth,
        density_percentile=args.density_percentile,
        minimum_component_triangles=args.minimum_component_triangles,
        poisson_threads=args.poisson_threads,
    )
    mesh_report = build_surface_mesh(mesh_args)
    run_report = {
        "schema": "noesis.pcf.single_room_review_surface_run.v1",
        "status": "review_only",
        "accepted_for_canonical_use": False,
        "room_name": prepared["manifest"]["room_name"],
        "owner_room_id": prepared["manifest"]["owner_room_id"],
        "ceiling": prepared["manifest"]["ceiling"],
        "source": prepared["manifest"]["source"],
        "input_manifest": str(prepared["output_manifest"].resolve()),
        "withheld_points": prepared["manifest"]["artifacts"]["withheld_single_view_points"],
        "surface_mesh_manifest": str(mesh_args.output_manifest.resolve()),
        "surface_mesh": mesh_report["output"],
        "support_metrics": prepared["manifest"]["support_metrics"],
        "frame_identity": prepared["manifest"]["frame_identity"],
        "menon_consumer": {
            "descriptor_endpoint": "/api/v1/scenes/current/review-assemblies/whole-home",
            "artifact_endpoint": "/api/v1/scenes/current/review-assemblies/whole-home/artifacts/multiroom_surface_mesh_glb",
            "active_publication": False,
            "single_room_whole_home_wrapper": False,
        },
    }
    run_path = output_dir / "run_report.json"
    run_path.write_text(json.dumps(run_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return run_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consensus-dir", type=Path, required=True)
    parser.add_argument("--alignment-dir", type=Path, required=True)
    parser.add_argument("--room-name", required=True)
    parser.add_argument("--owner", type=int, required=True, choices=sorted(OWNER_NAMES))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ceiling-mode", choices=("full_height", "cutaway"), default="full_height")
    parser.add_argument("--ceiling-cutaway-m", type=float, default=1.85)
    parser.add_argument("--full-height-margin-m", type=float, default=0.03)
    parser.add_argument("--voxel-size-m", type=float, default=0.08)
    parser.add_argument("--normal-radius-m", type=float, default=0.14)
    parser.add_argument("--support-distance-m", type=float, default=0.10)
    parser.add_argument("--poisson-depth", type=int, default=7)
    parser.add_argument("--density-percentile", type=float, default=2.0)
    parser.add_argument("--minimum-component-triangles", type=int, default=64)
    parser.add_argument("--poisson-threads", type=int, default=1)
    return parser


def main() -> int:
    report = build_single_room_surface(_parser().parse_args())
    print(json.dumps(report["surface_mesh"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
