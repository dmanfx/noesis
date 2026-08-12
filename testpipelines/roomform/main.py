#!/usr/bin/env python3
"""Run Roomform locally on an aligned room reconstruction."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_ROOM_ID = "living-room"
ROOM_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
EXPECTED_COLOR_SOURCE = "mapanything_persisted_zarr_rgb_texture"
AXIS_TRANSFORM = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)
DA3_TO_ROOMFORM_Z_UP = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class CachedCloud:
    room_id: str
    revision_dir: Path
    source_kind: str
    source_points_path: Path
    points: np.ndarray
    colors: np.ndarray
    stations: np.ndarray
    axis_transform: np.ndarray
    manifest: dict[str, Any]
    points_meta: dict[str, Any]
    provenance: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bounds(points: np.ndarray) -> dict[str, list[float]]:
    return {
        "min": [float(value) for value in np.min(points, axis=0)],
        "max": [float(value) for value in np.max(points, axis=0)],
    }


def _camera_station(points_meta: dict[str, Any]) -> np.ndarray:
    raw = np.asarray(points_meta.get("extrinsics_col_major"), dtype=np.float64)
    if raw.shape != (16,) or not np.all(np.isfinite(raw)):
        raise RuntimeError("room cache has no valid camera extrinsics")
    matrix = raw.reshape((4, 4), order="F")
    station = matrix[:3, 3]
    if not np.all(np.isfinite(station)):
        raise RuntimeError("room cache has a non-finite camera station")
    return station.astype(np.float32)


def _validated_room_id(room_id: str) -> str:
    value = room_id.strip().lower()
    if not ROOM_ID_PATTERN.fullmatch(value):
        raise RuntimeError(
            "room id must contain only lowercase letters, numbers, and hyphens"
        )
    return value


def find_cached_room_revision(
    virtual_twin_root: Path,
    room_id: str = DEFAULT_ROOM_ID,
) -> Path:
    room_id = _validated_room_id(room_id)
    candidates: list[tuple[int, Path]] = []
    revision_prefix = room_id.replace("-", "_")
    for path in virtual_twin_root.glob(f"vt_{revision_prefix}_stream_rgbmesh_*"):
        manifest_path = path / "manifest.json"
        points_path = path / "room_points.npz"
        meta_path = path / "room_points_meta.json"
        if not (manifest_path.is_file() and points_path.is_file() and meta_path.is_file()):
            continue
        manifest = _read_json(manifest_path)
        meta = _read_json(meta_path)
        if str(manifest.get("camera")) != room_id:
            continue
        if str(meta.get("camera")) != room_id:
            continue
        if str(meta.get("color_source")) != EXPECTED_COLOR_SOURCE:
            continue
        source = str(meta.get("source", "")).lower()
        if "mapanything" not in source:
            continue
        created = int(manifest.get("created_ts_us") or meta.get("generated_ts_us") or 0)
        candidates.append((created, path))
    if not candidates:
        raise RuntimeError(
            f"no RGB-backed {room_id} MapAnything reconstruction found under "
            f"{virtual_twin_root}"
        )
    return max(candidates, key=lambda row: row[0])[1]


def find_cached_living_room_revision(virtual_twin_root: Path) -> Path:
    return find_cached_room_revision(virtual_twin_root, DEFAULT_ROOM_ID)


def load_cached_cloud(
    revision_dir: Path,
    room_id: str = DEFAULT_ROOM_ID,
) -> CachedCloud:
    room_id = _validated_room_id(room_id)
    manifest = _read_json(revision_dir / "manifest.json")
    points_meta = _read_json(revision_dir / "room_points_meta.json")
    if (
        str(manifest.get("camera")) != room_id
        or str(points_meta.get("camera")) != room_id
    ):
        raise RuntimeError(f"refusing cache input that is not for {room_id}")
    if str(points_meta.get("color_source")) != EXPECTED_COLOR_SOURCE:
        raise RuntimeError(
            "the Roomform 55M checkpoint needs the RGB-backed MapAnything cache; "
            f"received color_source={points_meta.get('color_source')!r}"
        )
    data = np.load(revision_dir / "room_points.npz")
    points = np.asarray(data["points"], dtype=np.float32)
    colors = np.asarray(data["colors"], dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 1000:
        raise RuntimeError(f"invalid cached point cloud shape: {points.shape}")
    if colors.shape != points.shape:
        raise RuntimeError(
            f"cached RGB/point shape mismatch: colors={colors.shape} points={points.shape}"
        )
    valid = np.all(np.isfinite(points), axis=1)
    points = np.ascontiguousarray(points[valid])
    colors = np.ascontiguousarray(colors[valid, :3])
    return CachedCloud(
        room_id=room_id,
        revision_dir=revision_dir,
        source_kind="fixed_camera_mapanything_cache",
        source_points_path=revision_dir / "room_points.npz",
        points=points,
        colors=colors,
        stations=_camera_station(points_meta)[None],
        axis_transform=AXIS_TRANSFORM,
        manifest=manifest,
        points_meta=points_meta,
        provenance={
            "source_room_points_npz": str(revision_dir / "room_points.npz"),
            "source_mapanything_snapshots": (
                manifest.get("model_fingerprints", {})
                .get("mapanything", {})
                .get("snapshot_paths", [])
            ),
        },
    )


def _phone_provider(outputs_manifest: dict[str, Any]) -> tuple[str, str]:
    provider = str(outputs_manifest.get("provider") or "").strip().lower()
    legacy_model = outputs_manifest.get("model", {})
    legacy_model_id = (
        str(legacy_model.get("id") or "") if isinstance(legacy_model, dict) else ""
    )
    model_id = str(outputs_manifest.get("model_id") or legacy_model_id).strip()
    if not provider and "map-anything" in model_id.lower():
        provider = "mapanything"
    if provider not in {"mapanything", "da3"}:
        raise RuntimeError(
            f"unsupported phone-walk provider={provider!r} model_id={model_id!r}"
        )
    return provider, model_id


def load_phone_walk_cloud(
    scan_dir: Path,
    room_id: str = DEFAULT_ROOM_ID,
) -> CachedCloud:
    room_id = _validated_room_id(room_id)
    alignment_dir = scan_dir / "alignment"
    points_path = alignment_dir / "aligned_phone_points.glb"
    trajectory_path = alignment_dir / "aligned_camera_trajectory.json"
    alignment_path = alignment_dir / "alignment_report.json"
    outputs_manifest_path = scan_dir / "outputs" / "scan_outputs_manifest.json"
    for path in (points_path, trajectory_path, alignment_path, outputs_manifest_path):
        if not path.is_file():
            raise RuntimeError(f"phone-walk input is incomplete: missing {path}")

    alignment = _read_json(alignment_path)
    target = alignment.get("target", {})
    gate = alignment.get("quality_gate", {})
    if (
        alignment.get("status") != "passed"
        or gate.get("passed") is not True
        or target.get("camera_id") != room_id
    ):
        raise RuntimeError(
            f"refusing a phone walk that did not pass {room_id} alignment gates"
        )
    outputs_manifest = _read_json(outputs_manifest_path)
    provider, model_id = _phone_provider(outputs_manifest)

    scene = trimesh.load(points_path, force="scene")
    geometry_name = "phone_mapanything_aligned_rgb"
    if geometry_name not in scene.geometry:
        raise RuntimeError(
            f"aligned phone GLB has no {geometry_name!r} point-cloud geometry"
        )
    geometry = scene.geometry[geometry_name]
    points = np.asarray(geometry.vertices, dtype=np.float32)
    colors = getattr(getattr(geometry, "visual", None), "vertex_colors", None)
    if colors is None or len(colors) != len(points):
        raise RuntimeError("aligned phone-walk point cloud has no per-point RGB")
    colors = np.asarray(colors, dtype=np.uint8)[:, :3]
    valid = np.all(np.isfinite(points), axis=1)
    points = np.ascontiguousarray(points[valid])
    colors = np.ascontiguousarray(colors[valid])
    if len(points) < 1000:
        raise RuntimeError(f"invalid phone-walk point count: {len(points)}")

    trajectory = _read_json(trajectory_path)
    poses = np.asarray(trajectory.get("camera_poses"), dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not np.all(np.isfinite(poses)):
        raise RuntimeError("aligned phone-walk trajectory has invalid camera poses")
    stations = np.ascontiguousarray(poses[:, :3, 3], dtype=np.float32)
    coordinate_frame = str(trajectory.get("coordinate_frame", ""))
    expected_frame = str(target.get("coordinate_frame", ""))
    if not coordinate_frame or coordinate_frame != expected_frame:
        raise RuntimeError(
            "phone-walk points and camera trajectory do not share the aligned frame"
        )

    return CachedCloud(
        room_id=room_id,
        revision_dir=scan_dir,
        source_kind=f"aligned_{provider}_phone_walk",
        source_points_path=points_path,
        points=points,
        colors=colors,
        stations=stations,
        axis_transform=AXIS_TRANSFORM,
        manifest=outputs_manifest,
        points_meta={
            "coordinate_frame": coordinate_frame,
            "color_source": f"{provider}_aligned_phone_walk_rgb",
        },
        provenance={
            "source_aligned_points_glb": str(points_path),
            "source_aligned_camera_trajectory_json": str(trajectory_path),
            "source_alignment_report_json": str(alignment_path),
            "source_alignment_status": alignment.get("status"),
            "source_alignment_admission": alignment.get("admission"),
            "source_frame_count": outputs_manifest.get("view_count"),
            "source_provider": provider,
            "source_model_id": model_id,
            "source_model": outputs_manifest.get("model"),
            "target_revision": target.get("revision_id"),
        },
    )


def load_point_preserving_fusion(
    cloud_path: Path,
    camera_solution_path: Path,
    room_id: str = DEFAULT_ROOM_ID,
) -> CachedCloud:
    room_id = _validated_room_id(room_id)
    report_path = cloud_path.parent / "point_preserving_fusion_report.json"
    for path in (cloud_path, camera_solution_path, report_path):
        if not path.is_file():
            raise RuntimeError(f"point-preserving fusion input is missing {path}")

    report = _read_json(report_path)
    if report.get("schema") != "noesis.phone_walk.point_preserving_fusion.v1":
        raise RuntimeError(f"unexpected point-preserving fusion report: {report_path}")
    with np.load(cloud_path) as data:
        required = {"points", "colors", "weights", "sample_counts", "view_counts"}
        missing = sorted(required - set(data.files))
        if missing:
            raise RuntimeError(
                f"point-preserving fusion NPZ is missing: {', '.join(missing)}"
            )
        points = np.asarray(data["points"], dtype=np.float32)
        colors = np.asarray(data["colors"], dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 1000:
        raise RuntimeError(f"invalid point-preserving fusion shape: {points.shape}")
    if colors.shape != points.shape:
        raise RuntimeError(
            f"fusion RGB/point shape mismatch: colors={colors.shape} "
            f"points={points.shape}"
        )
    valid = np.all(np.isfinite(points), axis=1)
    points = np.ascontiguousarray(points[valid])
    colors = np.ascontiguousarray(colors[valid, :3])

    with np.load(camera_solution_path) as solution:
        poses = np.asarray(solution["camera_poses"], dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not np.all(np.isfinite(poses)):
        raise RuntimeError("fusion camera solution has invalid camera poses")
    stations = np.ascontiguousarray(poses[:, :3, 3], dtype=np.float32)
    if len(stations) != int(report.get("view_count", -1)):
        raise RuntimeError(
            "fusion point evidence and camera solution have different view counts"
        )

    return CachedCloud(
        room_id=room_id,
        revision_dir=cloud_path.parent,
        source_kind="da3_mapanything_point_preserving_fusion",
        source_points_path=cloud_path,
        points=points,
        colors=colors,
        stations=stations,
        axis_transform=DA3_TO_ROOMFORM_Z_UP,
        manifest=report,
        points_meta={
            "coordinate_frame": "da3_pose_carried_consensus_phone_metric_world",
            "color_source": "mapanything_da3_consensus_rgb",
        },
        provenance={
            "source_fusion_report_json": str(report_path),
            "source_fusion_camera_solution_npz": str(camera_solution_path),
            "source_raw_evidence": report.get("source_raw_evidence"),
            "source_frame_count": report.get("view_count"),
            "source_fusion_method": report.get("method"),
            "source_fusion_voxel_m": report.get("voxel_m"),
        },
    )


def write_roomform_scan(
    cloud: CachedCloud, output_dir: Path
) -> tuple[Path, Path, Path]:
    input_dir = output_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    points_z_up = np.ascontiguousarray(
        cloud.points @ cloud.axis_transform.T, dtype=np.float32
    )
    stations_z_up = np.ascontiguousarray(
        cloud.stations @ cloud.axis_transform.T, dtype=np.float32
    )
    scan_stem = f"{cloud.room_id}-{cloud.source_kind.replace('_', '-')}"
    scan_npz = input_dir / f"{scan_stem}.npz"
    np.savez_compressed(
        scan_npz,
        pts=points_z_up,
        pts_color=cloud.colors,
        stations=stations_z_up,
    )
    scan_ply = input_dir / f"{scan_stem}.ply"
    trimesh.PointCloud(points_z_up, colors=cloud.colors).export(scan_ply)
    provenance = {
        "schema": "noesis.roomform.input.v2",
        "camera": cloud.room_id,
        "source_kind": cloud.source_kind,
        "source_revision": cloud.revision_dir.name,
        "source_revision_dir": str(cloud.revision_dir),
        "source_points_path": str(cloud.source_points_path),
        "source_points_sha256": _sha256(cloud.source_points_path),
        "source_coordinate_frame": cloud.points_meta.get("coordinate_frame"),
        "source_color_source": cloud.points_meta.get("color_source"),
        "roomform_coordinate_frame": "metric_m_z_up_right_handed",
        "source_to_roomform_z_up_row_major": cloud.axis_transform.tolist(),
        "point_count": int(len(points_z_up)),
        "bounds_noesis_backend_world_m": _bounds(cloud.points),
        "bounds_roomform_z_up_m": _bounds(points_z_up),
        "station_count": int(len(stations_z_up)),
        "stations_roomform_z_up_m": stations_z_up.tolist(),
        "stations_contract_note": (
            "Preserved in ScanInput. The tested Roomform checkout does not "
            "consume stations while building evidence."
        ),
        "scan_npz": str(scan_npz),
        "scan_ply": str(scan_ply),
        **cloud.provenance,
    }
    manifest_path = input_dir / "input_manifest.json"
    manifest_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return scan_npz, scan_ply, manifest_path


def _add_roomform_to_path(roomform_root: Path) -> None:
    package = roomform_root / "roomform"
    if not package.is_dir():
        raise RuntimeError(f"Roomform checkout is invalid: {roomform_root}")
    sys.path.insert(0, str(roomform_root))


def run_shell_cuda(
    evidence: Any,
    checkpoint: Path,
    out_npz: Path,
    *,
    precision: str,
) -> tuple[Any, dict[str, Any]]:
    from roomform.contracts import PatchGraph
    from roomform.inference.local import build_input, load_checkpoint

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Roomform accuracy checkpoint")
    device_index = 0
    device = torch.device("cuda", device_index)
    dtype_by_name = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    amp_dtype = dtype_by_name[precision]
    torch.cuda.set_device(device_index)
    torch.cuda.init()
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = precision == "fp32"
    torch.backends.cudnn.allow_tf32 = precision == "fp32"
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)
    start = time.perf_counter()
    model, config = load_checkpoint(str(checkpoint), "cpu")
    model = model.to(device=device, memory_format=torch.channels_last_3d).eval()
    features = torch.from_numpy(build_input(evidence, config))[None]
    features = features.to(device=device, memory_format=torch.channels_last_3d)
    load_s = time.perf_counter() - start
    infer_start = time.perf_counter()
    amp_enabled = precision != "fp32"
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=amp_dtype, enabled=amp_enabled
    ):
        outputs = model(features)
    torch.cuda.synchronize(device)
    inference_s = time.perf_counter() - infer_start
    sx, sy, sz = evidence.shape
    nodes = torch.sigmoid(outputs[0])[0, :, :sx, :sy, :sz]
    edges = torch.sigmoid(outputs[1])[0, :, :sx, :sy, :sz]
    arrays: dict[str, np.ndarray] = {
        "node_probs": nodes.float().cpu().numpy().astype(np.float16),
        "edge_probs": edges.float().cpu().numpy().astype(np.float16),
    }
    output_index = 2
    if config.predict_offsets:
        arrays["offsets"] = (
            outputs[output_index][0, :, :sx, :sy, :sz]
            .float()
            .cpu()
            .numpy()
            .astype(np.float16)
        )
        output_index += 1
    if config.predict_openings:
        arrays["openings"] = (
            torch.sigmoid(outputs[output_index])[0, :, :sx, :sy, :sz]
            .float()
            .cpu()
            .numpy()
            .astype(np.float16)
        )
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **arrays)
    runtime = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device_index),
        "compute_capability": list(torch.cuda.get_device_capability(device_index)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "precision": precision,
        "channels_last_3d": True,
        "tf32": precision == "fp32",
        "cudnn_benchmark": True,
        "model_load_and_transfer_s": round(load_s, 3),
        "inference_s": round(inference_s, 3),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device_index)),
        "checkpoint_config": config.model_dump(),
    }
    shell = PatchGraph(
        npz_path=str(out_npz),
        vox_m=evidence.vox_m,
        origin=evidence.origin,
        shape=evidence.shape,
        node_threshold=0.5,
        edge_threshold=0.5,
        model_id=checkpoint.name,
    )
    del outputs, features, model
    torch.cuda.empty_cache()
    return shell, runtime


def _prediction_summary(patchgraph_path: Path, threshold: float = 0.5) -> dict[str, Any]:
    data = np.load(patchgraph_path)
    nodes = np.asarray(data["node_probs"], dtype=np.float32)
    openings = np.asarray(data["openings"], dtype=np.float32) if "openings" in data else None
    result: dict[str, Any] = {
        "node_threshold": threshold,
        "wall_cells": int(np.count_nonzero(nodes[0] >= threshold)),
        "floor_cells": int(np.count_nonzero(nodes[1] >= threshold)),
        "ceiling_cells": int(np.count_nonzero(nodes[2] >= threshold)),
        "wall_probability_max": float(np.max(nodes[0])),
        "floor_probability_max": float(np.max(nodes[1])),
        "ceiling_probability_max": float(np.max(nodes[2])),
    }
    if openings is not None:
        result.update(
            {
                "door_cells_at_0p69": int(np.count_nonzero(openings[0] >= 0.69)),
                "window_cells_at_0p70": int(np.count_nonzero(openings[1] >= 0.70)),
                "door_probability_max": float(np.max(openings[0])),
                "window_probability_max": float(np.max(openings[1])),
            }
        )
    return result


def _object_summary(objects: list[Any]) -> dict[str, Any]:
    by_class: dict[str, int] = {}
    for obj in objects:
        by_class[obj.cls] = by_class.get(obj.cls, 0) + 1
    return {
        "count": len(objects),
        "by_class": dict(sorted(by_class.items())),
        "wall_leak_flagged": sum(bool(obj.qa.leaking) for obj in objects),
    }


def _git_commit(checkout: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roomform-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--room-id", default=DEFAULT_ROOM_ID)
    parser.add_argument(
        "--virtual-twin-root",
        type=Path,
        default=REPO_ROOT / "data" / "virtual_twin" / "revisions",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source-revision", type=Path, default=None)
    source.add_argument("--phone-scan-dir", type=Path, default=None)
    source.add_argument("--fusion-cloud-npz", type=Path, default=None)
    parser.add_argument("--fusion-camera-solution", type=Path, default=None)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--vox", type=float, choices=(0.08,), default=0.08)
    parser.add_argument("--local-ptv3", action="store_true")
    parser.add_argument("--pointcept-root", type=Path, default=None)
    parser.add_argument("--pointcept-checkpoint", type=Path, default=None)
    parser.add_argument("--ptv3-max-points", type=int, default=150_000)
    parser.add_argument("--ptv3-grid-m", type=float, choices=(0.02,), default=0.02)
    parser.add_argument("--ptv3-tile-overlap-m", type=float, default=0.75)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    room_id = _validated_room_id(args.room_id)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    roomform_root = args.roomform_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise RuntimeError(f"Roomform checkpoint not found: {checkpoint}")
    _add_roomform_to_path(roomform_root)
    if args.fusion_cloud_npz is not None:
        if args.fusion_camera_solution is None:
            raise RuntimeError(
                "--fusion-cloud-npz requires --fusion-camera-solution"
            )
        cloud = load_point_preserving_fusion(
            args.fusion_cloud_npz.expanduser().resolve(),
            args.fusion_camera_solution.expanduser().resolve(),
            room_id,
        )
    elif args.phone_scan_dir is not None:
        cloud = load_phone_walk_cloud(
            args.phone_scan_dir.expanduser().resolve(),
            room_id,
        )
    else:
        revision = (
            args.source_revision.expanduser().resolve()
            if args.source_revision is not None
            else find_cached_room_revision(
                args.virtual_twin_root.expanduser().resolve(),
                room_id,
            )
        )
        cloud = load_cached_cloud(revision, room_id)
    print(
        f"[roomform] source={cloud.source_kind} points={len(cloud.points):,} "
        f"stations={len(cloud.stations)}",
        flush=True,
    )
    scan_npz, scan_ply, input_manifest = write_roomform_scan(cloud, output_dir)
    print(f"[roomform] input={scan_npz}", flush=True)

    from roomform.export import export_glb, export_object_meshes
    from roomform.pipe.evidence.build import build_evidence
    from roomform.pipe.fuse import fuse

    evidence_path = output_dir / "evidence.npz"
    evidence = build_evidence(str(scan_npz), str(evidence_path), float(args.vox))
    print(f"[roomform] evidence shape={evidence.shape}", flush=True)
    patchgraph_path = output_dir / "patchgraph.npz"
    shell, runtime = run_shell_cuda(
        evidence,
        checkpoint,
        patchgraph_path,
        precision=args.precision,
    )
    print(
        f"[roomform] shell complete in {runtime['inference_s']}s; "
        f"peak={runtime['peak_cuda_memory_bytes'] / (1024**3):.2f} GiB",
        flush=True,
    )
    labels_path: Path | None = None
    lifting_runtime: dict[str, Any] | None = None
    objects: list[Any] = []
    if args.local_ptv3:
        if args.pointcept_root is None or args.pointcept_checkpoint is None:
            raise RuntimeError(
                "--local-ptv3 requires --pointcept-root and --pointcept-checkpoint"
            )
        pointcept_root = args.pointcept_root.expanduser().resolve()
        pointcept_checkpoint = args.pointcept_checkpoint.expanduser().resolve()
        if not (pointcept_root / "pointcept").is_dir():
            raise RuntimeError(f"Pointcept checkout is invalid: {pointcept_root}")
        if not pointcept_checkpoint.is_file():
            raise RuntimeError(f"PTv3 checkpoint not found: {pointcept_checkpoint}")
        if args.ptv3_max_points < 10_000:
            raise RuntimeError("--ptv3-max-points must be at least 10000")
        from testpipelines.roomform.local_pointlabel import (
            lift_local_labels,
            segment_local,
        )

        labels_path = output_dir / "labels.npz"
        print(
            f"[roomform] local PTv3 start max_points={args.ptv3_max_points:,}",
            flush=True,
        )
        lifting_runtime = segment_local(
            scan_ply,
            labels_path,
            pointcept_root=pointcept_root,
            checkpoint=pointcept_checkpoint,
            max_points=args.ptv3_max_points,
            semantic_grid_m=args.ptv3_grid_m,
            tile_overlap_m=args.ptv3_tile_overlap_m,
        )
        objects = lift_local_labels(
            labels_path,
            tuple(evidence.origin),
            sample_grid_m=float(lifting_runtime["semantic_grid_m"]),
        )
        print(
            f"[roomform] PTv3 labels={lifting_runtime['semantic_points']:,} "
            f"grid={lifting_runtime['semantic_grid_m']:.4f}m "
            f"objects_before_fusion={len(objects)}",
            flush=True,
        )

    scene_path = output_dir / "scene.json"
    scene = fuse(shell, objects, str(scan_npz), evidence.origin, str(scene_path))
    glb_path = output_dir / "scene.glb"
    planar_glb_path = output_dir / "scene-planar.glb"
    export_glb(scene, str(glb_path), str(evidence_path))
    export_glb(
        scene,
        str(planar_glb_path),
        str(evidence_path),
        include_planar=True,
    )
    object_meshes = export_object_meshes(str(output_dir)) if objects else []
    print(
        f"[roomform] scene objects={len(scene.objects)} "
        f"object_glbs={len(object_meshes)}",
        flush=True,
    )
    report = {
        "schema": "noesis.roomform.run.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "camera": cloud.room_id,
        "source_kind": cloud.source_kind,
        "roomform_commit": _git_commit(roomform_root),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "input_manifest": str(input_manifest),
        "evidence": {
            "path": str(evidence_path),
            "shape": list(evidence.shape),
            "origin": list(evidence.origin),
            "vox_m": evidence.vox_m,
            "visibility_source": evidence.visibility_source,
        },
        "runtime": runtime,
        "lifting_runtime": lifting_runtime,
        "prediction": _prediction_summary(patchgraph_path),
        "objects": _object_summary(scene.objects),
        "artifacts": {
            "scene_json": str(scene_path),
            "scene_glb": str(glb_path),
            "scene_planar_glb": str(planar_glb_path),
            "patchgraph_npz": str(patchgraph_path),
            "labels_npz": str(labels_path) if labels_path is not None else None,
            "object_meshes": object_meshes,
        },
        "limitations": [
            "The tested Roomform checkout preserves stations in ScanInput but does not yet carve evidence from them.",
            "PTv3 produces ScanNet-20 semantic labels; instances are local connected components, so adjacent same-class objects can merge.",
            "Per-object GLBs are segmented RGB point clouds. Roomform's solid SAM3D object reconstruction remains a separate external FAL stage and is not invoked.",
        ],
    }
    report_path = output_dir / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
